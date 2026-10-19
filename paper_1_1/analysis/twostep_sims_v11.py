"""v1.1 port (2026-10-16): linear magnitude-gap regressor (gamma_cm * mean Dm of the stack), v1.1 scan stacks (jvs/{TL}_stacks/stacks3.pkl),
forward-modelled medians from jvs/{TL}_stacked_hN.npz, output in {O}/twostep_v11.  Otherwise identical to the 2026-10-06 1.0 version.

Two-step stacked analysis of the simulated realisations, for comparison with the forward-modelled stacked fit (2026-10-06).

Step 1: each stack (>= 50 members) is fit on its own with an NFW + miscentring model (log M, log c, f_mis, sigma_off free, flat
priors; no two-halo term, as the simulations contain none), evaluated at the stack's mean redshift; Hartlap-corrected stack
covariance, as the forward-modelled fit.
Step 2: weighted least squares of the per-stack (log M, log c) on the stacks' mean observables:
    log M = log M0 + alpha_Mlambda log(lambda/40) + beta_Mlambda log((1+z)/1.35)
    log c = log c0 + alpha_cm (log M - 14) + beta_cm log((1+z)/1.35) + gamma_cm Delta m
(log M of step 1 is used as the regressor of the second relation, as in the classic two-step method).
Output: one JSON per realisation with step-2 coefficients, their WLS errors, the truths, and the forward-modelled fit's medians.
usage: twostep_sims.py NAME [NAME ...]     (NAME = e.g. LSSTind1k1, DESfull3k2)
"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, sys, json, glob, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
O = os.environ.get("TWOSTEP_BASE", f"{DATA}/output"); OUT = os.environ.get("TWOSTEP_OUT", f"{DATA}/output/v1.1/twostep_v11"); os.makedirs(OUT, exist_ok=True)
FT = {}   # v1.1: forward medians and truths are read per realisation
R = np.logspace(-0.8, 0.6, 16); NB = 500; MIN_N = 50


def fit_stack(args):
    name, s = args
    from destacking.model import _nfw_components
    from nautilus import Prior, Sampler
    d = np.asarray(s["sigmas"], float); C = np.asarray(s["sigmas_cov"], float)
    ok = np.isfinite(d) & np.isfinite(np.diag(C)) & (np.diag(C) > 0)
    if ok.sum() < 4:
        return name, None
    Cs = C[np.ix_(ok, ok)]; Cs = Cs + 1e-6 * np.eye(ok.sum()) * np.mean(np.diag(Cs))
    Nr = int(s["n_stacked"]); Ci = (Nr - ok.sum() - 2) / (Nr - 1) * np.linalg.inv(Cs); z = float(s["z_mean"])   # Hartlap with N = member clusters (2026-10-08)
    pr = Prior(); pr.add_parameter("logM", dist=(13.0, 16.0)); pr.add_parameter("logc", dist=(-0.3, 1.7))
    pr.add_parameter("f_mis", dist=(0.0, 1.0)); pr.add_parameter("sigma_off", dist=(0.01, 1.5))

    def ll(P):
        P = np.atleast_2d(P); n = len(P)
        cen, mis = _nfw_components(np.full(n, z), 10 ** P[:, 0], 10 ** P[:, 1], R, SIGMA_OFF=P[:, 3], bin_nodes=0)
        m = (1 - P[:, 2:3]) * cen + P[:, 2:3] * mis; r = d[ok][None, :] - m[:, ok]
        out = -0.5 * np.einsum("ni,ij,nj->n", r, Ci, r); return np.where(np.isfinite(out), out, -1e30)
    smp = Sampler(pr, ll, n_live=600, vectorized=True, pass_dict=False, seed=1)
    smp.run(verbose=False, n_eff=1500)
    X, lw, _ = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    q = {}
    for j, p in enumerate(("logM", "logc")):
        o = np.argsort(X[:, j]); q[p] = list(np.interp([0.16, 0.5, 0.84], np.cumsum(w[o]), X[o, j]))
    return name, dict(q, z=z, lam=float(s["lambda_mean"]), dm=float(s["dm_mean"]), n=int(s["n_stacked"]))


def wls(y, e, cols):
    X = np.column_stack([np.ones(len(y))] + cols); W = 1 / e ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None])); b = cov @ X.T @ (W * y)
    return b, np.sqrt(np.diag(cov)), float(np.sum(W * (y - X @ b) ** 2)), len(y) - X.shape[1]


def forward_medians(name):
    if name in FT: return FT[name]["forward"]
    f = f"{O}/jvs/{name}_stacked_hN.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True); o = [str(p) for p in z["param_order"]]; X = z["samples"]; w = z["weights"] / z["weights"].sum(); out = {}
    for j, p in enumerate(o):
        s = np.argsort(X[:, j]); c = np.cumsum(w[s]); out[p] = list(np.interp([0.16, 0.5, 0.84], c, X[s, j]))
    return out


if __name__ == "__main__":
    with Pool(int(os.environ.get('NPROC', '32'))) as pool:
        for name in sys.argv[1:]:
            fo = f"{OUT}/{name}.json"
            if os.path.exists(fo):
                print("cached", name); continue
            S = pickle.load(open(f"{O}/jvs/{name}_stacks/stacks3.pkl", "rb"))
            S = S["stacks"] if isinstance(S, dict) and "stacks" in S and "z_bin" in S else S
            items = [(k, v) for k, v in S.items() if int(v["n_stacked"]) >= MIN_N]
            res = {k: v for k, v in pool.map(fit_stack, items) if v is not None}
            ks = sorted(res); a = lambda f: np.array([f(res[k]) for k in ks], float)
            logM = a(lambda r: r["logM"][1]); eM = a(lambda r: (r["logM"][2] - r["logM"][0]) / 2)
            logc = a(lambda r: r["logc"][1]); ec = a(lambda r: (r["logc"][2] - r["logc"][0]) / 2)
            lam, z, dm = a(lambda r: r["lam"]), a(lambda r: r["z"]), a(lambda r: r["dm"])
            lz = np.log10((1 + z) / 1.35)
            bM, sM, chiM, dofM = wls(logM, eM, [np.log10(lam / 40), lz])
            bc, sc, chic, dofc = wls(logc, ec, [logM - 14.0, lz, dm])   # v1.1: linear in Delta m
            truth = FT[name]["truth"] if name in FT else json.load(open(glob.glob(f"{O}/{name}_sim/simulation_*/ols.json")[0]))
            out = dict(name=name, n_stacks=len(ks), stacks=res,
                       twostep=dict(log10M0=[bM[0], sM[0]], alpha_mr=[bM[1], sM[1]], beta_mr=[bM[2], sM[2]], chi2_mr=[chiM, dofM],
                                    c0=[10 ** bc[0], 10 ** bc[0] * np.log(10) * sc[0]], alpha_cm=[bc[1], sc[1]], beta_cm=[bc[2], sc[2]],
                                    gamma_cm=[bc[3], sc[3]], chi2_cm=[chic, dofc]),
                       truth=truth, forward=forward_medians(name))
            json.dump(out, open(fo, "w"), indent=1)
            print(name, len(ks), "stacks; two-step gamma", round(bc[3], 3), "+-", round(sc[3], 3), "truth", round(truth["gamma_true_ols"], 3),
                  "forward", None if out["forward"] is None else round(out["forward"]["gamma_cm"][1], 3), flush=True)
