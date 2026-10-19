"""Stacks-only fits for the c-M figure (2026-10-02): each real stack fit on its own with free log M, log c, f_mis, sigma_off,
two-halo term included, flat priors (log c flat on [0, 1.6]); nautilus, vectorised likelihood.  Output: per-stack quantiles."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import pickle, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from multiprocessing import Pool
D = f"{DATA}/real_data/stacked_full_param_fit_f1.2"; OUT = f"{DATA}/output/harness_2026-09-29/results/stack_only_fits.json"
R = np.logspace(-0.8, 0.6, 16); NB = 500
def fit(name):
    import xlensing
    from destacking.model import _nfw_components
    from nautilus import Prior, Sampler
    s = pickle.load(open(f"{D}/stacks3.pkl", "rb"))[name]
    d = np.asarray(s["sigmas"], float); C = np.asarray(s["sigmas_cov"], float); ok = np.isfinite(d) & (np.diag(C) > 0)
    Cs = C[np.ix_(ok, ok)]; Cs = Cs + 1e-6 * np.eye(ok.sum()) * np.mean(np.diag(Cs)); Nr = int(s["n_stacked"]); Ci = (Nr - ok.sum() - 2) / (Nr - 1) * np.linalg.inv(Cs)   # Hartlap with N = member clusters (2026-10-08)
    z = float(s["z_mean"]); dsl = xlensing.model.Delta_Sigma_l(np.array([z]), R)[0] / 1e12
    pr = Prior(); pr.add_parameter("logM", dist=(13.3, 15.7)); pr.add_parameter("logc", dist=(0.0, 1.6))
    pr.add_parameter("f_mis", dist=(0.0, 1.0)); pr.add_parameter("sigma_off", dist=(0.01, 1.0))
    def ll(P):
        P = np.atleast_2d(P); n = len(P); M = 10 ** P[:, 0]; zz = np.full(n, z)
        cen, mis = _nfw_components(zz, M, 10 ** P[:, 1], R, SIGMA_OFF=P[:, 3], bin_nodes=0)
        mod = (1 - P[:, 2:3]) * cen + P[:, 2:3] * mis + xlensing.model.Bias(zz, M)[:, None] * dsl[None, :]
        r = d[ok][None, :] - mod[:, ok]; out = -0.5 * np.einsum("ni,ij,nj->n", r, Ci, r)
        return np.where(np.isfinite(out), out, -1e30)
    smp = Sampler(pr, ll, n_live=1000, vectorized=True, pass_dict=False, seed=1)
    smp.run(verbose=False, n_eff=3000)
    X, lw, _ = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    q = {}
    for j, p in enumerate(("logM", "logc", "f_mis", "sigma_off")):
        o = np.argsort(X[:, j]); q[p] = list(np.interp([0.16, 0.5, 0.84], np.cumsum(w[o]), X[o, j]))
    chi2 = float(-2 * ll(np.array([[q["logM"][1], q["logc"][1], q["f_mis"][1], q["sigma_off"][1]]]))[0])
    return name, dict(q, z=z, dm=float(s["dm_mean"]), n=int(s["n_stacked"]), nbins=int(ok.sum()), chi2_med=chi2)
if __name__ == "__main__":
    names = [str(n) for n in np.load(f"{DATA}/output/stacked_forward/real_minN50_v3.npz", allow_pickle=True)["stack_names"]]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        res = dict(pool.map(fit, names))
    json.dump(res, open(OUT, "w"), indent=1); print("done", len(res))
