"""Forward-modelled stacked fit of the real stacks under likelihood/model variants (2026-10-07).
  python stack_variants.py VARIANT [VARIANT ...]
Variants (paired with the reference run posteriors/stack.npz of gamma_diff_2026-10-04, same seed and settings):
  hartNs   Hartlap factor with the number of member clusters N_s instead of N_boot = 500           (MC9)
  sh       Sellentin & Heavens (2016) t-likelihood with N_s, no Hartlap factor                       (MC9)
  dmmis    f_mis free per magnitude-gap tercile (3 values) instead of one                           (MC3)
  dmsoff   f_mis and sigma_off free per magnitude-gap tercile                                        (MC3)
  f11      stacks measured with the joint fit's f = 1.1 background selection (and their own boosts)          (MC4a)
  f11r025  f11, bins above 0.25 Mpc only                                                               (MC4a)
  r025     f = 1.2 stacks, bins above 0.25 Mpc only, delta_Mlambda = 0                                   (MC4a)
  rshear   reduced shear: model Delta Sigma -> Delta Sigma * (1 + Sigma <Sc^-3>/<Sc^-2>) per member     (MC7)
Output: posteriors/stack_<variant>.npz next to the reference.
"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, sys, pickle, importlib.util, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from astropy.table import Table
from nautilus import Prior, Sampler
from destacking.dmform import DM_FORM, dm_regressor
assert DM_FORM == 'linear', 'v1.1 runs need DESTACKING_DM_FORM=linear'
RD = f"{DATA}/real_data"; OUT = f"{DATA}/output/v1.1/posteriors"
AUX = os.path.expanduser(f"{AUX}"); sys.path.insert(0, AUX)
spec = importlib.util.spec_from_file_location("sff", f"{AUX}/stacked_forward_fit.py"); sff = importlib.util.module_from_spec(spec); spec.loader.exec_module(sff)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
PRI = {"c0": stats.uniform(1, 14), "alpha_cm": stats.norm(-0.1, 0.5), "beta_cm": stats.norm(-0.4, 2), "gamma_cm": stats.norm(0, 1),
       "log_M_norm": stats.norm(0, 0.5), "alpha_mr": stats.norm(1.356, 0.5), "beta_mr": stats.norm(-0.3, 2), "f_mis": stats.uniform(0, 1),
       "sigma_off": stats.uniform(0, 1)}
NB = 500


def build(f11=False, r_min=0.0, likelihood="hartlap"):
    cat = Table.read(f"{RD}/wazp_real_data.fits")
    st, wt = ((f"{RD}/stacked_f1.1_3way/stacks3.pkl", f"{RD}/stacked_f1.1_3way/weights.npz") if f11 else
              (f"{RD}/stacked_full_param_fit_f1.2/stacks3.pkl", f"{RD}/stack_weights_f1.2.npz"))
    M = sff.StackedForwardModel(cat, pickle.load(open(st, "rb")), np.load(wt), min_n=50, two_halo=True, bin_nodes=0, r_min=r_min, likelihood=likelihood)
    return M


def run(variant):
    M = build(f11=variant.startswith("f11"), r_min=0.25 if variant.endswith("r025") else 0.0, likelihood="sh" if variant == "sh" else "hartlap"); names = list(BASE); prior = dict(PRI); base_model = M.model
    hart_nb = [(NB - m.sum() - 2) / (NB - 1) for m in M.masks]
    if variant in ("f11", "f11r025", "r025", "sh"):
        loglike = M.loglike
    elif variant == "hartNs_old":
        M.cinv = [c * ((n - m.sum() - 2) / (n - 1)) / h for c, n, m, h in zip(M.cinv, M.n_stacked, M.masks, hart_nb)]
        loglike = M.loglike
    elif variant == "sh_old":
        C0 = [c / h for c, h in zip(M.cinv, hart_nb)]
        def loglike(p):
            if p["c0"] <= 0 or p["sigma_off"] <= 0: return -np.inf
            mod = M.model(p); ll = 0.0
            for k in range(len(M.names)):
                r = M.data[k] - mod[k][M.masks[k]]
                if not np.all(np.isfinite(r)): return -np.inf
                n = M.n_stacked[k]; ll -= 0.5 * n * np.log1p((r @ C0[k] @ r) / (n - 1))
            return float(ll) if np.isfinite(ll) else -np.inf
    elif variant in ("dmmis", "dmsoff"):
        # tercile of each member, from the stack it belongs to (S is stacks x members)
        terc = np.zeros(M.S.shape[1], int)
        for k, nm in enumerate(M.names):
            terc[M.S[k].indices] = int(nm.split("_dm")[1])
        extra = [f"f_mis_t{t}" for t in range(3)] + ([f"sigma_off_t{t}" for t in range(3)] if variant == "dmsoff" else [])
        names = [n for n in BASE if n not in ("f_mis",) + (("sigma_off",) if variant == "dmsoff" else ())] + extra
        for e in extra: prior[e] = stats.uniform(0, 1)
        from destacking.model import _nfw_components
        import xlensing
        def model(p):
            logM = sff.mass_lambda_McClintock18(M.lam, M.z, Flambda=p["alpha_mr"], Gz=p["beta_mr"]) + p["log_M_norm"]
            logc = (np.log10(p["c0"]) + p["alpha_cm"] * (logM - sff.M_PIVOT_LOG) + p["beta_cm"] * np.log10((1 + M.z) / (1 + sff.Z_REF))
                    + p["gamma_cm"] * M.log_dm)
            fm = np.array([p[f"f_mis_t{t}"] for t in range(3)])[terc]
            so = np.array([p[f"sigma_off_t{t}"] for t in range(3)])[terc] if variant == "dmsoff" else np.full(len(M.z), p["sigma_off"])
            cen, mis = _nfw_components(M.z, 10.0 ** logM, 10.0 ** logc, M.radii, SIGMA_OFF=so, bin_nodes=M.bin_nodes)
            ds = (1 - fm[:, None]) * cen + fm[:, None] * mis + xlensing.model.Bias(M.z, 10.0 ** logM)[:, None] * M.ds_l
            with np.errstate(invalid="ignore", divide="ignore"):
                return (M.S @ (M.K * ds)) / M.norm
        def loglike(p):
            q = dict(p); q.setdefault("f_mis", 0.5); q.setdefault("sigma_off", 0.5)
            if p["c0"] <= 0 or any(p[k] <= 0 for k in p if k.startswith("sigma_off")): return -np.inf
            mod = model(p); ll = 0.0
            for k in range(len(M.names)):
                r = M.data[k] - mod[k][M.masks[k]]
                if not np.all(np.isfinite(r)): return -np.inf
                ll -= 0.5 * r @ M.cinv[k] @ r
            return float(ll) if np.isfinite(ll) else -np.inf
    elif variant == "rshear":
        from reduced_shear import kappa_factor
        def model(p):
            ds = base_model(p)       # stacked (K-weighted) model; correction applied per member below instead
            return ds
        import xlensing
        from destacking.model import _nfw_components
        def model_rs(p):
            logM = sff.mass_lambda_McClintock18(M.lam, M.z, Flambda=p["alpha_mr"], Gz=p["beta_mr"]) + p["log_M_norm"]
            logc = (np.log10(p["c0"]) + p["alpha_cm"] * (logM - sff.M_PIVOT_LOG) + p["beta_cm"] * np.log10((1 + M.z) / (1 + sff.Z_REF))
                    + p["gamma_cm"] * M.log_dm)
            cen, mis = _nfw_components(M.z, 10.0 ** logM, 10.0 ** logc, M.radii, SIGMA_OFF=np.full(len(M.z), p["sigma_off"]), bin_nodes=M.bin_nodes)
            ds = (1 - p["f_mis"]) * cen + p["f_mis"] * mis + xlensing.model.Bias(M.z, 10.0 ** logM)[:, None] * M.ds_l
            ds = ds * kappa_factor(M.z, 10.0 ** logM, 10.0 ** logc, M.radii, p["f_mis"], p["sigma_off"], f_sel=1.2)
            with np.errstate(invalid="ignore", divide="ignore"):
                return (M.S @ (M.K * ds)) / M.norm
        M.model = model_rs; loglike = M.loglike
    else:
        raise SystemExit(f"unknown variant {variant}")
    pr = Prior()
    for nm in names: pr.add_parameter(nm, dist=prior[nm])
    smp = Sampler(pr, loglike, n_live=2000, pool=int(os.environ.get("NPOOL", "14")), seed=1)
    smp.run(verbose=False, n_eff=3000)
    X, lw, ll = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    np.savez(f"{OUT}/stack_{variant}.npz", samples=X, weights=w, log_l=ll, param_order=np.array(names), log_z=smp.log_z, dm_form=DM_FORM)
    q = lambda j: np.interp([0.16, 0.5, 0.84], np.cumsum(w[np.argsort(X[:, j])]), np.sort(X[:, j]))
    print(variant, " ".join(f"{nm}={np.round(q(j), 3).tolist()}" for j, nm in enumerate(names)), f"logZ={smp.log_z:.2f}", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        if os.path.exists(f"{OUT}/stack_{v}.npz"): print("cached", v); continue
        run(v)
