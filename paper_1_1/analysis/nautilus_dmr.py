"""Posteriors with a magnitude-gap term in the mass-richness relation (2026-10-04).
  nautilus_dmr.py stack [--dmr]            forward-modelled stacked fit of the real stacks (stacked-fit priors)
  nautilus_dmr.py joint [--dmr] B0 B1      joint fit, cluster-bootstrap draws B0..B1-1 (fiducial real-data joint setup: f=1.1 6-bin
                                           bootstrap data with floor, 2-way boost, c0 ~ N(4.5,5), flat miscentring priors)
delta_Mlambda enters as log10 M += delta * log10(Delta m / 0.1), implemented exactly as an effective richness lambda*(dm/0.1)^(delta/alpha_mr);
prior N(0, 1).  Output: ~/destacking_data/output/gamma_diff_2026-10-04/posteriors/"""
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
RD = f"{DATA}/real_data"; OUT = f"{DATA}/output/v1.1/posteriors"; os.makedirs(OUT, exist_ok=True)
mode = sys.argv[1]; DMR = "--dmr" in sys.argv; args = [a for a in sys.argv[2:] if a != "--dmr"]
NAMES = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"] + (["delta_mr"] if DMR else [])
common = {"alpha_cm": stats.norm(-0.1, 0.5), "beta_cm": stats.norm(-0.4, 2), "gamma_cm": stats.norm(0, 1), "log_M_norm": stats.norm(0, 0.5),
          "alpha_mr": stats.norm(1.356, 0.5), "beta_mr": stats.norm(-0.3, 2), "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0, 1), "delta_mr": stats.norm(0, 1)}
if mode == "stack":
    AUX = os.path.expanduser(f"{AUX}"); sys.path.insert(0, AUX)
    spec = importlib.util.spec_from_file_location("sff", f"{AUX}/stacked_forward_fit.py"); sff = importlib.util.module_from_spec(spec); spec.loader.exec_module(sff)
    M = sff.StackedForwardModel(Table.read(f"{RD}/wazp_real_data.fits"), pickle.load(open(f"{RD}/stacked_full_param_fit_f1.2/stacks3.pkl", "rb")),
                                np.load(f"{RD}/stack_weights_f1.2.npz"), min_n=50, two_halo=True, bin_nodes=0, r_min=float(os.environ.get("RMIN", "0")))
    LAM0 = M.lam.copy(); PRI = dict(common, c0=stats.uniform(1, 14))
    def loglike(p):
        if DMR: M.lam = LAM0 * 10 ** (p["delta_mr"] * M.log_dm / max(p["alpha_mr"], 1e-3))
        return M.loglike(p)
    draws = [None]
else:
    from destacking.fitting import log_likelihood_combined as L
    from cluster_toolkit.boostfactors import boost_powerlaw_at_R
    R6 = np.logspace(-0.6, 0.6, 6)
    a = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/simulated_data.npz"); co = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/cluster_obs.npz", allow_pickle=True)
    dv, cov = a["data_vectors"].copy(), a["cov_matrices"].copy(); BF = pickle.load(open(f"{RD}/wazp_boost_fits.pkl", "rb"))["boost_fits"]
    for i, nm in enumerate(co["stack_name"]):
        f = BF.get(str(nm)) if nm else None
        if f is not None and not f.get("failed"): b = boost_powerlaw_at_R(R6, f["B0"], f["Rs"], f["alpha"]); dv[i] *= b; cov[i] *= np.outer(b, b)
    TAB = Table({k: np.asarray(co[k], float) for k in ("z", "Lambda", "delta_m")}); sel = np.isfinite(TAB["delta_m"]) & (TAB["delta_m"] > 0)
    TAB, DV, CV = TAB[sel], dv[sel], cov[sel]; PRI = dict(common, c0=stats.truncnorm((0 - 4.5) / 5, np.inf, loc=4.5, scale=5))
    draws = list(range(int(args[0]), int(args[1])))
    STATE = {}
    def loglike(p):
        t, d, c, lam0, ldm = STATE["t"], STATE["d"], STATE["c"], STATE["lam0"], STATE["ldm"]; n = len(t)
        if DMR: t["Lambda"] = lam0 * 10 ** (p["delta_mr"] * ldm / max(p["alpha_mr"], 1e-3))
        return L(p["c0"], p["alpha_cm"], p["gamma_cm"], t, d, c, R6, beta_cm=p["beta_cm"], log_M_norm=p["log_M_norm"], alpha_mr=p["alpha_mr"], beta_mr=p["beta_mr"],
                 FMISS=np.full(n, p["f_mis"]), SIGMA_OFF=np.full(n, p["sigma_off"]), two_halo=True, bin_nodes=0)
for b in draws:
    tag = f"{mode}{'_dmr' if DMR else ''}{'_rmin' + os.environ['RMIN'] if os.environ.get('RMIN') else ''}" + ("" if b is None else f"_b{b:03d}")
    if os.path.exists(f"{OUT}/{tag}.npz"): print("cached", tag); continue
    if b is not None:
        idx = np.random.default_rng(7000 + b).integers(0, len(TAB), len(TAB)); t = TAB[idx]
        STATE.update(t=t, d=DV[idx], c=CV[idx], lam0=np.asarray(t["Lambda"], float).copy(), ldm=dm_regressor(np.asarray(t["delta_m"], float), 0.1))
    pr = Prior()
    for nm in NAMES: pr.add_parameter(nm, dist=PRI[nm])
    smp = Sampler(pr, loglike, n_live=2000 if mode == "stack" else 1000, pool=int(os.environ.get("NPOOL", "12")), seed=b if b is not None else 1)
    smp.run(verbose=False, n_eff=3000)
    X, lw, ll = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    np.savez(f"{OUT}/{tag}.npz", samples=X, weights=w, log_l=ll, param_order=np.array(NAMES), log_z=smp.log_z, dm_form=DM_FORM)
    q = lambda j: np.interp([0.16, 0.5, 0.84], np.cumsum(w[np.argsort(X[:, j])]), np.sort(X[:, j]))
    print(tag, " ".join(f"{nm}={np.round(q(j), 3).tolist()}" for j, nm in enumerate(NAMES) if nm in ("gamma_cm", "delta_mr", "c0", "alpha_mr")), f"logZ={smp.log_z:.2f}", flush=True)
