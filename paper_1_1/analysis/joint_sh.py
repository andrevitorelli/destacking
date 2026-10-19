"""Real-data joint fit with a per-bin Sellentin & Heavens (2016) likelihood for the galaxy-bootstrap covariance (2026-10-08).
For cluster i and radial bin k, with residual r, bootstrap variance s2 (boost-corrected) and n_k background sources in the bin,
    ln L_ik = -(n_k / 2) ln[1 + r^2 / (s2 (n_k - 1))]
(the covariances are effectively diagonal); bins whose variance was floored at the analytic shape-noise value (below 0.5 Mpc) are
not estimates and keep the Gaussian -r^2 / (2 s2).  Same data and model as the paper's joint fit v9d (Delta m > 0, 2-way boost,
two-halo term, bin centres, c0 ~ N(4.5,5) > 0, flat miscentring priors).
  python joint_sh.py full                 full-sample fit
  python joint_sh.py boot B0 B1           cluster-bootstrap draws B0..B1-1 (same resampling as nautilus_dmr.py)
Output: ~/destacking_data/output/gamma_diff_2026-10-04/posteriors/joint_sh_{full,bNNN}.npz
"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, sys, pickle, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from astropy.table import Table
from nautilus import Prior, Sampler
from destacking.dmform import DM_FORM, dm_regressor
assert DM_FORM == 'linear', 'v1.1 runs need DESTACKING_DM_FORM=linear'
from destacking.model import forward_model_combined, forward_model_diemer
from cluster_toolkit.boostfactors import boost_powerlaw_at_R
RD = os.environ.get("SH_RD", f"{DATA}/real_data"); OUT = os.environ.get("SH_OUT", f"{DATA}/output/v1.1/posteriors")
MODEL = os.environ.get("SH_MODEL", "power")   # power (paper model) | dmr (delta_Mlambda free) | diemer (Diemer19 baseline + gamma)
R6 = np.logspace(-0.6, 0.6, 6)
NAMES = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
PRI = {"c0": stats.truncnorm((0 - 4.5) / 5, np.inf, loc=4.5, scale=5), "alpha_cm": stats.norm(-0.1, 0.5), "beta_cm": stats.norm(-0.4, 2),
       "gamma_cm": stats.norm(0, 1), "log_M_norm": stats.norm(0, 0.5), "alpha_mr": stats.norm(1.356, 0.5), "beta_mr": stats.norm(-0.3, 2),
       "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0, 1), "delta_mr": stats.norm(0, 1), "gamma": stats.norm(0, 1)}
if MODEL == "dmr": NAMES = NAMES + ["delta_mr"]
if MODEL == "diemer": NAMES = ["gamma", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
TAGP = {"power": "joint_sh", "dmr": "joint_sh_dmr", "diemer": "joint_sh_diemer"}[MODEL]
a = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/simulated_data.npz")
co = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/cluster_obs.npz", allow_pickle=True)
bc = np.load(f"{RD}/wazp_real_data_bootfloor_v4/bin_counts.npz")
dv, cov = a["data_vectors"].copy(), a["cov_matrices"].copy(); BF = pickle.load(open(f"{RD}/wazp_boost_fits.pkl", "rb"))["boost_fits"]
for i, nm in enumerate(co["stack_name"]):
    f = BF.get(str(nm)) if nm else None
    if f is not None and not f.get("failed"): b = boost_powerlaw_at_R(R6, f["B0"], f["Rs"], f["alpha"]); dv[i] *= b; cov[i] *= np.outer(b, b)
var = np.array([np.diag(c) for c in cov]); nk = bc["n_k"].astype(float); fl = bc["floored"]
TAB = Table({k: np.asarray(co[k], float) for k in ("z", "Lambda", "delta_m")})
sel = np.isfinite(TAB["delta_m"]) & (TAB["delta_m"] > 0)
TAB, DV, VAR, NK, FL = TAB[sel], dv[sel], var[sel], nk[sel], fl[sel]
STATE = {}


def loglike(p):
    t, d, v, n, flo = STATE["t"], STATE["d"], STATE["v"], STATE["n"], STATE["fl"]
    if p.get("c0", 1.0) <= 0 or p["sigma_off"] <= 0: return -np.inf
    try:
        kw = dict(log_M_norm=p["log_M_norm"], alpha_mr=p["alpha_mr"], beta_mr=p["beta_mr"], FMISS=np.full(len(t), p["f_mis"]),
                  SIGMA_OFF=np.full(len(t), p["sigma_off"]), two_halo=True, bin_nodes=0)
        if MODEL == "diemer":
            m = forward_model_diemer(p["gamma"], t, R6, **kw)
        else:
            if MODEL == "dmr":   # log10 M += delta log10(dm/0.1), as an effective richness (nautilus_dmr.py convention)
                t = t.copy(); t["Lambda"] = STATE["lam0"] * 10 ** (p["delta_mr"] * STATE["ldm"] / max(p["alpha_mr"], 1e-3))
            m = forward_model_combined(p["c0"], p["alpha_cm"], p["gamma_cm"], t, R6, beta_cm=p["beta_cm"], **kw)
    except Exception:
        return -np.inf
    dat = np.isfinite(d) & np.isfinite(v) & (v > 0)
    if not np.all(np.isfinite(m[dat])): return -np.inf          # a model that is not finite where there are data is rejected
    r2 = (d - m) ** 2; ok = dat
    q = np.where(ok, r2 / np.where(ok, v, 1.0), 0.0)
    sh = ok & ~flo & (n >= 2); ga = ok & ~sh
    ll = -0.5 * np.sum(n[sh] * np.log1p(q[sh] / (n[sh] - 1))) - 0.5 * np.sum(q[ga])
    return float(ll) if np.isfinite(ll) else -np.inf


def run(tag, idx):
    if os.path.exists(f"{OUT}/{tag}.npz") or os.path.exists(f"{OUT}/{tag}.skipped"): print("cached", tag); return
    STATE.update(t=TAB[idx], d=DV[idx], v=VAR[idx], n=NK[idx], fl=FL[idx], lam0=np.asarray(TAB["Lambda"], float)[idx],
                 ldm=dm_regressor(np.asarray(TAB["delta_m"], float)[idx], 0.1))
    pr = Prior()
    for nm in NAMES: pr.add_parameter(nm, dist=PRI[nm])
    b = int(tag.split("_b")[1]) if "_b" in tag else 1
    smp = Sampler(pr, loglike, n_live=1000, pool=int(os.environ.get("NPOOL", "15")), seed=b)
    ok = smp.run(verbose=False, n_eff=3000, timeout=float(os.environ.get("SH_TIMEOUT", "2400")))
    if ok is False or getattr(smp, "n_eff", 3000) < 1000:   # a draw that does not converge in the time limit is skipped
        print("NOT CONVERGED, skipped", tag, flush=True); open(f"{OUT}/{tag}.skipped", "w").write("timeout\n"); return
    X, lw, ll = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    np.savez(f"{OUT}/{tag}.npz", samples=X, weights=w, log_l=ll, param_order=np.array(NAMES), log_z=smp.log_z, dm_form=DM_FORM)
    qf = lambda j: np.interp([0.16, 0.5, 0.84], np.cumsum(w[np.argsort(X[:, j])]), np.sort(X[:, j]))
    print(tag, " ".join(f"{nm}={np.round(qf(j), 3).tolist()}" for j, nm in enumerate(NAMES)), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "full":
        run(f"{TAGP}_full", np.arange(len(TAB)))
    else:
        want = int(sys.argv[3]) - int(sys.argv[2]); b = int(sys.argv[2])
        while sum(os.path.exists(f"{OUT}/{TAGP}_b{k:03d}.npz") for k in range(int(sys.argv[2]), b)) < want and b < int(sys.argv[2]) + want + 10:
            run(f"{TAGP}_b{b:03d}", np.random.default_rng(7000 + b).integers(0, len(TAB), len(TAB))); b += 1
