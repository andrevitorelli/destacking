import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import sys, json, pickle, numpy as np, warnings; warnings.filterwarnings("ignore")
from destacking.model import _nfw_profiles
from cluster_toolkit.boostfactors import boost_powerlaw_at_R
RD = f"{DATA}/real_data"; R6 = np.logspace(-0.6, 0.6, 6); LN10 = np.log(10)
a = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/simulated_data.npz")
co = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/cluster_obs.npz", allow_pickle=True)
dv, cov = a["data_vectors"].copy(), a["cov_matrices"].copy(); BF = pickle.load(open(f"{RD}/wazp_boost_fits.pkl", "rb"))["boost_fits"]
for i, nm in enumerate(co["stack_name"]):
    f = BF.get(str(nm)) if nm else None
    if f is not None and not f.get("failed"): b = boost_powerlaw_at_R(R6, f["B0"], f["Rs"], f["alpha"]); dv[i] *= b; cov[i] *= np.outer(b, b)
var = np.array([np.diag(c) for c in cov]); var = np.where(np.isfinite(dv), var, np.nan)
z, lam, dm = (np.asarray(co[k], float) for k in ("z", "Lambda", "delta_m")); s = np.isfinite(dm) & (dm > 0)
z, lam, dm, var = z[s], lam[s], dm[s], var[s]
p = {k: v["med"] for k, v in json.load(open(f"{DATA}/output/v1.1/results/sh_summary.json"))["joint_sh"].items() if not k.startswith("_")}
logM = 14.489 + p["log_M_norm"] + p["alpha_mr"] * np.log10(lam / 40) + p["beta_mr"] * np.log10((1 + z) / 1.35)
for form, x in (("linear", dm), ("log", np.log10(dm / 0.1))):
    c = 10 ** (np.log10(p["c0"]) + p["alpha_cm"] * (logM - 14) + p["beta_cm"] * np.log10((1 + z) / 1.35) + p["gamma_cm"] * dm)
    for mis in (False, True):
        kw = dict(FMISS=np.full(len(z), p["f_mis"]), SIGMA_OFF=np.full(len(z), p["sigma_off"])) if mis else {}
        prof = lambda cc: _nfw_profiles(z, 10 ** logM, cc, R6, bin_nodes=0, **kw); h = 1e-3
        A = (prof(c * np.exp(h)) - prof(c * np.exp(-h))) / (2 * h); v = np.isfinite(var) & (var > 0); iv = np.where(v, 1 / np.where(v, var, 1), 0)
        r = {}
        for q, f in (("gamma", LN10 * x[:, None]), ("c0", np.ones_like(x)[:, None])):
            d2 = (A * f) ** 2 * iv; r[q] = round(float(np.sqrt(d2.sum() / d2[:, R6 >= 0.5].sum())), 2)
        print(form, "miscentred" if mis else "centred", len(z), r)
