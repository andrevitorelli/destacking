"""Paper 1.1 (2026-10-16): diemer19_alpha_cm_prediction.pdf from the v1.1 real-data joint fit.
Replaces the 2026-07-26 figure (notebooks/diemer19_alpha_cm_comparison.ipynb on run_2026-07-25, i.e. before the (1+z) mass-richness fix,
the two-halo term, the linear Dm term and the v1.1 covariance).  Same method as that notebook: masses of the real Dm>0 clusters from the
fitted mass-richness relation at z_ref=0.35 (where the beta_Mlambda term vanishes); power-law fits to the Diemer & Joyce (2019) c(M) over
(i) a log-uniform richness grid spanning the sample (each mass bin equally) and (ii) the real clusters weighted by their Fisher information
on alpha_cm (numerical derivative of the forward model at the joint-fit medians, bootstrap covariance with floor and boost, as in the fit)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
from cluster_toolkit.boostfactors import boost_powerlaw_at_R
from destacking.model import diemer19_c200m, forward_model_combined, _MC_M_PIVOT_LOG, _MC_Z_REF
from destacking.relations import mass_lambda_McClintock18
RD = f"{DATA}/real_data"; FIG = f"{FIGS}"
# v1.1 real-data joint fit: medians of the 50 per-bootstrap medians (paper Table tab:real_results)
med = dict(c0=8.35, alpha_cm=0.033, beta_cm=0.90, gamma_cm=0.057, log_M_norm=14.289 - 14.489, alpha_mr=1.042, beta_mr=-2.05,
           f_mis=0.357, sigma_off=0.390)
R6 = np.logspace(-0.6, 0.6, 6)
D = f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0"
a = np.load(f"{D}/simulated_data.npz"); co = np.load(f"{D}/cluster_obs.npz", allow_pickle=True)
dv, cov = a["data_vectors"].copy(), a["cov_matrices"].copy()
BF = pickle.load(open(f"{RD}/wazp_boost_fits.pkl", "rb"))["boost_fits"]
for i, nm in enumerate(co["stack_name"]):
    f = BF.get(str(nm)) if nm else None
    if f is not None and not f.get("failed"):
        b = boost_powerlaw_at_R(R6, f["B0"], f["Rs"], f["alpha"]); dv[i] *= b; cov[i] *= np.outer(b, b)
lam, dm = np.asarray(co["Lambda"], float), np.asarray(co["delta_m"], float)
sel = np.isfinite(dm) & (dm > 0) & np.isfinite(dv).any(1)
lam, dm, dv, cov = lam[sel], dm[sel], dv[sel], cov[sel]; N = len(lam)
logM = mass_lambda_McClintock18(lam, _MC_Z_REF, Flambda=med["alpha_mr"], Gz=med["beta_mr"]) + med["log_M_norm"]
cD = diemer19_c200m(10 ** logM, np.full(N, _MC_Z_REF))
lam_g = np.logspace(np.log10(max(25.0, lam.min())), np.log10(lam.max()), 200)
logM_g = mass_lambda_McClintock18(lam_g, _MC_Z_REF, Flambda=med["alpha_mr"], Gz=med["beta_mr"]) + med["log_M_norm"]
cD_g = diemer19_c200m(10 ** logM_g, np.full(len(logM_g), _MC_Z_REF))
def plfit(x, y, w=None):
    w = np.ones_like(x) if w is None else w; A = np.column_stack([np.ones_like(x), x - _MC_M_PIVOT_LOG])
    c = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y)); return 10 ** c[0], c[1]
c0_g, a_g = plfit(logM_g, np.log10(cD_g))
t = Table({"z": np.full(N, _MC_Z_REF), "Lambda": lam, "delta_m": dm})
kw = dict(beta_cm=med["beta_cm"], log_M_norm=med["log_M_norm"], alpha_mr=med["alpha_mr"], beta_mr=med["beta_mr"],
          FMISS=np.full(N, med["f_mis"]), SIGMA_OFF=np.full(N, med["sigma_off"]), bin_nodes=0)
h = 1e-3
dmod = (forward_model_combined(med["c0"], med["alpha_cm"] + h, med["gamma_cm"], t, R6, **kw)
        - forward_model_combined(med["c0"], med["alpha_cm"] - h, med["gamma_cm"], t, R6, **kw)) / (2 * h)
var = np.einsum("nii->ni", cov); ok = np.isfinite(dv) & np.isfinite(var) & (var > 0) & np.isfinite(dmod)
F = np.where(ok, dmod ** 2 / np.where(ok, var, 1), 0).sum(1)       # bootstrap covariances are effectively diagonal
c0_f, a_f = plfit(logM, np.log10(cD), F)
c0_c, a_c = plfit(logM, np.log10(cD))
top = np.argsort(logM)[int(0.8 * N):]
print(f"N={N}  logM range {logM.min():.3f}-{logM.max():.3f} median {np.median(logM):.3f}")
print(f"alpha_cm Diemer: log-uniform grid {a_g:+.4f}  real counts {a_c:+.4f}  Fisher-weighted {a_f:+.4f}")
print(f"top mass quintile carries {100 * F[top].sum() / F.sum():.0f}% of the Fisher information")
fmt = lambda v: "0.000" if abs(v) < 5e-4 else f"{v:+.3f}"
mline = np.linspace(logM_g.min(), logM_g.max(), 200)
fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})
ax.plot(mline, diemer19_c200m(10 ** mline, np.full(200, _MC_Z_REF)), color="k", lw=1.8, label=rf"Diemer & Joyce (2019), $z={_MC_Z_REF}$")
ax.plot(mline, c0_g * 10 ** (a_g * (mline - _MC_M_PIVOT_LOG)), "--", color="#1f77b4", lw=1.6, label=rf"unweighted grid fit: $\alpha_{{\rm cm}}={fmt(a_g)}$")
ax.plot(mline, c0_f * 10 ** (a_f * (mline - _MC_M_PIVOT_LOG)), ":", color="#d62728", lw=2.0, label=rf"Fisher-information-weighted fit: $\alpha_{{\rm cm}}={fmt(a_f)}$")
ax.set_ylabel(r"$c_{200m}$"); ax.legend(fontsize=8)
bins = np.linspace(logM.min(), logM.max(), 31); cen = 0.5 * (bins[1:] + bins[:-1])
bx.hist(logM, bins, color="0.75", edgecolor="0.55"); bx.set_ylabel(r"$N$ clusters", color="0.4")
b2 = bx.twinx(); fs, _ = np.histogram(logM, bins, weights=F); b2.plot(cen, 100 * fs / fs.sum(), "o-", ms=3, color="#d62728", lw=1.6)
b2.set_ylabel(r"% of total $\alpha_{\rm cm}$ Fisher info", color="#d62728")
bx.set_xlabel(r"$\log_{10}(M_{200m}/M_\odot)$")
for x in (ax, bx): x.tick_params(direction="in", top=True)
fig.savefig(f"{FIG}/diemer19_alpha_cm_prediction.pdf", bbox_inches="tight"); fig.savefig(f"{DATA}/output/v1.1/report_des/figs/diemer19_alpha_cm_prediction.png", dpi=80)
np.save(f"{DATA}/output/v1.1/report_des/diemer19_alpha_v11.npy", np.array([a_g, a_c, a_f]))
