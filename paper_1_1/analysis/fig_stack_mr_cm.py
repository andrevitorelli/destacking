"""Two-panel figure (2026-10-02): stacks-only fits (stack_only_fits.py) in the mass-richness and mass-concentration planes, coloured by
the stack's mean magnitude gap, with the forward-modelled stacked relations and Diemer & Joyce (2019).  Also a quick assembly-bias test:
weighted regression of the per-stack log M on log(lambda), log(1+z) and Delta m (v1.1: linear)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, pickle, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.halo import concentration
from destacking.relations import mass_lambda_McClintock18
D = f"{DATA}/real_data/stacked_full_param_fit_f1.2"; FIG = f"{FIGS}"
S = pickle.load(open(f"{D}/stacks3.pkl", "rb")); J = json.load(open(f"{DATA}/output/harness_2026-09-29/results/stack_only_fits.json"))
F = np.load(f"{DATA}/output/v1.1/stacked/real_minN50.npz", allow_pickle=True)
names = [str(n) for n in F["stack_names"]]; order = [str(p) for p in F["param_order"]]; w = F["weights"] / F["weights"].sum(); X = F["samples"]
rng = np.random.default_rng(3); idx = rng.choice(len(w), size=4000, p=w); SS = {p: X[idx, i] for i, p in enumerate(order)}
med = {p: float(np.interp(0.5, np.cumsum(w[np.argsort(X[:, i])]), np.sort(X[:, i]))) for i, p in enumerate(order)}
z = np.array([J[n]["z"] for n in names]); dm = np.array([J[n]["dm"] for n in names]); N = np.array([J[n]["n"] for n in names])
lam = np.array([float(S[n]["lambda_mean"]) for n in names])
M = np.array([J[n]["logM"] for n in names]); C = np.array([J[n]["logc"] for n in names])
def wls(y, e, extra):
    X_ = np.column_stack([np.ones(len(y)), np.log10(lam / 40), np.log10((1 + z) / 1.35), *extra]); W = 1 / e ** 2
    cov = np.linalg.inv(X_.T @ (X_ * W[:, None])); b = cov @ X_.T @ (W * y); chi2 = float(np.sum(W * (y - X_ @ b) ** 2)); return b, np.sqrt(np.diag(cov)), chi2
eM = (M[:, 2] - M[:, 0]) / 2; eC = (C[:, 2] - C[:, 0]) / 2
bM, sM, chiM = wls(M[:, 1], eM, [dm])
Xc = np.column_stack([np.ones(len(z)), M[:, 1] - 14.0, np.log10((1 + z) / 1.35), dm]); Wc = 1 / eC ** 2
covc = np.linalg.inv(Xc.T @ (Xc * Wc[:, None])); bc = covc @ Xc.T @ (Wc * C[:, 1]); sc_ = np.sqrt(np.diag(covc))
print(f"M-lambda regression on stacks: alpha={bM[1]:.3f}+-{sM[1]:.3f}, beta={bM[2]:.2f}+-{sM[2]:.2f}, Delta m slope delta={bM[3]:.3f}+-{sM[3]:.3f}, chi2={chiM:.1f}/{len(z)-4}")
print(f"c-M regression on stacks: Delta m slope gamma={bc[3]:.3f}+-{sc_[3]:.3f}")
cosmology.setCosmology("planck18"); h = cosmology.getCurrent().h
zm = float(np.average(z, weights=N)); lo, hi = np.percentile(dm, [16.7, 83.3])
fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw=dict(wspace=0.25))
kw = dict(c=dm, cmap="coolwarm", s=36, edgecolor="k", lw=0.4, zorder=3, vmin=dm.min(), vmax=dm.max())
ax = axs[0]
ax.errorbar(np.log10(lam), M[:, 1], yerr=[M[:, 1] - M[:, 0], M[:, 2] - M[:, 1]], fmt="none", ecolor="0.7", lw=0.8, zorder=2)
ax.scatter(np.log10(lam), M[:, 1], **kw)
ll = np.linspace(np.log10(25), np.log10(lam.max() * 1.1), 100)
band = np.array([mass_lambda_McClintock18(10 ** ll, np.full(100, zm), Flambda=a, Gz=b) + n for a, b, n in zip(SS["alpha_mr"], SS["beta_mr"], SS["log_M_norm"])])
ax.fill_between(ll, *np.percentile(band, [16, 84], axis=0), color="0.2", alpha=0.15, lw=0)
ax.plot(ll, mass_lambda_McClintock18(10 ** ll, np.full(100, zm), Flambda=med["alpha_mr"], Gz=med["beta_mr"]) + med["log_M_norm"], "-", color="0.2", lw=1.6, label=f"stacked fit ($z={zm:.2f}$), any $\\Delta m$")
ax.text(0.03, 0.96, "per-stack log M on log $\\lambda$, log(1+z), $\\Delta m$:\n" + f"$\\Delta m$ slope $={bM[3]:+.3f}\\pm{sM[3]:.3f}$", transform=ax.transAxes, va="top", fontsize=8.5)
ax.set_xlabel(r"$\log_{10}\lambda$ (stack mean)"); ax.set_ylabel(r"$\log_{10}(M_{200{\rm m}}/M_\odot)$ (per stack)"); ax.legend(fontsize=8, loc="lower right", frameon=False)
ax = axs[1]
ax.errorbar(M[:, 1], C[:, 1], xerr=[M[:, 1] - M[:, 0], M[:, 2] - M[:, 1]], yerr=[C[:, 1] - C[:, 0], C[:, 2] - C[:, 1]], fmt="none", ecolor="0.7", lw=0.8, zorder=2)
sc = ax.scatter(M[:, 1], C[:, 1], **kw)
mm = np.linspace(13.9, 15.0, 100)
ax.plot(mm, np.log10(concentration.concentration(10 ** mm * h, "200m", zm, model="diemer19")), "--", color="0.3", lw=1.5, label=f"Diemer & Joyce 2019 ($z={zm:.2f}$)")
for dmv, col, lab in ((lo, "#3b4cc0", "low"), (hi, "#b40426", "high")):
    band = np.log10(SS["c0"])[:, None] + SS["alpha_cm"][:, None] * (mm - 14.0) + SS["beta_cm"][:, None] * np.log10((1 + zm) / 1.35) + SS["gamma_cm"][:, None] * dmv
    ax.fill_between(mm, *np.percentile(band, [16, 84], axis=0), color=col, alpha=0.15, lw=0)
    ax.plot(mm, np.log10(med["c0"]) + med["alpha_cm"] * (mm - 14.0) + med["beta_cm"] * np.log10((1 + zm) / 1.35) + med["gamma_cm"] * dmv, "-", color=col, lw=1.6, label=rf"stacked fit, {lab} $\Delta m$ ({dmv:.2f} mag)")
ax.text(0.03, 0.96, "per-stack log c on log M, log(1+z), $\\Delta m$:\n" + f"$\\Delta m$ slope $={bc[3]:+.3f}\\pm{sc_[3]:.3f}$", transform=ax.transAxes, va="top", fontsize=8.5)
ax.set_xlabel(r"$\log_{10}(M_{200{\rm m}}/M_\odot)$ (per stack)"); ax.set_ylabel(r"$\log_{10} c_{200{\rm m}}$ (per stack)"); ax.legend(fontsize=8, loc="lower left", frameon=False)
for a in axs: a.tick_params(direction="in", top=True, right=True)
cb = fig.colorbar(sc, ax=axs, fraction=0.03, pad=0.02); cb.set_label(r"stack mean $\Delta m$ [mag]")
fig.savefig(f"{FIG}/stacked_mass_richness_concentration_dm.pdf", bbox_inches="tight"); fig.savefig(f"{DATA}/output/v1.1/results/stacked_mr_cm.png", dpi=75, bbox_inches="tight")
