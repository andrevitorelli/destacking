"""Paper 1.1 (2026-10-11): 3-panel example_pop_sim.pdf (gap histogram vs real, c-M coloured by gap, c-gap coloured by mass) from the option-C truth catalogue of DESsx0L (gamma=1/6, seed 4101)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
FIG = f"{FIGS}"; R = f"{DATA}/output/v1.1/report_des"
plt.rcParams.update({"font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
t = np.load(f"{R}/truth_DESsx0L.npz"); z, M, c, dm = (np.asarray(t[k], float) for k in ("z", "M_halo", "concentration", "delta_m"))
T = json.load(open(f"{R}/DESsx_summary_v11.json"))["DESsx0L"]["truth"]
c0, a, b, g = T["c0_true"], T["alpha_true"], T["beta_true"], T["gamma_true_ols"]
print(f"N={len(z)} truth c0={c0:.2f} alpha={a:.3f} beta={b:.3f} gamma={g:.3f}; dm median {np.median(dm):.2f} 16-84 {np.percentile(dm,16):.2f}-{np.percentile(dm,84):.2f} 99% {np.percentile(dm,99):.2f}")
h = 0.6766
def dm14(zz, mm):
    return 0.520 + 0.385 * np.exp(-0.617 * zz ** 1.21) + (-0.101 + 0.026 * zz) * np.log10(mm * h / 1e12)
cat = Table.read(f"{DATA}/real_data/wazp_real_data.fits"); d = np.asarray(cat["DELTA_M_I"], float); d = d[np.isfinite(d)]; real = d[d > 0]
print(f"real N={len(real)} (<=0: {np.sum(d<=0)}) median {np.median(real):.2f} 16-84 {np.percentile(real,16):.2f}-{np.percentile(real,84):.2f} 99% {np.percentile(real,99):.2f}")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
ax = axs[0]; bins = np.linspace(0, 3.2, 65)
for lab, v, col in (("simulated (DES-like)", dm, "#2a78d6"), (r"WaZP+DES-Y3, $\Delta m>0$", real, "k")):
    q16, me, q84 = np.percentile(v, [16, 50, 84])
    ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=col, label=f"{lab}\nmedian {me:.2f}, 16–84% {q16:.2f}–{q84:.2f}")
    ax.axvline(me, color=col, ls=":", lw=1.2)
ax.set_xlabel(r"$\Delta m$ [mag]"); ax.set_ylabel(r"probability density [mag$^{-1}$]"); ax.set_ylim(0, 1.45); ax.legend(fontsize=7.5, loc="upper right")
ii = np.random.default_rng(0).choice(len(z), min(len(z), 6000), replace=False)
jj = ii[np.argsort(dm[ii])]  # large gaps drawn on top
vlo, vhi = np.percentile(dm, [2, 98])
sc0 = axs[1].scatter(np.log10(M[jj]), c[jj], c=dm[jj], s=4, cmap="coolwarm", vmin=vlo, vmax=vhi, alpha=0.75, lw=0)
mm = np.linspace(14, 15, 50); axs[1].plot(mm, 10 ** dm14(0.4, 10 ** mm), "--", color="k", lw=1.8, label=r"Dutton & Macciò (2014), $z=0.4$")
axs[1].set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$"); axs[1].set_ylabel(r"$c_{200}$"); axs[1].set_yscale("log"); axs[1].legend(fontsize=8, loc="lower left")
cb0 = fig.colorbar(sc0, ax=axs[1], extend="both"); cb0.set_label(r"$\Delta m$ [mag]")
sc = axs[2].scatter(dm[ii], c[ii], c=np.log10(M[ii]), s=4, cmap="viridis", alpha=0.6, lw=0)
x = np.linspace(0, dm.max(), 100); lm = np.median(np.log10(M))
axs[2].plot(x, 10 ** (np.log10(c0) + a * (lm - 14) + b * np.log10((1 + np.median(z)) / 1.35) + g * x), "--", color="C3", lw=1.8,
            label=rf"injected relation at median mass ($10^{{{lm:.2f}}}\,M_\odot$)")
axs[2].set_xlabel(r"$\Delta m$ [mag]"); axs[2].set_ylabel(r"$c_{200}$"); axs[2].set_yscale("log"); axs[2].legend(fontsize=8, loc="lower right")
cb = fig.colorbar(sc, ax=axs[2]); cb.set_label(r"$\log_{10}(M_{200}/M_\odot)$")
from matplotlib.ticker import ScalarFormatter, NullFormatter
axs[0].tick_params(top=True, right=True)
for ax in axs[1:]:
    ax.tick_params(top=True, right=True, which="both"); ax.set_yticks([3, 4, 6, 8, 10]); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
for ax, l in zip(axs, "abc"): ax.text(0.03, 0.96, f"({l})", transform=ax.transAxes, va="top", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/example_pop_sim.pdf", bbox_inches="tight"); fig.savefig(f"{R}/figs/example_pop_sim.png", dpi=70); plt.close(fig)
