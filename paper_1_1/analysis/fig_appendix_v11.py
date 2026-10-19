"""Paper 1.1 copy (2026-10-11): map_vs_median (v1.1 fits) and paired_covariance_test (linear-regressor projection)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, glob, os, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
from destacking.model import _nfw_components
from destacking.relations import c_DuttonMaccio
O = f"{DATA}/output"; K = f"{O}/harness_2026-09-29"; FIG = f"{FIGS}"; V = f"{O}/v1.1"
A, B, C, MUT, GR = "#2a78d6", "#eb6834", "#2e9e6b", "#6b6b66", "#e6e6e3"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUT, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False, "axes.labelcolor": "#1f1f1e"})
def clean(ax, grid=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis="y", color=GR, lw=0.8); ax.set_axisbelow(True)
def logx(ax):
    ax.set_xscale("log"); ax.set_xticks([0.2, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "4"]); ax.xaxis.set_minor_formatter(mt.NullFormatter())
# 1. MAP vs median -------------------------------------------------------------------------------------------------
R = json.load(open(f"{V}/results/map_pjhpd_real_v11.json"))
P = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
LAB = {"c0": r"$c_0$", "alpha_cm": r"$\alpha_{\rm cm}$", "beta_cm": r"$\beta_{\rm cm}$", "gamma_cm": r"$\gamma_{\rm cm}$", "log_M_norm": r"$\log_{10}(M_0/M_\odot)$", "alpha_mr": r"$\alpha_{M\lambda}$", "beta_mr": r"$\beta_{M\lambda}$", "f_mis": r"$f_{\rm mis}$", "sigma_off": r"$\sigma_{\rm off}$ [Mpc]"}
fig, axs = plt.subplots(3, 3, figsize=(10.5, 7.4))
for ax, p in zip(axs.flat, P):
    off = 14.489 if p == "log_M_norm" else 0
    for i, (k, col) in enumerate((("sh", A), ("v11e", B), ("stacked", C))):
        q = R[k][p]; y = 2 - i
        ax.plot([q["p16"] + off, q["p84"] + off], [y + .12] * 2, "-", color=col, lw=5, alpha=.35, solid_capstyle="butt"); ax.plot(q["median"] + off, y + .12, "o", color=col, ms=6.5, mec="white")
        ax.plot([q["pj_lo"] + off, q["pj_hi"] + off], [y - .12] * 2, "-", color=col, lw=2, solid_capstyle="butt"); ax.plot(q["map_med"] + off, y - .12, "*", color=col, ms=10, mec="white")
    ax.set_yticks([]); ax.set_xlabel(LAB[p]); clean(ax, False)
h = [plt.Line2D([], [], color=c, lw=5, alpha=.5, label=l) for l, c in (("joint, bootstrap cov.", A), ("joint, analytic cov.", B), ("stacked", C))]
h += [plt.Line2D([], [], color=MUT, marker="o", ls="-", lw=5, alpha=.5, label="median, marginal 68%"), plt.Line2D([], [], color=MUT, marker="*", ls="-", lw=2, ms=9, label="MAP, PJ-HPD 68%")]
fig.legend(handles=h, loc="upper center", ncol=3, fontsize=9); fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(f"{FIG}/map_vs_median.pdf", bbox_inches="tight"); plt.close(fig)
# 2. bin occupancy -------------------------------------------------------------------------------------------------
# 4. paired e/-e covariance test -----------------------------------------------------------------------------------
r = np.load(f"{K}/results/des_z0.3_c3.npz"); Rr = r["R"]; m = r["model"]; n = int(r["N"]); NB = 20
def jk(fun):
    full = fun(np.ones(n, bool)); reps = []
    for b in np.array_split(np.arange(n), NB):
        mk = np.ones(n, bool); mk[b] = False; reps.append(fun(mk))
    reps = np.array(reps); return full, np.sqrt((NB - 1) / NB * np.sum((reps - reps.mean(0)) ** 2, 0))
def wm(copy, s, kind):
    d = r[f"{copy}_dv"]; vb = r[f"{copy}_var"]; va = r[f"{copy}_var_an"]; ok = np.isfinite(d) & np.isfinite(va) & (vb > 0)
    w = {"equal": np.ones_like(vb), "boot": 1 / np.where(ok, vb, 1), "analytic": 1 / np.where(ok, va, 1)}[kind]
    ww = np.where(ok[s], w[s], 0.0); dd = np.where(ok[s], d[s], 0.0); return (ww * dd).sum(0) / ww.sum(0)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for kind, col, lab in (("boot", B, "inverse-variance, bootstrap variance"), ("analytic", A, "inverse-variance, analytic variance"), ("equal", C, "equal weights")):
    v, e = jk(lambda s: (wm("exact", s, kind) + wm("exact_flip", s, kind)) / 2 / m - 1)
    axs[0].errorbar(Rr * (1 + 0.04 * {"boot": -1, "analytic": 0, "equal": 1}[kind]), 100 * v, 100 * e, fmt="o-", color=col, capsize=2, label=lab)
axs[0].axhline(0, color=MUT, lw=1); logx(axs[0]); axs[0].set_xlabel("R [Mpc]"); axs[0].set_ylabel(r"mean estimate / truth $-\,1$ [per cent]"); axs[0].legend(fontsize=8.5, loc="upper right"); clean(axs[0])
axs[0].text(.30, .62, r"paired $e$/$-e$ copies, $N=6000$ identical haloes" "\n" r"$\log_{10}M=14.5$, $c=3$, $z=0.3$ (DES-like)", transform=axs[0].transAxes, fontsize=8.5, color=MUT)
dj = json.load(open(f"{V}/results/des_clean_projection_linear.json")); par = dj["params"]
for i, p in enumerate(("gamma_cm", "c0", "beta_cm")):
    j = par.index(p); sc = np.std([x[j] for x in dj["analytic"]], ddof=1)
    for t_, (tag, col) in enumerate((("bootstrap", B), ("analytic", A))):
        v = np.array([x[j] for x in dj[tag]])
        axs[1].errorbar(i + 0.18 * (t_ - .5), v.mean() / sc, v.std(ddof=1) / np.sqrt(len(v)) / sc, fmt="o", color=col, ms=7, capsize=3, label=("bootstrap covariance" if i == 0 and t_ == 0 else "analytic covariance" if i == 0 else None))
axs[1].axhline(0, color=MUT, lw=1); axs[1].set_xticks([0, 1, 2]); axs[1].set_xticklabels([r"$\gamma_{\rm cm}$", r"$c_0$", r"$\beta_{\rm cm}$"])
axs[1].set_ylabel("mean projected shift of the fit\n(units of the analytic-covariance scatter)"); axs[1].legend(fontsize=9); clean(axs[1])
pass
fig.tight_layout(); fig.savefig(f"{FIG}/paired_covariance_test.pdf"); plt.close(fig)
print("paired done")
for k in ("boot", "analytic", "equal"):
    v, e = jk(lambda s: (wm("exact", s, k) + wm("exact_flip", s, k)) / 2 / m - 1); print(k, np.round(100 * v, 2), np.round(100 * e, 2))
for tag in ("bootstrap", "analytic"):
    for p in ("gamma_cm", "c0", "beta_cm"):
        j = par.index(p); v = np.array([x[j] for x in dj[tag]]); print(tag, p, round(v.mean(), 4), "+-", round(v.std(ddof=1) / np.sqrt(len(v)), 4), " scatter", round(v.std(ddof=1), 4))
