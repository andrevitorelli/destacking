"""Paper figure (2026-10-05; 2026-10-07: joint fit as smoothed distribution of per-bootstrap medians, referee MC2): (gamma_cm, delta_Mlambda) posteriors and gamma_cm marginals for the original model (delta = 0) and with
delta free, for the stacked fit (all bins; bins above 0.25 Mpc) and the joint fit (bins from 0.25 Mpc)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
P = f"{DATA}/output/v1.1/posteriors"; OUT = f"{FIGS}/gamma_delta_mr_posteriors.pdf"
JOINT, STACK, STACK2, MUT = "#2a78d6", "#eb6834", "#8a3a12", "#6b6b66"
plt.rcParams.update({"font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False, "axes.edgecolor": MUT})
def load(files):
    X, W = [], []
    for f in files:
        z = np.load(f, allow_pickle=True); X.append(z["samples"]); W.append(z["weights"] / z["weights"].sum()); o = [str(s) for s in z["param_order"]]
    X = np.concatenate(X); W = np.concatenate(W); return X, W / W.sum(), o
def loadv9c():
    X, W = [], []
    for b in sorted(glob.glob(f"{DATA}/output/spill/real_joint_v9c/run/simulation_*/bootstrap_*/gamma_posterior.npz")):
        z = np.load(b, allow_pickle=True); X.append(z["all_samples"]); W.append(z["weights"] / z["weights"].sum()); o = [str(t) for t in z["param_order"]]
    X = np.concatenate(X); W = np.concatenate(W); return X, W / W.sum(), o
S0 = load([f"{P}/stack.npz"]); S1 = load([f"{P}/stack_dmr.npz"]); S2 = load([f"{P}/stack_dmr_rmin0.25.npz"])
import sys; from bootmed import boot_medians
from fig_bootmed import smoothed
def medsmooth(files, key):
    names, M = boot_medians(files, key); X = smoothed(M); return X, np.ones(len(X)) / len(X), names
J0 = medsmooth(sorted(glob.glob(f"{DATA}/output/v1.1/posteriors/joint_sh_b[0-9][0-9][0-9].npz")), "samples")
J1 = medsmooth(sorted(glob.glob(f"{P}/joint_sh_dmr_b[0-9][0-9][0-9].npz")), "samples")
for lab, D in (("J0", J0), ("J1", J1)):
    names, M = boot_medians(sorted(glob.glob(f"{DATA}/output/v1.1/posteriors/joint_sh_b[0-9][0-9][0-9].npz")), "samples") if lab == "J0" else boot_medians(sorted(glob.glob(f"{P}/joint_sh_dmr_b[0-9][0-9][0-9].npz")), "samples")
    for p in ("gamma_cm", "delta_mr"):
        if p in names: print(lab, p, "n_boot", len(M), np.round(np.percentile(M[:, names.index(p)], [16, 50, 84]), 4))
def contour(ax, D, col, lab, ls="-"):
    X, W, o = D; x, y = X[:, o.index("gamma_cm")], X[:, o.index("delta_mr")]; sub = np.random.default_rng(1).choice(len(x), 20000, p=W)
    k = gaussian_kde(np.vstack([x[sub], y[sub]])); gx = np.linspace(-0.25, 0.25, 130); gy = np.linspace(-0.06, 0.16, 130); GX, GY = np.meshgrid(gx, gy)
    Z = k(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape); zs = np.sort(Z.ravel())[::-1]; cs = np.cumsum(zs) / zs.sum(); lv = [zs[np.searchsorted(cs, q)] for q in (0.95, 0.68)]
    ax.contourf(GX, GY, Z, levels=[lv[0], lv[1], Z.max()], colors=[col, col], alpha=0.18); ax.contour(GX, GY, Z, levels=lv, colors=[col], linewidths=1.2, linestyles=ls)
    ax.plot([], [], color=col, ls=ls, label=lab)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axs[0]
contour(ax, S1, STACK, "stacked, all bins"); contour(ax, S2, STACK2, r"stacked, bins above 0.25 Mpc", ls="--"); contour(ax, J1, JOINT, "joint (bins from 0.25 Mpc)")
ax.axvline(0, color=MUT, lw=0.8); ax.axhline(0, color=MUT, lw=0.8); ax.set_xlabel(r"$\gamma_{\rm cm}$"); ax.set_ylabel(r"$\delta_{M\lambda}$"); ax.legend(fontsize=8.5, loc="upper right")
ax.text(0.03, 0.04, r"$\delta_{M\lambda}$ free", transform=ax.transAxes, fontsize=9, color=MUT)
ax = axs[1]; g = np.linspace(-0.25, 0.3, 450)
for D, col, ls, lab in ((S0, STACK, "-", r"stacked, $\delta_{M\lambda}=0$ (original)"), (S1, STACK, "--", r"stacked, $\delta_{M\lambda}$ free"),
                        (S2, STACK2, ":", r"stacked, $\delta_{M\lambda}$ free, bins above 0.25 Mpc"),
                        (J0, JOINT, "-", r"joint, $\delta_{M\lambda}=0$ (original)"), (J1, JOINT, "--", r"joint, $\delta_{M\lambda}$ free")):
    X, W, o = D; x = X[:, o.index("gamma_cm")]; sub = np.random.default_rng(2).choice(len(x), 30000, p=W); ax.plot(g, gaussian_kde(x[sub])(g), ls=ls, color=col, lw=1.8, label=lab)
ax.axvline(0, color=MUT, lw=0.8); ax.set_xlabel(r"$\gamma_{\rm cm}$"); ax.set_yticks([]); ax.set_ylim(bottom=0); ax.legend(fontsize=8, loc="upper left")
for a in axs:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight"); fig.savefig(f"{DATA}/output/v1.1/results/dmr_paper.png", dpi=75, bbox_inches="tight")
X, W, o = S2
for p in ("gamma_cm", "delta_mr"):
    j = o.index(p); s = np.argsort(X[:, j]); c = np.cumsum(W[s]); print(p, np.round(np.interp([.16, .5, .84], c, X[s, j]), 3))
print("P(gamma<0)", round(float(W[X[:, o.index('gamma_cm')] < 0].sum()), 3))
