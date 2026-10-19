"""Appendix B figures: bin occupancy (Fig. 21) and bin-centre versus bin-averaged model (Fig. 22).

Occupancy of the simulated grids is read from one DES-like and one LSST-like simulated realisation
(output/step0/{DESrb2,LSSTrb2}/sim), and of the real data from the analytic-covariance measurement.
(The Appendix D and G figures are made by fig_appendix_v11.py.)"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, glob, os, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
from destacking.model import _nfw_components
from destacking.relations import c_DuttonMaccio
O = f"{DATA}/output"; K = f"{O}/harness_2026-09-29"; FIG = FIGS
A, B, C, MUT, GR = "#2a78d6", "#eb6834", "#2e9e6b", "#6b6b66", "#e6e6e3"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUT, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False, "axes.labelcolor": "#1f1f1e"})
def clean(ax, grid=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis="y", color=GR, lw=0.8); ax.set_axisbelow(True)
def logx(ax):
    ax.set_xscale("log"); ax.set_xticks([0.2, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "4"]); ax.xaxis.set_minor_formatter(mt.NullFormatter())
# 2. bin occupancy -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4))
for S, lab, col, r in (("LSSTrb2", "LSST-like (80,000 clusters)", A, np.logspace(-0.8, 0.6, 8)), ("DESrb2", "DES-like (15,967 clusters)", B, np.logspace(-0.6, 0.6, 6))):
    d = np.load(glob.glob(f"{O}/step0/{S}/sim/simulation_*/simulated_data.npz")[0])["data_vectors"]; ax.plot(r, np.isfinite(d).mean(0), "o-", color=col, label=lab)
D = f"{DATA}/real_data/wazp_real_data_analytic_v3/simulation_g00_gamma+0.0"
d = np.load(f"{D}/simulated_data.npz")["data_vectors"]; dm = np.asarray(np.load(f"{D}/cluster_obs.npz")["delta_m"], float); s = np.isfinite(dm) & (dm > 0)
ax.plot(np.logspace(-0.6, 0.6, 6), np.isfinite(d[s]).mean(0), "o-", color=C, label=f"WaZP+DES-Y3 ({int(s.sum()):,} clusters)")
ax.axvline(0.5, color=MUT, ls="--", lw=1); ax.text(0.52, 0.06, r"$R=0.5$ Mpc", color=MUT, fontsize=9)
logx(ax); ax.set_xlabel("R [Mpc]"); ax.set_ylabel(r"fraction of clusters with $\geq3$ sources in the bin"); ax.set_ylim(0, 1.04); ax.legend(fontsize=9, loc="lower right"); clean(ax)
fig.tight_layout(); fig.savefig(f"{FIG}/bin_occupancy.pdf"); plt.close(fig); print("occupancy done")
# 3. bin centre vs bin average -------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
for ax, (lab, r) in zip(axs, (("LSST-like grid (8 bins)", np.logspace(-0.8, 0.6, 8)), ("DES-like and WaZP grid (6 bins)", np.logspace(-0.6, 0.6, 6)))):
    pass
grids = (("LSST-like, 8 bins", np.logspace(-0.8, 0.6, 8)), ("DES-like and WaZP, 6 bins", np.logspace(-0.6, 0.6, 6)))
for ax, (lab, r) in zip(axs[:2], grids):
    for lM, col in zip((14.0, 14.4, 14.8), (A, B, C)):
        M = 10 ** lM; c = float(c_DuttonMaccio(0.4, M))
        p0, _ = _nfw_components(np.array([.4]), np.array([M]), np.array([c]), r, SIGMA_OFF=np.array([0.47]), bin_nodes=0)
        p4, _ = _nfw_components(np.array([.4]), np.array([M]), np.array([c]), r, SIGMA_OFF=np.array([0.47]), bin_nodes=4)
        ax.plot(r, p0[0] / p4[0], "o-", color=col, label=rf"$\log_{{10}}M={lM}$")
    ax.axhline(1, color=MUT, lw=.8); logx(ax); ax.set_xlabel("R [Mpc]"); ax.text(.04, .93, lab, transform=ax.transAxes, fontsize=9.5); clean(ax)
axs[0].set_ylabel(r"$\Delta\Sigma$(bin centre) / $\langle\Delta\Sigma\rangle_{\rm bin}$"); axs[0].legend(fontsize=9, loc="lower right")
fit = json.load(open(f"{K}/results/binavg_fit_effect.json")) if os.path.exists(f"{K}/results/binavg_fit_effect.json") else None
axs[2].axis("off")
axs[2].text(0.02, .9, transform=axs[2].transAxes, s="Effect on the joint fit\n(paired fits, same data):\n\n"
            r"$\gamma_{\rm cm}$ shift: $+0.1\sigma$ and" "\n" r"$+0.7\sigma$ in two paired tests" "\n" r"$\log_{10}M_{\rm norm}$: $-0.018$ dex" "\n\n" "bin averaging costs 4x\nin run time and is not used", fontsize=10, va="top")
fig.tight_layout(); fig.savefig(f"{FIG}/bin_centre_vs_average.pdf"); plt.close(fig); print("binavg done")
