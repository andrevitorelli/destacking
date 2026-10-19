"""Paper 1.1 (2026-10-16): nz_sim_vs_real_combined.pdf from the v1.1 scan realisations.
Left: cluster redshifts of the 84 DES-like and 48 LSST-like scan truth catalogues (report_des/nz_truth_v11.npz, collected on
the HPC cluster from {TL}_sim/simulation_g00*/truth_catalog.npz) against the 15,825 real WaZP clusters with Dm>0.  Right: source n(z),
Smail-Schechter of each configuration (truncated at 2.5) against the DES Y3 metacal DNF redshifts of real_data/metacal_filtered.npz.
Replaces the 2026-09-24 figure, made before the (1+z) mass-richness fix (f38a111), whose simulated n(z) showed a spurious low-z deficit."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.table import Table
FIG = f"{FIGS}"; R = f"{DATA}/output/v1.1/report_des"
RED, DES, LSST = "#d62728", "#1f77b4", "#ff7f0e"
plt.rcParams.update({"font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
sim = np.load(f"{R}/nz_truth_v11.npz")
cat = Table.read(f"{DATA}/real_data/wazp_real_data.fits")
zc, dm = np.asarray(cat["ZPHOT"], float), np.asarray(cat["DELTA_M_I"], float)
zc = zc[np.isfinite(dm) & (dm > 0)]
zs = np.load(f"{DATA}/real_data/metacal_filtered.npz", mmap_mode="r")["z"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
b = np.linspace(0.05, 0.72, 46)
axs[0].hist(zc, b, density=True, histtype="stepfilled", color=RED, alpha=0.25)
axs[0].hist(zc, b, density=True, histtype="step", color=RED, lw=1.8, label=f"real WaZP+DES-Y3 ({len(zc):,})")
axs[0].hist(sim["DES"], b, density=True, histtype="step", color=DES, lw=1.6, label=f"DES-like simulated (84 realisations)")
axs[0].hist(sim["LSST"], b, density=True, histtype="step", color=LSST, lw=1.6, label=f"LSST-like simulated (48 realisations)")
axs[0].set_xlabel("cluster redshift $z$"); axs[0].set_ylabel("$n(z)$ [normalised]"); axs[0].legend(fontsize=8.5, loc="upper left")
bs = np.linspace(0, 2.5, 251); h, _ = np.histogram(np.asarray(zs), bs, density=True); c = 0.5 * (bs[1:] + bs[:-1])
axs[1].fill_between(c, h, step="mid", color=RED, alpha=0.25, lw=0); axs[1].step(c, h, where="mid", color=RED, lw=1.8, label="real DES Y3 metacal")
x = np.linspace(0, 2.5, 500)
for (a, be, z0), col, lab in (((2.0, 1.5, 0.40), DES, "DES-like simulated"), ((2.0, 0.68, 0.11), LSST, "LSST-like simulated")):
    p = (x / z0) ** a * np.exp(-(x / z0) ** be); p /= np.trapezoid(p, x); axs[1].plot(x, p, color=col, lw=1.6, label=lab)
axs[1].set_xlabel("source redshift $z$"); axs[1].set_ylabel("$n(z)$ [normalised]"); axs[1].set_xlim(0, 2.5); axs[1].set_ylim(0); axs[1].legend(fontsize=8.5)
for ax in axs: ax.tick_params(top=True, right=True)
fig.tight_layout(); fig.savefig(f"{FIG}/nz_sim_vs_real_combined.pdf", bbox_inches="tight"); fig.savefig(f"{R}/figs/nz_sim_vs_real_combined.png", dpi=80)
print("real clusters", len(zc), "sources", len(zs))
