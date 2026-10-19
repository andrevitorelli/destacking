"""Paper 1.1 (2026-10-12): posterior width of gamma_cm per simulated fit, DES-like vs LSST-like, joint vs stacked
(fig sim_gamma_width.pdf), plus the numbers of the gamma-recovery table.  Same 48 LSST + 84 DES scan fits as the paper."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R11 = f"{DATA}/output/v1.1/report_des"; FIG = f"{FIGS}"
J, S, MUTED, GRID = "#2a78d6", "#eb6834", "#6b6b66", "#e6e6e3"
plt.rcParams.update({"font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
data = {"LSST-like": json.load(open(f"{R11}/lsst_scan_rows_v11_48.json")), "DES-like": json.load(open(f"{R11}/des_scan_rows_v11.json"))}
fig, ax = plt.subplots(figsize=(6.4, 3.4))
rows, ylab, y = {}, [], 0
for sv in ("LSST-like", "DES-like"):
    for meth, col in (("joint", J), ("stacked", S)):
        s = np.array([r["d"]["gamma_cm"][1] for r in data[sv] if r["method"] == meth]); rows[(sv, meth)] = s
        jit = np.random.default_rng(int(10 * y)).uniform(-0.18, 0.18, len(s))
        ax.plot(s, y + jit, "o", ms=3.5, color=col, alpha=0.45, mec="none")
        ax.plot(np.median(s), y, "|", ms=18, mew=2.5, color="k")
        ylab.append(f"{sv}, {meth}"); y += 1
    y += 0.5
yt = [0, 1, 2.5, 3.5]
ax.set_yticks(yt); ax.set_yticklabels(ylab); ax.invert_yaxis(); ax.set_xscale("log")
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
ax.xaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.05, 0.1])); ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.set_xlim(0.005, 0.1)
ax.set_xlabel(r"posterior 68% half-width of $\gamma_{\rm cm}$ per fit [mag$^{-1}$]")
ax.grid(axis="x", color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True); ax.tick_params(top=True, which="both")
fig.tight_layout(); fig.savefig(f"{FIG}/sim_gamma_width.pdf", bbox_inches="tight"); fig.savefig(f"{R11}/figs/sim_gamma_width.png", dpi=90); plt.close(fig)
for k, s in rows.items(): print(k, f"median {np.median(s):.4f} mean {s.mean():.4f} N={len(s)}")
for meth in ("joint", "stacked"): print(meth, "DES/LSST median ratio", round(np.median(rows[("DES-like", meth)]) / np.median(rows[("LSST-like", meth)]), 2))
