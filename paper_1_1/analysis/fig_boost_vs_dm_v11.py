"""Paper 1.1 (2026-10-17): boost_vs_dm.pdf, the boost excess B-1 inside 0.5 Mpc of the low, middle and high magnitude-gap tercile of each
redshift-richness cell of the real f=1.2 stacks (stacked_full_param_fit_f1.2/stacks3.pkl).  Same calculation as section 7 of
referee_2026-10-07/fig_report_1007.py; also prints the inverse-variance mean high-minus-low difference quoted in the paper."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A, B, C, MUT = "#2a78d6", "#eb6834", "#2e9e6b", "#6b6b66"
plt.rcParams.update({"font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
S3 = pickle.load(open(f"{DATA}/real_data/stacked_full_param_fit_f1.2/stacks3.pkl", "rb"))
Rb = np.logspace(-0.8, 0.6, 16); inner = Rb < 0.5
fig, ax = plt.subplots(figsize=(6.2, 3.4)); i = 0; diff, err = [], []
for zb in range(3):
    for nb in range(4):
        b, e = [], []
        for d in range(3):
            s = S3.get(f"z{zb}_ng{nb}_dm{d}")
            if s is None or s["n_stacked"] < 50: b = None; break
            bb = np.asarray(s["boosts"]) - 1; w = 1 / np.diag(np.asarray(s["boosts_cov"]))[inner]
            b.append(np.sum(w * bb[inner]) / w.sum()); e.append(1 / np.sqrt(w.sum()))
        if b:
            ax.errorbar(i + np.array([-0.2, 0, 0.2]), b, e, fmt="-o", ms=3.5, color=[A, C, B][zb], capsize=1.5,
                        label=f"redshift bin {zb + 1}" if nb == 0 else None)
            diff.append(b[2] - b[0]); err.append(np.hypot(e[2], e[0])); i += 1
ax.set_xticks(range(i)); ax.set_xticklabels([])
ax.set_xlabel(r"redshift-richness cell (low, middle, high $\Delta m$ tercile)"); ax.set_ylabel(r"$B-1$ inside $0.5$ Mpc")
ax.legend(fontsize=8.5); ax.tick_params(top=True, right=True)
fig.tight_layout(); fig.savefig(f"{FIGS}/boost_vs_dm.pdf", bbox_inches="tight")
fig.savefig(f"{DATA}/output/v1.1/report_des/figs/boost_vs_dm.png", dpi=90)
d, w = np.array(diff), 1 / np.array(err) ** 2
print(f"cells {len(d)}; high-minus-low B-1 inside 0.5 Mpc: {np.sum(w * d) / w.sum():+.4f} +- {1 / np.sqrt(w.sum()):.4f}")
