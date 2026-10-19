"""Paper 1.1 copy (2026-10-11): sim figures + table numbers from the v1.1 campaign (option-C gaps, gamma=k/36, linear Delta m).
x-axis = injected truth (OLS gamma of the realisation); offsets are fit minus that truth."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
K = os.path.expanduser(f"{DATA}/output/harness_2026-09-29"); FIG = f"{FIGS}"; R11 = f"{DATA}/output/v1.1/report_des"
J, S = "#2a78d6", "#eb6834"; MUTED, GRID = "#6b6b66", "#e6e6e3"
plt.rcParams.update({"font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
LAB = {"c0": r"$c_0$", "alpha_cm": r"$\alpha_{\rm cm}$", "beta_cm": r"$\beta_{\rm cm}$", "gamma_cm": r"$\gamma_{\rm cm}$",
       "log_M_norm": r"$\log_{10}M_0$", "alpha_mr": r"$\alpha_{M\lambda}$", "beta_mr": r"$\beta_{M\lambda}$",
       "f_mis": r"$f_{\rm mis}$", "sigma_off": r"$\sigma_{\rm off}$"}
P = list(LAB)
# LSST: the 48-fit scan of the paper (the 60-row file adds the 2026-10-11 slope-trace seeds)
data = {s: json.load(open(f"{R11}/{f}")) for s, f in (("LSST-like", "lsst_scan_rows_v11_48.json"), ("DES-like", "des_scan_rows_v11.json"))}
# Fig 5: gamma scans
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, (sv, rows) in zip(axs, data.items()):
    for meth, col, mk, dx in (("joint", J, "o", -0.0015), ("stacked", S, "s", 0.0015)):
        rr = [r for r in rows if r["method"] == meth]; ks = sorted({r["k"] for r in rr})
        for r in rr:
            ax.plot(r["k"] / 36 + dx, r["d"]["gamma_cm"][0], mk, ms=3, color=col, alpha=0.25, mec="none")
        grp = [[r["d"]["gamma_cm"][0] for r in rr if r["k"] == k] for k in ks]
        ax.errorbar(np.array(ks) / 36 + dx, [np.mean(g) for g in grp], [np.std(g, ddof=1) / np.sqrt(len(g)) for g in grp],
                    fmt=mk, ms=7, color=col, mec="white", mew=1, lw=1.8, capsize=0,
                    label=f"{meth} ({len(rr)} fits; mean $\\sigma(\\gamma_{{\\rm cm}})={np.mean([r['d']['gamma_cm'][1] for r in rr]):.3f}$)")
    ax.axhline(0, color=MUTED, lw=1); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel(r"injected $\gamma_{\rm cm}$"); ax.set_ylabel(r"recovered $-$ injected $\gamma_{\rm cm}$")
    ax.text(0.03, 0.95, sv, transform=ax.transAxes, va="top", fontsize=11, fontweight="bold"); ax.legend(fontsize=8.5, loc="lower left")
    ax.tick_params(top=True, right=True)
fig.tight_layout(); fig.savefig(f"{FIG}/gamma_scan_des_lsst.pdf", bbox_inches="tight"); fig.savefig(f"{R11}/figs/gscan_paper.png", dpi=60)
# joint vs stacked bias, both surveys
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
y = np.arange(len(P))[::-1]
for ax, (sv, rows) in zip(axs, data.items()):
    for meth, col, mk, dy in (("joint", J, "o", 0.14), ("stacked", S, "s", -0.14)):
        rr = [r for r in rows if r["method"] == meth]; b, e = [], []
        for p in P:
            d = np.array([r["d"][p][0] for r in rr]); s = np.mean([r["d"][p][1] for r in rr]); b.append(d.mean() / s); e.append(d.std(ddof=1) / np.sqrt(len(d)) / s)
        ax.errorbar(b, y + dy, xerr=e, fmt=mk, ms=6, color=col, mec="white", mew=1, lw=1.8, capsize=0, label=f"{meth} ({len(rr)} fits)")
    ax.axvline(0, color=MUTED, lw=1); ax.set_axisbelow(True)
    ax.set_xscale("symlog", linthresh=0.5, linscale=1.0)   # linear within |x|<0.5, logarithmic beyond (2026-10-12)
    for xv in (-0.5, 0.5): ax.axvline(xv, color=MUTED, lw=0.8, ls=":")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    lim = (-20, 2) if sv == "LSST-like" else (-4, 2)
    tk = [t for t in (-10, -3, -1, -0.5, 0, 0.5, 1) if lim[0] < t < lim[1]]
    ax.set_xlim(*lim); ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="x", color=GRID, lw=0.8, which="major")
    ax.set_yticks(y); ax.set_yticklabels([LAB[p] for p in P]); ax.set_xlabel(r"mean bias / mean posterior $\sigma$ (linear within $\pm0.5$)")
    ax.text(0.03, 0.97, sv, transform=ax.transAxes, va="top", fontsize=11, fontweight="bold"); ax.legend(fontsize=8.5, loc="lower left")
    ax.tick_params(top=True, right=True)
fig.tight_layout(); fig.savefig(f"{FIG}/sim_joint_vs_stacked_bias.pdf", bbox_inches="tight"); fig.savefig(f"{R11}/figs/jvsb_paper.png", dpi=60)
# numbers for the tables
out = {}
for sv, rows in data.items():
    for meth in ("joint", "stacked"):
        rr = [r for r in rows if r["method"] == meth]
        xs = np.array([r["gamma_true"] for r in rr]); ys = xs + np.array([r["d"]["gamma_cm"][0] for r in rr])
        A = np.vstack([xs, np.ones_like(xs)]).T; c, *_ = np.linalg.lstsq(A, ys, rcond=None); res = ys - A @ c
        cov = np.linalg.inv(A.T @ A) * res.var(ddof=2)
        out[(sv, meth)] = {"n": len(rr), "slope": (c[0], np.sqrt(cov[0, 0])), "offset": (c[1], np.sqrt(cov[1, 1]))}
        for p in P:
            d = np.array([r["d"][p][0] for r in rr]); s = np.array([r["d"][p][1] for r in rr])
            out[(sv, meth)][p] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d.std(ddof=1), s.mean())
for sv in data:
    print(f"\n{sv}")
    for meth in ("joint", "stacked"):
        o = out[(sv, meth)]
        print(f"  {meth}: N={o['n']} slope {o['slope'][0]:.3f}+-{o['slope'][1]:.3f} offset {o['offset'][0]:+.3f}+-{o['offset'][1]:.3f}")
        for p in P:
            m, se, sc, sg = o[p]; print(f"    {p:11s} {m:+.4f} {se:.4f} {sc:.4f} {sg:.4f}")
    print("  width ratio stacked/joint (mean sigma): " + " ".join(f"{p}={out[(sv,'stacked')][p][3]/out[(sv,'joint')][p][3]:.2f}" for p in P))

for sv, rows in data.items():
    for meth in ("joint", "stacked"):
        rr = [r for r in rows if r["method"] == meth]
        for k in sorted({r["k"] for r in rr}):
            g = np.array([r["d"]["gamma_cm"][0] for r in rr if r["k"] == k]); print(f"  {sv} {meth} k={k}: n={len(g)} mean {g.mean():+.4f} se {g.std(ddof=1)/np.sqrt(len(g)):.4f}  ({g.mean()/(g.std(ddof=1)/np.sqrt(len(g))):+.1f} se)")
