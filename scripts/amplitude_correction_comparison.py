"""
Compare the 51-point amplitude_correction scan to the 201-point reference
(no-A) scan, and show where A and log_M_norm landed.

Outputs (under gamma_scan_outputs/run_2026-05-27T09-41-05/):
  comparison_transfer_function.png  — γ̂ vs γ_true, both runs overlaid +
                                      linear fits.
  A_lmn_posteriors.png              — A and log_M_norm posteriors vs γ_true
                                      for the new run.
"""

from __future__ import annotations
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REF_RUN  = "gamma_scan_outputs/run_2026-05-26T18-27-50"
NEW_RUN  = "gamma_scan_outputs/run_2026-05-27T09-41-05"
OUT_DIR  = NEW_RUN


def load_summary(run_dir):
    with open(os.path.join(run_dir, "scan_summary.json")) as f:
        s = json.load(f)
    s.sort(key=lambda r: r["gamma_true"])
    gt = np.array([r["gamma_true"]         for r in s])
    gh = np.array([r["gamma_ensemble_med"] for r in s])
    sb = np.array([r["sigma_boot"]         for r in s])
    return gt, gh, sb


def weighted_linfit(x, y, yerr):
    sb_safe = np.where(yerr > 1e-6, yerr, np.nanmedian(yerr))
    w = 1.0 / sb_safe ** 2
    (a, b), cov = np.polyfit(x, y, deg=1, w=w, cov=True)
    return float(a), float(b), float(np.sqrt(cov[0,0])), float(np.sqrt(cov[1,1])), cov


def main():
    gt_r, gh_r, sb_r = load_summary(REF_RUN)
    gt_n, gh_n, sb_n = load_summary(NEW_RUN)

    a_r, b_r, sa_r, sb_r_int, _ = weighted_linfit(gt_r, gh_r, sb_r)
    a_n, b_n, sa_n, sb_n_int, _ = weighted_linfit(gt_n, gh_n, sb_n)

    print(f"Reference (201pt, no A):  γ̂ = ({a_r:.3f}±{sa_r:.3f})γ + ({b_r:+.3f}±{sb_r_int:.3f})")
    print(f"With-A (51pt):           γ̂ = ({a_n:.3f}±{sa_n:.3f})γ + ({b_n:+.3f}±{sb_n_int:.3f})")

    # ── plot 1: transfer functions overlaid ────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 10),
        gridspec_kw={"height_ratios": [3, 2]}, sharex=True,
    )
    g_lim = 1.3
    ax_top.plot([-g_lim, g_lim], [-g_lim, g_lim], "k--", lw=1.2,
                label=r"$\hat\gamma=\gamma_{\rm true}$", zorder=1)
    ax_top.errorbar(gt_r, gh_r, yerr=sb_r if hasattr(sb_r, '__iter__') else None,
                    fmt="o", color="steelblue", capsize=2.5, ms=3.5, alpha=0.45,
                    lw=0.8, label="reference (201pt, no A)", zorder=2)
    ax_top.errorbar(gt_n, gh_n, yerr=sb_n, fmt="s", color="crimson", capsize=3,
                    ms=5, alpha=0.85, lw=1.0,
                    label="with A∼N(0.80, 0.02)  (51pt)", zorder=3)
    # linear fits
    x = np.linspace(-g_lim, g_lim, 400)
    ax_top.plot(x, a_r*x + b_r, "--", color="steelblue", lw=2.0,
                label=rf"ref fit: slope={a_r:.3f}, offset={b_r:+.3f}")
    ax_top.plot(x, a_n*x + b_n, "-", color="crimson", lw=2.2,
                label=rf"+A fit: slope={a_n:.3f}, offset={b_n:+.3f}")
    ax_top.set_ylabel(r"$\hat\gamma$", fontsize=12)
    ax_top.set_xlim(-g_lim, g_lim)
    ax_top.set_ylim(-2, 2)
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.grid(alpha=0.3)
    ax_top.set_title("Transfer function: with vs without amplitude correction",
                     fontsize=11)

    # residuals
    ax_bot.axhline(0.0, color="k", lw=1.2, ls="--")
    ax_bot.plot(gt_r, gh_r - gt_r, "o", color="steelblue", ms=3.5, alpha=0.45,
                label="ref residual")
    ax_bot.plot(gt_n, gh_n - gt_n, "s", color="crimson", ms=5, alpha=0.85,
                label="+A residual")
    ax_bot.plot(x, (a_r-1)*x + b_r, "--", color="steelblue", lw=2.0,
                label=f"ref slope-1 = {a_r-1:+.3f}")
    ax_bot.plot(x, (a_n-1)*x + b_n, "-", color="crimson", lw=2.0,
                label=f"+A slope-1 = {a_n-1:+.3f}")
    ax_bot.set_xlabel(r"$\gamma_{\rm true}$", fontsize=12)
    ax_bot.set_ylabel(r"$\hat\gamma - \gamma_{\rm true}$", fontsize=12)
    ax_bot.set_ylim(-1.5, 1.5)
    ax_bot.legend(fontsize=9, loc="best")
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "comparison_transfer_function.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── plot 2: A and log_M_norm posteriors vs γ_true ──────────────────────
    with open(os.path.join(NEW_RUN, "A_lmn_posteriors.json")) as f:
        Adata = json.load(f)
    Adata.sort(key=lambda r: r["gamma_true"])
    g    = np.array([r["gamma_true"]  for r in Adata])
    A    = np.array([r["A_med"]      for r in Adata])
    Asig = np.array([(r["A_hi"]-r["A_lo"])/2 for r in Adata])
    lmn  = np.array([r["lmn_med"]    for r in Adata])
    lsig = np.array([(r["lmn_hi"]-r["lmn_lo"])/2 for r in Adata])

    # Reference run lmn for comparison
    with open(os.path.join(REF_RUN, "lmn_posteriors.json")) as f:
        Ldata = json.load(f)
    Ldata.sort(key=lambda r: r["gamma_true"])
    g_ref   = np.array([r["gamma_true"] for r in Ldata])
    lmn_ref = np.array([r["lmn_med"]    for r in Ldata])
    lsig_ref= np.array([(r["lmn_hi"]-r["lmn_lo"])/2 for r in Ldata])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)

    # A panel
    ax1.errorbar(g, A, yerr=Asig, fmt="s", color="crimson", capsize=3, ms=5,
                 label=r"$A$ posterior (with-A run)")
    ax1.axhspan(0.80 - 0.02, 0.80 + 0.02, alpha=0.18, color="grey",
                label=r"prior $A\sim N(0.80, 0.02)$")
    ax1.axhline(0.80, color="grey", ls="--", lw=1.0)
    ax1.axhline(1.00, color="k", ls=":", lw=1.0, label="A=1 (no correction)")
    ax1.set_ylabel(r"amplitude $A$", fontsize=12)
    ax1.set_title(r"$A$ and $\log M_{\rm norm}$ posteriors vs $\gamma_{\rm true}$",
                  fontsize=11)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(alpha=0.3)

    # log_M_norm panel
    ax2.errorbar(g_ref, lmn_ref, yerr=lsig_ref, fmt="o", color="steelblue",
                 capsize=2.5, ms=3.5, alpha=0.4,
                 label=r"ref (no A) $\log M_{\rm norm}$")
    ax2.errorbar(g, lmn, yerr=lsig, fmt="s", color="crimson", capsize=3, ms=5,
                 label=r"with-A $\log M_{\rm norm}$")
    ax2.axhspan(-0.10, +0.10, alpha=0.18, color="grey",
                label=r"prior $\log M_{\rm norm}\sim N(0, 0.10)$")
    ax2.axhline(0.0, color="grey", ls="--", lw=1.0)
    ax2.set_xlabel(r"$\gamma_{\rm true}$", fontsize=12)
    ax2.set_ylabel(r"$\log M_{\rm norm}$ [dex]", fontsize=12)
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "A_lmn_posteriors.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── numerical summary ──────────────────────────────────────────────────
    print()
    print("Summary table:")
    print(f"  reference (201pt):      slope={a_r:.3f}±{sa_r:.3f}  offset={b_r:+.3f}±{sb_r_int:.3f}")
    print(f"                          log_M_norm median over 201 sims = {np.median(lmn_ref):+.4f}")
    print(f"  with A (51pt):          slope={a_n:.3f}±{sa_n:.3f}  offset={b_n:+.3f}±{sb_n_int:.3f}")
    print(f"                          A median over 51 sims          = {np.median(A):.4f}")
    print(f"                          log_M_norm median over 51 sims  = {np.median(lmn):+.4f}")

    # Bias in the physical regime
    for name, x, y in [("ref", gt_r, gh_r), ("+A", gt_n, gh_n)]:
        mask = (x >= 0) & (x <= 0.5)
        res = y[mask] - x[mask]
        print(f"\n  Physical regime γ∈[0,0.5] for {name}  (N={mask.sum()}):")
        print(f"    mean residual = {res.mean():+.4f}")
        print(f"    RMS bias      = {np.sqrt((res**2).mean()):.4f}")


if __name__ == "__main__":
    main()
