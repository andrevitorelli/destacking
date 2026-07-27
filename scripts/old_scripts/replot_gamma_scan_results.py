"""
Re-render gamma_scan_results.png from an existing run's scan_summary.json with
two added overlays:

  (1) Weighted linear fit  γ̂ = a · γ_true + b  with ±1σ error region.
      The slope a quantifies the overall calibration: a < 1 means the
      estimator is compressed toward zero; b ≠ 0 is an additive offset.

  (2) A running median in bins of 10 consecutive points with the 68 % CL
      band shown around it.

Usage:
    python scripts/replot_gamma_scan_results.py [RUN_DIR]

RUN_DIR defaults to the most recently modified gamma_scan_outputs/run_*.
"""

from __future__ import annotations
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_latest_run():
    base = "gamma_scan_outputs"
    runs = sorted((d for d in os.listdir(base) if d.startswith("run_")),
                  key=lambda d: os.path.getmtime(os.path.join(base, d)),
                  reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run directories in {base}/")
    return os.path.join(base, runs[0])


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Run: {run_dir}")

    with open(os.path.join(run_dir, "scan_summary.json")) as f:
        scan = json.load(f)

    # Sort by γ_true so binning and fit operate on the natural ordering
    scan.sort(key=lambda r: r["gamma_true"])

    gt = np.array([r["gamma_true"]          for r in scan])
    gh = np.array([r["gamma_ensemble_med"]  for r in scan])
    sb = np.array([r["sigma_boot"]          for r in scan])
    g_ols = np.array([r.get("gamma_true_ols", r["gamma_true"]) for r in scan])

    n_clusters  = scan[0].get("n_clusters",  "?")
    n_bootstrap = scan[0].get("n_bootstrap", "?")

    residuals  = gh - gt
    bias_rms   = float(np.sqrt(np.mean(residuals ** 2)))
    mean_sigma = float(np.mean(sb))

    # ── (1) Weighted linear fit  γ̂ = a · γ_true + b ────────────────────────
    # Weights = 1/σ² with a tiny floor so σ=0 outliers don't dominate.
    sb_safe = np.where(sb > 1e-6, sb, np.nanmedian(sb))
    w = 1.0 / sb_safe ** 2
    (a, b), cov = np.polyfit(gt, gh, deg=1, w=w, cov=True)
    sigma_a = float(np.sqrt(cov[0, 0]))
    sigma_b = float(np.sqrt(cov[1, 1]))
    cov_ab  = float(cov[0, 1])

    x_grid = np.linspace(gt.min() - 0.05, gt.max() + 0.05, 400)
    y_fit  = a * x_grid + b
    # Variance of a linear combination:
    #   Var(a·x + b) = x² Var(a) + 2x Cov(a,b) + Var(b)
    y_var  = x_grid ** 2 * sigma_a ** 2 + 2 * x_grid * cov_ab + sigma_b ** 2
    y_sig  = np.sqrt(y_var)

    # ── (2) Running median in bins of 10 ────────────────────────────────────
    BIN_SIZE = 10
    bin_x, bin_med, bin_p16, bin_p84 = [], [], [], []
    for i in range(0, len(gt), BIN_SIZE):
        gt_b = gt[i:i + BIN_SIZE]
        gh_b = gh[i:i + BIN_SIZE]
        if len(gh_b) < 2:
            continue
        bin_x.append(float(np.median(gt_b)))
        bin_med.append(float(np.median(gh_b)))
        bin_p16.append(float(np.percentile(gh_b, 16)))
        bin_p84.append(float(np.percentile(gh_b, 84)))
    bin_x   = np.array(bin_x)
    bin_med = np.array(bin_med)
    bin_p16 = np.array(bin_p16)
    bin_p84 = np.array(bin_p84)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 11),
        gridspec_kw={"height_ratios": [3, 2]}, sharex=True,
    )

    # ----- top: γ̂ vs γ_true -----
    g_lim = max(1.3, gt.max() + 0.2)
    ax_top.plot([-g_lim, g_lim], [-g_lim, g_lim], "k--", lw=1.2,
                label="perfect recovery", zorder=1)

    # data
    ax_top.errorbar(gt, gh, yerr=sb, fmt="o", color="steelblue",
                    capsize=2.5, markersize=3.5, lw=0.9, alpha=0.55,
                    zorder=2, label=r"$\hat\gamma\pm\sigma$")
    # OLS truth (γ that the simulator imprints)
    ax_top.scatter(gt, g_ols, marker="x", s=18, color="darkorange",
                   alpha=0.35, zorder=3,
                   label=r"OLS truth $\gamma_{\rm OLS}$")

    # linear fit + ±1σ region
    ax_top.fill_between(x_grid, y_fit - y_sig, y_fit + y_sig,
                        color="crimson", alpha=0.20, zorder=4,
                        label=r"linear fit $\pm 1\sigma$")
    fit_lbl = (rf"fit: $\hat\gamma = ({a:.3f}\pm{sigma_a:.3f})\,\gamma_{{\rm true}} "
               rf"+ ({b:+.3f}\pm{sigma_b:.3f})$")
    ax_top.plot(x_grid, y_fit, color="crimson", lw=2.0, zorder=5,
                label=fit_lbl)

    # binned median + 68% CL band
    ax_top.fill_between(bin_x, bin_p16, bin_p84,
                        color="purple", alpha=0.20, zorder=4,
                        label="68% CL (bins of 10)")
    ax_top.plot(bin_x, bin_med, "D-", color="purple", lw=2.0, ms=7,
                zorder=6, label="median (bins of 10)")

    ax_top.set_ylabel(r"Estimated $\hat\gamma$", fontsize=12)
    ax_top.set_title(
        f"Gamma scan — true vs estimated "
        f"($N$={n_clusters}, {n_bootstrap} bootstrap draw(s)/γ)",
        fontsize=11,
    )
    ax_top.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_top.grid(alpha=0.3)
    ax_top.set_xlim(-g_lim, g_lim)
    ax_top.set_ylim(-2.0, 2.0)

    info_txt = (rf"RMS bias = {bias_rms:.4f}" "\n"
                rf"$\langle\sigma\rangle$ = {mean_sigma:.4f}" "\n"
                rf"slope−1 = {a-1:+.3f} ± {sigma_a:.3f}" "\n"
                rf"offset $b$ = {b:+.3f} ± {sigma_b:.3f}")
    ax_top.text(0.97, 0.04, info_txt, transform=ax_top.transAxes,
                fontsize=10, va="bottom", ha="right",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))

    # ----- bottom: residual -----
    ax_bot.axhline(0.0, color="k", lw=1.5, ls="--", label="unbiased")
    ax_bot.axhspan(-mean_sigma, mean_sigma, alpha=0.10, color="tomato",
                   label=rf"$\langle\sigma\rangle$ = {mean_sigma:.4f}")
    ax_bot.errorbar(gt, residuals, yerr=sb, fmt="o", color="steelblue",
                    capsize=2.5, markersize=3.5, lw=0.9, alpha=0.55,
                    label=r"residual $\pm\sigma$")

    # fit-derived residual = (a−1) γ_true + b, same Var as y_fit
    res_fit = (a - 1) * x_grid + b
    ax_bot.fill_between(x_grid, res_fit - y_sig, res_fit + y_sig,
                        color="crimson", alpha=0.20,
                        label=r"fit residual $\pm 1\sigma$")
    ax_bot.plot(x_grid, res_fit, color="crimson", lw=2.0,
                label=f"slope−1 = {a-1:+.3f}, offset = {b:+.3f}")

    # binned median residual + 68% CL
    ax_bot.fill_between(bin_x, bin_p16 - bin_x, bin_p84 - bin_x,
                        color="purple", alpha=0.20,
                        label="68% CL (bins of 10)")
    ax_bot.plot(bin_x, bin_med - bin_x, "D-", color="purple", lw=2.0, ms=7,
                label="median residual (bins of 10)")

    ax_bot.set_xlabel(r"True $\gamma$", fontsize=12)
    ax_bot.set_ylabel(r"$\hat\gamma - \gamma_{\rm true}$", fontsize=12)
    ax_bot.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax_bot.grid(alpha=0.3)
    ax_bot.set_ylim(-1.8, 1.8)

    plt.tight_layout()
    out = os.path.join(run_dir, "gamma_scan_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"  fit slope     = {a:.4f} ± {sigma_a:.4f}  (deviation from 1: {a-1:+.4f})")
    print(f"  fit intercept = {b:+.4f} ± {sigma_b:.4f}")
    print(f"  RMS bias = {bias_rms:.4f},  <σ> = {mean_sigma:.4f}")


if __name__ == "__main__":
    main()
