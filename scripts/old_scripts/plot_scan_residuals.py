"""
Generate per-simulation residual + posterior-overlay plots for a gamma_scan run.

For each simulation_g{k}_gamma{v} directory, produces:
  gamma_residuals.png      — per-bootstrap (gamma_med - gamma_true) with 68% CI error bars
  posteriors_overlay.png   — overlay of all bootstrap posterior KDEs

Usage:
  python scripts/plot_scan_residuals.py [RUN_DIR]

RUN_DIR defaults to the most-recently-modified gamma_scan_outputs/run_* directory.
"""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde, norm as sp_norm

PRIORS_GAMMA = sp_norm(loc=0.0, scale=3.5)


def find_latest_run():
    base = "gamma_scan_outputs"
    runs = sorted(
        (d for d in os.listdir(base) if d.startswith("run_")),
        key=lambda d: os.path.getmtime(os.path.join(base, d)),
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base}/")
    return os.path.join(base, runs[0])


def load_simulation(sim_dir):
    """Return (gamma_true, list_of_bootstrap_results).

    gamma_true is read from the per-bootstrap summary.json (exact float)
    rather than parsed from the directory name (rounded to 1 decimal).
    """
    results = []
    for name in sorted(os.listdir(sim_dir)):
        if not name.startswith("bootstrap_"):
            continue
        summary_path = os.path.join(sim_dir, name, "summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            results.append(json.load(f))

    if not results:
        return None, []
    gamma_true = float(results[0]["gamma_true"])
    return gamma_true, results


def plot_residuals(gamma_true, results, sim_dir):
    if not results:
        return
    seq_idx = np.arange(1, len(results) + 1)
    gm  = np.array([r["gamma_med"] for r in results])
    glo = np.array([r["gamma_lo"]  for r in results])
    ghi = np.array([r["gamma_hi"]  for r in results])
    yerr = np.vstack([gm - glo, ghi - gm])
    residuals  = gm - gamma_true
    mean_res   = float(np.mean(residuals))
    std_res    = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    mean_sigma = float(np.mean((ghi - glo) / 2.0))

    n_clusters = results[0].get("n_clusters", "?")
    n_resample = results[0].get("n_resample", n_clusters)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(seq_idx, residuals, yerr=yerr, fmt="o",
                color="steelblue", capsize=4, markersize=6,
                label=r"Posterior median $\pm$68% CI")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="Unbiased")
    ax.axhspan(mean_res - std_res, mean_res + std_res,
               alpha=0.10, color="tomato",
               label=rf"$\sigma_{{\rm boot}} = {std_res:.4f}$")
    ax.axhline(mean_res, color="tomato", lw=1.5, ls=":",
               label=rf"Mean residual = {mean_res:+.4f}")
    ax.set(
        xlabel="Bootstrap draw #",
        ylabel=r"$\hat\gamma - \gamma_{\rm true}$",
        title=(
            rf"$\gamma$ residuals — post-hoc bootstrap  "
            rf"($\gamma_{{\rm true}}={gamma_true:+.1f}$, "
            rf"$N={n_clusters}$, $n_{{\rm resample}}={n_resample}$)"
        ),
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    if mean_sigma > 0:
        ratio = std_res / mean_sigma
        ax.text(
            0.02, 0.02,
            rf"$\langle\sigma_\gamma^{{\rm post}}\rangle = {mean_sigma:.4f}$"
            "\n"
            rf"$\sigma_{{\rm boot}}/\langle\sigma_\gamma^{{\rm post}}\rangle = {ratio:.2f}$",
            transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"),
        )

    plt.tight_layout()
    out = os.path.join(sim_dir, "gamma_residuals.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_posteriors(gamma_true, results, sim_dir):
    if not results:
        return

    # Adaptive x-range centred on gamma_true
    half_width = max(1.5, abs(gamma_true) + 1.0)
    x_lo = gamma_true - half_width
    x_hi = gamma_true + half_width
    x_grid = np.linspace(x_lo, x_hi, 600)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis
    n    = len(results)

    for k, r in enumerate(results):
        npz_path = os.path.join(sim_dir, f"bootstrap_{r['bootstrap']:03d}", "gamma_posterior.npz")
        if not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path)
        samples = data["gamma_samples"]
        weights = data["weights"]
        try:
            kde = gaussian_kde(samples, weights=weights, bw_method=0.15)
            y   = kde(x_grid)
        except Exception:
            y, edges = np.histogram(samples, weights=weights, bins=40,
                                    range=(x_lo, x_hi), density=True)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), y,
                    color=cmap(k / max(n - 1, 1)), alpha=0.6, lw=1.0)
            continue
        ax.plot(x_grid, y, color=cmap(k / max(n - 1, 1)), alpha=0.6, lw=1.0)

    ax.plot(x_grid, PRIORS_GAMMA.pdf(x_grid),
            color="darkorange", lw=1.8, ls="--", label="Prior")
    ax.axvline(gamma_true, color="crimson", lw=2.0, ls="--",
               label=rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$")

    n_clusters = results[0].get("n_clusters", "?")
    ax.set(
        xlabel=r"$\gamma$",
        ylabel="Posterior density",
        title=(
            rf"$\gamma$ posteriors — post-hoc bootstrap  "
            rf"($\gamma_{{\rm true}}={gamma_true:+.1f}$, $N={n_clusters}$)"
        ),
        xlim=(x_lo, x_hi),
    )
    handles_extra = [Line2D([0], [0], color=cmap(0.5), lw=1.2,
                             label=rf"{n} posterior densities (KDE)")]
    ax.legend(handles=[*handles_extra, *ax.get_legend_handles_labels()[0]],
              loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(sim_dir, "posteriors_overlay.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Run directory: {run_dir}")

    sim_dirs = sorted(
        os.path.join(run_dir, d)
        for d in os.listdir(run_dir)
        if d.startswith("simulation_g")
    )

    for sim_dir in sim_dirs:
        gamma_true, results = load_simulation(sim_dir)
        if gamma_true is None or not results:
            print(f"  Skipping {os.path.basename(sim_dir)} (no results)")
            continue
        print(f"\n{os.path.basename(sim_dir)}  gamma_true={gamma_true:+.1f}  n_boots={len(results)}")
        plot_residuals(gamma_true, results, sim_dir)
        plot_posteriors(gamma_true, results, sim_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
