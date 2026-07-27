"""
Alternative residual + combined-posterior plots for a gamma_scan run.

Produces two figures saved to RUN_DIR:
  scan_transfer_function.png  — gamma_hat vs gamma_true (top) and bias (bottom)
  scan_combined_posteriors.png — one panel per gamma: all bootstrap posteriors
                                 pooled into a single combined posterior KDE

For each gamma_true the bootstrap posteriors are combined by giving every draw
equal weight (1/N_boot), then re-normalising.  The result is the mixture
posterior that averages over bootstrap resamplings of the cluster sample.

Usage:
  python scripts/plot_combined_posteriors.py [RUN_DIR]
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

PRIOR_GAMMA = sp_norm(loc=0.0, scale=3.5)


# ── helpers ──────────────────────────────────────────────────────────────────

def find_latest_run():
    base = "gamma_scan_outputs"
    runs = sorted(
        (d for d in os.listdir(base) if d.startswith("run_")),
        key=lambda d: os.path.getmtime(os.path.join(base, d)),
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run directories in {base}/")
    return os.path.join(base, runs[0])


def load_simulation(sim_dir):
    """Return (gamma_true, summary_list, combined_samples, combined_weights).

    gamma_true is read from the per-bootstrap summary.json (exact float),
    NOT parsed from the directory name (which is rounded to 1 decimal and
    collides for sub-0.1 grid spacings, e.g. 201-point scans).
    """
    if not os.path.isdir(sim_dir):
        return None, [], None, None

    summaries, all_samples, all_weights = [], [], []
    for name in sorted(os.listdir(sim_dir)):
        if not name.startswith("bootstrap_"):
            continue
        sj = os.path.join(sim_dir, name, "summary.json")
        nz = os.path.join(sim_dir, name, "gamma_posterior.npz")
        if not (os.path.exists(sj) and os.path.exists(nz)):
            continue
        with open(sj) as f:
            summaries.append(json.load(f))
        data = np.load(nz)
        all_samples.append(data["gamma_samples"])
        # Each draw contributes equally; normalise its weights to sum=1 then
        # divide by the number of draws so the pooled weights sum to 1.
        w = data["weights"].astype(float)
        all_weights.append(w / w.sum())          # normalise per draw

    if not summaries:
        return None, [], None, None
    gamma_true = float(summaries[0]["gamma_true"])

    n_boot = len(all_samples)
    combined_s = np.concatenate(all_samples)
    combined_w = np.concatenate([w / n_boot for w in all_weights])
    return gamma_true, summaries, combined_s, combined_w


def weighted_quantile(values, weights, q):
    sorter = np.argsort(values)
    cw = np.cumsum(weights[sorter]) / weights.sum()
    return float(np.interp(q, cw, values[sorter]))


# ── plot 1: transfer function / bias ─────────────────────────────────────────

def plot_transfer_function(sim_data, run_dir):
    """
    sim_data: list of (gamma_true, summaries, comb_s, comb_w)
    Top panel: gamma_hat vs gamma_true with σ error bars.
    Bottom panel: residual (gamma_hat - gamma_true) with σ error bars,
                  plus the mean residual and its 68% CL band.

    The per-point σ is the std of bootstrap medians when N_BOOTSTRAP > 1,
    or the single posterior's (hi − lo)/2 when it = 1 (BOOTSTRAP=0 case).
    """
    gammas_true, gammas_hat, sigmas_boot = [], [], []
    for gamma_true, summaries, _, _ in sim_data:
        if not summaries:
            continue
        gm = np.array([r["gamma_med"] for r in summaries])
        sg = np.array([r["sigma_gamma"] for r in summaries])
        gammas_true.append(gamma_true)
        gammas_hat.append(float(np.median(gm)))
        if len(gm) > 1:
            sigmas_boot.append(float(np.std(gm, ddof=1)))
        else:
            sigmas_boot.append(float(sg[0]))

    gt = np.array(gammas_true)
    gh = np.array(gammas_hat)
    sb = np.array(sigmas_boot)
    bias = gh - gt

    # Residual population stats (across γ_true grid points)
    bias_mean = float(np.mean(bias))
    bias_p16  = float(np.percentile(bias, 16))
    bias_p84  = float(np.percentile(bias, 84))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # top: transfer function
    ax = axes[0]
    ax.errorbar(gt, gh, yerr=sb, fmt="o", color="steelblue",
                capsize=5, markersize=7,
                label=r"$\hat\gamma_{\rm ens}\pm\sigma$")
    lo = min(gt.min(), gh.min()) - 0.15
    hi = max(gt.max(), gh.max()) + 0.15
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label=r"$\hat\gamma=\gamma_{\rm true}$")
    ax.set_ylabel(r"$\hat\gamma$ (ensemble median)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Gamma scan — estimator transfer function", fontsize=12)

    # bottom: residual
    ax2 = axes[1]
    ax2.errorbar(gt, bias, yerr=sb, fmt="o", color="steelblue",
                 capsize=5, markersize=7, zorder=3,
                 label=r"$\hat\gamma-\gamma_{\rm true}\,\pm\sigma$")
    ax2.axhline(0.0, color="k", lw=1, ls="--")
    ax2.axhspan(bias_p16, bias_p84, alpha=0.15, color="tomato",
                label=f"68% CL: [{bias_p16:+.3f}, {bias_p84:+.3f}]")
    ax2.axhline(bias_mean, color="tomato", lw=1.5, ls=":",
                label=f"mean residual = {bias_mean:+.3f}")
    ax2.set_xlabel(r"$\gamma_{\rm true}$", fontsize=12)
    ax2.set_ylabel(r"Residual $\hat\gamma-\gamma_{\rm true}$", fontsize=12)
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(run_dir, "scan_transfer_function.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(f"  mean residual = {bias_mean:+.4f}  68% CL = "
          f"[{bias_p16:+.4f}, {bias_p84:+.4f}]")


# ── plot 2: combined posteriors grid ─────────────────────────────────────────

def plot_combined_posteriors(sim_data, run_dir):
    """One subplot per gamma: prior, combined posterior KDE, true gamma line."""
    valid = [(gt, summaries, s, w)
             for gt, summaries, s, w in sim_data
             if summaries and s is not None]
    n = len(valid)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, (gamma_true, summaries, comb_s, comb_w) in enumerate(valid):
        ax = axes[idx // ncols][idx % ncols]

        # x-range centred on truth, wide enough to show the prior shoulder
        half = max(1.2, abs(gamma_true) + 0.8)
        x = np.linspace(gamma_true - half, gamma_true + half, 600)

        # combined posterior KDE
        try:
            kde = gaussian_kde(comb_s, weights=comb_w, bw_method=0.12)
            y_post = kde(x)
        except Exception:
            y_post, edges = np.histogram(comb_s, weights=comb_w, bins=60,
                                         range=(x[0], x[-1]), density=True)
            ax.fill_between(0.5 * (edges[1:] + edges[:-1]), y_post,
                            alpha=0.35, color="steelblue")
            ax.plot(0.5 * (edges[1:] + edges[:-1]), y_post,
                    color="steelblue", lw=1.5)
        else:
            ax.fill_between(x, y_post, alpha=0.35, color="steelblue",
                            label="Combined posterior")
            ax.plot(x, y_post, color="steelblue", lw=1.8,
                    label="Combined posterior")

        # prior
        ax.plot(x, PRIOR_GAMMA.pdf(x), color="darkorange",
                lw=1.5, ls="--", label="Prior")

        # true gamma
        ax.axvline(gamma_true, color="crimson", lw=2.0, ls="--",
                   label=rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$")

        # median and 68% CI from combined posterior
        med = weighted_quantile(comb_s, comb_w, 0.50)
        lo  = weighted_quantile(comb_s, comb_w, 0.16)
        hi  = weighted_quantile(comb_s, comb_w, 0.84)
        ax.axvline(med, color="steelblue", lw=1.5, ls=":",
                   label=rf"$\hat\gamma={med:+.3f}$")
        ax.axvspan(lo, hi, alpha=0.12, color="steelblue")

        n_boot = len(summaries)
        n_clust = summaries[0].get("n_clusters", "?")
        ax.set_title(
            rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$"
            rf"  ({n_boot} bootstraps, $N={n_clust}$)",
            fontsize=10,
        )
        ax.set_xlabel(r"$\gamma$", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(x[0], x[-1])
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)

    # hide unused axes
    for idx in range(len(valid), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Gamma scan — combined posteriors\n"
        "(all bootstrap draws pooled with equal weight per draw)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(run_dir, "scan_combined_posteriors.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Run directory: {run_dir}")

    sim_dirs = sorted(
        os.path.join(run_dir, d)
        for d in os.listdir(run_dir)
        if d.startswith("simulation_g")
    )

    sim_data = []
    for sim_dir in sim_dirs:
        gamma_true, summaries, comb_s, comb_w = load_simulation(sim_dir)
        if gamma_true is None:
            continue
        print(f"  {os.path.basename(sim_dir)}: {len(summaries)} bootstraps")
        sim_data.append((gamma_true, summaries, comb_s, comb_w))

    plot_transfer_function(sim_data, run_dir)
    plot_combined_posteriors(sim_data, run_dir)
    print("Done.")


if __name__ == "__main__":
    main()
