"""
Combined-posterior plot for a gamma_repeat_resampled run.

For each simulation, all bootstrap posteriors are pooled (equal weight per
draw) into a single combined posterior KDE shown in one panel.

Produces:
  repeat_combined_posteriors.png   — grid of combined posteriors, one per sim

Usage:
  python scripts/plot_repeat_combined_posteriors.py [RUN_DIR]

RUN_DIR defaults to the most-recently-modified
gamma_repeat_resampled_outputs/run_* directory.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm as sp_norm

PRIOR_GAMMA = sp_norm(loc=0.0, scale=3.5)


def find_latest_run():
    base = "gamma_repeat_resampled_outputs"
    runs = sorted(
        (d for d in os.listdir(base) if d.startswith("run_")),
        key=lambda d: os.path.getmtime(os.path.join(base, d)),
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run directories in {base}/")
    return os.path.join(base, runs[0])


def load_simulation(sim_dir):
    """Return (sim_idx, gamma_true, combined_samples, combined_weights, n_boot)."""
    sim_idx = int(os.path.basename(sim_dir).split("_")[-1])

    # Read gamma_true from the first available summary
    gamma_true = None
    all_samples, all_weights = [], []

    for name in sorted(os.listdir(sim_dir)):
        if not name.startswith("bootstrap_"):
            continue
        sj  = os.path.join(sim_dir, name, "summary.json")
        nz  = os.path.join(sim_dir, name, "gamma_posterior.npz")
        if not (os.path.exists(sj) and os.path.exists(nz)):
            continue
        if gamma_true is None:
            with open(sj) as f:
                gamma_true = json.load(f).get("gamma_true", 0.2)
        data = np.load(nz)
        w = data["weights"].astype(float)
        all_samples.append(data["gamma_samples"])
        all_weights.append(w / w.sum())   # normalise per draw

    if not all_samples:
        return sim_idx, None, None, None, 0

    n_boot = len(all_samples)
    comb_s = np.concatenate(all_samples)
    comb_w = np.concatenate([w / n_boot for w in all_weights])
    return sim_idx, gamma_true, comb_s, comb_w, n_boot


def weighted_quantile(values, weights, q):
    sorter = np.argsort(values)
    cw = np.cumsum(weights[sorter]) / weights.sum()
    return float(np.interp(q, cw, values[sorter]))


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Run directory: {run_dir}")

    sim_dirs = sorted(
        os.path.join(run_dir, d)
        for d in os.listdir(run_dir)
        if d.startswith("simulation_")
    )

    sims = []
    for sd in sim_dirs:
        sim_idx, gamma_true, comb_s, comb_w, n_boot = load_simulation(sd)
        if gamma_true is None:
            print(f"  simulation_{sim_idx:03d}: no posteriors, skipping")
            continue
        print(f"  simulation_{sim_idx:03d}: {n_boot} bootstraps, gamma_true={gamma_true}")
        sims.append((sim_idx, gamma_true, comb_s, comb_w, n_boot))

    if not sims:
        print("No data found.")
        return

    n     = len(sims)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    gamma_true_global = sims[0][1]   # same for all sims
    x = np.linspace(-1.5, 1.5, 600)

    for idx, (sim_idx, gamma_true, comb_s, comb_w, n_boot) in enumerate(sims):
        ax = axes[idx // ncols][idx % ncols]

        # combined posterior KDE
        try:
            kde    = gaussian_kde(comb_s, weights=comb_w, bw_method=0.12)
            y_post = kde(x)
        except Exception:
            y_post, edges = np.histogram(comb_s, weights=comb_w, bins=60,
                                         range=(x[0], x[-1]), density=True)
            ax.fill_between(0.5*(edges[1:]+edges[:-1]), y_post,
                            alpha=0.35, color="steelblue")
            ax.plot(0.5*(edges[1:]+edges[:-1]), y_post, color="steelblue", lw=1.5)
        else:
            ax.fill_between(x, y_post, alpha=0.35, color="steelblue",
                            label="Combined posterior")
            ax.plot(x, y_post, color="steelblue", lw=1.8)

        # prior
        ax.plot(x, PRIOR_GAMMA.pdf(x), color="darkorange",
                lw=1.5, ls="--", label="Prior")

        # true gamma
        ax.axvline(gamma_true, color="crimson", lw=2.0, ls="--",
                   label=rf"$\gamma_{{\rm true}}={gamma_true}$")

        # combined median + 68% CI
        med = weighted_quantile(comb_s, comb_w, 0.50)
        lo  = weighted_quantile(comb_s, comb_w, 0.16)
        hi  = weighted_quantile(comb_s, comb_w, 0.84)
        ax.axvline(med, color="steelblue", lw=1.5, ls=":",
                   label=rf"$\hat\gamma={med:+.3f}$")
        ax.axvspan(lo, hi, alpha=0.12, color="steelblue")

        ax.set_title(f"Simulation {sim_idx}  ({n_boot} bootstraps)", fontsize=10)
        ax.set_xlabel(r"$\gamma$", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(x[0], x[-1])
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)

    for idx in range(len(sims), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    n_clusters = 5000  # default; read from first summary if needed
    fig.suptitle(
        rf"Gamma repeat — combined posteriors ($\gamma_{{\rm true}}={gamma_true_global}$)"
        "\n(bootstrap draws pooled with equal weight per draw)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(run_dir, "repeat_combined_posteriors.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
