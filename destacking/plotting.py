"""Plotting for gamma-scan results (moved from scripts/gamma_scan.py).

Backend-agnostic: the caller is responsible for selecting a matplotlib
backend (e.g. ``matplotlib.use("Agg")``) before importing this module.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_gamma_scan_results(scan_results, run_dir, n_clusters, n_bootstrap,
                            bin_size=10):
    """Two-panel: top = true vs estimated; bottom = residuals.  Each panel
    overlays the median in bins of `bin_size` consecutive γ_true points
    (sorted), a 68% credible band from the within-bin 16th/84th
    percentiles, and a linear (γ_est vs γ_true) least-squares fit so the
    overall slope-1 and offset-0 deviations are quantified."""
    gamma_true = np.array([r["gamma_true"]          for r in scan_results])
    gamma_est  = np.array([r["gamma_ensemble_med"]   for r in scan_results])
    sigma_boot = np.array([r["sigma_gamma"]           for r in scan_results])
    gamma_ols  = np.array([r["gamma_true_ols"]        for r in scan_results])

    # Sort by γ_true so the binning is consecutive in γ.
    order      = np.argsort(gamma_true)
    g_t        = gamma_true[order]
    g_e        = gamma_est[order]
    s_b        = sigma_boot[order]
    g_ols_sort = gamma_ols[order]
    residuals  = g_e - g_t

    bias_rms   = float(np.sqrt(np.mean(residuals**2)))
    mean_sigma = float(np.mean(s_b))

    # Linear fit γ_est = (1 + slope) γ_true + offset  →  residual = slope γ + offset
    A = np.column_stack([g_t, np.ones_like(g_t)])
    coeff, *_ = np.linalg.lstsq(A, g_e, rcond=None)
    fit_slope, fit_offset = float(coeff[0]) - 1.0, float(coeff[1])
    # Error bars on slope / offset from the residual scatter
    cov = np.linalg.inv(A.T @ A) * float(np.var(g_e - A @ coeff, ddof=2))
    fit_slope_err  = float(np.sqrt(cov[0, 0]))
    fit_offset_err = float(np.sqrt(cov[1, 1]))

    # Binned median + 68% band in bins of `bin_size` consecutive points
    n = len(g_t)
    bin_edges = np.arange(0, n + bin_size, bin_size)
    bin_edges = bin_edges[bin_edges <= n]
    if bin_edges[-1] != n:
        bin_edges = np.append(bin_edges, n)
    bin_centres_x = []
    bin_med_est   = []
    bin_lo_est    = []
    bin_hi_est    = []
    bin_med_res   = []
    bin_lo_res    = []
    bin_hi_res    = []
    for s, e in zip(bin_edges[:-1], bin_edges[1:]):
        if e - s < 1:
            continue
        bin_centres_x.append(float(np.median(g_t[s:e])))
        bin_med_est.append(float(np.median(g_e[s:e])))
        bin_lo_est.append(float(np.percentile(g_e[s:e], 16)))
        bin_hi_est.append(float(np.percentile(g_e[s:e], 84)))
        bin_med_res.append(float(np.median(residuals[s:e])))
        bin_lo_res.append(float(np.percentile(residuals[s:e], 16)))
        bin_hi_res.append(float(np.percentile(residuals[s:e], 84)))
    bin_centres_x = np.asarray(bin_centres_x)
    bin_med_est   = np.asarray(bin_med_est)
    bin_lo_est    = np.asarray(bin_lo_est)
    bin_hi_est    = np.asarray(bin_hi_est)
    bin_med_res   = np.asarray(bin_med_res)
    bin_lo_res    = np.asarray(bin_lo_res)
    bin_hi_res    = np.asarray(bin_hi_res)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 10),
        gridspec_kw={"height_ratios": [3, 2]}, sharex=True,
    )

    # ── Top: recovery ─────────────────────────────────────────────────────────
    g_lim = 1.3
    g_grid = np.linspace(-g_lim, g_lim, 200)
    ax_top.plot([-g_lim, g_lim], [-g_lim, g_lim], "k--", lw=1.2,
                label="Perfect recovery", zorder=0)
    ax_top.errorbar(
        g_t, g_e, yerr=s_b,
        fmt="o", color="steelblue", capsize=4, markersize=5, lw=1.0,
        alpha=0.35, zorder=3,
        label=r"individual $\hat\gamma$ ± $\sigma_{\rm boot}$",
    )
    ax_top.fill_between(bin_centres_x, bin_lo_est, bin_hi_est,
                        color="purple", alpha=0.18, lw=0,
                        label=f"68% CL (bins of {bin_size})", zorder=4)
    ax_top.plot(bin_centres_x, bin_med_est, "o-", color="purple", lw=1.6,
                markersize=6, zorder=5,
                label=f"median estimate (bins of {bin_size})")
    ax_top.plot(g_grid, (1.0 + fit_slope) * g_grid + fit_offset,
                color="crimson", lw=2.0, zorder=6,
                label=(rf"fit: slope−1 = {fit_slope:+.3f} ± {fit_slope_err:.3f}, "
                       rf"offset = {fit_offset:+.3f} ± {fit_offset_err:.3f}"))
    ax_top.set_ylabel(r"Estimated $\hat\gamma$", fontsize=12)
    ax_top.set_title(
        rf"Gamma scan — true vs estimated"
        rf"  ($N={n_clusters}$, {n_bootstrap} bootstrap draws/γ)",
        fontsize=11,
    )
    ax_top.legend(fontsize=9, loc="lower right")
    ax_top.grid(alpha=0.3)
    ax_top.set_xlim(-g_lim, g_lim)
    ax_top.set_ylim(-2.0, 2.0)
    ax_top.text(
        0.03, 0.97,
        rf"RMS bias = {bias_rms:.4f}"
        "\n"
        rf"$\langle\sigma_{{\rm boot}}\rangle = {mean_sigma:.4f}$",
        transform=ax_top.transAxes, fontsize=11, va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="grey"),
    )

    # ── Bottom: residuals ─────────────────────────────────────────────────────
    ax_bot.axhline(0.0, color="k", lw=1.5, ls="--", label="Unbiased")
    ax_bot.errorbar(
        g_t, residuals, yerr=s_b,
        fmt="o", color="steelblue", capsize=4, markersize=5, lw=1.0,
        alpha=0.35,
        label=r"residual ± $\sigma_{\rm boot}$",
    )
    ax_bot.axhspan(-mean_sigma, mean_sigma, alpha=0.10, color="tomato",
                   label=rf"$\langle\sigma_{{\rm boot}}\rangle = {mean_sigma:.4f}$")
    ax_bot.fill_between(bin_centres_x, bin_lo_res, bin_hi_res,
                        color="purple", alpha=0.18, lw=0,
                        label=f"68% CL (bins of {bin_size})")
    ax_bot.plot(bin_centres_x, bin_med_res, "o-", color="purple", lw=1.6,
                markersize=6, label=f"median residual (bins of {bin_size})")
    ax_bot.plot(g_grid, fit_slope * g_grid + fit_offset,
                color="crimson", lw=2.0,
                label=(rf"fit: slope = {fit_slope:+.3f} ± {fit_slope_err:.3f}, "
                       rf"offset = {fit_offset:+.3f} ± {fit_offset_err:.3f}"))
    ax_bot.set_xlabel(r"True $\gamma$", fontsize=12)
    ax_bot.set_ylabel(r"$\hat\gamma_{\rm ens} - \gamma_{\rm true}$", fontsize=12)
    ax_bot.legend(fontsize=9, loc="upper left")
    ax_bot.grid(alpha=0.3)
    ax_bot.set_ylim(-1.8, 1.8)

    plt.tight_layout()
    out = os.path.join(run_dir, "gamma_scan_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def plot_posteriors_overlay(results, sim_subdir, gamma_true, gamma_prior,
                            n_clusters):
    """Overlay of all bootstrap γ posteriors for one gamma point, with the
    combined (mixture) posterior, the prior, and the truth line.  All curves
    are peak-normalised so the broad combined mixture stays visible next to
    the narrow per-fit posteriors.

    Parameters
    ----------
    results : list of dict
        Per-bootstrap summaries; each needs a ``bootstrap`` index matching a
        ``bootstrap_NNN/gamma_posterior.npz`` file under ``sim_subdir``.
    gamma_prior : frozen scipy.stats distribution
        The prior on γ (drawn for reference).
    n_clusters : int
        Sample size, reported in the title.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x_grid  = np.linspace(-2.5, 2.5, 800)
    cmap    = plt.cm.viridis
    n       = len(results)
    pool_s, pool_w = [], []
    for k, r in enumerate(results):
        npz_path = os.path.join(
            sim_subdir, f"bootstrap_{r['bootstrap']:03d}", "gamma_posterior.npz"
        )
        if not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path)
        samples = data["gamma_samples"]
        weights = data["weights"]
        pool_s.append(samples)
        pool_w.append(weights / weights.sum())
        try:
            kde = gaussian_kde(samples, weights=weights, bw_method=0.15)
            y = kde(x_grid)
            # Peak-normalise: narrow per-fit posteriors would otherwise dwarf
            # the broad combined mixture on a shared density scale.
            ax.plot(x_grid, y / y.max(),
                    color=cmap(k / max(n - 1, 1)), alpha=0.55, lw=1.0)
        except Exception:
            pass
    # Combined (mixture) posterior over all bootstraps — the measurement
    if pool_s:
        try:
            pool_s_c = np.concatenate(pool_s)
            pool_w_c = np.concatenate(pool_w)
            pool_w_c = pool_w_c / pool_w_c.sum()
            kde_comb = gaussian_kde(pool_s_c, weights=pool_w_c, bw_method=0.15)
            y = kde_comb(x_grid)
            ax.plot(x_grid, y / y.max(),
                    color="black", lw=3.0, label="Combined posterior")
        except Exception:
            pass
    y_prior = gamma_prior.pdf(x_grid)
    ax.plot(x_grid, y_prior / y_prior.max(),
            color="darkorange", lw=1.8, ls="--", label="Prior")
    ax.axvline(gamma_true, color="crimson", lw=2.0, ls="--",
               label=rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$")
    ax.set(xlabel=r"$\gamma$", ylabel="Posterior density (peak-normalised)",
           title=(rf"Bootstrap posteriors  "
                  rf"($\gamma_{{\rm true}}={gamma_true:+.1f}$, $N={n_clusters}$)"))
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(sim_subdir, "posteriors_overlay.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
