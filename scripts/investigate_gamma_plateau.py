"""
Investigate the γ̂ ≈ -0.5 plateau seen in the gamma scan.

For each saved negative-γ_true simulation in a gamma_scan run, this script:

  1. Loads the truth catalogue, cluster observables, simulated ΔΣ data
     and OLS-fit truth parameters (c0_true, α_true, β_true).
  2. Holds (c0, α, β, log_M_norm) fixed at the OLS truth and scans γ
     across a wide range.  Evaluates the EXACT same likelihood that the
     sampler uses (`log_likelihood_combined` with the same DELTA_M_REF,
     FMISS, SIGMA_OFF).
  3. For diagnostic γ values it computes the per-cluster c distribution
     and the average predicted ΔΣ profile, and compares against the
     simulated data mean.

Outputs three figures into RUN_DIR:

  plateau_loglike.png       — log-L(γ) scan for each γ_true (negatives)
  plateau_c_distribution.png — concentration histogram at γ ∈ {γ_true, −0.5, 0}
  plateau_mean_profile.png  — mean ΔΣ at γ ∈ {γ_true, −0.5, 0} vs data mean

Usage:
  python scripts/investigate_gamma_plateau.py [RUN_DIR]

This is a read-only diagnostic — it never writes into the existing run
directory's simulation subfolders, only into a new ``plateau/`` subdir.
"""

from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from destacking.fitting import log_likelihood_combined
from destacking.model import forward_model_combined, _MC_M_PIVOT_LOG, _MC_Z_REF

# ── Constants matching gamma_scan.py ────────────────────────────────────────
DELTA_M_REF = 0.1
RADII       = np.logspace(-0.8, 0.8, 12)
R_MIN_FIT   = 0.5

from astropy.cosmology import Planck18 as _planck18
F_CEN           = 0.64
SIGMA_OFF_MPC_H = 0.32
H_PLANCK        = float(_planck18.h)
SIGMA_OFF_MPC   = SIGMA_OFF_MPC_H / H_PLANCK


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


def parse_gamma(sim_dir):
    m = re.search(r"gamma([+-][0-9.]+)$", os.path.basename(sim_dir))
    return float(m.group(1)) if m else None


def load_sim(sim_dir):
    """Load truth catalogue, observables, data, covariance and OLS truth."""
    co  = np.load(os.path.join(sim_dir, "cluster_obs.npz"))
    obs = Table({k: co[k] for k in co.files})
    raw = np.load(os.path.join(sim_dir, "simulated_data.npz"))
    dv  = raw["data_vectors"]
    cv  = raw["cov_matrices"]
    with open(os.path.join(sim_dir, "ols.json")) as f:
        ols = json.load(f)
    tc  = np.load(os.path.join(sim_dir, "truth_catalog.npz"))
    truth = Table({k: tc[k] for k in tc.files})
    return obs, dv, cv, ols, truth


def loglike_scan(obs, dv_fit, cv_fit, R_fit, ols, gamma_grid, fmiss, sig_off):
    """Evaluate the full combined log-likelihood on a γ grid, with all
    other params fixed at the OLS truth values from the simulation."""
    out = np.empty_like(gamma_grid)
    for k, g in enumerate(gamma_grid):
        out[k] = log_likelihood_combined(
            c0=ols["c0_true"], alpha=ols["alpha_true"], gamma=float(g),
            cluster_table=obs, data_vectors=dv_fit, cov_matrices=cv_fit,
            radii=R_fit,
            beta=ols["beta_true"],
            M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
            delta_m_ref=DELTA_M_REF,
            log_M_norm=0.0,
            FMISS=fmiss, SIGMA_OFF=sig_off, BCG_B_MASS=None,
        )
    return out


def model_c_array(obs, ols, gamma):
    """Replicate forward_model_combined's logC formula -> per-cluster c."""
    z      = np.asarray(obs["z"])
    Lambda = np.asarray(obs["Lambda"])
    dm     = np.asarray(obs["delta_m"], dtype=float)
    from destacking.relations import mass_lambda_McClintock18
    logM   = mass_lambda_McClintock18(Lambda, z)
    dm_eff = np.where(np.isfinite(dm), dm, DELTA_M_REF)
    dm_eff = np.maximum(dm_eff, 1e-6)
    log_dm_r = np.log10(dm_eff / DELTA_M_REF)
    logC = (np.log10(ols["c0_true"])
            + ols["alpha_true"] * (logM - _MC_M_PIVOT_LOG)
            + ols["beta_true"]  * np.log10((1.0 + z) / (1.0 + _MC_Z_REF))
            + gamma * log_dm_r)
    return 10.0 ** logC


def mean_model_profile(obs, R_fit, ols, gamma, fmiss, sig_off):
    """Mean of forward-model ΔΣ over all clusters."""
    model = forward_model_combined(
        c0=ols["c0_true"], alpha=ols["alpha_true"], gamma=gamma,
        cluster_table=obs, radii=R_fit,
        beta=ols["beta_true"],
        M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
        delta_m_ref=DELTA_M_REF, log_M_norm=0.0,
        FMISS=fmiss, SIGMA_OFF=sig_off, BCG_B_MASS=None,
    )
    return model.mean(axis=0), model


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Run directory: {run_dir}")
    out_dir = os.path.join(run_dir, "plateau")
    os.makedirs(out_dir, exist_ok=True)

    # Pick negative-γ simulations
    sim_dirs = sorted(
        os.path.join(run_dir, d) for d in os.listdir(run_dir)
        if d.startswith("simulation_g")
    )
    sims = [(parse_gamma(d), d) for d in sim_dirs]
    sims = [(g, d) for g, d in sims if g is not None and g < 0.0]
    print(f"Found {len(sims)} negative-γ simulations.")
    if not sims:
        print("Nothing to investigate; exiting.")
        return

    # γ scan grid — denser near the peak, sparser at edges
    gamma_grid = np.unique(np.concatenate([
        np.linspace(-1.5, -1.05, 5),
        np.linspace(-1.0, 0.0, 41),
        np.linspace(0.05, 0.5, 5),
    ]))

    # ── Plot 1: log-L(γ) scan, one panel per γ_true ───────────────────────
    ncols = min(3, len(sims))
    nrows = (len(sims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)
    summary_rows = []

    # we also store per-sim c distributions and mean profiles for later plots
    cdist_records = []
    mean_records  = []

    keep = RADII >= R_MIN_FIT
    R_fit = RADII[keep]

    for idx, (gamma_true, sim_dir) in enumerate(sims):
        print(f"\n[{idx+1}/{len(sims)}] γ_true={gamma_true:+.1f}  {os.path.basename(sim_dir)}")
        obs, dv, cv, ols, truth = load_sim(sim_dir)
        N = len(obs)
        print(f"  N={N} clusters, OLS c0={ols['c0_true']:.3f}, α={ols['alpha_true']:.4f}, "
              f"β={ols['beta_true']:.4f}, γ_OLS={ols['gamma_true_ols']:+.4f}")

        dv_fit = dv[:, keep]
        cv_fit = cv[:, keep][:, :, keep]
        fmiss  = np.full(N, 1.0 - F_CEN)
        sigoff = np.full(N, SIGMA_OFF_MPC)

        ll = loglike_scan(obs, dv_fit, cv_fit, R_fit, ols,
                          gamma_grid, fmiss, sigoff)

        # locate ML, compare to γ_true
        i_peak = int(np.argmax(ll))
        g_peak = gamma_grid[i_peak]
        ll_peak = ll[i_peak]

        # also evaluate at exact γ_true and -0.5 for a clean readout
        ll_true = log_likelihood_combined(
            c0=ols["c0_true"], alpha=ols["alpha_true"], gamma=float(gamma_true),
            cluster_table=obs, data_vectors=dv_fit, cov_matrices=cv_fit,
            radii=R_fit, beta=ols["beta_true"],
            M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
            delta_m_ref=DELTA_M_REF, log_M_norm=0.0,
            FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
        )
        ll_m05 = log_likelihood_combined(
            c0=ols["c0_true"], alpha=ols["alpha_true"], gamma=-0.5,
            cluster_table=obs, data_vectors=dv_fit, cov_matrices=cv_fit,
            radii=R_fit, beta=ols["beta_true"],
            M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
            delta_m_ref=DELTA_M_REF, log_M_norm=0.0,
            FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
        )
        print(f"  log-L peak at γ={g_peak:+.3f}  (peak={ll_peak:.1f})")
        print(f"  log-L(γ_true={gamma_true:+.1f}) = {ll_true:.1f}")
        print(f"  log-L(γ=-0.5)               = {ll_m05:.1f}")
        print(f"  Δ(L_peak − L_true)          = {ll_peak - ll_true:+.1f}")

        summary_rows.append({
            "gamma_true": gamma_true,
            "g_peak": float(g_peak),
            "ll_peak": float(ll_peak),
            "ll_at_true": float(ll_true),
            "ll_at_minus_half": float(ll_m05),
            "ll_true_minus_peak": float(ll_true - ll_peak),
        })

        ax = axes[idx // ncols][idx % ncols]
        dll = ll - ll_peak
        # clip catastrophic drops at far-negative γ (numerical c-saturation)
        # so the actual shape near the peak is visible
        Y_FLOOR = -10000.0
        dll_clipped = np.maximum(dll, Y_FLOOR)
        ax.plot(gamma_grid, dll_clipped, "o-", color="steelblue", ms=3, lw=1.4)
        ax.axvline(gamma_true, color="crimson", ls="--", lw=1.5,
                   label=rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$")
        ax.axvline(g_peak, color="black", ls=":", lw=1.2,
                   label=rf"peak $\gamma={g_peak:+.3f}$")
        ax.axvline(-0.5, color="grey", ls="--", lw=1.0, alpha=0.6,
                   label=r"$\gamma=-0.5$")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\log L(\gamma) - \log L_{\rm peak}$ (clipped)")
        ax.set_title(rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$  (others at OLS)",
                     fontsize=10)
        # auto-fit y to the post-clip range so the peak shape is visible
        finite = np.isfinite(dll_clipped)
        if finite.any():
            ymin = float(np.min(dll_clipped[finite]))
            ax.set_ylim(max(ymin * 1.05, -8000), 50)
        ax.set_xlim(-1.55, 0.55)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

        # per-cluster c arrays at the three diagnostic γ values
        gammas_diag = [gamma_true, -0.5, 0.0]
        cdist_records.append({
            "gamma_true": gamma_true,
            "c": {g: model_c_array(obs, ols, g) for g in gammas_diag},
            "c_truth": np.asarray(truth["concentration"]),
        })

        # mean predicted profile and mean data profile (only on R_fit)
        means = {}
        for g in gammas_diag:
            mp, _ = mean_model_profile(obs, R_fit, ols, g, fmiss, sigoff)
            means[g] = mp
        mean_records.append({
            "gamma_true": gamma_true,
            "R_fit": R_fit,
            "model_means": means,
            "data_mean": dv_fit.mean(axis=0),
        })

    # hide unused panels
    for k in range(len(sims), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle("Profile log-likelihood vs γ "
                 "(c0, α, β fixed at OLS truth; same DELTA_M_REF=0.1)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "plateau_loglike.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {os.path.join(out_dir, 'plateau_loglike.png')}")

    # ── Plot 2: per-cluster c distribution ─────────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)
    bins = np.logspace(-2, 2.0, 60)
    for idx, rec in enumerate(cdist_records):
        ax = axes[idx // ncols][idx % ncols]
        gt = rec["gamma_true"]
        colors = {gt: "crimson", -0.5: "steelblue", 0.0: "darkgreen"}
        for g, col in colors.items():
            if g not in rec["c"]:
                continue
            c_vals = np.clip(rec["c"][g], bins[0], bins[-1])
            ax.hist(c_vals, bins=bins, histtype="step", color=col, lw=1.6,
                    label=rf"$c(\gamma={g:+.1f})$")
        c_truth = np.clip(rec["c_truth"], bins[0], bins[-1])
        ax.hist(c_truth, bins=bins, histtype="step", color="black", ls=":",
                lw=1.5, label="truth $c$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$c_{200}$")
        ax.set_ylabel("clusters / bin")
        ax.set_title(rf"$\gamma_{{\rm true}}={gt:+.1f}$", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for k in range(len(cdist_records), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle("Model-implied $c_{200}$ distribution at three γ values "
                 "vs the simulated truth", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "plateau_c_distribution.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(out_dir, 'plateau_c_distribution.png')}")

    # ── Plot 3: mean predicted profile vs mean data profile ───────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)
    for idx, rec in enumerate(mean_records):
        ax = axes[idx // ncols][idx % ncols]
        gt = rec["gamma_true"]
        R  = rec["R_fit"]
        ax.plot(R, rec["data_mean"], "ko-", lw=1.6, ms=5, label="data mean")
        colors = {gt: "crimson", -0.5: "steelblue", 0.0: "darkgreen"}
        for g, col in colors.items():
            if g in rec["model_means"]:
                ax.plot(R, rec["model_means"][g], color=col, lw=1.6,
                        marker="o", ms=4, alpha=0.85,
                        label=rf"model $\gamma={g:+.1f}$")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel(r"$R$ [Mpc]")
        ax.set_ylabel(r"$\langle\Delta\Sigma\rangle$  [$M_\odot/{\rm pc}^2$]")
        ax.set_title(rf"$\gamma_{{\rm true}}={gt:+.1f}$", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    for k in range(len(mean_records), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle("Mean predicted ΔΣ at three γ values vs simulated data mean",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "plateau_mean_profile.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(out_dir, 'plateau_mean_profile.png')}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\nSummary:")
    print(f"{'γ_true':>8}  {'γ_peak':>8}  {'logL_peak':>11}  "
          f"{'logL(γ_true)':>13}  {'logL(-0.5)':>11}  {'ΔL_true_peak':>13}")
    for row in summary_rows:
        print(f"{row['gamma_true']:+8.2f}  {row['g_peak']:+8.3f}  "
              f"{row['ll_peak']:11.1f}  {row['ll_at_true']:13.1f}  "
              f"{row['ll_at_minus_half']:11.1f}  "
              f"{row['ll_true_minus_peak']:+13.1f}")

    with open(os.path.join(out_dir, "plateau_summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSaved {os.path.join(out_dir, 'plateau_summary.json')}")


if __name__ == "__main__":
    main()
