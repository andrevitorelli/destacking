"""
Test whether the γ̂≈-0.5 plateau is a parametrization issue.

For each negative-γ_true simulation, we evaluate three things:

  (A) Spherical-NFW ΔΣ predicted from the EXACT truth (z, M_halo, c_truth)
      — no parametrization at all, no log-power-law fit to c.
  (B) Spherical-NFW ΔΣ predicted by the inference model at γ=γ_true with
      OLS-fit (c0, α, β) — the "best parametric realisation" of the truth.
  (C) Spherical-NFW ΔΣ predicted by the inference model at γ=-0.5.

We compare each of these against the simulated data (which used triaxial
NFW shears).  If (A) matches the data well, the bias is a parametrization
artefact.  If (A) and (B) match each other and both deviate from the data,
the bias comes from triaxial→spherical projection, NOT parametrization.

Also computes the total log-likelihood for each scenario to quantify the
preference.

Outputs into RUN_DIR/plateau/:
  truth_vs_param_profile.png  — mean ΔΣ comparison
  truth_vs_param_summary.json
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

from destacking.fitting import cholog_gaussian
from destacking.model import (
    forward_model_combined, _nfw_profiles, _MC_M_PIVOT_LOG, _MC_Z_REF,
)

from astropy.cosmology import Planck18 as _planck18

DELTA_M_REF = 0.1
RADII       = np.logspace(-0.8, 0.8, 12)
R_MIN_FIT   = 0.5
F_CEN       = 0.64
SIGMA_OFF_MPC_H = 0.32
SIGMA_OFF_MPC = SIGMA_OFF_MPC_H / float(_planck18.h)


def find_latest_run():
    base = "gamma_scan_outputs"
    runs = sorted(
        (d for d in os.listdir(base) if d.startswith("run_")),
        key=lambda d: os.path.getmtime(os.path.join(base, d)),
        reverse=True,
    )
    return os.path.join(base, runs[0])


def parse_gamma(sim_dir):
    m = re.search(r"gamma([+-][0-9.]+)$", os.path.basename(sim_dir))
    return float(m.group(1)) if m else None


def load_sim(sim_dir):
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


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Run directory: {run_dir}")
    out_dir = os.path.join(run_dir, "plateau")
    os.makedirs(out_dir, exist_ok=True)

    sim_dirs = sorted(
        os.path.join(run_dir, d) for d in os.listdir(run_dir)
        if d.startswith("simulation_g")
    )
    sims = [(parse_gamma(d), d) for d in sim_dirs]
    sims = [(g, d) for g, d in sims if g is not None and g <= 0.0]

    keep = RADII >= R_MIN_FIT
    R_fit = RADII[keep]

    ncols = min(3, len(sims))
    nrows = (len(sims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)

    summary = []
    for idx, (gamma_true, sim_dir) in enumerate(sims):
        print(f"\n[{idx+1}/{len(sims)}] γ_true={gamma_true:+.1f}  "
              f"{os.path.basename(sim_dir)}")
        obs, dv, cv, ols, truth = load_sim(sim_dir)
        N = len(obs)

        z_truth = np.asarray(truth["z"])
        M_truth = np.asarray(truth["M_halo"])
        c_truth = np.asarray(truth["concentration"])
        fmiss   = np.full(N, 1.0 - F_CEN)
        sigoff  = np.full(N, SIGMA_OFF_MPC)

        # (A) spherical NFW with true (z, M, c) — no parametrisation
        ds_truth = _nfw_profiles(
            z_truth, M_truth, c_truth, R_fit,
            FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
        )

        # (B) parametric inference model at γ=γ_true with OLS coefficients
        ds_param_true = forward_model_combined(
            c0=ols["c0_true"], alpha=ols["alpha_true"], gamma=float(gamma_true),
            cluster_table=obs, radii=R_fit,
            beta=ols["beta_true"],
            M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
            delta_m_ref=DELTA_M_REF, log_M_norm=0.0,
            FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
        )

        # (C) parametric inference model at γ=-0.5 with OLS coefficients
        ds_param_m05 = forward_model_combined(
            c0=ols["c0_true"], alpha=ols["alpha_true"], gamma=-0.5,
            cluster_table=obs, radii=R_fit,
            beta=ols["beta_true"],
            M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
            delta_m_ref=DELTA_M_REF, log_M_norm=0.0,
            FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
        )

        dv_fit = dv[:, keep]
        cv_fit = cv[:, keep][:, :, keep]

        ll_truth = cholog_gaussian(dv_fit - ds_truth,      cv_fit)
        ll_b     = cholog_gaussian(dv_fit - ds_param_true, cv_fit)
        ll_c     = cholog_gaussian(dv_fit - ds_param_m05,  cv_fit)

        print(f"  log-L (true c, no param)        = {ll_truth:.1f}")
        print(f"  log-L (param γ={gamma_true:+.1f}, OLS) = {ll_b:.1f}")
        print(f"  log-L (param γ=-0.5,  OLS)      = {ll_c:.1f}")
        print(f"  Δ(param γ_true − truth_c)       = {ll_b - ll_truth:+.1f}")
        print(f"  Δ(param γ=-0.5  − truth_c)      = {ll_c - ll_truth:+.1f}")

        summary.append({
            "gamma_true": float(gamma_true),
            "ll_truth_c":      float(ll_truth),
            "ll_param_true":   float(ll_b),
            "ll_param_minus05":float(ll_c),
        })

        # plot mean profiles
        ax = axes[idx // ncols][idx % ncols]
        ax.plot(R_fit, dv_fit.mean(axis=0),    "ko-", lw=1.6, ms=6, label="data mean (triaxial)")
        ax.plot(R_fit, ds_truth.mean(axis=0),  color="purple", lw=1.6, ms=5, marker="s",
                label="spherical NFW with true c")
        ax.plot(R_fit, ds_param_true.mean(axis=0), color="crimson", lw=1.6, ms=4, marker="o",
                label=rf"param $\gamma={gamma_true:+.1f}$")
        ax.plot(R_fit, ds_param_m05.mean(axis=0), color="steelblue", lw=1.6, ms=4, marker="o",
                label=r"param $\gamma=-0.5$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("R [Mpc]")
        ax.set_ylabel(r"$\langle\Delta\Sigma\rangle$  [$M_\odot/{\rm pc}^2$]")
        ax.set_title(rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

    for k in range(len(sims), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle("Mean ΔΣ: data (triaxial) vs spherical NFW with "
                 "true-c, parametric γ=γ_true, parametric γ=-0.5",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "truth_vs_param_profile.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    with open(os.path.join(out_dir, "truth_vs_param_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary table:")
    print(f"{'γ_true':>8}  {'ll_truth_c':>12}  {'ll_param_γt':>12}  "
          f"{'ll_param_-0.5':>13}")
    for row in summary:
        print(f"{row['gamma_true']:+8.2f}  {row['ll_truth_c']:12.1f}  "
              f"{row['ll_param_true']:12.1f}  {row['ll_param_minus05']:13.1f}")


if __name__ == "__main__":
    main()
