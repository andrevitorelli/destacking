"""
Test whether changing DELTA_M_REF to the median Δm shifts the γ̂ plateau.

For γ_true = -1.0 simulation, scan log L(γ) for two reference choices:
  D=0.1 (current)              with c0 = c0_old   = 4.575
  D=0.236 (median)             with c0 = c0_old * 10^(0.373*γ_test)
                                              ↑↑↑ varies along the scan
We also redo the test holding c0 fixed at 4.575 in both, to show how the
TIGHT c0 prior interacts with each choice of DELTA_M_REF.

Outputs into RUN_DIR/plateau/:
  delta_m_ref_test.png — log-L vs γ for the two DELTA_M_REF choices
"""

from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18 as _planck18

from destacking.fitting import log_likelihood_combined
from destacking.model import _MC_M_PIVOT_LOG, _MC_Z_REF

RADII     = np.logspace(-0.8, 0.8, 12)
R_MIN_FIT = 0.5
F_CEN     = 0.64
SIG_OFF   = 0.32 / float(_planck18.h)


def load_sim(sim_dir):
    co  = np.load(os.path.join(sim_dir, "cluster_obs.npz"))
    obs = Table({k: co[k] for k in co.files})
    raw = np.load(os.path.join(sim_dir, "simulated_data.npz"))
    with open(os.path.join(sim_dir, "ols.json")) as f:
        ols = json.load(f)
    return obs, raw["data_vectors"], raw["cov_matrices"], ols


def scan(obs, dv_fit, cv_fit, R_fit, ols, gamma_grid, D, fmiss, sigoff,
         c0_mode):
    """
    c0_mode = "consistent": c0 changes with γ so model c values match OLS truth
              "fixed_4.57": c0 held at 4.575 (mimics tight prior)
    """
    out = np.empty_like(gamma_grid)
    c0_old = ols["c0_true"]
    for k, g in enumerate(gamma_grid):
        if c0_mode == "consistent":
            # c0(D, γ) = c0_old * (D/0.1)^γ   makes per-cluster c values
            # invariant under reparametrisation:
            #   c0_new * (dm/D_new)^γ = c0_old * (dm/D_old)^γ
            # ⇒  c0_new / c0_old = (D_new/D_old)^γ
            c0 = c0_old * (D / 0.1) ** g
        else:
            c0 = c0_old   # = 4.575, i.e., tight-prior centre
        out[k] = log_likelihood_combined(
            c0=c0, alpha=ols["alpha_true"], gamma=float(g),
            cluster_table=obs, data_vectors=dv_fit, cov_matrices=cv_fit,
            radii=R_fit, beta=ols["beta_true"],
            M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
            delta_m_ref=D, log_M_norm=0.0,
            FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
        )
    return out


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 \
              else "gamma_scan_outputs/run_2026-05-24T10-19-32"
    sim_dir = os.path.join(run_dir, "simulation_g00_gamma-1.0")
    print(f"Using {sim_dir}")
    obs, dv, cv, ols = load_sim(sim_dir)
    print(f"OLS: c0={ols['c0_true']:.3f}, α={ols['alpha_true']:.4f}, "
          f"β={ols['beta_true']:.4f}, γ_OLS={ols['gamma_true_ols']:.4f}")

    # Confirm median Δm
    dm = np.asarray(obs["delta_m"], dtype=float)
    dm_finite = dm[np.isfinite(dm) & (dm > 0)]
    dm_median = float(np.median(dm_finite))
    print(f"Δm median = {dm_median:.4f}  (other percentiles: "
          f"16%={np.percentile(dm_finite,16):.3f}, "
          f"84%={np.percentile(dm_finite,84):.3f})")

    keep   = RADII >= R_MIN_FIT
    R_fit  = RADII[keep]
    dv_fit = dv[:, keep]
    cv_fit = cv[:, keep][:, :, keep]
    N      = len(obs)
    fmiss  = np.full(N, 1.0 - F_CEN)
    sigoff = np.full(N, SIG_OFF)

    gamma_grid = np.linspace(-1.3, 0.0, 27)
    refs = [0.1, 0.01]
    labels = [r"$D=0.1$ (current)", r"$D=0.01$ (low-$\Delta m$ tail)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    # ── Panel 1: consistent c0 (model c values match OLS truth at γ_test=γ_OLS) ──
    ax = axes[0]
    for D, lab in zip(refs, labels):
        ll = scan(obs, dv_fit, cv_fit, R_fit, ols, gamma_grid, D,
                  fmiss, sigoff, c0_mode="consistent")
        ll_peak = float(np.max(ll))
        g_peak  = float(gamma_grid[np.argmax(ll)])
        ax.plot(gamma_grid, ll - ll_peak, "o-", lw=1.4, ms=4,
                label=f"{lab}, peak γ={g_peak:+.3f}")
        print(f"[consistent c0]  D={D:.3f}: peak γ={g_peak:+.3f}  "
              f"ll_peak={ll_peak:.1f}")
    ax.axvline(-1.0, color="crimson", ls="--", lw=1.5,
               label=r"$\gamma_{\rm true}=-1$")
    ax.axvline(-0.5, color="grey", ls="--", lw=1.0, alpha=0.5,
               label=r"$\gamma=-0.5$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\log L(\gamma) - \log L_{\rm peak}$")
    ax.set_title("c₀ varies with γ to keep per-cluster c = OLS truth\n"
                 "(reparametrisation only)", fontsize=10)
    ax.set_ylim(-5000, 50)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # ── Panel 2: c0 fixed at 4.575 in both ─────────────────────────────────
    ax = axes[1]
    for D, lab in zip(refs, labels):
        ll = scan(obs, dv_fit, cv_fit, R_fit, ols, gamma_grid, D,
                  fmiss, sigoff, c0_mode="fixed_4.57")
        ll_peak = float(np.max(ll))
        g_peak  = float(gamma_grid[np.argmax(ll)])
        ax.plot(gamma_grid, ll - ll_peak, "o-", lw=1.4, ms=4,
                label=f"{lab}, peak γ={g_peak:+.3f}")
        print(f"[c0=4.575]       D={D:.3f}: peak γ={g_peak:+.3f}  "
              f"ll_peak={ll_peak:.1f}")
    ax.axvline(-1.0, color="crimson", ls="--", lw=1.5,
               label=r"$\gamma_{\rm true}=-1$")
    ax.axvline(-0.5, color="grey", ls="--", lw=1.0, alpha=0.5,
               label=r"$\gamma=-0.5$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\log L(\gamma) - \log L_{\rm peak}$")
    ax.set_title("c₀ FIXED at 4.575 (mimics tight prior)\n"
                 "shows how DELTA_M_REF interacts with the c0 prior", fontsize=10)
    ax.set_ylim(-5000, 50)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(r"$\gamma_{\rm true}=-1$ simulation — effect of "
                 r"DELTA_M_REF on the log-likelihood",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(run_dir, "plateau", "delta_m_ref_test_D0.01.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
