"""
REVIEWER VALIDATION SCRIPT — DELTA_M_REF choice / c0-gamma degeneracy check.

Test whether changing DELTA_M_REF shifts the recovered-gamma plateau for
the most negative-gamma_true simulation available in a gamma_inference.py
SIMULATION run (simulation: true). Scans log L(γ) for two reference
choices:
  D=0.1  (current)
  D=0.01 (low-Δm tail)
with c0 either (a) reparametrised alongside D so the per-cluster c values
stay pinned to the OLS truth (pure reparametrisation, should NOT move the
likelihood peak), or (b) held fixed at the OLS c0 (mimics a tight c0
prior, and is where a poor DELTA_M_REF choice interacting with that prior
CAN shift the peak).

Outputs into RUN_DIR/plateau/:
  delta_m_ref_test_D0.01.png — log-L vs γ for the two DELTA_M_REF choices,
                               one panel per c0 treatment

Usage:
  python test_delta_m_ref.py RUN_DIR
  python test_delta_m_ref.py   # uses the latest run in
                                # ~/destacking_data/output/

Fixed 2026-07-27: hardcoded a specific dead run directory and a specific
simulation_g00_gamma-1.0 subdirectory — now picks the most-negative
gamma_true simulation_g* checkpoint actually present in RUN_DIR (see
find_most_negative_gamma_sim()), and every gamma_true=-1-specific
axvline/title string is now computed from that value instead. Also fixed
the alpha/beta/gamma -> alpha_cm/beta_cm/gamma_cm kwarg rename and the
fixed 12-point RADII grid (now reconstructed per-run via load_radii()).
Verified against a fresh 3-gamma-point smoke-test scan (150 clusters,
most-negative point gamma_true=-0.3) — the four log-L(γ) scans give
distinct, finite peak locations as expected; the underlying question
(does reparametrisation alone move the peak, vs. does the interaction
with a fixed/tight c0 move it) needs a larger production run to answer
with real statistical power — this smoke test only confirms the script
itself runs correctly end to end.
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

R_MIN_FIT = 0.5
F_CEN     = 0.64
SIG_OFF   = 0.32 / float(_planck18.h)


def find_latest_run():
    base = os.path.expanduser("~/destacking_data/output")
    runs = sorted((d for d in os.listdir(base) if d.startswith("run_")),
                  key=lambda d: os.path.getmtime(os.path.join(base, d)),
                  reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run directories in {base}/")
    return os.path.join(base, runs[0])


def load_radii(run_dir):
    """Reconstruct the simulated radial grid from the run's saved YAML
    config. Falls back to the historical default grid if none is found."""
    import glob
    import yaml
    for cfg_path in glob.glob(os.path.join(run_dir, "*.yaml")):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        if "sim_log_inner_radius" in cfg:
            return np.logspace(
                cfg["sim_log_inner_radius"], cfg["sim_log_outer_radius"],
                int(cfg["sim_n_radii"]),
            )
    return np.logspace(-0.8, 0.8, 12)


def find_most_negative_gamma_sim(run_dir):
    """Pick the most negative-gamma_true simulation_g* checkpoint
    available -- this diagnostic is specifically about the negative-gamma
    plateau, so an arbitrary point isn't a substitute the way it is for
    the shape-distribution script."""
    import re
    sim_dirs = [d for d in os.listdir(run_dir) if d.startswith("simulation_g")]
    if not sim_dirs:
        raise FileNotFoundError(f"No simulation_g* checkpoints in {run_dir}")
    def _gamma(d):
        m = re.search(r"gamma([+-][0-9.]+)$", d)
        return float(m.group(1)) if m else np.inf
    return min(sim_dirs, key=_gamma)


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
            c0=c0, alpha_cm=ols["alpha_true"], gamma_cm=float(g),
            cluster_table=obs, data_vectors=dv_fit, cov_matrices=cv_fit,
            radii=R_fit, beta_cm=ols["beta_true"],
            M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
            delta_m_ref=D, log_M_norm=0.0,
            FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
        )
    return out


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    sim_dir = os.path.join(run_dir, find_most_negative_gamma_sim(run_dir))
    gamma_true = float(sim_dir.rsplit("gamma", 1)[1])
    print(f"Using {sim_dir}  (gamma_true={gamma_true:+.2f})")
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

    RADII  = load_radii(run_dir)
    keep   = RADII >= R_MIN_FIT
    R_fit  = RADII[keep]
    dv_fit = dv[:, keep]
    cv_fit = cv[:, keep][:, :, keep]
    N      = len(obs)
    fmiss  = np.full(N, 1.0 - F_CEN)
    sigoff = np.full(N, SIG_OFF)

    gamma_grid = np.linspace(min(gamma_true - 0.3, -1.3), 0.0, 27)
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
    ax.axvline(gamma_true, color="crimson", ls="--", lw=1.5,
               label=rf"$\gamma_{{\rm true}}={gamma_true:+.2f}$")
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
    ax.axvline(gamma_true, color="crimson", ls="--", lw=1.5,
               label=rf"$\gamma_{{\rm true}}={gamma_true:+.2f}$")
    ax.axvline(-0.5, color="grey", ls="--", lw=1.0, alpha=0.5,
               label=r"$\gamma=-0.5$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\log L(\gamma) - \log L_{\rm peak}$")
    ax.set_title("c₀ FIXED at 4.575 (mimics tight prior)\n"
                 "shows how DELTA_M_REF interacts with the c0 prior", fontsize=10)
    ax.set_ylim(-5000, 50)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(rf"$\gamma_{{\rm true}}={gamma_true:+.2f}$ simulation — effect of "
                 r"DELTA_M_REF on the log-likelihood",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_dir = os.path.join(run_dir, "plateau")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "delta_m_ref_test_D0.01.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
