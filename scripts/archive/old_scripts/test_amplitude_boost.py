"""
REVIEWER VALIDATION SCRIPT — amplitude-mismatch diagnostic for the
triaxial-vs-spherical bias.

Empirical test: does a global amplitude boost remove the γ̂ ≈ -0.5 plateau?

For each gamma_true <= 0.0 simulation in a gamma_inference.py SIMULATION
run (simulation: true), we evaluate log L(γ) under three model choices:

  (A) bare spherical NFW            (current pipeline)
  (B) spherical NFW × 1.25          (constant boost from sphere_vs_triaxial.py)
  (C) spherical NFW × A_opt(γ)      (best-fit per-γ amplitude — gives a
                                     lower bound on the residual bias if you
                                     also let an amplitude nuisance float)

(c0, α, β) are held at the OLS-truth values; γ is scanned.  We report the
log-L peak γ for each scheme to see whether the boost dissolves the plateau.

Context: this predates and is distinct from the later "amplitude
correction A as a free fit parameter" experiment, which project memory
records as closed (negligible effect, A now pinned at 1.0 in production
fits). This script instead asks a narrower question — does a *fixed*
amplitude mismatch explain the specific negative-γ plateau — which is
still a valid standalone diagnostic for a reviewer to reproduce.

Usage:
  python test_amplitude_boost.py RUN_DIR
  python test_amplitude_boost.py   # uses the latest run in
                                    # ~/destacking_data/output/

Fixed 2026-07-27: had no CLI arg support at all (RUN_DIR was hardcoded to
a since-deleted run_2026-05-24T10-19-32, as was the output path);
forward_model_combined's alpha/beta/gamma kwargs were renamed
alpha_cm/beta_cm/gamma_cm; the fixed 12-point RADII grid didn't match
every run's actual sim grid (now reconstructed via load_radii()); and the
closed-form A_opt computation didn't mask per-cluster NaN bins the way
destacking.fitting.cholog_gaussian does (too few background sources at a
given radius for a given cluster is a normal occurrence, not an error —
see cholog_gaussian's docstring), so a single NaN bin anywhere silently
poisoned the whole A_opt sum to NaN. Verified against a fresh
3-gamma-point smoke-test scan (150 clusters, gamma_true in {-0.3, 0.0}):
A_opt now comes out finite and in (0, 1) as expected for a small/noisy
sample, and the comparison plot renders correctly.
"""

from __future__ import annotations
import json
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import Planck18 as _planck18

from destacking.fitting import cholog_gaussian
from destacking.model import (
    forward_model_combined, _MC_M_PIVOT_LOG, _MC_Z_REF,
)

DELTA_M_REF = 0.1
R_MIN_FIT   = 0.5
F_CEN       = 0.64
SIGMA_OFF   = 0.32 / float(_planck18.h)

BOOST = 1.25


def find_latest_run():
    base = os.path.expanduser("~/destacking_data/output")
    runs = sorted((d for d in os.listdir(base) if d.startswith("run_")),
                  key=lambda d: os.path.getmtime(os.path.join(base, d)),
                  reverse=True)
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


def parse_gamma(sim_dir):
    m = re.search(r"gamma([+-][0-9.]+)$", os.path.basename(sim_dir))
    return float(m.group(1)) if m else None


def load_sim(sim_dir):
    co  = np.load(os.path.join(sim_dir, "cluster_obs.npz"))
    obs = Table({k: co[k] for k in co.files})
    raw = np.load(os.path.join(sim_dir, "simulated_data.npz"))
    with open(os.path.join(sim_dir, "ols.json")) as f:
        ols = json.load(f)
    return obs, raw["data_vectors"], raw["cov_matrices"], ols


def ll(model_arr, dv, cv):
    """Cholog Gaussian on residuals = dv - model_arr."""
    if not np.all(np.isfinite(model_arr)):
        return -np.inf
    return cholog_gaussian(dv - model_arr, cv)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Using run: {run_dir}")

    sim_dirs = sorted(os.path.join(run_dir, d)
                      for d in os.listdir(run_dir) if d.startswith("simulation_g"))
    sims = [(parse_gamma(d), d) for d in sim_dirs]
    sims = [(g, d) for g, d in sims if g is not None and g <= 0.0]

    RADII  = load_radii(run_dir)
    keep   = RADII >= R_MIN_FIT
    R_fit  = RADII[keep]

    gamma_grid = np.linspace(-1.3, 0.2, 31)

    rows = []
    for gamma_true, sim_dir in sims:
        obs, dv, cv, ols = load_sim(sim_dir)
        N      = len(obs)
        dv_fit = dv[:, keep]
        cv_fit = cv[:, keep][:, :, keep]
        fmiss  = np.full(N, 1.0 - F_CEN)
        sigoff = np.full(N, SIGMA_OFF)

        ll_bare, ll_boost, ll_opt, a_opt_list = [], [], [], []
        for g in gamma_grid:
            model = forward_model_combined(
                c0=ols["c0_true"], alpha_cm=ols["alpha_true"], gamma_cm=float(g),
                cluster_table=obs, radii=R_fit, beta_cm=ols["beta_true"],
                M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
                delta_m_ref=DELTA_M_REF, log_M_norm=0.0,
                FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
            )
            ll_bare.append(ll(model,         dv_fit, cv_fit))
            ll_boost.append(ll(model * BOOST, dv_fit, cv_fit))

            # Best-fit amplitude A that maximises log L(A·model) per γ.
            # Closed form: A_opt = Σ_i (m_i^T C_i^-1 d_i) / Σ_i (m_i^T C_i^-1 m_i)
            # Per-bin NaNs (too few sources at that radius for that cluster)
            # are masked exactly like cholog_gaussian does, not just
            # per-model finiteness -- a cluster with any NaN bin would
            # otherwise poison the whole sum with a NaN.
            num = 0.0
            den = 0.0
            for i in range(N):
                valid = np.isfinite(dv_fit[i])
                if not valid.any():
                    continue
                Li = np.linalg.cholesky(cv_fit[i][np.ix_(valid, valid)])
                yi = np.linalg.solve(Li, dv_fit[i][valid])
                mi = np.linalg.solve(Li, model[i][valid])
                num += float(np.dot(yi, mi))
                den += float(np.dot(mi, mi))
            A_opt = num / den if den > 0 else 1.0
            a_opt_list.append(A_opt)
            ll_opt.append(ll(model * A_opt, dv_fit, cv_fit))

        ll_bare  = np.asarray(ll_bare)
        ll_boost = np.asarray(ll_boost)
        ll_opt   = np.asarray(ll_opt)

        g_peak_bare  = float(gamma_grid[int(np.argmax(ll_bare))])
        g_peak_boost = float(gamma_grid[int(np.argmax(ll_boost))])
        g_peak_opt   = float(gamma_grid[int(np.argmax(ll_opt))])
        A_at_peak    = float(a_opt_list[int(np.argmax(ll_opt))])

        print(f"\nγ_true={gamma_true:+.1f}")
        print(f"  bare        peak γ = {g_peak_bare:+.3f}  (bias {g_peak_bare-gamma_true:+.3f})")
        print(f"  ×1.25 boost peak γ = {g_peak_boost:+.3f}  (bias {g_peak_boost-gamma_true:+.3f})")
        print(f"  free  A     peak γ = {g_peak_opt:+.3f}  (bias {g_peak_opt-gamma_true:+.3f}),  A_opt = {A_at_peak:.3f}")

        rows.append({
            "gamma_true":    gamma_true,
            "peak_bare":     g_peak_bare,
            "peak_boost":    g_peak_boost,
            "peak_opt":      g_peak_opt,
            "A_at_peak_opt": A_at_peak,
        })

    # Plot: peak γ vs γ_true under the three schemes
    gt = np.array([r["gamma_true"]  for r in rows])
    pb = np.array([r["peak_bare"]   for r in rows])
    pB = np.array([r["peak_boost"]  for r in rows])
    po = np.array([r["peak_opt"]    for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    g_lim = max(abs(gt.min()), abs(gt.max())) + 0.2
    ax.plot([-g_lim, g_lim], [-g_lim, g_lim], "k--", lw=1, label=r"$\hat\gamma=\gamma_{\rm true}$")
    ax.plot(gt, pb, "o-", color="steelblue", lw=1.6, ms=7, label="bare spherical NFW")
    ax.plot(gt, pB, "s-", color="darkorange", lw=1.6, ms=7, label=r"spherical $\times 1.25$ (uniform boost)")
    ax.plot(gt, po, "^-", color="crimson",    lw=1.6, ms=7, label=r"spherical $\times A_{\rm opt}(\gamma)$ (best-fit amplitude per $\gamma$)")
    ax.set_xlabel(r"$\gamma_{\rm true}$", fontsize=12)
    ax.set_ylabel(r"likelihood peak $\gamma$", fontsize=12)
    ax.set_title(r"Effect of a global amplitude correction on the negative-$\gamma$ plateau",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_dir = os.path.join(run_dir, "plateau")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "amplitude_boost_test.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
