"""
Empirical test: does a global amplitude boost remove the γ̂ ≈ -0.5 plateau?

For each negative-γ_true simulation in the bootstrap reference run we evaluate
log L(γ) under three model choices:

  (A) bare spherical NFW            (current pipeline)
  (B) spherical NFW × 1.25          (constant boost from sphere_vs_triaxial.py)
  (C) spherical NFW × A_opt(γ)      (best-fit per-γ amplitude — gives a
                                     lower bound on the residual bias if you
                                     also let an amplitude nuisance float)

(c0, α, β) are held at the OLS-truth values; γ is scanned.  We report the
log-L peak γ for each scheme to see whether the boost dissolves the plateau.
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
RADII       = np.logspace(-0.8, 0.8, 12)
R_MIN_FIT   = 0.5
F_CEN       = 0.64
SIGMA_OFF   = 0.32 / float(_planck18.h)

BOOST = 1.25


def find_latest_run():
    base = "gamma_scan_outputs"
    runs = sorted((d for d in os.listdir(base) if d.startswith("run_")),
                  key=lambda d: os.path.getmtime(os.path.join(base, d)),
                  reverse=True)
    return os.path.join(base, runs[0])


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
    run_dir = "gamma_scan_outputs/run_2026-05-24T10-19-32"   # reference run
    print(f"Using run: {run_dir}")

    sim_dirs = sorted(os.path.join(run_dir, d)
                      for d in os.listdir(run_dir) if d.startswith("simulation_g"))
    sims = [(parse_gamma(d), d) for d in sim_dirs]
    sims = [(g, d) for g, d in sims if g is not None and g <= 0.0]

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
                c0=ols["c0_true"], alpha=ols["alpha_true"], gamma=float(g),
                cluster_table=obs, radii=R_fit, beta=ols["beta_true"],
                M_pivot_log=_MC_M_PIVOT_LOG, z_ref=_MC_Z_REF,
                delta_m_ref=DELTA_M_REF, log_M_norm=0.0,
                FMISS=fmiss, SIGMA_OFF=sigoff, BCG_B_MASS=None,
            )
            ll_bare.append(ll(model,         dv_fit, cv_fit))
            ll_boost.append(ll(model * BOOST, dv_fit, cv_fit))

            # Best-fit amplitude A that maximises log L(A·model) per γ.
            # Closed form: A_opt = Σ_i (m_i^T C_i^-1 d_i) / Σ_i (m_i^T C_i^-1 m_i)
            num = 0.0
            den = 0.0
            for i in range(N):
                Li = np.linalg.cholesky(cv_fit[i])
                yi = np.linalg.solve(Li, dv_fit[i])
                mi = np.linalg.solve(Li, model[i])
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
    out = "gamma_scan_outputs/run_2026-05-24T10-19-32/plateau/amplitude_boost_test.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
