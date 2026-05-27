"""
Nested post-hoc bootstrap calibration.

Design
------
  for sim_idx = 0 .. N_SIMULATIONS-1:          # outer: independent simulations
      simulate ONE catalog of N_CLUSTERS clusters + lensing data
      for b = 0 .. N_BOOTSTRAP-1:              # inner: bootstrap draws
          idx  = bootstrap resample of {0..N_CLUSTERS-1}
          run nautilus on data[idx]  ->  γ̂^(sim, b)
      gamma_ensemble[sim] = median_b( γ̂^(sim, b) )
      sigma_boot[sim]     = std_b(    γ̂^(sim, b) )

Calibration check
-----------------
  std( gamma_ensemble )  ≈  mean( sigma_boot )

If this holds, the bootstrap correctly estimates the per-survey uncertainty.

Configuration via environment variables:

  N_SIMULATIONS    Independent outer simulations   (default 10)
  N_BOOTSTRAP      Bootstrap draws per simulation  (default 10)
  N_CLUSTERS       Clusters per simulation         (default 5000)
  N_WORKERS        Pool workers                    (default 15)
  N_RESAMPLE       Bootstrap draw size             (default N_CLUSTERS)
  RUN_DIR          Output directory                (default
                   gamma_repeat_resampled_outputs/run_YYYY-MM-DDTHH-MM-SS)
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import time
import traceback


def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    m = rem / 60
    return f"{int(h)}h{int(m):02d}m"


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocess as mp
import numpy as np
from astropy.cosmology import Planck18 as _planck18
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.stats import norm as sp_norm

import ngmix

from destacking.fitting import run_sampler_combined
from destacking.model import _MC_M_PIVOT_LOG, _MC_Z_REF, _nfw_profiles
from destacking.relations import c_DuttonMaccio
from destacking.simulation import (
    des_y3_cluster_pdf,
    des_y3_source_pdf,
    generate_truth_catalog,
    make_test_data,
    sample_des_y3_cluster_z,
    sample_des_y3_source_z,
)

import xlensing.cosmo as _xlc
import xlensing.sims as _xlsims
from xlensing.data import metacal_cluster_lensing, signal
from xlensing.sims import (
    apply_triaxial_NFW_shear_region,
    make_simple_random_cat,
    project_ellipsoid,
    r_vir,
    sample_triaxial_shapes,
)

try:
    import corner
except ImportError:
    corner = None

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Configuration ────────────────────────────────────────────────────────────
N_SIMULATIONS  = int(os.environ.get("N_SIMULATIONS", 10))
N_BOOTSTRAP    = int(os.environ.get("N_BOOTSTRAP", 10))
N_CLUSTERS     = int(os.environ.get("N_CLUSTERS", 5000))
N_WORKERS      = int(os.environ.get("N_WORKERS", 15))

_n_resample_env = os.environ.get("N_RESAMPLE", "")
N_RESAMPLE = int(_n_resample_env) if _n_resample_env.strip() else None

TRUE_GAMMA    = 0.2
MASTER_SEED   = None

DELTA_M_REF = 0.1
RADII       = np.logspace(-0.8, 0.8, 12)
R_MIN_FIT   = 0.5

DES_Y3_Z_CLUSTER_RANGE = (0.2, 0.65)
DES_Y3_Z_SRC_ALPHA     = 2.0
DES_Y3_Z_SRC_BETA      = 1.5
DES_Y3_Z_SRC_Z0        = 0.4
N_OVERSAMPLE_CLUSTER   = 5

F_CEN           = 0.64
SIGMA_OFF_MPC_H = 0.32
H_PLANCK        = float(_planck18.h)
SIGMA_OFF_MPC   = SIGMA_OFF_MPC_H / H_PLANCK

DENSITY         = 30
WIDTH_RAD       = 0.006
Z_SRC_RANGE     = (0.0, 2.5)
SHAPE_NOISE     = 0.27
NOISE_FLOOR_COV = 1e4

_log_r = np.log10(RADII)
_dlog  = np.diff(_log_r)[0]
_edges = np.logspace(_log_r[0] - _dlog / 2, _log_r[-1] + _dlog / 2, len(RADII) + 1)
BIN_LIMITS = np.column_stack([_edges[:-1], _edges[1:]])
N_BINS     = len(RADII)

PRIORS = {
    "c0":         sp_norm(loc=4.57, scale=0.05),
    "alpha":      sp_norm(loc=-0.1, scale=0.05),
    "beta":       sp_norm(loc=-0.4, scale=0.2),
    "gamma":      sp_norm(loc=0.0,  scale=3.5),
    "log_M_norm": sp_norm(loc=0.0,  scale=0.1),
}
PRIOR_LABELS = {
    "c0": r"$c_0$", "alpha": r"$\alpha$", "beta": r"$\beta$",
    "gamma": r"$\gamma$", "log_M_norm": r"$\log M_{\rm norm}$",
}
PARAM_ORDER = ["c0", "alpha", "beta", "gamma", "log_M_norm"]

N_LIVE  = 400
TIMEOUT = 600

_DEFAULT_RUN_TS = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR = os.environ.get(
    "RUN_DIR",
    os.path.join("gamma_repeat_resampled_outputs", f"run_{_DEFAULT_RUN_TS}"),
)
os.makedirs(RUN_DIR, exist_ok=True)


# ── Simulation pipeline (identical to gamma_repeat.py) ───────────────────────
def build_truth_catalog(gamma_true):
    n_pool     = N_CLUSTERS * N_OVERSAMPLE_CLUSTER
    truth_pool = generate_truth_catalog(
        n_pool, z_range=DES_Y3_Z_CLUSTER_RANGE, logM_range=(13.5, 15.0),
        delta_m_gamma=0.0, scatter_logc=0.0, seed=_seed(0),
    )
    z_pool = np.asarray(truth_pool["z"])
    w_pool = des_y3_cluster_pdf(z_pool, *DES_Y3_Z_CLUSTER_RANGE)
    w_pool = w_pool / w_pool.sum()
    rng_res = np.random.default_rng(_seed(100))
    idx     = rng_res.choice(n_pool, size=N_CLUSTERS, replace=False, p=w_pool)
    truth   = truth_pool[np.sort(idx)]

    dm_s    = np.asarray(truth["delta_m"])
    logdm_s = np.log10(np.maximum(dm_s, 1e-6) / DELTA_M_REF)
    truth["concentration"] = np.asarray(truth["concentration"]) * 10.0 ** (
        gamma_true * logdm_s
    )

    logM_s  = np.log10(truth["M_halo"])
    logc_s  = np.log10(truth["concentration"])
    logz1_s = np.log10((1.0 + truth["z"]) / (1.0 + _MC_Z_REF))
    X = np.column_stack(
        [np.ones(N_CLUSTERS), logM_s - _MC_M_PIVOT_LOG, logz1_s, logdm_s]
    )
    coeffs, *_ = np.linalg.lstsq(X, logc_s, rcond=None)
    return truth, {
        "c0_true":        float(10.0 ** coeffs[0]),
        "alpha_true":     float(coeffs[1]),
        "beta_true":      float(coeffs[2]),
        "gamma_true_ols": float(coeffs[3]),
    }


def _seed(offset):
    return None if MASTER_SEED is None else MASTER_SEED + int(offset)


def compute_triaxial_shapes(truth):
    rng_tri = np.random.default_rng(_seed(10))
    return sample_triaxial_shapes(
        truth["z"], truth["M_halo"],
        ba_range=(0.5, 1.0), ca_lo=0.3, isotropic_los=True, rng=rng_tri,
    )


def compute_miscentering():
    rng_mis     = np.random.default_rng(_seed(20))
    is_mis      = rng_mis.uniform(0, 1, N_CLUSTERS) > F_CEN
    off_mag_mpc = np.where(
        is_mis, rng_mis.rayleigh(scale=SIGMA_OFF_MPC, size=N_CLUSTERS), 0.0
    )
    phi_mis = rng_mis.uniform(0, 2 * np.pi, N_CLUSTERS)
    return is_mis, off_mag_mpc, phi_mis


_TRUTH       = None
_SHAPES      = None
_OFF_MAG_MPC = None
_PHI_MIS     = None
_DA_FAST     = None


def _process_cluster(i):
    z_i = float(_TRUTH["z"][i])
    M_i = float(_TRUTH["M_halo"][i])
    c_i = float(_TRUTH["concentration"][i])
    a_i = _SHAPES["r_vir"][i]

    gal_i  = make_simple_random_cat(
        DENSITY, WIDTH_RAD, Z_SRC_RANGE, shape_noise=0.0,
        seed=_seed(4000 + i), poisson_density=True,
    ).T
    n_gals = len(gal_i)

    rng_src = np.random.default_rng(_seed(5000 + i))
    gal_i[:, 2] = sample_des_y3_source_z(
        n_gals, rng_src,
        z_min=Z_SRC_RANGE[0], z_max=Z_SRC_RANGE[1],
        alpha=DES_Y3_Z_SRC_ALPHA, beta=DES_Y3_Z_SRC_BETA, z0=DES_Y3_Z_SRC_Z0,
    )

    rng_resp = np.random.default_rng(_seed(6000 + i))
    sizes    = rng_resp.lognormal(mean=np.log(0.5), sigma=0.5, size=n_gals)
    R_RESP   = np.clip(1.0 - (sizes / 1.5) ** 2, 0.05, 0.99)
    gal_i[:, 5] = R_RESP
    gal_i[:, 6] = 0.0
    gal_i[:, 7] = 0.0
    gal_i[:, 8] = R_RESP

    rng_sh = np.random.default_rng(_seed(1000 + i))
    sp_i   = ngmix.priors.shape.GPriorBA(sigma=SHAPE_NOISE, rng=rng_sh)
    e1, e2 = sp_i.sample2d(n_gals)
    gal_i[:, 3] = e1
    gal_i[:, 4] = e2

    mc_mask = gal_i[:, 2] > 1.1 * z_i + 0.2
    gal_mc  = gal_i[mc_mask]
    if len(gal_mc) < 20:
        return i, None, None

    cluster_arr = [
        0.0, 0.0, z_i, M_i, c_i, a_i,
        _SHAPES["q_ba"][i] * a_i, _SHAPES["q_ca"][i] * a_i,
        _SHAPES["theta_los"][i], _SHAPES["phi_los"][i],
    ]
    gal_sheared = apply_triaxial_NFW_shear_region(cluster_arr, gal_mc)

    sources_mc = gal_sheared[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]]
    DA_i       = _DA_FAST(0, z_i)
    off_rad_i  = _OFF_MAG_MPC[i] / DA_i
    cen_ra_i   = off_rad_i * np.cos(_PHI_MIS[i])
    cen_dec_i  = off_rad_i * np.sin(_PHI_MIS[i])

    bg = metacal_cluster_lensing((cen_ra_i, cen_dec_i, z_i), sources_mc.T,
                                  WIDTH_RAD * DA_i)
    if bg["Count"] < 20:
        return i, None, None

    stake = np.array([
        bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
        bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"],
    ])
    sig_t, _ = signal(stake, BIN_LIMITS)
    dv = np.nan_to_num(sig_t, nan=0.0)

    R      = bg["Radial Distance"]
    in_bin = (R[:, None] > BIN_LIMITS[:, 0]) & (R[:, None] < BIN_LIMITS[:, 1])
    cov_d  = np.full(N_BINS, NOISE_FLOOR_COV)
    for b in range(N_BINS):
        m   = in_bin[:, b]
        n_b = m.sum()
        if n_b >= 5:
            cov_d[b] = (
                SHAPE_NOISE * bg["Critical Density"][m].mean() / np.sqrt(n_b)
            ) ** 2
    return i, dv, cov_d


def simulate_lensing(truth, shapes, mis):
    global _TRUTH, _SHAPES, _OFF_MAG_MPC, _PHI_MIS, _DA_FAST

    z_tab    = np.linspace(0.0, 3.0, 5000)
    chi_tab  = _xlc.cosmology.comoving_distance(z_tab).value
    chi_of_z = interp1d(z_tab, chi_tab, kind="linear",
                        bounds_error=False, fill_value="extrapolate")

    def DA_fast(z1, z2):
        z2_a = np.atleast_1d(np.asarray(z2, dtype=float))
        res  = (chi_of_z(z2_a) - float(chi_of_z(float(z1)))) / (1.0 + z2_a)
        return float(res[0]) if np.ndim(z2) == 0 else res

    _xlc.DA    = DA_fast
    _xlsims.DA = DA_fast

    _TRUTH        = truth
    _SHAPES       = shapes
    _, _OFF_MAG_MPC, _PHI_MIS = mis
    _DA_FAST      = DA_fast

    data_vectors = np.zeros((N_CLUSTERS, N_BINS))
    cov_matrices = np.full((N_CLUSTERS, N_BINS, N_BINS), 0.0)
    for i in range(N_CLUSTERS):
        np.fill_diagonal(cov_matrices[i], NOISE_FLOOR_COV)

    with mp.Pool(N_WORKERS) as pool:
        for idx, dv, cov_d in pool.imap(
            _process_cluster, range(N_CLUSTERS), chunksize=10
        ):
            if dv is not None:
                data_vectors[idx] = dv
                np.fill_diagonal(cov_matrices[idx], cov_d)

    return data_vectors, cov_matrices


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _wq(v, w, q):
    sorter = np.argsort(v)
    cs     = np.cumsum(w[sorter]) / w.sum()
    return float(np.interp(q, cs, v[sorter]))


def plot_des_y3_nz(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    z_cl = np.linspace(*DES_Y3_Z_CLUSTER_RANGE, 400)
    p_cl = des_y3_cluster_pdf(z_cl, *DES_Y3_Z_CLUSTER_RANGE)
    axes[0].fill_between(z_cl, p_cl, color="steelblue", alpha=0.5)
    axes[0].plot(z_cl, p_cl, color="steelblue", lw=2)
    axes[0].set(xlabel="Cluster z", ylabel="n(z)",
                title=f"DES Y3 cluster n(z) on {DES_Y3_Z_CLUSTER_RANGE}")
    z_sr = np.linspace(0.01, 2.5, 600)
    p_sr = des_y3_source_pdf(z_sr, DES_Y3_Z_SRC_ALPHA,
                              DES_Y3_Z_SRC_BETA, DES_Y3_Z_SRC_Z0)
    axes[1].fill_between(z_sr, p_sr, color="tomato", alpha=0.5)
    axes[1].plot(z_sr, p_sr, color="tomato", lw=2)
    axes[1].set(xlabel="Source z", ylabel="n(z)",
                title=rf"DES Y3 source n(z) Smail($\alpha=2, \beta=1.5, z_0=0.4$)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "des_y3_nz.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_truth_distributions(truth, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].hist(truth["z"], bins=20, color="steelblue", edgecolor="white")
    axes[0, 0].set_xlabel("Redshift z")
    axes[0, 1].hist(np.log10(truth["M_halo"]), bins=20, color="steelblue", edgecolor="white")
    axes[0, 1].set_xlabel(r"$\log_{10}(M_{\rm halo})$"); axes[0, 1].set_yscale("log")
    axes[1, 0].hist(truth["concentration"], bins=20, color="steelblue", edgecolor="white")
    axes[1, 0].set_xlabel("Concentration")
    axes[1, 1].hist(truth["delta_m"], bins=20, color="steelblue", edgecolor="white")
    axes[1, 1].set_xlabel(r"$\Delta m$ [mag]"); axes[1, 1].set_yscale("log")
    plt.suptitle(rf"Truth catalogue  ($N={N_CLUSTERS}$, $\gamma_{{\rm true}}={TRUE_GAMMA}$)",
                 y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "truth_distributions.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mc_scatter(truth, ols, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    logM_s = np.log10(truth["M_halo"])
    sc = axes[0].scatter(
        logM_s, truth["concentration"], c=truth["delta_m"], cmap="plasma",
        s=8, alpha=0.6, vmin=0, vmax=4, rasterized=True,
    )
    plt.colorbar(sc, ax=axes[0], label=r"$\Delta m$ [mag]")
    z_med  = float(np.median(truth["z"]))
    M_grid = np.logspace(13.5, 15.0, 200)
    axes[0].plot(np.log10(M_grid), c_DuttonMaccio(z_med * np.ones(200), M_grid, c=1.0),
                 "k-", lw=2, label=rf"DM14 ($z={z_med:.2f}$)")
    axes[0].set(xlabel=r"$\log_{10}(M_{\rm halo})$", ylabel=r"$c_{200}$",
                title="M-c coloured by Δm"); axes[0].legend(fontsize=9)
    sc2 = axes[1].scatter(
        truth["delta_m"], truth["concentration"], c=logM_s, cmap="viridis",
        s=8, alpha=0.6, rasterized=True,
    )
    plt.colorbar(sc2, ax=axes[1], label=r"$\log_{10}(M_{\rm halo})$")
    axes[1].set(xlabel=r"$\Delta m$ [mag]", ylabel=r"$c_{200}$",
                title=r"$c_{200}$ vs Δm")
    plt.suptitle(
        rf"OLS truth: $c_0={ols['c0_true']:.2f}$, "
        rf"$\alpha={ols['alpha_true']:.3f}$, "
        rf"$\beta={ols['beta_true']:.3f}$, "
        rf"$\gamma={ols['gamma_true_ols']:.3f}$", y=1.02, fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "mc_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_triaxial_shapes(shapes, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(shapes["q_ba"], bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("b/a")
    axes[1].hist(shapes["q_ca"], bins=20, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("c/a")
    axes[2].hist(shapes["q_min"] / shapes["q_maj"], bins=20,
                 color="tomato", edgecolor="white")
    axes[2].set_xlabel(r"Projected $q_{\rm min}/q_{\rm maj}$")
    for ax in axes:
        ax.set_ylabel("Count")
    plt.suptitle("Triaxial shape distributions", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "triaxial_shapes.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_miscentering_offsets(mis, outdir):
    is_mis, off_mag_mpc, phi_mis = mis
    n_mis = int(is_mis.sum())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(off_mag_mpc[is_mis], bins=25, color="tomato",
                 edgecolor="white", density=True)
    if n_mis > 0:
        r_grid  = np.linspace(0, off_mag_mpc.max() * 1.1, 300)
        pdf_ray = r_grid / SIGMA_OFF_MPC**2 * np.exp(-r_grid**2 / (2 * SIGMA_OFF_MPC**2))
        axes[0].plot(r_grid, pdf_ray, "k-", lw=1.5,
                     label=rf"Rayleigh($\sigma={SIGMA_OFF_MPC:.3f}$ Mpc)")
        axes[0].legend()
    axes[0].set(xlabel="Offset magnitude [Mpc]", ylabel="PDF",
                title=f"Miscentred offset distribution  (N={n_mis})")
    for r_i, ph_i in zip(off_mag_mpc[is_mis][:200], phi_mis[is_mis][:200]):
        axes[1].plot(r_i * np.cos(ph_i), r_i * np.sin(ph_i), "o",
                     color="tomato", ms=2, alpha=0.4)
    axes[1].add_patch(plt.Circle((0, 0), SIGMA_OFF_MPC, fill=False,
                                  ls="--", color="k", lw=1))
    axes[1].set_aspect("equal")
    axes[1].set(xlabel="ΔRA [Mpc]", ylabel="ΔDec [Mpc]",
                title="Offset positions on sky")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "miscentering_offsets.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sample_profiles(data_vectors, cov_matrices, truth, is_mis, off_mag_mpc, outdir):
    idx  = np.round(np.linspace(0, N_CLUSTERS - 1, 6)).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    cmap_dm = plt.cm.plasma
    dm_all  = np.asarray(truth["delta_m"])
    vmin, vmax = 0.0, float(np.nanpercentile(dm_all, 95))
    for ax, i in zip(axes.ravel(), idx):
        errs   = np.sqrt(np.diag(cov_matrices[i]))
        real   = errs < np.sqrt(NOISE_FLOOR_COV)
        dm_i   = float(truth["delta_m"][i])
        col    = cmap_dm((dm_i - vmin) / (vmax - vmin))
        marker = "^" if is_mis[i] else "o"
        ax.errorbar(RADII[real], data_vectors[i, real], yerr=errs[real],
                    fmt=marker, markersize=4, capsize=4, color=col)
        ax.set_xscale("log"); ax.set_yscale("log")
        mis_str = f"off={off_mag_mpc[i]:.2f} Mpc" if is_mis[i] else "centred"
        ax.set_title(
            rf"$z={truth['z'][i]:.2f}$  $\log M={np.log10(truth['M_halo'][i]):.2f}$  "
            rf"$\Delta m={dm_i:.2f}$  {mis_str}", fontsize=9,
        )
        ax.set_xlabel("R [Mpc]")
        ax.set_ylabel(r"$\Delta\Sigma$ [M$_\odot$/pc$^2$]")
    plt.suptitle("Sample ΔΣ profiles  (▲ = miscentred)", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "sample_profiles.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_triaxiality_diagnostic(truth, data_vectors, cov_matrices, outdir):
    truth_profiles = _nfw_profiles(
        np.asarray(truth["z"]), np.asarray(truth["M_halo"]),
        np.asarray(truth["concentration"]), RADII,
    )
    frac_res   = (data_vectors - truth_profiles) / np.where(
        truth_profiles > 0, truth_profiles, np.nan
    )
    errs       = np.sqrt(np.diagonal(cov_matrices, axis1=1, axis2=2))
    real_mask  = errs < np.sqrt(NOISE_FLOOR_COV)
    frac_res_m = np.where(real_mask, frac_res, np.nan)
    med        = np.nanmedian(frac_res_m, axis=0)
    p16        = np.nanpercentile(frac_res_m, 16, axis=0)
    p84        = np.nanpercentile(frac_res_m, 84, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].fill_between(RADII, p16, p84, color="steelblue", alpha=0.25, label="16-84%")
    axes[0].plot(RADII, med, "-o", color="steelblue", label="Median")
    axes[0].axhline(0.0, color="k", lw=0.5, ls=":")
    axes[0].axvline(R_MIN_FIT, color="crimson", lw=1.5, ls="--",
                    label=f"R_MIN_FIT = {R_MIN_FIT} Mpc")
    axes[0].set(xscale="log", xlabel="R [Mpc]", ylim=(-1.5, 1.5),
                ylabel=r"$(\Delta\Sigma_{\rm data} - \Delta\Sigma_{\rm NFW,truth})/\Delta\Sigma_{\rm NFW,truth}$",
                title="Per-cluster fractional residual")
    axes[0].legend(fontsize=9)
    axes[1].plot(RADII, (p84 - p16) / 2.0, "-o", color="tomato",
                 label="Triaxiality scatter (half 68% spread)")
    axes[1].axvline(R_MIN_FIT, color="crimson", lw=1.5, ls="--")
    axes[1].set(xscale="log", xlabel="R [Mpc]", ylabel=r"$\sigma_{\rm res}$",
                title="Per-cluster fractional scatter vs R")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "triaxiality_diagnostic.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_corner_with_priors(points, weights, ols, outdir):
    if corner is None:
        return
    labels = [PRIOR_LABELS[n] for n in PARAM_ORDER]
    truths = [ols["c0_true"], ols["alpha_true"], ols["beta_true"],
              ols["gamma_true_ols"], 0.0]
    fig = corner.corner(
        points, weights=weights, labels=labels, truths=truths,
        truth_color="crimson", show_titles=True, title_fmt=".3f",
        smooth=1.0, color="steelblue", hist_kwargs={"density": True},
    )
    ndim_c = points.shape[1]
    axes   = np.array(fig.axes).reshape((ndim_c, ndim_c))
    for i, name in enumerate(PARAM_ORDER):
        ax       = axes[i, i]
        xlim     = ax.get_xlim()
        x_grid   = np.linspace(xlim[0], xlim[1], 400)
        pdf_vals = PRIORS[name].pdf(x_grid)
        if np.any(np.isfinite(pdf_vals) & (pdf_vals > 0)):
            ax.plot(x_grid, pdf_vals, color="darkorange", lw=1.6, ls="--",
                    alpha=0.9, zorder=5)
    handles = [
        Line2D([0], [0], color="steelblue",  lw=2,            label="Posterior"),
        Line2D([0], [0], color="darkorange", lw=1.6, ls="--", label="Prior"),
        Line2D([0], [0], color="crimson",    lw=1.5,           label="Truth"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=10, frameon=True)
    fig.savefig(os.path.join(outdir, "corner.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mc_recovery_band(points, weights, truth, ols, outdir):
    M_grid   = np.logspace(13.5, 15.0, 200)
    logM_g   = np.log10(M_grid)
    z_med    = float(np.median(truth["z"]))
    logz1_g  = np.log10((1.0 + z_med) / (1.0 + _MC_Z_REF))
    rng_plot = np.random.default_rng(0)
    idx_band = rng_plot.choice(len(points), size=300,
                                p=weights / weights.sum(), replace=True)
    dm_show    = [0.5, DELTA_M_REF, 2.0]
    colors     = ["royalblue", "steelblue", "teal"]
    colors_tru = ["navy", "darkblue", "darkcyan"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for dm_val, col, col_t in zip(dm_show, colors, colors_tru):
        log_dm_r = np.log10(max(dm_val, 1e-6) / DELTA_M_REF)
        C_band = np.array([
            10.0 ** (np.log10(points[k, 0])
                     + points[k, 1] * (logM_g - _MC_M_PIVOT_LOG)
                     + points[k, 2] * logz1_g
                     + points[k, 3] * log_dm_r)
            for k in idx_band
        ])
        ax.fill_between(logM_g, C_band.min(0), C_band.max(0), alpha=0.20, color=col)
        c_piv_true = (max(dm_val, 1e-6) / DELTA_M_REF) ** TRUE_GAMMA
        ax.plot(logM_g, c_DuttonMaccio(z_med * np.ones(200), M_grid, c=c_piv_true),
                "--", color=col_t, lw=1.5,
                label=rf"DM14 truth ($\Delta m={dm_val:.1f}$ mag)")
    sc = ax.scatter(np.log10(truth["M_halo"]), truth["concentration"],
                    c=truth["delta_m"], cmap="plasma", s=5, alpha=0.25,
                    vmin=0, vmax=4, rasterized=True, zorder=0)
    plt.colorbar(sc, ax=ax, label=r"$\Delta m$ [mag]")
    ax.set(xlabel=r"$\log_{10}(M_{\rm halo})$", ylabel=r"$c_{200}$")
    ax.set_title("M-c + γ recovery band  (post-hoc bootstrap, $R \\geq 0.5$ Mpc)",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "mc_recovery_band.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_marginal(points, weights, outdir):
    gamma_s = points[:, 3]
    lo      = _wq(gamma_s, weights, 0.16)
    med     = _wq(gamma_s, weights, 0.50)
    hi      = _wq(gamma_s, weights, 0.84)
    bias    = (med - TRUE_GAMMA) / max((hi - lo) / 2.0, 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(gamma_s, weights=weights, bins=40, density=True,
            color="steelblue", alpha=0.65, edgecolor="white", label="Posterior")
    ax.axvspan(lo, hi, alpha=0.20, color="steelblue",
               label=rf"68% CI [{lo:.3f}, {hi:.3f}]")
    ax.axvline(med,        color="steelblue", lw=1.5, ls="-",
               label=rf"Median $\hat\gamma={med:.3f}$")
    ax.axvline(TRUE_GAMMA, color="crimson",   lw=2.0, ls="--",
               label=rf"True $\gamma={TRUE_GAMMA}$")
    xlim   = ax.get_xlim()
    x_grid = np.linspace(xlim[0], xlim[1], 400)
    ax.plot(x_grid, PRIORS["gamma"].pdf(x_grid),
            color="darkorange", lw=2.0, ls="--", label="Prior")
    ax.set(xlabel=r"$\gamma$", ylabel="Density",
           title=rf"$\gamma$ marginal — post-hoc bootstrap  ($N={N_CLUSTERS}$, bias={bias:+.2f}σ)")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "gamma_marginal.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return gamma_s, weights, lo, med, hi


# ── Single bootstrap draw runner ─────────────────────────────────────────────
def run_one_bootstrap(b_idx, sim_subdir, truth, ols,
                      cluster_obs, data_vectors_fit, cov_matrices_fit, RADII_fit):
    """Run one bootstrap draw on a pre-simulated fixed dataset."""
    subdir       = os.path.join(sim_subdir, f"bootstrap_{b_idx:03d}")
    os.makedirs(subdir, exist_ok=True)
    summary_path = os.path.join(subdir, "summary.json")
    npz_path     = os.path.join(subdir, "gamma_posterior.npz")

    n_resample_actual = N_RESAMPLE if N_RESAMPLE is not None else N_CLUSTERS
    rng_boot  = np.random.default_rng()
    idx_boot  = rng_boot.choice(N_CLUSTERS, size=n_resample_actual, replace=True)
    n_unique  = int(np.unique(idx_boot).size)

    cluster_obs_boot      = cluster_obs[idx_boot]
    data_vectors_fit_boot = data_vectors_fit[idx_boot]
    cov_matrices_fit_boot = cov_matrices_fit[idx_boot]
    # Population-level miscentering constants — not resampled
    FMISS_boot     = np.full(n_resample_actual, 1.0 - F_CEN)
    SIGMA_OFF_boot = np.full(n_resample_actual, SIGMA_OFF_MPC)

    print(
        f"    bootstrap resample: {n_resample_actual} drawn "
        f"({n_unique} unique, {100*n_unique/N_CLUSTERS:.1f}%)",
        flush=True,
    )

    checkpoint = os.path.join(subdir, "nautilus_checkpoint.hdf5")
    sampler = run_sampler_combined(
        cluster_obs_boot, data_vectors_fit_boot, cov_matrices_fit_boot, RADII_fit,
        priors=PRIORS, delta_m_ref=DELTA_M_REF, fit_beta=True,
        FMISS=FMISS_boot, SIGMA_OFF=SIGMA_OFF_boot,
        n_live=N_LIVE, pool=N_WORKERS,
        verbose=True, filepath=checkpoint, resume=True, timeout=TIMEOUT,
    )
    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()

    plot_corner_with_priors(points, weights, ols, subdir)
    plot_mc_recovery_band(points, weights, truth, ols, subdir)
    gamma_s, w, lo, med, hi = plot_gamma_marginal(points, weights, subdir)
    np.savez_compressed(npz_path, gamma_samples=gamma_s, weights=w)

    result = {
        "bootstrap":   b_idx,
        "gamma_true":  TRUE_GAMMA,
        "gamma_med":   med,
        "gamma_lo":    lo,
        "gamma_hi":    hi,
        "sigma_gamma": (hi - lo) / 2.0,
        "n_clusters":  N_CLUSTERS,
        "n_resample":  n_resample_actual,
        "n_unique":    n_unique,
        **ols,
    }
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"    gamma_med = {med:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
        f"sigma = {(hi - lo) / 2.0:.4f}",
        flush=True,
    )
    return result


# ── Single simulation runner (simulate once, B bootstrap draws) ───────────────
def run_one_simulation(sim_idx, run_dir):
    sim_subdir   = os.path.join(run_dir, f"simulation_{sim_idx:03d}")
    os.makedirs(sim_subdir, exist_ok=True)
    ens_path     = os.path.join(sim_subdir, "ensemble_summary.json")

    print(f"\n  building truth catalogue ...", flush=True)
    truth, ols = build_truth_catalog(TRUE_GAMMA)
    print(
        f"    OLS truth: c0={ols['c0_true']:.3f}, "
        f"alpha={ols['alpha_true']:.4f}, "
        f"beta={ols['beta_true']:.4f}, "
        f"gamma_ols={ols['gamma_true_ols']:.4f}",
        flush=True,
    )

    print("  drawing triaxial shapes + miscentering ...", flush=True)
    shapes = compute_triaxial_shapes(truth)
    mis    = compute_miscentering()
    is_mis, off_mag_mpc, _ = mis

    print(f"  simulating lensing ({N_CLUSTERS} clusters) ...", flush=True)
    data_vectors, cov_matrices = simulate_lensing(truth, shapes, mis)

    cluster_obs, _, _ = make_test_data(
        truth, RADII, fractional_noise=0.2,
        poisson_richness=False, seed=_seed(1),
    )

    keep             = RADII >= R_MIN_FIT
    RADII_fit        = RADII[keep]
    data_vectors_fit = data_vectors[:, keep]
    cov_matrices_fit = cov_matrices[:, keep][:, :, keep]

    # Simulation-level plots (once per simulation)
    plot_truth_distributions(truth, sim_subdir)
    plot_mc_scatter(truth, ols, sim_subdir)
    plot_triaxial_shapes(shapes, sim_subdir)
    plot_miscentering_offsets(mis, sim_subdir)
    plot_sample_profiles(data_vectors, cov_matrices, truth, is_mis, off_mag_mpc, sim_subdir)
    plot_triaxiality_diagnostic(truth, data_vectors, cov_matrices, sim_subdir)

    # Bootstrap loop
    results      = []
    failures     = []
    attempt_idx  = 0
    max_attempts = 4 * N_BOOTSTRAP

    while len(results) < N_BOOTSTRAP and attempt_idx < max_attempts:
        b_subdir     = os.path.join(sim_subdir, f"bootstrap_{attempt_idx:03d}")
        summary_path = os.path.join(b_subdir, "summary.json")
        npz_path     = os.path.join(b_subdir, "gamma_posterior.npz")

        if os.path.exists(summary_path) and os.path.exists(npz_path):
            print(
                f"  [{len(results)+1}/{N_BOOTSTRAP}] bootstrap_{attempt_idx:03d} -- cached",
                flush=True,
            )
            with open(summary_path) as f:
                results.append(json.load(f))
            attempt_idx += 1
            continue

        print(
            f"\n  [{len(results)+1}/{N_BOOTSTRAP}] bootstrap_{attempt_idx:03d}",
            flush=True,
        )
        try:
            r = run_one_bootstrap(
                attempt_idx, sim_subdir,
                truth, ols, cluster_obs,
                data_vectors_fit, cov_matrices_fit, RADII_fit,
            )
            results.append(r)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            traceback.print_exc()
            failures.append((attempt_idx, type(exc).__name__))
        attempt_idx += 1

    if not results:
        raise RuntimeError(f"simulation_{sim_idx:03d}: all bootstrap draws failed")

    # Ensemble statistics for this simulation
    gamma_meds  = np.array([r["gamma_med"]   for r in results])
    sigma_posts = np.array([r["sigma_gamma"] for r in results])
    ens_med     = float(np.median(gamma_meds))
    sigma_boot  = float(np.std(gamma_meds, ddof=1)) if len(gamma_meds) > 1 else 0.0
    mean_sigma  = float(np.mean(sigma_posts))

    # Add realisation field for compatibility with aggregate plot functions
    for k, r in enumerate(results):
        r["realisation"] = k

    plot_gamma_residuals(results, sim_subdir)
    plot_posteriors_overlay(results, sim_subdir)

    ens_result = {
        "simulation":         sim_idx,
        "gamma_true":         TRUE_GAMMA,
        "gamma_ensemble_med": ens_med,
        "sigma_boot":         sigma_boot,
        "mean_sigma_post":    mean_sigma,
        "n_bootstrap":        len(results),
        "n_failures":         len(failures),
        **ols,
    }
    with open(ens_path, "w") as f:
        json.dump(ens_result, f, indent=2)
    print(
        f"  simulation_{sim_idx:03d} ensemble: "
        f"gamma={ens_med:+.4f}  sigma_boot={sigma_boot:.4f}  "
        f"mean_sigma_post={mean_sigma:.4f}",
        flush=True,
    )
    return ens_result


# ── Aggregate plots ──────────────────────────────────────────────────────────
def plot_gamma_residuals(results, run_dir):
    seq_idx    = np.arange(1, len(results) + 1)
    gm         = np.array([r["gamma_med"]  for r in results])
    glo        = np.array([r["gamma_lo"]   for r in results])
    ghi        = np.array([r["gamma_hi"]   for r in results])
    yerr       = np.vstack([gm - glo, ghi - gm])
    residuals  = gm - TRUE_GAMMA
    mean_res   = np.mean(residuals)
    std_res    = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
    mean_sigma = np.mean((ghi - glo) / 2.0)
    n_res_label = results[0].get("n_resample", N_CLUSTERS) if results else N_CLUSTERS

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(seq_idx, residuals, yerr=yerr, fmt="o",
                color="steelblue", capsize=4, markersize=6,
                label=r"Posterior median $\pm$68% CI")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="Unbiased")
    ax.axhspan(mean_res - std_res, mean_res + std_res, alpha=0.10, color="tomato",
               label=rf"$\sigma_{{\rm runs}} = {std_res:.4f}$")
    ax.axhline(mean_res, color="tomato", lw=1.5, ls=":",
               label=rf"Mean residual = {mean_res:+.4f}")
    ax.set(xlabel="Realisation #",
           ylabel=r"$\hat\gamma - \gamma_{\rm true}$",
           title=(rf"$\gamma$ residuals — post-hoc bootstrap  "
                  rf"($\gamma_{{\rm true}}={TRUE_GAMMA}$, "
                  rf"$N={N_CLUSTERS}$, $n_{{\rm resample}}={n_res_label}$)"))
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    if mean_sigma > 0:
        ratio = std_res / mean_sigma
        ax.text(
            0.02, 0.02,
            rf"$\langle\sigma_\gamma^{{\rm post}}\rangle = {mean_sigma:.4f}$"
            "\n"
            rf"$\sigma_{{\rm runs}} / \langle\sigma_\gamma^{{\rm post}}\rangle = {ratio:.2f}$"
            "\n"
            f"(1.0 = error bars well-calibrated)",
            transform=ax.transAxes, fontsize=10,
            va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"),
        )
    plt.tight_layout()
    out = os.path.join(run_dir, "gamma_residuals.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def plot_posteriors_overlay(results, run_dir):
    fig, ax  = plt.subplots(figsize=(9, 5))
    x_grid   = np.linspace(-0.3, 0.7, 600)
    cmap     = plt.cm.viridis
    n        = len(results)
    for k, r in enumerate(results):
        npz_path = os.path.join(
            run_dir, f"bootstrap_{r['bootstrap']:03d}", "gamma_posterior.npz"
        )
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
                                    range=(x_grid[0], x_grid[-1]), density=True)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), y,
                    color=cmap(k / max(n - 1, 1)), alpha=0.6, lw=1.0)
            continue
        ax.plot(x_grid, y, color=cmap(k / max(n - 1, 1)), alpha=0.6, lw=1.0)

    ax.plot(x_grid, PRIORS["gamma"].pdf(x_grid),
            color="darkorange", lw=1.8, ls="--", label="Prior")
    ax.axvline(TRUE_GAMMA, color="crimson", lw=2.0, ls="--",
               label=rf"$\gamma_{{\rm true}}={TRUE_GAMMA}$")
    ax.set(xlabel=r"$\gamma$", ylabel="Posterior density",
           title=(rf"$\gamma$ posteriors — post-hoc bootstrap  "
                  rf"($\gamma_{{\rm true}}={TRUE_GAMMA}$, $N={N_CLUSTERS}$)"))
    ax.set_xlim(x_grid[0], x_grid[-1])
    handles_extra = [Line2D([0], [0], color=cmap(0.5), lw=1.2,
                             label=rf"{n} posterior densities (KDE)")]
    ax.legend(handles=[*handles_extra, *ax.get_legend_handles_labels()[0]],
              loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(run_dir, "posteriors_overlay.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def plot_calibration_summary(sim_results, run_dir):
    """Outer calibration plot: ensemble γ̂ per simulation with σ_boot error bars."""
    sim_idx   = np.arange(1, len(sim_results) + 1)
    ens_meds  = np.array([r["gamma_ensemble_med"] for r in sim_results])
    sig_boots = np.array([r["sigma_boot"]         for r in sim_results])

    std_ens      = float(np.std(ens_meds, ddof=1)) if len(ens_meds) > 1 else 0.0
    mean_sigboot = float(np.mean(sig_boots))
    ratio        = std_ens / mean_sigboot if mean_sigboot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(sim_idx, ens_meds - TRUE_GAMMA, yerr=sig_boots,
                fmt="o", color="steelblue", capsize=5, markersize=7,
                label=r"Ensemble $\hat\gamma_{\rm ens}$ ± $\sigma_{\rm boot}$")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="Unbiased")
    ax.axhspan(-std_ens, std_ens, alpha=0.10, color="tomato",
               label=rf"$\sigma(\hat\gamma_{{\rm ens}}) = {std_ens:.4f}$")
    ax.set(
        xlabel="Simulation #",
        ylabel=r"$\hat\gamma_{\rm ens} - \gamma_{\rm true}$",
        title=(
            rf"Bootstrap calibration  ($\gamma_{{\rm true}}={TRUE_GAMMA}$, "
            rf"$N={N_CLUSTERS}$, {N_BOOTSTRAP} bootstrap draws/sim)"
        ),
    )
    ax.text(
        0.02, 0.02,
        rf"$\sigma(\hat\gamma_{{\rm ens}}) = {std_ens:.4f}$"
        "\n"
        rf"$\langle\sigma_{{\rm boot}}\rangle = {mean_sigboot:.4f}$"
        "\n"
        rf"ratio = {ratio:.2f}  (1.0 = calibrated)",
        transform=ax.transAxes, fontsize=10, va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="grey"),
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(run_dir, "calibration_summary.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def print_calibration_table(sim_results):
    print("\n" + "=" * 75, flush=True)
    print(
        f'{"sim":>4}  {"gamma_ens":>11}  {"sigma_boot":>11}  '
        f'{"mean_sig_post":>14}  {"bias[σ_boot]":>12}',
        flush=True,
    )
    print("-" * 75, flush=True)
    for r in sim_results:
        bias = (r["gamma_ensemble_med"] - TRUE_GAMMA) / max(r["sigma_boot"], 1e-12)
        print(
            f'{r["simulation"]:>4d}  {r["gamma_ensemble_med"]:>+11.4f}  '
            f'{r["sigma_boot"]:>11.4f}  '
            f'{r["mean_sigma_post"]:>14.4f}  '
            f'{bias:>+12.2f}',
            flush=True,
        )
    ens_meds  = [r["gamma_ensemble_med"] for r in sim_results]
    sig_boots = [r["sigma_boot"]         for r in sim_results]
    std_ens   = float(np.std(ens_meds, ddof=1)) if len(ens_meds) > 1 else 0.0
    mean_sb   = float(np.mean(sig_boots))
    print("-" * 75, flush=True)
    print(
        f"std(gamma_ens) = {std_ens:.4f}   "
        f"mean(sigma_boot) = {mean_sb:.4f}   "
        f"ratio = {std_ens/mean_sb:.2f}  (target 1.0)",
        flush=True,
    )


def main():
    n_resample_actual = N_RESAMPLE if N_RESAMPLE is not None else N_CLUSTERS
    print(
        f"Bootstrap calibration: N_SIMULATIONS={N_SIMULATIONS}, "
        f"N_BOOTSTRAP={N_BOOTSTRAP}, N_CLUSTERS={N_CLUSTERS}, "
        f"N_RESAMPLE={n_resample_actual}, N_WORKERS={N_WORKERS}",
        flush=True,
    )
    print(f"Output directory: {RUN_DIR}", flush=True)

    plot_des_y3_nz(RUN_DIR)

    sim_results = []
    run_t0      = time.time()

    for sim_idx in range(N_SIMULATIONS):
        sim_subdir = os.path.join(RUN_DIR, f"simulation_{sim_idx:03d}")
        ens_path   = os.path.join(sim_subdir, "ensemble_summary.json")

        print(
            f"\n[{sim_idx + 1}/{N_SIMULATIONS}] simulation_{sim_idx:03d}",
            flush=True,
        )
        iter_t0 = time.time()

        if os.path.exists(ens_path):
            print("  cached, loading ensemble_summary.json", flush=True)
            with open(ens_path) as f:
                sim_results.append(json.load(f))
            continue

        try:
            r = run_one_simulation(sim_idx, RUN_DIR)
            sim_results.append(r)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"  simulation_{sim_idx:03d} FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue

        iter_dt = time.time() - iter_t0
        done    = len(sim_results)
        avg_dt  = (time.time() - run_t0) / done
        eta     = avg_dt * (N_SIMULATIONS - done)
        print(
            f"  [{done}/{N_SIMULATIONS}] took {_fmt_duration(iter_dt)}; "
            f"ETA {_fmt_duration(eta)}",
            flush=True,
        )

    total_elapsed = time.time() - run_t0
    print(
        f"\nTotal wall time: {_fmt_duration(total_elapsed)}  "
        f"({len(sim_results)}/{N_SIMULATIONS} simulations completed)",
        flush=True,
    )

    if not sim_results:
        print("No simulations completed.", file=sys.stderr)
        return 1

    with open(os.path.join(RUN_DIR, "calibration_summary.json"), "w") as f:
        json.dump(sim_results, f, indent=2)

    plot_calibration_summary(sim_results, RUN_DIR)
    print_calibration_table(sim_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
