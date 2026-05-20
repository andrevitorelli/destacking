"""
Run the unbiased notebook pipeline N_REALISATIONS times to estimate how
well the gamma error bars are calibrated.

Each realisation:
  * Builds a DES~Y3-like truth catalogue with gamma_true = 0.2 imprinted as
    a direct log10(c) shift.
  * Draws CDM-like triaxial shapes via xlensing.sims.sample_triaxial_shapes.
  * Draws Rayleigh+delta miscentering offsets.
  * Simulates the triaxial-NFW lensing signal on an independent random
    background catalogue per cluster (Poisson density, DES Y3 source n(z)).
  * Builds the cluster observable table (deterministic richness).
  * Applies the radial cut R >= 0.5 Mpc.
  * Runs the combined nautilus sampler with the tight DM14 priors and a
    fixed-parameter population miscentering correction in the forward
    model.
  * Saves every notebook plot (truth distributions, M-c scatter, triaxial
    shapes, miscentering offsets, sample profiles, triaxiality residual
    diagnostic, corner, M-c recovery band, gamma marginal) into a per-
    realisation subdirectory together with the gamma posterior samples
    and the median/68% CI as JSON.

MASTER_SEED is hard-coded to None so that every realisation draws fresh
OS entropy.  The aggregate plots written into the run root are:

  gamma_residuals.png       (gamma_med - gamma_true) for every realisation
                            with 68% error bars and the std of the medians
                            compared against the average per-realisation
                            posterior sigma -- this is the error-bar
                            calibration test.

  posteriors_overlay.png    All N_REALISATIONS gamma posteriors smoothed and
                            overlaid on the same axes, against the truth
                            and the prior.

Configuration is exposed via environment variables:

  N_REALISATIONS  Number of realisations         (default 20)
  N_CLUSTERS      Clusters per realisation       (default 5000)
  N_WORKERS       Pool workers for lensing       (default 15)
  RUN_DIR         Output directory               (default
                  gamma_repeat_outputs/run_YYYY-MM-DDTHH-MM-SS).
                  Each invocation creates a new timestamped subdirectory;
                  set RUN_DIR to resume an interrupted run.

Cached realisation summaries (<RUN_DIR>/realisation_NNN/summary.json) are
reloaded on subsequent invocations of the same RUN_DIR so a failed run can
be resumed.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
import traceback


def _fmt_duration(seconds):
    """Format a duration in seconds as a short human-readable string."""
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

# ── Configuration (mirrors notebook) ────────────────────────────────────────
N_REALISATIONS = int(os.environ.get("N_REALISATIONS", 20))
N_CLUSTERS = int(os.environ.get("N_CLUSTERS", 5000))
N_WORKERS = int(os.environ.get("N_WORKERS", 15))

TRUE_GAMMA = 0.2
MASTER_SEED = None  # always None for this script — user wants run-to-run variability


def _seed(offset):
    """MASTER_SEED + offset, or None when MASTER_SEED is None."""
    return None if MASTER_SEED is None else MASTER_SEED + int(offset)


DELTA_M_REF = 0.1
RADII = np.logspace(-0.8, 0.8, 12)
R_MIN_FIT = 0.5

DES_Y3_Z_CLUSTER_RANGE = (0.2, 0.65)
DES_Y3_Z_SRC_ALPHA = 2.0
DES_Y3_Z_SRC_BETA = 1.5
DES_Y3_Z_SRC_Z0 = 0.4
N_OVERSAMPLE_CLUSTER = 5

F_CEN = 0.64
SIGMA_OFF_MPC_H = 0.32
H_PLANCK = float(_planck18.h)
SIGMA_OFF_MPC = SIGMA_OFF_MPC_H / H_PLANCK

DENSITY = 30
WIDTH_RAD = 0.006
Z_SRC_RANGE = (0.0, 2.5)
SHAPE_NOISE = 0.27
NOISE_FLOOR_COV = 1e4

_log_r = np.log10(RADII)
_dlog = np.diff(_log_r)[0]
_edges = np.logspace(_log_r[0] - _dlog / 2, _log_r[-1] + _dlog / 2, len(RADII) + 1)
BIN_LIMITS = np.column_stack([_edges[:-1], _edges[1:]])
N_BINS = len(RADII)

PRIORS = {
    "c0":         sp_norm(loc=4.57, scale=0.05),
    "alpha":      sp_norm(loc=-0.1, scale=0.05),
    "beta":       sp_norm(loc=-0.4, scale=0.2),
    "gamma":      sp_norm(loc=0.0,  scale=1.),
    "log_M_norm": sp_norm(loc=0.0,  scale=0.1),
}
PRIOR_LABELS = {
    "c0": r"$c_0$", "alpha": r"$\alpha$", "beta": r"$\beta$",
    "gamma": r"$\gamma$", "log_M_norm": r"$\log M_{\rm norm}$",
}
PARAM_ORDER = ["c0", "alpha", "beta", "gamma", "log_M_norm"]

N_LIVE = 400
TIMEOUT = 600
_DEFAULT_RUN_TS = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR = os.environ.get(
    "RUN_DIR", os.path.join("gamma_repeat_outputs", f"run_{_DEFAULT_RUN_TS}")
)
os.makedirs(RUN_DIR, exist_ok=True)


# ── Pipeline stages (identical to gamma_scan.py / notebook) ─────────────────
def build_truth_catalog(gamma_true):
    n_pool = N_CLUSTERS * N_OVERSAMPLE_CLUSTER
    truth_pool = generate_truth_catalog(
        n_pool, z_range=DES_Y3_Z_CLUSTER_RANGE, logM_range=(13.5, 15.0),
        delta_m_gamma=0.0, scatter_logc=0.0, seed=_seed(0),
    )
    z_pool = np.asarray(truth_pool["z"])
    w_pool = des_y3_cluster_pdf(z_pool, *DES_Y3_Z_CLUSTER_RANGE)
    w_pool = w_pool / w_pool.sum()
    rng_res = np.random.default_rng(_seed(100))
    idx = rng_res.choice(n_pool, size=N_CLUSTERS, replace=False, p=w_pool)
    truth = truth_pool[np.sort(idx)]

    dm_s = np.asarray(truth["delta_m"])
    logdm_s = np.log10(np.maximum(dm_s, 1e-6) / DELTA_M_REF)
    truth["concentration"] = np.asarray(truth["concentration"]) * 10.0 ** (
        gamma_true * logdm_s
    )

    logM_s = np.log10(truth["M_halo"])
    logc_s = np.log10(truth["concentration"])
    logz1_s = np.log10((1.0 + truth["z"]) / (1.0 + _MC_Z_REF))
    X = np.column_stack(
        [np.ones(N_CLUSTERS), logM_s - _MC_M_PIVOT_LOG, logz1_s, logdm_s]
    )
    coeffs, *_ = np.linalg.lstsq(X, logc_s, rcond=None)
    return truth, {
        "c0_true": float(10.0 ** coeffs[0]),
        "alpha_true": float(coeffs[1]),
        "beta_true": float(coeffs[2]),
        "gamma_true_ols": float(coeffs[3]),
    }


def compute_triaxial_shapes(truth):
    rng_tri = np.random.default_rng(_seed(10))
    return sample_triaxial_shapes(
        truth["z"], truth["M_halo"],
        ba_range=(0.5, 1.0), ca_lo=0.3, isotropic_los=True, rng=rng_tri,
    )


def compute_miscentering():
    rng_mis = np.random.default_rng(_seed(20))
    is_mis = rng_mis.uniform(0, 1, N_CLUSTERS) > F_CEN
    off_mag_mpc = np.where(
        is_mis, rng_mis.rayleigh(scale=SIGMA_OFF_MPC, size=N_CLUSTERS), 0.0
    )
    phi_mis = rng_mis.uniform(0, 2 * np.pi, N_CLUSTERS)
    return is_mis, off_mag_mpc, phi_mis


# Module-level worker state
_TRUTH = None
_SHAPES = None
_OFF_MAG_MPC = None
_PHI_MIS = None
_DA_FAST = None


def _process_cluster(i):
    z_i = float(_TRUTH["z"][i])
    M_i = float(_TRUTH["M_halo"][i])
    c_i = float(_TRUTH["concentration"][i])
    a_i = _SHAPES["r_vir"][i]

    gal_i = make_simple_random_cat(
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
    sizes = rng_resp.lognormal(mean=np.log(0.5), sigma=0.5, size=n_gals)
    R_RESP = np.clip(1.0 - (sizes / 1.5) ** 2, 0.05, 0.99)
    gal_i[:, 5] = R_RESP
    gal_i[:, 6] = 0.0
    gal_i[:, 7] = 0.0
    gal_i[:, 8] = R_RESP

    rng_sh = np.random.default_rng(_seed(1000 + i))
    sp_i = ngmix.priors.shape.GPriorBA(sigma=SHAPE_NOISE, rng=rng_sh)
    e1, e2 = sp_i.sample2d(n_gals)
    gal_i[:, 3] = e1
    gal_i[:, 4] = e2

    mc_mask = gal_i[:, 2] > 1.1 * z_i + 0.2
    gal_mc = gal_i[mc_mask]
    if len(gal_mc) < 20:
        return i, None, None

    cluster_arr = [
        0.0, 0.0, z_i, M_i, c_i, a_i,
        _SHAPES["q_ba"][i] * a_i, _SHAPES["q_ca"][i] * a_i,
        _SHAPES["theta_los"][i], _SHAPES["phi_los"][i],
    ]
    gal_sheared = apply_triaxial_NFW_shear_region(cluster_arr, gal_mc)

    sources_mc = gal_sheared[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]]
    radius_mpc = WIDTH_RAD * _DA_FAST(0, z_i)

    DA_i = _DA_FAST(0, z_i)
    off_rad_i = _OFF_MAG_MPC[i] / DA_i
    cen_ra_i = off_rad_i * np.cos(_PHI_MIS[i])
    cen_dec_i = off_rad_i * np.sin(_PHI_MIS[i])

    bg = metacal_cluster_lensing((cen_ra_i, cen_dec_i, z_i), sources_mc.T, radius_mpc)
    if bg["Count"] < 20:
        return i, None, None

    stake = np.array([
        bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
        bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"],
    ])
    sig_t, _ = signal(stake, BIN_LIMITS)
    dv = np.nan_to_num(sig_t, nan=0.0)

    R = bg["Radial Distance"]
    in_bin = (R[:, None] > BIN_LIMITS[:, 0]) & (R[:, None] < BIN_LIMITS[:, 1])
    cov_d = np.full(N_BINS, NOISE_FLOOR_COV)
    for b in range(N_BINS):
        m = in_bin[:, b]
        n_b = m.sum()
        if n_b >= 5:
            cov_d[b] = (
                SHAPE_NOISE * bg["Critical Density"][m].mean() / np.sqrt(n_b)
            ) ** 2
    return i, dv, cov_d


def simulate_lensing(truth, shapes, mis):
    global _TRUTH, _SHAPES, _OFF_MAG_MPC, _PHI_MIS, _DA_FAST

    z_tab = np.linspace(0.0, 3.0, 5000)
    chi_tab = _xlc.cosmology.comoving_distance(z_tab).value
    chi_of_z = interp1d(z_tab, chi_tab, kind="linear",
                        bounds_error=False, fill_value="extrapolate")

    def DA_fast(z1, z2):
        z2_a = np.atleast_1d(np.asarray(z2, dtype=float))
        res = (chi_of_z(z2_a) - float(chi_of_z(float(z1)))) / (1.0 + z2_a)
        return float(res[0]) if np.ndim(z2) == 0 else res

    _xlc.DA = DA_fast
    _xlsims.DA = DA_fast

    _TRUTH = truth
    _SHAPES = shapes
    _, _OFF_MAG_MPC, _PHI_MIS = mis
    _DA_FAST = DA_fast

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


# ── Plotting helpers ────────────────────────────────────────────────────────
def _wq(v, w, q):
    sorter = np.argsort(v)
    cs = np.cumsum(w[sorter]) / w.sum()
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
    z_med = float(np.median(truth["z"]))
    M_grid = np.logspace(13.5, 15.0, 200)
    C_dm0 = c_DuttonMaccio(z_med * np.ones(200), M_grid, c=1.0)
    axes[0].plot(np.log10(M_grid), C_dm0, "k-", lw=2,
                 label=rf"DM14 ($z={z_med:.2f}$, $\Delta m={DELTA_M_REF}$ mag)")
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
    q_maj, q_min = shapes["q_maj"], shapes["q_min"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(shapes["q_ba"], bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("b/a")
    axes[1].hist(shapes["q_ca"], bins=20, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("c/a")
    axes[2].hist(q_min / q_maj, bins=20, color="tomato", edgecolor="white")
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
        r_grid = np.linspace(0, off_mag_mpc.max() * 1.1, 300)
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
    idx = np.round(np.linspace(0, N_CLUSTERS - 1, 6)).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    cmap_dm = plt.cm.plasma
    dm_all = np.asarray(truth["delta_m"])
    vmin, vmax = 0.0, float(np.nanpercentile(dm_all, 95))
    for ax, i in zip(axes.ravel(), idx):
        errs = np.sqrt(np.diag(cov_matrices[i]))
        real = errs < np.sqrt(NOISE_FLOOR_COV)
        dm_i = float(truth["delta_m"][i])
        col = cmap_dm((dm_i - vmin) / (vmax - vmin))
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
    frac_res = (data_vectors - truth_profiles) / np.where(
        truth_profiles > 0, truth_profiles, np.nan
    )
    errs = np.sqrt(np.diagonal(cov_matrices, axis1=1, axis2=2))
    real_mask = errs < np.sqrt(NOISE_FLOOR_COV)
    frac_res_m = np.where(real_mask, frac_res, np.nan)
    med = np.nanmedian(frac_res_m, axis=0)
    p16 = np.nanpercentile(frac_res_m, 16, axis=0)
    p84 = np.nanpercentile(frac_res_m, 84, axis=0)
    sigma_res = (p84 - p16) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].fill_between(RADII, p16, p84, color="steelblue", alpha=0.25,
                         label="16-84%")
    axes[0].plot(RADII, med, "-o", color="steelblue", label="Median")
    axes[0].axhline(0.0, color="k", lw=0.5, ls=":")
    axes[0].axvline(R_MIN_FIT, color="crimson", lw=1.5, ls="--",
                    label=f"R_MIN_FIT = {R_MIN_FIT} Mpc")
    axes[0].set(xscale="log",
                xlabel="R [Mpc]", ylim=(-1.5, 1.5),
                ylabel=r"$(\Delta\Sigma_{\rm data} - \Delta\Sigma_{\rm NFW,truth})/\Delta\Sigma_{\rm NFW,truth}$",
                title="Per-cluster fractional residual")
    axes[0].legend(fontsize=9)

    axes[1].plot(RADII, sigma_res, "-o", color="tomato",
                 label="Triaxiality scatter (half 68% spread)")
    axes[1].axvline(R_MIN_FIT, color="crimson", lw=1.5, ls="--")
    axes[1].set(xscale="log", xlabel="R [Mpc]",
                ylabel=r"$\sigma_{\rm res}$",
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
        smooth=1.0, color="steelblue",
        hist_kwargs={"density": True},
    )
    ndim = points.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i, name in enumerate(PARAM_ORDER):
        ax = axes[i, i]
        xlim = ax.get_xlim()
        x_grid = np.linspace(xlim[0], xlim[1], 400)
        pdf_vals = PRIORS[name].pdf(x_grid)
        if np.any(np.isfinite(pdf_vals) & (pdf_vals > 0)):
            ax.plot(x_grid, pdf_vals, color="darkorange", lw=1.6, ls="--",
                    alpha=0.9, zorder=5)
    handles = [
        Line2D([0], [0], color="steelblue", lw=2,            label="Posterior"),
        Line2D([0], [0], color="darkorange", lw=1.6, ls="--", label="Prior"),
        Line2D([0], [0], color="crimson", lw=1.5,            label="Truth"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=10, frameon=True)
    fig.savefig(os.path.join(outdir, "corner.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mc_recovery_band(points, weights, truth, ols, outdir):
    M_grid = np.logspace(13.5, 15.0, 200)
    logM_g = np.log10(M_grid)
    z_med = float(np.median(truth["z"]))
    logz1_g = np.log10((1.0 + z_med) / (1.0 + _MC_Z_REF))
    c0_s    = points[:, 0]
    alpha_s = points[:, 1]
    beta_s  = points[:, 2]
    gamma_s = points[:, 3]
    rng_plot = np.random.default_rng(0)
    idx_band = rng_plot.choice(len(c0_s), size=300,
                               p=weights / weights.sum(), replace=True)
    dm_show = [0.5, DELTA_M_REF, 2.0]
    colors = ["royalblue", "steelblue", "teal"]
    colors_tru = ["navy", "darkblue", "darkcyan"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for dm_val, col, col_t in zip(dm_show, colors, colors_tru):
        log_dm_r = np.log10(max(dm_val, 1e-6) / DELTA_M_REF)
        label_dm = rf"$\Delta m = {dm_val:.1f}$ mag"
        C_band = np.array([
            10.0 ** (np.log10(c0_s[k])
                     + alpha_s[k] * (logM_g - _MC_M_PIVOT_LOG)
                     + beta_s[k]  * logz1_g
                     + gamma_s[k] * log_dm_r)
            for k in idx_band
        ])
        ax.fill_between(logM_g, C_band.min(0), C_band.max(0), alpha=0.20, color=col)
        c_piv_true = 1.0 * (max(dm_val, 1e-6) / DELTA_M_REF) ** TRUE_GAMMA
        C_true = c_DuttonMaccio(z_med * np.ones(200), M_grid, c=c_piv_true)
        ax.plot(logM_g, C_true, "--", color=col_t, lw=1.5,
                label=f"DM14 truth ({label_dm})")
    sc = ax.scatter(np.log10(truth["M_halo"]), truth["concentration"],
                    c=truth["delta_m"], cmap="plasma", s=5, alpha=0.25,
                    vmin=0, vmax=4, rasterized=True, zorder=0)
    plt.colorbar(sc, ax=ax, label=r"$\Delta m$ [mag]")
    ax.set(xlabel=r"$\log_{10}(M_{\rm halo})$", ylabel=r"$c_{200}$")
    ax.set_title("Combined M-c + γ recovery band  (radial cut $R \\geq 0.5$ Mpc)",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "mc_recovery_band.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_marginal(points, weights, outdir):
    gamma_s = points[:, 3]
    lo, med, hi = _wq(gamma_s, weights, 0.16), _wq(gamma_s, weights, 0.5), _wq(gamma_s, weights, 0.84)
    bias = (med - TRUE_GAMMA) / max((hi - lo) / 2.0, 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(gamma_s, weights=weights, bins=40, density=True,
            color="steelblue", alpha=0.65, edgecolor="white", label="Posterior")
    ax.axvspan(lo, hi, alpha=0.20, color="steelblue",
               label=rf"68% CI [{lo:.3f}, {hi:.3f}]")
    ax.axvline(med, color="steelblue", lw=1.5, ls="-",
               label=rf"Median $\hat\gamma={med:.3f}$")
    ax.axvline(TRUE_GAMMA, color="crimson", lw=2.0, ls="--",
               label=rf"True $\gamma={TRUE_GAMMA}$")
    xlim = ax.get_xlim()
    x_grid = np.linspace(xlim[0], xlim[1], 400)
    ax.plot(x_grid, PRIORS["gamma"].pdf(x_grid),
            color="darkorange", lw=2.0, ls="--", label="Prior")
    ax.set(xlabel=r"$\gamma$", ylabel="Density",
           title=rf"$\gamma$ marginal  ($N={N_CLUSTERS}$, bias={bias:+.2f}σ)")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "gamma_marginal.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return gamma_s, weights, lo, med, hi


# ── Single-realisation runner ───────────────────────────────────────────────
def run_one_realisation(idx, run_dir):
    subdir = os.path.join(run_dir, f"realisation_{idx:03d}")
    os.makedirs(subdir, exist_ok=True)

    summary_path = os.path.join(subdir, "summary.json")
    npz_path     = os.path.join(subdir, "gamma_posterior.npz")

    print("  building truth catalogue ...", flush=True)
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
    mis = compute_miscentering()
    is_mis, off_mag_mpc, phi_mis = mis

    print(f"  simulating lensing ({N_CLUSTERS} clusters) ...", flush=True)
    data_vectors, cov_matrices = simulate_lensing(truth, shapes, mis)

    cluster_obs, _, _ = make_test_data(
        truth, RADII, fractional_noise=0.2,
        poisson_richness=False, seed=_seed(1),
    )

    keep = RADII >= R_MIN_FIT
    RADII_fit = RADII[keep]
    data_vectors_fit = data_vectors[:, keep]
    cov_matrices_fit = cov_matrices[:, keep][:, :, keep]

    FMISS_FIT = np.full(N_CLUSTERS, 1.0 - F_CEN)
    SIGMA_OFF_FIT = np.full(N_CLUSTERS, SIGMA_OFF_MPC)

    # Nautilus checkpoint -- one HDF5 per realisation in the realisation
    # subdir, so a crashed sampler can resume from where it stopped.
    checkpoint = os.path.join(subdir, "nautilus_checkpoint.hdf5")
    print(
        f"  running sampler (verbose, checkpoint: {os.path.basename(checkpoint)})...",
        flush=True,
    )
    sampler = run_sampler_combined(
        cluster_obs, data_vectors_fit, cov_matrices_fit, RADII_fit,
        priors=PRIORS, delta_m_ref=DELTA_M_REF, fit_beta=True,
        FMISS=FMISS_FIT, SIGMA_OFF=SIGMA_OFF_FIT,
        n_live=N_LIVE, pool=N_WORKERS,
        verbose=True, filepath=checkpoint, resume=True, timeout= TIMEOUT,
    )
    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()

    # Per-realisation plots
    print("  rendering plots ...", flush=True)
    plot_truth_distributions(truth, subdir)
    plot_mc_scatter(truth, ols, subdir)
    plot_triaxial_shapes(shapes, subdir)
    plot_miscentering_offsets(mis, subdir)
    plot_sample_profiles(data_vectors, cov_matrices, truth, is_mis, off_mag_mpc, subdir)
    plot_triaxiality_diagnostic(truth, data_vectors, cov_matrices, subdir)
    plot_corner_with_priors(points, weights, ols, subdir)
    plot_mc_recovery_band(points, weights, truth, ols, subdir)
    gamma_s, w, lo, med, hi = plot_gamma_marginal(points, weights, subdir)

    # Save posterior samples for later overlay
    np.savez_compressed(npz_path, gamma_samples=gamma_s, weights=w)

    result = {
        "realisation": idx,
        "gamma_true": TRUE_GAMMA,
        "gamma_med": med,
        "gamma_lo":  lo,
        "gamma_hi":  hi,
        "sigma_gamma": (hi - lo) / 2.0,
        "n_clusters": N_CLUSTERS,
        **ols,
    }
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"  gamma_med = {med:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
        f"sigma_gamma = {(hi - lo) / 2.0:.4f}",
        flush=True,
    )
    return result


# ── Aggregate plots ─────────────────────────────────────────────────────────
def plot_gamma_residuals(results, run_dir):
    idx = np.array([r["realisation"] for r in results])
    gm  = np.array([r["gamma_med"]   for r in results])
    glo = np.array([r["gamma_lo"]    for r in results])
    ghi = np.array([r["gamma_hi"]    for r in results])
    yerr = np.vstack([gm - glo, ghi - gm])
    residuals = gm - TRUE_GAMMA

    # Calibration metrics
    mean_res    = np.mean(residuals)
    std_res     = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0
    mean_sigma  = np.mean((ghi - glo) / 2.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(idx, residuals, yerr=yerr, fmt="o",
                color="steelblue", capsize=4, markersize=6,
                label=r"Posterior median $\pm$68% CI")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="Unbiased")
    ax.axhspan(mean_res - std_res, mean_res + std_res, alpha=0.10, color="tomato",
               label=rf"$\sigma_{{\rm runs}} = {std_res:.4f}$")
    ax.axhline(mean_res, color="tomato", lw=1.5, ls=":",
               label=rf"Mean residual = {mean_res:+.4f}")
    ax.set(xlabel="Realisation index",
           ylabel=r"$\hat\gamma - \gamma_{\rm true}$",
           title=(rf"$\gamma$ residuals across {len(results)} realisations  "
                  rf"($\gamma_{{\rm true}}={TRUE_GAMMA}$, "
                  rf"$N={N_CLUSTERS}$, MASTER_SEED = None)"))
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    # Annotate the error-bar calibration test
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
    # Load posterior samples per realisation; build a smoothed density.
    fig, ax = plt.subplots(figsize=(9, 5))
    x_grid = np.linspace(-0.3, 0.7, 600)
    cmap = plt.cm.viridis
    n = len(results)
    for k, r in enumerate(results):
        npz_path = os.path.join(
            run_dir, f"realisation_{r['realisation']:03d}", "gamma_posterior.npz"
        )
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        samples = data["gamma_samples"]
        weights = data["weights"]
        # Weighted Gaussian KDE
        try:
            kde = gaussian_kde(samples, weights=weights, bw_method=0.15)
            y = kde(x_grid)
        except Exception:
            # Fallback: density histogram
            y, edges = np.histogram(samples, weights=weights, bins=40,
                                    range=(x_grid[0], x_grid[-1]), density=True)
            x_grid_h = 0.5 * (edges[1:] + edges[:-1])
            ax.plot(x_grid_h, y, color=cmap(k / max(n - 1, 1)), alpha=0.6, lw=1.0)
            continue
        ax.plot(x_grid, y, color=cmap(k / max(n - 1, 1)), alpha=0.6, lw=1.0)

    # Prior overlay
    ax.plot(x_grid, PRIORS["gamma"].pdf(x_grid),
            color="darkorange", lw=1.8, ls="--", label="Prior")
    ax.axvline(TRUE_GAMMA, color="crimson", lw=2.0, ls="--",
               label=rf"$\gamma_{{\rm true}}={TRUE_GAMMA}$")
    ax.set(xlabel=r"$\gamma$", ylabel="Posterior density",
           title=(rf"$\gamma$ posterior overlay across {n} realisations  "
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


def print_summary(results):
    print("\n" + "=" * 70, flush=True)
    print(
        f'{"idx":>4}  {"gamma_med":>11}  {"-1σ":>9}  {"+1σ":>9}  {"sigma":>9}  {"bias [σ]":>10}',
        flush=True,
    )
    print("-" * 70, flush=True)
    for r in results:
        bias = (r["gamma_med"] - TRUE_GAMMA) / max(r["sigma_gamma"], 1e-12)
        print(
            f'{r["realisation"]:>4d}  {r["gamma_med"]:>+11.4f}  '
            f'{r["gamma_med"] - r["gamma_lo"]:>9.4f}  '
            f'{r["gamma_hi"] - r["gamma_med"]:>9.4f}  '
            f'{r["sigma_gamma"]:>9.4f}  '
            f'{bias:>+10.2f}',
            flush=True,
        )


def main():
    print(
        f"Gamma repeat: N_REALISATIONS={N_REALISATIONS}, "
        f"TRUE_GAMMA={TRUE_GAMMA}, MASTER_SEED={MASTER_SEED}, "
        f"N_CLUSTERS={N_CLUSTERS}, N_WORKERS={N_WORKERS}",
        flush=True,
    )
    print(f"Output directory: {RUN_DIR}", flush=True)

    # The DES Y3 n(z) plot only needs to be drawn once.
    plot_des_y3_nz(RUN_DIR)

    results = []
    durations = []                         # freshly-computed iteration times only
    total = N_REALISATIONS
    run_t0 = time.time()
    for k in range(total):
        subdir = os.path.join(RUN_DIR, f"realisation_{k:03d}")
        summary_path = os.path.join(subdir, "summary.json")
        npz_path     = os.path.join(subdir, "gamma_posterior.npz")
        if os.path.exists(summary_path) and os.path.exists(npz_path):
            print(
                f"\n[{k + 1}/{total}] realisation_{k:03d} -- cached, loading",
                flush=True,
            )
            with open(summary_path) as f:
                results.append(json.load(f))
            continue

        print(f"\n[{k + 1}/{total}] realisation_{k:03d}", flush=True)
        iter_t0 = time.time()
        try:
            r = run_one_realisation(k, RUN_DIR)
        except Exception as exc:
            print(f"  FAILED for realisation {k}:", flush=True)
            traceback.print_exc()
            continue
        iter_dt = time.time() - iter_t0
        durations.append(iter_dt)
        results.append(r)

        remaining = total - (k + 1)
        avg = sum(durations) / len(durations)
        eta = avg * remaining
        print(
            f"  [{k + 1}/{total}] iteration took {_fmt_duration(iter_dt)}; "
            f"avg {_fmt_duration(avg)}; "
            f"ETA {_fmt_duration(eta)} for {remaining} remaining iteration"
            f"{'s' if remaining != 1 else ''}",
            flush=True,
        )

    total_elapsed = time.time() - run_t0
    print(
        f"\nTotal wall time: {_fmt_duration(total_elapsed)}  "
        f"({len(durations)} fresh realisation"
        f"{'s' if len(durations) != 1 else ''} timed)",
        flush=True,
    )

    if not results:
        print("No successful realisations; nothing to aggregate.", file=sys.stderr)
        return 1

    plot_gamma_residuals(results, RUN_DIR)
    plot_posteriors_overlay(results, RUN_DIR)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
