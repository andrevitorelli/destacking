"""
Gamma scan based on combined_recovery_lensing_triaxial_unbiased.ipynb.

For each value in GAMMA_VALUES the script repeats the full notebook pipeline:

  1. Build a DES~Y3-like cluster truth catalogue with that gamma imprinted as
     a direct log10(c) shift.
  2. Assign CDM-like triaxial halo shapes with isotropic LOS orientations.
  3. Draw Rayleigh + delta-function miscentering offsets.
  4. Simulate triaxial NFW lensing on a DES~Y3-like source field (parallel
     multiprocess pool).
  5. Build the cluster observable table (deterministic richness, no Eddington
     bias).
  6. Apply the radial cut R >= R_MIN_FIT.
  7. Run the combined nautilus sampler with explicit priors (tight DM14
     Gaussians on c0, alpha, beta; broad N(0,1) on gamma) and a fixed
     population miscentering correction in the forward model
     (FMISS = 1 - F_CEN, SIGMA_OFF = SIGMA_OFF_MPC).
  8. Extract the gamma posterior median and 68% credible interval and cache
     the summary to disk as JSON.

After all gammas are processed, two plots are produced inside the run-specific
output directory (see RUN_DIR below):

  <RUN_DIR>/gamma_recovered_vs_true.png   gamma_true vs gamma_post
  <RUN_DIR>/gamma_residuals.png           residual gamma_post - gamma_true

Configuration is exposed via environment variables:

  N_CLUSTERS    Number of clusters per run     (default 10000)
  GAMMA_VALUES  Comma-separated gamma list     (default -0.4,-0.2,0.0,0.2,0.4)
  N_WORKERS     Pool workers for lensing/sampler (default 15)
  MASTER_SEED   Master RNG seed (None or int)  (default None — fresh entropy
                every RNG; set to an integer for a fully reproducible run)
  RUN_DIR       Explicit output directory      (default
                gamma_scan_outputs/run_YYYY-MM-DDTHH-MM-SS).  Each invocation
                creates a fresh timestamped subdirectory so multiple scans do
                not clobber each other.  Point RUN_DIR at an existing folder
                to resume a previously-cached scan.

Cached results in <RUN_DIR>/gamma_*.json are reloaded on subsequent
invocations of the same RUN_DIR, so a failed run can be resumed by passing
RUN_DIR=<previous folder>.
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
import multiprocess as mp
import numpy as np
from astropy.cosmology import Planck18 as _planck18
from scipy.interpolate import interp1d
from scipy.stats import norm as sp_norm

import ngmix

from destacking.fitting import run_sampler_combined
from destacking.model import _MC_M_PIVOT_LOG, _MC_Z_REF
from destacking.simulation import (
    generate_truth_catalog,
    make_test_data,
    des_y3_cluster_pdf,
    sample_des_y3_cluster_z,
    des_y3_source_pdf,
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

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
# ── Configuration (mirrors the notebook) ────────────────────────────────────
N_CLUSTERS = 5000#int(os.environ.get("N_CLUSTERS", 10000))
GAMMA_VALUES = np.linspace(-.5,.5,20)
# GAMMA_VALUES = [
#     float(g) for g in os.environ.get("GAMMA_VALUES", "-0.4,-0.2,0.0,0.2,0.4").split(",")
# ]
# ── Centralised seeds ──
# MASTER_SEED defaults to None so that every RNG in the pipeline draws fresh
# OS entropy (a different realisation per invocation, just like the notebook
# default).  Set the MASTER_SEED env var to an integer for a fully
# reproducible run; every other seed is derived from this single value via
# the _seed() helper below.
def _parse_master_seed(env_value):
    if env_value is None:
        return None
    s = env_value.strip().lower()
    if s in ("", "none", "null"):
        return None
    return int(s)

MASTER_SEED = _parse_master_seed(os.environ.get("MASTER_SEED"))

def _seed(offset):
    """MASTER_SEED + offset, or None when MASTER_SEED is None."""
    return None if MASTER_SEED is None else MASTER_SEED + int(offset)

N_WORKERS = int(os.environ.get("N_WORKERS", 15))

DELTA_M_REF = 0.1
RADII = np.logspace(-0.8, 0.8, 12)
R_MIN_FIT = 0.5

# DES Y3 redshift distributions
DES_Y3_Z_CLUSTER_RANGE = (0.2, 0.65)
DES_Y3_Z_SRC_ALPHA = 2.0
DES_Y3_Z_SRC_BETA = 1.5
DES_Y3_Z_SRC_Z0 = 0.4
N_OVERSAMPLE_CLUSTER = 5

# Miscentering (simulation + fitting use the same values)
F_CEN = 0.64
SIGMA_OFF_MPC_H = 0.32
H_PLANCK = float(_planck18.h)
SIGMA_OFF_MPC = SIGMA_OFF_MPC_H / H_PLANCK

# Survey
DENSITY = 30
WIDTH_RAD = 0.006
Z_SRC_RANGE = (0.0, 2.5)
SHAPE_NOISE = 0.27
NOISE_FLOOR_COV = 1e4

# Radial bin layout
_log_r = np.log10(RADII)
_dlog = np.diff(_log_r)[0]
_edges = np.logspace(_log_r[0] - _dlog / 2, _log_r[-1] + _dlog / 2, len(RADII) + 1)
BIN_LIMITS = np.column_stack([_edges[:-1], _edges[1:]])
N_BINS = len(RADII)

# Priors (matching the notebook).  c0 centred on the DM14 prediction
# c_DM14(M=1e14, z=z_ref=0.35) = 4.571 — Fix A from the 2026-05-19 bias report
# — so the c0-gamma degeneracy no longer leaks a misaligned c0 centroid into
# a low gamma bias.
PRIORS = {
    "c0":         sp_norm(loc=4.57, scale=0.05),
    "alpha":      sp_norm(loc=-0.1, scale=0.05),
    "beta":       sp_norm(loc=-0.4, scale=0.2),
    "gamma":      sp_norm(loc=0.0,  scale=1.0),
    "log_M_norm": sp_norm(loc=0.0,  scale=0.1),
}

N_LIVE = 400

# Output directory: timestamped per run so successive scans do not clobber
# each other.  Override via the RUN_DIR env var to resume a previous run
# (the script will reload its cached gamma_*.json files from there).
_DEFAULT_RUN_TS = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
OUTDIR = os.environ.get(
    "RUN_DIR", os.path.join("gamma_scan_outputs", f"run_{_DEFAULT_RUN_TS}")
)
os.makedirs(OUTDIR, exist_ok=True)


# DES Y3 distribution functions are imported from destacking.simulation.


# ── Pipeline stages ─────────────────────────────────────────────────────────
def build_truth_catalog(gamma_true):
    n_pool = N_CLUSTERS * N_OVERSAMPLE_CLUSTER
    truth_pool = generate_truth_catalog(
        n_pool,
        z_range=DES_Y3_Z_CLUSTER_RANGE,
        logM_range=(13.5, 15.0),
        delta_m_gamma=0.0,
        scatter_logc=0.0,
        seed=_seed(0),
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
    truth_params = {
        "c0_true": float(10.0 ** coeffs[0]),
        "alpha_true": float(coeffs[1]),
        "beta_true": float(coeffs[2]),
        "gamma_true_ols": float(coeffs[3]),
    }
    return truth, truth_params


def compute_triaxial_shapes(truth):
    """Thin wrapper around xlensing.sims.sample_triaxial_shapes."""
    rng_tri = np.random.default_rng(_seed(10))
    s = sample_triaxial_shapes(
        truth["z"], truth["M_halo"],
        ba_range=(0.5, 1.0), ca_lo=0.3, isotropic_los=True, rng=rng_tri,
    )
    return s["r_vir"], s["q_ba"], s["q_ca"], s["theta_los"], s["phi_los"]


def compute_miscentering():
    rng_mis = np.random.default_rng(_seed(20))
    is_mis = rng_mis.uniform(0, 1, N_CLUSTERS) > F_CEN
    off_mag_mpc = np.where(
        is_mis, rng_mis.rayleigh(scale=SIGMA_OFF_MPC, size=N_CLUSTERS), 0.0
    )
    phi_mis = rng_mis.uniform(0, 2 * np.pi, N_CLUSTERS)
    return is_mis, off_mag_mpc, phi_mis


# Module-level globals consumed by _process_cluster (set before each Pool dispatch)
_TRUTH = None
_R_VIR_VALS = None
_Q_BA = None
_Q_CA = None
_THETA_LOS = None
_PHI_LOS = None
_OFF_MAG_MPC = None
_PHI_MIS = None
_DA_FAST = None


def _process_cluster(i):
    z_i = float(_TRUTH["z"][i])
    M_i = float(_TRUTH["M_halo"][i])
    c_i = float(_TRUTH["concentration"][i])
    a_i = _R_VIR_VALS[i]

    # Independent random background catalogue for this cluster with Poisson density.
    gal_i = make_simple_random_cat(
        DENSITY, WIDTH_RAD, Z_SRC_RANGE, shape_noise=0.0,
        seed=_seed(4000 + i), poisson_density=True,
    ).T
    n_gals = len(gal_i)

    # Override source z with DES Y3 source n(z).
    rng_src = np.random.default_rng(_seed(5000 + i))
    gal_i[:, 2] = sample_des_y3_source_z(
        n_gals, rng_src,
        z_min=Z_SRC_RANGE[0], z_max=Z_SRC_RANGE[1],
        alpha=DES_Y3_Z_SRC_ALPHA, beta=DES_Y3_Z_SRC_BETA, z0=DES_Y3_Z_SRC_Z0,
    )

    # Lognormal-size based metacal responses.
    rng_resp = np.random.default_rng(_seed(6000 + i))
    sizes  = rng_resp.lognormal(mean=np.log(0.5), sigma=0.5, size=n_gals)
    R_RESP = np.clip(1.0 - (sizes / 1.5) ** 2, 0.05, 0.99)
    gal_i[:, 5] = R_RESP
    gal_i[:, 6] = 0.0
    gal_i[:, 7] = 0.0
    gal_i[:, 8] = R_RESP

    # Intrinsic shapes drawn AFTER n_gals is known.
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
        0.0, 0.0, z_i, M_i, c_i,
        a_i, _Q_BA[i] * a_i, _Q_CA[i] * a_i,
        _THETA_LOS[i], _PHI_LOS[i],
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

    stake = np.array(
        [
            bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
            bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"],
        ]
    )
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
    """Run the triaxial-NFW lensing simulation; return (data, cov).

    Each cluster gets an independent random background catalogue drawn inside
    the worker with Poisson density (no shared base catalogue across clusters).
    """
    global _TRUTH, _R_VIR_VALS, _Q_BA, _Q_CA
    global _THETA_LOS, _PHI_LOS, _OFF_MAG_MPC, _PHI_MIS, _DA_FAST

    # Comoving-distance spline patch (xlensing DA replacement)
    z_tab = np.linspace(0.0, 3.0, 5000)
    chi_tab = _xlc.cosmology.comoving_distance(z_tab).value
    chi_of_z = interp1d(
        z_tab, chi_tab, kind="linear", bounds_error=False, fill_value="extrapolate"
    )

    def DA_fast(z1, z2):
        z2_a = np.atleast_1d(np.asarray(z2, dtype=float))
        res = (chi_of_z(z2_a) - float(chi_of_z(float(z1)))) / (1.0 + z2_a)
        return float(res[0]) if np.ndim(z2) == 0 else res

    _xlc.DA = DA_fast
    _xlsims.DA = DA_fast

    _TRUTH = truth
    _R_VIR_VALS, _Q_BA, _Q_CA, _THETA_LOS, _PHI_LOS = shapes
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


def run_one_gamma(gamma_true):
    """Full single-gamma pipeline. Returns posterior-summary dict.

    All RNG seeds in the pipeline are derived from the module-level
    MASTER_SEED via the _seed() helper.
    """
    print(f"\n=== gamma_true = {gamma_true:+.3f} ===", flush=True)

    truth, truth_params = build_truth_catalog(gamma_true)
    print(
        f"  Truth params: c0={truth_params['c0_true']:.3f}, "
        f"alpha={truth_params['alpha_true']:.4f}, "
        f"beta={truth_params['beta_true']:.4f}, "
        f"gamma_ols={truth_params['gamma_true_ols']:.4f}",
        flush=True,
    )

    shapes = compute_triaxial_shapes(truth)
    mis = compute_miscentering()

    print(f"  Running lensing simulation ({N_CLUSTERS} clusters)...", flush=True)
    data_vectors, cov_matrices = simulate_lensing(truth, shapes, mis)

    cluster_obs, _, _ = make_test_data(
        truth, RADII, fractional_noise=0.2, poisson_richness=False,
        seed=_seed(1),
    )

    keep = RADII >= R_MIN_FIT
    RADII_fit = RADII[keep]
    data_vectors_fit = data_vectors[:, keep]
    cov_matrices_fit = cov_matrices[:, keep][:, :, keep]

    FMISS_FIT = np.full(N_CLUSTERS, 1.0 - F_CEN)
    SIGMA_OFF_FIT = np.full(N_CLUSTERS, SIGMA_OFF_MPC)

    # Nautilus checkpoint file -- one per gamma so a crashed sampler can
    # resume from where it stopped instead of restarting from scratch.
    checkpoint = os.path.join(OUTDIR, f"checkpoint_{gamma_true:+.3f}.hdf5")
    print(
        f"  Running sampler (verbose, checkpoint: {os.path.basename(checkpoint)})...",
        flush=True,
    )
    sampler = run_sampler_combined(
        cluster_obs,
        data_vectors_fit,
        cov_matrices_fit,
        RADII_fit,
        priors=PRIORS,
        delta_m_ref=DELTA_M_REF,
        fit_beta=True,
        FMISS=FMISS_FIT,
        SIGMA_OFF=SIGMA_OFF_FIT,
        n_live=N_LIVE,
        pool=N_WORKERS,
        verbose=True,
        filepath=checkpoint,
        resume=True,
    )

    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()

    gamma_samples = points[:, 3]  # order: c0, alpha, beta, gamma, log_M_norm

    def wq(v, q):
        sorter = np.argsort(v)
        cs = np.cumsum(weights[sorter]) / weights.sum()
        return np.interp(q, cs, v[sorter])

    lo = float(wq(gamma_samples, 0.16))
    med = float(wq(gamma_samples, 0.5))
    hi = float(wq(gamma_samples, 0.84))

    result = {
        "gamma_true": float(gamma_true),
        "gamma_med": med,
        "gamma_lo": lo,
        "gamma_hi": hi,
        "sigma_gamma": float((hi - lo) / 2.0),
        "bias_sigma": float((med - gamma_true) / ((hi - lo) / 2.0)),
        "n_clusters": N_CLUSTERS,
        "n_live": N_LIVE,
        **truth_params,
    }
    print(
        f"  gamma_recovered = {med:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
        f"bias = {result['bias_sigma']:+.2f}σ",
        flush=True,
    )
    return result


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_results(results):
    gt = np.array([r["gamma_true"] for r in results])
    gm = np.array([r["gamma_med"] for r in results])
    glo = np.array([r["gamma_lo"] for r in results])
    ghi = np.array([r["gamma_hi"] for r in results])
    yerr = np.vstack([gm - glo, ghi - gm])

    # Plot 1: gamma_true vs gamma_recovered
    fig, ax = plt.subplots(figsize=(7, 6))
    g_lo = min(gt.min(), gm.min()) - 0.1
    g_hi = max(gt.max(), gm.max()) + 0.1
    ax.plot([g_lo, g_hi], [g_lo, g_hi], "k--", lw=1, label="Identity")
    ax.errorbar(
        gt, gm, yerr=yerr, fmt="o", color="steelblue",
        capsize=4, markersize=6, label="Posterior (68% CI)",
    )
    ax.set_xlabel(r"$\gamma_{\rm true}$", fontsize=13)
    ax.set_ylabel(r"$\hat\gamma$ (posterior median)", fontsize=13)
    ax.set_title(
        rf"$\gamma$ recovery scan  ($N={N_CLUSTERS}$ clusters per run, "
        rf"radial cut $R \geq {R_MIN_FIT}$\,Mpc)",
        fontsize=11,
    )
    ax.set_xlim(g_lo, g_hi)
    ax.set_ylim(g_lo, g_hi)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out1 = f"{OUTDIR}/gamma_recovered_vs_true.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}", flush=True)

    # Plot 2: residuals
    fig, ax = plt.subplots(figsize=(7, 4.5))
    residuals = gm - gt
    ax.axhline(0.0, color="k", ls="--", lw=1, label="Unbiased")
    ax.errorbar(
        gt, residuals, yerr=yerr, fmt="o", color="steelblue",
        capsize=4, markersize=6,
    )
    ax.set_xlabel(r"$\gamma_{\rm true}$", fontsize=13)
    ax.set_ylabel(r"$\hat\gamma - \gamma_{\rm true}$", fontsize=13)
    ax.set_title(
        rf"$\gamma$ residuals  ($N={N_CLUSTERS}$ clusters per run)",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out2 = f"{OUTDIR}/gamma_residuals.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out2}", flush=True)


def main():
    print(
        f"Gamma scan: N_CLUSTERS={N_CLUSTERS}, GAMMA_VALUES={GAMMA_VALUES}, "
        f"MASTER_SEED={MASTER_SEED}, N_WORKERS={N_WORKERS}",
        flush=True,
    )
    print(f"Output directory: {OUTDIR}", flush=True)
    results = []
    durations = []                         # freshly-computed iteration times only
    total = len(GAMMA_VALUES)
    scan_t0 = time.time()
    for k, g in enumerate(GAMMA_VALUES):
        cache_file = f"{OUTDIR}/gamma_{g:+.3f}.json"
        if os.path.exists(cache_file):
            print(
                f"\n[{k + 1}/{total}] gamma_true = {g:+.3f} -- cached, loading",
                flush=True,
            )
            with open(cache_file) as f:
                r = json.load(f)
            results.append(r)
            continue

        iter_t0 = time.time()
        try:
            r = run_one_gamma(g)
        except Exception as exc:  # don't kill the whole scan
            print(f"  FAILED for gamma_true = {g:+.3f}:", flush=True)
            traceback.print_exc()
            continue
        iter_dt = time.time() - iter_t0
        durations.append(iter_dt)

        with open(cache_file, "w") as f:
            json.dump(r, f, indent=2)
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

    total_elapsed = time.time() - scan_t0
    print(
        f"\nScan total wall time: {_fmt_duration(total_elapsed)}  "
        f"({len(durations)} fresh iteration"
        f"{'s' if len(durations) != 1 else ''} timed)",
        flush=True,
    )

    if not results:
        print("No successful runs; nothing to plot.", file=sys.stderr)
        return 1

    plot_results(results)

    # Summary table
    print("\n" + "=" * 60, flush=True)
    print(
        f'{"gamma_true":>11}  {"gamma_med":>11}  {"-1σ":>10}  {"+1σ":>10}  {"bias [σ]":>10}',
        flush=True,
    )
    print("-" * 60, flush=True)
    for r in results:
        print(
            f'{r["gamma_true"]:>+11.4f}  {r["gamma_med"]:>+11.4f}  '
            f'{r["gamma_med"] - r["gamma_lo"]:>10.4f}  '
            f'{r["gamma_hi"] - r["gamma_med"]:>10.4f}  '
            f'{r["bias_sigma"]:>+10.2f}',
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
