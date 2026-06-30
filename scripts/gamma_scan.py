"""
Gamma scan with optional post-hoc bootstrap.

Design
------
  GAMMA_VALUES = np.linspace(-1, +1, 11)

  for each gamma_true in GAMMA_VALUES:          # 11 independent simulations
      simulate N_CLUSTERS=10000 clusters       → saved to disk
      for b = 0 .. N_BOOTSTRAP-1:              # 50 bootstrap draws per gamma
          idx  = bootstrap resample of {0..N_CLUSTERS-1}     (BOOTSTRAP=1)
                 OR np.arange(N_CLUSTERS)  full sample       (BOOTSTRAP=0)
          run nautilus on data[idx]  →  γ̂^b
      gamma_ensemble = median_b(γ̂^b)
      sigma_boot     = std_b(γ̂^b)   (= 0 when BOOTSTRAP=0 and N_BOOTSTRAP=1)

  Final output:
      gamma_scan_results.png    two-panel plot: γ_true vs γ_est + residuals
      scan_summary.json         all results
      progress.json             live progress (updated after each gamma)

Resumability
------------
  Any interruption is safe.  Re-run the EXACT same command (or set
  RUN_DIR=<existing-dir>) to continue from where it stopped.

  Per-gamma directory layout:
    simulation_g{k:02d}_gamma{v:+.1f}/
      simulated_data.npz      data_vectors, cov_matrices  (skip re-sim)
      truth_catalog.npz       z, M_halo, concentration, delta_m
      cluster_obs.npz         z, Lambda, delta_m
      ols.json                OLS truth coefficients
      ensemble_summary.json   final per-gamma result  (skip gamma if present)
      bootstrap_{b:03d}/
        summary.json          per-draw result  (skip draw if present)
        gamma_posterior.npz

Configuration via environment variables
----------------------------------------
  N_CLUSTERS    Clusters per simulation         (default 10000)
  N_GAMMA       Number of γ values in scan      (default 11; evenly spaced
                in [-1, +1] inclusive)
  N_BOOTSTRAP   Bootstrap draws per gamma       (default 50; forced to 1
                when BOOTSTRAP=0)
  N_WORKERS     Pool workers                    (default 15)
  N_RESAMPLE    Bootstrap draw size             (default N_CLUSTERS;
                ignored when BOOTSTRAP=0)
  BOOTSTRAP     1/0 — enable post-hoc bootstrap (default 1).  When 0 the
                full N_CLUSTERS sample is used once per gamma with no
                resampling; N_BOOTSTRAP is forced to 1.
  RUN_DIR       Output directory                (default
                gamma_scan_outputs/run_YYYY-MM-DDTHH-MM-SS)
                Set to an existing path to resume a previous run.
"""

from __future__ import annotations

import datetime
import json
import os
import pickle
import sys
import time
import traceback

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
    apply_NFW_shear_region,
    apply_triaxial_NFW_shear_region,
    make_simple_random_cat,
    r_vir,
    sample_triaxial_shapes,
)

try:
    import corner
except ImportError:
    corner = None

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Scan definition ───────────────────────────────────────────────────────────
# ── YAML configuration ────────────────────────────────────────────────────────
# Override the defaults below by either (i) passing a YAML file as the first
# CLI argument or (ii) setting the CONFIG environment variable to its path.
# YAML wins over the CONFIG env-var, which wins over the per-knob env vars,
# which win over the hard-coded defaults.  A template is at
# scripts/gamma_scan_config_example.yaml.
def _load_yaml_config():
    path = None
    # Allow sys.argv[1] when it does not look like an internal nautilus arg
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.yaml')
                              or sys.argv[1].endswith('.yml')):
        path = sys.argv[1]
    elif os.environ.get("CONFIG", "").strip():
        path = os.environ["CONFIG"].strip()
    if path is None:
        return {}
    try:
        import yaml as _yaml
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot load config {path}: PyYAML not installed ({exc!r})"
        )
    with open(path) as fh:
        cfg = _yaml.safe_load(fh) or {}
    print(f"[gamma_scan] loaded YAML config from {path}", flush=True)
    return cfg

_CFG = _load_yaml_config()


def _cfg(key, env_name, default, cast=None):
    """YAML > env var > default.  Cast applied for env-var path only; YAML
    values are taken as-is (PyYAML already gives ints/floats/bools/lists)."""
    if key in _CFG:
        return _CFG[key]
    if env_name and env_name in os.environ:
        v = os.environ[env_name]
        return cast(v) if cast else v
    return default


N_GAMMA      = int(_cfg("n_gamma", "N_GAMMA", 51, cast=int))
_GAMMA_RANGE = _cfg("gamma_range", None, [-1.0, 1.0])
GAMMA_VALUES = np.linspace(float(_GAMMA_RANGE[0]), float(_GAMMA_RANGE[1]),
                           N_GAMMA)

# ── Configuration ─────────────────────────────────────────────────────────────
N_CLUSTERS  = int(_cfg("n_clusters",  "N_CLUSTERS",  15000, cast=int))
N_BOOTSTRAP = int(_cfg("n_bootstrap", "N_BOOTSTRAP", 1,     cast=int))
N_WORKERS   = int(_cfg("n_workers",   "N_WORKERS",   15,    cast=int))

_n_resample_env = os.environ.get("N_RESAMPLE", "")
N_RESAMPLE = int(_n_resample_env) if _n_resample_env.strip() else None

def _yaml_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")

# triaxiality: true (default) = triaxial NFW; false = pure spherical.
TRIAXIALITY  = _yaml_bool(_cfg("triaxiality",
                               "TRIAXIALITY",
                               not _yaml_bool(os.environ.get("SPHERICAL", "0"))))
SPHERICAL    = not TRIAXIALITY
MISCENTERING = _yaml_bool(_cfg("miscentering", "MISCENTERING", True))
BOOTSTRAP = os.environ.get("BOOTSTRAP", "0").strip().lower() \
            not in ("0", "false", "no", "off")
if not BOOTSTRAP:
    N_BOOTSTRAP = 1

DELTA_M_REF = 0.1
RADII       = np.logspace(-0.8, 0.6, 8)
R_MIN_FIT   = 0.5

_Z_CL_RANGE              = _cfg("z_cluster_range", None, [0.10, 0.60])
DES_Y3_Z_CLUSTER_RANGE = (float(_Z_CL_RANGE[0]), float(_Z_CL_RANGE[1]))
WAZP_LOGM_RANGE        = (14.0, 15.0)
WAZP_LAMBDA_MIN        = 25.0
DES_Y3_Z_SRC_ALPHA     = float(_cfg("smail_alpha", "SMAIL_ALPHA", 2.0,
                                    cast=float))
DES_Y3_Z_SRC_BETA      = float(_cfg("smail_beta",  "SMAIL_BETA",  1.5,
                                    cast=float))
DES_Y3_Z_SRC_Z0        = float(_cfg("smail_z0",    "SMAIL_Z0",    0.4,
                                    cast=float))
N_OVERSAMPLE_CLUSTER   = 5

F_CEN           = 0.64
SIGMA_OFF_MPC_H = 0.32
H_PLANCK        = float(_planck18.h)
SIGMA_OFF_MPC   = SIGMA_OFF_MPC_H / H_PLANCK

DENSITY         = float(_cfg("density", "DENSITY", 7.5, cast=float))
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
    # NB: A (the global multiplicative amplitude correction) is omitted
    # so that destacking.fitting.run_sampler_combined fixes it at 1.0.
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
    os.path.join("gamma_scan_outputs", f"run_{_DEFAULT_RUN_TS}"),
)
os.makedirs(RUN_DIR, exist_ok=True)

# ── Utilities ─────────────────────────────────────────────────────────────────

def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    return f"{int(h)}h{int(rem/60):02d}m"


def _seed(g_idx, offset):
    """Deterministic seed unique per (gamma-index, offset) — ensures same
    simulation data on resume even across restarts."""
    return g_idx * 100_000 + int(offset)


def gamma_dir_name(g_idx, gamma_true):
    # Directory names are only a human-readable label; g_idx already makes
    # them unique. Plot scripts must read the exact gamma_true value from
    # the per-bootstrap summary.json, not parse it from this name.
    return f"simulation_g{g_idx:02d}_gamma{gamma_true:+.1f}"


def _wq(v, w, q):
    sorter = np.argsort(v)
    cs     = np.cumsum(w[sorter]) / w.sum()
    return float(np.interp(q, cs, v[sorter]))


# ── Simulation pipeline ───────────────────────────────────────────────────────

def build_truth_catalog(gamma_true, g_idx):
    # WaZP-matched truth: oversample at logM∈[14,15], then keep the rows
    # passing Λ>WAZP_LAMBDA_MIN and Δm>0 until N_CLUSTERS pass the cut.
    # The McClintock+18 Λ-inversion is run inside make_test_data; we
    # reuse it here purely as a selection filter and discard the noisy
    # ΔΣ vectors that step also produces.
    from astropy.table import vstack as _vstack

    batch_size = max(int(N_CLUSTERS / 0.35), 5000)
    selected   = []
    n_sel      = 0
    batch      = 0
    while n_sel < N_CLUSTERS:
        batch += 1
        truth_b = generate_truth_catalog(
            batch_size,
            z_range=DES_Y3_Z_CLUSTER_RANGE,
            logM_range=WAZP_LOGM_RANGE,
            delta_m_gamma=0.0, scatter_logc=0.0,
            seed=_seed(g_idx, batch),
        )
        cluster_obs_b, _, _ = make_test_data(
            truth_b, RADII, poisson_richness=True,
            seed=_seed(g_idx, 50_000 + batch),
        )
        truth_b["Lambda"] = cluster_obs_b["Lambda"]
        keep = ((truth_b["Lambda"]  > WAZP_LAMBDA_MIN)
              & (truth_b["delta_m"] > 0.0)
              & np.isfinite(truth_b["delta_m"])
              & (truth_b["z"] < DES_Y3_Z_CLUSTER_RANGE[1])
              & (truth_b["z"] >= DES_Y3_Z_CLUSTER_RANGE[0]))
        truth_b = truth_b[keep]
        selected.append(truth_b)
        n_sel += len(truth_b)
    truth = _vstack(selected)[:N_CLUSTERS]

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


def compute_triaxial_shapes(truth, g_idx):
    rng_tri = np.random.default_rng(_seed(g_idx, 10))
    return sample_triaxial_shapes(
        truth["z"], truth["M_halo"],
        ba_range=(0.5, 1.0), ca_lo=0.3, isotropic_los=True, rng=rng_tri,
    )


def compute_miscentering(g_idx):
    if not MISCENTERING:
        # All clusters perfectly centred — used for the no-miscentering
        # sensitivity test selected via miscentering: false in YAML.
        return (np.zeros(N_CLUSTERS, dtype=bool),
                np.zeros(N_CLUSTERS),
                np.zeros(N_CLUSTERS))
    rng_mis     = np.random.default_rng(_seed(g_idx, 20))
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
_G_IDX_SIM   = 0


def _process_cluster(i):
    z_i = float(_TRUTH["z"][i])
    M_i = float(_TRUTH["M_halo"][i])
    c_i = float(_TRUTH["concentration"][i])
    a_i = _SHAPES["r_vir"][i]
    g_i = _G_IDX_SIM

    gal_i  = make_simple_random_cat(
        DENSITY, WIDTH_RAD, Z_SRC_RANGE, shape_noise=0.0,
        seed=_seed(g_i, 4000 + i), poisson_density=True,
    ).T
    n_gals = len(gal_i)

    rng_src = np.random.default_rng(_seed(g_i, 5000 + i))
    gal_i[:, 2] = sample_des_y3_source_z(
        n_gals, rng_src,
        z_min=Z_SRC_RANGE[0], z_max=Z_SRC_RANGE[1],
        alpha=DES_Y3_Z_SRC_ALPHA, beta=DES_Y3_Z_SRC_BETA, z0=DES_Y3_Z_SRC_Z0,
    )

    rng_resp = np.random.default_rng(_seed(g_i, 6000 + i))
    sizes    = rng_resp.lognormal(mean=np.log(0.5), sigma=0.5, size=n_gals)
    R_RESP   = np.clip(1.0 - (sizes / 1.5) ** 2, 0.05, 0.99)
    gal_i[:, 5] = R_RESP
    gal_i[:, 6] = 0.0
    gal_i[:, 7] = 0.0
    gal_i[:, 8] = R_RESP

    rng_sh = np.random.default_rng(_seed(g_i, 1000 + i))
    sp_i   = ngmix.priors.shape.GPriorBA(sigma=SHAPE_NOISE, rng=rng_sh)
    e1, e2 = sp_i.sample2d(n_gals)
    gal_i[:, 3] = e1
    gal_i[:, 4] = e2

    mc_mask = gal_i[:, 2] > 1.1 * z_i + 0.2
    gal_mc  = gal_i[mc_mask]
    if len(gal_mc) < 20:
        return i, None, None

    if SPHERICAL:
        # Pure spherical NFW: set b=c=a so the projected ellipse degenerates
        # to a circle and the triaxial-shear pipeline collapses analytically
        # to the Wright-Brainerd 1999 spherical-NFW formula.  This reuses the
        # fast vectorised path of apply_triaxial_NFW_shear_region rather than
        # the per-galaxy Python loop in apply_NFW_shear_region.
        cluster_arr = [
            0.0, 0.0, z_i, M_i, c_i, a_i,
            a_i, a_i,                                # b = c = a → sphere
            _SHAPES["theta_los"][i], _SHAPES["phi_los"][i],
        ]
        gal_sheared = apply_triaxial_NFW_shear_region(cluster_arr, gal_mc)
    else:
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
        return i, None, None, None

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
    # Float32 bg for storage; downstream stacked_signal / single_cluster
    # don't care about precision.
    bg_arr = stake.astype("float32")
    return i, dv, cov_d, bg_arr


def simulate_lensing(truth, shapes, mis, g_idx):
    global _TRUTH, _SHAPES, _OFF_MAG_MPC, _PHI_MIS, _DA_FAST, _G_IDX_SIM

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
    _G_IDX_SIM    = g_idx

    data_vectors = np.zeros((N_CLUSTERS, N_BINS))
    cov_matrices = np.full((N_CLUSTERS, N_BINS, N_BINS), 0.0)
    bgs          = [None] * N_CLUSTERS
    for i in range(N_CLUSTERS):
        np.fill_diagonal(cov_matrices[i], NOISE_FLOOR_COV)

    with mp.Pool(N_WORKERS) as pool:
        for idx, dv, cov_d, bg in pool.imap(
            _process_cluster, range(N_CLUSTERS), chunksize=10
        ):
            if dv is not None:
                data_vectors[idx] = dv
                np.fill_diagonal(cov_matrices[idx], cov_d)
                bgs[idx]          = bg

    return data_vectors, cov_matrices, bgs


# ── Notebook-compatible per-gamma sim data ────────────────────────────────────

_NOTEBOOK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'notebooks', 'data',
)


def _write_notebook_compat_sim(g_idx, gamma_true, truth, cluster_obs, bgs):
    """Save sim_g{idx:03d}_real_data.fits + sim_g{idx:03d}_bgs.pkl in the
    layout that wazp_lensing_signals.ipynb and
    wazp_mass_richness_by_delta_m.ipynb expect when SIMS=True and
    SIM_PREFIX='sim_g{idx:03d}'."""
    import json as _json
    import datetime as _datetime
    from astropy.table import Table as _Table
    from astropy.io    import fits as _fits  # noqa: F401  (Table.write needs fits)

    # Distinct prefix per shear/centering combination so non-default
    # configs don't overwrite the canonical sim_g{idx:03d}_*.  Override
    # with the YAML 'sim_prefix' or env SIM_PREFIX_ROOT if you want a
    # custom run name.
    prefix_default = ('sim_g' if (not SPHERICAL and MISCENTERING)
                       else ('sim_sph_g' if SPHERICAL and MISCENTERING
                       else ('sim_nomis_g' if not SPHERICAL
                       else 'sim_sph_nomis_g')))
    prefix_root = _cfg("sim_prefix", "SIM_PREFIX_ROOT", prefix_default)
    prefix      = f'{prefix_root}{g_idx:03d}'
    fits_p  = os.path.join(_NOTEBOOK_DATA_DIR, f'{prefix}_real_data.fits')
    bgs_p   = os.path.join(_NOTEBOOK_DATA_DIR, f'{prefix}_bgs.pkl')
    meta_p  = os.path.join(_NOTEBOOK_DATA_DIR, f'{prefix}_metadata.json')
    os.makedirs(_NOTEBOOK_DATA_DIR, exist_ok=True)

    # Cluster catalogue: distribute on a non-overlapping RA/Dec grid; the
    # notebook only reads ID/ZPHOT/NGALS/DELTA_M_I when SIMS=True so the
    # angular positions are purely formal.
    n        = len(truth)
    side     = int(np.ceil(np.sqrt(n)))
    iy, ix   = np.divmod(np.arange(n), side)
    step_deg = 0.6
    ra       = (10.0 + ix * step_deg).astype(float)
    dec      = (-60.0 + iy * step_deg).astype(float)
    cid      = np.arange(n, dtype=np.int64)

    tab = _Table({
        'ID':            cid,
        'RA':            ra,
        'DEC':           dec,
        'NGALS':         np.asarray(cluster_obs['Lambda']).astype('float32'),
        'ZPHOT':         np.asarray(truth['z']).astype('float32'),
        'DELTA_M_I':     np.asarray(truth['delta_m']).astype('float32'),
        'DELTA_M_I_ERR': np.full(n, 0.01, dtype='float32'),
    })
    tab.write(fits_p, overwrite=True)

    bgs_by_id = {int(cid[i]): bg
                 for i, bg in enumerate(bgs) if bg is not None}
    with open(bgs_p, 'wb') as fh:
        pickle.dump(bgs_by_id, fh)

    meta = {
        'gamma':         float(gamma_true),
        'g_idx':         int(g_idx),
        'n_truth':       int(n),
        'n_usable_bgs':  int(len(bgs_by_id)),
        'density':       DENSITY,
        'radii_layout':  'np.logspace(-0.8, 0.6, 8)',
        'shape_noise':   SHAPE_NOISE,
        'z_src_range':   list(Z_SRC_RANGE),
        'generated_at':  _datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(meta_p, 'w') as fh:
        _json.dump(meta, fh, indent=2)

    print(f"    notebook-compat sim written: "
          f"{prefix}_real_data.fits ({n:,} clusters), "
          f"{prefix}_bgs.pkl ({len(bgs_by_id):,} bgs)", flush=True)


# ── Simulation checkpointing ──────────────────────────────────────────────────

def save_simulation_checkpoint(sim_subdir, truth, ols, cluster_obs,
                                data_vectors, cov_matrices):
    """Persist all arrays needed to run bootstrap draws without re-simulating."""
    np.savez_compressed(
        os.path.join(sim_subdir, "simulated_data.npz"),
        data_vectors=data_vectors,
        cov_matrices=cov_matrices,
    )
    np.savez_compressed(
        os.path.join(sim_subdir, "truth_catalog.npz"),
        z=np.asarray(truth["z"]),
        M_halo=np.asarray(truth["M_halo"]),
        concentration=np.asarray(truth["concentration"]),
        delta_m=np.asarray(truth["delta_m"]),
    )
    np.savez_compressed(
        os.path.join(sim_subdir, "cluster_obs.npz"),
        z=np.asarray(cluster_obs["z"]),
        Lambda=np.asarray(cluster_obs["Lambda"]),
        delta_m=np.asarray(cluster_obs["delta_m"]),
    )
    with open(os.path.join(sim_subdir, "ols.json"), "w") as f:
        json.dump(ols, f, indent=2)


def load_simulation_checkpoint(sim_subdir):
    """Reload checkpointed simulation data. Returns astropy Tables."""
    from astropy.table import Table

    raw = np.load(os.path.join(sim_subdir, "simulated_data.npz"))
    data_vectors = raw["data_vectors"]
    cov_matrices = raw["cov_matrices"]

    tc    = np.load(os.path.join(sim_subdir, "truth_catalog.npz"))
    truth = Table({k: tc[k] for k in tc.files})

    co          = np.load(os.path.join(sim_subdir, "cluster_obs.npz"))
    cluster_obs = Table({k: co[k] for k in co.files})

    with open(os.path.join(sim_subdir, "ols.json")) as f:
        ols = json.load(f)

    return truth, ols, cluster_obs, data_vectors, cov_matrices


# ── Bootstrap draw runner ─────────────────────────────────────────────────────

def run_one_bootstrap(b_idx, sim_subdir, gamma_true, ols,
                      cluster_obs, data_vectors_fit, cov_matrices_fit, RADII_fit):
    """Run one fit on a pre-simulated fixed dataset.

    With BOOTSTRAP=1 (default) draws ``n_resample_actual`` clusters with
    replacement; with BOOTSTRAP=0 uses the full N_CLUSTERS sample.
    """
    subdir       = os.path.join(sim_subdir, f"bootstrap_{b_idx:03d}")
    os.makedirs(subdir, exist_ok=True)
    summary_path = os.path.join(subdir, "summary.json")
    npz_path     = os.path.join(subdir, "gamma_posterior.npz")

    if BOOTSTRAP:
        n_resample_actual = N_RESAMPLE if N_RESAMPLE is not None else N_CLUSTERS
        rng_boot = np.random.default_rng()
        idx_boot = rng_boot.choice(N_CLUSTERS, size=n_resample_actual, replace=True)
        n_unique = int(np.unique(idx_boot).size)
        print(
            f"    bootstrap resample: {n_resample_actual} drawn "
            f"({n_unique} unique, {100*n_unique/N_CLUSTERS:.1f}%)",
            flush=True,
        )
    else:
        n_resample_actual = N_CLUSTERS
        idx_boot          = np.arange(N_CLUSTERS)
        n_unique          = N_CLUSTERS
        print("    no bootstrap — using full sample", flush=True)

    cluster_obs_boot      = cluster_obs[idx_boot]
    data_vectors_fit_boot = data_vectors_fit[idx_boot]
    cov_matrices_fit_boot = cov_matrices_fit[idx_boot]
    FMISS_boot     = np.full(n_resample_actual, 1.0 - F_CEN)
    SIGMA_OFF_boot = np.full(n_resample_actual, SIGMA_OFF_MPC)

    checkpoint = os.path.join(subdir, "nautilus_checkpoint.hdf5")
    sampler = run_sampler_combined(
        cluster_obs_boot, data_vectors_fit_boot, cov_matrices_fit_boot, RADII_fit,
        priors=PRIORS, delta_m_ref=DELTA_M_REF, fit_beta=True,
        FMISS=FMISS_boot, SIGMA_OFF=SIGMA_OFF_boot,
        n_live=N_LIVE, pool=N_WORKERS,
        verbose=True, filepath=checkpoint, resume=True, timeout=TIMEOUT,
    )
    points, log_w, _ = sampler.posterior()
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()

    gamma_s = points[:, 3]
    lo  = _wq(gamma_s, weights, 0.16)
    med = _wq(gamma_s, weights, 0.50)
    hi  = _wq(gamma_s, weights, 0.84)
    # Persist the full posterior (all PARAM_ORDER columns) so the corner
    # plot can be regenerated later without re-running nautilus.
    np.savez_compressed(npz_path,
                        gamma_samples=gamma_s,
                        weights=weights,
                        all_samples=points,
                        param_order=np.asarray(PARAM_ORDER))

    # Per-fit corner plot inside this bootstrap subdir.  When BOOTSTRAP=0
    # there is exactly one bootstrap per gamma, so this lands at
    # sim_subdir/bootstrap_000/posterior_corner.png — i.e. "inside the
    # gamma's directory" for the typical no-bootstrap run.
    try:
        import corner as _corner
        labels = [PRIOR_LABELS.get(p, p) for p in PARAM_ORDER]
        truths = [None] * len(PARAM_ORDER)
        if "gamma" in PARAM_ORDER:
            truths[PARAM_ORDER.index("gamma")] = float(gamma_true)
        if "c0" in PARAM_ORDER:
            truths[PARAM_ORDER.index("c0")] = float(ols.get("c0_true",
                                                            np.nan))
        if "alpha" in PARAM_ORDER:
            truths[PARAM_ORDER.index("alpha")] = float(ols.get("alpha_true",
                                                               np.nan))
        if "beta" in PARAM_ORDER:
            truths[PARAM_ORDER.index("beta")] = float(ols.get("beta_true",
                                                              np.nan))
        fig = _corner.corner(
            points, weights=weights,
            labels=labels,
            truths=truths,
            truth_color="crimson",
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True, title_fmt=".3f",
            plot_datapoints=False, fill_contours=True,
            levels=(0.39, 0.86),
            smooth=1.0,
        )
        fig.suptitle(
            f"posterior — γ_true={gamma_true:+.3f}, "
            f"γ_med={med:+.3f} ± {(hi - lo) / 2.0:.3f}",
            y=1.02, fontsize=11,
        )
        corner_path = os.path.join(subdir, "posterior_corner.png")
        fig.savefig(corner_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    corner plot → {corner_path}", flush=True)
    except Exception as exc:
        print(f"    corner plot skipped: {exc!r}", flush=True)

    result = {
        "bootstrap":     b_idx,
        "bootstrap_on":  bool(BOOTSTRAP),
        "gamma_true":    float(gamma_true),
        "gamma_med":     float(med),
        "gamma_lo":      float(lo),
        "gamma_hi":      float(hi),
        "sigma_gamma":   float((hi - lo) / 2.0),
        "n_clusters":    N_CLUSTERS,
        "n_resample":    n_resample_actual,
        "n_unique":      n_unique,
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


# ── Per-gamma simulation + bootstrap loop ─────────────────────────────────────

def run_one_gamma(g_idx, gamma_true, run_dir):
    """Simulate once + run N_BOOTSTRAP bootstrap draws for a single gamma value."""
    sim_subdir = os.path.join(run_dir, gamma_dir_name(g_idx, gamma_true))
    os.makedirs(sim_subdir, exist_ok=True)
    ens_path   = os.path.join(sim_subdir, "ensemble_summary.json")

    # ── Simulation (checkpoint-aware) ────────────────────────────────────────
    data_npz = os.path.join(sim_subdir, "simulated_data.npz")
    if os.path.exists(data_npz):
        print("  loading simulation from checkpoint ...", flush=True)
        truth, ols, cluster_obs, data_vectors, cov_matrices = \
            load_simulation_checkpoint(sim_subdir)
    else:
        print(f"  building truth catalogue (γ={gamma_true:+.1f}) ...", flush=True)
        truth, ols = build_truth_catalog(gamma_true, g_idx)
        print(
            f"    OLS: c0={ols['c0_true']:.3f}, "
            f"α={ols['alpha_true']:.4f}, β={ols['beta_true']:.4f}, "
            f"γ_ols={ols['gamma_true_ols']:.4f}",
            flush=True,
        )

        print("  triaxial shapes + miscentering ...", flush=True)
        shapes = compute_triaxial_shapes(truth, g_idx)
        mis    = compute_miscentering(g_idx)

        print(f"  simulating lensing ({N_CLUSTERS} clusters) ...", flush=True)
        data_vectors, cov_matrices, bgs = simulate_lensing(truth, shapes, mis, g_idx)

        cluster_obs, _, _ = make_test_data(
            truth, RADII, fractional_noise=0.2,
            poisson_richness=False, seed=_seed(g_idx, 1),
        )

        print("  saving simulation checkpoint ...", flush=True)
        save_simulation_checkpoint(
            sim_subdir, truth, ols, cluster_obs, data_vectors, cov_matrices
        )

        # Also write notebook-compatible per-gamma sim data so the
        # wazp_lensing_signals / wazp_mass_richness_by_delta_m notebooks
        # can consume this gamma's catalogue and bgs by setting
        # SIMS=True and SIM_PREFIX=sim_g{idx:03d}.
        _write_notebook_compat_sim(g_idx, gamma_true, truth, cluster_obs, bgs)

    keep             = RADII >= R_MIN_FIT
    RADII_fit        = RADII[keep]
    data_vectors_fit = data_vectors[:, keep]
    cov_matrices_fit = cov_matrices[:, keep][:, :, keep]

    # ── Bootstrap loop ───────────────────────────────────────────────────────
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
                gamma_true, ols, cluster_obs,
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
        raise RuntimeError(f"{gamma_dir_name(g_idx, gamma_true)}: all bootstrap draws failed")

    # ── Ensemble statistics ───────────────────────────────────────────────────
    gamma_meds  = np.array([r["gamma_med"]   for r in results])
    sigma_posts = np.array([r["sigma_gamma"] for r in results])
    ens_med     = float(np.median(gamma_meds))
    mean_sigma  = float(np.mean(sigma_posts))
    if len(gamma_meds) > 1:
        # Multi-draw bootstrap: error from spread of bootstrap medians
        sigma_boot = float(np.std(gamma_meds, ddof=1))
    else:
        # Single fit (BOOTSTRAP=0): error is the single posterior's 16/84
        # half-width, since std-of-one is undefined.
        sigma_boot = float(sigma_posts[0])

    ens_result = {
        "g_idx":              g_idx,
        "gamma_true":         float(gamma_true),
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
        f"  γ={gamma_true:+.1f}  γ_est={ens_med:+.4f}  "
        f"σ_boot={sigma_boot:.4f}  bias={ens_med - gamma_true:+.4f}",
        flush=True,
    )

    # posteriors overlay for this gamma
    try:
        _plot_posteriors_overlay(results, sim_subdir, gamma_true)
    except Exception:
        pass

    return ens_result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_gamma_scan_results(scan_results, run_dir, bin_size=10):
    """Two-panel: top = true vs estimated; bottom = residuals.  Each panel
    overlays the median in bins of `bin_size` consecutive γ_true points
    (sorted), a 68% credible band from the within-bin 16th/84th
    percentiles, and a linear (γ_est vs γ_true) least-squares fit so the
    overall slope-1 and offset-0 deviations are quantified."""
    gamma_true = np.array([r["gamma_true"]          for r in scan_results])
    gamma_est  = np.array([r["gamma_ensemble_med"]   for r in scan_results])
    sigma_boot = np.array([r["sigma_boot"]            for r in scan_results])
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
        rf"  ($N={N_CLUSTERS}$, {N_BOOTSTRAP} bootstrap draws/γ)",
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


def _plot_posteriors_overlay(results, sim_subdir, gamma_true):
    fig, ax = plt.subplots(figsize=(9, 5))
    x_grid  = np.linspace(-2.5, 2.5, 800)
    cmap    = plt.cm.viridis
    n       = len(results)
    for k, r in enumerate(results):
        npz_path = os.path.join(
            sim_subdir, f"bootstrap_{r['bootstrap']:03d}", "gamma_posterior.npz"
        )
        if not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path)
        samples = data["gamma_samples"]
        weights = data["weights"]
        try:
            kde = gaussian_kde(samples, weights=weights, bw_method=0.15)
            ax.plot(x_grid, kde(x_grid),
                    color=cmap(k / max(n - 1, 1)), alpha=0.55, lw=1.0)
        except Exception:
            pass
    ax.plot(x_grid, PRIORS["gamma"].pdf(x_grid),
            color="darkorange", lw=1.8, ls="--", label="Prior")
    ax.axvline(gamma_true, color="crimson", lw=2.0, ls="--",
               label=rf"$\gamma_{{\rm true}}={gamma_true:+.1f}$")
    ax.set(xlabel=r"$\gamma$", ylabel="Posterior density",
           title=(rf"Bootstrap posteriors  "
                  rf"($\gamma_{{\rm true}}={gamma_true:+.1f}$, $N={N_CLUSTERS}$)"))
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(sim_subdir, "posteriors_overlay.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def print_scan_table(scan_results):
    print("\n" + "=" * 82, flush=True)
    print(
        f'{"g_idx":>6}  {"gamma_true":>10}  {"gamma_est":>10}  '
        f'{"sigma_boot":>10}  {"bias":>8}  {"bias/sig":>8}',
        flush=True,
    )
    print("-" * 82, flush=True)
    for r in scan_results:
        bias = r["gamma_ensemble_med"] - r["gamma_true"]
        bsig = bias / max(r["sigma_boot"], 1e-12)
        print(
            f'{r["g_idx"]:>6d}  {r["gamma_true"]:>+10.3f}  '
            f'{r["gamma_ensemble_med"]:>+10.4f}  '
            f'{r["sigma_boot"]:>10.4f}  '
            f'{bias:>+8.4f}  '
            f'{bsig:>+8.2f}',
            flush=True,
        )
    residuals = np.array([r["gamma_ensemble_med"] - r["gamma_true"] for r in scan_results])
    sig_boots = np.array([r["sigma_boot"] for r in scan_results])
    print("-" * 82, flush=True)
    print(
        f"RMS bias = {np.sqrt(np.mean(residuals**2)):.4f}   "
        f"mean(sigma_boot) = {np.mean(sig_boots):.4f}",
        flush=True,
    )


def write_progress(scan_results, run_dir, run_t0, n_total):
    elapsed = time.time() - run_t0
    n_done  = len(scan_results)
    eta     = (elapsed / n_done * (n_total - n_done)) if n_done > 0 else float("nan")
    prog = {
        "timestamp":  datetime.datetime.now().isoformat(),
        "elapsed_s":  round(elapsed, 1),
        "n_done":     n_done,
        "n_total":    n_total,
        "eta_s":      round(eta, 1) if np.isfinite(eta) else None,
        "completed": [
            {
                "g_idx":      r["g_idx"],
                "gamma_true": r["gamma_true"],
                "gamma_est":  round(r["gamma_ensemble_med"], 4),
                "sigma_boot": round(r["sigma_boot"], 4),
                "bias":       round(r["gamma_ensemble_med"] - r["gamma_true"], 4),
            }
            for r in scan_results
        ],
    }
    with open(os.path.join(run_dir, "progress.json"), "w") as f:
        json.dump(prog, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_resample_actual = N_RESAMPLE if N_RESAMPLE is not None else N_CLUSTERS
    n_total           = len(GAMMA_VALUES)

    print(
        f"Gamma scan: {n_total} gamma values from {GAMMA_VALUES[0]:+.1f} to {GAMMA_VALUES[-1]:+.1f}\n"
        f"  N_CLUSTERS={N_CLUSTERS}, BOOTSTRAP={'on' if BOOTSTRAP else 'off'}, "
        f"N_BOOTSTRAP={N_BOOTSTRAP}, "
        f"N_RESAMPLE={n_resample_actual if BOOTSTRAP else N_CLUSTERS}, "
        f"N_WORKERS={N_WORKERS}, "
        f"SHEAR={'SPHERICAL' if SPHERICAL else 'TRIAXIAL'}, "
        f"MISCENTERING={'on' if MISCENTERING else 'off'}\n"
        f"  density={DENSITY} gal/arcmin^2, "
        f"Smail(α={DES_Y3_Z_SRC_ALPHA},β={DES_Y3_Z_SRC_BETA},"
        f"z0={DES_Y3_Z_SRC_Z0}), "
        f"z_cl∈[{DES_Y3_Z_CLUSTER_RANGE[0]:.2f},"
        f"{DES_Y3_Z_CLUSTER_RANGE[1]:.2f}]",
        flush=True,
    )
    print(f"Output directory: {RUN_DIR}", flush=True)

    scan_results = []
    run_t0       = time.time()

    for g_idx, gamma_true in enumerate(GAMMA_VALUES):
        sim_subdir = os.path.join(RUN_DIR, gamma_dir_name(g_idx, gamma_true))
        ens_path   = os.path.join(sim_subdir, "ensemble_summary.json")

        print(
            f"\n[{g_idx + 1}/{n_total}] gamma={gamma_true:+.1f}  "
            f"({gamma_dir_name(g_idx, gamma_true)})",
            flush=True,
        )
        iter_t0 = time.time()

        if os.path.exists(ens_path):
            print("  cached — loading ensemble_summary.json", flush=True)
            with open(ens_path) as f:
                scan_results.append(json.load(f))
            write_progress(scan_results, RUN_DIR, run_t0, n_total)
            continue

        try:
            r = run_one_gamma(g_idx, gamma_true, RUN_DIR)
            scan_results.append(r)
        except KeyboardInterrupt:
            print(
                f"\nInterrupted after {len(scan_results)}/{n_total} gamma values.\n"
                f"Resume with:  RUN_DIR={RUN_DIR} python gamma_scan.py",
                flush=True,
            )
            _finalize(scan_results, RUN_DIR, run_t0, n_total)
            return 2
        except Exception as exc:
            print(f"  gamma={gamma_true:+.1f} FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue

        iter_dt = time.time() - iter_t0
        done    = len(scan_results)
        avg_dt  = (time.time() - run_t0) / done
        eta     = avg_dt * (n_total - done)
        print(
            f"  [{done}/{n_total}] took {_fmt_duration(iter_dt)}; "
            f"ETA {_fmt_duration(eta)}",
            flush=True,
        )
        write_progress(scan_results, RUN_DIR, run_t0, n_total)

    total_elapsed = time.time() - run_t0
    print(
        f"\nTotal wall time: {_fmt_duration(total_elapsed)}  "
        f"({len(scan_results)}/{n_total} gamma values completed)",
        flush=True,
    )

    if not scan_results:
        print("No simulations completed.", file=sys.stderr)
        return 1

    _finalize(scan_results, RUN_DIR, run_t0, n_total)
    return 0


def _finalize(scan_results, run_dir, run_t0, n_total):
    with open(os.path.join(run_dir, "scan_summary.json"), "w") as f:
        json.dump(scan_results, f, indent=2)
    write_progress(scan_results, run_dir, run_t0, n_total)
    if len(scan_results) >= 2:
        try:
            plot_gamma_scan_results(scan_results, run_dir)
        except Exception as exc:
            print(f"Warning: scan plot failed: {exc}", flush=True)
    print_scan_table(scan_results)


if __name__ == "__main__":
    sys.exit(main())
