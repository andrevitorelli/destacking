"""
Gamma inference with separated simulation and inference phases -- Diemer
& Joyce (2019) mass-concentration relation variant.

Identical to gamma_inference.py in every respect (phases, resumability,
YAML config, bootstrap/combination logic) EXCEPT the mass-concentration
relation: instead of the free power-law c0/alpha_cm/beta_cm/gamma_cm
model, mass and redshift dependence come directly from the Diemer & Joyce
(2019) c200m(M, z) relation (destacking.model.diemer19_c200m /
forward_model_diemer), and the ONLY concentration-related free parameter
is 'gamma' -- the magnitude-gap slope, playing the same role gamma_cm
does in the combined model, layered on top of the Diemer19 baseline:

    log10(c) = log10(c_Diemer19(M200m, z)) + gamma * log10(Δm/Δm_ref)

gamma=0 reproduces the literature prediction exactly, independent of Δm.
See destacking/model.py's forward_model_diemer docstring and project
report reports/2026-07-26_boost_factors_variance_floor_alpha_cm.tex
(Section 5) for the physical motivation and the mass-definition
consistency check (M200m throughout, matching xlensing.model.
NFW_Delta_Sigma -- Diemer19 is natively 200c in colossus, so
diemer19_c200m interpolates a precomputed M200m grid rather than calling
colossus with mdef='200m' directly, which has a vectorized-redshift bug).

Does exactly what gamma_scan.py does, but the per-gamma work is split into
two independently toggleable phases:

  simulation: true/false   generate + measure the synthetic cluster sample,
                           saving the standard checkpoint files per gamma
  inference:  true/false   run the bootstrap nautilus fits and combine the
                           posteriors, reading the checkpoints

YAML keys (in addition to everything gamma_scan.py accepts):
  simulation:  true        # default
  inference:   true        # default
  sim_log_inner_radius: -0.8  # SIMULATED radial grid [Mpc]:
  sim_log_outer_radius:  0.6  #   np.logspace(inner, outer, n)
  sim_n_radii:           8
  fit_log_inner_radius: -0.8  # FIT radial grid: the radii the data vectors
  fit_log_outer_radius:  0.6  #   live on.  Defaults to the sim grid on full
  fit_n_radii:           8    #   runs; REQUIRED in inference-only mode.
  r_min_fit:             0.5  # minimum radius [Mpc] for bins entering the fit
  fit_params: {...}        # PREFERRED: full per-parameter fit control (see
                           #   "Per-parameter fit control" below). Mutually
                           #   exclusive with the three legacy keys below.
  fit_miscentering: false  # LEGACY (no fit_params): sample population-level
                           #   f_mis and sigma_off as free fit parameters
                           #   (flat priors) instead of fixing them at
                           #   fit_fmiss / fit_sigma_off_mpc_h
  data_folder: <path>      # REQUIRED when simulation is false: a directory
                           # containing simulation_g{k:02d}_gamma{v:+.1f}/
                           # subdirectories in the standard checkpoint
                           # layout (simulated_data.npz, truth_catalog.npz,
                           # cluster_obs.npz, ols.json), e.g. the output of
                           # a previous simulation-only run or of
                           # gamma_scan.py.

Modes
-----
  simulation=true,  inference=true   end-to-end, identical to gamma_scan.py
  simulation=true,  inference=false  produce simulations only (e.g. on a
                                     machine with many cores, to be fitted
                                     elsewhere)
  simulation=false, inference=true   fit pre-existing data from data_folder;
                                     fit outputs go to this run's RUN_DIR,
                                     the data folder is never written to

Design
------
  GAMMA_VALUES = np.linspace(-1, +1, 11)

  for each gamma_true in GAMMA_VALUES:          # 11 independent simulations
      simulate N_CLUSTERS=10000 clusters       → saved to disk
      for b = 0 .. N_BOOTSTRAP-1:              # 50 bootstrap draws per gamma
          idx  = bootstrap resample of {0..N_CLUSTERS-1}     (BOOTSTRAP=1)
                 OR np.arange(N_CLUSTERS)  full sample       (BOOTSTRAP=0)
          run nautilus on data[idx]  →  posterior_b(γ)
      combined posterior = equal-weight mixture of the posterior_b
      gamma_ensemble = median of the combined posterior
      sigma_gamma    = 16-84 half-width of the combined posterior
      (std_b of the per-bootstrap medians is kept as a diagnostic,
       "sigma_med_scatter")

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
  TIMEOUT       Wall-time budget per nautilus fit in seconds (default 600;
                YAML key: timeout).  Scale with N_CLUSTERS — too small a
                budget cuts the sampler off mid-run and produces collapsed
                (near-zero-width) posteriors.
  RUN_DIR       Output directory                (default
                ~/destacking_data/output/run_YYYY-MM-DDTHH-MM-SS)
                Set to an existing path to resume a previous run.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as _planck18
from scipy.stats import norm as sp_norm
from scipy.stats import uniform as sp_uniform
from scipy.stats import truncnorm as sp_truncnorm


from destacking.fitting import run_sampler_diemer
from destacking.io import (
    load_simulation_checkpoint,
    save_simulation_checkpoint,
    write_notebook_compat_sim,
)
from destacking.plotting import (
    overlay_priors_on_corner,
    plot_combined_corner as _plot_combined_corner_impl,
    plot_gamma_scan_results as _plot_gamma_scan_results,
    plot_posteriors_overlay as _plot_posteriors_overlay_impl,
)
from destacking.simulation import (
    SimulationConfig,
    build_truth_catalog as _sim_build_truth_catalog,
    make_test_data,
    compute_miscentering as _sim_compute_miscentering,
    compute_triaxial_shapes,
    simulate_lensing as _sim_simulate_lensing,
)
from destacking.utils import (
    deterministic_seed as _seed,
    fmt_duration as _fmt_duration,
    gamma_dir_name,
    weighted_quantile as _wq,
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
    """Returns (config dict, path to the YAML file or None)."""
    path = None
    # Allow sys.argv[1] when it does not look like an internal nautilus arg
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.yaml')
                              or sys.argv[1].endswith('.yml')):
        path = sys.argv[1]
    elif os.environ.get("CONFIG", "").strip():
        path = os.environ["CONFIG"].strip()
    if path is None:
        return {}, None
    try:
        import yaml as _yaml
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot load config {path}: PyYAML not installed ({exc!r})"
        )
    with open(path) as fh:
        cfg = _yaml.safe_load(fh) or {}
    print(f"[gamma_scan] loaded YAML config from {path}", flush=True)
    return cfg, path

_CFG, _CFG_PATH = _load_yaml_config()


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
N_CLUSTERS      = int(_cfg("n_clusters",      "N_CLUSTERS",      15000, cast=int))
N_BOOTSTRAP     = int(_cfg("n_bootstrap",     "N_BOOTSTRAP",     1,     cast=int))
N_WORKERS       = int(_cfg("n_workers",       "N_WORKERS",       15,    cast=int))
N_COV_BOOTSTRAP = int(_cfg("n_cov_bootstrap", "N_COV_BOOTSTRAP", 200,   cast=int))

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
BOOTSTRAP = _yaml_bool(_cfg("bootstrap", "BOOTSTRAP", False))
if not BOOTSTRAP:
    N_BOOTSTRAP = 1

# Phase toggles: run the simulation phase, the inference phase, or both.
# When simulation is off, inference reads checkpoints from DATA_FOLDER.
SIMULATION  = _yaml_bool(_cfg("simulation", "SIMULATION", True))
INFERENCE   = _yaml_bool(_cfg("inference",  "INFERENCE",  True))
DATA_FOLDER = _cfg("data_folder", "DATA_FOLDER", None)

DELTA_M_REF = 0.1
# ── Radial bins ───────────────────────────────────────────────────────────
# Simulation grid — the ΔΣ points measured and stored per cluster:
#   SIM_RADII = np.logspace(sim_log_inner_radius, sim_log_outer_radius,
#                           sim_n_radii)   [Mpc]
# Fit grid — the radii the inference assumes the data vectors live on.
# On a full run (simulation on) it defaults to the simulation grid; in
# inference-only mode the fit_* keys are REQUIRED and must describe the
# grid of the data in data_folder.  r_min_fit then selects which of those
# bins enter the likelihood.
for _old_key in ("log_inner_radius", "log_outer_radius", "n_radii"):
    if _old_key in _CFG:
        raise SystemExit(
            f"config key '{_old_key}' was split into sim_{_old_key} / "
            f"fit_{_old_key} — update the YAML "
            f"(see gamma_inference_config_example.yaml)")

SIM_LOG_INNER_RADIUS = float(_cfg("sim_log_inner_radius",
                                  "SIM_LOG_INNER_RADIUS", -0.8, cast=float))
SIM_LOG_OUTER_RADIUS = float(_cfg("sim_log_outer_radius",
                                  "SIM_LOG_OUTER_RADIUS", 0.6, cast=float))
SIM_N_RADII          = int(_cfg("sim_n_radii", "SIM_N_RADII", 8, cast=int))
SIM_RADII = np.logspace(SIM_LOG_INNER_RADIUS, SIM_LOG_OUTER_RADIUS,
                        SIM_N_RADII)

_fit_keys = ("fit_log_inner_radius", "fit_log_outer_radius", "fit_n_radii")
_fit_vals = [_cfg(k, k.upper(), None) for k in _fit_keys]
if any(v is not None for v in _fit_vals):
    if any(v is None for v in _fit_vals):
        _missing = [k for k, v in zip(_fit_keys, _fit_vals) if v is None]
        raise SystemExit(f"incomplete fit radial grid: missing {_missing}")
    FIT_LOG_INNER_RADIUS = float(_fit_vals[0])
    FIT_LOG_OUTER_RADIUS = float(_fit_vals[1])
    FIT_N_RADII          = int(_fit_vals[2])
    FIT_RADII = np.logspace(FIT_LOG_INNER_RADIUS, FIT_LOG_OUTER_RADIUS,
                            FIT_N_RADII)
    if SIMULATION and not (FIT_N_RADII == SIM_N_RADII
                           and np.allclose(FIT_RADII, SIM_RADII)):
        raise SystemExit(
            "fit radial grid differs from the simulation grid, but this run "
            "simulates its own data (simulation: true), so the data vectors "
            "will be measured on the sim grid.  Drop the fit_* radii keys "
            "or make them match the sim_* ones.")
elif SIMULATION:
    # Full run with sims: fit on the simulated grid.
    FIT_LOG_INNER_RADIUS = SIM_LOG_INNER_RADIUS
    FIT_LOG_OUTER_RADIUS = SIM_LOG_OUTER_RADIUS
    FIT_N_RADII          = SIM_N_RADII
    FIT_RADII            = SIM_RADII.copy()
elif INFERENCE:
    raise SystemExit(
        "inference-only mode requires the fit radial grid explicitly: set "
        "fit_log_inner_radius, fit_log_outer_radius and fit_n_radii to "
        "describe the radii of the data in data_folder.")
else:
    FIT_RADII = SIM_RADII.copy()   # nothing will run anyway

# Minimum radius [Mpc] for bins entering the fit (bins below are kept in
# the data products but excluded from the likelihood).
R_MIN_FIT   = float(_cfg("r_min_fit", "R_MIN_FIT", 0.5, cast=float))

# Optional per-cluster boost correction: 'boosts: <path>' points at a
# wazp_stacked_analysis.py Stage 2b boost_fits pickle (data/wazp_boost_fits.pkl),
# keyed by stack name ("z{i}_ng{j}_dm{k}"), each a
# {B0, Rs, alpha, ...} cluster_toolkit.boostfactors.boost_powerlaw_at_R fit.
# Ported from gamma_inference.py (this script forked before that feature
# was added there and never received it) so the Diemer19-baseline fit can
# use the same boost-corrected real-data measurement as the combined-model
# production run. See gamma_inference.py's own comment on this block for
# the full rationale.
BOOSTS_FILE = _cfg("boosts", "BOOSTS_FILE", None)
# Two-halo term in the fitted model (added 2026-09-26).  true for REAL data,
# which contain the two-halo signal; false (default) for simulated data,
# because the simulations shear sources with isolated haloes only.
TWO_HALO = _yaml_bool(_cfg("two_halo", "TWO_HALO", False))
# Area-weighted average of the model over each radial bin (Gauss-Legendre nodes
# per bin; 0 = evaluate at the bin centre, the behaviour before 2026-09-28).
BIN_AVERAGE_NODES = int(_cfg("bin_average_nodes", "BIN_AVERAGE_NODES", 0, cast=int))
if BOOSTS_FILE is not None:
    with open(BOOSTS_FILE, "rb") as _bf:
        import pickle as _pickle
        _boost_data = _pickle.load(_bf)
    BOOST_FITS = _boost_data["boost_fits"]
else:
    BOOST_FITS = None

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

# Miscentering model parameters (also used by the fit's FMISS when
# miscentering is on): f_cen = correctly-centred fraction, sigma_off_mpc_h
# = Rayleigh scale of the offset in Mpc/h.
F_CEN           = float(_cfg("f_cen", "F_CEN", 0.64, cast=float))
SIGMA_OFF_MPC_H = float(_cfg("sigma_off_mpc_h", "SIGMA_OFF_MPC_H", 0.32,
                             cast=float))
H_PLANCK        = float(_planck18.h)
SIGMA_OFF_MPC   = SIGMA_OFF_MPC_H / H_PLANCK

DENSITY         = float(_cfg("density",     "DENSITY",     7.5,  cast=float))
SHAPE_NOISE     = float(_cfg("shape_noise", "SHAPE_NOISE", 0.27, cast=float))
# See gamma_inference.py's FLOOR_RMAX comment for the full rationale.
FLOOR_RMAX      = _cfg("floor_rmax", "FLOOR_RMAX", None, cast=float)
# Physical target aperture radius [Mpc], converted per-cluster via each
# cluster's own DA(z) inside destacking.simulation._process_cluster -- see
# scripts/gamma_inference.py's WIDTH_RAD comment for the full story (fixed
# 2026-09-05; this sibling script shares the same underlying pipeline and
# had the same latent bug).
WIDTH_RAD       = 5.0
Z_SRC_RANGE     = (0.0, 2.5)

_log_r = np.log10(SIM_RADII)
_dlog  = np.diff(_log_r)[0]
_edges = np.logspace(_log_r[0] - _dlog / 2, _log_r[-1] + _dlog / 2,
                     len(SIM_RADII) + 1)
BIN_LIMITS = np.column_stack([_edges[:-1], _edges[1:]])
N_BINS     = len(SIM_RADII)

# ── Per-parameter fit control ─────────────────────────────────────────────
#
# Which of the model's parameters are sampled (with a chosen prior) versus
# held fixed (at a declared value) is controlled by the YAML 'fit_params'
# block — one entry per parameter, e.g.:
#
#   fit_params:
#     gamma:        {fit: true,  prior: gaussian, loc: 0.0,  scale: 0.3}
#     log_M_norm:   {fit: true,  prior: gaussian, loc: 0.0,  scale: 0.1}
#     alpha_mr:     {fit: false, value: 1.356}    # McClintock+18 F_lambda
#     beta_mr:      {fit: false, value: -0.30}    # McClintock+18 G_z
#     f_mis:        {fit: true,  prior: flat, lo: 0.0,  hi: 1.0}
#     sigma_off:    {fit: true,  prior: flat, lo: 0.05, hi: 1.5}
#
# 'gamma' is the sole concentration-related parameter in this variant (see
# the module docstring): the magnitude-gap slope layered on top of the
# Diemer & Joyce (2019) c200m(M, z) prediction, replacing the
# c0/alpha_cm/beta_cm/gamma_cm power-law relation gamma_inference.py fits
# (gamma plays the same role gamma_cm does there). '_mr' parameters (alpha_mr,
# beta_mr) belong to the mass-richness relation, same meaning as in
# gamma_inference.py.
#
# All six names are required when 'fit_params' is given — every parameter
# is either fit (with an explicit prior) or fixed (with an explicit value),
# no silent defaults.  'gamma' must have fit: true (it is the science
# target of this pipeline). Legacy configs that omit 'fit_params' keep the
# pre-existing hardcoded behaviour exactly (see _build_fit_params below).
_FIT_PARAM_LABELS = {
    "gamma":      r"$\gamma$",
    "log_M_norm": r"$\log M_{\rm norm}$",
    "alpha_mr":   r"$\alpha_{\rm mr}$",
    "beta_mr":    r"$\beta_{\rm mr}$",
    "f_mis":      r"$f_{\rm mis}$",
    "sigma_off":  r"$\sigma_{\rm off}$ [Mpc]",
}
_FIT_PARAM_NAMES = tuple(_FIT_PARAM_LABELS)

# f_mis is a fraction and sigma_off is a scale radius — both are physically
# bounded regardless of what the user's Gaussian prior says, so a 'gaussian'
# fit_params entry for these two names is truncated to its physical range
# rather than left as an unbounded normal (which can sample negative
# sigma_off or f_mis outside [0, 1] once the prior is widened).
_GAUSSIAN_PRIOR_BOUNDS = {
    "f_mis":     (0.0, 1.0),
    "sigma_off": (0.0, None),
}


def _prior_from_spec(name, spec):
    """One fit_params.<name> entry -> ('sample', frozen_dist) or
    ('fixed', float). Raises SystemExit with a specific message on any
    malformed or incomplete spec — there is no silent fallback."""
    if not isinstance(spec, dict) or "fit" not in spec:
        raise SystemExit(
            f"fit_params.{name} must be a mapping with a boolean 'fit' key"
        )
    if _yaml_bool(spec["fit"]):
        kind = spec.get("prior")
        if kind == "gaussian":
            for k in ("loc", "scale"):
                if k not in spec:
                    raise SystemExit(
                        f"fit_params.{name}: prior 'gaussian' requires '{k}'"
                    )
            loc, scale = float(spec["loc"]), float(spec["scale"])
            bounds = _GAUSSIAN_PRIOR_BOUNDS.get(name)
            if bounds is not None:
                lo, hi = bounds
                a = -np.inf if lo is None else (lo - loc) / scale
                b = np.inf if hi is None else (hi - loc) / scale
                return "sample", sp_truncnorm(a, b, loc=loc, scale=scale)
            return "sample", sp_norm(loc=loc, scale=scale)
        if kind == "flat":
            for k in ("lo", "hi"):
                if k not in spec:
                    raise SystemExit(
                        f"fit_params.{name}: prior 'flat' requires '{k}'"
                    )
            lo, hi = float(spec["lo"]), float(spec["hi"])
            if not hi > lo:
                raise SystemExit(f"fit_params.{name}: 'hi' must exceed 'lo'")
            return "sample", sp_uniform(loc=lo, scale=hi - lo)
        raise SystemExit(
            f"fit_params.{name}: 'prior' must be 'gaussian' or 'flat' "
            f"(got {kind!r})"
        )
    else:
        if "value" not in spec:
            raise SystemExit(
                f"fit_params.{name}: fit is false, so 'value' is required"
            )
        return "fixed", float(spec["value"])


def _build_fit_params():
    """Resolve every parameter to either a prior (sampled) or a fixed
    value, returning (priors, fixed) dicts for run_sampler_diemer.

    Preferred path: the explicit 'fit_params' YAML block, giving full
    control over what is fit and its prior, with fixed parameters
    requiring a precise declared value.

    Legacy path (no 'fit_params' in the config): gamma/log_M_norm always
    sampled with fixed Gaussian priors, alpha_mr/beta_mr fixed at the
    McClintock+18 values, and f_mis/sigma_off controlled by the older
    fit_miscentering/fit_fmiss/fit_sigma_off_mpc_h keys — mirrors
    gamma_inference.py's legacy path (minus the power-law c-M parameters,
    which this variant doesn't have).
    """
    raw = _cfg("fit_params", None, None)
    legacy_keys = ("fit_miscentering", "fit_fmiss", "fit_sigma_off_mpc_h")

    if raw is not None:
        used_legacy = [k for k in legacy_keys if k in _CFG]
        if used_legacy:
            raise SystemExit(
                f"fit_params is set; remove the legacy key(s) {used_legacy} "
                f"and control f_mis/sigma_off through fit_params instead"
            )
        missing = [n for n in _FIT_PARAM_NAMES if n not in raw]
        if missing:
            raise SystemExit(
                f"fit_params is missing entries for {missing} — every one "
                f"of {list(_FIT_PARAM_NAMES)} must be declared (fit or fixed)"
            )
        priors, fixed = {}, {}
        for name in _FIT_PARAM_NAMES:
            kind, val = _prior_from_spec(name, raw[name])
            (priors if kind == "sample" else fixed)[name] = val
        if "gamma" not in priors:
            raise SystemExit(
                "fit_params.gamma must have fit: true — gamma is "
                "the science target of this pipeline"
            )
        return priors, fixed

    # ── Legacy path ──────────────────────────────────────────────────────
    priors = {
        "gamma":      sp_norm(loc=0.0,  scale=0.3),
        "log_M_norm": sp_norm(loc=0.0,  scale=0.1),
    }
    fixed = {"alpha_mr": 1.356, "beta_mr": -0.30}

    fit_miscentering = _yaml_bool(
        _cfg("fit_miscentering", "FIT_MISCENTERING", False)
    )
    if fit_miscentering:
        priors["f_mis"]     = sp_uniform(loc=0.0,  scale=1.0)
        priors["sigma_off"] = sp_uniform(loc=0.05, scale=1.45)
    else:
        # Fit-model miscentering (what the likelihood assumes), independent
        # of the simulation truth (F_CEN/SIGMA_OFF_MPC).  Defaults keep
        # model and truth consistent: fit_fmiss = 1 - f_cen when
        # miscentering is on, 0 when off.
        _fit_fmiss_raw = _cfg("fit_fmiss", "FIT_FMISS", None)
        if _fit_fmiss_raw is None:
            fixed["f_mis"] = (1.0 - F_CEN) if MISCENTERING else 0.0
        else:
            fixed["f_mis"] = float(_fit_fmiss_raw)
        _fit_sig_h = float(_cfg("fit_sigma_off_mpc_h", "FIT_SIGMA_OFF_MPC_H",
                                SIGMA_OFF_MPC_H, cast=float))
        fixed["sigma_off"] = _fit_sig_h / H_PLANCK
    return priors, fixed


PRIORS, FIXED_PARAMS = _build_fit_params()
PRIOR_LABELS = dict(_FIT_PARAM_LABELS)

N_LIVE  = 400
# Wall-time budget per nautilus fit [s].  Scale with N_CLUSTERS: the 600 s
# default is adequate for ~1500 clusters; larger samples need proportionally
# more or the sampler is cut off mid-run and returns degenerate posteriors.
TIMEOUT = int(_cfg("timeout", "TIMEOUT", 600, cast=int))

_DEFAULT_RUN_TS = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR = os.environ.get(
    "RUN_DIR",
    os.path.join(os.path.expanduser("~/destacking_data/output"), f"run_{_DEFAULT_RUN_TS}"),
)
# RUN_DIR is only a path here; it is created (and the config copied into
# it) inside main(), so importing this module for analysis leaves no
# empty run directories behind.

# ── Simulation pipeline configuration ─────────────────────────────────────────────

# The pipeline itself lives in destacking.simulation; this script only
# assembles its configuration from the YAML/env knobs above and provides
# signature-compatible wrappers for the run loop below.
SIM_CFG = SimulationConfig(
    n_clusters=N_CLUSTERS,
    z_cluster_range=DES_Y3_Z_CLUSTER_RANGE,
    logm_range=WAZP_LOGM_RANGE,
    lambda_min=WAZP_LAMBDA_MIN,
    delta_m_ref=DELTA_M_REF,
    density=DENSITY,
    shape_noise=SHAPE_NOISE,
    width_rad=WIDTH_RAD,
    z_src_range=Z_SRC_RANGE,
    z_src_alpha=DES_Y3_Z_SRC_ALPHA,
    z_src_beta=DES_Y3_Z_SRC_BETA,
    z_src_z0=DES_Y3_Z_SRC_Z0,
    spherical=SPHERICAL,
    miscentering=MISCENTERING,
    f_cen=F_CEN,
    sigma_off_mpc=SIGMA_OFF_MPC,
    radii=SIM_RADII,
    bin_limits=BIN_LIMITS,
    n_cov_bootstrap=N_COV_BOOTSTRAP,
    n_workers=N_WORKERS,
    floor_rmax=FLOOR_RMAX,
)


def build_truth_catalog(gamma_true, g_idx):
    return _sim_build_truth_catalog(gamma_true, g_idx, SIM_CFG)


def compute_miscentering(g_idx):
    return _sim_compute_miscentering(g_idx, SIM_CFG)


def simulate_lensing(truth, shapes, mis, g_idx):
    return _sim_simulate_lensing(truth, shapes, mis, g_idx, SIM_CFG)


# ── Notebook-compatible per-gamma sim data ────────────────────────────────────

_NOTEBOOK_DATA_DIR = os.path.expanduser("~/destacking_data/sims")


def _write_notebook_compat_sim(g_idx, gamma_true, truth, cluster_obs, bgs):
    # Distinct prefix per shear/centering combination so non-default
    # configs don't overwrite the canonical sim_g{idx:03d}_*.  Override
    # with the YAML 'sim_prefix' or env SIM_PREFIX_ROOT if you want a
    # custom run name.
    prefix_default = ('sim_g' if (not SPHERICAL and MISCENTERING)
                       else ('sim_sph_g' if SPHERICAL and MISCENTERING
                       else ('sim_nomis_g' if not SPHERICAL
                       else 'sim_sph_nomis_g')))
    prefix_root = _cfg("sim_prefix", "SIM_PREFIX_ROOT", prefix_default)
    return write_notebook_compat_sim(
        g_idx, gamma_true, truth, cluster_obs, bgs,
        out_dir=_NOTEBOOK_DATA_DIR, prefix_root=prefix_root,
        density=DENSITY, shape_noise=SHAPE_NOISE, z_src_range=Z_SRC_RANGE,
        radii_layout=(f"np.logspace({SIM_LOG_INNER_RADIUS}, "
                      f"{SIM_LOG_OUTER_RADIUS}, {SIM_N_RADII})"),
    )


def _param_truths(param_order, ols, gamma_true):
    """Truth-line values for a corner plot, one per name in param_order (or
    None where no truth is known/applicable).  alpha_mr/beta_mr from the
    McClintock+18 values; f_mis/sigma_off from the simulation's
    F_CEN/SIGMA_OFF_MPC.  'gamma' has no truth line: the simulation phase
    (unchanged from gamma_inference.py) injects Dutton-Maccio-based truth
    concentrations, not a Diemer19-relative offset, so there is no
    consistent ground truth to compare a fitted gamma against, in
    simulation or on real data."""
    truths = [None] * len(param_order)
    if "alpha_mr" in param_order:
        truths[param_order.index("alpha_mr")] = 1.356
    if "beta_mr" in param_order:
        truths[param_order.index("beta_mr")] = -0.30
    if "f_mis" in param_order:
        truths[param_order.index("f_mis")]     = 1.0 - F_CEN
        truths[param_order.index("sigma_off")] = SIGMA_OFF_MPC
    return truths


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

    # Resample from the actual data length (not the configured n_clusters),
    # so inference-only runs on external data folders are self-consistent.
    n_data = len(data_vectors_fit)
    if BOOTSTRAP:
        n_resample_actual = N_RESAMPLE if N_RESAMPLE is not None else n_data
        rng_boot = np.random.default_rng()
        idx_boot = rng_boot.choice(n_data, size=n_resample_actual, replace=True)
        n_unique = int(np.unique(idx_boot).size)
        print(
            f"    bootstrap resample: {n_resample_actual} drawn "
            f"({n_unique} unique, {100*n_unique/n_data:.1f}%)",
            flush=True,
        )
    else:
        n_resample_actual = n_data
        idx_boot          = np.arange(n_data)
        n_unique          = n_data
        print("    no bootstrap — using full sample", flush=True)

    cluster_obs_boot      = cluster_obs[idx_boot]
    data_vectors_fit_boot = data_vectors_fit[idx_boot]
    cov_matrices_fit_boot = cov_matrices_fit[idx_boot]

    checkpoint = os.path.join(subdir, "nautilus_checkpoint.hdf5")
    fit_t0  = time.time()
    # PRIORS/FIXED_PARAMS (module-level, from _build_fit_params) declare
    # every parameter as either sampled or fixed at a precise value — see
    # the "Per-parameter fit control" section above.  No FMISS/SIGMA_OFF
    # arrays are built here any more: f_mis/sigma_off go through the same
    # sample-or-fixed mechanism as every other parameter.
    sampler = run_sampler_diemer(
        cluster_obs_boot, data_vectors_fit_boot, cov_matrices_fit_boot, RADII_fit,
        priors=PRIORS, fixed=FIXED_PARAMS, delta_m_ref=DELTA_M_REF,
        two_halo=TWO_HALO,
        bin_nodes=BIN_AVERAGE_NODES,
        n_live=N_LIVE, pool=N_WORKERS,
        verbose=True, filepath=checkpoint, resume=True, timeout=TIMEOUT,
    )
    fit_wall_s = time.time() - fit_t0
    converged  = bool(getattr(sampler, "converged", True))
    # The set (and order) of SAMPLED parameters — only these are columns of
    # `points` below.  Authoritative: mirrors nautilus.Prior.keys exactly,
    # so it is correct for whatever subset PRIORS declares this run.
    param_order = list(getattr(sampler, "param_order", list(PRIORS)))
    # Sampler diagnostics for post-hoc fit-quality assessment: a poor fit
    # shows up as converged=False, fit_wall_s ~ timeout_s, and/or low n_eff.
    fit_diag = {
        "fit_wall_s": round(fit_wall_s, 1),
        "timeout_s":  TIMEOUT,
        "converged":  converged,
        "n_like":     int(getattr(sampler, "n_like", -1)),
        "n_eff":      float(getattr(sampler, "n_eff", np.nan)),
        "log_z":      float(getattr(sampler, "log_z", np.nan)),
        # Information gain from prior to posterior, in nats:
        # D_KL(posterior || prior) = <log L>_posterior - log Z. Always
        # reported (attached by run_sampler_combined itself).
        "kl_nats":    float(getattr(sampler, "kl_nats", np.nan)),
        # Per-parameter marginal KL(posterior_i || prior_i) and the
        # joint-minus-sum-of-marginals residual (= mutual information the
        # fit induced between parameters, since the priors are independent
        # across parameters — see run_sampler_combined's docstring/comment).
        "marginal_kl_nats": {
            k: float(v)
            for k, v in getattr(sampler, "marginal_kl_nats", {}).items()
        },
        "kl_residual_nats": float(getattr(sampler, "kl_residual_nats", np.nan)),
    }
    points, log_w, _ = sampler.posterior()
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()

    gamma_s = points[:, param_order.index("gamma")]
    lo  = _wq(gamma_s, weights, 0.16)
    med = _wq(gamma_s, weights, 0.50)
    hi  = _wq(gamma_s, weights, 0.84)

    mis_summary = {}
    if "f_mis" in param_order:
        for pname in ("f_mis", "sigma_off"):
            col = points[:, param_order.index(pname)]
            p_lo  = _wq(col, weights, 0.16)
            p_med = _wq(col, weights, 0.50)
            p_hi  = _wq(col, weights, 0.84)
            mis_summary[f"{pname}_med"]   = float(p_med)
            mis_summary[f"{pname}_sigma"] = float((p_hi - p_lo) / 2.0)
        print(
            f"    miscentering fit: f_mis = {mis_summary['f_mis_med']:.4f} "
            f"± {mis_summary['f_mis_sigma']:.4f}, "
            f"sigma_off = {mis_summary['sigma_off_med']:.4f} "
            f"± {mis_summary['sigma_off_sigma']:.4f} Mpc",
            flush=True,
        )
    # Persist the full posterior (all param_order columns) so the corner
    # plot can be regenerated later without re-running nautilus.
    np.savez_compressed(npz_path,
                        gamma_samples=gamma_s,
                        weights=weights,
                        all_samples=points,
                        param_order=np.asarray(param_order))

    # Per-fit corner plot inside this bootstrap subdir.  When BOOTSTRAP=0
    # there is exactly one bootstrap per gamma, so this lands at
    # sim_subdir/bootstrap_000/posterior_corner.png — i.e. "inside the
    # gamma's directory" for the typical no-bootstrap run.
    try:
        import corner as _corner
        labels = [PRIOR_LABELS.get(p, p) for p in param_order]
        truths = _param_truths(param_order, ols, gamma_true)
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
            hist_kwargs={"density": True},
        )
        overlay_priors_on_corner(fig, param_order, PRIORS)
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
        **mis_summary,
        **fit_diag,
        "n_clusters":    n_data,
        "n_resample":    n_resample_actual,
        "n_unique":      n_unique,
        **ols,
    }
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"    gamma_med = {med:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
        f"sigma = {(hi - lo) / 2.0:.4f}  "
        f"{'converged' if converged else 'NOT CONVERGED (timeout/n_like_max)'}",
        flush=True,
    )
    print(
        f"    fit diagnostics: wall={fit_diag['fit_wall_s']:.0f}s"
        f"/{TIMEOUT}s budget, n_like={fit_diag['n_like']}, "
        f"N_eff={fit_diag['n_eff']:.0f}, log Z={fit_diag['log_z']:.2f}, "
        f"KL={fit_diag['kl_nats']:.2f} nats",
        flush=True,
    )
    _marg = fit_diag["marginal_kl_nats"]
    if _marg:
        _marg_str = ", ".join(f"{n}={v:.2f}" for n, v in _marg.items())
        print(
            f"    marginal KL [nats]: {_marg_str}  "
            f"(residual/mutual-info={fit_diag['kl_residual_nats']:.2f})",
            flush=True,
        )
    return result


# ── Simulation phase ──────────────────────────────────────────────────────────

def simulate_one_gamma(g_idx, gamma_true, sim_subdir):
    """Simulation phase for one gamma: build the truth catalogue, simulate
    the lensing measurement of every cluster, and save the standard
    checkpoint (plus the notebook-compatible data products).

    Checkpoint-aware: if sim_subdir already holds simulated_data.npz the
    phase is skipped, so interrupted runs resume for free.
    """
    os.makedirs(sim_subdir, exist_ok=True)
    data_npz = os.path.join(sim_subdir, "simulated_data.npz")
    if os.path.exists(data_npz):
        print("  simulation checkpoint exists — skipping simulation phase",
              flush=True)
        return

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
        truth, SIM_RADII, fractional_noise=0.2,
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


# ── Inference phase ───────────────────────────────────────────────────────────

def _apply_boost_correction(cluster_obs, data_vectors_fit, cov_matrices_fit,
                            RADII_fit):
    """Multiply each cluster's data vector/covariance by its stack's fitted
    boost(R), from BOOST_FITS (loaded from the `boosts:` config path).
    Clusters with no stack_name (outside every z/richness/delta_m bin, or
    a checkpoint predating stack labelling) are left uncorrected — flagged
    once via a summary count, not per-cluster, to keep the log readable.

    Ported verbatim from gamma_inference.py — see that script's copy for
    the full rationale comment."""
    from cluster_toolkit.boostfactors import boost_powerlaw_at_R

    if "stack_name" not in cluster_obs.colnames:
        print("    WARNING: boosts requested but this checkpoint's "
              "cluster_obs has no 'stack_name' column (predates "
              "prepare_real_data.py's stack labelling) — no boost "
              "correction applied to any cluster", flush=True)
        return data_vectors_fit, cov_matrices_fit

    stack_names = np.asarray(cluster_obs["stack_name"])
    data_vectors_fit = data_vectors_fit.copy()
    cov_matrices_fit = cov_matrices_fit.copy()

    n_corrected, n_missing_stack, n_unknown_stack = 0, 0, 0
    boost_cache = {}
    for i, name in enumerate(stack_names):
        if not name:
            n_missing_stack += 1
            continue
        fit = BOOST_FITS.get(name)
        if fit is None or fit.get("failed"):
            n_unknown_stack += 1
            continue
        boost_vec = boost_cache.get(name)
        if boost_vec is None:
            boost_vec = boost_powerlaw_at_R(
                RADII_fit, fit["B0"], fit["Rs"], fit["alpha"],
            )
            boost_cache[name] = boost_vec
        data_vectors_fit[i] *= boost_vec
        cov_matrices_fit[i] *= np.outer(boost_vec, boost_vec)
        n_corrected += 1

    print(f"    boost correction applied: {n_corrected:,} clusters "
          f"({n_missing_stack:,} no stack, {n_unknown_stack:,} stack not "
          f"in {BOOSTS_FILE!r})", flush=True)
    return data_vectors_fit, cov_matrices_fit


def infer_one_gamma(g_idx, gamma_true, data_subdir, out_subdir, run_t0=None):
    """Inference phase for one gamma: read the checkpointed data from
    ``data_subdir``, run N_BOOTSTRAP bootstrap nautilus fits, and combine
    the posteriors.  All outputs (bootstrap_NNN/, ensemble_summary.json,
    posteriors_overlay.png) are written under ``out_subdir``; the data
    directory is only read, so pre-existing data folders stay pristine.

    When this script also ran the simulation phase, data_subdir and
    out_subdir are the same directory — reproducing gamma_scan.py exactly.
    """
    gamma_t0   = time.time()
    n_total_g  = len(GAMMA_VALUES)
    os.makedirs(out_subdir, exist_ok=True)
    ens_path   = os.path.join(out_subdir, "ensemble_summary.json")

    print("  loading data ...", flush=True)
    truth, ols, cluster_obs, data_vectors, cov_matrices = \
        load_simulation_checkpoint(data_subdir)

    # Two-halo rule (2026-09-26): real data contain the two-halo signal, the
    # simulations (isolated haloes) do not, so the model must match.  Real-data
    # checkpoints (prepare_real_data.py) carry no truth coefficients (ols={}).
    _is_real = not ols
    if TWO_HALO != _is_real:
        raise RuntimeError(
            f"two_halo={TWO_HALO} but {data_subdir} is "
            f"{'REAL' if _is_real else 'SIMULATED'} data: real-data fits must "
            "set two_halo: true, simulated-data fits must not."
        )

    if data_vectors.shape[1] != len(FIT_RADII):
        raise RuntimeError(
            f"radial-bin mismatch: data in {data_subdir} has "
            f"{data_vectors.shape[1]} bins but the fit grid has "
            f"fit_n_radii={len(FIT_RADII)} "
            f"(logspace({FIT_LOG_INNER_RADIUS}, {FIT_LOG_OUTER_RADIUS}, "
            f"{len(FIT_RADII)}))"
        )

    keep             = FIT_RADII >= R_MIN_FIT
    RADII_fit        = FIT_RADII[keep]
    data_vectors_fit = data_vectors[:, keep]
    cov_matrices_fit = cov_matrices[:, keep][:, :, keep]

    if BOOST_FITS is not None:
        data_vectors_fit, cov_matrices_fit = _apply_boost_correction(
            cluster_obs, data_vectors_fit, cov_matrices_fit, RADII_fit,
        )

    # ── Bootstrap loop ───────────────────────────────────────────────────────
    load_phase_dur = time.time() - gamma_t0   # wall time for data loading
    results       = []
    failures      = []
    attempt_idx   = 0
    max_attempts  = 4 * N_BOOTSTRAP
    b_durations   = []   # wall times of non-cached bootstrap fits

    while len(results) < N_BOOTSTRAP and attempt_idx < max_attempts:
        b_subdir     = os.path.join(out_subdir, f"bootstrap_{attempt_idx:03d}")
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
        b_t0 = time.time()
        try:
            r = run_one_bootstrap(
                attempt_idx, out_subdir,
                gamma_true, ols, cluster_obs,
                data_vectors_fit, cov_matrices_fit, RADII_fit,
            )
            results.append(r)
            b_dt = time.time() - b_t0
            b_durations.append(b_dt)
            b_done  = len(results)
            avg_b   = float(np.mean(b_durations))
            eta_b   = avg_b * (N_BOOTSTRAP - b_done)
            est_gamma = load_phase_dur + N_BOOTSTRAP * avg_b
            eta_scan  = eta_b + (n_total_g - g_idx - 1) * est_gamma
            print(
                f"    fit took {_fmt_duration(b_dt)}; "
                f"avg/fit {_fmt_duration(avg_b)}; "
                f"ETA this γ {_fmt_duration(eta_b)}; "
                f"ETA full scan {_fmt_duration(eta_scan)}",
                flush=True,
            )
            # ── Live result after every completed bootstrap ──────────────────
            g_meds_so_far = np.array([res["gamma_med"] for res in results])
            run_mean = float(np.mean(g_meds_so_far))
            run_std  = float(np.std(g_meds_so_far, ddof=1)) if b_done > 1 else float("nan")
            print(
                f"  >>> LIVE [{b_done}/{N_BOOTSTRAP}]  "
                f"γ_true={gamma_true:+.4f}  "
                f"γ_this={r['gamma_med']:+.4f} ± {r['sigma_gamma']:.4f}  "
                f"|  running mean={run_mean:+.4f}  std={run_std:.4f}",
                flush=True,
            )
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
    # Primary measurement: the MIXTURE of the bootstrap posteriors.  Each
    # bootstrap's weights already sum to 1 when saved, so pooling samples
    # with weights/N_b gives every bootstrap equal mass.  The mixture width
    # carries both the within-fit posterior width and the between-bootstrap
    # scatter, so it stays honest even where individual posteriors are
    # overconfident.
    # Quality gate: fits nautilus reports as NOT converged (stopped by the
    # timeout or n_like_max) are excluded from the mixture — their collapsed
    # posteriors would otherwise contaminate it with full weight.  Fits from
    # runs predating the flag (no "converged" key) are included.  If no fit
    # converged, fall back to pooling everything rather than reporting
    # nothing.
    gamma_meds  = np.array([r["gamma_med"]   for r in results])
    sigma_posts = np.array([r["sigma_gamma"] for r in results])
    conv_mask   = [r.get("converged", True) is not False for r in results]
    use_results = [r for r, ok in zip(results, conv_mask) if ok]
    n_converged = len(use_results)
    if n_converged == 0:
        print(
            "    WARNING: no converged fits — combining all bootstraps anyway",
            flush=True,
        )
        use_results = results
    pool_s, pool_w = [], []
    pool_all, pool_all_w = [], []
    combined_param_order = None
    for r in use_results:
        npz_path = os.path.join(
            out_subdir, f"bootstrap_{r['bootstrap']:03d}", "gamma_posterior.npz"
        )
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=True)
        w = data["weights"]
        pool_s.append(data["gamma_samples"])
        pool_w.append(w / w.sum())
        # All-parameter pooling for the combined corner plot.  param_order
        # is fixed for the whole run (PRIORS/FIXED_PARAMS are module-level
        # globals), so every bootstrap should agree; a mismatch (e.g. a
        # checkpoint from a differently-configured run) is skipped rather
        # than silently concatenated into garbage columns.
        this_order = [str(p) for p in data["param_order"]]
        if combined_param_order is None:
            combined_param_order = this_order
        if this_order == combined_param_order:
            pool_all.append(data["all_samples"])
            pool_all_w.append(w / w.sum())

    if pool_s:
        pool_s = np.concatenate(pool_s)
        pool_w = np.concatenate(pool_w)
        pool_w = pool_w / pool_w.sum()
        comb_lo  = float(_wq(pool_s, pool_w, 0.16))
        comb_med = float(_wq(pool_s, pool_w, 0.50))
        comb_hi  = float(_wq(pool_s, pool_w, 0.84))
    else:
        # No saved posteriors (should not happen) — fall back to medians
        comb_med = float(np.median(gamma_meds))
        comb_lo  = comb_med - float(np.median(sigma_posts))
        comb_hi  = comb_med + float(np.median(sigma_posts))
    sigma_comb = (comb_hi - comb_lo) / 2.0

    # Diagnostics: the previous summary statistics
    med_scatter = (float(np.std(gamma_meds, ddof=1)) if len(gamma_meds) > 1
                   else float(sigma_posts[0]))

    ens_result = {
        "g_idx":              g_idx,
        "gamma_true":         float(gamma_true),
        "gamma_ensemble_med": comb_med,
        "gamma_ensemble_lo":  comb_lo,
        "gamma_ensemble_hi":  comb_hi,
        "sigma_gamma":        sigma_comb,
        # diagnostics (pre-mixture statistics)
        "gamma_med_of_meds":  float(np.median(gamma_meds)),
        "sigma_med_scatter":  med_scatter,
        "mean_sigma_post":    float(np.mean(sigma_posts)),
        "n_bootstrap":        len(results),
        "n_converged":        n_converged,
        "n_failures":         len(failures),
        # Aggregated fit diagnostics (NaN for pre-diagnostic runs)
        "median_fit_wall_s":  float(np.median(
            [r["fit_wall_s"] for r in results if "fit_wall_s" in r] or [np.nan])),
        "median_n_eff":       float(np.median(
            [r["n_eff"] for r in results if "n_eff" in r] or [np.nan])),
        # Median information gain (nats) from prior to posterior across
        # this gamma's bootstrap fits — see fit_diag['kl_nats'] per fit.
        "median_kl_nats":     float(np.median(
            [r["kl_nats"] for r in results if "kl_nats" in r] or [np.nan])),
        **ols,
    }
    with open(ens_path, "w") as f:
        json.dump(ens_result, f, indent=2)
    print(
        f"  γ={gamma_true:+.1f}  γ_est={comb_med:+.4f} ± {sigma_comb:.4f} "
        f"(combined posterior, {n_converged}/{len(results)} converged fits)  "
        f"bias={comb_med - gamma_true:+.4f}  "
        f"median KL={ens_result['median_kl_nats']:.2f} nats",
        flush=True,
    )

    # posteriors overlay for this gamma
    try:
        _plot_posteriors_overlay(results, out_subdir, gamma_true)
    except Exception:
        pass

    # Combined corner plot: the mixture posterior over EVERY fit parameter
    # (not just gamma), pooling all converged bootstraps the same way as
    # the gamma-only combination above.
    if pool_all and combined_param_order:
        try:
            samples_all = np.concatenate(pool_all)
            weights_all = np.concatenate(pool_all_w)
            weights_all = weights_all / weights_all.sum()
            corner_path = os.path.join(out_subdir, "combined_corner.png")
            out = _plot_combined_corner(
                samples_all, weights_all, combined_param_order, corner_path,
                ols, gamma_true, n_converged, N_CLUSTERS,
            )
            if out:
                print(f"    combined corner plot → {out}", flush=True)
        except Exception as exc:
            print(f"    combined corner plot skipped: {exc!r}", flush=True)

    return ens_result


# ── Reporting ─────────────────────────────────────────────────────────────────

# Plot implementations live in destacking.plotting; these wrappers inject
# the scan-level configuration so run-loop call sites stay unchanged.

def plot_gamma_scan_results(scan_results, run_dir):
    return _plot_gamma_scan_results(scan_results, run_dir,
                                    n_clusters=N_CLUSTERS,
                                    n_bootstrap=N_BOOTSTRAP)


def _plot_posteriors_overlay(results, sim_subdir, gamma_true):
    return _plot_posteriors_overlay_impl(results, sim_subdir, gamma_true,
                                         gamma_prior=PRIORS["gamma"],
                                         n_clusters=N_CLUSTERS)


def _plot_combined_corner(samples, weights, param_order, out_path, ols,
                          gamma_true, n_bootstrap, n_clusters):
    labels = [PRIOR_LABELS.get(p, p) for p in param_order]
    truths = _param_truths(param_order, ols, gamma_true)
    return _plot_combined_corner_impl(
        samples, weights, param_order, out_path,
        labels=labels, truths=truths, gamma_true=gamma_true,
        n_bootstrap=n_bootstrap, n_clusters=n_clusters, priors=PRIORS,
    )


def print_scan_table(scan_results):
    print("\n" + "=" * 94, flush=True)
    print(
        f'{"g_idx":>6}  {"gamma_true":>10}  {"gamma_est":>10}  '
        f'{"sigma_gamma":>11}  {"bias":>8}  {"bias/sig":>8}  {"KL[nats]":>9}',
        flush=True,
    )
    print("-" * 94, flush=True)
    for r in scan_results:
        bias = r["gamma_ensemble_med"] - r["gamma_true"]
        bsig = bias / max(r["sigma_gamma"], 1e-12)
        kl = r.get("median_kl_nats", np.nan)
        print(
            f'{r["g_idx"]:>6d}  {r["gamma_true"]:>+10.3f}  '
            f'{r["gamma_ensemble_med"]:>+10.4f}  '
            f'{r["sigma_gamma"]:>11.4f}  '
            f'{bias:>+8.4f}  '
            f'{bsig:>+8.2f}  '
            f'{kl:>9.2f}',
            flush=True,
        )
    residuals = np.array([r["gamma_ensemble_med"] - r["gamma_true"] for r in scan_results])
    sig_gammas = np.array([r["sigma_gamma"] for r in scan_results])
    kl_nats = np.array([r.get("median_kl_nats", np.nan) for r in scan_results])
    print("-" * 94, flush=True)
    print(
        f"RMS bias = {np.sqrt(np.mean(residuals**2)):.4f}   "
        f"mean(sigma_gamma) = {np.mean(sig_gammas):.4f}   "
        f"mean(KL) = {np.nanmean(kl_nats):.2f} nats",
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
                "sigma_gamma": round(r["sigma_gamma"], 4),
                "bias":       round(r["gamma_ensemble_med"] - r["gamma_true"], 4),
                "median_kl_nats": (round(r["median_kl_nats"], 3)
                                   if np.isfinite(r.get("median_kl_nats", np.nan))
                                   else None),
            }
            for r in scan_results
        ],
    }
    with open(os.path.join(run_dir, "progress.json"), "w") as f:
        json.dump(prog, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SIMULATION and not INFERENCE:
        print("Nothing to do: simulation and inference are both false.",
              file=sys.stderr)
        return 1
    if not SIMULATION:
        if not DATA_FOLDER:
            print("simulation is false but no data_folder was given — set the "
                  "'data_folder' YAML key (or DATA_FOLDER env var) to a "
                  "directory holding simulation_g*_gamma*/ checkpoints.",
                  file=sys.stderr)
            return 1
        if not os.path.isdir(DATA_FOLDER):
            print(f"data_folder does not exist: {DATA_FOLDER}", file=sys.stderr)
            return 1

    os.makedirs(RUN_DIR, exist_ok=True)
    # Provenance: keep a copy of the run's config in the output root.  Only
    # for a fresh run — on resume (RUN_DIR reused) the original record is kept.
    if _CFG_PATH is not None:
        cfg_dest = os.path.join(RUN_DIR, os.path.basename(_CFG_PATH))
        if not os.path.exists(cfg_dest):
            shutil.copy2(_CFG_PATH, cfg_dest)

    n_resample_actual = N_RESAMPLE if N_RESAMPLE is not None else N_CLUSTERS
    n_total           = len(GAMMA_VALUES)

    mode = (
        "simulation + inference" if SIMULATION and INFERENCE else
        "simulation only"        if SIMULATION else
        f"inference only (data from {DATA_FOLDER})"
    )
    print(
        f"Gamma inference [{mode}]: {n_total} gamma values "
        f"from {GAMMA_VALUES[0]:+.1f} to {GAMMA_VALUES[-1]:+.1f}\n"
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

    scan_results   = []
    n_gamma_failed = 0
    run_t0         = time.time()

    for g_idx, gamma_true in enumerate(GAMMA_VALUES):
        out_subdir  = os.path.join(RUN_DIR, gamma_dir_name(g_idx, gamma_true))
        data_subdir = (out_subdir if SIMULATION
                       else os.path.join(DATA_FOLDER,
                                         gamma_dir_name(g_idx, gamma_true)))
        ens_path = os.path.join(out_subdir, "ensemble_summary.json")

        print(
            f"\n[{g_idx + 1}/{n_total}] gamma={gamma_true:+.1f}  "
            f"({gamma_dir_name(g_idx, gamma_true)})",
            flush=True,
        )
        iter_t0 = time.time()

        try:
            if SIMULATION:
                simulate_one_gamma(g_idx, gamma_true, out_subdir)

            if not INFERENCE:
                continue

            if os.path.exists(ens_path):
                print("  cached — loading ensemble_summary.json", flush=True)
                with open(ens_path) as f:
                    scan_results.append(json.load(f))
                write_progress(scan_results, RUN_DIR, run_t0, n_total)
                continue

            if not os.path.exists(os.path.join(data_subdir,
                                               "simulated_data.npz")):
                print(f"  no data for this gamma point in {data_subdir} — "
                      f"skipping", file=sys.stderr)
                continue

            r = infer_one_gamma(g_idx, gamma_true, data_subdir, out_subdir,
                                run_t0=run_t0)
            scan_results.append(r)
        except KeyboardInterrupt:
            print(
                f"\nInterrupted after {len(scan_results)}/{n_total} gamma values.\n"
                f"Resume with:  RUN_DIR={RUN_DIR} python gamma_inference.py",
                flush=True,
            )
            if INFERENCE:
                _finalize(scan_results, RUN_DIR, run_t0, n_total)
            return 2
        except Exception as exc:
            print(f"  gamma={gamma_true:+.1f} FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_gamma_failed += 1
            continue

        iter_dt = time.time() - iter_t0
        done    = max(len(scan_results), g_idx + 1)
        avg_dt  = (time.time() - run_t0) / max(done, 1)
        eta     = avg_dt * (n_total - done)
        print(
            f"  [{done}/{n_total}] took {_fmt_duration(iter_dt)}; "
            f"ETA {_fmt_duration(eta)}",
            flush=True,
        )
        if INFERENCE:
            write_progress(scan_results, RUN_DIR, run_t0, n_total)

    total_elapsed = time.time() - run_t0
    print(
        f"\nTotal wall time: {_fmt_duration(total_elapsed)}",
        flush=True,
    )

    if not INFERENCE:
        n_ok = n_total - n_gamma_failed
        print(f"Simulation-only run complete: {n_ok}/{n_total} gamma points "
              f"in {RUN_DIR}"
              + (f"  ({n_gamma_failed} FAILED)" if n_gamma_failed else ""),
              flush=True)
        return 0 if n_gamma_failed == 0 else 1

    if not scan_results:
        print("No gamma points completed.", file=sys.stderr)
        return 1

    _finalize(scan_results, RUN_DIR, run_t0, n_total)
    return 0


def _finalize(scan_results, run_dir, run_t0, n_total):
    if not scan_results:
        return
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
