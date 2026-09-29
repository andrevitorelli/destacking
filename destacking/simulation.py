import numpy as np
from astropy import table
from destacking.relations import (
    c_DuttonMaccio,
    lum_BCG_zheng07,
    sample_zheng07_batch,
    sample_halo_masses_tinker08,
    HOD_samples,
)

# McClintock+18 pivot constants (mirrored for inversion in make_test_data)
_LOG10_LAMBDA0 = np.log10(40.0)
_LOG10_Z0      = np.log10(1.0 + 0.35)   # (1+z) form, matches
#                                        relations.mass_lambda_McClintock18
#                                        (changed 2026-09-24)
_LOG10_M0      = 14.489
_F_LAMBDA      = 1.356
_G_Z           = -0.30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_powerlaw_lf(N, alpha, L_min, L_max, rng):
    """
    Sample N luminosities from φ(L) ∝ L^alpha with L in [L_min, L_max].

    Uses the exact inverse-CDF.  Works for any alpha ≠ -1, including the
    typical faint-end slope alpha ≈ -1.2 where alpha+1 < 0.
    """
    ap1 = alpha + 1.0
    u = rng.uniform(0.0, 1.0, N)
    Lmin_p = L_min ** ap1
    Lmax_p = L_max ** ap1
    return (u * (Lmax_p - Lmin_p) + Lmin_p) ** (1.0 / ap1)


def _pivot_from_delta_m(delta_m, c_pivot_0, delta_m_ref, gamma):
    """
    Magnitude-gap-dependent pivot parameter for c_DuttonMaccio.

    c_pivot = c_pivot_0 * (Δm / Δm_ref)^gamma

    Clusters with no detected satellites (Δm = NaN) fall back to c_pivot_0.
    """
    finite = np.isfinite(delta_m)
    c_param = np.full(len(delta_m), c_pivot_0, dtype=float)
    dm = np.where(finite, delta_m, delta_m_ref)   # safe placeholder
    c_param[finite] = c_pivot_0 * (dm[finite] / delta_m_ref) ** gamma
    return c_param


# ---------------------------------------------------------------------------
# Shear-catalog simulation (wraps xlensing)
# ---------------------------------------------------------------------------

def simcluster(gal_density, total_radius, z_range, cluster, shape_noise, seed=31415):
    import xlensing
    cat = xlensing.sims.make_simple_random_cat(
        density=gal_density,
        width_rad=total_radius / xlensing.cosmo.DA(0, 0.3),
        zrange=z_range,
        shape_noise=shape_noise,
        seed=seed,
    ).T
    cat = xlensing.sims.apply_NFW_shear_region(cluster, cat)
    cat = table.Table(cat, names=['ra', 'dec', 'z', 'e1', 'e2', 'R11', 'R12', 'R21', 'R22', 'w'])
    return cat


# ---------------------------------------------------------------------------
# Truth catalog generator
# ---------------------------------------------------------------------------

def generate_truth_catalog(
    N,
    z_range=(0.1, 0.6),
    logM_range=(13.5, 15.0),
    HOD_sample=None,
    delta_m_gamma=0.5,
    c_pivot_0=1.0,
    delta_m_ref=1.0,
    scatter_logc=0.1,
    sat_alpha=-1.2,
    sat_L_min=1e8,
    seed=None,
):
    """
    Generate a random truth catalog of N galaxy clusters.

    Each cluster receives a redshift, halo mass, NFW concentration, BCG
    stellar mass and luminosity, brightest-satellite luminosity, and
    magnitude gap drawn from physically motivated distributions.

    The NFW concentration follows Dutton & Maccio (2014) with a pivot mass
    that scales as a power law of the magnitude gap:

        c_pivot = c_pivot_0 * (Δm / delta_m_ref)^delta_m_gamma

    This encodes the idea that clusters with a large magnitude gap (fossil
    systems) assembled earlier and are more concentrated.

    Parameters
    ----------
    N : int
        Number of clusters to generate.
    z_range : (float, float)
        Uniform redshift range (z_min, z_max).
    logM_range : (float, float)
        log10 mass limits [M_sun] used as integration bounds for the
        Tinker+08 halo mass function (default 13.5–15.0).
    HOD_sample : dict or None
        HOD parameter dictionary (see relations.HOD_samples).  Defaults to
        HOD_samples['SDSS21.0'].
    delta_m_gamma : float
        Power-law index of the pivot–magnitude-gap relation.  gamma > 0
        makes fossil clusters (large Δm) more concentrated.
    c_pivot_0 : float
        Pivot parameter for c_DuttonMaccio when Δm = delta_m_ref.
    delta_m_ref : float
        Reference magnitude gap [mag] for the pivot normalisation.
    scatter_logc : float
        Standard deviation of lognormal intrinsic scatter in concentration
        (in dex).
    sat_alpha : float
        Faint-end slope of the satellite luminosity function (φ ∝ L^alpha).
        Typical value ≈ -1.2.
    sat_L_min : float
        Minimum satellite luminosity [L_sun] (survey detection limit).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    astropy.table.Table
        Columns: z, M_halo [M_sun], concentration, M_BCG [M_sun],
        L_BCG [L_sun], L_brightest_sat [L_sun], delta_m [mag], N_sat.
        Clusters without detected satellites are discarded and resampled,
        so all rows have finite delta_m and N_sat ≥ 1.
    """
    rng = np.random.default_rng(seed)
    # sample_zheng07_batch draws from the global np.random state; seed it too
    # so that a given seed always produces an identical catalog end-to-end.
    if seed is not None:
        np.random.seed(int(np.random.default_rng(seed).integers(2**31)))

    if HOD_sample is None:
        HOD_sample = HOD_samples['SDSS18.0']

    z_keep, M_halo_keep, M_BCG_keep, L_BCG_keep = [], [], [], []
    L_sat_keep, delta_m_keep, N_sat_keep        = [], [], []

    max_attempts = 100
    for _ in range(max_attempts):
        if len(z_keep) >= N:
            break
        n_try = max(N - len(z_keep), 16)

        z_try      = rng.uniform(z_range[0], z_range[1], n_try)
        M_halo_try = sample_halo_masses_tinker08(z_try, logM_range[0], logM_range[1], rng)

        # log10(M_BCG) ≈ 0.4*log10(M_halo) + 6 → ~10^11.6 M_sun at M_halo=10^14
        log_M_BCG_try = 0.4 * np.log10(M_halo_try) + 6.0 + rng.normal(0.0, 0.15, n_try)
        M_BCG_try     = 10.0 ** log_M_BCG_try
        L_BCG_try     = lum_BCG_zheng07(M_BCG_try)

        _, N_sat_try = sample_zheng07_batch(M_halo_try, HOD_sample)

        for i in range(n_try):
            n_sat = int(N_sat_try[i])
            if n_sat == 0 or L_BCG_try[i] <= sat_L_min:
                continue
            L_sats = _sample_powerlaw_lf(n_sat, sat_alpha, sat_L_min, L_BCG_try[i], rng)
            L_max  = L_sats.max()
            # Δm = 2.5 log10(L_BCG / L_brightest_sat) > 0 by construction
            dm     = 2.5 * np.log10(L_BCG_try[i] / L_max)

            z_keep.append(z_try[i])
            M_halo_keep.append(M_halo_try[i])
            M_BCG_keep.append(M_BCG_try[i])
            L_BCG_keep.append(L_BCG_try[i])
            L_sat_keep.append(L_max)
            delta_m_keep.append(dm)
            N_sat_keep.append(n_sat)
            if len(z_keep) >= N:
                break
    else:
        raise RuntimeError(
            f"Could not draw {N} clusters with satellites in {max_attempts} batches; "
            f"check HOD parameters and mass range."
        )

    z               = np.asarray(z_keep[:N])
    M_halo          = np.asarray(M_halo_keep[:N])
    M_BCG           = np.asarray(M_BCG_keep[:N])
    L_BCG           = np.asarray(L_BCG_keep[:N])
    L_brightest_sat = np.asarray(L_sat_keep[:N])
    delta_m         = np.asarray(delta_m_keep[:N])
    N_sat           = np.asarray(N_sat_keep[:N])

    c_param       = _pivot_from_delta_m(delta_m, c_pivot_0, delta_m_ref, delta_m_gamma)
    concentration = c_DuttonMaccio(z, M_halo, c=c_param)
    concentration *= 10.0 ** rng.normal(0.0, scatter_logc, N)   # intrinsic scatter

    return table.Table({
        'z':               z,
        'M_halo':          M_halo,
        'concentration':   concentration,
        'M_BCG':           M_BCG,
        'L_BCG':           L_BCG,
        'L_brightest_sat': L_brightest_sat,
        'delta_m':         delta_m,
        'N_sat':           N_sat,
    })


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

def make_test_data(
    truth_catalog,
    radii,
    fractional_noise=0.2,
    noise_floor=0.01,
    poisson_richness=True,
    seed=None,
    sigma_lnlambda_intr=0.0,
):
    """
    Build synthetic observed profiles and covariances from a truth catalog.

    Profiles are computed from the true mass and concentration (bypassing
    the richness-mass relation to avoid compounding scatter).  Gaussian
    noise is then added.  Richness is assigned by inverting
    mass_lambda_McClintock18 on the true mass and applying Poisson scatter.

    Parameters
    ----------
    truth_catalog : astropy.table.Table
        Output of generate_truth_catalog.  Required columns:
        'z', 'M_halo', 'concentration', 'delta_m'.
    radii : array (M,)
        Projected radii in Mpc.
    fractional_noise : float
        Fractional Gaussian noise per profile bin (e.g. 0.2 = 20%).
    noise_floor : float
        Minimum absolute noise [M_sun/pc²] to prevent zero covariance.
    poisson_richness : bool
        If True (default), apply Poisson scatter to richness Λ around the
        deterministic McClintock+18 inversion of the true mass.  Set to
        False to bypass this scatter — useful when comparing the recovered
        model parameters against truth, since the forward model uses Λ to
        estimate mass and Poisson scatter introduces an errors-in-variables
        bias.
    seed : int or None
    sigma_lnlambda_intr : float
        Lognormal intrinsic scatter in ln(Λ) at fixed mass, applied to the
        deterministic inversion BEFORE any Poisson scatter.  Default 0.0
        draws nothing, so the random stream (and every existing
        simulation) is unchanged.  Added 2026-09-26 to test the joint
        fit's deterministic Λ→M assumption.

    Returns
    -------
    cluster_table : astropy.table.Table
        Observable table — columns 'z', 'Lambda', 'delta_m'.
    data_vectors : ndarray (N, M)
        Noisy ΔΣ profiles.
    cov_matrices : ndarray (N, M, M)
        Diagonal covariance matrices built from the per-bin noise.
    """
    # Local import to avoid circular dependency: model.py imports from this module.
    from destacking.model import _nfw_profiles

    rng = np.random.default_rng(seed)

    z  = np.asarray(truth_catalog['z'])
    M  = np.asarray(truth_catalog['M_halo'])
    C  = np.asarray(truth_catalog['concentration'])
    dm = np.asarray(truth_catalog['delta_m'], dtype=float)
    N  = len(z)

    true_profiles = _nfw_profiles(z, M, C, radii)                  # (N, Nrad)

    sigma        = np.maximum(fractional_noise * true_profiles, noise_floor)
    data_vectors = true_profiles + rng.normal(0.0, sigma)

    Nrad = len(radii)
    cov_matrices = np.zeros((N, Nrad, Nrad))
    for i in range(N):
        np.fill_diagonal(cov_matrices[i], sigma[i] ** 2)

    # Richness: invert McClintock+18 on the true mass, then Poisson scatter
    logLambda_true = (
        (np.log10(M) - _LOG10_M0 - _G_Z * (np.log10(1.0 + z) - _LOG10_Z0))
        / _F_LAMBDA + _LOG10_LAMBDA0
    )
    Lambda_true = 10.0 ** logLambda_true
    if sigma_lnlambda_intr > 0.0:
        Lambda_true = Lambda_true * np.exp(
            rng.normal(0.0, sigma_lnlambda_intr, N))
    if poisson_richness:
        Lambda_obs = rng.poisson(Lambda_true).astype(float)
        Lambda_obs = np.maximum(Lambda_obs, 1.0)   # at least 1 to keep log finite
    else:
        Lambda_obs = Lambda_true.astype(float)

    cluster_table = table.Table({
        'z':       z,
        'Lambda':  Lambda_obs,
        'delta_m': dm,
    })

    return cluster_table, data_vectors, cov_matrices


# ---------------------------------------------------------------------------
# Simulate DES distributions
# ---------------------------------------------------------------------------


def des_y3_cluster_pdf(z, z_min=0.2, z_max=0.65):
    """DES Y3 redMaPPer-like cluster n(z): volume-weighted on [z_min, z_max].

    Approximation: P(z) ∝ z^2 (flat-cosmology volume element), normalised on
    the cluster redshift range.  Captures the rough increase of cluster
    abundance with z over the redMaPPer Y3 range without imposing a specific
    selection function.
    """
    z = np.atleast_1d(z).astype(float)
    in_range = (z >= z_min) & (z <= z_max)
    pdf = np.where(in_range, z**2, 0.0)
    z_grid = np.linspace(z_min, z_max, 1000)
    norm = np.trapezoid(z_grid**2, z_grid)
    return pdf / norm

def sample_des_y3_cluster_z(N, rng, z_min=0.2, z_max=0.65):
    """Draw N cluster redshifts from des_y3_cluster_pdf by inverse-CDF."""
    z_grid = np.linspace(z_min, z_max, 5000)
    pdf    = des_y3_cluster_pdf(z_grid, z_min, z_max)
    cdf    = np.cumsum(pdf); cdf /= cdf[-1]
    u      = rng.uniform(0, 1, N)
    return np.interp(u, cdf, z_grid)

def des_y3_source_pdf(z, alpha=2.0, beta=1.5, z0=0.4):
    """DES Y3 source n(z): Smail-Schechter form  P(z) ∝ (z/z0)^alpha * exp(-(z/z0)^beta).

    Defaults chosen to roughly match the combined DES Y3 metacalibration source
    n(z) (mean ~0.65, peak near 0.4-0.5).  Cite Myles+21, MacCrann+21 for the
    canonical SOMPZ-calibrated n(z).
    """
    z = np.atleast_1d(z).astype(float)
    pdf = np.where(z > 0, (z/z0)**alpha * np.exp(-(z/z0)**beta), 0.0)
    z_grid = np.linspace(1e-3, 5.0, 5000)
    norm = np.trapezoid((z_grid/z0)**alpha * np.exp(-(z_grid/z0)**beta), z_grid)
    return pdf / norm

def sample_des_y3_source_z(N, rng, z_min=0.0, z_max=2.5,
                           alpha=2.0, beta=1.5, z0=0.4):
    """Draw N source redshifts from des_y3_source_pdf by inverse-CDF."""
    z_grid = np.linspace(max(z_min, 1e-3), z_max, 5000)
    pdf    = des_y3_source_pdf(z_grid, alpha, beta, z0)
    cdf    = np.cumsum(pdf); cdf /= cdf[-1]
    u      = rng.uniform(0, 1, N)
    return np.interp(u, cdf, z_grid)


# ---------------------------------------------------------------------------
# Survey simulation pipeline (moved from scripts/gamma_scan.py)
# ---------------------------------------------------------------------------

import multiprocess as _mp
from dataclasses import dataclass

from scipy.interpolate import interp1d as _interp1d

import xlensing.cosmo as _xlc
import xlensing.sims as _xlsims
from xlensing.data import bootstrap_signal, metacal_cluster_lensing
from xlensing.sims import (
    apply_triaxial_NFW_shear_region,
    make_simple_random_cat,
    sample_triaxial_shapes,
)

from destacking.utils import deterministic_seed as _seed

# NB: destacking.model imports from this module (_pivot_from_delta_m), so the
# McClintock pivot constants are imported lazily inside build_truth_catalog
# to avoid a circular import at load time.


@dataclass
class SimulationConfig:
    """Everything the survey-simulation pipeline needs to know.

    Mirrors the configuration constants of scripts/gamma_scan.py so the
    pipeline can be driven by any script (simulation scans today, real-data
    ingestion tools tomorrow) without reading that script's globals.
    """
    # Sample
    n_clusters:      int
    z_cluster_range: tuple            # (z_min, z_max) of the cluster sample
    logm_range:      tuple            # oversampling range for the truth draw
    lambda_min:      float            # richness selection threshold
    delta_m_ref:     float            # magnitude-gap pivot [mag]
    # Source survey
    density:     float                # source density [gal/arcmin^2]
    shape_noise: float                # per-component ellipticity RMS
    width_rad:   float                # aperture PHYSICAL radius [Mpc] --
                                       # converted to angular half-width
                                       # per-cluster via that cluster's own
                                       # DA(z), not a fixed reference z.
                                       # See _process_cluster's comment.
    z_src_range: tuple                # (z_min, z_max) for sources
    z_src_alpha: float                # Smail n(z) parameters
    z_src_beta:  float
    z_src_z0:    float
    # Physics switches
    spherical:    bool
    miscentering: bool
    f_cen:         float              # fraction of correctly centred clusters
    sigma_off_mpc: float              # Rayleigh scale of the offset [Mpc]
    # Measurement
    radii:           np.ndarray       # bin-centre radii [Mpc]
    bin_limits:      np.ndarray       # (n_bins, 2) bin edges [Mpc]
    n_cov_bootstrap: int              # galaxy bootstrap draws per cluster
    # Execution
    n_workers: int = 15
    # Shot-noise variance floor for bootstrap_signal (see _process_cluster's
    # comment) -- None (default) disables it, matching legacy behaviour.
    # Added 2026-09-07: simulated inferences are meant to be a final
    # validation of the exact code used on real data (prepare_real_data.py
    # already applies this floor), so the simulation pipeline should match
    # it too -- boost correction is the one deliberate exception, since
    # simulations have perfect knowledge of which galaxies are background
    # sources and so have nothing to correct for.
    floor_rmax: float = None
    # Lognormal intrinsic scatter in ln(Lambda) at fixed mass, applied in
    # the selection draw (build_truth_catalog -> make_test_data).  0.0
    # (default) reproduces every existing simulation exactly.
    sigma_lnlambda_intr: float = 0.0
    # Triaxial shear model passed to xlensing.sims.apply_triaxial_NFW_shear_region:
    # "exact" (default from 2026-09-28: true lensing of the projected triaxial NFW,
    # with the line-of-sight amplitude) or "pseudo" (the earlier pseudo-elliptical
    # approximation, ~19% low on average; only to reproduce older campaigns).
    triaxial_shear_model: str = "exact"

    @property
    def n_bins(self):
        return len(self.radii)


def build_truth_catalog(gamma_true, g_idx, cfg):
    # WaZP-matched truth: oversample at logM in cfg.logm_range, then keep the
    # rows passing Lambda>cfg.lambda_min and dm>0 until n_clusters pass the
    # cut.  The McClintock+18 Lambda-inversion is run inside make_test_data;
    # we reuse it here purely as a selection filter and discard the noisy
    # dSigma vectors that step also produces.
    from astropy.table import vstack as _vstack

    from destacking.model import _MC_M_PIVOT_LOG, _MC_Z_REF

    batch_size = max(int(cfg.n_clusters / 0.35), 5000)
    selected   = []
    n_sel      = 0
    batch      = 0
    while n_sel < cfg.n_clusters:
        batch += 1
        truth_b = generate_truth_catalog(
            batch_size,
            z_range=cfg.z_cluster_range,
            logM_range=cfg.logm_range,
            delta_m_gamma=0.0, scatter_logc=0.0,
            seed=_seed(g_idx, batch),
        )
        cluster_obs_b, _, _ = make_test_data(
            truth_b, cfg.radii, poisson_richness=True,
            seed=_seed(g_idx, 50_000 + batch),
            sigma_lnlambda_intr=cfg.sigma_lnlambda_intr,
        )
        truth_b["Lambda"] = cluster_obs_b["Lambda"]
        keep = ((truth_b["Lambda"]  > cfg.lambda_min)
              & (truth_b["delta_m"] > 0.0)
              & np.isfinite(truth_b["delta_m"])
              & (truth_b["z"] < cfg.z_cluster_range[1])
              & (truth_b["z"] >= cfg.z_cluster_range[0]))
        truth_b = truth_b[keep]
        selected.append(truth_b)
        n_sel += len(truth_b)
    truth = _vstack(selected)[:cfg.n_clusters]

    dm_s    = np.asarray(truth["delta_m"])
    logdm_s = np.log10(np.maximum(dm_s, 1e-6) / cfg.delta_m_ref)
    truth["concentration"] = np.asarray(truth["concentration"]) * 10.0 ** (
        gamma_true * logdm_s
    )

    logM_s  = np.log10(truth["M_halo"])
    logc_s  = np.log10(truth["concentration"])
    logz1_s = np.log10((1.0 + truth["z"]) / (1.0 + _MC_Z_REF))
    X = np.column_stack(
        [np.ones(cfg.n_clusters), logM_s - _MC_M_PIVOT_LOG, logz1_s, logdm_s]
    )
    coeffs, *_ = np.linalg.lstsq(X, logc_s, rcond=None)
    return truth, {
        "c0_true":        float(10.0 ** coeffs[0]),
        "alpha_true":     float(coeffs[1]),
        "beta_true":      float(coeffs[2]),
        "gamma_true_ols": float(coeffs[3]),
    }


def mass_richness_truth(M_halo, Lambda, z, lambda_piv=40.0, z_piv=0.35):
    """Effective mass-richness relation of a simulated sample.

    OLS of log10 M on log10(Λ/λ_piv) and log10((1+z)/(1+z_piv)), i.e. the
    <log10 M | Λ, z> the joint fit's forward model targets.  With intrinsic
    or Poisson richness scatter this differs from the injected McClintock+18
    inversion (the mass function and the Λ cut shift <M|Λ>), so it -- not
    the injected slopes -- is the truth to compare mass-richness posteriors
    against.  Returns log10M0 (at the pivots), alpha_mr, beta_mr and the
    residual scatter sigma_log10M (dex).
    """
    logM = np.log10(np.asarray(M_halo, dtype=float))
    X = np.column_stack([
        np.ones(len(logM)),
        np.log10(np.asarray(Lambda, dtype=float) / lambda_piv),
        np.log10((1.0 + np.asarray(z, dtype=float)) / (1.0 + z_piv)),
    ])
    coeffs, *_ = np.linalg.lstsq(X, logM, rcond=None)
    resid = logM - X @ coeffs
    return {
        "logM0_mr_true":       float(coeffs[0]),
        "alpha_mr_true":       float(coeffs[1]),
        "beta_mr_true":        float(coeffs[2]),
        "sigma_log10M_lambda": float(np.std(resid)),
    }


def compute_triaxial_shapes(truth, g_idx):
    rng_tri = np.random.default_rng(_seed(g_idx, 10))
    return sample_triaxial_shapes(
        truth["z"], truth["M_halo"],
        ba_range=(0.5, 1.0), ca_lo=0.3, isotropic_los=True, rng=rng_tri,
    )


def compute_miscentering(g_idx, cfg):
    if not cfg.miscentering:
        # All clusters perfectly centred — used for the no-miscentering
        # sensitivity test selected via miscentering: false in YAML.
        return (np.zeros(cfg.n_clusters, dtype=bool),
                np.zeros(cfg.n_clusters),
                np.zeros(cfg.n_clusters))
    rng_mis     = np.random.default_rng(_seed(g_idx, 20))
    is_mis      = rng_mis.uniform(0, 1, cfg.n_clusters) > cfg.f_cen
    off_mag_mpc = np.where(
        is_mis, rng_mis.rayleigh(scale=cfg.sigma_off_mpc, size=cfg.n_clusters),
        0.0,
    )
    phi_mis = rng_mis.uniform(0, 2 * np.pi, cfg.n_clusters)
    return is_mis, off_mag_mpc, phi_mis


# Worker state shared with _process_cluster through fork (set by
# simulate_lensing immediately before the pool dispatch).
_CFG         = None
_TRUTH       = None
_SHAPES      = None
_OFF_MAG_MPC = None
_PHI_MIS     = None
_DA_FAST     = None
_G_IDX_SIM   = 0


def _process_cluster(i):
    """Simulate and measure the lensing signal of one cluster, end to end.

    Runs inside a ``multiprocess.Pool`` worker.  All shared inputs come
    from the module globals set by :func:`simulate_lensing` before the
    pool dispatch (``_CFG``, ``_TRUTH``, ``_SHAPES``, ``_OFF_MAG_MPC``,
    ``_PHI_MIS``, ``_DA_FAST``, ``_G_IDX_SIM``) — the function's only
    argument is the cluster index, so ``pool.imap`` can stream over
    ``range(cfg.n_clusters)``.

    Pipeline stages
    ---------------
    1. Draw a random source-galaxy catalog around the cluster (Poisson
       density ``cfg.density`` gal/arcmin^2, aperture physical radius
       ``cfg.width_rad`` Mpc, converted to this cluster's own angular
       half-width via its DA(z)).
    2. Assign DES Y3-like source redshifts (Smail distribution), lognormal
       metacal responses, and intrinsic shapes (per-component Gaussian
       with RMS ``cfg.shape_noise``, redrawn until |e| < 1).
    3. Cut to sources behind the cluster (z_s > 1.1 z_l + 0.2 — the
       "margin" background-selection strategy's default; this is the only
       place that filter is applied, since metacal_cluster_lensing itself
       no longer filters by redshift).
    4. Shear the sources with the triaxial NFW model using the cluster's
       true (M, c, shape, orientation), scaling each galaxy's shear by its
       metacal response so the response-corrected estimator is unbiased;
       ``cfg.spherical`` collapses the triaxial code to the spherical
       Wright-Brainerd profile by setting b = c = a.
    5. Measure from a possibly miscentered position (offset ``_OFF_MAG_MPC``
       at angle ``_PHI_MIS``) via ``metacal_cluster_lensing``.
    6. Estimate the per-cluster data vector and covariance with
       ``bootstrap_signal`` — the same estimator, same shot-noise variance
       floor (``sigma_e=cfg.shape_noise``, ``floor_rmax=cfg.floor_rmax``) as
       the real-data measurement (prepare_real_data.py), added 2026-09-07 so
       a simulated inference validates the exact code path real data goes
       through. No boost correction is the one deliberate difference:
       simulations have perfect knowledge of which galaxies are background
       sources, so there is nothing to correct for. Bins that fail the
       bootstrap (too few galaxies for a finite, positive variance even
       after the floor) are NaN in the data vector and are dropped from
       this cluster's likelihood by cholog_gaussian.

    Reproducibility: every random stage uses ``_seed(g_idx, offset + i)``
    with disjoint offset blocks (1000 shapes, 4000 positions, 5000
    redshifts, 6000 responses), so any cluster can be regenerated
    identically without running the others.

    Returns
    -------
    (i, dv, cov, bg_arr) : tuple
        ``i``      — the cluster index (echoed back for imap bookkeeping).
        ``dv``     — (n_bins,) bootstrap-mean dSigma_t [Msun/pc^2], NaN in
                     failed bins.
        ``cov``    — (n_bins, n_bins) bootstrap covariance; failed
                     rows/columns hold inert identity placeholders that
                     the likelihood never reads.
        ``bg_arr`` — (6, N_src) float32 stake (Sigma_crit, e_t, e_x, W, R, M)
                     kept so notebooks can re-stack the raw signal.
    (i, None, None, None)
        Cluster rejected: fewer than 20 sources behind it, fewer than 20
        inside the measurement aperture, or no bin survived the bootstrap.
        The caller leaves its data vector all-NaN, so it is skipped by the
        likelihood rather than contributing zeros.
    """
    cfg = _CFG
    # True cluster properties for index i (gamma enters through _TRUTH's
    # concentrations, generated upstream by build_truth_catalog).
    z_i = float(_TRUTH["z"][i])
    M_i = float(_TRUTH["M_halo"][i])
    c_i = float(_TRUTH["concentration"][i])
    a_i = _SHAPES["r_vir"][i]
    g_i = _G_IDX_SIM

    # cfg.width_rad is a PHYSICAL target aperture radius [Mpc] (matching
    # the RADIUS_MPC=5.0 convention used everywhere else in the codebase --
    # xlensing.data.stacked_signal calls, prepare_real_data.py,
    # scripts/auxiliary_scripts/gen_sim_data_for_notebooks.py), converted
    # to an angular half-width using THIS cluster's own DA_i -- not a
    # single fixed reference redshift. Found 2026-09-05: every version of
    # this pipeline back to the original prototype notebook (which called
    # simcluster(total_radius=5, ...)) converted a physical radius using a
    # SINGLE fixed z=0.3 reference (destacking.simulation.simcluster,
    # DA(0, 0.3)) rather than each cluster's actual z, so low-z clusters'
    # simulated field was smaller in physical Mpc than the stacking
    # analysis's fixed-Mpc radial grid needs, while high-z clusters got an
    # unnecessarily large one. gen_sim_data_for_notebooks.py (a separate,
    # simpler standalone generator) already does this correctly
    # per-cluster; this ports that same pattern into the shared
    # _process_cluster pipeline every gamma_inference*.py script uses.
    # 0.3 Mpc buffer beyond the target radius, same convention as that
    # script, so the actual measurement radius (Stage 5 below) sits safely
    # inside the generated field's edge, not exactly on it.
    DA_i = _DA_FAST(0, z_i)
    width_rad_i = (cfg.width_rad + 0.3) / DA_i

    # Stage 1 — random source positions (metacal catalog layout:
    # ra, dec, z, e1, e2, R11, R12, R21, R22, w).  Shapes/z/responses are
    # placeholders here and get overwritten below with their own streams.
    gal_i  = make_simple_random_cat(
        cfg.density, width_rad_i, cfg.z_src_range, shape_noise=0.0,
        seed=_seed(g_i, 4000 + i), poisson_density=True,
    ).T
    n_gals = len(gal_i)

    # Stage 2a — source redshifts from the DES Y3 Smail fit.
    rng_src = np.random.default_rng(_seed(g_i, 5000 + i))
    gal_i[:, 2] = sample_des_y3_source_z(
        n_gals, rng_src,
        z_min=cfg.z_src_range[0], z_max=cfg.z_src_range[1],
        alpha=cfg.z_src_alpha, beta=cfg.z_src_beta, z0=cfg.z_src_z0,
    )

    # Stage 2b — diagonal metacal response tied to a lognormal size proxy
    # (small/noisy galaxies respond less); off-diagonal terms zero.
    rng_resp = np.random.default_rng(_seed(g_i, 6000 + i))
    sizes    = rng_resp.lognormal(mean=np.log(0.5), sigma=0.5, size=n_gals)
    R_RESP   = np.clip(1.0 - (sizes / 1.5) ** 2, 0.05, 0.99)
    gal_i[:, 5] = R_RESP
    gal_i[:, 6] = 0.0
    gal_i[:, 7] = 0.0
    gal_i[:, 8] = R_RESP

    # Stage 2c — intrinsic shapes: per-component Gaussian with |e| < 1
    # enforced by redraw.  ngmix's GPriorBA sampler is unreliable at small
    # sigma (its rejection sampling floors at rms ~ 0.04) and its (1-g^2)^2
    # truncation shifts the delivered rms below sigma, so we draw directly:
    # cfg.shape_noise is the per-component ellipticity RMS by construction.
    rng_sh = np.random.default_rng(_seed(g_i, 1000 + i))
    e1 = rng_sh.normal(0.0, cfg.shape_noise, n_gals)
    e2 = rng_sh.normal(0.0, cfg.shape_noise, n_gals)
    bad = e1 ** 2 + e2 ** 2 >= 1.0
    while bad.any():
        n_bad = int(bad.sum())
        e1[bad] = rng_sh.normal(0.0, cfg.shape_noise, n_bad)
        e2[bad] = rng_sh.normal(0.0, cfg.shape_noise, n_bad)
        bad = e1 ** 2 + e2 ** 2 >= 1.0
    gal_i[:, 3] = e1
    gal_i[:, 4] = e2

    # Stage 3 — keep only sources behind the cluster, applied early so we
    # only shear survivors. This is the "margin" background-selection
    # strategy (destacking.data.background_mask_margin's default
    # factor/offset) applied directly rather than via that helper, since
    # this is per-galaxy array code, not a call into metacal_cluster_lensing
    # (which itself applies no redshift filtering any more — see
    # destacking.data.select_background and project memory
    # "xlensing background selection moved to caller").
    mc_mask = gal_i[:, 2] > 1.1 * z_i + 0.2
    gal_mc  = gal_i[mc_mask]
    if len(gal_mc) < 20:
        return i, None, None, None

    # Stage 4 — inject the NFW shear.  Cluster tuple layout expected by
    # apply_triaxial_NFW_shear_region:
    # (ra, dec, z, M200, c200, a, b, c, theta_los, phi_los), axes in Mpc.
    # The per-galaxy metacal response (col 5; R11 = R22, off-diag 0) scales
    # the injected shear so the response-corrected signal() estimator
    # recovers the true amplitude — without it every dSigma comes out
    # inflated by 1/<Rt> ~ 1.2.
    resp_i = gal_mc[:, 5]
    if cfg.spherical:
        # Pure spherical NFW: set b=c=a so the projected ellipse degenerates
        # to a circle and the triaxial-shear pipeline collapses analytically
        # to the Wright-Brainerd 1999 spherical-NFW formula.  This reuses the
        # fast vectorised path of apply_triaxial_NFW_shear_region rather than
        # the per-galaxy Python loop in apply_NFW_shear_region.
        cluster_arr = [
            0.0, 0.0, z_i, M_i, c_i, a_i,
            a_i, a_i,                                # b = c = a -> sphere
            _SHAPES["theta_los"][i], _SHAPES["phi_los"][i],
        ]
        gal_sheared = apply_triaxial_NFW_shear_region(cluster_arr, gal_mc,
                                                      response=resp_i,
                                                      model=_CFG.triaxial_shear_model)
    else:
        cluster_arr = [
            0.0, 0.0, z_i, M_i, c_i, a_i,
            _SHAPES["q_ba"][i] * a_i, _SHAPES["q_ca"][i] * a_i,
            _SHAPES["theta_los"][i], _SHAPES["phi_los"][i],
        ]
        gal_sheared = apply_triaxial_NFW_shear_region(cluster_arr, gal_mc,
                                                      response=resp_i,
                                                      model=_CFG.triaxial_shear_model)

    # Stage 5 — measure from the (possibly miscentered) observed position.
    # Column reorder puts the weight (col 9) after the shapes, matching the
    # metacal_cluster_lensing source layout
    # (ra, dec, z, e1, e2, w, R11, R12, R21, R22).  The true center is at
    # the origin; the offset (drawn in Mpc upstream) converts to radians
    # through the angular diameter distance.
    sources_mc = gal_sheared[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]]
    # DA_i already computed above (per-cluster aperture conversion) -- reused
    # here, not recomputed.
    off_rad_i  = _OFF_MAG_MPC[i] / DA_i
    cen_ra_i   = off_rad_i * np.cos(_PHI_MIS[i])
    cen_dec_i  = off_rad_i * np.sin(_PHI_MIS[i])

    # cfg.width_rad is now the physical target radius [Mpc] directly (see
    # Stage 1's comment) -- the generated field extends 0.3 Mpc beyond
    # this, so the measurement radius sits safely inside its edge.
    bg = metacal_cluster_lensing((cen_ra_i, cen_dec_i, z_i), sources_mc.T,
                                  cfg.width_rad)
    if bg["Count"] < 20:
        return i, None, None, None

    # Standard xlensing stake layout: rows (Sigma_crit, e_t, e_x, W, R, M).
    stake = np.array([
        bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
        bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"],
    ])

    # Stage 6 — data vector and covariance from the same bootstrap estimator
    # (signal(), no boost correction) so both are consistent with the NFW
    # model in the likelihood, which also does not include a boost. No
    # boost correction is the one deliberate difference from the real-data
    # measurement (prepare_real_data.py): simulations have perfect
    # knowledge of which galaxies are background sources, so there is
    # nothing to correct for. Everything else uses the identical
    # bootstrap_signal call, including the sigma_e/floor_rmax shot-noise
    # variance floor (added 2026-09-07 -- see cfg.floor_rmax's docstring)
    # so a simulated inference is a genuine validation of the same code
    # path the real data goes through, not a different one that happens
    # to look similar. bootstrap_signal handles failure per bin: bins
    # without enough galaxies for a finite, positive bootstrap variance
    # (post-floor) come back as NaN, and those NaN entries stay in the
    # data vector so cholog_gaussian drops exactly the failed bins from
    # this cluster's likelihood.
    sigmas, sigmas_cov, _, _ = bootstrap_signal(
        stake, cfg.bin_limits, Nboot=cfg.n_cov_bootstrap,
        sigma_e=cfg.shape_noise, floor_rmax=cfg.floor_rmax,
    )
    valid_bins = np.isfinite(sigmas)
    if not valid_bins.any():
        return i, None, None, None
    dv = sigmas

    # Failed bins get placeholder identity entries in the stored covariance;
    # they are never used because cholog_gaussian masks on NaN in dv.
    cov = np.where(np.isfinite(sigmas_cov), sigmas_cov, 0.0)
    cov[~valid_bins, ~valid_bins] = 1.0

    # Float32 bg for storage; downstream stacked_signal doesn't need precision.
    bg_arr = stake.astype("float32")
    return i, dv, cov, bg_arr


def simulate_lensing(truth, shapes, mis, g_idx, cfg):
    global _CFG, _TRUTH, _SHAPES, _OFF_MAG_MPC, _PHI_MIS, _DA_FAST, _G_IDX_SIM

    z_tab    = np.linspace(0.0, 3.0, 5000)
    chi_tab  = _xlc.cosmology.comoving_distance(z_tab).value
    chi_of_z = _interp1d(z_tab, chi_tab, kind="linear",
                         bounds_error=False, fill_value="extrapolate")

    def DA_fast(z1, z2):
        z2_a = np.atleast_1d(np.asarray(z2, dtype=float))
        res  = (chi_of_z(z2_a) - float(chi_of_z(float(z1)))) / (1.0 + z2_a)
        return float(res[0]) if np.ndim(z2) == 0 else res

    _xlc.DA    = DA_fast
    _xlsims.DA = DA_fast
    _CFG          = cfg
    _TRUTH        = truth
    _SHAPES       = shapes
    _, _OFF_MAG_MPC, _PHI_MIS = mis
    _DA_FAST      = DA_fast
    _G_IDX_SIM    = g_idx

    # Clusters that fail _process_cluster keep an all-NaN data vector, so
    # cholog_gaussian skips them entirely (per-bin NaN masking).  The identity
    # covariance is a placeholder that is never read for skipped entries.
    data_vectors = np.full((cfg.n_clusters, cfg.n_bins), np.nan)
    cov_matrices = np.zeros((cfg.n_clusters, cfg.n_bins, cfg.n_bins))
    for i in range(cfg.n_clusters):
        np.fill_diagonal(cov_matrices[i], 1.0)
    bgs          = [None] * cfg.n_clusters

    with _mp.Pool(cfg.n_workers) as pool:
        for idx, dv, cov, bg in pool.imap(
            _process_cluster, range(cfg.n_clusters), chunksize=10
        ):
            if dv is not None:
                data_vectors[idx] = dv
                cov_matrices[idx] = cov   # full (n_bins, n_bins) matrix
                bgs[idx]          = bg

    # Drop the per-run module state the (now closed) worker pool read, so a
    # multi-gamma job does not keep the previous point's truth/shapes alive.
    _TRUTH = _SHAPES = _OFF_MAG_MPC = _PHI_MIS = None

    return data_vectors, cov_matrices, bgs
