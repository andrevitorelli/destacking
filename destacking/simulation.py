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
_LOG10_Z0      = np.log10(0.35)
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
        (np.log10(M) - _LOG10_M0 - _G_Z * (np.log10(z) - _LOG10_Z0))
        / _F_LAMBDA + _LOG10_LAMBDA0
    )
    Lambda_true = 10.0 ** logLambda_true
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
    norm = np.trapz(z_grid**2, z_grid)
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
    norm = np.trapz((z_grid/z0)**alpha * np.exp(-(z_grid/z0)**beta), z_grid)
    return pdf / norm

def sample_des_y3_source_z(N, rng, z_min=0.0, z_max=2.5,
                           alpha=2.0, beta=1.5, z0=0.4):
    """Draw N source redshifts from des_y3_source_pdf by inverse-CDF."""
    z_grid = np.linspace(max(z_min, 1e-3), z_max, 5000)
    pdf    = des_y3_source_pdf(z_grid, alpha, beta, z0)
    cdf    = np.cumsum(pdf); cdf /= cdf[-1]
    u      = rng.uniform(0, 1, N)
    return np.interp(u, cdf, z_grid)
