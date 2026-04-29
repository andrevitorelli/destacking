import numpy as np
from astropy import table
from destacking.relations import (
    c_DuttonMaccio,
    lum_BCG_zheng07,
    sample_zheng07_batch,
    HOD_samples,
)


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
        Uniform range in log10(M_halo / M_sun).
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
        L_brightest_sat and delta_m are NaN for clusters with no detected
        satellites.
    """
    rng = np.random.default_rng(seed)
    # sample_zheng07_batch draws from the global np.random state; seed it too
    # so that a given seed always produces an identical catalog end-to-end.
    if seed is not None:
        np.random.seed(int(np.random.default_rng(seed).integers(2**31)))

    if HOD_sample is None:
        HOD_sample = HOD_samples['SDSS21.0']

    # --- Redshifts and halo masses ---
    z     = rng.uniform(z_range[0],    z_range[1],    N)
    logM  = rng.uniform(logM_range[0], logM_range[1], N)
    M_halo = 10.0 ** logM

    # --- BCG stellar mass and luminosity ---
    # Abundance-matching inspired scaling: log10(M_BCG) ≈ 0.4*log10(M_halo) + 6
    # gives M_BCG ~ 10^11.6 M_sun at M_halo = 10^14, with 0.15 dex scatter.
    log_M_BCG = 0.4 * logM + 6.0 + rng.normal(0.0, 0.15, N)
    M_BCG  = 10.0 ** log_M_BCG
    L_BCG  = lum_BCG_zheng07(M_BCG)    # [L_sun]

    # --- Number of satellites from HOD ---
    _, N_sat = sample_zheng07_batch(M_halo, HOD_sample)

    # --- Satellite luminosities → magnitude gap ---
    # Satellites are sampled from a power-law LF with L in [sat_L_min, L_BCG],
    # so the BCG is guaranteed brighter than every satellite by construction.
    L_brightest_sat = np.full(N, np.nan)
    delta_m         = np.full(N, np.nan)

    for i in range(N):
        n = int(N_sat[i])
        if n == 0 or L_BCG[i] <= sat_L_min:
            continue
        L_sats = _sample_powerlaw_lf(n, sat_alpha, sat_L_min, L_BCG[i], rng)
        L_brightest_sat[i] = L_sats.max()
        # Δm = m_brightest_sat − m_BCG = 2.5 log10(L_BCG / L_brightest_sat) > 0
        delta_m[i] = 2.5 * np.log10(L_BCG[i] / L_brightest_sat[i])

    # --- Concentration with magnitude-gap-dependent pivot ---
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
