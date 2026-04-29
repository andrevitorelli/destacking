import numpy as np
from astropy import table
import xlensing

from destacking.relations import mass_lambda_McClintock18, c_DuttonMaccio
from destacking.fitting import cholog_gaussian
from destacking.simulation import _pivot_from_delta_m

# McClintock+18 pivot constants (mirrored from relations.py for inversion)
_LOG10_LAMBDA0 = np.log10(40.0)
_LOG10_Z0      = np.log10(0.35)
_LOG10_M0      = 14.489
_F_LAMBDA      = 1.356
_G_Z           = -0.30


# ---------------------------------------------------------------------------
# Internal: NFW profile computation
# ---------------------------------------------------------------------------

def _nfw_profiles(z, M200m, C200m, radii):
    """NFW ΔΣ profiles for N clusters at M radii, no miscentering or BCG term.

    Returns array of shape (N, M) in M_sun/pc².
    """
    N = len(z)
    return xlensing.model.NFW_Delta_Sigma(
        M200m      = M200m,
        C200m      = C200m,
        Z          = z,
        FMISS      = np.zeros(N),
        SIGMA_OFF  = np.ones(N) * 0.1,   # irrelevant when FMISS=0
        BCG_B_MASS = np.zeros(N),
        radii      = radii,
    )['NFW Signal']   # (N, M)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def forward_model(
    gamma,
    cluster_table,
    radii,
    c_pivot_0=1.0,
    delta_m_ref=1.0,
    log_M_norm=0.0,
):
    """
    Predict NFW ΔΣ profiles for a set of clusters given parameters.

    The forward model is:
      1. Mass from richness:   M = 10^(mass_lambda_McClintock18(Lambda, z) + log_M_norm)
      2. Pivot from mag gap:   c_param = c_pivot_0 * (Δm / delta_m_ref)^gamma
      3. Concentration:        C = c_DuttonMaccio(z, M, c=c_param)
      4. Profile:              ΔΣ = NFW_Delta_Sigma(M, C, z, ...)

    Parameters
    ----------
    gamma : float
        Power-law index of the pivot–magnitude-gap relation.
    cluster_table : astropy.table.Table
        Must have columns 'z', 'Lambda', 'delta_m'.
        delta_m may contain NaN (clusters without detected satellites);
        those fall back to pivot c_pivot_0.
    radii : array (M,)
        Projected radii in Mpc.
    c_pivot_0 : float
        Pivot parameter at delta_m = delta_m_ref.
    delta_m_ref : float
        Reference magnitude gap [mag] for pivot normalisation.
    log_M_norm : float
        Log10 mass normalisation offset applied to all clusters (dex).
        Marginalised over in run_sampler to absorb mass-richness calibration
        uncertainty.  Default 0 → no correction.

    Returns
    -------
    ndarray (N, M)
        Predicted ΔΣ profiles in M_sun/pc².
    """
    z      = np.asarray(cluster_table['z'])
    Lambda = np.asarray(cluster_table['Lambda'])
    dm     = np.asarray(cluster_table['delta_m'], dtype=float)

    logM  = mass_lambda_McClintock18(Lambda, z) + log_M_norm
    M200m = 10.0 ** logM

    c_param = _pivot_from_delta_m(dm, c_pivot_0, delta_m_ref, gamma)
    C200m   = c_DuttonMaccio(z, M200m, c=c_param)

    return _nfw_profiles(z, M200m, C200m, radii)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

def make_test_data(
    truth_catalog,
    radii,
    fractional_noise=0.2,
    noise_floor=0.01,
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
        Output of simulation.generate_truth_catalog.  Required columns:
        'z', 'M_halo', 'concentration', 'delta_m'.
    radii : array (M,)
        Projected radii in Mpc.
    fractional_noise : float
        Fractional Gaussian noise per profile bin (e.g. 0.2 = 20%).
    noise_floor : float
        Minimum absolute noise [M_sun/pc²] to prevent zero covariance.
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
    rng = np.random.default_rng(seed)

    z  = np.asarray(truth_catalog['z'])
    M  = np.asarray(truth_catalog['M_halo'])
    C  = np.asarray(truth_catalog['concentration'])
    dm = np.asarray(truth_catalog['delta_m'], dtype=float)
    N  = len(z)

    # True profiles from exact truth masses and concentrations
    true_profiles = _nfw_profiles(z, M, C, radii)                  # (N, Nrad)

    # Gaussian noise
    sigma        = np.maximum(fractional_noise * true_profiles, noise_floor)
    data_vectors = true_profiles + rng.normal(0.0, sigma)

    # Diagonal covariance matrices
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
    Lambda_obs  = rng.poisson(Lambda_true).astype(float)
    Lambda_obs  = np.maximum(Lambda_obs, 1.0)   # at least 1 to keep log finite

    cluster_table = table.Table({
        'z':       z,
        'Lambda':  Lambda_obs,
        'delta_m': dm,
    })

    return cluster_table, data_vectors, cov_matrices


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def log_likelihood(
    gamma,
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    c_pivot_0=1.0,
    delta_m_ref=1.0,
    log_M_norm=0.0,
):
    """
    Log-likelihood for gamma given cluster observables and ΔΣ data.

    Evaluates cholog_gaussian on (data − model) residuals.

    Returns -inf for unphysical parameter values or if the forward model
    raises an exception.
    """
    if gamma < 0:
        return -np.inf
    try:
        model     = forward_model(gamma, cluster_table, radii, c_pivot_0, delta_m_ref, log_M_norm)
        residuals = data_vectors - model
        return cholog_gaussian(residuals, cov_matrices)
    except Exception:
        return -np.inf


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def run_sampler(
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    gamma_range=(0.0, 3.0),
    c_pivot_0=1.0,
    delta_m_ref=1.0,
    log_M_norm_sigma=0.1,
    n_live=500,
    pool=None,
    verbose=True,
):
    """
    Run the nautilus nested sampler to constrain gamma and log_M_norm.

    Parameters
    ----------
    cluster_table : astropy.table.Table
        Cluster observables — columns 'z', 'Lambda', 'delta_m'.
    data_vectors : ndarray (N, M)
    cov_matrices : ndarray (N, M, M)
    radii : array (M,)
    gamma_range : (float, float)
        Bounds of the flat prior on gamma.
    c_pivot_0, delta_m_ref : float
        Fixed model hyperparameters (not sampled).
    log_M_norm_sigma : float
        Width (sigma) of the Gaussian prior on log_M_norm [dex].
        Represents mass-richness calibration uncertainty.
    n_live : int
        Number of nautilus live points.
    pool : int or None
        Parallel workers.  None → single process.
    verbose : bool

    Returns
    -------
    sampler : nautilus.Sampler
        Completed sampler.  Call sampler.posterior() for samples.
        Parameter order: gamma (index 0), log_M_norm (index 1).
    """
    from nautilus import Prior, Sampler
    from scipy.stats import uniform as sp_uniform, norm as sp_norm

    prior = Prior()
    prior.add_parameter(
        'gamma',
        dist=sp_uniform(
            loc=gamma_range[0],
            scale=gamma_range[1] - gamma_range[0],
        ),
    )
    prior.add_parameter(
        'log_M_norm',
        dist=sp_norm(loc=0.0, scale=log_M_norm_sigma),
    )

    def _loglike(param_dict):
        return log_likelihood(
            param_dict['gamma'],
            cluster_table,
            data_vectors,
            cov_matrices,
            radii,
            c_pivot_0=c_pivot_0,
            delta_m_ref=delta_m_ref,
            log_M_norm=param_dict['log_M_norm'],
        )

    sampler = Sampler(prior, _loglike, n_live=n_live, pool=pool)
    sampler.run(verbose=verbose)
    return sampler
