import numpy as np
from scipy.linalg import solve_triangular

from destacking.model import (
    forward_model, forward_model_mc, forward_model_combined,
    _MC_M_PIVOT_LOG, _MC_Z_REF, _MC_DM_REF,
)

from scipy.stats import truncnorm as sp_truncnorm, norm as sp_norm, uniform as sp_uniform


def cholog_gaussian(data_vectors, cov_matrices, extra_diag=None):
    """
    Simultaneous fitting of N datavectors of size k.
    Optimized using Cholesky decomposition for better numerical stability.

    Parameters
    ----------
    data_vectors : ndarray (N, k)
        Residual vectors (data minus model).
    cov_matrices : ndarray (N, k, k)
        Per-cluster covariance matrices (e.g. shape noise only).
    extra_diag : ndarray (N, k) or None
        Optional extra variance to add to the diagonal of each covariance
        matrix before factorization.  Used to inject per-cluster intrinsic
        scatter (e.g. from triaxiality) on top of the measurement noise.
    """
    n, k = data_vectors.shape
    total_log_lik = 0.0
    log_2pi = np.log(2 * np.pi)

    for i in range(n):
        cov_i = cov_matrices[i] if extra_diag is None else (
            cov_matrices[i] + np.diag(extra_diag[i])
        )
        try:
            L = np.linalg.cholesky(cov_i)
            # Use triangular solver — correct and stable for a Cholesky factor.
            # np.linalg.solve uses a general LU solver which can raise
            # "Singular matrix" for triangular systems on some LAPACK builds.
            y = solve_triangular(L, data_vectors[i], lower=True)
            quad_form = np.dot(y, y)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            total_log_lik += -0.5 * (k * log_2pi + log_det + quad_form)
        except Exception:
            return -np.inf

    # Guard against silent NaN propagation (e.g. from NaN in extra_diag when
    # the forward model returns non-finite values for extreme parameters).
    return total_log_lik if np.isfinite(total_log_lik) else -np.inf


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
    gamma_range=(-1.0, 1.0),
    gamma_prior_sigma=0.5,
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
        Truncation bounds for the gamma prior.
    gamma_prior_sigma : float
        Sigma of the truncated-normal prior on gamma, centred at 0.
        Slightly informative default (0.5) spans ±2σ within [-1, 1].
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

    lo, hi = gamma_range
    a = (lo - 0.0) / gamma_prior_sigma
    b = (hi - 0.0) / gamma_prior_sigma

    prior = Prior()
    prior.add_parameter(
        'gamma',
        dist=sp_truncnorm(a=a, b=b, loc=0.0, scale=gamma_prior_sigma),
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
    return sampler  # gamma sampler


# ---------------------------------------------------------------------------
# M–c relation: likelihood and sampler
# ---------------------------------------------------------------------------

def log_likelihood_mc(
    c0,
    alpha,
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    beta=0.0,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    log_M_norm=0.0,
):
    """
    Log-likelihood for the M–c relation parameters given ΔΣ data.

    Returns -inf for unphysical c0 or if the forward model raises.
    """
    if c0 <= 0:
        return -np.inf
    try:
        model     = forward_model_mc(c0, alpha, cluster_table, radii,
                                     beta=beta, M_pivot_log=M_pivot_log,
                                     z_ref=z_ref, log_M_norm=log_M_norm)
        residuals = data_vectors - model
        return cholog_gaussian(residuals, cov_matrices)
    except Exception:
        return -np.inf


def run_sampler_mc(
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    c0_prior_loc=5.0,
    c0_prior_scale=5.0,
    alpha_prior_loc=0.0,
    alpha_prior_scale=0.5,
    beta_prior_loc=0.0,
    beta_prior_scale=1.0,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    log_M_norm_sigma=0.1,
    fit_beta=True,
    n_live=300,
    pool=None,
    verbose=True,
):
    """
    Run the nautilus nested sampler to constrain the M–c relation.

    Fits c0 (normalisation), alpha (mass slope), optionally beta (redshift
    evolution), and log_M_norm (mass-richness calibration offset).

    Parameters
    ----------
    cluster_table : astropy.table.Table
        Cluster observables — columns 'z' and 'Lambda'.  'delta_m' unused.
    data_vectors : ndarray (N, M)
    cov_matrices : ndarray (N, M, M)
    radii : array (M,)
    c0_prior_loc : float
        Centre of the Gaussian prior on c0 [dimensionless].
    c0_prior_scale : float
        Sigma of the Gaussian prior on c0.
    alpha_prior_loc : float
        Centre of the Gaussian prior on the mass slope alpha.
    alpha_prior_scale : float
        Sigma of the Gaussian prior on alpha.
    beta_prior_loc : float
        Centre of the Gaussian prior on the redshift-evolution slope beta.
        Ignored when fit_beta=False.
    beta_prior_scale : float
        Sigma of the Gaussian prior on beta.
        Ignored when fit_beta=False.
    M_pivot_log : float
        log10(M_pivot / M_sun).  Fixed; not sampled.
    z_ref : float
        Reference redshift for c0.  Fixed; not sampled.
    log_M_norm_sigma : float
        Gaussian prior sigma on log_M_norm [dex].
    fit_beta : bool
        If True, include beta as a free parameter (3 + 1 = 4 total).
        If False, fix beta = 0 and sample only (c0, alpha, log_M_norm).
    n_live : int
    pool : int or None
    verbose : bool

    Returns
    -------
    sampler : nautilus.Sampler
        Completed sampler.  Call sampler.posterior() for samples.
        Parameter order: c0, alpha, [beta,] log_M_norm.
    """
    from nautilus import Prior, Sampler

    prior = Prior()
    prior.add_parameter(
        'c0',
        dist=sp_norm(loc=c0_prior_loc, scale=c0_prior_scale),
    )
    prior.add_parameter(
        'alpha',
        dist=sp_norm(loc=alpha_prior_loc, scale=alpha_prior_scale),
    )
    if fit_beta:
        prior.add_parameter(
            'beta',
            dist=sp_norm(loc=beta_prior_loc, scale=beta_prior_scale),
        )
    prior.add_parameter(
        'log_M_norm',
        dist=sp_norm(loc=0.0, scale=log_M_norm_sigma),
    )

    def _loglike(p):
        return log_likelihood_mc(
            p['c0'], p['alpha'],
            cluster_table, data_vectors, cov_matrices, radii,
            beta=p.get('beta', 0.0),
            M_pivot_log=M_pivot_log,
            z_ref=z_ref,
            log_M_norm=p['log_M_norm'],
        )

    sampler = Sampler(prior, _loglike, n_live=n_live, pool=pool)
    sampler.run(verbose=verbose)
    return sampler


# ---------------------------------------------------------------------------
# Combined M–c + magnitude-gap: likelihood and sampler
# ---------------------------------------------------------------------------

def log_likelihood_combined(
    c0,
    alpha,
    gamma,
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    beta=0.0,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    delta_m_ref=_MC_DM_REF,
    f_intrinsic=0.0,
    log_M_norm=0.0,
):
    """
    Log-likelihood for the combined M–c + magnitude-gap model.

    Parameters
    ----------
    f_intrinsic : float
        Fractional per-cluster intrinsic scatter added in quadrature to the
        diagonal of each covariance matrix: extra_var[i,b] = (f_int * |model[i,b]|)².
        Represents unmodelled scatter from triaxiality, miscentering, etc.
        Must be >= 0.  Default 0 (shape noise only).

    Returns -inf for unphysical c0/f_intrinsic or if the forward model raises.
    """
    if c0 <= 0 or f_intrinsic < 0:
        return -np.inf
    try:
        model = forward_model_combined(
            c0, alpha, gamma, cluster_table, radii,
            beta=beta, M_pivot_log=M_pivot_log, z_ref=z_ref,
            delta_m_ref=delta_m_ref, log_M_norm=log_M_norm,
        )
        if not np.all(np.isfinite(model)):
            return -np.inf
        residuals = data_vectors - model
        if f_intrinsic > 0:
            extra_diag = (f_intrinsic * np.abs(model)) ** 2
            return cholog_gaussian(residuals, cov_matrices, extra_diag=extra_diag)
        return cholog_gaussian(residuals, cov_matrices)
    except Exception:
        return -np.inf


def run_sampler_combined(
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    c0_prior_loc=5.0,
    c0_prior_scale=5.0,
    alpha_prior_loc=0.0,
    alpha_prior_scale=0.5,
    beta_prior_loc=0.0,
    beta_prior_scale=1.0,
    gamma_range=(-1.0, 1.0),
    gamma_prior_sigma=0.5,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    delta_m_ref=_MC_DM_REF,
    log_M_norm_sigma=0.1,
    fit_beta=True,
    fit_intrinsic=False,
    f_intrinsic_scale=0.5,
    n_live=400,
    pool=None,
    filepath=None,
    verbose=True,
):
    """
    Run the nautilus nested sampler for the combined M–c + magnitude-gap model.

    Simultaneously constrains c0 (normalisation), alpha (mass slope),
    optionally beta (redshift evolution), gamma (magnitude-gap slope [dex/mag]),
    and log_M_norm (mass-richness calibration offset).

    gamma is d(log10 c)/d(Δm [mag]).  Positive gamma shifts all clusters above
    the M–c baseline (since Δm ≥ 0).

    Parameters
    ----------
    cluster_table : astropy.table.Table
        Cluster observables — columns 'z', 'Lambda', and 'delta_m'.
    data_vectors : ndarray (N, M)
    cov_matrices : ndarray (N, M, M)
    radii : array (M,)
    c0_prior_loc, c0_prior_scale : float
        Gaussian prior on c0.
    alpha_prior_loc, alpha_prior_scale : float
        Gaussian prior on alpha.
    beta_prior_loc, beta_prior_scale : float
        Gaussian prior on beta.  Ignored when fit_beta=False.
    gamma_range : (float, float)
        Truncation bounds for the TruncNormal prior on gamma [dex/dex].
    gamma_prior_sigma : float
        Sigma of the TruncNormal prior on gamma, centred at 0.
    M_pivot_log, z_ref : float
        Fixed model hyperparameters (not sampled).
    delta_m_ref : float
        Reference magnitude gap [mag].  Default 1.0.
    log_M_norm_sigma : float
        Gaussian prior sigma on log_M_norm [dex].
    fit_beta : bool
        If True, include beta as a free parameter.
        If False, fix beta = 0.
    fit_intrinsic : bool
        If True, add f_intrinsic as a free nuisance parameter representing
        fractional per-cluster scatter (e.g. from triaxiality).  The covariance
        matrix for each cluster is augmented as
        cov[i,b,b] += (f_intrinsic * |model[i,b]|)².
        Recommended when lensing data are generated from triaxial halos or
        when the shape-noise-only covariance is known to be underestimated.
        Default False (backward compatible).
    f_intrinsic_scale : float
        Upper bound of the Uniform(0, f_intrinsic_scale) prior on f_intrinsic.
        Ignored when fit_intrinsic=False.  Default 0.5.

    Parameter order in posterior (fit_beta=True, fit_intrinsic=True):
        c0, alpha, beta, gamma, f_intrinsic, log_M_norm
    Parameter order (fit_beta=True, fit_intrinsic=False):
        c0, alpha, beta, gamma, log_M_norm
    n_live : int
    pool : int or None
    filepath : str or None
        Path for nautilus checkpoint file.  The sampler saves its state
        periodically; if the file already exists the run resumes from it.
        If None (default) no checkpointing is done.
    verbose : bool

    Returns
    -------
    sampler : nautilus.Sampler
        Sampler object — complete if the run finished normally, otherwise
        containing whatever chains were accumulated before the error.
        Call sampler.posterior() to extract samples.
    """
    import warnings
    from nautilus import Prior, Sampler

    lo, hi = gamma_range
    a = (lo - 0.0) / gamma_prior_sigma
    b = (hi - 0.0) / gamma_prior_sigma

    prior = Prior()
    prior.add_parameter('c0',    dist=sp_norm(loc=c0_prior_loc,    scale=c0_prior_scale))
    prior.add_parameter('alpha', dist=sp_norm(loc=alpha_prior_loc, scale=alpha_prior_scale))
    if fit_beta:
        prior.add_parameter('beta', dist=sp_norm(loc=beta_prior_loc, scale=beta_prior_scale))
    prior.add_parameter('gamma',
        dist=sp_truncnorm(a=a, b=b, loc=0.0, scale=gamma_prior_sigma))
    if fit_intrinsic:
        prior.add_parameter('f_intrinsic',
            dist=sp_uniform(loc=0.0, scale=f_intrinsic_scale))
    prior.add_parameter('log_M_norm', dist=sp_norm(loc=0.0, scale=log_M_norm_sigma))

    def _loglike(p):
        return log_likelihood_combined(
            p['c0'], p['alpha'], p['gamma'],
            cluster_table, data_vectors, cov_matrices, radii,
            beta=p.get('beta', 0.0),
            M_pivot_log=M_pivot_log,
            z_ref=z_ref,
            delta_m_ref=delta_m_ref,
            f_intrinsic=p.get('f_intrinsic', 0.0),
            log_M_norm=p['log_M_norm'],
        )

    sampler = Sampler(prior, _loglike, n_live=n_live, pool=pool,
                      filepath=filepath)
    try:
        sampler.run(verbose=verbose)
    except Exception as exc:
        warnings.warn(
            f"Nautilus sampler raised {type(exc).__name__}: {exc}\n"
            "Returning sampler with chains accumulated so far. "
            "Call sampler.posterior() to extract available samples."
        )
    return sampler