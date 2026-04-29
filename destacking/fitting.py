import numpy as np

from destacking.model import forward_model


def cholog_gaussian(data_vectors, cov_matrices):
    """
    Simultaneous fitting of N datavectors of size k.
    Optimized using Cholesky decomposition for better numerical stability
    """
    n, k = data_vectors.shape
    total_log_lik = 0
    log_2pi = np.log(2 * np.pi)
    
    for i in range(n):
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_matrices[i])
            # Solve L * y = x (forward substitution)
            y = np.linalg.solve(L, data_vectors[i])
            # Compute quadratic form
            quad_form = np.sum(y**2)
            # Log determinant
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Log likelihood
            log_lik = -0.5 * (k * log_2pi + log_det + quad_form)
            total_log_lik += log_lik
            
        except np.linalg.LinAlgError:
            return -np.inf

    return total_log_lik


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