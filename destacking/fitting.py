import numpy as np

from destacking.model import (
    forward_model, forward_model_mc, forward_model_combined,
    _MC_M_PIVOT_LOG, _MC_Z_REF, _MC_DM_REF,
)

from scipy.stats import truncnorm as sp_truncnorm, norm as sp_norm, uniform as sp_uniform


def cholog_gaussian(data_vectors, cov_matrices):
    """
    Simultaneous Gaussian log-likelihood for N data vectors of size k.

    Bins where ``data_vectors[i]`` is NaN are treated as masked: only the
    finite bins participate in the Cholesky likelihood for that cluster.
    This lets per-cluster bin masks (too few expected galaxies) propagate
    from the data-prep step without requiring a separate mask array.

    A cluster with zero valid bins contributes nothing to the likelihood.
    Returns ``-np.inf`` if any cluster yields a non-finite contribution
    on its valid bins, so the nautilus sampler never sees NaN.

    Parameters
    ----------
    data_vectors : ndarray (N, k)
        Per-cluster residual vectors (data minus model).  NaN marks invalid bins.
    cov_matrices : ndarray (N, k, k)
        Per-cluster covariance matrices.  Rows/cols for NaN bins are unused.
    """
    n, k = data_vectors.shape
    total_log_lik = 0.0
    log_2pi = np.log(2 * np.pi)

    for i in range(n):
        valid = np.isfinite(data_vectors[i])      # (k,) — per-bin mask
        if not valid.any():
            continue                               # no valid bins — skip cluster
        dv_i  = data_vectors[i][valid]
        cov_i = cov_matrices[i][np.ix_(valid, valid)]
        k_eff = int(valid.sum())
        try:
            L = np.linalg.cholesky(cov_i)
            y = np.linalg.solve(L, dv_i)
            quad_form = float(np.dot(y, y))
            log_det   = 2.0 * float(np.sum(np.log(np.diag(L))))
            log_lik   = -0.5 * (k_eff * log_2pi + log_det + quad_form)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return -np.inf
        if not np.isfinite(log_lik):
            return -np.inf
        total_log_lik += log_lik

    if not np.isfinite(total_log_lik):
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

    Returns -inf for unphysical parameter values, for forward-model values
    that contain NaN/inf, or if the forward model raises an exception.
    """
    try:
        model     = forward_model(gamma, cluster_table, radii, c_pivot_0, delta_m_ref, log_M_norm)
        if not np.all(np.isfinite(model)):
            return -np.inf
        residuals = data_vectors - model
        ll = cholog_gaussian(residuals, cov_matrices)
        return ll if np.isfinite(ll) else -np.inf
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

    Returns -inf for unphysical c0, for NaN/inf in the forward model, or
    if the forward model raises.
    """
    if c0 <= 0:
        return -np.inf
    try:
        model     = forward_model_mc(c0, alpha, cluster_table, radii,
                                     beta=beta, M_pivot_log=M_pivot_log,
                                     z_ref=z_ref, log_M_norm=log_M_norm)
        if not np.all(np.isfinite(model)):
            return -np.inf
        residuals = data_vectors - model
        ll = cholog_gaussian(residuals, cov_matrices)
        return ll if np.isfinite(ll) else -np.inf
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
    log_M_norm=0.0,
    A=1.0,
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
):
    """
    Log-likelihood for the combined M–c + magnitude-gap model.

    FMISS, SIGMA_OFF, BCG_B_MASS are passed through to forward_model_combined
    as fixed per-cluster miscentering / BCG arrays.  Default None means
    centered NFW with no BCG.  ``A`` is a global multiplicative amplitude
    correction (default 1.0).

    Returns -inf for unphysical c0 or A, for NaN/inf in the forward model,
    or if the forward model raises.  Never returns NaN.
    """
    if c0 <= 0 or A <= 0:
        return -np.inf
    try:
        model     = forward_model_combined(
            c0, alpha, gamma, cluster_table, radii,
            beta=beta, M_pivot_log=M_pivot_log, z_ref=z_ref,
            delta_m_ref=delta_m_ref, log_M_norm=log_M_norm, A=A,
            FMISS=FMISS, SIGMA_OFF=SIGMA_OFF, BCG_B_MASS=BCG_B_MASS,
        )
        if not np.all(np.isfinite(model)):
            return -np.inf
        residuals = data_vectors - model
        ll = cholog_gaussian(residuals, cov_matrices)
        return ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


def run_sampler_combined(
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    priors=None,
    # Legacy individual kwargs — used only when priors is None
    c0_prior_loc=5.0,
    c0_prior_scale=5.0,
    alpha_prior_loc=0.0,
    alpha_prior_scale=0.5,
    beta_prior_loc=0.0,
    beta_prior_scale=1.0,
    gamma_range=(-1.0, 1.0),
    gamma_prior_sigma=0.5,
    log_M_norm_sigma=0.1,
    # Model hyperparameters
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    delta_m_ref=_MC_DM_REF,
    fit_beta=True,
    # Fixed per-cluster miscentering / BCG arrays (not sampled)
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
    n_live=400,
    pool=None,
    verbose=True,
    filepath=None,
    resume=True,
    timeout=600,
):
    """
    Run the nautilus nested sampler for the combined M–c + magnitude-gap model.

    Simultaneously constrains c0 (normalisation), alpha (mass slope),
    optionally beta (redshift evolution), gamma (magnitude-gap slope [dex/mag]),
    and log_M_norm (mass-richness calibration offset).

    The recommended way to declare priors is via the ``priors`` dict.  For
    backward compatibility, when ``priors=None`` the function builds an
    equivalent dict from the individual ``*_prior_*`` legacy kwargs.

    Parameters
    ----------
    cluster_table : astropy.table.Table
        Cluster observables — columns 'z', 'Lambda', and 'delta_m'.
    data_vectors : ndarray (N, M)
    cov_matrices : ndarray (N, M, M)
    radii : array (M,)
    priors : dict[str, scipy.stats distribution] or None
        Mapping from parameter name to a frozen ``scipy.stats`` distribution
        (anything that exposes ``.pdf`` and is accepted by
        ``nautilus.Prior.add_parameter``).
        Required keys: ``'c0'``, ``'alpha'``, ``'gamma'``, ``'log_M_norm'``.
        Required additionally when ``fit_beta=True``: ``'beta'``.
        Optional: ``'A'`` --- global multiplicative amplitude correction.
        If supplied, A is sampled and appended at the END of the parameter
        order (after log_M_norm), preserving the indexing of every earlier
        parameter.  If absent, A is fixed at 1.0.
        Example::

            from scipy.stats import norm, truncnorm
            priors = {
                'c0':         norm(loc=5.0, scale=5.0),
                'alpha':      norm(loc=0.0, scale=0.5),
                'beta':       norm(loc=0.0, scale=1.0),
                'gamma':      truncnorm(a=-2, b=2, loc=0.0, scale=0.5),
                'log_M_norm': norm(loc=0.0, scale=0.1),
                'A':          norm(loc=0.80, scale=0.02),   # optional
            }
        Default ``None`` — fall back to the individual legacy kwargs below.
    c0_prior_loc, c0_prior_scale : float
        Gaussian prior on c0 (legacy, used only when ``priors`` is None).
    alpha_prior_loc, alpha_prior_scale : float
        Gaussian prior on alpha (legacy).
    beta_prior_loc, beta_prior_scale : float
        Gaussian prior on beta (legacy).  Ignored when fit_beta=False.
    gamma_range : (float, float)
        Truncation bounds for the TruncNormal prior on gamma (legacy).
    gamma_prior_sigma : float
        Sigma of the TruncNormal prior on gamma, centred at 0 (legacy).
    log_M_norm_sigma : float
        Gaussian prior sigma on log_M_norm [dex] (legacy).
    M_pivot_log, z_ref : float
        Fixed model hyperparameters (not sampled).
    delta_m_ref : float
        Reference magnitude gap [mag].  Default 1.0.
    fit_beta : bool
        If True, include beta as a free parameter.
        Parameter order: c0, alpha, beta, gamma, log_M_norm, [A].
        If False, fix beta = 0.
        Parameter order: c0, alpha, gamma, log_M_norm, [A].
        A is appended only when ``priors`` contains an ``'A'`` key.
    FMISS : array_like (N,) or None
        Per-cluster miscentered-signal fraction.  Held fixed (not sampled).
        Default None → zeros (centered NFW).
    SIGMA_OFF : array_like (N,) or None
        Per-cluster Rayleigh scale of the miscentering offset distribution
        [Mpc].  Held fixed.  Default None → 0.1 Mpc (irrelevant when FMISS=0).
    BCG_B_MASS : array_like (N,) or None
        Per-cluster baryonic BCG mass [M_sun].  Held fixed.  Default None → 0.
    n_live : int
    pool : int or None
    verbose : bool
    filepath : str or None
        Optional HDF5 path passed straight to ``nautilus.Sampler`` for
        periodic checkpointing.  When set, the sampler dumps its state to
        this file as it makes progress; if the file already exists and
        ``resume=True``, the sampler resumes from that state rather than
        starting over.
    resume : bool
        If ``True`` (default) and ``filepath`` exists, resume from the saved
        state.  Has no effect when ``filepath`` is ``None``.
    timeout : int
        Time in seconds to complete the sampler run.

    Returns
    -------
    sampler : nautilus.Sampler
        Completed sampler.  Call sampler.posterior() for samples.
    """
    from nautilus import Prior, Sampler

    if priors is None:
        lo, hi = gamma_range
        a = (lo - 0.0) / gamma_prior_sigma
        b = (hi - 0.0) / gamma_prior_sigma
        priors = {
            'c0':         sp_norm(loc=c0_prior_loc,    scale=c0_prior_scale),
            'alpha':      sp_norm(loc=alpha_prior_loc, scale=alpha_prior_scale),
            'beta':       sp_norm(loc=beta_prior_loc,  scale=beta_prior_scale),
            'gamma':      sp_truncnorm(a=a, b=b, loc=0.0, scale=gamma_prior_sigma),
            'log_M_norm': sp_norm(loc=0.0, scale=log_M_norm_sigma),
        }

    required = {'c0', 'alpha', 'gamma', 'log_M_norm'}
    if fit_beta:
        required.add('beta')
    missing = required - set(priors)
    if missing:
        raise KeyError(
            f"priors dict is missing required keys: {sorted(missing)}"
        )

    nprior = Prior()
    nprior.add_parameter('c0',    dist=priors['c0'])
    nprior.add_parameter('alpha', dist=priors['alpha'])
    if fit_beta:
        nprior.add_parameter('beta', dist=priors['beta'])
    nprior.add_parameter('gamma',      dist=priors['gamma'])
    nprior.add_parameter('log_M_norm', dist=priors['log_M_norm'])
    sample_A = 'A' in priors
    if sample_A:
        nprior.add_parameter('A', dist=priors['A'])

    def _loglike(p):
        return log_likelihood_combined(
            p['c0'], p['alpha'], p['gamma'],
            cluster_table, data_vectors, cov_matrices, radii,
            beta=p.get('beta', 0.0),
            M_pivot_log=M_pivot_log,
            z_ref=z_ref,
            delta_m_ref=delta_m_ref,
            log_M_norm=p['log_M_norm'],
            A=p.get('A', 1.0),
            FMISS=FMISS,
            SIGMA_OFF=SIGMA_OFF,
            BCG_B_MASS=BCG_B_MASS,
        )

    sampler = Sampler(
        nprior, _loglike,
        n_live=n_live, pool=pool,
        filepath=filepath, resume=resume,
    )
    sampler.run(verbose=verbose,timeout=timeout)
    return sampler


# ---------------------------------------------------------------------------
# Bootstrap-resampled combined likelihood and sampler
# ---------------------------------------------------------------------------

def log_likelihood_combined_resampled(
    c0,
    alpha,
    gamma,
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    n_resample,
    replace=True,
    beta=0.0,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    delta_m_ref=_MC_DM_REF,
    log_M_norm=0.0,
    A=1.0,
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
    rng=None,
):
    """
    Resampled variant of log_likelihood_combined.

    On each call draws a fresh random subset of ``n_resample`` clusters
    (with or without replacement) and evaluates the combined likelihood on
    that subset only.  All per-cluster arrays — cluster_table, data_vectors,
    cov_matrices, FMISS, SIGMA_OFF, BCG_B_MASS — are subsampled with the
    same index set so they remain consistent.

    This implements a bootstrap-over-clusters approach: the stochasticity
    introduced on each likelihood call causes the posterior to broaden,
    reflecting uncertainty from cluster sample variance rather than shape
    noise alone.

    Parameters
    ----------
    n_resample : int
        Number of clusters to draw on each call.  Set to the full sample
        size N for a standard bootstrap (draw N with replacement).
    replace : bool
        If True (default), sample with replacement (bootstrap).
        If False, draw a random subsample without replacement.
    rng : np.random.Generator or None
        Optional RNG instance.  If None, a fresh generator seeded from OS
        entropy is created on each call — the intended behaviour inside a
        sampler where each evaluation must be independently random.

    All other parameters are identical to ``log_likelihood_combined``.
    """
    N = len(data_vectors)
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(N, size=n_resample, replace=replace)

    sub_table  = cluster_table[idx]
    sub_dv     = data_vectors[idx]
    sub_cov    = cov_matrices[idx]
    sub_fmiss  = FMISS[idx]      if FMISS      is not None else None
    sub_sigoff = SIGMA_OFF[idx]  if SIGMA_OFF  is not None else None
    sub_bcg    = BCG_B_MASS[idx] if BCG_B_MASS is not None else None

    return log_likelihood_combined(
        c0, alpha, gamma,
        sub_table, sub_dv, sub_cov, radii,
        beta=beta,
        M_pivot_log=M_pivot_log,
        z_ref=z_ref,
        delta_m_ref=delta_m_ref,
        log_M_norm=log_M_norm,
        A=A,
        FMISS=sub_fmiss,
        SIGMA_OFF=sub_sigoff,
        BCG_B_MASS=sub_bcg,
    )


def run_sampler_combined_resampled(
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    n_resample=None,
    replace=True,
    priors=None,
    # Legacy individual kwargs — used only when priors is None
    c0_prior_loc=5.0,
    c0_prior_scale=5.0,
    alpha_prior_loc=0.0,
    alpha_prior_scale=0.5,
    beta_prior_loc=0.0,
    beta_prior_scale=1.0,
    gamma_range=(-1.0, 1.0),
    gamma_prior_sigma=0.5,
    log_M_norm_sigma=0.1,
    # Model hyperparameters
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    delta_m_ref=_MC_DM_REF,
    fit_beta=True,
    # Fixed per-cluster miscentering / BCG arrays (not sampled)
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
    n_live=400,
    pool=None,
    verbose=True,
    filepath=None,
    resume=True,
    timeout=600,
):
    """
    Run the nautilus sampler with a bootstrap-resampled cluster likelihood.

    Identical to ``run_sampler_combined`` except that on every likelihood
    evaluation a fresh random draw of ``n_resample`` clusters is used,
    implementing an in-sampler bootstrap over the cluster sample.  The
    posterior therefore marginalises over cluster sample variance in
    addition to the usual shape-noise uncertainty.

    Parameters
    ----------
    n_resample : int or None
        Number of clusters drawn on each likelihood call.  Defaults to the
        full sample size N (standard bootstrap with replacement).
    replace : bool
        If True (default), sample with replacement (bootstrap).
        If False, draw without replacement (subsampling).

    All other parameters are identical to ``run_sampler_combined``.
    """
    from nautilus import Prior, Sampler

    N = len(data_vectors)
    if n_resample is None:
        n_resample = N

    if priors is None:
        lo, hi = gamma_range
        a = (lo - 0.0) / gamma_prior_sigma
        b = (hi - 0.0) / gamma_prior_sigma
        priors = {
            'c0':         sp_norm(loc=c0_prior_loc,    scale=c0_prior_scale),
            'alpha':      sp_norm(loc=alpha_prior_loc, scale=alpha_prior_scale),
            'beta':       sp_norm(loc=beta_prior_loc,  scale=beta_prior_scale),
            'gamma':      sp_truncnorm(a=a, b=b, loc=0.0, scale=gamma_prior_sigma),
            'log_M_norm': sp_norm(loc=0.0, scale=log_M_norm_sigma),
        }

    required = {'c0', 'alpha', 'gamma', 'log_M_norm'}
    if fit_beta:
        required.add('beta')
    missing = required - set(priors)
    if missing:
        raise KeyError(
            f"priors dict is missing required keys: {sorted(missing)}"
        )

    nprior = Prior()
    nprior.add_parameter('c0',    dist=priors['c0'])
    nprior.add_parameter('alpha', dist=priors['alpha'])
    if fit_beta:
        nprior.add_parameter('beta', dist=priors['beta'])
    nprior.add_parameter('gamma',      dist=priors['gamma'])
    nprior.add_parameter('log_M_norm', dist=priors['log_M_norm'])
    if 'A' in priors:
        nprior.add_parameter('A', dist=priors['A'])

    def _loglike(p):
        return log_likelihood_combined_resampled(
            p['c0'], p['alpha'], p['gamma'],
            cluster_table, data_vectors, cov_matrices, radii,
            n_resample=n_resample,
            replace=replace,
            beta=p.get('beta', 0.0),
            M_pivot_log=M_pivot_log,
            z_ref=z_ref,
            delta_m_ref=delta_m_ref,
            log_M_norm=p['log_M_norm'],
            A=p.get('A', 1.0),
            FMISS=FMISS,
            SIGMA_OFF=SIGMA_OFF,
            BCG_B_MASS=BCG_B_MASS,
        )

    sampler = Sampler(
        nprior, _loglike,
        n_live=n_live, pool=pool,
        filepath=filepath, resume=resume,
    )
    sampler.run(verbose=verbose, timeout=timeout)
    return sampler