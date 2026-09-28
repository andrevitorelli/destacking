import numpy as np

from destacking.model import (
    forward_model, forward_model_mc, forward_model_combined,
    forward_model_diemer,
    _MC_M_PIVOT_LOG, _MC_Z_REF, _MC_DM_REF,
)

from scipy.stats import truncnorm as sp_truncnorm, norm as sp_norm, uniform as sp_uniform
from scipy.stats import gaussian_kde


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

    Vectorized over clusters (no per-cluster Python loop): profiling on a
    real 80,000-cluster likelihood call showed the old per-cluster loop
    (``np.linalg.cholesky``/``solve`` on each cluster's tiny k×k matrix
    individually) cost ~93% of the total call time, almost entirely
    Python-loop and numpy-dispatch overhead rather than arithmetic (e.g.
    building the ``np.ix_`` mask index cost more than the Cholesky
    decomposition itself). Each cluster's own (possibly different) set of
    NaN bins is handled by replacing, per cluster, the invalid rows/cols of
    its covariance with a decoupled identity block and its invalid
    residual entries with 0 -- this contributes exactly 0 to that
    cluster's quadratic form and log-determinant (log(1)=0), and the
    per-cluster normalization constant still uses that cluster's true
    k_eff (valid-bin count), not the padded k. A block-diagonal matrix's
    Cholesky factor is the direct sum of each block's factor, so this is
    algebraically identical to the old valid-submatrix-only computation,
    just done for every cluster in one batched (N,k,k) call instead of N
    separate Python-level ones. See tests/test_fitting.py for the
    equivalence checks (including against the pre-vectorization
    implementation) this relies on.

    Parameters
    ----------
    data_vectors : ndarray (N, k)
        Per-cluster residual vectors (data minus model).  NaN marks invalid bins.
    cov_matrices : ndarray (N, k, k)
        Per-cluster covariance matrices.  Rows/cols for NaN bins are unused.
    """
    n, k = data_vectors.shape
    log_2pi = np.log(2 * np.pi)

    valid = np.isfinite(data_vectors)                  # (n, k)
    k_eff = valid.sum(axis=1)                          # (n,)
    dv    = np.where(valid, data_vectors, 0.0)         # (n, k)

    invalid = ~valid
    mask2d  = invalid[:, :, None] | invalid[:, None, :]   # (n, k, k)
    cov     = np.where(mask2d, 0.0, cov_matrices)

    idx_n = np.arange(n)[:, None]
    idx_k = np.arange(k)[None, :]
    diag_cur = cov[idx_n, idx_k, idx_k]
    cov[idx_n, idx_k, idx_k] = np.where(invalid, 1.0, diag_cur)

    try:
        L = np.linalg.cholesky(cov)                     # batched, (n, k, k)
        # dv[..., None]: solve's batched-vector-RHS broadcasting is
        # ambiguous when b has exactly one fewer dimension than a (numpy
        # 2.2 dispatches it as a single (m,n) matrix RHS, not a batch of
        # vectors) -- an explicit trailing axis of size 1 makes the "batch
        # of single-column matrices" case unambiguous instead.
        y = np.linalg.solve(L, dv[..., None])[..., 0]   # batched, (n, k)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return -np.inf

    quad_form = np.sum(y * y, axis=1)                                       # (n,)
    log_det   = 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)  # (n,)
    log_lik   = -0.5 * (k_eff * log_2pi + log_det + quad_form)              # (n,)

    if not np.all(np.isfinite(log_lik)):
        return -np.inf

    total_log_lik = float(np.sum(log_lik))
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
    alpha_cm,
    gamma_cm,
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    beta_cm=0.0,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    delta_m_ref=_MC_DM_REF,
    log_M_norm=0.0,
    alpha_mr=1.356,
    beta_mr=-0.30,
    A=1.0,
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
    two_halo=False,
    bin_nodes=0,
):
    """
    Log-likelihood for the combined M–c + magnitude-gap model.

    FMISS, SIGMA_OFF, BCG_B_MASS are passed through to forward_model_combined
    as fixed per-cluster miscentering / BCG arrays.  Default None means
    centered NFW with no BCG.  ``alpha_mr``/``beta_mr`` are the
    mass-richness relation's richness/redshift slopes (default McClintock+18
    values).  ``A`` is a global multiplicative amplitude correction
    (default 1.0).

    Returns -inf for unphysical c0 or A, for NaN/inf in the forward model,
    or if the forward model raises.  Never returns NaN.
    """
    if c0 <= 0 or A <= 0:
        return -np.inf
    try:
        model     = forward_model_combined(
            c0, alpha_cm, gamma_cm, cluster_table, radii,
            beta_cm=beta_cm, M_pivot_log=M_pivot_log, z_ref=z_ref,
            delta_m_ref=delta_m_ref, log_M_norm=log_M_norm,
            alpha_mr=alpha_mr, beta_mr=beta_mr, A=A,
            FMISS=FMISS, SIGMA_OFF=SIGMA_OFF, BCG_B_MASS=BCG_B_MASS,
            two_halo=two_halo,
            bin_nodes=bin_nodes,
        )
        if not np.all(np.isfinite(model)):
            return -np.inf
        residuals = data_vectors - model
        ll = cholog_gaussian(residuals, cov_matrices)
        return ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


# Every parameter the combined model exposes, in the order they are added
# to the nautilus Prior when sampled.  c0, alpha_cm, gamma_cm and
# log_M_norm have no hardcoded default: legacy callers relying on the old
# "always required" behaviour must still supply them via `priors` (or,
# now, `fixed`).  The other entries default exactly to the pre-existing
# hardcoded behaviour (beta_cm=0, A=1, McClintock+18 alpha_mr/beta_mr) so
# old callers that never heard of these new knobs are unaffected.
#
# Naming: _cm = mass-concentration relation, _mr = mass-richness relation
# (both parametrised by an alpha/beta slope pair, plus gamma_cm for the
# magnitude-gap slope) -- distinct namespaces so both relations can be fit
# simultaneously without name collisions.
_COMBINED_PARAMS = (
    'c0', 'alpha_cm', 'beta_cm', 'gamma_cm', 'log_M_norm',
    'alpha_mr', 'beta_mr', 'A',
)
_COMBINED_PARAM_SOFT_DEFAULTS = {
    'beta_cm': 0.0, 'A': 1.0, 'alpha_mr': 1.356, 'beta_mr': -0.30,
}


def run_sampler_combined(
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    priors=None,
    fixed=None,
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
    # Fixed per-cluster miscentering / BCG arrays (legacy pathway; prefer
    # 'f_mis'/'sigma_off' in priors/fixed — see below)
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
    two_halo=False,
    bin_nodes=0,
    n_live=400,
    pool=None,
    verbose=True,
    filepath=None,
    resume=True,
    timeout=600,
):
    """
    Run the nautilus nested sampler for the combined M–c + magnitude-gap model.

    Every parameter the model exposes — c0, alpha_cm, beta_cm, gamma_cm,
    log_M_norm (mass-concentration relation + mass normalisation),
    alpha_mr, beta_mr (mass-richness relation), f_mis, sigma_off
    (population-level miscentering) — is either SAMPLED (its name is a key
    of ``priors``, mapping to a frozen ``scipy.stats`` distribution) or
    FIXED (its name is a key of ``fixed``, mapping to a precise float). A
    parameter named in neither raises immediately, except for the soft-
    defaulted legacy set {beta_cm, A, alpha_mr, beta_mr} (see below) and
    f_mis/sigma_off, which fall back to the FMISS/SIGMA_OFF array kwargs
    when unaddressed — there is no other silent default.

    ``gamma_cm`` and ``c0`` have no default at all: they must always appear
    in ``priors`` or ``fixed``. ``alpha_cm``/``log_M_norm`` are likewise
    always required — this reproduces the historical "always required"
    behaviour of this function, with ``fixed`` now offering an explicit
    alternative to sampling for any of them. ``beta_cm``/``A``/
    ``alpha_mr``/``beta_mr`` silently default to their pre-existing
    hardcoded values (0, 1, 1.356, -0.30) when addressed by neither dict,
    purely for backward compatibility with callers written before those
    parameters existed as fittable knobs.

    ``f_mis``/``sigma_off`` (population-level miscentering: one value
    applied to every cluster, not fit per cluster) are special-cased: when
    'f_mis' is a key of ``priors``, both it and 'sigma_off' must be (they
    are sampled together); when 'f_mis' is a key of ``fixed``, both it and
    'sigma_off' must be; otherwise the legacy per-cluster FMISS/SIGMA_OFF
    array kwargs are used as before (default None -> centered, 0.1 Mpc).

    Parameter order in ``sampler.posterior()``'s point array — and hence in
    ``sampler.param_order``, the definitive list this function attaches to
    the returned sampler — is ``_COMBINED_PARAMS`` order (c0, alpha_cm,
    beta_cm, gamma_cm, log_M_norm, alpha_mr, beta_mr, A), filtered to
    whichever of those are actually sampled, with f_mis/sigma_off appended
    last if sampled.

    The recommended way to declare priors is via the ``priors``/``fixed``
    dicts.  For backward compatibility, when ``priors`` is ``None`` the
    function builds an equivalent dict from the individual ``*_prior_*``
    legacy kwargs (c0, alpha_cm, beta_cm, gamma_cm, log_M_norm all sampled).

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
        ``nautilus.Prior.add_parameter``) for every SAMPLED parameter.
        Example::

            from scipy.stats import norm, truncnorm, uniform
            priors = {
                'c0':         norm(loc=5.0, scale=5.0),
                'alpha_cm':   norm(loc=0.0, scale=0.5),
                'beta_cm':    norm(loc=0.0, scale=1.0),
                'gamma_cm':   truncnorm(a=-2, b=2, loc=0.0, scale=0.5),
                'log_M_norm': norm(loc=0.0, scale=0.1),
                'alpha_mr':   norm(loc=1.356, scale=0.1),     # optional
                'f_mis':      uniform(loc=0.0, scale=1.0),    # optional
                'sigma_off':  uniform(loc=0.05, scale=1.45),  # required with f_mis
            }
        Default ``None`` — fall back to the individual legacy kwargs below.
    fixed : dict[str, float] or None
        Mapping from parameter name to a precise fixed value, for every
        FIXED (not sampled) parameter you want to control explicitly
        instead of relying on the soft defaults above.  Example::

            fixed = {'beta_cm': -0.4018, 'alpha_mr': 1.2, 'beta_mr': -0.25}
    c0_prior_loc, c0_prior_scale : float
        Gaussian prior on c0 (legacy, used only when ``priors`` is None).
    alpha_prior_loc, alpha_prior_scale : float
        Gaussian prior on alpha_cm (legacy).
    beta_prior_loc, beta_prior_scale : float
        Gaussian prior on beta_cm (legacy).  Ignored when fit_beta=False.
    gamma_range : (float, float)
        Truncation bounds for the TruncNormal prior on gamma_cm (legacy).
    gamma_prior_sigma : float
        Sigma of the TruncNormal prior on gamma_cm, centred at 0 (legacy).
    log_M_norm_sigma : float
        Gaussian prior sigma on log_M_norm [dex] (legacy).
    M_pivot_log, z_ref : float
        Fixed model hyperparameters (not sampled).
    delta_m_ref : float
        Reference magnitude gap [mag].  Default 1.0.
    fit_beta : bool
        Legacy shim, only consulted when 'beta_cm' is in neither ``priors``
        nor ``fixed``: if False, fixes beta_cm=0 (matches the pre-
        ``fixed``-dict behaviour); if True (default), beta_cm falls
        through to its soft default of 0 anyway, so this is a no-op in the
        new pathway.
    FMISS : array_like (N,) or None
        Legacy per-cluster miscentered-signal fraction, used only when
        f_mis is addressed by neither ``priors`` nor ``fixed``.
        Default None → zeros (centered NFW).
    SIGMA_OFF : array_like (N,) or None
        Legacy per-cluster Rayleigh scale of the miscentering offset
        [Mpc], used only when sigma_off is addressed by neither ``priors``
        nor ``fixed``.  Default None → 0.1 Mpc (irrelevant when FMISS=0).
    BCG_B_MASS : array_like (N,) or None
        Per-cluster baryonic BCG mass [M_sun].  Always fixed (not
        fittable).  Default None → 0.
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
        ``sampler.param_order`` is the list of sampled parameter names, in
        the exact order of the columns returned by ``sampler.posterior()``.
        ``sampler.converged`` is True unless nautilus stopped on the
        timeout/n_like_max rather than actual convergence.
    """
    from nautilus import Prior, Sampler

    if priors is None:
        lo, hi = gamma_range
        a = (lo - 0.0) / gamma_prior_sigma
        b = (hi - 0.0) / gamma_prior_sigma
        priors = {
            'c0':         sp_norm(loc=c0_prior_loc,    scale=c0_prior_scale),
            'alpha_cm':   sp_norm(loc=alpha_prior_loc, scale=alpha_prior_scale),
            'beta_cm':    sp_norm(loc=beta_prior_loc,  scale=beta_prior_scale),
            'gamma_cm':   sp_truncnorm(a=a, b=b, loc=0.0, scale=gamma_prior_sigma),
            'log_M_norm': sp_norm(loc=0.0, scale=log_M_norm_sigma),
        }
    priors = dict(priors)
    fixed  = dict(fixed) if fixed is not None else {}

    # fit_beta legacy shim: only matters if beta_cm is otherwise unaddressed.
    if not fit_beta and 'beta_cm' not in priors and 'beta_cm' not in fixed:
        fixed['beta_cm'] = 0.0

    always_required = ('c0', 'alpha_cm', 'gamma_cm', 'log_M_norm')
    missing = [n for n in always_required
              if n not in priors and n not in fixed]
    if missing:
        raise KeyError(
            f"the following parameters need either a prior (to sample) or "
            f"a fixed value: {sorted(missing)}"
        )
    for name, default in _COMBINED_PARAM_SOFT_DEFAULTS.items():
        if name not in priors and name not in fixed:
            fixed[name] = default

    sample_mis_prior = ('f_mis' in priors) or ('sigma_off' in priors)
    if sample_mis_prior and not ('f_mis' in priors and 'sigma_off' in priors):
        raise KeyError("sampling miscentering requires both 'f_mis' and "
                       "'sigma_off' priors")
    fixed_mis = ('f_mis' in fixed) or ('sigma_off' in fixed)
    if fixed_mis and not ('f_mis' in fixed and 'sigma_off' in fixed):
        raise KeyError("fixing miscentering requires both 'f_mis' and "
                       "'sigma_off' values")

    nprior = Prior()
    for name in _COMBINED_PARAMS:
        if name in priors:
            nprior.add_parameter(name, dist=priors[name])
    if sample_mis_prior:
        nprior.add_parameter('f_mis',     dist=priors['f_mis'])
        nprior.add_parameter('sigma_off', dist=priors['sigma_off'])
    param_order = list(nprior.keys)

    n_cl = len(data_vectors)

    def _val(p, name):
        return p[name] if name in p else fixed[name]

    def _loglike(p):
        if 'f_mis' in p:
            fmiss_arr = np.full(n_cl, p['f_mis'])
            soff_arr  = np.full(n_cl, p['sigma_off'])
        elif fixed_mis:
            fmiss_arr = np.full(n_cl, fixed['f_mis'])
            soff_arr  = np.full(n_cl, fixed['sigma_off'])
        else:
            fmiss_arr = FMISS
            soff_arr  = SIGMA_OFF
        return log_likelihood_combined(
            _val(p, 'c0'), _val(p, 'alpha_cm'), _val(p, 'gamma_cm'),
            cluster_table, data_vectors, cov_matrices, radii,
            beta_cm=_val(p, 'beta_cm'),
            M_pivot_log=M_pivot_log,
            z_ref=z_ref,
            delta_m_ref=delta_m_ref,
            log_M_norm=_val(p, 'log_M_norm'),
            alpha_mr=_val(p, 'alpha_mr'),
            beta_mr=_val(p, 'beta_mr'),
            A=_val(p, 'A'),
            FMISS=fmiss_arr,
            SIGMA_OFF=soff_arr,
            BCG_B_MASS=BCG_B_MASS,
            two_halo=two_halo,
            bin_nodes=bin_nodes,
        )

    sampler = Sampler(
        nprior, _loglike,
        n_live=n_live, pool=pool,
        filepath=filepath, resume=resume,
    )
    converged = sampler.run(verbose=verbose, timeout=timeout)
    # nautilus returns False when stopped by timeout/n_like_max rather
    # than convergence.  Exposed as an attribute to keep the return
    # signature unchanged for existing callers.
    sampler.converged   = bool(converged)
    sampler.param_order = param_order
    # Information gain from prior to posterior, in nats, via the nested-
    # sampling identity D_KL(posterior || prior) = <log L>_posterior - log Z
    # (same convention used in notebooks/wazp_mass_richness_by_delta_m.ipynb).
    # Always computed and attached so every caller gets it for free.
    try:
        _points, _log_w, _log_l = sampler.posterior()
        _w = np.exp(_log_w - _log_w.max())
        _w = _w / _w.sum()
        _finite = np.isfinite(_log_l)
        if _finite.any():
            sampler.kl_nats = float(
                np.sum(_w[_finite] * _log_l[_finite]) - sampler.log_z
            )
            # Per-parameter marginal KL(posterior_i || prior_i): weighted
            # 1D KDE of the marginal posterior evaluated against the
            # analytic prior pdf, Monte-Carlo averaged over the posterior
            # samples themselves.  Priors are independent across sampled
            # parameters, so joint_kl - sum(marginal_kl) is the mutual
            # information the data induced between them — i.e. how much
            # of the total information gain went into correlations /
            # degeneracies (e.g. mass-concentration) rather than pinning
            # down individual parameters.
            _wf = _w[_finite]
            _wf = _wf / _wf.sum()
            marginal_kl = {}
            for i, name in enumerate(param_order):
                try:
                    x = _points[_finite, i]
                    # np.log(kde.pdf(x)), not kde.logpdf(x): the latter is
                    # broken for weighted KDEs on some scipy builds (returns
                    # all-NaN despite a well-conditioned covariance/factor).
                    # Clip the density away from exactly 0 first: an
                    # isolated low-weight tail sample can otherwise
                    # underflow to log(0) = -inf, which combined with its
                    # (near-)zero weight in the sum is a 0 * -inf -> NaN.
                    pdf_q = gaussian_kde(x, weights=_wf).pdf(x)
                    log_q = np.log(np.clip(pdf_q, 1e-300, None))
                    log_p = priors[name].logpdf(x)
                    marginal_kl[name] = float(np.sum(_wf * (log_q - log_p)))
                except Exception:
                    marginal_kl[name] = float("nan")
            sampler.marginal_kl_nats = marginal_kl
            _finite_marg = [v for v in marginal_kl.values() if np.isfinite(v)]
            sampler.kl_residual_nats = (
                float(sampler.kl_nats - sum(_finite_marg))
                if np.isfinite(sampler.kl_nats)
                and len(_finite_marg) == len(marginal_kl)
                else float("nan")
            )
        else:
            sampler.kl_nats = float("nan")
            sampler.marginal_kl_nats = {n: float("nan") for n in param_order}
            sampler.kl_residual_nats = float("nan")
    except Exception:
        sampler.kl_nats = float("nan")
        sampler.marginal_kl_nats = {n: float("nan") for n in param_order}
        sampler.kl_residual_nats = float("nan")
    return sampler


# ---------------------------------------------------------------------------
# Diemer & Joyce (2019) M-c relation: likelihood and sampler
# ---------------------------------------------------------------------------

def log_likelihood_diemer(
    gamma,
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    delta_m_ref=_MC_DM_REF,
    log_M_norm=0.0,
    alpha_mr=1.356,
    beta_mr=-0.30,
    A=1.0,
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
    two_halo=False,
    bin_nodes=0,
):
    """
    Log-likelihood for the Diemer & Joyce (2019) M-c relation model
    (``forward_model_diemer``) -- gamma is the magnitude-gap slope applied
    on top of the theoretical c200m(M, z) prediction, the only
    concentration-related free parameter in this model (no
    c0/alpha_cm/beta_cm).

    Returns -inf for unphysical A, for NaN/inf in the forward model
    (including a cluster whose mass falls outside the Diemer19
    interpolation grid), or if the forward model raises. Never returns NaN.
    """
    if A <= 0:
        return -np.inf
    try:
        model = forward_model_diemer(
            gamma, cluster_table, radii,
            delta_m_ref=delta_m_ref,
            log_M_norm=log_M_norm, alpha_mr=alpha_mr, beta_mr=beta_mr, A=A,
            FMISS=FMISS, SIGMA_OFF=SIGMA_OFF, BCG_B_MASS=BCG_B_MASS,
            two_halo=two_halo,
            bin_nodes=bin_nodes,
        )
        if not np.all(np.isfinite(model)):
            return -np.inf
        residuals = data_vectors - model
        ll = cholog_gaussian(residuals, cov_matrices)
        return ll if np.isfinite(ll) else -np.inf
    except Exception:
        return -np.inf


# Every parameter this model exposes. 'gamma' has no default (always
# required, sampled or fixed); the rest soft-default to the same values as
# _COMBINED_PARAM_SOFT_DEFAULTS for consistency with forward_model_combined.
_DIEMER_PARAMS = ('gamma', 'log_M_norm', 'alpha_mr', 'beta_mr', 'A')
_DIEMER_PARAM_SOFT_DEFAULTS = {
    'log_M_norm': 0.0, 'A': 1.0, 'alpha_mr': 1.356, 'beta_mr': -0.30,
}


def run_sampler_diemer(
    cluster_table,
    data_vectors,
    cov_matrices,
    radii,
    priors=None,
    fixed=None,
    delta_m_ref=_MC_DM_REF,
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
    two_halo=False,
    bin_nodes=0,
    n_live=400,
    pool=None,
    verbose=True,
    filepath=None,
    resume=True,
    timeout=600,
):
    """
    Run the nautilus nested sampler for the Diemer & Joyce (2019) M-c
    relation model. Mirrors ``run_sampler_combined``'s priors/fixed
    interface exactly (see its docstring), restricted to this model's
    smaller parameter set: 'gamma' (always required, sampled or fixed --
    the magnitude-gap slope on top of the Diemer19 baseline, same role as
    ``gamma_cm`` in the combined model), 'log_M_norm'/'alpha_mr'/
    'beta_mr'/'A' (soft-defaulted), and optional 'f_mis'/'sigma_off'
    (population-level miscentering, sampled or fixed together, else
    falling back to the FMISS/SIGMA_OFF array kwargs).

    Parameter order in ``sampler.posterior()``/``sampler.param_order`` is
    ``_DIEMER_PARAMS`` order (gamma, log_M_norm, alpha_mr, beta_mr, A),
    filtered to whichever are actually sampled, with f_mis/sigma_off
    appended last if sampled.

    Returns
    -------
    sampler : nautilus.Sampler
        Completed sampler, with the same ``.converged``/``.param_order``/
        ``.kl_nats``/``.marginal_kl_nats``/``.kl_residual_nats``
        attributes ``run_sampler_combined`` attaches.
    """
    from nautilus import Prior, Sampler

    priors = dict(priors) if priors is not None else {}
    fixed  = dict(fixed) if fixed is not None else {}

    if 'gamma' not in priors and 'gamma' not in fixed:
        raise KeyError(
            "'gamma' needs either a prior (to sample) or a fixed value"
        )
    for name, default in _DIEMER_PARAM_SOFT_DEFAULTS.items():
        if name not in priors and name not in fixed:
            fixed[name] = default

    sample_mis_prior = ('f_mis' in priors) or ('sigma_off' in priors)
    if sample_mis_prior and not ('f_mis' in priors and 'sigma_off' in priors):
        raise KeyError("sampling miscentering requires both 'f_mis' and "
                       "'sigma_off' priors")
    fixed_mis = ('f_mis' in fixed) or ('sigma_off' in fixed)
    if fixed_mis and not ('f_mis' in fixed and 'sigma_off' in fixed):
        raise KeyError("fixing miscentering requires both 'f_mis' and "
                       "'sigma_off' values")

    nprior = Prior()
    for name in _DIEMER_PARAMS:
        if name in priors:
            nprior.add_parameter(name, dist=priors[name])
    if sample_mis_prior:
        nprior.add_parameter('f_mis',     dist=priors['f_mis'])
        nprior.add_parameter('sigma_off', dist=priors['sigma_off'])
    param_order = list(nprior.keys)

    n_cl = len(data_vectors)

    def _val(p, name):
        return p[name] if name in p else fixed[name]

    def _loglike(p):
        if 'f_mis' in p:
            fmiss_arr = np.full(n_cl, p['f_mis'])
            soff_arr  = np.full(n_cl, p['sigma_off'])
        elif fixed_mis:
            fmiss_arr = np.full(n_cl, fixed['f_mis'])
            soff_arr  = np.full(n_cl, fixed['sigma_off'])
        else:
            fmiss_arr = FMISS
            soff_arr  = SIGMA_OFF
        return log_likelihood_diemer(
            _val(p, 'gamma'),
            cluster_table, data_vectors, cov_matrices, radii,
            delta_m_ref=delta_m_ref,
            log_M_norm=_val(p, 'log_M_norm'),
            alpha_mr=_val(p, 'alpha_mr'),
            beta_mr=_val(p, 'beta_mr'),
            A=_val(p, 'A'),
            FMISS=fmiss_arr,
            SIGMA_OFF=soff_arr,
            BCG_B_MASS=BCG_B_MASS,
            two_halo=two_halo,
            bin_nodes=bin_nodes,
        )

    sampler = Sampler(
        nprior, _loglike,
        n_live=n_live, pool=pool,
        filepath=filepath, resume=resume,
    )
    converged = sampler.run(verbose=verbose, timeout=timeout)
    sampler.converged   = bool(converged)
    sampler.param_order = param_order
    try:
        _points, _log_w, _log_l = sampler.posterior()
        _w = np.exp(_log_w - _log_w.max())
        _w = _w / _w.sum()
        _finite = np.isfinite(_log_l)
        if _finite.any():
            sampler.kl_nats = float(
                np.sum(_w[_finite] * _log_l[_finite]) - sampler.log_z
            )
            _wf = _w[_finite]
            _wf = _wf / _wf.sum()
            marginal_kl = {}
            for i, name in enumerate(param_order):
                try:
                    x = _points[_finite, i]
                    pdf_q = gaussian_kde(x, weights=_wf).pdf(x)
                    log_q = np.log(np.clip(pdf_q, 1e-300, None))
                    log_p = priors[name].logpdf(x)
                    marginal_kl[name] = float(np.sum(_wf * (log_q - log_p)))
                except Exception:
                    marginal_kl[name] = float("nan")
            sampler.marginal_kl_nats = marginal_kl
            _finite_marg = [v for v in marginal_kl.values() if np.isfinite(v)]
            sampler.kl_residual_nats = (
                float(sampler.kl_nats - sum(_finite_marg))
                if np.isfinite(sampler.kl_nats)
                and len(_finite_marg) == len(marginal_kl)
                else float("nan")
            )
        else:
            sampler.kl_nats = float("nan")
            sampler.marginal_kl_nats = {n: float("nan") for n in param_order}
            sampler.kl_residual_nats = float("nan")
    except Exception:
        sampler.kl_nats = float("nan")
        sampler.marginal_kl_nats = {n: float("nan") for n in param_order}
        sampler.kl_residual_nats = float("nan")
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
        beta_cm=beta,
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
    converged = sampler.run(verbose=verbose, timeout=timeout)
    # nautilus returns False when stopped by timeout/n_like_max rather
    # than convergence.  Exposed as an attribute to keep the return
    # signature unchanged for existing callers.
    sampler.converged = bool(converged)
    return sampler