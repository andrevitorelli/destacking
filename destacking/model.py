import numpy as np
import xlensing

from destacking.relations import mass_lambda_McClintock18, c_DuttonMaccio
from destacking.simulation import _pivot_from_delta_m

_MC_M_PIVOT_LOG = 14.0   # log10(M_sun) — fixed pivot mass for M-c parametrisation
_MC_Z_REF       = 0.35   # fixed reference redshift for c0 normalisation
_MC_DM_REF      = 1.0    # reference magnitude gap [mag]


# ---------------------------------------------------------------------------
# Diemer & Joyce (2019) concentration-mass relation, in the M200m/C200m
# definition this package uses throughout (xlensing.model.NFW_Delta_Sigma's
# M200m/C200m are both spherical-overdensity-wrt-mean-matter-density
# quantities -- see r_vir_m, which integrates cosmo.rhoM, not cosmo.rhocz).
#
# colossus's 'diemer19' model is natively defined ONLY for the 200c mass
# definition (colossus.halo.concentration.models['diemer19'].mdefs ==
# ['200c']); requesting mdef='200m' still works (colossus converts M200m to
# M200c internally, evaluates the native model, converts the resulting
# concentration back), and this is the physically correct call to make --
# our M really is an M200m value, and treating it as if it were M200c
# outright would systematically bias the predicted concentration (M200m
# exceeds M200c for the same halo, typically by 15-30%). But that
# conversion path breaks for vector-valued z (a colossus bug: internally it
# hits an "ambiguous truth value of an array" error and silently returns
# -1 for every element) even though it works fine for a scalar z with a
# vector M. Since this module needs concentrations for many clusters at
# many different redshifts on every likelihood call, we precompute
# log10(c200m) on a (z, log10 M) grid via colossus at fixed scalar z (safe,
# vectorized over M) ONCE, lazily, and interpolate at runtime -- fast and
# free of the array-z bug.
_DIEMER19_LOGM_RANGE = (12.5, 16.0)   # log10(M200m/Msun), comfortably wider
_DIEMER19_Z_RANGE    = (0.02, 0.85)   # than any WaZP-like sample's range
_DIEMER19_N_LOGM      = 71
_DIEMER19_N_Z          = 42
_diemer19_interp = None   # lazily-built scipy.interpolate.RegularGridInterpolator


def _build_diemer19_interp():
    import warnings
    from colossus.cosmology import cosmology as _colossus_cosmology
    from colossus.halo import concentration as _colossus_concentration
    from scipy.interpolate import RegularGridInterpolator

    # Match xlensing.cosmo's Planck18 cosmology (see xlensing/cosmo.py).
    # colossus.cosmology.getCurrent() RAISES (not returns None) when no
    # cosmology has been set yet in this process/worker -- there is no
    # query-only way to check, so probe with try/except instead.
    try:
        _colossus_cosmology.getCurrent()
    except Exception:
        _colossus_cosmology.setCosmology('planck18')

    logM_grid = np.linspace(*_DIEMER19_LOGM_RANGE, _DIEMER19_N_LOGM)
    z_grid    = np.linspace(*_DIEMER19_Z_RANGE, _DIEMER19_N_Z)
    M_grid    = 10.0 ** logM_grid

    log10c_table = np.empty((len(z_grid), len(logM_grid)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, z in enumerate(z_grid):
            c = _colossus_concentration.concentration(
                M_grid, '200m', z=float(z), model='diemer19')
            # colossus returns -1 (not NaN) for points outside the model's
            # calibrated range; propagate as NaN so a fit that wanders there
            # gets -inf from the likelihood, not a silently wrong c=-1.
            c = np.where(np.isfinite(c) & (c > 0), c, np.nan)
            log10c_table[i] = np.log10(c)

    return RegularGridInterpolator(
        (z_grid, logM_grid), log10c_table,
        bounds_error=False, fill_value=None,   # extrapolate at the edges
    )


def diemer19_c200m(M200m, z):
    """log10(c200m) from the Diemer & Joyce (2019) relation, interpolated
    from a precomputed (z, log10 M200m) grid -- see the module-level note
    above for why this isn't just a direct colossus call. Vectorized over
    both M200m and z (they may be different lengths-worth of scalars, but
    must broadcast together -- typically both (N,) per-cluster arrays).

    Returns
    -------
    ndarray
        c200m (linear, not log) at each (M200m, z), NaN outside the grid's
        calibrated mass/redshift range.
    """
    global _diemer19_interp
    if _diemer19_interp is None:
        _diemer19_interp = _build_diemer19_interp()
    M200m = np.asarray(M200m, dtype=float)
    z     = np.asarray(z, dtype=float)
    logM  = np.log10(M200m)
    pts   = np.column_stack([np.broadcast_to(z, logM.shape),
                             np.broadcast_to(logM, logM.shape)])
    return 10.0 ** _diemer19_interp(pts)


# ---------------------------------------------------------------------------
# Internal: NFW profile computation
# ---------------------------------------------------------------------------

# Default model evaluation: at the logarithmic bin centre (bin_nodes=0).
# bin_nodes > 0 instead returns the area-weighted average over each radial bin
# with that many Gauss-Legendre nodes (accurate to ~3e-6 for NFW with 4 nodes),
# which is the exact counterpart of the annulus-averaged measurement.
#
# History: made the default (4 nodes) on 2026-09-28, reverted to 0 on 2026-09-29.
# Paired LSST-like fits of IDENTICAL simulated data with the two models showed
# that averaging does not change the gamma_cm result: LSSTsph -> LSSTsphBA
# 0.1894 -> 0.1906 (+0.1 sigma) and LSSTclean -> LSSTcleanBA 0.1842 -> 0.1884
# (+0.7 sigma, part of it unseeded bootstrap resampling); c0, alpha_cm, beta_cm
# move by <= 1 sigma.  The bin-centre error (1-3.5 %, nearly independent of mass
# and concentration) is absorbed almost entirely by the mass normalisation:
# log_M_norm -0.017/-0.018 (centre) vs +0.001 (average), i.e. a ~0.018 dex
# shift that cancels in the differential gamma_cm.  Averaging costs ~4x per
# likelihood call, which pushed LSST fits against their per-fit time budget, so
# the cheaper bin-centre model is used for the campaign.  DES-like bins are
# 0.24 dex wide vs LSST's 0.20 dex; the centre error scales ~ width^2, i.e. ~1.4x.
BIN_AVERAGE_NODES_DEFAULT = 0


def bin_average_nodes(radii, n_nodes):
    """Radii and weights for the area-weighted average of a profile over each radial bin.

    ``radii`` are the bin centres of a logarithmic grid (the repository's
    convention: bin edges halfway between neighbouring centres in log R, the
    outermost edges half a bin beyond the first/last centre).  For sources
    spread uniformly over the sky, the expected stacked/binned measurement in a
    bin is the area-weighted mean,

        <f>_bin = int f(R) 2 pi R dR / [pi (R2^2 - R1^2)]
                = mean of f over t = R^2 in [R1^2, R2^2],

    evaluated here with ``n_nodes`` Gauss-Legendre points in t.  Returns
    (node_radii (M, K), weights (K,)) with sum(weights) == 1.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.size < 2:
        raise ValueError("bin averaging needs at least two radial bins to define edges")
    lr = np.log10(radii)
    d = np.diff(lr)
    edges = np.concatenate([[lr[0] - d[0] / 2], 0.5 * (lr[:-1] + lr[1:]), [lr[-1] + d[-1] / 2]])
    r1, r2 = 10.0 ** edges[:-1], 10.0 ** edges[1:]
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    t = r1[:, None] ** 2 + (r2[:, None] ** 2 - r1[:, None] ** 2) * 0.5 * (x[None, :] + 1.0)
    return np.sqrt(t), 0.5 * w


def _nfw_profiles(z, M200m, C200m, radii, FMISS=None, SIGMA_OFF=None,
                  BCG_B_MASS=None, two_halo=False, bin_nodes=BIN_AVERAGE_NODES_DEFAULT):
    """One-halo NFW ΔΣ profiles for N clusters at M radii (no 2-halo term).

    The returned signal is ``xlensing.model.NFW_Delta_Sigma['Signal']
    − ['Two-halo term']``, i.e. the mixture
    ``(1 − FMISS)·NFW_centered + FMISS·NFW_miscentered + BCG``.
    With the default FMISS=zeros, BCG=zeros this reduces to the centered NFW
    signal exactly as before.

    Parameters
    ----------
    z, M200m, C200m : array_like (N,)
        Per-cluster redshift, M200m [M_sun], and c200m.
    radii : array_like (M,)
        Projected radii [Mpc].
    FMISS : array_like (N,) or None
        Per-cluster miscentered-signal fraction (0 = always centered,
        1 = always miscentered, intermediate = mixture).  Default ``None`` → zeros.
    SIGMA_OFF : array_like (N,) or None
        Per-cluster Rayleigh scale of the miscentering offset distribution
        [Mpc].  Irrelevant where FMISS=0.  Default ``None`` → 0.1 Mpc.
    BCG_B_MASS : array_like (N,) or None
        Per-cluster baryonic BCG mass [M_sun].  Default ``None`` → 0.

    Returns
    -------
    ndarray (N, M)
        ΔΣ in M_sun/pc² (one-halo: centered + miscentered NFW + BCG, no 2-halo).
    """
    N = len(z)
    if FMISS is None:
        FMISS = np.zeros(N)
    if SIGMA_OFF is None:
        SIGMA_OFF = np.ones(N) * 0.1
    if BCG_B_MASS is None:
        BCG_B_MASS = np.zeros(N)

    # bin_nodes > 0: area-weighted average over each radial bin (added
    # 2026-09-28) instead of evaluating at the bin centre -- see bin_average_nodes.
    radii = np.asarray(radii, dtype=float)
    if radii.size < 2:
        bin_nodes = 0          # a single radius defines no bin: evaluate at it
    if bin_nodes:
        nodes, wts = bin_average_nodes(radii, bin_nodes)
        eval_radii = nodes.ravel()
    else:
        eval_radii = radii
    out = xlensing.model.NFW_Delta_Sigma(
        M200m      = M200m,
        C200m      = C200m,
        Z          = z,
        FMISS      = FMISS,
        SIGMA_OFF  = SIGMA_OFF,
        BCG_B_MASS = BCG_B_MASS,
        radii      = eval_radii,
    )
    # two_halo=True keeps xlensing's two-halo term (Johnston+07 linear bias
    # x matter correlation) -- used for REAL data, which contain it.  The
    # simulations shear sources with isolated haloes only, so fits to them
    # must keep the default one-halo model (added 2026-09-26).
    sig = out['Signal'] if two_halo else out['Signal'] - out['Two-halo term']
    if bin_nodes:
        sig = (sig.reshape(len(z), len(radii), -1) * wts).sum(axis=-1)
    return sig


def _nfw_components(z, M200m, C200m, radii, SIGMA_OFF=None, bin_nodes=BIN_AVERAGE_NODES_DEFAULT):
    """Centered and miscentered one-halo NFW ΔΣ, returned separately.

    Same physics as ``_nfw_profiles`` (xlensing's NFW + Rayleigh
    miscentering table, no BCG) but (i) skips the two-halo term, which
    ``_nfw_profiles`` computes only to subtract it again (~half of its
    cost), and (ii) keeps the two components apart so a caller can mix
    them at the likelihood level, per cluster, instead of mixing the
    profiles.  ``(1-f)*centered + f*miscentered`` reproduces
    ``_nfw_profiles(..., FMISS=f, SIGMA_OFF=...)`` with BCG_B_MASS=0.

    Returns
    -------
    centered, miscentered : ndarray (N, M) each, in M_sun/pc².
    """
    xm = xlensing.model
    z = np.asarray(z, dtype=float)
    M200m = np.asarray(M200m, dtype=float)
    C200m = np.asarray(C200m, dtype=float)
    if SIGMA_OFF is None:
        SIGMA_OFF = np.full(len(z), 0.1)
    rs   = xm.r_vir_m(z, M200m) / C200m
    fact = 2 * rs * xm.NFW_delta_c(C200m) * xm.cosmo.rhoM(z) / 1e12
    radii = np.asarray(radii, dtype=float)
    if radii.size < 2:
        bin_nodes = 0                              # a single radius defines no bin
    if bin_nodes:                                  # area-weighted bin average
        nodes, wts = bin_average_nodes(radii, bin_nodes)
        eval_radii = nodes.ravel()
    else:
        eval_radii = radii
    X    = eval_radii[np.newaxis, :] / rs[:, np.newaxis]
    centered    = fact[:, None] * (xm.SigBar(X) - xm.SigX(X))
    miscentered = fact[:, None] * xm.Delta_Sigma_NFW_off_x(X, np.asarray(SIGMA_OFF) / rs)
    if bin_nodes:
        shape = (len(z), len(radii), -1)
        centered = (centered.reshape(shape) * wts).sum(-1)
        miscentered = (miscentered.reshape(shape) * wts).sum(-1)
    return centered, miscentered


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
      1. Mass from richness:  M = 10^(mass_lambda_McClintock18(Lambda, z) + log_M_norm)
      2. Concentration:       c = c_DuttonMaccio(z, M, c=c_pivot)
                              where c_pivot = c_pivot_0 * (Δm / delta_m_ref)^gamma
      3. Profile:             ΔΣ = NFW_Delta_Sigma(M, c, z, ...)

    gamma is d(log10 c) / d(log10(Δm/Δm_ref)).  Clusters with Δm = NaN receive
    no gamma correction (c_pivot = c_pivot_0).

    Parameters
    ----------
    gamma : float
        Magnitude-gap concentration slope [dex/dex].
    cluster_table : astropy.table.Table
        Must have columns 'z', 'Lambda', 'delta_m'.
    radii : array (M,)
        Projected radii in Mpc.
    c_pivot_0 : float
        Pivot parameter at delta_m = delta_m_ref.
    delta_m_ref : float
        Reference magnitude gap [mag].
    log_M_norm : float
        Log10 mass normalisation offset [dex].  Default 0.

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
# Forward model — mass–concentration relation recovery
# ---------------------------------------------------------------------------

def forward_model_mc(
    c0,
    alpha,
    cluster_table,
    radii,
    beta=0.0,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    log_M_norm=0.0,
):
    """
    Predict NFW ΔΣ profiles using a parametric power-law M–c relation.

    The forward model is:
      1. Mass from richness:  M = 10^(mass_lambda_McClintock18(Lambda, z) + log_M_norm)
      2. Concentration:       log10(c) = log10(c0)
                                        + alpha * (log10(M) − M_pivot_log)
                                        + beta  * log10((1+z) / (1+z_ref))
      3. Profile:             ΔΣ = NFW_Delta_Sigma(M, c, z, …)

    Parameters
    ----------
    c0 : float
        Concentration at M = 10^M_pivot_log M_sun and z = z_ref.
    alpha : float
        Logarithmic slope  d log10(c) / d log10(M).
        Dutton–Maccio value at cluster masses: ≈ −0.09.
    cluster_table : astropy.table.Table
        Must have columns 'z' and 'Lambda'.  'delta_m' is not used.
    radii : array (M,)
        Projected radii in Mpc.
    beta : float
        Redshift evolution:  d log10(c) / d log10(1+z).
        Dutton–Maccio value: ≈ −0.43.  Default 0 (evolution ignored).
    M_pivot_log : float
        log10(M_pivot / M_sun).  Default 14.0.
    z_ref : float
        Reference redshift for the c0 normalisation.  Default 0.35.
    log_M_norm : float
        Log10 mass normalisation offset [dex]; marginalised over to absorb
        mass–richness calibration uncertainty.  Default 0.

    Returns
    -------
    ndarray (N, M)
        Predicted ΔΣ profiles in M_sun/pc².
    """
    z      = np.asarray(cluster_table['z'])
    Lambda = np.asarray(cluster_table['Lambda'])

    logM  = mass_lambda_McClintock18(Lambda, z) + log_M_norm
    M200m = 10.0 ** logM

    logC  = (np.log10(c0)
             + alpha * (logM - M_pivot_log)
             + beta  * np.log10((1.0 + z) / (1.0 + z_ref)))
    C200m = 10.0 ** logC

    return _nfw_profiles(z, M200m, C200m, radii)


# ---------------------------------------------------------------------------
# Forward model — combined M–c relation + magnitude-gap (gamma)
# ---------------------------------------------------------------------------

def forward_model_combined(
    c0,
    alpha_cm,
    gamma_cm,
    cluster_table,
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
    bin_nodes=BIN_AVERAGE_NODES_DEFAULT,
):
    """
    Predict NFW ΔΣ profiles using a combined M–c + magnitude-gap model
    with an optional global amplitude correction.

    Two scaling relations combine: the mass-richness relation (subscript
    _mr, McClintock+18 form) sets M from Lambda and z; the mass-
    concentration relation (subscript _cm) sets c from M, z, and the
    magnitude gap Δm:

        log10(M) = log10(M0) + alpha_mr * (log10(Lambda) − log10(Lambda0))
                              + beta_mr  * (log10(1+z) − log10(1+z0))
                              + log_M_norm

        log10(c) = log10(c0)
                 + alpha_cm * (log10(M) − M_pivot_log)
                 + beta_cm  * log10((1+z) / (1+z_ref))
                 + gamma_cm * log10(Δm / delta_m_ref)

    gamma_cm is d(log10 c) / d(log10(Δm/Δm_ref)).  Clusters without
    detected satellites (delta_m = NaN) receive no gamma_cm correction
    (log10 ratio = 0).

    The signal returned is

        A · [(1 − FMISS)·NFW_centered + FMISS·NFW_miscentered + BCG]

    from ``xlensing.model.NFW_Delta_Sigma`` (with the 2-halo term subtracted)
    and a global multiplicative amplitude factor ``A``.  A is intended to
    absorb the population-mean orientation/triaxiality bias measured from
    simulations (the spherical-vs-triaxial deficit), and would typically be
    given a tight Gaussian prior centred on the simulation-calibrated value.

    Parameters
    ----------
    c0 : float
        Concentration normalisation at M = 10^M_pivot_log, z = z_ref, Δm = delta_m_ref.
    alpha_cm : float
        Mass-concentration relation: logarithmic mass slope
        d log10(c) / d log10(M).
    gamma_cm : float
        Mass-concentration relation: magnitude-gap slope
        d log10(c) / d log10(Δm/Δm_ref).
    cluster_table : astropy.table.Table
        Must have columns 'z', 'Lambda', and 'delta_m'.
    radii : array (M,)
        Projected radii in Mpc.
    beta_cm : float
        Mass-concentration relation: redshift-evolution slope
        d log10(c) / d log10(1+z).  Default 0.
    M_pivot_log : float
        log10(M_pivot / M_sun).  Default 14.0.
    z_ref : float
        Reference redshift for c0.  Default 0.35.
    delta_m_ref : float
        Reference magnitude gap [mag].  Default 1.0.
    log_M_norm : float
        Log10 mass normalisation offset [dex].  Default 0.
    alpha_mr : float
        Mass-richness relation: richness slope F_lambda in
        mass_lambda_McClintock18.  Default 1.356 (McClintock+18).  Free
        this when the mass-richness relation is not assumed known.
    beta_mr : float
        Mass-richness relation: redshift-evolution slope G_z in
        mass_lambda_McClintock18.  Default -0.30 (McClintock+18).
    A : float
        Global multiplicative amplitude correction.  Default 1.0 (no
        correction) reproduces the previous forward model.
    FMISS, SIGMA_OFF, BCG_B_MASS : array_like (N,) or None
        Optional per-cluster miscentering and BCG arrays passed through to
        ``_nfw_profiles``.  Default ``None`` → centered, no BCG.

    Returns
    -------
    ndarray (N, M)
        Predicted ΔΣ profiles in M_sun/pc².
    """
    z      = np.asarray(cluster_table['z'])
    Lambda = np.asarray(cluster_table['Lambda'])
    dm     = np.asarray(cluster_table['delta_m'], dtype=float)

    logM  = mass_lambda_McClintock18(Lambda, z, Flambda=alpha_mr,
                                     Gz=beta_mr) + log_M_norm
    M200m = 10.0 ** logM

    dm_eff   = np.where(np.isfinite(dm), dm, delta_m_ref)
    dm_eff   = np.maximum(dm_eff, 1e-6)   # avoid log10(0)
    log_dm_r = np.log10(dm_eff / delta_m_ref)

    logC = (np.log10(c0)
            + alpha_cm * (logM - M_pivot_log)
            + beta_cm  * np.log10((1.0 + z) / (1.0 + z_ref))
            + gamma_cm * log_dm_r)
    C200m = 10.0 ** logC

    return A * _nfw_profiles(
        z, M200m, C200m, radii,
        FMISS=FMISS, SIGMA_OFF=SIGMA_OFF, BCG_B_MASS=BCG_B_MASS,
        two_halo=two_halo,
        bin_nodes=bin_nodes,
    )


# ---------------------------------------------------------------------------
# Forward model — Diemer & Joyce (2019) M-c relation, single normalisation
# ---------------------------------------------------------------------------

def forward_model_diemer(
    gamma,
    cluster_table,
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
    bin_nodes=BIN_AVERAGE_NODES_DEFAULT,
):
    """
    Predict NFW ΔΣ profiles using the Diemer & Joyce (2019) mass-
    concentration relation as the baseline (mass and redshift dependence
    both taken directly from the theoretical model, not fit), with a
    single free magnitude-gap slope ``gamma`` as the only concentration-
    related parameter:

        log10(M) = log10(M0) + alpha_mr * (log10(Lambda) − log10(Lambda0))
                              + beta_mr  * (log10(1+z) − log10(1+z0))
                              + log_M_norm

        log10(c) = log10(c_Diemer19(M200m, z))
                 + gamma * log10(Δm / delta_m_ref)

    Unlike ``forward_model_combined``, there is no separate mass slope or
    redshift slope for concentration -- Diemer19 already encodes the mass
    and redshift dependence (including its non-power-law curvature, see
    project report
    reports/2026-07-26_boost_factors_variance_floor_alpha_cm.tex) -- but
    ``gamma`` plays the same role ``gamma_cm`` does in
    ``forward_model_combined``: the magnitude-gap slope,
    d log10(c) / d log10(Δm/Δm_ref), layered on top of the Diemer19
    baseline instead of a free power-law one. ``gamma=0`` reproduces the
    literature prediction exactly, independent of Δm. Clusters without a
    detected satellite (delta_m = NaN) receive no gamma correction
    (log10 ratio = 0), same convention as ``forward_model_combined``.

    Parameters
    ----------
    gamma : float
        Mass-concentration relation: magnitude-gap slope
        d log10(c) / d log10(Δm/Δm_ref), applied on top of the Diemer19
        c200m(M200m, z) baseline.
    cluster_table : astropy.table.Table
        Must have columns 'z', 'Lambda', and 'delta_m'.
    radii : array (M,)
        Projected radii in Mpc.
    delta_m_ref : float
        Reference magnitude gap [mag]. Default 1.0.
    log_M_norm, alpha_mr, beta_mr : float
        Mass-richness relation parameters, same meaning as in
        ``forward_model_combined``.
    A : float
        Global multiplicative amplitude correction. Default 1.0.
    FMISS, SIGMA_OFF, BCG_B_MASS : array_like (N,) or None
        Optional per-cluster miscentering and BCG arrays passed through to
        ``_nfw_profiles``. Default ``None`` → centered, no BCG.

    Returns
    -------
    ndarray (N, M)
        Predicted ΔΣ profiles in M_sun/pc².
    """
    z      = np.asarray(cluster_table['z'])
    Lambda = np.asarray(cluster_table['Lambda'])
    dm     = np.asarray(cluster_table['delta_m'], dtype=float)

    logM  = mass_lambda_McClintock18(Lambda, z, Flambda=alpha_mr,
                                     Gz=beta_mr) + log_M_norm
    M200m = 10.0 ** logM

    dm_eff   = np.where(np.isfinite(dm), dm, delta_m_ref)
    dm_eff   = np.maximum(dm_eff, 1e-6)   # avoid log10(0)
    log_dm_r = np.log10(dm_eff / delta_m_ref)

    logC  = np.log10(diemer19_c200m(M200m, z)) + gamma * log_dm_r
    C200m = 10.0 ** logC

    return A * _nfw_profiles(
        z, M200m, C200m, radii,
        FMISS=FMISS, SIGMA_OFF=SIGMA_OFF, BCG_B_MASS=BCG_B_MASS,
        two_halo=two_halo,
        bin_nodes=bin_nodes,
    )
