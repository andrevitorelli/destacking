import numpy as np
import xlensing

from destacking.relations import mass_lambda_McClintock18, c_DuttonMaccio
from destacking.simulation import _pivot_from_delta_m

_MC_M_PIVOT_LOG = 14.0   # log10(M_sun) — fixed pivot mass for M-c parametrisation
_MC_Z_REF       = 0.35   # fixed reference redshift for c0 normalisation
_MC_DM_REF      = 1.0    # reference magnitude gap [mag]


# ---------------------------------------------------------------------------
# Internal: NFW profile computation
# ---------------------------------------------------------------------------

def _nfw_profiles(z, M200m, C200m, radii, FMISS=None, SIGMA_OFF=None,
                  BCG_B_MASS=None):
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

    out = xlensing.model.NFW_Delta_Sigma(
        M200m      = M200m,
        C200m      = C200m,
        Z          = z,
        FMISS      = FMISS,
        SIGMA_OFF  = SIGMA_OFF,
        BCG_B_MASS = BCG_B_MASS,
        radii      = radii,
    )
    return out['Signal'] - out['Two-halo term']


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
    alpha,
    gamma,
    cluster_table,
    radii,
    beta=0.0,
    M_pivot_log=_MC_M_PIVOT_LOG,
    z_ref=_MC_Z_REF,
    delta_m_ref=_MC_DM_REF,
    log_M_norm=0.0,
    FMISS=None,
    SIGMA_OFF=None,
    BCG_B_MASS=None,
):
    """
    Predict NFW ΔΣ profiles using a combined M–c + magnitude-gap model.

    The full parametric concentration is:

        log10(c) = log10(c0)
                 + alpha * (log10(M) − M_pivot_log)
                 + beta  * log10((1+z) / (1+z_ref))
                 + gamma * log10(Δm / delta_m_ref)

    gamma is d(log10 c) / d(log10(Δm/Δm_ref)).  Clusters without detected
    satellites (delta_m = NaN) receive no gamma correction (log10 ratio = 0).

    The signal returned is the one-halo mixture
    ``(1 − FMISS)·NFW_centered + FMISS·NFW_miscentered + BCG`` from
    ``xlensing.model.NFW_Delta_Sigma`` with the 2-halo term subtracted.

    Parameters
    ----------
    c0 : float
        Concentration normalisation at M = 10^M_pivot_log, z = z_ref, Δm = delta_m_ref.
    alpha : float
        Logarithmic slope  d log10(c) / d log10(M).
    gamma : float
        Magnitude-gap slope  d log10(c) / d log10(Δm/Δm_ref).
    cluster_table : astropy.table.Table
        Must have columns 'z', 'Lambda', and 'delta_m'.
    radii : array (M,)
        Projected radii in Mpc.
    beta : float
        Redshift-evolution slope  d log10(c) / d log10(1+z).  Default 0.
    M_pivot_log : float
        log10(M_pivot / M_sun).  Default 14.0.
    z_ref : float
        Reference redshift for c0.  Default 0.35.
    delta_m_ref : float
        Reference magnitude gap [mag].  Default 1.0.
    log_M_norm : float
        Log10 mass normalisation offset [dex].  Default 0.
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

    logM  = mass_lambda_McClintock18(Lambda, z) + log_M_norm
    M200m = 10.0 ** logM

    dm_eff   = np.where(np.isfinite(dm), dm, delta_m_ref)
    dm_eff   = np.maximum(dm_eff, 1e-6)   # avoid log10(0)
    log_dm_r = np.log10(dm_eff / delta_m_ref)

    logC = (np.log10(c0)
            + alpha * (logM - M_pivot_log)
            + beta  * np.log10((1.0 + z) / (1.0 + z_ref))
            + gamma * log_dm_r)
    C200m = 10.0 ** logC

    return _nfw_profiles(z, M200m, C200m, radii,
                        FMISS=FMISS, SIGMA_OFF=SIGMA_OFF, BCG_B_MASS=BCG_B_MASS)
