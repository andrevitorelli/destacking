import numpy as np
import xlensing

from destacking.relations import mass_lambda_McClintock18, c_DuttonMaccio
from destacking.simulation import _pivot_from_delta_m


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
