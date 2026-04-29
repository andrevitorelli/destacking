import numpy as np
from scipy.special import erf

HOD_samples={
    'SDSS18.0':{
        'logM_min': 11.35,
        'sigma_logM': 0.25,
        'logM_0': 12.20,
        'logM_1': 12.40,
        'alpha': 0.83,
    },
    'SDSS21.0': {
        'logM_min': 12.79,
        'sigma_logM': 0.39,
        'logM_0': 11.92,
        'logM_1': 13.94,
        'alpha': 1.15,
    },
    'SDSS22.0':{
        'logM_min': 14.22,
        'sigma_logM': 0.77,
        'logM_0': 14.00,
        'logM_1': 14.69,
        'alpha': 0.87,
    }
    
}


def sample_zheng07_batch(M_halos, HOD_sample):
    """
    Parametrized Halo Occupation distribution from Zheng+07
    Vectorized sampling for an array of halo masses.
    Returns arrays of (N_cen, N_sat) for each halo.
    """
    logM = np.log10(M_halos)
    logM_min= HOD_sample['logM_min']
    sigma_logM= HOD_sample['sigma_logM']
    logM_0= HOD_sample['logM_0']
    logM_1= HOD_sample['logM_1']
    alpha= HOD_sample['alpha']
    
    
    # Mean central occupation
    mean_cen = 0.5 * (1 + erf((logM - logM_min) / sigma_logM))
    
    # Mean satellite occupation (vectorized with masking)
    M_0 = 10**logM_0
    M_1 = 10**logM_1
    mean_sat = np.zeros_like(M_halos)
    mask = M_halos > M_0
    mean_sat[mask] = ((M_halos[mask] - M_0) / M_1)**alpha
    
    # Sample centrals (Bernoulli)
    N_cen = np.random.binomial(n=1, p=mean_cen)
    
    # Sample satellites (Poisson, only where N_cen=1)
    N_sat = np.zeros_like(M_halos, dtype=int)
    sat_mask = N_cen == 1
    N_sat[sat_mask] = np.random.poisson(lam=mean_sat[sat_mask])
    
    return N_cen, N_sat

#### mass concentration
def c_DuttonMaccio(z, m,c=1., h=1.):
  """Concentration from c(M) relation in Dutton & Maccio (2014).
  
  Code taken from Jes Ford on github.
  
  Parameters
  ----------
  z : float or array_like
      Redshift(s) of halos.
  m : float or array_like
      Mass(es) of halos (m200 definition), in units of solar masses.
  h : float, optional
      Hubble parameter. Default is from Planck13.
  Returns
  ----------
  ndarray
      Concentration values (c200) for halos.
  References
  ----------
  Calculation from Planck-based results of simulations presented in:
  A.A. Dutton & A.V. Maccio, "Cold dark matter haloes in the Planck era:
  evolution of structural parameters for Einasto and NFW profiles,"
  Monthly Notices of the Royal Astronomical Society, Volume 441, Issue 4,
  p.3359-3374, 2014.
  """


  a = 0.52 + 0.385 * np.exp(-0.617 * (z**1.21))  # EQ 10
  b = -0.101 + 0.026 * z                         # EQ 11

  logc200 = a + b * np.log10(m * h / (c*1e12))  # EQ 7 modified

  concentration = 10.**logc200
  return concentration

#####mass richness
def mass_lambda_McClintock18(Lambda,z):
  """
  Mass-richness relation of DES clusters (McClintock+18)
  """
  #pivots
  log10Lambda0 = np.log10(40)
  log10z0 = np.log10(0.35)
  log10M0 = 14.489
  #scalings
  Flambda = 1.356
  Gz = -0.30
    
  mlog10M200 = log10M0 + Flambda*(np.log10(Lambda)-log10Lambda0) + Gz*(np.log10(z) - log10z0)
    
  return mlog10M200


####BCG Luminosity 
def lum_BCG_zheng07(M_BCG):
    """
    BCG Luminosity as a function of mass
    """
    L0_lumbcg= 2.8e9 #SDSS
    Mc_lumbcg=3.7e9
    k_lumbcg = 0.0255
    a_lumbcg = 29.78
    b_lumbcg =29.5
    Mratio = M_BCG/Mc_lumbcg
    return L0_lumbcg*Mratio**a_lumbcg*(1+Mratio**(b_lumbcg*k_lumbcg))**(-1/k_lumbcg)