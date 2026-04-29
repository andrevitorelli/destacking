


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
  #pivots
  log10Lambda0 = np.log10(40)
  log10z0 = np.log10(0.35)
  log10M0 = 14.489
  #scalings
  Flambda = 1.356
  Gz = -0.30
    
  mlog10M200 = log10M0 + Flambda*(np.log10(Lambda)-log10Lambda0) + Gz*(np.log10(z) - log10z0)
    
  return mlog10M200