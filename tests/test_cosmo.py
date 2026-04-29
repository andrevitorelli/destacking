import numpy as np
import pytest
from astropy.cosmology import Planck18
from astropy import units as u
from destacking.cosmo import critical_surface_density


cosmo = Planck18


def test_critical_surface_density_positive():
    sigma = critical_surface_density(0.3, 1.0, cosmo)
    assert sigma.value > 0


def test_critical_surface_density_has_units():
    sigma = critical_surface_density(0.3, 1.0, cosmo)
    # Check it carries astropy units and can be converted to M☉/pc²
    sigma_converted = sigma.to(u.M_sun / u.pc**2)
    assert sigma_converted.value > 0


def test_critical_surface_density_physical_range():
    # Σ_crit for z_l=0.3, z_s=1 is ~2000–4000 M☉/pc² with Planck18
    sigma = critical_surface_density(0.3, 1.0, cosmo).to(u.M_sun / u.pc**2)
    assert 500 < sigma.value < 10000


def test_critical_surface_density_decreases_with_source_redshift():
    # Higher z_s → greater lensing efficiency → lower Σ_crit
    s1 = critical_surface_density(0.3, 0.8, cosmo)
    s2 = critical_surface_density(0.3, 2.0, cosmo)
    assert s2 < s1


def test_critical_surface_density_depends_on_lens_redshift():
    s1 = critical_surface_density(0.2, 1.0, cosmo)
    s2 = critical_surface_density(0.5, 1.0, cosmo)
    assert s1 != s2


def test_critical_surface_density_diverges_as_lens_approaches_source():
    # As z_l → z_s, D_ls → 0 and Σ_crit → ∞
    s_close = critical_surface_density(0.95, 1.0, cosmo)
    s_far   = critical_surface_density(0.3,  1.0, cosmo)
    assert s_close > s_far
