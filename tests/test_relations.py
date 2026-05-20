import numpy as np
import pytest
from destacking.relations import (
    c_DuttonMaccio,
    mass_lambda_McClintock18,
    sample_zheng07_batch,
    lum_BCG_zheng07,
    HOD_samples,
)


# ---------------------------------------------------------------------------
# c_DuttonMaccio
# ---------------------------------------------------------------------------

def test_c_DuttonMaccio_positive():
    assert c_DuttonMaccio(0.3, 1e14) > 0


def test_c_DuttonMaccio_physical_range():
    # Concentrations for galaxy clusters are typically 2–10
    c = c_DuttonMaccio(0.3, 1e14)
    assert 1.0 < c < 20.0


def test_c_DuttonMaccio_decreases_with_mass():
    # Higher mass → lower concentration (well-known trend)
    c_low  = c_DuttonMaccio(0.3, 1e13)
    c_high = c_DuttonMaccio(0.3, 1e15)
    assert c_low > c_high


def test_c_DuttonMaccio_decreases_with_redshift():
    # Higher z → less concentrated (halos form later, less time to concentrate)
    c_z0 = c_DuttonMaccio(0.0, 1e14)
    c_z1 = c_DuttonMaccio(1.0, 1e14)
    assert c_z0 > c_z1


def test_c_DuttonMaccio_vectorized():
    z = np.array([0.0, 0.3, 0.5, 1.0])
    m = np.array([1e13, 1e14, 1e14, 1e15])
    result = c_DuttonMaccio(z, m)
    assert result.shape == (4,)
    assert np.all(result > 0)


def test_c_DuttonMaccio_pivot_parameter():
    # c parameter rescales the pivot mass: larger c → lower effective mass → higher concentration
    c1 = c_DuttonMaccio(0.3, 1e14, c=1.0)
    c2 = c_DuttonMaccio(0.3, 1e14, c=2.0)
    assert c1 != c2


# ---------------------------------------------------------------------------
# mass_lambda_McClintock18
# ---------------------------------------------------------------------------

def test_mass_lambda_McClintock18_pivot():
    # At Lambda=40, z=0.35 → exactly log10M0 = 14.489
    result = mass_lambda_McClintock18(40, 0.35)
    np.testing.assert_allclose(result, 14.489, rtol=1e-6)


def test_mass_lambda_McClintock18_increases_with_richness():
    m1 = mass_lambda_McClintock18(20,  0.3)
    m2 = mass_lambda_McClintock18(40,  0.3)
    m3 = mass_lambda_McClintock18(100, 0.3)
    assert m1 < m2 < m3


def test_mass_lambda_McClintock18_decreases_with_redshift():
    # Gz = -0.30 → higher z → lower mass at fixed richness
    m_lo = mass_lambda_McClintock18(40, 0.2)
    m_hi = mass_lambda_McClintock18(40, 0.6)
    assert m_lo > m_hi


def test_mass_lambda_McClintock18_physical_range():
    # log10(M) should be in [12, 16] for realistic richnesses
    m = mass_lambda_McClintock18(40, 0.35)
    assert 12 < m < 16


def test_mass_lambda_McClintock18_vectorized():
    Lambda = np.array([10, 30, 60, 100])
    z = np.full(4, 0.3)
    result = mass_lambda_McClintock18(Lambda, z)
    assert result.shape == (4,)
    assert np.all(np.diff(result) > 0)


# ---------------------------------------------------------------------------
# sample_zheng07_batch
# ---------------------------------------------------------------------------

@pytest.fixture
def hod():
    return HOD_samples['SDSS21.0']


def test_sample_zheng07_batch_shapes(hod):
    rng = np.random.default_rng(0)
    M = rng.uniform(1e12, 1e15, 100)
    N_cen, N_sat = sample_zheng07_batch(M, hod)
    assert N_cen.shape == (100,)
    assert N_sat.shape == (100,)


def test_sample_zheng07_batch_N_cen_binary(hod):
    M = np.full(200, 1e14)
    N_cen, _ = sample_zheng07_batch(M, hod)
    assert set(np.unique(N_cen)).issubset({0, 1})


def test_sample_zheng07_batch_N_sat_nonnegative(hod):
    M = np.full(200, 1e14)
    _, N_sat = sample_zheng07_batch(M, hod)
    assert np.all(N_sat >= 0)


def test_sample_zheng07_batch_sat_only_where_cen(hod):
    # Satellites are only assigned when N_cen = 1
    M = np.full(500, 1e14)
    N_cen, N_sat = sample_zheng07_batch(M, hod)
    assert np.all(N_sat[N_cen == 0] == 0)


def test_sample_zheng07_batch_high_mass_has_cen(hod):
    # Very massive halos should nearly always host a central
    rng = np.random.default_rng(42)
    M = np.full(500, 1e16)
    N_cen, _ = sample_zheng07_batch(M, hod)
    assert N_cen.mean() > 0.99


def test_sample_zheng07_batch_low_mass_mostly_empty(hod):
    # Very low-mass halos should rarely host a central
    M = np.full(500, 1e9)
    N_cen, _ = sample_zheng07_batch(M, hod)
    assert N_cen.mean() < 0.1


# ---------------------------------------------------------------------------
# lum_BCG_zheng07
# ---------------------------------------------------------------------------

def test_lum_BCG_positive():
    assert lum_BCG_zheng07(1e11) > 0


def test_lum_BCG_increases_with_mass():
    L1 = lum_BCG_zheng07(1e10)
    L2 = lum_BCG_zheng07(1e12)
    assert L2 > L1


def test_lum_BCG_vectorized():
    M = np.array([1e10, 1e11, 1e12])
    L = lum_BCG_zheng07(M)
    assert L.shape == (3,)
    assert np.all(L > 0)


# ---------------------------------------------------------------------------
# sample_halo_masses_tinker08
# ---------------------------------------------------------------------------

def test_sample_halo_masses_tinker08_shape_and_range():
    from destacking.relations import sample_halo_masses_tinker08
    rng = np.random.default_rng(0)
    z = np.array([0.2, 0.3, 0.4, 0.5])
    M = sample_halo_masses_tinker08(z, logM_min=13.5, logM_max=15.0, rng=rng)
    assert M.shape == z.shape
    logM = np.log10(M)
    assert np.all(logM >= 13.5)
    assert np.all(logM <= 15.0)


def test_sample_halo_masses_tinker08_reproducible():
    from destacking.relations import sample_halo_masses_tinker08
    z = np.array([0.3, 0.5])
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    M1 = sample_halo_masses_tinker08(z, logM_min=13.5, logM_max=15.0, rng=rng1)
    M2 = sample_halo_masses_tinker08(z, logM_min=13.5, logM_max=15.0, rng=rng2)
    np.testing.assert_array_equal(M1, M2)


def test_sample_halo_masses_tinker08_scalar_z_accepted():
    # Function uses np.atleast_1d internally so a scalar z must work.
    from destacking.relations import sample_halo_masses_tinker08
    rng = np.random.default_rng(0)
    M = sample_halo_masses_tinker08(0.3, logM_min=13.5, logM_max=15.0, rng=rng)
    assert M.shape == (1,)
    assert 10**13.5 <= M[0] <= 10**15.0
