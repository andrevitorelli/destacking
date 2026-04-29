import numpy as np
import pytest
from destacking.simulation import (
    generate_truth_catalog,
    _sample_powerlaw_lf,
    _pivot_from_delta_m,
)


# ---------------------------------------------------------------------------
# _sample_powerlaw_lf
# ---------------------------------------------------------------------------

def test_powerlaw_lf_bounds():
    rng = np.random.default_rng(0)
    L = _sample_powerlaw_lf(1000, -1.2, 1e8, 1e10, rng)
    assert np.all(L >= 1e8)
    assert np.all(L <= 1e10)


def test_powerlaw_lf_shape():
    rng = np.random.default_rng(1)
    L = _sample_powerlaw_lf(500, -1.2, 1e8, 1e10, rng)
    assert L.shape == (500,)


def test_powerlaw_lf_skewed_faint():
    # For alpha=-1.2 (faint-end slope), most galaxies should be near L_min
    rng = np.random.default_rng(2)
    L = _sample_powerlaw_lf(5000, -1.2, 1e8, 1e10, rng)
    assert np.median(L) < (1e8 + 1e10) / 2


def test_powerlaw_lf_positive():
    rng = np.random.default_rng(3)
    L = _sample_powerlaw_lf(200, -1.2, 1e8, 1e10, rng)
    assert np.all(L > 0)


# ---------------------------------------------------------------------------
# _pivot_from_delta_m
# ---------------------------------------------------------------------------

def test_pivot_at_ref_returns_c_pivot_0():
    dm = np.array([1.0, 1.0, 1.0])
    result = _pivot_from_delta_m(dm, c_pivot_0=2.0, delta_m_ref=1.0, gamma=0.5)
    np.testing.assert_allclose(result, 2.0)


def test_pivot_increases_with_delta_m():
    dm = np.array([0.5, 1.0, 2.0])
    result = _pivot_from_delta_m(dm, c_pivot_0=1.0, delta_m_ref=1.0, gamma=0.5)
    assert np.all(np.diff(result) > 0)


def test_pivot_nan_falls_back_to_c_pivot_0():
    dm = np.array([1.5, np.nan, 2.0])
    result = _pivot_from_delta_m(dm, c_pivot_0=3.0, delta_m_ref=1.0, gamma=0.5)
    assert result[1] == 3.0


def test_pivot_gamma_zero_gives_constant():
    dm = np.array([0.1, 1.0, 5.0])
    result = _pivot_from_delta_m(dm, c_pivot_0=1.5, delta_m_ref=1.0, gamma=0.0)
    np.testing.assert_allclose(result, 1.5)


# ---------------------------------------------------------------------------
# generate_truth_catalog — structure
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog():
    return generate_truth_catalog(50, seed=42)


def test_catalog_length(catalog):
    assert len(catalog) == 50


def test_catalog_columns(catalog):
    for col in ['z', 'M_halo', 'concentration', 'M_BCG', 'L_BCG',
                'L_brightest_sat', 'delta_m', 'N_sat']:
        assert col in catalog.colnames


def test_catalog_redshift_in_range(catalog):
    assert np.all(catalog['z'] >= 0.1)
    assert np.all(catalog['z'] <= 0.6)


def test_catalog_mass_in_range(catalog):
    logM = np.log10(catalog['M_halo'])
    assert np.all(logM >= 13.5)
    assert np.all(logM <= 15.0)


# ---------------------------------------------------------------------------
# generate_truth_catalog — physical content
# ---------------------------------------------------------------------------

def test_concentration_positive(catalog):
    assert np.all(catalog['concentration'] > 0)


def test_concentration_physical_range(catalog):
    assert np.all(catalog['concentration'] > 1)
    assert np.all(catalog['concentration'] < 20)


def test_L_BCG_positive(catalog):
    assert np.all(catalog['L_BCG'] > 0)


def test_M_BCG_positive(catalog):
    assert np.all(catalog['M_BCG'] > 0)


def test_N_sat_nonnegative(catalog):
    assert np.all(catalog['N_sat'] >= 0)


def test_delta_m_positive_where_finite(catalog):
    finite = ~np.isnan(catalog['delta_m'])
    assert np.all(catalog['delta_m'][finite] > 0)


def test_delta_m_nan_iff_no_satellites(catalog):
    no_sat = catalog['N_sat'] == 0
    has_sat = catalog['N_sat'] > 0
    assert np.all(np.isnan(catalog['delta_m'][no_sat]))
    # Clusters with satellites should mostly have finite delta_m
    # (could still be NaN if L_BCG <= sat_L_min, unlikely with default params)
    assert np.isfinite(catalog['delta_m'][has_sat]).sum() > 0


def test_brightest_sat_fainter_than_BCG(catalog):
    # By construction, the brightest satellite must be fainter than the BCG
    finite = ~np.isnan(catalog['L_brightest_sat'])
    assert np.all(catalog['L_brightest_sat'][finite] < catalog['L_BCG'][finite])


def test_delta_m_consistent_with_luminosities(catalog):
    # delta_m = 2.5 * log10(L_BCG / L_brightest_sat)
    finite = ~np.isnan(catalog['delta_m'])
    expected = 2.5 * np.log10(
        catalog['L_BCG'][finite] / catalog['L_brightest_sat'][finite]
    )
    np.testing.assert_allclose(catalog['delta_m'][finite], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# generate_truth_catalog — reproducibility and parameter effects
# ---------------------------------------------------------------------------

def test_seed_reproducible():
    cat1 = generate_truth_catalog(20, seed=7)
    cat2 = generate_truth_catalog(20, seed=7)
    np.testing.assert_array_equal(cat1['M_halo'], cat2['M_halo'])
    np.testing.assert_array_equal(cat1['concentration'], cat2['concentration'])


def test_different_seeds_differ():
    cat1 = generate_truth_catalog(20, seed=1)
    cat2 = generate_truth_catalog(20, seed=2)
    assert not np.allclose(cat1['M_halo'], cat2['M_halo'])


def test_larger_gamma_increases_concentration_spread():
    # Higher gamma amplifies the pivot effect → wider concentration distribution
    cat_low  = generate_truth_catalog(500, delta_m_gamma=0.0, seed=0)
    cat_high = generate_truth_catalog(500, delta_m_gamma=2.0, seed=0)
    assert cat_high['concentration'].std() > cat_low['concentration'].std()


def test_custom_z_range():
    cat = generate_truth_catalog(30, z_range=(0.3, 0.4), seed=0)
    assert np.all(cat['z'] >= 0.3)
    assert np.all(cat['z'] <= 0.4)


def test_custom_logM_range():
    cat = generate_truth_catalog(30, logM_range=(14.0, 14.5), seed=0)
    logM = np.log10(cat['M_halo'])
    assert np.all(logM >= 14.0)
    assert np.all(logM <= 14.5)
