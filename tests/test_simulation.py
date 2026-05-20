import numpy as np
import pytest
from destacking.simulation import (
    generate_truth_catalog,
    make_test_data,
    _sample_powerlaw_lf,
    _pivot_from_delta_m,
)

RADII = np.logspace(-0.6, 0.6, 8)


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


# ---------------------------------------------------------------------------
# make_test_data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def truth_for_data():
    return generate_truth_catalog(30, seed=42, delta_m_gamma=0.5)


def test_make_test_data_shapes(truth_for_data):
    cluster_obs, data, covs = make_test_data(truth_for_data, RADII, seed=0)
    N, M = len(truth_for_data), len(RADII)
    assert data.shape   == (N, M)
    assert covs.shape   == (N, M, M)
    assert len(cluster_obs) == N


def test_make_test_data_columns(truth_for_data):
    cluster_obs, _, _ = make_test_data(truth_for_data, RADII, seed=0)
    for col in ('z', 'Lambda', 'delta_m'):
        assert col in cluster_obs.colnames


def test_make_test_data_lambda_positive(truth_for_data):
    cluster_obs, _, _ = make_test_data(truth_for_data, RADII, seed=0)
    assert np.all(cluster_obs['Lambda'] >= 1.0)


def test_make_test_data_covs_diagonal_positive(truth_for_data):
    _, _, covs = make_test_data(truth_for_data, RADII, seed=0)
    for i in range(len(truth_for_data)):
        diag = np.diag(covs[i])
        assert np.all(diag > 0)


def test_make_test_data_covs_off_diagonal_zero(truth_for_data):
    _, _, covs = make_test_data(truth_for_data, RADII, seed=0)
    for i in range(len(truth_for_data)):
        off = covs[i].copy()
        np.fill_diagonal(off, 0.0)
        assert np.all(off == 0.0)


def test_make_test_data_data_positive(truth_for_data):
    _, data, _ = make_test_data(truth_for_data, RADII, fractional_noise=0.05, seed=0)
    assert np.all(data > 0)


def test_make_test_data_seed_reproducible(truth_for_data):
    _, d1, _ = make_test_data(truth_for_data, RADII, seed=99)
    _, d2, _ = make_test_data(truth_for_data, RADII, seed=99)
    np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# DES Y3 cluster n(z): des_y3_cluster_pdf + sample_des_y3_cluster_z
# ---------------------------------------------------------------------------

from destacking.simulation import (
    des_y3_cluster_pdf,
    sample_des_y3_cluster_z,
    des_y3_source_pdf,
    sample_des_y3_source_z,
)


def test_des_y3_cluster_pdf_zero_outside_range():
    p = des_y3_cluster_pdf(np.array([0.0, 0.1, 0.19, 0.66, 1.0]),
                           z_min=0.2, z_max=0.65)
    assert np.all(p == 0.0)


def test_des_y3_cluster_pdf_positive_inside_range():
    z = np.linspace(0.2, 0.65, 50)
    p = des_y3_cluster_pdf(z, z_min=0.2, z_max=0.65)
    assert np.all(p > 0.0)


def test_des_y3_cluster_pdf_normalised():
    z = np.linspace(0.2, 0.65, 5000)
    p = des_y3_cluster_pdf(z, z_min=0.2, z_max=0.65)
    integral = np.trapezoid(p, z) if hasattr(np, "trapezoid") else np.trapz(p, z)
    np.testing.assert_allclose(integral, 1.0, rtol=1e-3)


def test_sample_des_y3_cluster_z_in_range():
    rng = np.random.default_rng(0)
    z = sample_des_y3_cluster_z(2000, rng, z_min=0.2, z_max=0.65)
    assert z.shape == (2000,)
    assert np.all(z >= 0.2) and np.all(z <= 0.65)


def test_sample_des_y3_cluster_z_volume_weighted():
    # P(z) ∝ z**2 puts more weight at higher z; the sample mean should
    # exceed the midpoint (0.425) of the range.
    rng = np.random.default_rng(1)
    z = sample_des_y3_cluster_z(5000, rng, z_min=0.2, z_max=0.65)
    assert z.mean() > 0.425


def test_sample_des_y3_cluster_z_reproducible():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    z1 = sample_des_y3_cluster_z(500, rng1)
    z2 = sample_des_y3_cluster_z(500, rng2)
    np.testing.assert_array_equal(z1, z2)


# ---------------------------------------------------------------------------
# DES Y3 source n(z): des_y3_source_pdf + sample_des_y3_source_z
# ---------------------------------------------------------------------------

def test_des_y3_source_pdf_zero_at_zero():
    # Smail-Schechter is zero at z=0 because (z/z0)**alpha vanishes.
    p = des_y3_source_pdf(np.array([0.0]))
    assert p[0] == 0.0


def test_des_y3_source_pdf_positive_for_positive_z():
    z = np.linspace(0.05, 2.5, 50)
    p = des_y3_source_pdf(z, alpha=2.0, beta=1.5, z0=0.4)
    assert np.all(p > 0.0)


def test_des_y3_source_pdf_normalised():
    z = np.linspace(0.001, 5.0, 5000)
    p = des_y3_source_pdf(z, alpha=2.0, beta=1.5, z0=0.4)
    integral = np.trapezoid(p, z) if hasattr(np, "trapezoid") else np.trapz(p, z)
    # The PDF is normalised on z in [1e-3, 5]; allow some integration error.
    np.testing.assert_allclose(integral, 1.0, rtol=5e-3)


def test_sample_des_y3_source_z_in_range():
    rng = np.random.default_rng(0)
    z = sample_des_y3_source_z(2000, rng, z_min=0.0, z_max=2.5,
                                 alpha=2.0, beta=1.5, z0=0.4)
    assert z.shape == (2000,)
    assert np.all(z >= 0.0) and np.all(z <= 2.5)


def test_sample_des_y3_source_z_mean_near_expected():
    # Smail with alpha=2, beta=1.5, z0=0.4 → sample mean ≈ 0.65–0.7.
    rng = np.random.default_rng(0)
    z = sample_des_y3_source_z(5000, rng, alpha=2.0, beta=1.5, z0=0.4)
    assert 0.5 < z.mean() < 0.85


def test_sample_des_y3_source_z_reproducible():
    rng1 = np.random.default_rng(11)
    rng2 = np.random.default_rng(11)
    z1 = sample_des_y3_source_z(500, rng1)
    z2 = sample_des_y3_source_z(500, rng2)
    np.testing.assert_array_equal(z1, z2)


# ---------------------------------------------------------------------------
# simcluster — wraps xlensing.sims.make_simple_random_cat +
# xlensing.sims.apply_NFW_shear_region.  Run a minimal smoke test that
# verifies it returns an astropy Table with the expected columns and the
# right number of rows.
# ---------------------------------------------------------------------------

def test_simcluster_smoke():
    from destacking.simulation import simcluster
    import xlensing  # noqa: F401  -- only run if xlensing is importable

    # A 14-Msun cluster at z=0.3, c=5, ra/dec/z = 0,0,0.3
    cluster = [0.0, 0.0, 0.3, 1e14, 5.0]
    cat = simcluster(
        gal_density=5,    # low density to keep the call fast
        total_radius=2.0,
        z_range=(0.4, 1.5),
        cluster=cluster,
        shape_noise=0.0,
        seed=42,
    )
    assert set(["ra", "dec", "z", "e1", "e2",
                "R11", "R12", "R21", "R22", "w"]).issubset(cat.colnames)
    assert len(cat) > 0
