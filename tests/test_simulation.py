import numpy as np
import pytest
from destacking.simulation import (
    generate_truth_catalog,
    make_test_data,
    _sample_powerlaw_lf,
    _pivot_from_delta_m,
    SimulationConfig,
    build_truth_catalog,
    compute_triaxial_shapes,
    compute_miscentering,
)

RADII = np.logspace(-0.6, 0.6, 8)


def _bin_limits(radii):
    # Bin-edge convention from CLAUDE.md / scripts/gamma_inference.py:
    # half-bin-width offset in log-space around each radius.
    log_r = np.log10(radii)
    dlog  = np.diff(log_r)[0]
    edges = np.logspace(log_r[0] - dlog / 2, log_r[-1] + dlog / 2,
                        len(radii) + 1)
    return np.column_stack([edges[:-1], edges[1:]])


def _make_sim_config(n_clusters=3, n_workers=1, miscentering=True,
                     spherical=False):
    return SimulationConfig(
        n_clusters=n_clusters,
        z_cluster_range=(0.10, 0.60),
        logm_range=(13.5, 15.0),
        lambda_min=20.0,
        delta_m_ref=0.1,
        density=7.5,
        shape_noise=0.27,
        width_rad=0.006,
        z_src_range=(0.0, 2.5),
        z_src_alpha=2.0,
        z_src_beta=1.5,
        z_src_z0=0.4,
        spherical=spherical,
        miscentering=miscentering,
        f_cen=0.64,
        sigma_off_mpc=0.32,
        radii=RADII,
        bin_limits=_bin_limits(RADII),
        n_cov_bootstrap=20,
        n_workers=n_workers,
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


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

def test_simulation_config_fields_accessible():
    cfg = _make_sim_config()
    assert cfg.n_clusters == 3
    assert cfg.z_cluster_range == (0.10, 0.60)
    assert cfg.n_workers == 1


def test_simulation_config_n_bins_property():
    cfg = _make_sim_config()
    assert cfg.n_bins == len(RADII)


def test_simulation_config_n_workers_default():
    # n_workers has a default (=15) so it can be omitted.
    cfg = SimulationConfig(
        n_clusters=1, z_cluster_range=(0.1, 0.6), logm_range=(13.5, 15.0),
        lambda_min=20.0, delta_m_ref=0.1, density=7.5, shape_noise=0.27,
        width_rad=0.006, z_src_range=(0.0, 2.5), z_src_alpha=2.0,
        z_src_beta=1.5, z_src_z0=0.4, spherical=False, miscentering=True,
        f_cen=0.64, sigma_off_mpc=0.32, radii=RADII,
        bin_limits=_bin_limits(RADII), n_cov_bootstrap=20,
    )
    assert cfg.n_workers == 15


# ---------------------------------------------------------------------------
# build_truth_catalog
# ---------------------------------------------------------------------------

def test_build_truth_catalog_returns_n_clusters():
    cfg = _make_sim_config(n_clusters=25)
    truth, ols = build_truth_catalog(0.0, g_idx=0, cfg=cfg)
    assert len(truth) == 25


def test_build_truth_catalog_columns():
    cfg = _make_sim_config(n_clusters=10)
    truth, ols = build_truth_catalog(0.0, g_idx=0, cfg=cfg)
    for col in ("z", "M_halo", "concentration", "delta_m", "Lambda"):
        assert col in truth.colnames


def test_build_truth_catalog_selection_cuts_applied():
    cfg = _make_sim_config(n_clusters=10)
    truth, ols = build_truth_catalog(0.0, g_idx=1, cfg=cfg)
    assert np.all(np.asarray(truth["Lambda"]) > cfg.lambda_min)
    assert np.all(np.asarray(truth["delta_m"]) > 0.0)
    assert np.all(np.isfinite(np.asarray(truth["delta_m"])))
    z = np.asarray(truth["z"])
    assert np.all(z >= cfg.z_cluster_range[0])
    assert np.all(z < cfg.z_cluster_range[1])


def test_build_truth_catalog_ols_keys():
    cfg = _make_sim_config(n_clusters=15)
    truth, ols = build_truth_catalog(0.2, g_idx=2, cfg=cfg)
    for key in ("c0_true", "alpha_true", "beta_true", "gamma_true_ols"):
        assert key in ols
        assert np.isfinite(ols[key])


def test_build_truth_catalog_gamma_true_ols_recovers_injected_gamma():
    # The OLS fit is a linear (c0, alpha, beta, gamma) approximation of the
    # true (Dutton-Maccio-based) concentration relation, so it recovers
    # gamma_true closely but not exactly -- the mismatch is the linear
    # model's residual against the true nonlinear c-M-z-dm relation.
    cfg = _make_sim_config(n_clusters=200)
    truth, ols = build_truth_catalog(0.4, g_idx=3, cfg=cfg)
    assert ols["gamma_true_ols"] == pytest.approx(0.4, abs=0.01)


def test_build_truth_catalog_reproducible_with_same_g_idx():
    cfg = _make_sim_config(n_clusters=10)
    truth1, ols1 = build_truth_catalog(0.1, g_idx=4, cfg=cfg)
    truth2, ols2 = build_truth_catalog(0.1, g_idx=4, cfg=cfg)
    np.testing.assert_array_equal(
        np.asarray(truth1["M_halo"]), np.asarray(truth2["M_halo"])
    )
    assert ols1 == ols2


def test_build_truth_catalog_different_g_idx_differ():
    cfg = _make_sim_config(n_clusters=10)
    truth1, _ = build_truth_catalog(0.1, g_idx=5, cfg=cfg)
    truth2, _ = build_truth_catalog(0.1, g_idx=6, cfg=cfg)
    assert not np.allclose(
        np.asarray(truth1["M_halo"]), np.asarray(truth2["M_halo"])
    )


# ---------------------------------------------------------------------------
# compute_triaxial_shapes
# ---------------------------------------------------------------------------

def test_compute_triaxial_shapes_keys_and_length():
    cfg = _make_sim_config(n_clusters=10)
    truth, _ = build_truth_catalog(0.0, g_idx=7, cfg=cfg)
    shapes = compute_triaxial_shapes(truth, g_idx=7)
    for key in ("q_ba", "q_ca", "theta_los", "phi_los", "r_vir"):
        assert key in shapes
        assert len(shapes[key]) == len(truth)


def test_compute_triaxial_shapes_axis_ratios_in_range():
    cfg = _make_sim_config(n_clusters=20)
    truth, _ = build_truth_catalog(0.0, g_idx=8, cfg=cfg)
    shapes = compute_triaxial_shapes(truth, g_idx=8)
    assert np.all(shapes["q_ba"] >= 0.5) and np.all(shapes["q_ba"] <= 1.0)
    assert np.all(shapes["q_ca"] >= 0.3)
    assert np.all(shapes["q_ca"] <= shapes["q_ba"])


def test_compute_triaxial_shapes_reproducible():
    cfg = _make_sim_config(n_clusters=10)
    truth, _ = build_truth_catalog(0.0, g_idx=9, cfg=cfg)
    s1 = compute_triaxial_shapes(truth, g_idx=9)
    s2 = compute_triaxial_shapes(truth, g_idx=9)
    np.testing.assert_array_equal(s1["q_ba"], s2["q_ba"])
    np.testing.assert_array_equal(s1["theta_los"], s2["theta_los"])


# ---------------------------------------------------------------------------
# compute_miscentering
# ---------------------------------------------------------------------------

def test_compute_miscentering_disabled_returns_all_centred():
    cfg = _make_sim_config(n_clusters=15, miscentering=False)
    is_mis, off_mag_mpc, phi_mis = compute_miscentering(g_idx=0, cfg=cfg)
    assert is_mis.shape == (15,)
    assert not np.any(is_mis)
    np.testing.assert_array_equal(off_mag_mpc, 0.0)
    np.testing.assert_array_equal(phi_mis, 0.0)


def test_compute_miscentering_enabled_has_some_miscentered():
    cfg = _make_sim_config(n_clusters=500, miscentering=True)
    is_mis, off_mag_mpc, phi_mis = compute_miscentering(g_idx=1, cfg=cfg)
    # f_cen=0.64 -> ~36% miscentered; just check it's neither 0% nor 100%.
    frac = is_mis.mean()
    assert 0.1 < frac < 0.6


def test_compute_miscentering_offset_nonnegative():
    cfg = _make_sim_config(n_clusters=200, miscentering=True)
    is_mis, off_mag_mpc, phi_mis = compute_miscentering(g_idx=2, cfg=cfg)
    assert np.all(off_mag_mpc >= 0.0)
    # Centred clusters have exactly zero offset.
    np.testing.assert_array_equal(off_mag_mpc[~is_mis], 0.0)


def test_compute_miscentering_phi_in_full_circle():
    cfg = _make_sim_config(n_clusters=200, miscentering=True)
    _, _, phi_mis = compute_miscentering(g_idx=3, cfg=cfg)
    assert np.all(phi_mis >= 0.0) and np.all(phi_mis <= 2 * np.pi)


def test_compute_miscentering_reproducible():
    cfg = _make_sim_config(n_clusters=50, miscentering=True)
    r1 = compute_miscentering(g_idx=4, cfg=cfg)
    r2 = compute_miscentering(g_idx=4, cfg=cfg)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# simulate_lensing — full per-cluster simulation + measurement pipeline.
# Marked slow: launches a real (tiny) multiprocess pool and runs the full
# xlensing shear-injection + bootstrap_signal measurement per cluster.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_simulate_lensing_returns_expected_shapes():
    from destacking.simulation import simulate_lensing
    cfg = _make_sim_config(n_clusters=3, n_workers=1)
    truth, _ = build_truth_catalog(0.0, g_idx=100, cfg=cfg)
    shapes = compute_triaxial_shapes(truth, g_idx=100)
    mis = compute_miscentering(g_idx=100, cfg=cfg)

    data_vectors, cov_matrices, bgs = simulate_lensing(
        truth, shapes, mis, g_idx=100, cfg=cfg,
    )
    assert data_vectors.shape == (cfg.n_clusters, cfg.n_bins)
    assert cov_matrices.shape == (cfg.n_clusters, cfg.n_bins, cfg.n_bins)
    assert len(bgs) == cfg.n_clusters


@pytest.mark.slow
def test_simulate_lensing_produces_some_finite_clusters():
    from destacking.simulation import simulate_lensing
    cfg = _make_sim_config(n_clusters=3, n_workers=1)
    truth, _ = build_truth_catalog(0.0, g_idx=101, cfg=cfg)
    shapes = compute_triaxial_shapes(truth, g_idx=101)
    mis = compute_miscentering(g_idx=101, cfg=cfg)

    data_vectors, cov_matrices, bgs = simulate_lensing(
        truth, shapes, mis, g_idx=101, cfg=cfg,
    )
    # At least one cluster/bin should have succeeded (not every entry NaN);
    # a total dropout would indicate the pipeline is fundamentally broken.
    assert np.isfinite(data_vectors).any()


@pytest.mark.slow
def test_simulate_lensing_spherical_mode_runs():
    from destacking.simulation import simulate_lensing
    cfg = _make_sim_config(n_clusters=2, n_workers=1, spherical=True,
                           miscentering=False)
    truth, _ = build_truth_catalog(0.0, g_idx=102, cfg=cfg)
    shapes = compute_triaxial_shapes(truth, g_idx=102)
    mis = compute_miscentering(g_idx=102, cfg=cfg)

    data_vectors, cov_matrices, bgs = simulate_lensing(
        truth, shapes, mis, g_idx=102, cfg=cfg,
    )
    assert data_vectors.shape == (cfg.n_clusters, cfg.n_bins)
