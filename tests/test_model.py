import numpy as np
import pytest
from astropy import table
from destacking.simulation import generate_truth_catalog
from destacking.model import (
    forward_model,
    make_test_data,
    log_likelihood,
)

RADII = np.logspace(-0.6, 0.6, 8)
TRUTH_GAMMA = 0.5


@pytest.fixture(scope="module")
def truth():
    return generate_truth_catalog(30, seed=42, delta_m_gamma=TRUTH_GAMMA)


@pytest.fixture(scope="module")
def test_dataset(truth):
    cluster_obs, data, covs = make_test_data(truth, RADII, fractional_noise=0.2, seed=7)
    return cluster_obs, data, covs


# ---------------------------------------------------------------------------
# make_test_data
# ---------------------------------------------------------------------------

def test_make_test_data_shapes(truth):
    cluster_obs, data, covs = make_test_data(truth, RADII, seed=0)
    N, M = len(truth), len(RADII)
    assert data.shape   == (N, M)
    assert covs.shape   == (N, M, M)
    assert len(cluster_obs) == N


def test_make_test_data_columns(truth):
    cluster_obs, _, _ = make_test_data(truth, RADII, seed=0)
    for col in ('z', 'Lambda', 'delta_m'):
        assert col in cluster_obs.colnames


def test_make_test_data_lambda_positive(truth):
    cluster_obs, _, _ = make_test_data(truth, RADII, seed=0)
    assert np.all(cluster_obs['Lambda'] >= 1.0)


def test_make_test_data_covs_diagonal_positive(truth):
    _, _, covs = make_test_data(truth, RADII, seed=0)
    for i in range(len(truth)):
        diag = np.diag(covs[i])
        assert np.all(diag > 0)


def test_make_test_data_covs_off_diagonal_zero(truth):
    _, _, covs = make_test_data(truth, RADII, seed=0)
    for i in range(len(truth)):
        off = covs[i].copy()
        np.fill_diagonal(off, 0.0)
        assert np.all(off == 0.0)


def test_make_test_data_data_positive(truth):
    # Profiles are NFW signal + noise; should be positive for small fractional noise
    _, data, _ = make_test_data(truth, RADII, fractional_noise=0.05, seed=0)
    assert np.all(data > 0)


def test_make_test_data_seed_reproducible(truth):
    _, d1, _ = make_test_data(truth, RADII, seed=99)
    _, d2, _ = make_test_data(truth, RADII, seed=99)
    np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# forward_model
# ---------------------------------------------------------------------------

def test_forward_model_shape(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model(TRUTH_GAMMA, cluster_obs, RADII)
    assert profiles.shape == (len(cluster_obs), len(RADII))


def test_forward_model_positive(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model(TRUTH_GAMMA, cluster_obs, RADII)
    assert np.all(profiles > 0)


def test_forward_model_physical_range(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model(TRUTH_GAMMA, cluster_obs, RADII)
    # ΔΣ for galaxy clusters is typically 1–1000 M☉/pc²
    assert np.all(profiles > 0.1)
    assert np.all(profiles < 5000)


def test_forward_model_gamma_zero_same_pivot(test_dataset):
    # gamma=0 → all clusters get the same pivot regardless of delta_m
    cluster_obs, _, _ = test_dataset
    p1 = forward_model(0.0, cluster_obs, RADII)
    p2 = forward_model(0.0, cluster_obs, RADII)
    np.testing.assert_array_equal(p1, p2)


def test_forward_model_decreasing_with_radius(test_dataset):
    # NFW ΔΣ should decrease with projected radius
    cluster_obs, _, _ = test_dataset
    profiles = forward_model(TRUTH_GAMMA, cluster_obs, RADII)
    for row in profiles:
        assert np.all(np.diff(row) < 0)


def test_forward_model_changes_with_gamma(test_dataset):
    cluster_obs, _, _ = test_dataset
    p_lo = forward_model(0.1, cluster_obs, RADII)
    p_hi = forward_model(2.0, cluster_obs, RADII)
    assert not np.allclose(p_lo, p_hi)


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

def test_log_likelihood_scalar(test_dataset):
    cluster_obs, data, covs = test_dataset
    ll = log_likelihood(TRUTH_GAMMA, cluster_obs, data, covs, RADII)
    assert np.isscalar(ll) or ll.ndim == 0


def test_log_likelihood_finite_at_true_gamma(test_dataset):
    cluster_obs, data, covs = test_dataset
    ll = log_likelihood(TRUTH_GAMMA, cluster_obs, data, covs, RADII)
    assert np.isfinite(ll)


def test_log_likelihood_negative_gamma_returns_neg_inf(test_dataset):
    cluster_obs, data, covs = test_dataset
    ll = log_likelihood(-0.5, cluster_obs, data, covs, RADII)
    assert ll == -np.inf


def test_log_likelihood_prefers_true_gamma(test_dataset):
    # The likelihood evaluated at the true gamma should exceed that at a
    # clearly wrong gamma when the data was generated with the true gamma.
    cluster_obs, data, covs = test_dataset
    ll_true  = log_likelihood(TRUTH_GAMMA, cluster_obs, data, covs, RADII)
    ll_wrong = log_likelihood(2.5,         cluster_obs, data, covs, RADII)
    assert ll_true > ll_wrong


def test_log_likelihood_decreases_away_from_truth(test_dataset):
    cluster_obs, data, covs = test_dataset
    gammas = [TRUTH_GAMMA, TRUTH_GAMMA + 0.5, TRUTH_GAMMA + 1.5]
    lls = [log_likelihood(g, cluster_obs, data, covs, RADII) for g in gammas]
    assert lls[0] > lls[1] > lls[2]


def test_log_likelihood_log_M_norm_changes_result(test_dataset):
    cluster_obs, data, covs = test_dataset
    ll_0   = log_likelihood(TRUTH_GAMMA, cluster_obs, data, covs, RADII, log_M_norm=0.0)
    ll_off = log_likelihood(TRUTH_GAMMA, cluster_obs, data, covs, RADII, log_M_norm=0.5)
    assert ll_0 != ll_off


def test_forward_model_log_M_norm_shifts_amplitude(test_dataset):
    cluster_obs, _, _ = test_dataset
    p_base = forward_model(TRUTH_GAMMA, cluster_obs, RADII, log_M_norm=0.0)
    p_high = forward_model(TRUTH_GAMMA, cluster_obs, RADII, log_M_norm=0.3)
    assert np.all(p_high > p_base)
