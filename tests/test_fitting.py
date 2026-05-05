import numpy as np
import pytest
from scipy.stats import multivariate_normal
from destacking.fitting import cholog_gaussian, log_likelihood
from destacking.simulation import generate_truth_catalog, make_test_data

RADII = np.logspace(-0.6, 0.6, 8)
TRUTH_GAMMA = 0.5


def _make_data(n=10, k=8, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    # Random PD covariance matrices via A @ A.T + eps*I
    covs = []
    for _ in range(n):
        A = rng.standard_normal((k, k))
        covs.append(A @ A.T + np.eye(k) * 0.5)
    return np.zeros((n, k)), np.array(covs)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

def test_cholog_gaussian_returns_scalar():
    vecs, covs = _make_data()
    result = cholog_gaussian(vecs, covs)
    assert np.isscalar(result) or result.ndim == 0


def test_cholog_gaussian_zero_residuals_is_maximum():
    vecs, covs = _make_data()
    ll_zero = cholog_gaussian(np.zeros_like(vecs), covs)
    ll_nonzero = cholog_gaussian(np.ones_like(vecs), covs)
    assert ll_zero > ll_nonzero


def test_cholog_gaussian_nonzero_residuals_lower():
    rng = np.random.default_rng(7)
    vecs, covs = _make_data(rng=rng)
    residuals = rng.standard_normal(vecs.shape)
    ll_base = cholog_gaussian(vecs, covs)
    ll_noisy = cholog_gaussian(residuals, covs)
    assert ll_base > ll_noisy


def test_cholog_gaussian_non_pd_returns_neg_inf():
    n, k = 3, 4
    vecs = np.zeros((n, k))
    # Singular (rank-deficient) matrix
    covs = np.zeros((n, k, k))
    result = cholog_gaussian(vecs, covs)
    assert result == -np.inf


def test_cholog_gaussian_single_cluster():
    k = 5
    vecs = np.zeros((1, k))
    covs = np.eye(k)[np.newaxis]
    result = cholog_gaussian(vecs, covs)
    # For identity covariance and zero residual: -k/2 * log(2π)
    expected = -0.5 * k * np.log(2 * np.pi)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_cholog_gaussian_additive_over_clusters():
    # Sum of individual log-likelihoods must equal joint call
    rng = np.random.default_rng(3)
    n, k = 6, 4
    vecs, covs = _make_data(n=n, k=k, rng=rng)
    vecs = rng.standard_normal((n, k)) * 0.2

    total = cholog_gaussian(vecs, covs)
    per_cluster = sum(
        cholog_gaussian(vecs[i:i+1], covs[i:i+1]) for i in range(n)
    )
    np.testing.assert_allclose(total, per_cluster, rtol=1e-10)


def test_cholog_gaussian_consistent_with_scipy():
    rng = np.random.default_rng(5)
    n, k = 4, 6
    vecs, covs = _make_data(n=n, k=k, rng=rng)
    vecs = rng.standard_normal((n, k)) * 0.3

    result = cholog_gaussian(vecs, covs)
    expected = sum(
        multivariate_normal.logpdf(vecs[i], mean=np.zeros(k), cov=covs[i])
        for i in range(n)
    )
    np.testing.assert_allclose(result, expected, rtol=1e-8)


# ---------------------------------------------------------------------------
# log_likelihood
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_dataset():
    truth = generate_truth_catalog(30, seed=42, delta_m_gamma=TRUTH_GAMMA)
    return make_test_data(truth, RADII, fractional_noise=0.2, seed=7)


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
