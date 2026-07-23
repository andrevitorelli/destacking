import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm as sp_norm
from destacking.fitting import (
    cholog_gaussian,
    log_likelihood,
    log_likelihood_mc,
    log_likelihood_combined,
)
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
    # fractional_noise=0 + poisson_richness=False make the data deterministic
    # given the truth catalogue, so the likelihood is sharply peaked at the
    # truth gamma and the monotonicity assertions below hold exactly.  Both
    # noise sources must be off: fractional_noise controls the ΔΣ residuals
    # and poisson_richness controls the errors-in-variables scatter that
    # propagates through the M(Λ) inversion in the forward model.
    truth = generate_truth_catalog(30, seed=42, delta_m_gamma=TRUTH_GAMMA)
    return make_test_data(
        truth, RADII,
        fractional_noise=0.0,
        poisson_richness=False,
        seed=7,
    )


def test_log_likelihood_scalar(test_dataset):
    cluster_obs, data, covs = test_dataset
    ll = log_likelihood(TRUTH_GAMMA, cluster_obs, data, covs, RADII)
    assert np.isscalar(ll) or ll.ndim == 0


def test_log_likelihood_finite_at_true_gamma(test_dataset):
    cluster_obs, data, covs = test_dataset
    ll = log_likelihood(TRUTH_GAMMA, cluster_obs, data, covs, RADII)
    assert np.isfinite(ll)


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


# ---------------------------------------------------------------------------
# log_likelihood_mc — parametric M-c
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mc_dataset():
    truth = generate_truth_catalog(20, seed=42, delta_m_gamma=0.0)
    cluster_obs, data_vectors, cov_matrices = make_test_data(
        truth, RADII, fractional_noise=0.2, poisson_richness=False, seed=7,
    )
    return cluster_obs, data_vectors, cov_matrices


def test_log_likelihood_mc_finite(mc_dataset):
    cluster_obs, data, cov = mc_dataset
    ll = log_likelihood_mc(5.0, -0.1, cluster_obs, data, cov, RADII)
    assert np.isfinite(ll)


def test_log_likelihood_mc_c0_zero_returns_neg_inf(mc_dataset):
    cluster_obs, data, cov = mc_dataset
    assert log_likelihood_mc(0.0, -0.1, cluster_obs, data, cov, RADII) == -np.inf
    assert log_likelihood_mc(-3.0, -0.1, cluster_obs, data, cov, RADII) == -np.inf


def test_log_likelihood_mc_truth_beats_offset(mc_dataset):
    # Larger c0 offset from a plausible truth should give a worse fit
    cluster_obs, data, cov = mc_dataset
    ll_good = log_likelihood_mc(5.0, -0.1, cluster_obs, data, cov, RADII)
    ll_bad  = log_likelihood_mc(50.0, -0.1, cluster_obs, data, cov, RADII)
    assert ll_good > ll_bad


# ---------------------------------------------------------------------------
# log_likelihood_combined — M-c + gamma
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def combined_dataset():
    truth = generate_truth_catalog(20, seed=42, delta_m_gamma=0.0)
    cluster_obs, data_vectors, cov_matrices = make_test_data(
        truth, RADII, fractional_noise=0.2, poisson_richness=False, seed=7,
    )
    return cluster_obs, data_vectors, cov_matrices


def test_log_likelihood_combined_finite(combined_dataset):
    cluster_obs, data, cov = combined_dataset
    ll = log_likelihood_combined(5.0, -0.1, 0.0, cluster_obs, data, cov, RADII)
    assert np.isfinite(ll)


def test_log_likelihood_combined_c0_invalid_returns_neg_inf(combined_dataset):
    cluster_obs, data, cov = combined_dataset
    assert log_likelihood_combined(0.0, -0.1, 0.0,
                                   cluster_obs, data, cov, RADII) == -np.inf


def test_log_likelihood_combined_extreme_gamma_returns_neg_inf(combined_dataset):
    # Extreme gamma + small delta_m_ref overflows the forward model;
    # the NaN-guard must convert this to -inf (not NaN).
    cluster_obs, data, cov = combined_dataset
    ll = log_likelihood_combined(
        5.0, -0.1, 500.0, cluster_obs, data, cov, RADII,
        delta_m_ref=1e-4,
    )
    assert ll == -np.inf


def test_log_likelihood_combined_never_returns_nan(combined_dataset):
    # Sweep a wide range of parameters; output must always be finite or -inf.
    cluster_obs, data, cov = combined_dataset
    rng = np.random.default_rng(0)
    for _ in range(15):
        c0    = float(rng.uniform(0.5, 20.0))
        alpha = float(rng.uniform(-0.5, 0.5))
        gamma = float(rng.uniform(-5.0, 5.0))
        ll = log_likelihood_combined(c0, alpha, gamma,
                                     cluster_obs, data, cov, RADII)
        assert not np.isnan(ll), (
            f"NaN return for (c0,alpha,gamma)=({c0},{alpha},{gamma})"
        )


def test_log_likelihood_combined_gamma_zero_matches_mc(combined_dataset):
    # gamma=0 should yield the same log-likelihood as the bare M-c version.
    cluster_obs, data, cov = combined_dataset
    ll_combined = log_likelihood_combined(
        5.0, -0.1, 0.0, cluster_obs, data, cov, RADII,
        beta_cm=-0.3, log_M_norm=0.0,
    )
    ll_mc = log_likelihood_mc(
        5.0, -0.1, cluster_obs, data, cov, RADII,
        beta=-0.3, log_M_norm=0.0,
    )
    assert np.isclose(ll_combined, ll_mc, rtol=1e-10)


def test_log_likelihood_combined_FMISS_changes_value(combined_dataset):
    cluster_obs, data, cov = combined_dataset
    N = len(cluster_obs)
    ll_centred = log_likelihood_combined(
        5.0, -0.1, 0.0, cluster_obs, data, cov, RADII,
        FMISS=np.zeros(N), SIGMA_OFF=np.ones(N) * 0.3,
    )
    ll_misc = log_likelihood_combined(
        5.0, -0.1, 0.0, cluster_obs, data, cov, RADII,
        FMISS=np.ones(N), SIGMA_OFF=np.ones(N) * 0.3,
    )
    assert ll_centred != ll_misc


# ---------------------------------------------------------------------------
# Sampler smoke tests — minimal n_live to keep total wall time small.
# Marked as `slow` so users can de-select with -m "not slow".
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_sampler_mc_runs_and_returns_finite(mc_dataset):
    from destacking.fitting import run_sampler_mc
    cluster_obs, data, cov = mc_dataset
    sampler = run_sampler_mc(
        cluster_obs, data, cov, RADII,
        n_live=30, fit_beta=False, verbose=False,
    )
    points, log_w, log_l = sampler.posterior()
    # Nautilus may return -inf log-weights for points with vanishing
    # likelihood; what we require is that the sampler ran, produced points,
    # and contains at least some finite-weight samples.
    assert points.shape[0] > 0
    assert np.any(np.isfinite(log_w))
    # No NaNs
    assert not np.any(np.isnan(log_w))


@pytest.mark.slow
def test_run_sampler_combined_runs_and_returns_finite(combined_dataset):
    from destacking.fitting import run_sampler_combined
    cluster_obs, data, cov = combined_dataset
    priors = {
        "c0":         sp_norm(loc=5.0, scale=0.5),
        "alpha_cm":   sp_norm(loc=-0.1, scale=0.05),
        "beta_cm":    sp_norm(loc=-0.4, scale=0.2),
        "gamma_cm":   sp_norm(loc=0.0,  scale=1.0),
        "log_M_norm": sp_norm(loc=0.0,  scale=0.1),
    }
    sampler = run_sampler_combined(
        cluster_obs, data, cov, RADII,
        priors=priors, fit_beta=True,
        n_live=30, verbose=False,
    )
    points, log_w, log_l = sampler.posterior()
    assert points.shape[0] > 0
    assert points.shape[1] == 5  # c0, alpha_cm, beta_cm, gamma_cm, log_M_norm
    assert sampler.param_order == ["c0", "alpha_cm", "beta_cm", "gamma_cm",
                                   "log_M_norm"]
    assert np.any(np.isfinite(log_w))
    assert not np.any(np.isnan(log_w))
