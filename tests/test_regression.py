import numpy as np
import pytest
from destacking.simulation import generate_truth_catalog, make_test_data
from destacking.fitting import run_sampler

RADII      = np.logspace(-0.6, 0.6, 8)
TRUE_GAMMA = 0.5


def _weighted_quantile(values, quantiles, weights):
    sorter = np.argsort(values)
    v = values[sorter]
    cumw = np.cumsum(weights[sorter]) / weights.sum()
    return np.interp(quantiles, cumw, v)


@pytest.mark.slow
def test_gamma_recovery():
    """
    End-to-end regression: the sampler recovers the injected gamma within
    its 95% credible interval.

    Uses scatter_logc=0 (no intrinsic concentration scatter) so the model
    is correctly specified and recovery is unbiased.  Run with:
        pytest -m slow
    """
    truth = generate_truth_catalog(
        50,
        delta_m_gamma=TRUE_GAMMA,
        scatter_logc=0.0,
        seed=42,
    )
    cluster_obs, data, covs = make_test_data(
        truth, RADII, fractional_noise=0.05, seed=43
    )
    sampler = run_sampler(
        cluster_obs, data, covs, RADII,
        gamma_range=(0.0, 3.0),
        n_live=100,
        verbose=False,
    )

    points, log_w, _ = sampler.posterior()
    gamma_samples = points[:, 0]
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()

    glo, ghi = _weighted_quantile(gamma_samples, [0.025, 0.975], weights)
    assert glo < TRUE_GAMMA < ghi, (
        f"True gamma={TRUE_GAMMA} not inside 95% CI [{glo:.3f}, {ghi:.3f}]"
    )
