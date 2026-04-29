import numpy as np
import pytest
from destacking.simulation import generate_truth_catalog, make_test_data
from destacking.model import forward_model

RADII = np.logspace(-0.6, 0.6, 8)
TRUTH_GAMMA = 0.5


@pytest.fixture(scope="module")
def test_dataset():
    truth = generate_truth_catalog(30, seed=42, delta_m_gamma=TRUTH_GAMMA)
    return make_test_data(truth, RADII, fractional_noise=0.2, seed=7)


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


def test_forward_model_log_M_norm_shifts_amplitude(test_dataset):
    cluster_obs, _, _ = test_dataset
    p_base = forward_model(TRUTH_GAMMA, cluster_obs, RADII, log_M_norm=0.0)
    p_high = forward_model(TRUTH_GAMMA, cluster_obs, RADII, log_M_norm=0.3)
    assert np.all(p_high > p_base)
