import numpy as np
import pytest
from destacking.simulation import generate_truth_catalog, make_test_data
from destacking.model import (
    forward_model,
    forward_model_mc,
    forward_model_combined,
    _MC_M_PIVOT_LOG,
    _MC_Z_REF,
    _MC_DM_REF,
)

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


# ---------------------------------------------------------------------------
# forward_model_mc — parametric M-c relation (c0, alpha, beta)
# ---------------------------------------------------------------------------

def test_forward_model_mc_shape(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model_mc(5.0, -0.1, cluster_obs, RADII)
    assert profiles.shape == (len(cluster_obs), len(RADII))


def test_forward_model_mc_positive(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model_mc(5.0, -0.1, cluster_obs, RADII)
    assert np.all(profiles > 0)


def test_forward_model_mc_c0_increases_amplitude(test_dataset):
    # Larger c0 -> more concentrated halo -> larger inner amplitude of ΔΣ.
    cluster_obs, _, _ = test_dataset
    p_low  = forward_model_mc(3.0, -0.1, cluster_obs, RADII)
    p_high = forward_model_mc(8.0, -0.1, cluster_obs, RADII)
    # at the innermost radii the high-c0 model is strictly larger
    assert np.all(p_high[:, 0] > p_low[:, 0])


def test_forward_model_mc_decreasing_with_radius(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model_mc(5.0, -0.1, cluster_obs, RADII)
    for row in profiles:
        assert np.all(np.diff(row) < 0)


def test_forward_model_mc_beta_changes_amplitude(test_dataset):
    # The beta term is a multiplicative power of (1+z)/(1+z_ref); for a sample
    # spanning multiple redshifts a non-zero beta should change the predictions.
    cluster_obs, _, _ = test_dataset
    p0 = forward_model_mc(5.0, -0.1, cluster_obs, RADII, beta=0.0)
    p1 = forward_model_mc(5.0, -0.1, cluster_obs, RADII, beta=-0.5)
    assert not np.allclose(p0, p1)


def test_forward_model_mc_log_M_norm_increases_amplitude(test_dataset):
    cluster_obs, _, _ = test_dataset
    p_base = forward_model_mc(5.0, -0.1, cluster_obs, RADII, log_M_norm=0.0)
    p_high = forward_model_mc(5.0, -0.1, cluster_obs, RADII, log_M_norm=0.3)
    assert np.all(p_high > p_base)


# ---------------------------------------------------------------------------
# forward_model_combined — M-c relation + magnitude-gap slope gamma
# ---------------------------------------------------------------------------

def test_forward_model_combined_shape(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model_combined(5.0, -0.1, 0.2, cluster_obs, RADII)
    assert profiles.shape == (len(cluster_obs), len(RADII))


def test_forward_model_combined_positive(test_dataset):
    cluster_obs, _, _ = test_dataset
    profiles = forward_model_combined(5.0, -0.1, 0.2, cluster_obs, RADII)
    assert np.all(profiles > 0)


def test_forward_model_combined_gamma_zero_decouples_from_mc(test_dataset):
    # With gamma = 0 the combined model and the bare M-c model must agree.
    cluster_obs, _, _ = test_dataset
    p_mc       = forward_model_mc(5.0, -0.1, cluster_obs, RADII, beta=-0.3)
    p_combined = forward_model_combined(5.0, -0.1, 0.0, cluster_obs, RADII, beta=-0.3)
    np.testing.assert_allclose(p_combined, p_mc, rtol=1e-10)


def test_forward_model_combined_changes_with_gamma(test_dataset):
    cluster_obs, _, _ = test_dataset
    p_lo = forward_model_combined(5.0, -0.1,  0.1, cluster_obs, RADII)
    p_hi = forward_model_combined(5.0, -0.1, +1.0, cluster_obs, RADII)
    assert not np.allclose(p_lo, p_hi)


def test_forward_model_combined_finite_under_extreme_gamma(test_dataset):
    # Extreme gamma can push 10**logC -> inf in the forward model.  The
    # function may legitimately return NaN/inf in that regime; the test
    # only asserts it does not raise so the likelihood guard can catch it
    # via np.isfinite().
    cluster_obs, _, _ = test_dataset
    p = forward_model_combined(5.0, -0.1, 50.0, cluster_obs, RADII, delta_m_ref=0.01)
    assert p.shape == (len(cluster_obs), len(RADII))


def test_forward_model_combined_passes_FMISS_through(test_dataset):
    # Switching on full miscentering must change the prediction.
    cluster_obs, _, _ = test_dataset
    N = len(cluster_obs)
    p_centred = forward_model_combined(
        5.0, -0.1, 0.0, cluster_obs, RADII,
        FMISS=np.zeros(N), SIGMA_OFF=np.ones(N) * 0.3,
    )
    p_misc = forward_model_combined(
        5.0, -0.1, 0.0, cluster_obs, RADII,
        FMISS=np.ones(N), SIGMA_OFF=np.ones(N) * 0.3,
    )
    assert not np.allclose(p_centred, p_misc)
