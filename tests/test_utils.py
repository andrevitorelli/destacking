import numpy as np
import pytest

from destacking.utils import (
    fmt_duration,
    deterministic_seed,
    gamma_dir_name,
    weighted_quantile,
)


# ---------------------------------------------------------------------------
# fmt_duration
# ---------------------------------------------------------------------------

def test_fmt_duration_seconds():
    assert fmt_duration(42.0) == "42.0s"


def test_fmt_duration_seconds_boundary_just_under_minute():
    assert fmt_duration(59.9) == "59.9s"


def test_fmt_duration_minutes():
    assert fmt_duration(187) == "3m07s"


def test_fmt_duration_minutes_boundary_at_exactly_one_minute():
    assert fmt_duration(60) == "1m00s"


def test_fmt_duration_hours():
    assert fmt_duration(7500) == "2h05m"


def test_fmt_duration_hours_boundary_at_exactly_one_hour():
    assert fmt_duration(3600) == "1h00m"


# ---------------------------------------------------------------------------
# deterministic_seed
# ---------------------------------------------------------------------------

def test_deterministic_seed_formula():
    assert deterministic_seed(3, 10) == 300_010


def test_deterministic_seed_zero_index():
    assert deterministic_seed(0, 7) == 7


def test_deterministic_seed_reproducible():
    assert deterministic_seed(5, 2) == deterministic_seed(5, 2)


def test_deterministic_seed_unique_per_g_idx():
    assert deterministic_seed(1, 10) != deterministic_seed(2, 10)


def test_deterministic_seed_accepts_float_offset():
    # offset is explicitly int()-cast so a float offset shouldn't raise
    assert deterministic_seed(0, 10.0) == 10


# ---------------------------------------------------------------------------
# gamma_dir_name
# ---------------------------------------------------------------------------

def test_gamma_dir_name_positive():
    assert gamma_dir_name(0, 0.5) == "simulation_g00_gamma+0.5"


def test_gamma_dir_name_negative():
    assert gamma_dir_name(3, -1.2) == "simulation_g03_gamma-1.2"


def test_gamma_dir_name_zero_has_explicit_plus_sign():
    assert gamma_dir_name(1, 0.0) == "simulation_g01_gamma+0.0"


def test_gamma_dir_name_index_zero_padded():
    assert gamma_dir_name(7, 0.1).startswith("simulation_g07_")


def test_gamma_dir_name_index_beyond_two_digits():
    # g_idx format is `{:02d}` -- must not truncate for >=100
    assert gamma_dir_name(123, 0.0) == "simulation_g123_gamma+0.0"


# ---------------------------------------------------------------------------
# weighted_quantile
# ---------------------------------------------------------------------------

def test_weighted_quantile_uniform_weights_interpolates_on_raw_cumsum():
    # weighted_quantile interpolates directly on cumsum(w)/sum(w) with no
    # bin-centering correction, so it does NOT reproduce np.median's
    # convention exactly -- pin the documented behaviour instead.
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.ones_like(v)
    result = weighted_quantile(v, w, 0.5)
    np.testing.assert_allclose(result, 2.5, atol=1e-8)


def test_weighted_quantile_returns_float():
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0])
    result = weighted_quantile(v, w, 0.5)
    assert isinstance(result, float)


def test_weighted_quantile_min_and_max():
    v = np.array([10.0, 20.0, 30.0])
    w = np.array([1.0, 1.0, 1.0])
    assert weighted_quantile(v, w, 0.0) == pytest.approx(10.0)
    assert weighted_quantile(v, w, 1.0) == pytest.approx(30.0)


def test_weighted_quantile_unsorted_input_handled():
    v = np.array([3.0, 1.0, 2.0])
    w = np.array([1.0, 1.0, 1.0])
    result_unsorted = weighted_quantile(v, w, 0.5)
    result_sorted = weighted_quantile(np.sort(v), w, 0.5)
    np.testing.assert_allclose(result_unsorted, result_sorted)


def test_weighted_quantile_concentrated_weight_dominates():
    # Nearly all weight on the smallest value -> median should sit near it
    v = np.array([1.0, 2.0, 100.0])
    w = np.array([1e6, 1.0, 1.0])
    result = weighted_quantile(v, w, 0.5)
    assert result < 2.0


def test_weighted_quantile_accepts_python_lists():
    result = weighted_quantile([1, 2, 3], [1, 1, 1], 0.5)
    np.testing.assert_allclose(result, 1.5)
