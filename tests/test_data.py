import numpy as np
import pytest
from destacking.data import (
    background_mask_margin,
    select_background,
    BACKGROUND_SELECTION_STRATEGIES,
)


def test_background_mask_margin_default_threshold():
    z_cluster = 0.5
    z_source = np.array([0.1, 0.75001, 0.9, 1.2, 0.3])
    mask = background_mask_margin(z_source, z_cluster)
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_background_mask_margin_custom_factor_offset():
    z_cluster = 0.4
    z_source = np.array([0.4, 0.5, 0.6])
    mask = background_mask_margin(z_source, z_cluster, factor=1.0, offset=0.1)
    np.testing.assert_array_equal(mask, [False, False, True])


def test_select_background_dispatches_by_name():
    z_cluster = 0.5
    z_source = np.array([0.1, 0.9])
    mask = select_background("margin", z_source, z_cluster)
    np.testing.assert_array_equal(mask, [False, True])


def test_select_background_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown background selection strategy"):
        select_background("not_a_real_strategy", np.array([0.9]), 0.5)


def test_background_selection_strategies_registry_contains_margin():
    assert "margin" in BACKGROUND_SELECTION_STRATEGIES
    assert BACKGROUND_SELECTION_STRATEGIES["margin"] is background_mask_margin
