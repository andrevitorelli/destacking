import numpy as np
import pytest
from destacking.data import (
    background_mask_margin,
    select_background,
    BACKGROUND_SELECTION_STRATEGIES,
    _stack_bin_idx,
    assign_stack_bins,
    magnitude_gaps,
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


# ---------------------------------------------------------------------------
# _stack_bin_idx
# ---------------------------------------------------------------------------

def test_stack_bin_idx_assigns_correct_bins():
    edges = np.array([0.1, 0.3, 0.5, 0.7])
    values = np.array([0.05, 0.15, 0.35, 0.55, 0.75])
    idx = _stack_bin_idx(values, edges)
    np.testing.assert_array_equal(idx, [-1, 0, 1, 2, -1])


def test_stack_bin_idx_boundary_values_included():
    edges = np.array([0.1, 0.3, 0.5, 0.7])
    # edges[0] and edges[-1] are both inclusive endpoints
    idx = _stack_bin_idx(np.array([0.1, 0.7]), edges)
    assert idx[0] != -1
    assert idx[1] != -1


def test_stack_bin_idx_all_out_of_range():
    edges = np.array([1.0, 2.0, 3.0])
    idx = _stack_bin_idx(np.array([-5.0, 10.0]), edges)
    np.testing.assert_array_equal(idx, [-1, -1])


# ---------------------------------------------------------------------------
# assign_stack_bins
# ---------------------------------------------------------------------------

def test_assign_stack_bins_output_shapes():
    n = 20
    rng = np.random.default_rng(0)
    z       = rng.uniform(0.05, 0.75, n)
    ngals   = rng.uniform(20, 90, n)
    delta_m = rng.uniform(0.1, 1.0, n)
    z_bin, ng_bin, dm_bin, dm_median, stack_name = assign_stack_bins(
        z, ngals, delta_m
    )
    assert z_bin.shape == (n,)
    assert ng_bin.shape == (n,)
    assert dm_bin.shape == (n,)
    assert stack_name.shape == (n,)
    assert stack_name.dtype.kind in "SU"


def test_assign_stack_bins_out_of_range_z_gets_no_stack():
    # z far outside STACK_Z_EDGES -> z_bin=-1 -> never gets a stack name
    n = 10
    z       = np.full(n, 99.0)
    ngals   = np.full(n, 30.0)
    delta_m = np.full(n, 0.5)
    z_bin, ng_bin, dm_bin, dm_median, stack_name = assign_stack_bins(
        z, ngals, delta_m
    )
    np.testing.assert_array_equal(z_bin, -1)
    assert all(s == "" for s in stack_name)


def test_assign_stack_bins_nonpositive_delta_m_excluded():
    # Non-finite or <=0 delta_m must never receive a stack, even if z/ngals
    # are perfectly in-range and there are plenty of cell-mates.
    n = 10
    z       = np.full(n, 0.2)   # inside [0.10, 0.30) -> z_bin=0
    ngals   = np.full(n, 30.0)  # inside [25, 35) -> ng_bin=0
    delta_m = np.array([0.5, 0.6, 0.7, np.nan, -0.1, 0.0,
                        np.inf, 0.8, 0.9, 1.0])
    z_bin, ng_bin, dm_bin, dm_median, stack_name = assign_stack_bins(
        z, ngals, delta_m
    )
    bad = ~np.isfinite(delta_m) | (delta_m <= 0)
    assert np.all(dm_bin[bad] == -1)
    assert all(stack_name[bad] == "")


def test_assign_stack_bins_median_split_gives_valid_stack_names():
    # Two full cells (>=2 valid delta_m each) so the median split fires and
    # every cluster gets a "z{i}_ng{j}_dm{k}" name.
    n = 6
    z       = np.full(n, 0.2)    # z_bin=0
    ngals   = np.full(n, 30.0)   # ng_bin=0
    delta_m = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    z_bin, ng_bin, dm_bin, dm_median, stack_name = assign_stack_bins(
        z, ngals, delta_m
    )
    assert np.all(z_bin == 0)
    assert np.all(ng_bin == 0)
    assert np.all(dm_bin >= 0)
    for s in stack_name:
        assert s == "z0_ng0_dm0" or s == "z0_ng0_dm1"
    assert dm_median[(0, 0)] == pytest.approx(np.median(delta_m))


def test_assign_stack_bins_single_member_cell_gets_no_dm_bin():
    # A (z, ngals) cell with fewer than 2 valid delta_m entries can't
    # define a median split -> dm_bin stays -1 for that lone cluster.
    n = 1
    z_bin, ng_bin, dm_bin, dm_median, stack_name = assign_stack_bins(
        np.array([0.2]), np.array([30.0]), np.array([0.5]),
    )
    assert dm_bin[0] == -1
    assert stack_name[0] == ""
    assert dm_median[(0, 0)] is None


# ---------------------------------------------------------------------------
# magnitude_gaps
# ---------------------------------------------------------------------------

def test_magnitude_gaps_certain_membership_is_deterministic():
    # p_mem=1 for the brightest inner satellite collapses the weighted sum
    # to the simple non-probabilistic magnitude gap.
    mag_gap, err = magnitude_gaps(
        mag_bcg=20.0,
        mag_mem=np.array([21.0, 22.0]),
        p_mem=np.array([1.0, 1.0]),
        r_mem=np.array([0.1, 0.2]),
        r_cut=0.5,
    )
    assert mag_gap == pytest.approx(1.0)
    assert err == pytest.approx(0.0, abs=1e-10)


def test_magnitude_gaps_no_inner_members_returns_nan():
    mag_gap, err = magnitude_gaps(
        mag_bcg=20.0,
        mag_mem=np.array([21.0, 22.0]),
        p_mem=np.array([1.0, 1.0]),
        r_mem=np.array([0.9, 1.2]),   # both outside r_cut
        r_cut=0.5,
    )
    assert np.isnan(mag_gap)
    assert np.isnan(err)

def test_magnitude_gaps_outer_members_ignored():
    # An outer member (r > r_cut) with a brighter-than-inner magnitude must
    # not affect the result -- only r < r_cut members count.
    gap_without_outer, err_without_outer = magnitude_gaps(
        mag_bcg=20.0,
        mag_mem=np.array([21.0]),
        p_mem=np.array([1.0]),
        r_mem=np.array([0.1]),
        r_cut=0.5,
    )
    gap_with_outer, err_with_outer = magnitude_gaps(
        mag_bcg=20.0,
        mag_mem=np.array([21.0, 15.0]),
        p_mem=np.array([1.0, 1.0]),
        r_mem=np.array([0.1, 5.0]),
        r_cut=0.5,
    )
    assert gap_with_outer == pytest.approx(gap_without_outer)
    assert err_with_outer == pytest.approx(err_without_outer)


def test_magnitude_gaps_partial_probability_weighted_average():
    # Hand-computed: p=[0.5,0.5], deltas=[1,2] -> mag_gap=1.0, err=sqrt(0.5)
    mag_gap, err = magnitude_gaps(
        mag_bcg=20.0,
        mag_mem=np.array([21.0, 22.0]),
        p_mem=np.array([0.5, 0.5]),
        r_mem=np.array([0.1, 0.2]),
        r_cut=0.5,
    )
    assert mag_gap == pytest.approx(1.0)
    assert err == pytest.approx(np.sqrt(0.5))


def test_magnitude_gaps_single_inner_member():
    mag_gap, err = magnitude_gaps(
        mag_bcg=20.0,
        mag_mem=np.array([22.0]),
        p_mem=np.array([1.0]),
        r_mem=np.array([0.1]),
        r_cut=0.5,
    )
    assert mag_gap == pytest.approx(2.0)
    assert err == pytest.approx(0.0, abs=1e-10)
