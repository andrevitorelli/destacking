import numpy as np
from astropy import table


# ── Background-galaxy selection ─────────────────────────────────────────────
# xlensing.data.lensfit_cluster_lensing/metacal_cluster_lensing used to
# hardcode this choice (z_source > 1.1*z_cluster + margin) and apply it
# internally; that filtering was removed so it can be controlled and
# compared from the caller instead — every strategy below returns a
# boolean mask over the source arrays, applied BEFORE calling
# metacal_cluster_lensing/lensfit_cluster_lensing (source redshift is
# still passed through unfiltered-by-xlensing, since it feeds Sigma_crit).

def background_mask_margin(z_source, z_cluster, factor=1.1, offset=0.2):
    """Background selection: z_source > factor*z_cluster + offset.

    This is the original xlensing hardcoded default (lensfit used
    offset=0.1, metacal used offset=0.2; factor=1.1 in both), kept here as
    the baseline strategy now that selection is caller-controlled.
    """
    return np.asarray(z_source) > factor * np.asarray(z_cluster) + offset


BACKGROUND_SELECTION_STRATEGIES = {
    "margin": background_mask_margin,
}


def select_background(strategy, z_source, z_cluster, **kwargs):
    """Boolean mask selecting background sources for one cluster, dispatched
    by strategy name (see BACKGROUND_SELECTION_STRATEGIES). Add a new
    strategy by defining a `background_mask_<name>(z_source, z_cluster,
    **kwargs) -> bool array` function and registering it in
    BACKGROUND_SELECTION_STRATEGIES — no other code needs to change.
    """
    try:
        fn = BACKGROUND_SELECTION_STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown background selection strategy {strategy!r} — "
            f"choose from {sorted(BACKGROUND_SELECTION_STRATEGIES)}"
        )
    return fn(z_source, z_cluster, **kwargs)


# ── Stack (z x richness x delta_m) bin assignment ───────────────────────────
# Shared by wazp_stacked_analysis.py (which fits a boost model per stack)
# and prepare_real_data.py (which labels each cluster with its stack so
# gamma_inference.py can look up and apply that stack's boost correction).
# Both MUST use this one function — duplicating the binning logic risks a
# cluster being labelled into a different stack than the one its boost
# model was actually fit from.
STACK_Z_EDGES  = np.array([0.10, 0.30, 0.50, 0.70])
STACK_NG_EDGES = np.array([25., 35., 55., 85., np.inf])


def _stack_bin_idx(values, edges):
    v = np.asarray(values, dtype=float)
    idx = np.digitize(v, edges[1:-1], right=False)
    out = ~((v >= edges[0]) & (v <= edges[-1]))
    idx[out] = -1
    return idx


def assign_stack_bins(z, ngals, delta_m,
                      z_edges=STACK_Z_EDGES, ng_edges=STACK_NG_EDGES):
    """3 z x N_NG richness x 2 delta_m (median-split per (z, NGALS) cell)
    stack assignment. delta_m entries that are non-finite or <= 0 (cluster-
    finder artefacts) never enter the per-cell median computation and never
    get a stack — matches wazp_stacked_analysis.py's own DELTA_M_I>0
    pre-filter even when the caller (e.g. prepare_real_data.py) passes an
    unfiltered catalogue.

    Returns
    -------
    z_bin, ng_bin, dm_bin : int arrays (N,)
        -1 where a cluster falls outside every bin (or, for dm_bin, has no
        valid delta_m or too few cell-mates to define a median).
    dm_median_per_cell : dict[(i, j) -> float or None]
    stack_name : str array (N,)
        ``"z{i}_ng{j}_dm{k}"`` where all three bins are valid, else ``""``.
    """
    n_ng = len(ng_edges) - 1
    z       = np.asarray(z, dtype=float)
    ngals   = np.asarray(ngals, dtype=float)
    delta_m = np.asarray(delta_m, dtype=float)
    n = len(z)

    z_bin  = _stack_bin_idx(z, z_edges)
    ng_bin = _stack_bin_idx(ngals, ng_edges)
    dm_valid = np.isfinite(delta_m) & (delta_m > 0)

    dm_bin = np.full(n, -1, dtype=int)
    dm_median_per_cell = {}
    for i in range(len(z_edges) - 1):
        for j in range(n_ng):
            cell = (z_bin == i) & (ng_bin == j) & dm_valid
            if cell.sum() < 2:
                dm_median_per_cell[(i, j)] = None
                continue
            cell_dm = delta_m[cell]
            med = float(np.median(cell_dm))
            dm_median_per_cell[(i, j)] = med
            glob_idx = np.where(cell)[0]
            dm_bin[glob_idx[cell_dm < med]] = 0
            dm_bin[glob_idx[cell_dm >= med]] = 1

    valid = (z_bin >= 0) & (ng_bin >= 0) & (dm_bin >= 0)
    stack_name = np.full(n, "", dtype=object)
    for i in np.where(valid)[0]:
        stack_name[i] = f"z{z_bin[i]}_ng{ng_bin[i]}_dm{dm_bin[i]}"

    return z_bin, ng_bin, dm_bin, dm_median_per_cell, stack_name.astype(str)


def magnitude_gaps(mag_bcg, mag_mem, p_mem, r_mem, r_cut=.5):
    """
    Calculates the expected magnitude gap between the central galaxy and the
    brightest satellite galaxy within a radius of r_cut, considering the
    probability of membership.

    Parameters
    ----------
    mag_bcg : float
        Magnitude of the central galaxy
    mag_mem : array_like (N,)
        Magnitudes of the satellite galaxies
    p_mem : array_like (N,)
        Probabilities of membership of satellite galaxies
    r_mem : array_like (N,)
        Radius of the member galaxies from the cluster centre
    r_cut : float
        Cut radius to define inner satellites from outskirt satellites

    Returns
    -------

    mag_gap : float
        Expected value of the magnitude gap
    error_mag_gap:
        Square root of variance of the magnitude gap
    """
    centre_galaxies = r_mem < r_cut
    tabled_members = table.Table([mag_mem, p_mem, r_mem],
                                 names=["mag", "prob", "rad"])
    centre_members = tabled_members[centre_galaxies]
    centre_size = sum(centre_galaxies)

    if centre_size == 0:
        return np.nan, np.nan

    centre_members.sort('mag')
    deltas = centre_members['mag'] - mag_bcg
    deltas2 = deltas * deltas
    anti_probs = 1 - centre_members['prob']
    factor = [np.prod(anti_probs[0:n]) for n in range(centre_size)]
    probs = centre_members['prob'] * factor

    mag_gap = sum(probs * deltas)
    error_mag_gap = np.sqrt(sum(probs * deltas2) - mag_gap ** 2)

    return mag_gap, error_mag_gap
