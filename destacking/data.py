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


# ── Per-cluster data vector with an analytic shape-noise covariance ─────────
# Added 2026-09-29.  The per-cluster galaxy-bootstrap covariance
# (xlensing.data.bootstrap_signal) correlates with the value it accompanies:
# a ratio estimator's bootstrap variance depends on the measured value, so an
# inverse-variance likelihood up-weights bins that fluctuate high.  In
# LSST-like simulations this inflated inner bins by 3 % -> 0.2 % and biased the
# joint fit's c0 (+0.135) and beta_cm (+0.074); at DES-like density single
# bins beyond floor_rmax with ~500x too-small variances took over whole fits.
# The covariance below uses no individual shape: each galaxy contributes its
# weight, Sigma_crit and response, and a POPULATION shape-noise dispersion
# sigma_e (estimated from the cross-shear component of the whole source sample,
# which carries no lensing signal, or supplied per galaxy class).  Shape noise
# is independent between radial bins (each galaxy is in one bin), so the
# covariance is diagonal.

def population_sigma_e(stakes, min_count=1):
    """Per-component shape-noise dispersion from the cross shear of a sample.

    ``stakes`` is an iterable of xlensing stakes (rows Sigma_crit, e_t, e_x,
    W, R, M).  Returns sqrt(sum W e_x^2 / sum W) over all galaxies: the cross
    component has no lensing signal, so its weighted rms is the per-component
    dispersion of the measured shapes (intrinsic + measurement noise).
    """
    num = den = 0.0
    n = 0
    for st in stakes:
        if st is None:
            continue
        w = np.asarray(st[3], dtype=float)
        ex = np.asarray(st[2], dtype=float)
        num += float(np.sum(w * ex ** 2)); den += float(np.sum(w)); n += len(w)
    if n < min_count or den <= 0:
        raise ValueError("no galaxies to estimate sigma_e from")
    return float(np.sqrt(num / den))


def analytic_signal(stake, bin_limits, sigma_e, min_gal=3):
    """Direct Delta Sigma estimator and its analytic shape-noise covariance.

        DeltaSigma_b = sum_i e_t,i W_i / Sigma_cr,i  /  sum_i R_i W_i / Sigma_cr,i^2
        Var_b        = sum_i sigma_e,i^2 (W_i / Sigma_cr,i)^2 / (sum_i R_i W_i / Sigma_cr,i^2)^2

    (the xlensing ``signal`` estimator, with R = 1 + M).  ``sigma_e`` is a
    scalar or a per-galaxy array.  Bins with fewer than ``min_gal`` galaxies
    are NaN in the data vector (and get a unit placeholder variance), so the
    likelihood's NaN masking drops them.  Returns (dv, cov) with cov diagonal.
    """
    sig, et, ex, w, R, M = (np.asarray(stake[k], dtype=float) for k in range(6))
    bl = np.asarray(bin_limits, dtype=float)
    se2 = np.broadcast_to(np.asarray(sigma_e, dtype=float) ** 2, w.shape)
    in_bin = (R[:, None] > bl[:, 0]) & (R[:, None] < bl[:, 1])
    den = ((1.0 + M) * w / sig ** 2) @ in_bin
    num = (et * w / sig) @ in_bin
    var = (se2 * (w / sig) ** 2) @ in_bin
    n = in_bin.sum(0)
    ok = (n >= min_gal) & (den > 0)
    dv = np.where(ok, num / np.where(ok, den, 1.0), np.nan)
    v = np.where(ok, var / np.where(ok, den, 1.0) ** 2, 1.0)
    return dv, np.diag(v)


def bin_counts_from_backgrounds(bgs, bin_limits, variances, sigma_e, floor_rmax=0.5):
    """Per-cluster, per-bin source counts and floor flags for the Sellentin & Heavens likelihood (2026-10-11).

    ``bgs`` is a list (index = cluster row) or a dict {cluster row: stake} of xlensing stakes (rows Sigma_crit, e_t,
    e_x, W, R, M), ``variances`` the (N, Nbins) bootstrap variances of the checkpoint.  A bin is flagged as floored when
    its lower edge is below ``floor_rmax`` and its variance equals the shape-noise floor
    sigma_e^2 / sum((1+M) W / Sigma_crit^2) of xlensing.data.bootstrap_signal; such a bin's variance is not an estimate,
    so it keeps the Gaussian likelihood.  Returns (n_k int array, floored bool array), both (N, Nbins).
    """
    bl = np.asarray(bin_limits, dtype=float)
    var = np.asarray(variances, dtype=float)
    n_cl, n_b = var.shape
    nk = np.zeros((n_cl, n_b), int)
    floor = np.full((n_cl, n_b), np.nan)
    items = bgs.items() if isinstance(bgs, dict) else enumerate(bgs)
    for cid, st in items:
        i = int(cid)
        if st is None or not (0 <= i < n_cl):
            continue
        sig, w, R, M = (np.asarray(st[k], dtype=float) for k in (0, 3, 4, 5))
        inb = (R[:, None] > bl[:, 0]) & (R[:, None] < bl[:, 1])
        nk[i] = inb.sum(0)
        K = ((1.0 + M) * w / sig ** 2).astype(np.float32).astype(float) @ inb   # float32 as in bootstrap_signal
        floor[i] = np.where(K > 0, sigma_e ** 2 / np.where(K > 0, K, 1.0), np.nan)
    rmax = np.inf if floor_rmax is None else floor_rmax
    floored = ((bl[:, 0] < rmax)[None, :] & np.isfinite(floor) & np.isfinite(var)
               & (np.abs(var - floor) <= 1e-4 * floor))
    return nk, floored
