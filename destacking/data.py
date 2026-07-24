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
