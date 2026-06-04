import numpy as np
from astropy import table


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
