"""Expected contamination of candidate cleaner samples, from the eSCALE random catalogues (2026-11-27).

Real matches: the per-magnitude catalogues ~/data/eSCALE_project/eSCALE_<mag>.fits (one row per X-ray source and
magnitude limit; flags p80/p90/p95).  Randoms: the 12 random files (build_random_pool.LEVEL: magnitude limit and level).
Both get the cuts of build_escale_sample.py row by row (0.08 < z < 0.25 at the matched group's redshift; X-ray peak in
DES Y3 with >= 80% of a 3 Mpc disc covered; not in the three bad tiles; X-ray-optical offset <= OFF_MAX; finite flux),
the randoms with the optical position un-shifted (+10 deg in RA) for the offset.  An X-ray source passes level L if any
of its brightest-limit partner has flag L (real) or that pair lies in a level-L random file, as in our sample
(the merged catalogue takes each X-ray source's row from the brightest-limit catalogue); "any" = any level.  Expected spurious
fraction = N_rand / N_real, counted both as unique X-ray sources and as unique optical groups (each X-ray source's
partner from its brightest-limit catalogue; our sample keeps one X-ray source per optical group).

Options: which magnitude catalogues are used (all four merged; -20 only; ...), whether the match must appear in at least
two magnitude catalogues (at that level), the offset limit, and a flux limit.

    python contamination_options.py
"""
import glob
import os

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from build_random_pool import LEVEL, SHIFT

E = os.path.expanduser("~/data/eSCALE_project")
D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
BAD_TILES = (71150, 42144, 38132)
MAGS = ["20", "19.5", "19", "18.5"]
OCC = np.load(f"{D}/des_metacal_counts_nside512.npz")["cnt"] > 0


def flag(col):
    return ~np.asarray(col.mask) if hasattr(col, "mask") else np.asarray(col) == 1


def base_cuts(cid, ra_x, dec_x, ra_o, dec_o, z, flux):
    """Row-wise cuts (all but the offset limit); returns keep mask and offset [Mpc]."""
    DA = COSMO.angular_diameter_distance(np.clip(z, 1e-3, None)).to_value(u.Mpc)
    off = np.radians(SkyCoord(ra_o, dec_o, unit="deg").separation(SkyCoord(ra_x, dec_x, unit="deg")).deg) * DA
    zc = (z > 0.08) & (z < 0.25)
    vec = hp.ang2vec(ra_x, dec_x, lonlat=True)
    cover = np.zeros(len(z))
    for n in np.flatnonzero(zc):
        cover[n] = OCC[hp.query_disc(512, vec[n], 3.0 / DA[n])].mean()
    in_des = OCC[hp.ang2pix(512, ra_x, dec_x, lonlat=True)] & (cover >= 0.8)
    keep = zc & in_des & ~np.isin(cid // 1000, BAD_TILES) & np.isfinite(flux) & (flux > 0)
    return keep, off


def load():
    R = []
    for mag in MAGS:
        t = Table.read(f"{E}/eSCALE_{mag}.fits")
        R.append(dict(cid=np.asarray(t["CID6722"], np.int64), grp=np.asarray(t["ClusterID"], np.int64), mag=np.full(len(t), mag), ra_x=np.asarray(t["RA_peak"], float),
                      dec_x=np.asarray(t["DEC_peak"], float), ra_o=np.asarray(t["RA"], float), dec_o=np.asarray(t["DEC"], float),
                      z=np.asarray(t["z_splus"], float), flux=np.asarray(t["Flux0623"], float),
                      **{f"p{l}": flag(t[f"p{l}"]) for l in (80, 90, 95)}))
    real = {k: np.concatenate([r[k] for r in R]) for k in R[0]}
    Q = []
    for f in sorted(glob.glob(f"{E}/randoms/XMass_CID6722_rnd*_cleanflux3_final.fits")):
        mag, lev = LEVEL[os.path.basename(f).split("rnd")[1].split("_clean")[0]]
        t = Table.read(f)
        Q.append(dict(cid=np.asarray(t["CID6722"], np.int64), grp=np.asarray(t["ClusterID"], np.int64), mag=np.full(len(t), mag), ra_x=np.asarray(t["RA_peak"], float),
                      dec_x=np.asarray(t["DEC_peak"], float), ra_o=(np.asarray(t["RA"], float) + SHIFT) % 360,
                      dec_o=np.asarray(t["DEC"], float), z=np.asarray(t["z_splus"], float), flux=np.asarray(t["Flux0623"], float),
                      **{f"p{l}": np.full(len(t), lev == l) for l in (80, 90, 95)}))
    rnd = {k: np.concatenate([q[k] for q in Q]) for k in Q[0]}
    for d in (real, rnd):
        d["keep"], d["off"] = base_cuts(d["cid"], d["ra_x"], d["dec_x"], d["ra_o"], d["dec_o"], d["z"], d["flux"])
    return real, rnd


def count(d, level, mags, off_max, min_mags, flux_min):
    """As our sample is built: each X-ray source takes its partner (and flags) from the brightest-limit catalogue in
    `mags` where it is matched (for randoms: the union of the level files containing that same pair); the cuts are then
    applied to that partner.  min_mags > 1 also requires a match at this level in at least min_mags catalogues.
    Returns (unique X-ray sources, unique (catalogue, optical group) partners)."""
    lev = (d["p80"] | d["p90"] | d["p95"]) if level == "any" else d[f"p{level}"]
    inm = np.isin(d["mag"], mags)
    rank = np.array([MAGS.index(m) for m in d["mag"]])
    idx = np.flatnonzero(inm)
    o = idx[np.lexsort((rank[idx], d["cid"][idx]))]
    n_src, keys = 0, set()
    cid_o = d["cid"][o]
    starts = np.r_[0, np.flatnonzero(np.diff(cid_o)) + 1, len(o)]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        rows = o[s0:s1]
        b = rows[0]                                                    # brightest-limit catalogue for this source
        pair = rows[(d["mag"][rows] == d["mag"][b]) & (d["grp"][rows] == d["grp"][b])]
        if not lev[pair].any():
            continue
        if not (d["keep"][b] and d["off"][b] <= off_max and d["flux"][b] > flux_min):
            continue
        if min_mags > 1 and len(set(d["mag"][rows[lev[rows]]])) < min_mags:
            continue
        n_src += 1
        keys.add((d["mag"][b], int(d["grp"][b])))
    return n_src, len(keys)


def main():
    real, rnd = load()
    opts = [("all four catalogues merged (baseline)", MAGS, 2.0, 1, 0),
            ("-20 catalogue only", ["20"], 2.0, 1, 0),
            ("matched in >= 2 magnitude catalogues", MAGS, 2.0, 2, 0),
            ("offset <= 1 Mpc", MAGS, 1.0, 1, 0),
            ("offset <= 0.5 Mpc", MAGS, 0.5, 1, 0),
            ("flux > 1e-13", MAGS, 2.0, 1, 1e-13),
            ("-20 only, offset <= 1 Mpc", ["20"], 1.0, 1, 0),
            ("-20 only, offset <= 0.5 Mpc", ["20"], 0.5, 1, 0),
            (">= 2 catalogues, offset <= 1 Mpc", MAGS, 1.0, 2, 0),
            ("offset <= 0.5 Mpc, flux > 1e-13", MAGS, 0.5, 1, 1e-13)]
    print("| option | level | X-ray sources: real / random / fraction | optical groups: real / random / fraction |\n|---|---|---|---|")
    for lab, mags, om, mm, fm in opts:
        for level in ("any", 80, 90, 95):
            (nr, gr), (nq, gq) = count(real, level, mags, om, mm, fm), count(rnd, level, mags, om, mm, fm)
            print(f"| {lab} | {'full' if level == 'any' else 'p' + str(level)} | {nr} / {nq} / {nq / max(nr, 1):.2f} | {gr} / {gq} / {gq / max(gr, 1):.2f} |")
    # offset distributions
    for nm, d in (("real", real), ("random", rnd)):
        m = d["keep"] & (d["off"] <= 2.0)
        print(f"{nm}: offset percentiles 25/50/75 [Mpc] {np.round(np.percentile(d['off'][m], [25, 50, 75]), 2)}")


if __name__ == "__main__":
    main()
