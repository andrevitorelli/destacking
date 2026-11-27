"""Pool of spurious X-ray-optical matches ("randoms") for the contamination-injection test (2026-11-24).

Input: ~/data/eSCALE_project/randoms/XMass_CID6722_rnd<mag>_<a>_<b>_cleanflux3_final.fits (from the eSCALE team).
Each row is a real eRASS1 X-ray source (CID6722; its own peak position, flux and flux error) matched to an S-PLUS
optical group of a catalogue shifted by -10 deg in RA (RA_opt = RA_peak - 10 deg, checked: median 9.99-10.02 deg,
Dec unchanged), i.e. a chance match.  z_splus, the richnesses and Lx (recomputed at that z) belong to the chance
partner.  For each optical magnitude limit (-18.5, -19, -19.5, -20) there are three files: the randoms that pass
the p80, p90 and p95 matching thresholds (largest to smallest file, confirmed by the team's file names 2026-11-27; N_rand / N_real = 0.25-0.28, 0.14-0.17,
0.07-0.08, i.e. 1 - purity).

The pool applies the cuts of build_escale_sample.py with the optical position un-shifted (+10 deg in RA) for the
S-PLUS-X-ray offset:  0.08 < z < 0.25; X-ray peak in DES Y3 (>= 80% of a 3 Mpc disc); not in tiles
071150/042144/038132; offset <= 2 Mpc.  One partner per X-ray source (the same source can be chance-matched in
several magnitude catalogues): the partner from the brightest-limit (-20) catalogue first, else -19.5, -19, -18.5.

Output: ~/destacking_data/lx_splus_erosita_des/randoms/escale_random_pool.fits with the columns of
escale_des_sample.fits (+ MAG, FLUX, FLUX_ERR, RP80/RP90/RP95 = the pair appears in that level's random file; TIER and the contamination
counts use these, as the real catalogue uses the flags of its brightest-limit row; AP80/90/95 = any magnitude, reference only),
ID = 1e9 + CID (never collides with the real S-PLUS IDs), and contamination.json with the per-level counts and the highest-flag tiers
(tier = highest purity flag; the flags are not nested, every real cluster has at least one) used by the mixture model.
usage: python build_random_pool.py
"""
import glob
import json
import os

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

SRC = os.path.expanduser("~/data/eSCALE_project/randoms")
D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
OUT = f"{D}/randoms"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
BAD_TILES = (71150, 42144, 38132)
Z_MIN, Z_MAX, OFF_MAX, R_CHECK, MIN_COVER, SHIFT = 0.08, 0.25, 2.0, 3.0, 0.8, 10.0
# file -> (magnitude limit, purity level); first identified by size within each magnitude, CONFIRMED 2026-11-27 by the
# eSCALE team's Drive names (shortcuts XMass_CID6722_rnd<mag>_p<level>.fits; ~/data/eSCALE_project/escale_randoms_filename_map.csv)
LEVEL = {"185_74_176": ("18.5", 80), "185_74_132": ("18.5", 90), "185_37_138": ("18.5", 95),
         "19_63_164": ("19", 80), "19_25_132": ("19", 90), "19_27_122": ("19", 95),
         "195_72_124": ("19.5", 80), "195_26_118": ("19.5", 90), "195_45_130": ("19.5", 95),
         "20_145_129": ("20", 80), "20_83_168": ("20", 90), "20_33_127": ("20", 95)}
MAG_ORDER = ["20", "19.5", "19", "18.5"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(f"{SRC}/XMass_CID6722_rnd*_cleanflux3_final.fits")):
        key = os.path.basename(f).split("rnd")[1].split("_clean")[0]
        mag, lev = LEVEL[key]
        t = Table.read(f)
        for r in t:
            rows.append((int(r["CID6722"]), int(r["ClusterID"]), mag, lev, float(r["RA_peak"]), float(r["DEC_peak"]),
                         float(r["RA"]), float(r["DEC"]), float(r["z_splus"]), float(r["Lx"]), float(r["eLx"]),
                         float(r["Flux0623"]), float(r["eFlux0623"]), float(r["M200c"]), float(r["R_ame"])))
    a = np.array(rows, dtype=[("cid", "i8"), ("opt", "i8"), ("mag", "U4"), ("lev", "i4"), ("ra", "f8"), ("dec", "f8"),
                              ("ra_opt", "f8"), ("dec_opt", "f8"), ("z", "f8"), ("lx", "f8"), ("elx", "f8"),
                              ("flux", "f8"), ("eflux", "f8"), ("m200", "f8"), ("lam", "f8")])
    log = [f"random rows {len(a)} (unique X-ray sources {len(np.unique(a['cid']))})"]

    # cuts, as build_escale_sample.py
    z = a["z"]
    DA = COSMO.angular_diameter_distance(np.clip(z, 1e-3, None)).to_value(u.Mpc)
    off = np.radians(SkyCoord((a["ra_opt"] + SHIFT) % 360, a["dec_opt"], unit="deg").separation(
        SkyCoord(a["ra"], a["dec"], unit="deg")).deg) * DA
    occ = np.load(f"{D}/des_metacal_counts_nside512.npz")["cnt"] > 0
    vec = hp.ang2vec(a["ra"], a["dec"], lonlat=True)
    cover = np.array([occ[hp.query_disc(512, v, R_CHECK / d)].mean() for v, d in zip(vec, DA)])
    in_des = occ[hp.ang2pix(512, a["ra"], a["dec"], lonlat=True)] & (cover >= MIN_COVER)
    keep = np.ones(len(a), bool)
    for lab, m in [("0.08 < z < 0.25", (z > Z_MIN) & (z < Z_MAX)), ("X-ray peak in DES", in_des),
                   ("not in bad tiles", ~np.isin(a["cid"] // 1000, BAD_TILES)), ("offset <= 2 Mpc", off <= OFF_MAX),
                   ("finite flux and Lx", np.isfinite(a["flux"]) & (a["flux"] > 0) & np.isfinite(a["lx"]) & (a["lx"] > 0))]:
        keep &= m
        log.append(f"  after {lab:22s}: {keep.sum()} rows, {len(np.unique(a['cid'][keep]))} X-ray sources")
    a, off = a[keep], off[keep]

    # per-level counts (one per X-ray source) with ANY-magnitude flags (used until 2026-11-27; kept for reference)
    counts_any = {lev: int(len(np.unique(a["cid"][a["lev"] == lev]))) for lev in (80, 90, 95)}
    # one partner per X-ray source: brightest magnitude limit first; flags = that pair is in the level's file
    pair_lev = {}
    for r in a:
        pair_lev.setdefault((r["cid"], r["opt"], r["mag"]), set()).add(int(r["lev"]))
    best = {}
    for k, r in enumerate(a):
        c = r["cid"]
        if c not in best or MAG_ORDER.index(r["mag"]) < MAG_ORDER.index(a[best[c]]["mag"]) \
                or (r["mag"] == a[best[c]]["mag"] and r["lev"] < a[best[c]]["lev"]):
            best[c] = k
    sel = np.array(sorted(best.values()))
    p, offp = a[sel], off[sel]
    Ez = COSMO.efunc(p["z"])
    out = Table()
    out["ID"] = (1_000_000_000 + p["cid"]).astype(np.int64)
    out["CID"] = p["cid"]
    out["RA"], out["DEC"] = p["ra"], p["dec"]
    out["RA_OPT"], out["DEC_OPT"] = (p["ra_opt"] + SHIFT) % 360, p["dec_opt"]
    out["Z"], out["EZ"] = p["z"], Ez
    out["LX"], out["LX_ERR"], out["LX_EZ"] = p["lx"], p["elx"], p["lx"] / Ez
    out["M200C_CAT"] = p["m200"]
    out["OFFSET_MPC"] = offp
    out["LAMBDA_AME"] = p["lam"]
    out["MAG"] = p["mag"]
    out["OPT_ID"] = p["opt"]
    out["FLUX"], out["FLUX_ERR"] = p["flux"], p["eflux"]
    for lev in (80, 90, 95):
        out[f"RP{lev}"] = np.array([lev in pair_lev[(r["cid"], r["opt"], r["mag"])] for r in p])
    # any-magnitude flags per X-ray source (as the real catalogue's flags), for the purity tiers of the mixture model
    for lev in (80, 90, 95):
        cids = set(a["cid"][a["lev"] == lev].tolist())
        out[f"AP{lev}"] = np.isin(p["cid"], list(cids))
    # TIER from the flags of the chosen (brightest-limit) partner, as the real merged catalogue takes each X-ray source's
    # row and flags from its brightest-limit catalogue (checked 2026-11-27: 3813/3813); the AP (any-magnitude) flags
    # over-counted randoms at p95/p90 relative to the real flags.
    out["TIER"] = np.where(out["RP95"], 95, np.where(out["RP90"], 90, 80))
    counts = {lev: int(np.sum(out[f"RP{lev}"])) for lev in (80, 90, 95)}
    out.meta["COSMO"] = "FlatLambdaCDM H0=70 Om0=0.3"
    out.meta["CENTRE"] = "X-ray peak; optical partner = chance match (catalogue shifted by 10 deg in RA)"
    out.write(f"{OUT}/escale_random_pool.fits", overwrite=True)
    log.append(f"pool: {len(out)} X-ray sources with one chance partner each "
               f"(magnitude limits {dict(zip(*np.unique(p['mag'], return_counts=True)))})")
    log.append(f"per-level unique X-ray sources after cuts (brightest-partner flags): {counts}; any-magnitude flags: {counts_any}")
    real = Table.read(f"{D}/escale_des_sample.fits")
    n_real = {80: int(real["P80"].sum()), 90: int(real["P90"].sum()), 95: int(real["P95"].sum()), "all": len(real)}
    tr = np.where(np.asarray(real["P95"], bool), 95, np.where(np.asarray(real["P90"], bool), 90, 80))
    tiers = {t: dict(n_random=int((out["TIER"] == t).sum()), n_real=int((tr == t).sum())) for t in (95, 90, 80)}
    for t in tiers:
        tiers[t]["q"] = tiers[t]["n_random"] / tiers[t]["n_real"]
    log.append("highest-flag tiers (random / real = q): " + ", ".join(f"{t}: {v['n_random']}/{v['n_real']} = {v['q']:.3f}" for t, v in tiers.items()))
    json.dump(dict(n_random_after_cuts=counts, n_random_after_cuts_anymag=counts_any, n_real=n_real, tiers=tiers, flags="brightest-limit partner (RP)",
                   ratio={lev: counts[lev] / n_real[lev] for lev in (80, 90, 95)}), open(f"{OUT}/contamination.json", "w"), indent=1)
    log.append(f"real sample: {n_real}; N_rand/N_real: " + ", ".join(f"p{l} {counts[l] / n_real[l]:.3f}" for l in (80, 90, 95)))
    print("\n".join(log))
    open(f"{OUT}/build_random_pool.log", "w").write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
