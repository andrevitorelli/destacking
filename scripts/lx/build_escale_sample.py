"""Build the eSCALE x DES Y3 lensing sample for the Lx-M analysis (2026-10-21).

Input: gas_mass_uniqueIDs_AME_r200.fits (eSCALE, Doubrawa et al. 2026, SCALE II) and the DES Y3
metacal source-count HEALPix map made by notebooks/lx_splus_erosita_exploration.ipynb.

Selection (agreed with the user 2026-10-21):
  * one row per S-PLUS cluster (ClusterID_1): the row with the smallest eLx/Lx (tile duplicates);
  * 0.08 < z_splus < 0.25 (the SCALE II range; excludes the Horologium-Reticulum region);
  * X-ray peak inside the DES Y3 footprint, with >= 80% of a 3 Mpc disc covered by source pixels;
  * drop eRASS1 tiles 071150, 042144, 038132 (optical-X-ray offsets of order a degree) and any
    cluster whose S-PLUS-X-ray offset exceeds 2 Mpc (likely a wrong match).
Centre for lensing: the X-ray peak (RA_peak, DEC_peak).  Redshift: z_splus.
Cosmology: flat LCDM H0=70, Om0=0.3 (the catalogue's).

Output columns: ID, RA, DEC (X-ray peak), Z, LX, LX_ERR, LX_EZ (Lx / E(z)), EZ, M200C_CAT (Leauthaud+10,
reference only), RA_OPT, DEC_OPT, OFFSET_MPC, LAMBDA_AME (R_amec-20), P80, P90, P95, CID.
usage: python build_escale_sample.py [out.fits]
"""
import os, sys
import numpy as np, healpy as hp
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
import astropy.units as u

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
CAT = f"{D}/gas_mass_uniqueIDs_AME_r200.fits"
SRC_MAP = f"{D}/des_metacal_counts_nside512.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{D}/escale_des_sample.fits"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
BAD_TILES = (71150, 42144, 38132)
Z_MIN, Z_MAX, OFF_MAX, R_CHECK, MIN_COVER = 0.08, 0.25, 2.0, 3.0, 0.8

cat = Table.read(CAT)
order = np.lexsort((np.asarray(cat["eLx"] / cat["Lx"]), np.asarray(cat["ClusterID_1"])))
_, first = np.unique(np.asarray(cat["ClusterID_1"])[order], return_index=True)
cl = cat[order[first]]
log = [f"rows {len(cat)} -> unique S-PLUS clusters {len(cl)}"]

z = np.asarray(cl["z_splus"], float)
DA = COSMO.angular_diameter_distance(z).to_value(u.Mpc)
off = np.radians(SkyCoord(cl["RA"], cl["DEC"], unit="deg").separation(
    SkyCoord(cl["RA_peak"], cl["DEC_peak"], unit="deg")).deg) * DA

nside = 512
occ = np.load(SRC_MAP)["cnt"] > 0
ra, dec = np.asarray(cl["RA_peak"], float), np.asarray(cl["DEC_peak"], float)
vec = hp.ang2vec(ra, dec, lonlat=True)
cover = np.array([occ[hp.query_disc(nside, v, R_CHECK / d)].mean() for v, d in zip(vec, DA)])
in_des = occ[hp.ang2pix(nside, ra, dec, lonlat=True)] & (cover >= MIN_COVER)
tile = np.asarray(cl["CID6722"]) // 1000

cuts = [("0.08 < z < 0.25", (z > Z_MIN) & (z < Z_MAX)),
        ("X-ray peak inside DES (>=80% of 3 Mpc disc)", in_des),
        ("not in tiles 071150/042144/038132", ~np.isin(tile, BAD_TILES)),
        (f"S-PLUS-X-ray offset <= {OFF_MAX} Mpc", off <= OFF_MAX)]
keep = np.ones(len(cl), bool)
for lab, m in cuts:
    keep &= m
    log.append(f"  after {lab:45s}: {keep.sum()}")
s = cl[keep]
zs = z[keep]
Ez = COSMO.efunc(zs)
out = Table()
out["ID"] = np.asarray(s["ClusterID_1"], np.int64)
out["CID"] = np.asarray(s["CID6722"], np.int64)
out["RA"], out["DEC"] = np.asarray(s["RA_peak"], float), np.asarray(s["DEC_peak"], float)
out["RA_OPT"], out["DEC_OPT"] = np.asarray(s["RA"], float), np.asarray(s["DEC"], float)
out["Z"] = zs
out["EZ"] = Ez
out["LX"] = np.asarray(s["Lx"], float)
out["LX_ERR"] = np.asarray(s["eLx"], float)
out["LX_EZ"] = out["LX"] / Ez
out["M200C_CAT"] = np.asarray(s["M200c"], float)
out["OFFSET_MPC"] = off[keep]
out["LAMBDA_AME"] = np.asarray(s["R_amec-20"], float)
for c in ("p80", "p90", "p95"):
    out[c.upper()] = ~np.asarray(s[c].mask) if hasattr(s[c], "mask") else np.ones(len(s), bool)
out.meta["COSMO"] = "FlatLambdaCDM H0=70 Om0=0.3"
out.meta["CENTRE"] = "X-ray peak (RA_peak, DEC_peak)"
out.write(OUT, overwrite=True)
log.append(f"wrote {len(out)} clusters to {OUT}")
log.append(f"  median z = {np.median(zs):.4f}, median log10(Lx/E) = {np.median(np.log10(out['LX_EZ'])):.3f}, "
           f"median log10(M200c_cat E) = {np.median(np.log10(out['M200C_CAT'] * Ez)):.3f}")
print("\n".join(log))
