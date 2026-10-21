"""Per-cluster lensing measurement of the eSCALE x DES Y3 sample (2026-10-21).

For every cluster (X-ray peak centre, z_splus) it stores
  * the raw background array (stake: Sigma_crit, e_t, e_x, W, R, M), from which the stacks and their member
    weights K_c(R) are built (16 bins, 0.16-3.98 Mpc, as in paper 1.1);
  * the per-cluster data vector and covariance on the joint fit's 6-bin grid (0.25-3.98 Mpc centres): galaxy
    bootstrap (200 draws) with each inner (< 0.5 Mpc) bin's variance raised, where needed, to the analytic
    shape-noise variance (population sigma_e) -- the measurement of paper 1.1's joint fit (bootfloor_v4);
  * per-bin source counts n_k and the 'floored' flags for the Sellentin & Heavens likelihood.
Settings identical to paper 1.1 except: cosmology flat LCDM 70/0.3 (the catalogue's), and one background
selection for all frameworks (margin z_s > 1.1 z_l + 0.2).

usage: python measure_escale.py [n_workers]
writes ~/destacking_data/lx_splus_erosita_des/measurement/{stakes.pkl, joint6.npz, meta.json}
"""
import os, sys, json, time, pickle
import numpy as np

from destacking import lx
lx.set_cosmology()
import xlensing.cosmo as xc
from xlensing.data import bootstrap_signal, metacal_cluster_lensing
from destacking.data import analytic_signal, select_background
from astropy.table import Table

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
OUT = f"{D}/measurement"
SAMPLE = f"{D}/escale_des_sample.fits"
SRC = os.path.expanduser("~/destacking_data/real_data/metacal_filtered.npz")
RADIUS = 5.3
BG = dict(factor=1.1, offset=0.2)
N_BOOT, FLOOR_RMAX, MIN_SOURCES = 200, 0.5, 20


def bins(log_in, log_out, n):
    r = np.logspace(log_in, log_out, n); d = np.diff(np.log10(r))[0]
    e = np.logspace(np.log10(r[0]) - d / 2, np.log10(r[-1]) + d / 2, n + 1)
    return r, np.column_stack([e[:-1], e[1:]])


R6, BL6 = bins(-0.6, 0.6, 6)
R16, BL16 = bins(-0.8, 0.6, 16)
_TREE = _SR = _SIGE = None


def work(args):
    i, cid, ra, dec, z = args
    try:
        ra_r, dec_r = np.deg2rad(ra), np.deg2rad(dec)
        idx = _TREE.query_radius(np.array([[dec_r, ra_r]]), r=RADIUS / float(xc.DA(0, z)))[0]
        idx = idx[select_background("margin", _SR["z"][idx], z, **BG)]
        if len(idx) == 0:
            return i, None, None, None, 0
        src = tuple(_SR[k][idx] for k in ("ra", "dec", "z", "e1", "e2", "w", "R11", "R12", "R21", "R22"))
        bg = metacal_cluster_lensing((ra_r, dec_r, z), src, radius=RADIUS)
        if bg["Count"] < MIN_SOURCES:
            return i, None, None, None, int(bg["Count"])
        stake = np.array([bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
                          bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"]])
        dv, cov, _, _ = bootstrap_signal(stake, BL6, Nboot=N_BOOT)
        _, an_cov = analytic_signal(stake, BL6, _SIGE, min_gal=1)
        an = np.diag(an_cov); d = np.diag(cov).copy()
        inner = BL6[:, 0] < FLOOR_RMAX
        ok = np.isfinite(d) & np.isfinite(an) & inner
        floored = ok & (an > d)
        cov = cov.copy(); cov[np.diag_indices_from(cov)] = np.where(floored, an, d)
        valid = np.isfinite(dv)
        cov = np.where(np.isfinite(cov), cov, 0.0); cov[~valid, ~valid] = 1.0
        nk = ((stake[4][:, None] > BL6[:, 0]) & (stake[4][:, None] < BL6[:, 1])).sum(0)
        return i, stake.astype(np.float32), (dv, cov, nk, floored), None, int(bg["Count"])
    except Exception as exc:
        print(f"  cluster {cid} FAILED: {exc!r}", flush=True)
        return i, None, None, None, 0


def main():
    global _TREE, _SR, _SIGE
    import multiprocess as mp
    from sklearn.neighbors import BallTree
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    os.makedirs(OUT, exist_ok=True)
    cat = Table.read(SAMPLE)
    t0 = time.time()
    f = np.load(SRC)
    _SR = {k: f[k] for k in ("ra", "dec", "z", "e1", "e2", "w", "R11", "R12", "R21", "R22")}
    w = _SR["w"].astype(float)
    _SIGE = float(np.sqrt(np.sum(w * (_SR["e1"].astype(float) ** 2 + _SR["e2"].astype(float) ** 2)) / (2 * np.sum(w))))
    print(f"{len(cat)} clusters; {len(w):,} sources; sigma_e (population) = {_SIGE:.4f} ({time.time()-t0:.0f}s)", flush=True)
    _TREE = BallTree(np.column_stack([_SR["dec"], _SR["ra"]]), metric="haversine")
    print(f"BallTree built ({time.time()-t0:.0f}s)", flush=True)
    n = len(cat)
    DV = np.full((n, 6), np.nan); COV = np.tile(np.eye(6), (n, 1, 1)); NK = np.zeros((n, 6), int)
    FL = np.zeros((n, 6), bool); NSRC = np.zeros(n, int); stakes = {}
    jobs = [(i, int(cat["ID"][i]), float(cat["RA"][i]), float(cat["DEC"][i]), float(cat["Z"][i])) for i in range(n)]
    with mp.Pool(nw) as pool:
        for k, (i, st, res, _, ns) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            NSRC[i] = ns
            if st is not None:
                stakes[int(cat["ID"][i])] = st
                DV[i], COV[i], NK[i], FL[i] = res
            if (k + 1) % 100 == 0 or k + 1 == n:
                dt = time.time() - t0
                print(f"  {k+1:5d}/{n}  ok {len(stakes):5d}  elapsed {dt/60:.1f} min", flush=True)
    pickle.dump(stakes, open(f"{OUT}/stakes.pkl", "wb"), protocol=4)
    np.savez(f"{OUT}/joint6.npz", ids=np.asarray(cat["ID"]), data_vectors=DV, cov_matrices=COV, n_k=NK, floored=FL,
             n_src=NSRC, radii=R6, bin_limits=BL6)
    json.dump({"cosmology": "FlatLambdaCDM H0=70 Om0=0.3", "centre": "X-ray peak", "background": BG, "radius_mpc": RADIUS,
               "sigma_e_population": _SIGE, "n_boot": N_BOOT, "floor_rmax": FLOOR_RMAX, "min_sources": MIN_SOURCES,
               "n_clusters": n, "n_measured": len(stakes)}, open(f"{OUT}/meta.json", "w"), indent=1)
    print(f"done: {len(stakes)}/{n} clusters measured ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
