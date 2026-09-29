"""Rebuild a simulation checkpoint's per-cluster data vectors and covariances from its saved
background galaxies, with the direct estimator and the analytic shape-noise covariance
(destacking.data.analytic_signal) instead of the galaxy-bootstrap mean/covariance (2026-09-29).

Reads SRC_SIM_DIR (simulated_data.npz, truth_catalog.npz, cluster_obs.npz, ols.json) and the
notebook-compatible background file BGS_PKL ({cluster ID: stake}, ID = checkpoint row), and writes
DST_SIM_DIR with the same truth/observables and new simulated_data.npz.  sigma_e is the population
dispersion estimated from the cross shear of every source in the file (destacking.data.population_sigma_e)
unless --sigma-e is given.  Clusters without saved backgrounds keep an all-NaN data vector.

usage: python scripts/auxiliary_scripts/remeasure_checkpoint.py SRC_SIM_DIR BGS_PKL DST_SIM_DIR
           [--log-inner -0.8 --log-outer 0.6 --n-radii 8] [--sigma-e X] [--min-gal 3]
"""
import argparse
import json
import os
import pickle
import shutil
import time

import numpy as np

from destacking.data import analytic_signal, population_sigma_e

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("bgs"); ap.add_argument("dst")
ap.add_argument("--log-inner", type=float, default=-0.8)
ap.add_argument("--log-outer", type=float, default=0.6)
ap.add_argument("--n-radii", type=int, default=8)
ap.add_argument("--sigma-e", type=float, default=None)
ap.add_argument("--min-gal", type=int, default=3)
a = ap.parse_args()

t0 = time.time()
radii = np.logspace(a.log_inner, a.log_outer, a.n_radii)
lr = np.log10(radii); d = (lr[1] - lr[0]) / 2
bin_limits = np.column_stack([10 ** (lr - d), 10 ** (lr + d)])
old = np.load(os.path.join(a.src, "simulated_data.npz"))
n_cl, n_b = old["data_vectors"].shape
assert n_b == a.n_radii, f"checkpoint has {n_b} bins, grid has {a.n_radii}"
with open(a.bgs, "rb") as fh:
    bgs = pickle.load(fh)
print(f"loaded {len(bgs)} background sets ({time.time() - t0:.0f}s)", flush=True)
sigma_e = a.sigma_e if a.sigma_e is not None else population_sigma_e(bgs.values())
print(f"sigma_e = {sigma_e:.4f} ({'given' if a.sigma_e is not None else 'population, from cross shear'})", flush=True)

dv = np.full((n_cl, n_b), np.nan)
cov = np.zeros((n_cl, n_b, n_b))
cov[:, np.arange(n_b), np.arange(n_b)] = 1.0
for cid, st in bgs.items():
    i = int(cid)
    if st is None or not (0 <= i < n_cl):
        continue
    dv[i], cov[i] = analytic_signal(st, bin_limits, sigma_e, min_gal=a.min_gal)
del bgs

os.makedirs(a.dst, exist_ok=True)
for f in ("truth_catalog.npz", "cluster_obs.npz"):
    shutil.copy2(os.path.join(a.src, f), os.path.join(a.dst, f))
with open(os.path.join(a.src, "ols.json")) as fh:
    ols = json.load(fh)
ols["remeasured"] = {"source": os.path.abspath(a.src), "bgs": os.path.abspath(a.bgs),
                     "estimator": "direct", "covariance": "analytic shape noise (diagonal)",
                     "sigma_e": sigma_e, "min_gal": a.min_gal, "date": "2026-09-29"}
with open(os.path.join(a.dst, "ols.json"), "w") as fh:
    json.dump(ols, fh, indent=2)
np.savez_compressed(os.path.join(a.dst, "simulated_data.npz"), data_vectors=dv, cov_matrices=cov)
v_old = np.isfinite(old["data_vectors"]).mean(0); v_new = np.isfinite(dv).mean(0)
print("valid fraction per bin  old: " + " ".join(f"{x:.3f}" for x in v_old))
print("                        new: " + " ".join(f"{x:.3f}" for x in v_new))
print(f"wrote {a.dst} ({time.time() - t0:.0f}s)")
