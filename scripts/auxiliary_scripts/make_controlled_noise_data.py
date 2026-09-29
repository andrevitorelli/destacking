"""Controlled-noise copies of a simulation checkpoint, to localise a gamma_cm bias (2026-09-29).

Takes an existing simulated checkpoint (simulation_gXX_*/ with simulated_data.npz, truth_catalog.npz,
cluster_obs.npz, ols.json) and writes a copy in which ONLY the per-cluster data vectors are replaced:

  noiseless : d_i = m_i, the true model profile of cluster i -- the fit's own model
              (destacking.model._nfw_profiles, centred, area-weighted bin average) evaluated at the
              cluster's TRUE M200m and c200m.
  gaussian  : d_i = m_i + L_i n_i, with L_i the Cholesky factor of cluster i's own stored covariance
              (restricted to its valid bins) and n_i ~ N(0, 1): noise that is statistically
              independent of the covariance estimate, unlike the simulated galaxy-level measurement.

Everything else is kept bit-for-bit: truth, observables, covariances, and the NaN mask of failed bins
(a bin missing in the original stays missing, so the likelihood sees the same bins).

Reading the refit:
  noiseless biased            -> the likelihood/masking/weighting (or the model) is biased without noise
  noiseless ok, gaussian ok,
    original simulation biased -> the bias comes from how the measured noise enters (e.g. the
                                  correlation between each cluster's data and its bootstrap covariance)
  gaussian biased             -> noise-driven bias of the likelihood itself (e.g. Jensen / low-S/N)

Only centred (miscentering: false) sims are supported: with miscentring the true profile depends on each
cluster's realised offset, which is not stored.

usage:
  python scripts/auxiliary_scripts/make_controlled_noise_data.py SRC_SIM_SUBDIR DST_SIM_SUBDIR \
         --mode noiseless|gaussian [--seed 1] [--bin-nodes 4] [--log-inner -0.8 --log-outer 0.6 --n-radii 8]
"""
import argparse
import json
import os
import shutil

import numpy as np

from destacking.model import _nfw_profiles

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--mode", choices=["noiseless", "gaussian"], required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--bin-nodes", type=int, default=4)
ap.add_argument("--log-inner", type=float, default=-0.8)
ap.add_argument("--log-outer", type=float, default=0.6)
ap.add_argument("--n-radii", type=int, default=8)
args = ap.parse_args()

radii = np.logspace(args.log_inner, args.log_outer, args.n_radii)
sim = np.load(os.path.join(args.src, "simulated_data.npz"))
dv, cov = sim["data_vectors"], sim["cov_matrices"]
tc = np.load(os.path.join(args.src, "truth_catalog.npz"))
assert dv.shape == (len(tc["z"]), len(radii)), (dv.shape, len(radii))

model = _nfw_profiles(tc["z"], tc["M_halo"], tc["concentration"], radii, bin_nodes=args.bin_nodes)
valid = np.isfinite(dv)
new = np.full_like(dv, np.nan)
if args.mode == "noiseless":
    new[valid] = model[valid]
else:
    rng = np.random.default_rng(args.seed)
    for i in range(len(dv)):
        v = valid[i]
        if not v.any():
            continue
        L = np.linalg.cholesky(cov[i][np.ix_(v, v)])
        new[i, v] = model[i, v] + L @ rng.standard_normal(v.sum())

os.makedirs(args.dst, exist_ok=True)
for f in ("truth_catalog.npz", "cluster_obs.npz"):
    shutil.copy2(os.path.join(args.src, f), os.path.join(args.dst, f))
with open(os.path.join(args.src, "ols.json")) as f:
    ols = json.load(f)
ols["controlled_noise"] = {"source": os.path.abspath(args.src), "mode": args.mode,
                           "seed": args.seed if args.mode == "gaussian" else None, "bin_nodes": args.bin_nodes}
with open(os.path.join(args.dst, "ols.json"), "w") as f:
    json.dump(ols, f, indent=2)
np.savez_compressed(os.path.join(args.dst, "simulated_data.npz"), data_vectors=new, cov_matrices=cov)

# summary: how far the original measurement sits from the truth, bin by bin (inverse-variance mean)
sd = np.sqrt(np.einsum("nii->ni", cov))
for name, d in (("original", dv), (args.mode, new)):
    r = np.where(valid, d / model, np.nan)
    w = np.where(valid, (model / sd) ** 2, 0.0)
    print(f"{name:10s} ivar-mean data/model per bin: "
          + " ".join(f"{np.nansum(w[:, k] * r[:, k]) / w[:, k].sum():.4f}" for k in range(len(radii))))
print(f"wrote {args.dst}  ({valid.sum()} valid bins in {len(dv)} clusters)")
