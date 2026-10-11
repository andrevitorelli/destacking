"""Per-cluster, per-bin source counts and floor flags of a simulation's galaxy-bootstrap checkpoint, for the per-bin
Sellentin & Heavens (2016) likelihood (likelihood: sellentin_heavens in gamma_inference.py), mirroring the real-data
bin_counts.npz (2026-10-11).  A bin is 'floored' when its lower edge is below floor_rmax and its bootstrap variance equals the
shape-noise floor sigma_e^2 / sum((1+M) W / Sigma_crit^2) of xlensing.data.bootstrap_signal; those bins keep the Gaussian.
usage: python sim_bin_counts.py SIM_DIR BGS_PKL --log-inner -0.6 --log-outer 0.6 --n-radii 6 --sigma-e 0.27 --floor-rmax 0.5
writes SIM_DIR/bin_counts.npz (n_k, floored, bin_limits)"""
import argparse, os, pickle, time, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("sim"); ap.add_argument("bgs")
ap.add_argument("--log-inner", type=float, required=True); ap.add_argument("--log-outer", type=float, required=True)
ap.add_argument("--n-radii", type=int, required=True); ap.add_argument("--sigma-e", type=float, required=True)
ap.add_argument("--floor-rmax", type=float, default=0.5)
a = ap.parse_args(); t0 = time.time()
lr = np.log10(np.logspace(a.log_inner, a.log_outer, a.n_radii)); d = (lr[1] - lr[0]) / 2
BL = np.column_stack([10 ** (lr - d), 10 ** (lr + d)])
var = np.einsum("nii->ni", np.load(os.path.join(a.sim, "simulated_data.npz"))["cov_matrices"])
n_cl, n_b = var.shape; assert n_b == a.n_radii
bgs = pickle.load(open(a.bgs, "rb"))
from destacking.data import bin_counts_from_backgrounds
nk, floored = bin_counts_from_backgrounds(bgs, BL, var, a.sigma_e, a.floor_rmax)
np.savez(os.path.join(a.sim, "bin_counts.npz"), n_k=nk, floored=floored, bin_limits=BL)
ok = np.isfinite(var)
print(f"{a.sim}: floored fraction of valid bins " + " ".join(f"{floored[ok[:, k], k].mean():.3f}" for k in range(n_b)),
      "| median n_k " + " ".join(f"{np.median(nk[ok[:, k], k]):.0f}" for k in range(n_b)),
      "| valid bins with n_k<3: " + str(int(np.sum(ok & (nk < 3)))), f"({time.time()-t0:.0f}s)", flush=True)
