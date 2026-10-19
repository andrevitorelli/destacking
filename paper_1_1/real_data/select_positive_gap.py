"""Copy a prepare_real_data.py data folder keeping only clusters with a positive magnitude gap.

The paper excludes the clusters with Delta m <= 0 (142 of 15,967).  This writes
SRC_dmpos/ next to SRC/ with every per-cluster array of simulated_data.npz,
truth_catalog.npz and cluster_obs.npz restricted to delta_m > 0.

usage: python select_positive_gap.py SRC_DATA_FOLDER
"""
import glob
import os
import shutil
import sys

import numpy as np


def main():
    src = os.path.abspath(sys.argv[1].rstrip("/"))
    dst = src + "_dmpos"
    for sub in sorted(glob.glob(os.path.join(src, "simulation_*"))):
        out = os.path.join(dst, os.path.basename(sub)); os.makedirs(out, exist_ok=True)
        obs = np.load(os.path.join(sub, "cluster_obs.npz"), allow_pickle=True)
        keep = np.asarray(obs["delta_m"], float) > 0
        n = len(keep)
        for name in ("simulated_data.npz", "truth_catalog.npz", "cluster_obs.npz"):
            z = np.load(os.path.join(sub, name), allow_pickle=True)
            arrs = {k: (z[k][keep] if z[k].ndim and len(z[k]) == n else z[k]) for k in z.files}
            np.savez(os.path.join(out, name), **arrs)
        for name in os.listdir(sub):
            if name.endswith((".json", ".yaml")):
                shutil.copy(os.path.join(sub, name), out)
        print(f"{sub}: kept {keep.sum()} of {n} clusters -> {out}")


if __name__ == "__main__":
    main()
