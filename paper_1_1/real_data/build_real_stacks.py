"""Build the 3-way real-data stacks and the member weights of the forward-modelled stacked fit.

Bins the WaZP clusters with Delta m > 0 into 3 redshift x 4 richness cells and magnitude-gap
terciles (stacked_3way_dm_test.build_stacks_3way), measures each stack with
xlensing.data.stacked_signal (500 cluster bootstraps, 16 radial bins 0.16-3.98 Mpc,
boost-corrected), and stores each member's lensing weight K_c(R) (Eq. 14 of the paper).

Input: the per-cluster background arrays written by wazp_stacked_analysis.py stage 1 for the
same selection factor (real_data/wazp_stacked_bkgs.pkl for f = 1.1, ..._f1.2.pkl for f = 1.2).

usage: python build_real_stacks.py FACTOR
  FACTOR = 1.2 (paper's stacked analysis) -> real_data/stacked_full_param_fit_f1.2/stacks3.pkl
                                              real_data/stack_weights_f1.2.npz
  FACTOR = 1.1 (joint fit's selection)    -> real_data/stacked_f1.1_3way/{stacks3.pkl, weights.npz}
"""
import os as _os, sys as _sys
_sys.path[:0] = [_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))]
from paths import REAL, AUX  # noqa: E402
import importlib.util
import os
import pickle
import sys
import time

import numpy as np
from astropy.table import Table

sys.path.insert(0, AUX)
_spec = importlib.util.spec_from_file_location("s3", os.path.join(AUX, "stacked_3way_dm_test.py"))
s3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s3)
wsa = s3.wsa


def main():
    factor = float(sys.argv[1])
    if factor == 1.2:
        bkg_file = f"{REAL}/wazp_stacked_bkgs_f1.2.pkl"
        out_dir = f"{REAL}/stacked_full_param_fit_f1.2"; weights_file = f"{REAL}/stack_weights_f1.2.npz"
    elif factor == 1.1:
        bkg_file = f"{REAL}/wazp_stacked_bkgs.pkl"
        out_dir = f"{REAL}/stacked_f1.1_3way"; weights_file = f"{out_dir}/weights.npz"
    else:
        raise SystemExit("FACTOR must be 1.1 or 1.2")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    cat = Table.read(f"{REAL}/wazp_real_data.fits")
    cat = cat[np.isfinite(cat["DELTA_M_I"]) & (cat["DELTA_M_I"] > 0)]
    bk = pickle.load(open(bkg_file, "rb"))
    bkg_by_id = bk["bkg_by_id"] if isinstance(bk, dict) and "bkg_by_id" in bk else bk
    print(f"{len(cat)} clusters, {len(bkg_by_id)} backgrounds ({time.time() - t0:.0f}s)", flush=True)
    stacks, z_bin, ng_bin, dm_bin = s3.build_stacks_3way(cat, bkg_by_id)
    pickle.dump({"stacks": stacks, "z_bin": z_bin, "ng_bin": ng_bin, "dm_bin": dm_bin},
                open(f"{out_dir}/stacks3.pkl", "wb"))
    print(f"stacks built ({time.time() - t0:.0f}s)", flush=True)
    radii = np.asarray(wsa.radii); BL = np.asarray(wsa.BINS_LIMS)
    ids = np.array(sorted(int(i) for i in bkg_by_id), dtype=np.int64); K = np.zeros((len(ids), len(radii)))
    for n, cid in enumerate(ids):
        sig, et, ex, w, R, M = bkg_by_id[int(cid)][:6]
        ib = (R[:, None] > BL[:, 0]) & (R[:, None] < BL[:, 1])
        K[n] = np.dot((1 + M) * w / sig ** 2, ib)
    np.savez(weights_file, ids=ids, K=K)
    print(f"wrote {out_dir}/stacks3.pkl and {weights_file} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
