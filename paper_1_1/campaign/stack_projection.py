"""Stacked analogue of dv_vs_truth.py: build the 3-way stacks of one simulated realisation exactly as the
stacked analysis does (stacked_3way_dm_test stages 0+2), compute K_c(R) as stack_weights.py, then compare
the stacked data with the stack model at every member's TRUE M and c and project the difference onto the
fit parameters (c0, alpha_cm, beta_cm, gamma_cm, log_M_norm):
    delta_theta = F^-1 sum_s J_s^T Cinv_s (d_s - m_s),   F = sum_s J_s^T Cinv_s J_s
with Cinv_s the Hartlap-corrected inverse bootstrap covariance of stacked_forward_fit.StackedForwardModel.
No per-stack NFW fits, no sampling.  (2026-09-29, gamma_cm bias study.)
usage: python stack_projection.py SIM_PREFIX SIM_CHECKPOINT_DIR OUT_DIR [N_WORKERS]
"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np

from paths import REPO  # noqa: E402
AUX = os.path.join(REPO, "scripts/auxiliary_scripts")
sys.path.insert(0, AUX)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


s3 = _load("stacked_3way_dm_test", os.path.join(AUX, "stacked_3way_dm_test.py"))
sff = _load("stacked_forward_fit", os.path.join(AUX, "stacked_forward_fit.py"))
wsa = s3.wsa
from destacking.model import _nfw_profiles, _MC_M_PIVOT_LOG, _MC_Z_REF, _MC_DM_REF

prefix, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
wsa.NWORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 32
os.makedirs(out, exist_ok=True)
for k, f in (("STACKS_PKL", "stacks3.pkl"), ("FITS_PKL", "fits3.pkl"), ("ML_PKL", "ml_fits3.pkl"),
             ("CMR_PKL", "cmr_fits3.pkl"), ("BOOST_PKL", "boost_fits3.pkl")):
    setattr(wsa, k, os.path.join(out, f))
t0 = time.time()
radii = np.logspace(-0.8, 0.6, 16); N = len(radii)
e = np.logspace(np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
                np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2, N + 1)
BL = np.array([[e[i], e[i + 1]] for i in range(N)])
if os.path.exists(os.path.join(out, "stacks3.pkl")) and os.path.exists(os.path.join(out, "weights.npz")):
    from astropy.table import Table
    catalog = Table.read(os.path.join(os.path.expanduser(f"{DATA}/sims"), f"{prefix}000_real_data.fits"))
    catalog = catalog[np.isfinite(catalog["DELTA_M_I"]) & (catalog["DELTA_M_I"] > 0)]
    d = pickle.load(open(os.path.join(out, "stacks3.pkl"), "rb"))
    stacks, z_bin, ng_bin, dm_bin = d["stacks"], d["z_bin"], d["ng_bin"], d["dm_bin"]
    w_ = np.load(os.path.join(out, "weights.npz")); weights = {"ids": w_["ids"], "K": w_["K"]}
    print("resumed from existing stacks3.pkl + weights.npz", flush=True)
else:
  catalog, bkg_by_id = s3.load_sim_catalog_and_bkgs(os.path.expanduser(f"{DATA}/sims"), prefix)
  print(f"loaded {len(catalog)} clusters ({time.time() - t0:.0f}s)", flush=True)
  stacks, z_bin, ng_bin, dm_bin = s3.build_stacks_3way(catalog, bkg_by_id)
  with open(os.path.join(out, "stacks3.pkl"), "wb") as fh:
      pickle.dump({"stacks": stacks, "z_bin": z_bin, "ng_bin": ng_bin, "dm_bin": dm_bin}, fh)
  print(f"stacks built ({time.time() - t0:.0f}s)", flush=True)

  # K_c(R) exactly as scripts/configs/stack_weights.py
  ids = np.array(sorted(bkg_by_id), dtype=np.int64)
  K = np.zeros((len(ids), N))
  for n, cid in enumerate(ids):
      sig, et, ex, w, R, M = bkg_by_id[int(cid)][:6]
      ib = (R[:, None] > BL[:, 0]) & (R[:, None] < BL[:, 1])
      K[n] = np.dot((1 + M) * w / sig ** 2, ib)
  weights = {"ids": ids, "K": K}
  np.savez(os.path.join(out, "weights.npz"), **weights)
  del bkg_by_id

fm = sff.StackedForwardModel(catalog, {"stacks": stacks, "z_bin": z_bin}, weights, min_n=50,
                             two_halo=False, bin_nodes=4)
assert np.allclose(fm.radii, radii)
tc = np.load(os.path.join(ckpt, "truth_catalog.npz"))
ols = json.load(open(os.path.join(ckpt, "ols.json")))
rows = np.asarray(catalog["ID"])[fm.catalog_rows].astype(int)          # ID == truth row
z, Mt, ct = tc["z"][rows], tc["M_halo"][rows], tc["concentration"][rows]
assert np.allclose(z, fm.z)
C0, ALPHA = ols["c0_true"], ols["alpha_true"]
LN10 = np.log(10)
prof = lambda MM, cc: _nfw_profiles(z, MM, cc, radii, bin_nodes=4)
h = 1e-3
m = prof(Mt, ct)
dm_dlnc = (prof(Mt, ct * np.exp(h)) - prof(Mt, ct * np.exp(-h))) / (2 * h)
dm_dlnM = (prof(Mt * np.exp(h), ct) - prof(Mt * np.exp(-h), ct)) / (2 * h)
dlnc = [np.full_like(z, 1 / C0), LN10 * (np.log10(Mt) - _MC_M_PIVOT_LOG),
        LN10 * np.log10((1 + z) / (1 + _MC_Z_REF)), LN10 * fm.log_dm]
Jc = [dm_dlnc * x[:, None] for x in dlnc] + [dm_dlnM * LN10 + dm_dlnc * ALPHA * LN10]
stackify = lambda a: (fm.S @ (fm.K * a)) / fm.norm
m_s = stackify(m); J_s = [stackify(j) for j in Jc]
PAR = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm"]
F = np.zeros((5, 5)); Sv = np.zeros(5); per_stack = []
for k in range(len(fm.names)):
    msk = fm.masks[k]; r = fm.data[k] - m_s[k][msk]
    J = np.stack([j[k][msk] for j in J_s], 1)
    F += J.T @ fm.cinv[k] @ J; s = J.T @ fm.cinv[k] @ r; Sv += s; per_stack.append(s)
Finv = np.linalg.inv(F); dtheta = Finv @ Sv
dg_stack = np.array([Finv[3] @ s for s in per_stack])
dm_of = np.array([stacks[nm]["dm_bin"] for nm in fm.names])
print(f"\n{prefix}: {len(fm.names)} stacks, {fm.S.shape[1]} member clusters")
print("  predicted shifts:", "  ".join(f"{p}={v:+.4f}" for p, v in zip(PAR, dtheta)),
      f"  [sigma(gamma) Fisher = {np.sqrt(Finv[3, 3]):.4f}]")
print("  gamma shift by dm tercile of the stacks:", " ".join(f"{dg_stack[dm_of == t].sum():+.4f}" for t in range(3)))
np.savez(os.path.join(out, "projection.npz"), dtheta=dtheta, Finv=Finv, dg_stack=dg_stack, dm_of=dm_of,
         names=np.array(fm.names))
print(f"done ({time.time() - t0:.0f}s)")
