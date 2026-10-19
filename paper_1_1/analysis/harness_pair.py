"""Paired (e -> -e) harness: is the simulated inner-bin data vector biased, and do bootstrap weights
correlate with the data?  (2026-09-29)

N identical spherical, centred haloes go through the simulator's own per-cluster pipeline
(destacking.simulation.simulate_lensing: Smail n(z), shape noise, lognormal metacal responses, exact
shear composition, bootstrap_signal).  Every configuration is run twice with identical positions,
redshifts and responses (same seeds), the second time with each intrinsic ellipticity negated, by
wrapping xlensing.sims.add_shears.  Pair averages cancel shape noise to first order.

usage: python harness_pair.py LOGM C Z N OUT.npz
"""
import sys
import time
import warnings

import numpy as np
from astropy.table import Table

warnings.filterwarnings("ignore")
import xlensing.sims as xs
from xlensing.data import signal
from destacking.simulation import SimulationConfig, simulate_lensing
from destacking.model import _nfw_profiles

lM, c, z, N, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
# optional survey settings: density beta z0 log_inner log_outer n_bins (default LSST-like)
DENS, BETA, Z0, LIN, LOUT, NB = (float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]),
                                 float(sys.argv[10]), int(sys.argv[11])) if len(sys.argv) > 6 else (30.0, 0.68, 0.11, -0.8, 0.6, 8)
R = np.logspace(LIN, LOUT, NB); lr = np.log10(R); d = np.diff(lr)[0]
BL = np.column_stack([10 ** (lr - d / 2), 10 ** (lr + d / 2)])
SIG_E = 0.27
cfg = SimulationConfig(n_clusters=N, z_cluster_range=(0.1, 0.6), logm_range=(14, 15), lambda_min=25, delta_m_ref=0.1,
                       density=DENS, shape_noise=SIG_E, width_rad=5.0, z_src_range=(0.0, 2.5),
                       z_src_alpha=2.0, z_src_beta=BETA, z_src_z0=Z0, spherical=True, miscentering=False,
                       f_cen=1.0, sigma_off_mpc=0.47, radii=R, bin_limits=BL, n_cov_bootstrap=200, n_workers=14,
                       floor_rmax=0.5)
M = 10 ** lM
truth = Table({"z": np.full(N, z), "M_halo": np.full(N, M), "concentration": np.full(N, c), "delta_m": np.full(N, 0.5)})
shapes = {"q_ba": np.ones(N), "q_ca": np.ones(N), "theta_los": np.zeros(N), "phi_los": np.zeros(N), "r_vir": np.full(N, 1.0)}
mis = (np.zeros(N, bool), np.zeros(N), np.zeros(N))
model = _nfw_profiles(np.array([z]), np.array([M]), np.array([c]), R, bin_nodes=4)[0]

_orig = xs.add_shears
MODES = {
    "exact":      lambda e, g: _orig(e, g),
    "exact_flip": lambda e, g: _orig(-e, g),
}
res = {"R": R, "model": model, "logM": lM, "c": c, "z": z, "N": N}
for mode, fn in MODES.items():
    xs.add_shears = fn
    t0 = time.time()
    dv, cov, bgs = simulate_lensing(truth, shapes, mis, 7, cfg)
    direct = np.full_like(dv, np.nan); var_an = np.full_like(dv, np.nan); var_rot = np.full_like(dv, np.nan)
    for i, st in enumerate(bgs):
        if st is None:
            continue
        st = np.asarray(st, dtype=float)
        direct[i] = signal(st, BL)[0]
        sig, w, Rg, Mb = st[0], st[3], st[4], st[5]
        inb = (Rg[:, None] > BL[:, 0]) & (Rg[:, None] < BL[:, 1])
        den = ((1 + Mb) * w / sig ** 2) @ inb
        var_an[i] = SIG_E ** 2 * ((w / sig) ** 2 @ inb) / den ** 2      # analytic shape-noise variance
        et, ex = st[1], st[2]
        var_rot[i] = (((w / sig) ** 2 * (et ** 2 + ex ** 2) / 2) @ inb) / den ** 2   # random-rotation variance
    res[mode + "_dv"] = dv
    res[mode + "_var"] = np.einsum("nii->ni", cov)
    res[mode + "_direct"] = direct
    res[mode + "_var_an"] = var_an
    res[mode + "_var_rot"] = var_rot
    print(f"{mode:12s} done in {time.time() - t0:.0f}s", flush=True)
xs.add_shears = _orig
np.savez(out, **res)
print("saved", out)
