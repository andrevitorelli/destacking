"""Simulated data vectors vs synthetic (noise-free) profiles, and which data drive gamma_cm.

For each dataset (checkpoint dir), compares d_i (stored data vectors) with m_i, the fit's own model at
each cluster's TRUE M200m and c200m (area-weighted bin average, as the measurement), and computes the
first-order parameter shift of the 5-parameter joint fit (c0, alpha_cm, beta_cm, gamma_cm, log_M_norm):
    S = sum_i J_i^T C_i^-1 (d_i - m_i),  F = sum_i J_i^T C_i^-1 J_i,  dtheta = F^-1 S
decomposed exactly by radial bin and cluster group.
usage: python dv_vs_truth.py NAME=CHECKPOINT_DIR [NAME=DIR ...]
"""
import sys
import numpy as np
from destacking.model import _nfw_profiles, _MC_M_PIVOT_LOG, _MC_Z_REF, _MC_DM_REF

import os
_g = [float(x) for x in os.environ.get("DVT_GRID", "-0.8,0.6,8").split(",")]
R = np.logspace(_g[0], _g[1], int(_g[2]))
LN10 = np.log(10)
PAR = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm"]
C0, ALPHA = 4.559, -0.0922           # truth values at the fit's pivots (ols.json)


def load(d):
    s = np.load(f"{d}/simulated_data.npz"); t = np.load(f"{d}/truth_catalog.npz")
    return s["data_vectors"], s["cov_matrices"], t


def analyse(name, d):
    dv, cov, t = load(d)
    z, M, c, dm = t["z"], t["M_halo"], t["concentration"], t["delta_m"]
    prof = lambda MM, cc: _nfw_profiles(z, MM, cc, R, bin_nodes=4)
    m = prof(M, c)
    h = 1e-3
    dm_dlnc = (prof(M, c * np.exp(h)) - prof(M, c * np.exp(-h))) / (2 * h)
    dm_dlnM = (prof(M * np.exp(h), c) - prof(M * np.exp(-h), c)) / (2 * h)
    ldm = np.where(np.isfinite(dm) & (dm > 0), np.log10(dm / _MC_DM_REF), 0.0)
    dlnc = np.stack([np.full_like(z, 1 / C0), LN10 * (np.log10(M) - _MC_M_PIVOT_LOG),
                     LN10 * np.log10((1 + z) / (1 + _MC_Z_REF)), LN10 * ldm], 1)        # (N, 4)
    J = dm_dlnc[:, :, None] * dlnc[:, None, :]                                            # (N, 8, 4)
    JM = (dm_dlnM * LN10 + dm_dlnc * ALPHA * LN10)[:, :, None]                            # log_M_norm
    J = np.concatenate([J, JM], 2)                                                        # (N, 8, 5)
    valid = np.isfinite(dv)
    r = np.where(valid, dv - m, 0.0)
    J = np.where(valid[:, :, None], J, 0.0)
    Cinv_r = np.linalg.solve(cov, r[:, :, None])[:, :, 0]                                 # (N, 8)
    Cinv_J = np.linalg.solve(cov, J)                                                      # (N, 8, 5)
    Cinv_r = np.where(valid, Cinv_r, 0.0); Cinv_J = np.where(valid[:, :, None], Cinv_J, 0.0)
    F = np.einsum("nkp,nkq->pq", J, Cinv_J)
    Finv = np.linalg.inv(F)
    S_nk = J * Cinv_r[:, :, None]                                                         # (N, 8, 5) score contributions
    dtheta = Finv @ S_nk.sum((0, 1))
    g = PAR.index("gamma_cm")
    dg_nk = np.einsum("p,nkp->nk", Finv[g], S_nk)                                         # gamma shift per cluster & bin
    sd = np.sqrt(np.einsum("nii->ni", cov)); pull = np.where(valid, (dv - m) / sd, np.nan)
    print(f"\n=== {name}: {len(z)} clusters, {valid.sum()} valid bins")
    print("  predicted shifts (first order, vs truth):", "  ".join(f"{p}={v:+.4f}" for p, v in zip(PAR, dtheta)),
          f"   [sigma(gamma) Fisher = {np.sqrt(Finv[g, g]):.4f}]")
    print("  R [Mpc]          " + " ".join(f"{x:8.2f}" for x in R))
    print("  gamma shift/bin  " + " ".join(f"{x:+8.4f}" for x in dg_nk.sum(0)))
    print("  pull mean        " + " ".join(f"{np.nanmean(pull[:, k]):+8.3f}" for k in range(len(R))))
    print("  pull std         " + " ".join(f"{np.nanstd(pull[:, k]):8.3f}" for k in range(len(R))))
    print("  valid fraction   " + " ".join(f"{valid[:, k].mean():8.3f}" for k in range(len(R))))
    groups = {"dm": ldm, "z": z, "logM": np.log10(M)}
    for gname, x in groups.items():
        q = np.quantile(x, [0, 1 / 3, 2 / 3, 1])
        for k3, lab in enumerate(["low", "mid", "high"]):
            sel = (x >= q[k3]) & (x <= q[k3 + 1])
            print(f"  {gname:4s} {lab:4s} gamma shift: {dg_nk[sel].sum():+.4f}   per bin: "
                  + " ".join(f"{v:+.4f}" for v in dg_nk[sel].sum(0)))
    return dtheta, dg_nk


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, d = arg.split("=", 1)
        analyse(name, d)


def piece_sigma(name, d):
    """gamma-shift contribution of (Delta-m tercile x bin) cells, with its expected noise sigma
    sqrt(Finv_g^T F_cell Finv_g) for Gaussian noise with the stored covariances, and mean pulls."""
    dv, cov, t = load(d)
    z, M, c, dm = t["z"], t["M_halo"], t["concentration"], t["delta_m"]
    prof = lambda MM, cc: _nfw_profiles(z, MM, cc, R, bin_nodes=4)
    m = prof(M, c); h = 1e-3
    dm_dlnc = (prof(M, c * np.exp(h)) - prof(M, c * np.exp(-h))) / (2 * h)
    dm_dlnM = (prof(M * np.exp(h), c) - prof(M * np.exp(-h), c)) / (2 * h)
    ldm = np.where(np.isfinite(dm) & (dm > 0), np.log10(dm / _MC_DM_REF), 0.0)
    dlnc = np.stack([np.full_like(z, 1 / C0), LN10 * (np.log10(M) - _MC_M_PIVOT_LOG),
                     LN10 * np.log10((1 + z) / (1 + _MC_Z_REF)), LN10 * ldm], 1)
    J = np.concatenate([dm_dlnc[:, :, None] * dlnc[:, None, :], (dm_dlnM * LN10 + dm_dlnc * ALPHA * LN10)[:, :, None]], 2)
    valid = np.isfinite(dv); r = np.where(valid, dv - m, 0.0); J = np.where(valid[:, :, None], J, 0.0)
    Cinv_r = np.where(valid, np.linalg.solve(cov, r[:, :, None])[:, :, 0], 0.0)
    Cinv_J = np.where(valid[:, :, None], np.linalg.solve(cov, J), 0.0)
    Finv = np.linalg.inv(np.einsum("nkp,nkq->pq", J, Cinv_J)); fg = Finv[3]
    a = np.einsum("p,nkp->nk", fg, J)          # per-cluster, per-bin projection of the gamma row
    dg = a * Cinv_r                            # gamma-shift contribution
    # noise variance of sum over a cell of a_k (C^-1 r)_k: a^T C^-1 a restricted to the cell's bins
    Cinv_a = np.where(valid, np.linalg.solve(cov, np.where(valid, a, 0.0)[:, :, None])[:, :, 0], 0.0)
    sd = np.sqrt(np.einsum("nii->ni", cov)); pull = np.where(valid, (dv - m) / sd, np.nan)
    q = np.quantile(ldm, [0, 1 / 3, 2 / 3, 1])
    print(f"\n=== {name}: gamma contribution [sigma]  (mean pull) by Delta-m tercile x bin")
    print("            " + " ".join(f"{x:>17.2f}" for x in R))
    for k3, lab in enumerate(["low", "mid", "high"]):
        sel = (ldm >= q[k3]) & (ldm <= q[k3 + 1])
        cells = []
        for k in range(8):
            v = dg[sel, k].sum(); s = np.sqrt(np.sum(a[sel, k] * Cinv_a[sel, k]))
            cells.append(f"{v:+.4f}[{v / s:+.1f}]({np.nanmean(pull[sel, k]):+.3f})")
        print(f"  {lab:4s} dm  " + " ".join(f"{c_:>17s}" for c_ in cells))
    sel = ldm >= q[2]
    v = dg[sel, :2].sum(); s = np.sqrt(np.sum((a * Cinv_a)[sel, :2]))
    print(f"  high-dm, 0.16+0.25 Mpc: {v:+.4f} = {v / s:+.2f} sigma (sigma {s:.4f})")
