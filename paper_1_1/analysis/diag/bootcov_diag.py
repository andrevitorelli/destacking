"""2026-10-11: why do some DES-like bootstrap-covariance joint fits crash?  Compare bootstrap vs analytic per-cluster variances
(sims: {TL}_sim vs {TL}_sim_analytic; real: bootfloor_v4 vs analytic_v3), bin by bin.  No model needed.
  ratio      = bootstrap var / analytic var (same cluster, same bin)
  maxshare   = largest single-cluster share of sum(1/var) in the bin (1/var weights)
  maxsnr2    = largest dv^2/var in the bin (one cluster's chi^2 weight against zero signal)"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import sys, glob, numpy as np
O = f"{DATA}/output"; RD = f"{DATA}/real_data"
def load(d):
    a = np.load(glob.glob(f"{d}/simulation_g00*/simulated_data.npz")[0]); dv = a["data_vectors"]; v = np.einsum("nii->ni", a["cov_matrices"])
    return dv, v
def stats(tag, db, da):
    (b, vb), (a, va) = load(db), load(da)
    assert b.shape == a.shape, (tag, b.shape, a.shape)
    ok = np.isfinite(b) & np.isfinite(vb) & (vb > 0) & np.isfinite(va) & (va > 0) & np.isfinite(a)
    r = np.where(ok, vb / np.where(ok, va, 1), np.nan)
    out = []
    for k in range(b.shape[1]):
        m = ok[:, k]; rk = r[m, k]; wb = 1 / vb[m, k]; wa = 1 / va[m, k]
        out.append((m.mean(), np.median(rk), np.mean(rk < 0.5), np.mean(rk < 0.2), rk.min(), wb.max() / wb.sum(), wa.max() / wa.sum(),
                    (b[m, k] ** 2 / vb[m, k]).max(), (a[m, k] ** 2 / va[m, k]).max()))
    print(f"\n{tag}  N={len(b)}")
    print("  bin valid  med(ratio) f<0.5  f<0.2  min(ratio)  maxshare boot/an      maxsnr2 boot/an")
    for k, s in enumerate(out):
        print(f"  {k}   {s[0]:.2f}   {s[1]:5.2f}    {s[2]:.3f}  {s[3]:.3f}  {s[4]:.2e}   {s[5]:.4f} / {s[6]:.4f}   {s[7]:9.1f} / {s[8]:7.1f}")
for tl in sys.argv[1:]:
    if tl == "REAL": stats("REAL (bootfloor_v4 vs analytic_v3)", f"{RD}/wazp_real_data_bootfloor_v4", f"{RD}/wazp_real_data_analytic_v3")
    else: stats(tl, f"{O}/{tl}_sim", f"{O}/{tl}_sim_analytic")
