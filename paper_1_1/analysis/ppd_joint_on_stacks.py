"""Posterior predictive of the joint fit for the 33 real stacks (2026-10-07)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import sys, json, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
from stack_variants import build, BASE
from bootmed import boot_medians, wq
M = build()
R = M.radii
names, meds = boot_medians(sorted(glob.glob(f"{DATA}/output/v1.1/posteriors/joint_sh_b[0-9][0-9][0-9].npz")), "samples")
st = np.load(f"{DATA}/output/v1.1/stacked/real_minN50.npz", allow_pickle=True); ns = [str(p) for p in st["param_order"]]
w = st["weights"] / st["weights"].sum()
pst = {p: float(wq(st["samples"][:, j], w, 0.5)) for j, p in enumerate(ns)}
pj = {p: float(np.median(meds[:, j])) for j, p in enumerate(names)}
def chi2_parts(p):
    mod = M.model(p); out = []
    for k, nm in enumerate(M.names):
        m = M.masks[k]; r = M.data[k] - mod[k][m]
        L = np.linalg.cholesky(M.cinv[k]); z = L.T @ r          # whitened residuals, sum z^2 = chi2
        out.append((nm, R[m], r, z, float(r @ M.cinv[k] @ r)))
    return out
res = {}
for lab, p in (("stacked", pst), ("joint", pj)):
    parts = chi2_parts(p); res[lab] = parts
    tot = sum(x[4] for x in parts); npts = sum(len(x[2]) for x in parts)
    by_t = {t: sum(x[4] for x in parts if x[0].endswith(f"dm{t}")) for t in range(3)}
    print(f"{lab:8s} chi2={tot:.1f} for {npts} points; by tercile low/mid/high = " + ", ".join(f"{by_t[t]:.1f}" for t in range(3)))
# chi2 difference split by radius and tercile, with the residual-sign pattern
d = {}
for (nm, Rk, r_s, z_s, c_s), (_, _, r_j, z_j, c_j) in zip(res["stacked"], res["joint"]):
    t = int(nm.split("_dm")[1])
    for rr in (("R<0.25", Rk < 0.25), ("0.25-1", (Rk >= 0.25) & (Rk < 1)), ("R>1", Rk >= 1)):
        pass
# per-bin decomposition with the full covariance is not additive; use the whitened-residual chi2 difference per stack, and the
# inverse-variance-weighted mean fractional residual (data/model - 1) per tercile and radial range for the joint parameters
modj = M.model(pj); mods = M.model(pst)
for lab, mod in (("joint", modj), ("stacked", mods)):
    print(lab, "mean fractional residual (data/model-1), inverse-variance weighted:")
    for t in range(3):
        row = []
        for rng_lab, lo, hi in (("R<0.25", 0, 0.25), ("0.25-1", 0.25, 1.0), ("R>1", 1.0, 99)):
            num = den = 0.0
            for k, nm in enumerate(M.names):
                if not nm.endswith(f"dm{t}"): continue
                m = M.masks[k]; Rk = R[m]; sel = (Rk >= lo) & (Rk < hi)
                if not sel.any(): continue
                C = np.linalg.inv(M.cinv[k]); v = np.diag(C)[sel]; mm = mod[k][m][sel]
                f = M.data[k][sel] / mm - 1; wv = (mm ** 2) / v
                num += np.sum(wv * f); den += np.sum(wv)
            row.append(f"{rng_lab}: {num/den:+.3f}+-{1/np.sqrt(den):.3f}")
        print(f"  tercile {t}: " + "  ".join(row))
json.dump({lab: {"chi2": sum(x[4] for x in res[lab]), "by_stack": {x[0]: x[4] for x in res[lab]}} for lab in res},
          open(f"{DATA}/output/v1.1/results/ppd_joint_on_stacks.json", "w"), indent=1)
