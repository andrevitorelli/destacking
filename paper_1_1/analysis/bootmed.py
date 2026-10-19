"""Bootstrap-median summaries (2026-10-07): median and 16-84 percentiles of the per-bootstrap posterior medians."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import numpy as np, glob, json, sys
def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]; return np.interp(q, c, x[o])
def boot_medians(files, key_samples):
    meds = []; names = None
    for f in files:
        z = np.load(f, allow_pickle=True); names = [str(p) for p in z["param_order"]]
        X = z[key_samples]; w = z["weights"] / z["weights"].sum()
        meds.append([wq(X[:, j], w, 0.5) for j in range(X.shape[1])])
    return names, np.array(meds)
def summary(names, meds):
    out = {}
    for j, p in enumerate(names):
        lo, md, hi = np.percentile(meds[:, j], [16, 50, 84]); out[p] = [float(md), float(md - lo), float(hi - md), float(meds[:, j].std(ddof=1))]
    return out
if __name__ == "__main__":
    v9c = sorted(glob.glob(f"{DATA}/output/real_runs/joint_v9c/simulation_g00_gamma+0.0/bootstrap_*/gamma_posterior.npz"))
    n, m = boot_medians(v9c, "all_samples"); S = summary(n, m); S["_n"] = len(v9c)
    st = np.load(f"{DATA}/output/stacked_forward/real_minN50_v3.npz", allow_pickle=True)
    ns = [str(p) for p in st["param_order"]]; w = st["weights"] / st["weights"].sum()
    ST = {p: [float(v) for v in (lambda q: [q[1], q[1]-q[0], q[2]-q[1]])(wq(st["samples"][:, j], w, [0.16, .5, .84]))] for j, p in enumerate(ns)}
    json.dump({"joint_v9c_bootmed": S, "stacked": ST}, open(f"{DATA}/output/referee_2026-10-07/bootmed_v9c.json", "w"), indent=1)
    print(f"{'param':11s} {'joint (boot-med)':>26s} {'stacked':>26s}  diff[sig]  ratio st/jt")
    for p in n:
        j = S[p]; s = ST[p]; sj = (j[1]+j[2])/2; ss = (s[1]+s[2])/2
        print(f"{p:11s} {j[0]:9.3f} -{j[1]:.3f} +{j[2]:.3f}      {s[0]:9.3f} -{s[1]:.3f} +{s[2]:.3f}   {(j[0]-s[0])/np.hypot(sj,ss):+.2f}   {ss/sj:.2f}")
