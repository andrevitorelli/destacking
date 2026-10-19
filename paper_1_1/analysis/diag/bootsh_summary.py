"""2026-10-11: analytic-Gaussian (joint) vs bootstrap-Gaussian (jointB) vs bootstrap-Sellentin-Heavens (jointS) joint fits of the
v1.1 scan realisations, all 9 parameters; plus the k=5 triaxiality-off (k5T0L) / miscentering-off (k5M0L) test.  Writes bootsh_rows.json."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import glob, json, re, numpy as np
O = f"{DATA}/output"
P = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
def q(x, w, p):
    o = np.argsort(x); c = np.cumsum(w[o]); return np.interp(p, c / c[-1], x[o])
def fit(T, suf, fmis=0.36):
    o = glob.glob(f"{O}/{T}_sim/simulation_g00*/ols.json"); f = glob.glob(f"{O}/jvs/{T}_{suf}/simulation_g00*/bootstrap_000/gamma_posterior.npz")
    if not (o and f): return None
    o = json.load(open(o[0])); z = np.load(f[0], allow_pickle=True); X, W, nm = z["all_samples"], z["weights"], list(z["param_order"])
    tru = {"c0": o["c0_true"], "alpha_cm": o["alpha_true"], "beta_cm": o["beta_true"], "gamma_cm": o["gamma_true_ols"], "log_M_norm": 0.0,
           "alpha_mr": o["alpha_mr_true"], "beta_mr": o["beta_mr_true"], "f_mis": fmis, "sigma_off": 0.32 / 0.6766}
    out = {"gamma_true": tru["gamma_cm"]}
    for i, p in enumerate(nm):
        lo, me, hi = q(X[:, i], W, [0.16, 0.5, 0.84]); out[p] = (me - tru[p], (hi - lo) / 2)
    return out
rows = []
for sv, pat in (("DES", "DESfull{r}k{k}L"), ("LSST", "LSSTind{r}k{k}L")):
    for k in range(7):
        for r in range(1, 13):
            T = pat.format(r=r, k=k)
            for suf in ("joint", "jointB", "jointS"):
                f = fit(T, suf)
                if f: rows.append({"survey": sv, "k": k, "r": r, "method": suf, **f})
for r in range(1, 13):
    for v, suf, fm in (("full", "", 0.36), ("T0", "T0", 0.36), ("M0", "M0", 0.0)):
        f = fit(f"LSSTind{r}k5{suf}L", "joint", fm)
        if f: rows.append({"survey": "LSSTk5", "k": 5, "r": r, "method": v, **f})
json.dump(rows, open(f"{O}/bootsh_rows.json", "w"))
LAB = {"joint": "analytic, Gaussian", "jointB": "bootstrap, Gaussian", "jointS": "bootstrap, SH"}
for sv in ("DES", "LSST"):
    R = [x for x in rows if x["survey"] == sv]
    print(f"\n===== {sv}")
    for m in ("joint", "jointB", "jointS"):
        rr = [x for x in R if x["method"] == m]
        xs = np.array([x["gamma_true"] for x in rr]); ys = xs + np.array([x["gamma_cm"][0] for x in rr])
        A = np.vstack([xs, np.ones_like(xs)]).T; c, *_ = np.linalg.lstsq(A, ys, rcond=None); res = ys - A @ c; cov = np.linalg.inv(A.T @ A) * res.var(ddof=2)
        d0 = np.array([x["gamma_cm"][0] for x in rr if x["k"] == 0])
        print(f"  {LAB[m]:20s} N={len(rr):3d} slope {c[0]:.3f}+-{np.sqrt(cov[0,0]):.3f}  gamma=0 offset {d0.mean():+.4f}+-{d0.std(ddof=1)/np.sqrt(len(d0)):.4f} (n={len(d0)})")
        print("      bias/<sigma>:   " + " ".join(f"{p}={np.mean([x[p][0] for x in rr])/np.mean([x[p][1] for x in rr]):+.2f}" for p in P))
        print("      scatter/<sigma>:" + " ".join(f"{p}={np.std([x[p][0] for x in rr],ddof=1)/np.mean([x[p][1] for x in rr]):.2f}" for p in P))
    a = {(x["k"], x["r"]): x for x in R if x["method"] == "joint"}; s = {(x["k"], x["r"]): x for x in R if x["method"] == "jointS"}
    ks = sorted(set(a) & set(s))
    print(f"  paired SH - analytic, mean +- s.e.m. in units of analytic sigma (N={len(ks)}):")
    print("      " + " ".join(f"{p}={np.mean([(s[k][p][0]-a[k][p][0])/a[k][p][1] for k in ks]):+.2f}+-{np.std([(s[k][p][0]-a[k][p][0])/a[k][p][1] for k in ks],ddof=1)/np.sqrt(len(ks)):.2f}" for p in P))
print("\n===== LSST k=5 (gamma=5/36): joint gamma (fit - truth), same seeds")
K = {v: {x["r"]: x["gamma_cm"][0] for x in rows if x["survey"] == "LSSTk5" and x["method"] == v} for v in ("full", "T0", "M0")}
for v in ("full", "T0", "M0"):
    d = np.array(list(K[v].values())); print(f"  {v:4s} N={len(d):2d} mean {d.mean():+.4f}+-{d.std(ddof=1)/np.sqrt(len(d)):.4f}  scatter {d.std(ddof=1):.4f}")
for v in ("T0", "M0"):
    rs = sorted(set(K[v]) & set(K["full"])); d = np.array([K[v][r] - K["full"][r] for r in rs])
    print(f"  paired {v} - full: {d.mean():+.4f}+-{d.std(ddof=1)/np.sqrt(len(d)):.4f} (N={len(rs)})")
