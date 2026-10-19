# v1.1 (2026-10-10): option-C linear-Dm campaign (*L chains)
"""2026-10-09: as scan_summary.py, but stacked fits with the Hartlap factor from the member clusters (*_stacked_hN.npz);
LSSTind1k0 dropped from both methods (inputs deleted). Writes {lsst,des}_scan_rows_v11.json for paper_sim_figs.py."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import glob, json, sys, numpy as np
O = f"{DATA}/output"
SURVEY = sys.argv[1] if len(sys.argv) > 1 else "LSST"
P = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
def q(x, w, p):
    o = np.argsort(x); c = np.cumsum(w[o]); return np.interp(p, c / c[-1], x[o])
def summ(X, W, names):
    return {n: q(X[:, i], W, [0.16, 0.5, 0.84]) for i, n in enumerate(names)}
pat = "LSSTind{r}k{k}L" if SURVEY == "LSST" else "DESfull{r}k{k}L"
rows = []   # (k, r, method, {p: (med-truth, sigma)})
for k in range(7):
    for r in range(1, 13):
        T = pat.format(r=r, k=k)
        if T == "__none__": continue   # LSSTind1k0 restored 2026-10-09 (catalogue rebuilt)
        o = glob.glob(f"{O}/{T}_sim/simulation_g00*/ols.json")
        if not o: continue
        o = json.load(open(o[0]))
        tru = {"c0": o["c0_true"], "alpha_cm": o["alpha_true"], "beta_cm": o["beta_true"], "gamma_cm": o["gamma_true_ols"],
               "log_M_norm": 0.0, "alpha_mr": o["alpha_mr_true"], "beta_mr": o["beta_mr_true"], "f_mis": 0.36, "sigma_off": 0.32 / 0.6766}
        f = glob.glob(f"{O}/jvs/{T}_joint/simulation_g00*/bootstrap_000/gamma_posterior.npz")
        if f:
            z = np.load(f[0], allow_pickle=True); s = summ(z["all_samples"], z["weights"], list(z["param_order"]))
            rows.append((k, r, "joint", tru["gamma_cm"], {p: (s[p][1] - tru[p], (s[p][2] - s[p][0]) / 2) for p in P}))
        f = f"{O}/jvs/{T}_stacked_hN.npz"
        try:
            z = np.load(f, allow_pickle=True); s = summ(z["samples"], z["weights"], list(z["param_order"]))
            rows.append((k, r, "stacked", tru["gamma_cm"], {p: (s[p][1] - tru[p], (s[p][2] - s[p][0]) / 2) for p in P}))
        except FileNotFoundError:
            pass
print(f"{SURVEY}: per injected gamma -- gamma_cm (fit - truth): mean +- s.e.m. [N] | scatter | mean sigma")
for meth in ("joint", "stacked"):
    print(f"  {meth}")
    xs, ys, es = [], [], []
    for k in range(7):
        d = [x for x in rows if x[0] == k and x[2] == meth]
        if not d: continue
        g = np.array([x[4]["gamma_cm"][0] for x in d]); s = np.array([x[4]["gamma_cm"][1] for x in d]); gt = d[0][3]
        sem = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
        print(f"    gamma_true {gt:+.4f}: {g.mean():+.4f} +- {sem:.4f} [{len(g)}] | {g.std(ddof=1) if len(g)>1 else np.nan:.4f} | {s.mean():.4f}")
        for x in d:
            xs.append(gt); ys.append(gt + x[4]["gamma_cm"][0]); es.append(x[4]["gamma_cm"][1])
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([xs, np.ones_like(xs)]).T; coef, res, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ coef; cov = np.linalg.inv(A.T @ A) * resid.var(ddof=2)
    print(f"    recovered vs true (all realisations, equal weight): slope {coef[0]:.4f} +- {np.sqrt(cov[0,0]):.4f}, offset {coef[1]:+.4f} +- {np.sqrt(cov[1,1]):.4f}, N={len(xs)}")
    print(f"    all parameters, pooled over gammas: mean (fit - truth) +- s.e.m. | scatter | mean sigma")
    for p in P:
        d = np.array([x[4][p][0] for x in rows if x[2] == meth]); s = np.array([x[4][p][1] for x in rows if x[2] == meth])
        print(f"      {p:11s} {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f} | {d.std(ddof=1):.4f} | {s.mean():.4f}")

# Rows of every chain found (the LSST-like file includes the 12 trace realisations r = 7..12 at k = 5, 6),
# and, for LSST-like, the 48-chain paper scan without them (read by paper_sim_figs_v11.py).
REP = f"{DATA}/output/v1.1/report_des"; _os.makedirs(REP, exist_ok=True)
_all = [dict(k=x[0], r=x[1], method=x[2], gamma_true=float(x[3]), d={p: [float(v) for v in x[4][p]] for p in P}) for x in rows]
json.dump(_all, open(f"{REP}/{SURVEY.lower()}_scan_rows_v11.json", "w"), indent=1)
if SURVEY == "LSST":
    json.dump([x for x in _all if not (x["k"] in (5, 6) and x["r"] > 6)], open(f"{REP}/lsst_scan_rows_v11_48.json", "w"), indent=1)
