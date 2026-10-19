"""2026-10-16: joint vs forward-modelled stacked vs two-step stacked fits of the same v1.1 scan realisations (48 LSST-like, 84 DES-like).
Per parameter: mean bias (fit - truth) +- s.e.m., scatter between realisations (std), mean reported uncertainty, and scatter ratios to joint."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, glob, re, numpy as np
R = f"{DATA}/output/v1.1"
P = [("gamma_cm", "gamma_cm", "gamma_true_ols"), ("alpha_cm", "alpha_cm", "alpha_true"), ("beta_cm", "beta_cm", "beta_true"),
     ("c0", "c0", "c0_true"), ("alpha_mr", "alpha_mr", "alpha_mr_true"), ("beta_mr", "beta_mr", "beta_mr_true"), ("log_M_norm", "log10M0", None)]
out = {}
for sv, rf, pat in (("LSST", "report_des/lsst_scan_rows_v11_48.json", "LSSTind"), ("DES", "report_des/des_scan_rows_v11.json", "DESfull")):
    rows = json.load(open(f"{R}/{rf}")); M = {(r["method"], r["k"], r["r"]): r for r in rows}
    T = {}
    for f in glob.glob(f"{R}/twostep_v11/{pat}*.json"):
        d = json.load(open(f)); r, k = map(int, re.search(r"(\d+)k(\d)L", d["name"]).groups()); T[(k, r)] = d
    keys = sorted(set(T) & {(k, r) for (m, k, r) in M if m == "joint"} & {(k, r) for (m, k, r) in M if m == "stacked"})
    print(f"\n===== {sv}-like: {len(keys)} realisations with all three methods")
    print(f"  {'param':10s} | {'bias joint':>17s} {'bias forward':>17s} {'bias two-step':>17s} | {'scatter J':>9s} {'F':>8s} {'2S':>8s} | F/J  2S/J | width/scatter J F 2S")
    for p, tp, tt in P:
        bj = np.array([M[("joint",) + kk]["d"][p][0] for kk in keys]); wj = np.array([M[("joint",) + kk]["d"][p][1] for kk in keys])
        bf = np.array([M[("stacked",) + kk]["d"][p][0] for kk in keys]); wf = np.array([M[("stacked",) + kk]["d"][p][1] for kk in keys])
        tv = np.array([T[kk]["twostep"][tp][0] for kk in keys]); tw = np.array([T[kk]["twostep"][tp][1] for kk in keys])
        tru = np.array([T[kk]["truth"][tt] if tt else 14.489 for kk in keys]); bt = tv - tru
        s = lambda x: x.std(ddof=1); e = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        print(f"  {p:10s} | {bj.mean():+8.4f}+-{e(bj):.4f} {bf.mean():+8.4f}+-{e(bf):.4f} {bt.mean():+8.4f}+-{e(bt):.4f} | {s(bj):9.4f} {s(bf):8.4f} {s(bt):8.4f} |"
              f" {s(bf)/s(bj):4.2f} {s(bt)/s(bj):5.2f} | {wj.mean()/s(bj):4.2f} {wf.mean()/s(bf):4.2f} {tw.mean()/s(bt):4.2f}")
        out[f"{sv}_{p}"] = dict(bias=[bj.mean(), bf.mean(), bt.mean()], sem=[e(bj), e(bf), e(bt)], scatter=[s(bj), s(bf), s(bt)], width=[wj.mean(), wf.mean(), tw.mean()])
    # recovery slope of gamma for the two-step
    x = np.array([T[kk]["truth"]["gamma_true_ols"] for kk in keys]); y = np.array([T[kk]["twostep"]["gamma_cm"][0] for kk in keys])
    A = np.vstack([x, np.ones_like(x)]).T; c, *_ = np.linalg.lstsq(A, y, rcond=None); res = y - A @ c; cov = np.linalg.inv(A.T @ A) * res.var(ddof=2)
    d0 = np.array([T[kk]["twostep"]["gamma_cm"][0] - T[kk]["truth"]["gamma_true_ols"] for kk in keys if kk[0] == 0])
    print(f"  two-step gamma: slope {c[0]:.3f}+-{np.sqrt(cov[0,0]):.3f}, offset {c[1]:+.4f}+-{np.sqrt(cov[1,1]):.4f}; gamma=0 mean {d0.mean():+.4f}+-{d0.std(ddof=1)/np.sqrt(len(d0)):.4f} (n={len(d0)})")
    out[f"{sv}_twostep_slope"] = [c[0], np.sqrt(cov[0, 0]), c[1], np.sqrt(cov[1, 1])]
json.dump(out, open(f"{R}/diag/threeway_summary.json", "w"), indent=1)
