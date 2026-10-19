"""2026-10-16 diagnostic (NOT for the paper): why is the two-step stacked analysis biased in the v1.1 simulations?
For each scan realisation, rebuild each stack's members from its bin edges (z, Lambda, Dm tercile), check the count against n_stacked,
and compare the per-stack fit (twostep_v11/{TL}.json) with the members' true mean log10 M and log10 c.  Then redo step 2 with
  base : fitted logM, logc; regressors log10(<lambda>), <z>, <Dm>  (= the two-step analysis)
  A    : fitted logM, logc; member means of log10 lambda, log10(1+z), Dm               (cause 2: stack means vs member means)
  B    : as A, but true member-mean log10 M as the regressor of the c-M relation     (cause 3: fitted-mass regressor)
  C    : true member-mean log10 M and log10 c, member-mean observables               (perfect step 1: what remains is step-2 aggregation)
usage: python twostep_diag.py OUT.json   (run where $DESTACKING_DATA/output is visible)"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import glob, json, sys, numpy as np
O = f"{DATA}/output"
def wls(y, e, cols):
    X = np.column_stack([np.ones(len(y))] + cols); W = 1 / e ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None])); return cov @ X.T @ (W * y)
names = [l.split()[1] for l in open(f"{O}/scan_list.txt")]
res, bad = [], []
import pickle
for tl in names:
    d = json.load(open(f"{O}/twostep_v11/{tl}.json")); st = d["stacks"]; tr = d["truth"]
    sd = glob.glob(f"{O}/{tl}_sim/simulation_g00*")[0]
    t = np.load(f"{sd}/truth_catalog.npz"); co = np.load(f"{sd}/cluster_obs.npz", allow_pickle=True)
    logM_t, logc_t = np.log10(t["M_halo"]), np.log10(t["concentration"])
    lam, z, dm = np.asarray(co["Lambda"], float), np.asarray(co["z"], float), np.asarray(co["delta_m"], float)
    S = pickle.load(open(f"{O}/jvs/{tl}_stacks/stacks3.pkl", "rb")); S = S["stacks"] if "stacks" in S and "z_bin" in S else S
    rows = []
    for k, v in st.items():
        m = S[k]; sel = (z >= m["z_lo"]) & (z < m["z_hi"]) & (lam >= m["ng_lo"]) & (lam < m["ng_hi"]) & np.isfinite(dm) & (dm > 0)
        terc = np.asarray(m["dm_terciles"], float); b = int(m["dm_bin"])
        lo = -np.inf if b == 0 else terc[b - 1] if len(terc) == 2 else terc[b]; hi = np.inf if b == 2 else terc[b] if len(terc) == 2 else terc[b + 1]
        sel &= (dm >= lo) & (dm < hi)
        if abs(int(sel.sum()) - v["n"]) > 2: bad.append((tl, k, int(sel.sum()), v["n"])); continue
        rows.append(dict(fM=v["logM"][1], eM=(v["logM"][2] - v["logM"][0]) / 2, fc=v["logc"][1], ec=(v["logc"][2] - v["logc"][0]) / 2,
                         tM=logM_t[sel].mean(), tc=logc_t[sel].mean(), lam_m=np.log10(v["lam"]), z_m=np.log10((1 + v["z"]) / 1.35), dm_m=v["dm"],
                         lam_i=np.log10(lam[sel]).mean(), z_i=np.log10((1 + z[sel]) / 1.35).mean(), dm_i=dm[sel].mean()))
    if len(rows) < 8: continue
    a = lambda k: np.array([r[k] for r in rows])
    fM, eM, fc, ec, tM, tc = a("fM"), a("eM"), a("fc"), a("ec"), a("tM"), a("tc")
    out = dict(name=tl, k=int(tl.split("k")[1][0]), dM=float(np.mean(fM - tM)), dc=float(np.mean(fc - tc)),
               dM_z=list(np.polyfit(a("z_m"), fM - tM, 1)), dc_z=list(np.polyfit(a("z_m"), fc - tc, 1)))
    tru = dict(gamma=tr["gamma_true_ols"], alpha_cm=tr["alpha_true"], beta_cm=tr["beta_true"], logc0=np.log10(tr["c0_true"]),
               alpha_mr=tr["alpha_mr_true"], beta_mr=tr["beta_mr_true"], logM0=14.489)
    for var, (Mv, cv, Mreg, lamr, zr, dmr) in dict(
            base=(fM, fc, fM, a("lam_m"), a("z_m"), a("dm_m")), A=(fM, fc, fM, a("lam_i"), a("z_i"), a("dm_i")),
            B=(fM, fc, tM, a("lam_i"), a("z_i"), a("dm_i")), C=(tM, tc, tM, a("lam_i"), a("z_i"), a("dm_i"))).items():
        bM = wls(Mv, eM, [lamr - np.log10(40), zr]); bc = wls(cv, ec, [Mreg - 14.0, zr, dmr])
        out[var] = dict(logM0=bM[0] - tru["logM0"], alpha_mr=bM[1] - tru["alpha_mr"], beta_mr=bM[2] - tru["beta_mr"],
                        logc0=bc[0] - tru["logc0"], alpha_cm=bc[1] - tru["alpha_cm"], beta_cm=bc[2] - tru["beta_cm"], gamma=bc[3] - tru["gamma"])
    res.append(out)
json.dump(dict(res=res, bad=bad), open(sys.argv[1], "w"), indent=1, default=float)
print(len(res), "realisations;", len(bad), "stacks with membership mismatch", bad[:3])
