"""Summaries of the real-data joint fit with the per-bin Sellentin-Heavens likelihood (2026-10-09), for the paper:
bootstrap medians (median, 16-84%), KL (joint = <log L> - log Z; marginal via weighted KDE; residual), within/between widths,
differences and width ratios against the stacked fit (v4, Hartlap with N_s), robustness variants (full-sample), pivot correlations,
MAP + PJ-HPD (Joachimi+21) for the MAP appendix.  Output: sh_summary.json"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import glob, json, sys, numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from bootmed import wq
P = f"{DATA}/output/v1.1/posteriors"
PRI = {"c0": stats.truncnorm((0 - 4.5) / 5, np.inf, loc=4.5, scale=5), "alpha_cm": stats.norm(-0.1, 0.5), "beta_cm": stats.norm(-0.4, 2),
       "gamma_cm": stats.norm(0, 1), "log_M_norm": stats.norm(0, 0.5), "alpha_mr": stats.norm(1.356, 0.5), "beta_mr": stats.norm(-0.3, 2),
       "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0, 1), "delta_mr": stats.norm(0, 1), "gamma": stats.norm(0, 1)}


def load(f):
    z = np.load(f, allow_pickle=True); o = [str(t) for t in z["param_order"]]; w = z["weights"] / z["weights"].sum()
    return o, z["samples"], w, z["log_l"], float(z["log_z"])


def kl(o, X, w, ll, lz):
    joint = float(np.sum(w * ll) - lz); marg = {}
    ii = np.random.default_rng(0).choice(len(w), 4000, p=w)          # equal-weight subsample for the marginal KDEs (O(n^2))
    for j, p in enumerate(o):
        x = X[ii, j]; q = np.log(np.clip(gaussian_kde(x).pdf(x), 1e-300, None)); marg[p] = float(np.mean(q - PRI[p].logpdf(x)))
    return joint, marg


def boot_summary(prefix):
    fs = sorted(glob.glob(f"{P}/{prefix}_b[0-9][0-9][0-9].npz")); med = []; within = []; KL = []; MK = []
    for f in fs:
        o, X, w, ll, lz = load(f); med.append([wq(X[:, j], w, 0.5) for j in range(X.shape[1])])
        m = np.sum(w[:, None] * X, 0); within.append(np.sqrt(np.sum(w[:, None] * (X - m) ** 2, 0)))
        j_, mk = kl(o, X, w, ll, lz); KL.append(j_); MK.append([mk[p] for p in o])
    med = np.array(med); S = {}
    for j, p in enumerate(o):
        lo, me, hi = np.percentile(med[:, j], [16, 50, 84]); S[p] = dict(med=float(me), lo=float(me - lo), hi=float(hi - me), between=float(med[:, j].std(ddof=1)),
                                                                          within=float(np.sqrt(np.mean(np.array(within)[:, j] ** 2))), kl=float(np.median(np.array(MK)[:, j])))
    S["_kl_joint"] = float(np.median(KL)); S["_kl_resid"] = float(np.median(np.array(KL) - np.array(MK).sum(1))); S["_n"] = len(fs)
    return S, o, fs


if __name__ == "__main__":
    out = {}
    S, o, fs = boot_summary("joint_sh"); out["joint_sh"] = S
    st = np.load(f"{DATA}/output/v1.1/stacked/real_minN50.npz", allow_pickle=True); so = [str(t) for t in st["param_order"]]; ws = st["weights"] / st["weights"].sum()
    print(f"joint SH: {S['_n']} boots; KL joint {S['_kl_joint']:.2f} resid {S['_kl_resid']:.2f}")
    for p in o:
        s = S[p]; lo_s, me_s, hi_s = wq(st["samples"][:, so.index(p)], ws, [.16, .5, .84]); sj = (s["lo"] + s["hi"]) / 2; ss = (hi_s - lo_s) / 2
        s.update(diff=float((s["med"] - me_s) / np.hypot(sj, ss)), ratio=float(ss / sj))
        print(f"  {p:11s} {s['med']:8.4f} -{s['lo']:.4f} +{s['hi']:.4f}  KL {s['kl']:.2f}  within {s['within']:.4f} between {s['between']:.4f}  diff {s['diff']:+.2f}  ratio {s['ratio']:.2f}")
    # variants (full-sample)
    out["variants"] = {}
    for f in sorted(glob.glob(f"{P}/joint_full_sh_*.npz")):
        o2, X, w, ll, lz = load(f); g = o2.index("gamma_cm"); lo, me, hi = wq(X[:, g], w, [.16, .5, .84]); r = dict(gamma=[float(me), float(hi - me), float(me - lo)])
        for p in o2:
            if p.startswith("f_mis_t") or p in ("c0", "alpha_cm"): q = wq(X[:, o2.index(p)], w, [.16, .5, .84]); r[p] = [float(q[1]), float(q[2] - q[1]), float(q[1] - q[0])]
        if "zbin" in f: q = wq(X[:, o2.index("log_M_norm")], w, [.16, .5, .84]); r["log_M_norm"] = [float(q[1]), float(q[2] - q[1]), float(q[1] - q[0])]
        out["variants"][f.split("joint_full_sh_")[1][:-4]] = r
        print("  variant", f.split("joint_full_sh_")[1][:-4], np.round(r["gamma"], 3).tolist(), {k: np.round(v, 3).tolist() for k, v in r.items() if k != "gamma"})
    # pivot correlations on the pooled posterior
    X = np.concatenate([load(f)[1] for f in fs]); W = np.concatenate([load(f)[2] / len(fs) for f in fs])
    c = {p: X[:, o.index(p)] for p in o}; lc = np.log10(c["c0"])
    def wc(a, b):
        ma, mb = np.sum(W * a), np.sum(W * b); return float(np.sum(W * (a - ma) * (b - mb)) / np.sqrt(np.sum(W * (a - ma) ** 2) * np.sum(W * (b - mb) ** 2)))
    l2 = lc + c["alpha_cm"] * 0.3 + c["gamma_cm"] * 0.63   # v1.1: linear in Delta m, pivot Delta m = 0.63
    out["pivot"] = dict(old_gamma=wc(lc, c["gamma_cm"]), old_alpha=wc(lc, c["alpha_cm"]), new_gamma=wc(l2, c["gamma_cm"]), new_alpha=wc(l2, c["alpha_cm"]))
    print("  pivot corr", {k: round(v, 2) for k, v in out["pivot"].items()})
    # MAP + PJ-HPD
    import map_pjhpd as M
    boots = []
    for f in fs:
        o3, X3, w3, ll3, _ = load(f); lp = ll3 + sum(PRI[p].logpdf(X3[:, j]) for j, p in enumerate(o3)); boots.append((X3, w3, lp))
    pooledX = np.concatenate([b[0] for b in boots]); pooledW = np.concatenate([b[1] for b in boots]) / len(boots); rel = np.concatenate([b[2] - b[2].max() for b in boots])
    maps = np.array([b[0][np.argmax(b[2])] for b in boots]); pj = M.pjhpd(pooledX, pooledW, rel); out["map"] = {}
    for j, p in enumerate(o):
        m = M.marginal_hpd_and_median(pooledX[:, j], pooledW)
        out["map"][p] = dict(median=m["median"], p16=m["p16"], p84=m["p84"], map_med=float(np.median(maps[:, j])), pj_lo=float(pj[j, 0]), pj_hi=float(pj[j, 1]))
    json.dump(out, open(f"{DATA}/output/v1.1/results/sh_summary.json", "w"), indent=1, default=float)
    print("ok")
