"""MAP + PJ-HPD vs marginal median for the real-data joint (bootstrapped) and stacked fits (2026-10-02)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, glob, json, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from nautilus import Prior, Sampler
import map_pjhpd as M
O = f"{DATA}/output"
LEGACY = {"c0": stats.truncnorm(-0.45, np.inf, loc=4.5, scale=10), "alpha_cm": stats.norm(-0.088, 3), "beta_cm": stats.norm(-0.4, 10), "gamma_cm": stats.norm(0, 10.5),
          "log_M_norm": stats.norm(0, 3), "alpha_mr": stats.norm(1.356, 3), "beta_mr": stats.norm(-0.3, 3), "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0, 1)}
RUNS = {"v5": (f"{O}/run_2026-09-26T04-53-48/simulation_*", LEGACY), "v7": (f"{O}/real_runs/joint_v7/simulation_g00_gamma+0.0", M.PRIORS)}
def boot_posteriors(pat, priors):
    out = []
    for d in sorted(glob.glob(glob.glob(pat)[0] + "/bootstrap_*")):
        if not os.path.exists(d + "/summary.json"): continue
        z = np.load(d + "/gamma_posterior.npz", allow_pickle=True); order = [str(t) for t in z["param_order"]]
        pr = Prior()
        for p in order: pr.add_parameter(p, dist=priors[p])
        s = Sampler(pr, lambda p: 0.0, n_live=1000, filepath=d + "/nautilus_checkpoint.hdf5", resume=True)
        X, lw, ll = s.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
        out.append((X, w, ll + M.log_prior(X, order, priors)))
    return order, out
def summarise(order, boots):
    pooledX = np.concatenate([b[0] for b in boots]); pooledW = np.concatenate([b[1] for b in boots]) / len(boots)
    rel = np.concatenate([b[2] - b[2].max() for b in boots])           # log posterior relative to each bootstrap's own maximum
    maps = np.array([b[0][np.argmax(b[2])] for b in boots])
    pj = M.pjhpd(pooledX, pooledW, rel)
    res = {}
    for j, p in enumerate(order):
        m = M.marginal_hpd_and_median(pooledX[:, j], pooledW)
        res[p] = dict(median=m["median"], mode=m["mode"], p16=m["p16"], p84=m["p84"], map_med=float(np.median(maps[:, j])), map_p16=float(np.percentile(maps[:, j], 16)),
                      map_p84=float(np.percentile(maps[:, j], 84)), pj_lo=float(pj[j, 0]), pj_hi=float(pj[j, 1]))
    return res, maps
def stacked(path, priors, tag):
    z = np.load(path, allow_pickle=True); order = [str(t) for t in z["param_order"]]; X = z["samples"]; w = z["weights"] / z["weights"].sum()
    lp = z["log_l"] + M.log_prior(X, order, priors); pj = M.pjhpd(X, w, lp); i = np.argmax(lp); res = {}
    for j, p in enumerate(order):
        m = M.marginal_hpd_and_median(X[:, j], w)
        res[p] = dict(median=m["median"], mode=m["mode"], p16=m["p16"], p84=m["p84"], map_med=float(X[i, j]), map_p16=np.nan, map_p84=np.nan, pj_lo=float(pj[j, 0]), pj_hi=float(pj[j, 1]))
    return order, res
if __name__ == "__main__":
    allres = {}
    for k, (pat, pri) in RUNS.items():
        order, boots = boot_posteriors(pat, pri); allres[k], maps = summarise(order, boots); allres[k]["_maps"] = maps.tolist(); print("done", k, len(boots), flush=True)
    order, allres["stacked v3"] = stacked(f"{O}/stacked_forward/real_minN50_v3.npz", M.PRIORS, "stacked"); print("done stacked")
    json.dump(allres, open("../results/map_pjhpd_real.json", "w"), indent=1)
    for k, r in allres.items():
        print(f"\n{k}:  param      median[16,84]            mode     MAP(med of boots; [16,84] of boots)     PJ-HPD 68%")
        for p in order:
            q = r[p]; print(f"   {p:11s} {q['median']:8.3f} [{q['p16']:7.3f},{q['p84']:7.3f}]  {q['mode']:8.3f}  {q['map_med']:8.3f} [{q['map_p16']:7.3f},{q['map_p84']:7.3f}]   [{q['pj_lo']:7.3f},{q['pj_hi']:7.3f}]")
