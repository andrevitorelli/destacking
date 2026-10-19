"""Paper 1.1 (2026-10-11): MAP + PJ-HPD for the analytic-covariance joint fit (the HPC cluster joint_v11e) and the forward-modelled stacked fit;
joint SH fit taken from results/sh_summary.json. Output: results/map_pjhpd_real_v11.json (same layout as harness results/map_pjhpd_real.json)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import sys, json, numpy as np
from scipy import stats
import map_pjhpd as M
from map_summary_real import boot_posteriors, summarise, stacked
V = f"{DATA}/output/v1.1"
P9C = dict(M.PRIORS, c0=stats.truncnorm((0 - 4.5) / 5, np.inf, loc=4.5, scale=5))
P9B = dict(P9C, f_mis=stats.norm(0.294, 0.178), sigma_off=stats.norm(0.422, 0.128))
R = {"sh": json.load(open(f"{V}/results/sh_summary.json"))["map"]}
order, boots = boot_posteriors(f"{V}/real_runs/joint_v11e/simulation_g00_gamma+0.0", P9B); R["v11e"], maps = summarise(order, boots); R["v11e"]["_n"] = len(boots)
order, R["stacked"] = stacked(f"{V}/stacked/real_minN50.npz", M.PRIORS, "stacked")
for k in ("sh", "v11e", "stacked"):
    print(k)
    for p in order:
        q = R[k][p]; print(f"   {p:11s} {q['median']:8.3f} [{q['p16']:7.3f},{q['p84']:7.3f}]  MAP {q['map_med']:8.3f}  PJ [{q['pj_lo']:7.3f},{q['pj_hi']:7.3f}]")
json.dump(R, open(f"{V}/results/map_pjhpd_real_v11.json", "w"), indent=1, default=float)
