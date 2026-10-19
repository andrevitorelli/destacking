"""2026-10-11: miscentering-off (k{k}M0L) minus full-physics (k{k}L) joint gamma, paired by seed, at k = 2, 3, 5, 6."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import glob, json, numpy as np
O = f"{DATA}/output"
def g(T):
    f = glob.glob(f"{O}/jvs/{T}_joint/simulation_g00*/bootstrap_000/summary.json")
    if not f: return None
    d = json.load(open(f[0])); return d["gamma_med"] - d["gamma_true_ols"], d["gamma_true_ols"]
print(" k  gamma    N  full (fit-true)       M0 (fit-true)        paired M0-full        shift/gamma")
for k in (2, 3, 5, 6):
    P = [(g(f"LSSTind{r}k{k}L"), g(f"LSSTind{r}k{k}M0L")) for r in range(1, 13)]
    P = [p for p in P if p[0] and p[1]]
    a = np.array([p[0][0] for p in P]); b = np.array([p[1][0] for p in P]); d = b - a; gt = np.mean([p[0][1] for p in P])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(f" {k}  {gt:.4f}  {len(P):2d}  {a.mean():+.4f}+-{se(a):.4f}   {b.mean():+.4f}+-{se(b):.4f}   {d.mean():+.4f}+-{se(d):.4f}   {d.mean()/gt:+.3f}+-{se(d)/gt:.3f}")
