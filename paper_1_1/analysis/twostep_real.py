"""Two-step analysis of the real stacks from the per-stack fits in stack_only_fits.json (2026-10-08; same WLS as twostep_sims.py)."""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import json, sys, pickle, numpy as np
def wls(y, e, cols):
    X = np.column_stack([np.ones(len(y))] + cols); W = 1 / e ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None])); b = cov @ X.T @ (W * y)
    return b, np.sqrt(np.diag(cov)), float(np.sum(W * (y - X @ b) ** 2)), len(y) - X.shape[1]
ST = pickle.load(open(f"{DATA}/real_data/stacked_full_param_fit_f1.2/stacks3.pkl", "rb"))
S = json.load(open(f"{DATA}/output/harness_2026-09-29/results/stack_only_fits.json"))
S = S.get("fits", S); ks = sorted(k for k, v in S.items() if isinstance(v, dict) and v.get("n", 0) >= 50)
a = lambda f: np.array([f(S[k]) for k in ks], float)
logM = a(lambda r: r["logM"][1]); eM = a(lambda r: (r["logM"][2] - r["logM"][0]) / 2); logc = a(lambda r: r["logc"][1]); ec = a(lambda r: (r["logc"][2] - r["logc"][0]) / 2)
lam = np.array([ST[k]["lambda_mean"] for k in ks], float); z = a(lambda r: r["z"]); dm = a(lambda r: r["dm"]); lz = np.log10((1 + z) / 1.35)
bM, sM, chiM, dofM = wls(logM, eM, [np.log10(lam / 40), lz]); bc, sc, chic, dofc = wls(logc, ec, [logM - 14.0, lz, dm])   # v1.1: linear in Delta m
out = dict(log10M0=[bM[0], sM[0]], alpha_mr=[bM[1], sM[1]], beta_mr=[bM[2], sM[2]], chi2_mr=[chiM, dofM], c0=[10 ** bc[0], 10 ** bc[0] * np.log(10) * sc[0]],
           alpha_cm=[bc[1], sc[1]], beta_cm=[bc[2], sc[2]], gamma_cm=[bc[3], sc[3]], chi2_cm=[chic, dofc])
print(len(ks), {k: np.round(v, 3).tolist() for k, v in out.items()})
json.dump(out, open(f"{DATA}/output/v1.1/results/twostep_real.json" if "--write" in sys.argv else f"{DATA}/output/v1.1/results/twostep_real_check.json", "w"), indent=1, default=float)
