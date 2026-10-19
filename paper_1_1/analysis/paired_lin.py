import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, sys, json, numpy as np
os.environ["DVT_GRID"] = "-0.6,0.6,6"
import dv_vs_truth as D
from destacking.model import _MC_DM_REF
FORM = sys.argv[1]
def shift(d):
    dv, cov, t = D.load(d); z, M, c, dm = t["z"], t["M_halo"], t["concentration"], t["delta_m"]
    prof = lambda MM, cc: D._nfw_profiles(z, MM, cc, D.R, bin_nodes=4); m = prof(M, c); h = 1e-3
    a = (prof(M, c * np.exp(h)) - prof(M, c * np.exp(-h))) / (2 * h); b = (prof(M * np.exp(h), c) - prof(M * np.exp(-h), c)) / (2 * h)
    ok = np.isfinite(dm) & (dm > 0)
    x = np.where(ok, np.log10(dm / _MC_DM_REF), 0.0) if FORM == "log" else np.where(ok, dm, 0.0)
    dl = np.stack([np.full_like(z, 1 / D.C0), D.LN10 * (np.log10(M) - D._MC_M_PIVOT_LOG), D.LN10 * np.log10((1 + z) / (1 + D._MC_Z_REF)), D.LN10 * x], 1)
    J = np.concatenate([a[:, :, None] * dl[:, None, :], (b * D.LN10 + a * D.ALPHA * D.LN10)[:, :, None]], 2)
    v = np.isfinite(dv); r = np.where(v, dv - m, 0.0); J = np.where(v[:, :, None], J, 0.0)
    Cr = np.where(v, np.linalg.solve(cov, r[:, :, None])[:, :, 0], 0.0); CJ = np.where(v[:, :, None], np.linalg.solve(cov, J), 0.0)
    F = np.einsum("nkp,nkq->pq", J, CJ); return np.linalg.solve(F, np.einsum("nkp,nk->p", J, Cr)), np.std(x[ok])
B = f"{DATA}/output/harness_2026-09-29/results"
out = {}
for lab in ("desclean", "desclean_an"):
    S = []; sx = []
    for i in range(24):
        s, x = shift(f"{B}/{lab}/D{i}"); S.append(s); sx.append(x)
    S = np.array(S); out[lab] = S.tolist()
    print(FORM, lab, "mean", np.round(S.mean(0), 4), "se", np.round(S.std(0, ddof=1) / np.sqrt(24), 4), "scatter", np.round(S.std(0, ddof=1), 4), "std(x)", round(np.mean(sx), 3))
if FORM == "linear":
    json.dump({"bootstrap": out["desclean"], "analytic": out["desclean_an"], "params": D.PAR, "dm_form": "linear"}, open(f"{DATA}/output/v1.1/results/des_clean_projection_linear.json", "w"))
