"""LSST-like joint-slope trace, step 1 (2026-10-11): first-order shift of the 9-parameter joint fit, linearised at the truth
(bin-centre model, injected miscentring), decomposed into
  A relation : model(true c) - model(c from the fitted power law at truth)   [DM14 curvature not captured by the power law]
  B binavg   : bin-averaged - bin-centre model at true c                      [bin-centre approximation]
  C data     : data vector - bin-averaged model at true c                     [noise, triaxiality, miscentring draws]
dtheta = F^-1 J^T C^-1 r, diagonal analytic covariance. Usage: python lsst_projection.py OUT.json DIR...  (DIR = .../simulation_g00_*)"""
import sys, json, os, numpy as np
from destacking.model import _nfw_components
LN10 = np.log(10); H = 0.6766; F_MIS = 0.36; S_OFF = 0.32 / H
R = np.logspace(-0.8, 0.6, 8)
PAR = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
def mod(z, M, c, so, bn):
    cen, mis = _nfw_components(z, M, c, R, SIGMA_OFF=np.full(len(z), so), bin_nodes=bn); return (1 - F_MIS) * cen + F_MIS * mis, cen, mis
out = {}
for d in sys.argv[2:]:
    name = d.rstrip("/").split("/")[-2].replace("_sim_analytic", "")
    t = np.load(f"{d}/truth_catalog.npz"); o = np.load(f"{d}/cluster_obs.npz"); s = np.load(f"{d}/simulated_data.npz"); T = json.load(open(f"{d}/ols.json"))
    z, M, c, dm = (np.asarray(t[k], float) for k in ("z", "M_halo", "concentration", "delta_m")); lam = np.asarray(o["Lambda"], float)
    lx, lzm = np.log10(lam / 40), np.log10((1 + z) / 1.35)
    crel = 10 ** (np.log10(T["c0_true"]) + T["alpha_true"] * (np.log10(M) - 14.0) + T["beta_true"] * lzm + T["gamma_true_ols"] * dm)
    dv = s["data_vectors"]; var = np.einsum("nii->ni", s["cov_matrices"]); v = np.isfinite(dv) & np.isfinite(var) & (var > 0); iv = np.where(v, 1 / np.where(v, var, 1), 0.0)
    m0, cen0, mis0 = mod(z, M, crel, S_OFF, 0); h = 1e-3
    dlnc = (mod(z, M, crel * np.exp(h), S_OFF, 0)[0] - mod(z, M, crel * np.exp(-h), S_OFF, 0)[0]) / (2 * h)
    dlnM = (mod(z, M * np.exp(h), crel, S_OFF, 0)[0] - mod(z, M * np.exp(-h), crel, S_OFF, 0)[0]) / (2 * h)
    dso = (mod(z, M, crel, S_OFF + 0.005, 0)[0] - mod(z, M, crel, S_OFF - 0.005, 0)[0]) / 0.01
    dlogM = LN10 * (dlnM + T["alpha_true"] * dlnc)                        # mass shift also moves c through alpha_cm
    J = np.stack([dlnc / T["c0_true"], dlnc * LN10 * (np.log10(M) - 14.0)[:, None], dlnc * LN10 * lzm[:, None], dlnc * LN10 * dm[:, None],
                  dlogM, dlogM * lx[:, None], dlogM * lzm[:, None], mis0 - cen0, dso], 2)
    J = np.where(v[:, :, None], J, 0.0)
    F = np.einsum("nkp,nk,nkq->pq", J, iv, J); Finv = np.linalg.inv(F)
    mt0 = mod(z, M, c, S_OFF, 0)[0]; mt4 = mod(z, M, c, S_OFF, 4)[0]
    res = {"A_relation": mt0 - m0, "B_binavg": mt4 - mt0, "C_data": np.where(v, dv, 0) - mt4}
    r = {"n": int(len(z)), "gamma_true": T["gamma_true_ols"], "fisher_sigma_gamma": float(np.sqrt(Finv[3, 3])),
         "median_data_over_model_by_bin": [float(np.nanmedian(np.where(v[:, k], dv[:, k], np.nan) / mt4[:, k])) for k in range(len(R))]}
    for lab, rr in res.items():
        rr = np.where(v, rr, 0.0); S = np.einsum("nkp,nk,nk->p", J, iv, rr); th = Finv @ S
        g_bin = np.einsum("p,nkp,nk,nk->k", Finv[3], J, iv, rr)
        terc = np.digitize(dm, np.quantile(dm, [1 / 3, 2 / 3])); g_t = [float(np.einsum("p,nkp,nk,nk->", Finv[3], J[terc == q], iv[terc == q], rr[terc == q])) for q in range(3)]
        r[lab] = {"shift": dict(zip(PAR, map(float, th))), "gamma_by_bin": g_bin.tolist(), "gamma_by_tercile": g_t}
    r["total_gamma"] = sum(r[l]["shift"]["gamma_cm"] for l in res); out[name] = r
    print(f"{name}: g_true {T['gamma_true_ols']:.4f}  dgamma A {r['A_relation']['shift']['gamma_cm']:+.4f} B {r['B_binavg']['shift']['gamma_cm']:+.4f} "
          f"C {r['C_data']['shift']['gamma_cm']:+.4f} total {r['total_gamma']:+.4f}  Fisher sigma {r['fisher_sigma_gamma']:.4f}", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
