"""eSCALE-like mock with REAL DES Y3 lensing noise, for validating the Lx-M fits (2026-10-21).

1. Haloes from the Tinker+08 mass function (M200c, flat LCDM 70/0.3) over 0.08 < z < 0.25.
2. True Lx from the fiducial relation (Pederneiras+25 eRASS1 at our pivots: alpha 0.47, beta 1.11, gamma 0,
   sigma_int 0.17 dex) and a correlated optical richness (the assumed model of fit_escale_sel.OPT).
3. Measured 0.6-2.3 keV flux = Lx / kappa(z) + Gaussian noise, with flux errors drawn from the real eSCALE clusters
   inside DES; selection: flux S/N >= 2 and the erf optical threshold on measured richness.
4. Lensing: mock clusters are paired, by redshift rank and without reuse, with real eSCALE clusters; each mock
   cluster takes its partner's background sources (positions, Sigma_crit, weights, responses) and its z.  The real
   cross shear of every source is the noise; the signal is injected as e_t = (1+m) DS_true(R) / Sigma_crit + e_x.
   DS_true: NFW with Diemer19 c200c, miscentring mixture (f_mis 0.3, sigma_off 0.15 Mpc), two-halo term.
5. Measured exactly like the data: per-cluster 6-bin galaxy bootstrap with the analytic floor (joint), and 3 x 5
   z x Lx/E stacks on 16 bins with a 500-draw cluster bootstrap.  No boost: the mock has no member contamination.
6. Per-stack single-halo fits (two-step first step), as in fit_escale.py.

usage: python make_escale_mock.py SEED [AREA_DEG2]
writes ~/destacking_data/lx_splus_erosita_des/mocks/mock_<SEED>.npz (fit with fit_escale_sel.py --data ...)
"""
import os, pickle, sys, time, json
import numpy as np
from astropy.table import Table
from destacking import lx
lx.set_cosmology()
import xlensing.cosmo as xc
from xlensing.data import bootstrap_signal
from destacking.data import analytic_signal
from destacking.selection import Population

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_escale_sel as fs
import build_escale_stacks as bes

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
TRUTH = dict(alpha=0.47, beta=1.11, gamma=0.0, sig_int=0.17, f_mis=0.3, sigma_off=0.15)
R6 = np.logspace(-0.6, 0.6, 6); d6 = np.diff(np.log10(R6))[0]
e6 = np.logspace(np.log10(R6[0]) - d6 / 2, np.log10(R6[-1]) + d6 / 2, 7); BL6 = np.column_stack([e6[:-1], e6[1:]])


def main():
    seed = int(sys.argv[1]); area = float(sys.argv[2]) if len(sys.argv) > 2 else 830.0
    rng = np.random.default_rng(seed); t0 = time.time()
    cat = Table.read(f"{D}/escale_des_sample.fits")
    raw = Table.read(f"{D}/gas_mass_uniqueIDs_AME_r200.fits")
    row = {int(c): k for k, c in enumerate(raw["CID6722"])}
    ix = np.array([row[int(c)] for c in cat["CID"]])
    F_r, eF_r = np.asarray(raw["Flux0623"], float)[ix], np.asarray(raw["eFlux0623"], float)[ix]
    DL_r = xc.cosmology.luminosity_distance(np.asarray(cat["Z"])).to_value("cm")
    K_MED = np.median(np.asarray(cat["LX"]) / (4 * np.pi * DL_r ** 2 * F_r))
    stakes = pickle.load(open(f"{D}/measurement/stakes.pkl", "rb"))

    # 1-3: population, observables, selection
    pop = Population(area, (0.08, 0.25))
    s = pop.sample(rng=rng, logM_min=11.8)
    N = len(s["z"])
    mu_l = lx.lx_from_mass(s["logM"], s["z"], TRUTH["alpha"], TRUTH["beta"], TRUTH["gamma"], fs.PIV)
    mu_lam = fs.OPT["lam0"] + fs.OPT["b_lam"] * (s["logM"] - fs.PIV.log_M)
    cov = np.array([[TRUTH["sig_int"] ** 2, fs.OPT["rho"] * TRUTH["sig_int"] * fs.OPT["sig_lam"]],
                    [fs.OPT["rho"] * TRUTH["sig_int"] * fs.OPT["sig_lam"], fs.OPT["sig_lam"] ** 2]])
    dev = rng.multivariate_normal([0, 0], cov, N)
    logLx, loglam = mu_l + dev[:, 0], mu_lam + dev[:, 1]
    conv = 4 * np.pi * xc.cosmology.luminosity_distance(s["z"]).to_value("cm") ** 2 * K_MED
    sF = rng.choice(eF_r, N)
    Fo = 10 ** logLx / conv + sF * rng.standard_normal(N)
    lam_o = loglam + fs.OPT["lam_err"] * rng.standard_normal(N)
    from scipy.special import erf
    p_opt = 0.5 * (1 + erf((lam_o - fs.OPT["x50"]) / (np.sqrt(2) * fs.OPT["width"])))
    sel = (Fo >= 2 * sF) & (rng.random(N) < p_opt)
    idx = np.flatnonzero(sel)
    print(f"parent {N}, selected {len(idx)} (real sample {len(cat)}) ({time.time()-t0:.0f}s)", flush=True)

    # 4: pair with real clusters by redshift rank, without reuse
    n = min(len(idx), len(cat))
    idx = rng.choice(idx, n, replace=False) if len(idx) > n else idx
    real_rows = np.sort(rng.choice(len(cat), n, replace=False)) if len(cat) > n else np.arange(len(cat))
    mo = idx[np.argsort(s["z"][idx])]
    rr = real_rows[np.argsort(np.asarray(cat["Z"])[real_rows])]
    zr = np.asarray(cat["Z"], float)[rr]
    logM_true = s["logM"][mo]
    print(f"paired {n}: median |z_mock - z_template| = {np.median(np.abs(s['z'][mo] - zr)):.4f}", flush=True)

    # true profiles on a fine radial grid, then per source
    Rf = np.logspace(np.log10(0.09), np.log10(5.4), 160)
    ds_f = lx.nfw_profiles_lx(zr, logM_true, Rf, f_mis=TRUTH["f_mis"], sigma_off=TRUTH["sigma_off"], two_halo=True)
    new_stakes, DV, COV, NK, FL = [], np.full((n, 6), np.nan), np.tile(np.eye(6), (n, 1, 1)), np.zeros((n, 6)), np.zeros((n, 6), bool)
    w_all = np.concatenate([stakes[int(cat["ID"][r])][3].astype(float) for r in rr])
    ex_all = np.concatenate([stakes[int(cat["ID"][r])][2].astype(float) for r in rr])
    sige = float(np.sqrt(np.sum(w_all * ex_all ** 2) / np.sum(w_all)) * 1.0)
    for j, r in enumerate(rr):
        st = stakes[int(cat["ID"][r])].astype(float).copy()
        dsg = np.interp(np.log(st[4]), np.log(Rf), ds_f[j])
        st[1] = (1 + st[5]) * dsg / st[0] + st[2]
        new_stakes.append(st)
        dv, cv, _, _ = bootstrap_signal(st, BL6, Nboot=200)
        _, an = analytic_signal(st, BL6, sige, min_gal=1); an = np.diag(an); dd = np.diag(cv).copy()
        inner = BL6[:, 0] < 0.5; ok = np.isfinite(dd) & np.isfinite(an) & inner; fl = ok & (an > dd)
        cv = cv.copy(); cv[np.diag_indices_from(cv)] = np.where(fl, an, dd)
        val = np.isfinite(dv); cv = np.where(np.isfinite(cv), cv, 0.0); cv[~val, ~val] = 1.0
        DV[j], COV[j], FL[j] = dv, cv, fl
        NK[j] = ((st[4][:, None] > BL6[:, 0]) & (st[4][:, None] < BL6[:, 1])).sum(0)
    print(f"per-cluster measurement done ({time.time()-t0:.0f}s); shape-noise sigma_e (cross shear) = {sige:.3f}", flush=True)

    # 5: stacks without boost (16 bins, cluster bootstrap of sum num / sum den)
    R16, BL16 = bes.R16, bes.BL16
    num = np.zeros((n, 16)); den = np.zeros((n, 16))
    for j, st in enumerate(new_stakes):
        ib = (st[4][:, None] > BL16[:, 0]) & (st[4][:, None] < BL16[:, 1])
        num[j] = (st[1] * st[3] / st[0]) @ ib; den[j] = ((1 + st[5]) * st[3] / st[0] ** 2) @ ib
    x = dict(z=zr, flux_obs=Fo[mo], flux_err=sF[mo], flux_conv=(4 * np.pi * xc.cosmology.luminosity_distance(zr).to_value("cm") ** 2 * K_MED),
             logLx_obs=np.log10(np.clip(Fo[mo], 1e-30, None) * conv[mo]))
    x["flux_conv"] = 10 ** x["logLx_obs"] / x["flux_obs"]
    logLxE = x["logLx_obs"] - np.log10(lx.efunc(zr))
    zb, lb, _, _ = bes.assign_bins(zr, logLxE)
    s_names, s_d, s_C, s_n, s_rows = [], [], [], [], []
    stacks_pkl = {}
    for i in range(3):
        for k in range(5):
            m = np.flatnonzero((zb == i) & (lb == k))
            boot = np.array([num[b].sum(0) / den[b].sum(0) for b in (rng.choice(m, len(m)) for _ in range(500))])
            d = num[m].sum(0) / den[m].sum(0); C = np.cov(boot.T)
            nm = f"z{i}_lx{k}"
            s_names.append(nm); s_d.append(d); s_C.append(C); s_n.append(len(m)); s_rows.append(m)
            stacks_pkl[nm] = dict(sigmas=d, sigmas_cov=C, n_stacked=len(m), z_mean=float(zr[m].mean()), logLxE_mean=float(logLxE[m].mean()))
    print(f"stacks done ({time.time()-t0:.0f}s): N per stack {s_n}", flush=True)

    # 6: per-stack fits (two-step first step), reusing fit_escale.fit_one_stack on a mock stacks file
    out_dir = f"{D}/mocks"; os.makedirs(out_dir, exist_ok=True)
    tmpdir = f"{out_dir}/stacks_{seed}"; os.makedirs(tmpdir, exist_ok=True)
    pickle.dump(dict(stacks=stacks_pkl, radii=R16), open(f"{tmpdir}/stacks.pkl", "wb"))
    import fit_escale as fe
    fe.M = tmpdir
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPOOL", "15"))) as pool:
        res = dict(pool.map(fe.fit_one_stack, s_names))
    s_logM_fit = [res[nm]["logM"] for nm in s_names]
    print(f"per-stack fits done ({time.time()-t0:.0f}s)", flush=True)

    np.savez(f"{out_dir}/mock_{seed}.npz", **{f"x_{k}": v for k, v in x.items()},
             j_d=DV, j_v=np.array([np.diag(c) for c in COV]), j_n=NK, j_fl=FL, j_R=R6,
             s_names=np.array(s_names), s_d=np.array(s_d), s_C=np.array(s_C), s_n=np.array(s_n),
             s_rows=np.array(s_rows, dtype=object), s_logM_fit=np.array(s_logM_fit), K16=den, R16=R16, two_halo=True,
             logM_true=logM_true, truth=json.dumps(TRUTH), opt=json.dumps(fs.OPT), seed=seed, area=area)
    print(f"wrote {out_dir}/mock_{seed}.npz ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
