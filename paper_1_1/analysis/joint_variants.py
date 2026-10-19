"""Full-sample (no bootstrap) real-data joint fits under data/model variants (2026-10-07).
Same setup as the paper's joint fit (v9c: f=1.1 6-bin bootstrap data with floor, 2-way power-law boost, two-halo term, bin-centre model,
c0 ~ N(4.5,5) truncated at 0, flat miscentring priors). Each variant is compared with the full-sample baseline 'full' (same seed),
so the shifts are paired; their size is judged against the bootstrap-median widths of the paper.
  python joint_variants.py VARIANT [VARIANT ...]
  full        baseline
  dmle0       include the clusters with Delta m <= 0, with Delta m floored at 0.05 mag                     (minor 10)
  dm2sig      drop clusters with Delta m < 2 sigma(Delta m)                                                  (minor 10)
  dmmis       f_mis free per magnitude-gap tercile (terciles within each redshift-richness cell, as the stacks)   (MC3)
  dmsoff      f_mis and sigma_off free per tercile                                                            (MC3)
  rshear      reduced-shear factor on the model                                                               (MC7)
  zbinK       clusters in redshift bin K = 0,1,2 (0.1-0.3-0.5-0.7) only, beta_Mlambda = beta_cm = 0           (MC7)
  boostNN     boost power laws refit to draw NN of the 2-way stacked boost factors from their covariance     (MC7)
  pivot       samples of 'full' re-expressed at M_piv = 10^14.3, Delta m_piv = 0.63 (no refit; correlations only) (MC6a)
Output: posteriors/joint_full_<variant>.npz
"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, sys, pickle, importlib.util, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from scipy.optimize import curve_fit
from astropy.table import Table
from nautilus import Prior, Sampler
from destacking.dmform import DM_FORM, dm_regressor
assert DM_FORM == 'linear', 'v1.1 runs need DESTACKING_DM_FORM=linear'
from destacking.fitting import log_likelihood_combined as L, cholog_gaussian
from destacking.model import forward_model_combined
from destacking.relations import mass_lambda_McClintock18
from cluster_toolkit.boostfactors import boost_powerlaw_at_R
RD = os.environ.get("SH_RD", f"{DATA}/real_data"); OUT = os.environ.get("SH_OUT", f"{DATA}/output/v1.1/posteriors")
LIK = os.environ.get("JV_LIK", "gauss")   # gauss | sh (per-bin Sellentin & Heavens, see joint_sh.py)
PFX = "joint_full_sh_" if LIK == "sh" else "joint_full_"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
AUX = os.path.expanduser(f"{AUX}"); sys.path.insert(0, AUX)
R6 = np.logspace(-0.6, 0.6, 6)
NAMES = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
PRI = {"c0": stats.truncnorm((0 - 4.5) / 5, np.inf, loc=4.5, scale=5), "alpha_cm": stats.norm(-0.1, 0.5), "beta_cm": stats.norm(-0.4, 2),
       "gamma_cm": stats.norm(0, 1), "log_M_norm": stats.norm(0, 0.5), "alpha_mr": stats.norm(1.356, 0.5), "beta_mr": stats.norm(-0.3, 2),
       "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0, 1)}
Z_EDGES = (0.1, 0.3, 0.5, 0.7)


def boost_fits_from_draw(seed):
    S = pickle.load(open(f"{RD}/wazp_stacked_stacks.pkl", "rb")); R = np.asarray(S["radii"], float); rng = np.random.default_rng(seed); out = {}
    for name, s in S["stacks"].items():
        B = np.asarray(s["boosts"], float); C = np.asarray(s["boosts_cov"], float)
        Bd = rng.multivariate_normal(B, C)
        try:
            p, _ = curve_fit(lambda r, b0, rs, a: boost_powerlaw_at_R(r, b0, rs, a), R, Bd, p0=(0.3, 1.0, -1.0),
                             bounds=((0.0, 0.01, -10.0), (20.0, 50.0, 10.0)), sigma=C, absolute_sigma=True, maxfev=20000)
            out[name] = {"B0": p[0], "Rs": p[1], "alpha": p[2], "failed": False}
        except Exception:
            out[name] = {"failed": True}
    return out


def load(variant):
    a = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/simulated_data.npz")
    co = np.load(f"{RD}/wazp_real_data_bootfloor_v4/simulation_g00_gamma+0.0/cluster_obs.npz", allow_pickle=True)
    cat = Table.read(f"{RD}/wazp_real_data.fits")
    dv, cov = a["data_vectors"].copy(), a["cov_matrices"].copy()
    BF = boost_fits_from_draw(int(variant[5:])) if variant.startswith("boost") else pickle.load(open(f"{RD}/wazp_boost_fits.pkl", "rb"))["boost_fits"]
    for i, nm in enumerate(co["stack_name"]):
        f = BF.get(str(nm)) if nm else None
        if f is not None and not f.get("failed"): b = boost_powerlaw_at_R(R6, f["B0"], f["Rs"], f["alpha"]); dv[i] *= b; cov[i] *= np.outer(b, b)
    T = Table({k: np.asarray(co[k], float) for k in ("z", "Lambda", "delta_m")}); dm = np.asarray(T["delta_m"]); err = np.asarray(cat["DELTA_M_I_ERR"], float)
    if variant == "dmle0":
        sel = np.isfinite(dm) & np.isfinite(dv).any(1)   # v1.1: linear in Delta m, no floor needed
    elif variant == "dm2sig":
        sel = np.isfinite(dm) & (dm > 0) & (dm >= 2 * err)
    else:
        sel = np.isfinite(dm) & (dm > 0)
    if variant.startswith("zbin"):
        k = int(variant[4:]); z = np.asarray(T["z"]); sel &= (z >= Z_EDGES[k]) & (z < Z_EDGES[k + 1])
    terc = None
    if variant in ("dmmis", "dmsoff"):
        spec = importlib.util.spec_from_file_location("s3", f"{AUX}/stacked_3way_dm_test.py"); s3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s3)
        zb, nb, db, _, _ = s3.assign_stack_bins_3way(np.asarray(cat["ZPHOT"]), np.asarray(cat["NGALS"]), np.asarray(cat["DELTA_M_I"]),
                                                     z_edges=s3.wsa.Z_EDGES, ng_edges=s3.wsa.NG_EDGES)
        db = np.asarray(db).copy(); dmc = np.asarray(cat["DELTA_M_I"], float); bad = (db < 0) & (dmc > 0)
        db[bad] = np.digitize(dmc[bad], np.percentile(dmc[dmc > 0], [100 / 3, 200 / 3]))   # z < 0.1: outside the stack cells, global terciles
        terc = db[sel]
    bc = np.load(f"{RD}/wazp_real_data_bootfloor_v4/bin_counts.npz")
    return T[sel], dv[sel], cov[sel], terc, bc["n_k"][sel].astype(float), bc["floored"][sel]


def run(variant):
    T, D, Cv, terc, NK, FL = load(variant); n = len(T); VAR = np.array([np.diag(c) for c in Cv])
    def sh_ll(mod):
        dat = np.isfinite(D) & np.isfinite(VAR) & (VAR > 0)
        if not np.all(np.isfinite(mod[dat])): return -np.inf
        r2 = (D - mod) ** 2; ok = dat
        q = np.where(ok, r2 / np.where(ok, VAR, 1.0), 0.0); shm = ok & ~FL & (NK >= 2); ga = ok & ~shm
        ll = -0.5 * np.sum(NK[shm] * np.log1p(q[shm] / (NK[shm] - 1))) - 0.5 * np.sum(q[ga])
        return float(ll) if np.isfinite(ll) else -np.inf
    names = list(NAMES); prior = dict(PRI); fixed = {}
    if variant.startswith("zbin"):
        fixed = {"beta_mr": 0.0, "beta_cm": 0.0}; names = [x for x in names if x not in fixed]
    if variant in ("dmmis", "dmsoff"):
        ex = [f"f_mis_t{t}" for t in range(3)] + ([f"sigma_off_t{t}" for t in range(3)] if variant == "dmsoff" else [])
        names = [x for x in names if not (x == "f_mis" or (variant == "dmsoff" and x == "sigma_off"))] + ex
        for e in ex: prior[e] = stats.uniform(0, 1)
    if variant == "rshear":
        from reduced_shear import kappa_factor
    z = np.asarray(T["z"]); lam = np.asarray(T["Lambda"]); ldm = dm_regressor(np.asarray(T["delta_m"]), 0.1)

    def loglike(q):
        p = dict(fixed, **q)
        if variant in ("dmmis", "dmsoff"):
            fm = np.array([p[f"f_mis_t{t}"] for t in range(3)])[terc]
            so = np.array([p[f"sigma_off_t{t}"] for t in range(3)])[terc] if variant == "dmsoff" else np.full(n, p["sigma_off"])
        else:
            fm = np.full(n, p["f_mis"]); so = np.full(n, p["sigma_off"])
        kw = dict(beta_cm=p["beta_cm"], log_M_norm=p["log_M_norm"], alpha_mr=p["alpha_mr"], beta_mr=p["beta_mr"], FMISS=fm, SIGMA_OFF=so,
                  two_halo=True, bin_nodes=0)
        if variant != "rshear" and LIK != "sh":
            return L(p["c0"], p["alpha_cm"], p["gamma_cm"], T, D, Cv, R6, **kw)
        if variant != "rshear":
            if p["c0"] <= 0 or np.any(so <= 0): return -np.inf
            try:
                mod = forward_model_combined(p["c0"], p["alpha_cm"], p["gamma_cm"], T, R6, **kw)
            except Exception:
                return -np.inf
            return sh_ll(mod) if np.all(np.isfinite(mod)) else -np.inf
        if p["c0"] <= 0 or np.any(so <= 0): return -np.inf
        try:
            mod = forward_model_combined(p["c0"], p["alpha_cm"], p["gamma_cm"], T, R6, **kw)
            logM = mass_lambda_McClintock18(lam, z, Flambda=p["alpha_mr"], Gz=p["beta_mr"]) + p["log_M_norm"]
            logc = np.log10(p["c0"]) + p["alpha_cm"] * (logM - 14.0) + p["beta_cm"] * np.log10((1 + z) / 1.35) + p["gamma_cm"] * ldm
            mod = mod * kappa_factor(z, 10 ** logM, 10 ** logc, R6, p["f_mis"], p["sigma_off"], f_sel=1.1)
            if not np.all(np.isfinite(mod)): return -np.inf
            if LIK == "sh": return sh_ll(mod)
            ll = cholog_gaussian(D - mod, Cv); return ll if np.isfinite(ll) else -np.inf
        except Exception:
            return -np.inf
    pr = Prior()
    for nm in names: pr.add_parameter(nm, dist=prior[nm])
    smp = Sampler(pr, loglike, n_live=1000, pool=int(os.environ.get("NPOOL", "14")), seed=1)
    smp.run(verbose=False, n_eff=3000)
    X, lw, ll = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    np.savez(f"{OUT}/{PFX}{variant}.npz", samples=X, weights=w, log_l=ll, param_order=np.array(names), log_z=smp.log_z, n_clusters=n, dm_form=DM_FORM)
    qf = lambda j: np.interp([0.16, 0.5, 0.84], np.cumsum(w[np.argsort(X[:, j])]), np.sort(X[:, j]))
    print(variant, f"N={n}", " ".join(f"{nm}={np.round(qf(j), 3).tolist()}" for j, nm in enumerate(names)), f"logZ={smp.log_z:.2f}", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        if os.path.exists(f"{OUT}/{PFX}{v}.npz"): print("cached", v); continue
        run(v)
