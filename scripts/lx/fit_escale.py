"""Deterministic Lx-M fits of the eSCALE x DES Y3 sample with the three frameworks (2026-10-21).

Relation (Pederneiras et al. 2025, Eq. 1), no scatter, inverted exactly for the mass of each cluster:

    log10(Lx E(z)^-1 / L_piv) = alpha + beta log10(M200c E(z) / M_piv) + gamma log10((1+z)/(1+z_piv))

Pivots at the sample medians: log L_piv = 43.26, log M_piv = 14.1, z_piv = 0.18.  Concentration: Diemer & Joyce
(2019) c200c(M200c, z), not fitted.  Miscentring: global f_mis, sigma_off (X-ray peak centres).  Two-halo term
included (real data).  Cosmology flat LCDM 70/0.3.  Free: alpha, beta, gamma, f_mis, sigma_off.

  python fit_escale.py joint      per-cluster fit (6 bins, 0.25-3.98 Mpc), each cluster's Delta Sigma and
                                  covariance boost-corrected with its stack's B(R); per-bin Sellentin-Heavens
                                  likelihood, Gaussian for bins whose variance is the shape-noise floor
                                  (paper 1.1's joint_sh.py)
  python fit_escale.py forward    forward-modelled stacks: stack model = sum_c K_c(R) DS_c(R) / sum_c K_c(R),
                                  Sellentin-Heavens likelihood with N = member clusters (16 bins, 0.16-3.98 Mpc)
  python fit_escale.py twostep    per-stack NFW fit (log M200c, f_mis, sigma_off; Diemer c), then weighted least
                                  squares of log10(M E/M_piv) on log10(<Lx/E>/L_piv) and log10((1+<z>)/(1+z_piv)),
                                  converted to (alpha, beta, gamma)
Output: ~/destacking_data/lx_splus_erosita_des/fits/{joint,forward,twostep}.npz / twostep.json
"""
import json, os, pickle, sys, time
import numpy as np
from scipy import stats
from astropy.table import Table
from destacking import lx
lx.set_cosmology()
import xlensing.model as xm
from nautilus import Prior, Sampler

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
M = f"{D}/measurement"
OUT = f"{D}/fits"
PIV = lx.Pivots(43.26, 14.1, 0.18)
NAMES = ["alpha", "beta", "gamma", "f_mis", "sigma_off"]
PRIORS = {"alpha": stats.uniform(-2, 4), "beta": stats.uniform(0.3, 2.7), "gamma": stats.uniform(-15, 30),
          "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0.01, 0.99)}
NPOOL = int(os.environ.get("NPOOL", "15"))


def load_sample():
    cat = Table.read(f"{D}/escale_des_sample.fits")
    a = np.load(f"{M}/joint6.npz")
    assert np.all(a["ids"] == np.asarray(cat["ID"]))
    return cat, a


def masses(p, logLx, z):
    return lx.mass_from_lx(logLx, z, p["alpha"], p["beta"], p["gamma"], PIV)


def quantiles(X, w, names):
    out = {}
    for j, n in enumerate(names):
        o = np.argsort(X[:, j]); out[n] = np.interp([0.16, 0.5, 0.84], np.cumsum(w[o]), X[o, j]).tolist()
    return out


def run_sampler(loglike, tag, extra=None):
    pr = Prior()
    for n in NAMES:
        pr.add_parameter(n, dist=PRIORS[n])
    t0 = time.time()
    smp = Sampler(pr, loglike, n_live=1000, pool=NPOOL, seed=1)
    smp.run(verbose=True, n_eff=3000)
    X, lw, ll = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    q = quantiles(X, w, NAMES)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{tag}.npz", samples=X, weights=w, log_l=ll, param_order=np.array(NAMES), log_z=smp.log_z,
             pivots=np.array([PIV.log_L, PIV.log_M, PIV.z]), **(extra or {}))
    print(f"{tag}: done in {(time.time() - t0) / 60:.1f} min")
    for n in NAMES:
        lo, me, hi = q[n]; print(f"  {n:10s} {me:+.3f} +{hi - me:.3f}/-{me - lo:.3f}")
    return q


# ── joint ────────────────────────────────────────────────────────────────────
class Joint:
    def __init__(self):
        cat, a = load_sample()
        b = np.load(f"{M}/boosts.npz"); assert np.all(b["ids"] == a["ids"])
        bst = b["boost6"]
        self.R = a["radii"]
        self.d = a["data_vectors"] * bst
        self.v = np.array([np.diag(c) for c in a["cov_matrices"]]) * bst ** 2
        self.n = a["n_k"].astype(float); self.fl = a["floored"]
        self.z = np.asarray(cat["Z"], float); self.llx = np.log10(np.asarray(cat["LX"], float))
        self.ds_l = xm.Delta_Sigma_l(self.z, self.R) / 1e12
        self.ok = np.isfinite(self.d) & np.isfinite(self.v) & (self.v > 0)
        self.sh = self.ok & ~self.fl & (self.n >= 2)
        self.ga = self.ok & ~self.sh
        self.N = len(self.z)

    def model(self, p):
        return lx.nfw_profiles_lx(self.z, masses(p, self.llx, self.z), self.R, f_mis=p["f_mis"],
                                  sigma_off=p["sigma_off"], two_halo=True, ds_l=self.ds_l)

    def loglike(self, p):
        m = self.model(p)
        if not np.all(np.isfinite(m[self.ok])):
            return -np.inf
        q = np.where(self.ok, (self.d - m) ** 2 / np.where(self.ok, self.v, 1.0), 0.0)
        ll = -0.5 * np.sum(self.n[self.sh] * np.log1p(q[self.sh] / (self.n[self.sh] - 1))) - 0.5 * np.sum(q[self.ga])
        return float(ll) if np.isfinite(ll) else -np.inf


# ── forward-modelled stacks ──────────────────────────────────────────────────
class Forward:
    def __init__(self, min_n=50):
        cat, _ = load_sample()
        st = pickle.load(open(f"{M}/stacks.pkl", "rb"))
        W = np.load(f"{M}/weights.npz"); assert np.all(W["ids"] == np.asarray(cat["ID"]))
        self.R = st["radii"]
        row = {int(i): k for k, i in enumerate(cat["ID"])}
        self.names, self.data, self.cinv, self.nst, self.rows = [], [], [], [], []
        for nm in sorted(st["stacks"]):
            s = st["stacks"][nm]
            if s["n_stacked"] < min_n:
                continue
            dd = np.asarray(s["sigmas"], float); C = np.asarray(s["sigmas_cov"], float)
            ok = np.isfinite(dd) & (np.diag(C) > 0)
            assert ok.all(), nm
            C = C + 1e-6 * np.eye(len(dd)) * np.mean(np.diag(C))
            self.names.append(nm); self.data.append(dd); self.cinv.append(np.linalg.inv(C)); self.nst.append(s["n_stacked"])
            self.rows.append(np.array([row[int(i)] for i in s["ids"]]))
        allr = np.unique(np.concatenate(self.rows))
        pos = {r: j for j, r in enumerate(allr)}
        self.mem = [np.array([pos[r] for r in rr]) for rr in self.rows]
        self.z = np.asarray(cat["Z"], float)[allr]; self.llx = np.log10(np.asarray(cat["LX"], float))[allr]
        self.K = W["K"][allr]
        self.ds_l = xm.Delta_Sigma_l(self.z, self.R) / 1e12

    def model(self, p):
        ds = lx.nfw_profiles_lx(self.z, masses(p, self.llx, self.z), self.R, f_mis=p["f_mis"],
                                sigma_off=p["sigma_off"], two_halo=True, ds_l=self.ds_l)
        return [(self.K[m] * ds[m]).sum(0) / self.K[m].sum(0) for m in self.mem]

    def loglike(self, p):
        mod = self.model(p); ll = 0.0
        for dd, ci, n, mm in zip(self.data, self.cinv, self.nst, mod):
            r = dd - mm
            if not np.all(np.isfinite(r)):
                return -np.inf
            ll -= 0.5 * n * np.log1p((r @ ci @ r) / (n - 1))
        return float(ll) if np.isfinite(ll) else -np.inf


# ── two-step ─────────────────────────────────────────────────────────────────
def fit_one_stack(name):
    lx.set_cosmology()
    import xlensing.model as xm2
    st = pickle.load(open(f"{M}/stacks.pkl", "rb")); s = st["stacks"][name]; R = st["radii"]
    d = np.asarray(s["sigmas"], float); C = np.asarray(s["sigmas_cov"], float)
    C = C + 1e-6 * np.eye(len(d)) * np.mean(np.diag(C)); Ci = np.linalg.inv(C); n = s["n_stacked"]
    z = float(s["z_mean"]); dsl = xm2.Delta_Sigma_l(np.array([z]), R) / 1e12
    pr = Prior(); pr.add_parameter("logM", dist=(12.5, 15.5)); pr.add_parameter("f_mis", dist=(0.0, 1.0))
    pr.add_parameter("sigma_off", dist=(0.01, 1.0))

    def ll(P):
        P = np.atleast_2d(P); k = len(P); zz = np.full(k, z)
        mod = lx.nfw_profiles_lx(zz, P[:, 0], R, f_mis=P[:, 1], sigma_off=P[:, 2], two_halo=True,
                                 ds_l=np.repeat(dsl, k, 0))
        r = d[None] - mod
        out = -0.5 * n * np.log1p(np.einsum("ni,ij,nj->n", r, Ci, r) / (n - 1))
        return np.where(np.isfinite(out), out, -1e30)

    smp = Sampler(pr, ll, n_live=1000, vectorized=True, pass_dict=False, seed=1)
    smp.run(verbose=False, n_eff=3000)
    X, lw, _ = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    q = quantiles(X, w, ["logM", "f_mis", "sigma_off"])
    return name, dict(q, z=z, logLxE=s["logLxE_mean"], n=n)


def twostep():
    from multiprocessing import Pool
    st = pickle.load(open(f"{M}/stacks.pkl", "rb"))
    names = sorted(st["stacks"])
    with Pool(min(NPOOL, len(names))) as pool:
        res = dict(pool.map(fit_one_stack, names))
    logM = np.array([res[k]["logM"][1] for k in names]); eM = np.array([(res[k]["logM"][2] - res[k]["logM"][0]) / 2 for k in names])
    z = np.array([res[k]["z"] for k in names]); llx = np.array([res[k]["logLxE"] for k in names])
    Ez = lx.efunc(z)
    y = logM + np.log10(Ez) - PIV.log_M
    X = np.column_stack([np.ones(len(y)), llx - PIV.log_L, np.log10((1 + z) / (1 + PIV.z))])
    Wt = 1 / eM ** 2
    cov = np.linalg.inv(X.T @ (X * Wt[:, None])); b = cov @ X.T @ (Wt * y)
    chi2 = float(np.sum(Wt * (y - X @ b) ** 2)); dof = len(y) - 3
    # M|Lx coefficients (A, B, C) -> Lx|M (alpha, beta, gamma): beta = 1/B, alpha = -A/B, gamma = -C/B
    draws = np.random.default_rng(1).multivariate_normal(b, cov, 200000)
    conv = np.column_stack([-draws[:, 0] / draws[:, 1], 1 / draws[:, 1], -draws[:, 2] / draws[:, 1]])
    qs = {n: np.percentile(conv[:, j], [16, 50, 84]).tolist() for j, n in enumerate(["alpha", "beta", "gamma"])}
    out = dict(stacks=res, A_B_C=b.tolist(), cov_ABC=cov.tolist(), chi2=chi2, dof=dof, lx_given_m=qs,
               pivots=[PIV.log_L, PIV.log_M, PIV.z])
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/twostep.json", "w"), indent=1, default=float)
    print(f"two-step: {len(names)} stacks, chi2/dof = {chi2:.1f}/{dof}")
    for k in names:
        r = res[k]; print(f"  {k}: N={r['n']} z={r['z']:.3f} <log Lx/E>={r['logLxE']:.2f} logM200c={r['logM'][1]:.3f} "
                          f"+{r['logM'][2]-r['logM'][1]:.3f}/-{r['logM'][1]-r['logM'][0]:.3f}  f_mis={r['f_mis'][1]:.2f}")
    print(f"  M|Lx: A={b[0]:+.3f}+/-{np.sqrt(cov[0,0]):.3f}  B={b[1]:+.3f}+/-{np.sqrt(cov[1,1]):.3f}  C={b[2]:+.3f}+/-{np.sqrt(cov[2,2]):.3f}")
    for n, (lo, me, hi) in qs.items():
        print(f"  {n:6s} {me:+.3f} +{hi - me:.3f}/-{me - lo:.3f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "joint":
        J = Joint(); print(f"joint: {J.N} clusters, {int(J.ok.sum())} bins with data")
        run_sampler(J.loglike, "joint", dict(n_clusters=J.N))
    elif mode == "forward":
        F = Forward(); print(f"forward: {len(F.names)} stacks, {len(F.z)} member clusters")
        run_sampler(F.loglike, "forward", dict(stack_names=np.array(F.names), n_stacked=np.array(F.nst)))
    elif mode == "twostep":
        twostep()
    else:
        raise SystemExit("mode: joint | forward | twostep")
