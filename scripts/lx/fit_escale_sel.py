"""Selection-modelled Lx-M fits of the eSCALE x DES Y3 sample with the three frameworks (2026-10-21).

Same relation, pivots, concentrations, miscentring and two-halo term as fit_escale.py, but each cluster's mass is
no longer the deterministic inverse of the relation.  It is the distribution (destacking.selection.MassPrior)

    w_ck(theta) = P(log M_k | F_hat_c, sigma_F,c, z_c, I, theta)
                propto dn/dlog M (M_k, z_c) int dlog Lx N(log Lx; <log Lx | M_k, z_c>, sigma_int) N(F_hat_c; Lx/kappa_c, sigma_F,c)
                       [x P(optical detection | M_k, z_c, F_hat_c)   (option B, --selection AB)]

with sigma_int free, the measured 0.6-2.3 keV flux and its error, and each cluster's own luminosity-to-flux
conversion kappa_c = Lx/F.  The X-ray S/N >= 2 cut acts on the measured flux and cancels (option A).
--selection AB adds the ASSUMED optical stage of notebooks/lx_selection_modelling.ipynb (richness-mass relation
log10 lambda = 1.0 + 0.8 (log M - 14.1), 0.2 dex scatter, correlation 0.3 with Lx, erf threshold at
log10 lambda = 0.8, width 0.1, 0.1 dex richness error): a sensitivity test until the real optical selection exists.

  joint    ln L = sum_c ln sum_k w_ck L_c(Delta Sigma_c | M_k)        (per-bin Sellentin-Heavens, as in fit_escale.py)
  forward  Delta Sigma_s = sum_c K_c sum_k w_ck Delta Sigma_ck / sum_c K_c   (SH, N = members)
  twostep  per-stack masses of fit_escale.py twostep, compared with the lensing-weighted mean mass the model
           predicts for each stack, log10[ sum_c K_c sum_k w_ck 10^M_k / sum_c K_c ] (K_c at ~1 Mpc), Gaussian.
Free: alpha, beta, gamma, sigma_int (+ f_mis, sigma_off for joint/forward).

usage: python fit_escale_sel.py MODE --selection {A,AB} [--subset {all,p95,p90}] [--data PATH_TO_MOCK.npz] [--tag T]
Output: ~/destacking_data/lx_splus_erosita_des/fits_sel/<mode>_<selection>_<subset>[_<tag>].npz
"""
import argparse, json, os, pickle, sys, time
import numpy as np
from scipy import stats
from astropy.table import Table
from destacking import lx
lx.set_cosmology()
import xlensing.model as xm
from destacking.model import _nfw_components
from destacking.selection import Population, Relation, ObservableModel, Stage, Selection, MassPrior, erf_completeness

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
M = f"{D}/measurement"
OUT = f"{D}/fits_sel"
PIV = lx.Pivots(43.26, 14.1, 0.18)
GRID = np.linspace(11.8, 15.8, 61)
SOFF = np.linspace(0.01, 1.0, 34)
OPT = dict(lam0=1.0, b_lam=0.8, sig_lam=0.2, rho=0.3, x50=0.8, width=0.1, lam_err=0.1)
PRI = {"alpha": stats.uniform(-2, 4), "beta": stats.uniform(0.3, 2.7), "gamma": stats.uniform(-15, 30),
       "sig_int": stats.uniform(0.02, 0.98), "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0.01, 0.99)}
NPOOL = int(os.environ.get("NPOOL", "15"))


# ── selection model ──────────────────────────────────────────────────────────
def make_prior(selection, z_range=(0.08, 0.25)):
    rel_lx = Relation("logLx", lambda m, z, t: lx.lx_from_mass(m, z, t["alpha"], t["beta"], t["gamma"], PIV),
                      lambda t: t["sig_int"])
    rel_lam = Relation("logLam", lambda m, z, t: OPT["lam0"] + OPT["b_lam"] * (m - PIV.log_M), lambda t: OPT["sig_lam"])
    om = ObservableModel([rel_lx, rel_lam], corr=lambda t: np.array([[1, OPT["rho"]], [OPT["rho"], 1]]))
    xray = Stage("X-ray flux S/N >= 2", ("flux_obs", "flux_err"),
                 lambda d: (np.asarray(d["flux_obs"]) >= 2 * np.asarray(d["flux_err"])).astype(float))
    stages = [xray] + ([erf_completeness("optical detection", "logLam_obs", OPT["x50"], OPT["width"])]
                       if selection == "AB" else [])

    def flux_meas(true, d):
        F = np.asarray(d["flux_obs"])[:, None, None]; s = np.asarray(d["flux_err"])[:, None, None]
        k = np.asarray(d["flux_conv"])[:, None, None]
        return -0.5 * ((F - 10 ** true / k) / s) ** 2

    return MassPrior(Population(1000, z_range), om, Selection(stages), conditioned=["logLx"], logM_grid=GRID,
                     gh_nodes=12, measurement={"logLx": flux_meas}, known_extra=("flux_obs", "flux_err", "flux_conv"))


def weights(mp, xdata, p):
    th = dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"], sig_int=p["sig_int"])
    return mp.weights(xdata, th, lat_err={"logLam": OPT["lam_err"]})


# ── data ─────────────────────────────────────────────────────────────────────
def load_real(subset):
    cat = Table.read(f"{D}/escale_des_sample.fits")
    raw = Table.read(f"{D}/gas_mass_uniqueIDs_AME_r200.fits")
    row = {int(c): k for k, c in enumerate(raw["CID6722"])}
    ix = np.array([row[int(c)] for c in cat["CID"]])
    F, eF = np.asarray(raw["Flux0623"], float)[ix], np.asarray(raw["eFlux0623"], float)[ix]
    keep = np.ones(len(cat), bool) if subset == "all" else np.asarray(cat[subset.upper()], bool)
    a = np.load(f"{M}/joint6.npz"); b = np.load(f"{M}/boosts.npz")
    xdata = {"z": np.asarray(cat["Z"], float), "flux_obs": F, "flux_err": eF,
             "flux_conv": np.asarray(cat["LX"], float) / F, "logLx_obs": np.log10(np.asarray(cat["LX"], float))}
    joint = dict(d=a["data_vectors"] * b["boost6"], v=np.array([np.diag(c) for c in a["cov_matrices"]]) * b["boost6"] ** 2,
                 n=a["n_k"].astype(float), fl=a["floored"], R=a["radii"])
    st = pickle.load(open(f"{M}/stacks.pkl", "rb")); W = np.load(f"{M}/weights.npz")
    idrow = {int(i): k for k, i in enumerate(cat["ID"])}
    stacks = [dict(name=nm, d=np.asarray(s["sigmas"], float), C=np.asarray(s["sigmas_cov"], float), n=s["n_stacked"],
                   rows=np.array([idrow[int(i)] for i in s["ids"]])) for nm, s in sorted(st["stacks"].items())]
    ts = json.load(open(f"{D}/fits/twostep.json"))["stacks"]
    for s in stacks:
        s["logM_fit"] = ts[s["name"]]["logM"]
    if subset != "all":   # stacked frameworks need fresh stacks for a subset: only the joint fit is subset-aware
        stacks = None
    return dict(x={k: v[keep] for k, v in xdata.items()}, joint={k: (v[keep] if k != "R" else v) for k, v in joint.items()},
                stacks=stacks, K16=W["K"], R16=st["radii"], keep=keep, N=int(keep.sum()))


def load_mock(path):
    m = np.load(path, allow_pickle=True)
    x = {k: m[f"x_{k}"] for k in ("z", "flux_obs", "flux_err", "flux_conv", "logLx_obs")}
    joint = dict(d=m["j_d"], v=m["j_v"], n=m["j_n"], fl=m["j_fl"], R=m["j_R"])
    stacks = [dict(name=str(nm), d=d, C=C, n=int(n), rows=r, logM_fit=list(lf)) for nm, d, C, n, r, lf in
              zip(m["s_names"], m["s_d"], m["s_C"], m["s_n"], m["s_rows"], m["s_logM_fit"])]
    return dict(x=x, joint=joint, stacks=stacks, K16=m["K16"], R16=m["R16"], keep=np.ones(len(x["z"]), bool),
                N=len(x["z"]), two_halo=bool(m["two_halo"]))


# ── profile tables on the mass grid ──────────────────────────────────────────
def profile_tables(z, R, two_halo=True):
    """cen (N,K,B), mis (S,N,K,B) on the sigma_off grid, th (N,K,B): one-halo centred / miscentred and two-halo."""
    N, K = len(z), len(GRID)
    Z = np.repeat(z, K); LM = np.tile(GRID, N)
    c = lx.diemer19_c200c(LM, Z)
    Mx, cx = lx.to_xlensing_mass(10 ** LM, c)
    cen, _ = _nfw_components(Z, Mx, cx, R, SIGMA_OFF=np.full(len(Z), 0.1), bin_nodes=0)
    mis = np.empty((len(SOFF), N, K, len(R)), np.float32)
    for j, so in enumerate(SOFF):
        _, m_ = _nfw_components(Z, Mx, cx, R, SIGMA_OFF=np.full(len(Z), so), bin_nodes=0)
        mis[j] = m_.reshape(N, K, -1)
    th = np.zeros((N, K, len(R)), np.float32)
    if two_halo:
        th = (xm.Bias(Z, Mx)[:, None] * (xm.Delta_Sigma_l(Z, R) / 1e12)).reshape(N, K, -1)
    return cen.reshape(N, K, -1).astype(np.float32), mis, th.astype(np.float32)


def mis_at(mis, so):
    j = np.clip(np.searchsorted(SOFF, so) - 1, 0, len(SOFF) - 2)
    t = (so - SOFF[j]) / (SOFF[j + 1] - SOFF[j])
    return (1 - t) * mis[j] + t * mis[j + 1]


# ── likelihoods ──────────────────────────────────────────────────────────────
class Joint:
    def __init__(self, data, selection):
        self.x = data["x"]; j = data["joint"]
        self.d, self.v, self.n, self.fl = j["d"], j["v"], j["n"], j["fl"]
        self.ok = np.isfinite(self.d) & np.isfinite(self.v) & (self.v > 0)
        self.sh = self.ok & ~self.fl & (self.n >= 2); self.ga = self.ok & ~self.sh
        self.d0 = np.where(self.ok, self.d, 0.0); self.v0 = np.where(self.ok, self.v, 1.0)
        self.det = selection == "det"
        self.R, self.two_halo = j["R"], data.get("two_halo", True)
        if self.det:
            self.ds_l = xm.Delta_Sigma_l(self.x["z"], self.R) / 1e12
        else:
            self.cen, self.mis, self.th = profile_tables(self.x["z"], j["R"], self.two_halo)
            self.mp = make_prior(selection)
        self.N = len(self.x["z"])

    def loglike(self, p):
        if self.det:
            lm = lx.mass_from_lx(self.x["logLx_obs"], self.x["z"], p["alpha"], p["beta"], p["gamma"], PIV)
            ds = lx.nfw_profiles_lx(self.x["z"], lm, self.R, f_mis=p["f_mis"], sigma_off=p["sigma_off"],
                                    two_halo=self.two_halo, ds_l=self.ds_l)[:, None, :]
            if not np.all(np.isfinite(ds[self.ok[:, None, :]])):
                return -np.inf
            w = np.ones((self.N, 1))
        else:
            w = weights(self.mp, self.x, p)
            ds = (1 - p["f_mis"]) * self.cen + p["f_mis"] * mis_at(self.mis, p["sigma_off"]) + self.th     # (N,K,B)
        q = (self.d0[:, None, :] - ds) ** 2 / self.v0[:, None, :]
        ll = (-0.5 * np.where(self.sh[:, None, :], self.n[:, None, :] * np.log1p(q / np.maximum(self.n[:, None, :] - 1, 1)), 0).sum(-1)
              - 0.5 * np.where(self.ga[:, None, :], q, 0).sum(-1))                                      # (N,K)
        a = np.log(np.clip(w, 1e-300, None)) + ll
        mx = a.max(1, keepdims=True)
        tot = float(np.sum(mx[:, 0] + np.log(np.exp(a - mx).sum(1))))
        return tot if np.isfinite(tot) else -np.inf


class Forward:
    def __init__(self, data, selection, min_n=50):
        self.x = data["x"]
        self.st = [s for s in data["stacks"] if s["n"] >= min_n]
        self.K = data["K16"]
        for s in self.st:
            C = s["C"] + 1e-6 * np.eye(len(s["d"])) * np.mean(np.diag(s["C"]))
            s["Ci"] = np.linalg.inv(C)
        self.det = selection == "det"
        self.R, self.two_halo = data["R16"], data.get("two_halo", True)
        if self.det:
            self.ds_l = xm.Delta_Sigma_l(self.x["z"], self.R) / 1e12
        else:
            self.cen, self.mis, self.th = profile_tables(self.x["z"], data["R16"], self.two_halo)
            self.mp = make_prior(selection)
        self.N = len(self.x["z"])

    def loglike(self, p):
        if self.det:
            lm = lx.mass_from_lx(self.x["logLx_obs"], self.x["z"], p["alpha"], p["beta"], p["gamma"], PIV)
            dsw = lx.nfw_profiles_lx(self.x["z"], lm, self.R, f_mis=p["f_mis"], sigma_off=p["sigma_off"],
                                     two_halo=self.two_halo, ds_l=self.ds_l)
        else:
            w = weights(self.mp, self.x, p)
            ds = (1 - p["f_mis"]) * self.cen + p["f_mis"] * mis_at(self.mis, p["sigma_off"]) + self.th
            dsw = np.einsum("nk,nkb->nb", w, ds)                                                      # (N,B)
        ll = 0.0
        for s in self.st:
            r = s["rows"]; mod = (self.K[r] * dsw[r]).sum(0) / self.K[r].sum(0)
            res = s["d"] - mod
            if not np.all(np.isfinite(res)):
                return -np.inf
            ll -= 0.5 * s["n"] * np.log1p((res @ s["Ci"] @ res) / (s["n"] - 1))
        return float(ll) if np.isfinite(ll) else -np.inf


class TwoStep:
    """Per-stack fitted masses vs the model's lensing-weighted mean stack mass."""
    def __init__(self, data, selection):
        self.x = data["x"]; self.st = data["stacks"]; self.K = data["K16"][:, 10]          # K_c at R ~ 1 Mpc
        self.y = np.array([s["logM_fit"][1] for s in self.st])
        self.e = np.array([(s["logM_fit"][2] - s["logM_fit"][0]) / 2 for s in self.st])
        self.mp = make_prior(selection)

    def loglike(self, p):
        w = weights(self.mp, self.x, p)
        Mm = w @ 10 ** GRID
        pred = np.array([np.log10((self.K[s["rows"]] * Mm[s["rows"]]).sum() / self.K[s["rows"]].sum()) for s in self.st])
        return float(-0.5 * np.sum(((self.y - pred) / self.e) ** 2))


def twostep_det(data, tag):
    """Deterministic two-step: WLS of the per-stack masses on <log Lx/E> and the redshift term (as fit_escale.py)."""
    st = data["stacks"]; x = data["x"]
    lm = np.array([s["logM_fit"][1] for s in st]); e = np.array([(s["logM_fit"][2] - s["logM_fit"][0]) / 2 for s in st])
    z = np.array([x["z"][s["rows"]].mean() for s in st])
    llx = np.array([np.mean(x["logLx_obs"][s["rows"]] - np.log10(lx.efunc(x["z"][s["rows"]]))) for s in st])
    y = lm + np.log10(lx.efunc(z)) - PIV.log_M
    X = np.column_stack([np.ones(len(y)), llx - PIV.log_L, np.log10((1 + z) / (1 + PIV.z))]); Wt = 1 / e ** 2
    cov = np.linalg.inv(X.T @ (X * Wt[:, None])); b = cov @ X.T @ (Wt * y)
    dr = np.random.default_rng(1).multivariate_normal(b, cov, 200000)
    conv = np.column_stack([-dr[:, 0] / dr[:, 1], 1 / dr[:, 1], -dr[:, 2] / dr[:, 1]])
    q = {n: np.percentile(conv[:, j], [16, 50, 84]).tolist() for j, n in enumerate(["alpha", "beta", "gamma"])}
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(lx_given_m=q, chi2=float(np.sum(Wt * (y - X @ b) ** 2)), dof=len(y) - 3), open(f"{OUT}/{tag}.json", "w"), indent=1)
    print(f"{tag}: N={data['N']}")
    for n, (lo, me, hi) in q.items():
        print(f"  {n:10s} {me:+.3f} +{hi - me:.3f}/-{me - lo:.3f}", flush=True)


def run(mode, selection, data, tag):
    if mode == "twostep" and selection == "det":
        return twostep_det(data, tag)
    names = ["alpha", "beta", "gamma"] + (["sig_int"] if selection != "det" else []) + (["f_mis", "sigma_off"] if mode != "twostep" else [])
    L = {"joint": Joint, "forward": Forward, "twostep": TwoStep}[mode](data, selection)
    from nautilus import Prior, Sampler
    pr = Prior()
    for n in names:
        pr.add_parameter(n, dist=PRI[n])
    p0 = dict(alpha=0.3, beta=1.1, gamma=0.0, sig_int=0.2, f_mis=0.4, sigma_off=0.2)
    if mode == "twostep" and selection == "det":
        return
    t = time.time(); [L.loglike(p0) for _ in range(3)]
    print(f"{tag}: N={data['N']}, one likelihood call {1e3 * (time.time() - t) / 3:.0f} ms", flush=True)
    t0 = time.time()
    smp = Sampler(pr, L.loglike, n_live=1000, pool=NPOOL, seed=1)
    smp.run(verbose=True, n_eff=3000)
    X, lw, ll = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{tag}.npz", samples=X, weights=w, log_l=ll, param_order=np.array(names), log_z=smp.log_z,
             n_clusters=data["N"], selection=selection, mode=mode, opt=json.dumps(OPT))
    print(f"{tag}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for j, n in enumerate(names):
        o = np.argsort(X[:, j]); lo, me, hi = np.interp([.16, .5, .84], np.cumsum(w[o]), X[o, j])
        print(f"  {n:10s} {me:+.3f} +{hi - me:.3f}/-{me - lo:.3f}", flush=True)
    # weight at the grid edges for the posterior median (sanity: the grid must contain the prior)
    pm = {n: float(np.interp(0.5, np.cumsum(w[np.argsort(X[:, j])]), np.sort(X[:, j]))) for j, n in enumerate(names)}
    pm.setdefault("f_mis", 0.0); pm.setdefault("sigma_off", 0.1)
    if selection == "det":
        return
    ww = weights(L.mp, L.x, pm)
    print(f"  max prior weight in the lowest/highest 2 grid points: {ww[:, :2].sum(1).max():.2e} / {ww[:, -2:].sum(1).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["joint", "forward", "twostep"])
    ap.add_argument("--selection", choices=["det", "A", "AB"], required=True)
    ap.add_argument("--subset", default="all", choices=["all", "p95", "p90"])
    ap.add_argument("--data", default=None, help="mock .npz (see make_escale_mock.py); default: real data")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    data = load_mock(a.data) if a.data else load_real(a.subset)
    if a.mode != "joint" and data["stacks"] is None:
        raise SystemExit("stacked frameworks need stacks built for this subset (only --subset all has them)")
    tag = f"{a.mode}_{a.selection}_{a.subset}" + (f"_{a.tag}" if a.tag else "")
    run(a.mode, a.selection, data, tag)
