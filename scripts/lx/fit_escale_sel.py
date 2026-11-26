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
# Optical stage (option B).  Version 2 (2026-11-10), from L. Doubrawa (priv. comm., SCALE II):
#  - richness-mass relation for the M_r < -20 AME richness (linmix on the 95%-purity sample, with the catalogue's
#    L_X-based M200c): log10(lambda/11.51) = 0.006 + 0.350 log10(M/1.53e14), scatter 0.184 dex;
#  - completeness of X-ray-detected systems against log10(lambda) from SCALE II Fig. 8 (purple curve, 1 - y),
#    fitted (unweighted; counts per bin unclear) with floor + (1 - floor) Phi((log lambda - x50)/width).
# rho (correlation of the lambda and L_X scatters) and lam_err (AME richness error, dex) remain assumptions.
# Version 1 (placeholder, all fits before 2026-11-10): lam0=1.0 at logM_piv=14.1, b_lam=0.8, sig_lam=0.2, rho=0.3,
# x50=0.8, width=0.1, floor=0, lam_err=0.1.
OPT = dict(version=2, lam0=np.log10(11.51) + 0.006, logM_piv=np.log10(1.53e14), b_lam=0.350, sig_lam=0.184, rho=0.3,
           x50=1.307, width=0.307, floor=0.245, lam_err=0.1, lam_min=None, w_min=0.1)
# lam_min (version 2b, 2026-11-10): Fig. 8 is a completeness RELATIVE to X-ray systems identified by some optical
# catalogue, so its low-richness floor is not an absolute detection probability. With lam_min set, the stage is
#   P = Phi((log lam - log lam_min)/w_min) * [floor + (1 - floor) Phi((log lam - x50)/width)],
# i.e. completeness goes to zero below a minimum richness (SCALE II matching needs sum P_mem >~ 2-5).
PRI = {"alpha": stats.uniform(-2, 4), "beta": stats.uniform(0.3, 2.7), "gamma": stats.uniform(-15, 30),
       "sig_int": stats.uniform(0.02, 0.98), "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0.01, 0.99),
       "q95": stats.uniform(0, 1), "q90": stats.uniform(0, 1), "q80": stats.uniform(0, 1), "q0": stats.uniform(0, 1)}
NPOOL = int(os.environ.get("NPOOL", "15"))
FIX = {}     # --fix NAME=VALUE (2026-11-26 diagnostics)
STACKS = dict(tag="", min_n=50)     # --stacks TAG: measurement/subset_<subset>_<TAG>; --min-n: smallest stack used (forward)


# ── selection model ──────────────────────────────────────────────────────────
def make_prior(selection, z_range=(0.08, 0.25)):
    rel_lx = Relation("logLx", lambda m, z, t: lx.lx_from_mass(m, z, t["alpha"], t["beta"], t["gamma"], PIV),
                      lambda t: t["sig_int"])
    rel_lam = Relation("logLam", lambda m, z, t: OPT["lam0"] + OPT["b_lam"] * (m - OPT["logM_piv"]), lambda t: OPT["sig_lam"])
    om = ObservableModel([rel_lx, rel_lam], corr=lambda t: np.array([[1, OPT["rho"]], [OPT["rho"], 1]]))
    xray = Stage("X-ray flux S/N >= 2", ("flux_obs", "flux_err"),
                 lambda d: (np.asarray(d["flux_obs"]) >= 2 * np.asarray(d["flux_err"])).astype(float))
    if OPT["lam_min"] is None:
        opt_stage = erf_completeness("optical detection", "logLam_obs", OPT["x50"], OPT["width"], OPT["floor"])
    else:   # one stage: both factors read the same latent richness, so they must be averaged together
        from scipy.special import erf as _erf
        lmin, wmin = np.log10(OPT["lam_min"]), OPT["w_min"]
        opt_stage = Stage("optical detection with minimum richness", ("logLam_obs",),
                          lambda d: 0.5 * (1 + _erf((d["logLam_obs"] - lmin) / (np.sqrt(2) * wmin)))
                          * (OPT["floor"] + (1 - OPT["floor"]) * 0.5 * (1 + _erf((d["logLam_obs"] - OPT["x50"]) / (np.sqrt(2) * OPT["width"])))))
    stages = [xray] + ([opt_stage] if selection == "AB" else [])

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
    if subset in ("t90", "t80"):     # disjoint highest-flag tiers (2026-11-26): P90 not P95; P80 only
        P = {k: np.asarray(cat[k], bool) for k in ("P95", "P90", "P80")}
        keep = (P["P90"] & ~P["P95"]) if subset == "t90" else (P["P80"] & ~P["P90"] & ~P["P95"])
    else:
        keep = np.ones(len(cat), bool) if subset == "all" else np.asarray(cat[subset.upper()], bool)
    a = np.load(f"{M}/joint6.npz"); b = np.load(f"{M}/boosts.npz")
    xdata = {"z": np.asarray(cat["Z"], float), "flux_obs": F, "flux_err": eF,
             "flux_conv": np.asarray(cat["LX"], float) / F, "logLx_obs": np.log10(np.asarray(cat["LX"], float))}
    joint = dict(d=a["data_vectors"] * b["boost6"], v=np.array([np.diag(c) for c in a["cov_matrices"]]) * b["boost6"] ** 2,
                 n=a["n_k"].astype(float), fl=a["floored"], R=a["radii"])
    sd = (M if subset == "all" else f"{M}/subset_{subset}") + (f"_{STACKS['tag']}" if STACKS["tag"] else "")
    stacks, K16, R16 = None, None, None
    if os.path.exists(f"{sd}/stacks.pkl"):
        st = pickle.load(open(f"{sd}/stacks.pkl", "rb")); W = np.load(f"{sd}/weights.npz")
        sub_ids = np.asarray(cat["ID"])[keep]
        assert np.all(W["ids"] == sub_ids), "stack weights must follow the subset's catalogue order"
        idrow = {int(i): k for k, i in enumerate(sub_ids)}
        stacks = [dict(name=nm, d=np.asarray(s["sigmas"], float), C=np.asarray(s["sigmas_cov"], float), n=s["n_stacked"],
                       rows=np.array([idrow[int(i)] for i in s["ids"]])) for nm, s in sorted(st["stacks"].items())]
        K16, R16 = W["K"], st["radii"]
        if subset == "all":
            ts = json.load(open(f"{D}/fits/twostep.json"))["stacks"]
        else:     # per-stack single-halo fits of the subset's stacks (two-step first step), cached
            cache = f"{sd}/stack_fits.json"
            if not os.path.exists(cache):
                import fit_escale as fe
                from multiprocessing import Pool
                fe.M = sd
                with Pool(min(NPOOL, len(stacks))) as pool:
                    json.dump(dict(pool.map(fe.fit_one_stack, [s["name"] for s in stacks])), open(cache, "w"), indent=1, default=float)
            ts = json.load(open(cache))
        for s in stacks:
            s["logM_fit"] = ts[s["name"]]["logM"]
    tier = np.where(np.asarray(cat["P95"], bool), 95, np.where(np.asarray(cat["P90"], bool), 90,
                    np.where(np.asarray(cat["P80"], bool), 80, 0)))      # highest flag; never 0 in the real sample
    xdata["tier"] = tier
    return dict(x={k: v[keep] for k, v in xdata.items()}, joint={k: (v[keep] if k != "R" else v) for k, v in joint.items()},
                stacks=stacks, K16=K16, R16=R16, keep=keep, N=int(keep.sum()))


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


# ── contamination (chance X-ray-optical matches), 2026-11-24 ─────────────────
# Mixture per cluster: L_c = (1 - q_c) sum_k w_ck L_c(DS | M_k) + q_c L_c(DS | chance-match template).
# q_c by purity tier from the random catalogues (randoms/contamination.json, counts after our cuts):
#   tier = highest purity flag (95; 90 = P90 not P95; 80 = only P80; the flags are not nested), q = N_rand/N_real per
#   tier, both after our cuts (contamination.json "tiers"; random flags = any magnitude catalogue, like the real ones).  Injected samples (validation): every
#   member gets the sample-average q = (n95 + n_add)/(N95 + n_add), i.e. the model does not know who was injected.
# Template: boosted random-pool profiles fitted as DS_b = A_b (L_X E^-1 / 10^43.2)^s, plus a per-bin extra variance
#   sig2_b (profile-to-profile scatter beyond measurement noise); template_half="A" uses only CID-even pool members
#   (and injection then draws only CID-odd ones), so validation never fits the template on the injected randoms.
CONTAM = dict(mode="none", q_below80=0.85, template_half="all", chance="template", qlx=False, qflat=None, chance_mis=False)
# chance_mis (2026-11-26, forward only): the mean chance-match profile gets the same miscentring (f_mis, sigma_off) as
#   the genuine model; the centred population over-predicts the random pool's mean signal in the innermost bin
#   (0.25 Mpc: 56 against 31 +- 7.5 M_sun/pc^2) while matching it from 0.44 Mpc out.
# qflat (2026-11-25 diagnostic): the same q for every member (e.g. p95 alone at the q of p95 + 289 injected).
# qlx=True (2026-11-25): q_c depends on (log L_X E^-1, z), since the randoms are fainter and at lower z than the real
#   clusters (a flat q under-absorbs chance matches at low L_X and steepens beta).  q_c = posterior odds of a weighted
#   logistic regression, randoms (class 1) against the sample's own members (class 0), in (log L_X E^-1, z):
#   odds(x) = N_ch f_ch(x) / (N_sample f_sample(x)) = expected chance fraction at x.  Real data: per highest-flag tier,
#   randoms of that tier against the sample's members of that tier.  Injected samples: CID-even randoms (never
#   injected) weighted to n_inj (all tiers) plus tier-95 ones weighted to N_rand(95), against the whole sample.
# chance="population" (2026-11-25): instead of one mean template, a chance match is a halo (centred NFW + two-halo
#   term at the assigned redshift) with log M ~ N(a_ch + b_ch (log L_X E^-1 - 43.2), s_ch) on the mass grid, the three
#   parameters fitted (maximum likelihood) on the random pool (or its CID-even half), cached in
#   randoms/chance_population_<half>.json.  Each chance match then has a realistic spread of possible signals.


def tier_q():
    """q per highest-flag tier (95: P95; 90: P90, not P95; 80: only P80) = N_rand(tier) / N_real(tier) after our cuts
    (the flags are not nested; every real cluster has at least one, so tier 0 never occurs in the real sample)."""
    c = json.load(open(f"{D}/randoms/contamination.json"))
    q = {int(t): v["q"] for t, v in c["tiers"].items()}
    q[0] = CONTAM["q_below80"]
    return q


def _logit_odds(x1, w1, x0, xe):
    """Weighted logistic regression (intercept + linear in the columns of x), class 1 = x1 (weights w1), class 0 = x0
    (weight 1); returns the posterior odds at xe, clipped to [1e-3, 0.95]."""
    from scipy.optimize import minimize
    X = np.vstack([x1, x0]); y = np.r_[np.ones(len(x1)), np.zeros(len(x0))]; w = np.r_[w1, np.ones(len(x0))]
    mu, sd = X.mean(0), X.std(0)
    A = np.c_[np.ones(len(X)), (X - mu) / sd]

    def nll(c):
        t = A @ c
        return np.sum(w * (np.logaddexp(0, t) - y * t)), A.T @ (w * (1 / (1 + np.exp(-t)) - y))
    c = minimize(nll, np.zeros(A.shape[1]), jac=True, method="BFGS").x
    return np.clip(np.exp(np.c_[np.ones(len(xe)), (xe - mu) / sd] @ c), 1e-3, 0.95), c


def q_lx(x, tier, n_inj):
    """Per-cluster chance probability in (log L_X E^-1, z); see CONTAM["qlx"]."""
    pool = Table.read(f"{D}/randoms/escale_random_pool.fits")
    xp = np.c_[np.log10(np.asarray(pool["LX_EZ"], float)), np.asarray(pool["Z"], float)]
    xs = np.c_[x["logLx_obs"] - np.log10(lx.efunc(x["z"])), x["z"]]
    tp = np.asarray(pool["TIER"]); c = json.load(open(f"{D}/randoms/contamination.json"))
    q = np.empty(len(xs))
    if n_inj:
        ev = np.asarray(pool["CID"]) % 2 == 0
        n95 = c["n_random_after_cuts"]["95"]
        x1 = np.vstack([xp[ev], xp[ev & (tp == 95)]])
        w1 = np.r_[np.full(ev.sum(), n_inj / ev.sum()), np.full((ev & (tp == 95)).sum(), n95 / (ev & (tp == 95)).sum())]
        q[:], coef = _logit_odds(x1, w1, xs, xs)
        print(f"q(L_X, z): injected sample, logistic coefficients {np.round(coef, 2)}", flush=True)
    else:
        for t in np.unique(tier):
            m = tier == t
            q[m], coef = _logit_odds(xp[tp == t], np.ones((tp == t).sum()), xs[m], xs[m])
            print(f"q(L_X, z): tier {t}, logistic coefficients {np.round(coef, 2)}, mean q {q[m].mean():.3f}", flush=True)
    return q


def chance_template(half="all"):
    """Fit DS_b = A_b (L/1e43.2)^s to the (boosted) random-pool profiles; returns (A_b, s, sig2_b, n_used)."""
    from cluster_toolkit.boostfactors import boost_powerlaw_at_R
    from scipy.optimize import least_squares
    pool = Table.read(f"{D}/randoms/escale_random_pool.fits"); a = np.load(f"{D}/randoms/measurement/joint6.npz")
    use = np.ones(len(pool), bool) if half == "all" else (np.asarray(pool["CID"]) % 2 == 0)
    st = pickle.load(open(f"{M}/stacks.pkl", "rb"))
    z = np.asarray(pool["Z"], float); llx = np.log10(np.asarray(pool["LX_EZ"], float))
    b6 = np.ones((len(pool), 6))
    for n in range(len(pool)):
        zb = int(np.clip(np.searchsorted(st["z_edges"], z[n], side="right") - 1, 0, len(st["z_edges"]) - 2))
        ed = st["lx_edges"][zb]; lb = int(np.clip(np.searchsorted(ed, llx[n], side="right") - 1, 0, len(ed) - 2))
        f = st["boost_fits"][f"z{zb}_lx{lb}"]
        if not f["failed"]:
            b6[n] = boost_powerlaw_at_R(a["radii"], f["B0"], f["Rs"], f["alpha"])
    d = (a["data_vectors"] * b6)[use]; v = (np.array([np.diag(c) for c in a["cov_matrices"]]) * b6 ** 2)[use]
    x = (llx - 43.2)[use]
    ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
    res = lambda t: np.where(ok, (np.where(ok, d, 0) - t[:6] * 10 ** (t[6] * x)[:, None]) / np.sqrt(np.where(ok, v, 1)), 0).ravel()
    t = least_squares(res, x0=np.r_[np.full(6, 10.0), 0.5]).x
    A, sl = t[:6], t[6]
    r2 = np.where(ok, (np.where(ok, d, 0) - A * 10 ** (sl * x)[:, None]) ** 2 - np.where(ok, v, 0), np.nan)
    sig2 = np.clip(np.nanmean(r2, 0), 0, None)
    return A, sl, sig2, int(use.sum())


def _bin_ll(d0, v0, n, sh, ga, ds):
    """Per-cluster, per-mass log-likelihood (N,K) for model profiles ds (N,K,B), as in Joint.loglike."""
    q = (d0[:, None, :] - ds) ** 2 / v0[:, None, :]
    return (-0.5 * np.where(sh[:, None, :], n[:, None, :] * np.log1p(q / np.maximum(n[:, None, :] - 1, 1)), 0).sum(-1)
            - 0.5 * np.where(ga[:, None, :], q, 0).sum(-1))


def _lognormal_grid(mu, s):
    w = np.exp(-0.5 * ((GRID[None, :] - mu[:, None]) / s) ** 2)
    return w / w.sum(1, keepdims=True)


def chance_population(half="all"):
    """Fit (a_ch, b_ch, s_ch) of the chance-match mass distribution on the random pool (cached)."""
    cache = f"{D}/randoms/chance_population_{half}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    from cluster_toolkit.boostfactors import boost_powerlaw_at_R
    from scipy.optimize import minimize
    pool = Table.read(f"{D}/randoms/escale_random_pool.fits"); a = np.load(f"{D}/randoms/measurement/joint6.npz")
    use = np.ones(len(pool), bool) if half == "all" else (np.asarray(pool["CID"]) % 2 == 0)
    st = pickle.load(open(f"{M}/stacks.pkl", "rb"))
    z = np.asarray(pool["Z"], float); llx = np.log10(np.asarray(pool["LX_EZ"], float))
    b6 = np.ones((len(pool), 6))
    for k in range(len(pool)):
        zb = int(np.clip(np.searchsorted(st["z_edges"], z[k], side="right") - 1, 0, len(st["z_edges"]) - 2))
        ed = st["lx_edges"][zb]; lb = int(np.clip(np.searchsorted(ed, llx[k], side="right") - 1, 0, len(ed) - 2))
        f = st["boost_fits"][f"z{zb}_lx{lb}"]
        if not f["failed"]:
            b6[k] = boost_powerlaw_at_R(a["radii"], f["B0"], f["Rs"], f["alpha"])
    d = (a["data_vectors"] * b6)[use]; v = (np.array([np.diag(c) for c in a["cov_matrices"]]) * b6 ** 2)[use]
    n = a["n_k"][use].astype(float); fl = a["floored"][use]; z, x = z[use], (llx - 43.2)[use]
    ok = np.isfinite(d) & np.isfinite(v) & (v > 0); sh = ok & ~fl & (n >= 2); ga = ok & ~sh
    cen, _, th = profile_tables(z, a["radii"], True)
    ll = _bin_ll(np.where(ok, d, 0), np.where(ok, v, 1), n, sh, ga, cen + th)          # (N,K), fixed
    def nll(t):
        if t[2] <= 0.05:
            return 1e30
        w = _lognormal_grid(t[0] + t[1] * x, t[2])
        A = np.log(np.clip(w, 1e-300, None)) + ll; mx = A.max(1, keepdims=True)
        return -float(np.sum(mx[:, 0] + np.log(np.exp(A - mx).sum(1))))
    best = None
    for t0 in ([13.5, 0.5, 0.5], [13.0, 1.0, 1.0], [14.0, 0.3, 0.3], [12.5, 0.5, 1.5]):
        r = minimize(nll, t0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-3, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    # curvature errors (finite differences) for reporting
    h = np.array([0.02, 0.02, 0.02]); H = np.zeros((3, 3)); t = best.x
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h[i], np.eye(3)[j] * h[j]
            H[i, j] = (nll(t + ei + ej) - nll(t + ei - ej) - nll(t - ei + ej) + nll(t - ei - ej)) / (4 * h[i] * h[j])
    err = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
    out = dict(a_ch=float(t[0]), b_ch=float(t[1]), s_ch=float(t[2]), err=err.tolist(), n_used=int(use.sum()),
               nll=float(best.fun), half=half)
    json.dump(out, open(cache, "w"), indent=1)
    return out


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
        self.contam = CONTAM["mode"] != "none" and not self.det
        if self.contam:
            llx = self.x["logLx_obs"] - np.log10(lx.efunc(self.x["z"])) - 43.2
            if CONTAM["chance"] == "population":
                cp = chance_population(CONTAM["template_half"])
                wch = _lognormal_grid(cp["a_ch"] + cp["b_ch"] * llx, cp["s_ch"])
                Ach = np.log(np.clip(wch, 1e-300, None)) + _bin_ll(self.d0, self.v0, self.n, self.sh, self.ga, self.cen + self.th)
                mx = Ach.max(1, keepdims=True)
                self.ll_chance = mx[:, 0] + np.log(np.exp(Ach - mx).sum(1))
                A, sl, sig2, nu = np.zeros(6), 0.0, np.zeros(6), cp["n_used"]
                print(f"chance population: log M = {cp['a_ch']:.2f} + {cp['b_ch']:.2f} (log L/E - 43.2), scatter {cp['s_ch']:.2f} dex "
                      f"(errors {np.round(cp['err'], 2)}; from {cp['n_used']} randoms)", flush=True)
            else:
                A, sl, sig2, nu = chance_template(CONTAM["template_half"])
                ds_c = A[None, :] * 10 ** (sl * llx)[:, None]
                vv = self.v0 + sig2[None, :]
                qq = (self.d0 - ds_c) ** 2 / vv
                # Gaussian bins carry the variance-ratio term, since the template's extra variance changes the normalisation
                self.ll_chance = (-0.5 * np.where(self.sh, self.n * np.log1p(qq / np.maximum(self.n - 1, 1)), 0).sum(-1)
                                  - 0.5 * np.where(self.ga, qq + np.log(vv / self.v0), 0).sum(-1))
            self.tier = np.asarray(self.x["tier"])
            tq = tier_q()
            n_inj = int((self.tier == -1).sum())
            if n_inj:
                c = json.load(open(f"{D}/randoms/contamination.json"))
                qa = (c["n_random_after_cuts"]["95"] + n_inj) / (c["n_real"]["95"] + n_inj)
                self.q_fixed = np.full(self.N, qa)
            else:
                self.q_fixed = np.array([tq[int(t)] for t in self.tier])
            if CONTAM["qflat"] is not None:
                self.q_fixed = np.full(self.N, CONTAM["qflat"])
            if CONTAM["qlx"]:
                self.q_fixed = q_lx(self.x, self.tier, n_inj)
                self.q_shape = self.q_fixed / np.array([tq[int(t)] if t >= 0 else 1.0 for t in self.tier])
            print(f"contamination mixture ({CONTAM['mode']}): template from {nu} randoms, A_b = {np.round(A, 1)}, "
                  f"slope {sl:.2f}, extra sigma_b = {np.round(np.sqrt(sig2), 1)}; q by tier {dict((k, round(v, 3)) for k, v in tq.items())}"
                  + (f"; injected sample: q = {self.q_fixed[0]:.3f} for all" if n_inj else ""), flush=True)

    def q_of(self, p):
        if CONTAM["mode"] == "free":
            tq = tier_q()     # fall back to the random-count values for any q not passed (e.g. the timing call in run)
            qt = np.select([self.tier == 95, self.tier == 90, self.tier == 80],
                           [p.get("q95", tq[95]), p.get("q90", tq[90]), p.get("q80", tq[80])], p.get("q0", tq[0]))
            return np.clip(qt * self.q_shape, 1e-3, 0.95) if CONTAM["qlx"] else qt
        return self.q_fixed

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
        lg = mx[:, 0] + np.log(np.exp(a - mx).sum(1))
        if self.contam:
            q = np.clip(self.q_of(p), 1e-9, 1 - 1e-9)
            lg = np.logaddexp(np.log1p(-q) + lg, np.log(q) + self.ll_chance)
        tot = float(np.sum(lg))
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
        # Chance-match dilution (2026-11-26): each member's expected profile is (1 - q_c) x genuine + q_c x the MEAN
        # chance-match profile, the latter from the halo population fitted on the random pool (chance_population: centred
        # NFW + two-halo, log M ~ N(a_ch + b_ch (log L_X E^-1 - 43.2), s_ch)) at the member's z and L_X; q_c fixed by
        # highest-flag tier from the randoms (tier_q).  No per-cluster classification: the stack is only diluted.
        self.contam = CONTAM["mode"] == "fixed" and not self.det
        if CONTAM["mode"] == "free":
            raise SystemExit("forward: only --contam fixed (q is degenerate with the amplitude in a stack)")
        if self.contam:
            cp = chance_population(CONTAM["template_half"])
            llx = self.x["logLx_obs"] - np.log10(lx.efunc(self.x["z"])) - 43.2
            wch = _lognormal_grid(cp["a_ch"] + cp["b_ch"] * llx, cp["s_ch"])
            self.wch = wch
            self.ds_ch = np.einsum("nk,nkb->nb", wch, self.cen + self.th)                                  # (N,B)
            tq = tier_q()
            self.q = np.array([tq[int(t)] for t in np.asarray(self.x["tier"])])
            for s in self.st:
                r = s["rows"]
                s["q_eff"] = float((self.K[r] * self.q[r, None]).sum() / self.K[r].sum())
            print(f"forward dilution: q by tier {dict((k, round(v, 3)) for k, v in tq.items())}; source-weighted q per stack "
                  + ", ".join(f"{s['name']} {s['q_eff']:.2f}" for s in self.st), flush=True)

    def loglike(self, p):
        if self.det:
            lm = lx.mass_from_lx(self.x["logLx_obs"], self.x["z"], p["alpha"], p["beta"], p["gamma"], PIV)
            dsw = lx.nfw_profiles_lx(self.x["z"], lm, self.R, f_mis=p["f_mis"], sigma_off=p["sigma_off"],
                                     two_halo=self.two_halo, ds_l=self.ds_l)
        else:
            w = weights(self.mp, self.x, p)
            ds = (1 - p["f_mis"]) * self.cen + p["f_mis"] * mis_at(self.mis, p["sigma_off"]) + self.th
            dsw = np.einsum("nk,nkb->nb", w, ds)                                                      # (N,B)
            if self.contam:
                ds_ch = np.einsum("nk,nkb->nb", self.wch, ds) if CONTAM["chance_mis"] else self.ds_ch
                dsw = (1 - self.q)[:, None] * dsw + self.q[:, None] * ds_ch
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


def resample(data, idx):
    """Cluster bootstrap of the per-cluster (joint) data: the same clusters, drawn with replacement."""
    out = dict(data)
    out["x"] = {k: v[idx] for k, v in data["x"].items()}
    out["joint"] = {k: (v[idx] if k != "R" else v) for k, v in data["joint"].items()}
    out["stacks"], out["N"] = None, len(idx)
    return out


def load_injected(base, n_add, seed):
    """Contamination-injection test (2026-11-24): the real `base` subset plus n_add spurious X-ray-optical matches
    drawn without replacement (seed) from the random pool (build_random_pool.py; measured with measure_escale.py).
    Pool members that are the same X-ray source as a base cluster are excluded.  Each random gets the boost of the
    full-sample stack its (z, Lx/E) falls in, as the real clusters do (load_real uses the full-sample boosts for
    every subset).  Joint fit only (no stacks)."""
    from cluster_toolkit.boostfactors import boost_powerlaw_at_R
    data = load_real(base)
    pool = Table.read(f"{D}/randoms/escale_random_pool.fits")
    a = np.load(f"{D}/randoms/measurement/joint6.npz")
    assert np.all(a["ids"] == np.asarray(pool["ID"]))
    base_cid = set(np.asarray(Table.read(f"{D}/escale_des_sample.fits")["CID"])[data["keep"]].astype(int))
    avail = np.flatnonzero(~np.isin(np.asarray(pool["CID"]).astype(int), list(base_cid)))
    if CONTAM["template_half"] == "A":     # validation: template from half A, inject only from half B
        avail = avail[np.asarray(pool["CID"])[avail] % 2 == 1]
    if n_add > len(avail):
        raise SystemExit(f"only {len(avail)} randoms available")
    pick = np.sort(np.random.default_rng(seed).choice(avail, n_add, replace=False))
    st = pickle.load(open(f"{M}/stacks.pkl", "rb"))
    z = np.asarray(pool["Z"], float)[pick]; llx = np.log10(np.asarray(pool["LX_EZ"], float)[pick])
    zb = np.clip(np.searchsorted(st["z_edges"], z, side="right") - 1, 0, len(st["z_edges"]) - 2)
    b6 = np.ones((n_add, 6))
    for n in range(n_add):
        ed = st["lx_edges"][zb[n]]
        lb = int(np.clip(np.searchsorted(ed, llx[n], side="right") - 1, 0, len(ed) - 2))
        f = st["boost_fits"][f"z{zb[n]}_lx{lb}"]
        if not f["failed"]:
            b6[n] = boost_powerlaw_at_R(a["radii"], f["B0"], f["Rs"], f["alpha"])
    F, eF = np.asarray(pool["FLUX"], float)[pick], np.asarray(pool["FLUX_ERR"], float)[pick]
    LX = np.asarray(pool["LX"], float)[pick]
    xr = {"z": z, "flux_obs": F, "flux_err": eF, "flux_conv": LX / F, "logLx_obs": np.log10(LX), "tier": np.full(n_add, -1)}
    jr = dict(d=a["data_vectors"][pick] * b6, v=np.array([np.diag(c) for c in a["cov_matrices"][pick]]) * b6 ** 2,
              n=a["n_k"][pick].astype(float), fl=a["floored"][pick])
    out = dict(data)
    out["x"] = {k: np.concatenate([data["x"][k], xr[k]]) for k in data["x"]}
    out["joint"] = {k: (np.concatenate([data["joint"][k], jr[k]]) if k != "R" else data["joint"][k]) for k in data["joint"]}
    out["stacks"], out["N"], out["n_injected"] = None, data["N"] + n_add, n_add
    print(f"injection: base {base} N={data['N']} + {n_add} randoms (seed {seed}; {len(avail)} available) -> N={out['N']}",
          flush=True)
    return out


def run(mode, selection, data, tag, timeout=None):
    if mode == "twostep" and selection == "det":
        return twostep_det(data, tag)
    names = ["alpha", "beta", "gamma"] + (["sig_int"] if selection != "det" else []) + (["f_mis", "sigma_off"] if mode != "twostep" else [])
    if mode == "joint" and CONTAM["mode"] == "free":
        tiers = set(np.asarray(data["x"]["tier"]).tolist())
        names += [f"q{t}" for t in (95, 90, 80, 0) if t in tiers]
    names = [n for n in names if n not in FIX]
    L = (Forward(data, selection, min_n=STACKS["min_n"]) if mode == "forward"
         else {"joint": Joint, "twostep": TwoStep}[mode](data, selection))
    if FIX:     # fixed parameters (--fix NAME=VALUE), merged into every likelihood call
        _ll = L.loglike
        L.loglike = lambda p: _ll({**p, **FIX})
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
    ok = smp.run(verbose=True, n_eff=3000, timeout=timeout) if timeout else smp.run(verbose=True, n_eff=3000)
    if timeout and (ok is False or getattr(smp, "n_eff", 3000) < 1000):   # as paper 1.1 joint_sh.py: skip, draw another
        print(f"{tag}: NOT CONVERGED within {timeout:.0f} s, skipped", flush=True)
        os.makedirs(OUT, exist_ok=True); open(f"{OUT}/{tag}.skipped", "w").write("timeout\n")
        return
    X, lw, ll = smp.posterior(); w = np.exp(lw - lw.max()); w /= w.sum()
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{tag}.npz", samples=X, weights=w, log_l=ll, param_order=np.array(names), log_z=smp.log_z,
             n_clusters=data["N"], selection=selection, mode=mode, opt=json.dumps(OPT), contam=json.dumps(CONTAM))
    print(f"{tag}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for j, n in enumerate(names):
        o = np.argsort(X[:, j]); lo, me, hi = np.interp([.16, .5, .84], np.cumsum(w[o]), X[o, j])
        print(f"  {n:10s} {me:+.3f} +{hi - me:.3f}/-{me - lo:.3f}", flush=True)
    # weight at the grid edges for the posterior median (sanity: the grid must contain the prior)
    pm = {n: float(np.interp(0.5, np.cumsum(w[np.argsort(X[:, j])]), np.sort(X[:, j]))) for j, n in enumerate(names)}
    pm.update(FIX); pm.setdefault("f_mis", 0.0); pm.setdefault("sigma_off", 0.1)
    if selection == "det":
        return
    ww = weights(L.mp, L.x, pm)
    print(f"  max prior weight in the lowest/highest 2 grid points: {ww[:, :2].sum(1).max():.2e} / {ww[:, -2:].sum(1).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["joint", "forward", "twostep"])
    ap.add_argument("--selection", choices=["det", "A", "AB"], required=True)
    ap.add_argument("--subset", default="all", choices=["all", "p95", "p90", "p80", "t90", "t80"],
                    help="t90 / t80: disjoint highest-flag tiers (P90 not P95; P80 only); p95 is the 95 tier")
    ap.add_argument("--data", default=None, help="mock .npz (see make_escale_mock.py); default: real data")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lam-min", type=float, default=None, help="minimum AME richness of the optical stage (OPT version 2b)")
    ap.add_argument("--boot", type=int, nargs=2, default=None, metavar=("B0", "B1"),
                    help="joint only: run cluster bootstraps until B1-B0 have converged, starting at draw B0 "
                         "(resampling seed 7000+b, as paper 1.1); unconverged draws are skipped and replaced (at most 10 extra)")
    ap.add_argument("--timeout", type=float, default=2400.0, help="per-bootstrap sampler time limit [s]")
    ap.add_argument("--contam", default="none", choices=["none", "fixed", "free"],
                    help="chance-match mixture (joint) or dilution (forward, fixed only), q by purity tier from the randoms or free")
    ap.add_argument("--template-half", default="all", choices=["all", "A"], help="A: template from CID-even randoms, inject CID-odd")
    ap.add_argument("--q-below80", type=float, default=0.85)
    ap.add_argument("--q-flat", type=float, default=None, help="diagnostic: the same fixed q for every member")
    ap.add_argument("--chance-mis", action="store_true", help="forward dilution: chance profile miscentred like genuine clusters")
    ap.add_argument("--stacks", default="", help="stack layout tag (measurement/subset_<subset>_<TAG>, from build_escale_stacks.py --out-tag)")
    ap.add_argument("--min-n", type=int, default=50, help="forward: smallest stack used")
    ap.add_argument("--gauss-prior", nargs="*", default=[], metavar="NAME=MU,SIGMA",
                    help="Gaussian prior instead of the flat one, e.g. gamma=0,0.79 (Pederneiras+25)")
    ap.add_argument("--fix", nargs="*", default=[], metavar="NAME=VALUE", help="fix parameters (diagnostics)")
    ap.add_argument("--q-lx", action="store_true", help="q per cluster from (L_X, z) (logistic regression on the randoms)")
    ap.add_argument("--chance", default="template", choices=["template", "population"],
                    help="chance-match lensing model: mean template, or halo population fitted on the randoms")
    ap.add_argument("--inject", type=int, nargs=2, default=None, metavar=("N_ADD", "SEED"),
                    help="joint only: add N_ADD random-pool spurious matches (seed SEED) to --subset (load_injected)")
    a = ap.parse_args()
    if a.lam_min is not None:
        OPT["lam_min"] = a.lam_min
    FIX.update({k: float(v) for k, v in (x.split("=") for x in a.fix)})
    STACKS.update(tag=a.stacks, min_n=a.min_n)
    GP = {}
    for x in a.gauss_prior:
        k, v = x.split("="); mu, sg = map(float, v.split(","))
        PRI[k] = stats.norm(mu, sg); GP[k] = (mu, sg)
    CONTAM.update(mode=a.contam, template_half=a.template_half, q_below80=a.q_below80, chance=a.chance, qlx=a.q_lx, qflat=a.q_flat, chance_mis=a.chance_mis)
    if a.inject:
        data = load_injected(a.subset, *a.inject)
    else:
        data = load_mock(a.data) if a.data else load_real(a.subset)
    if a.mode != "joint" and data["stacks"] is None:
        raise SystemExit("stacked frameworks need stacks built for this subset (only --subset all has them)")
    tag = (f"{a.mode}_{a.selection}_{a.subset}" + (f"_inj{a.inject[0]}s{a.inject[1]}" + ("hB" if a.template_half == "A" else "") if a.inject else "")
           + (f"_cont{a.contam}" + ("A" if a.template_half == "A" else "") + ("pop" if a.chance == "population" else "") + ("qlx" if a.q_lx else "") + (f"q{a.q_flat:g}" if a.q_flat is not None else "") + ("mis" if a.chance_mis else "")
              if a.contam != "none" else "") + "".join(f"_fix{k}{v:g}" for k, v in sorted(FIX.items())) + "".join(f"_gp{k}{m:g}_{g:g}" for k, (m, g) in sorted(GP.items())) + (f"_st{a.stacks}" if a.stacks else "")
           + (f"_minn{a.min_n}" if a.min_n != 50 else "") + (f"_{a.tag}" if a.tag else ""))
    if a.boot:
        if a.mode != "joint":
            raise SystemExit("--boot is for the joint fit")
        b0, b1 = a.boot; want = b1 - b0; b = b0
        done = lambda: sum(os.path.exists(f"{OUT}/{tag}_b{k:03d}.npz") for k in range(b0, b))
        while done() < want and b < b1 + 10:
            tb = f"{tag}_b{b:03d}"
            if not (os.path.exists(f"{OUT}/{tb}.npz") or os.path.exists(f"{OUT}/{tb}.skipped")):
                idx = np.random.default_rng(7000 + b).integers(0, data["N"], data["N"])
                run(a.mode, a.selection, resample(data, idx), tb, timeout=a.timeout)
            b += 1
    else:
        run(a.mode, a.selection, data, tag)
