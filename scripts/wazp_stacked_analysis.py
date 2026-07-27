"""
wazp_stacked_analysis.py

Stacked-signal cross-check of the WaZP + DES-Y3 destacking (joint
per-cluster) fit in gamma_scan_outputs/run_2026-07-23T15-34-36. That fit
found alpha_cm incompatible with zero on real data — flagged as a likely
data/pipeline problem (see project memory
wazp-desy3-alpha-cm-anomaly). This script re-analyses the SAME real
catalogues the traditional way: bin clusters by (z, richness, magnitude
gap), stack each bin's ΔΣ profile, fit NFW + miscentering per stack, then
fit mass-richness and mass-concentration power laws per magnitude-gap
group and compare them — the stacking methodology and bin definitions are
carried over directly from notebooks/wazp_mass_richness_by_delta_m.ipynb
(SIMS=False branch), which is the "guide" for this script.

Inputs (already built by prepare_real_data.py, reused as-is; real data lives
under ~/destacking_data/real_data/, see REAL_DATA_DIR below):
  wazp_real_data.fits    compact WaZP cluster catalogue (ID, RA, DEC,
                          NGALS, ZPHOT, DELTA_M_I, DELTA_M_I_ERR)
  metacal_filtered.npz   pre-filtered DES-Y3 metacal+DNF source
                          catalogue (ra, dec, z, e1, e2, w, R11..R22)

Pipeline (6 stages, each checkpointed under data/wazp_stacked_*.pkl /
data/wazp_boost_fits.pkl and gated by USE_CACHE so a re-run picks up where
it left off):

  1. bkgs       per-cluster RAW lensing background arrays (BallTree cone
                query + xlensing.data.metacal_cluster_lensing) — kept raw
                (not reduced to a single data vector) so the same
                per-cluster measurement can be stacked under any binning.
  2. stacks     bin into 3 z x 4 NGALS x 2 delta_m (median-split per
                (z, NGALS) cell, via destacking.data.assign_stack_bins) =
                up to 24 stacks; each stacked via xlensing.data.stacked_signal,
                which boost-corrects the signal AND its covariance
                internally (do NOT re-apply the boost — see project memory
                "stacked_signal applies the boost internally").
  2b. boost_fits fit cluster_toolkit.boostfactors.boost_powerlaw_at_R
                (B(R) = 1 + B0*(R/Rs)^alpha, 3 params) to each stack's
                boost(R) profile from stage 2, by GLS against the full
                boosts_cov. Saved to data/wazp_boost_fits.pkl, keyed by
                stack name ("z{i}_ng{j}_dm{k}") — prepare_real_data.py
                labels each cluster with this same stack name (also via
                assign_stack_bins) so gamma_inference.py can look up and
                apply the matching stack's boost correction to that
                cluster's individual ΔΣ profile (its `boosts:` config
                key) — the joint fit has no boost correction of its own
                (bootstrap_signal, not single_cluster), unlike stage 2
                here. See project memory
                wazp-desy3-alpha-cm-informative-prior.
  3. stack_fits NFW + miscentering nested-sampling fit per stack:
                log_M, C, FMISS, SIGMA_OFF (BCG_B_MASS fixed at 0).
  4. ml_fits    mass-richness power law per delta_m group (low/high),
                bootstrap linear regression over stacks (propagates both
                the per-stack log_M posterior and the per-cluster
                richness/z scatter within each stack).
  5. cmr_fits   mass-concentration power law per delta_m group, same
                bootstrap-regression method (propagates the log_M/C
                posterior correlation within each stack).

Usage: python wazp_stacked_analysis.py
Then: notebooks/wazp_stacked_plots.ipynb regenerates every plot from the
saved pickles alone, with no recomputation.
"""

import os
import pickle
import time

import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18 as COSMO
from astropy import units as u
from scipy.stats import norm as sp_norm, truncnorm as sp_truncnorm
from sklearn.neighbors import BallTree
from nautilus import Prior, Sampler

import xlensing.data
import xlensing.model
from destacking.data import select_background, assign_stack_bins
from cluster_toolkit.boostfactors import boost_powerlaw_at_R
from scipy.optimize import curve_fit

# ── Config ────────────────────────────────────────────────────────────────
# Real WaZP+DES-Y3 data and outputs (including the boost fits below) live
# outside the repo -- moved 2026-07-27, see project memory
# sim-data-relocated / real-data-relocated.
REAL_DATA_DIR = os.path.expanduser("~/destacking_data/real_data")

WAZP_CATALOG = f"{REAL_DATA_DIR}/wazp_real_data.fits"
FILTERED_NPZ = f"{REAL_DATA_DIR}/metacal_filtered.npz"

# Output file tag: distinguishes cache/output files between background-
# selection strategies/parameters so different runs can be compared side
# by side instead of clobbering each other. "" reproduces the original
# (factor=1.1, the xlensing-default-matching baseline) filenames — back to
# baseline: the boost fit needs to match wazp_real_data (factor=1.1), which
# is what the current joint fit (wazp_desy3_inference.yaml) reads.
RUN_TAG = ""

BKG_CACHE  = f"{REAL_DATA_DIR}/wazp_stacked_bkgs{RUN_TAG}.pkl"
STACKS_PKL = f"{REAL_DATA_DIR}/wazp_stacked_stacks{RUN_TAG}.pkl"
FITS_PKL   = f"{REAL_DATA_DIR}/wazp_stacked_fits{RUN_TAG}.pkl"
ML_PKL     = f"{REAL_DATA_DIR}/wazp_stacked_ml_fits{RUN_TAG}.pkl"
CMR_PKL    = f"{REAL_DATA_DIR}/wazp_stacked_cmr_fits{RUN_TAG}.pkl"
BOOST_PKL  = f"{REAL_DATA_DIR}/wazp_boost_fits{RUN_TAG}.pkl"

USE_CACHE  = True     # set False to force a clean rebuild of every stage
RADIUS_MPC = 5.0
NBOOT      = 500      # cluster-bootstrap draws for stacked_signal & NFW cov
NLIVE      = 400
NWORKERS   = 12

# Background-galaxy selection: xlensing.data.metacal_cluster_lensing applies
# no redshift filtering internally any more (moved here so different
# selection strategies can be tried and compared — see
# destacking.data.BACKGROUND_SELECTION_STRATEGIES). factor=1.1 reproduces
# the old hardcoded xlensing default exactly — back to baseline (see
# RUN_TAG above); the factor=1.2 experiment is on hold.
BACKGROUND_STRATEGY = "margin"
BACKGROUND_STRATEGY_KWARGS = {"factor": 1.1, "offset": 0.2}
TIMEOUT_S  = 300.0     # per-stack nautilus wall-clock cap
MIN_N_FOR_FIT = 10     # stacks below this are excluded from ml_fits/cmr_fits
N_DRAWS    = 1000      # bootstrap iterations for the M-lambda / c-M fits

# Radial grid: 16 log-spaced bins -0.8..0.6 (matches
# wazp_mass_richness_by_delta_m.ipynb; finer than the 8-bin joint-fit grid
# since a per-stack 4-parameter NFW fit can support more bins).
radii  = np.logspace(-0.8, 0.6, 16)
_N     = len(radii)
_edges = np.logspace(
    np.log10(radii[0])  + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    _N + 1,
)
BINS_LIMS = np.array([[_edges[i], _edges[i + 1]] for i in range(_N)])

# Bin definitions: 3 z x 4 NGALS x 2 delta_m (median split per cell) —
# identical to wazp_mass_richness_by_delta_m.ipynb §2.
Z_EDGES  = np.array([0.10, 0.30, 0.50, 0.70])
NG_EDGES = np.array([25., 35., 55., 85., np.inf])
N_NG     = len(NG_EDGES) - 1

LAM_PIV, Z_PIV   = 40.0, 0.35     # mass-richness pivot
CMR_M_PIV, CMR_Z_PIV = 1e14, 0.35 # mass-concentration pivot

PARAM_ORDER = ["log_M", "C", "FMISS", "SIGMA_OFF"]
LOGM_MU, LOGM_SIG = 14.5, 2.0
C_MU, C_SIG       = 5.0, 3.0
C_LO, C_HI        = 0.1, 20.0
PRIOR_LOGM = sp_norm(loc=LOGM_MU, scale=LOGM_SIG)
PRIOR_C    = sp_truncnorm((C_LO - C_MU) / C_SIG, (C_HI - C_MU) / C_SIG,
                          loc=C_MU, scale=C_SIG)


def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    return f"{int(h)}h{int(rem / 60):02d}m"


# ── Stage 1: per-cluster raw bkg arrays ──────────────────────────────────

_TREE = None
_SR   = None


def _bkg_worker(args):
    """Mirrors wazp_mass_richness_by_delta_m.ipynb's bkg_worker (SIMS=False
    branch): BallTree cone query + metacal_cluster_lensing, returning the
    raw 6-row bkg array (Sigma_crit, e_t, e_x, weight, R, m) — NOT reduced
    to a fixed data vector, so it can be stacked under any binning later."""
    cid, ra_deg, dec_deg, z_cl = args
    try:
        ra_r, dec_r = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
        D_A = float(COSMO.angular_diameter_distance(z_cl).to_value(u.Mpc))
        if not (D_A > 0):
            return cid, None, 0
        ang_r = RADIUS_MPC / D_A
        idx = _TREE.query_radius(np.array([[dec_r, ra_r]]), r=ang_r)[0]
        if len(idx) == 0:
            return cid, None, 0

        # Background selection is caller-side now — xlensing applies no
        # redshift filtering of its own (see project memory
        # "xlensing background selection moved to caller").
        bg_mask = select_background(BACKGROUND_STRATEGY, _SR["z"][idx], z_cl,
                                    **BACKGROUND_STRATEGY_KWARGS)
        idx = idx[bg_mask]
        if len(idx) == 0:
            return cid, None, 0

        sources = (
            _SR["ra"][idx], _SR["dec"][idx], _SR["z"][idx],
            _SR["e1"][idx], _SR["e2"][idx], _SR["w"][idx],
            _SR["R11"][idx], _SR["R12"][idx], _SR["R21"][idx], _SR["R22"][idx],
        )
        res = xlensing.data.metacal_cluster_lensing(
            (ra_r, dec_r, z_cl), sources, radius=RADIUS_MPC,
        )
        if res["Count"] == 0:
            return cid, None, 0
        bkg = np.array([
            np.asarray(res["Critical Density"], dtype="float32"),
            np.asarray(res["Tangential Shear"], dtype="float32"),
            np.asarray(res["Cross Shear"],      dtype="float32"),
            np.asarray(res["Weights"],          dtype="float32"),
            np.asarray(res["Radial Distance"],  dtype="float32"),
            np.asarray(res["Mult. Bias"],       dtype="float32"),
        ])
        return cid, bkg, int(res["Count"])
    except Exception:
        return cid, None, 0


def build_bkgs(catalog, sr):
    global _TREE, _SR
    if os.path.exists(BKG_CACHE) and USE_CACHE:
        with open(BKG_CACHE, "rb") as fh:
            bkg_by_id = pickle.load(fh)
        print(f"  bkgs: loaded {len(bkg_by_id):,} cached bkgs from {BKG_CACHE}",
              flush=True)
        return bkg_by_id

    print(f"  bkgs: building BallTree over {len(sr['ra']):,} sources ...",
          flush=True)
    t0 = time.time()
    _TREE = BallTree(np.column_stack([sr["dec"], sr["ra"]]), metric="haversine")
    _SR = sr
    print(f"  bkgs: BallTree built in {time.time() - t0:.1f}s", flush=True)

    work = [
        (int(catalog["ID"][i]), float(catalog["RA"][i]),
         float(catalog["DEC"][i]), float(catalog["ZPHOT"][i]))
        for i in range(len(catalog))
    ]
    print(f"  bkgs: measuring {len(work):,} clusters with {NWORKERS} workers",
          flush=True)

    import multiprocess as mp
    bkg_by_id, n_ok = {}, 0
    t0 = time.time()
    with mp.Pool(NWORKERS) as pool:
        for k, (cid, bkg, n_src) in enumerate(
            pool.imap_unordered(_bkg_worker, work, chunksize=4)
        ):
            if bkg is not None:
                bkg_by_id[cid] = bkg
                n_ok += 1
            if (k + 1) % 1000 == 0 or (k + 1) == len(work):
                dt = time.time() - t0
                eta = dt * (len(work) - k - 1) / (k + 1)
                print(f"    {k + 1:>7,}/{len(work):,}  good {n_ok:>7,}  "
                      f"elapsed {_fmt_duration(dt)}  eta {_fmt_duration(eta)}",
                      flush=True)

    with open(BKG_CACHE, "wb") as fh:
        pickle.dump(bkg_by_id, fh)
    print(f"  bkgs: {n_ok:,}/{len(work):,} usable, saved to {BKG_CACHE}",
          flush=True)
    return bkg_by_id


# ── Stage 2: bin + stack ──────────────────────────────────────────────────

def build_stacks(catalog, bkg_by_id):
    if os.path.exists(STACKS_PKL) and USE_CACHE:
        with open(STACKS_PKL, "rb") as fh:
            d = pickle.load(fh)
        print(f"  stacks: loaded {len(d['stacks'])} cached stacks from "
              f"{STACKS_PKL}", flush=True)
        return d["stacks"], d["z_bin"], d["ng_bin"], d["dm_bin"], \
               d["dm_median_per_cell"]

    # Shared with prepare_real_data.py (destacking.data.assign_stack_bins)
    # so a cluster's stack label always matches the boost model it was
    # fit from — see Stage 6 (fit_boost_models) below.
    z_bin, ng_bin, dm_bin, dm_median_per_cell, _stack_name = assign_stack_bins(
        catalog["ZPHOT"], catalog["NGALS"], catalog["DELTA_M_I"],
        z_edges=Z_EDGES, ng_edges=NG_EDGES,
    )

    stacks = {}
    boost_outer = []
    for i in range(3):
        for j in range(N_NG):
            for k in range(2):
                name = f"z{i}_ng{j}_dm{k}"
                mask = (z_bin == i) & (ng_bin == j) & (dm_bin == k)
                if not mask.any():
                    continue
                ids = np.asarray(catalog["ID"])[mask]
                bkgs = [bkg_by_id[int(cid)] for cid in ids if int(cid) in bkg_by_id]
                if not bkgs:
                    continue
                t0 = time.time()
                sigmas, boosts, boosts_cov, sigmas_cov, xigmas, xigmas_cov = (
                    xlensing.data.stacked_signal(bkgs, BINS_LIMS, NBOOT)
                )
                cell_z   = np.asarray(catalog["ZPHOT"])[mask]
                cell_lam = np.asarray(catalog["NGALS"])[mask]
                cell_dm  = np.asarray(catalog["DELTA_M_I"])[mask]
                stacks[name] = {
                    "z_bin": i, "ng_bin": j, "dm_bin": k,
                    "z_lo": float(Z_EDGES[i]), "z_hi": float(Z_EDGES[i + 1]),
                    "ng_lo": float(NG_EDGES[j]), "ng_hi": float(NG_EDGES[j + 1]),
                    "dm_median": dm_median_per_cell[(i, j)],
                    "z_mean": float(cell_z.mean()),
                    "lambda_mean": float(cell_lam.mean()),
                    "dm_mean": float(cell_dm.mean()),
                    "n_stacked": len(bkgs),
                    "sigmas": np.asarray(sigmas),
                    "sigmas_cov": np.asarray(sigmas_cov),
                    "boosts": np.asarray(boosts),
                    "boosts_cov": np.asarray(boosts_cov),
                    "xigmas": np.asarray(xigmas),
                    "xigmas_cov": np.asarray(xigmas_cov),
                }
                b_err_outer = float(np.sqrt(boosts_cov[-1, -1]))
                boost_outer.append(float(boosts[-1]))
                print(f"  stacks: {name}  <z>={cell_z.mean():.3f}  "
                      f"<lam>={cell_lam.mean():6.1f}  <dm>={cell_dm.mean():.2f}  "
                      f"N={len(bkgs):,}  B_outer={boosts[-1]:.3f}"
                      f"+/-{b_err_outer:.3f}  ({time.time() - t0:.1f}s)",
                      flush=True)

    if boost_outer:
        b_outer = np.array(boost_outer)
        print(f"\n  Outer-bin boost over {len(boost_outer)} stacks: "
              f"median={np.median(b_outer):.3f} mean={b_outer.mean():.3f} "
              f"min={b_outer.min():.3f} max={b_outer.max():.3f}", flush=True)
        if not (0.85 < np.median(b_outer) < 1.15):
            print("  WARNING: outer-bin boost median far from 1 — check "
                  "binning / random count", flush=True)

    with open(STACKS_PKL, "wb") as fh:
        pickle.dump({
            "radii": radii, "bins_lims": BINS_LIMS,
            "z_edges": Z_EDGES, "ng_edges": NG_EDGES,
            "dm_median_per_cell": dm_median_per_cell,
            "stacks": stacks, "z_bin": z_bin, "ng_bin": ng_bin, "dm_bin": dm_bin,
        }, fh)
    n_expected = 3 * N_NG * 2
    print(f"\n  stacks: produced {len(stacks)}/{n_expected}, saved to "
          f"{STACKS_PKL}", flush=True)
    return stacks, z_bin, ng_bin, dm_bin, dm_median_per_cell


# ── Stage 2b: fit a boost(R) model per stack ─────────────────────────────
#
# B(R) = 1 + B0*(R/Rs)^alpha  (cluster_toolkit.boostfactors.boost_powerlaw_at_R)
# fit by generalised least squares (full boosts_cov as the covariance,
# matching how the boost was itself estimated — a joint cluster bootstrap
# with the signal, see the module docstring) against each stack's boost
# profile from Stage 2. Used by prepare_real_data.py + gamma_inference.py
# to correct individual-cluster ΔΣ profiles by their stack's fitted boost
# (see gamma_inference.py's `boosts:` config key) — motivated by
# [[project_wazp_desy3_root_cause_hypotheses]]: the joint fit has no boost
# correction at all (bootstrap_signal, not single_cluster), unlike this
# stacked pipeline (stacked_signal).
BOOST_P0     = (0.3, 1.0, -1.0)          # B0, Rs [Mpc], alpha — cluster_toolkit's own test values
BOOST_BOUNDS = ((0.0, 0.01, -10.0), (20.0, 50.0, 10.0))


def _boost_model(R, B0, Rs, alpha):
    return boost_powerlaw_at_R(R, B0, Rs, alpha)


def fit_boost_models(stacks):
    if os.path.exists(BOOST_PKL) and USE_CACHE:
        with open(BOOST_PKL, "rb") as fh:
            d = pickle.load(fh)
        print(f"  boost_fits: loaded {len(d['boost_fits'])} cached fits from "
              f"{BOOST_PKL}", flush=True)
        return d["boost_fits"]

    boost_fits = {}
    for name, s in stacks.items():
        R   = radii
        B   = np.asarray(s["boosts"], dtype=float)
        cov = np.asarray(s["boosts_cov"], dtype=float)
        try:
            popt, pcov = curve_fit(
                _boost_model, R, B, p0=BOOST_P0, bounds=BOOST_BOUNDS,
                sigma=cov, absolute_sigma=True, maxfev=20000,
            )
            B0, Rs, alpha = (float(x) for x in popt)
            perr = np.sqrt(np.diag(pcov))
            resid = B - _boost_model(R, *popt)
            cov_reg = cov + 1e-8 * np.eye(len(R)) * np.mean(np.diag(cov))
            chi2 = float(resid @ np.linalg.solve(cov_reg, resid))
            boost_fits[name] = {
                "B0": B0, "Rs": Rs, "alpha": alpha,
                "B0_err": float(perr[0]), "Rs_err": float(perr[1]),
                "alpha_err": float(perr[2]), "cov": pcov,
                "chi2": chi2, "dof": len(R) - 3, "failed": False,
            }
            print(f"  boost_fits: {name}  B0={B0:.3f}+/-{perr[0]:.3f}  "
                  f"Rs={Rs:.3f}+/-{perr[1]:.3f}  alpha={alpha:+.3f}"
                  f"+/-{perr[2]:.3f}  chi2/dof={chi2:.1f}/{len(R)-3}",
                  flush=True)
        except Exception as exc:
            boost_fits[name] = {"failed": True, "error": repr(exc)}
            print(f"  boost_fits: {name}  FAILED  {exc!r}", flush=True)

    with open(BOOST_PKL, "wb") as fh:
        pickle.dump({"radii": radii, "boost_fits": boost_fits}, fh)
    n_ok = sum(1 for f in boost_fits.values() if not f.get("failed"))
    print(f"\n  boost_fits: {n_ok}/{len(boost_fits)} ok, saved to {BOOST_PKL}",
          flush=True)
    return boost_fits


# ── Stage 3: per-stack NFW + miscentering fit ────────────────────────────

def _make_prior():
    p = Prior()
    p.add_parameter("log_M", dist=PRIOR_LOGM)
    p.add_parameter("C", dist=PRIOR_C)
    p.add_parameter("FMISS", dist=(0.0, 1.0))
    p.add_parameter("SIGMA_OFF", dist=(0.01, 1.0))
    return p


def _run_sampler(loglike, n_live, pool=None, n_points_min=None,
                 split_threshold=100):
    kwargs = dict(n_live=n_live, pool=pool, split_threshold=split_threshold)
    if n_points_min is not None:
        kwargs["n_points_min"] = n_points_min
    samp = Sampler(_make_prior(), loglike, **kwargs)
    success = samp.run(verbose=False, timeout=TIMEOUT_S)
    return samp, bool(success)


def _kl_div_from_sampler(samp):
    """KL(posterior || prior) via D_KL = <log L>_post - log Z, in nats —
    same identity used throughout gamma_inference.py/destacking.fitting."""
    pts, lw, log_l = samp.posterior()
    w = np.exp(lw - lw.max())
    w = w / w.sum()
    finite = np.isfinite(log_l)
    if not finite.any():
        return pts, lw, log_l, float("nan")
    kl = float(np.sum(w[finite] * log_l[finite]) - samp.log_z)
    return pts, lw, log_l, kl


def fit_one(item, pool=None):
    name, s = item
    data   = np.asarray(s["sigmas"], dtype=float)
    cov_in = np.asarray(s["sigmas_cov"], dtype=float)
    z_eff  = float(s["z_mean"])

    common = {
        "z_mean": s["z_mean"], "lambda_mean": s["lambda_mean"],
        "dm_mean": s["dm_mean"], "dm_bin": s["dm_bin"],
        "n_stacked": s["n_stacked"],
    }

    if not np.all(np.isfinite(data)):
        return name, {"failed": True,
                      "error": f"NaN/Inf in sigmas "
                               f"({int((~np.isfinite(data)).sum())} bins)",
                      **common}
    if not np.all(np.isfinite(cov_in)):
        return name, {"failed": True, "error": "NaN/Inf in sigmas_cov", **common}
    diag_mean = float(np.mean(np.diag(cov_in)))
    if not np.isfinite(diag_mean) or diag_mean <= 0:
        return name, {"failed": True,
                      "error": f"degenerate sigmas_cov diag (mean={diag_mean})",
                      **common}
    cov_reg = cov_in + 1e-6 * np.eye(len(data)) * diag_mean
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError as exc:
        return name, {"failed": True, "error": f"cov inversion failed: {exc!r}",
                      **common}
    if not np.all(np.isfinite(cov_inv)):
        return name, {"failed": True, "error": "NaN/Inf in cov_inv", **common}

    def loglike(p):
        try:
            M = 10.0 ** p["log_M"]
            if not np.isfinite(M) or M <= 0:
                return -np.inf
            out = xlensing.model.NFW_Delta_Sigma(
                M200m=np.array([M]), C200m=np.array([p["C"]]),
                Z=np.array([z_eff]), FMISS=np.array([p["FMISS"]]),
                SIGMA_OFF=np.array([p["SIGMA_OFF"]]),
                BCG_B_MASS=np.array([0.0]), radii=radii,
            )
            model_ds = np.asarray(out["Signal"][0], dtype=float)
            if not np.all(np.isfinite(model_ds)):
                return -np.inf
            res = data - model_ds
            ll = float(-0.5 * res @ cov_inv @ res)
            return ll if np.isfinite(ll) else -np.inf
        except Exception:
            return -np.inf

    try:
        samp, ok = _run_sampler(loglike, n_live=NLIVE, pool=pool)
        if not ok:
            return name, {"failed": True,
                          "error": f"timeout > {TIMEOUT_S:.0f}s (first attempt)",
                          **common}
    except ValueError:
        try:
            samp, ok = _run_sampler(loglike, n_live=max(NLIVE * 4, 2000),
                                    pool=pool, n_points_min=50,
                                    split_threshold=400)
            if not ok:
                return name, {"failed": True,
                              "error": f"timeout > {TIMEOUT_S:.0f}s (retry)",
                              **common}
        except Exception as exc2:
            return name, {"failed": True, "error": repr(exc2), **common}
    except Exception as exc:
        return name, {"failed": True, "error": repr(exc), **common}

    pts, lw, log_l, kl_div = _kl_div_from_sampler(samp)
    w = np.exp(lw - lw.max())
    w = w / w.sum()

    def wq(x, q):
        srt = np.argsort(x)
        cum = np.cumsum(w[srt])
        return float(np.interp(q, cum, x[srt]))

    mean = {p: float(np.average(pts[:, i], weights=w))
            for i, p in enumerate(PARAM_ORDER)}
    qq = {p: (wq(pts[:, i], 0.16), wq(pts[:, i], 0.84))
          for i, p in enumerate(PARAM_ORDER)}
    return name, {
        "failed": False, "mean": mean, "quantiles": qq,
        "samples": pts.astype("float32"),
        "log_weights": lw.astype("float32"),
        "log_likes": log_l.astype("float32"),
        "log_z": float(samp.log_z), "kl_div": kl_div,
        **common,
    }


def build_stack_fits(stacks):
    fits_out = {}
    if os.path.exists(FITS_PKL) and USE_CACHE:
        with open(FITS_PKL, "rb") as fh:
            fits_out = pickle.load(fh)
        print(f"  stack_fits: resumed {len(fits_out)} cached fits from "
              f"{FITS_PKL}", flush=True)
    elif os.path.exists(FITS_PKL):
        print(f"  stack_fits: USE_CACHE=False, refitting all stacks",
              flush=True)

    items_todo = [(n, s) for n, s in stacks.items() if n not in fits_out]
    print(f"  stack_fits: fitting {len(items_todo)} stacks sequentially "
          f"(Pool of {NWORKERS} workers per fit)", flush=True)

    import multiprocess as mp
    if items_todo:
        t0 = time.time()
        with mp.Pool(NWORKERS) as nautilus_pool:
            for k, (name, s) in enumerate(items_todo):
                name_out, out = fit_one((name, s), pool=nautilus_pool)
                fits_out[name_out] = out
                with open(FITS_PKL, "wb") as fh:
                    pickle.dump(fits_out, fh)

                dt = time.time() - t0
                eta = dt * (len(items_todo) - k - 1) / (k + 1)
                n_clust = out.get("n_stacked", 0)
                if out.get("failed"):
                    print(f"  [{k + 1:>2}/{len(items_todo)}] {name_out}  "
                          f"N={n_clust:>5,}: FAILED {out['error'][:80]}  "
                          f"elapsed {_fmt_duration(dt)}  eta {_fmt_duration(eta)}",
                          flush=True)
                else:
                    m = out["mean"]
                    kl = out.get("kl_div", float("nan"))
                    print(f"  [{k + 1:>2}/{len(items_todo)}] {name_out}  "
                          f"N={n_clust:>5,}: logM={m['log_M']:.3f}  "
                          f"C={m['C']:.2f}  fmiss={m['FMISS']:.2f}  "
                          f"sig_off={m['SIGMA_OFF']:.3f}  KL={kl:5.2f} nats  "
                          f"elapsed {_fmt_duration(dt)}  eta {_fmt_duration(eta)}",
                          flush=True)

    n_ok   = sum(1 for f in fits_out.values() if not f.get("failed"))
    n_fail = sum(1 for f in fits_out.values() if f.get("failed"))
    print(f"\n  stack_fits: {len(fits_out)} total ({n_ok} ok, {n_fail} "
          f"failed) -> {FITS_PKL}", flush=True)
    return fits_out


# ── Stages 4/5: mass-richness and mass-concentration power laws ─────────

def distribution_linear_fit(X_per_stack, Y_per_stack, Nboot=1000, rng=None):
    """Multivariate linear bootstrap fit Y = X.beta + intercept across N
    stacks, propagating the per-stack input distributions on X and Y —
    see wazp_mass_richness_by_delta_m.ipynb for the full derivation. Each
    of Nboot iterations picks one random paired draw per stack and does an
    OLS fit; the (Nboot, d+1) output IS the bootstrap posterior on the fit
    parameters."""
    rng = np.random.default_rng() if rng is None else rng
    N = len(X_per_stack)
    if N == 0 or N != len(Y_per_stack):
        raise ValueError("X_per_stack/Y_per_stack must have same nonzero length")

    sizes = np.empty(N, dtype=int)
    d = None
    for k in range(N):
        Xk = np.atleast_2d(np.asarray(X_per_stack[k], dtype=float))
        Yk = np.atleast_1d(np.asarray(Y_per_stack[k], dtype=float))
        sizes[k] = Xk.shape[0]
        d = Xk.shape[1] if d is None else d

    params_boot = np.full((Nboot, d + 1), np.nan)
    Xb, Yb = np.empty((N, d)), np.empty(N)
    for b in range(Nboot):
        idxs = rng.integers(0, sizes)
        for k in range(N):
            Xb[k] = X_per_stack[k][idxs[k]]
            Yb[k] = Y_per_stack[k][idxs[k]]
        try:
            params, *_ = np.linalg.lstsq(
                np.column_stack([Xb, np.ones(N)]), Yb, rcond=None,
            )
            params_boot[b] = params
        except np.linalg.LinAlgError:
            pass
    return params_boot


def weighted_posterior_indices(log_w, n_draw, rng):
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()
    return rng.choice(len(log_w), size=n_draw, p=w)


def build_stack_cluster_arrays(stacks, catalog, z_bin, ng_bin, dm_bin):
    out = {}
    for name, st in stacks.items():
        i, j, k = st["z_bin"], st["ng_bin"], st["dm_bin"]
        mask = (z_bin == i) & (ng_bin == j) & (dm_bin == k)
        out[name] = {
            "lambda": np.asarray(catalog["NGALS"])[mask].astype(float),
            "z": np.asarray(catalog["ZPHOT"])[mask].astype(float),
        }
    return out


def build_pairs_M_lambda(fits, cluster_arr, dm_bin_value, min_n, rng, n_draws):
    X_per_stack, Y_per_stack, names_used = [], [], []
    for name, f in fits.items():
        if f.get("dm_bin") != dm_bin_value or f.get("failed", False):
            continue
        if f.get("n_stacked", 0) < min_n:
            continue
        lam_clu = cluster_arr[name]["lambda"]
        z_clu   = cluster_arr[name]["z"]
        if len(lam_clu) == 0:
            continue
        cidx = rng.integers(0, len(lam_clu), size=n_draws)
        lam_samp, z_samp = lam_clu[cidx], z_clu[cidx]

        pts = np.asarray(f["samples"])
        lw  = np.asarray(f["log_weights"])
        idx_post = weighted_posterior_indices(lw, n_draws, rng)
        logM_samp = pts[idx_post, PARAM_ORDER.index("log_M")]

        X = np.column_stack([
            np.log10(lam_samp / LAM_PIV),
            np.log10((1.0 + z_samp) / (1.0 + Z_PIV)),
        ])
        X_per_stack.append(X)
        Y_per_stack.append(logM_samp.astype(float))
        names_used.append(name)
    return X_per_stack, Y_per_stack, names_used


def build_pairs_c_M(fits, cluster_arr, dm_bin_value, min_n, rng, n_draws):
    X_per_stack, Y_per_stack, names_used = [], [], []
    for name, f in fits.items():
        if f.get("dm_bin") != dm_bin_value or f.get("failed", False):
            continue
        if f.get("n_stacked", 0) < min_n:
            continue
        pts = np.asarray(f["samples"])
        lw  = np.asarray(f["log_weights"])
        idx_post = weighted_posterior_indices(lw, n_draws, rng)
        logM_samp = pts[idx_post, PARAM_ORDER.index("log_M")]
        C_samp    = pts[idx_post, PARAM_ORDER.index("C")]
        valid = np.isfinite(logM_samp) & np.isfinite(C_samp) & (C_samp > 0)
        if valid.sum() < 10:
            continue
        logM_samp = logM_samp[valid]
        logC_samp = np.log10(C_samp[valid])

        z_clu = cluster_arr[name]["z"]
        if len(z_clu) == 0:
            continue
        z_samp = z_clu[rng.integers(0, len(z_clu), size=len(logM_samp))]

        X = np.column_stack([
            logM_samp - np.log10(CMR_M_PIV),
            np.log10((1.0 + z_samp) / (1.0 + CMR_Z_PIV)),
        ])
        X_per_stack.append(X)
        Y_per_stack.append(logC_samp.astype(float))
        names_used.append(name)
    return X_per_stack, Y_per_stack, names_used


def build_ml_fits(fits_out, cluster_arr):
    if os.path.exists(ML_PKL) and USE_CACHE:
        with open(ML_PKL, "rb") as fh:
            d = pickle.load(fh)
        print(f"  ml_fits: loaded cached fits from {ML_PKL}", flush=True)
        return d["ml_fits"]

    rng = np.random.default_rng(42)
    n_failed = sum(1 for f in fits_out.values() if f.get("failed"))
    n_lowN = sum(1 for f in fits_out.values()
                 if not f.get("failed") and f.get("n_stacked", 0) < MIN_N_FOR_FIT)
    print(f"  ml_fits: excluding {n_failed} failed + {n_lowN} with "
          f"N<{MIN_N_FOR_FIT}", flush=True)

    ml_fits = {}
    for dm_k in (0, 1):
        X_per_stack, Y_per_stack, names_used = build_pairs_M_lambda(
            fits_out, cluster_arr, dm_k, MIN_N_FOR_FIT, rng, N_DRAWS,
        )
        if len(X_per_stack) < 4:
            print(f"  ml_fits: dm={dm_k}: only {len(X_per_stack)} usable "
                  f"stacks, skipping", flush=True)
            continue
        params_boot = distribution_linear_fit(X_per_stack, Y_per_stack,
                                              Nboot=N_DRAWS, rng=rng)
        mean_v = np.nanmean(params_boot, axis=0)
        cov    = np.cov(params_boot.T, ddof=1)
        sigma  = np.sqrt(np.diag(cov))
        ml_fits[dm_k] = {
            "mean": {"alpha": float(mean_v[0]), "beta": float(mean_v[1]),
                     "logM0": float(mean_v[2])},
            "cov": cov, "sigma": sigma,
            "samples": params_boot.astype("float32"),
            "n_stacks": len(X_per_stack), "stacks_used": names_used,
            "n_draws": N_DRAWS,
        }
        print(f"  ml_fits: dm={dm_k} ({'low' if dm_k == 0 else 'high'} dm)  "
              f"logM0={mean_v[2]:+.3f}+/-{sigma[2]:.3f}  "
              f"alpha={mean_v[0]:+.3f}+/-{sigma[0]:.3f}  "
              f"beta={mean_v[1]:+.3f}+/-{sigma[1]:.3f}  "
              f"(N={len(X_per_stack)} stacks)", flush=True)

    with open(ML_PKL, "wb") as fh:
        pickle.dump({"lam_piv": LAM_PIV, "z_piv": Z_PIV,
                     "min_n_for_fit": MIN_N_FOR_FIT, "n_draws": N_DRAWS,
                     "ml_fits": ml_fits}, fh)
    return ml_fits


def build_cmr_fits(fits_out, cluster_arr):
    if os.path.exists(CMR_PKL) and USE_CACHE:
        with open(CMR_PKL, "rb") as fh:
            d = pickle.load(fh)
        print(f"  cmr_fits: loaded cached fits from {CMR_PKL}", flush=True)
        return d["cmr_fits"]

    rng = np.random.default_rng(42)
    cmr_fits = {}
    for dm_k in (0, 1):
        X_per_stack, Y_per_stack, names_used = build_pairs_c_M(
            fits_out, cluster_arr, dm_k, MIN_N_FOR_FIT, rng, N_DRAWS,
        )
        if len(X_per_stack) < 4:
            print(f"  cmr_fits: dm={dm_k}: only {len(X_per_stack)} usable "
                  f"stacks, skipping", flush=True)
            continue
        params_boot = distribution_linear_fit(X_per_stack, Y_per_stack,
                                              Nboot=N_DRAWS, rng=rng)
        mean_v = np.nanmean(params_boot, axis=0)
        cov    = np.cov(params_boot.T, ddof=1)
        sigma  = np.sqrt(np.diag(cov))
        cmr_fits[dm_k] = {
            "mean": {"alpha": float(mean_v[0]), "beta": float(mean_v[1]),
                     "logC0": float(mean_v[2])},
            "cov": cov, "sigma": sigma,
            "samples": params_boot.astype("float32"),
            "n_stacks": len(X_per_stack), "stacks_used": names_used,
            "n_draws": N_DRAWS,
        }
        print(f"  cmr_fits: dm={dm_k} ({'low' if dm_k == 0 else 'high'} dm)  "
              f"logC0={mean_v[2]:+.3f}+/-{sigma[2]:.3f}  "
              f"alpha={mean_v[0]:+.3f}+/-{sigma[0]:.3f}  "
              f"beta={mean_v[1]:+.3f}+/-{sigma[1]:.3f}  "
              f"(N={len(X_per_stack)} stacks)", flush=True)

    with open(CMR_PKL, "wb") as fh:
        pickle.dump({"M_piv": CMR_M_PIV, "z_piv": CMR_Z_PIV,
                     "min_n_for_fit": MIN_N_FOR_FIT, "n_draws": N_DRAWS,
                     "cmr_fits": cmr_fits}, fh)
    return cmr_fits


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs("data", exist_ok=True)

    print("Stage 0: loading catalogues", flush=True)
    catalog = Table.read(WAZP_CATALOG)
    keep = np.isfinite(catalog["DELTA_M_I"]) & (catalog["DELTA_M_I"] > 0)
    n0 = len(catalog)
    catalog = catalog[keep]
    print(f"  {n0:,} clusters loaded, {len(catalog):,} kept after "
          f"DELTA_M_I > 0 cut", flush=True)

    # Materialise into a plain dict of real ndarrays — NOT kept as the lazy
    # NpzFile np.load() returns. That object holds an open zip handle;
    # forked worker processes sharing it race on the same underlying file
    # position and corrupt each other's lazy reads (BadZipFile errors on
    # effectively every cluster). Same fix prepare_real_data.py already
    # applies when it reloads this same cache.
    _f = np.load(FILTERED_NPZ)
    sr = {k: _f[k] for k in
         ("ra", "dec", "z", "e1", "e2", "w", "R11", "R12", "R21", "R22")}
    print(f"  {len(sr['ra']):,} pre-filtered sources loaded", flush=True)

    print("\nStage 1: per-cluster bkg arrays", flush=True)
    bkg_by_id = build_bkgs(catalog, sr)

    print("\nStage 2: bin + stack", flush=True)
    stacks, z_bin, ng_bin, dm_bin, dm_median_per_cell = build_stacks(
        catalog, bkg_by_id,
    )

    print("\nStage 2b: fit boost(R) model per stack", flush=True)
    fit_boost_models(stacks)

    print("\nStage 3: NFW + miscentering fit per stack", flush=True)
    fits_out = build_stack_fits(stacks)

    cluster_arr = build_stack_cluster_arrays(stacks, catalog, z_bin, ng_bin, dm_bin)

    print("\nStage 4: mass-richness power law per delta_m group", flush=True)
    ml_fits = build_ml_fits(fits_out, cluster_arr)

    print("\nStage 5: mass-concentration power law per delta_m group", flush=True)
    cmr_fits = build_cmr_fits(fits_out, cluster_arr)

    print(f"\nTotal wall time: {_fmt_duration(time.time() - t0)}", flush=True)
    print("\nAll stages done. Run notebooks/wazp_stacked_plots.ipynb to "
          "regenerate the plots from the saved pickles.", flush=True)


if __name__ == "__main__":
    main()
