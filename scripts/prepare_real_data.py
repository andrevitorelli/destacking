"""
Prepare a real cluster + shear catalogue as a gamma_inference.py data_folder.

Reproduces, as a standalone/resumable script, the pipeline previously
prototyped in notebooks/wazp_cluster_selection.ipynb (build the compact
WaZP cluster catalogue, with the magnitude-gap computed from the member
catalogue) and notebooks/wazp_lensing_signals.ipynb (per-cluster weak
lensing measurement against a DES-Y3-like metacalibration catalogue), and
writes the result in the exact on-disk layout destacking.io expects, so it
can be fed straight to gamma_inference.py's ``data_folder`` in
inference-only mode:

    <out_dir>/simulation_g{g_idx:02d}_gamma{gamma_true:+.1f}/
        simulated_data.npz     data_vectors (N, n_bins), cov_matrices (N, n_bins, n_bins)
        truth_catalog.npz      z, M_halo, concentration, delta_m (placeholders — see below)
        cluster_obs.npz        z, Lambda, delta_m   <- REAL: ZPHOT, NGALS, DELTA_M_I
        ols.json               {} (no truth scaling-relation coefficients for real data)

gamma is not a truth field here: real clusters have no known gamma, so
``gamma_true`` in the output directory name and metadata is a pure label
(default 0.0). Point gamma_inference.py's ``data_folder`` at ``out_dir``
with ``simulation: false`` and a matching ``gamma_range`` covering that one
label (e.g. ``[0.0, 0.0]``, ``n_gamma: 1``); read off gamma_est/sigma_gamma
and ignore every "bias" column, which compares against the label, not a
truth. See gamma_inference.py's own docstring, and the note in that
script's answer about gamma_true being decorative in the inference path.

The measurement stage deliberately uses xlensing.data.bootstrap_signal, NOT
xlensing.data.single_cluster: single_cluster applies a boost correction
(B = N_obs/N_random) that is inconsistent with the boost-free NFW model
gamma_inference.py fits (this was a bias source in earlier simulation work
— see [[metacal-response-sim-bug]] project memory). bootstrap_signal's
galaxy-level bootstrap gives the data vector and covariance from the same
estimator, with no boost, exactly matching what the simulation pipeline in
destacking.simulation produces for synthetic clusters.

Background-galaxy selection happens here, NOT inside xlensing:
xlensing.data.metacal_cluster_lensing no longer filters sources by
redshift internally (that hardcoded cut was removed so it can be
controlled and compared from the caller). ``background_strategy`` /
``background_strategy_kwargs`` select and parameterise the selection
(destacking.data.select_background); the default ("margin", factor=1.1,
offset=0.2) reproduces the old hardcoded xlensing behaviour exactly.

Usage
-----
    python prepare_real_data.py my_real_data_config.yaml

See real_data_config_example.yaml for every key and its default.

Resumability
------------
Both expensive stages cache to disk and are skipped on a matching re-run:
  - the metacal+photo-z pre-filter -> filtered_cache (use_cache gates reuse)
  - the per-cluster measurement loop -> <out_dir>/<subdir>/.partial.npz,
    checkpointed every ``checkpoint_every`` clusters and resumed by index
    on restart (use_cache gates reuse; set false to force a clean redo).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np


# ── Config loading ──────────────────────────────────────────────────────────

def _load_yaml_config(path):
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    print(f"[prepare_real_data] loaded config from {path}", flush=True)
    return cfg


def _cfg(cfg, key, default, cast=None, required=False):
    if key not in cfg or cfg[key] is None:
        if required:
            raise SystemExit(f"config key '{key}' is required")
        return default
    v = cfg[key]
    return cast(v) if cast is not None else v


def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    return f"{int(h)}h{int(rem/60):02d}m"


# ── Stage 1: compact cluster catalogue (WaZP cluster + member -> one table) ──

def build_cluster_catalog(cfg):
    """Return the compact real-data cluster Table: ID, RA, DEC, NGALS,
    ZPHOT, DELTA_M_I, DELTA_M_I_ERR.

    If ``cluster_catalog_file`` already exists and use_cache is set, it is
    loaded directly (this is the common case: the catalogue was already
    built, e.g. by a previous run of this script, or by
    notebooks/wazp_cluster_selection.ipynb). Otherwise, when
    ``clusters_file``/``members_file`` are given, the compact catalogue is
    (re)built from the raw WaZP cluster + member FITS catalogues and
    written to ``cluster_catalog_file``.
    """
    from astropy.io import fits
    from astropy.table import Table
    from astropy.cosmology import Planck18 as COSMO
    from astropy import units as u

    out_path   = _cfg(cfg, "cluster_catalog_file", None)
    use_cache  = _cfg(cfg, "use_cache", True)
    clusters_f = _cfg(cfg, "clusters_file", None)
    members_f  = _cfg(cfg, "members_file", None)

    if out_path and use_cache and os.path.exists(out_path):
        tab = Table.read(out_path)
        print(f"  cluster catalogue: loaded {len(tab):,} rows (cached) "
              f"from {out_path}", flush=True)
        return tab

    if not (clusters_f and members_f):
        if out_path and os.path.exists(out_path):
            tab = Table.read(out_path)
            print(f"  cluster catalogue: loaded {len(tab):,} rows "
                  f"from {out_path} (use_cache=false, but no raw catalogues "
                  f"given to rebuild from)", flush=True)
            return tab
        raise SystemExit(
            "no cluster catalogue available: set cluster_catalog_file to an "
            "existing compact catalogue, or set clusters_file + members_file "
            "to build one from the raw WaZP catalogues"
        )

    richness_min = _cfg(cfg, "richness_min", 25.0, cast=float)
    z_max        = _cfg(cfg, "z_max", 0.7, cast=float)
    require_bcg  = _cfg(cfg, "require_bcg", True)
    r_cut        = _cfg(cfg, "mag_gap_r_cut_mpc", 0.5, cast=float)

    print(f"  building cluster catalogue from {clusters_f} + {members_f} "
          f"...", flush=True)
    with fits.open(clusters_f, memmap=True) as h:
        cl = h[1].data.copy()
    with fits.open(members_f, memmap=True) as h:
        members = Table(h[1].data)
    print(f"    {len(cl):,} clusters, {len(members):,} member rows",
          flush=True)

    # Projected distance of every member to its parent cluster centre [Mpc]
    max_id    = int(cl["ID"].max())
    id_to_idx = np.full(max_id + 1, -1, dtype=np.int64)
    id_to_idx[cl["ID"]] = np.arange(len(cl))
    parent_idx = id_to_idx[np.asarray(members["ID_CLUSTER"])]
    if (parent_idx < 0).any():
        n_bad = int((parent_idx < 0).sum())
        print(f"    WARNING: {n_bad:,} member rows reference unknown "
              f"cluster IDs — dropped", flush=True)
        keep_member = parent_idx >= 0
        members, parent_idx = members[keep_member], parent_idx[keep_member]

    def _angsep_rad(ra1, dec1, ra2, dec2):
        r1, d1 = np.deg2rad(ra1), np.deg2rad(dec1)
        r2, d2 = np.deg2rad(ra2), np.deg2rad(dec2)
        a = (np.sin((d2 - d1) / 2.0)) ** 2 + \
            np.cos(d1) * np.cos(d2) * (np.sin((r2 - r1) / 2.0)) ** 2
        return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    parent_ra, parent_dec = cl["RA"][parent_idx], cl["DEC"][parent_idx]
    parent_z = cl["ZPHOT"][parent_idx]
    theta_rad = _angsep_rad(parent_ra, parent_dec,
                            np.asarray(members["RA"]),
                            np.asarray(members["DEC"]))
    D_A_cluster = COSMO.angular_diameter_distance(cl["ZPHOT"]).to_value(u.Mpc)
    r_mpc = (theta_rad * D_A_cluster[parent_idx]).astype("float32")

    # Magnitude gap per cluster (i-band), BCG vs brightest inner satellite
    from destacking.data import magnitude_gaps

    mem_idc   = np.asarray(members["ID_CLUSTER"])
    order     = np.argsort(mem_idc, kind="stable")
    mem_idc_s = mem_idc[order]
    mag_i_s   = np.asarray(members["MAG_I"])[order]
    pmem_s    = np.asarray(members["PMEM"])[order]
    r_mpc_s   = r_mpc[order]
    bcg_s     = np.asarray(members["FLAG_CG"])[order]
    start_idx = np.searchsorted(mem_idc_s, cl["ID"], side="left")
    end_idx   = np.searchsorted(mem_idc_s, cl["ID"], side="right")

    delta_m   = np.full(len(cl), np.nan, dtype="float64")
    delta_m_e = np.full(len(cl), np.nan, dtype="float64")
    t0 = time.time()
    for k in range(len(cl)):
        s, e = start_idx[k], end_idx[k]
        if e == s:
            continue
        bcg_mask = bcg_s[s:e]
        if not bcg_mask.any():
            continue
        bcg_pos = s + int(np.argmax(bcg_mask))
        mag_bcg = mag_i_s[bcg_pos]
        sat_slice = np.r_[s:bcg_pos, bcg_pos + 1:e]
        if sat_slice.size == 0:
            continue
        mag_gap, mag_gap_e = magnitude_gaps(
            mag_bcg, mag_i_s[sat_slice], pmem_s[sat_slice],
            r_mpc_s[sat_slice], r_cut=r_cut,
        )
        delta_m[k], delta_m_e[k] = mag_gap, mag_gap_e
        if (k + 1) % 20000 == 0:
            dt = time.time() - t0
            print(f"    magnitude gaps: {k+1:,}/{len(cl):,}  "
                  f"({dt:.0f}s elapsed)", flush=True)

    cl_tab = Table(cl)
    cl_tab["DELTA_M_I"]     = delta_m.astype("float32")
    cl_tab["DELTA_M_I_ERR"] = delta_m_e.astype("float32")

    bcg_cluster_ids = np.unique(mem_idc[np.asarray(members["FLAG_CG"])])
    cut_richness = cl_tab["NGALS"] > richness_min
    cut_z        = cl_tab["ZPHOT"] < z_max
    cut_has_bcg  = (np.isin(cl_tab["ID"], bcg_cluster_ids)
                    if require_bcg else np.ones(len(cl_tab), dtype=bool))
    selection = cut_richness & cut_z & cut_has_bcg
    n0 = len(cl_tab)
    print(f"    selection: NGALS>{richness_min}: {int(cut_richness.sum()):,}  "
          f"ZPHOT<{z_max}: {int(cut_z.sum()):,}  "
          f"has BCG: {int(cut_has_bcg.sum()):,}  "
          f"-> {int(selection.sum()):,}/{n0:,}", flush=True)

    selected = cl_tab[selection]
    final = Table()
    final["ID"]            = selected["ID"]
    final["RA"]            = selected["RA"]
    final["DEC"]           = selected["DEC"]
    final["NGALS"]         = selected["NGALS"]
    final["ZPHOT"]         = selected["ZPHOT"]
    final["DELTA_M_I"]     = selected["DELTA_M_I"]
    final["DELTA_M_I_ERR"] = selected["DELTA_M_I_ERR"]

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        final.write(out_path, overwrite=True)
        print(f"    wrote {len(final):,} cluster rows to {out_path}",
              flush=True)
    return final


# ── Stage 2: metacal + photo-z pre-filter ────────────────────────────────────

def build_source_prefilter(cfg):
    """Return a dict of 1-D arrays (ra, dec, z, e1, e2, w, R11, R12, R21,
    R22) — the metacal source catalogue, cut to the standard selection
    (flags==0, snr in [10,1000], size_ratio>0.5, T>0, zmean_sof>0) and
    row-matched against a photo-z (DNF) catalogue on coadd_object_id order.
    Streams both HDF5 files in chunks so memory stays bounded regardless of
    catalogue size. Cached to ``filtered_cache`` (gated by use_cache).
    """
    import h5py

    cache      = _cfg(cfg, "filtered_cache",
                      os.path.expanduser("~/destacking_data/real_data/metacal_filtered.npz"))
    use_cache  = _cfg(cfg, "use_cache", True)
    metacal_f  = _cfg(cfg, "metacal_file", None, required=True)
    dnf_f      = _cfg(cfg, "dnf_file", None, required=True)
    chunk_read = _cfg(cfg, "chunk_read", 10_000_000, cast=int)

    if cache and use_cache and os.path.exists(cache):
        f = np.load(cache)
        sr = {k: f[k] for k in ("ra", "dec", "z", "e1", "e2", "w",
                                "R11", "R12", "R21", "R22")}
        print(f"  source pre-filter: loaded {len(sr['ra']):,} sources "
              f"(cached) from {cache}", flush=True)
        return sr

    print(f"  building source pre-filter from {metacal_f} + {dnf_f} ...",
          flush=True)
    cols = {k: [] for k in ("ra", "dec", "z", "e1", "e2", "w",
                            "R11", "R12", "R21", "R22")}
    with h5py.File(metacal_f, "r") as fm, h5py.File(dnf_f, "r") as fz:
        g, gz = fm["catalog/unsheared"], fz["catalog/unsheared"]
        n_tot = g["ra"].shape[0]
        if n_tot != gz["zmean_sof"].shape[0]:
            raise RuntimeError("metacal/DNF row count mismatch")
        if int(g["coadd_object_id"][0]) != int(gz["coadd_object_id"][0]) or \
           int(g["coadd_object_id"][n_tot - 1]) != int(gz["coadd_object_id"][n_tot - 1]):
            raise RuntimeError(
                "metacal/DNF coadd_object_id mismatch at row 0 or -1 — "
                "the two files are not row-aligned"
            )
        t0 = time.time()
        for s in range(0, n_tot, chunk_read):
            e = min(s + chunk_read, n_tot)
            flags = g["flags"][s:e]
            snr   = g["snr"][s:e]
            sr_   = g["size_ratio"][s:e]
            T     = g["T"][s:e]
            zmean = gz["zmean_sof"][s:e]
            sel = ((flags == 0) & (snr > 10) & (snr < 1000)
                   & (sr_ > 0.5) & (T > 0) & (zmean > 0))
            if not sel.any():
                continue
            idx = np.where(sel)[0]
            cols["ra"].append(np.deg2rad(g["ra"][s:e][idx]).astype("float64"))
            cols["dec"].append(np.deg2rad(g["dec"][s:e][idx]).astype("float64"))
            cols["z"].append(zmean[idx].astype("float32"))
            cols["e1"].append(g["e_1"][s:e][idx].astype("float32"))
            cols["e2"].append(g["e_2"][s:e][idx].astype("float32"))
            cols["w"].append(g["weight"][s:e][idx].astype("float32"))
            cols["R11"].append(g["R11"][s:e][idx].astype("float32"))
            cols["R12"].append(g["R12"][s:e][idx].astype("float32"))
            cols["R21"].append(g["R21"][s:e][idx].astype("float32"))
            cols["R22"].append(g["R22"][s:e][idx].astype("float32"))
            dt   = time.time() - t0
            kept = sum(len(x) for x in cols["ra"])
            eta  = dt * (n_tot - e) / max(e, 1)
            print(f"    {e:>11,}/{n_tot:,}  kept {kept:>11,}  "
                  f"elapsed {dt:6.1f}s  eta {eta:6.1f}s", flush=True)

    sr = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"    total kept: {len(sr['ra']):,}", flush=True)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, **sr)
        print(f"    wrote pre-filter cache to {cache}", flush=True)
    return sr


# ── Stage 3: per-cluster lensing measurement ─────────────────────────────────

# Pool-worker globals, set right before Pool creation so fork() shares them
# read-only (copy-on-write) — no per-task pickling of the tree/arrays.
_TREE         = None
_SR           = None
_RADIUS_MPC   = None
_BIN_LIMITS   = None
_N_COV_BOOT   = None
_MIN_SOURCES  = None
_BG_STRATEGY  = None
_BG_KWARGS    = None
_SIGMA_E      = None
_FLOOR_RMAX   = None


def _measure_cluster(args):
    """Mirrors destacking.simulation._process_cluster's stages 5-6 (measure
    + bootstrap_signal), but on REAL sources found within a BallTree cone
    query instead of an injected shear catalogue. Returns
    (i, dv, cov, n_src) or (i, None, None, 0) on failure."""
    import xlensing.cosmo as _xlc
    from xlensing.data import bootstrap_signal, metacal_cluster_lensing
    from destacking.data import select_background

    i, cid, ra_deg, dec_deg, z_cl = args
    try:
        ra_r, dec_r = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
        D_A = float(_xlc.DA(0, z_cl))
        if not (D_A > 0):
            return i, None, None, 0
        ang_r = _RADIUS_MPC / D_A
        idx = _TREE.query_radius(np.array([[dec_r, ra_r]]), r=ang_r)[0]
        if len(idx) == 0:
            return i, None, None, 0

        # Background selection is caller-side now — xlensing applies no
        # redshift filtering of its own (see project memory
        # "xlensing background selection moved to caller"). Apply the
        # configured strategy before building the source tuple.
        bg_mask = select_background(_BG_STRATEGY, _SR["z"][idx], z_cl,
                                    **_BG_KWARGS)
        idx = idx[bg_mask]
        if len(idx) == 0:
            return i, None, None, 0

        sources = (
            _SR["ra"][idx], _SR["dec"][idx], _SR["z"][idx],
            _SR["e1"][idx], _SR["e2"][idx], _SR["w"][idx],
            _SR["R11"][idx], _SR["R12"][idx], _SR["R21"][idx], _SR["R22"][idx],
        )
        bg = metacal_cluster_lensing((ra_r, dec_r, z_cl), sources,
                                     radius=_RADIUS_MPC)
        if bg["Count"] < _MIN_SOURCES:
            return i, None, None, int(bg["Count"])

        # Standard xlensing stake layout: rows (Sigma_crit, e_t, e_x, W, R, M).
        stake = np.array([
            bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
            bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"],
        ])

        # bootstrap_signal, NOT single_cluster: no boost correction, so the
        # data vector and covariance are consistent with the boost-free NFW
        # model gamma_inference.py fits. See module docstring.
        #
        # sigma_e/floor_rmax: floors each bin's bootstrap variance at the
        # shot-noise-only lower bound sigma_e**2/sum((1+M)W/Sigma_crit**2)
        # for bins below floor_rmax. Galaxy-level bootstrap with only 2-3
        # galaxies in an inner annulus can land on an implausibly tiny
        # variance by chance (their weighted ratio barely moves across
        # resamples), which single-handedly dominates the joint likelihood
        # once r_min_fit no longer excludes that bin — see project memory
        # wazp-desy3-root-cause-hypotheses.
        sigmas, sigmas_cov, _, _ = bootstrap_signal(
            stake, _BIN_LIMITS, Nboot=_N_COV_BOOT,
            sigma_e=_SIGMA_E, floor_rmax=_FLOOR_RMAX,
        )
        valid_bins = np.isfinite(sigmas)
        if not valid_bins.any():
            return i, None, None, int(bg["Count"])

        cov = np.where(np.isfinite(sigmas_cov), sigmas_cov, 0.0)
        cov[~valid_bins, ~valid_bins] = 1.0
        return i, sigmas, cov, int(bg["Count"])
    except Exception as exc:
        print(f"    cluster {cid} FAILED: {exc!r}", flush=True)
        return i, None, None, 0


def measure_all_clusters(catalog, sr, cfg, sim_subdir):
    """Measure every cluster in ``catalog`` against the pre-filtered source
    sample ``sr``. Resumable: partial results are checkpointed to
    ``<sim_subdir>/.partial.npz`` every ``checkpoint_every`` clusters and
    reloaded on a matching re-run (gated by use_cache)."""
    global _TREE, _SR, _RADIUS_MPC, _BIN_LIMITS, _N_COV_BOOT, _MIN_SOURCES
    global _BG_STRATEGY, _BG_KWARGS, _SIGMA_E, _FLOOR_RMAX
    import multiprocess as mp
    from sklearn.neighbors import BallTree

    radius_mpc      = _cfg(cfg, "radius_mpc", 5.0, cast=float)
    n_cov_bootstrap  = _cfg(cfg, "n_cov_bootstrap", 200, cast=int)
    min_sources      = _cfg(cfg, "min_sources", 20, cast=int)
    n_workers        = _cfg(cfg, "n_workers", 8, cast=int)
    checkpoint_every  = _cfg(cfg, "checkpoint_every", 1000, cast=int)
    use_cache        = _cfg(cfg, "use_cache", True)

    # Shot-noise variance floor for low-galaxy-count bins (see
    # _measure_cluster's comment). None (default) reproduces the old
    # behaviour exactly -- no floor.
    sigma_e    = _cfg(cfg, "sigma_e", None, cast=float)
    floor_rmax = _cfg(cfg, "floor_rmax", None, cast=float)

    # Background-galaxy selection: caller-side now (xlensing applies no
    # redshift filtering internally any more). Default ("margin", factor
    # 1.1, offset 0.2) reproduces the old hardcoded xlensing metacal
    # behaviour exactly; override to test other selection strategies.
    bg_strategy = _cfg(cfg, "background_strategy", "margin")
    bg_kwargs   = _cfg(cfg, "background_strategy_kwargs",
                       {"factor": 1.1, "offset": 0.2})

    log_inner = _cfg(cfg, "log_inner_radius", -0.8, cast=float)
    log_outer = _cfg(cfg, "log_outer_radius", 0.6, cast=float)
    n_radii   = _cfg(cfg, "n_radii", 8, cast=int)
    radii = np.logspace(log_inner, log_outer, n_radii)
    log_r = np.log10(radii)
    dlog  = np.diff(log_r)[0]
    edges = np.logspace(log_r[0] - dlog / 2, log_r[-1] + dlog / 2,
                        len(radii) + 1)
    bin_limits = np.column_stack([edges[:-1], edges[1:]])
    n_bins = len(radii)

    n = len(catalog)
    data_vectors = np.full((n, n_bins), np.nan)
    cov_matrices = np.zeros((n, n_bins, n_bins))
    for i in range(n):
        np.fill_diagonal(cov_matrices[i], 1.0)
    n_src_arr = np.zeros(n, dtype=np.int64)
    done_mask = np.zeros(n, dtype=bool)

    partial_path = os.path.join(sim_subdir, ".partial.npz")
    if use_cache and os.path.exists(partial_path):
        p = np.load(partial_path)
        data_vectors = p["data_vectors"]
        cov_matrices = p["cov_matrices"]
        n_src_arr    = p["n_src_arr"]
        done_mask    = p["done_mask"]
        print(f"  measurement: resuming from checkpoint, "
              f"{int(done_mask.sum()):,}/{n:,} clusters already done",
              flush=True)

    todo = np.where(~done_mask)[0]
    if len(todo) == 0:
        print("  measurement: all clusters already done (cached)", flush=True)
        return data_vectors, cov_matrices, n_src_arr, radii, bin_limits

    print(f"  measurement: building BallTree over {len(sr['ra']):,} "
          f"sources ...", flush=True)
    tree = BallTree(np.column_stack([sr["dec"], sr["ra"]]), metric="haversine")

    _TREE, _SR = tree, sr
    _RADIUS_MPC, _BIN_LIMITS = radius_mpc, bin_limits
    _N_COV_BOOT, _MIN_SOURCES = n_cov_bootstrap, min_sources
    _BG_STRATEGY, _BG_KWARGS = bg_strategy, bg_kwargs
    _SIGMA_E, _FLOOR_RMAX = sigma_e, floor_rmax

    work = [
        (int(i), int(catalog["ID"][i]), float(catalog["RA"][i]),
         float(catalog["DEC"][i]), float(catalog["ZPHOT"][i]))
        for i in todo
    ]
    print(f"  measurement: {len(work):,} clusters to measure "
          f"({n_workers} workers, radius={radius_mpc} Mpc, "
          f"n_cov_bootstrap={n_cov_bootstrap})", flush=True)

    t0 = time.time()
    n_ok = 0
    with mp.Pool(n_workers) as pool:
        for k, (i, dv, cov, n_src) in enumerate(
            pool.imap_unordered(_measure_cluster, work, chunksize=4)
        ):
            done_mask[i] = True
            n_src_arr[i] = n_src
            if dv is not None:
                data_vectors[i] = dv
                cov_matrices[i] = cov
                n_ok += 1
            if (k + 1) % 200 == 0 or (k + 1) == len(work):
                dt  = time.time() - t0
                eta = dt * (len(work) - k - 1) / (k + 1)
                print(f"    {k+1:>8,}/{len(work):,}  good {n_ok:>8,}  "
                      f"elapsed {_fmt_duration(dt)}  eta {_fmt_duration(eta)}",
                      flush=True)
            if (k + 1) % checkpoint_every == 0:
                os.makedirs(sim_subdir, exist_ok=True)
                np.savez(partial_path, data_vectors=data_vectors,
                         cov_matrices=cov_matrices, n_src_arr=n_src_arr,
                         done_mask=done_mask)

    os.makedirs(sim_subdir, exist_ok=True)
    np.savez(partial_path, data_vectors=data_vectors,
             cov_matrices=cov_matrices, n_src_arr=n_src_arr,
             done_mask=done_mask)
    print(f"  measurement done: {n_ok:,}/{n:,} clusters usable "
          f"(dropped: fewer than {min_sources} sources in the aperture, "
          f"or every radial bin failed the galaxy bootstrap)", flush=True)
    return data_vectors, cov_matrices, n_src_arr, radii, bin_limits


# ── Stage 4: write the gamma_inference.py checkpoint ─────────────────────────

def write_checkpoint(catalog, data_vectors, cov_matrices, radii, cfg,
                     sim_subdir):
    from astropy.table import Table
    from destacking.io import save_simulation_checkpoint
    from destacking.data import assign_stack_bins

    n = len(catalog)
    z       = np.asarray(catalog["ZPHOT"], dtype=float)
    lam     = np.asarray(catalog["NGALS"], dtype=float)
    delta_m = np.asarray(catalog["DELTA_M_I"], dtype=float)

    delta_m_min = _cfg(cfg, "delta_m_min", 0.0, cast=float)
    valid_dm = np.isfinite(delta_m) & (delta_m > delta_m_min)
    if not valid_dm.all():
        print(f"  {int((~valid_dm).sum()):,}/{n:,} clusters have "
              f"DELTA_M_I <= {delta_m_min} or non-finite — kept in the "
              f"output (the model falls back to the delta_m_ref pivot for "
              f"these, i.e. no gamma correction) but flagged for review",
              flush=True)

    # Stack membership (3 z x 4 NGALS x 2 delta_m, median-split per cell) —
    # same binning wazp_stacked_analysis.py stacks clusters and fits a
    # boost(R) model under (destacking.data.assign_stack_bins, shared so
    # the two never disagree). Lets gamma_inference.py's `boosts:` config
    # key look up and apply each cluster's stack's fitted boost correction.
    _, _, _, _, stack_name = assign_stack_bins(z, lam, delta_m)
    n_stacked = int((stack_name != "").sum())
    print(f"  {n_stacked:,}/{n:,} clusters assigned a boost-fit stack "
          f"({n - n_stacked:,} outside the z/richness/delta_m bin ranges)",
          flush=True)

    # cluster_obs: the REAL observables the fit reads (z, Lambda, delta_m),
    # plus stack_name for the optional boost correction.
    cluster_obs = Table({"z": z, "Lambda": lam, "delta_m": delta_m,
                         "stack_name": stack_name})

    # truth_catalog: schema-compatible placeholder. M_halo/concentration are
    # genuinely unknown for real data and are never read by the inference
    # path (destacking.io.load_simulation_checkpoint loads this file but
    # gamma_inference.infer_one_gamma does not use its contents) — kept only
    # so the checkpoint has the same shape as a simulated one.
    truth = Table({
        "z":             z,
        "M_halo":        np.full(n, np.nan),
        "concentration": np.full(n, np.nan),
        "delta_m":       delta_m,
    })

    ols = {}   # no scaling-relation truth for real data

    save_simulation_checkpoint(sim_subdir, truth, ols, cluster_obs,
                               data_vectors, cov_matrices)
    print(f"  wrote checkpoint to {sim_subdir}", flush=True)
    print(f"    simulated_data.npz : data_vectors {data_vectors.shape}, "
          f"cov_matrices {cov_matrices.shape}", flush=True)
    print(f"    radii [Mpc]        : {np.round(radii, 3).tolist()}",
          flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: python {sys.argv[0]} config.yaml")
    cfg = _load_yaml_config(sys.argv[1])

    out_dir     = _cfg(cfg, "out_dir", None, required=True)
    g_idx       = _cfg(cfg, "g_idx", 0, cast=int)
    gamma_true  = _cfg(cfg, "gamma_true", 0.0, cast=float)

    from destacking.utils import gamma_dir_name
    sim_subdir = os.path.join(out_dir, gamma_dir_name(g_idx, gamma_true))
    os.makedirs(sim_subdir, exist_ok=True)

    # Provenance: keep a copy of this run's config alongside the checkpoint.
    import shutil
    cfg_dest = os.path.join(sim_subdir, os.path.basename(sys.argv[1]))
    if not os.path.exists(cfg_dest):
        shutil.copy2(sys.argv[1], cfg_dest)

    t0 = time.time()
    print("Stage 1: cluster catalogue", flush=True)
    catalog = build_cluster_catalog(cfg)

    print("Stage 2: source pre-filter", flush=True)
    sr = build_source_prefilter(cfg)

    print("Stage 3: per-cluster measurement", flush=True)
    data_vectors, cov_matrices, n_src_arr, radii, bin_limits = \
        measure_all_clusters(catalog, sr, cfg, sim_subdir)

    print("Stage 4: writing checkpoint", flush=True)
    write_checkpoint(catalog, data_vectors, cov_matrices, radii, cfg,
                     sim_subdir)

    print(f"\nTotal wall time: {_fmt_duration(time.time() - t0)}", flush=True)
    print(f"\nTo fit this data, point gamma_inference.py at it with e.g.:\n"
          f"  simulation: false\n"
          f"  inference:  true\n"
          f"  data_folder: {out_dir}\n"
          f"  gamma_range: [{gamma_true}, {gamma_true}]\n"
          f"  n_gamma: 1\n"
          f"  fit_log_inner_radius: {_cfg(cfg, 'log_inner_radius', -0.8)}\n"
          f"  fit_log_outer_radius: {_cfg(cfg, 'log_outer_radius', 0.6)}\n"
          f"  fit_n_radii: {_cfg(cfg, 'n_radii', 8)}\n"
          f"(fit_* MUST match the log_inner_radius/log_outer_radius/n_radii "
          f"used here). See real_data_config_example.yaml.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
