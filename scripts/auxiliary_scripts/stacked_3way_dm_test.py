#!/usr/bin/env python3
"""
One-off exploratory test (2026-09-09): does splitting into 3 Delta-m
groups (terciles) instead of 2 (median) make stacking's mass-concentration
constraint on the magnitude-gap dependence better or worse? Hypothesis
(user): worse, since each group already has few stacks (~10-12 of 24) at
a 2-way split, and 3-way splits the same fixed 24-stack budget further.

NOT a modification of the standard pipeline: assign_stack_bins,
build_stacks, and build_cmr_fits are all hardcoded to a 2-way (median)
Delta-m split (destacking.data.assign_stack_bins; wazp_stacked_analysis.py's
`for k in range(2)` loops) by design, matching the notebooks/real-data
convention (see project instruction: "Only two bins in Delta M"). This
script reimplements just the 3-way binning + stacking + cmr-fit loop as a
standalone, one-off comparison, reusing every other stage
(xlensing.data.stacked_signal, build_stack_fits, build_stack_cluster_arrays,
build_pairs_c_M, distribution_linear_fit) unmodified from
wazp_stacked_analysis.py -- those are already generic over how many
Delta-m groups there are.

Usage: python scripts/auxiliary_scripts/stacked_3way_dm_test.py <config.yaml>
(same config format as stacked_sim_analysis.py: sim_prefix, sims_dir,
output_dir, n_workers)
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import importlib.util

import numpy as np
import yaml
from astropy.table import Table

_WSA_PATH = os.path.join(os.path.dirname(__file__), "wazp_stacked_analysis.py")
_spec = importlib.util.spec_from_file_location("wazp_stacked_analysis", _WSA_PATH)
wsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsa)

import xlensing  # noqa: E402  (after wsa exec, matching its own import order)


def _cfg(cfg, key, default=None, cast=None):
    v = cfg.get(key, default)
    return cast(v) if (cast is not None and v is not None) else v


def load_sim_catalog_and_bkgs(sims_dir, sim_prefix):
    prefix = f"{sim_prefix}000"
    cat_path = os.path.join(sims_dir, f"{prefix}_real_data.fits")
    bgs_path = os.path.join(sims_dir, f"{prefix}_bgs.pkl")
    catalog = Table.read(cat_path)
    keep = np.isfinite(catalog["DELTA_M_I"]) & (catalog["DELTA_M_I"] > 0)
    n0 = len(catalog)
    catalog = catalog[keep]
    print(f"  {n0:,} clusters loaded, {len(catalog):,} kept after "
          f"DELTA_M_I > 0 cut", flush=True)
    with open(bgs_path, "rb") as fh:
        bkg_by_id = pickle.load(fh)
    print(f"  {len(bkg_by_id):,} precomputed simulated bkg arrays loaded "
          f"from {bgs_path}", flush=True)
    return catalog, bkg_by_id


def assign_stack_bins_3way(z, ngals, delta_m, z_edges, ng_edges):
    """Same per-(z,NGALS)-cell logic as destacking.data.assign_stack_bins,
    but a tercile split (3 groups, k=0/1/2 low/mid/high) instead of a
    median split (2 groups) within each cell. Cells with fewer than 3
    valid delta_m entries can't support 2 tercile cuts and get no dm_bin
    (-1), same spirit as the original's `cell.sum() < 2` guard.
    """
    n_ng = len(ng_edges) - 1
    z, ngals, delta_m = (np.asarray(a, dtype=float) for a in (z, ngals, delta_m))
    n = len(z)

    z_bin = wsa.assign_stack_bins.__globals__["_stack_bin_idx"](z, z_edges)
    ng_bin = wsa.assign_stack_bins.__globals__["_stack_bin_idx"](ngals, ng_edges)
    dm_valid = np.isfinite(delta_m) & (delta_m > 0)

    dm_bin = np.full(n, -1, dtype=int)
    dm_terciles_per_cell = {}
    for i in range(len(z_edges) - 1):
        for j in range(n_ng):
            cell = (z_bin == i) & (ng_bin == j) & dm_valid
            if cell.sum() < 3:
                dm_terciles_per_cell[(i, j)] = None
                continue
            cell_dm = delta_m[cell]
            t1, t2 = np.percentile(cell_dm, [100 / 3, 200 / 3])
            dm_terciles_per_cell[(i, j)] = (float(t1), float(t2))
            glob_idx = np.where(cell)[0]
            dm_bin[glob_idx[cell_dm < t1]] = 0
            dm_bin[glob_idx[(cell_dm >= t1) & (cell_dm < t2)]] = 1
            dm_bin[glob_idx[cell_dm >= t2]] = 2

    valid = (z_bin >= 0) & (ng_bin >= 0) & (dm_bin >= 0)
    stack_name = np.full(n, "", dtype=object)
    for idx in np.where(valid)[0]:
        stack_name[idx] = f"z{z_bin[idx]}_ng{ng_bin[idx]}_dm{dm_bin[idx]}"
    z_bin = np.where(z_bin >= 0, z_bin, -1)
    ng_bin = np.where(ng_bin >= 0, ng_bin, -1)
    return z_bin, ng_bin, dm_bin, dm_terciles_per_cell, stack_name


def build_stacks_3way(catalog, bkg_by_id):
    z_bin, ng_bin, dm_bin, dm_terciles_per_cell, _ = assign_stack_bins_3way(
        catalog["ZPHOT"], catalog["NGALS"], catalog["DELTA_M_I"],
        z_edges=wsa.Z_EDGES, ng_edges=wsa.NG_EDGES,
    )
    stacks = {}
    for i in range(3):
        for j in range(wsa.N_NG):
            for k in range(3):
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
                    xlensing.data.stacked_signal(bkgs, wsa.BINS_LIMS, wsa.NBOOT)
                )
                cell_z = np.asarray(catalog["ZPHOT"])[mask]
                cell_lam = np.asarray(catalog["NGALS"])[mask]
                cell_dm = np.asarray(catalog["DELTA_M_I"])[mask]
                stacks[name] = {
                    "z_bin": i, "ng_bin": j, "dm_bin": k,
                    "z_lo": float(wsa.Z_EDGES[i]), "z_hi": float(wsa.Z_EDGES[i + 1]),
                    "ng_lo": float(wsa.NG_EDGES[j]), "ng_hi": float(wsa.NG_EDGES[j + 1]),
                    "dm_terciles": dm_terciles_per_cell[(i, j)],
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
                print(f"  stacks3: {name}  <z>={cell_z.mean():.3f}  "
                      f"<lam>={cell_lam.mean():6.1f}  <dm>={cell_dm.mean():.2f}  "
                      f"N={len(bkgs):,}  B_outer={boosts[-1]:.3f}  "
                      f"({time.time() - t0:.1f}s)", flush=True)
    n_expected = 3 * wsa.N_NG * 3
    print(f"\n  stacks3: produced {len(stacks)}/{n_expected}", flush=True)
    return stacks, z_bin, ng_bin, dm_bin


def build_cmr_fits_3way(fits_out, cluster_arr, min_n_for_fit, n_draws):
    rng = np.random.default_rng(42)
    cmr_fits = {}
    for dm_k in (0, 1, 2):
        X_per_stack, Y_per_stack, names_used = wsa.build_pairs_c_M(
            fits_out, cluster_arr, dm_k, min_n_for_fit, rng, n_draws,
        )
        if len(X_per_stack) < 4:
            print(f"  cmr_fits3: dm={dm_k}: only {len(X_per_stack)} usable "
                  f"stacks, skipping", flush=True)
            continue
        params_boot = wsa.distribution_linear_fit(X_per_stack, Y_per_stack,
                                                   Nboot=n_draws, rng=rng)
        mean_v = np.nanmean(params_boot, axis=0)
        cov = np.cov(params_boot.T, ddof=1)
        sigma = np.sqrt(np.diag(cov))
        cmr_fits[dm_k] = {
            "mean": {"alpha": float(mean_v[0]), "beta": float(mean_v[1]),
                     "logC0": float(mean_v[2])},
            "cov": cov, "sigma": sigma,
            "n_stacks": len(X_per_stack), "stacks_used": names_used,
        }
        print(f"  cmr_fits3: dm={dm_k} (tercile {dm_k+1}/3)  "
              f"logC0={mean_v[2]:+.3f}+/-{sigma[2]:.3f}  "
              f"alpha={mean_v[0]:+.3f}+/-{sigma[0]:.3f}  "
              f"beta={mean_v[1]:+.3f}+/-{sigma[1]:.3f}  "
              f"(N={len(X_per_stack)} stacks)", flush=True)
    return cmr_fits


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: python {sys.argv[0]} config.yaml")
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)

    sim_prefix = _cfg(cfg, "sim_prefix")
    sims_dir = os.path.expanduser(_cfg(cfg, "sims_dir", "~/destacking_data/sims"))
    output_dir = os.path.expanduser(_cfg(
        cfg, "output_dir", f"~/destacking_data/output/stacked3/{sim_prefix}",
    ))
    os.makedirs(output_dir, exist_ok=True)
    wsa.NWORKERS = _cfg(cfg, "n_workers", 32, cast=int)
    n_draws = 1000
    min_n_for_fit = wsa.MIN_N_FOR_FIT

    t0 = time.time()
    print(f"3-way Delta-m stacked test: sim_prefix={sim_prefix!r} "
          f"output_dir={output_dir}", flush=True)

    print("\nStage 0: loading simulated catalogue + bkg arrays", flush=True)
    catalog, bkg_by_id = load_sim_catalog_and_bkgs(sims_dir, sim_prefix)

    print("\nStage 2 (3-way): bin + stack", flush=True)
    stacks, z_bin, ng_bin, dm_bin = build_stacks_3way(catalog, bkg_by_id)
    with open(os.path.join(output_dir, "stacks3.pkl"), "wb") as fh:
        pickle.dump({"stacks": stacks, "z_bin": z_bin, "ng_bin": ng_bin,
                     "dm_bin": dm_bin}, fh)

    print("\nStage 3: NFW + miscentering fit per stack (unmodified)", flush=True)
    fits_out = wsa.build_stack_fits(stacks)
    with open(os.path.join(output_dir, "fits3.pkl"), "wb") as fh:
        pickle.dump(fits_out, fh)

    cluster_arr = wsa.build_stack_cluster_arrays(stacks, catalog, z_bin, ng_bin, dm_bin)

    print("\nStage 5 (3-way): mass-concentration power law per delta_m tercile", flush=True)
    cmr_fits = build_cmr_fits_3way(fits_out, cluster_arr, min_n_for_fit, n_draws)
    with open(os.path.join(output_dir, "cmr_fits3.pkl"), "wb") as fh:
        pickle.dump({"cmr_fits": cmr_fits}, fh)

    print(f"\nTotal wall time: {wsa._fmt_duration(time.time() - t0)}", flush=True)
    print(f"All stages done. Outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
