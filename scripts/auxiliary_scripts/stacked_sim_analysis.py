#!/usr/bin/env python3
"""
Stacked-analysis comparison run for SIMULATED destacking data, per
cluster_runs.md: "the stacked runs should be ran as done in the notebook,
with the same simulated data used on the joint-fit runs."

This is the simulated-data counterpart of wazp_stacked_analysis.py (the
real-WaZP+DES-Y3 stacked-analysis script), reusing that script's stages
directly rather than reimplementing them -- only Stage 1 (per-cluster bkg
arrays) differs, since simulated bgs are already precomputed by
gamma_inference.py's simulation phase (via destacking.io.write_notebook_compat_sim)
instead of measured from a real metacal catalog.

Data contract (must match a gamma_inference.py YAML config's `sim_prefix`
key exactly, WITHOUT the g_idx suffix -- every single-gamma config has
g_idx=0, so the full prefix on disk is `{sim_prefix}000`):
    ~/destacking_data/sims/{sim_prefix}000_real_data.fits   (cluster catalog:
        ID, RA, DEC, NGALS, ZPHOT, DELTA_M_I, DELTA_M_I_ERR)
    ~/destacking_data/sims/{sim_prefix}000_bgs.pkl          (dict[int ID ->
        ndarray(6, N_gal) float32], same (Sigma_crit, e_t, e_x, W, R, M)
        layout xlensing.data.metacal_cluster_lensing produces -- i.e.
        exactly what build_bkgs() would have returned for real data, so
        every downstream wazp_stacked_analysis.py stage is unmodified.)

Usage:
    python scripts/auxiliary_scripts/stacked_sim_analysis.py <config.yaml>

See scripts/configs/stacked_*.yaml for the config format (sim_prefix,
output_dir, n_workers).
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

# ── Load wazp_stacked_analysis.py as a module (it's a script, not part of
# the destacking package) and monkeypatch its output-path/config globals
# to point at this run's own namespaced directory instead of the real-data
# one. Every stage function reads those globals directly (USE_CACHE,
# NWORKERS, *_PKL), so this is the same "point it elsewhere" mechanism the
# real script itself uses via RUN_TAG, just done from outside since we
# don't want to touch that working, real-data-facing script at all.
_WSA_PATH = os.path.join(os.path.dirname(__file__), "wazp_stacked_analysis.py")
_spec = importlib.util.spec_from_file_location("wazp_stacked_analysis", _WSA_PATH)
wsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsa)


def _cfg(cfg, key, default=None, cast=None):
    v = cfg.get(key, default)
    return cast(v) if (cast is not None and v is not None) else v


def load_sim_catalog_and_bkgs(sims_dir, sim_prefix):
    """Load a gamma_inference.py-produced notebook-compat simulated
    dataset -- the exact (catalog, bkg_by_id) pair build_bkgs() would
    have produced for real data."""
    prefix = f"{sim_prefix}000"  # g_idx=0 for every single-gamma HPC config
    cat_path = os.path.join(sims_dir, f"{prefix}_real_data.fits")
    bgs_path = os.path.join(sims_dir, f"{prefix}_bgs.pkl")
    if not os.path.exists(cat_path) or not os.path.exists(bgs_path):
        raise FileNotFoundError(
            f"Simulated data not found for sim_prefix={sim_prefix!r}: "
            f"expected {cat_path} and {bgs_path}. This script must run "
            f"AFTER the matching gamma_inference.py simulation stage "
            f"(same sim_prefix) has completed."
        )
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


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: python {sys.argv[0]} config.yaml")
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)

    sim_prefix = _cfg(cfg, "sim_prefix")
    if not sim_prefix:
        raise SystemExit("config must set 'sim_prefix' (matching the "
                          "gamma_inference.py config that produced the data)")
    sims_dir   = os.path.expanduser(_cfg(cfg, "sims_dir", "~/destacking_data/sims"))
    output_dir = os.path.expanduser(_cfg(
        cfg, "output_dir",
        f"~/destacking_data/output/stacked/{sim_prefix}",
    ))
    os.makedirs(output_dir, exist_ok=True)

    # Point every wazp_stacked_analysis.py stage at this run's own
    # namespaced output directory -- fixes the notebook's known bug
    # (its c-M output path is hardcoded to the real-data location even
    # under SIMS=True); everything here is consistently namespaced by
    # construction instead.
    wsa.USE_CACHE  = _cfg(cfg, "use_cache", True, cast=bool)
    wsa.NWORKERS   = _cfg(cfg, "n_workers", 32, cast=int)
    wsa.BKG_CACHE  = os.path.join(output_dir, "bkgs.pkl")  # unused (bkgs are precomputed), kept for interface parity
    wsa.STACKS_PKL = os.path.join(output_dir, "stacks.pkl")
    wsa.FITS_PKL   = os.path.join(output_dir, "fits.pkl")
    wsa.ML_PKL     = os.path.join(output_dir, "ml_fits.pkl")
    wsa.CMR_PKL    = os.path.join(output_dir, "cmr_fits.pkl")
    wsa.BOOST_PKL  = os.path.join(output_dir, "boost_fits.pkl")

    t0 = time.time()
    print(f"Stacked-analysis run: sim_prefix={sim_prefix!r} "
          f"output_dir={output_dir}", flush=True)

    print("\nStage 0: loading simulated catalogue + bkg arrays", flush=True)
    catalog, bkg_by_id = load_sim_catalog_and_bkgs(sims_dir, sim_prefix)

    print("\nStage 2: bin + stack", flush=True)
    stacks, z_bin, ng_bin, dm_bin, dm_median_per_cell = wsa.build_stacks(
        catalog, bkg_by_id,
    )

    print("\nStage 2b: fit boost(R) model per stack", flush=True)
    try:
        wsa.fit_boost_models(stacks)
    except Exception as exc:
        # Diagnostic only -- the NFW fit below doesn't consume this (boost
        # is already baked into stacked_signal's returned sigmas); a clean
        # simulation with no injected member-dilution can legitimately
        # have boost(R)=1 everywhere, which is fine to fail to "fit" as a
        # power law. Don't let it block the rest of the pipeline.
        print(f"  WARNING: boost model fit failed/skipped: {exc!r}", flush=True)

    print("\nStage 3: NFW + miscentering fit per stack", flush=True)
    fits_out = wsa.build_stack_fits(stacks)

    cluster_arr = wsa.build_stack_cluster_arrays(stacks, catalog, z_bin, ng_bin, dm_bin)

    print("\nStage 4: mass-richness power law per delta_m group", flush=True)
    ml_fits = wsa.build_ml_fits(fits_out, cluster_arr)

    print("\nStage 5: mass-concentration power law per delta_m group", flush=True)
    cmr_fits = wsa.build_cmr_fits(fits_out, cluster_arr)

    print(f"\nTotal wall time: {wsa._fmt_duration(time.time() - t0)}", flush=True)
    print(f"All stages done. Outputs in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
