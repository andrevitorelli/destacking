"""Simulation checkpointing and notebook-compatible data products.

Moved from scripts/gamma_scan.py so simulation scans and future real-data
ingestion scripts share one on-disk format.
"""

import json
import os
import pickle

import numpy as np


# ── Simulation checkpointing ──────────────────────────────────────────────────

def save_simulation_checkpoint(sim_subdir, truth, ols, cluster_obs,
                                data_vectors, cov_matrices):
    """Persist all arrays needed to run bootstrap draws without re-simulating."""
    np.savez_compressed(
        os.path.join(sim_subdir, "simulated_data.npz"),
        data_vectors=data_vectors,
        cov_matrices=cov_matrices,
    )
    np.savez_compressed(
        os.path.join(sim_subdir, "truth_catalog.npz"),
        z=np.asarray(truth["z"]),
        M_halo=np.asarray(truth["M_halo"]),
        concentration=np.asarray(truth["concentration"]),
        delta_m=np.asarray(truth["delta_m"]),
        # true (error-free) magnitude gap, present only with delta_m_reference
        **({"delta_m_true": np.asarray(truth["delta_m_true"])} if "delta_m_true" in truth.colnames else {}),
    )
    co_arrays = {
        "z":       np.asarray(cluster_obs["z"]),
        "Lambda":  np.asarray(cluster_obs["Lambda"]),
        "delta_m": np.asarray(cluster_obs["delta_m"]),
    }
    # Optional: stack membership for the boost correction (see
    # gamma_inference.py's `boosts:` config key). Only present for
    # real-data checkpoints from prepare_real_data.py — simulated
    # checkpoints have no "stack" concept and omit it, and
    # load_simulation_checkpoint's generic Table({k: co[k] for k in
    # co.files}) means callers that don't ask for it never notice either
    # way.
    if "stack_name" in getattr(cluster_obs, "colnames", cluster_obs):
        co_arrays["stack_name"] = np.asarray(cluster_obs["stack_name"])
    np.savez_compressed(os.path.join(sim_subdir, "cluster_obs.npz"), **co_arrays)
    from destacking.dmform import DM_FORM
    with open(os.path.join(sim_subdir, "ols.json"), "w") as f:
        json.dump(dict(ols, dm_form=DM_FORM), f, indent=2)


def load_simulation_checkpoint(sim_subdir):
    """Reload checkpointed simulation data. Returns astropy Tables."""
    from astropy.table import Table

    raw = np.load(os.path.join(sim_subdir, "simulated_data.npz"))
    data_vectors = raw["data_vectors"]
    cov_matrices = raw["cov_matrices"]

    tc    = np.load(os.path.join(sim_subdir, "truth_catalog.npz"))
    truth = Table({k: tc[k] for k in tc.files})

    co          = np.load(os.path.join(sim_subdir, "cluster_obs.npz"))
    cluster_obs = Table({k: co[k] for k in co.files})

    with open(os.path.join(sim_subdir, "ols.json")) as f:
        ols = json.load(f)

    return truth, ols, cluster_obs, data_vectors, cov_matrices


# ── Notebook-compatible per-gamma sim data ────────────────────────────────────

def write_notebook_compat_sim(g_idx, gamma_true, truth, cluster_obs, bgs,
                              out_dir, prefix_root,
                              density=None, shape_noise=None,
                              z_src_range=None,
                              radii_layout='np.logspace(-0.8, 0.6, 8)'):
    """Save {prefix}_real_data.fits + {prefix}_bgs.pkl + {prefix}_metadata.json
    in the layout that wazp_lensing_signals.ipynb and
    wazp_mass_richness_by_delta_m.ipynb expect when SIMS=True and
    SIM_PREFIX='{prefix}', with prefix = f'{prefix_root}{g_idx:03d}'.

    ``density``, ``shape_noise`` and ``z_src_range`` are recorded in the
    metadata file only (they do not affect the data products).
    """
    import datetime as _datetime
    from astropy.table import Table as _Table
    from astropy.io    import fits as _fits  # noqa: F401  (Table.write needs fits)

    prefix  = f'{prefix_root}{g_idx:03d}'
    fits_p  = os.path.join(out_dir, f'{prefix}_real_data.fits')
    bgs_p   = os.path.join(out_dir, f'{prefix}_bgs.pkl')
    meta_p  = os.path.join(out_dir, f'{prefix}_metadata.json')
    os.makedirs(out_dir, exist_ok=True)

    # Cluster catalogue: distribute on a non-overlapping RA/Dec grid; the
    # notebook only reads ID/ZPHOT/NGALS/DELTA_M_I when SIMS=True so the
    # angular positions are purely formal.
    n        = len(truth)
    side     = int(np.ceil(np.sqrt(n)))
    iy, ix   = np.divmod(np.arange(n), side)
    step_deg = 0.6
    ra       = (10.0 + ix * step_deg).astype(float)
    dec      = (-60.0 + iy * step_deg).astype(float)
    cid      = np.arange(n, dtype=np.int64)

    tab = _Table({
        'ID':            cid,
        'RA':            ra,
        'DEC':           dec,
        'NGALS':         np.asarray(cluster_obs['Lambda']).astype('float32'),
        'ZPHOT':         np.asarray(truth['z']).astype('float32'),
        'DELTA_M_I':     np.asarray(truth['delta_m']).astype('float32'),
        'DELTA_M_I_ERR': np.full(n, 0.01, dtype='float32'),
    })
    tab.write(fits_p, overwrite=True)

    bgs_by_id = {int(cid[i]): bg
                 for i, bg in enumerate(bgs) if bg is not None}
    with open(bgs_p, 'wb') as fh:
        pickle.dump(bgs_by_id, fh)

    meta = {
        'gamma':         float(gamma_true),
        'g_idx':         int(g_idx),
        'n_truth':       int(n),
        'n_usable_bgs':  int(len(bgs_by_id)),
        'density':       density,
        'radii_layout':  radii_layout,
        'shape_noise':   shape_noise,
        'z_src_range':   list(z_src_range) if z_src_range is not None else None,
        'generated_at':  _datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(meta_p, 'w') as fh:
        json.dump(meta, fh, indent=2)

    print(f"    notebook-compat sim written: "
          f"{prefix}_real_data.fits ({n:,} clusters), "
          f"{prefix}_bgs.pkl ({len(bgs_by_id):,} bgs)", flush=True)
