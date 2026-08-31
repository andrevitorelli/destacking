#!/usr/bin/env python3
"""Generate a simulated WaZP-like cluster catalogue + per-cluster
backgrounds in the file layout that wazp_lensing_signals.ipynb and
wazp_mass_richness_by_delta_m.ipynb can load directly when their
SIMS=True toggle is enabled.

Cluster generation mirrors the WaZP selection (Λ>25, Δm>0,
0.10≤z<0.60); the per-cluster shear injection is gamma_scan-style
(spherical NFW; Z_SRC_RANGE=(0, 2.5), ρ=7.5 gal/arcmin², σ_e=0.27,
5 Mpc cone).  Optionally a non-zero Δm-c gamma correlation can be
imposed via GAMMA=<value> in [-1, 1].

Outputs (written to notebooks/data/, paths configurable via env):
    sim_real_data.fits        cluster catalogue: ID, RA, DEC, ZPHOT,
                              NGALS, DELTA_M_I, DELTA_M_I_ERR
    sim_bgs.pkl               dict: ID -> bg (6 × N_gal) ndarray
    sim_metadata.json         gamma value, seed, run timestamp

Environment variables (all optional):
    N_TRUTH      target number of selected clusters (default 15000)
    GAMMA        Δm-c gamma value in [-1, 1] (default 0.0)
    SEED         random seed (default 2026)
    DENSITY      gal/arcmin² (default 7.5, matches WaZP-filtered density)
    NBOOT        bootstrap iterations for the per-cluster reduction
                 (default 50; only used if we run single_cluster here)
    OUT_PREFIX   filename prefix (default 'sim'; suffix '_real_data.fits'
                 etc. are appended)
"""

import os
import json
import time
import pickle
import datetime
import numpy as np
import multiprocess as mp
from astropy.table import Table, vstack
from astropy.io    import fits

import xlensing.data
import xlensing.sims as xlsims
from xlensing.cosmo import DA

from destacking.simulation import generate_truth_catalog, make_test_data
from destacking.relations  import c_DuttonMaccio


# ── parameters ──────────────────────────────────────────────────────────────
N_TRUTH    = int(os.environ.get('N_TRUTH', 15000))
GAMMA      = float(os.environ.get('GAMMA', 0.0))
SEED       = int(os.environ.get('SEED', 2026))
DENSITY    = float(os.environ.get('DENSITY', 7.5))
NBOOT      = int(os.environ.get('NBOOT', 50))
OUT_PREFIX = os.environ.get('OUT_PREFIX', 'sim')

# Layout matches wazp_lensing_signals
RADIUS_MPC  = 5.0
SHAPE_NOISE = 0.27
Z_SRC_RANGE = (0.0, 2.5)
DELTA_M_REF = 0.1   # gamma_scan's reference Δm for the Δm-c modulation
NWORKERS    = max(1, mp.cpu_count() - 1)

OUT_DIR  = '/home/andre/github/destacking/notebooks/data'
OUT_FITS = os.path.join(OUT_DIR, f'{OUT_PREFIX}_real_data.fits')
OUT_BGS  = os.path.join(OUT_DIR, f'{OUT_PREFIX}_bgs.pkl')
OUT_META = os.path.join(OUT_DIR, f'{OUT_PREFIX}_metadata.json')

os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Truth catalogue with the WaZP selection ──────────────────────────────
def build_truth():
    print(f'Generating truth catalogue for GAMMA={GAMMA:.3f}, '
          f'N_TRUTH={N_TRUTH:,}, SEED={SEED}', flush=True)
    selected = []
    n_sel    = 0
    n_drawn  = 0
    batch    = 0
    radii_tmp = np.logspace(-0.8, 0.6, 8)
    t0       = time.time()
    while n_sel < N_TRUTH:
        batch += 1
        truth = generate_truth_catalog(
            N=max(int(N_TRUTH / 0.35), 5000),
            z_range=(0.10, 0.60),
            logM_range=(14.0, 15.0),
            delta_m_gamma=0.0,    # Δm-c is applied after the draw, post-hoc
            c_pivot_0=1.0,
            scatter_logc=0.1,
            seed=SEED + batch,
        )
        cluster_obs, _, _ = make_test_data(
            truth, radii_tmp, poisson_richness=True, seed=SEED + batch,
        )
        truth['Lambda'] = cluster_obs['Lambda']
        keep = ((truth['Lambda']  > 25.0)
              & (truth['delta_m'] > 0.0)
              & np.isfinite(truth['delta_m'])
              & (truth['z']     < 0.60)
              & (truth['z']     >= 0.10))
        truth_sel = truth[keep]
        selected.append(truth_sel)
        n_sel  += len(truth_sel)
        n_drawn += len(truth)
        print(f'  batch {batch}: {len(truth_sel):,} pass '
              f'(running total {n_sel:,}/{N_TRUTH:,}, elapsed {time.time()-t0:.1f}s)',
              flush=True)
    truth_sel = vstack(selected)[:N_TRUTH]

    # Apply the gamma Δm-c modulation: c -> c * (Δm/Δm_ref)^gamma
    dm = np.asarray(truth_sel['delta_m'])
    log_dm = np.log10(np.maximum(dm, 1e-6) / DELTA_M_REF)
    truth_sel['concentration'] = (np.asarray(truth_sel['concentration'])
                                  * 10.0 ** (GAMMA * log_dm))
    print(f'Truth done: {len(truth_sel):,} clusters '
          f'(efficiency {len(truth_sel)/n_drawn*100:.1f}%)', flush=True)
    return truth_sel


# ── 2. Per-cluster shear injection + bg reduction ───────────────────────────
_TRUTH_ARR = None   # numpy view shared across workers


def _init_DA():
    """Build a comoving-distance interpolator inside each worker for fast DA."""
    from scipy.interpolate import interp1d
    import xlensing.cosmo as _xlc
    z_grid  = np.linspace(0.005, 1.5, 600)
    chi     = _xlc.cosmology.comoving_distance(z_grid).value
    chi_of  = interp1d(z_grid, chi, kind='linear',
                       bounds_error=False, fill_value='extrapolate')

    def DA_fast(z1, z2):
        z2_a = np.atleast_1d(np.asarray(z2, dtype=float))
        res = (chi_of(z2_a) - float(chi_of(float(z1)))) / (1.0 + z2_a)
        return float(res[0]) if np.ndim(z2) == 0 else res
    return DA_fast


def per_cluster_worker(args):
    i, z_i, M_i, c_i, seed = args
    DA_fast = _init_DA()
    try:
        DA_lens = float(DA_fast(0, z_i))
        # WIDTH_RAD: half-width of the square catalogue containing the cone
        width_rad = (RADIUS_MPC + 0.3) / DA_lens

        cat = xlsims.make_simple_random_cat(
            density=DENSITY, width_rad=width_rad,
            zrange=Z_SRC_RANGE, shape_noise=SHAPE_NOISE,
            seed=int(seed), poisson_density=True,
        ).T
        if len(cat) == 0:
            return i, None
        cat = xlsims.apply_NFW_shear_region(
            np.array([0.0, 0.0, z_i, M_i, c_i]), cat,
        )
        sources_mc = cat[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]].T
        res = xlensing.data.metacal_cluster_lensing(
            (0.0, 0.0, z_i), sources_mc, radius=RADIUS_MPC,
        )
        if res['Count'] == 0:
            return i, None
        bg = np.array([
            np.asarray(res['Critical Density'], dtype='float32'),
            np.asarray(res['Tangential Shear'], dtype='float32'),
            np.asarray(res['Cross Shear'],      dtype='float32'),
            np.asarray(res['Weights'],          dtype='float32'),
            np.asarray(res['Radial Distance'],  dtype='float32'),
            np.asarray(res['Mult. Bias'],       dtype='float32'),
        ])
        return i, bg
    except Exception as exc:
        return i, repr(exc)


def simulate_per_cluster(truth):
    n = len(truth)
    args = [
        (i,
         float(truth['z'][i]),
         float(truth['M_halo'][i]),
         float(truth['concentration'][i]),
         SEED * 7 + i)
        for i in range(n)
    ]
    print(f'Running per-cluster shear sim on {n:,} clusters '
          f'with {NWORKERS} workers...', flush=True)
    bgs = [None] * n
    n_good = 0
    t0 = time.time()
    with mp.Pool(NWORKERS) as pool:
        for k, (i, bg) in enumerate(pool.imap_unordered(
                per_cluster_worker, args, chunksize=8)):
            if isinstance(bg, np.ndarray):
                bgs[i]  = bg
                n_good += 1
            if (k + 1) % 500 == 0 or (k + 1) == n:
                dt = time.time() - t0
                print(f'  {k+1:>6,}/{n:,}  good={n_good:>6,}  '
                      f'elapsed {dt/60:5.1f} min  '
                      f'eta {dt*(n-k-1)/(k+1)/60:5.1f} min', flush=True)
    print(f'Per-cluster sim done in {(time.time()-t0)/60:.1f} min, '
          f'{n_good:,} usable bgs', flush=True)
    return bgs


# ── 3. Write the FITS catalogue and the bgs pickle ──────────────────────────
def write_outputs(truth, bgs):
    # Distribute clusters on a non-overlapping RA/Dec grid (purely formal —
    # the notebooks query positions but the actual sources are pre-computed).
    n = len(truth)
    side = int(np.ceil(np.sqrt(n)))
    # ~30 arcmin spacing in RA, Dec — comfortably larger than 2 × 5 Mpc cone
    grid_step_deg = 0.6
    iy, ix = np.divmod(np.arange(n), side)
    ra  = (10.0 + ix * grid_step_deg).astype(float)
    dec = (-60.0 + iy * grid_step_deg).astype(float)

    cluster_id = np.arange(n, dtype=np.int64)
    tab = Table({
        'ID':           cluster_id,
        'RA':           ra,
        'DEC':          dec,
        'NGALS':        np.asarray(truth['Lambda']).astype('float32'),
        'ZPHOT':        np.asarray(truth['z']).astype('float32'),
        'DELTA_M_I':    np.asarray(truth['delta_m']).astype('float32'),
        'DELTA_M_I_ERR': np.full(n, 0.01, dtype='float32'),
    })
    tab.write(OUT_FITS, overwrite=True)
    print(f'Wrote cluster catalogue: {OUT_FITS}', flush=True)

    bgs_by_id = {int(cluster_id[i]): bg
                 for i, bg in enumerate(bgs) if bg is not None}
    with open(OUT_BGS, 'wb') as fh:
        pickle.dump(bgs_by_id, fh)
    print(f'Wrote {len(bgs_by_id):,} per-cluster bgs: {OUT_BGS}', flush=True)

    meta = {
        'gamma':         GAMMA,
        'seed':          SEED,
        'n_truth':       int(N_TRUTH),
        'n_usable_bgs':  int(len(bgs_by_id)),
        'density':       DENSITY,
        'radius_mpc':    RADIUS_MPC,
        'shape_noise':   SHAPE_NOISE,
        'z_src_range':   list(Z_SRC_RANGE),
        'generated_at':  datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(OUT_META, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'Wrote metadata: {OUT_META}', flush=True)


# ── driver ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    truth = build_truth()
    bgs   = simulate_per_cluster(truth)
    write_outputs(truth, bgs)
    print(f'\nAll done in {(time.time()-t0)/60:.1f} min')
