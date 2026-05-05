#!/usr/bin/env python3
"""
measure_cluster_lensing.py

Measures per-cluster ΔΣ weak-lensing profiles for DES Y6 WaZP (or Y3 redMaPPer)
clusters using the DES Y3 metacal shape catalog, via xlensing.data functions.

Pipeline
--------
1. Build a spatial source index from the 291 GB metacal catalog (positions + quality
   flags).  Cached to disk after first run (~10 min) and reused on subsequent runs.
2. Build a KD-tree on the ~90M filtered source positions for fast cone searches.
3. For each cluster, query the KD-tree, load shape data from HDF5 (sorted fancy
   indexing for efficiency), estimate per-source photo-z, and call
   xlensing.data.metacal_cluster_lensing() + signal().
4. Save per-cluster data vectors, diagonal covariance matrices, and cluster metadata
   as a single .npz file compatible with destacking.fitting.run_sampler().
5. Plot a weighted-stacked ΔΣ signal as a sanity check.

Photo-z note
------------
The metacal HDF5 catalog contains no photometric redshifts.  This script estimates
a rough per-galaxy photo-z from the i-band flux magnitude (assuming DES zero point
ZP = 30, i.e. mag = 30 − 2.5 log10(flux)):

    z_phot ≈ 0.15 × (mag_i − 17.9) + 0.02 × (mag_r − mag_i)

Calibrated so that the median z_phot ≈ 0.63 matches the known DES Y3 metacal
source n(z).  This is adequate for the background cut z_s > 1.1·z_cl + 0.2 and
for relative Σ_crit weighting within a redshift bin, but introduces scatter in the
absolute Σ_crit normalisation.  Replace estimate_photo_z() with a lookup from a
matched DNF/BPZ catalog for production analysis.
"""

import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import multiprocess as mp
import os, time, warnings
from pathlib import Path
from tqdm.auto import tqdm

import xlensing.cosmo as _xlc
import xlensing.sims as _xlsims
from xlensing.data import metacal_cluster_lensing, signal

# ── Fast D_A patch (comoving-distance spline, ~1000× faster than scipy.quad) ─
_z_tab      = np.linspace(0.0, 4.0, 10_000)
_chi_tab    = _xlc.cosmology.comoving_distance(_z_tab).value
_chi_spline = interp1d(_z_tab, _chi_tab, kind='linear',
                       bounds_error=False, fill_value='extrapolate')

def _DA_fast(z1, z2):
    z2_a = np.atleast_1d(np.asarray(z2, dtype=float))
    res  = (_chi_spline(z2_a) - float(_chi_spline(float(z1)))) / (1.0 + z2_a)
    return float(res[0]) if np.ndim(z2) == 0 else res

_xlc.DA    = _DA_fast
_xlsims.DA = _DA_fast

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR     = '/mnt/c/Users/andre/DATA'
METACAL_FILE = f'{DATA_DIR}/DESY3_metacal_v03-004.h5'

# Choose cluster catalog: 'wazp' or 'redmapper'
CATALOG_MODE = 'wazp'
WAZP_FILE    = f'{DATA_DIR}/y6a2_dnf_wazp_v5.0.12.6801_clusters.fits'
RM_FILE      = f'{DATA_DIR}/y3_redmapper_v6.4.22+2_release.h5'

OUT_DIR      = Path('cluster_lensing_output')
INDEX_FILE   = Path('metacal_source_index.npz')   # cached source positions

# ── Radial binning ────────────────────────────────────────────────────────────
RADII      = np.logspace(-0.3, 0.6, 8)            # bin centres [Mpc]
_log_r     = np.log10(RADII)
_dlog      = np.diff(_log_r)[0]
_edges     = np.logspace(_log_r[0] - _dlog/2, _log_r[-1] + _dlog/2, len(RADII) + 1)
BIN_LIMITS = np.column_stack([_edges[:-1], _edges[1:]])
N_BINS     = len(RADII)
R_MAX_MPC  = float(_edges[-1])                    # outer edge of last bin [Mpc]
NOISE_FLOOR_COV = 1e4                             # M_sun²/pc⁴ for empty bins

# ── Cluster selection ─────────────────────────────────────────────────────────
Z_MIN        = 0.10
Z_MAX        = 0.65
RICHNESS_MIN = 15           # NGALS for WaZP; lambda for redMaPPer
N_CLUSTERS_MAX = 2000       # None to process all passing clusters

# ── Metacal quality cuts ──────────────────────────────────────────────────────
SNR_MIN        = 10.0
SNR_MAX        = 1000.0
SIZE_RATIO_MIN = 0.5
METACAL_CHUNK  = 10_000_000  # objects per chunk when scanning catalog

# ── Parallelism ───────────────────────────────────────────────────────────────
N_WORKERS = 12

# ═══════════════════════════════════════════════════════════════════════════════
# Photo-z estimator
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_photo_z(flux_r, flux_i):
    """
    Rough per-galaxy photo-z from DES i-band magnitude (ZP = 30).

    Calibrated to reproduce the DES Y3 metacal source median z ≈ 0.63.
    PLACEHOLDER — replace with matched DNF/BPZ catalog for production.
    """
    eps   = 1e-6
    mag_i = np.where(flux_i > eps, 30.0 - 2.5 * np.log10(np.maximum(flux_i, eps)), 99.0)
    mag_r = np.where(flux_r > eps, 30.0 - 2.5 * np.log10(np.maximum(flux_r, eps)), 99.0)
    color_ri = np.clip(mag_r - mag_i, -3.0, 4.0)
    z_phot   = 0.15 * (mag_i - 17.9) + 0.02 * color_ri
    return np.clip(z_phot, 0.05, 3.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Source position index
# ═══════════════════════════════════════════════════════════════════════════════

def build_or_load_source_index():
    """
    Scan the full metacal catalog once, apply quality cuts, save (index, RA, DEC)
    arrays.  Subsequent calls just load the cached .npz.

    Returns
    -------
    src_idx : int64 array (M,)   original HDF5 row indices of passing sources
    src_ra  : float64 array (M,) RA  [deg]
    src_dec : float64 array (M,) Dec [deg]
    """
    if INDEX_FILE.exists():
        print(f'Loading cached source index from {INDEX_FILE} …')
        d = np.load(INDEX_FILE)
        print(f'  {len(d["indices"]):,} sources loaded.')
        return d['indices'], d['ra'], d['dec']

    print('Building source position index (one-time scan of 399 M metacal objects) …')
    t0 = time.time()

    idx_chunks = []
    ra_chunks  = []
    dec_chunks = []

    with h5py.File(METACAL_FILE, 'r') as f:
        g = f['catalog/unsheared']
        N = int(g['ra'].shape[0])
        for start in tqdm(range(0, N, METACAL_CHUNK), desc='Scanning metacal'):
            s     = slice(start, min(start + METACAL_CHUNK, N))
            flags = g['flags'][s]
            snr   = g['snr'][s]
            sr    = g['size_ratio'][s]
            T     = g['T'][s]
            cut   = ((flags == 0) &
                     (snr  >  SNR_MIN) & (snr  < SNR_MAX) &
                     (sr   >  SIZE_RATIO_MIN) &
                     (T    >  0.0))
            local = np.where(cut)[0]
            idx_chunks.append((local + start).astype(np.int64))
            ra_chunks.append(g['ra'][s][cut])
            dec_chunks.append(g['dec'][s][cut])

    src_idx = np.concatenate(idx_chunks)
    src_ra  = np.concatenate(ra_chunks)
    src_dec = np.concatenate(dec_chunks)

    np.savez(INDEX_FILE, indices=src_idx, ra=src_ra, dec=src_dec)
    print(f'Index saved to {INDEX_FILE}  '
          f'({len(src_idx):,} sources, {time.time()-t0:.0f} s)')
    return src_idx, src_ra, src_dec

# ═══════════════════════════════════════════════════════════════════════════════
# Cluster catalog loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_wazp_clusters():
    """Return astropy Table with columns z, Lambda, delta_m, ra, dec, name."""
    with fits.open(WAZP_FILE, memmap=True) as h:
        cl = h[1].data
    sel = (
        (cl['ZPHOT']  >= Z_MIN) &
        (cl['ZPHOT']  <= Z_MAX) &
        (cl['NGALS']  >= RICHNESS_MIN) &
        cl['IN_COSMO']
    )
    cl = cl[sel]
    if N_CLUSTERS_MAX is not None:
        order = np.argsort(cl['NGALS'])[::-1]
        cl    = cl[order[:N_CLUSTERS_MAX]]
    tbl = Table({
        'z':       cl['ZPHOT'].astype(float),
        'Lambda':  cl['NGALS'].astype(float),
        'delta_m': np.full(len(cl), np.nan),   # placeholder — compute from member catalog
        'ra':      cl['RA'].astype(float),
        'dec':     cl['DEC'].astype(float),
        'name':    cl['NAME'],
        'snr':     cl['SNR'].astype(float),
    })
    return tbl


def load_redmapper_clusters():
    """Return astropy Table with columns z, Lambda, delta_m, ra, dec, name."""
    with h5py.File(RM_FILE, 'r') as f:
        gc     = f['catalog/cluster']
        rm_z   = gc['z_lambda'][:]
        rm_lam = gc['lambda_chisq'][:]
        rm_ra  = gc['ra'][:]
        rm_dec = gc['dec'][:]
        rm_m   = gc['m200m'][:]
        lgt_sel = f['index/redmapper/lgt20/select'][:]

    sel = ((rm_z[lgt_sel]  >= Z_MIN) &
           (rm_z[lgt_sel]  <= Z_MAX) &
           (rm_lam[lgt_sel] >= RICHNESS_MIN))
    idx = lgt_sel[sel]
    if N_CLUSTERS_MAX is not None:
        order = np.argsort(rm_lam[idx])[::-1]
        idx   = idx[order[:N_CLUSTERS_MAX]]

    tbl = Table({
        'z':       rm_z[idx].astype(float),
        'Lambda':  rm_lam[idx].astype(float),
        'delta_m': np.full(len(idx), np.nan),  # placeholder
        'ra':      rm_ra[idx].astype(float),
        'dec':     rm_dec[idx].astype(float),
        'name':    [f'RM_{i}' for i in idx],
        'm200m':   rm_m[idx].astype(float),
    })
    return tbl

# ═══════════════════════════════════════════════════════════════════════════════
# Worker (per-cluster lensing measurement)
# ═══════════════════════════════════════════════════════════════════════════════

# Worker-level globals (set via initializer after fork)
_h5_group    = None
_src_idx_arr = None


def _worker_init(src_idx_shared):
    global _h5_group, _src_idx_arr
    _src_idx_arr = src_idx_shared
    _h5f         = h5py.File(METACAL_FILE, 'r')
    _h5_group    = _h5f['catalog/unsheared']


def _process_cluster(args):
    """
    Measure ΔΣ for one cluster.

    args = (ci, ra_deg, dec_deg, z_cl, local_idx)
      local_idx : indices into _src_idx_arr for sources inside the search cone.

    Returns (ci, dv, cov_d)  or  (ci, None, None) if too few sources.
    """
    ci, ra_deg, dec_deg, z_cl, local_idx = args

    if len(local_idx) < 20:
        return ci, None, None

    # Map local → original HDF5 row indices; sort for cache-friendly reads
    hdf5_idx = np.sort(_src_idx_arr[local_idx])

    g    = _h5_group
    ra   = g['ra'][hdf5_idx]
    dec  = g['dec'][hdf5_idx]
    e1   = g['e_1'][hdf5_idx]
    e2   = g['e_2'][hdf5_idx]
    w    = g['weight'][hdf5_idx]
    R11  = g['R11'][hdf5_idx]
    R12  = g['R12'][hdf5_idx]
    R21  = g['R21'][hdf5_idx]
    R22  = g['R22'][hdf5_idx]
    fr   = g['flux_r'][hdf5_idx]
    fi   = g['flux_i'][hdf5_idx]

    # Keep only finite, physically valid entries
    ok = (np.isfinite(e1) & np.isfinite(e2) &
          np.isfinite(R11) & np.isfinite(R22) &
          (fi > 0) & (fr > 0) & (w > 0))
    if ok.sum() < 20:
        return ci, None, None

    ra  = ra[ok];   dec = dec[ok]
    e1  = e1[ok];   e2  = e2[ok]
    w   = w[ok]
    R11 = R11[ok];  R12 = R12[ok]
    R21 = R21[ok];  R22 = R22[ok]
    fr  = fr[ok];   fi  = fi[ok]

    z_src = estimate_photo_z(fr, fi)

    # Pack source array in the 10-column format expected by metacal_cluster_lensing:
    # [RA_rad, DEC_rad, z_src, e1, e2, W, R11, R12, R21, R22]
    sources = np.array([
        np.deg2rad(ra), np.deg2rad(dec), z_src,
        e1, e2, w, R11, R12, R21, R22,
    ])

    try:
        bg = metacal_cluster_lensing(
            (np.deg2rad(ra_deg), np.deg2rad(dec_deg), z_cl),
            sources,
            R_MAX_MPC,
        )
    except Exception:
        return ci, None, None

    if bg['Count'] < 5:
        return ci, None, None

    stake = np.array([
        bg['Critical Density'],
        bg['Tangential Shear'],
        bg['Cross Shear'],
        bg['Weights'],
        bg['Radial Distance'],
        bg['Mult. Bias'],
    ])

    dv, _ = signal(stake, BIN_LIMITS)
    dv     = np.nan_to_num(dv, nan=0.0)

    # Per-bin diagonal shape-noise covariance
    R      = bg['Radial Distance']
    sig_c  = bg['Critical Density']
    in_bin = (R[:, None] > BIN_LIMITS[:, 0]) & (R[:, None] < BIN_LIMITS[:, 1])
    cov_d  = np.full(N_BINS, NOISE_FLOOR_COV)
    for b in range(N_BINS):
        m   = in_bin[:, b]
        n_b = int(m.sum())
        if n_b >= 5:
            # σ(ΔΣ)² ≈ (σ_e · <Σ_crit>)² / n_b   with σ_e = 0.25 (per component)
            cov_d[b] = (0.25 * float(sig_c[m].mean()) / np.sqrt(n_b)) ** 2

    return ci, dv, cov_d

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Build / load source position index ────────────────────────────────────
    src_idx, src_ra, src_dec = build_or_load_source_index()

    # 2. Build spatial KD-tree ─────────────────────────────────────────────────
    # Use (ra·cos(dec), dec) in degrees: Euclidean distance ≈ great-circle distance
    # for separations ≪ 1 radian.  Max cone radius here is ≲ 0.5°, so error < 0.1%.
    print('Building KD-tree …')
    t0 = time.time()
    dec_rad_src = np.deg2rad(src_dec)
    X_src       = np.column_stack([src_ra * np.cos(dec_rad_src), src_dec])
    tree        = cKDTree(X_src)
    print(f'KD-tree ready in {time.time()-t0:.1f} s')

    # 3. Load cluster catalog ──────────────────────────────────────────────────
    if CATALOG_MODE == 'wazp':
        clusters = load_wazp_clusters()
        print(f'WaZP clusters selected: {len(clusters):,}'
              f'  (z=[{Z_MIN},{Z_MAX}], NGALS≥{RICHNESS_MIN}, IN_COSMO)')
    else:
        clusters = load_redmapper_clusters()
        print(f'redMaPPer clusters selected: {len(clusters):,}'
              f'  (z=[{Z_MIN},{Z_MAX}], λ≥{RICHNESS_MIN})')

    n_cl = len(clusters)

    # 4. Pre-query spatial index for all clusters ──────────────────────────────
    print('Querying spatial index for all clusters …')
    tasks = []
    n_src_per_cl = []
    for ci in tqdm(range(n_cl), desc='Spatial query'):
        z_cl   = float(clusters['z'][ci])
        ra_cl  = float(clusters['ra'][ci])
        dec_cl = float(clusters['dec'][ci])
        # Angular search radius in degrees (flat-sky)
        DA_cl      = _DA_fast(0, z_cl)
        r_deg      = np.rad2deg(R_MAX_MPC / DA_cl)
        query_pt   = np.array([ra_cl * np.cos(np.deg2rad(dec_cl)), dec_cl])
        local_idx  = np.array(tree.query_ball_point(query_pt, r=r_deg), dtype=np.int32)
        tasks.append((ci, ra_cl, dec_cl, z_cl, local_idx))
        n_src_per_cl.append(len(local_idx))

    n_src_arr = np.array(n_src_per_cl)
    print(f'Sources per cluster:  median={np.median(n_src_arr):.0f}'
          f'  p5={np.percentile(n_src_arr,5):.0f}'
          f'  p95={np.percentile(n_src_arr,95):.0f}')

    # 5. Measure lensing signals (parallel) ────────────────────────────────────
    data_vectors = np.zeros((n_cl, N_BINS))
    cov_matrices = np.full((n_cl, N_BINS, N_BINS), 0.0)
    for i in range(n_cl):
        np.fill_diagonal(cov_matrices[i], NOISE_FLOOR_COV)

    t0 = time.time()
    with mp.Pool(
        N_WORKERS,
        initializer=_worker_init,
        initargs=(src_idx,),
    ) as pool:
        for ci, dv, cov_d in tqdm(
            pool.imap(_process_cluster, tasks, chunksize=10),
            total=n_cl, desc='Measuring ΔΣ',
        ):
            if dv is not None:
                data_vectors[ci] = dv
                np.fill_diagonal(cov_matrices[ci], cov_d)

    n_good = int((data_vectors.sum(axis=1) != 0).sum())
    print(f'\nDone in {(time.time()-t0)/60:.1f} min — '
          f'{n_good}/{n_cl} clusters have a signal.')

    # 6. Save results ──────────────────────────────────────────────────────────
    outfile = OUT_DIR / 'data_vectors.npz'
    np.savez(
        outfile,
        data_vectors  = data_vectors,
        cov_matrices  = cov_matrices,
        radii         = RADII,
        bin_limits    = BIN_LIMITS,
        cluster_z     = np.array(clusters['z']),
        cluster_ngals = np.array(clusters['Lambda']),
        cluster_ra    = np.array(clusters['ra']),
        cluster_dec   = np.array(clusters['dec']),
        cluster_delta_m = np.array(clusters['delta_m']),
    )
    print(f'Saved {outfile}')

    # Also write an astropy-Table FITS version compatible with run_sampler()
    fits_tbl = Table({
        'z':       clusters['z'],
        'Lambda':  clusters['Lambda'],
        'delta_m': clusters['delta_m'],
    })
    fits_out = OUT_DIR / 'cluster_obs.fits'
    fits_tbl.write(fits_out, overwrite=True)
    print(f'Saved {fits_out}')

    # 7. Stacked signal plot ────────────────────────────────────────────────────
    _plot_stacked_signal(clusters, data_vectors, cov_matrices, n_good)


def _plot_stacked_signal(clusters, data_vectors, cov_matrices, n_good):
    import matplotlib.pyplot as plt

    good = data_vectors.sum(axis=1) != 0
    dv_g = data_vectors[good]
    cv_g = cov_matrices[good]

    # Inverse-variance weighted stack
    inv_var = 1.0 / np.diagonal(cv_g, axis1=1, axis2=2)   # (N, M)
    sig_stack = np.sum(inv_var * dv_g, axis=0) / np.sum(inv_var, axis=0)
    err_stack = 1.0 / np.sqrt(np.sum(inv_var, axis=0))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: stacked ΔΣ profile
    ax = axes[0]
    ax.errorbar(RADII, sig_stack, yerr=err_stack,
                fmt='o', ms=6, capsize=4, color='steelblue', lw=1.5)
    ax.set_xscale('log')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('R  [Mpc]', fontsize=12)
    ax.set_ylabel(r'$\Delta\Sigma$  [M$_\odot$/pc$^2$]', fontsize=12)
    ax.set_title(f'Stacked ΔΣ  ({n_good} clusters)', fontsize=12)

    # Panel 2: redshift distribution of processed clusters
    ax = axes[1]
    ax.hist(clusters['z'][good], bins=30, color='steelblue',
            edgecolor='none', alpha=0.8)
    ax.set_xlabel('Cluster redshift $z$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Redshift distribution', fontsize=12)

    # Panel 3: richness vs redshift scatter
    ax = axes[2]
    sc = ax.scatter(clusters['z'][good], clusters['Lambda'][good],
                    s=4, alpha=0.3, c='steelblue', rasterized=True)
    ax.set_xlabel('Cluster redshift $z$', fontsize=12)
    ax.set_ylabel(r'Richness ($N_{\rm gals}$ or $\lambda$)', fontsize=12)
    ax.set_title('Processed cluster sample', fontsize=12)

    plt.suptitle('DES Cluster Weak-Lensing Measurements', fontsize=13)
    plt.tight_layout()
    plot_file = OUT_DIR / 'stacked_signal.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved {plot_file}')


if __name__ == '__main__':
    main()
