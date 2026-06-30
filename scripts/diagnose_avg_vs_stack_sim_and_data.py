#!/usr/bin/env python3
"""
Compare the unweighted average of per-cluster ΔΣ profiles to the stacked
ΔΣ profile, both in a gamma_scan-style simulation and in the WaZP real
data.  No covariance, no inverse-variance weighting, no fitting: just a
plain mean of per-cluster profiles vs the galaxy-level stack.

Simulation regime (matching gamma_scan.py)
------------------------------------------
  DENSITY     = 30 gal/arcmin^2
  WIDTH_RAD   = 0.006 rad     (cone radius ≈ 6.6 Mpc at z_lens=0.4)
  SHAPE_NOISE = 0.27          (DES Y3-like, σ_e per component)
  Z_LENS      = 0.4           (single cluster mass/redshift)
  M_TRUE      = 2e14 Msun, C_TRUE = 5
  N_CLUSTERS  = configurable (default 200)
  RADII       = np.logspace(-0.8, 0.6, 8)  (wazp_lensing_signals layout)

Real-data regime (matching wazp_lensing_signals.ipynb)
------------------------------------------------------
  reads data/wazp_lensing_signals.pkl  (per-cluster ΔΣ from single_cluster)
  reads data/wazp_mr_dm_stacks.pkl     (stacked ΔΣ from stacked_signal)
  groups by the (z, NGALS, Δm) stack definition of
  wazp_mass_richness_by_delta_m.ipynb

Output:
  diagnose_avg_vs_stack_sim_and_data.png
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import xlensing.data
import xlensing.model
import xlensing.sims as xlsims
from xlensing.cosmo import DA

NCL_SIM     = int(sys.argv[1]) if len(sys.argv) > 1 else 30

# ── simulation parameters (mirror gamma_scan.py constants where applicable) ──
DENSITY     = 7.5
WIDTH_RAD   = 0.006
SHAPE_NOISE = 0.27
Z_SRC_RANGE = (0.0, 2.5)
Z_LENS      = 0.4
M_TRUE      = 2.0e14
C_TRUE      = 5.0
NBOOT       = 50    # matches gamma_scan.py's N_BOOTSTRAP

# Wazp_lensing_signals radial layout
radii = np.logspace(-0.8, 0.6, 8)
Nbin  = len(radii)
edges = np.logspace(
    np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    Nbin + 1,
)
BINS = np.array([[edges[i], edges[i + 1]] for i in range(Nbin)])

# Wazp_mass_richness_by_delta_m radial layout (for the real-data stacked side)
mr_radii = np.logspace(-0.6, 0.6, 8)

OUT_PNG = 'diagnose_avg_vs_stack_sim_and_data.png'


# ═════════════════════════════════════════════════════════════════════════════
# Simulation arm
# ═════════════════════════════════════════════════════════════════════════════

def run_sim(n_clusters):
    """Run a gamma_scan-style sim of n_clusters spherical-NFW clusters at fixed
    (M, c, z).  Returns:
        per_cluster_sigmas  (n_used, Nbin)  -- boost-corrected per-cluster ΔΣ
        avg_uniform         (Nbin,)         -- unweighted mean
        sig_stack           (Nbin,)         -- stacked_signal on the same bgs
        ds_true             (Nbin,)         -- analytical NFW truth
    """
    DA_lens = float(DA(0, Z_LENS))

    truth = xlensing.model.NFW_Delta_Sigma(
        M200m=np.array([M_TRUE]), C200m=np.array([C_TRUE]),
        Z=np.array([Z_LENS]),
        FMISS=np.array([0.0]), SIGMA_OFF=np.array([0.1]),
        BCG_B_MASS=np.array([0.0]),
        radii=radii,
    )
    ds_true = np.asarray(truth['Signal'][0])

    per_cluster_sigmas = []
    all_bgs            = []
    for ci in range(n_clusters):
        # Source catalogue uniform in the WIDTH_RAD square, with shape noise.
        # make_simple_random_cat returns shape (10, N) with simplecat column
        # order (RA, DEC, Z, E1, E2, R11, R12, R21, R22, W); apply_NFW_shear
        # and downstream code expect (N, 10) row-major.
        cat = xlsims.make_simple_random_cat(
            density=DENSITY, width_rad=WIDTH_RAD,
            zrange=Z_SRC_RANGE, shape_noise=SHAPE_NOISE,
            seed=10_000 + ci,
        ).T   # (N, 10)
        # Apply spherical NFW shear at the cluster centre (0, 0, z, M, c).
        cluster = np.array([0.0, 0.0, Z_LENS, M_TRUE, C_TRUE])
        cat     = xlsims.apply_NFW_shear_region(cluster, cat)   # mutates cols 3,4
        # Reorder columns to metacal_cluster_lensing's expected order
        # (RA, DEC, Z, E1, E2, W, R11, R12, R21, R22) -- W moves from col 9 to col 5.
        sources_mc = cat[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]].T   # (10, N) tuple of arrays
        res = xlensing.data.metacal_cluster_lensing(
            (0.0, 0.0, Z_LENS), sources_mc, radius=WIDTH_RAD * DA_lens,
        )
        if res['Count'] == 0:
            continue
        bg = np.array([
            np.asarray(res['Critical Density']),
            np.asarray(res['Tangential Shear']),
            np.asarray(res['Cross Shear']),
            np.asarray(res['Weights']),
            np.asarray(res['Radial Distance']),
            np.asarray(res['Mult. Bias']),
        ])
        all_bgs.append(bg)
        sig, *_ = xlensing.data.single_cluster([bg], BINS, Nboot=NBOOT)
        per_cluster_sigmas.append(np.asarray(sig))

    per_cluster_sigmas = np.asarray(per_cluster_sigmas)   # (N, Nbin)
    # Unweighted mean across clusters, masking NaN per-bin
    avg_uniform = np.nanmean(per_cluster_sigmas, axis=0)

    sig_stack, *_ = xlensing.data.stacked_signal(all_bgs, BINS, Nboot=NBOOT)
    sig_stack = np.asarray(sig_stack)

    return per_cluster_sigmas, avg_uniform, sig_stack, ds_true


# ═════════════════════════════════════════════════════════════════════════════
# Real-data arm
# ═════════════════════════════════════════════════════════════════════════════

def run_real(stack_name='all'):
    """Average per-cluster sigmas from wazp_lensing_signals.pkl against
    stacked sigmas from wazp_mr_dm_stacks.pkl.  Returns dict with the
    radii, avg, stacked, residual percent."""
    pkl_ind  = '/home/andre/github/destacking/notebooks/data/wazp_lensing_signals.pkl'
    pkl_stk  = '/home/andre/github/destacking/notebooks/data/wazp_mr_dm_stacks.pkl'
    wazp_fit = '/home/andre/github/destacking/notebooks/data/wazp_real_data.fits'
    for p in (pkl_ind, pkl_stk, wazp_fit):
        if not os.path.exists(p):
            print(f'missing {p}; skipping real-data arm')
            return None

    with open(pkl_ind, 'rb') as fh:
        ind = pickle.load(fh)
    with open(pkl_stk, 'rb') as fh:
        stk = pickle.load(fh)
    wazp_all = Table.read(wazp_fit)
    # Same Δm > 0 cut as wazp_mass_richness_by_delta_m
    keep = np.isfinite(wazp_all['DELTA_M_I']) & (wazp_all['DELTA_M_I'] > 0)
    wazp = wazp_all[keep]

    radii_ind = np.asarray(ind['radii'])
    z_edges   = stk['z_edges']
    ng_edges  = stk['ng_edges']
    dm_med    = stk['dm_median_per_cell']
    stacks    = stk['stacks']

    def _bin_idx(values, edges):
        v   = np.asarray(values, dtype=float)
        idx = np.digitize(v, edges[1:-1], right=False)
        out = ~((v >= edges[0]) & (v <= edges[-1]))
        idx[out] = -1
        return idx
    z_bin  = _bin_idx(wazp['ZPHOT'], z_edges)
    ng_bin = _bin_idx(wazp['NGALS'], ng_edges)
    dm_bin = np.full(len(wazp), -1, dtype=int)
    for (ii, jj), med in dm_med.items():
        if med is None: continue
        cell = (z_bin == ii) & (ng_bin == jj)
        cell_dm = np.asarray(wazp['DELTA_M_I'])[cell]
        glob = np.where(cell)[0]
        dm_bin[glob[cell_dm <  med]] = 0
        dm_bin[glob[cell_dm >= med]] = 1

    good_by_id = {r['ID']: r for r in ind['results']
                  if r.get('sigmas') is not None}

    # Pool all surviving clusters into a single "all" stack: simple mean of
    # per-cluster ΔΣ vs the population-weighted aggregate of all stacked
    # sigmas (also unweighted average of stacked sigmas across the 24
    # stacks, weighted by n_stacked for a roughly equivalent comparison).
    rows = []
    for name, s_stk in stacks.items():
        ii, jj, kk = s_stk['z_bin'], s_stk['ng_bin'], s_stk['dm_bin']
        mask  = (z_bin == ii) & (ng_bin == jj) & (dm_bin == kk)
        ids   = np.asarray(wazp['ID'])[mask]
        sigs  = []
        for cid in ids:
            r = good_by_id.get(int(cid))
            if r is None: continue
            s = np.asarray(r['sigmas'], dtype=float)
            if not np.all(np.isfinite(s)): continue
            sigs.append(s)
        if not sigs: continue
        sigs = np.asarray(sigs)
        avg_unw = sigs.mean(axis=0)
        rows.append({
            'name':      name,
            'z_lo':      s_stk['z_lo'],
            'z_hi':      s_stk['z_hi'],
            'ng_lo':     s_stk['ng_lo'],
            'ng_hi':     s_stk['ng_hi'],
            'dm_bin':    kk,
            'n_used':    len(sigs),
            'avg_unw':   avg_unw,
            'stk_sig':   np.asarray(s_stk['sigmas']),
            'stk_err':   np.sqrt(np.diag(np.asarray(s_stk['sigmas_cov']))),
        })

    return {
        'radii_ind': radii_ind,
        'mr_radii':  np.asarray(stk['radii']),
        'rows':      rows,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════════

print(f'Running simulation arm with N={NCL_SIM} clusters '
      f'(DENSITY={DENSITY}, WIDTH_RAD={WIDTH_RAD}, σe={SHAPE_NOISE})')
per_cluster_sigmas, avg_uniform, sig_stack, ds_true = run_sim(NCL_SIM)

print()
print(f'Simulation: {len(per_cluster_sigmas)} usable clusters')
print(f'  radii  : {np.round(radii, 3).tolist()}')
print(f'  truth  : {np.round(ds_true, 1).tolist()}')
print(f'  stack  : {np.round(sig_stack, 1).tolist()}')
print(f'  avg    : {np.round(avg_uniform, 1).tolist()}')
print(f'  (avg - stack) / stack [%]  : '
      f'{np.round((avg_uniform / sig_stack - 1) * 100, 1).tolist()}')
print(f'  (stack - truth) / truth [%]: '
      f'{np.round((sig_stack / ds_true - 1) * 100, 1).tolist()}')
print(f'  (avg - truth)  / truth [%] : '
      f'{np.round((avg_uniform / ds_true - 1) * 100, 1).tolist()}')

print()
print('Running real-data arm (caches from wazp_*.pkl)')
real = run_real()


# ═════════════════════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# -- sim panel: relative residual (avg − stack)/stack ------------------------
ax = axes[0]
ax.axhline(0.0, color='k', ls='--', alpha=0.6)
ax.plot(radii, (avg_uniform / sig_stack - 1) * 100, 'o-',
        label=f'avg(per-cluster) vs stack  (N={len(per_cluster_sigmas)})',
        color='crimson', ms=6, lw=1.5)
# Sanity overlay: (avg-truth)/truth and (stack-truth)/truth
ax.plot(radii, (sig_stack    / ds_true - 1) * 100, 's--',
        label='stack vs truth', color='steelblue', ms=5, alpha=0.7)
ax.plot(radii, (avg_uniform  / ds_true - 1) * 100, 'd--',
        label='avg vs truth',  color='goldenrod', ms=5, alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel(r'$R$ [Mpc]')
ax.set_ylabel('residual  [%]')
ax.set_ylim(-100, 100)
ax.set_title(rf'Sim (gamma\_scan regime): '
             rf'M={M_TRUE:.0e}, c={C_TRUE}, z={Z_LENS}')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=9, loc='upper right')

# -- real panel: residual per stack (one curve per stack, viridis by NGALS) --
ax = axes[1]
ax.axhline(0.0, color='k', ls='--', alpha=0.6)
if real is not None:
    # Group stacks into NGALS buckets for clarity (single panel pooled)
    cmap = plt.cm.viridis
    rows = real['rows']
    # Color per (dm_bin)
    for row in rows:
        # Interpolate the stacked signal onto the per-cluster radii grid
        log_mr = np.log10(real['mr_radii'])
        log_tg = np.log10(real['radii_ind'])
        ok     = np.isfinite(row['stk_sig'])
        stk_at = np.interp(log_tg, log_mr[ok], row['stk_sig'][ok],
                            left=np.nan, right=np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = (row['avg_unw'] - stk_at) / stk_at * 100.0
        color = 'crimson' if row['dm_bin'] == 1 else 'steelblue'
        ax.plot(real['radii_ind'], res, '-', color=color, alpha=0.35, lw=1)

    # Population means for context
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='steelblue', lw=2, label='low Δm  stacks'),
        Line2D([0], [0], color='crimson',  lw=2, label='high Δm stacks'),
    ]
    ax.legend(handles=handles, fontsize=9, loc='upper right')
ax.set_xscale('log')
ax.set_xlabel(r'$R$ [Mpc]')
ax.set_ylabel('residual  [%]')
ax.set_ylim(-100, 100)
ax.set_title('Real WaZP data: avg(per-cluster) vs stack per (z, λ, Δm)')
ax.grid(alpha=0.3, which='both')

plt.suptitle(
    'Unweighted mean of per-cluster ΔΣ vs galaxy-level stacked ΔΣ',
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f'\nSaved plot to {OUT_PNG}')
