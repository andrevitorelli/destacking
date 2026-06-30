#!/usr/bin/env python3
"""Reproduce the §9 wazp_lensing_signals.ipynb plot on the REAL WaZP
data, in the same code as full_wazp_sim_section9_plot.py, so the two
sit side-by-side in the report with identical formatting."""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


# ── parameters (same as the notebook's §9 cell) ─────────────────────────────
PKL_IND           = '/home/andre/github/destacking/notebooks/data/wazp_lensing_signals.pkl'
PKL_STK           = '/home/andre/github/destacking/notebooks/data/wazp_mr_dm_stacks.pkl'
FITS_REAL         = '/home/andre/github/destacking/notebooks/data/wazp_real_data.fits'
OUT_PNG           = 'wazp_section9_replot.png'
MIN_BOOST_INCLUDE = 0.0


def _bin_idx(values, edges):
    v   = np.asarray(values, dtype=float)
    idx = np.digitize(v, edges[1:-1], right=False)
    out = ~((v >= edges[0]) & (v <= edges[-1]))
    idx[out] = -1
    return idx


with open(PKL_IND, 'rb') as fh: ind = pickle.load(fh)
with open(PKL_STK, 'rb') as fh: stk = pickle.load(fh)
wazp_all = Table.read(FITS_REAL)
keep = np.isfinite(wazp_all['DELTA_M_I']) & (wazp_all['DELTA_M_I'] > 0)
wazp = wazp_all[keep]

radii_ind = np.asarray(ind['radii'])
mr_radii  = np.asarray(stk['radii'])
z_edges   = stk['z_edges']
ng_edges  = stk['ng_edges']
dm_med    = stk['dm_median_per_cell']
stacks    = stk['stacks']
N_NG      = len(ng_edges) - 1

z_bin_arr  = _bin_idx(wazp['ZPHOT'], z_edges)
ng_bin_arr = _bin_idx(wazp['NGALS'], ng_edges)
dm_bin_arr = np.full(len(wazp), -1, dtype=int)
for (ii, jj), med in dm_med.items():
    if med is None: continue
    cell = (z_bin_arr == ii) & (ng_bin_arr == jj)
    cell_dm = np.asarray(wazp['DELTA_M_I'])[cell]
    glob = np.where(cell)[0]
    dm_bin_arr[glob[cell_dm <  med]] = 0
    dm_bin_arr[glob[cell_dm >= med]] = 1

good_by_id = {r['ID']: r for r in ind['results']
              if r.get('sigmas') is not None}


def avg_per_stack(ii, jj, kk):
    mask = (z_bin_arr == ii) & (ng_bin_arr == jj) & (dm_bin_arr == kk)
    ids  = np.asarray(wazp['ID'])[mask]
    sigs, vars_d = [], []
    for cid in ids:
        r = good_by_id.get(int(cid))
        if r is None: continue
        s = np.asarray(r['sigmas'],     dtype=float)
        b = np.asarray(r['boosts'],     dtype=float)
        v = np.diag(np.asarray(r['sigmas_cov'], dtype=float))
        ok = (np.isfinite(s) & np.isfinite(v) & (v > 0)
              & np.isfinite(b) & (b > 0))
        if not ok.all(): continue
        if MIN_BOOST_INCLUDE > 0 and (b < MIN_BOOST_INCLUDE).any():
            continue
        sigs.append(s); vars_d.append(v)
    if not sigs: return None
    sigs   = np.asarray(sigs)
    vars_d = np.asarray(vars_d)
    w = 1.0 / vars_d
    mean = (sigs * w).sum(axis=0) / w.sum(axis=0)
    err  = np.sqrt(1.0 / w.sum(axis=0))
    return mean, err, len(sigs)


def interp_to_radii(stk_sig, stk_err, target):
    log_mr = np.log10(mr_radii)
    log_tg = np.log10(target)
    valid  = np.isfinite(stk_sig) & np.isfinite(stk_err)
    sig_i  = np.interp(log_tg, log_mr[valid], stk_sig[valid],
                        left=np.nan, right=np.nan)
    err_i  = np.interp(log_tg, log_mr[valid], stk_err[valid],
                        left=np.nan, right=np.nan)
    return sig_i, err_i


ng_colors = [plt.cm.viridis(x) for x in np.linspace(0.0, 0.95, N_NG)]
dm_titles = {0: 'low Δm', 1: 'high Δm'}

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
for kk in range(2):
    for ii in range(3):
        ax = axes[kk, ii]
        for jj in range(N_NG):
            name = f'z{ii}_ng{jj}_dm{kk}'
            if name not in stacks: continue
            s_stk   = stacks[name]
            stk_sig = np.asarray(s_stk['sigmas'])
            stk_err = np.sqrt(np.diag(np.asarray(s_stk['sigmas_cov'])))
            grp = avg_per_stack(ii, jj, kk)
            if grp is None: continue
            mean_ind, sig_ind, Nused = grp
            stk_at_r, stk_err_at_r = interp_to_radii(stk_sig, stk_err, radii_ind)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel     = (mean_ind - stk_at_r) / stk_at_r * 100.0
                rel_err = (np.sqrt(sig_ind**2 + stk_err_at_r**2)
                           / np.abs(stk_at_r) * 100.0)
            color  = ng_colors[jj]
            hi_lbl = (f'{s_stk["ng_hi"]:.0f}'
                      if np.isfinite(s_stk['ng_hi']) else r'\infty')
            ax.errorbar(radii_ind, rel, yerr=rel_err,
                        fmt='o-', ms=5, color=color, capsize=2.5, mew=0.8,
                        lw=1.2, alpha=0.9,
                        label=rf'$\lambda\in[{s_stk["ng_lo"]:.0f},{hi_lbl})$  '
                              f'N={Nused:,}')
        ax.axhline(0.0, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.set_xscale('log')
        if kk == 1:
            ax.set_xlabel(r'$R$ [Mpc]')
        if ii == 0:
            ax.set_ylabel(
                rf'{dm_titles[kk]}' + '\n'
                + r'$(\langle\Delta\Sigma\rangle_{\rm ind} - '
                  r'\Delta\Sigma_{\rm stack}) / \Delta\Sigma_{\rm stack}$  [%]'
            )
        ax.set_title(rf'$z\in[{z_edges[ii]:.2f},{z_edges[ii+1]:.2f})$',
                     fontsize=10)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(-50, 50)

plt.suptitle(r'Relative residual: per-cluster average vs stacked '
             r'(real WaZP data; both boost-corrected)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f'Saved {OUT_PNG}')
