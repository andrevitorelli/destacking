#!/usr/bin/env python3
"""Reproduce the §9 diagnostic plot of wazp_lensing_signals.ipynb on the
full WaZP-mirror simulation.

Layout: 2 rows (low Δm / high Δm) × 3 columns (z slices); per panel, 4
NGALS curves (viridis) with the inverse-variance-weighted mean of
per-cluster ΔΣ vs the stacked ΔΣ residual in %, with errorbars and
y-axis clipped to ±50%.

The per-cluster sigmas + sigmas_cov are recomputed from the cached bgs
in full_wazp_sim_results.pkl so this script does NOT redo any shear
injection.  Stacked signals come from a fresh stacked_signal call on
each stack's bgs.
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocess import Pool, cpu_count

import xlensing.data


# ── parameters (must match full_wazp_sim_avg_vs_stack.py) ───────────────────
CKPT_PKL          = 'full_wazp_sim_results.pkl'
TRUTH_PKL         = 'full_wazp_sim_truth.pkl'   # produced by re-running build_catalogue
OUT_PNG           = 'full_wazp_sim_section9_plot.png'
NBOOT             = 50
NWORKERS          = max(1, cpu_count() - 1)
MIN_BOOST_INCLUDE = 0.0   # same as the §9 default

radii = np.logspace(-0.8, 0.6, 8)
Nbin  = len(radii)
edges = np.logspace(
    np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    Nbin + 1,
)
BINS_LIMS = np.array([[edges[i], edges[i + 1]] for i in range(Nbin)])

Z_EDGES  = np.array([0.10, 0.30, 0.50, 0.60])
NG_EDGES = np.array([25., 50., 75., 100., np.inf])
N_NG     = len(NG_EDGES) - 1


def _bin_idx(values, edges):
    v   = np.asarray(values, dtype=float)
    idx = np.digitize(v, edges[1:-1], right=False)
    out = ~((v >= edges[0]) & (v <= edges[-1]))
    idx[out] = -1
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Re-derive the truth catalogue (deterministic given seeds)
# ─────────────────────────────────────────────────────────────────────────────

def regenerate_truth_catalogue():
    if os.path.exists(TRUTH_PKL):
        with open(TRUTH_PKL, 'rb') as fh:
            return pickle.load(fh)
    print('Re-generating truth catalogue (matching seeds of '
          'full_wazp_sim_avg_vs_stack.py)...', flush=True)
    from destacking.simulation import generate_truth_catalog, make_test_data
    from astropy.table import vstack

    N_TARGET   = 15000
    batch_size = max(int(N_TARGET / 0.35), 5000)
    selected   = []
    n_sel      = 0
    batch      = 0
    while n_sel < N_TARGET:
        batch += 1
        truth = generate_truth_catalog(
            N=batch_size,
            z_range=(0.10, 0.60),
            logM_range=(14.0, 15.0),
            delta_m_gamma=0.0,
            c_pivot_0=1.0,
            scatter_logc=0.1,
            seed=2026 + batch,
        )
        cluster_obs, _, _ = make_test_data(
            truth, radii, poisson_richness=True, seed=2026 + batch,
        )
        truth['Lambda'] = cluster_obs['Lambda']
        keep = (
            (truth['Lambda']  > 25.0)
            & (truth['delta_m'] > 0.0)
            & np.isfinite(truth['delta_m'])
            & (truth['z']     < 0.60)
            & (truth['z']     >= 0.10)
        )
        selected.append(truth[keep])
        n_sel += int(keep.sum())
    truth_sel = vstack(selected)[:N_TARGET]
    with open(TRUTH_PKL, 'wb') as fh:
        pickle.dump(truth_sel, fh)
    return truth_sel


# ─────────────────────────────────────────────────────────────────────────────
# Re-run single_cluster on the cached bgs to get sigmas + sigmas_cov + boosts
# ─────────────────────────────────────────────────────────────────────────────

def reduce_one(args):
    i, bg = args
    if bg is None:
        return i, None
    try:
        sig, boost, _, sigcov, _, _ = xlensing.data.single_cluster(
            [bg], BINS_LIMS, Nboot=NBOOT,
        )
        return i, {
            'sigmas':     np.asarray(sig,    dtype='float32'),
            'boosts':     np.asarray(boost,  dtype='float32'),
            'sigmas_cov': np.asarray(sigcov, dtype='float32'),
        }
    except Exception as exc:
        return i, repr(exc)


def rerun_per_cluster(bgs):
    n = len(bgs)
    print(f'Re-running single_cluster on {n:,} cached bgs '
          f'with {NWORKERS} workers (no shear injection)...', flush=True)
    out = [None] * n
    t0 = time.time()
    with Pool(NWORKERS) as pool:
        for k, (i, res) in enumerate(
                pool.imap_unordered(reduce_one, list(enumerate(bgs)),
                                    chunksize=16)):
            if isinstance(res, dict):
                out[i] = res
            if (k + 1) % 1000 == 0 or (k + 1) == n:
                dt  = time.time() - t0
                eta = dt * (n - k - 1) / (k + 1)
                print(f'  {k+1:>6,}/{n:,}  elapsed {dt/60:5.1f} min  '
                      f'eta {eta/60:5.1f} min', flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stack assignment, per-stack inv-var avg, per-stack stacked_signal, plot
# ─────────────────────────────────────────────────────────────────────────────

def per_stack(truth, per_cluster, bgs, z_bin_arr, ng_bin_arr, dm_bin_arr):
    results = {}
    for i in range(3):
        for j in range(N_NG):
            for k in range(2):
                name = f'z{i}_ng{j}_dm{k}'
                mask = (z_bin_arr == i) & (ng_bin_arr == j) & (dm_bin_arr == k)
                idxs = np.where(mask)[0]
                if idxs.size == 0:
                    continue
                # Inverse-variance weighted mean of per-cluster sigmas
                sigs, vars_d = [], []
                bgs_k = []
                for ii in idxs:
                    r = per_cluster[ii]
                    if r is None:
                        continue
                    s = r['sigmas']
                    b = r['boosts']
                    v = np.diag(r['sigmas_cov'])
                    ok = (np.isfinite(s) & np.isfinite(v) & (v > 0)
                          & np.isfinite(b) & (b > 0))
                    if not ok.all():
                        continue
                    if MIN_BOOST_INCLUDE > 0 and (b < MIN_BOOST_INCLUDE).any():
                        continue
                    sigs.append(s.astype(float))
                    vars_d.append(v.astype(float))
                    bgs_k.append(bgs[ii])
                if len(sigs) < 5:
                    continue
                sigs   = np.asarray(sigs)
                vars_d = np.asarray(vars_d)
                w   = 1.0 / vars_d
                mean = (sigs * w).sum(axis=0) / w.sum(axis=0)
                err  = np.sqrt(1.0 / w.sum(axis=0))
                # Stacked signal on the same bgs
                sig_stack, *_ = xlensing.data.stacked_signal(
                    bgs_k, BINS_LIMS, Nboot=NBOOT,
                )
                stk_err = np.sqrt(np.diag(_[2]))
                results[name] = {
                    'z_bin':  i, 'ng_bin': j, 'dm_bin': k,
                    'n_used': len(sigs),
                    'avg':    mean,
                    'avg_err': err,
                    'stack':  np.asarray(sig_stack),
                    'stk_err': stk_err,
                }
                print(f'  {name}: N={len(sigs):>5,}', flush=True)
    return results


def make_section9_plot(stacks):
    dm_colors = {0: 'steelblue', 1: 'crimson'}
    dm_titles = {0: 'low Δm', 1: 'high Δm'}
    ng_colors = [plt.cm.viridis(x) for x in np.linspace(0.0, 0.95, N_NG)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for kk in range(2):
        for ii in range(3):
            ax = axes[kk, ii]
            for jj in range(N_NG):
                name = f'z{ii}_ng{jj}_dm{kk}'
                if name not in stacks:
                    continue
                row = stacks[name]
                stk = row['stack']
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel     = (row['avg'] - stk) / stk * 100.0
                    rel_err = (np.sqrt(row['avg_err']**2 + row['stk_err']**2)
                               / np.abs(stk) * 100.0)
                color  = ng_colors[jj]
                hi_lbl = (f'{NG_EDGES[jj+1]:.0f}'
                          if np.isfinite(NG_EDGES[jj+1]) else r'\infty')
                ax.errorbar(radii, rel, yerr=rel_err,
                            fmt='o-', ms=5, color=color, capsize=2.5,
                            mew=0.8, lw=1.2, alpha=0.9,
                            label=(rf'$\lambda\in[{NG_EDGES[jj]:.0f},{hi_lbl})$  '
                                   f'N={row["n_used"]:,}'))
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
            ax.set_title(rf'$z\in[{Z_EDGES[ii]:.2f},{Z_EDGES[ii+1]:.2f})$',
                         fontsize=10)
            ax.grid(alpha=0.3, which='both')
            ax.legend(fontsize=7, loc='upper right')
            ax.set_ylim(-50, 50)
    plt.suptitle(r'Relative residual: per-cluster average vs stacked '
                 r'(simulation, ρ=7.5 gal/arcmin², N=15,000 truth)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'\nSaved {OUT_PNG}', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with open(CKPT_PKL, 'rb') as fh:
        d = pickle.load(fh)
    bgs = d['bgs']

    truth = regenerate_truth_catalogue()
    z_bin  = _bin_idx(truth['z'],      Z_EDGES)
    ng_bin = _bin_idx(truth['Lambda'], NG_EDGES)
    dm_bin = np.full(len(truth), -1, dtype=int)
    for i in range(3):
        for j in range(N_NG):
            cell = (z_bin == i) & (ng_bin == j)
            if cell.sum() < 2:
                continue
            cell_dm = np.asarray(truth['delta_m'])[cell]
            med = float(np.median(cell_dm))
            glob = np.where(cell)[0]
            dm_bin[glob[cell_dm <  med]] = 0
            dm_bin[glob[cell_dm >= med]] = 1

    per_cluster = rerun_per_cluster(bgs)
    print()
    stacks = per_stack(truth, per_cluster, bgs, z_bin, ng_bin, dm_bin)
    make_section9_plot(stacks)
