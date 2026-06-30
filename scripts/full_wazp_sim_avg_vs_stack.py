#!/usr/bin/env python3
"""
Full WaZP-style simulated catalogue with realistic mass function, HOD,
magnitude gaps, and a matched galaxy density.  Then replicate the §9
diagnostic of wazp_lensing_signals.ipynb -- unweighted mean of
per-cluster ΔΣ vs galaxy-level stacked ΔΣ per (z, NGALS, Δm) stack --
on both the simulation and the WaZP data, and compare.

Pipeline
========

1. Truth catalogue via destacking.simulation.generate_truth_catalog
   (Tinker-08 mass function, Zheng-07 HOD, power-law LF for satellites
   -> mass, concentration, BCG luminosity, brightest-satellite L, Δm).
2. Observable richness Λ via destacking.simulation.make_test_data
   (McClintock+18 inversion + Poisson scatter).
3. Apply the WaZP selection: Λ > 25, Δm > 0, z < 0.60.
4. For each cluster, simulate a metacal source catalogue at
   DENSITY = 7.5 gal/arcmin^2 (matches the WaZP pre-filtered density),
   inject the analytic NFW shear with xlensing.sims.apply_NFW_shear_region,
   reduce with metacal_cluster_lensing → single_cluster.
5. Bin into the same 24 stacks as wazp_mass_richness_by_delta_m
   (3 z × 4 NGALS × 2 per-cell-median Δm).
6. For each stack, take the unweighted mean of per-cluster ΔΣ and run
   stacked_signal on the same bgs.  Plot the residual
   (avg − stack) / stack [%].
7. Repeat step 6 on the cached WaZP data; produce a side-by-side plot.

Outputs
-------
  scripts/full_wazp_sim_avg_vs_stack.png
  scripts/full_wazp_sim_results.pkl   (checkpointed per-cluster bgs)
"""

import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from multiprocess import Pool, cpu_count

import xlensing.data
import xlensing.sims as xlsims
from xlensing.cosmo import DA

from destacking.simulation import generate_truth_catalog, make_test_data


# ── parameters ──────────────────────────────────────────────────────────────
N_TRUTH     = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
DENSITY     = 7.5      # gal/arcmin^2, matches the WaZP-filtered density
RADIUS_MPC  = 5.0      # cone radius matches wazp_lensing_signals
SHAPE_NOISE = 0.27     # DES Y3 σ_e per component
NBOOT       = 50
NWORKERS    = max(1, cpu_count() - 1)

CKPT_PKL    = 'full_wazp_sim_results.pkl'
OUT_PNG     = 'full_wazp_sim_avg_vs_stack.png'

# wazp_lensing_signals radial layout
radii = np.logspace(-0.8, 0.6, 8)
Nbin  = len(radii)
edges = np.logspace(
    np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    Nbin + 1,
)
BINS_LIMS = np.array([[edges[i], edges[i + 1]] for i in range(Nbin)])

# wazp_mass_richness_by_delta_m stack edges (3 z × 4 NGALS × 2 Δm)
Z_EDGES  = np.array([0.10, 0.30, 0.50, 0.60])
NG_EDGES = np.array([25., 50., 75., 100., np.inf])
N_NG     = len(NG_EDGES) - 1


def _bin_idx(values, edges):
    v   = np.asarray(values, dtype=float)
    idx = np.digitize(v, edges[1:-1], right=False)
    out = ~((v >= edges[0]) & (v <= edges[-1]))
    idx[out] = -1
    return idx


# ═════════════════════════════════════════════════════════════════════════════
# 1-3. Truth catalogue + observable Λ + selection
# ═════════════════════════════════════════════════════════════════════════════

def build_catalogue():
    """Generate enough truth clusters so that N_TRUTH=N_TARGET clusters
    survive the WaZP selection (Λ>25, Δm>0, 0.1≤z<0.6).  Truth is drawn
    in batches with logM_range=(14.0, 15.0) -- McClintock+18 implies
    Λ≈25 at log10M≈14.2 at z=0.35 -- so the selection efficiency is
    roughly ~40% per batch.  We keep drawing until N_TARGET pass."""
    N_TARGET   = N_TRUTH
    batch_size = max(int(N_TARGET / 0.35), 5000)
    print(f'Target: {N_TARGET:,} selected clusters '
          f'(Λ>25, Δm>0, 0.1≤z<0.6).  Drawing in batches of '
          f'{batch_size:,}.', flush=True)

    selected = []
    n_sel    = 0
    n_drawn  = 0
    batch    = 0
    t0       = time.time()
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
        n_drawn += len(truth)
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
        truth_sel = truth[keep]
        selected.append(truth_sel)
        n_sel += len(truth_sel)
        print(f'  batch {batch}: {len(truth_sel):,} pass '
              f'({len(truth_sel)/len(truth)*100:.1f}%)  '
              f'running total: {n_sel:,}/{N_TARGET:,}  '
              f'(drew {n_drawn:,} so far, elapsed {(time.time()-t0)/60:.1f} min)',
              flush=True)

    from astropy.table import vstack
    truth_sel = vstack(selected)[:N_TARGET]
    print(f'Final selected sample: {len(truth_sel):,} clusters '
          f'(drew {n_drawn:,} truth clusters total; '
          f'overall efficiency {len(truth_sel)/n_drawn*100:.1f}%)',
          flush=True)
    return truth_sel


# ═════════════════════════════════════════════════════════════════════════════
# 4. Per-cluster shear simulation and ΔΣ reduction
# ═════════════════════════════════════════════════════════════════════════════

# Pre-compute the angular-diameter table once for fast lookup in workers.
_DA_GRID = None
_DA_VAL  = None


def _init_DA_table():
    global _DA_GRID, _DA_VAL
    _DA_GRID = np.linspace(0.005, 1.0, 400)
    _DA_VAL  = np.array([float(DA(0, z)) for z in _DA_GRID])


def DA_fast(z):
    return float(np.interp(z, _DA_GRID, _DA_VAL))


def per_cluster_worker(args):
    """Return (i, sigmas, bg) for one cluster.  bg is the (6, N_gal) array
    needed by stacked_signal downstream."""
    i, z_i, M_i, c_i, seed = args
    try:
        DA_lens = DA_fast(z_i)
        # Convert cone radius to angular radians for make_simple_random_cat.
        # width_rad is the HALF-width of the square footprint, so set it to
        # the desired angular cone radius (slightly oversized).
        width_rad = (RADIUS_MPC + 0.3) / DA_lens

        cat = xlsims.make_simple_random_cat(
            density=DENSITY, width_rad=width_rad,
            zrange=(0.0, 2.5), shape_noise=SHAPE_NOISE,
            seed=int(seed), poisson_density=True,
        ).T
        if len(cat) == 0:
            return i, None, None

        cat = xlsims.apply_NFW_shear_region(
            np.array([0.0, 0.0, z_i, M_i, c_i]), cat,
        )
        sources_mc = cat[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]].T
        res = xlensing.data.metacal_cluster_lensing(
            (0.0, 0.0, z_i), sources_mc, radius=RADIUS_MPC,
        )
        if res['Count'] == 0:
            return i, None, None
        bg = np.array([
            np.asarray(res['Critical Density'], dtype='float32'),
            np.asarray(res['Tangential Shear'], dtype='float32'),
            np.asarray(res['Cross Shear'],      dtype='float32'),
            np.asarray(res['Weights'],          dtype='float32'),
            np.asarray(res['Radial Distance'],  dtype='float32'),
            np.asarray(res['Mult. Bias'],       dtype='float32'),
        ])
        sig, *_ = xlensing.data.single_cluster([bg], BINS_LIMS, Nboot=NBOOT)
        return i, np.asarray(sig, dtype='float32'), bg
    except Exception as exc:
        return i, None, repr(exc)


def run_simulation(truth_sel):
    """Run per-cluster shear sim + single_cluster on every selected
    cluster.  Checkpoints results to CKPT_PKL."""
    if os.path.exists(CKPT_PKL):
        with open(CKPT_PKL, 'rb') as fh:
            d = pickle.load(fh)
        if (d.get('n_truth') == len(truth_sel)
            and d.get('density') == DENSITY
            and d.get('radius') == RADIUS_MPC
            and d.get('nboot') == NBOOT):
            print(f'Resuming from cached {CKPT_PKL} '
                  f'({d["n_done"]:,}/{len(truth_sel):,} done)', flush=True)
            return d

    n = len(truth_sel)
    sigmas = np.full((n, Nbin), np.nan, dtype='float32')
    bgs    = [None] * n
    n_done = 0

    args = [
        (i,
         float(truth_sel['z'][i]),
         float(truth_sel['M_halo'][i]),
         float(truth_sel['concentration'][i]),
         100_000 + i)
        for i in range(n)
    ]

    print(f'Running per-cluster shear sim + single_cluster on {n:,} clusters '
          f'with {NWORKERS} workers...', flush=True)
    _init_DA_table()
    t0 = time.time()
    with Pool(NWORKERS) as pool:
        for k, (i, sig_i, bg_i) in enumerate(
                pool.imap_unordered(per_cluster_worker, args, chunksize=8)):
            if sig_i is not None and isinstance(bg_i, np.ndarray):
                sigmas[i] = sig_i
                bgs[i]    = bg_i
                n_done   += 1
            if (k + 1) % 500 == 0 or (k + 1) == n:
                dt  = time.time() - t0
                eta = dt * (n - k - 1) / (k + 1)
                print(f'  {k+1:>6,}/{n:,}  good={n_done:>6,}  '
                      f'elapsed {dt/60:5.1f} min  eta {eta/60:5.1f} min',
                      flush=True)
            if (k + 1) % 5000 == 0:
                # Periodic checkpoint
                with open(CKPT_PKL, 'wb') as fh:
                    pickle.dump({
                        'n_truth': n, 'n_done': n_done,
                        'density': DENSITY, 'radius': RADIUS_MPC,
                        'nboot':   NBOOT,
                        'sigmas':  sigmas, 'bgs': bgs,
                    }, fh)

    with open(CKPT_PKL, 'wb') as fh:
        pickle.dump({
            'n_truth': n, 'n_done': n_done,
            'density': DENSITY, 'radius': RADIUS_MPC, 'nboot': NBOOT,
            'sigmas':  sigmas, 'bgs': bgs,
        }, fh)
    print(f'Saved {n_done:,} per-cluster reductions to {CKPT_PKL} '
          f'(total {(time.time()-t0)/60:.1f} min)', flush=True)
    return {'sigmas': sigmas, 'bgs': bgs, 'n_done': n_done}


# ═════════════════════════════════════════════════════════════════════════════
# 5-6. Stack assignment, average + stacked_signal per stack
# ═════════════════════════════════════════════════════════════════════════════

def assign_stack_bins(truth_sel):
    z_bin  = _bin_idx(truth_sel['z'],       Z_EDGES)
    ng_bin = _bin_idx(truth_sel['Lambda'],  NG_EDGES)
    dm_bin = np.full(len(truth_sel), -1, dtype=int)
    dm_medians = {}
    for i in range(3):
        for j in range(N_NG):
            cell = (z_bin == i) & (ng_bin == j)
            if cell.sum() < 2:
                dm_medians[(i, j)] = None
                continue
            cell_dm = np.asarray(truth_sel['delta_m'])[cell]
            med = float(np.median(cell_dm))
            dm_medians[(i, j)] = med
            glob = np.where(cell)[0]
            dm_bin[glob[cell_dm <  med]] = 0
            dm_bin[glob[cell_dm >= med]] = 1
    return z_bin, ng_bin, dm_bin, dm_medians


def stack_diagnostics(sigmas, bgs, z_bin, ng_bin, dm_bin):
    """Per (z, ng, dm) stack, return dict of name -> (avg, stack, n_used)."""
    out = {}
    for i in range(3):
        for j in range(N_NG):
            for k in range(2):
                name = f'z{i}_ng{j}_dm{k}'
                mask = (z_bin == i) & (ng_bin == j) & (dm_bin == k)
                idxs = np.where(mask)[0]
                if idxs.size == 0:
                    continue
                # Filter to clusters with valid signal + bg
                sigs_k = []
                bgs_k  = []
                for ii in idxs:
                    s = sigmas[ii]
                    if not np.all(np.isfinite(s)): continue
                    if bgs[ii] is None: continue
                    sigs_k.append(s)
                    bgs_k.append(bgs[ii])
                if len(bgs_k) < 5:
                    continue
                sigs_k = np.asarray(sigs_k)
                avg_unw = sigs_k.mean(axis=0)
                sig_stack, *_ = xlensing.data.stacked_signal(
                    bgs_k, BINS_LIMS, Nboot=NBOOT,
                )
                out[name] = {
                    'z_bin': i, 'ng_bin': j, 'dm_bin': k,
                    'n_used':  len(bgs_k),
                    'avg_unw': avg_unw,
                    'stack':   np.asarray(sig_stack),
                }
                print(f'  {name}: N={len(bgs_k):>5,}', flush=True)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Real-data side: same diagnostic on the cached WaZP results
# ═════════════════════════════════════════════════════════════════════════════

def real_diagnostic():
    pkl_ind = '/home/andre/github/destacking/notebooks/data/wazp_lensing_signals.pkl'
    pkl_stk = '/home/andre/github/destacking/notebooks/data/wazp_mr_dm_stacks.pkl'
    fits    = '/home/andre/github/destacking/notebooks/data/wazp_real_data.fits'
    for p in (pkl_ind, pkl_stk, fits):
        if not os.path.exists(p):
            print(f'missing {p}; skipping real-data arm')
            return None

    with open(pkl_ind, 'rb') as fh: ind = pickle.load(fh)
    with open(pkl_stk, 'rb') as fh: stk = pickle.load(fh)
    wazp_all = Table.read(fits)
    keep = np.isfinite(wazp_all['DELTA_M_I']) & (wazp_all['DELTA_M_I'] > 0)
    wazp = wazp_all[keep]

    radii_ind = np.asarray(ind['radii'])
    z_edges   = stk['z_edges']
    ng_edges  = stk['ng_edges']
    dm_med    = stk['dm_median_per_cell']
    stacks    = stk['stacks']
    mr_radii  = np.asarray(stk['radii'])

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

    out = {}
    for name, s_stk in stacks.items():
        ii, jj, kk = s_stk['z_bin'], s_stk['ng_bin'], s_stk['dm_bin']
        mask = (z_bin == ii) & (ng_bin == jj) & (dm_bin == kk)
        ids  = np.asarray(wazp['ID'])[mask]
        sigs = []
        for cid in ids:
            r = good_by_id.get(int(cid))
            if r is None: continue
            s = np.asarray(r['sigmas'], dtype=float)
            if not np.all(np.isfinite(s)): continue
            sigs.append(s)
        if not sigs: continue
        sigs = np.asarray(sigs)
        avg_unw = sigs.mean(axis=0)

        # Interpolate stacked signal from mr_radii onto radii_ind
        log_mr = np.log10(mr_radii)
        log_tg = np.log10(radii_ind)
        valid  = np.isfinite(s_stk['sigmas'])
        stk_at = np.interp(log_tg, log_mr[valid],
                            np.asarray(s_stk['sigmas'])[valid],
                            left=np.nan, right=np.nan)
        out[name] = {
            'z_bin': ii, 'ng_bin': jj, 'dm_bin': kk,
            'n_used':  len(sigs),
            'avg_unw': avg_unw,
            'stack':   stk_at,
            'radii':   radii_ind,
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Plot side-by-side
# ═════════════════════════════════════════════════════════════════════════════

def plot(sim, real):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, results, title, x in (
        (axes[0], sim,  'Full simulation (Tinker+HOD, ρ=7.5 gal/arcmin², '
                        f'N={N_TRUTH:,} truth)', radii),
        (axes[1], real, 'Real WaZP data',
                        np.asarray(list(real.values())[0]['radii'])
                        if real else radii),
    ):
        ax.axhline(0.0, color='k', ls='--', alpha=0.5)
        if results is None:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center')
            continue
        for name, row in results.items():
            avg = row['avg_unw']; stk = row['stack']
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = (avg - stk) / stk * 100.0
            color = 'crimson' if row['dm_bin'] == 1 else 'steelblue'
            ax.plot(x, rel, '-', color=color, alpha=0.35, lw=1)
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0], [0], color='steelblue', lw=2, label='low Δm'),
            Line2D([0], [0], color='crimson',  lw=2, label='high Δm'),
        ], loc='upper right', fontsize=10)
        ax.set_xscale('log')
        ax.set_xlabel(r'$R$ [Mpc]')
        ax.set_ylim(-100, 100)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel(r'$(\langle\Delta\Sigma\rangle_{\rm ind} '
                       r'- \Delta\Sigma_{\rm stack}) / \Delta\Sigma_{\rm stack}$  [%]')
    plt.suptitle(
        'avg(per-cluster ΔΣ) vs stacked ΔΣ per (z, λ, Δm) stack',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'Saved plot to {OUT_PNG}')


# ═════════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    truth_sel = build_catalogue()
    sim_state = run_simulation(truth_sel)
    sigmas    = sim_state['sigmas']
    bgs       = sim_state['bgs']
    print(f'\nValid per-cluster reductions: {sim_state["n_done"]:,}')

    z_bin, ng_bin, dm_bin, dm_medians = assign_stack_bins(truth_sel)
    print('\nPer-stack populations (sim, after good-cluster filter):')
    sim_results = stack_diagnostics(sigmas, bgs, z_bin, ng_bin, dm_bin)

    print('\nRunning real-data diagnostic (cached WaZP results)...')
    real_results = real_diagnostic()

    plot(sim_results, real_results)
