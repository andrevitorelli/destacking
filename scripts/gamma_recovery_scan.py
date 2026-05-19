#!/usr/bin/env python3
"""
gamma_recovery_scan.py

Scans 100 values of gamma (pivot-magnitude-gap power-law index) from 0.0 to 3.0.
For each gamma:
  1. Generates a truth catalog of N_CLUSTERS triaxial clusters.
  2. Simulates per-cluster ΔΣ profiles via the fast xlensing pipeline (parallel).
  3. Fits gamma with the destacking likelihood (nautilus nested sampler).
  4. Saves the posterior chain.

Results are checkpointed: already-completed gamma values are skipped on restart.

Final output (in OUTDIR):
  scatter_true_vs_recovered.png   — true vs median recovered gamma with 68% CI bars
  combined_posteriors_shifted.png — all shifted posteriors (gamma_hat - gamma_true)
                                    to expose systematic biases
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocess as mp
from scipy.interpolate import interp1d

# ── xlensing / destacking ────────────────────────────────────────────────────
import xlensing.cosmo as _xlc
import xlensing.sims as _xlsims
from xlensing.sims import (
    make_simple_random_cat,
    apply_triaxial_NFW_shear_region,
    r_vir,
    project_ellipsoid,
)
from xlensing.data import metacal_cluster_lensing, signal
import ngmix

from destacking.simulation import generate_truth_catalog, make_test_data
from destacking.fitting import run_sampler

# ── Parameters ────────────────────────────────────────────────────────────────
GAMMA_VALUES    = np.linspace(-1., 1. , 100)
N_CLUSTERS      = 10000          # clusters per gamma value
N_LIVE          = 200          # nautilus live points
N_WORKERS       = 15
SEED_BASE       = 42           # seed for gamma i = SEED_BASE + i*1000

RADII           = np.logspace(-0.6, 0.6, 8)
DENSITY         = 30
WIDTH_RAD       = 0.006
Z_SRC_RANGE     = (0.4, 1.5)
SHAPE_NOISE     = 0.25
NOISE_FLOOR_COV = 1e4

OUTDIR = 'gamma_scan_results'
os.makedirs(OUTDIR, exist_ok=True)

log_r      = np.log10(RADII)
dlog       = np.diff(log_r)[0]
edges      = np.logspace(log_r[0] - dlog/2, log_r[-1] + dlog/2, len(RADII) + 1)
bin_limits = np.column_stack([edges[:-1], edges[1:]])
N_BINS     = len(RADII)

# ── Fast DA: chi(z) spline replaces per-element scipy.quad ───────────────────
_z_tab    = np.linspace(0.0, 3.0, 5000)
_chi_tab  = _xlc.cosmology.comoving_distance(_z_tab).value
_chi_of_z = interp1d(_z_tab, _chi_tab, kind='linear',
                     bounds_error=False, fill_value='extrapolate')

def _DA_fast(z1, z2):
    z2_a = np.atleast_1d(np.asarray(z2, dtype=float))
    res  = (_chi_of_z(z2_a) - float(_chi_of_z(float(z1)))) / (1.0 + z2_a)
    return float(res[0]) if np.ndim(z2) == 0 else res

_xlc.DA    = _DA_fast   # data.py uses cosmo.DA(...)
_xlsims.DA = _DA_fast   # sims.py imported DA at module level

# ── Shared background source field (built once, reused for all gamma) ─────────
rng_base  = np.random.default_rng(99)
base_cat  = make_simple_random_cat(DENSITY, WIDTH_RAD, Z_SRC_RANGE, shape_noise=0.0, seed=0).T
N_GALS    = len(base_cat)
sizes     = rng_base.lognormal(mean=np.log(0.5), sigma=0.5, size=N_GALS)
R_RESP    = np.clip(1.0 - (sizes / 1.5)**2, 0.05, 0.99)
base_cat[:, 5] = R_RESP
base_cat[:, 6] = 0.0
base_cat[:, 7] = 0.0
base_cat[:, 8] = R_RESP

print(f"Source field: {N_GALS} galaxies | "
      f"{N_WORKERS} workers | {N_CLUSTERS} clusters/gamma | "
      f"{len(GAMMA_VALUES)} gamma values")


# ── Per-cluster worker ─────────────────────────────────────────────────────────
# All cluster-varying data is passed via the task tuple so the pool can be
# reused across gamma iterations (workers inherit base_cat, fast DA, etc. via fork).

def _process_cluster(task):
    i, z_i, M_i, c_i, a_i, q_ba_i, q_ca_i, theta_los_i, phi_los_i = task

    rng_i  = np.random.default_rng(1000 + i)
    sp_i   = ngmix.priors.shape.GPriorBA(sigma=SHAPE_NOISE, rng=rng_i)
    e1, e2 = sp_i.sample2d(N_GALS)

    gal_i = base_cat.copy()
    gal_i[:, 3] = e1
    gal_i[:, 4] = e2

    # Pre-filter to metacal background before shear injection (~half the galaxies)
    mc_mask = gal_i[:, 2] > 1.1 * z_i + 0.2
    gal_mc  = gal_i[mc_mask]
    if len(gal_mc) < 20:
        return i, None, None

    cluster_arr = [0., 0., z_i, M_i, c_i,
                   a_i, q_ba_i * a_i, q_ca_i * a_i,
                   theta_los_i, phi_los_i]
    gal_sheared = apply_triaxial_NFW_shear_region(cluster_arr, gal_mc)

    # Reorder to metacal format: [RA, DEC, z, E1, E2, W, R11, R12, R21, R22]
    sources_mc = gal_sheared[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]]
    radius_mpc = WIDTH_RAD * _DA_fast(0, z_i)
    bg = metacal_cluster_lensing((0., 0., z_i), sources_mc.T, radius_mpc)
    if bg['Count'] < 20:
        return i, None, None

    stake = np.array([
        bg['Critical Density'], bg['Tangential Shear'], bg['Cross Shear'],
        bg['Weights'], bg['Radial Distance'], bg['Mult. Bias'],
    ])
    sig_t, _ = signal(stake, bin_limits)
    dv = np.nan_to_num(sig_t, nan=0.0)

    R      = bg['Radial Distance']
    in_bin = (R[:, None] > bin_limits[:, 0]) & (R[:, None] < bin_limits[:, 1])
    cov_d  = np.full(N_BINS, NOISE_FLOOR_COV)
    for b in range(N_BINS):
        m   = in_bin[:, b]
        n_b = m.sum()
        if n_b >= 5:
            cov_d[b] = (SHAPE_NOISE * bg['Critical Density'][m].mean() / np.sqrt(n_b)) ** 2

    return i, dv, cov_d


def simulate_lensing(gamma_true, seed):
    """Simulate per-cluster ΔΣ profiles for one gamma value."""
    rng_tri = np.random.default_rng(seed + 10)

    truth = generate_truth_catalog(
        N_CLUSTERS,
        z_range=(0.2, 0.6),
        logM_range=(13.5, 15.0),
        delta_m_gamma=gamma_true,
        scatter_logc=0.0,
        seed=seed,
    )
    cluster_obs, _, _ = make_test_data(truth, RADII, fractional_noise=0.2, seed=seed + 1)

    r_vir_vals    = np.array([r_vir(float(z), float(M))
                               for z, M in zip(truth['z'], truth['M_halo'])])
    q_ba          = rng_tri.uniform(0.5, 1.0, N_CLUSTERS)
    q_ca          = rng_tri.uniform(0.3, q_ba)
    theta_los_arr = np.arccos(rng_tri.uniform(0.0, 1.0, N_CLUSTERS))
    phi_los_arr   = rng_tri.uniform(0.0, 2 * np.pi, N_CLUSTERS)

    tasks = [
        (i,
         float(truth['z'][i]), float(truth['M_halo'][i]), float(truth['concentration'][i]),
         r_vir_vals[i], q_ba[i], q_ca[i], theta_los_arr[i], phi_los_arr[i])
        for i in range(N_CLUSTERS)
    ]

    data_vectors = np.zeros((N_CLUSTERS, N_BINS))
    cov_matrices = np.full((N_CLUSTERS, N_BINS, N_BINS), 0.)
    for i in range(N_CLUSTERS):
        np.fill_diagonal(cov_matrices[i], NOISE_FLOOR_COV)

    with mp.Pool(N_WORKERS) as pool:
        for idx, dv, cov_d in pool.imap(_process_cluster, tasks, chunksize=20):
            if dv is not None:
                data_vectors[idx] = dv
                np.fill_diagonal(cov_matrices[idx], cov_d)

    return cluster_obs, data_vectors, cov_matrices


def weighted_quantile(values, quantiles, weights):
    sorter = np.argsort(values)
    v, w   = values[sorter], weights[sorter]
    cumw   = np.cumsum(w) / np.sum(w)
    return np.interp(quantiles, cumw, v)


# ── Main scan loop ─────────────────────────────────────────────────────────────
print(f"Scanning {len(GAMMA_VALUES)} gamma values: "
      f"{GAMMA_VALUES[0]:.3f} → {GAMMA_VALUES[-1]:.3f}")
print(f"Results in: {OUTDIR}/\n")

t_scan_start = time.time()

for gi, gamma_true in enumerate(GAMMA_VALUES):
    outfile = os.path.join(OUTDIR, f'chain_{gi:03d}_gamma{gamma_true:.4f}.npz')
    if os.path.exists(outfile):
        print(f"[{gi+1:3d}/100] gamma={gamma_true:.4f}  skipped (already done)")
        continue

    t0 = time.time()
    seed = SEED_BASE + gi * 1000

    print(f"[{gi+1:3d}/100] gamma={gamma_true:.4f}  sim...", end='', flush=True)
    cluster_obs, data_vectors, cov_matrices = simulate_lensing(gamma_true, seed)
    t_sim = time.time() - t0

    print(f" {t_sim:.1f}s  fit...", end='', flush=True)
    sampler = run_sampler(
        cluster_obs, data_vectors, cov_matrices, RADII,
        gamma_range=(-1.0, 1.0),
        n_live=N_LIVE,
        verbose=False,
        pool=N_WORKERS,
    )
    points, log_w, log_l = sampler.posterior()
    t_total = time.time() - t0

    weights_s  = np.exp(log_w - log_w.max())
    weights_s /= weights_s.sum()
    glo, gmd, ghi = weighted_quantile(points[:, 0], [0.16, 0.50, 0.84], weights_s)

    np.savez(outfile, gamma_true=gamma_true, points=points, log_w=log_w, log_l=log_l)

    elapsed = time.time() - t_scan_start
    remaining = elapsed / (gi + 1) * (len(GAMMA_VALUES) - gi - 1)
    print(f" {t_total:.1f}s  "
          f"recovered={gmd:.3f} +{ghi-gmd:.3f}/-{gmd-glo:.3f}  "
          f"ETA {remaining/60:.0f}min")

print(f"\nScan complete in {(time.time()-t_scan_start)/60:.1f} min — loading chains for plots.")

# ── Load all chains ────────────────────────────────────────────────────────────
gamma_true_arr = []
gamma_med_arr  = []
gamma_lo_arr   = []   # distance to 16th percentile
gamma_hi_arr   = []   # distance to 84th percentile
all_shifted    = []   # list of (shifted_gamma_samples, weights)

for gi, gamma_true in enumerate(GAMMA_VALUES):
    outfile = os.path.join(OUTDIR, f'chain_{gi:03d}_gamma{gamma_true:.4f}.npz')
    if not os.path.exists(outfile):
        print(f"  Warning: missing chain for gamma={gamma_true:.4f}, skipping.")
        continue

    d      = np.load(outfile)
    pts    = d['points']
    lw     = d['log_w']

    w      = np.exp(lw - lw.max())
    w     /= w.sum()

    glo, gmd, ghi = weighted_quantile(pts[:, 0], [0.16, 0.50, 0.84], w)

    gamma_true_arr.append(float(d['gamma_true']))
    gamma_med_arr.append(gmd)
    gamma_lo_arr.append(gmd - glo)
    gamma_hi_arr.append(ghi - gmd)
    all_shifted.append((pts[:, 0] - float(d['gamma_true']), w))

gamma_true_arr = np.array(gamma_true_arr)
gamma_med_arr  = np.array(gamma_med_arr)
gamma_lo_arr   = np.array(gamma_lo_arr)
gamma_hi_arr   = np.array(gamma_hi_arr)
bias_arr       = gamma_med_arr - gamma_true_arr

print(f"Loaded {len(gamma_true_arr)} chains.")
print(f"Median bias: {np.median(bias_arr):.4f}")
print(f"RMS bias:    {np.sqrt(np.mean(bias_arr**2)):.4f}")

# ── Plot 1: true vs recovered gamma ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

lim = [GAMMA_VALUES[0] - 0.1, GAMMA_VALUES[-1] + 0.1]

ax = axes[0]
ax.errorbar(
    gamma_true_arr, gamma_med_arr,
    yerr=[gamma_lo_arr, gamma_hi_arr],
    fmt='o', ms=4, capsize=2, lw=0.8, color='steelblue', alpha=0.7,
    label='Recovered (median ± 68%)',
)
ax.plot(lim, lim, 'k--', lw=1.5, label='Unbiased')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'True $\gamma$', fontsize=13)
ax.set_ylabel(r'Recovered $\hat{\gamma}$', fontsize=13)
ax.set_title(f'γ recovery — {N_CLUSTERS} clusters/γ,  n_live={N_LIVE}', fontsize=11)
ax.legend(fontsize=10)

ax = axes[1]
ax.scatter(gamma_true_arr, bias_arr, s=18, color='tomato', alpha=0.8, zorder=3)
ax.axhline(0, color='k', lw=1.5, ls='--', label='Zero bias')
med_bias = np.median(bias_arr)
ax.axhline(med_bias, color='tomato', lw=1.5, ls='-',
           label=f'Median = {med_bias:+.3f}')
ax.fill_between(lim, -0.15, 0.15, color='k', alpha=0.06, label='|bias| < 0.15')
ax.set_xlim(lim)
ax.set_xlabel(r'True $\gamma$', fontsize=13)
ax.set_ylabel(r'$\hat{\gamma} - \gamma_{\rm true}$', fontsize=13)
ax.set_title('Point-estimate bias', fontsize=11)
ax.legend(fontsize=10)

plt.suptitle('Triaxial NFW lensing — γ recovery (spherical NFW model)',
             y=1.02, fontsize=12)
plt.tight_layout()
out1 = os.path.join(OUTDIR, 'scatter_true_vs_recovered.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out1}")

# ── Plot 2: combined shifted posteriors ───────────────────────────────────────
x_grid = np.linspace(-2.5, 2.5, 1000)
kdes   = []
for shifted_samples, w in all_shifted:
    h, edges_h = np.histogram(shifted_samples, bins=300, range=(-3.0, 3.0),
                               weights=w, density=True)
    centres    = 0.5 * (edges_h[:-1] + edges_h[1:])
    kdes.append(np.interp(x_grid, centres, h, left=0.0, right=0.0))

kdes = np.array(kdes)   # shape (n_gamma, n_x)

fig, ax = plt.subplots(figsize=(10, 5))

# Light individual traces
for kde in kdes:
    ax.plot(x_grid, kde, color='steelblue', lw=0.4, alpha=0.15)

# Envelope and median
ax.fill_between(x_grid,
                np.percentile(kdes, 16, axis=0),
                np.percentile(kdes, 84, axis=0),
                alpha=0.35, color='steelblue', label='16th–84th percentile band')
ax.plot(x_grid, np.median(kdes, axis=0),
        color='steelblue', lw=2.5, label='Median posterior')

ax.axvline(0, color='crimson', lw=2, ls='--', label='Zero bias')
ax.axvline(med_bias, color='tomato', lw=1.5, ls='-',
           label=f'Median point-estimate bias = {med_bias:+.3f}')

ax.set_xlabel(r'$\hat{\gamma} - \gamma_{\rm true}$', fontsize=13)
ax.set_ylabel('Posterior density', fontsize=12)
ax.set_title(
    f'Combined shifted posteriors over {len(gamma_true_arr)} γ values\n'
    '(triaxial simulation, spherical NFW model — zero = unbiased)',
    fontsize=11,
)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(bottom=0)

plt.tight_layout()
out2 = os.path.join(OUTDIR, 'combined_posteriors_shifted.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out2}")
print("Done.")
