#!/usr/bin/env python3
"""
Diagnose why the per-cluster average of single_cluster ΔΣ profiles
appears to go toward zero at small radii in wazp_lensing_signals.ipynb.

Hypothesis: at low source counts in inner bins (a few per cluster), the
per-cluster boost-corrected ΔΣ from the bootstrap is biased — either by
selection effects from dropping zero-N_b bootstrap iterations or by
inverse-variance averaging picking up systematically biased outliers.

Strategy: synthesise a population of clusters with KNOWN NFW shear
injected on uniformly-distributed source galaxies.  In this setup the
true per-cluster boost is 1.0 in every bin by construction (no
cluster-member contamination), so any deviation between
single_cluster's output and analytical NFW truth is purely a
per-cluster-estimator issue, not a real-cluster contamination issue.

For each scenario we:
  1. Run single_cluster on every synthetic cluster.
  2. Compute the per-bin inverse-variance-weighted mean of per-cluster ΔΣ.
  3. Run stacked_signal on the same clusters (galaxy-level estimator).
  4. Compare both estimators against analytical NFW truth.
  5. Sweep the per-cluster source count to see where any bias turns on.

If single_cluster is biased low at small R (where N_inner per cluster
is a few), we have reproduced the wazp_lensing_signals symptom and can
diagnose its source.  If single_cluster matches truth here, the symptom
in the real-data residual comes from somewhere else (e.g. the per-
cluster boost in the presence of real cluster contamination).

Usage:
    python diagnose_single_cluster_inner_bin_bias.py [n_clusters] [n_gal_per_cluster]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import xlensing.data
import xlensing.model
from xlensing.cosmo import DA, lightspeed, gravity

# ── parameters ──────────────────────────────────────────────────────────────
NCL    = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NGAL   = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

M_TRUE = 2.0e14      # M200m in M_sun
C_TRUE = 5.0         # concentration
Z_LENS = 0.4
RADIUS = 5.0         # cone radius in Mpc (proper)
NBOOT  = 200
SIG_E  = 0.27        # shape noise per component

OUT_PNG = 'diagnose_single_cluster_inner_bin_bias.png'

# Radial bins matching the wazp_lensing_signals notebook
radii = np.logspace(-0.8, 0.6, 8)
Nbin  = len(radii)
edges = np.logspace(
    np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    Nbin + 1,
)
BINS = np.array([[edges[i], edges[i + 1]] for i in range(Nbin)])

# ── analytical NFW truth at the bin centres ─────────────────────────────────
truth = xlensing.model.NFW_Delta_Sigma(
    M200m=np.array([M_TRUE]),
    C200m=np.array([C_TRUE]),
    Z=np.array([Z_LENS]),
    FMISS=np.array([0.0]),
    SIGMA_OFF=np.array([0.1]),
    BCG_B_MASS=np.array([0.0]),
    radii=radii,
)
ds_true = np.asarray(truth['Signal'][0])

# ── synthetic cluster generator ─────────────────────────────────────────────
def synth_cluster(seed, n_gal=NGAL):
    """Return a (6, n_gal) bkg array with the convention used by
    single_cluster: rows (Σ_crit, e_t, e_x, W, R, M).  Sources are
    uniform in the cone — no cluster-member contamination — so the true
    per-bin boost is 1.0 in every bin."""
    rng = np.random.default_rng(seed)
    r   = RADIUS * np.sqrt(rng.uniform(0.0, 1.0, n_gal))   # uniform in disk
    z_src = rng.uniform(1.1 * Z_LENS + 0.2, 1.5, n_gal)

    DA_l  = float(DA(0, Z_LENS))
    DA_s  = np.array([float(DA(0, z)) for z in z_src])
    DA_ls = np.array([float(DA(Z_LENS, z)) for z in z_src])
    # Σ_crit in M_sun / pc² to match xlensing's NFW_Delta_Sigma output convention
    # (xlensing.data divides the natural M_sun / Mpc² value by 1e12).
    sigcrit = ((lightspeed**2) / (4.0 * np.pi * gravity)
               * DA_s / (DA_l * DA_ls)) / 1e12

    # Per-galaxy ΔΣ from analytical NFW evaluated at the source radius
    ds_g = xlensing.model.NFW_Delta_Sigma(
        M200m=np.array([M_TRUE]), C200m=np.array([C_TRUE]),
        Z=np.array([Z_LENS]),
        FMISS=np.array([0.0]), SIGMA_OFF=np.array([0.1]),
        BCG_B_MASS=np.array([0.0]),
        radii=r,
    )['Signal'][0]
    gamma_t = ds_g / sigcrit
    e_t = gamma_t + rng.normal(0.0, SIG_E, n_gal)
    e_x =          rng.normal(0.0, SIG_E, n_gal)

    bkg = np.array([
        sigcrit,
        e_t,
        e_x,
        np.ones(n_gal),
        r,
        np.zeros(n_gal),
    ])
    return bkg


# ── run single_cluster on each synthetic cluster ────────────────────────────
print(f'Synthesising {NCL} clusters with {NGAL} sources each '
      f'(M={M_TRUE:.0e}, c={C_TRUE}, z={Z_LENS})...')

sigmas_list, boosts_list, sigcov_list = [], [], []
all_bkgs = []
for ci in range(NCL):
    bkg = synth_cluster(seed=2026 + ci)
    all_bkgs.append(bkg)
    out = xlensing.data.single_cluster([bkg], BINS, Nboot=NBOOT)
    sig, boo, _, sigcov, _, _ = out
    sigmas_list.append(sig)
    boosts_list.append(boo)
    sigcov_list.append(sigcov)
    if ci == 0:
        # Per-bin galaxy counts of the first cluster, for context
        in_bin = (bkg[4][:, None] > BINS[:, 0]) & (bkg[4][:, None] < BINS[:, 1])
        print(f'  cluster 0 per-bin N_gal: {in_bin.sum(axis=0).tolist()}')

sigmas_arr = np.array(sigmas_list)     # (NCL, Nbin)
boosts_arr = np.array(boosts_list)     # (NCL, Nbin)
vars_d     = np.array([np.diag(c) for c in sigcov_list])

# Per-bin inverse-variance-weighted mean across clusters
sigs_avg     = np.full(Nbin, np.nan)
sigs_avg_err = np.full(Nbin, np.nan)
n_used       = np.zeros(Nbin, dtype=int)
for b in range(Nbin):
    sel = np.isfinite(sigmas_arr[:, b]) & np.isfinite(vars_d[:, b]) & (vars_d[:, b] > 0)
    if sel.sum() == 0:
        continue
    s = sigmas_arr[sel, b]
    v = vars_d[sel, b]
    w = 1.0 / v
    sigs_avg[b]     = float((s * w).sum() / w.sum())
    sigs_avg_err[b] = float(np.sqrt(1.0 / w.sum()))
    n_used[b]       = int(sel.sum())

# ── stacked_signal on the same clusters (galaxy-level estimator) ────────────
sig_stack, boo_stack, _, sigcov_stack, _, _ = xlensing.data.stacked_signal(
    all_bkgs, BINS, Nboot=NBOOT,
)
sig_stack_err = np.sqrt(np.diag(sigcov_stack))

# ── report ──────────────────────────────────────────────────────────────────
print()
print('Bin centres (Mpc)        :', np.round(radii, 3).tolist())
print('Truth ΔΣ                 :', np.round(ds_true, 1).tolist())
print('Stacked ΔΣ               :', np.round(sig_stack, 1).tolist())
print('Per-cluster avg ΔΣ       :', np.round(sigs_avg, 1).tolist())
print('Per-cluster avg / truth−1:',
      np.round((sigs_avg / ds_true - 1) * 100, 1).tolist(), '%')
print('Stacked / truth − 1      :',
      np.round((sig_stack / ds_true - 1) * 100, 1).tolist(), '%')
print('Mean per-cluster boost   :', np.round(boosts_arr.mean(axis=0), 3).tolist())
print('Stacked boost            :', np.round(boo_stack, 3).tolist())
print('N clusters used per bin  :', n_used.tolist())
print()

# Distribution of per-cluster boosts in the innermost bin
b0 = boosts_arr[:, 0]
print(f'Inner-bin per-cluster boost: '
      f'min={b0.min():.3f}  med={np.median(b0):.3f}  '
      f'max={b0.max():.3f}  zero-fraction={(b0 == 0).mean():.2%}')

# ── per-cluster diagnostic at the innermost bin ─────────────────────────────
# Compute actual N_inner (count of original galaxies inside the bin), the
# per-cluster reported sigma, and the per-cluster reported bootstrap σ.
inner_lo, inner_hi = BINS[0]
n_inner_per_cluster = np.array([
    int(((bkg[4] > inner_lo) & (bkg[4] < inner_hi)).sum())
    for bkg in all_bkgs
])
sig_inner = sigmas_arr[:, 0]
err_inner = np.sqrt(vars_d[:, 0])

# Sort by reported weight (1/var) so the heaviest contributors come first.
finite = np.isfinite(sig_inner) & np.isfinite(err_inner) & (err_inner > 0)
order  = np.argsort(-1.0 / err_inner[finite]**2)
idxs   = np.where(finite)[0][order]

print()
print('Innermost-bin per-cluster top-15 by inv-var weight (truth ≈ '
      f'{ds_true[0]:.1f}):')
print(f'  {"i":>3} {"N_in":>4} {"boost":>7} {"σ_reported":>11} '
      f'{"err_boot":>9} {"weight":>9}')
for k, i in enumerate(idxs[:15]):
    print(f'  {i:>3} {n_inner_per_cluster[i]:>4} {boosts_arr[i, 0]:>7.3f} '
          f'{sig_inner[i]:>11.2f} {err_inner[i]:>9.2f} '
          f'{1.0/err_inner[i]**2:>9.5f}')

# Empirical scatter of per-cluster sigmas — compare to mean reported err.
empirical_std = float(np.std(sig_inner[finite], ddof=1))
mean_reported_std = float(np.mean(err_inner[finite]))
print()
print('At the innermost bin across the sample of clusters:')
print(f'  empirical std of per-cluster ΔΣ   : {empirical_std:.1f}')
print(f'  mean of per-cluster bootstrap σ   : {mean_reported_std:.1f}')
print(f'  ratio (empirical / bootstrap)     : '
      f'{empirical_std / max(mean_reported_std, 1e-30):.2f}')
print('  → If ratio >> 1, the bootstrap is UNDER-estimating the true '
      'per-cluster variance and the inverse-variance average is '
      'over-weighting noisy low-N clusters.')

# Compare three averaging schemes
def weighted_mean(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan, np.nan
    v = values[valid]; w = weights[valid]
    m   = float((v * w).sum() / w.sum())
    se  = float(np.sqrt((w**2 * (v - m)**2).sum()) / w.sum())
    return m, se

w_invvar  = 1.0 / vars_d[:, 0]
w_ninner  = n_inner_per_cluster.astype(float)
w_uniform = np.ones(NCL)

m_iv, se_iv = weighted_mean(sig_inner, w_invvar)
m_nN, se_nN = weighted_mean(sig_inner, w_ninner)
m_un, se_un = weighted_mean(sig_inner, w_uniform)
print()
print('Innermost-bin estimators (truth = '
      f'{ds_true[0]:.1f}):')
print(f'  inv-var weighted     : {m_iv:7.2f}  '
      f'(bias vs truth: {(m_iv/ds_true[0] - 1) * 100:+.1f}%)')
print(f'  N_inner weighted     : {m_nN:7.2f}  '
      f'(bias vs truth: {(m_nN/ds_true[0] - 1) * 100:+.1f}%)')
print(f'  uniform weighted     : {m_un:7.2f}  '
      f'(bias vs truth: {(m_un/ds_true[0] - 1) * 100:+.1f}%)')
print(f'  stacked_signal       : {sig_stack[0]:7.2f}  '
      f'(bias vs truth: {(sig_stack[0]/ds_true[0] - 1) * 100:+.1f}%)')

# ── plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(radii, ds_true, 'k-',  lw=1.5, label='truth (analytical NFW)')
ax.errorbar(radii, sig_stack, yerr=sig_stack_err,
            fmt='s-', ms=6, alpha=0.85, label=f'stacked_signal  (N={NCL})')
ax.errorbar(radii, sigs_avg,  yerr=sigs_avg_err,
            fmt='o--', ms=6, alpha=0.85,
            label=f'inv-var avg of per-cluster ΔΣ')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$R$ [Mpc, proper]')
ax.set_ylabel(r'$\Delta\Sigma$ [$M_\odot/\mathrm{pc}^2$]')
ax.set_title(f'M={M_TRUE:.0e}, c={C_TRUE}, z={Z_LENS}, '
             f'{NGAL} sources/cluster')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')

ax = axes[1]
ax.axhline(0.0, color='k', ls='--', alpha=0.5)
ax.plot(radii, (sig_stack / ds_true - 1) * 100, 's-', ms=6,
        label='stacked / truth − 1', alpha=0.85)
ax.plot(radii, (sigs_avg / ds_true - 1) * 100, 'o--', ms=6,
        label='per-cluster avg / truth − 1', alpha=0.85)
ax.set_xscale('log')
ax.set_xlabel(r'$R$ [Mpc, proper]')
ax.set_ylabel('Residual vs truth  [%]')
ax.set_ylim(-60, 60)
ax.set_title(f'NCL={NCL}, NBOOT={NBOOT}')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f'Saved plot to {OUT_PNG}')
