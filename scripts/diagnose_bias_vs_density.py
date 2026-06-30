#!/usr/bin/env python3
"""
Sweep the per-cluster source count NGAL and show the inner-bin bias of
the inv-var-weighted per-cluster average turn on / off as N_inner drops
into the few-galaxy regime.

The prototype destacking notebook
(`notebooks/prototype_destacking.ipynb`) ran in the high-NGAL regime
(density 30 gal/arcmin² in an 8 Mpc cone at z=0.3, with shape noise
divided by sqrt(100) ⇒ effectively many-thousand high-S/N galaxies per
cluster).  In that regime the per-cluster bootstrap variance accurately
reflects the per-cluster scatter and no single cluster dominates the
average.

The wazp_lensing_signals notebook runs in the opposite regime: ~1500
sources per cone at z=0.4 with full DES Y3 shape noise, giving 1–3
galaxies in the innermost radial bin per cluster.  In that regime the
bootstrap variance is much smaller than the true (shape-noise-dominated)
per-cluster variance and the inv-var average collapses.

This script reproduces the transition by varying NGAL across the two
regimes at fixed cluster parameters.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import xlensing.data
import xlensing.model
from xlensing.cosmo import DA, lightspeed, gravity

NCL    = 60
NGAL_SWEEP = [500, 1500, 5000, 20000, 80000]   # per-cluster source count
M_TRUE = 2.0e14
C_TRUE = 5.0
Z_LENS = 0.4
RADIUS = 5.0
NBOOT  = 200
SIG_E  = 0.27

OUT_PNG = 'diagnose_bias_vs_density.png'

radii = np.logspace(-0.8, 0.6, 8)
Nbin  = len(radii)
edges = np.logspace(
    np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    Nbin + 1,
)
BINS = np.array([[edges[i], edges[i + 1]] for i in range(Nbin)])

truth = xlensing.model.NFW_Delta_Sigma(
    M200m=np.array([M_TRUE]), C200m=np.array([C_TRUE]),
    Z=np.array([Z_LENS]),
    FMISS=np.array([0.0]), SIGMA_OFF=np.array([0.1]),
    BCG_B_MASS=np.array([0.0]),
    radii=radii,
)
ds_true = np.asarray(truth['Signal'][0])


def synth_cluster(seed, n_gal, sig_e=SIG_E):
    rng = np.random.default_rng(seed)
    r   = RADIUS * np.sqrt(rng.uniform(0.0, 1.0, n_gal))
    z_src = rng.uniform(1.1 * Z_LENS + 0.2, 1.5, n_gal)
    DA_l  = float(DA(0, Z_LENS))
    DA_s  = np.array([float(DA(0, z)) for z in z_src])
    DA_ls = np.array([float(DA(Z_LENS, z)) for z in z_src])
    sigcrit = ((lightspeed**2) / (4.0 * np.pi * gravity)
               * DA_s / (DA_l * DA_ls)) / 1e12
    ds_g = xlensing.model.NFW_Delta_Sigma(
        M200m=np.array([M_TRUE]), C200m=np.array([C_TRUE]),
        Z=np.array([Z_LENS]),
        FMISS=np.array([0.0]), SIGMA_OFF=np.array([0.1]),
        BCG_B_MASS=np.array([0.0]),
        radii=r,
    )['Signal'][0]
    gamma_t = ds_g / sigcrit
    e_t = gamma_t + rng.normal(0.0, sig_e, n_gal)
    e_x =          rng.normal(0.0, sig_e, n_gal)
    return np.array([sigcrit, e_t, e_x, np.ones(n_gal), r, np.zeros(n_gal)])


def run_one(n_gal):
    sigmas, vars_d, bkgs_all = [], [], []
    n_inner_arr = []
    for ci in range(NCL):
        bkg = synth_cluster(2026 + ci, n_gal)
        out = xlensing.data.single_cluster([bkg], BINS, Nboot=NBOOT)
        sig, _, _, sigcov, _, _ = out
        sigmas.append(sig)
        vars_d.append(np.diag(sigcov))
        bkgs_all.append(bkg)
        n_inner_arr.append(int(((bkg[4] > BINS[0, 0])
                                & (bkg[4] < BINS[0, 1])).sum()))
    sigmas = np.asarray(sigmas)
    vars_d = np.asarray(vars_d)

    # inv-var average
    avg_iv = np.full(Nbin, np.nan)
    for b in range(Nbin):
        ok = np.isfinite(sigmas[:, b]) & np.isfinite(vars_d[:, b]) & (vars_d[:, b] > 0)
        if not ok.any():
            continue
        w = 1.0 / vars_d[ok, b]
        avg_iv[b] = float((sigmas[ok, b] * w).sum() / w.sum())

    # stacked
    sig_stack, _, _, sigcov_stack, _, _ = xlensing.data.stacked_signal(
        bkgs_all, BINS, Nboot=NBOOT,
    )

    return {
        'avg_iv':       avg_iv,
        'sig_stack':    np.asarray(sig_stack),
        'mean_n_inner': float(np.mean(n_inner_arr)),
        'sigmas':       sigmas,
        'vars_d':       vars_d,
    }


# ── run the sweep ───────────────────────────────────────────────────────────
print(f'Sweeping NGAL across {NGAL_SWEEP} at fixed NCL={NCL}, '
      f'sig_e={SIG_E}, cone={RADIUS}Mpc, z={Z_LENS}')
print()
out_by_n = {n: run_one(n) for n in NGAL_SWEEP}

# ── plot: per-bin bias of inv-var average vs stacked, four regimes ──────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes_flat = axes.flatten()
for ax, n in zip(axes_flat, NGAL_SWEEP[:4]):
    r = out_by_n[n]
    ax.axhline(0.0, color='k', ls='--', alpha=0.5, lw=1)
    ax.plot(radii, (r['avg_iv']    / ds_true - 1) * 100, 'o--', ms=6, alpha=0.85,
            label=r'per-cluster inv-var avg / truth − 1')
    ax.plot(radii, (r['sig_stack'] / ds_true - 1) * 100, 's-', ms=6, alpha=0.85,
            label=r'stacked_signal / truth − 1')
    ax.set_xscale('log')
    ax.set_xlabel(r'$R$ [Mpc, proper]')
    ax.set_ylabel('Residual vs truth [%]')
    ax.set_ylim(-110, 110)
    ax.set_title(
        rf'NGAL={n:,}  ⟨N_inner⟩={r["mean_n_inner"]:.1f}  per cluster',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='both')

plt.suptitle(
    'Per-cluster inv-var average diverges from truth at low N_inner; '
    'stacked_signal stays consistent across regimes',
    fontsize=12, y=1.00,
)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f'Saved plot to {OUT_PNG}')

# ── summary table ───────────────────────────────────────────────────────────
print()
print(f'{"NGAL":>7}  {"⟨N_in⟩":>7}  '
      f'{"bias_innermost(iv)":>20}  {"bias_innermost(stack)":>22}')
for n in NGAL_SWEEP:
    r = out_by_n[n]
    iv_bias = (r['avg_iv'][0]    / ds_true[0] - 1) * 100
    sk_bias = (r['sig_stack'][0] / ds_true[0] - 1) * 100
    print(f'{n:>7,}  {r["mean_n_inner"]:>7.2f}  '
          f'{iv_bias:>19.1f}%  {sk_bias:>21.1f}%')

# ── variance ratio diagnostic: empirical std / mean bootstrap σ vs NGAL ─────
print()
print('Variance underestimation factor (empirical std / mean bootstrap σ) '
      'at innermost bin:')
print(f'{"NGAL":>7}  {"emp std":>10}  {"boot σ":>10}  {"ratio":>8}')
for n in NGAL_SWEEP:
    r = out_by_n[n]
    s0 = r['sigmas'][:, 0]
    v0 = r['vars_d'][:, 0]
    ok = np.isfinite(s0) & np.isfinite(v0) & (v0 > 0)
    emp  = float(np.std(s0[ok], ddof=1)) if ok.sum() > 1 else float('nan')
    boot = float(np.mean(np.sqrt(v0[ok])))    if ok.any() else float('nan')
    print(f'{n:>7,}  {emp:>10.2f}  {boot:>10.2f}  '
          f'{emp / max(boot, 1e-30):>8.2f}')
