#!/usr/bin/env python3
"""Side-by-side histograms of per-cluster ΔΣ at the innermost bin
for the full WaZP-mirror simulation and the real WaZP data, to expose
the extreme outliers that dominate the unweighted-mean diagnostic of
wazp_lensing_signals.ipynb §9."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('full_wazp_sim_results.pkl', 'rb') as fh:
    sim = pickle.load(fh)
sim_sig0 = sim['sigmas'][:, 0]
sim_sig0 = sim_sig0[np.isfinite(sim_sig0)]

with open('/home/andre/github/destacking/notebooks/data/wazp_lensing_signals.pkl',
          'rb') as fh:
    real = pickle.load(fh)
real_sig0 = np.array([r['sigmas'][0] for r in real['results']
                       if r.get('sigmas') is not None])
real_sig0 = real_sig0[np.isfinite(real_sig0)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Both share the same x range so the comparison is direct.  Use symmetric log
# bin spacing so the outlier tail is visible.
xlim = 5e4
bins_lin = np.linspace(-xlim, xlim, 121)
for ax, data, label, color in (
    (axes[0], sim_sig0,  f'Simulation  (N={len(sim_sig0):,})',  'steelblue'),
    (axes[1], real_sig0, f'Real WaZP   (N={len(real_sig0):,})', 'crimson'),
):
    ax.hist(np.clip(data, -xlim, xlim), bins=bins_lin, color=color, alpha=0.85,
            edgecolor='none')
    med = np.median(data); std = np.std(data)
    n_outl = np.sum(np.abs(data) > 1e4)
    ax.axvline(med, color='k', ls='--', lw=1, label=f'median = {med:.0f}')
    ax.set_title(label, fontsize=11)
    ax.set_xlabel(r'per-cluster $\Delta\Sigma$ at $R = 0.158$ Mpc '
                  r'[$M_\odot/\mathrm{pc}^2$]')
    ax.set_ylabel('# clusters per bin')
    ax.set_yscale('log')
    ax.set_xlim(-xlim, xlim)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.text(0.02, 0.95,
            f'std = {std:,.0f}\n|ΔΣ|>10⁴: {n_outl:,} ({n_outl/len(data)*100:.2f}%)\n'
            f'max |ΔΣ|: {np.max(np.abs(data)):.0f}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Per-cluster ΔΣ at the innermost bin: simulation vs WaZP data',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('diagnose_outlier_histograms.png', dpi=120, bbox_inches='tight')
print('saved diagnose_outlier_histograms.png')
