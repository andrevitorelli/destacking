# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`destacking` investigates the joint (un-stacked) fitting of individual cluster weak-lensing ΔΣ profiles to recover scaling relations between cluster observables (magnitude gap, BCG offset) and NFW concentration. The core idea: instead of stacking signals and fitting a mean model, fit all N per-cluster profiles simultaneously under a shared hierarchical likelihood, gaining statistical power.

The package depends on `xlensing` (https://github.com/andrevitorelli/xlensing) for ΔΣ measurement and NFW modeling, plus `nautilus` for nested sampling.

Other dependencies: `numpy`, `scipy`, `astropy`, `multiprocess`, `colossus`, `cluster_toolkit`, `galsim`.

## Package state

Most Python modules are stubs. The implemented code is:

- **`destacking/cosmo.py`** — `critical_surface_density(z_l, z_s, cosmo)` using astropy angular diameter distances (returns unscaled Quantity; caller must `.to()` the desired units).
- **`destacking/relations.py`** — Two scaling relations:
  - `c_DuttonMaccio(z, m, c=1., h=1.)` — NFW concentration from Dutton & Maccio (2014), returns c200 for M200 in M☉.
  - `mass_lambda_McClintock18(Lambda, z)` — log10(M200) from richness λ (McClintock et al. 2018), pivot at λ=40, z=0.35.
- **`destacking/fitting.py`** — stub (`import numpy as np; import nautilus`).
- **`destacking/model.py`**, **`data.py`**, **`simulation.py`**, **`__init__.py`** — empty stubs.

## Method and prototype pipeline

The full working pipeline lives in `notebooks/prototype_destacking.ipynb`. The key steps:

1. **Simulate clusters** — `xlensing.sims.make_simple_random_cat()` + `xlensing.sims.apply_NFW_shear_region()` → per-cluster galaxy catalogs. Parallelized with `multiprocess.Pool`.

2. **Measure individual ΔΣ profiles** — `xlensing.data.lensfit_cluster_lensing(cluster, sources, radius)` per cluster, then `xlensing.data.single_cluster([bkg], bins_lims, Nboot)` → `(sigmas, sigmas_cov, xigmas, xigmas_cov)`.

3. **Persist** — `notebooks/data/single_sigmas_{n}.npy` and `single_sigmas_cov_{n}.npy` (plus xigmas variants) for N=500 clusters; `notebooks/cluster_params.npy` (columns: ra, dec, z, mass, conc); `notebooks/clusterbkgs.npy`.

4. **Joint likelihood** — `cholog_gaussian(data_vectors - models, cov_matrices)` sums Cholesky-based log-likelihoods over all clusters. The model for each cluster is `xlensing.model.NFW_Delta_Sigma(...)['NFW Signal']` evaluated at the cluster's redshift.

5. **Sampling** — `nautilus.Sampler(prior, loglike, n_live=1000, pool=15).run()`. Prior uses `nautilus.Prior`; parameters `logMhalo` (log10 M☉) and `conc` are sampled. Posterior extracted via `sampler.posterior()`.

## Bin layout convention

Radial bins are log-spaced with edges offset by half a bin width so that `radii` array values fall at bin centers:

```python
radii = np.logspace(-0.6, 0.6, 8)  # 8 radii in Mpc
N = len(radii)
bins_lims = np.logspace(
    np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    np.log10(radii[N-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2,
    N + 1)
bins_lims = np.array([[bins_lims[i], bins_lims[i+1]] for i in range(N)])
```

## Key result (from prototype)

Stacking 500 clusters gives σ(logM)≈0.06, σ(c)≈1.5. Joint (destacking) fitting of the same 500 individual profiles gives σ(logM)≈0.03, σ(c)≈1.1 — roughly 2× tighter constraints.
