# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`destacking` investigates the joint (un-stacked) fitting of individual-cluster weak-lensing ΔΣ profiles to recover scaling relations between cluster observables (magnitude gap, richness) and NFW mass/concentration — fitting all N per-cluster profiles simultaneously under one shared hierarchical likelihood, instead of stacking signals and fitting a mean model, for tighter constraints.

The package depends on `xlensing` (https://github.com/andrevitorelli/xlensing, not on PyPI — install separately) for ΔΣ measurement and NFW modeling, plus `nautilus-sampler` for nested sampling. See `pyproject.toml` for the full pinned dependency list.

**For anything beyond a quick orientation, read `docs/MANUAL.md` first** — it documents installation, core concepts (gamma/gamma_cm, boost correction, miscentering, the stack-membership/boost(R) pipeline), config-key references, output formats, and known caveats, all sourced from the docstrings/comments below. This file is a short map, not a substitute.

## Package state (current, `develop` branch)

Not stubs — this is an actively developed, ~8,000-line pipeline across `destacking/` + `scripts/`. Module-by-module:

- **`destacking/cosmo.py`** — `critical_surface_density(z_l, z_s, cosmo)`.
- **`destacking/relations.py`** — `c_DuttonMaccio` (Dutton & Maccio 2014 c-M), `mass_lambda_McClintock18` (McClintock+18 M-λ), HOD/luminosity-function sampling helpers used by the truth-catalog generator.
- **`destacking/simulation.py`** — truth-catalog generation, synthetic-profile generation (`make_test_data`), and the full survey-simulation pipeline (`SimulationConfig`, `simulate_lensing`, `_process_cluster`) moved here from the original `scripts/gamma_scan.py`: per-cluster triaxial/spherical NFW shearing, DES-Y3-like source density/n(z)/metacal-response injection, optional miscentering, `bootstrap_signal` measurement.
- **`destacking/model.py`** — forward models: `forward_model` (magnitude-gap-only), `forward_model_mc`/`forward_model_combined` (power-law M-c relation, optionally + magnitude-gap term + amplitude/miscentering/BCG terms), `forward_model_diemer` (Diemer & Joyce 2019 M-c baseline + magnitude-gap term). `_nfw_profiles` wraps `xlensing.model.NFW_Delta_Sigma`.
- **`destacking/fitting.py`** — `cholog_gaussian` (per-cluster NaN-masked Cholesky likelihood), `run_sampler*` family driving `nautilus` for each forward model above.
- **`destacking/data.py`** — caller-side background-source selection (`select_background`), and `assign_stack_bins` — the one canonical z×richness×Δm binning shared by `wazp_stacked_analysis.py` (stacks + boost fits) and `prepare_real_data.py` (per-cluster stack-membership labels), so a boost(R) fit and its application can never disagree on which cluster belongs to which stack.
- **`destacking/io.py`** — simulation checkpoint read/write (`save_simulation_checkpoint`/`load_simulation_checkpoint`) and notebook-compatible sim output.
- **`destacking/plotting.py`** — scan-result and corner-plot helpers.
- **`destacking/utils.py`** — small shared helpers (deterministic seeding, duration formatting).

Two main scripts, kept directly in `scripts/` (YAML-configured, `python scripts/foo.py config.yaml`):
- **`scripts/gamma_inference.py`** — the main simulation+inference driver (see its module docstring for the full config surface: phase toggles, radial grids, `fit_params` per-parameter fit control, boost correction via `boosts:`, miscentering).
- **`scripts/gamma_inference_diemer.py`** — its Diemer19-baseline sibling.

Everything else that supports those two (data prep, cross-checks, plotting) lives in `scripts/auxiliary_scripts/`:
- **`prepare_real_data.py`** — builds a `gamma_inference.py`-compatible `data_folder` from real WaZP + DES-Y3-like metacal catalogs (boost-free measurement by design; see MANUAL §3).
- **`wazp_stacked_analysis.py`** — independent stacked-analysis cross-check (stacks, boost fits, per-stack NFW fits, mass-richness/mass-concentration power-law fits).
- **`make_corner_plots.py`** — renders a corner plot for every per-bootstrap fit in one or more `gamma_inference.py` run directories.
- **`gen_sim_data_for_notebooks.py`** — generates notebook-compatible simulated data for the `SIMS=True` toggle in `prototype_destacking.ipynb`/`wazp_lensing_signals.ipynb`/`wazp_mass_richness_by_delta_m.ipynb`.
- **`sphere_vs_triaxial.py`**, **`add_stacked_comparison.py`**, **`deficit_vs_concentration.py`** — the triaxiality-vs-sphericity stacking-bias diagnostic suite (the latter two extend the first's output).

- **`scripts/archive/`** — non-functional debug scripts/configs, kept for history only; see `scripts/archive/README.md`. Do not use as a starting point for new work.

Real and simulated data/outputs live outside the repo, under `~/destacking_data/{sims,real_data,output}/` (moved out 2026-07-27 for disk-space reasons) — see path conventions in `gamma_inference.py`/`prepare_real_data.py`.

## Quickstart (no external data needed)

```
python scripts/gamma_inference.py scripts/gamma_inference_config_example.yaml
```
Fully self-contained: simulates its own clusters and fits them. This is the right first thing to run to confirm an environment is set up correctly, or for anyone without access to the real WaZP/DES-Y3 catalogs. See `docs/MANUAL.md` §4 for this and the real-data/stacked-cross-check walkthroughs.

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

## Key result (from the original prototype)

Stacking 500 clusters gives σ(logM)≈0.06, σ(c)≈1.5. Joint (destacking) fitting of the same 500 individual profiles gives σ(logM)≈0.03, σ(c)≈1.1 — roughly 2× tighter constraints. The current real-data result (WaZP+DES-Y3, boost- and floor-corrected) is `alpha_cm=+0.076 (+0.075/-0.071)`; see `reports/2026-07-26_boost_factors_variance_floor_alpha_cm.tex`.
