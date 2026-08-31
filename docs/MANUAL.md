# destacking — User Manual

This manual is assembled from the docstrings and inline commentary already
in the codebase (`destacking/*.py`, `scripts/*.py`, YAML config templates).
It does not introduce new behavior claims — every statement here traces to
a specific comment, docstring, or code path cited inline. Where the code
and this manual disagree, the code is authoritative; please open an issue.

## 1. What this package does

`destacking` investigates **joint (un-stacked) fitting of individual-cluster
weak-lensing ΔΣ profiles**, instead of the traditional approach of stacking
many clusters' profiles and fitting one mean model. The core idea: fit all
*N* per-cluster profiles simultaneously under a shared hierarchical
likelihood (one scaling-relation model, evaluated separately at each
cluster's own observables), which recovers scaling-relation parameters
(a mass–concentration relation, and its dependence on a magnitude-gap
observable) with tighter constraints than stacking the same data (a
prototype comparison in `CLAUDE.md` found stacking 500 clusters gives
σ(logM)≈0.06, σ(c)≈1.5, vs. σ(logM)≈0.03, σ(c)≈1.1 for the joint fit of the
same clusters — roughly 2× tighter).

Scientifically, the package targets the **mass–concentration (c–M)
relation** for galaxy clusters, and in particular whether it depends on the
magnitude gap Δm (difference in r-band magnitude between the BCG and
brightest satellite within a fixed radius) — a proxy for cluster assembly
history/dynamical state. It supports both:
- **Simulated clusters**, end-to-end, to validate the method and quantify
  biases (triaxiality, miscentering, shot noise, source density) under
  known truth.
- **Real data**, specifically WaZP cluster catalogs cross-matched against a
  DES-Y3-like metacalibration shear catalog.

The package depends on `xlensing` (https://github.com/andrevitorelli/xlensing)
for ΔΣ measurement (galaxy-level bootstrap estimators) and NFW modeling,
and `nautilus` for nested sampling.

## 2. Installation

### Declared in `pyproject.toml`
```toml
[project]
requires-python = ">=3.9"
dependencies = ["numpy", "scipy", "astropy"]

[project.optional-dependencies]
fitting = ["nautilus-sampler"]
```
Install with `pip install -e .` (editable) or `pip install .` from the repo
root. `pip install -e .[fitting]` adds `nautilus-sampler`.

### Not declared, but required by the code actually run

A grep of every `import`/`from` statement across `destacking/` and
`scripts/` turns up dependencies the packaging does not list:
`xlensing` (external, install via
`pip install git+https://github.com/andrevitorelli/xlensing` — noted only
as a comment in `pyproject.toml`, not a real dependency spec), `colossus`
(Diemer & Joyce 2019 concentration model, Tinker+08 halo mass function),
`cluster_toolkit` (boost-factor fitting), `corner` (posterior corner
plots), `h5py` (metacal HDF5 catalogs), `multiprocess` (parallel
simulation/measurement), `ngmix` (used by `gamma_recovery_scan.py`),
`scikit-learn` (`sklearn.neighbors.BallTree` in `wazp_stacked_analysis.py`),
`tqdm`, `PyYAML` (all script configs are YAML). See §9 for this as a 1.0
punch-list item — a fresh `pip install -e .` today does **not** get you a
runnable pipeline; you additionally need at minimum:
```
pip install git+https://github.com/andrevitorelli/xlensing nautilus-sampler \
            colossus cluster_toolkit corner h5py multiprocess ngmix \
            scikit-learn tqdm pyyaml
```

## 3. Core concepts

**Destacking vs. stacking.** Traditional cluster weak lensing stacks many
clusters' shear signal into one mean ΔΣ(R) profile per bin (e.g. per
richness bin) and fits one model to it. Destacking instead evaluates the
model separately for every individual cluster (using its own z, richness,
Δm) and sums per-cluster Cholesky log-likelihoods
(`destacking.fitting.cholog_gaussian`) into one joint likelihood — no
information is thrown away by binning.

**gamma / gamma_cm.** The magnitude-gap slope of the mass–concentration
relation: `d log10(c) / d log10(Δm/Δm_ref)`. This is the primary science
target parameter across most of the fitting code (`forward_model`,
`forward_model_combined`, `forward_model_diemer`). In the simulation
pipeline, `gamma_true` is the truth value injected into the simulated
concentrations; `gamma_est`/`gamma_ensemble_med` is what the sampler
recovers. **For real data there is no truth** — `gamma_true` in
`prepare_real_data.py`'s output is a pure directory-naming/metadata label,
not a measured or true quantity (see its module docstring).

**alpha_cm / beta_cm / c0.** In `forward_model_combined`/
`forward_model_mc`, the mass–concentration relation is parametrized as a
power law: `log10(c) = log10(c0) + alpha_cm*(log10(M) - M_pivot_log) +
beta_cm*log10((1+z)/(1+z_ref)) + gamma_cm*log10(Δm/Δm_ref)`. `alpha_cm` is
the mass slope (Dutton & Maccio 2014 predicts ≈ -0.09 at cluster masses);
`beta_cm` is the redshift-evolution slope. `forward_model_diemer` instead
takes the Diemer & Joyce (2019) M–c relation directly from theory
(non-power-law, precomputed on a (z, log M) grid and interpolated — see
`destacking/model.py`'s `diemer19_c200m`) and only fits the magnitude-gap
slope `gamma` on top of it.

**Boost correction.** `xlensing.data.single_cluster` applies a boost
correction (B = N_obs/N_random, correcting for cluster-member/foreground
contamination of the "background" sample) that the boost-free NFW model
these scripts fit does not include. Two different, deliberately
inconsistent choices are made in two different pipelines:
- The **joint per-cluster fit** (`prepare_real_data.py` →
  `gamma_inference.py`) uses `xlensing.data.bootstrap_signal`, which
  applies **no boost**, matching its boost-free forward model exactly. An
  *optional*, separately-fit boost(R) correction can be layered on top via
  the `boosts:` config key (see §5, §6) — without it, the joint fit has no
  contamination correction at all.
- The **stacked cross-check** (`wazp_stacked_analysis.py`) uses
  `xlensing.data.stacked_signal`, which **boost-corrects internally**
  (signal and covariance both) — do not re-apply the boost to its output.

**Miscentering (FMISS / f_mis, SIGMA_OFF / sigma_off).** A fraction of
clusters `1 - f_cen` are offset from their true center by a Rayleigh-
distributed magnitude with scale `sigma_off`. The forward model's ΔΣ is a
mixture `(1-FMISS)·NFW_centered + FMISS·NFW_miscentered`. The **simulation
truth** (`f_cen`, `sigma_off_mpc_h`) and the **fit model's assumption**
(`fit_fmiss`/`f_mis`, `fit_sigma_off_mpc_h`/`sigma_off`) are independent
config knobs — a known historical bug (fixed 2026-07-17, see
`gamma_inference.py`) had the fit silently assuming 36% miscentering even
for perfectly centered simulations, biasing recovered gamma by +10%. They
now default to matching (fit assumes truth) but can be deliberately set
apart, e.g. to test model misspecification.

**Stack membership / boost(R) pipeline.** `destacking.data.assign_stack_bins`
implements one canonical binning (3 redshift × 4 richness × 2 magnitude-gap
median-split bins = up to 24 "stacks") shared by two scripts so they never
disagree: `wazp_stacked_analysis.py` fits a per-stack boost(R) power law
(`cluster_toolkit.boostfactors.boost_powerlaw_at_R`, saved to
`wazp_boost_fits.pkl`) from `xlensing.data.stacked_signal`'s stage-2
output; `prepare_real_data.py` labels each individual cluster with its
`stack_name`, so `gamma_inference.py`'s `_apply_boost_correction` can look
up and apply that cluster's stack's fitted boost(R) to its own ΔΣ vector
and covariance before the joint fit (`boosts:` config key). This is a
deterministic point-estimate correction applied once, with no boost-
uncertainty propagation.

**Simulation vs. real-data mode.** `gamma_inference.py` has two
independently toggleable phases (`simulation: true/false`,
`inference: true/false`): a full run simulates its own data and fits it in
one go; `simulation: false` requires `data_folder` pointing at
pre-existing checkpoints (either from a prior simulation-only run, or from
`prepare_real_data.py`'s real-data output) and fits those instead.

## 4. End-to-end walkthroughs

### (a) Pure simulation → inference

```
python scripts/gamma_inference.py scripts/gamma_inference_config_example.yaml
```
This is fully self-contained — no external/real data needed. Per the
config template and `gamma_inference.py`'s module docstring, for each
`gamma_true` in `linspace(*gamma_range, n_gamma)` it:
1. Generates a truth catalog and simulates `n_clusters` triaxial (or
   spherical, `triaxiality: false`) clusters via
   `destacking.simulation.simulate_lensing` (parallelized,
   `multiprocess.Pool`), injecting DES-Y3-like source density/shape
   noise/n(z), applying (optionally) miscentering.
2. Measures each cluster's ΔΣ via `xlensing.data.bootstrap_signal` (no
   boost — see §3) and checkpoints to
   `<RUN_DIR>/simulation_g{k:02d}_gamma{v:+.1f}/` (`simulated_data.npz`,
   `truth_catalog.npz`, `cluster_obs.npz`, `ols.json` — written by
   `destacking.io.save_simulation_checkpoint`).
3. Runs `N_BOOTSTRAP` nautilus fits (cluster-level bootstrap resamples, or
   the full sample once if `bootstrap: false`) via
   `destacking.fitting.run_sampler_combined`, combining posteriors into an
   equal-weight mixture; extracts `gamma_ensemble_med` and a 16–84
   `sigma_gamma`.
4. Writes `gamma_scan_results.png`, `scan_summary.json`, `progress.json`
   under `RUN_DIR` (default
   `~/destacking_data/output/run_<timestamp>`, overridable via `RUN_DIR`
   env var or resumed by pointing at an existing one — **every stage is
   checkpointed and safe to interrupt/re-run**, per the module docstring's
   "Resumability" section).

A quick smoke test: shrink `n_clusters` (e.g. 200–1000), `n_gamma` to 1–3,
and `timeout` to something small (this scales with sample size — too small
a budget truncates the nautilus sampler and produces a collapsed,
near-zero-width posterior, per the `TIMEOUT` docstring note).

### (b) Real WaZP + DES-Y3-like data

Two-stage pipeline (this needs proprietary survey catalogs an outside
reader will not have — see §9):
```
python scripts/prepare_real_data.py my_real_data_config.yaml   # measurement
python scripts/gamma_inference.py wazp_desy3_inference_boost_floor6.yaml  # fit
```
`prepare_real_data.py` (see its full module docstring and
`scripts/real_data_config_example.yaml`) builds (or loads a pre-built)
compact WaZP cluster catalog, pre-filters and caches a metacalibration
source catalog (HDF5, streamed in `chunk_read`-sized chunks), then for each
cluster: applies caller-side background selection
(`destacking.data.select_background`, "margin" strategy = the old
xlensing hardcoded default), measures ΔΣ via `bootstrap_signal` with an
optional shot-noise variance floor (`sigma_e`/`floor_rmax` — see §7 caveat
on why this matters), and writes a `gamma_inference.py`-compatible
checkpoint including each cluster's `stack_name` for the optional boost
correction. Both expensive stages are resumable/cached (`use_cache`,
per-cluster `.partial.npz` checkpoints every `checkpoint_every` clusters).

Then point `gamma_inference.py` at it with `simulation: false`,
`data_folder: <out_dir>`, `gamma_range: [0.0, 0.0]`, `n_gamma: 1` (there is
no truth gamma for real data — every "bias" column in the output compares
against this decorative label, not a real truth, so ignore it), and the
`fit_*_radius`/`fit_n_radii` keys set to match the radial grid
`prepare_real_data.py` measured on. Add `boosts: <path to
wazp_boost_fits.pkl>` to apply the stack-level boost correction (§3, §6).

### (c) Stacked cross-check

```
python scripts/wazp_stacked_analysis.py
```
Re-analyzes the same real catalogs the traditional way: bins clusters via
`assign_stack_bins`, stacks each bin with `xlensing.data.stacked_signal`
(boost-corrected internally), fits per-stack NFW+miscentering, then fits
mass–richness and mass–concentration power laws per magnitude-gap group by
bootstrap regression over stacks. Six stages, each checkpointed to
`data/wazp_stacked_*.pkl` / `data/wazp_boost_fits.pkl`, gated by
`USE_CACHE`. `notebooks/wazp_stacked_plots.ipynb` regenerates every plot
from the saved pickles with no recomputation. This is the independent
methodology the joint per-cluster fit's results were cross-checked against
(see project history in `reports/2026-07-26_boost_factors_variance_floor_alpha_cm.tex`).

## 5. Config file reference

All scripts read a YAML file (`python script.py config.yaml`) with a
uniform precedence: **YAML value > environment variable (its UPPER_CASE
name) > built-in default** (`gamma_inference.py`'s `_cfg` helper, mirrored
in `prepare_real_data.py`). Every key below is enumerated directly from
`_cfg(...)` call sites in the scripts, not inferred.

### `gamma_inference.py` (and `gamma_inference_diemer.py`, same shape)

| Key | Default | Meaning |
|---|---|---|
| `simulation` | `true` | Run the simulate+measure phase |
| `inference` | `true` | Run the bootstrap-fit phase |
| `data_folder` | — | Required when `simulation: false`; directory of checkpoints |
| `n_gamma` | 51 | Number of gamma_true grid points |
| `gamma_range` | `[-1.0, 1.0]` | `np.linspace` endpoints |
| `n_clusters` | 15000 | Clusters per simulated gamma |
| `n_bootstrap` | 1 | Bootstrap fit draws per gamma (forced to 1 if `bootstrap: false`) |
| `n_workers` | 15 | Pool size |
| `n_cov_bootstrap` | 200 | Galaxy-level bootstrap draws per cluster (data vector + cov) |
| `triaxiality` | `true` | `false` → pure spherical NFW |
| `miscentering` | `true` | `false` → all clusters perfectly centered (simulation truth) |
| `bootstrap` | `false` | Enable cluster-level bootstrap resampling in the fit |
| `f_cen` | 0.64 | Truth: correctly-centered fraction |
| `sigma_off_mpc_h` | 0.32 | Truth: Rayleigh offset scale [Mpc/h] |
| `density` | 7.5 | Source density [gal/arcmin²] (7.5 = DES-Y3-like; see §7 caveat) |
| `shape_noise` | 0.27 | Per-component intrinsic ellipticity RMS |
| `smail_alpha`/`smail_beta`/`smail_z0` | 2.0/1.5/0.4 | Source n(z) Smail-distribution params |
| `sim_log_inner_radius`/`sim_log_outer_radius`/`sim_n_radii` | -0.8/0.6/8 | Simulated radial grid `np.logspace(...)` [Mpc] |
| `fit_log_inner_radius`/`fit_log_outer_radius`/`fit_n_radii` | (= sim grid) | Fit radial grid; **required** in inference-only mode to describe `data_folder`'s grid |
| `r_min_fit` | 0.5 | Minimum radius [Mpc] entering the likelihood (bins below are measured but excluded) |
| `wild_outlier_mask` | `false` | Mask individual (cluster, bin) entries outside a value range instead of a hard radius cut — **found statistically invalid with a one-sided range in project history; if ever re-enabled, the range must be two-sided** (see §7) |
| `wild_outlier_rmax`/`wild_outlier_min`/`wild_outlier_max` | 0.5/0.0/10000.0 | Range/scope for the above |
| `boosts` | `None` | Path to a `wazp_stacked_analysis.py` Stage-2b boost_fits pickle; applies per-cluster boost(R) by `stack_name` lookup (see §3) |
| `z_cluster_range` | `[0.10, 0.60]` | Cluster redshift range (simulation) |
| `f_mis`/`sigma_off` (via `fit_params`, see below) | — | Fit-model miscentering assumption |
| `fit_params` | — | **Preferred**: full per-parameter fit control, see below |
| `fit_miscentering`, `fit_fmiss`, `fit_sigma_off_mpc_h` | `false`/derived from truth | **Legacy** alternative to `fit_params` for f_mis/sigma_off only; mutually exclusive with `fit_params` |
| `timeout` | 600 | Wall-time budget [s] per nautilus fit — scale with `n_clusters` or fits collapse |
| `sim_prefix` | derived | Override for notebook-compat sim file prefix |

**`fit_params` block** (preferred over the legacy keys): one entry per
model parameter, either `{fit: true, prior: gaussian, loc: ..., scale: ...}`
(optionally `lo`/`hi` for truncation — `f_mis`/`sigma_off` are truncated to
their physical range `[0,1]`/`[0,∞)` by default even without explicit
`lo`/`hi`), `{fit: true, prior: flat, lo: ..., hi: ...}`, or
`{fit: false, value: ...}`. All nine names must be declared:
`c0, alpha_cm, beta_cm, gamma_cm, log_M_norm, alpha_mr, beta_mr, f_mis,
sigma_off`. `gamma_cm` must have `fit: true` — it is the science target.

Environment-variable equivalents exist for most keys (their UPPER_CASE
form, e.g. `N_CLUSTERS`, `TIMEOUT`, `RUN_DIR`) — see the module docstring's
"Configuration via environment variables" section.

### `prepare_real_data.py`

See `scripts/real_data_config_example.yaml` (reproduced with inline
comments) for the full annotated list: cluster-catalog source
(`cluster_catalog_file` or `clusters_file`+`members_file` to build one),
selection cuts (`richness_min`, `z_max`, `require_bcg`,
`mag_gap_r_cut_mpc`, `delta_m_min`), source catalog (`metacal_file`,
`dnf_file`, `filtered_cache`, `chunk_read`), measurement
(`radius_mpc`, `background_strategy`/`background_strategy_kwargs`,
`log_inner_radius`/`log_outer_radius`/`n_radii` — **must match** the
`fit_*` grid later given to `gamma_inference.py`, `n_cov_bootstrap`,
`min_sources`, `sigma_e`/`floor_rmax` for the variance floor — see §7),
execution (`n_workers`, `checkpoint_every`, `use_cache`), and output
(`out_dir`, `g_idx`, `gamma_true` — decorative label only).

## 6. Output / checkpoint format

`destacking.io.save_simulation_checkpoint` / `load_simulation_checkpoint`
define the on-disk layout every downstream script expects, one directory
per gamma point (`simulation_g{k:02d}_gamma{v:+.1f}/`):
- `simulated_data.npz` — `data_vectors` (N, n_bins), `cov_matrices`
  (N, n_bins, n_bins). NaN entries mark bins with no valid measurement for
  that cluster (dropped per-cluster, per-bin, by
  `cholog_gaussian`'s masking — not treated as zero).
- `truth_catalog.npz` — `z`, `M_halo`, `concentration`, `delta_m`. For real
  data, `M_halo`/`concentration` are all-NaN placeholders (genuinely
  unknown, never read by the inference path) kept only for schema
  compatibility.
- `cluster_obs.npz` — `z`, `Lambda`, `delta_m`, and (real-data checkpoints
  only) `stack_name` for the optional boost correction.
- `ols.json` — OLS truth coefficients for simulated data (`{}` for real
  data — no truth scaling relation exists).

Per-bootstrap fit output lives under
`<subdir>/bootstrap_{b:03d}/summary.json` + `gamma_posterior.npz`
(full posterior, all `param_order` columns) + `posterior_corner.png`.
Per-gamma combined results: `ensemble_summary.json`,
`posteriors_overlay.png`, `combined_corner.png`. Scan-level:
`gamma_scan_results.png`, `scan_summary.json`, `progress.json` (live,
updated after each gamma — useful for monitoring a long-running scan).

`nautilus.Sampler.posterior()` gives raw posterior samples/weights; the
attached `sampler.param_order` is the definitive column order (see
`run_sampler_combined`'s docstring) — always read parameter identity from
this, not by position convention, since which parameters are sampled vs.
fixed varies by config.

## 7. Known caveats (from code comments, not guesses)

- **DES-Y3-like source density (7.5 gal/arcmin²) is information-poor for
  concentration.** Quantified in project history (see
  `wazp_desy3_root_cause_hypotheses` investigation, referenced in
  `gamma_inference.py`'s `wild_outlier_mask` comment): for typical WaZP
  cluster masses at this density, the NFW scale radius r_s falls inside or
  right at the edge of the radius range where per-cluster ΔΣ is even
  finitely measurable (bins below ~0.5–0.6 Mpc have finite-measurement
  fractions as low as 38% at the innermost radius). This is a genuine
  survey-depth finding, not a pipeline bug — LSST-like density
  (40 gal/arcmin²) is the intended production regime.
- **`wild_outlier_mask` with a one-sided range is statistically invalid.**
  A one-sided cut (only masking `ΔΣ ≤ min`) manufactures a fake excess
  inner signal by asymmetrically truncating noise that genuinely straddles
  zero — confirmed to bias the fit toward extreme concentration/slope
  values. If ever re-enabled, `wild_outlier_min` must be negative
  (two-sided), matching the current default range's intent.
- **Boost correction is inconsistent by design between the two real-data
  pipelines** (§3) — the joint per-cluster fit has none unless `boosts:`
  is explicitly configured; the stacked cross-check always applies one
  internally. Do not compare their raw ΔΣ data vectors directly without
  accounting for this.
- **`r_min_fit=0.5` is not an arbitrary cut** — it sits almost exactly at
  the radius where per-cluster measurement reliability sharply improves
  for this survey depth (see the finite-fraction quantification above).
  Relaxing it without also addressing data quality reproduces the
  numerical pathology the `sigma_e`/`floor_rmax` variance floor was built
  to fix (a handful of near-zero-galaxy-count bins can otherwise get an
  implausibly tiny bootstrap variance and dominate the entire joint
  likelihood — see `prepare_real_data.py`'s inline comment on
  `sigma_e`/`floor_rmax`).
- **The stack-level c–M relation from
  `notebooks/wazp_mass_richness_by_delta_m.ipynb` is flagged internally as
  unphysical** (project history, suspected DES-Y3 high-shear calibration
  at inner radii) and carries no in-notebook warning cell as of this
  writing — see §9.
- **Legacy fit-model miscentering defaulted to a value inconsistent with
  simulation truth** (fixed; see the `fit_fmiss`/`fit_params` discussion)
  — any results predating that fix need re-quoting, not re-trusting.

## 8. Where to look next

- `CLAUDE.md` — architectural summary (currently describes an April-2026
  snapshot of the package as mostly stubs; substantially out of date
  relative to the current `develop` branch — see §9).
- `reports/2026-07-26_boost_factors_variance_floor_alpha_cm.tex` — full
  writeup of the boost-factor/variance-floor/Diemer19-comparison work
  underlying the current real-data alpha_cm result.
- `notebooks/` — prototype and analysis notebooks, including
  `wazp_stacked_plots.ipynb` (regenerates every stacked-analysis plot from
  cached pickles).

---

## 9. Assessment: readiness for a 1.0 release linked in a paper

Scope: can an external reader (referee, interested colleague) clone this
repo and productively test/reproduce the pipeline? Assessed directly
against the current `develop` branch (`HEAD` at the time of writing:
`79e20cc`), not by assumption.

**Status: items 1, 2, 3, 5, 6, 7, 8 below are fixed** (2026-08-31, on
`develop`, uncommitted pending review). Item 4 is intentionally left
alone — it's discussed directly in the paper, so no in-repo caveat is
needed.

### Must fix before linking in a paper

1. ~~**Packaging does not list real dependencies.**~~ **FIXED**:
   `pyproject.toml` now declares `nautilus-sampler`, `colossus`,
   `cluster_toolkit`, `corner`, `h5py`, `multiprocess`, `scikit-learn`,
   `tqdm`, `pyyaml`, `matplotlib` as core dependencies, plus an `xlensing`
   extra pulling in `ngmix` (for `gamma_recovery_scan.py` only). `xlensing`
   itself still can't be a PEP 508 dependency (not on PyPI) — the
   `pyproject.toml` comment and README Quickstart both give the
   `pip install git+https://...` command explicitly.
2. ~~**No LICENSE file.**~~ **FIXED**: added `LICENSE` (MIT, matching the
   sibling `xlensing` repo by the same author) and `license = {file =
   "LICENSE"}` in `pyproject.toml`. Note: the repo's `.gitignore` blocks
   every extensionless file by default (`*` then allow-list by
   extension) — a bare `LICENSE` was silently untracked until a `!LICENSE`
   line was added; worth remembering if other extensionless files
   (`Makefile`, `Dockerfile`, ...) are added later.
3. ~~**`main` is 103 commits behind `develop`.**~~ **FIXED locally**:
   fast-forwarded `main` to `develop`'s `79e20cc` (`git checkout main &&
   git merge --ff-only develop`) — a pure fast-forward since `develop..main`
   was empty, nothing lost. **Not pushed to `origin/main`** — that's a
   separate, deliberate step for the user to take.
4. **The unphysical WaZP c–M result has no in-repo warning.** Left as-is
   by explicit instruction — the paper itself discusses this result, so a
   separate in-repo caveat isn't needed.
5. ~~**`CLAUDE.md` is badly stale.**~~ **FIXED**: rewritten to describe the
   actual current module/script layout, points to this manual for detail,
   and includes the no-external-data quickstart.

### Should fix — real but lower blast radius

6. ~~**Dead debug scripts/configs.**~~ **FIXED**: the 5 confirmed-dead
   `hisnr_*.yaml` configs (verified via `grep` that each points at a
   `gamma_scan_outputs/run_2026-0[5-7]-*` directory absent from both the
   repo and `~/destacking_data/output/`) and all 8 files under
   `scripts/old_scripts/` moved to `scripts/archive/` (`old_scripts/`
   subfolder preserved) with a short `scripts/archive/README.md`
   explaining why. `hisnr_tri_mis.yaml`/`hisnr_triaxial.yaml` were checked
   and do **not** reference a deleted path — left in place in `scripts/`.
7. ~~**No top-level pointer to this manual.**~~ **FIXED**: `README.md` now
   has a "Documentation" section linking `docs/MANUAL.md`.
8. ~~**Synthetic quickstart not flagged as the golden path.**~~ **FIXED**:
   `README.md` and `CLAUDE.md` both now lead with a "Quickstart" section
   naming `gamma_inference_config_example.yaml` explicitly as the
   no-external-data first thing to run.

### Nice to have

9. **CLI ergonomics**: all scripts are run as `python scripts/foo.py
   config.yaml`, no installed console-script entry points. Fine for a
   research codebase; a `[project.scripts]` block would be a nice-to-have
   polish, not a blocker.
10. **Tests**: 224 tests pass in ~21s (`pytest -q -m "not slow"`, run
    directly against `develop`); 9 tests are marked `slow` (full nautilus
    sampler / real multiprocess pipeline runs) and were not run to
    completion in this assessment (a naive `pytest -q` including them
    exceeded 5 minutes without finishing) — that's expected for sampler-
    based tests and not itself a problem, but consider whether CI should
    run them on a longer budget/schedule rather than leaving them
    permanently unverified.
11. **`.coverage`, `.pytest_cache`, `__pycache__`, `.ipynb_checkpoints`
    directories are committed/present in the working tree** (not
    necessarily tracked by git — not checked here — but worth a
    `.gitignore` audit before tagging so a fresh clone doesn't ship stale
    cache artifacts).

### What's already in good shape

- Core module docstrings (`gamma_inference.py`, `prepare_real_data.py`,
  `wazp_stacked_analysis.py`, `destacking/simulation.py`, `destacking/
  model.py`, `destacking/fitting.py`) are genuinely thorough — this
  manual was assembled almost entirely by reorganizing existing
  documentation, not writing new explanations from scratch.
- Config templates (`gamma_inference_config_example.yaml`,
  `real_data_config_example.yaml`) are well-commented and functional as
  quickstarts.
- Resumability/checkpointing is real and documented at every stage
  (simulation, measurement, per-bootstrap fit).
- 224 fast unit tests currently pass.
