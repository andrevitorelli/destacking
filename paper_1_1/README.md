# Reproducing paper 1.1

*Joint fit of galaxy cluster scaling relations from individual weak lensing data:
comparison with stacked analysis and a real data test* (Vitorelli & Cypriano).

This folder holds the configurations and scripts that produced every number,
table and figure of the paper. The steps below go from the public catalogues
to the final figures. Each step lists the command, what it writes, and roughly
how long it takes. All scripts take their paths from two environment variables,
so nothing depends on the machine they were first run on.

```
paper_1_1/
  paths.py              data and figure locations (DESTACKING_DATA, DESTACKING_FIGS)
  campaign/             the simulation campaign: config templates, chain list, runner, summaries
  real_data/            real-data configurations and helpers
  analysis/             real-data fits, summaries, tables and every paper figure
  analysis/diag/        simulation diagnostics quoted in the text (Sec. 4, App. E, App. F)
```

The commands below assume you run them from the repository root.

---

## 0. Set-up

### Software

```bash
micromamba create -n destacking -f scripts/environment.yml   # or conda/mamba
micromamba activate destacking
pip install git+https://github.com/tmcclintock/cluster_toolkit
pip install git+https://github.com/andrevitorelli/xlensing
pip install -e .                                              # this repository
pip install colossus                                          # Diemer & Joyce (2019) c(M), Sec. 6.4
```

Every paper run uses the linear magnitude-gap model. Set this in every shell
you use:

```bash
export DESTACKING_DM_FORM=linear
```

### Data tree

```bash
export DESTACKING_DATA=/path/to/destacking_data     # default: ~/destacking_data
export DESTACKING_FIGS=/path/to/paper/figs          # default: ./figs
```

Everything is read from and written under `$DESTACKING_DATA`:

| Folder | Contents |
|---|---|
| `raw/` | the public input catalogues (step 1) |
| `real_data/` | the WaZP cluster catalogue, the source cache, per-cluster measurements, stacks, boost fits |
| `sims/` | simulated cluster catalogues and per-cluster background arrays |
| `output/` | one folder per simulation chain (`<chain>_sim`, `<chain>_sim_analytic`), fits in `output/jvs/` |
| `output/v1.1/` | paper 1.1 configurations, posteriors, summaries (`results/`, `report_des/`) |

### Compute

* **Simulation campaign:** the 132 chains of the paper took about 350
  core-days in total (32 cores per chain, about 2 hours each).
* **Real-data fits:** a few days on a 16-core workstation.
* **Figures and tables:** minutes, once the fits exist.

---

## 1. Input catalogues (public)

Download the following into `$DESTACKING_DATA/raw/`:

| File | Content | Source |
|---|---|---|
| `DESY3_metacal_v03-004.h5` | DES Y3 metacalibration shape catalogue | DES Y3 release, https://des.ncsa.illinois.edu/releases |
| `DESY3_GOLD_2_2.1_DNF.h5` | DES Y3 DNF photometric redshifts | DES Y3 release |
| `y6a2_dnf_wazp_v5.0.12.6801_clusters.fits` | WaZP clusters (DES Y6, v5.0.12) | DES Y6 release |
| `y6a2_dnf_wazp_v5.0.12.6801_members.fits` | WaZP members, used for the magnitude gaps | DES Y6 release |

---

## 2. Real data: per-cluster measurements (Sec. 5)

**Bootstrap-covariance measurement (used by the joint fit).** This step also
builds the cluster catalogue `real_data/wazp_real_data.fits` and the filtered
source cache:

```bash
python scripts/auxiliary_scripts/prepare_real_data.py paper_1_1/real_data/prepare_real_data_bootstrap.yaml
python paper_1_1/real_data/select_positive_gap.py $DESTACKING_DATA/real_data/wazp_real_data_bootfloor_v4
```

**Analytic-covariance measurement (Appendix E).** This version estimates
σ_e separately for 50 classes of source weight:

```bash
python scripts/auxiliary_scripts/prepare_real_data.py paper_1_1/real_data/prepare_real_data_analytic.yaml
python paper_1_1/real_data/select_positive_gap.py $DESTACKING_DATA/real_data/wazp_real_data_wclass_v5
```

What the steps above write:
* `real_data/wazp_real_data.fits`: 15,967 clusters, with ZPHOT, NGALS and
  DELTA_M_I.
* `real_data/wazp_real_data_bootfloor_v4[_dmpos]/` and
  `real_data/wazp_real_data_wclass_v5[_dmpos]/`: one data vector and
  covariance per cluster. The `_dmpos` copies keep the 15,825 clusters with
  Δm > 0.
* Run time: a few hours each. Both steps are resumable.

## 3. Real data: stacks and boost factors (Sec. 5.4–5.5)

```bash
# stage 1 of the stacked analysis: per-cluster background arrays, 2-way stacks and the boost-factor fits
python scripts/auxiliary_scripts/wazp_stacked_analysis.py                          # f = 1.1 (joint fit's selection)
WAZP_STACK_FACTOR=1.2 python scripts/auxiliary_scripts/wazp_stacked_analysis.py    # f = 1.2 (stacked analysis)
# 3-way (redshift x richness x gap tercile) stacks and the member weights K_c(R) of the forward model
python paper_1_1/real_data/build_real_stacks.py 1.2
python paper_1_1/real_data/build_real_stacks.py 1.1
```

What this writes:
* `real_data/wazp_stacked_bkgs[_f1.2].pkl`.
* `real_data/wazp_boost_fits.pkl`: the per-cluster boost factors of the joint
  fit (Eq. 21, Fig. 11).
* `real_data/stacked_full_param_fit_f1.2/stacks3.pkl` and
  `real_data/stack_weights_f1.2.npz`: the 36 stacks of Sec. 5.5.
* `real_data/stacked_f1.1_3way/`: the stacks used for Table 7.

## 4. Real data: fits (Sec. 5.5, 5.6, 6)

Run these from the `analysis/` folder. `NPOOL` sets the number of cores.

```bash
cd paper_1_1/analysis
export NPOOL=15 SH_TIMEOUT=2400 JV_LIK=sh
D=$DESTACKING_DATA/real_data; V=$DESTACKING_DATA/output/v1.1
mkdir -p $V/stacked $V/posteriors $V/results $V/report_des/figs $V/diag

# forward-modelled stacked fit (Sec. 5.5; Table 4, Figs. 12, 13, 16), Hartlap and Sellentin-Heavens likelihoods
for L in hartlap sh; do t=$([ $L = sh ] && echo _sh); python ../../scripts/auxiliary_scripts/stacked_forward_fit.py \
  --catalog $D/wazp_real_data.fits --stacks $D/stacked_full_param_fit_f1.2/stacks3.pkl --weights $D/stack_weights_f1.2.npz \
  --out $V/stacked/real_minN50$t.npz --data real --min-n 50 --bin-nodes 0 --n-live 2000 --priors physical --workers 15 --likelihood $L; done

python nautilus_dmr.py stack                    # stacked fit, reference for Sec. 6.5
python nautilus_dmr.py stack --dmr              # stacked fit with delta_Mlambda free (Table 6)
RMIN=0.25 python nautilus_dmr.py stack --dmr    # ... bins above 0.25 Mpc only (Table 6)
python stack_variants.py f11r025 r025 f11 dmmis dmsoff rshear sh   # stacked robustness refits (Table 7)
python stack_only_fits.py                       # per-stack single-halo fits (Fig. 14, App. F)
python twostep_real.py --write                  # two-step analysis of the real stacks (Table 9)

python joint_sh.py boot 0 50                    # joint fit, 50 cluster bootstraps (Sec. 5.6; Table 4, Fig. 15)
SH_MODEL=dmr python joint_sh.py boot 0 50       # joint fit with delta_Mlambda free (Table 6, Fig. 18)
SH_MODEL=diemer python joint_sh.py boot 0 50    # Diemer & Joyce (2019) baseline (Sec. 6.4, Fig. 17b)
python joint_variants.py full rshear dmle0 dm2sig dmmis dmsoff zbin0 zbin1 zbin2 \
       boost01 boost02 boost03 boost04 boost05 boost06 boost07 boost08          # joint robustness refits (Table 7)

# joint fit with the analytic covariance (App. E, G), 50 bootstraps
RUN_DIR=$V/real_runs/joint_v11e python ../../scripts/gamma_inference.py ../real_data/joint_fit_analytic.yaml
```

Run times on 15 cores:
* each stacked fit: about 1 hour;
* each set of 50 joint bootstraps: about 6 hours;
* the variants: a day in total.

The posteriors go to `$V/posteriors/` and `$V/stacked/`.

## 5. Simulation campaign (Sec. 3, 4; App. D, E, F)

### 5a. Configurations

```bash
python paper_1_1/campaign/make_configs.py      # writes $DESTACKING_DATA/output/v1.1/configs/
```

`campaign/chains.csv` lists the 132 chains of the paper:
* 84 DES-like realisations (`DESfull<r>k<k>L`);
* 48 LSST-like realisations (`LSSTind<r>k<k>L`);
* the injected γ_cm = k/36 and the random-seed offset of each.

The templates in `campaign/templates/` hold everything else: population,
sources, triaxiality, miscentring, radial grids, priors.

`make_configs.py` also writes:
* the 12 extra LSST-like realisations of the Sec. 4 slope trace;
* the Sec. 4 controls, with miscentring off (`…M0L`) or triaxiality off
  (`…T0L`);
* the Appendix E refits with the bootstrap covariance (`_jointB`, `_jointS`);
* the 24 clean realisations of Appendix D (`DESclean0..23`).

### 5b. Running the chains

Each chain goes through four stages: simulate, then prepare (analytic
re-measurement and stacks), then the joint fit, then the forward-modelled
stacked fit.

```bash
paper_1_1/campaign/run_chain.sh DESfull1k3L 32
```

Every stage is skipped if its output exists, so the script can be re-run.
On a cluster, submit one job per chain with 32 cores and about 4 hours of
wall time. For example, with Slurm:

```bash
for c in $(tail -n +2 paper_1_1/campaign/chains.csv | cut -d, -f1); do
  sbatch -c 32 -t 04:00:00 --wrap "paper_1_1/campaign/run_chain.sh $c 32"; done
```

Additional runs quoted in the paper:

| Runs | Command |
|---|---|
| Sec. 4 slope trace: LSST-like r = 7..12 at k = 5, 6 | `run_chain.sh LSSTind<r>k<k>L 32 "simulate prepare joint"` |
| Sec. 4 miscentring-off controls (k = 2, 3, 5, 6) | `run_chain.sh LSSTind<r>k<k>M0L 32 "simulate prepare joint"` |
| Sec. 4 triaxiality-off control (k = 5) | `run_chain.sh LSSTind<r>k5T0L 32 "simulate prepare joint"` |
| App. E bootstrap-covariance refits (84 DES-like + 60 LSST-like) | `run_chain.sh <chain> 32 "jointB jointS"` |
| App. D clean realisations (i = 0..23) | `RUN_DIR=$DESTACKING_DATA/output/DESclean<i>_sim python scripts/gamma_inference.py $DESTACKING_DATA/output/v1.1/configs/DESclean<i>_simulate.yaml`, then re-measure it with the analytic covariance as in `run_chain.sh` (prepare stage) |

### 5c. Campaign summaries

Run these from `paper_1_1/campaign`, then `paper_1_1/analysis`:

```bash
python scan_summary_v11.py DES; python scan_summary_v11.py LSST
cd ../analysis
python twostep_sims_v11.py $(tail -n +2 ../campaign/chains.csv | cut -d, -f1)   # -> output/v1.1/twostep_v11/<chain>.json
python diag/threeway_summary.py
python diag/bootsh_summary.py
python diag/m0_scan.py
```

What each produces:
* `scan_summary_v11.py`: the scan rows in `output/v1.1/report_des/`, read
  for Tables 2–3 and Figs. 5–7.
* `twostep_sims_v11.py`: the two-step analysis of every simulated stack
  (App. F).
* `diag/threeway_summary.py`: Table 5 (simulation columns) and Table 10.
* `diag/bootsh_summary.py`: the App. E numbers.
* `diag/m0_scan.py`: the Sec. 4 miscentring control.

## 6. Tables and figures

Run these from `paper_1_1/analysis` once steps 2–5 are done. Figures go to
`$DESTACKING_FIGS`.

**Tables**

| Table | Script |
|---|---|
| 1 (priors) | in the configs: `campaign/templates/*_joint.yaml`, `analysis/joint_sh.py` |
| 2, 3 | `paper_sim_figs_v11.py` (printed) |
| 4, 7 | `sh_summary.py` (→ `results/sh_summary.json`) |
| 5 | `sh_summary.py` (real) and `diag/threeway_summary.py` (simulations) |
| 6 | `fig_dmr_bootmed.py` and `nautilus_dmr.py` outputs |
| 8 | `fisher_cut.py`, `fisher_real.py` |
| 9 | `twostep_real.py` |
| 10 | `diag/threeway_summary.py` |
| 11 | `map_v11.py` (→ `results/map_pjhpd_real_v11.json`) |

Numbers quoted in the text:
* `ppd_joint_on_stacks.py`: the posterior predictive check, Sec. 6.6.
* `reduced_shear.py`: the reduced-shear factor, Sec. 6.8.
* `fig_stack_mr_cm.py`: the per-stack regressions, Sec. 5.5.
* `fig_boost_vs_dm_v11.py`: App. C.
* `paired_lin.py` and `harness_pair.py`: App. D.
* `lsst_projection.py`: the Sec. 4 trace.

**Figures**

| Figure | File | Script |
|---|---|---|
| 1 | example_pop_sim.pdf | `paper_v11_extra_figs.py` |
| 2 | nz_sim_vs_real_combined.pdf | `fig_nz_v11.py` |
| 5, 7 | gamma_scan_des_lsst.pdf, sim_joint_vs_stacked_bias.pdf | `paper_sim_figs_v11.py` |
| 6 | sim_gamma_width.pdf | `fig_width_v11.py` |
| 12 | stacked_forward_fit_data.pdf | `fig_stackfit_v11.py` |
| 13, 15, 16, 17b | stacked_forward_corner.pdf, joint_fit_posterior.pdf, real_data_joint_vs_stacked.pdf, diemer_gamma_comparison.pdf | `fig_bootmed.py` |
| 14 | stacked_mass_richness_concentration_dm.pdf | `fig_stack_mr_cm.py` |
| 17a | diemer19_alpha_cm_prediction.pdf | `fig_diemer19_v11.py` (run before `fig_bootmed.py`, which reads its output) |
| 18 | gamma_delta_mr_posteriors.pdf | `fig_dmr_bootmed.py` |
| 20 | triaxial_dsigma_effect.pdf | `scripts/auxiliary_scripts/triaxial_dsigma_effect.py --out $DESTACKING_FIGS` |
| 21, 22 | bin_occupancy.pdf, bin_centre_vs_average.pdf | `fig_binning_appendix.py` |
| 23 | boost_vs_dm.pdf | `fig_boost_vs_dm_v11.py` |
| 24, 25 | paired_covariance_test.pdf, map_vs_median.pdf | `fig_appendix_v11.py` (App. D inputs, see below) |

The figures above, rerun from this repository on the original data products,
are byte-identical to the published PDFs.

**Appendix D inputs.**

1. Run the paired e/−e test with DES-like settings. This writes
   `output/harness_2026-09-29/results/des_z0.3_c3.npz`, 6000 identical haloes
   with log M = 14.5, c = 3 at z = 0.3, in 6 bins from 0.25 Mpc:

   ```bash
   python harness_pair.py 14.5 3 0.3 6000 $DESTACKING_DATA/output/harness_2026-09-29/results/des_z0.3_c3.npz 6.7 1.5 0.4 -0.6 0.6 6
   ```
2. Copy each clean realisation into `output/harness_2026-09-29/results/`:
   * `output/DESclean<i>_sim/simulation_g00_*/` goes to `desclean/D<i>/`;
   * its analytic-covariance re-measurement goes to `desclean_an/D<i>/`.
3. Run `python paired_lin.py linear`, which writes
   `output/v1.1/results/des_clean_projection_linear.json`.

Figures 3, 4, 8–11 and 19 illustrate the inputs: the simulated axis ratios
and offsets, the WaZP population, the boost factors and the triaxial
examples. They were made in the notebooks `notebooks/triaxial_shapes_3d.ipynb`
(Figs. 3 and 19) and `notebooks/wazp_stacked_plots.ipynb` (Figs. 10 and 11),
and directly from `real_data/wazp_real_data.fits` and the simulated truth
catalogues (Figs. 4, 8, 9). No numbers in the paper depend on them.

---

## Notes

* **Folder names.** Some intermediate folders keep the names they had during
  the analysis, such as `output/harness_2026-09-29/` (Appendix B and D
  inputs) and `output/v1.1/real_runs/` (the analytic-covariance joint fit).
  The scripts expect these names.
* **Two-halo term.** The simulations contain no two-halo term and no
  contaminating galaxies. Their fits omit the two-halo term and use unit
  boost factors (Sec. 3).
* **Run-to-run differences.** Nested sampling uses fixed seeds, so reruns
  should match closely. Small differences at the level of the Monte Carlo
  error (below 0.03σ) can come from different numbers of worker processes.
