# destacking

Joint (un-stacked) fitting of individual galaxy-cluster weak-lensing profiles
to constrain cluster scaling relations.

Instead of stacking clusters and fitting a mean profile, `destacking` fits
every cluster's ΔΣ profile at once under one shared likelihood:
* the mass of each cluster is tied to its richness and redshift (the
  mass-richness relation);
* its concentration is tied to its mass, redshift and magnitude gap (the
  mass-concentration relation).

The package also implements a forward-modelled stacked fit that uses exactly
the same model, and the simulations used to compare the two.

This is a research repository under active development. The lensing
measurement and the ΔΣ model live in the
[xlensing](https://github.com/andrevitorelli/xlensing) package.

## Reproducing the paper

**[`paper_1_1/README.md`](paper_1_1/README.md)** is a step-by-step guide to
reproducing *Joint fit of galaxy cluster scaling relations from individual
weak lensing data: comparison with stacked analysis and a real data test*
(Vitorelli & Cypriano). It covers:
* the public input catalogues;
* the real-data measurements and fits;
* the 132-chain simulation campaign;
* every table and figure.

All configurations and scripts are in `paper_1_1/`.

## Installation

```bash
micromamba create -n destacking -f scripts/environment.yml    # or conda / mamba
micromamba activate destacking
pip install git+https://github.com/tmcclintock/cluster_toolkit
pip install git+https://github.com/andrevitorelli/xlensing
pip install -e .
```

Data and outputs are read from and written to `$DESTACKING_DATA`, which
defaults to `~/destacking_data`. Config files may use `~` and
`$DESTACKING_DATA` in their paths.

## Quick start

No external data are needed. This simulates its own clusters end to end:

```bash
python scripts/gamma_inference.py scripts/gamma_inference_config_example.yaml
```

## Documentation

[`docs/MANUAL.md`](docs/MANUAL.md) covers installation, core concepts, the
config reference, walkthroughs for simulated and real data, the output
formats and known caveats.

## Layout

| Path | Contents |
|---|---|
| `destacking/` | the package: model, fitting, simulation, data selection, I/O |
| `scripts/gamma_inference.py` | the main simulation + joint-fit driver (YAML configured) |
| `scripts/auxiliary_scripts/` | real-data preparation, stacked analysis, forward-modelled stacked fit |
| `paper_1_1/` | everything needed to reproduce the paper |
| `notebooks/` | exploratory notebooks |
| `scripts/archive/` | superseded scripts and configurations, kept for the record |

## License

MIT (see `LICENSE`).
