# Destacking

This work consists of a joint fitting of the weak lensing signal of galaxy clusters to study observable relations. The initial idea is to investigate relationships between cluster observables such as the magnitude gap (that is, the difference in magnitude between the BCG and the brightest satellite in the r band inside a radius of half R_200), or the offset between the BCG and the weighted luminosity center, and the cluster concentration. The idea is to calibrate the hight of the pivot of the mass-concentration relation as a function of the observable, using a small hierarchical bayesian model.


## Documentation

Full manual (installation, core concepts, config reference, walkthroughs,
known caveats): [`docs/MANUAL.md`](docs/MANUAL.md).

## Quickstart

No external data needed — this simulates its own clusters end to end:
```
pip install -e .
pip install git+https://github.com/andrevitorelli/xlensing
python scripts/gamma_inference.py scripts/gamma_inference_config_example.yaml
```
This is the recommended first thing to run, including for readers without
access to the real WaZP/DES-Y3 catalogs used in the paper. See
`docs/MANUAL.md` §4 for this walkthrough plus the real-data and
stacked-cross-check pipelines.

## Dependencies
- Numpy, scipy, astropy, nautilus-sampler, colossus, cluster_toolkit,
  corner, h5py, multiprocess, scikit-learn, tqdm, pyyaml, matplotlib
  (see `pyproject.toml`)
- xlensing (https://www.github.com/andrevitorelli/xlensing) — not on PyPI,
  install separately


## Method example

1. Sample a mass from a mass-observable relation for each cluster in the data, given that observable. 
2. Using a concentration-observable relation, define the pivot height for the mass-concentration relation as a part of the parameter sample process.
3. Generate NFW profiles for each of the clusters in the sample.
4. Calculate the loglikelihood respective to the data

In this example, only one parameter is fitted (the scaling between the height of the pivot of the mass-concentration relation and the observable.)


## Development tasks

- Initial Simulation
    - Generate a sample of cluster masses from a mass function
    - Generate a cluster richness from an HOD
    - Generate cluster magnitude gaps by sampling galaxy luminosities from the luminosity function
    - Generate their concentrations, using a mass-concentration relation, but changing the pivot point with some power-law function of the magnitude gap
    - Generate NFW profiles of the excess mass density (Delta Sigma) directly, no shear catalog at this point
- First application: use the method above to recover the parameter that controls the height of the pivot point respective to the magnitude gap
- Further reiterations
    - Generate background galaxy fields and distort them with the clusters and repeat the analysis
    - Generate triaxial clusters and repeat the analysis
- Future work:
    - investigate the collider effect when getting mass-concentration relations from data, when stacking clusters per richness (as richness will be a collider between environmental effects and mass)
    - add cosmological sampling to investigate the role of neutrinos