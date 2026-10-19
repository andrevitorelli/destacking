"""Locations used by the paper 1.1 reproduction scripts.

Everything is resolved from environment variables, so no script contains a
machine-specific path:

DESTACKING_DATA   root of the data tree (default ~/destacking_data), laid out as
                    real_data/           WaZP catalogue, metacalibration cache, real-data products
                    sims/                simulated cluster catalogues and background arrays
                    output/              simulation runs (one folder per chain) and fits
                    output/v1.1/         paper 1.1 posteriors, summaries and intermediate results
DESTACKING_FIGS   where the paper figures are written (default ./figs in the current directory)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # repository root
AUX = os.path.join(REPO, "scripts", "auxiliary_scripts")                      # pipeline scripts
DATA = os.path.abspath(os.path.expanduser(os.environ.get("DESTACKING_DATA", "~/destacking_data")))
FIGS = os.path.abspath(os.path.expanduser(os.environ.get("DESTACKING_FIGS", "figs")))
REAL = os.path.join(DATA, "real_data")
SIMS = os.path.join(DATA, "sims")
OUT = os.path.join(DATA, "output")
V11 = os.path.join(OUT, "v1.1")
os.makedirs(FIGS, exist_ok=True)
