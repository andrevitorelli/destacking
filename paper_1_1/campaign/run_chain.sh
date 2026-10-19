#!/bin/bash
# Run one paper 1.1 simulation chain end to end on one machine:
#   1. simulate   draw the cluster population + sources, shear them, measure every cluster (gamma_inference.py)
#   2. prepare    re-measure with the analytic covariance (remeasure_checkpoint.py) and build the
#                 3-way stacks and member weights of the stacked analysis (stack_projection.py)
#   3. joint      nine-parameter joint fit (gamma_inference.py, inference only)
#   4. stacked    forward-modelled stacked fit (stacked_forward_fit.py)
# Each stage is skipped if its output already exists, so the script can be re-run after an interruption.
#
# Extra stages (Appendix E): jointB and jointS refit the same realisation with the galaxy-bootstrap covariance and
# a Gaussian or per-bin Sellentin-Heavens likelihood.
#
# usage: paper_1_1/campaign/run_chain.sh CHAIN [N_WORKERS] [STAGES]
#   e.g. run_chain.sh DESfull1k3L 32                        (simulate prepare joint stacked)
#        run_chain.sh LSSTind1k5M0L 32 "simulate prepare joint"   (Section 4 controls: no stacked fit)
#        run_chain.sh DESfull1k3L 32 "jointB jointS"         (Appendix E refits)
# Needs the configs written by make_configs.py.  Run from anywhere; set DESTACKING_DATA (default ~/destacking_data).
# Typical wall time with 32 cores: DES-like ~2 h, LSST-like ~2.5 h (simulate ~15 min, joint 0.5-1 h, stacked ~1 h).
set -euo pipefail
CHAIN=$1; NW=${2:-32}; STAGES=" ${3:-simulate prepare joint stacked} "
has(){ [[ "$STAGES" == *" $1 "* ]]; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${DESTACKING_DATA:-$HOME/destacking_data}; OUT=$DATA/output; SIMS=$DATA/sims; CFG=$OUT/v1.1/configs
export DESTACKING_DATA=$DATA DESTACKING_DM_FORM=linear
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$REPO"
grid=(); case $CHAIN in DES*) grid=(--log-inner -0.6 --log-outer 0.6 --n-radii 6);; esac

# 1. simulate
if has simulate && ! ls "$OUT/${CHAIN}_sim"/simulation_g00_*/simulated_data.npz >/dev/null 2>&1; then
  RUN_DIR=$OUT/${CHAIN}_sim python scripts/gamma_inference.py "$CFG/${CHAIN}_simulate.yaml"
fi
SIM=$(ls -d "$OUT/${CHAIN}_sim"/simulation_g00_* | head -1); SUB=$(basename "$SIM")

# 2. prepare: analytic-covariance re-measurement and the stacks
if has prepare; then
  [ -f "$OUT/${CHAIN}_sim_analytic/$SUB/simulated_data.npz" ] || \
    python scripts/auxiliary_scripts/remeasure_checkpoint.py "$SIM" "$SIMS/${CHAIN}_g000_bgs.pkl" "$OUT/${CHAIN}_sim_analytic/$SUB" "${grid[@]}"
  if has stacked || [[ "$CHAIN" != *M0L && "$CHAIN" != *T0L ]]; then
    [ -f "$OUT/jvs/${CHAIN}_stacks/weights.npz" ] || \
      python paper_1_1/campaign/stack_projection.py "${CHAIN}_g" "$SIM" "$OUT/jvs/${CHAIN}_stacks" "$NW"
  fi
fi

# 3. joint fits (joint: analytic covariance; jointB / jointS: bootstrap covariance, Appendix E)
for J in joint jointB jointS; do
  has $J || continue
  ls "$OUT/jvs/${CHAIN}_$J"/simulation_g00*/bootstrap_000/gamma_posterior.npz >/dev/null 2>&1 || \
    RUN_DIR=$OUT/jvs/${CHAIN}_$J python scripts/gamma_inference.py "$CFG/${CHAIN}_$J.yaml"
done

# 4. forward-modelled stacked fit
has stacked && [ ! -f "$OUT/jvs/${CHAIN}_stacked_hN.npz" ] && \
  python scripts/auxiliary_scripts/stacked_forward_fit.py --catalog "$SIMS/${CHAIN}_g000_real_data.fits" \
    --stacks "$OUT/jvs/${CHAIN}_stacks/stacks3.pkl" --weights "$OUT/jvs/${CHAIN}_stacks/weights.npz" \
    --out "$OUT/jvs/${CHAIN}_stacked_hN.npz" --data sim --min-n 50 --n-live 2000 --workers "$NW"
echo "chain $CHAIN done"
