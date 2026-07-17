"""Refit a gamma run with the model evaluated as the annulus-averaged dSigma.

Patches destacking.model._nfw_profiles so that, for each radial bin, the
model is the count-weighted average of dSigma(R) over the annulus
(uniform source density: weight ∝ R dR) instead of the value at the bin
centre.  Then delegates to gamma_inference.main() in inference-only mode.

Usage: python refit_annulus_model.py hisnr_annulus_refit.yaml
"""

import sys

import numpy as np

import gamma_inference as gi
import destacking.model as dmodel

K_SUB = 8   # sub-annuli per radial bin

_orig_nfw_profiles = dmodel._nfw_profiles

# Per bin-centre radius: linear sub-grid across the annulus and the
# count weights (uniform surface density -> weight ∝ R dR).
_SUB = {}
for r, (lo, hi) in zip(gi.SIM_RADII, gi.BIN_LIMITS):
    edges = np.linspace(lo, hi, K_SUB + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])
    w     = mids * np.diff(edges)
    _SUB[round(float(r), 10)] = (mids, w / w.sum())


def _nfw_profiles_annulus(z, M200m, C200m, radii, **kwargs):
    mids_all, w_all = [], []
    for r in np.asarray(radii):
        mids, w = _SUB[round(float(r), 10)]
        mids_all.append(mids)
        w_all.append(w)
    radii_sub = np.concatenate(mids_all)                    # (M*K,)
    out = _orig_nfw_profiles(z, M200m, C200m, radii_sub, **kwargs)
    n_cl, n_bins = out.shape[0], len(radii)
    out = out.reshape(n_cl, n_bins, K_SUB)
    W   = np.stack(w_all)                                   # (M, K)
    return np.einsum('nmk,mk->nm', out, W)


dmodel._nfw_profiles = _nfw_profiles_annulus
print(f"[refit] model patched: annulus-averaged dSigma "
      f"({K_SUB} sub-annuli per bin, count-weighted)", flush=True)

if __name__ == "__main__":
    sys.exit(gi.main())
