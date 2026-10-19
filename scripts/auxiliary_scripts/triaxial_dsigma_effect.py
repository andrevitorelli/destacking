"""Effect of triaxiality on the measured NFW ΔΣ profile (2026-09-28).

Same method as sphere_vs_triaxial.py, updated to the current xlensing API
and to the paper's radial range: for one representative halo, N triaxial
shapes/orientations are drawn from the simulation prior
(xlensing.sims.sample_triaxial_shapes); for each, ONE zero-shape-noise
source field is sheared twice through the same function
(apply_triaxial_NFW_shear_region) -- once with b=c=a (exactly spherical, the
simulation's own spherical path) and once with the drawn axes -- and both are
measured with the pipeline's estimator (metacal_cluster_lensing + signal).
The ratio isolates the geometric projection effect.

It then fits a spherical one-halo NFW (M, c) to the orientation-averaged
triaxial profile, over the full radial range and over R >= 0.5 Mpc, to show
what a spherical fit infers.

Outputs (figures + npz) go to --out (default: reports/figs).
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from scipy.optimize import least_squares

from xlensing.data import metacal_cluster_lensing, signal
from xlensing.sims import (apply_triaxial_NFW_shear_region, make_simple_random_cat,
                           sample_triaxial_shapes)

from destacking.model import _nfw_components

Z, M, C = 0.4, 2.0e14, 4.0   # defaults; --logm / --c override
RADII = np.logspace(-0.8, 0.6, 16)                   # stacked-analysis grid
_d = np.diff(np.log10(RADII))[0]
_e = np.logspace(np.log10(RADII[0]) - _d / 2, np.log10(RADII[-1]) + _d / 2, len(RADII) + 1)
BINS = np.column_stack([_e[:-1], _e[1:]])


def measure(cluster, gal, width_mpc):
    g = apply_triaxial_NFW_shear_region(cluster, gal.copy(), response=gal[:, 5])
    src = g[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]]
    bg = metacal_cluster_lensing((0.0, 0.0, Z), src.T, width_mpc)
    stake = np.array([bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
                      bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"]])
    return signal(stake, BINS)[0]


def sph_model(logM, logc):
    cen, _ = _nfw_components(np.array([Z]), np.array([10 ** logM]), np.array([10 ** logc]), RADII)
    return cen[0]


def main():
    global M, C
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../../reports/figs"))
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--c", type=float, default=C)
    ap.add_argument("--logm", type=float, default=np.log10(M))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    M, C = 10 ** a.logm, a.c
    rng = np.random.default_rng(a.seed)
    DA = Planck18.angular_diameter_distance(Z).value
    width_rad = 1.05 * BINS[-1, 1] / DA
    width_mpc = width_rad * DA
    sh = sample_triaxial_shapes(np.full(a.n, Z), np.full(a.n, M), rng=rng)
    ds_s, ds_t = np.zeros((a.n, len(RADII))), np.zeros((a.n, len(RADII)))
    t0 = time.time()
    for i in range(a.n):
        gal = make_simple_random_cat(a.density, width_rad, (0.8, 2.5), 0.0,
                                     seed=int(rng.integers(1, 2**31 - 1))).T   # (N_gal, 10)
        ax_ = sh["r_vir"][i]
        base = [0.0, 0.0, Z, M, C, ax_]
        ds_s[i] = measure(base + [ax_, ax_, sh["theta_los"][i], sh["phi_los"][i]], gal, width_mpc)
        ds_t[i] = measure(base + [sh["q_ba"][i] * ax_, sh["q_ca"][i] * ax_,
                                  sh["theta_los"][i], sh["phi_los"][i]], gal, width_mpc)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{a.n}  ({time.time() - t0:.0f} s)", flush=True)
    ratio = ds_t / ds_s
    r_mean = np.nanmean(ratio, 0); r_lo, r_hi = np.nanpercentile(ratio, [16, 84], 0)
    cos_t = np.abs(np.cos(sh["theta_los"]))
    q = sh["q_ca"]   # line of sight along the major axis (cos ~ 1) is the most elongated view
    # spherical NFW fit to the orientation-averaged triaxial profile, expressed as
    # ratio x spherical model (removes binning differences between model and measurement)
    target = r_mean * sph_model(np.log10(M), np.log10(C))
    fits = {}
    for lab, sel in [("0.16-4 Mpc", RADII > 0), ("0.5-4 Mpc", RADII >= 0.5)]:
        f = least_squares(lambda p: np.log(sph_model(*p)[sel] / target[sel]), [np.log10(M), np.log10(C)])
        fits[lab] = (f.x[0] - np.log10(M), 10 ** f.x[1] / C)
        print(f"  spherical fit {lab}: dlogM = {fits[lab][0]:+.3f} dex, c_fit/c_true = {fits[lab][1]:.3f}")
    np.savez(os.path.join(a.out, f"triaxial_dsigma_effect{a.tag}.npz"), radii=RADII, ds_sph=ds_s, ds_tri=ds_t,
             q_ba=sh["q_ba"], q_ca=q, theta_los=sh["theta_los"], z=Z, M=M, c=C)

    plt.rcParams.update({"font.size": 11, "xtick.direction": "in", "ytick.direction": "in"})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    ax.plot(RADII, np.nanmean(ds_s, 0), "k-", lw=2, label="spherical")
    ax.plot(RADII, np.nanmean(ds_t, 0), "C3-", lw=2, label="triaxial, orientation average")
    ax.fill_between(RADII, *np.nanpercentile(ds_t, [16, 84], 0), color="C3", alpha=0.2, lw=0, label="triaxial, 16-84%")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$R$ [Mpc]"); ax.set_ylabel(r"$\Delta\Sigma$ [$M_\odot\,{\rm pc}^{-2}$]")
    ax.set_title(rf"$z={Z}$, $\log_{{10}}M_{{200}}={np.log10(M):.2f}$, $c={C:g}$, $N={a.n}$ shapes", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax = axs[1]
    ax.axhline(1, color="k", lw=0.8)
    ax.plot(RADII, r_mean, "C3-", lw=2, label="all orientations")
    ax.fill_between(RADII, r_lo, r_hi, color="C3", alpha=0.15, lw=0)
    edges = np.quantile(cos_t, [0, 1 / 3, 2 / 3, 1])
    for k, (lab, col) in enumerate([(r"$|\cos\theta_{\rm LOS}|$ low", "C0"), ("middle", "C2"), ("high", "C1")]):
        m = (cos_t >= edges[k]) & (cos_t <= edges[k + 1])
        ax.plot(RADII, np.nanmean(ratio[m], 0), "--", color=col, lw=1.4, label=f"{lab} tercile")
    ax.set_xscale("log"); ax.set_xlabel(r"$R$ [Mpc]"); ax.set_ylabel(r"$\Delta\Sigma_{\rm tri}/\Delta\Sigma_{\rm sph}$")
    ax.axvspan(RADII[0] / 1.1, 0.5, color="0.9", zorder=0, lw=0); ax.text(0.2, ax.get_ylim()[0] + 0.02, "inner bins", fontsize=8, color="0.4")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[2]
    ax.axhline(1, color="k", lw=0.8)
    ax.plot(RADII, target / sph_model(np.log10(M), np.log10(C)), "C3-", lw=2, label="triaxial average / truth")
    for (lab, (dm, cr)), col in zip(fits.items(), ["C0", "C2"]):
        best = sph_model(np.log10(M) + dm, np.log10(C * cr))
        ax.plot(RADII, best / sph_model(np.log10(M), np.log10(C)), "--", color=col, lw=1.6,
                label=rf"spherical fit, {lab}: $\Delta\log M={dm:+.3f}$, $c/c_{{\rm true}}={cr:.2f}$")
    ax.set_xscale("log"); ax.set_xlabel(r"$R$ [Mpc]"); ax.set_ylabel(r"ratio to true spherical profile")
    ax.axvspan(RADII[0] / 1.1, 0.5, color="0.9", zorder=0, lw=0); ax.legend(fontsize=8, frameon=False)
    for ax in axs:
        ax.tick_params(top=True, right=True, which="both")
        ax.set_xticks([0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.5", "1", "2"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"triaxial_dsigma_effect{a.tag}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(a.out, f"triaxial_dsigma_effect{a.tag}.png"), dpi=110, bbox_inches="tight")
    print("mean ratio per bin:", np.round(r_mean, 3))


if __name__ == "__main__":
    main()
