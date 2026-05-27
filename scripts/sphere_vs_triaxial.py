"""
Sphere-vs-triaxial NFW ΔΣ comparison at zero shape noise.

For a fixed cluster (z, M, c) we sample N triaxial shapes (b/a, c/a, θ_LOS,
φ_LOS) from the same prior used in gamma_scan.py and, for each, generate ONE
zero-shape-noise background galaxy field which is then sheared twice:
  (a) spherical NFW with (M, c)
  (b) triaxial NFW with (M, c, a, b, c, θ_LOS, φ_LOS)
The two sheared catalogs are passed through the same metacal_cluster_lensing →
signal binning pipeline used by gamma_scan, yielding two ΔΣ profiles per
cluster.  With shape_noise = 0 and zero intrinsic ellipticities, the only
signal in the recovered ΔΣ is the lensing-induced one, so the difference
between the two measurements is purely the triaxial-vs-spherical geometric
projection effect.

Outputs under ``sphere_vs_triaxial_outputs/`` (created if missing):
  profiles.png             — population mean profiles + per-cluster band
  fractional_residual.png  — (ΔΣ_tri − ΔΣ_sph) / ΔΣ_sph vs R
  by_orientation.png       — split by cos(θ_LOS) bins
  by_triaxiality.png       — split by Franx+91 T parameter
  summary.json             — numerical summary (means, per-bin stats)
"""

from __future__ import annotations
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as _planck18

import xlensing
from xlensing.sims import (
    apply_NFW_shear_region,
    apply_triaxial_NFW_shear_region,
    make_simple_random_cat,
    sample_triaxial_shapes,
    r_vir,
)
from xlensing.data import metacal_cluster_lensing, signal


# ── Fixed cluster parameters ────────────────────────────────────────────────
N_CLUSTERS = 100              # number of triaxial shape draws
Z_CLUSTER  = 0.4
M_CLUSTER  = 1.0e14           # M_sun
C_CLUSTER  = 5.0              # canonical Dutton-Maccio

# ── Background galaxy field ─────────────────────────────────────────────────
DENSITY     = 30              # gal/arcmin^2
WIDTH_RAD   = 0.006           # ≈ 20.6 arcmin half-width
Z_SRC_RANGE = (0.6, 2.5)      # all > Z_CLUSTER

# ── Radial bins (same as gamma_scan) ────────────────────────────────────────
RADII  = np.logspace(-0.8, 0.8, 12)
_log_r = np.log10(RADII)
_dlog  = np.diff(_log_r)[0]
_edges = np.logspace(_log_r[0] - _dlog/2, _log_r[-1] + _dlog/2, len(RADII) + 1)
BIN_LIMITS = np.column_stack([_edges[:-1], _edges[1:]])

OUT_DIR = "sphere_vs_triaxial_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MASTER_SEED = 20260526


def make_galaxies(seed):
    """Zero-shape-noise galaxy catalog.

    make_simple_random_cat with shape_noise=0 already sets:
        e1 = e2 = 0,  R11 = R22 = 1, R12 = R21 = 0,  w = 1
    so the raw catalog has zero intrinsic ellipticity and unity response.
    """
    cat = make_simple_random_cat(
        DENSITY, WIDTH_RAD, Z_SRC_RANGE, shape_noise=0.0,
        seed=seed, poisson_density=False,
    ).T
    return cat


def measure_dsigma(cluster_arr, base_galaxies, shear_fn, width_mpc):
    """Apply shear and measure binned ΔΣ via metacal_cluster_lensing+signal."""
    g = base_galaxies.copy()
    g = shear_fn(cluster_arr, g)
    sources_mc = g[:, [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]]
    bg = metacal_cluster_lensing((0.0, 0.0, Z_CLUSTER), sources_mc.T, width_mpc)
    stake = np.array([
        bg["Critical Density"], bg["Tangential Shear"], bg["Cross Shear"],
        bg["Weights"], bg["Radial Distance"], bg["Mult. Bias"],
    ])
    sig_t, _ = signal(stake, BIN_LIMITS)
    return np.nan_to_num(sig_t, nan=0.0)


def main():
    rng = np.random.default_rng(MASTER_SEED)

    DA = float(_planck18.angular_diameter_distance(Z_CLUSTER).value)
    width_mpc = WIDTH_RAD * DA
    print(f"Cluster: z={Z_CLUSTER}, M={M_CLUSTER:.2e} M_sun, c={C_CLUSTER}")
    print(f"DA(z) = {DA:.2f} Mpc;  field full-width = {2*width_mpc:.2f} Mpc")
    print(f"N_CLUSTERS = {N_CLUSTERS},  galaxies/arcmin^2 = {DENSITY}")

    z_arr = np.full(N_CLUSTERS, Z_CLUSTER)
    M_arr = np.full(N_CLUSTERS, M_CLUSTER)
    shapes = sample_triaxial_shapes(z_arr, M_arr, rng=rng)

    q_ba   = shapes["q_ba"]
    q_ca   = shapes["q_ca"]
    theta  = shapes["theta_los"]
    phi    = shapes["phi_los"]
    a_axis = shapes["r_vir"]
    T      = (1 - q_ba**2) / (1 - q_ca**2 + 1e-12)

    print(f"\nShape distribution (N={N_CLUSTERS}):")
    print(f"  b/a   median={np.median(q_ba):.3f}  "
          f"16-84%={np.percentile(q_ba,16):.3f}-{np.percentile(q_ba,84):.3f}")
    print(f"  c/a   median={np.median(q_ca):.3f}  "
          f"16-84%={np.percentile(q_ca,16):.3f}-{np.percentile(q_ca,84):.3f}")
    print(f"  T     median={np.median(T):.3f}")
    print(f"  Prolate (T>2/3):  {np.mean(T>2/3):.2f}")
    print(f"  Triaxial:         {np.mean((T>=1/3)&(T<=2/3)):.2f}")
    print(f"  Oblate  (T<1/3):  {np.mean(T<1/3):.2f}")

    ds_sph = np.zeros((N_CLUSTERS, len(RADII)))
    ds_tri = np.zeros((N_CLUSTERS, len(RADII)))

    t0 = time.time()
    for i in range(N_CLUSTERS):
        seed_i = int(rng.integers(1, 2**31 - 1))
        galaxies = make_galaxies(seed_i)

        cluster_sph = [0.0, 0.0, Z_CLUSTER, M_CLUSTER, C_CLUSTER]
        ds_sph[i] = measure_dsigma(cluster_sph, galaxies,
                                   apply_NFW_shear_region, width_mpc)

        a = a_axis[i]
        b = q_ba[i] * a
        c = q_ca[i] * a
        cluster_tri = [0.0, 0.0, Z_CLUSTER, M_CLUSTER, C_CLUSTER,
                       a, b, c, theta[i], phi[i]]
        ds_tri[i] = measure_dsigma(cluster_tri, galaxies,
                                   apply_triaxial_NFW_shear_region, width_mpc)

        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            eta = dt / (i + 1) * (N_CLUSTERS - i - 1)
            print(f"  [{i+1}/{N_CLUSTERS}]  elapsed {dt:.0f}s  eta {eta:.0f}s",
                  flush=True)

    # ── Summary statistics ──────────────────────────────────────────────────
    sph_mean = np.nanmean(ds_sph, axis=0)
    tri_mean = np.nanmean(ds_tri, axis=0)
    sph_med  = np.nanmedian(ds_sph, axis=0)
    tri_med  = np.nanmedian(ds_tri, axis=0)
    sph_std  = np.nanstd(ds_sph, axis=0, ddof=1)
    tri_std  = np.nanstd(ds_tri, axis=0, ddof=1)

    frac_resid = (ds_tri - ds_sph) / np.where(ds_sph != 0, ds_sph, np.nan)
    frac_mean  = np.nanmean(frac_resid, axis=0)
    frac_med   = np.nanmedian(frac_resid, axis=0)
    frac_p16   = np.nanpercentile(frac_resid, 16, axis=0)
    frac_p84   = np.nanpercentile(frac_resid, 84, axis=0)

    print("\nPer-radius mean ΔΣ (M_sun/pc^2):")
    print(f"{'R[Mpc]':>10} {'sph mean':>11} {'tri mean':>11} "
          f"{'tri-sph':>10} {'frac.med':>10}  {'68% band':>16}")
    for k, R in enumerate(RADII):
        print(f"{R:>10.3f} {sph_mean[k]:>11.3f} {tri_mean[k]:>11.3f} "
              f"{tri_mean[k]-sph_mean[k]:>+10.3f} "
              f"{100*frac_med[k]:>+9.2f}%  "
              f"[{100*frac_p16[k]:>+5.1f}%,{100*frac_p84[k]:>+5.1f}%]")

    summary = {
        "config": {
            "N_CLUSTERS": N_CLUSTERS,
            "Z_CLUSTER":  Z_CLUSTER,
            "M_CLUSTER":  M_CLUSTER,
            "C_CLUSTER":  C_CLUSTER,
            "DENSITY":    DENSITY,
            "WIDTH_RAD":  WIDTH_RAD,
        },
        "radii_mpc":         RADII.tolist(),
        "sph_mean":          sph_mean.tolist(),
        "tri_mean":          tri_mean.tolist(),
        "sph_med":           sph_med.tolist(),
        "tri_med":           tri_med.tolist(),
        "sph_std":           sph_std.tolist(),
        "tri_std":           tri_std.tolist(),
        "frac_resid_mean":   frac_mean.tolist(),
        "frac_resid_median": frac_med.tolist(),
        "frac_resid_p16":    frac_p16.tolist(),
        "frac_resid_p84":    frac_p84.tolist(),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(OUT_DIR, "profiles.npz"),
        radii=RADII, ds_sph=ds_sph, ds_tri=ds_tri,
        q_ba=q_ba, q_ca=q_ca, theta_los=theta, phi_los=phi, T=T,
    )

    # ── Plot 1: mean profiles ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i in range(N_CLUSTERS):
        ax.plot(RADII, ds_tri[i], color="steelblue", alpha=0.08, lw=0.8)
    ax.plot(RADII, sph_mean, "o-", color="black", lw=2.2, ms=7,
            label="spherical NFW (population mean)")
    ax.plot(RADII, tri_mean, "s-", color="steelblue", lw=2.2, ms=7,
            label="triaxial NFW (population mean)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R\;[\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\Delta\Sigma\;[M_\odot/\mathrm{pc}^2]$")
    ax.set_title(rf"$\Delta\Sigma$ profiles  "
                 rf"(z={Z_CLUSTER}, M={M_CLUSTER:.1e}, c={C_CLUSTER}, "
                 rf"shape noise = 0)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "profiles.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: fractional residual ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.fill_between(RADII, 100*frac_p16, 100*frac_p84,
                    color="steelblue", alpha=0.25, label="68% per-cluster band")
    ax.plot(RADII, 100*frac_med, "s-", color="steelblue", lw=2.0, ms=6,
            label="median over clusters")
    ax.plot(RADII, 100*frac_mean, "o--", color="black", lw=1.6, ms=5,
            label="mean over clusters")
    ax.axhline(0, color="grey", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$R\;[\mathrm{Mpc}]$")
    ax.set_ylabel(r"$(\Delta\Sigma_{\rm tri}-\Delta\Sigma_{\rm sph})/\Delta\Sigma_{\rm sph}\;[\%]$")
    ax.set_title("Per-cluster fractional difference, triaxial vs spherical",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fractional_residual.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: split by cos(theta_LOS) ────────────────────────────────────
    cos_theta = np.cos(theta)
    bins = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    labels = [r"edge-on ($\cos\theta<0.33$)",
              r"intermediate ($0.33\leq\cos\theta<0.66$)",
              r"face-on ($\cos\theta\geq 0.66$)"]
    colors = ["#cc0033", "#aa6600", "#003388"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for (lo, hi), lab, col in zip(bins, labels, colors):
        mask = (cos_theta >= lo) & (cos_theta < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        m = np.nanmean(frac_resid[mask], axis=0) * 100
        ax.plot(RADII, m, "o-", color=col, lw=2.0, ms=6,
                label=f"{lab}  (N={n})")
    ax.axhline(0, color="grey", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$R\;[\mathrm{Mpc}]$")
    ax.set_ylabel(r"mean $(\Delta\Sigma_{\rm tri}-\Delta\Sigma_{\rm sph})/\Delta\Sigma_{\rm sph}\;[\%]$")
    ax.set_title("Fractional residual split by line-of-sight orientation",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "by_orientation.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: split by triaxiality T ─────────────────────────────────────
    bins_T = [(0.0, 1/3), (1/3, 2/3), (2/3, 1.01)]
    labels_T = [r"oblate ($T<1/3$)",
                r"triaxial ($1/3\leq T\leq 2/3$)",
                r"prolate ($T>2/3$)"]
    colors_T = ["#003388", "#888888", "#cc0033"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for (lo, hi), lab, col in zip(bins_T, labels_T, colors_T):
        mask = (T >= lo) & (T < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        m = np.nanmean(frac_resid[mask], axis=0) * 100
        ax.plot(RADII, m, "o-", color=col, lw=2.0, ms=6,
                label=f"{lab}  (N={n})")
    ax.axhline(0, color="grey", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$R\;[\mathrm{Mpc}]$")
    ax.set_ylabel(r"mean $(\Delta\Sigma_{\rm tri}-\Delta\Sigma_{\rm sph})/\Delta\Sigma_{\rm sph}\;[\%]$")
    ax.set_title("Fractional residual split by intrinsic triaxiality T",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "by_triaxiality.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote results to {OUT_DIR}/")


if __name__ == "__main__":
    main()
