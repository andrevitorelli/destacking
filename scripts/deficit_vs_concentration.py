"""
Sphere-vs-triaxial ΔΣ deficit as a function of concentration.

For each c in C_VALUES we repeat the sphere_vs_triaxial.py measurement at
fixed (z=0.4, M=10^14 M_sun) with the SAME triaxial shape draws and the
SAME background galaxy catalogue per cluster.  Only the NFW concentration
in the shear application changes across c values — both spherical and
triaxial paths see it.  This isolates the c-dependence of the deficit:

    deficit(c)  =  ⟨ ΔΣ_tri − ΔΣ_sph ⟩ / ⟨ ΔΣ_sph ⟩   averaged over R

Output (under sphere_vs_triaxial_outputs/):
  deficit_vs_concentration.png   — mean / 68% band / orientation splits
  deficit_vs_c.npz               — per-(cluster, c) profiles
"""

from __future__ import annotations
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as _planck18

from xlensing.sims import (
    apply_NFW_shear_region,
    apply_triaxial_NFW_shear_region,
    make_simple_random_cat,
    sample_triaxial_shapes,
)
from xlensing.data import metacal_cluster_lensing, signal


# ── Fixed cluster identity, varying c ──────────────────────────────────────
N_CLUSTERS  = 50
Z_CLUSTER   = 0.4
M_CLUSTER   = 1.0e14
C_VALUES    = np.geomspace(0.5, 30.0, 13)   # log-spaced over the test range
R_MIN_FIT   = 0.5                            # only average over the fit range

# Background galaxy field
DENSITY     = 30
WIDTH_RAD   = 0.006
Z_SRC_RANGE = (0.6, 2.5)

RADII  = np.logspace(-0.8, 0.8, 12)
_log_r = np.log10(RADII)
_dlog  = np.diff(_log_r)[0]
_edges = np.logspace(_log_r[0] - _dlog/2, _log_r[-1] + _dlog/2, len(RADII) + 1)
BIN_LIMITS = np.column_stack([_edges[:-1], _edges[1:]])

OUT_DIR = "sphere_vs_triaxial_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
MASTER_SEED = 20260526


def make_galaxies(seed):
    """Zero-shape-noise background-galaxy catalogue (same call as sphere_vs_triaxial.py)."""
    return make_simple_random_cat(
        DENSITY, WIDTH_RAD, Z_SRC_RANGE, shape_noise=0.0,
        seed=seed, poisson_density=False,
    ).T


def measure_dsigma(cluster_arr, base_galaxies, shear_fn, width_mpc):
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
    print(f"z={Z_CLUSTER}, M={M_CLUSTER:.2e}, N={N_CLUSTERS} clusters, "
          f"{len(C_VALUES)} c-values, R_fit ≥ {R_MIN_FIT} Mpc")
    print(f"C_VALUES = {[f'{c:.2f}' for c in C_VALUES]}")

    # Single shape draw — same shapes used for every c value.
    shapes = sample_triaxial_shapes(
        np.full(N_CLUSTERS, Z_CLUSTER),
        np.full(N_CLUSTERS, M_CLUSTER),
        rng=rng,
    )
    q_ba   = shapes["q_ba"];   q_ca = shapes["q_ca"]
    theta  = shapes["theta_los"]
    a_axis = shapes["r_vir"]
    T = (1 - q_ba**2) / (1 - q_ca**2 + 1e-12)
    cos_theta = np.cos(theta)

    # Single seed per cluster — same galaxy field reused across c values.
    gal_seeds = rng.integers(1, 2**31 - 1, size=N_CLUSTERS)

    n_R = len(RADII)
    keep = RADII >= R_MIN_FIT
    ds_sph = np.zeros((len(C_VALUES), N_CLUSTERS, n_R))
    ds_tri = np.zeros_like(ds_sph)

    t0 = time.time()
    for ic, c in enumerate(C_VALUES):
        for i in range(N_CLUSTERS):
            galaxies = make_galaxies(int(gal_seeds[i]))

            cluster_sph = [0.0, 0.0, Z_CLUSTER, M_CLUSTER, float(c)]
            ds_sph[ic, i] = measure_dsigma(cluster_sph, galaxies,
                                            apply_NFW_shear_region, width_mpc)

            a = a_axis[i]; b = q_ba[i] * a; c_ax = q_ca[i] * a
            cluster_tri = [0.0, 0.0, Z_CLUSTER, M_CLUSTER, float(c),
                           a, b, c_ax, float(theta[i]),
                           float(shapes["phi_los"][i])]
            ds_tri[ic, i] = measure_dsigma(cluster_tri, galaxies,
                                            apply_triaxial_NFW_shear_region,
                                            width_mpc)
        dt = time.time() - t0
        eta = dt / (ic + 1) * (len(C_VALUES) - ic - 1)
        print(f"  c={c:.2f}  [{ic+1}/{len(C_VALUES)}]  "
              f"elapsed {dt:.0f}s  eta {eta:.0f}s", flush=True)

    # Per-cluster fractional residual averaged over the fit range
    # frac[c, i] = mean_k_in_keep( (tri - sph) / sph ).
    frac_per_R = (ds_tri - ds_sph) / np.where(ds_sph != 0, ds_sph, np.nan)
    frac_avg   = np.nanmean(frac_per_R[:, :, keep], axis=2)    # (n_c, N)

    mean_def = np.nanmean(frac_avg, axis=1) * 100
    med_def  = np.nanmedian(frac_avg, axis=1) * 100
    p16      = np.nanpercentile(frac_avg, 16, axis=1) * 100
    p84      = np.nanpercentile(frac_avg, 84, axis=1) * 100

    # Orientation splits (same shapes/orientations across all c values)
    bins_o = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    labels_o = [r"edge-on ($\cos\theta<0.33$)",
                r"intermediate",
                r"face-on ($\cos\theta\ge 0.66$)"]
    colors_o = ["#cc0033", "#aa6600", "#003388"]
    mean_orient = []
    for (lo, hi) in bins_o:
        m = (cos_theta >= lo) & (cos_theta < hi)
        if m.sum() == 0:
            mean_orient.append(np.full(len(C_VALUES), np.nan))
        else:
            mean_orient.append(np.nanmean(frac_avg[:, m], axis=1) * 100)

    # Save numerical output
    np.savez_compressed(
        os.path.join(OUT_DIR, "deficit_vs_c.npz"),
        c_values=C_VALUES, radii=RADII,
        ds_sph=ds_sph, ds_tri=ds_tri,
        frac_per_R=frac_per_R, frac_avg=frac_avg,
        cos_theta=cos_theta, T=T, q_ba=q_ba, q_ca=q_ca,
    )
    print(f"\nSaved deficit_vs_c.npz")

    print(f"\n{'c':>7} {'mean[%]':>10} {'median[%]':>11} "
          f"{'p16[%]':>9} {'p84[%]':>9}")
    for ic, c in enumerate(C_VALUES):
        print(f"{c:>7.2f} {mean_def[ic]:>+10.2f} {med_def[ic]:>+11.2f} "
              f"{p16[ic]:>+9.2f} {p84[ic]:>+9.2f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_between(C_VALUES, p16, p84, alpha=0.18, color="steelblue",
                    label="68% per-cluster band")
    ax.plot(C_VALUES, med_def, "s-", color="steelblue", lw=2.0, ms=7,
            label="median over clusters", zorder=4)
    ax.plot(C_VALUES, mean_def, "o--", color="black", lw=1.6, ms=6,
            label="mean over clusters", zorder=5)
    for lab, col, ys in zip(labels_o, colors_o, mean_orient):
        if np.all(np.isnan(ys)):
            continue
        ax.plot(C_VALUES, ys, "^:", color=col, lw=1.4, ms=6, alpha=0.9,
                label=lab)
    ax.axhline(0, color="grey", ls="--", lw=1.0)
    # Physical c range
    ax.axvspan(1.0, 10.0, alpha=0.08, color="green", zorder=0)
    ax.text(np.sqrt(10.0), ax.get_ylim()[1]*0.95,
            "physical range  $c\\in[1,10]$",
            color="darkgreen", fontsize=9, ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlabel(r"NFW concentration $c$", fontsize=12)
    ax.set_ylabel(r"$\langle(\Delta\Sigma_{\rm tri}-\Delta\Sigma_{\rm sph})/\Delta\Sigma_{\rm sph}\rangle_R\;[\%]$",
                  fontsize=11)
    ax.set_title(
        rf"Triaxial-vs-spherical $\Delta\Sigma$ deficit vs concentration  "
        rf"(z={Z_CLUSTER}, M={M_CLUSTER:.0e} $M_\odot$, $N={N_CLUSTERS}$ shape draws)",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "deficit_vs_concentration.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
