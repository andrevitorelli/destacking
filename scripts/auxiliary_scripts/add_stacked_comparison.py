"""
Add a stacked-cluster comparison to the sphere_vs_triaxial outputs.

For identical-noise, identical-(M, z) clusters the *stacked* measurement is
the population mean of the per-cluster signals.  This is what a traditional
weak-lensing stacking analysis would obtain.  We then fit a spherical NFW
model to the stacked triaxial signal and recover an effective concentration
c_eff, quantifying the bias that a stacking analysis would inherit from
triaxiality.

Updates the following plots (data already exists, no shear re-computation):

  profiles.png                — add stacked triaxial + best-fit spherical
                                NFW (c_eff)
  fractional_residual.png     — add stacked deficit + best-fit-NFW deviation
  deficit_vs_concentration.png — 2-panel: original deficit (top) + c_eff
                                 vs c_input (bottom)
"""

from __future__ import annotations
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from destacking.model import _nfw_profiles


Z_CLUSTER = 0.4
M_CLUSTER = 1.0e14
R_MIN_FIT = 0.5         # same cut the inference pipeline uses
OUT_DIR   = "sphere_vs_triaxial_outputs"


def spherical_nfw(R, c, M=M_CLUSTER, z=Z_CLUSTER):
    """One-halo centered spherical NFW ΔΣ at radii R for (M, c, z)."""
    return _nfw_profiles(
        np.array([z]), np.array([M]),
        np.array([max(float(c), 0.05)]),
        R,
        FMISS=np.zeros(1), SIGMA_OFF=np.full(1, 0.1), BCG_B_MASS=np.zeros(1),
    ).flatten()


def fit_c(R, ds_stack, sigma=None, c0=5.0):
    """Fit a spherical NFW (M, z fixed) to a stacked ΔΣ; return (c, σ_c)."""
    try:
        popt, pcov = curve_fit(spherical_nfw, R, ds_stack, p0=[c0],
                               sigma=sigma, absolute_sigma=False,
                               bounds=(0.05, 100.0))
        return float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception as exc:
        print(f"    fit failed: {exc}")
        return float("nan"), float("nan")


# ── profiles.png + fractional_residual.png (single-c case) ──────────────────

def update_single_c_plots():
    npz = os.path.join(OUT_DIR, "profiles.npz")
    if not os.path.exists(npz):
        print(f"missing {npz} — skip")
        return None
    d = np.load(npz)
    R = d["radii"]; ds_sph = d["ds_sph"]; ds_tri = d["ds_tri"]
    N = ds_sph.shape[0]

    sph_stack = np.nanmean(ds_sph, axis=0)
    tri_stack = np.nanmean(ds_tri, axis=0)
    tri_sem   = np.nanstd(ds_tri, axis=0, ddof=1) / np.sqrt(N)

    keep = R >= R_MIN_FIT
    c_check, _      = fit_c(R[keep], sph_stack[keep], c0=5.0)
    c_eff, c_eff_e  = fit_c(R[keep], tri_stack[keep], sigma=tri_sem[keep], c0=5.0)
    fit_model       = spherical_nfw(R, c_eff)
    print(f"\n[c=5 single-c data, fitting only R >= {R_MIN_FIT} Mpc]")
    print(f"  fit-to-spherical-stack  c = {c_check:.3f}  (input = 5.0)")
    print(f"  fit-to-triaxial-stack   c_eff = {c_eff:.3f} ± {c_eff_e:.3f}  "
          f"(bias vs input 5.0: {c_eff - 5.0:+.3f})")

    # ── profiles.png ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.8))
    for i in range(N):
        ax.plot(R, ds_tri[i], color="steelblue", alpha=0.07, lw=0.7)
    ax.plot(R, sph_stack, "o-", color="black", lw=2.2, ms=7,
            label=r"spherical NFW @ $c=5$ (stacked $=$ single cluster)")
    ax.plot(R, tri_stack, "s-", color="steelblue", lw=2.2, ms=7,
            label="stacked triaxial NFW (mean over orientations)")
    ax.plot(R, fit_model, "--", color="crimson", lw=2.2,
            label=rf"best-fit spherical NFW to stack:  "
                  rf"$c_{{\rm eff}}={c_eff:.2f}\pm{c_eff_e:.2f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$R\;[\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\Delta\Sigma\;[M_\odot/\mathrm{pc}^2]$")
    ax.set_title(rf"$\Delta\Sigma$ profiles  "
                 rf"($z={Z_CLUSTER}$, $M={M_CLUSTER:.0e}$, $c=5$, shape noise=0, $N={N}$)",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "profiles.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(OUT_DIR, 'profiles.png')}")

    # ── fractional_residual.png ────────────────────────────────────────
    frac = (ds_tri - ds_sph) / np.where(ds_sph != 0, ds_sph, np.nan)
    f_mean = np.nanmean(frac, axis=0) * 100
    f_med  = np.nanmedian(frac, axis=0) * 100
    f_p16  = np.nanpercentile(frac, 16, axis=0) * 100
    f_p84  = np.nanpercentile(frac, 84, axis=0) * 100
    stack_frac = (tri_stack - sph_stack) / sph_stack * 100
    fit_frac   = (fit_model - sph_stack) / sph_stack * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(R, f_p16, f_p84, color="steelblue", alpha=0.22,
                    label="68% per-cluster band")
    ax.plot(R, f_med, "s-", color="steelblue", lw=2.0, ms=6,
            label="median per-cluster")
    ax.plot(R, f_mean, "o--", color="black", lw=1.5, ms=5,
            label="mean per-cluster")
    ax.plot(R, stack_frac, "D-", color="darkgreen", lw=2.4, ms=7,
            label=r"stacked triaxial vs spherical $c=5$")
    ax.plot(R, fit_frac, ":", color="crimson", lw=2.4,
            label=rf"best-fit spherical NFW ($c_{{\rm eff}}={c_eff:.2f}$) vs $c=5$")
    ax.axhline(0, color="grey", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$R\;[\mathrm{Mpc}]$")
    ax.set_ylabel(r"$(\Delta\Sigma - \Delta\Sigma_{\rm sph, c=5})/\Delta\Sigma_{\rm sph, c=5}\;[\%]$")
    ax.set_title("Fractional residual vs spherical-NFW($c=5$) reference",
                 fontsize=10)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fractional_residual.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(OUT_DIR, 'fractional_residual.png')}")
    return c_eff, c_eff_e


# ── deficit_vs_concentration.png with c_eff vs c_input ─────────────────────

def update_deficit_plot():
    npz = os.path.join(OUT_DIR, "deficit_vs_c.npz")
    if not os.path.exists(npz):
        print(f"\nmissing {npz} — skip")
        return None
    d = np.load(npz)
    C_VALUES  = d["c_values"]
    R         = d["radii"]
    ds_sph    = d["ds_sph"]
    ds_tri    = d["ds_tri"]
    cos_theta = d["cos_theta"]
    keep = R >= R_MIN_FIT

    print(f"\n[deficit-vs-c data, {len(C_VALUES)} c values, fitting only R >= {R_MIN_FIT} Mpc]")
    c_eff_arr     = np.zeros(len(C_VALUES))
    c_eff_err_arr = np.zeros(len(C_VALUES))
    for ic, c in enumerate(C_VALUES):
        tri_stack = np.nanmean(ds_tri[ic], axis=0)
        tri_sem   = np.nanstd(ds_tri[ic], axis=0, ddof=1) / np.sqrt(ds_tri.shape[1])
        c_eff_arr[ic], c_eff_err_arr[ic] = fit_c(
            R[keep], tri_stack[keep], sigma=tri_sem[keep], c0=float(c)
        )
        print(f"  c_in={c:>6.2f}  c_eff={c_eff_arr[ic]:>6.3f}±{c_eff_err_arr[ic]:.3f}  "
              f"bias={c_eff_arr[ic]-c:>+7.3f}  "
              f"frac={(c_eff_arr[ic]-c)/c*100:>+6.1f}%")

    # ── recompute the deficit panel data ─────────────────────────────────
    frac_per_R = (ds_tri - ds_sph) / np.where(ds_sph != 0, ds_sph, np.nan)
    frac_avg = np.nanmean(frac_per_R[:, :, keep], axis=2)
    mean_def = np.nanmean(frac_avg, axis=1) * 100
    med_def  = np.nanmedian(frac_avg, axis=1) * 100
    p16      = np.nanpercentile(frac_avg, 16, axis=1) * 100
    p84      = np.nanpercentile(frac_avg, 84, axis=1) * 100

    bins_o   = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    labels_o = [r"edge-on ($\cos\theta<0.33$)",
                "intermediate",
                r"face-on ($\cos\theta\geq 0.66$)"]
    colors_o = ["#cc0033", "#aa6600", "#003388"]
    mean_orient = []
    for lo, hi in bins_o:
        m = (cos_theta >= lo) & (cos_theta < hi)
        if m.sum() == 0:
            mean_orient.append(np.full(len(C_VALUES), np.nan))
        else:
            mean_orient.append(np.nanmean(frac_avg[:, m], axis=1) * 100)

    # ── plot: 2-panel ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.4]},
    )

    # Top: fractional deficit (as before, with stacked label)
    ax1.fill_between(C_VALUES, p16, p84, alpha=0.18, color="steelblue",
                     label="68% per-cluster band")
    ax1.plot(C_VALUES, med_def, "s-", color="steelblue", lw=2.0, ms=7,
             label="median over clusters", zorder=4)
    ax1.plot(C_VALUES, mean_def, "o--", color="black", lw=1.6, ms=6,
             label="mean over clusters  ($\\equiv$ stacked deficit)", zorder=5)
    for lab, col, ys in zip(labels_o, colors_o, mean_orient):
        if not np.all(np.isnan(ys)):
            ax1.plot(C_VALUES, ys, "^:", color=col, lw=1.4, ms=6, alpha=0.9,
                     label=lab)
    ax1.axhline(0, color="grey", ls="--", lw=1.0)
    ax1.axvspan(1.0, 10.0, alpha=0.08, color="green", zorder=0)
    ax1.text(np.sqrt(10), ax1.get_ylim()[1] * 0.95
             if ax1.get_ylim()[1] > 0 else 0,
             r"physical range  $c\in[1,10]$",
             color="darkgreen", fontsize=9, ha="center", va="top")
    ax1.set_xscale("log")
    ax1.set_ylabel(
        r"$\langle(\Delta\Sigma_{\rm tri}-\Delta\Sigma_{\rm sph})/\Delta\Sigma_{\rm sph}\rangle_R\;[\%]$",
        fontsize=10,
    )
    ax1.set_title(
        "Triaxial-vs-spherical $\\Delta\\Sigma$ deficit (top) and "
        "concentration inferred from stacking (bottom)",
        fontsize=11,
    )
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(alpha=0.3, which="both")

    # Bottom: c_eff vs c_input
    ax2.plot([C_VALUES[0], C_VALUES[-1]], [C_VALUES[0], C_VALUES[-1]],
             "k--", lw=1.2, label=r"$c_{\rm eff}=c_{\rm input}$ (no stacking bias)")
    ax2.errorbar(C_VALUES, c_eff_arr, yerr=c_eff_err_arr, fmt="o-",
                 color="crimson", lw=2.0, ms=7, capsize=3,
                 label="best-fit spherical NFW to stacked triaxial")
    ax2.axvspan(1.0, 10.0, alpha=0.08, color="green", zorder=0)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"input concentration $c$", fontsize=12)
    ax2.set_ylabel(r"recovered $c_{\rm eff}$ from stacked fit", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "deficit_vs_concentration.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.join(OUT_DIR, 'deficit_vs_concentration.png')}")
    return c_eff_arr, c_eff_err_arr


if __name__ == "__main__":
    update_single_c_plots()
    update_deficit_plot()
