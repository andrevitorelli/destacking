"""
Characterise the simulated triaxial-halo shape distribution.

For one negative-γ simulation, re-sample the same triaxial shapes that
gamma_scan.py used (deterministic seed _seed(g_idx, 10)) and plot:
  * b/a and c/a marginals
  * the triaxiality parameter T = (a² − b²)/(a² − c²)
    (T<0.5 oblate-leaning, T>0.5 prolate-leaning)
  * the projected axis ratio q_min/q_maj on the sky
  * the line-of-sight direction (theta_los) distribution

Output:
  RUN_DIR/plateau/triaxiality_distribution.png
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from xlensing.sims import sample_triaxial_shapes


def _seed(g_idx, offset):
    return g_idx * 100_000 + int(offset)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 \
              else "gamma_scan_outputs/run_2026-05-24T10-19-32"
    sim_dir = os.path.join(run_dir, "simulation_g00_gamma-1.0")
    print(f"Using {sim_dir}")

    tc = np.load(os.path.join(sim_dir, "truth_catalog.npz"))
    z = np.asarray(tc["z"])
    M = np.asarray(tc["M_halo"])
    print(f"N={len(z)} clusters,  M_halo: median={np.median(M):.2e},  "
          f"z median={np.median(z):.3f}")

    # Reproduce shapes used in the run (g_idx=0, offset=10)
    rng = np.random.default_rng(_seed(0, 10))
    shapes = sample_triaxial_shapes(
        z, M, ba_range=(0.5, 1.0), ca_lo=0.3, isotropic_los=True, rng=rng,
    )

    q_ba      = shapes["q_ba"]
    q_ca      = shapes["q_ca"]
    theta_los = shapes["theta_los"]
    q_maj     = shapes["q_maj"]
    q_min     = shapes["q_min"]
    proj_ratio = q_min / q_maj

    # Triaxiality parameter: T = (a²-b²)/(a²-c²),  with a=1
    T = (1 - q_ba**2) / (1 - q_ca**2 + 1e-12)

    # Classify
    frac_oblate    = float(np.mean(T < 1/3))
    frac_triaxial  = float(np.mean((T >= 1/3) & (T <= 2/3)))
    frac_prolate   = float(np.mean(T > 2/3))

    print(f"\nAxis ratios:")
    print(f"  b/a:  median={np.median(q_ba):.3f}  "
          f"16%={np.percentile(q_ba,16):.3f}  84%={np.percentile(q_ba,84):.3f}")
    print(f"  c/a:  median={np.median(q_ca):.3f}  "
          f"16%={np.percentile(q_ca,16):.3f}  84%={np.percentile(q_ca,84):.3f}")
    print(f"\nTriaxiality parameter T (Franx+91 convention):")
    print(f"  T:    median={np.median(T):.3f}  "
          f"16%={np.percentile(T,16):.3f}  84%={np.percentile(T,84):.3f}")
    print(f"\nFractions:")
    print(f"  oblate    (T<1/3):  {frac_oblate:.3f}")
    print(f"  triaxial  (1/3<=T<=2/3): {frac_triaxial:.3f}")
    print(f"  prolate   (T>2/3):  {frac_prolate:.3f}")
    print(f"\nProjected sky axis ratio q_min/q_maj:")
    print(f"  median={np.median(proj_ratio):.3f}  "
          f"16%={np.percentile(proj_ratio,16):.3f}  "
          f"84%={np.percentile(proj_ratio,84):.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.hist(q_ba, bins=40, color="steelblue", alpha=0.7, label=r"$b/a$")
    ax.hist(q_ca, bins=40, color="crimson",   alpha=0.7, label=r"$c/a$")
    ax.set_xlabel("3D axis ratio")
    ax.set_ylabel("clusters / bin")
    ax.set_title("Intrinsic axis ratios", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(T, bins=40, color="darkorange", edgecolor="black", lw=0.4)
    ax.axvline(1/3, color="grey", ls="--", lw=1.2)
    ax.axvline(2/3, color="grey", ls="--", lw=1.2)
    ax.text(0.10, 0.92, f"oblate\n{frac_oblate:.0%}", transform=ax.transAxes,
            color="navy", fontsize=10, ha="left", va="top")
    ax.text(0.45, 0.92, f"triaxial\n{frac_triaxial:.0%}", transform=ax.transAxes,
            color="black", fontsize=10, ha="left", va="top")
    ax.text(0.80, 0.92, f"prolate\n{frac_prolate:.0%}", transform=ax.transAxes,
            color="maroon", fontsize=10, ha="left", va="top")
    ax.set_xlabel(r"Triaxiality $T=(a^2-b^2)/(a^2-c^2)$")
    ax.set_ylabel("clusters / bin")
    ax.set_title("Triaxiality parameter (Franx+91)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    h = ax.hist2d(q_ba, q_ca, bins=40, range=[[0.45, 1.0], [0.25, 1.0]],
                  cmap="viridis")
    ax.plot([0.3, 1.0], [0.3, 1.0], "w--", lw=1.2,
            label=r"$c/a = b/a$ (prolate limit)")
    ax.axhline(0.3, color="w", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$b/a$")
    ax.set_ylabel(r"$c/a$")
    ax.set_title(r"Joint distribution in $(b/a,\,c/a)$ plane", fontsize=10)
    ax.legend(fontsize=9, loc="lower right", facecolor="white",
              edgecolor="grey")
    plt.colorbar(h[3], ax=ax, label="clusters")

    ax = axes[1, 1]
    ax.hist(proj_ratio, bins=40, color="seagreen", edgecolor="black", lw=0.4)
    ax.set_xlabel(r"Projected $q_{\min}/q_{\max}$")
    ax.set_ylabel("clusters / bin")
    ax.set_title("Sky-projected axis ratio  (1=circular)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    fig.suptitle(
        r"Triaxial-shape distribution used in gamma_scan.py  "
        r"($b/a\!\sim\!U(0.5,1)$, $c/a\!\sim\!U(0.3,b/a)$, isotropic LOS)",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    out_dir = os.path.join(run_dir, "plateau")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "triaxiality_distribution.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
