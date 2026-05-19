"""
Generate the distribution figures referenced in report_2026-05-18.tex.

Reproduces the inline distribution plots from
notebooks/combined_recovery_lensing_triaxial_unbiased.ipynb at a representative
sample size so the report can include them without running the full sampler.

Output PNGs are written to notebooks/.
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as _planck18

from destacking.simulation import generate_truth_catalog
from destacking.relations import c_DuttonMaccio
from destacking.model import _MC_M_PIVOT_LOG, _MC_Z_REF

from xlensing.sims import r_vir, project_ellipsoid

# ---- Notebook configuration (a smaller N for fast figure generation) ----
N_CLUSTERS = int(os.environ.get("N_CLUSTERS", 4000))
N_OVERSAMPLE = 5
TRUE_GAMMA = 0.2
DELTA_M_REF = 0.1
DES_Y3_Z_CLUSTER_RANGE = (0.2, 0.65)
DES_Y3_Z_SRC_ALPHA = 2.0
DES_Y3_Z_SRC_BETA = 1.5
DES_Y3_Z_SRC_Z0 = 0.4
F_CEN = 0.64
SIGMA_OFF_MPC_H = 0.32
h = float(_planck18.h)
SIGMA_OFF_MPC = SIGMA_OFF_MPC_H / h

SEED = 42

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)


def des_y3_cluster_pdf(z, z_min=0.2, z_max=0.65):
    z = np.atleast_1d(z).astype(float)
    in_range = (z >= z_min) & (z <= z_max)
    pdf = np.where(in_range, z**2, 0.0)
    z_grid = np.linspace(z_min, z_max, 1000)
    norm = np.trapz(z_grid**2, z_grid)
    return pdf / norm


def des_y3_source_pdf(z, alpha=2.0, beta=1.5, z0=0.4):
    z = np.atleast_1d(z).astype(float)
    pdf = np.where(z > 0, (z / z0) ** alpha * np.exp(-((z / z0) ** beta)), 0.0)
    z_grid = np.linspace(1e-3, 5.0, 5000)
    norm = np.trapz((z_grid / z0) ** alpha * np.exp(-((z_grid / z0) ** beta)), z_grid)
    return pdf / norm


# ───────────────────────── 1. DES Y3 n(z) figure ─────────────────────────
def fig_des_y3_nz():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    z_cl = np.linspace(*DES_Y3_Z_CLUSTER_RANGE, 400)
    axes[0].fill_between(
        z_cl, des_y3_cluster_pdf(z_cl, *DES_Y3_Z_CLUSTER_RANGE),
        color="steelblue", alpha=0.5,
    )
    axes[0].plot(z_cl, des_y3_cluster_pdf(z_cl, *DES_Y3_Z_CLUSTER_RANGE),
                 color="steelblue", lw=2)
    axes[0].set_xlabel("Cluster z")
    axes[0].set_ylabel("n(z)")
    axes[0].set_title(f"DES Y3 cluster n(z)  (volume-weighted on {DES_Y3_Z_CLUSTER_RANGE})")

    z_sr = np.linspace(0.01, 2.5, 600)
    axes[1].fill_between(
        z_sr,
        des_y3_source_pdf(z_sr, DES_Y3_Z_SRC_ALPHA, DES_Y3_Z_SRC_BETA, DES_Y3_Z_SRC_Z0),
        color="tomato", alpha=0.5,
    )
    axes[1].plot(
        z_sr,
        des_y3_source_pdf(z_sr, DES_Y3_Z_SRC_ALPHA, DES_Y3_Z_SRC_BETA, DES_Y3_Z_SRC_Z0),
        color="tomato", lw=2,
    )
    axes[1].set_xlabel("Source z")
    axes[1].set_ylabel("n(z)")
    axes[1].set_title(
        rf"DES Y3 source n(z)  (Smail: $\alpha={DES_Y3_Z_SRC_ALPHA}$, "
        rf"$\beta={DES_Y3_Z_SRC_BETA}$, $z_0={DES_Y3_Z_SRC_Z0}$)"
    )

    plt.tight_layout()
    out = f"{OUT}/des_y3_nz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out)


# ─────────────────────── 2. Truth catalogue + scatter ────────────────────
def build_truth():
    print(f"Building truth catalogue with N={N_CLUSTERS} (oversample x{N_OVERSAMPLE})...")
    _N_POOL = N_CLUSTERS * N_OVERSAMPLE
    truth_pool = generate_truth_catalog(
        _N_POOL,
        z_range=DES_Y3_Z_CLUSTER_RANGE,
        logM_range=(13.5, 15.0),
        delta_m_gamma=0.0,
        scatter_logc=0.0,
        seed=SEED,
    )
    z_pool = np.asarray(truth_pool["z"])
    w_pool = des_y3_cluster_pdf(z_pool, *DES_Y3_Z_CLUSTER_RANGE)
    w_pool /= w_pool.sum()
    rng_res = np.random.default_rng(SEED + 100)
    idx = rng_res.choice(_N_POOL, size=N_CLUSTERS, replace=False, p=w_pool)
    truth = truth_pool[np.sort(idx)]

    dm_s = np.asarray(truth["delta_m"])
    logdm_s = np.log10(np.maximum(dm_s, 1e-6) / DELTA_M_REF)
    truth["concentration"] = (
        np.asarray(truth["concentration"]) * 10.0 ** (TRUE_GAMMA * logdm_s)
    )

    logM_s = np.log10(truth["M_halo"])
    logc_s = np.log10(truth["concentration"])
    logz1_s = np.log10((1.0 + truth["z"]) / (1.0 + _MC_Z_REF))
    X = np.column_stack([np.ones(N_CLUSTERS), logM_s - _MC_M_PIVOT_LOG, logz1_s, logdm_s])
    coeffs, *_ = np.linalg.lstsq(X, logc_s, rcond=None)
    logc0_true, alpha_true, beta_true, gamma_true = coeffs
    c0_true = 10.0**logc0_true
    return truth, (c0_true, alpha_true, beta_true, gamma_true, logM_s, logdm_s)


def fig_truth_distributions(truth):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].hist(truth["z"], bins=20, color="steelblue", edgecolor="white")
    axes[0, 0].set_xlabel("Redshift z")
    axes[0, 1].hist(np.log10(truth["M_halo"]), bins=20, color="steelblue", edgecolor="white")
    axes[0, 1].set_xlabel(r"$\log_{10}(M_{\rm halo})$")
    axes[0, 1].set_yscale("log")
    axes[1, 0].hist(truth["concentration"], bins=20, color="steelblue", edgecolor="white")
    axes[1, 0].set_xlabel("Concentration")
    axes[1, 1].hist(truth["delta_m"], bins=20, color="steelblue", edgecolor="white")
    axes[1, 1].set_xlabel(r"Magnitude gap $\Delta m$ [mag]")
    axes[1, 1].set_yscale("log")
    plt.suptitle(
        rf"Truth catalogue  ($N={N_CLUSTERS}$, $\gamma_{{\rm true}}={TRUE_GAMMA}$)",
        y=0.99,
    )
    plt.tight_layout()
    out = f"{OUT}/truth_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out)


def fig_mc_scatter(truth, ols):
    c0_true, alpha_true, beta_true, gamma_true, logM_s, logdm_s = ols

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sc = ax.scatter(
        logM_s, truth["concentration"], c=truth["delta_m"], cmap="plasma",
        s=8, alpha=0.6, vmin=0, vmax=4, rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label=r"$\Delta m$ [mag]")
    z_med = float(np.median(truth["z"]))
    M_grid = np.logspace(13.5, 15.0, 200)
    C_dm0 = c_DuttonMaccio(z_med * np.ones(200), M_grid, c=1.0)
    ax.plot(np.log10(M_grid), C_dm0, "k-", lw=2,
            label=rf"DM14 ($z={z_med:.2f}$, $\Delta m={DELTA_M_REF}$ mag)")
    ax.set_xlabel(r"$\log_{10}(M_{\rm halo})$", fontsize=12)
    ax.set_ylabel(r"Concentration $c_{200}$", fontsize=12)
    ax.set_title("M-c coloured by magnitude gap", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    sc2 = ax.scatter(
        truth["delta_m"], truth["concentration"], c=logM_s, cmap="viridis",
        s=8, alpha=0.6, rasterized=True,
    )
    plt.colorbar(sc2, ax=ax, label=r"$\log_{10}(M_{\rm halo})$")
    ax.set_xlabel(r"Magnitude gap $\Delta m$ [mag]", fontsize=12)
    ax.set_ylabel(r"Concentration $c_{200}$", fontsize=12)
    ax.set_title("Concentration vs magnitude gap", fontsize=11)

    plt.suptitle(
        rf"Truth: $c_0={c0_true:.2f}$, $\alpha={alpha_true:.3f}$, "
        rf"$\beta={beta_true:.3f}$, $\gamma={gamma_true:.3f}$",
        y=1.02, fontsize=12,
    )
    plt.tight_layout()
    out = f"{OUT}/mc_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out)


# ───────────────────────── 3. Triaxial shape figure ──────────────────────
def fig_triaxial(truth):
    print("Computing triaxial shapes...")
    rng_tri = np.random.default_rng(SEED + 10)
    r_vir_vals = np.array(
        [r_vir(float(z), float(M)) for z, M in zip(truth["z"], truth["M_halo"])]
    )
    q_ba = rng_tri.uniform(0.5, 1.0, N_CLUSTERS)
    q_ca = rng_tri.uniform(0.3, q_ba)
    theta = np.arccos(rng_tri.uniform(0.0, 1.0, N_CLUSTERS))
    phi = rng_tri.uniform(0.0, 2 * np.pi, N_CLUSTERS)
    q_maj = np.zeros(N_CLUSTERS)
    q_min = np.zeros(N_CLUSTERS)
    for i in range(N_CLUSTERS):
        a = r_vir_vals[i]
        b, c = q_ba[i] * a, q_ca[i] * a
        q_maj[i], q_min[i], _ = project_ellipsoid(a, b, c, theta[i], phi[i])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(q_ba, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("b/a")
    axes[1].hist(q_ca, bins=20, color="steelblue", edgecolor="white")
    axes[1].set_xlabel("c/a")
    axes[2].hist(q_min / q_maj, bins=20, color="tomato", edgecolor="white")
    axes[2].set_xlabel(r"Projected $q_{\rm min}/q_{\rm maj}$")
    for ax in axes:
        ax.set_ylabel("Count")
    plt.suptitle("Triaxial shape distributions", y=1.02)
    plt.tight_layout()
    out = f"{OUT}/triaxial_shapes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out)


# ─────────────────────── 4. Miscentering offsets figure ──────────────────
def fig_miscentering():
    rng_mis = np.random.default_rng(SEED + 20)
    is_mis = rng_mis.uniform(0, 1, N_CLUSTERS) > F_CEN
    off_mag = np.where(
        is_mis, rng_mis.rayleigh(scale=SIGMA_OFF_MPC, size=N_CLUSTERS), 0.0
    )
    phi_mis = rng_mis.uniform(0, 2 * np.pi, N_CLUSTERS)
    n_mis = int(is_mis.sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(off_mag[is_mis], bins=25, color="tomato", edgecolor="white", density=True)
    r_grid = np.linspace(0, off_mag.max() * 1.1, 300)
    pdf_ray = r_grid / SIGMA_OFF_MPC**2 * np.exp(-r_grid**2 / (2 * SIGMA_OFF_MPC**2))
    axes[0].plot(r_grid, pdf_ray, "k-", lw=1.5,
                 label=rf"Rayleigh($\sigma={SIGMA_OFF_MPC:.3f}$ Mpc)")
    axes[0].set_xlabel("Offset magnitude [Mpc]")
    axes[0].set_ylabel("PDF")
    axes[0].set_title(f"Miscentred offset distribution  (N={n_mis})")
    axes[0].legend()

    for r_i, ph_i in zip(off_mag[is_mis][:200], phi_mis[is_mis][:200]):
        axes[1].plot(r_i * np.cos(ph_i), r_i * np.sin(ph_i), "o",
                     color="tomato", ms=2, alpha=0.4)
    axes[1].add_patch(
        plt.Circle((0, 0), SIGMA_OFF_MPC, fill=False, ls="--", color="k", lw=1)
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel(r"$\Delta$RA [Mpc]")
    axes[1].set_ylabel(r"$\Delta$Dec [Mpc]")
    axes[1].set_title("Offset positions on sky")
    plt.tight_layout()
    out = f"{OUT}/miscentering_offsets.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", out)


if __name__ == "__main__":
    fig_des_y3_nz()
    truth, ols = build_truth()
    fig_truth_distributions(truth)
    fig_mc_scatter(truth, ols)
    fig_triaxial(truth)
    fig_miscentering()
    print("Done.")
