"""Figures and a results table for the optical-selection study (2026-11-11 note).

  1. lxm_2026-11-11_optical_stage.png: the optical completeness against richness (SCALE II Fig. 8 points, the
     placeholder, OPT v2 and v2 x minimum-richness factor) and the implied P(optical | M) at z = 0.18.
  2. lxm_2026-11-11_prior_pileup.png: per-cluster mass priors w_ck of the p95 sample for each optical stage, at the
     placeholder joint fit's posterior median (same theta for all, so only the selection stage differs).
  3. lxm_2026-11-11_beta_gamma.png: beta, gamma, sigma_int for every sample/framework/optical stage available.
Also prints the results table (markdown) used in the note.

    python plot_optical_stage.py [--out DIR]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.stats import norm

import fit_escale_sel as fs

D = fs.D
F = fs.OUT
V1 = dict(version=1, lam0=1.0, logM_piv=14.1, b_lam=0.8, sig_lam=0.2, rho=0.3, x50=0.8, width=0.1, floor=0.0,
          lam_err=0.1, lam_min=None, w_min=0.1)
V2 = dict(fs.OPT, lam_min=None)
STAGES = [("placeholder", "", V1, "0.5"), ("Lia, no $\\lambda_{\\min}$", "_opt2", V2, "C3"),
          ("Lia, $\\lambda_{\\min}=3$", "_opt2lmin3", dict(V2, lam_min=3.0), "C1"),
          ("Lia, $\\lambda_{\\min}=5$", "_opt2lmin5", dict(V2, lam_min=5.0), "C0")]
SAMPLES = [("full", "all"), ("p90", "p90"), ("p95", "p95")]


def completeness(loglam, o):
    c = o["floor"] + (1 - o["floor"]) * 0.5 * (1 + erf((loglam - o["x50"]) / (np.sqrt(2) * o["width"])))
    if o["lam_min"] is not None:
        c = c * 0.5 * (1 + erf((loglam - np.log10(o["lam_min"])) / (np.sqrt(2) * o["w_min"])))
    return c


def p_opt_given_m(logm, o):
    """Marginal P(optical | M): completeness averaged over log lambda ~ N(mean(M), sig_lam^2 + lam_err^2)."""
    x = np.linspace(-1.5, 3.5, 2001)
    mu = o["lam0"] + o["b_lam"] * (logm - o["logM_piv"])
    s = np.hypot(o["sig_lam"], o["lam_err"])
    return np.trapezoid(completeness(x, o)[None] * norm.pdf(x[None], mu[:, None], s), x, axis=1)


def quant(path, name):
    d = np.load(path, allow_pickle=True)
    j = list(d["param_order"]).index(name)
    x, w = d["samples"][:, j], d["weights"]
    o = np.argsort(x)
    return np.interp([.16, .5, .84], np.cumsum(w[o]), x[o])


def boot_quant(prefix, name):
    meds = [quant(f, name)[1] for f in sorted(glob.glob(f"{prefix}_b[0-9][0-9][0-9].npz"))]
    return (np.percentile(meds, [16, 50, 84]), len(meds)) if len(meds) >= 10 else (None, len(meds))


def fig_stage(out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lia_x = np.array([0.43636214, 0.51861583, 0.60086953, 0.68312322, 0.76537692, 0.84763062, 0.92988431, 1.01213801,
                      1.09439171, 1.1766454, 1.2588991, 1.3411528, 1.42340649, 1.50566019, 1.58791389, 1.67016758,
                      1.75242128, 1.83467497])
    lia_y = np.array([0.625, 0.76190476, 0.77358491, 0.75257732, 0.79041916, 0.72580645, 0.69278997, 0.65588915,
                      0.53472222, 0.50483092, 0.43396226, 0.2810219, 0.20657277, 0.21126761, 0.21621622, 0.14285714,
                      0.06086957, 0.0])
    x = np.linspace(-0.2, 2.0, 400)
    ax[0].plot(lia_x, 1 - lia_y, "ko", ms=4, label="SCALE II Fig. 8 (1 − purple), Doubrawa priv. comm.")
    for lab, _, o, c in STAGES:
        ax[0].plot(x, completeness(x, o), color=c, lw=2, label=lab)
    ax[0].set(xlabel=r"$\log_{10}\lambda$ (AME, $M_r<-20$)", ylabel="optical completeness $C(\\lambda)$",
              ylim=(-0.03, 1.05))
    ax[0].legend(fontsize=8, loc="lower right")
    m = np.linspace(11.8, 15.5, 300)
    for lab, _, o, c in STAGES:
        ax[1].plot(m, p_opt_given_m(m, o), color=c, lw=2, label=lab)
    ax[1].set(xlabel=r"$\log_{10} M_{200c}\ [M_\odot]$", ylabel=r"$P(\mathrm{optical}\mid M)$", ylim=(-0.03, 1.05))
    ax[1].axvspan(11.8, 11.8 + 2 * (fs.GRID[1] - fs.GRID[0]), color="k", alpha=0.08, label="lowest 2 grid points")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/lxm_2026-11-11_optical_stage.png", dpi=150)


def fig_prior(out):
    p = {n: quant(f"{F}/joint_AB_p95.npz", n)[1] for n in ["alpha", "beta", "gamma", "sig_int"]}
    data = fs.load_real("p95")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for lab, _, o, c in STAGES:
        fs.OPT.clear(); fs.OPT.update(o)
        w = fs.weights(fs.make_prior("AB"), data["x"], p)
        ax[0].plot(fs.GRID, w.mean(0), color=c, lw=2, label=lab)
        floor = w[:, :2].sum(1)
        ax[1].hist(floor, bins=np.linspace(0, 0.3, 31), histtype="step", color=c, lw=2, label=lab)
        mean_m = (w * fs.GRID).sum(1)
        rows.append((lab, floor.max(), np.median(mean_m)))
    ax[0].set(xlabel=r"$\log_{10} M_{200c}\ [M_\odot]$", ylabel=r"mean over clusters of $w_{ck}$",
              title=f"p95 (N = {data['N']}), $\\theta$ = placeholder joint median")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="per-cluster prior weight in the lowest 2 grid points", ylabel="clusters", yscale="log")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/lxm_2026-11-11_prior_pileup.png", dpi=150)
    print("\n| optical stage | max floor weight | median prior-mean log M |\n|---|---|---|")
    for lab, fmax, mm in rows:
        print(f"| {lab} | {fmax:.3f} | {mm:.2f} |")


def fig_results(out):
    pars = [("beta", r"$\beta$"), ("gamma", r"$\gamma$"), ("sig_int", r"$\sigma_{\rm int}$")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    lines = ["| sample | N | optical stage | framework | β | γ | σ_int |", "|---|---|---|---|---|---|---|"]
    y = 0; ticks = []
    for sname, sub in SAMPLES:
        for lab, tag, o, c in STAGES:
            for mode, mk in [("forward", "s"), ("joint", "o")]:
                path = f"{F}/{mode}_AB_{sub}{tag}.npz"
                bq, nb = boot_quant(f"{F}/{mode}_AB_{sub}{tag}", "beta") if mode == "joint" else (None, 0)
                row = []
                for i, (pn, _) in enumerate(pars):
                    if bq is not None:
                        q, _ = boot_quant(f"{F}/{mode}_AB_{sub}{tag}", pn)
                    elif os.path.exists(path):
                        q = quant(path, pn)
                    else:
                        q = None
                    if q is not None:
                        ax[i].errorbar(q[1], y, xerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt=mk, color=c,
                                       mfc=c if mode == "joint" else "white", capsize=2)
                    row.append("—" if q is None else f"{q[1]:+.2f} +{q[2] - q[1]:.2f} −{q[1] - q[0]:.2f}")
                if any(r != "—" for r in row):
                    n = int(np.load(path)["n_clusters"]) if os.path.exists(path) else {"all": 1556, "p90": 1024, "p95": 496}[sub]
                    lines.append(f"| {sname} | {n} | {lab.replace('$', '').replace(chr(92), '')} | {mode}"
                                 + (f" (b, {nb})" if bq is not None else "") + " | " + " | ".join(row) + " |")
                ticks.append((y, f"{sname} {mode}"))
                y -= 1
        y -= 0.8
    for a, (pn, pl) in zip(ax, pars):
        a.set_xlabel(pl)
    ax[0].axvline(1.11, color="k", ls=":", lw=1); ax[1].axvline(0.0, color="k", ls=":", lw=1)
    ax[0].axvspan(0.96, 1.26, color="k", alpha=0.07, label="Pederneiras+25 eRASS1")
    ax[1].axvspan(-0.79, 0.79, color="k", alpha=0.07)
    ax[0].set_yticks([t for t, _ in ticks]); ax[0].set_yticklabels([l for _, l in ticks], fontsize=7)
    ax[1].set_xlim(-15, 15)
    hs = [plt.Line2D([], [], color=c, lw=2, label=lab) for lab, _, _, c in STAGES]
    hs += [plt.Line2D([], [], color="k", marker="s", mfc="white", ls="", label="forward stacks"),
           plt.Line2D([], [], color="k", marker="o", ls="", label="joint (filled; bootstrap if ≥10 draws)")]
    fig.legend(handles=hs, fontsize=8, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{out}/lxm_2026-11-11_beta_gamma.png", dpi=150)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/figures")
    ap.add_argument("--skip-prior", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_stage(a.out)
    fig_results(a.out)
    if not a.skip_prior:
        fig_prior(a.out)
