"""Extrapolation of the L_X-M parameters to zero contamination ("p100") from the purity tiers (2026-11-26).

Main estimate: the three disjoint highest-flag tiers (95 = p95; 90 = P90 not P95; 80 = P80 only), independent samples,
each fitted alone (joint A+B, optical stage v2, lambda_min = 5, no mixture), against the expected spurious fraction
f = N_rand / N_real per tier from the randoms (contamination.json "tiers").  Weighted straight line per parameter,
weights 1/sigma^2 with sigma = half the 68% posterior interval; value at f = 0 with its 1-sigma error from the fit
covariance.  Cross-check: the overlapping samples p95 / p90 / full (correlated, so their line errors are too small).
Comparison: the slope d(param)/df from the injection test (p95 + n chance matches, means over draws), i.e. what pure
contamination does.

    python plot_extrapolation.py [--out DIR]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
F = f"{D}/fits_sel"
T = "opt2lmin5"
PARS = [("beta", r"$\beta$"), ("gamma", r"$\gamma$"), ("alpha", r"$\alpha$"), ("sig_int", r"$\sigma_{\rm int}$ [dex]")]


def quant(path, name):
    d = np.load(path, allow_pickle=True)
    po = list(d["param_order"])
    j = po.index(name)
    x, w = d["samples"][:, j], d["weights"]
    o = np.argsort(x)
    return np.interp([.16, .5, .84], np.cumsum(w[o]), x[o])


def boot_meds(prefix, name):
    """Medians of each cluster-bootstrap posterior (files <prefix>_b<NNN>.npz), or None if fewer than 10."""
    fs = [f for f in glob.glob(f"{prefix}_b*.npz") if re.fullmatch(r"_b\d+\.npz", f[len(prefix):])]
    return np.array([quant(f, name)[1] for f in fs]) if len(fs) >= 10 else None


def wline(x, y, s):
    """Weighted straight line y = a + b x; returns (a, b), covariance, chi2."""
    A = np.c_[np.ones(len(x)), x] / s[:, None]
    C = np.linalg.inv(A.T @ A)
    p = C @ A.T @ (y / s)
    return p, C, float(np.sum(((y - p[0] - p[1] * x) / s) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/figures")
    a = ap.parse_args()
    c = json.load(open(f"{D}/randoms/contamination.json"))
    ft = {int(k): v["q"] for k, v in c["tiers"].items()}
    tiers = [("tier 95 (= p95)", "p95", ft[95]), ("tier 90 (P90, not P95)", "t90", ft[90]), ("tier 80 (P80 only)", "t80", ft[80])]
    over = [("p95", "p95", 76 / 496), ("p90", "p90", c["ratio"]["90"]), ("full", "all", (76 + 265 + 353) / 1556)]
    inj = {}
    for f in glob.glob(f"{F}/joint_AB_p95_inj*s*_{T}.npz"):
        m = re.search(r"_inj(\d+)s(\d+)_" + T + r"\.npz$", f)
        if m:
            inj.setdefault(int(m.group(1)), []).append(f)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
    rows = ["| parameter | tiers: value at f = 0 | tiers: slope d/df | χ² (1 dof) | overlapping samples: value at f = 0 | injection: slope d/df |",
            "|---|---|---|---|---|---|"]
    xx = np.linspace(0, 0.8, 50)
    for i, (pn, pl) in enumerate(PARS):
        # tiers
        x = np.array([f for *_, f in tiers])
        bm = [boot_meds(f"{F}/joint_AB_{s}_{T}", pn) for _, s, _ in tiers]
        if all(b is not None for b in bm):     # bootstrap: median of medians, 16-84% spread; f = 0 value from random triples
            q = np.array([np.percentile(b, [16, 50, 84]) for b in bm])
            y = q[:, 1]; s = (q[:, 2] - q[:, 0]) / 2
            p, C, chi2 = wline(x, y, s)
            rng = np.random.default_rng(1)
            tri = np.array([[b[rng.integers(len(b))] for b in bm] for _ in range(20000)])
            A = np.c_[np.ones(3), x] / s[:, None]
            P = (np.linalg.inv(A.T @ A) @ A.T @ (tri / s).T).T
            t0 = np.percentile(P[:, 0], [16, 50, 84]); t1 = np.percentile(P[:, 1], [16, 50, 84])
            kind = "bootstrap"
        else:
            q = np.array([quant(f"{F}/joint_AB_{s}_{T}.npz", pn) for _, s, _ in tiers])
            y = q[:, 1]; s = (q[:, 2] - q[:, 0]) / 2
            p, C, chi2 = wline(x, y, s)
            t0 = np.array([p[0] - np.sqrt(C[0, 0]), p[0], p[0] + np.sqrt(C[0, 0])])
            t1 = np.array([p[1] - np.sqrt(C[1, 1]), p[1], p[1] + np.sqrt(C[1, 1])])
            kind = "single fits"
        ax[i].errorbar(x, y, yerr=[y - q[:, 0], q[:, 2] - y], fmt="o", color="C0", ms=8, mec="k", capsize=3,
                       label="disjoint tiers (independent; bootstrap median and 16–84%)" if i == 0 else None, zorder=5)
        band = np.sqrt(C[0, 0] + 2 * xx * C[0, 1] + xx ** 2 * C[1, 1])
        ax[i].plot(xx, p[0] + p[1] * xx, "-", color="C0")
        ax[i].fill_between(xx, p[0] + p[1] * xx - band, p[0] + p[1] * xx + band, color="C0", alpha=0.15)
        ax[i].errorbar(0, t0[1], yerr=[[t0[1] - t0[0]], [t0[2] - t0[1]]], fmt="*", color="C3", ms=14, mec="k", capsize=3, zorder=6,
                       label="extrapolated to f = 0" if i == 0 else None)
        # overlapping samples
        qo = np.array([quant(f"{F}/joint_AB_{s}_{T}.npz", pn) for _, s, _ in over])
        xo = np.array([f for *_, f in over]); so = (qo[:, 2] - qo[:, 0]) / 2
        po, Co, _ = wline(xo, qo[:, 1], so)
        ax[i].plot(xo + 0.01, qo[:, 1], "s", color="0.5", ms=6, label="p95 / p90 / full (overlapping)" if i == 0 else None)
        ax[i].plot(xx, po[0] + po[1] * xx, "--", color="0.5", lw=1)
        # injection slope, anchored at the p95 point
        xs, ys = [], []
        for n in sorted(inj):
            xs.append((76 + n) / (496 + n)); ys.append(np.mean([quant(f, pn)[1] for f in inj[n]]))
        if xs:
            xs = np.r_[ft[95], xs]; ys = np.r_[y[0], ys]
            pi = np.polyfit(xs, ys, 1)
            ax[i].plot(xs, ys, "^", color="C1", ms=6, label="p95 + injected chance matches (mean of draws)" if i == 0 else None)
            ax[i].plot(xx, np.polyval(pi, xx), ":", color="C1", lw=1.5)
            sl_inj = f"{pi[0]:+.2f}"
        else:
            sl_inj = "–"
        ax[i].set_xlabel("expected fraction of spurious matches f")
        ax[i].set_ylabel(pl)
        ax[i].set_xlim(-0.03, 0.8)
        rows.append(f"| {pn} | {t0[1]:+.2f} +{t0[2] - t0[1]:.2f} −{t0[1] - t0[0]:.2f} | {t1[1]:+.2f} +{t1[2] - t1[1]:.2f} −{t1[1] - t1[0]:.2f} | {chi2:.2f} | "
                    f"{po[0]:+.2f} ± {np.sqrt(Co[0, 0]):.2f} (too small: correlated) | {sl_inj} |")
    ax[0].axhspan(0.96, 1.26, color="k", alpha=0.07, label="Pederneiras+25 eRASS1")
    ax[1].axhspan(-0.79, 0.79, color="k", alpha=0.07)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    N = {s: int(np.load(f"{F}/joint_AB_{s}_{T}.npz")["n_clusters"]) for _, s, _ in tiers}
    fig.suptitle("Extrapolation to zero contamination from the purity tiers (joint A+B, λ_min = 5, no mixture; "
                 f"N = {N['p95']} / {N['t90']} / {N['t80']}; tiers: 50 cluster bootstraps each)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/lxm_2026-11-26_extrapolation.png", dpi=150)
    print("\n".join(rows))
    print("tiers f:", {s: round(f, 3) for _, s, f in tiers}, "N:", N)
    for nm, s_, f in tiers:
        out = []
        for pn, _ in PARS:
            b = boot_meds(f"{F}/joint_AB_{s_}_{T}", pn)
            v = np.percentile(b, [16, 50, 84]) if b is not None else quant(f"{F}/joint_AB_{s_}_{T}.npz", pn)
            out.append(f"{pn} {v[1]:+.2f} +{v[2] - v[1]:.2f} −{v[1] - v[0]:.2f}")
        print(nm, f"({'bootstrap, ' + str(len(b)) if b is not None else 'single fit'})", " | ".join(out))


if __name__ == "__main__":
    main()
