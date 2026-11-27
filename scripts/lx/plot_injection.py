"""Contamination-injection test: figure and table (2026-11-24).

Real p95 sample + n chance matches from the random pool (build_random_pool.py) against the real p90, p80 and full
samples, all joint A+B fits with the optical stage v2 and lambda_min = 5.  x axis: expected fraction of spurious
matches, from the random catalogues (contamination.json: N_rand / N_real after our cuts), and for the injected
samples (n_spurious(p95) + n) / (N_p95 + n).

    python plot_injection.py [--out DIR]
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
TAG = "opt2lmin5"
PARS = [("alpha", r"$\alpha$"), ("beta", r"$\beta$"), ("gamma", r"$\gamma$"), ("sig_int", r"$\sigma_{\rm int}$ [dex]")]


def quant(path, name):
    d = np.load(path, allow_pickle=True)
    j = list(d["param_order"]).index(name)
    x, w = d["samples"][:, j], d["weights"]
    o = np.argsort(x)
    return np.interp([.16, .5, .84], np.cumsum(w[o]), x[o])


def boot(prefix, name):
    fs = [f for f in glob.glob(f"{prefix}_b*.npz") if re.fullmatch(r"_b\d+\.npz", f[len(prefix):])]
    if len(fs) < 10:
        return None
    return np.percentile([quant(f, name)[1] for f in fs], [16, 50, 84])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/figures")
    a = ap.parse_args()
    c = json.load(open(f"{D}/randoms/contamination.json"))
    nr = {int(k): v for k, v in c["n_random_after_cuts"].items()}
    nreal = c["n_real"]
    c_of = {"p95": nr[95] / nreal["95"], "p90": nr[90] / nreal["90"], "p80": nr[80] / nreal["80"]}
    n95, s95 = nreal["95"], nr[95]

    inj = {}
    for f in glob.glob(f"{F}/joint_AB_p95_inj*s*_{TAG}.npz"):
        m = re.search(r"_inj(\d+)s(\d+)_" + TAG + r"\.npz$", f)     # plain injections only (not hB / mixture runs)
        if not m:
            continue
        n, s = map(int, m.groups())
        inj.setdefault(n, {})[s] = f
    real = {"p95": f"{F}/joint_AB_p95_{TAG}", "p90": f"{F}/joint_AB_p90_{TAG}", "p80": f"{F}/joint_AB_p80_{TAG}",
            "full": f"{F}/joint_AB_all_{TAG}"}

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    rows = ["| sample | N | expected spurious fraction | β | γ | α | σ_int |", "|---|---|---|---|---|---|---|"]

    def fmt(q):
        return f"{q[1]:+.2f} +{q[2] - q[1]:.2f} −{q[1] - q[0]:.2f}"

    # real samples
    for name, pre in real.items():
        qs = []
        for i, (pn, _) in enumerate(PARS):
            q = boot(pre, pn)
            kind = "bootstrap"
            if q is None:
                if not os.path.exists(pre + ".npz"):
                    qs = None
                    break
                q, kind = quant(pre + ".npz", pn), "single fit"
            qs.append(q)
        if qs is None:
            continue
        x = c_of.get(name, c_of["p80"])
        N = int(np.load(pre + ".npz")["n_clusters"]) if os.path.exists(pre + ".npz") else None
        for i, q in enumerate(qs):
            ax[i].errorbar(x + (0.012 if name == "full" else 0), q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="D",
                           color="k" if name != "full" else "0.5", mfc="k" if name != "full" else "white", ms=7, capsize=3,
                           zorder=5, label="real samples" if (i == 0 and name == "p95") else (
                               "real full sample (≥ p80 level)" if i == 0 and name == "full" else None))
            if i == 1:
                below = name in ("p80",)
                ax[i].annotate(name, (x + (0.012 if name == "full" else 0), q[0] if below else q[2]),
                               textcoords="offset points", xytext=(-10 if below else 6, -10 if below else 4),
                               ha="center", fontsize=8)
        rows.append(f"| real {name} ({kind}) | {N} | {'≥ ' if name == 'full' else ''}{x:.2f} | "
                    + " | ".join(fmt(qs[k]) for k in (1, 2, 0, 3)) + " |")

    # injected samples
    for n in sorted(inj):
        x = (s95 + n) / (n95 + n)
        meds = {pn: [] for pn, _ in PARS}
        for s, f in sorted(inj[n].items()):
            for i, (pn, _) in enumerate(PARS):
                q = quant(f, pn)
                meds[pn].append(q[1])
                ax[i].errorbar(x + 0.004 * (s - 3), q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="o", color="C0",
                               alpha=0.35, ms=3, capsize=0, lw=0.8)
        for i, (pn, _) in enumerate(PARS):
            m = np.array(meds[pn])
            ax[i].plot(x, m.mean(), "o", color="C0", ms=8, mec="k",
                       label="p95 + injected randoms (mean of draws; faint: each draw)" if (i == 0 and n == min(inj)) else None)
        rows.append(f"| p95 + {n} randoms ({len(inj[n])} draws) | {n95 + n} | {x:.2f} | "
                    + " | ".join(f"{np.mean(meds[pn]):+.2f} (draw spread {np.std(meds[pn]):.2f})" for pn in ("beta", "gamma", "alpha", "sig_int")) + " |")
    # connect: base p95 + injected means
    for i, (pn, _) in enumerate(PARS):
        xs, ys = [], []
        if os.path.exists(real["p95"] + ".npz"):
            q = boot(real["p95"], pn); q = q if q is not None else quant(real["p95"] + ".npz", pn)
            xs.append(c_of["p95"]); ys.append(q[1])
        for n in sorted(inj):
            xs.append((s95 + n) / (n95 + n)); ys.append(np.mean([quant(f, pn)[1] for f in inj[n].values()]))
        ax[i].plot(xs, ys, "-", color="C0", lw=1.5, zorder=1)
    ax[1].axhspan(0.96, 1.26, color="k", alpha=0.07, label="Pederneiras+25 eRASS1")
    ax[2].axhspan(-0.79, 0.79, color="k", alpha=0.07)
    for i, (pn, pl) in enumerate(PARS):
        ax[i].set_xlabel("expected fraction of spurious matches")
        ax[i].set_ylabel(pl)
        for nm in ("p95", "p90", "p80"):
            ax[i].axvline(c_of[nm], color="k", lw=0.5, ls=":")
    h, l = [], []
    for a_ in ax[:2]:
        hh, ll = a_.get_legend_handles_labels(); h += hh; l += ll
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Contamination injection: real p95 + chance X-ray–optical matches vs real lower-purity samples "
                 "(joint A+B, λ_min = 5)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/lxm_2026-11-24_injection.png", dpi=150)
    print("\n".join(rows))
    print(f"\ncontamination: {c_of}; p95 expected spurious {s95}/{n95}")


if __name__ == "__main__":
    main()
