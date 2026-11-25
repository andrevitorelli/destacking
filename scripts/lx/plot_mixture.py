"""Chance-match mixture model: validation on injected samples and real-data fits (2026-11-25).

    python plot_mixture.py [--out DIR]
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
F = f"{D}/fits_sel"
T = "opt2lmin5"


def quant(path, name):
    d = np.load(path, allow_pickle=True)
    po = list(d["param_order"])
    if name not in po:
        return None
    j = po.index(name)
    x, w = d["samples"][:, j], d["weights"]
    o = np.argsort(x)
    return np.interp([.16, .5, .84], np.cumsum(w[o]), x[o])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/figures")
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    # validation: beta, gamma against injected n
    fsp = {0: 76 / 496, 124: 200 / 620, 289: 365 / 785}
    series = [("no mixture (randoms from whole pool)", "{n}s[0-9]_" + T, "0.5", "s"),
              ("no mixture (randoms from half B)", "{n}s[0-9]hB_" + T, "C1", "^"),
              ("mixture, q fixed (template from half A)", "{n}s[0-9]hB_contfixedA_" + T, "C0", "o")]
    rows = []
    for i, pn in enumerate(["beta", "gamma"]):
        for lab, pat, c, mk in series:
            xs, ys, es = [], [], []
            base = f"{F}/joint_AB_p95_{'contfixed_' if 'mixture, q' in lab else ''}{T}.npz"
            q0 = quant(base, pn)
            xs.append(fsp[0]); ys.append(q0[1]); es.append(0)
            for n in (124, 289):
                fs = sorted(glob.glob(f"{F}/joint_AB_p95_inj" + pat.format(n=n) + ".npz"))
                if not fs:
                    continue
                m = [quant(f, pn)[1] for f in fs]
                for f in fs:
                    q = quant(f, pn)
                    ax[i].errorbar(fsp[n] + {"0.5": -0.006, "C1": 0.0, "C0": 0.006}[c], q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]],
                                   fmt=mk, color=c, alpha=0.3, ms=3, lw=0.8)
                xs.append(fsp[n]); ys.append(np.mean(m)); es.append(np.std(m))
                if i == 0:
                    rows.append((lab, n, len(fs)))
            ax[i].plot(xs, ys, "-" + mk, color=c, ms=8, mec="k", label=lab)
        ax[i].set_xlabel("expected fraction of spurious matches (p95 + injected)")
        ax[i].set_ylabel({"beta": r"$\beta$", "gamma": r"$\gamma$"}[pn])
        ax[i].set_title("validation: p95 + injected chance matches", fontsize=9)
    ax[0].legend(fontsize=7, loc="upper left")
    # real data
    samples = [("p95", "p95", 0.153), ("p90", "p90", 0.323), ("full", "all", 0.446)]
    for i, pn in enumerate(["beta", "gamma"]):
        k = 2 + i
        for lab, tag, c, mk, dx in [("no mixture", "", "0.3", "D", -0.006), ("mixture, q fixed by tier", "contfixed_", "C0", "o", 0.006)]:
            xs, ys = [], []
            for nm, sub, x in samples:
                q = quant(f"{F}/joint_AB_{sub}_{tag}{T}.npz", pn)
                ax[k].errorbar(x + dx, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt=mk, color=c, capsize=3, ms=7,
                               label=lab if nm == "p95" else None)
                xs.append(x + dx); ys.append(q[1])
            ax[k].plot(xs, ys, "-", color=c, lw=1)
        q = quant(f"{F}/joint_AB_all_contfree_{T}.npz", pn)
        ax[k].errorbar(0.446 + 0.018, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="x", color="C3", capsize=3, ms=8,
                       label="full, mixture with q free")
        for nm, sub, x in samples:
            ax[k].annotate(nm, (x, ax[k].get_ylim()[0]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax[k].set_xlabel("expected fraction of spurious matches (randoms)")
        ax[k].set_ylabel({"beta": r"$\beta$", "gamma": r"$\gamma$"}[pn])
        ax[k].set_title("real samples", fontsize=9)
    ax[2].axhspan(0.96, 1.26, color="k", alpha=0.07, label="Pederneiras+25 eRASS1")
    ax[3].axhspan(-0.79, 0.79, color="k", alpha=0.07)
    ax[2].legend(fontsize=7, loc="upper right")
    fig.suptitle("Chance-match mixture model (joint A+B, λ_min = 5): validation fails at high contamination; free q runs away",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/lxm_2026-11-25_mixture.png", dpi=150)
    print(rows)


if __name__ == "__main__":
    main()
