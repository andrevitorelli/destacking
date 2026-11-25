"""Chance-match mixture models compared: validation on injected samples and real-data fits (2026-11-25).

Models: no mixture; mean template (q flat by tier); halo population (q flat by tier); halo population with q(L_X, z).
Validation: real p95 + n chance matches from pool half B, chance model fitted on half A; target = the same model on p95 alone.

    python plot_population.py [--out DIR]
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
# label, validation suffix, real-data tag, colour, marker
MODELS = [("no mixture", "hB_", "", "0.45", "s"),
          ("mean template, flat q", "hB_contfixedA_", "contfixed_", "C1", "^"),
          ("halo population, flat q", "hB_contfixedApop_", "contfixedpop_", "C0", "o"),
          ("halo population, q(L_X, z)", "hB_contfixedApopqlx_", "contfixedpopqlx_", "C2", "D")]
FREE = [("template", "contfree_", "C1"), ("population", "contfreepop_", "C0"), ("population, q(L_X, z)", "contfreepopqlx_", "C2")]
FSP = {0: 76 / 496, 124: 200 / 620, 289: 365 / 785}
REAL = [("p95", "p95", 0.153), ("p90", "p90", 0.323), ("full", "all", 0.446)]
LAB = {"beta": r"$\beta$", "gamma": r"$\gamma$", "sig_int": r"$\sigma_{\rm int}$ [dex]"}


def quant(path, name):
    if not os.path.exists(path):
        return None
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
    pars = ["beta", "gamma", "sig_int"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    rows = ["| model | sample | N fits | β | γ | σ_int |", "|---|---|---|---|---|---|"]
    for k, (lab, vs, rt, c, mk) in enumerate(MODELS):
        dx = 0.006 * (k - 1.5)
        # validation (top row): base = same model on p95 alone
        stats = {}
        for i, pn in enumerate(pars):
            q0 = quant(f"{F}/joint_AB_p95_{rt}{T}.npz", pn)
            if q0 is None:
                continue
            xs, ys = [FSP[0] + dx], [q0[1]]
            ax[0, i].errorbar(FSP[0] + dx, q0[1], yerr=[[q0[1] - q0[0]], [q0[2] - q0[1]]], fmt=mk, color=c, ms=7, mec="k", capsize=2)
            for n in (124, 289):
                fs = sorted(glob.glob(f"{F}/joint_AB_p95_inj{n}s[0-9]{vs}{T}.npz"))
                if not fs:
                    continue
                qs = np.array([quant(f, pn) for f in fs])
                ax[0, i].errorbar(np.full(len(fs), FSP[n] + dx), qs[:, 1], yerr=[qs[:, 1] - qs[:, 0], qs[:, 2] - qs[:, 1]],
                                  fmt=mk, color=c, alpha=0.25, ms=3, lw=0.7)
                xs.append(FSP[n] + dx); ys.append(qs[:, 1].mean())
                stats.setdefault(n, {})[pn] = (qs[:, 1].mean(), qs[:, 1].std(), len(fs))
            ax[0, i].plot(xs, ys, "-" + mk, color=c, ms=8, mec="k", label=lab)
            ax[0, i].axhline(q0[1], color=c, lw=0.6, ls=":")
            stats.setdefault(0, {})[pn] = (q0[1], 0.0, 1)
        stats = {n: s for n, s in stats.items() if n or len(stats) > 1}
        for n in sorted(stats):
            s = stats[n]
            if len(s) == len(pars):
                rows.append(f"| {lab} | p95{' + ' + str(n) if n else ' alone'} | {s['beta'][2]} | "
                            + " | ".join(f"{s[p][0]:+.2f}" + (f" (spread {s[p][1]:.2f})" if n else "") for p in pars) + " |")
        # real samples (bottom row)
        for i, pn in enumerate(pars):
            xs, ys = [], []
            for nm, sub, x in REAL:
                q = quant(f"{F}/joint_AB_{sub}_{rt}{T}.npz", pn)
                if q is None:
                    continue
                ax[1, i].errorbar(x + dx, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt=mk, color=c, capsize=2, ms=7, mec="k")
                xs.append(x + dx); ys.append(q[1])
            ax[1, i].plot(xs, ys, "-", color=c, lw=1)
    # population model with q free in each sample (purple crosses, offset)
    for i, pn in enumerate(pars):
        xs, ys = [], []
        for nm, sub, x in REAL:
            q = quant(f"{F}/joint_AB_{sub}_contfreepop_{T}.npz", pn)
            if q is None:
                continue
            ax[1, i].errorbar(x + 0.016, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt="X", color="C4", capsize=2, ms=8, mec="k",
                              label="halo population, q free in each sample" if (i == 0 and nm == "p95") else None)
            xs.append(x + 0.016); ys.append(q[1])
        ax[1, i].plot(xs, ys, "--", color="C4", lw=1)
    FREE_ = [f for f in FREE if f[1] != "contfreepop_"]
    for fl, tag, c in FREE_:
        for i, pn in enumerate(pars):
            q = quant(f"{F}/joint_AB_all_{tag}{T}.npz", pn)
            if q is not None:
                ax[1, i].errorbar(0.446 + 0.03 + 0.008 * ["C1", "C0", "C2"].index(c), q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]],
                                  fmt="x", color=c, capsize=2, ms=8, mew=2, label=f"full, q free ({fl})" if i == 0 else None)
    for i, pn in enumerate(pars):
        ax[0, i].set_title("validation: p95 + injected chance matches\n(dotted: same model on p95 alone = target)", fontsize=9)
        ax[0, i].set_xlabel("expected fraction of spurious matches")
        ax[1, i].set_title("real samples", fontsize=9, pad=14)
        ax[1, i].set_xlabel("expected fraction of spurious matches (randoms)")
        for r in (0, 1):
            ax[r, i].set_ylabel(LAB[pn])
        for nm, sub, x in REAL:
            ax[1, i].axvline(x, color="k", lw=0.4, ls=":")
            ax[1, i].text(x, 1.01, nm, transform=ax[1, i].get_xaxis_transform(), ha="center", fontsize=8)
    ax[0, 0].legend(fontsize=7, loc="upper left")
    ax[1, 0].legend(fontsize=7, loc="upper right")
    ax[1, 0].axhspan(0.96, 1.26, color="k", alpha=0.07)
    ax[1, 1].axhspan(-0.79, 0.79, color="k", alpha=0.07)
    fig.suptitle("Chance-match mixture models (joint A+B, λ_min = 5); grey bands: Pederneiras+25 eRASS1", fontsize=11)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/lxm_2026-11-25_population.png", dpi=150)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
