"""
Render a corner plot for every per-bootstrap fit in one or more gamma_scan
runs, by reloading the posterior from the nautilus checkpoint.

For each ``simulation_g*/bootstrap_*/nautilus_checkpoint.hdf5`` we:
  - detect whether the fit had 5 or 6 sampled parameters (with or without
    the amplitude correction A),
  - resume the sampler to recover the weighted posterior samples,
  - render a corner plot with the appropriate parameter labels and prior
    centres (plus γ_true) marked as truth lines,
  - save it next to the checkpoint as ``corner.png``.

Skips fits that already have a corner.png on disk.

Usage:
    python scripts/make_corner_plots.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations
import json
import os
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import corner
from nautilus import Prior, Sampler
from scipy.stats import norm as sp_norm


PRIOR_DEFS = {
    "c0":         sp_norm(loc=4.57, scale=0.05),
    "alpha":      sp_norm(loc=-0.1, scale=0.05),
    "beta":       sp_norm(loc=-0.4, scale=0.2),
    "gamma":      sp_norm(loc=0.0,  scale=3.5),
    "log_M_norm": sp_norm(loc=0.0,  scale=0.1),
    "A":          sp_norm(loc=0.80, scale=0.02),
}
PARAM_ORDER_5 = ["c0", "alpha", "beta", "gamma", "log_M_norm"]
PARAM_ORDER_6 = PARAM_ORDER_5 + ["A"]
LABELS = {
    "c0":         r"$c_0$",
    "alpha":      r"$\alpha$",
    "beta":       r"$\beta$",
    "gamma":      r"$\gamma$",
    "log_M_norm": r"$\log M_{\rm norm}$",
    "A":          r"$A$",
}
PRIOR_CENTERS = {n: float(d.mean()) for n, d in PRIOR_DEFS.items()}


def dummy_loglike(p):
    return 0.0


def detect_n_params(ckpt_path):
    """Inspect the nautilus checkpoint to find the parameter count."""
    with h5py.File(ckpt_path, "r") as f:
        # nautilus stores neural-emulator coefficient arrays shaped
        # (n_params, hidden_dim) in bound_*/neural_bound_*/emulator/coefs_0_0.
        def find(grp):
            for key in grp:
                item = grp[key]
                if key == "coefs_0_0" and hasattr(item, "shape") \
                        and len(item.shape) == 2:
                    return int(item.shape[0])
                if hasattr(item, "keys"):
                    n = find(item)
                    if n is not None:
                        return n
            return None
        n = find(f)
        if n is None:
            # fall back: any (N, k) 'points' dataset
            def find_points(grp):
                for key in grp:
                    item = grp[key]
                    if key == "points" and hasattr(item, "shape") \
                            and len(item.shape) == 2:
                        return int(item.shape[1])
                    if hasattr(item, "keys"):
                        v = find_points(item)
                        if v is not None:
                            return v
                return None
            n = find_points(f)
        if n is None:
            raise RuntimeError(f"could not detect param count in {ckpt_path}")
        return n


def build_sampler(ckpt_path, n_params):
    order = PARAM_ORDER_5 if n_params == 5 else PARAM_ORDER_6
    prior = Prior()
    for name in order:
        prior.add_parameter(name, dist=PRIOR_DEFS[name])
    return Sampler(prior, dummy_loglike, n_live=400,
                   filepath=ckpt_path, resume=True, pool=None), order


def render_one(ckpt, summary, out_path):
    n_params = detect_n_params(ckpt)
    sampler, order = build_sampler(ckpt, n_params)
    pts, log_w, _ = sampler.posterior()
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()

    labels = [LABELS[p] for p in order]
    truths = [PRIOR_CENTERS[p] for p in order]
    gt = None
    if os.path.exists(summary):
        with open(summary) as f:
            gt = float(json.load(f)["gamma_true"])
        truths[order.index("gamma")] = gt   # mark the true γ explicitly

    fig = corner.corner(
        pts, weights=w, labels=labels,
        truths=truths, truth_color="crimson",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt=".3f",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        levels=(0.39, 0.86),   # 1σ/2σ contour levels for 2D Gaussian
    )

    # Overlay prior PDFs on the diagonal 1D histograms.  corner's diagonals
    # are normalised (density=True), so a scipy.stats .pdf(x) value plots on
    # the same axis.  We extend the y-range if the prior PDF peak exceeds
    # the current histogram limit.
    K = len(order)
    for i, name in enumerate(order):
        ax_diag = fig.axes[i * K + i]
        xmin, xmax = ax_diag.get_xlim()
        x = np.linspace(xmin, xmax, 400)
        prior_pdf = PRIOR_DEFS[name].pdf(x)
        ax_diag.plot(x, prior_pdf, color="darkorange",
                     ls="--", lw=1.4, alpha=0.9, zorder=10)
        ymin, ymax = ax_diag.get_ylim()
        ax_diag.set_ylim(ymin, max(ymax, float(prior_pdf.max()) * 1.05))

    # Legend in the upper-right (corner.corner leaves the upper-triangular
    # subplots blank, so we can drop a legend there without overlapping).
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="crimson",   lw=1.4, ls="-",
               label=r"truth ($\gamma_{\rm true}$ / prior centre)"),
        Line2D([0], [0], color="darkorange", lw=1.4, ls="--",
               label="prior PDF"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=10, frameon=True)

    title = os.path.basename(os.path.dirname(os.path.dirname(out_path)))
    if gt is not None:
        title += f"   $\\gamma_{{\\rm true}}={gt:+.3f}$"
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def process_run(run_dir, force=False):
    sim_dirs = sorted(d for d in os.listdir(run_dir)
                      if d.startswith("simulation_g"))
    print(f"\n=== Run: {run_dir}  ({len(sim_dirs)} simulations) ===")
    done = skipped = errored = 0
    t0 = time.time()
    total_targets = 0
    targets = []
    for sd in sim_dirs:
        sim_path = os.path.join(run_dir, sd)
        for boot in sorted(os.listdir(sim_path)):
            if not boot.startswith("bootstrap_"):
                continue
            sub  = os.path.join(sim_path, boot)
            ckpt = os.path.join(sub, "nautilus_checkpoint.hdf5")
            sj   = os.path.join(sub, "summary.json")
            out  = os.path.join(sub, "corner.png")
            if not os.path.exists(ckpt):
                continue
            total_targets += 1
            if os.path.exists(out) and not force:
                skipped += 1
                continue
            targets.append((ckpt, sj, out, sd, boot))

    print(f"  {len(targets)} new, {skipped} already done, {total_targets} total")
    for idx, (ckpt, sj, out, sd, boot) in enumerate(targets):
        try:
            render_one(ckpt, sj, out)
            done += 1
        except Exception as exc:
            print(f"  ERROR {sd}/{boot}: {exc}")
            errored += 1
            plt.close("all")
        if (idx + 1) % 10 == 0 or idx + 1 == len(targets):
            dt = time.time() - t0
            eta = dt / (idx + 1) * (len(targets) - idx - 1)
            print(f"    [{idx+1}/{len(targets)}]  "
                  f"elapsed {dt:.0f}s  eta {eta:.0f}s", flush=True)

    print(f"  finished: {done} rendered, {skipped} cached, {errored} errors  "
          f"({time.time()-t0:.0f}s)")


def main():
    args = sys.argv[1:]
    force = False
    if "--force" in args:
        force = True
        args = [a for a in args if a != "--force"]
    if not args:
        sys.exit("usage: make_corner_plots.py [--force] RUN_DIR [RUN_DIR ...]")
    for rd in args:
        process_run(rd, force=force)


if __name__ == "__main__":
    main()
