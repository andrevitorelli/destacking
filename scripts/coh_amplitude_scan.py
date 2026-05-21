"""
Tuning scan over the rank-1 covariance inflation amplitude.

For each ``coh_amplitude`` value in AMPLITUDES, the script repeats the
gamma_repeat pipeline (DES Y3-like simulation + spherical-NFW inference
with the Sherman-Morrison rank-1 covariance inflation) for
``N_REAL_PER_AMP`` realisations of fresh OS entropy.  For every
amplitude the calibration ratio
   sigma_runs / <sigma_post>
is measured: the value of ``coh_amplitude`` that brings this ratio closest
to 1 absorbs the missing per-cluster coherent variance correctly.

Three diagnostic plots are produced at the end:

  coh_amplitude_scan_calibration.png   calibration ratio vs amplitude
  coh_amplitude_scan_widths.png        <sigma_post> and sigma_runs vs amplitude
  coh_amplitude_scan_bias.png          residual bias vs amplitude

Configuration
-------------

  AMPLITUDES        Comma-separated list           (default 0.0,0.1,0.2,0.3,0.4,0.6)
  N_REAL_PER_AMP    Realisations per amplitude     (default 10)
  N_CLUSTERS        Clusters per realisation       (default 5000)
  N_WORKERS         Pool workers for the sampler   (default 15)
  RUN_DIR           Output directory               (default
                    coh_tuning_outputs/run_YYYY-MM-DDTHH-MM-SS).
                    Override to resume an interrupted scan.

Cached per-(amplitude, realisation) summaries live under
   <RUN_DIR>/amp_<amp>/realisation_<idx>/summary.json
and are reused on subsequent invocations of the same RUN_DIR.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Re-use the full gamma_repeat pipeline as a library.  We mutate its
# module-level COH_AMPLITUDE attribute between scans (no subprocesses).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gamma_repeat as gr  # noqa: E402


# ── Configuration ──────────────────────────────────────────────────────────
def _parse_amplitudes(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]

AMPLITUDES = _parse_amplitudes(
    os.environ.get("AMPLITUDES", "0.0,0.1,0.2,0.3,0.4,0.6")
)
N_REAL_PER_AMP = int(os.environ.get("N_REAL_PER_AMP", 10))

# Make sure the inner gamma_repeat sees the right NCLUSTERS / NWORKERS via env
# (they are read at import time).  If the caller already exported them, this
# is a no-op.
os.environ.setdefault("N_CLUSTERS", "5000")
os.environ.setdefault("N_WORKERS", "15")

_DEFAULT_TS = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
RUN_DIR = os.environ.get(
    "RUN_DIR", os.path.join("coh_tuning_outputs", f"run_{_DEFAULT_TS}")
)
os.makedirs(RUN_DIR, exist_ok=True)


def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    m = rem / 60
    return f"{int(h)}h{int(m):02d}m"


# ── One realisation at a given amplitude ─────────────────────────────────
def _run_at_amplitude(amp, amp_dir, n_realisations):
    """Run n_realisations of gamma_repeat at ``coh_amplitude = amp``.

    Returns the list of successful result dicts (one per realisation).
    Failed realisations have their subdirectories removed.
    """
    # Tell the gamma_repeat module to use this amplitude on every subsequent
    # sampler call.  COH_AMPLITUDE is looked up at runtime from gr's globals.
    gr.COH_AMPLITUDE = float(amp)
    print(
        f"\n=== coh_amplitude = {amp:.3f}  ({n_realisations} realisations) "
        f"=> {amp_dir} ===",
        flush=True,
    )

    results = []
    durations = []
    failures = []
    attempt_idx = 0
    max_attempts = 4 * n_realisations

    while len(results) < n_realisations and attempt_idx < max_attempts:
        subdir = os.path.join(amp_dir, f"realisation_{attempt_idx:03d}")
        summary_path = os.path.join(subdir, "summary.json")
        npz_path     = os.path.join(subdir, "gamma_posterior.npz")
        if os.path.exists(summary_path) and os.path.exists(npz_path):
            print(
                f"  [{len(results) + 1}/{n_realisations}] "
                f"realisation_{attempt_idx:03d} -- cached",
                flush=True,
            )
            with open(summary_path) as f:
                results.append(json.load(f))
            attempt_idx += 1
            continue

        print(
            f"  [{len(results) + 1}/{n_realisations}] "
            f"realisation_{attempt_idx:03d} (attempt {attempt_idx + 1})",
            flush=True,
        )
        t0 = time.time()
        try:
            r = gr.run_one_realisation(attempt_idx, amp_dir)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            dt = time.time() - t0
            print(
                f"    FAILED in {_fmt_duration(dt)}: {type(exc).__name__}",
                flush=True,
            )
            traceback.print_exc()
            shutil.rmtree(subdir, ignore_errors=True)
            failures.append((attempt_idx, type(exc).__name__))
            durations.append(dt)
            attempt_idx += 1
            continue

        dt = time.time() - t0
        durations.append(dt)
        results.append(r)
        remaining = n_realisations - len(results)
        avg = sum(durations) / len(durations) if durations else dt
        # ETA accounting for empirical failure rate
        total = len(results) + len(failures)
        fail_rate = len(failures) / total if total else 0.0
        success_rate = max(1.0 - fail_rate, 1e-3)
        eta = avg * (remaining / success_rate)
        print(
            f"    iteration took {_fmt_duration(dt)}; avg {_fmt_duration(avg)}; "
            f"ETA {_fmt_duration(eta)} for {remaining} more"
            + (f"  ({len(failures)} failed)" if failures else ""),
            flush=True,
        )
        attempt_idx += 1

    if failures:
        print(f"  Failed attempts: {failures}", flush=True)
    return results


# ── Aggregation per amplitude ────────────────────────────────────────────
def _aggregate(results, true_gamma=0.2):
    gm  = np.array([r["gamma_med"]    for r in results])
    glo = np.array([r["gamma_lo"]     for r in results])
    ghi = np.array([r["gamma_hi"]     for r in results])
    sigs = np.array([r["sigma_gamma"] for r in results])
    res = gm - true_gamma
    return {
        "n":             len(results),
        "mean_gamma":    float(gm.mean()),
        "median_gamma":  float(np.median(gm)),
        "std_gamma":     float(gm.std(ddof=1)) if len(gm) > 1 else 0.0,
        "mean_sigma":    float(sigs.mean()),
        "median_sigma":  float(np.median(sigs)),
        "mean_residual": float(res.mean()),
        "std_residual":  float(res.std(ddof=1)) if len(res) > 1 else 0.0,
        "calibration_ratio": (
            float(gm.std(ddof=1) / sigs.mean()) if len(gm) > 1 and sigs.mean() > 0 else float("nan")
        ),
    }


# ── Final plots ──────────────────────────────────────────────────────────
def _plot_results(per_amp, run_dir):
    amps   = np.array([a for a, _ in per_amp])
    agg    = [s for _, s in per_amp]
    cal    = np.array([s["calibration_ratio"] for s in agg])
    s_runs = np.array([s["std_gamma"]         for s in agg])
    s_post = np.array([s["mean_sigma"]        for s in agg])
    bias   = np.array([s["mean_residual"]     for s in agg])
    n_runs = np.array([s["n"]                 for s in agg])

    # Plot 1: calibration ratio vs amplitude
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(amps, cal, "-o", color="steelblue", markersize=6)
    ax.axhline(1.0, color="k", lw=1, ls="--",
               label="target: $\\sigma_{\\rm runs} / \\langle\\sigma_{\\rm post}\\rangle = 1$")
    # If the curve crosses 1, mark the linear-interpolated root
    for i in range(len(amps) - 1):
        if (cal[i] - 1.0) * (cal[i + 1] - 1.0) < 0:
            frac = (1.0 - cal[i]) / (cal[i + 1] - cal[i])
            root = amps[i] + frac * (amps[i + 1] - amps[i])
            ax.axvline(root, color="crimson", lw=1.2, ls=":",
                       label=f"crossing at coh_amplitude $\\approx {root:.3f}$")
    ax.set_xlabel("coh_amplitude")
    ax.set_ylabel(r"$\sigma_{\rm runs} / \langle\sigma_{\rm post}\rangle$")
    ax.set_title(
        f"Calibration ratio vs rank-1 inflation amplitude  "
        f"(N_real per amp = {N_REAL_PER_AMP})",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    out = os.path.join(run_dir, "coh_amplitude_scan_calibration.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

    # Plot 2: widths
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(amps, s_runs, "-o", color="tomato",   label=r"$\sigma_{\rm runs}$ (empirical)")
    ax.plot(amps, s_post, "-s", color="steelblue", label=r"$\langle\sigma_{\rm post}\rangle$")
    ax.set_xlabel("coh_amplitude")
    ax.set_ylabel(r"$\sigma$ (gamma)")
    ax.set_title("Empirical run-to-run scatter vs mean posterior width", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(run_dir, "coh_amplitude_scan_widths.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

    # Plot 3: bias
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(amps, bias, yerr=s_runs / np.sqrt(n_runs), fmt="o",
                color="steelblue", capsize=4, markersize=6,
                label="mean residual $\\pm \\sigma_{\\rm mean}$")
    ax.axhline(0.0, color="k", lw=1, ls="--", label="unbiased")
    ax.set_xlabel("coh_amplitude")
    ax.set_ylabel(r"$\langle \hat\gamma \rangle - \gamma_{\rm true}$")
    ax.set_title("Recovered gamma bias vs coh_amplitude", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(run_dir, "coh_amplitude_scan_bias.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def _print_summary(per_amp):
    print("\n" + "=" * 78, flush=True)
    print(
        f'{"amp":>6}  {"N":>3}  {"mean_gamma":>11}  {"std_gamma":>10}  '
        f'{"<sigma>":>9}  {"ratio":>7}  {"bias":>9}',
        flush=True,
    )
    print("-" * 78, flush=True)
    for amp, s in per_amp:
        print(
            f'{amp:>+6.3f}  {s["n"]:>3d}  {s["mean_gamma"]:>+11.4f}  '
            f'{s["std_gamma"]:>10.4f}  {s["mean_sigma"]:>9.4f}  '
            f'{s["calibration_ratio"]:>7.2f}  {s["mean_residual"]:>+9.4f}',
            flush=True,
        )


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print(
        f"COH_AMPLITUDE scan: AMPLITUDES={AMPLITUDES}, "
        f"N_REAL_PER_AMP={N_REAL_PER_AMP}, "
        f"N_CLUSTERS={gr.N_CLUSTERS}, "
        f"N_WORKERS={gr.N_WORKERS}",
        flush=True,
    )
    print(f"Output directory: {RUN_DIR}", flush=True)

    # The DES Y3 n(z) plot needs to be drawn only once.
    gr.plot_des_y3_nz(RUN_DIR)

    per_amp = []
    overall_t0 = time.time()
    for amp in AMPLITUDES:
        amp_dir = os.path.join(RUN_DIR, f"amp_{amp:.3f}")
        os.makedirs(amp_dir, exist_ok=True)
        results = _run_at_amplitude(amp, amp_dir, N_REAL_PER_AMP)
        if not results:
            print(f"  no successful realisations at amp={amp}; skipping", flush=True)
            continue
        agg = _aggregate(results, true_gamma=gr.TRUE_GAMMA)
        per_amp.append((amp, agg))
        # Save per-amplitude aggregate
        with open(os.path.join(amp_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)

    print(
        f"\nTotal wall time: {_fmt_duration(time.time() - overall_t0)}",
        flush=True,
    )

    if not per_amp:
        print("No amplitude produced successful results.", file=sys.stderr)
        return 1

    # Combined results JSON for downstream consumption / report builds
    out_json = os.path.join(RUN_DIR, "scan_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {"true_gamma": gr.TRUE_GAMMA,
             "n_real_per_amp": N_REAL_PER_AMP,
             "n_clusters": gr.N_CLUSTERS,
             "amplitudes": [a for a, _ in per_amp],
             "aggregates": [s for _, s in per_amp]},
            f, indent=2,
        )
    print(f"Saved {out_json}", flush=True)

    _plot_results(per_amp, RUN_DIR)
    _print_summary(per_amp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
