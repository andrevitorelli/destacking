"""Small shared utilities for the destacking pipelines.

Extracted from scripts/gamma_scan.py so that simulation scans and future
real-data ingestion scripts share one implementation.
"""

import numpy as np


def fmt_duration(seconds):
    """Compact human-readable duration: 42.0s, 3m07s, 2h05m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    return f"{int(h)}h{int(rem/60):02d}m"


def deterministic_seed(g_idx, offset):
    """Deterministic seed unique per (gamma-index, offset) — ensures same
    simulation data on resume even across restarts."""
    return g_idx * 100_000 + int(offset)


def gamma_dir_name(g_idx, gamma_true):
    """Per-gamma output directory name.

    Directory names are only a human-readable label; g_idx already makes
    them unique. Plot scripts must read the exact gamma_true value from
    the per-bootstrap summary.json, not parse it from this name.
    """
    return f"simulation_g{g_idx:02d}_gamma{gamma_true:+.1f}"


def weighted_quantile(v, w, q):
    """Quantile q of values ``v`` under weights ``w`` (linear interpolation
    on the normalised cumulative weight)."""
    v = np.asarray(v)
    w = np.asarray(w)
    sorter = np.argsort(v)
    cs     = np.cumsum(w[sorter]) / w.sum()
    return float(np.interp(q, cs, v[sorter]))
