"""Functional form of the magnitude-gap terms of the scaling relations.

paper 1.1 (2026-10-10): the magnitude gap enters directly, gamma_cm * Delta_m (and
delta_Mlambda * Delta_m), since Delta_m is already a logarithmic quantity.  paper 1.0 used
gamma_cm * log10(Delta_m / Delta_m_ref).  Select with DESTACKING_DM_FORM = linear (default) | log.
Non-finite gaps contribute nothing (regressor 0) in both forms.
"""
import os
import numpy as np

DM_FORM = os.environ.get("DESTACKING_DM_FORM", "linear").strip().lower()
if DM_FORM not in ("linear", "log"):
    raise ValueError(f"DESTACKING_DM_FORM must be 'linear' or 'log', got {DM_FORM!r}")


def dm_regressor(dm, delta_m_ref, form=None):
    """Regressor multiplying gamma_cm / delta_Mlambda: Delta_m (linear) or log10(Delta_m/ref) (log)."""
    form = form or DM_FORM
    dm = np.asarray(dm, dtype=float)
    ok = np.isfinite(dm)
    if form == "linear":
        return np.where(ok, dm, 0.0)
    return np.where(ok, np.log10(np.maximum(np.where(ok, dm, delta_m_ref), 1e-6) / delta_m_ref), 0.0)
