"""X-ray luminosity - mass relation for the eSCALE x DES Y3 analysis (added 2026-10-21).

Everything specific to fitting Lx-M with the three frameworks lives here:

* ``set_cosmology``: flat LCDM for xlensing (distances, Sigma_crit, NFW, two-halo tables) and colossus
  (Diemer & Joyce 2019 concentrations, halo mass function).  Must be called before any lensing is measured
  or modelled; it reloads ``xlensing.model`` so the tables built at import time follow the new cosmology.
* ``lx_from_mass`` / ``mass_from_lx``: the Pederneiras et al. (2025) relation

      log10(Lx E(z)^-1 / L_piv) = alpha + beta log10(M200c E(z) / M_piv) + gamma log10((1+z)/(1+z_piv))

  and its exact inverse (no scatter).
* ``diemer19_c200c``: c200c(M200c, z) of Diemer & Joyce (2019), native 200c definition, tabulated once.
* ``to_xlensing_mass``: M200c, c200c -> the (M, c) pair that xlensing's ``NFW_Delta_Sigma`` expects.  xlensing
  calls it M200m but defines it with rho = Omega_m0 rho_crit(z) (xlensing.cosmo.rhoM), i.e. an overdensity of
  200 Omega_m0 with respect to rho_crit(z).  The conversion keeps the NFW profile (rho_s, r_s) fixed, so it is
  exact for NFW and does not depend on colossus' own 200m definition.
* ``nfw_profiles_lx``: one-halo (+ optional two-halo) Delta Sigma for given M200c, with Diemer19 concentrations.
"""
from __future__ import annotations

import importlib

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

H0_DEFAULT, OM0_DEFAULT = 70.0, 0.3
_STATE = {"H0": None, "Om0": None, "c_interp": None}

LOGM200C_RANGE = (11.0, 16.3)
Z_RANGE = (0.01, 0.6)


def set_cosmology(H0=H0_DEFAULT, Om0=OM0_DEFAULT, sigma8=0.81, ns=0.965, Ob0=0.049):
    """Flat LCDM for xlensing and colossus.  Idempotent; returns the astropy cosmology."""
    from astropy.cosmology import FlatLambdaCDM
    import xlensing.cosmo as xc
    import xlensing.model as xm
    from colossus.cosmology import cosmology as ccos

    if _STATE["H0"] == H0 and _STATE["Om0"] == Om0:
        return xc.cosmology
    xc.cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)
    xc.h, xc.OmegaM, xc.OmegaK = H0 / 100.0, Om0, 0.0
    importlib.reload(xm)   # growth factor, sigma(R), rho_c0 and the two-halo tables are built at import time
    name = f"flat_{H0:g}_{Om0:g}"
    ccos.addCosmology(name, {"flat": True, "H0": H0, "Om0": Om0, "Ob0": Ob0, "sigma8": sigma8, "ns": ns})
    ccos.setCosmology(name)
    _STATE.update(H0=H0, Om0=Om0, c_interp=None)
    return xc.cosmology


def restore_default_cosmology():
    """Back to xlensing's and destacking's defaults (Planck18), e.g. after a test."""
    from astropy.cosmology import Planck18
    import xlensing.cosmo as xc
    import xlensing.model as xm
    from colossus.cosmology import cosmology as ccos
    xc.cosmology = Planck18
    xc.h, xc.OmegaM, xc.OmegaK = Planck18.h, Planck18.Om0, Planck18.Ok0
    importlib.reload(xm)
    ccos.setCosmology("planck18")
    _STATE.update(H0=None, Om0=None, c_interp=None)


def efunc(z):
    import xlensing.cosmo as xc
    return np.asarray(xc.cosmology.efunc(np.asarray(z, float)), float)


# ── the scaling relation ──────────────────────────────────────────────────────
class Pivots:
    """Pivots of the Lx-M relation: log10 L_piv [erg/s], log10 M_piv [Msun], z_piv."""

    def __init__(self, log_L=43.26, log_M=14.1, z=0.18):
        self.log_L, self.log_M, self.z = float(log_L), float(log_M), float(z)

    def __repr__(self):
        return f"Pivots(log_L={self.log_L}, log_M={self.log_M}, z={self.z})"


def lx_from_mass(logM200c, z, alpha, beta, gamma, piv: Pivots):
    """log10 Lx [erg/s] (not E(z)-scaled) for log10 M200c [Msun]."""
    Ez = efunc(z)
    return (piv.log_L + np.log10(Ez) + alpha + beta * (np.asarray(logM200c) + np.log10(Ez) - piv.log_M)
            + gamma * np.log10((1 + np.asarray(z)) / (1 + piv.z)))


def mass_from_lx(logLx, z, alpha, beta, gamma, piv: Pivots):
    """log10 M200c [Msun] from log10 Lx [erg/s]: the exact inverse of ``lx_from_mass`` (no scatter)."""
    Ez = efunc(z)
    y = np.asarray(logLx) - np.log10(Ez) - piv.log_L - alpha - gamma * np.log10((1 + np.asarray(z)) / (1 + piv.z))
    return piv.log_M - np.log10(Ez) + y / beta


# ── concentration and mass definitions ───────────────────────────────────────
def _build_c_interp(n_m=107, n_z=60):
    import warnings
    from colossus.halo import concentration as cc
    lm = np.linspace(*LOGM200C_RANGE, n_m)
    zz = np.linspace(*Z_RANGE, n_z)
    tab = np.empty((n_z, n_m))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, z in enumerate(zz):
            # colossus takes masses in Msun/h
            c = cc.concentration(10 ** lm * (_STATE["H0"] / 100.0), "200c", z=float(z), model="diemer19")
            tab[i] = np.log10(np.where(c > 0, c, np.nan))
    return RegularGridInterpolator((zz, lm), tab, bounds_error=False, fill_value=np.nan)


def diemer19_c200c(logM200c, z):
    """Diemer & Joyce (2019) c200c at log10 M200c [Msun] and z; NaN outside the tabulated range."""
    if _STATE["H0"] is None:
        raise RuntimeError("call destacking.lx.set_cosmology() first")
    if _STATE["c_interp"] is None:
        _STATE["c_interp"] = _build_c_interp()
    logM200c, z = np.broadcast_arrays(np.asarray(logM200c, float), np.asarray(z, float))
    return 10 ** _STATE["c_interp"](np.column_stack([z.ravel(), logM200c.ravel()])).reshape(logM200c.shape)


def _mu(x):
    return np.log1p(x) - x / (1 + x)


_C_GRID = np.logspace(np.log10(0.5), np.log10(40.0), 400)
_CX_CACHE = {}


def _cx_of_c(delta_ratio):
    """c_X(c_200c) for an overdensity Delta_X = delta_ratio * 200 (both w.r.t. rho_crit(z)), tabulated."""
    key = round(float(delta_ratio), 10)
    if key not in _CX_CACHE:
        cx = np.array([brentq(lambda x: _mu(x) / x ** 3 - delta_ratio * _mu(c) / c ** 3, 1e-3, 1e3) for c in _C_GRID])
        _CX_CACHE[key] = cx
    return _CX_CACHE[key]


def to_xlensing_mass(M200c, c200c):
    """(M, c) in xlensing's mass definition (Delta = 200 Omega_m0 w.r.t. rho_crit(z)) for an NFW halo of M200c, c200c."""
    import xlensing.cosmo as xc
    ratio = xc.OmegaM                                     # Delta_X / 200
    c200c = np.asarray(c200c, float)
    cx = np.interp(np.log(c200c), np.log(_C_GRID), _cx_of_c(ratio), left=np.nan, right=np.nan)
    Mx = np.asarray(M200c, float) * _mu(cx) / _mu(c200c)
    return Mx, cx


def nfw_profiles_lx(z, logM200c, radii, f_mis=0.0, sigma_off=0.1, two_halo=True, bin_nodes=0, ds_l=None):
    """Delta Sigma [Msun/pc^2] for N clusters with Diemer19 concentrations: (1-f) centred + f miscentred
    (+ two-halo).  ``ds_l`` may pass precomputed xlensing.model.Delta_Sigma_l(z, radii)/1e12 (N, M)."""
    import xlensing.model as xm
    from destacking.model import _nfw_components
    z = np.asarray(z, float)
    logM200c = np.asarray(logM200c, float)
    c = diemer19_c200c(logM200c, z)
    Mx, cx = to_xlensing_mass(10 ** logM200c, c)
    so = np.broadcast_to(np.asarray(sigma_off, float), z.shape)
    cen, mis = _nfw_components(z, Mx, cx, radii, SIGMA_OFF=so, bin_nodes=bin_nodes)
    f = np.asarray(f_mis, float)
    f = f[:, None] if f.ndim else f
    ds = (1 - f) * cen + f * mis
    if two_halo:
        if ds_l is None:
            ds_l = xm.Delta_Sigma_l(z, np.asarray(radii, float)) / 1e12
        ds = ds + xm.Bias(z, Mx)[:, None] * ds_l
    return ds
