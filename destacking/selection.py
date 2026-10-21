"""Generic population, observable, measurement and selection model for cluster scaling relations (2026-10-21).

The lensing frameworks of this package (joint fit, forward-modelled stacks, two-step stacks) need, for each
selected cluster c, the distribution of its true mass given what was measured about it.  This module builds
that distribution on a grid of log10 M,

    w_ck(theta) = P(log M_k | O_hat_c, z_c, I, theta),    sum_k w_ck = 1,

from four independent pieces, each a small object with a declared interface:

* ``Population``: the halo mass function times the comoving volume (colossus; any mass definition/model).
* ``ObservableModel``: P(O | M, z, theta) for one or more observables, multivariate log-normal (log10 space),
  with pluggable mean relations and a scatter covariance that may include correlations.
* measurement errors: per-cluster log10-space Gaussian errors, passed with the data (``O_err``).
* ``Selection``: a product of stages.  Each stage is a function of MEASURED quantities (and sky position,
  redshift) only, and declares which variables it uses (``depends_on``).

Why the selection usually drops out.  By Bayes' theorem

    P(M | O_hat, z, I) = P(I | O_hat, M, z) P(M | O_hat, z) / P(I | O_hat, z).

If every selection stage depends only on variables we condition on (the measured observables in the model, z,
sky position, measurement errors), P(I | O_hat, M, z) = P(I | O_hat, z) and the ratio is one:

    P(M | O_hat, z, I) = P(M | O_hat, z)  propto  dn/dlog M (M, z) N(O_hat; mu(M, z), Sigma_int + Sigma_meas,c).

Malmquist and Eddington biases are both in this expression (through the slope of the mass function and the
total scatter); the selection function itself never needs to be written down ("option A").

When a stage depends on a variable that is NOT conditioned on -- for example an optical detection threshold
on richness when only Lx is modelled -- that stage does not cancel, and the module multiplies in

    P(I_u | M, z, O_hat) = E[ p_stage(o_u_hat) | M, z, O_hat ]

averaging over the conditional distribution of the unconditioned observable o_u given M and the conditioned
observables (Gauss-Hermite quadrature) -- "option B".  ``MassPrior.report()`` says which option each stage
needs, so the same code runs any survey's selection.

The deterministic (no scatter, no errors) limit is a delta function at the inverted relation, which is what the
deterministic fits of the Lx-M analysis use (destacking.lx.mass_from_lx).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.special import erf

LN10 = np.log(10.0)


# ── population ──────────────────────────────────────────────────────────────
class Population:
    """dn/dlog10M (comoving, Mpc^-3) times dV/dz over ``area_deg2`` (Mpc^3 per unit z), from colossus.

    The cosmology is colossus' current one (set it first, e.g. with destacking.lx.set_cosmology).
    """

    def __init__(self, area_deg2, z_range, mdef="200c", model="tinker08", tabulate=True):
        self.area_deg2 = float(area_deg2)
        self.z_range = tuple(map(float, z_range))
        self.mdef, self.model = mdef, model
        self._tab = None
        if tabulate:
            from scipy.interpolate import RegularGridInterpolator
            zz = np.linspace(max(self.z_range[0] - 0.02, 0.0), self.z_range[1] + 0.02, 41)
            lm = np.linspace(11.0, 16.5, 221)
            Z, L = np.meshgrid(zz, lm, indexing="ij")
            self._tab = RegularGridInterpolator((zz, lm), np.log(self._dn_dlogM_exact(L, Z)),
                                                bounds_error=False, fill_value=None)

    def dn_dlogM(self, logM, z):
        """Comoving number density per dex of mass [Mpc^-3], logM in log10 Msun (no h); tabulated (log-linear
        interpolation on a 41 x 221 (z, log M) grid) unless the Population was built with tabulate=False."""
        if self._tab is None:
            return self._dn_dlogM_exact(logM, z)
        logM, z = np.broadcast_arrays(np.asarray(logM, float), np.asarray(z, float))
        return np.exp(self._tab(np.column_stack([z.ravel(), logM.ravel()]))).reshape(logM.shape)

    def _dn_dlogM_exact(self, logM, z):
        from colossus.cosmology import cosmology as ccos
        from colossus.lss import mass_function as mf
        h = ccos.getCurrent().h
        logM = np.asarray(logM, float)
        z = np.asarray(z, float)
        out = np.empty(np.broadcast(logM, z).shape)
        lb, zb = np.broadcast_arrays(logM, z)
        for zz in np.unique(np.round(zb, 4)):
            m = np.round(zb, 4) == zz
            dndlnM = mf.massFunction(10 ** lb[m] * h, float(zz), mdef=self.mdef, model=self.model, q_out="dndlnM")
            out[m] = dndlnM * h ** 3 * LN10            # (Mpc/h)^-3 per ln M -> Mpc^-3 per dex
        return out

    def dV_dz(self, z):
        """Comoving volume per unit redshift over the survey area [Mpc^3]."""
        from colossus.cosmology import cosmology as ccos
        c = ccos.getCurrent()
        z = np.asarray(z, float)
        zf = z.ravel()
        Dc = (c.comovingDistance(z_min=np.zeros_like(zf), z_max=zf, transverse=True) / c.h).reshape(z.shape)  # Mpc
        dVdzdOmega = 299792.458 / (100.0 * c.h) * Dc ** 2 / c.Ez(z)   # Mpc^3 / sr
        return dVdzdOmega * self.area_deg2 * (np.pi / 180) ** 2

    def sample(self, n_expected=None, logM_min=13.0, logM_max=15.6, rng=None, n_z=200, n_m=400):
        """Poisson draw of (log10 M, z) from the population.  Returns a dict of arrays."""
        rng = np.random.default_rng(rng)
        zz = np.linspace(*self.z_range, n_z)
        lm = np.linspace(logM_min, logM_max, n_m)
        Z, L = np.meshgrid(zz, lm, indexing="ij")
        dens = self.dn_dlogM(L, Z) * self.dV_dz(Z)               # per unit z per dex
        cell = dens * (zz[1] - zz[0]) * (lm[1] - lm[0])
        N = rng.poisson(cell.sum()) if n_expected is None else int(n_expected)
        p = cell.ravel() / cell.sum()
        k = rng.choice(p.size, size=N, p=p)
        z = Z.ravel()[k] + (rng.random(N) - 0.5) * (zz[1] - zz[0])
        logM = L.ravel()[k] + (rng.random(N) - 0.5) * (lm[1] - lm[0])
        return {"logM": logM, "z": np.clip(z, *self.z_range)}


# ── observables ─────────────────────────────────────────────────────────────
@dataclass
class Relation:
    """One observable: log10 O = mean(log10 M, z, theta), with log-normal intrinsic scatter ``scatter(theta)`` [dex].

    ``mean`` and ``scatter`` take the parameter dict theta; names are free, so any relation can be plugged in.
    """
    name: str
    mean: Callable
    scatter: Callable


class ObservableModel:
    """Multivariate log-normal P(log10 O | log10 M, z, theta) for several observables.

    ``corr`` (optional) is a function theta -> correlation matrix of the intrinsic scatter (default: identity).
    """

    def __init__(self, relations: Sequence[Relation], corr: Callable | None = None):
        self.relations = list(relations)
        self.names = [r.name for r in self.relations]
        self.corr = corr

    def mean(self, logM, z, theta):
        return np.stack([r.mean(logM, z, theta) for r in self.relations], axis=-1)

    def cov_int(self, theta):
        s = np.array([r.scatter(theta) for r in self.relations], float)
        R = np.eye(len(s)) if self.corr is None else np.asarray(self.corr(theta), float)
        return R * np.outer(s, s)

    def sample(self, logM, z, theta, rng=None):
        rng = np.random.default_rng(rng)
        mu = self.mean(logM, z, theta)
        C = self.cov_int(theta)
        return mu + rng.multivariate_normal(np.zeros(len(self.names)), C, size=len(np.atleast_1d(logM)))


# ── selection ───────────────────────────────────────────────────────────────
@dataclass
class Stage:
    """One selection stage: ``prob(data) -> P(selected)`` in [0, 1] per cluster, where ``data`` is a dict of
    arrays.  ``depends_on`` lists the keys it reads (measured observables are named '<observable>_obs')."""
    name: str
    depends_on: tuple
    prob: Callable


def hard_cut(name, var, lo=-np.inf, hi=np.inf):
    return Stage(name, (var,), lambda d: ((d[var] > lo) & (d[var] < hi)).astype(float))


def erf_completeness(name, var, x50, width):
    """P = [1 + erf((x - x50) / (sqrt(2) width))] / 2 -- e.g. a detection threshold in a noisy measured quantity."""
    return Stage(name, (var,), lambda d: 0.5 * (1 + erf((d[var] - x50) / (np.sqrt(2) * width))))


def signal_to_noise_cut(name, var, err_var, snr_min):
    """Measured log-observable with log-space error: keep if S/N = 1 / (ln10 sigma_log) >= snr_min.
    Depends only on the measurement error, e.g. the eSCALE sigma_Lx / Lx < 0.5 cut (S/N >= 2)."""
    return Stage(name, (err_var,), lambda d: (1.0 / (LN10 * np.asarray(d[err_var])) >= snr_min).astype(float))


@dataclass
class Selection:
    stages: list = field(default_factory=list)

    def prob(self, data):
        p = 1.0
        for s in self.stages:
            p = p * s.prob(data)
        return p

    def variables(self):
        return sorted({v for s in self.stages for v in s.depends_on})


# ── mass prior ──────────────────────────────────────────────────────────────
class MassPrior:
    """Per-cluster P(log M | O_hat, z, I, theta) on a log10 M grid.

    Parameters
    ----------
    population : Population
    observables : ObservableModel
    selection : Selection
    conditioned : the observable names that are measured and used (e.g. ['logLx']).  Others in ``observables``
        are latent: they may only enter through selection stages (option B).
    always_known : variables that never depend on mass beyond what is modelled (redshift, sky position,
        measurement errors).  A stage depending only on these and on conditioned observables cancels.
    logM_grid : grid of log10 M [Msun].
    gh_nodes : Gauss-Hermite nodes for option-B averages.
    """

    def __init__(self, population, observables, selection, conditioned, logM_grid=None,
                 always_known=("z", "ra", "dec"), gh_nodes=24, measurement=None, known_extra=()):
        self.pop, self.obs, self.sel = population, observables, selection
        self.cond = list(conditioned)
        self.lat = [n for n in observables.names if n not in self.cond]
        self.grid = np.linspace(12.8, 15.8, 151) if logM_grid is None else np.asarray(logM_grid, float)
        # measurement: optional {observable: callable(true_log10 (N, K, G), data) -> log P(data | true)} replacing
        # the default Gaussian-in-log10 error for a conditioned observable (e.g. a Gaussian in linear flux, or
        # Poisson counts).  Only for a single conditioned observable; the intrinsic scatter is then integrated
        # numerically (Gauss-Hermite).  ``known_extra`` names further data keys it reads (e.g. 'flux_err').
        self.meas = dict(measurement or {})
        if self.meas and (len(self.cond) != 1 or set(self.meas) != set(self.cond)):
            raise NotImplementedError("a custom measurement model is supported for a single conditioned observable")
        known = (set(always_known) | {f"{n}_obs" for n in self.cond} | {f"{n}_err" for n in self.cond}
                 | set(known_extra))
        self.cancel, self.keep = [], []
        for s in selection.stages:
            (self.cancel if set(s.depends_on) <= known else self.keep).append(s)
        for s in self.keep:
            unknown = [v for v in s.depends_on if v not in known]
            bad = [v for v in unknown if not (v.endswith("_obs") and v[:-4] in self.lat)]
            if bad:
                raise ValueError(f"stage {s.name!r} depends on {bad}: neither conditioned on nor a modelled latent "
                                 f"observable -- add it to the ObservableModel or to always_known")
        self.gh_x, self.gh_w = np.polynomial.hermite_e.hermegauss(gh_nodes)
        self.gh_w = self.gh_w / self.gh_w.sum()

    def report(self):
        lines = ["selection stage -> treatment"]
        for s in self.cancel:
            lines.append(f"  {s.name:32s} depends on {s.depends_on}: conditioned on -> cancels (option A)")
        for s in self.keep:
            lines.append(f"  {s.name:32s} depends on {s.depends_on}: NOT conditioned on -> P(I|M) term (option B)")
        return "\n".join(lines)

    def log_weights(self, data, theta, lat_err=None):
        """Unnormalised log w_ck for N clusters (rows) on the mass grid (columns).

        ``data`` holds 'z' and, for each conditioned observable n, '<n>_obs' (log10) and '<n>_err' (log10-space
        1-sigma measurement error; 0 for none).  ``lat_err`` maps a latent observable to its measurement error
        (scalar or per cluster) for the option-B average.
        """
        z = np.asarray(data["z"], float)
        N, K = len(z), len(self.grid)
        L = np.broadcast_to(self.grid, (N, K))
        Zg = np.broadcast_to(z[:, None], (N, K))
        logw = np.log(self.pop.dn_dlogM(L, Zg))
        mu = self.obs.mean(L, Zg, theta)                       # (N, K, n_obs)
        C = self.obs.cov_int(theta)
        ic = [self.obs.names.index(n) for n in self.cond]
        if self.meas:
            return logw + self._log_obs_custom(data, theta, mu, C, ic[0], lat_err)
        if ic:
            x = np.stack([np.asarray(data[f"{n}_obs"], float) for n in self.cond], -1)        # (N, nc)
            e = np.stack([np.broadcast_to(np.asarray(data.get(f"{n}_err", 0.0), float), (N,)) for n in self.cond], -1)
            Ccc = C[np.ix_(ic, ic)][None] + np.einsum("ni,ij->nij", e ** 2, np.eye(len(ic)))   # (N, nc, nc)
            r = x[:, None, :] - mu[..., ic]                                                  # (N, K, nc)
            Ci = np.linalg.inv(Ccc)
            logw = logw - 0.5 * np.einsum("nki,nij,nkj->nk", r, Ci, r) - 0.5 * np.log(np.linalg.det(Ccc))[:, None]
        for s in self.keep:
            logw = logw + np.log(np.clip(self._p_select_latent(s, data, theta, mu, C, ic, x if ic else None,
                                                               e if ic else None, lat_err), 1e-300, None))
        return logw

    def _log_obs_custom(self, data, theta, mu, C, c, lat_err):
        """log of  int d o_true N(o_true; mu_c, C_cc) P(data | o_true) prod_keep E[p_stage | o_true]  (G-H in o_true)."""
        meas = self.meas[self.cond[0]]
        N, K = mu.shape[:2]
        sc = np.sqrt(C[c, c])
        true = mu[..., c][..., None] + sc * self.gh_x[None, None, :]              # (N, K, G)
        lw = np.log(self.gh_w)[None, None, :] + meas(true, data)
        for s in self.keep:
            lat = [v[:-4] for v in s.depends_on if v.endswith("_obs") and v[:-4] in self.lat]
            if len(lat) != 1:
                raise NotImplementedError("option B is implemented for stages reading one latent observable")
            u = self.obs.names.index(lat[0])
            m_u = mu[..., u][..., None] + (C[u, c] / sc) * self.gh_x[None, None, :]
            v_u = C[u, u] - C[u, c] ** 2 / C[c, c] + (0.0 if lat_err is None else float(lat_err.get(lat[0], 0.0)) ** 2)
            p = np.zeros_like(true)
            base = {k: (np.asarray(v)[:, None, None] * np.ones((1, K, len(self.gh_x))) if np.ndim(v) else v)
                    for k, v in data.items()}
            for xg, wg in zip(self.gh_x, self.gh_w):
                d = dict(base); d[f"{lat[0]}_obs"] = m_u + np.sqrt(max(v_u, 0.0)) * xg
                p += wg * s.prob(d)
            lw = lw + np.log(np.clip(p, 1e-300, None))
        mx = lw.max(axis=-1, keepdims=True)
        return (mx + np.log(np.exp(lw - mx).sum(-1, keepdims=True)))[..., 0]

    def _p_select_latent(self, stage, data, theta, mu, C, ic, x, e, lat_err):
        """E[p_stage] over the latent observables the stage reads, given M (grid) and the conditioned data.
        One latent observable per stage (Gauss-Hermite); the conditional mean/variance follow from the joint
        Gaussian of intrinsic scatter + measurement errors."""
        lat = [v[:-4] for v in stage.depends_on if v.endswith("_obs") and v[:-4] in self.lat]
        if len(lat) != 1:
            raise NotImplementedError("option B is implemented for stages reading one latent observable")
        u = self.obs.names.index(lat[0])
        N, K = mu.shape[:2]
        mu_u = mu[..., u]
        var_u = C[u, u] + (0.0 if lat_err is None else np.asarray(lat_err.get(lat[0], 0.0), float) ** 2)
        var_u = np.broadcast_to(var_u, (N,))[:, None] * np.ones((1, K))
        if ic:
            Cuc = C[u, ic]                                                 # (nc,)
            Ccc = C[np.ix_(ic, ic)][None] + np.einsum("ni,ij->nij", e ** 2, np.eye(len(ic)))
            Ci = np.linalg.inv(Ccc)
            r = x[:, None, :] - mu[..., ic]
            mu_u = mu_u + np.einsum("i,nij,nkj->nk", Cuc, Ci, r)
            var_u = var_u - np.einsum("i,nij,j->n", Cuc, Ci, Cuc)[:, None]
        sd = np.sqrt(np.clip(var_u, 0, None))
        out = np.zeros((N, K))
        base = {k: (np.asarray(v)[:, None] * np.ones((1, K)) if np.ndim(v) else v) for k, v in data.items()}
        for xg, wg in zip(self.gh_x, self.gh_w):
            d = dict(base); d[f"{lat[0]}_obs"] = mu_u + sd * xg
            out += wg * stage.prob(d)
        return out

    def weights(self, data, theta, lat_err=None):
        lw = self.log_weights(data, theta, lat_err)
        lw = lw - lw.max(axis=1, keepdims=True)
        w = np.exp(lw)
        return w / w.sum(axis=1, keepdims=True)

    def moments(self, data, theta, lat_err=None):
        """Posterior mean and standard deviation of log10 M per cluster."""
        w = self.weights(data, theta, lat_err)
        m = w @ self.grid
        return m, np.sqrt(np.clip(w @ self.grid ** 2 - m ** 2, 0, None))
