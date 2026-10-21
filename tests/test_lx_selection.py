"""Tests for destacking.lx (Lx-M relation, mass definitions) and destacking.selection (2026-10-21)."""
import numpy as np
import pytest

from destacking import lx
from destacking.selection import (MassPrior, ObservableModel, Population, Relation, Selection, Stage,
                                  erf_completeness, hard_cut)

PIV = lx.Pivots(43.26, 14.1, 0.18)


@pytest.fixture(autouse=True, scope="module")
def _cosmology():
    lx.set_cosmology()
    yield
    lx.restore_default_cosmology()


def test_relation_inverse():
    z = np.array([0.1, 0.18, 0.24])
    lm = np.array([13.4, 14.0, 14.7])
    ll = lx.lx_from_mass(lm, z, 0.1, 1.3, 0.5, PIV)
    assert np.allclose(lx.mass_from_lx(ll, z, 0.1, 1.3, 0.5, PIV), lm)


def test_xlensing_mass_keeps_nfw_scale_radius():
    import xlensing.cosmo as xc, xlensing.model as xm
    z, M, c = 0.2, 3e14, 4.5
    Mx, cx = lx.to_xlensing_mass(M, c)
    r200c = (3 * M / (4 * np.pi * 200 * xc.rhocz(z))) ** (1 / 3)
    assert np.isclose(xm.r_vir_m(z, Mx) / cx, r200c / c, rtol=1e-4)


def test_diemer_range():
    c = lx.diemer19_c200c([12.0, 14.0, 15.5], [0.18] * 3)
    assert np.all(np.isfinite(c)) and np.all((c > 2) & (c < 12))


def _model(theta):
    rel = Relation("logLx", lambda m, z, t: lx.lx_from_mass(m, z, t["alpha"], t["beta"], t["gamma"], PIV),
                   lambda t: t["sig"])
    lam = Relation("logLam", lambda m, z, t: 1.2 + 0.9 * (m - 14.1), lambda t: 0.2)
    return ObservableModel([rel, lam], corr=lambda t: np.array([[1, 0.3], [0.3, 1]]))


def test_stage_classification_and_unknown_variable():
    th = dict(alpha=0.0, beta=1.3, gamma=0.0, sig=0.2)
    pop = Population(100, (0.1, 0.2))
    sel = Selection([hard_cut("flux", "logLx_obs", lo=42.5), erf_completeness("opt", "logLam_obs", 1.0, 0.1)])
    mp = MassPrior(pop, _model(th), sel, conditioned=["logLx"])
    assert [s.name for s in mp.cancel] == ["flux"] and [s.name for s in mp.keep] == ["opt"]
    with pytest.raises(ValueError):
        MassPrior(pop, _model(th), Selection([hard_cut("x", "unmodelled_obs")]), conditioned=["logLx"])


def test_option_a_recovers_mean_mass_under_lx_selection():
    """Selection on measured Lx only: the conditional prior's mean mass matches the truth of the selected
    clusters, while the deterministic inversion is biased (Eddington)."""
    th = dict(alpha=0.0, beta=1.3, gamma=0.0, sig=0.25)
    om = ObservableModel([_model(th).relations[0]])
    pop = Population(800, (0.1, 0.25))
    rng = np.random.default_rng(3)
    s = pop.sample(rng=rng, logM_min=12.8)
    o = om.sample(s["logM"], s["z"], th, rng=rng)[:, 0] + 0.1 * rng.standard_normal(len(s["z"]))
    keep = o > 43.0
    d = {"z": s["z"][keep], "logLx_obs": o[keep], "logLx_err": np.full(keep.sum(), 0.1)}
    mp = MassPrior(pop, om, Selection([hard_cut("flux", "logLx_obs", lo=43.0)]), conditioned=["logLx"])
    m, _ = mp.moments(d, th)
    det = lx.mass_from_lx(d["logLx_obs"], d["z"], th["alpha"], th["beta"], th["gamma"], PIV)
    assert abs(np.mean(s["logM"][keep] - m)) < 0.03
    assert np.mean(s["logM"][keep] - det) < -0.08
