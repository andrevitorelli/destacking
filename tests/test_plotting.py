import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.stats import norm as sp_norm

from destacking.plotting import (
    plot_gamma_scan_results,
    plot_posteriors_overlay,
    overlay_priors_on_corner,
    plot_combined_corner,
)


# ---------------------------------------------------------------------------
# plot_gamma_scan_results
# ---------------------------------------------------------------------------

def _make_scan_results(n=11, seed=0):
    rng = np.random.default_rng(seed)
    gamma_true = np.linspace(-0.5, 1.0, n)
    return [
        {
            "gamma_true":         float(g),
            "gamma_ensemble_med": float(g + rng.normal(0, 0.02)),
            "sigma_gamma":        float(0.05 + 0.01 * rng.random()),
            "gamma_true_ols":     float(g + rng.normal(0, 0.001)),
        }
        for g in gamma_true
    ]


def test_plot_gamma_scan_results_creates_file(tmp_path):
    scan_results = _make_scan_results()
    plot_gamma_scan_results(scan_results, str(tmp_path), n_clusters=1000,
                            n_bootstrap=5)
    assert (tmp_path / "gamma_scan_results.png").exists()


def test_plot_gamma_scan_results_file_nonempty(tmp_path):
    scan_results = _make_scan_results()
    plot_gamma_scan_results(scan_results, str(tmp_path), n_clusters=1000,
                            n_bootstrap=5)
    assert (tmp_path / "gamma_scan_results.png").stat().st_size > 0


def test_plot_gamma_scan_results_handles_single_point_gracefully():
    # A degenerate 2-point scan is the minimum for a linear fit (lstsq
    # needs >= 2 points for a 2-parameter fit) -- must not raise.
    scan_results = _make_scan_results(n=2)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        plot_gamma_scan_results(scan_results, d, n_clusters=100, n_bootstrap=1)
        assert os.path.exists(os.path.join(d, "gamma_scan_results.png"))


# ---------------------------------------------------------------------------
# plot_posteriors_overlay
# ---------------------------------------------------------------------------

def _write_bootstrap_posteriors(sim_subdir, n_boot=4, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for b in range(n_boot):
        boot_dir = sim_subdir / f"bootstrap_{b:03d}"
        boot_dir.mkdir(parents=True, exist_ok=True)
        samples = rng.normal(0.3, 0.1, size=500)
        weights = rng.uniform(0.5, 1.5, size=500)
        np.savez(boot_dir / "gamma_posterior.npz",
                gamma_samples=samples, weights=weights)
        results.append({"bootstrap": b})
    return results


def test_plot_posteriors_overlay_creates_file(tmp_path):
    results = _write_bootstrap_posteriors(tmp_path)
    plot_posteriors_overlay(
        results, str(tmp_path), gamma_true=0.3,
        gamma_prior=sp_norm(loc=0.0, scale=3.5), n_clusters=1000,
    )
    assert (tmp_path / "posteriors_overlay.png").exists()


def test_plot_posteriors_overlay_missing_npz_skipped_not_raised(tmp_path):
    # A results entry with no matching gamma_posterior.npz on disk must be
    # silently skipped, not raise.
    results = [{"bootstrap": 0}, {"bootstrap": 99}]  # 99 has no file
    (tmp_path / "bootstrap_000").mkdir()
    rng = np.random.default_rng(1)
    np.savez(tmp_path / "bootstrap_000" / "gamma_posterior.npz",
             gamma_samples=rng.normal(0, 1, 200),
             weights=np.ones(200))
    plot_posteriors_overlay(
        results, str(tmp_path), gamma_true=0.0,
        gamma_prior=sp_norm(loc=0.0, scale=3.5), n_clusters=500,
    )
    assert (tmp_path / "posteriors_overlay.png").exists()


def test_plot_posteriors_overlay_empty_results_does_not_raise(tmp_path):
    plot_posteriors_overlay(
        [], str(tmp_path), gamma_true=0.0,
        gamma_prior=sp_norm(loc=0.0, scale=3.5), n_clusters=100,
    )
    assert (tmp_path / "posteriors_overlay.png").exists()


# ---------------------------------------------------------------------------
# overlay_priors_on_corner
# ---------------------------------------------------------------------------

def _make_corner_fig(n_params=2, n_samples=1000, seed=0):
    import corner as _corner
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 1, size=(n_samples, n_params))
    param_order = [f"p{i}" for i in range(n_params)]
    fig = _corner.corner(
        samples, labels=param_order,
        hist_kwargs={"density": True},
    )
    return fig, param_order


def test_overlay_priors_on_corner_returns_true_and_adds_legend():
    fig, param_order = _make_corner_fig(n_params=2)
    priors = {name: sp_norm(loc=0.0, scale=1.0) for name in param_order}
    drawn = overlay_priors_on_corner(fig, param_order, priors)
    assert drawn is True
    assert len(fig.legends) > 0


def test_overlay_priors_on_corner_skips_missing_priors():
    fig, param_order = _make_corner_fig(n_params=2)
    # Only one of the two params has a prior with a .pdf method.
    priors = {param_order[0]: sp_norm(loc=0.0, scale=1.0)}
    drawn = overlay_priors_on_corner(fig, param_order, priors)
    assert drawn is True


def test_overlay_priors_on_corner_no_matching_priors_returns_false():
    fig, param_order = _make_corner_fig(n_params=2)
    drawn = overlay_priors_on_corner(fig, param_order, priors={})
    assert drawn is False


def test_overlay_priors_on_corner_non_pdf_object_skipped():
    fig, param_order = _make_corner_fig(n_params=2)
    priors = {name: object() for name in param_order}   # no .pdf attribute
    drawn = overlay_priors_on_corner(fig, param_order, priors)
    assert drawn is False


def test_overlay_priors_on_corner_too_few_axes_returns_false():
    fig, param_order = _make_corner_fig(n_params=2)
    # Claim 5 params when the figure only has a 2x2 grid of axes.
    fake_param_order = param_order + ["p2", "p3", "p4"]
    priors = {name: sp_norm(loc=0.0, scale=1.0) for name in fake_param_order}
    drawn = overlay_priors_on_corner(fig, fake_param_order, priors)
    assert drawn is False


# ---------------------------------------------------------------------------
# plot_combined_corner
# ---------------------------------------------------------------------------

def test_plot_combined_corner_creates_file(tmp_path):
    rng = np.random.default_rng(0)
    n, k = 800, 3
    samples = rng.normal(0, 1, size=(n, k))
    weights = rng.uniform(0.1, 1.0, size=n)
    param_order = ["c0", "alpha_cm", "gamma_cm"]
    out_path = str(tmp_path / "combined_corner.png")

    result = plot_combined_corner(
        samples, weights, param_order, out_path,
        gamma_true=0.3, n_bootstrap=10, n_clusters=1000,
    )
    assert result == out_path
    assert (tmp_path / "combined_corner.png").exists()


def test_plot_combined_corner_with_priors_overlay(tmp_path):
    rng = np.random.default_rng(1)
    n, k = 500, 2
    samples = rng.normal(0, 1, size=(n, k))
    weights = np.ones(n)
    param_order = ["a", "b"]
    priors = {name: sp_norm(loc=0.0, scale=1.0) for name in param_order}
    out_path = str(tmp_path / "with_priors.png")

    result = plot_combined_corner(
        samples, weights, param_order, out_path, priors=priors,
    )
    assert result == out_path
    assert (tmp_path / "with_priors.png").exists()


def test_plot_combined_corner_returns_none_when_corner_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "corner", None)
    rng = np.random.default_rng(2)
    samples = rng.normal(0, 1, size=(100, 2))
    weights = np.ones(100)
    result = plot_combined_corner(
        samples, weights, ["a", "b"], str(tmp_path / "unused.png"),
    )
    assert result is None
    assert not (tmp_path / "unused.png").exists()


def test_plot_combined_corner_returns_none_on_failure_not_raises(tmp_path):
    # Malformed samples (wrong shape vs param_order) should be caught and
    # turned into a None return, not propagate an exception.
    result = plot_combined_corner(
        samples=np.zeros((10, 3)),
        weights=np.ones(10),
        param_order=["a", "b"],   # mismatched: 2 names for 3 columns
        out_path=str(tmp_path / "bad.png"),
    )
    assert result is None
