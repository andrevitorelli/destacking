import json
from destacking.dmform import DM_FORM
import pickle

import numpy as np
import pytest
from astropy.table import Table

from destacking.io import (
    save_simulation_checkpoint,
    load_simulation_checkpoint,
    write_notebook_compat_sim,
)


def _make_truth(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return Table({
        "z":             rng.uniform(0.1, 0.6, n),
        "M_halo":        10 ** rng.uniform(13.5, 15.0, n),
        "concentration": rng.uniform(2.0, 8.0, n),
        "delta_m":       rng.uniform(0.0, 2.0, n),
    })


def _make_cluster_obs(n=6, seed=1, with_stack=False):
    rng = np.random.default_rng(seed)
    cols = {
        "z":       rng.uniform(0.1, 0.6, n),
        "Lambda":  rng.uniform(20.0, 100.0, n),
        "delta_m": rng.uniform(0.0, 2.0, n),
    }
    if with_stack:
        cols["stack_name"] = np.array([f"bin{i % 2}" for i in range(n)])
    return Table(cols)


# ---------------------------------------------------------------------------
# save_simulation_checkpoint / load_simulation_checkpoint round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip_basic(tmp_path):
    n, k = 6, 5
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    ols = {"c0_true": 4.5, "alpha_true": -0.1, "beta_true": -0.3}
    data_vectors = np.random.default_rng(2).normal(size=(n, k))
    cov_matrices = np.stack([np.eye(k) for _ in range(n)])

    save_simulation_checkpoint(
        str(tmp_path), truth, ols, cluster_obs, data_vectors, cov_matrices,
    )
    (truth_r, ols_r, cluster_obs_r, dv_r, cv_r) = load_simulation_checkpoint(
        str(tmp_path)
    )

    np.testing.assert_allclose(dv_r, data_vectors)
    np.testing.assert_allclose(cv_r, cov_matrices)
    np.testing.assert_allclose(np.asarray(truth_r["z"]), np.asarray(truth["z"]))
    np.testing.assert_allclose(
        np.asarray(truth_r["M_halo"]), np.asarray(truth["M_halo"])
    )
    np.testing.assert_allclose(
        np.asarray(truth_r["concentration"]), np.asarray(truth["concentration"])
    )
    np.testing.assert_allclose(
        np.asarray(cluster_obs_r["Lambda"]), np.asarray(cluster_obs["Lambda"])
    )
    assert ols_r == dict(ols, dm_form=DM_FORM)


def test_checkpoint_writes_expected_files(tmp_path):
    n, k = 4, 3
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    ols = {"a": 1}
    data_vectors = np.zeros((n, k))
    cov_matrices = np.stack([np.eye(k) for _ in range(n)])

    save_simulation_checkpoint(
        str(tmp_path), truth, ols, cluster_obs, data_vectors, cov_matrices,
    )
    for name in ("simulated_data.npz", "truth_catalog.npz",
                 "cluster_obs.npz", "ols.json"):
        assert (tmp_path / name).exists()


def test_checkpoint_ols_json_is_human_readable(tmp_path):
    n, k = 3, 2
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    ols = {"c0_true": 4.5}
    save_simulation_checkpoint(
        str(tmp_path), truth, ols, cluster_obs,
        np.zeros((n, k)), np.stack([np.eye(k)] * n),
    )
    with open(tmp_path / "ols.json") as f:
        loaded = json.load(f)
    # the checkpoint also records the magnitude-gap form of the run (destacking.dmform)
    assert loaded == dict(ols, dm_form=DM_FORM)


def test_checkpoint_stack_name_included_when_present(tmp_path):
    n, k = 4, 2
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n, with_stack=True)
    save_simulation_checkpoint(
        str(tmp_path), truth, {}, cluster_obs,
        np.zeros((n, k)), np.stack([np.eye(k)] * n),
    )
    _, _, cluster_obs_r, _, _ = load_simulation_checkpoint(str(tmp_path))
    assert "stack_name" in cluster_obs_r.colnames
    np.testing.assert_array_equal(
        np.asarray(cluster_obs_r["stack_name"]),
        np.asarray(cluster_obs["stack_name"]),
    )


def test_checkpoint_stack_name_omitted_when_absent(tmp_path):
    n, k = 4, 2
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n, with_stack=False)
    save_simulation_checkpoint(
        str(tmp_path), truth, {}, cluster_obs,
        np.zeros((n, k)), np.stack([np.eye(k)] * n),
    )
    _, _, cluster_obs_r, _, _ = load_simulation_checkpoint(str(tmp_path))
    assert "stack_name" not in cluster_obs_r.colnames


def test_checkpoint_covariance_shape_preserved(tmp_path):
    n, k = 5, 7
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    rng = np.random.default_rng(9)
    cov_matrices = np.stack([
        A @ A.T + np.eye(k) for A in rng.standard_normal((n, k, k))
    ])
    save_simulation_checkpoint(
        str(tmp_path), truth, {}, cluster_obs,
        np.zeros((n, k)), cov_matrices,
    )
    _, _, _, _, cv_r = load_simulation_checkpoint(str(tmp_path))
    assert cv_r.shape == (n, k, k)
    np.testing.assert_allclose(cv_r, cov_matrices)


# ---------------------------------------------------------------------------
# write_notebook_compat_sim
# ---------------------------------------------------------------------------

def test_write_notebook_compat_sim_creates_expected_files(tmp_path):
    n = 5
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    bgs = [{"ra": 1.0, "dec": 2.0}] * n

    write_notebook_compat_sim(
        g_idx=0, gamma_true=0.5, truth=truth, cluster_obs=cluster_obs,
        bgs=bgs, out_dir=str(tmp_path), prefix_root="sim_g",
        density=7.5, shape_noise=0.27, z_src_range=(0.2, 2.0),
    )
    prefix = "sim_g000"
    assert (tmp_path / f"{prefix}_real_data.fits").exists()
    assert (tmp_path / f"{prefix}_bgs.pkl").exists()
    assert (tmp_path / f"{prefix}_metadata.json").exists()


def test_write_notebook_compat_sim_fits_table_columns_and_length(tmp_path):
    n = 8
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    bgs = [{"dummy": i} for i in range(n)]

    write_notebook_compat_sim(
        g_idx=1, gamma_true=-0.2, truth=truth, cluster_obs=cluster_obs,
        bgs=bgs, out_dir=str(tmp_path), prefix_root="sim_g",
    )
    tab = Table.read(tmp_path / "sim_g001_real_data.fits")
    assert len(tab) == n
    for col in ("ID", "RA", "DEC", "NGALS", "ZPHOT", "DELTA_M_I",
                "DELTA_M_I_ERR"):
        assert col in tab.colnames
    np.testing.assert_allclose(
        np.asarray(tab["ZPHOT"]), np.asarray(truth["z"]), rtol=1e-5,
    )


def test_write_notebook_compat_sim_drops_none_bgs(tmp_path):
    n = 4
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    bgs = [{"a": 1}, None, {"a": 3}, None]

    write_notebook_compat_sim(
        g_idx=2, gamma_true=0.0, truth=truth, cluster_obs=cluster_obs,
        bgs=bgs, out_dir=str(tmp_path), prefix_root="sim_g",
    )
    with open(tmp_path / "sim_g002_bgs.pkl", "rb") as f:
        bgs_by_id = pickle.load(f)
    assert len(bgs_by_id) == 2
    assert set(bgs_by_id.keys()) == {0, 2}


def test_write_notebook_compat_sim_metadata_fields(tmp_path):
    n = 3
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    bgs = [{"a": 1}] * n

    write_notebook_compat_sim(
        g_idx=4, gamma_true=0.3, truth=truth, cluster_obs=cluster_obs,
        bgs=bgs, out_dir=str(tmp_path), prefix_root="sim_g",
        density=7.5, shape_noise=0.27, z_src_range=(0.2, 2.0),
    )
    with open(tmp_path / "sim_g004_metadata.json") as f:
        meta = json.load(f)
    assert meta["gamma"] == pytest.approx(0.3)
    assert meta["g_idx"] == 4
    assert meta["n_truth"] == n
    assert meta["n_usable_bgs"] == n
    assert meta["density"] == 7.5
    assert meta["shape_noise"] == 0.27
    assert meta["z_src_range"] == [0.2, 2.0]
    assert "generated_at" in meta


def test_write_notebook_compat_sim_prefix_depends_on_g_idx(tmp_path):
    n = 2
    truth = _make_truth(n)
    cluster_obs = _make_cluster_obs(n)
    bgs = [{"a": 1}] * n

    write_notebook_compat_sim(
        g_idx=42, gamma_true=0.0, truth=truth, cluster_obs=cluster_obs,
        bgs=bgs, out_dir=str(tmp_path), prefix_root="myprefix_g",
    )
    assert (tmp_path / "myprefix_g042_real_data.fits").exists()
