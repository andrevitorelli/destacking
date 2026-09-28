"""Forward-modelled stacked fit of the mass-richness and mass-concentration
relations (added 2026-09-26).

Instead of fitting each stack with a free NFW (log M, c) and regressing
those per-stack values on the stacks' richness/redshift/magnitude gap
afterwards, this fits the scaling-relation parameters DIRECTLY to the
stacked ΔΣ profiles.  Each stack's model is the average of its members'
model profiles, combined with exactly the weights the stacked measurement
uses (xlensing.data.stacked_signal's per-cluster, per-bin
K_c(R) = Σ_gal (1+m) w / Σ_crit²):

    ΔΣ_model,s(R) = Σ_{c∈s} K_c(R) ΔΣ_c(R; θ) / Σ_{c∈s} K_c(R)

with ΔΣ_c from the same forward model as the joint fit
(destacking.model: M from the mass-richness relation, c from the power-law
mass-concentration relation, global f_mis/σ_off miscentering).  The whole
distribution of every member's (λ, z, Δm) therefore enters, and the
measurement uncertainty enters through each stack's bootstrap covariance.
The only difference from the joint fit is that the data are stacked.

Two-halo rule (as in gamma_inference.py): --data real includes the two-halo
term, which real data contain; --data sim omits it, because the simulations
shear sources with isolated haloes only.

Minimum stack size: each stack's covariance is a cluster bootstrap over its
N members, so it has rank <= N-1; with 16 radial bins a 10-cluster stack's
covariance is singular and its inverse dominates the whole likelihood (found
2026-09-26: five N=10 stacks carried 37,000 of a 37,789 chi^2 at truth on
the DES-like simulation).  --min-n defaults to 50 and must be >= 2 x bins.

Inputs: the stacked analysis's stacks3.pkl (stacked_3way_dm_test.py or the
real-data notebook), the cluster catalogue it was built from (ID, ZPHOT,
NGALS, DELTA_M_I), and the per-cluster K_c(R) weights (stack_weights npz:
ids, K).  Membership is recomputed with stacked_3way_dm_test's
assign_stack_bins_3way and checked against every stack's n_stacked.

usage:
  python scripts/auxiliary_scripts/stacked_forward_fit.py \\
      --catalog cat.fits --stacks stacks3.pkl --weights weights.npz \\
      --out result.npz --data {real,sim} [--min-n 50] [--n-live 2000] [--workers 12]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import time

import numpy as np
from astropy.table import Table
from scipy import stats
from scipy.sparse import csr_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "stacked_3way_dm_test", os.path.join(_HERE, "stacked_3way_dm_test.py"))
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)
wsa = s3.wsa

import xlensing  # noqa: E402
from nautilus import Prior, Sampler  # noqa: E402

from destacking.model import _nfw_components, bin_average_nodes  # noqa: E402
from destacking.relations import mass_lambda_McClintock18  # noqa: E402

DELTA_M_REF = 0.1          # same as gamma_inference.py
M_PIVOT_LOG = 14.0
Z_REF = 0.35
LOG10_M0_MCCLINTOCK = 14.489

# Same wide priors as the real-data joint fit (v4) and the simulated
# jointfull runs, so joint and forward-stacked are directly comparable.
PARAMS = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log_M_norm",
          "alpha_mr", "beta_mr", "f_mis", "sigma_off"]


def make_prior():
    p = Prior()
    p.add_parameter("c0", dist=stats.truncnorm((0.0 - 4.5) / 10.0, np.inf,
                                               loc=4.5, scale=10.0))
    p.add_parameter("alpha_cm", dist=stats.norm(-0.088, 3.0))
    p.add_parameter("beta_cm", dist=stats.norm(-0.4, 10.0))
    p.add_parameter("gamma_cm", dist=stats.norm(0.0, 10.5))
    p.add_parameter("log_M_norm", dist=stats.norm(0.0, 3.0))
    p.add_parameter("alpha_mr", dist=stats.norm(1.356, 3.0))
    p.add_parameter("beta_mr", dist=stats.norm(-0.30, 3.0))
    p.add_parameter("f_mis", dist=stats.uniform(0.0, 1.0))
    p.add_parameter("sigma_off", dist=stats.uniform(0.0, 1.0))
    return p


class StackedForwardModel:
    """Precomputed members, weights and stacked data; callable log-likelihood."""

    def __init__(self, catalog, stacks, weights, min_n=50, two_halo=False, bin_nodes=0,
                 n_boot_cov=None):
        stacks = stacks["stacks"] if "stacks" in stacks and "z_bin" in stacks else stacks
        keep = np.isfinite(catalog["DELTA_M_I"]) & (catalog["DELTA_M_I"] > 0)
        cat = catalog[keep]
        z_bin, ng_bin, dm_bin, _, _ = s3.assign_stack_bins_3way(
            cat["ZPHOT"], cat["NGALS"], cat["DELTA_M_I"],
            z_edges=wsa.Z_EDGES, ng_edges=wsa.NG_EDGES)
        id_to_row = {int(i): n for n, i in enumerate(weights["ids"])}
        ids = np.asarray(cat["ID"]).astype(np.int64)
        has_w = np.array([int(i) in id_to_row for i in ids])

        self.radii = np.asarray(wsa.radii, dtype=float)
        n_boot = wsa.NBOOT if n_boot_cov is None else n_boot_cov
        rows, cols, names, data, cinv, masks, used = [], [], [], [], [], [], []
        member_rows = []
        for name in sorted(stacks):
            s = stacks[name]
            m = ((z_bin == s["z_bin"]) & (ng_bin == s["ng_bin"])
                 & (dm_bin == s["dm_bin"]) & has_w)
            n = int(m.sum())
            if n != s["n_stacked"]:
                raise RuntimeError(f"{name}: recomputed membership {n} != "
                                   f"n_stacked {s['n_stacked']}")
            if n < min_n:
                continue
            d = np.asarray(s["sigmas"], float)
            C = np.asarray(s["sigmas_cov"], float)
            ok = np.isfinite(d) & np.isfinite(np.diag(C)) & (np.diag(C) > 0)
            if ok.sum() < 3:
                continue
            Cs = C[np.ix_(ok, ok)]
            Cs = Cs + 1e-6 * np.eye(ok.sum()) * np.mean(np.diag(Cs))
            # Hartlap et al. (2007) debiasing of the inverse of a covariance
            # estimated from n_boot bootstrap resamples.
            hart = (n_boot - ok.sum() - 2) / (n_boot - 1)
            k = len(names)
            names.append(name); data.append(d[ok]); masks.append(ok)
            cinv.append(hart * np.linalg.inv(Cs)); used.append(n)
            member_rows.append(np.flatnonzero(m))
            rows.extend([k] * n); cols.extend(np.flatnonzero(m).tolist())

        # compact member set: only clusters that are in some used stack
        allm = np.unique(np.concatenate(member_rows))
        pos = {r: j for j, r in enumerate(allm)}
        cols = [pos[c] for c in cols]
        self.S = csr_matrix((np.ones(len(rows)), (rows, cols)),
                            shape=(len(names), len(allm)))
        sub = cat[allm]
        # members' row indices in the input catalogue (for diagnostics)
        self.catalog_rows = np.flatnonzero(np.asarray(keep))[allm]
        self.z = np.asarray(sub["ZPHOT"], float)
        self.lam = np.asarray(sub["NGALS"], float)
        self.log_dm = np.log10(np.asarray(sub["DELTA_M_I"], float) / DELTA_M_REF)
        self.K = np.asarray(weights["K"], float)[
            [id_to_row[int(i)] for i in np.asarray(sub["ID"])]]
        self.norm = self.S @ self.K                      # (n_stacks, n_bins)
        self.names, self.data, self.cinv, self.masks = names, data, cinv, masks
        self.n_stacked = np.array(used)
        self.two_halo = two_halo
        self.bin_nodes = bin_nodes
        if two_halo:
            if bin_nodes:                      # bin-average the two-halo term too
                nodes, wts = bin_average_nodes(self.radii, bin_nodes)
                dl = xlensing.model.Delta_Sigma_l(self.z, nodes.ravel()) / 1e12
                self.ds_l = (dl.reshape(len(self.z), len(self.radii), -1) * wts).sum(-1)
            else:
                self.ds_l = xlensing.model.Delta_Sigma_l(self.z, self.radii) / 1e12

    def model(self, p):
        logM = (mass_lambda_McClintock18(self.lam, self.z, Flambda=p["alpha_mr"],
                                         Gz=p["beta_mr"]) + p["log_M_norm"])
        logc = (np.log10(p["c0"]) + p["alpha_cm"] * (logM - M_PIVOT_LOG)
                + p["beta_cm"] * np.log10((1 + self.z) / (1 + Z_REF))
                + p["gamma_cm"] * self.log_dm)
        M = 10.0 ** logM
        cen, mis = _nfw_components(self.z, M, 10.0 ** logc, self.radii,
                                   SIGMA_OFF=np.full(len(self.z), p["sigma_off"]),
                                   bin_nodes=self.bin_nodes)
        ds = (1 - p["f_mis"]) * cen + p["f_mis"] * mis
        if self.two_halo:
            ds = ds + xlensing.model.Bias(self.z, M)[:, None] * self.ds_l
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.S @ (self.K * ds)) / self.norm   # (n_stacks, n_bins)

    def loglike(self, p):
        if p["c0"] <= 0 or p["sigma_off"] <= 0:
            return -np.inf
        mod = self.model(p)
        ll = 0.0
        for k in range(len(self.names)):
            r = self.data[k] - mod[k][self.masks[k]]
            if not np.all(np.isfinite(r)):
                return -np.inf
            ll -= 0.5 * r @ self.cinv[k] @ r
        # Extreme prior draws can overflow the quadratic form (inf - inf =
        # NaN); nautilus needs -inf there, as in destacking.fitting.
        return float(ll) if np.isfinite(ll) else -np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--stacks", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True, choices=["real", "sim"],
                    help="real -> two-halo term on; sim -> off (see module docstring)")
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--bin-nodes", type=int, default=0,
                    help="area-weighted average of the model over each radial bin with this many "
                         "Gauss-Legendre nodes (0 = evaluate at the bin centre)")
    ap.add_argument("--n-live", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.min_n < 2 * len(wsa.radii):
        raise SystemExit(f"--min-n {a.min_n} < 2 x {len(wsa.radii)} radial bins: "
                         "bootstrap covariances of such stacks are (near-)singular")

    fm = StackedForwardModel(Table.read(a.catalog), pickle.load(open(a.stacks, "rb")),
                             np.load(a.weights), min_n=a.min_n, two_halo=(a.data == "real"),
                             bin_nodes=a.bin_nodes)
    print(f"{len(fm.names)} stacks (N>={a.min_n}), {fm.S.shape[1]:,} member clusters, "
          f"data={a.data} two_halo={fm.two_halo} bin_nodes={a.bin_nodes}", flush=True)
    t0 = time.time()
    sampler = Sampler(make_prior(), fm.loglike, n_live=a.n_live, pool=a.workers)
    sampler.run(verbose=True)
    pts, logw, logl = sampler.posterior()
    w = np.exp(logw - logw.max()); w /= w.sum()
    np.savez(a.out, samples=pts, weights=w, log_l=logl, param_order=np.array(PARAMS),
             stack_names=np.array(fm.names), n_stacked=fm.n_stacked,
             min_n=a.min_n, two_halo=fm.two_halo, data=a.data, log_z=sampler.log_z)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.out}")
    for j, name in enumerate(PARAMS):
        o = np.argsort(pts[:, j]); c = np.cumsum(w[o])
        lo, me, hi = np.interp([0.16, 0.5, 0.84], c, pts[o, j])
        print(f"  {name:11s} {me:+.3f} +{hi - me:.3f}/-{me - lo:.3f}")


if __name__ == "__main__":
    main()
