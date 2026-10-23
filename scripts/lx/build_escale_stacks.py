"""Stacks, member weights and boost factors for the eSCALE Lx-M analysis (2026-10-21).

Bins: 3 redshift bins (sample terciles) x 5 Lx/E(z) bins (quintiles within each redshift bin), ~100 clusters each.
Each stack is measured with xlensing.data.stacked_signal on paper 1.1's 16-bin grid (0.16-3.98 Mpc centres,
500 cluster bootstraps, boost-corrected).  Writes
  stacks.pkl   {name: {z_bin, lx_bin, n_stacked, ids, z_mean, logLxE_mean, sigmas, sigmas_cov, boosts, boosts_cov}}
  weights.npz  ids, K (N, 16): K_c(R) = sum_gal (1+m) w / Sigma_crit^2 (forward-modelled stacks)
  boosts.npz   per-cluster boost on the joint fit's 6-bin grid, from its stack's fitted B(R) = 1 + B0 (R/Rs)^alpha
usage: python build_escale_stacks.py
"""
import os, pickle, time
import numpy as np
from scipy.optimize import curve_fit
from destacking import lx
lx.set_cosmology()
import xlensing
from cluster_toolkit.boostfactors import boost_powerlaw_at_R
from astropy.table import Table

D = os.path.expanduser("~/destacking_data/lx_splus_erosita_des")
M = f"{D}/measurement"
NZ, NL, NBOOT = 3, 5, 500
R16 = np.logspace(-0.8, 0.6, 16); d = np.diff(np.log10(R16))[0]
e = np.logspace(np.log10(R16[0]) - d / 2, np.log10(R16[-1]) + d / 2, 17); BL16 = np.column_stack([e[:-1], e[1:]])


def assign_bins(z, logLxE, nz=NZ, nl=NL):
    """Redshift terciles, then Lx/E(z) quintiles within each redshift bin.  Returns (z_bin, lx_bin, edges)."""
    z_edges = np.quantile(z, np.linspace(0, 1, nz + 1)); z_edges[0] -= 1e-9; z_edges[-1] += 1e-9
    zb = np.clip(np.searchsorted(z_edges, z, side="right") - 1, 0, nz - 1)
    lb = np.zeros(len(z), int); l_edges = {}
    for i in range(nz):
        m = zb == i
        ed = np.quantile(logLxE[m], np.linspace(0, 1, nl + 1)); ed[0] -= 1e-9; ed[-1] += 1e-9
        lb[m] = np.clip(np.searchsorted(ed, logLxE[m], side="right") - 1, 0, nl - 1); l_edges[i] = ed
    return zb, lb, z_edges, l_edges


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="all", choices=["all", "p95", "p90"])
    ap.add_argument("--nz", type=int, default=NZ); ap.add_argument("--nl", type=int, default=NL)
    a = ap.parse_args()
    out = M if a.subset == "all" else f"{M}/subset_{a.subset}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    np.random.seed(20261021)
    cat = Table.read(f"{D}/escale_des_sample.fits")
    stakes = pickle.load(open(f"{M}/stakes.pkl", "rb"))
    has = np.array([int(i) in stakes for i in cat["ID"]])
    if a.subset != "all":
        has &= np.asarray(cat[a.subset.upper()], bool)
    cat = cat[has]
    z = np.asarray(cat["Z"], float); llx = np.log10(np.asarray(cat["LX_EZ"], float))
    zb, lb, z_edges, l_edges = assign_bins(z, llx, a.nz, a.nl)
    stacks = {}
    for i in range(a.nz):
        for j in range(a.nl):
            m = (zb == i) & (lb == j)
            ids = np.asarray(cat["ID"])[m]
            bk = [stakes[int(c)].astype(float) for c in ids]
            sig, B, Bcov, cov, xs, xcov = xlensing.data.stacked_signal(bk, BL16, NBOOT)
            name = f"z{i}_lx{j}"
            stacks[name] = dict(z_bin=i, lx_bin=j, n_stacked=int(m.sum()), ids=ids, z_mean=float(z[m].mean()),
                                logLxE_mean=float(llx[m].mean()), logLx_mean=float(np.log10(cat["LX"][m]).mean()),
                                sigmas=np.asarray(sig), sigmas_cov=np.asarray(cov), boosts=np.asarray(B),
                                boosts_cov=np.asarray(Bcov), xigmas=np.asarray(xs), xigmas_cov=np.asarray(xcov))
            print(f"  {name}: N={m.sum():3d}  <z>={z[m].mean():.3f}  <log Lx/E>={llx[m].mean():.2f}  "
                  f"B(inner)={B[0]:.3f} B(outer)={B[-1]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    # boost fits
    fits = {}
    for name, s in stacks.items():
        try:
            p, pc = curve_fit(lambda R, B0, Rs, a: boost_powerlaw_at_R(R, B0, Rs, a), R16, s["boosts"], p0=(0.3, 1.0, -1.0),
                              bounds=((0.0, 0.01, -10.0), (20.0, 50.0, 10.0)), sigma=s["boosts_cov"], absolute_sigma=True, maxfev=20000)
            fits[name] = dict(B0=p[0], Rs=p[1], alpha=p[2], failed=False)
        except Exception as exc:
            fits[name] = dict(failed=True, error=repr(exc))
        print(f"  boost fit {name}: {fits[name]}", flush=True)
    pickle.dump(dict(stacks=stacks, z_edges=z_edges, lx_edges=l_edges, radii=R16, bin_limits=BL16, boost_fits=fits,
                     nboot=NBOOT), open(f"{out}/stacks.pkl", "wb"))
    ids_all = np.asarray(cat["ID"]); K = np.zeros((len(ids_all), 16))
    for n, c in enumerate(ids_all):
        sg, et, ex, w, R, mb = stakes[int(c)].astype(float)
        K[n] = np.dot((1 + mb) * w / sg ** 2, (R[:, None] > BL16[:, 0]) & (R[:, None] < BL16[:, 1]))
    np.savez(f"{out}/weights.npz", ids=ids_all, K=K, radii=R16, bin_limits=BL16)
    R6 = np.logspace(-0.6, 0.6, 6); boost6 = np.ones((len(ids_all), 6)); stack_of = np.empty(len(ids_all), object)
    for n in range(len(ids_all)):
        nm = f"z{zb[n]}_lx{lb[n]}"; stack_of[n] = nm; f = fits[nm]
        if not f["failed"]:
            boost6[n] = boost_powerlaw_at_R(R6, f["B0"], f["Rs"], f["alpha"])
    np.savez(f"{out}/boosts.npz", ids=ids_all, boost6=boost6, stack=stack_of.astype(str), radii=R6)
    print(f"done ({time.time()-t0:.0f}s): {len(stacks)} stacks, {len(ids_all)} clusters", flush=True)


if __name__ == "__main__":
    main()
