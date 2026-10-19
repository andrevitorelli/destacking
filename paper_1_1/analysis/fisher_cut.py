"""Fisher sigma growth when bins below 0.5 Mpc are removed (other parameters fixed), linear Delta m regressor (paper 1.1)."""
import sys, glob, numpy as np
from destacking.model import _nfw_profiles
LN10 = np.log(10)
def ratios(z, M, c, x, var, R):
    prof = lambda cc: _nfw_profiles(z, M, cc, R, bin_nodes=0); h = 1e-3
    a = (prof(c * np.exp(h)) - prof(c * np.exp(-h))) / (2 * h)            # d model / d ln c
    v = np.isfinite(var) & (var > 0); iv = np.where(v, 1 / np.where(v, var, 1), 0.0)
    out = {}
    for p, f in (("gamma", LN10 * x[:, None]), ("c0", np.ones_like(x)[:, None])):
        d2 = (a * f) ** 2 * iv; out[p] = np.sqrt(d2.sum() / d2[:, R >= 0.5].sum())
    return out
if __name__ == "__main__":
    for d, grid in zip(sys.argv[1::2], sys.argv[2::2]):
        g = [float(t) for t in grid.split(",")]; R = np.logspace(g[0], g[1], int(g[2]))
        s = np.load(glob.glob(f"{d}/simulated_data.npz")[0]); t = np.load(glob.glob(f"{d}/truth_catalog.npz")[0])
        var = np.einsum("nii->ni", s["cov_matrices"]); var = np.where(np.isfinite(s["data_vectors"]), var, np.nan)
        print(d, len(R), "bins removed", int((R < 0.5).sum()), {k: round(float(v), 2) for k, v in ratios(t["z"], t["M_halo"], t["concentration"], np.asarray(t["delta_m"], float), var, R).items()})
