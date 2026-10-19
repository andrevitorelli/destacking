"""Reduced-shear factor for the Delta Sigma model (2026-10-07).

The estimator measures g_t = gamma_t / (1 - kappa); to first order in kappa the expected estimate of Delta Sigma is
Delta Sigma * (1 + Sigma * <Sigma_crit^-3> / <Sigma_crit^-2>), the averages over the selected sources behind a lens at z
(weights Sigma_crit^-2 as in the estimator; source redshifts from the DES Y3 DNF sample; background cut z_s > f z_l + 0.2).
Sigma of the miscentred component is approximated by the centred Sigma at sqrt(R^2 + 2 sigma_off^2) (Rayleigh mean square offset).
"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import os, numpy as np
from astropy.cosmology import Planck18 as C
import astropy.units as u, astropy.constants as k
from xlensing import cosmo as xc
from xlensing.model import NFW_delta_c, r_vir_m, SigX
_FAC = (k.c ** 2 / (4 * np.pi * k.G)).to(u.Msun / u.Mpc).value
_CACHE = {}


def ratio_grid(f=float(os.environ.get("RS_F", "1.1")), n=300000):
    if f in _CACHE: return _CACHE[f]
    fn = f"{DATA}/output/referee_2026-10-07/rs_ratio_f{f}.npz"
    if os.path.exists(fn):
        d = np.load(fn); _CACHE[f] = (d["zg"], d["r"]); return _CACHE[f]
    src = np.load(f"{DATA}/real_data/metacal_filtered.npz")["z"]
    zs_all = src[np.random.default_rng(0).choice(len(src), n, replace=False)]
    zg = np.linspace(0.05, 0.75, 36); r = []
    for zl in zg:
        zs = zs_all[zs_all > f * zl + 0.2]
        Dl = C.angular_diameter_distance(zl).value; Ds = C.angular_diameter_distance(zs).value; Dls = C.angular_diameter_distance(zl, zs).value
        inv = Dl * Dls / Ds / _FAC
        r.append(np.mean(inv ** 3) / np.mean(inv ** 2))
    _CACHE[f] = (zg, np.array(r)); np.savez(fn, zg=zg, r=np.array(r)); return _CACHE[f]


def sigma_nfw(z, M, c, R):
    rs = r_vir_m(z, M) / c; amp = 2 * rs * NFW_delta_c(c) * xc.rhoM(z)
    x = np.atleast_2d(R) / rs[:, None]
    return amp[:, None] * SigX(x)


def kappa_factor(z, M, c, R, f_mis, sigma_off, f_sel=None):
    zg, rg = ratio_grid() if f_sel is None else ratio_grid(f_sel)
    z = np.atleast_1d(z); M = np.atleast_1d(M); c = np.atleast_1d(c); R = np.asarray(R, float)
    so = np.broadcast_to(np.asarray(sigma_off, float), z.shape); fm = np.broadcast_to(np.asarray(f_mis, float), z.shape)
    cen = sigma_nfw(z, M, c, R)
    mis = np.array([sigma_nfw(z[i:i + 1], M[i:i + 1], c[i:i + 1], np.sqrt(R ** 2 + 2 * so[i] ** 2))[0] for i in range(len(z))]) if np.ndim(sigma_off) else \
        sigma_nfw(z, M, c, np.sqrt(R ** 2 + 2 * float(sigma_off) ** 2))
    sig = (1 - fm[:, None]) * cen + fm[:, None] * mis
    return 1 + sig * np.interp(z, zg, rg)[:, None]


if __name__ == "__main__":
    z = np.array([0.45]); M = np.array([10 ** 14.3]); c = np.array([5.0]); R = np.logspace(-0.6, 0.6, 6)
    print(kappa_factor(z, M, c, R, 0.0, 0.4), kappa_factor(z, M, c, R, 0.35, 0.4))
