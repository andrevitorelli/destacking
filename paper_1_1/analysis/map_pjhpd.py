"""MAP + projected-joint-HPD (PJ-HPD) summaries from nautilus posteriors (Joachimi+2021, arXiv:2007.01844, Sect. 6.4, Eqs. 48-52).
For samples theta_n with weights w_n and log posterior density lp_n (= log L + log prior):
  MAP      = theta at max lp  (sample-based here; can be refined by an optimiser)
  PJ-HPD   = sort samples by decreasing lp; step down the list recording, per parameter, the interval [min,max] of all values seen so far;
             stop when the 1-D marginal posterior mass inside that interval reaches alpha; interpolate the boundaries."""
import numpy as np
from scipy import stats

PRIORS = {"c0": stats.uniform(1, 14), "alpha_cm": stats.norm(-0.1, 0.5), "beta_cm": stats.norm(-0.4, 2), "gamma_cm": stats.norm(0, 1),
          "log_M_norm": stats.norm(0, 0.5), "alpha_mr": stats.norm(1.356, 0.5), "beta_mr": stats.norm(-0.30, 2),
          "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0, 1)}

def log_prior(X, order, priors=PRIORS):
    return sum(priors[p].logpdf(X[:, i]) for i, p in enumerate(order))

def marginal_mass(x, w, a, b):
    return w[(x >= a) & (x <= b)].sum() / w.sum()

def pjhpd(X, w, lp, alpha=0.6827, nstep=400):
    """Return (lo, hi) per parameter column of X."""
    order = np.argsort(-lp); Xs = X[order]; n, d = X.shape
    cmin = np.minimum.accumulate(Xs, axis=0); cmax = np.maximum.accumulate(Xs, axis=0)
    res = []
    idx = np.unique(np.round(np.geomspace(1, n, nstep)).astype(int)) - 1
    for j in range(d):
        x = X[:, j]; o = np.argsort(x); xs, ws = x[o], w[o] / w.sum(); cw = np.cumsum(ws)
        def mass(a, b): return np.interp(b, xs, cw) - np.interp(a, xs, cw)
        prev = None
        for k in idx:
            m = mass(cmin[k, j], cmax[k, j])
            if m >= alpha:
                if prev is None: res.append((cmin[k, j], cmax[k, j])); break
                kp, mp = prev; f = (alpha - mp) / (m - mp)
                res.append((cmin[kp, j] + f * (cmin[k, j] - cmin[kp, j]), cmax[kp, j] + f * (cmax[k, j] - cmax[kp, j]))); break
            prev = (k, m)
        else:
            res.append((cmin[-1, j], cmax[-1, j]))
    return np.array(res)

def marginal_hpd_and_median(x, w, alpha=0.6827):
    o = np.argsort(x); xs, ws = x[o], w[o] / w.sum(); cw = np.cumsum(ws)
    med = np.interp(0.5, cw, xs); lo16, hi84 = np.interp([(1 - alpha) / 2, (1 + alpha) / 2], cw, xs)
    # marginal mode from weighted KDE; shortest interval containing alpha
    from scipy.stats import gaussian_kde
    sub = np.random.default_rng(0).choice(len(xs), size=min(20000, len(xs)), p=ws)
    g = np.linspace(xs[0], xs[-1], 600); k = gaussian_kde(x[o][sub])(g); mode = g[np.argmax(k)]
    j = np.searchsorted(cw, cw[:, None] * 0 + alpha + cw[None, :1] * 0) if False else None
    lo_i = np.arange(len(xs)); hi_i = np.searchsorted(cw, cw + alpha); ok = hi_i < len(xs)
    w_int = xs[hi_i[ok]] - xs[lo_i[ok]]; b = np.argmin(w_int)
    return dict(median=med, mode=mode, p16=lo16, p84=hi84, hpd=(xs[lo_i[ok][b]], xs[hi_i[ok][b]]))
