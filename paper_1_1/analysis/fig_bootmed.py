"""Paper figures with the joint fit summarised by the distribution of its per-bootstrap posterior medians (2026-10-07).

The joint-fit distributions are smoothed bootstraps that preserve the mean and covariance of the medians:
x* = mean + (x_i - mean + h * eps) / sqrt(1 + h^2), eps ~ N(0, Cov(medians)), h = Silverman bandwidth factor.
Regenerates joint_fit_posterior.pdf, stacked_forward_corner.pdf (no header), real_data_joint_vs_stacked.pdf, diemer_gamma_comparison.pdf.
"""
import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import glob, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.figure import Figure
Figure.suptitle = lambda self, *a, **k: None          # no figure headers in paper figures
DIAG = {}                                               # name -> (lo, med, hi) to impose on the diagonal panels of the next corner
_save = Figure.savefig
def _savefig(self, *a, **k):
    if DIAG.get("vals"):
        vals, labs = DIAG["vals"], DIAG["labs"]; n = len(vals)
        for i in range(n):
            ax = self.axes[i * n + i]
            for ln in list(ax.lines):
                xd = np.asarray(ln.get_xdata(), float)
                if xd.size == 2 and xd[0] == xd[1] and ln.get_linestyle() == "--" and ln.get_color() in ("k", "black", "#000000"): ln.remove()
            lo, me, hi = vals[i]
            for v in (lo, me, hi): ax.axvline(v, color="k", ls="--", lw=1)
            ax.set_title(f"{labs[i]} = ${me:.3f}^{{+{hi-me:.3f}}}_{{-{me-lo:.3f}}}$", fontsize=10)
    return _save(self, *a, **k)
Figure.savefig = _savefig
from scipy import stats
from scipy.stats import gaussian_kde
_dg = np.load(f'{DATA}/output/v1.1/report_des/diemer19_alpha_v11.npy')  # grid, counts, Fisher slopes (fig_diemer19_v11.py)
DIEMER_BAND = (min(_dg[0], _dg[2]), max(_dg[0], _dg[2]))   # span of the unweighted-grid and Fisher-weighted Diemer slopes
from bootmed import boot_medians, wq
from destacking.plotting import plot_combined_corner
O = f"{DATA}/output"; FIG = f"{FIGS}"
ORDER = ["c0", "alpha_cm", "beta_cm", "gamma_cm", "log10M0", "alpha_mr", "beta_mr", "f_mis", "sigma_off"]
LAB = {"c0": r"$c_0$", "alpha_cm": r"$\alpha_{\rm cm}$", "beta_cm": r"$\beta_{\rm cm}$", "gamma_cm": r"$\gamma_{\rm cm}$",
       "log10M0": r"$\log_{10}M_0$", "alpha_mr": r"$\alpha_{M\lambda}$", "beta_mr": r"$\beta_{M\lambda}$",
       "f_mis": r"$f_{\rm mis}$", "sigma_off": r"$\sigma_{\rm off}$"}
PRI = {"c0": stats.uniform(1, 14), "alpha_cm": stats.norm(-0.1, 0.5), "beta_cm": stats.norm(-0.4, 2),
       "gamma_cm": stats.norm(0, 1), "log10M0": stats.norm(14.489, 0.5), "alpha_mr": stats.norm(1.356, 0.5), "beta_mr": stats.norm(-0.3, 2),
       "f_mis": stats.uniform(0, 1), "sigma_off": stats.uniform(0, 1)}
PRI9C = dict(PRI, c0=stats.truncnorm((0 - 4.5) / 5, np.inf, loc=4.5, scale=5))


def smoothed(meds, n=40000, seed=3):
    """Variance-preserving smoothed bootstrap of the medians (rows = bootstraps)."""
    rng = np.random.default_rng(seed); m, d = meds.shape
    h = (4 / (d + 2)) ** (1 / (d + 4)) * m ** (-1 / (d + 4)); mu = meds.mean(0); C = np.atleast_2d(np.cov(meds.T))
    x = meds[rng.integers(0, m, n)] + h * rng.multivariate_normal(np.zeros(d), C, n)
    return mu + (x - mu) / np.sqrt(1 + h ** 2)


def medians(run, key="all_samples"):
    fs = sorted(glob.glob(run)) if "*" in run or "[" in run else sorted(glob.glob(f"{O}/{run}/simulation_*/bootstrap_*/gamma_posterior.npz"))
    if "*" in run or "[" in run: key = "samples"
    names, M = boot_medians(fs, key)
    S = {p: M[:, j] for j, p in enumerate(names)}
    if "log_M_norm" in S: S["log10M0"] = S["log_M_norm"] + 14.489
    return S


def ranges(X, w):
    out = []
    for j in range(X.shape[1]):
        lo, hi = wq(X[:, j], w, [0.003, 0.997]); p = 0.08 * (hi - lo); out.append((lo - p, hi + p))
    return out


def stacked():
    z = np.load(f"{O}/v1.1/stacked/real_minN50.npz", allow_pickle=True); o = [str(p) for p in z["param_order"]]
    S = {p: z["samples"][:, o.index(p)] for p in o}; S["log10M0"] = S["log_M_norm"] + 14.489
    return S, z["weights"] / z["weights"].sum()


if __name__ == "__main__":
    J = medians(f"{DATA}/output/v1.1/posteriors/joint_sh_b[0-9][0-9][0-9].npz"); Mj = np.column_stack([J[p] for p in ORDER]); Xj = smoothed(Mj); wj = np.ones(len(Xj)) / len(Xj)
    print("check: percentiles of medians vs smoothed draws (gamma):", np.percentile(J["gamma_cm"], [16, 50, 84]).round(4),
          np.percentile(Xj[:, 3], [16, 50, 84]).round(4))
    DIAG.update(vals=[tuple(np.percentile(J[p], [16, 50, 84])) for p in ORDER], labs=[LAB[p] for p in ORDER])
    plot_combined_corner(Xj, wj, ORDER, f"{FIG}/joint_fit_posterior.pdf", labels=[LAB[p] for p in ORDER], truths=None, gamma_true=None,
                         priors=PRI9C, smooth1d=True, plot_range=ranges(Xj, wj))
    DIAG.clear()
    S, w = stacked(); Y = np.column_stack([S[p] for p in ORDER])
    plot_combined_corner(Y, w, ORDER, f"{FIG}/stacked_forward_corner.pdf", labels=[LAB[p] for p in ORDER], truths=None, gamma_true=None,
                         priors=PRI, smooth1d=True, plot_range=ranges(Y, w))
    # joint vs stacked marginals
    P = [("gamma_cm", r"$\gamma_{\rm cm}$"), ("alpha_cm", r"$\alpha_{\rm cm}$"), ("beta_cm", r"$\beta_{\rm cm}$"),
         ("alpha_mr", r"$\alpha_{M\lambda}$"), ("beta_mr", r"$\beta_{M\lambda}$"), ("log10M0", r"$\log_{10}(M_0/M_\odot)$")]
    plt.rcParams.update({"font.size": 10.5, "xtick.direction": "in", "ytick.direction": "in"})
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (p, lab) in zip(axs.flat, P):
        xj = Xj[:, ORDER.index(p)]; xs = S[p]
        lo = min(np.percentile(xj, 0.3), wq(xs, w, 0.003)); hi = max(np.percentile(xj, 99.7), wq(xs, w, 0.997)); pad = 0.1 * (hi - lo)
        g = np.linspace(lo - pad, hi + pad, 400)
        ax.plot(g, gaussian_kde(xj)(g), color="C0", lw=1.8, label="joint (15,783 clusters)")
        ax.plot(g, gaussian_kde(xs, weights=w, bw_method=0.25)(g), color="C3", lw=1.8, label="forward-modelled stacks (33 stacks)")
        if p == "alpha_cm": ax.axvspan(*DIEMER_BAND, color="0.75", alpha=0.7, lw=0, label="Diemer19")
        ax.set_xlabel(lab); ax.set_yticks([]); ax.set_ylim(bottom=0); ax.tick_params(top=True, which="both")
    h, l = axs.flat[0].get_legend_handles_labels(); hb, lb = axs.flat[1].get_legend_handles_labels()
    for hh, ll in zip(hb, lb):
        if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{FIG}/real_data_joint_vs_stacked.pdf", bbox_inches="tight"); plt.close(fig)
    # Diemer baseline vs power law (both joint, bootstrap medians)
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(6.5, 4.2)); g = np.linspace(-0.12, 0.25, 500)
    D = medians(f"{DATA}/output/v1.1/posteriors/joint_sh_diemer_b[0-9][0-9][0-9].npz"); kd = "gamma_cm" if "gamma_cm" in D else "gamma"
    for x, lab, col in ((J["gamma_cm"], "power-law $c(M,z)$", "C0"), (D[kd], "Diemer & Joyce (2019) baseline", "C3")):
        lo, me, hi = np.percentile(x, [16, 50, 84]); xs_ = smoothed(x[:, None])[:, 0]
        ax.plot(g, gaussian_kde(xs_)(g), color=col, lw=2, label=f"{lab}: ${me:.3f}^{{+{hi-me:.3f}}}_{{-{me-lo:.3f}}}$")
        print("diemer fig", lab, round(me, 4), round(me - lo, 4), round(hi - me, 4))
    ax.axvline(0, color="0.5", lw=0.8, ls=":"); ax.set_xlabel(r"$\gamma_{\rm cm}$"); ax.set_ylabel("density of bootstrap medians"); ax.set_ylim(bottom=0)
    ax.tick_params(top=True, right=True, which="both"); ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.savefig(f"{FIG}/diemer_gamma_comparison.pdf", bbox_inches="tight"); plt.close(fig)
    print("ok")
