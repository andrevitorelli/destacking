import os as _os, sys as _sys  # paper_1_1: locate paths.py and sibling helpers
_sys.path[:0] = [_os.path.dirname(_os.path.abspath(__file__)), _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                 _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))]
from paths import DATA, FIGS, REPO, AUX  # noqa: E402,F401
import pickle, numpy as np, importlib.util, warnings, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "xtick.direction": "in", "ytick.direction": "in"})
spec = importlib.util.spec_from_file_location("sff", f"{AUX}/stacked_forward_fit.py")
sff = importlib.util.module_from_spec(spec); spec.loader.exec_module(sff)
from astropy.table import Table
D = f"{DATA}/real_data"
st = pickle.load(open(f"{D}/stacked_full_param_fit_f1.2/stacks3.pkl", "rb"))
fm = sff.StackedForwardModel(Table.read(f"{D}/wazp_real_data.fits"), st, np.load(f"{D}/stack_weights_f1.2.npz"), min_n=50, two_halo=True)
z = np.load(f"{DATA}/output/v1.1/stacked/real_minN50.npz", allow_pickle=True)
o = list(z["param_order"]); w = z["weights"]; X = z["samples"]
rng = np.random.default_rng(0); idx = rng.choice(len(w), 200, p=w / w.sum())
models = np.array([fm.model(dict(zip(o, X[i]))) for i in idx])          # (200, n_stacks, n_bins)
lo, me, hi = np.percentile(models, [16, 50, 84], axis=0)
R = fm.radii; zlab = ["$0.1<z<0.3$", "$0.3<z<0.5$", "$0.5<z<0.7$"]; nglab = ["$25$--$35$", "$35$--$55$", "$55$--$85$", "$>85$"]
fig, axs = plt.subplots(4, 3, figsize=(10.5, 11), sharex=True, sharey=True)
cols = ["C0", "C2", "C3"]; dml = ["low", "mid", "high"]
for k, name in enumerate(fm.names):
    s = st[name]; i, j, d = s["z_bin"], s["ng_bin"], s["dm_bin"]; ax = axs[3 - j, i]       # richness increases upward, like the stack cells of the population figure
    ok = fm.masks[k]; e = np.sqrt(np.diag(s["sigmas_cov"]))[ok]; f = 1.0 + 0.04 * (d - 1)
    ax.errorbar(R[ok] * f, fm.data[k], yerr=e, fmt="o", ms=3, color=cols[d], capsize=0, lw=0.8,
                label=f"{dml[d]} $\\Delta m$ ($N={s['n_stacked']}$)")
    ax.plot(R, me[k], color=cols[d], lw=1.2); ax.fill_between(R, lo[k], hi[k], color=cols[d], alpha=0.2, lw=0)
for i in range(3):
    for j in range(4):
        ax = axs[3 - j, i]; ax.set_xscale("log"); ax.set_yscale("log"); ax.tick_params(top=True, right=True, which="both")
        ax.set_xticks([0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.5", "1", "2"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if ax.has_data(): ax.legend(fontsize=7, frameon=False, loc="lower left")
        else: ax.text(0.5, 0.5, "no stack with $N\\geq50$", transform=ax.transAxes, ha="center", fontsize=8, color="0.4")
    axs[0, i].set_title(zlab[i]); axs[-1, i].set_xlabel(r"$R$ [Mpc]")
for j in range(4): axs[3 - j, 0].set_ylabel(r"$\Delta\Sigma$ [$M_\odot\,{\rm pc}^{-2}$]" + f"\n$\\lambda$ {nglab[j]}")
axs[0, 0].set_ylim(0.5, 400)   # shared y axis
fig.tight_layout(); fig.savefig(f"{FIGS}/stacked_forward_fit_data.pdf", bbox_inches="tight")
fig.savefig(f"{DATA}/output/v1.1/results/stackfit.png", dpi=60, bbox_inches="tight")
chi = sum(float(r @ ci @ r) for r, ci in [(fm.data[k] - me[k][fm.masks[k]], fm.cinv[k]) for k in range(len(fm.names))])
print("chi2 at median model:", round(chi, 1), "bins:", sum(len(d) for d in fm.data), "stacks:", len(fm.names))
