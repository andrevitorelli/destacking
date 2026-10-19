"""Per-cluster stacking weights K_c(R) = sum_gal (1+m) w / Sigma_crit^2 in the
stacked analysis's 16 radial bins (xlensing.data.stacked_signal's sums_K),
so a forward model of a stack is sum_c K_c(R) DS_c(R) / sum_c K_c(R).
usage: python stack_weights.py bgs.pkl out.npz"""
import sys, pickle, numpy as np
radii = np.logspace(-0.8, 0.6, 16); N = len(radii)
e = np.logspace(np.log10(radii[0]) + (np.log10(radii[0]) - np.log10(radii[1])) / 2,
                np.log10(radii[-1]) - (np.log10(radii[0]) - np.log10(radii[1])) / 2, N + 1)
BL = np.array([[e[i], e[i + 1]] for i in range(N)])
bk = pickle.load(open(sys.argv[1], "rb"))
ids = np.array(sorted(bk), dtype=np.int64)
K = np.zeros((len(ids), N)); C = np.zeros((len(ids), N), dtype=np.int64)
for n, cid in enumerate(ids):
    sig, et, ex, w, R, M = bk[int(cid)][:6]
    ib = (R[:, None] > BL[:, 0]) & (R[:, None] < BL[:, 1])
    K[n] = np.dot((1 + M) * w / sig**2, ib); C[n] = ib.sum(0)
np.savez(sys.argv[2], ids=ids, K=K, counts=C, radii=radii, bins=BL)
print(len(ids), "clusters", K.shape)
