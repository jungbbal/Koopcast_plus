"""Why averaging fails: is K_j garbage on states outside cluster j?

Experiment 7c refuted the stalling story -- the uniform average overshoots by 13x
at the FIRST step, so this is not a spectral/horizon effect at all. A one-step
blow-up points somewhere else: each K_j is an affine map fit only on its own
region of state space, and

    (sum_j pi_j K_j) psi  =  sum_j pi_j (K_j psi)

so a soft mixture necessarily evaluates K_j on states that cluster j never saw.
If K_j extrapolates badly off its own domain, the mixture inherits that no matter
how well each K_j behaves at home.

Measures the one-step velocity prediction error of K_j on states assigned to j
(in-domain) versus on states assigned elsewhere (off-domain), on held-out zara2.
Ratio >> 1 confirms the mechanism.

Run: python exp/07d_offdomain.py
"""
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np
from sklearn.cluster import KMeans

import zlab
from bsk import fit_lstsq
from bsk.observables import Observable, pairs
from bsk.stability import fit_scaled_ridge

OUT = _HERE.parents[1] / "out"
OBS = Observable(5, True, "delay-5 +nonlin")
RIDGE, MIN_COUNT = 1e-2, 50


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def main():
    X, Y = pairs(zlab.load("train"), OBS)
    Xt, Yt = pairs([zlab.load_zara2_scene()], OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)
    res = {"cells": []}
    print("one-step velocity RMSE of K_j, in-domain vs off-domain (zara2)\n")
    print(f"  {'M':>3s}{'in-domain':>12s}{'off-domain':>12s}{'ratio':>9s}"
          f"{'global':>10s}")
    for M in [4, 8, 16]:
        km = KMeans(n_clusters=M, n_init=10, random_state=0).fit(vel_block(X))
        C, lab = km.cluster_centers_, km.labels_
        Ks = np.stack([fit_scaled_ridge(X[lab == j], Y[lab == j], RIDGE, len(X))
                       if (lab == j).sum() >= MIN_COUNT else K_glob
                       for j in range(M)])
        assign = np.argmin(((vel_block(Xt)[:, None, :] - C[None]) ** 2).sum(-1), 1)
        vt = Yt[:, 2:4]                                   # true v_{t+1}
        errs_in, errs_off = [], []
        for j in range(M):
            pred = Xt @ Ks[j].T
            e = ((pred[:, 2:4] - vt) ** 2).sum(axis=1)     # squared vel error
            m = assign == j
            if m.sum() == 0:
                continue
            errs_in.append(e[m])
            errs_off.append(e[~m])
        ein = float(np.sqrt(np.mean(np.concatenate(errs_in))))
        eoff = float(np.sqrt(np.mean(np.concatenate(errs_off))))
        pg = Xt @ K_glob.T
        eg = float(np.sqrt(np.mean(((pg[:, 2:4] - vt) ** 2).sum(axis=1))))
        res["cells"].append({"M": M, "rmse_in": ein, "rmse_off": eoff,
                             "ratio": eoff / ein, "rmse_global": eg})
        print(f"  {M:3d}{ein:12.4f}{eoff:12.4f}{eoff/ein:9.1f}x{eg:10.4f}")
    (OUT / "07d_offdomain.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / '07d_offdomain.json'}")


if __name__ == "__main__":
    main()
