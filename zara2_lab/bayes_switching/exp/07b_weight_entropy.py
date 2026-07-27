"""Check: at each tau, how soft is the mixture actually?

Experiment 7 found soft ~= hard for tau <= 0.5 and worse beyond. That reading is
only valid if the mixture was genuinely soft in that range. If the softmax is
effectively one-hot at tau=0.5, then "soft ties hard" is a tautology -- soft *was*
hard -- and the experiment tested nothing.

Reports mean max-weight and normalised entropy H/log(M) over the test states.
The uniform-average control sits at H/log(M) = 1 by construction.

Run: python exp/07b_weight_entropy.py
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
from bsk.observables import Observable, pairs

OUT = _HERE.parents[1] / "out"
MS = [4, 8, 16]
OBS = Observable(5, True, "delay-5 +nonlin")
TAUS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def main():
    X, _ = pairs(zlab.load("train"), OBS)
    Xt, _ = pairs([zlab.load_zara2_scene()], OBS)
    res = {"cells": []}
    print("normalised entropy H/log(M) and mean max-weight, on zara2 states\n")
    for M in MS:
        km = KMeans(n_clusters=M, n_init=10, random_state=0).fit(vel_block(X))
        C = km.cluster_centers_
        scale = float(np.mean(((vel_block(X) - C[km.labels_]) ** 2).sum(axis=1)))
        d = ((vel_block(Xt)[:, None, :] - C[None]) ** 2).sum(axis=-1)   # (n, M)
        print(f"  M={M:2d}  {'tau':>6s}{'H/logM':>9s}{'max pi':>9s}"
              f"{'eff. #modes':>13s}")
        for tau in TAUS:
            z = -d / (tau * scale)
            z -= z.max(axis=1, keepdims=True)
            w = np.exp(z)
            w /= w.sum(axis=1, keepdims=True)
            H = -(w * np.log(np.maximum(w, 1e-300))).sum(axis=1)
            hn = float(H.mean() / np.log(M))
            mx = float(w.max(axis=1).mean())
            eff = float(np.exp(H).mean())
            res["cells"].append({"M": M, "tau": tau, "H_norm": hn,
                                 "mean_max_pi": mx, "eff_modes": eff})
            print(f"        {tau:6.2f}{hn:9.3f}{mx:9.3f}{eff:13.2f}")
        print()
    (OUT / "07b_weight_entropy.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / '07b_weight_entropy.json'}")


if __name__ == "__main__":
    main()
