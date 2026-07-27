"""Why does operator averaging collapse? Test the stalling hypothesis directly.

Experiment 7: the uniform average mean_j K_j scores ADE 1.07 at M=8 while every
individual K_j scores ~0.31 -- and rho(K_unif)=0.984 < 1 while the K_j sit at
~1.09. The proposed mechanism: the K_j do not commute and their velocity-block
eigenvectors point differently, so averaging destroys the marginally-stable mode
that carries constant speed. The prediction then decays to a standstill.

That is a *claim about the predicted trajectory*, so measure it there rather than
inferring it from rho: track predicted speed ||p_{h} - p_{h-1}|| over the horizon,
normalised by the last observed speed. Stalling means the ratio decays toward 0.
Ground truth is the reference -- real pedestrians do slow down somewhat.

Run: python exp/07c_stall.py
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
M = 8
OBS = Observable(5, True, "delay-5 +nonlin")
RIDGE = 1e-2
MIN_COUNT = 50
N_OBS, N_PRED = 8, 12


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def speed_profile(scenes, K_of):
    """Mean predicted step-length at each horizon h, normalised by the last
    observed step length."""
    prof, base = [], []
    L = N_OBS + N_PRED
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1):
                w = traj[s:s + L]
                psi, _ = OBS.lift(w[:N_OBS])
                if len(psi) == 0:
                    continue
                x = psi[-1]
                v0 = np.linalg.norm(w[N_OBS - 1] - w[N_OBS - 2])
                if v0 < 1e-6:
                    continue
                prev, steps = x[:2].copy(), []
                for _ in range(N_PRED):
                    x = K_of(x) @ x
                    if not np.isfinite(x).all():
                        break
                    steps.append(np.linalg.norm(x[:2] - prev))
                    prev = x[:2].copy()
                if len(steps) == N_PRED:
                    prof.append(np.asarray(steps) / v0)
                    base.append(v0)
    return np.mean(prof, axis=0)


def truth_profile(scenes):
    prof = []
    L = N_OBS + N_PRED
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1):
                w = traj[s:s + L]
                v0 = np.linalg.norm(w[N_OBS - 1] - w[N_OBS - 2])
                if v0 < 1e-6:
                    continue
                st = np.linalg.norm(np.diff(w[N_OBS - 1:], axis=0), axis=1)
                prof.append(st / v0)
    return np.mean(prof, axis=0)


def main():
    test = [zlab.load_zara2_scene()]
    X, Y = pairs(zlab.load("train"), OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)
    km = KMeans(n_clusters=M, n_init=10, random_state=0).fit(vel_block(X))
    C, lab = km.cluster_centers_, km.labels_
    Ks = np.stack([fit_scaled_ridge(X[lab == j], Y[lab == j], RIDGE, len(X))
                   if (lab == j).sum() >= MIN_COUNT else K_glob for j in range(M)])
    K_unif = Ks.mean(axis=0)
    scale = float(np.mean(((vel_block(X) - C[lab]) ** 2).sum(axis=1)))

    def soft(tau):
        def f(x):
            d = ((vel_block(x) - C) ** 2).sum(axis=-1)
            z = -d / (tau * scale)
            z -= z.max()
            w = np.exp(z)
            return np.tensordot(w / w.sum(), Ks, axes=1)
        return f

    variants = {
        "ground truth": None,
        "hard": lambda x: Ks[int(np.argmin(((vel_block(x) - C) ** 2).sum(axis=-1)))],
        "soft tau=1": soft(1.0),
        "soft tau=4": soft(4.0),
        "uniform avg": lambda x: K_unif,
        "global K": lambda x: K_glob,
    }
    res = {}
    print(f"predicted step length / last observed step length, M={M}, zara2\n")
    hs = [1, 3, 6, 9, 12]
    print(f"  {'variant':14s}" + "".join(f"{f'h={h}':>9s}" for h in hs))
    for name, fn in variants.items():
        p = truth_profile(test) if fn is None else speed_profile(test, fn)
        res[name] = p.tolist()
        print(f"  {name:14s}" + "".join(f"{p[h-1]:9.3f}" for h in hs))
    (OUT / "07c_stall.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / '07c_stall.json'}")


if __name__ == "__main__":
    main()
