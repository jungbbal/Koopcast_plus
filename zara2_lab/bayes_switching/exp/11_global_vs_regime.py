"""Q: at each model class, does splitting into regimes add anything over M=1?

Experiment 10 gave a clean monotone ladder up to +5.2%, but every rung was run at
M=8, so it cannot separate two very different explanations:

  (a) regimes matter   -- a given model class improves when split by velocity regime
  (b) the model matters -- v <- A v + b is simply a better global model than
                           constant velocity, and the M=8 column was measuring that

This is the control that decides the project's headline. Each rung is rerun across
M in {1, 2, 4, 8, 16} on the same k-means partitions. Read the ROWS: if a row is
flat in M, that model class gets nothing from regimes and its gain over CV belongs
to the model class alone.

'vlin' at M=1 is a global linear velocity map -- a baseline stronger than constant
velocity that the earlier experiments never ran, so it may absorb much of what was
being credited to switching.

Run: python exp/11_global_vs_regime.py
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
from bsk.observables import Observable, const_vel, pairs
from bsk.stability import fit_scaled_ridge


OUT = _HERE.parents[1] / "out"
OBS = Observable(5, True, "delay-5 +nonlin")
RIDGE, MIN_COUNT = 1e-2, 50
MS = [1, 2, 4, 8, 16]
SEEDS = [0, 1, 2]
N_OBS, N_PRED = 8, 12


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def fit_accel(Xc, Yc):
    return (Yc[:, 2:4] - Xc[:, 2:4]).mean(axis=0)


def step_accel(a, v, vprev):
    return v + a


def fit_vlin(Xc, Yc):
    F = np.concatenate([Xc[:, 2:4], np.ones((len(Xc), 1))], 1)
    return np.linalg.solve(F.T @ F + RIDGE * np.eye(3), F.T @ Yc[:, 2:4])


def step_vlin(W, v, vprev):
    return np.concatenate([v, [1.0]]) @ W


def fit_vhist(Xc, Yc):
    F = np.concatenate([Xc[:, 2:4], Xc[:, 4:6], np.ones((len(Xc), 1))], 1)
    return np.linalg.solve(F.T @ F + RIDGE * np.eye(5), F.T @ Yc[:, 2:4])


def step_vhist(W, v, vprev):
    return np.concatenate([v, vprev, [1.0]]) @ W


LADDER = [("accel", fit_accel, step_accel, 2),
          ("vlin", fit_vlin, step_vlin, 6),
          ("vlin+hist", fit_vhist, step_vhist, 10)]


def rollout_small(scenes, params, C, step):
    ades, fdes = [], []
    L = N_OBS + N_PRED
    one = C is None
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1):
                w = traj[s:s + L]
                psi, _ = OBS.lift(w[:N_OBS])
                if len(psi) == 0:
                    continue
                x = psi[-1]
                p, v, vprev = x[:2].copy(), x[2:4].copy(), x[4:6].copy()
                lift, pred = x.copy(), []
                for _ in range(N_PRED):
                    if one:
                        j = 0
                    else:
                        lift[2:4], lift[4:6] = v, vprev
                        j = int(np.argmin(((vel_block(lift) - C) ** 2).sum(-1)))
                    vn = step(params[j], v, vprev)
                    p = p + vn
                    vprev, v = v, vn
                    pred.append(p.copy())
                pred = np.asarray(pred)
                if not np.isfinite(pred).all() or np.abs(pred).max() > 1e3:
                    pred = np.clip(np.nan_to_num(pred), -1e3, 1e3)
                d = np.linalg.norm(pred - w[N_OBS:], axis=-1)
                ades.append(d.mean())
                fdes.append(d[-1])
    return float(np.mean(ades)), float(np.mean(fdes))


def rollout_koop(scenes, Ks, C):
    ades, fdes = [], []
    L = N_OBS + N_PRED
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1):
                w = traj[s:s + L]
                psi, _ = OBS.lift(w[:N_OBS])
                if len(psi) == 0:
                    continue
                x, pred = psi[-1].copy(), []
                for _ in range(N_PRED):
                    j = 0 if C is None else int(
                        np.argmin(((vel_block(x) - C) ** 2).sum(-1)))
                    x = Ks[j] @ x
                    pred.append(x[:2].copy())
                pred = np.asarray(pred)
                if not np.isfinite(pred).all() or np.abs(pred).max() > 1e3:
                    pred = np.clip(np.nan_to_num(pred), -1e3, 1e3)
                d = np.linalg.norm(pred - w[N_OBS:], axis=-1)
                ades.append(d.mean())
                fdes.append(d[-1])
    return float(np.mean(ades)), float(np.mean(fdes))


def main():
    OUT.mkdir(exist_ok=True)
    test = [zlab.load_zara2_scene()]
    X, Y = pairs(zlab.load("train"), OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)
    ade_cv, _ = const_vel(test)
    res = {"const_vel_ade": ade_cv, "cells": []}

    parts = {}
    for M in MS:
        if M == 1:
            parts[M] = [(np.zeros(len(X), int), None)]
            continue
        parts[M] = []
        for seed in SEEDS:
            km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
            parts[M].append((km.labels_, km.cluster_centers_))

    print(f"=== ADE, LOO (train -> zara2). CV = {ade_cv:.4f} ===\n")
    print(f"  {'model':14s}" + "".join(f"{f'M={m}':>10s}" for m in MS)
          + f"{'M=8 vs M=1':>13s}")

    def record(name, npar, ades):
        row = []
        for M, a in zip(MS, ades):
            res["cells"].append({"model": name, "M": M, "ade": a,
                                 "vs_cv_pct": 100 * (ade_cv - a) / ade_cv})
            row.append(a)
        gain = 100 * (row[0] - row[MS.index(8)]) / row[0]
        print(f"  {name:14s}" + "".join(f"{a:10.4f}" for a in row)
              + f"{gain:+12.2f}%")
        return row

    for name, fit, step, npar in LADDER:
        ades = []
        for M in MS:
            vals = []
            for lab, C in parts[M]:
                params = {j: (fit(X[lab == j], Y[lab == j])
                              if (lab == j).sum() >= MIN_COUNT else fit(X, Y))
                          for j in range(M)}
                vals.append(rollout_small(test, params, C, step)[0])
            ades.append(float(np.mean(vals)))
        record(name, npar, ades)

    ades = []
    for M in MS:
        vals = []
        for lab, C in parts[M]:
            Ks = np.stack([fit_scaled_ridge(X[lab == j], Y[lab == j], RIDGE, len(X))
                           if (lab == j).sum() >= MIN_COUNT else K_glob
                           for j in range(M)])
            vals.append(rollout_koop(test, Ks, C)[0])
        ades.append(float(np.mean(vals)))
    record("koopman", OBS.dim ** 2, ades)

    (OUT / "11_global_vs_regime.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / '11_global_vs_regime.json'}")


if __name__ == "__main__":
    main()
