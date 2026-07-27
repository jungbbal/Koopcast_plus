"""Q: with stability equalised across M, does the M-curve say discrete or continuous?

Experiments 4-5 could not answer this: max rho(K_j) climbed 1.00 -> 3.03 as M
went 1 -> 32, so the sweep measured how many operators could be fit stably, not
how many regimes exist. Here every M is fit under the same stability discipline
and the sweep is re-read.

Methods compared at each M:
  plain       -- fixed ridge (the exp 4/5 setting; the thing to beat)
  scaled      -- lambda_j = lambda_0 * N / N_j, small clusters regularised harder
  proj/excess -- fixed ridge, then eigenvalues with |lambda| > 1 capped at 1
  proj/all    -- fixed ridge, then the literal lambda/max(1,|lambda|/(1-eps)) rule
  scaled+proj -- both

'proj/all' is expected to underperform and is included as a check on the
structural-eigenvalue argument in stability.py: psi carries a Jordan block at
lambda = 1 that IS the position integration, so shrinking every eigenvalue to
1-eps should bias predictions toward standing still. If proj/all does not lose,
that argument is wrong.

Reported per cell: ADE (mean over k-means seeds), max rho, and max ||K^12||,
because rho <= 1 alone does not bound a non-normal 12-step rollout.

Run: python exp/06_stable_M_sweep.py
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
from bsk.observables import Observable, const_vel, pairs, rollout
from bsk.stability import diagnose, fit_scaled_ridge, project_spectrum

OUT = _HERE.parents[1] / "out"
MS = [1, 2, 4, 8, 16, 32]
SEEDS = [0, 1, 2]
OBS = Observable(5, True, "delay-5 +nonlin")
RIDGE = 1e-2
EPS = 0.01
MIN_COUNT = 50

METHODS = ["plain", "scaled", "proj/excess", "proj/all", "scaled+proj"]


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def build_operators(X, Y, lab, M, method, K_glob):
    """One K per cluster under the given stability discipline."""
    Ks, infos = [], []
    for j in range(M):
        m = lab == j
        if m.sum() < MIN_COUNT:
            Ks.append(K_glob)
            infos.append({"cond_V": 1.0, "n_projected": 0})
            continue
        if method in ("scaled", "scaled+proj"):
            K = fit_scaled_ridge(X[m], Y[m], RIDGE, len(X))
        else:
            K = fit_lstsq(X[m], Y[m], RIDGE)
        info = {"cond_V": 1.0, "n_projected": 0}
        if method == "proj/excess" or method == "scaled+proj":
            K, info = project_spectrum(K, eps=0.0, mode="excess")
        elif method == "proj/all":
            K, info = project_spectrum(K, eps=EPS, mode="all")
        Ks.append(K)
        infos.append(info)
    return np.stack(Ks), infos


def main():
    OUT.mkdir(exist_ok=True)
    zara2 = zlab.load_zara2_scene()
    fit_scenes, test_scenes = zlab.load("train"), [zara2]
    ade_cv, _ = const_vel(test_scenes)
    X, Y = pairs(fit_scenes, OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)

    print(f"=== LOO (train -> zara2), observable={OBS.name}, "
          f"ridge={RIDGE:.0e}, eps={EPS}, seeds={SEEDS} ===")
    print(f"const-velocity ADE={ade_cv:.4f}\n")

    # cache the partitions so every method sees identical clusters
    parts = {}
    for M in MS:
        for seed in SEEDS:
            if M == 1:
                parts[(M, seed)] = (np.zeros(len(X), int), None)
            else:
                km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
                parts[(M, seed)] = (km.labels_, km.cluster_centers_)
            if M == 1:
                break

    results = {"const_vel_ade": ade_cv, "cells": []}
    print(f"  {'method':13s} " + "".join(f"{f'M={m}':>10s}" for m in MS))

    table = {}
    for method in METHODS:
        cells = []
        for M in MS:
            ades, rhos, amps, conds = [], [], [], []
            seeds = SEEDS if M > 1 else [0]
            for seed in seeds:
                lab, C = parts[(M, seed)]
                Ks, infos = build_operators(X, Y, lab, M, method, K_glob)
                if C is None:
                    assign = lambda x: 0                            # noqa: E731
                else:
                    def assign(x, C=C):
                        return int(np.argmin(((vel_block(x) - C) ** 2).sum(axis=1)))
                d = [diagnose(k) for k in Ks]
                rhos.append(max(x["rho"] for x in d))
                amps.append(max(x["amp12"] for x in d))
                conds.append(max(i["cond_V"] for i in infos))
                ades.append(rollout(test_scenes, OBS, lambda x: Ks[assign(x)])[0])
            cell = {"method": method, "M": M, "ade": float(np.mean(ades)),
                    "ade_std": float(np.std(ades)), "max_rho": max(rhos),
                    "max_amp12": max(amps), "max_cond_V": max(conds)}
            results["cells"].append(cell)
            cells.append(cell)
        table[method] = cells
        print(f"  {method:13s} " + "".join(f"{c['ade']:10.4f}" for c in cells))

    print(f"\n  {'method':13s} " + "".join(f"{f'rho M={m}':>10s}" for m in MS))
    for method in METHODS:
        print(f"  {method:13s} " + "".join(f"{c['max_rho']:10.3f}" for c in table[method]))

    print(f"\n  {'method':13s} " + "".join(f"{f'|K^12| {m}':>11s}" for m in MS))
    for method in METHODS:
        print(f"  {method:13s} " + "".join(
            f"{min(c['max_amp12'], 9999):11.1f}" for c in table[method]))

    best = min(results["cells"], key=lambda c: c["ade"])
    print(f"\nbest: {best['method']} M={best['M']} ADE={best['ade']:.4f} "
          f"({100*(ade_cv-best['ade'])/ade_cv:+.1f}% vs CV)")

    (OUT / "06_stable_M_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT / '06_stable_M_sweep.json'}")


if __name__ == "__main__":
    main()
