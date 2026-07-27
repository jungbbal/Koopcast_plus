"""Q: is a soft mixture better than hard switching -- and if so, is it just averaging?

Experiment 6 fixed the confound that made the M-sweep unreadable: with scaled
ridge every K_j is fit stably, so a soft mixture's gain can no longer be
explained as "averaging away an unstable operator". That makes this the right
moment to ask the question.

  hard   psi <- K_{argmax_j pi_j} psi
  soft   psi <- (sum_j pi_j K_j) psi        pi from a softmax over cluster distance

THE CONTROL THAT DECIDES IT. A soft mixture beating hard switching does NOT show
that smooth regime transitions matter. sum_j pi_j K_j is a convex combination, so
it is *shrunk toward the mean operator* -- a plain variance-reduction effect that
has nothing to do with regimes. Two controls separate the two stories:

  uniform  psi <- (mean_j K_j) psi          state-independent average: pure shrinkage
  global   psi <- K psi                     one operator fit on everything

If soft ~= uniform, the gain is operator averaging and the pi_t carry no
information. Only soft < uniform licenses "the state-dependent mixture matters".

The temperature sweep is the same statement as a curve: tau -> 0 reproduces hard,
tau -> inf reproduces uniform. Where the minimum sits is the answer.

Run: python exp/07_hard_vs_soft.py
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
from bsk.stability import diagnose, fit_scaled_ridge

OUT = _HERE.parents[1] / "out"
MS = [4, 8, 16]
SEEDS = [0, 1, 2]
OBS = Observable(5, True, "delay-5 +nonlin")
RIDGE = 1e-2
MIN_COUNT = 50
TAUS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def build_operators(X, Y, lab, M, K_glob):
    """One scaled-ridge K per cluster -- the exp-6 winner, held fixed here."""
    Ks, sizes = [], []
    for j in range(M):
        m = lab == j
        sizes.append(int(m.sum()))
        if m.sum() < MIN_COUNT:
            Ks.append(K_glob)
            continue
        Ks.append(fit_scaled_ridge(X[m], Y[m], RIDGE, len(X)))
    return np.stack(Ks), sizes


def make_weights(C, scale):
    """pi_j(psi) ∝ exp(-||f - c_j||^2 / (tau * scale)); returns a closure factory."""
    def weights(x, tau):
        d = ((vel_block(x) - C) ** 2).sum(axis=-1)
        z = -d / max(tau * scale, 1e-12)
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()
    return weights


def main():
    OUT.mkdir(exist_ok=True)
    zara2 = zlab.load_zara2_scene()
    fit_scenes, test_scenes = zlab.load("train"), [zara2]
    ade_cv, fde_cv = const_vel(test_scenes)
    X, Y = pairs(fit_scenes, OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)

    print(f"=== LOO (train -> zara2), observable={OBS.name}, ridge={RIDGE:.0e}, "
          f"seeds={SEEDS} ===")
    print(f"const-velocity   ADE={ade_cv:.4f}  FDE={fde_cv:.4f}")
    ade_g, fde_g, clip_g, n_win = rollout(test_scenes, OBS, lambda x: K_glob)
    print(f"global K (M=1)   ADE={ade_g:.4f}  FDE={fde_g:.4f}  "
          f"clipped={clip_g}/{n_win}\n")

    results = {"const_vel": {"ade": ade_cv, "fde": fde_cv},
               "global": {"ade": ade_g, "fde": fde_g, "clipped": clip_g},
               "cells": []}

    for M in MS:
        print(f"--- M={M} " + "-" * 62)
        rows = {}
        for seed in SEEDS:
            km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
            lab, C = km.labels_, km.cluster_centers_
            Ks, sizes = build_operators(X, Y, lab, M, K_glob)
            K_unif = Ks.mean(axis=0)
            # distance scale: typical squared distance to the assigned centre,
            # so tau is dimensionless and comparable across M and seeds
            scale = float(np.mean(((vel_block(X) - C[lab]) ** 2).sum(axis=1)))
            weights = make_weights(C, scale)

            def hard_K(x, Ks=Ks, C=C):
                return Ks[int(np.argmin(((vel_block(x) - C) ** 2).sum(axis=-1)))]

            trials = [("hard", hard_K), ("uniform", lambda x, K=K_unif: K)]
            for tau in TAUS:
                def soft_K(x, Ks=Ks, tau=tau, weights=weights):
                    return np.tensordot(weights(x, tau), Ks, axes=1)
                trials.append((f"soft tau={tau}", soft_K))

            for name, fn in trials:
                a, f, c, _ = rollout(test_scenes, OBS, fn)
                rows.setdefault(name, []).append((a, f, c))

            if seed == SEEDS[0]:
                d = [diagnose(k) for k in Ks]
                print(f"  sizes={sorted(sizes)[:3]}..{sorted(sizes)[-1]}  "
                      f"max rho(K_j)={max(x['rho'] for x in d):.3f}  "
                      f"rho(K_unif)={diagnose(K_unif)['rho']:.3f}")

        print(f"  {'variant':16s}{'ADE':>9s}{'std':>8s}{'FDE':>9s}"
              f"{'vsCV':>8s}{'vs hard':>9s}{'clip':>6s}")
        ade_hard = float(np.mean([r[0] for r in rows["hard"]]))
        for name, rs in rows.items():
            a = float(np.mean([r[0] for r in rs]))
            s = float(np.std([r[0] for r in rs]))
            f = float(np.mean([r[1] for r in rs]))
            c = int(max(r[2] for r in rs))
            results["cells"].append({"M": M, "variant": name, "ade": a,
                                     "ade_std": s, "fde": f, "max_clipped": c})
            print(f"  {name:16s}{a:9.4f}{s:8.4f}{f:9.4f}"
                  f"{100*(ade_cv-a)/ade_cv:+7.1f}%{100*(ade_hard-a)/ade_hard:+8.2f}%"
                  f"{c:6d}")
        print()

    best = min(results["cells"], key=lambda c: c["ade"])
    print(f"best: M={best['M']} {best['variant']} ADE={best['ade']:.4f} "
          f"({100*(ade_cv-best['ade'])/ade_cv:+.1f}% vs CV)")
    (OUT / "07_hard_vs_soft.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT / '07_hard_vs_soft.json'}")


if __name__ == "__main__":
    main()
