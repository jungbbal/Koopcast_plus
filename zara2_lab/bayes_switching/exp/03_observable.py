"""Q: can ONE global Koopman operator beat constant velocity, given a better lift?

The gate before any switching work. Experiments 1-2 showed K_j on psi=[p,v,1]
can only encode a per-token constant acceleration, and every variant lost to
constant velocity in closed loop. So: hold M=1, sweep the observable, and ask
only whether ADE_global-K < ADE_CV.

Two protocols, because experiment 2 showed the operators are scene-specific:
  within  -- fit and test inside zara2, split by agent (no domain shift)
  loo     -- fit on the ETH-UCY train split, test on zara2 (the real benchmark)

Metric is closed-loop 12-step ADE/FDE in metres, not one-step MSE. One-step
residual is dominated by components that are free by construction (see
koopman.py) and says almost nothing about rollout quality.

Ridge is swept too: a K fit at ridge=1e-6 can have eigenvalues > 1 and diverge
over 12 steps, which is a property of the fit, not of the observable.

Run: python exp/03_observable.py
"""
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np

import zlab
from bsk import fit_lstsq
from bsk.observables import OBSERVABLES, const_vel, pairs, rollout

OUT = _HERE.parents[1] / "out"
RIDGES = [1e-6, 1e-4, 1e-2, 1e-1]


def agent_split(sc, frac=0.5, seed=0):
    ids = np.unique(sc.agents)
    perm = np.random.default_rng(seed).permutation(ids)
    a = set(perm[: int(len(perm) * frac)].tolist())
    masks = [np.array([x in a for x in sc.agents]),
             np.array([x not in a for x in sc.agents])]
    return [zlab.Scene(f"{sc.name}_{k}", sc.frames[m], sc.agents[m], sc.xy[m])
            for k, m in zip(("fit", "held"), masks)]


def run(fit_scenes, test_scenes, label, results):
    ade_cv, fde_cv = const_vel(test_scenes)
    print(f"\n=== {label} ===")
    print(f"  {'const-velocity':22s} {'':>8s} ADE={ade_cv:.4f}  FDE={fde_cv:.4f}")
    print(f"  {'observable':22s} {'ridge':>8s} {'ADE':>8s} {'FDE':>8s} "
          f"{'vs CV':>8s} {'rho(K)':>7s} {'div':>5s}")
    results[label] = {"const_vel": {"ade": ade_cv, "fde": fde_cv}, "runs": []}

    for obs in OBSERVABLES:
        X, Y = pairs(fit_scenes, obs)
        best = None
        for ridge in RIDGES:
            K = fit_lstsq(X, Y, ridge)
            rho = float(np.abs(np.linalg.eigvals(K)).max())
            ade, fde, n_clip, n = rollout(test_scenes, obs, lambda _x, K=K: K)
            row = {"observable": obs.name, "dim": obs.dim, "ridge": ridge,
                   "ade": ade, "fde": fde, "rho": rho,
                   "n_diverged": n_clip, "n_windows": n}
            results[label]["runs"].append(row)
            if best is None or ade < best["ade"]:
                best = row
        b = best
        print(f"  {b['observable']:22s} {b['ridge']:8.0e} {b['ade']:8.4f} "
              f"{b['fde']:8.4f} {100*(ade_cv-b['ade'])/ade_cv:+7.1f}% "
              f"{b['rho']:7.3f} {b['n_diverged']:5d}")
    return results


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    zara2 = zlab.load_zara2_scene()
    fit_sc, held_sc = agent_split(zara2)

    run([fit_sc], [held_sc], "within-zara2 (agent split)", results)
    run(zlab.load("train"), [zara2], "loo (train split -> zara2)", results)

    (OUT / "03_observable.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / '03_observable.json'}")
    print("\n(best ridge per observable shown; all runs in the json)")


if __name__ == "__main__":
    main()
