"""Q: is any of the gain actually coming from *switching*?

Experiment 8 measured mean self-transition = 0.965 at M=8. Over a 12-step rollout
that implies the token changes well under once per window. If so, "switching
Koopman" is a misnomer for what is winning: the model would be picking one local
operator per trajectory and holding it -- a static mixture of experts on velocity
features, with a piecewise-linear-regression story, not a regime-switching one.

The decisive comparison is cheap:

  switching   token recomputed from the propagated psi at every step (the exp 6-8 model)
  frozen      token computed once from the last observed state, then held for all 12
  oracle-1    frozen, but ALSO reports how often the switching variant actually moved

If frozen == switching, switching contributes nothing and the correct description
of the result is local linear regression in velocity space. If frozen is clearly
worse, the temporal re-assignment is doing real work and the switching framing
survives.

Also reported: how zara2 states distribute over regimes that were defined on the
training scenes. Experiment 8 found regimes 1 and 3 draw 91-95% of their training
mass from biwi_eth. If zara2 rarely lands in them, those operators are scene
tokens that happen to be harmless rather than transferable primitives.

Run: python exp/09_is_it_switching.py
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
MS = [4, 8, 16]
SEEDS = [0, 1, 2]
N_OBS, N_PRED = 8, 12


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def rollout_modes(scenes, Ks, C, K_glob):
    """One pass returning ADE/FDE for switching and frozen, plus switch counts."""
    acc = {"switch": ([], []), "frozen": ([], [])}
    n_changes, n_win = [], 0
    L = N_OBS + N_PRED
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1):
                w = traj[s:s + L]
                psi, _ = OBS.lift(w[:N_OBS])
                if len(psi) == 0:
                    continue
                n_win += 1
                j0 = int(np.argmin(((vel_block(psi[-1]) - C) ** 2).sum(-1)))
                for mode in ("switch", "frozen"):
                    x, pred, seq = psi[-1].copy(), [], []
                    for _ in range(N_PRED):
                        j = (int(np.argmin(((vel_block(x) - C) ** 2).sum(-1)))
                             if mode == "switch" else j0)
                        seq.append(j)
                        x = Ks[j] @ x
                        pred.append(x[:2].copy())
                    pred = np.asarray(pred)
                    if not np.isfinite(pred).all() or np.abs(pred).max() > 1e3:
                        pred = np.clip(np.nan_to_num(pred), -1e3, 1e3)
                    d = np.linalg.norm(pred - w[N_OBS:], axis=-1)
                    acc[mode][0].append(d.mean())
                    acc[mode][1].append(d[-1])
                    if mode == "switch":
                        n_changes.append(sum(a != b for a, b in zip(seq[:-1], seq[1:])))
    out = {m: (float(np.mean(a)), float(np.mean(f))) for m, (a, f) in acc.items()}
    return out, float(np.mean(n_changes)), float(np.mean(np.array(n_changes) > 0)), n_win


def main():
    OUT.mkdir(exist_ok=True)
    test = [zlab.load_zara2_scene()]
    X, Y = pairs(zlab.load("train"), OBS)
    Xt, _ = pairs(test, OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)
    ade_cv, fde_cv = const_vel(test)
    res = {"const_vel": {"ade": ade_cv, "fde": fde_cv}, "cells": [], "usage": {}}

    print(f"=== LOO (train -> zara2), CV ADE={ade_cv:.4f} ===\n")
    print(f"  {'M':>3s}{'switching':>11s}{'frozen':>11s}{'diff':>9s}"
          f"{'switches/12':>13s}{'win w/ switch':>15s}")
    for M in MS:
        rows = {"switch": [], "frozen": []}
        nc, fr = [], []
        for seed in SEEDS:
            km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
            C, lab = km.cluster_centers_, km.labels_
            Ks = np.stack([fit_scaled_ridge(X[lab == j], Y[lab == j], RIDGE, len(X))
                           if (lab == j).sum() >= MIN_COUNT else K_glob
                           for j in range(M)])
            out, c, f, _ = rollout_modes(test, Ks, C, K_glob)
            for m in rows:
                rows[m].append(out[m])
            nc.append(c)
            fr.append(f)
        a_s = float(np.mean([r[0] for r in rows["switch"]]))
        a_f = float(np.mean([r[0] for r in rows["frozen"]]))
        res["cells"].append({
            "M": M, "ade_switch": a_s, "ade_frozen": a_f,
            "fde_switch": float(np.mean([r[1] for r in rows["switch"]])),
            "fde_frozen": float(np.mean([r[1] for r in rows["frozen"]])),
            "mean_switches": float(np.mean(nc)),
            "frac_windows_with_switch": float(np.mean(fr))})
        print(f"  {M:3d}{a_s:11.4f}{a_f:11.4f}{100*(a_f-a_s)/a_s:+8.2f}%"
              f"{np.mean(nc):13.2f}{np.mean(fr):15.2f}")

    # which regimes do zara2 states actually land in?
    km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(vel_block(X))
    C = km.cluster_centers_
    a_tr = np.bincount(km.labels_, minlength=8) / len(X)
    a_te = np.bincount(np.argmin(((vel_block(Xt)[:, None] - C[None]) ** 2).sum(-1), 1),
                       minlength=8) / len(Xt)
    res["usage"] = {"train": a_tr.tolist(), "zara2": a_te.tolist()}
    print(f"\n  M=8 regime occupancy   " + "".join(f"{j:7d}" for j in range(8)))
    print(f"    train                " + "".join(f"{x:7.3f}" for x in a_tr))
    print(f"    zara2 (test)         " + "".join(f"{x:7.3f}" for x in a_te))

    (OUT / "09_is_it_switching.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / '09_is_it_switching.json'}")


if __name__ == "__main__":
    main()
