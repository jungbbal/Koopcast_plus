"""Q: does the 18x18 Koopman operator earn its parameters?

Experiment 9 showed the token is frozen for 90% of windows, so the winning model
is really "pick one local operator from the velocity vector, apply it 12 times".
That is a piecewise-linear approximation of a velocity field -- and if that is all
it is, a far smaller local model should capture the same gain.

A ladder of local models over the SAME M=8 k-means partition, so only the
per-regime model class changes:

  cv              p_{t+h} = p + h*v                        0 params
  accel-global    v <- v + a                               2
  accel-regime    v <- v + a_j                             2M
  vlin-regime     v <- A_j v + b_j                         6M
  vlin+hist       v <- A_j v + B_j v_{t-1} + b_j          10M
  koopman-regime  psi <- K_j psi (delay-5 +nonlin)        324M   <- the current model

Reading: the smallest rung that reaches ~0.305 is the honest description of the
result. If accel-regime already gets there, the Koopman lift, the delays and the
nonlinear terms were all decoration and the finding is "different speed groups
have different mean acceleration".

Both frozen and switching assignment are reported for every rung, since exp 9
found they barely differ and that should hold across the ladder if the
mixture-of-experts reading is right.

Run: python exp/10_complexity_ladder.py
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
M = 8
SEEDS = [0, 1, 2]
N_OBS, N_PRED = 8, 12


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


# --------------------------------------------------------------------------- #
# local model classes: fit(Xc, Yc) -> params ; step(params, v, vprev, psi) -> v_next
# --------------------------------------------------------------------------- #
def fit_accel(Xc, Yc):
    return (Yc[:, 2:4] - Xc[:, 2:4]).mean(axis=0)                    # a


def step_accel(a, v, vprev, psi):
    return v + a


def fit_vlin(Xc, Yc):
    F = np.concatenate([Xc[:, 2:4], np.ones((len(Xc), 1))], 1)       # [v, 1]
    return np.linalg.solve(F.T @ F + RIDGE * np.eye(3), F.T @ Yc[:, 2:4])


def step_vlin(W, v, vprev, psi):
    return np.concatenate([v, [1.0]]) @ W


def fit_vhist(Xc, Yc):
    F = np.concatenate([Xc[:, 2:4], Xc[:, 4:6], np.ones((len(Xc), 1))], 1)
    return np.linalg.solve(F.T @ F + RIDGE * np.eye(5), F.T @ Yc[:, 2:4])


def step_vhist(W, v, vprev, psi):
    return np.concatenate([v, vprev, [1.0]]) @ W


LADDER = [
    ("accel-regime", fit_accel, step_accel, 2),
    ("vlin-regime", fit_vlin, step_vlin, 6),
    ("vlin+hist", fit_vhist, step_vhist, 10),
]


def rollout_small(scenes, params, C, step, frozen):
    """Rollout where only the velocity block is modelled; position integrates it."""
    ades, fdes = [], []
    L = N_OBS + N_PRED
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1):
                w = traj[s:s + L]
                psi, _ = OBS.lift(w[:N_OBS])
                if len(psi) == 0:
                    continue
                x = psi[-1]
                p, v, vprev = x[:2].copy(), x[2:4].copy(), x[4:6].copy()
                j0 = int(np.argmin(((vel_block(x) - C) ** 2).sum(-1)))
                lift = x.copy()
                pred = []
                for _ in range(N_PRED):
                    if frozen:
                        j = j0
                    else:
                        lift[2:4], lift[4:6] = v, vprev
                        j = int(np.argmin(((vel_block(lift) - C) ** 2).sum(-1)))
                    vn = step(params[j], v, vprev, lift)
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


def rollout_koop(scenes, Ks, C, frozen):
    ades, fdes = [], []
    L = N_OBS + N_PRED
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1):
                w = traj[s:s + L]
                psi, _ = OBS.lift(w[:N_OBS])
                if len(psi) == 0:
                    continue
                x = psi[-1].copy()
                j0 = int(np.argmin(((vel_block(x) - C) ** 2).sum(-1)))
                pred = []
                for _ in range(N_PRED):
                    j = j0 if frozen else int(
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
    ade_cv, fde_cv = const_vel(test)
    res = {"const_vel": {"ade": ade_cv, "fde": fde_cv}, "rungs": []}

    print(f"=== LOO (train -> zara2), M={M}, seeds={SEEDS} ===\n")
    print(f"  {'model':18s}{'params':>8s}{'ADE frozen':>12s}{'ADE switch':>12s}"
          f"{'FDE switch':>12s}{'vs CV':>9s}")
    print(f"  {'const-velocity':18s}{0:8d}{ade_cv:12.4f}{ade_cv:12.4f}"
          f"{fde_cv:12.4f}{0.0:+8.1f}%")

    # global constant acceleration -- the M=1 rung of the same ladder
    a_g = fit_accel(X, Y)
    ade_ag, fde_ag = rollout_small(test, {0: a_g}, np.zeros((1, 2 * OBS.n_delay)),
                                   step_accel, True)
    res["rungs"].append({"model": "accel-global", "params": 2,
                         "ade_frozen": ade_ag, "ade_switch": ade_ag, "fde": fde_ag})
    print(f"  {'accel-global':18s}{2:8d}{ade_ag:12.4f}{ade_ag:12.4f}"
          f"{fde_ag:12.4f}{100*(ade_cv-ade_ag)/ade_cv:+8.1f}%")

    parts = []
    for seed in SEEDS:
        km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
        parts.append((km.labels_, km.cluster_centers_))

    for name, fit, step, npar in LADDER:
        af, asw, fsw = [], [], []
        for lab, C in parts:
            params = {}
            for j in range(M):
                m = lab == j
                params[j] = fit(X[m], Y[m]) if m.sum() >= MIN_COUNT else fit(X, Y)
            af.append(rollout_small(test, params, C, step, True)[0])
            a, f = rollout_small(test, params, C, step, False)
            asw.append(a)
            fsw.append(f)
        r = {"model": name, "params": npar * M, "ade_frozen": float(np.mean(af)),
             "ade_switch": float(np.mean(asw)), "fde": float(np.mean(fsw)),
             "ade_switch_std": float(np.std(asw))}
        res["rungs"].append(r)
        print(f"  {name:18s}{npar*M:8d}{r['ade_frozen']:12.4f}"
              f"{r['ade_switch']:12.4f}{r['fde']:12.4f}"
              f"{100*(ade_cv-r['ade_switch'])/ade_cv:+8.1f}%")

    af, asw, fsw = [], [], []
    for lab, C in parts:
        Ks = np.stack([fit_scaled_ridge(X[lab == j], Y[lab == j], RIDGE, len(X))
                       if (lab == j).sum() >= MIN_COUNT else K_glob
                       for j in range(M)])
        af.append(rollout_koop(test, Ks, C, True)[0])
        a, f = rollout_koop(test, Ks, C, False)
        asw.append(a)
        fsw.append(f)
    r = {"model": "koopman-regime", "params": OBS.dim ** 2 * M,
         "ade_frozen": float(np.mean(af)), "ade_switch": float(np.mean(asw)),
         "fde": float(np.mean(fsw)), "ade_switch_std": float(np.std(asw))}
    res["rungs"].append(r)
    print(f"  {'koopman-regime':18s}{r['params']:8d}{r['ade_frozen']:12.4f}"
          f"{r['ade_switch']:12.4f}{r['fde']:12.4f}"
          f"{100*(ade_cv-r['ade_switch'])/ade_cv:+8.1f}%")

    (OUT / "10_complexity_ladder.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / '10_complexity_ladder.json'}")


if __name__ == "__main__":
    main()
