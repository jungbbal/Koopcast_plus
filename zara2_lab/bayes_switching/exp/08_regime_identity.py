"""Q: what are the M=8 regimes -- and are they anything more than speed bins?

Experiment 7 established that hard switching wins and that the K_j are genuinely
local operators (3x worse off their own domain), so asking "what are they" is now
a well-posed question rather than an invitation to name noise.

Two things are measured, and the second is what keeps the first honest.

1. CONDITIONAL MOTION STATISTICS per cluster: E[||v||], E[|dtheta|], E[dspeed],
   the transition matrix p(q_{t+1}|q_t), and the training-scene composition.
   Scene composition is the confound check the plan asks for: if cluster j draws
   80% of its mass from one source file, it is a scene token, not a motion
   primitive, and it should not transfer.

2. PARAMETER-MATCHED ALTERNATIVE PARTITIONS at the same M. Naming clusters from
   their statistics is unfalsifiable on its own -- any partition of velocity space
   yields a describable table. So the same pipeline is rerun with the partition
   swapped for deliberately impoverished ones:

     speed-quantile  M bins on ||v_t|| alone       -- "regimes are just speed"
     heading-octant  M bins on direction of v_t    -- "regimes are just direction"
     speed x dtheta  2D k-means on (speed, dtheta) -- the handcrafted primitive axes
     random          size-matched label shuffle    -- the parameter-count control

   Every variant gets M operators, the same scaled ridge, the same rollout. If
   speed-quantile matches k-means, the regimes are speed bins and the velocity-
   delay lift bought nothing. If heading-octant matches, they are scene-specific
   and LOO transfer was luck.

Run: python exp/08_regime_identity.py
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
OBS = Observable(5, True, "delay-5 +nonlin")
RIDGE, MIN_COUNT = 1e-2, 50
MS = [4, 8]
SEEDS = [0, 1, 2]


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def motion_feats(psi):
    """(speed, dtheta, dspeed) from the lift -- v_t = psi[2:4], v_{t-1} = psi[4:6]."""
    v, vp = psi[..., 2:4], psi[..., 4:6]
    sp, spp = np.linalg.norm(v, axis=-1), np.linalg.norm(vp, axis=-1)
    dth = np.arctan2(v[..., 1], v[..., 0]) - np.arctan2(vp[..., 1], vp[..., 0])
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    return sp, dth, sp - spp


# --------------------------------------------------------------------------- #
# partitions: each returns (labels on X, assign_fn(psi) -> int)
# --------------------------------------------------------------------------- #
def part_kmeans(X, M, seed):
    km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
    C = km.cluster_centers_
    return km.labels_, lambda x: int(np.argmin(((vel_block(x) - C) ** 2).sum(-1)))


def part_speed(X, M, seed):
    sp, _, _ = motion_feats(X)
    edges = np.quantile(sp, np.linspace(0, 1, M + 1)[1:-1])
    lab = np.searchsorted(edges, sp)

    def assign(x):
        s, _, _ = motion_feats(x)
        return int(np.searchsorted(edges, s))
    return lab, assign


def part_heading(X, M, seed):
    v = X[..., 2:4]
    th = np.arctan2(v[:, 1], v[:, 0])
    lab = np.floor((th + np.pi) / (2 * np.pi) * M).astype(int) % M

    def assign(x):
        t = np.arctan2(x[3], x[2])
        return int(np.floor((t + np.pi) / (2 * np.pi) * M)) % M
    return lab, assign


def part_speed_dtheta(X, M, seed):
    sp, dth, _ = motion_feats(X)
    F = np.stack([sp, dth], 1)
    mu, sd = F.mean(0), F.std(0) + 1e-12
    km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit((F - mu) / sd)
    C = km.cluster_centers_

    def assign(x):
        s, d, _ = motion_feats(x)
        f = (np.array([s, d]) - mu) / sd
        return int(np.argmin(((f - C) ** 2).sum(-1)))
    return km.labels_, assign


def part_random(X, M, seed):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, M, len(X))
    return lab, lambda x, rng=np.random.default_rng(seed + 999): int(rng.integers(0, M))


PARTS = {"k-means (vel-delay)": part_kmeans, "speed-quantile": part_speed,
         "heading-octant": part_heading, "speed x dtheta": part_speed_dtheta,
         "random": part_random}


def build(X, Y, lab, M, K_glob):
    return np.stack([
        fit_scaled_ridge(X[lab == j], Y[lab == j], RIDGE, len(X))
        if (lab == j).sum() >= MIN_COUNT else K_glob for j in range(M)])


def main():
    OUT.mkdir(exist_ok=True)
    test = [zlab.load_zara2_scene()]
    train_scenes = zlab.load("train")
    X, Y = pairs(train_scenes, OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)
    ade_cv, _ = const_vel(test)

    # per-scene lifts, so cluster scene-composition can be measured
    per_scene = {sc.name: pairs([sc], OBS)[0] for sc in train_scenes}

    res = {"const_vel_ade": ade_cv, "partitions": [], "regimes": []}
    print(f"=== LOO (train -> zara2), CV ADE={ade_cv:.4f} ===\n")

    # ------------------------------------------------------------------ #
    # 1. partition comparison
    # ------------------------------------------------------------------ #
    print("ADE by partition (parameter-matched: M operators each)\n")
    print(f"  {'partition':22s}" + "".join(f"{f'M={m}':>16s}" for m in MS))
    for name, fn in PARTS.items():
        cells = []
        for M in MS:
            ades = []
            for seed in SEEDS:
                lab, assign = fn(X, M, seed)
                Ks = build(X, Y, lab, M, K_glob)
                ades.append(rollout(test, OBS, lambda x: Ks[assign(x)])[0])
                if name in ("speed-quantile", "heading-octant"):
                    break            # deterministic, seed does nothing
            a, s = float(np.mean(ades)), float(np.std(ades))
            res["partitions"].append({"partition": name, "M": M, "ade": a,
                                      "ade_std": s,
                                      "vs_cv_pct": 100 * (ade_cv - a) / ade_cv})
            cells.append(f"{a:.4f}±{s:.4f}")
        print(f"  {name:22s}" + "".join(f"{c:>16s}" for c in cells))

    # ------------------------------------------------------------------ #
    # 2. what the M=8 k-means regimes are
    # ------------------------------------------------------------------ #
    M = 8
    lab, assign = part_kmeans(X, M, 0)
    Ks = build(X, Y, lab, M, K_glob)
    sp, dth, dsp = motion_feats(X)
    v = X[:, 2:4]
    heading = np.arctan2(v[:, 1], v[:, 0])

    print(f"\n\nM={M} k-means regimes (train), sorted by mean speed\n")
    order = sorted(range(M), key=lambda j: sp[lab == j].mean())
    print(f"  {'j':>3s}{'N':>7s}{'frac':>7s}{'E|v|':>8s}{'E|dth|':>8s}"
          f"{'E dsp':>8s}{'rho':>7s}{'top scene':>18s}{'share':>7s}")
    scene_lab = {n: assign_batch(Xs, assign) for n, Xs in per_scene.items()}
    for j in order:
        m = lab == j
        tot = {n: float((sl == j).sum()) for n, sl in scene_lab.items()}
        s_tot = sum(tot.values())
        comp = {n: tot[n] / max(s_tot, 1) for n in tot}
        top = max(comp, key=comp.get)
        rec = {"j": int(j), "N": int(m.sum()), "frac": float(m.mean()),
               "speed": float(sp[m].mean()), "abs_dtheta": float(np.abs(dth[m]).mean()),
               "dspeed": float(dsp[m].mean()),
               "heading_mean": float(np.arctan2(np.sin(heading[m]).mean(),
                                                np.cos(heading[m]).mean())),
               "heading_conc": float(np.hypot(np.sin(heading[m]).mean(),
                                              np.cos(heading[m]).mean())),
               "rho": diagnose(Ks[j])["rho"], "scene_composition": comp}
        res["regimes"].append(rec)
        print(f"  {j:3d}{m.sum():7d}{m.mean():7.3f}{sp[m].mean():8.4f}"
              f"{np.abs(dth[m]).mean():8.4f}{dsp[m].mean():+8.4f}"
              f"{rec['rho']:7.3f}{top:>18s}{comp[top]:7.2f}")

    # heading concentration: 0 = isotropic (motion primitive), 1 = one direction
    print("\n  heading concentration |E[e^{i theta}]| per regime "
          "(0=isotropic, 1=single direction):")
    print("   " + "".join(f"{r['heading_conc']:7.2f}" for r in
                          sorted(res['regimes'], key=lambda r: r['j'])))

    # transition matrix, within-track only
    P = transitions(train_scenes, assign, M)
    res["transition"] = P.tolist()
    print(f"\n  p(q_{{t+1}}|q_t), rows sorted by speed (diag = self-transition):")
    print(f"  {'j':>3s}" + "".join(f"{k:7d}" for k in order) + f"{'self':>9s}")
    for j in order:
        print(f"  {j:3d}" + "".join(f"{P[j, k]:7.2f}" for k in order)
              + f"{P[j, j]:9.2f}")
    res["mean_self_transition"] = float(np.diag(P).mean())
    print(f"\n  mean self-transition = {np.diag(P).mean():.3f} "
          f"(1/M = {1/M:.3f} would mean no persistence)")

    (OUT / "08_regime_identity.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / '08_regime_identity.json'}")


def assign_batch(X, assign):
    return np.array([assign(x) for x in X])


def transitions(scenes, assign, M):
    """p(q_{t+1}|q_t) accumulated inside tracks only -- never across a boundary."""
    Cm = np.zeros((M, M))
    for sc in scenes:
        for traj in sc.tracks(min_len=OBS.need + 3).values():
            psi, _ = OBS.lift(traj)
            if len(psi) < 2:
                continue
            q = assign_batch(psi, assign)
            for a, b in zip(q[:-1], q[1:]):
                Cm[a, b] += 1
    return Cm / np.maximum(Cm.sum(1, keepdims=True), 1)


if __name__ == "__main__":
    main()
