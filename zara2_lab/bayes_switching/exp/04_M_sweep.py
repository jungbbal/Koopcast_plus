"""Q: are there a few discrete motion regimes, or is the variation continuous?

The decisive measurement. Hold the observable fixed, sweep the number of local
operators M in {1,2,4,8,16,32}, and look at the SHAPE of the ADE curve:

  saturates at small M  -> a few discrete regimes exist; a hard token/HMM is right
  keeps improving       -> the dynamics vary continuously; a soft/Bayesian
                           mixture is the right model and hard tokens are just
                           a piecewise approximation of a continuum

Partition is k-means on the velocity-delay block of psi (not the handcrafted
tokens, which cap out at 5 and are 88% 'straight' on zara2). Assignment during
rollout uses the *propagated* psi, so nothing from the future leaks in.

The random control is re-run at EVERY M and matters more here than anywhere
else: M operators always fit better, so the only meaningful quantity is the gap
between k-means and a size-matched random partition at the same M.

Run: python exp/04_M_sweep.py
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

OUT = _HERE.parents[1] / "out"
MS = [1, 2, 4, 8, 16, 32]
OBS = Observable(5, True, "delay-5 +nonlin")   # best global lift from exp 03
RIDGE = 1e-2
MIN_COUNT = 50


def agent_split(sc, frac=0.5, seed=0):
    ids = np.unique(sc.agents)
    perm = np.random.default_rng(seed).permutation(ids)
    a = set(perm[: int(len(perm) * frac)].tolist())
    masks = [np.array([x in a for x in sc.agents]),
             np.array([x not in a for x in sc.agents])]
    return [zlab.Scene(f"{sc.name}_{k}", sc.frames[m], sc.agents[m], sc.xy[m])
            for k, m in zip(("fit", "held"), masks)]


def vel_block(psi: np.ndarray) -> np.ndarray:
    """The velocity-delay part of psi -- what the partition is defined on."""
    return psi[..., 2:2 + 2 * OBS.n_delay]


def fit_local(X, Y, lab, M, K_glob):
    """One K per group, falling back to the global K where data is thin."""
    Ks = []
    for j in range(M):
        m = lab == j
        Ks.append(fit_lstsq(X[m], Y[m], RIDGE) if m.sum() >= MIN_COUNT else K_glob)
    return np.stack(Ks)


def run(fit_scenes, test_scenes, label, results, seed=0):
    rng = np.random.default_rng(seed)
    ade_cv, fde_cv = const_vel(test_scenes)
    X, Y = pairs(fit_scenes, OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)

    print(f"\n=== {label} ===   observable={OBS.name} (dim {OBS.dim}), ridge={RIDGE:.0e}")
    print(f"  const-velocity   ADE={ade_cv:.4f}  FDE={fde_cv:.4f}   "
          f"(fit samples={len(X)})")
    print(f"  {'M':>3s} {'kmeans ADE':>11s} {'random ADE':>11s} {'km vs CV':>9s} "
          f"{'km vs rnd':>10s} {'thin':>5s} {'div':>4s}")
    rows = []

    for M in MS:
        if M == 1:
            lab_fit = np.zeros(len(X), int)
            assign = lambda x: 0                                    # noqa: E731
        else:
            km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
            lab_fit = km.labels_
            C = km.cluster_centers_
            def assign(x, C=C):
                return int(np.argmin(((vel_block(x) - C) ** 2).sum(axis=1)))

        n_thin = int(sum(np.sum(lab_fit == j) < MIN_COUNT for j in range(M)))
        Ks = fit_local(X, Y, lab_fit, M, K_glob)
        ade_k, fde_k, div_k, _ = rollout(test_scenes, OBS, lambda x: Ks[assign(x)])

        # size-matched random control: same group sizes, no motion information
        lab_rnd = rng.permutation(lab_fit)
        Ks_r = fit_local(X, Y, lab_rnd, M, K_glob)
        ade_r, fde_r, div_r, _ = rollout(
            test_scenes, OBS, lambda _x: Ks_r[rng.integers(M)])

        rows.append({"M": M, "ade_kmeans": ade_k, "fde_kmeans": fde_k,
                     "ade_random": ade_r, "fde_random": fde_r,
                     "n_thin_clusters": n_thin,
                     "n_diverged": div_k + div_r})
        print(f"  {M:3d} {ade_k:11.4f} {ade_r:11.4f} "
              f"{100*(ade_cv-ade_k)/ade_cv:+8.1f}% "
              f"{100*(ade_r-ade_k)/ade_r:+9.1f}% {n_thin:5d} {div_k+div_r:4d}")

    results[label] = {"const_vel": {"ade": ade_cv, "fde": fde_cv},
                      "observable": OBS.name, "ridge": RIDGE, "sweep": rows}


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    zara2 = zlab.load_zara2_scene()
    fit_sc, held_sc = agent_split(zara2)
    run([fit_sc], [held_sc], "within-zara2 (agent split)", results)
    run(zlab.load("train"), [zara2], "loo (train split -> zara2)", results)
    (OUT / "04_M_sweep.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / '04_M_sweep.json'}")


if __name__ == "__main__":
    main()
