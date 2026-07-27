"""Q: is the M=8 peak from experiment 4 real, or one lucky k-means seed?

Experiment 4 found the best result so far -- LOO, M=8, ADE 0.3062 vs CV 0.3219
(+4.9%) -- on a single seed, and experiment 2 already showed this setup is
seed-sensitive. So: re-run the interesting M values across k-means seeds and
report mean +/- std. A peak that does not survive this is not a peak.

Also records the spectral radius of the least stable K_j, to explain the M=32
blow-up (ADE 1.05): with 32 groups some operators are fit on few samples and
come out with eigenvalues > 1, which a 12-step rollout amplifies.

Run: python exp/05_seed_stability.py
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
SEEDS = range(5)
OBS = Observable(5, True, "delay-5 +nonlin")
RIDGE = 1e-2
MIN_COUNT = 50


def vel_block(psi):
    return psi[..., 2:2 + 2 * OBS.n_delay]


def main():
    OUT.mkdir(exist_ok=True)
    zara2 = zlab.load_zara2_scene()
    fit_scenes, test_scenes = zlab.load("train"), [zara2]
    ade_cv, _ = const_vel(test_scenes)
    X, Y = pairs(fit_scenes, OBS)
    K_glob = fit_lstsq(X, Y, RIDGE)

    print(f"=== LOO (train -> zara2), {len(list(SEEDS))} k-means seeds ===")
    print(f"const-velocity ADE={ade_cv:.4f}\n")
    print(f"  {'M':>3s} {'ADE mean':>9s} {'std':>7s} {'min':>7s} {'max':>7s} "
          f"{'vs CV':>8s} {'max rho':>8s}")
    results = {"const_vel_ade": ade_cv, "sweep": []}

    for M in MS:
        ades, rhos = [], []
        for seed in SEEDS:
            if M == 1:
                lab = np.zeros(len(X), int)
                assign = lambda x: 0                                # noqa: E731
            else:
                km = KMeans(n_clusters=M, n_init=10, random_state=seed).fit(vel_block(X))
                lab, C = km.labels_, km.cluster_centers_
                def assign(x, C=C):
                    return int(np.argmin(((vel_block(x) - C) ** 2).sum(axis=1)))
            Ks = np.stack([
                fit_lstsq(X[lab == j], Y[lab == j], RIDGE)
                if (lab == j).sum() >= MIN_COUNT else K_glob
                for j in range(M)
            ])
            rhos.append(max(float(np.abs(np.linalg.eigvals(k)).max()) for k in Ks))
            ades.append(rollout(test_scenes, OBS, lambda x: Ks[assign(x)])[0])
            if M == 1:
                break
        a = np.array(ades)
        results["sweep"].append({"M": M, "ades": a.tolist(),
                                 "mean": float(a.mean()), "std": float(a.std()),
                                 "max_rho": max(rhos)})
        print(f"  {M:3d} {a.mean():9.4f} {a.std():7.4f} {a.min():7.4f} "
              f"{a.max():7.4f} {100*(ade_cv-a.mean())/ade_cv:+7.1f}% {max(rhos):8.3f}")

    (OUT / "05_seed_stability.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / '05_seed_stability.json'}")


if __name__ == "__main__":
    main()
