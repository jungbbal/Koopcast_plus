"""Sanity pass: what is actually in the box, and what does trivial get?

Run: python exp/00_sanity.py
Everything after this should be measured against the constant-velocity number
printed here -- it is the honest floor for "did my idea do anything".
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

import zlab


def describe(split):
    scenes = zlab.load(split)
    print(f"\n=== split: {split} ===")
    tot_rows = tot_agents = 0
    for sc in scenes:
        zlab.assert_contiguous(sc)
        tr = sc.tracks()
        lens = np.array([len(t) for t in tr.values()])
        print(f"  {sc.name:24s} rows={len(sc.xy):6d}  agents={sc.n_agents:5d}  "
              f"track_len med={np.median(lens):5.1f} max={lens.max():4d}  "
              f"x=[{sc.xy[:,0].min():6.2f},{sc.xy[:,0].max():6.2f}] "
              f"y=[{sc.xy[:,1].min():6.2f},{sc.xy[:,1].max():6.2f}]")
        tot_rows += len(sc.xy)
        tot_agents += sc.n_agents
    w = zlab.windows(scenes)
    print(f"  -> {tot_rows} rows, {tot_agents} agents, {len(w)} windows (8/12, stride 1)")
    return scenes, w


def const_velocity(obs):
    """Extrapolate the last observed velocity. The baseline that refuses to die."""
    v = obs[:, -1] - obs[:, -2]                      # (W, 2)
    steps = np.arange(1, zlab.PRED_LEN + 1)[None, :, None]
    return obs[:, -1][:, None, :] + v[:, None, :] * steps


def main():
    train_scenes, train_w = describe("train")
    describe("val")
    test_scenes, test_w = describe("test")

    print("\n=== constant-velocity baseline (obs 8 -> pred 12) ===")
    for name, w in [("train split", train_w), ("zara2 scene (test)", test_w)]:
        obs, gt = zlab.split_obs_pred(w)
        pr = const_velocity(obs)
        print(f"  {name:20s} ADE={zlab.ade(pr, gt).mean():.3f}  "
              f"FDE={zlab.fde(pr, gt).mean():.3f}   (n={len(w)})")

    print("\n=== speed distribution, m/s ===")
    for name, scenes in [("train split", train_scenes), ("zara2 scene", test_scenes)]:
        sp = np.concatenate([
            np.linalg.norm(np.diff(t, axis=0), axis=-1).ravel() / zlab.DT
            for sc in scenes for t in sc.tracks(min_len=2).values()
        ])
        q = np.percentile(sp, [5, 25, 50, 75, 95])
        print(f"  {name:14s} p5={q[0]:.2f} p25={q[1]:.2f} med={q[2]:.2f} "
              f"p75={q[3]:.2f} p95={q[4]:.2f}  frac_static(<0.1)={np.mean(sp<0.1):.3f}")


if __name__ == "__main__":
    main()
