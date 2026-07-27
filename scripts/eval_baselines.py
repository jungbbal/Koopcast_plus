#!/usr/bin/env python3
"""
Compare KoopCast++ against the vendored CANVAS baselines on the ETH/UCY
leave-one-out test splits, all through kpp's neighbour-aware ``evaluate_scene``
so every model sees an identical set of (agent, t0) targets.

Usage:
    python scripts/eval_baselines.py                    # all scenes, working baselines
    python scripts/eval_baselines.py zara1 eth          # selected scenes
    python scripts/eval_baselines.py --models stgcnn socialvae --scenes zara1
"""
import argparse
import sys
import pathlib
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kpp.data import load_ethucy, ETHUCY_SCENES
from kpp.eval import evaluate_scene
from kpp.predictors import ConstantVelocity
from kpp.baselines import make_baseline, WORKING

# ours + the working vendored baselines
DEFAULT_MODELS = ["cv", *WORKING]


def build(name, scene):
    if name == "cv":
        return ConstantVelocity(pred_len=12)
    if name == "koopcastpp":
        from kpp.predictors import KoopCastPP
        return KoopCastPP(scene, pred_len=12)
    return make_baseline(name, scene, pred_len=12, history_len=8, dt=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", help="scenes (default: all ETH/UCY)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--scenes", dest="scenes_opt", nargs="+", default=None)
    args = ap.parse_args()

    scenes = args.scenes_opt or args.scenes or list(ETHUCY_SCENES)
    scenes = [s.strip().lower() for s in scenes]

    print(f"scenes : {scenes}")
    print(f"models : {args.models}\n")
    header = f"{'scene':<8}{'model':<18}{'ADE':>9}{'FDE':>9}{'n':>8}"
    print(header); print("-" * len(header))

    agg = {m: [] for m in args.models}
    for scene in scenes:
        ds = load_ethucy(scene, "test")
        for m in args.models:
            try:
                r = evaluate_scene(build(m, scene), ds)
                if r is None:
                    print(f"{scene:<8}{m:<18}{'--':>9}{'--':>9}{'0':>8}")
                    continue
                print(f"{scene:<8}{m:<18}{r.ade_mean:>9.4f}{r.fde_mean:>9.4f}{r.n_samples:>8}")
                agg[m].append((r.ade_mean, r.fde_mean))
            except Exception as e:
                print(f"{scene:<8}{m:<18}  ERROR: {type(e).__name__}: {e}")
        print("-" * len(header))

    print("\nAVG over scenes")
    print("-" * len(header))
    for m in args.models:
        if agg[m]:
            import numpy as np
            a = np.mean([x[0] for x in agg[m]])
            f = np.mean([x[1] for x in agg[m]])
            print(f"{'AVG':<8}{m:<18}{a:>9.4f}{f:>9.4f}")


if __name__ == "__main__":
    main()
