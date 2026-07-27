#!/usr/bin/env python3
"""
Score Trajectron++ on the ETH/UCY leave-one-out test splits.

Compares the CANVAS-shipped pretrained checkpoints against anything retrained by
``scripts/train_trajectron.py``, so the retraining pipeline can be checked for
reproducibility rather than taken on faith.

Trajectron++ does not go through ``kpp.eval.evaluate_scene`` -- see
``kpp/baselines/README.md`` for why (upstream ``get_timesteps_data`` demands
future ground truth). Target counts still match the other baselines exactly.

Usage:
    python scripts/eval_trajectron.py                     # pretrained, all scenes
    python scripts/eval_trajectron.py --retrained         # retrained, all scenes
    python scripts/eval_trajectron.py --both zara1 eth    # side by side
"""
import argparse
import glob
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from kpp.baselines.trajectron_eval import evaluate_trajectron  # noqa: E402

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
RUNS = ROOT / "runs" / "trajectron"


def retrained_dir(scene: str, ts: int = 100):
    """Newest run dir for `scene` that actually holds a `ts`-epoch checkpoint."""
    cands = [d for d in sorted(glob.glob(str(RUNS / scene / "models_*")))
             if (pathlib.Path(d) / f"model_registrar-{ts}.pt").exists()]
    return cands[-1] if cands else None


def score(scene: str, which: str, ts: int = 100):
    if which == "pretrained":
        return evaluate_trajectron(scene, ts=100)
    d = retrained_dir(scene, ts)
    if d is None:
        return None
    return evaluate_trajectron(scene, model_dir=d, ts=ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=None)
    ap.add_argument("--retrained", action="store_true", help="score retrained only")
    ap.add_argument("--both", action="store_true", help="pretrained and retrained")
    ap.add_argument("--ts", type=int, default=100, help="retrained epoch to load")
    args = ap.parse_args()

    scenes = [s.strip().lower() for s in (args.scenes or SCENES)]
    kinds = ["pretrained", "retrained"] if args.both else \
            (["retrained"] if args.retrained else ["pretrained"])

    hdr = f"{'scene':<8}{'variant':<12}{'ADE':>9}{'FDE':>9}{'n':>8}"
    print(hdr); print("-" * len(hdr))
    agg = {k: [] for k in kinds}
    for sc in scenes:
        for k in kinds:
            r = score(sc, k, args.ts)
            if r is None:
                print(f"{sc:<8}{k:<12}{'--':>9}{'--':>9}{'(no ckpt)':>8}")
                continue
            print(f"{sc:<8}{k:<12}{r['ade_mean']:>9.4f}{r['fde_mean']:>9.4f}{r['n_samples']:>8}")
            agg[k].append((r["ade_mean"], r["fde_mean"]))
        if args.both:
            print("-" * len(hdr))

    print("\n" + "-" * len(hdr))
    for k in kinds:
        if agg[k]:
            a = np.mean([x[0] for x in agg[k]])
            f = np.mean([x[1] for x in agg[k]])
            print(f"{'AVG':<8}{k:<12}{a:>9.4f}{f:>9.4f}")


if __name__ == "__main__":
    main()
