#!/usr/bin/env python3
"""
KoopCast++ online adaptation: tune eta on validation, report on test.

The adaptation rate ``eta`` is a hyperparameter, so picking it by test score
would be test-set tuning. This script selects eta per scene on the **val**
split and only then scores the **test** split, alongside the baselines.

For reference it also prints the oracle (eta chosen on test) so the gap between
"honestly tuned" and "best possible" is visible -- the oracle row is a diagnostic
upper bound, not a reportable number.

Adaptation only fires when the predictor receives more than ``obs_len`` observed
steps, hence ``full_history=True``; that adds only frames at or before t0, and
the prediction input plus the target set are unchanged (see kpp/baselines/README).

Usage:
    python scripts/eval_adaptive.py                    # ETH/UCY + snu-asri
    python scripts/eval_adaptive.py --scenes zara1
    python scripts/eval_adaptive.py --etas 0 0.01 0.05
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from kpp.data import load_ethucy, load, ETHUCY_SCENES  # noqa: E402
from kpp.data.loaders import load_taa_npy  # noqa: E402
from kpp.eval import evaluate_scene  # noqa: E402
from kpp.predictors import ConstantVelocity, KoopCastPP  # noqa: E402
from kpp.baselines import make_baseline  # noqa: E402

ETAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
BASELINES = ["socialvae", "stgcnn", "eigen"]
SNU_VAL = ROOT / "data" / "raw" / "snu-asri-train" / "val_1.npy"


def splits(scene):
    """(val, test) TrajDatasets for a scene."""
    if scene in ETHUCY_SCENES:
        return load_ethucy(scene, "val"), load_ethucy(scene, "test")
    if scene == "snu-asri":
        return load_taa_npy(SNU_VAL, title="snu-asri/val", fps=2.5), load("snu-asri")
    if scene == "snu-asri-ood":
        # no OOD-specific val; tune on the in-distribution lobby val
        return load_taa_npy(SNU_VAL, title="snu-asri/val", fps=2.5), load("snu-asri-ood")
    raise ValueError(f"unknown scene {scene}")


def kcpp(scene, eta):
    art = "snu-asri" if scene.startswith("snu-asri") else scene
    return KoopCastPP(art, pred_len=12, eta=eta)


def sweep(scene, ds, etas):
    """{eta: ADE} for KoopCast++ on one dataset."""
    out = {}
    for e in etas:
        r = evaluate_scene(kcpp(scene, e), ds, full_history=True)
        out[e] = (r.ade_mean, r.fde_mean) if r else (float("inf"),) * 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=list(ETHUCY_SCENES) + ["snu-asri", "snu-asri-ood"])
    ap.add_argument("--etas", nargs="+", type=float, default=ETAS)
    args = ap.parse_args()

    hdr = (f"{'scene':<14}{'model':<24}{'ADE':>9}{'FDE':>9}{'eta':>7}")
    print(hdr); print("=" * len(hdr))
    rows = {}

    for scene in args.scenes:
        val, test = splits(scene)

        val_sweep = sweep(scene, val, args.etas)
        best_eta = min(val_sweep, key=lambda e: val_sweep[e][0])

        test_sweep = sweep(scene, test, args.etas)
        oracle_eta = min(test_sweep, key=lambda e: test_sweep[e][0])

        static = test_sweep[0.0]
        tuned = test_sweep[best_eta]
        oracle = test_sweep[oracle_eta]

        print(f"{scene:<14}{'KoopCast++ static':<24}{static[0]:>9.4f}{static[1]:>9.4f}{0.0:>7.3f}")
        print(f"{scene:<14}{'KoopCast++ adaptive':<24}{tuned[0]:>9.4f}{tuned[1]:>9.4f}{best_eta:>7.3f}"
              f"   <- eta from val")
        print(f"{scene:<14}{'  (oracle, diagnostic)':<24}{oracle[0]:>9.4f}{oracle[1]:>9.4f}{oracle_eta:>7.3f}")

        r = evaluate_scene(ConstantVelocity(pred_len=12), test)
        print(f"{scene:<14}{'ConstantVelocity':<24}{r.ade_mean:>9.4f}{r.fde_mean:>9.4f}{'-':>7}")
        base = {"cv": (r.ade_mean, r.fde_mean)}
        for b in BASELINES:
            try:
                rb = evaluate_scene(make_baseline(b, scene, pred_len=12, history_len=8, dt=0.4), test)
                print(f"{scene:<14}{b:<24}{rb.ade_mean:>9.4f}{rb.fde_mean:>9.4f}{'-':>7}")
                base[b] = (rb.ade_mean, rb.fde_mean)
            except Exception as e:
                print(f"{scene:<14}{b:<24}  ERROR {type(e).__name__}: {str(e)[:30]}")
        rows[scene] = dict(static=static, tuned=tuned, oracle=oracle,
                           best_eta=best_eta, **base)
        print("-" * len(hdr), flush=True)

    # ---- summary -------------------------------------------------------- #
    print("\nSUMMARY  (adaptive eta tuned on val, scored on test)")
    h2 = f"{'scene':<14}{'static':>9}{'adaptive':>10}{'gain':>8}{'eta':>7}{'best baseline':>18}"
    print(h2); print("=" * len(h2))
    for sc, r in rows.items():
        gain = 100 * (r["tuned"][0] - r["static"][0]) / r["static"][0]
        cands = {k: v[0] for k, v in r.items()
                 if k in ("cv", *BASELINES) and isinstance(v, tuple)}
        bname = min(cands, key=cands.get) if cands else "-"
        print(f"{sc:<14}{r['static'][0]:>9.4f}{r['tuned'][0]:>10.4f}{gain:>+7.1f}%"
              f"{r['best_eta']:>7.3f}{bname + ' ' + format(cands[bname], '.4f'):>18}")


if __name__ == "__main__":
    main()
