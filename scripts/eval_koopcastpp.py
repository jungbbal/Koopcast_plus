#!/usr/bin/env python3
"""
Evaluate KoopCast++ on the ETH/UCY leave-one-out test splits and compare it,
sample-for-sample, against the ConstantVelocity baseline.

Both predictors are scored with ``evaluate_scene`` (neighbour-aware windows), so
they see exactly the same set of (agent, t0) targets per scene.

Usage:
    python scripts/eval_koopcastpp.py            # all trained scenes
    python scripts/eval_koopcastpp.py zara1 eth  # selected
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kpp.data import load_ethucy, ETHUCY_SCENES
from kpp.predictors import ConstantVelocity, KoopCastPP
from kpp.predictors.koopcastpp import available_datasets
from kpp.eval import evaluate_scene


def main():
    want = [a.strip().lower() for a in sys.argv[1:]] or list(ETHUCY_SCENES)
    trained = set(available_datasets())
    scenes = [s for s in want if s in trained]
    missing = [s for s in want if s not in trained]
    if missing:
        print(f"(skipping untrained scenes: {missing})")

    cv = ConstantVelocity(pred_len=12)
    rows = []
    print(f"\n{'scene':<8}{'model':<22}{'ADE':>9}{'FDE':>9}{'n':>8}")
    print("-" * 56)
    for sc in scenes:
        ds = load_ethucy(sc, "test")
        kpp = KoopCastPP(sc)
        r_cv = evaluate_scene(cv, ds)
        r_kp = evaluate_scene(kpp, ds)
        rows.append((sc, r_cv, r_kp))
        print(f"{sc:<8}{'ConstantVelocity':<22}{r_cv.ade_mean:>9.4f}{r_cv.fde_mean:>9.4f}{r_cv.n_samples:>8}")
        print(f"{sc:<8}{kpp.name:<22}{r_kp.ade_mean:>9.4f}{r_kp.fde_mean:>9.4f}{r_kp.n_samples:>8}")
        d_ade = 100.0 * (r_kp.ade_mean - r_cv.ade_mean) / r_cv.ade_mean
        d_fde = 100.0 * (r_kp.fde_mean - r_cv.fde_mean) / r_cv.fde_mean
        print(f"{'':8}{'  -> KoopCast++ vs CV':<22}{d_ade:>+8.1f}%{d_fde:>+8.1f}%   (negative = better)")
        print("-" * 56)

    if rows:
        cv_ade = sum(r[1].ade_mean for r in rows) / len(rows)
        cv_fde = sum(r[1].fde_mean for r in rows) / len(rows)
        kp_ade = sum(r[2].ade_mean for r in rows) / len(rows)
        kp_fde = sum(r[2].fde_mean for r in rows) / len(rows)
        print(f"{'AVG':<8}{'ConstantVelocity':<22}{cv_ade:>9.4f}{cv_fde:>9.4f}")
        print(f"{'AVG':<8}{'KoopCast++(static)':<22}{kp_ade:>9.4f}{kp_fde:>9.4f}")


if __name__ == "__main__":
    main()
