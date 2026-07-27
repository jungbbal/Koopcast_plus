#!/usr/bin/env python3
"""
End-to-end sanity check for the self-contained core.

    raw -> TrajDataset -> windows -> ConstantVelocity -> ADE/FDE

Run from the repo root:  python scripts/smoke.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kpp.data import load, list_datasets
from kpp.predictors import ConstantVelocity
from kpp.eval import evaluate


def main() -> None:
    print("registered datasets:", list_datasets(), "\n")
    cv = ConstantVelocity(pred_len=12)
    for key in list_datasets():
        try:
            ds = load(key)
        except FileNotFoundError as e:
            print(f"  {key:<14} SKIP ({e})")
            continue
        res = evaluate(cv, ds, obs_len=8, pred_len=12)
        if res is None:
            print(f"  {key:<14} no valid 8->12 windows")
            continue
        print(" ", res)


if __name__ == "__main__":
    main()
