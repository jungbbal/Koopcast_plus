"""
Prediction metrics + evaluation loop -- all in-process.

ADE = mean over time of L2(pred, gt); FDE = L2 at the final step. Reported as
mean +/- std over windows. No subprocess, no stdout parsing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from ..data import TrajDataset, agent_windows, stack_windows, scene_windows
from ..predictors.base import Predictor


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-window Average Displacement Error. pred/gt: (N, P, 2) -> (N,)."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def fde(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-window Final Displacement Error. pred/gt: (N, P, 2) -> (N,)."""
    return np.linalg.norm(pred[:, -1, :] - gt[:, -1, :], axis=-1)


@dataclass
class EvalResult:
    predictor: str
    dataset: str
    n_samples: int
    ade_mean: float
    ade_std: float
    fde_mean: float
    fde_std: float

    def as_row(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"{self.predictor:<20} {self.dataset:<14} "
                f"ADE={self.ade_mean:.4f}±{self.ade_std:.4f}  "
                f"FDE={self.fde_mean:.4f}±{self.fde_std:.4f}  (n={self.n_samples})")


def evaluate(predictor: Predictor, ds: TrajDataset, *,
             obs_len: int = 8, pred_len: int = 12,
             stride: int = 1) -> Optional[EvalResult]:
    """Run a predictor over every valid window of a dataset and score it."""
    wins = agent_windows(ds, obs_len=obs_len, pred_len=pred_len, stride=stride)
    obs, gt = stack_windows(wins)
    if len(wins) == 0:
        return None
    pred = np.asarray(predictor.predict(obs), dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"predictor returned {pred.shape}, expected {gt.shape}")
    a, f = ade(pred, gt), fde(pred, gt)
    return EvalResult(
        predictor=predictor.name, dataset=ds.title, n_samples=len(wins),
        ade_mean=float(a.mean()), ade_std=float(a.std()),
        fde_mean=float(f.mean()), fde_std=float(f.std()),
    )


def evaluate_scene(predictor: Predictor, ds: TrajDataset, *,
                   obs_len: int = 8, pred_len: int = 12,
                   stride: int = 1, full_history: bool = False) -> Optional[EvalResult]:
    """Neighbor-aware evaluation: one ``predict_scene`` call per (scene, t0),
    scored over the targets. Works for any predictor (marginal models fall back
    to per-agent prediction), so all predictors share the same sample set.

    full_history extends each agent's observed history backwards past obs_len
    (still ending at t0, so nothing from the future leaks). The prediction
    target set is identical either way; it only matters for predictors that
    adapt online from the observed stream -- KoopCast++ with eta > 0 reads the
    extra steps, while every other predictor slices the last obs_len anyway.
    """
    sws = scene_windows(ds, obs_len=obs_len, pred_len=pred_len, stride=stride,
                        full_history=full_history)
    a_list, f_list = [], []
    for sw in sws:
        if not sw.targets:
            continue
        preds = predictor.predict_scene(sw.obs)
        for t in sw.targets:
            p = preds.get(t.agent_id)
            if p is None:
                continue
            p = np.asarray(p, dtype=float)[:pred_len]
            a_list.append(float(np.linalg.norm(p - t.pred, axis=-1).mean()))
            f_list.append(float(np.linalg.norm(p[-1] - t.pred[-1])))
    if not a_list:
        return None
    a, f = np.asarray(a_list), np.asarray(f_list)
    return EvalResult(
        predictor=predictor.name, dataset=ds.title, n_samples=len(a_list),
        ade_mean=float(a.mean()), ade_std=float(a.std()),
        fde_mean=float(f.mean()), fde_std=float(f.std()),
    )
