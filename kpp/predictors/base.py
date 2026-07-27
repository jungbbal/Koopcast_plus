"""
Predictor interface.

A predictor maps an observed track to a future track. This is the single
contract every model (the constant-velocity baseline, KoopCast++,
anything you train later) implements, so the evaluation/control loops never
special-case a model.

Batch convention (plain numpy, in-memory; no disk, no subprocess):
    predict(obs) where obs is (N, obs_len, 2)  ->  pred (N, pred_len, 2)
"""
from __future__ import annotations

import abc
from typing import Dict

import numpy as np


class Predictor(abc.ABC):
    """Common contract for every model.

    Two entry points:
      * ``predict(obs)``        -- marginal, single-agent batch (N,H,2)->(N,P,2).
      * ``predict_scene(obs)``  -- neighbor-aware; takes ``{id: (h,2)}`` for all
        agents at one instant, returns ``{id: (P,2)}``. The default ignores
        neighbours and just runs ``predict`` per agent, so marginal models work
        on scene windows for free; social models (KoopCast++) override
        it to use the context.
    """
    #: number of future steps this predictor emits
    pred_len: int = 12
    #: whether this predictor actually uses neighbour context
    social: bool = False

    def __init__(self, pred_len: int = 12) -> None:
        self.pred_len = pred_len

    @abc.abstractmethod
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs: (N, obs_len, 2) -> pred: (N, pred_len, 2)."""

    def predict_scene(self, obs: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Default marginal fallback: predict each agent independently."""
        out: Dict[int, np.ndarray] = {}
        for aid, hist in obs.items():
            hist = np.asarray(hist, dtype=float)
            if hist.shape[0] < 2:
                continue
            out[aid] = self.predict(hist[None, ...])[0]
        return out

    @property
    def name(self) -> str:
        return type(self).__name__
