"""
Constant-velocity baseline.

The simplest honest predictor: estimate velocity from the last step(s) of the
observation and extrapolate linearly. Exists mainly to prove the
load -> window -> predict -> ADE/FDE pipeline end to end; drop-in target for
swapping in real models that implement the same ``Predictor`` interface.
"""
from __future__ import annotations

import numpy as np

from .base import Predictor


class ConstantVelocity(Predictor):
    def __init__(self, pred_len: int = 12, n_steps: int = 1) -> None:
        super().__init__(pred_len=pred_len)
        self.n_steps = n_steps  # how many trailing steps to average velocity over

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        last = obs[:, -1, :]                                   # (N, 2)
        k = min(self.n_steps, obs.shape[1] - 1)
        vel = (obs[:, -1, :] - obs[:, -1 - k, :]) / k         # (N, 2) per-step
        steps = np.arange(1, self.pred_len + 1)[None, :, None]  # (1, P, 1)
        return last[:, None, :] + steps * vel[:, None, :]      # (N, P, 2)
