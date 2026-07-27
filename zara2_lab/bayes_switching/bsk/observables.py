"""Observable (lifting) functions and closed-loop rollout for a general psi.

Every observable here is built so that psi_t is computable from positions up to
time t alone -- nothing from the future enters, so a rollout that iterates
psi <- K psi is a legitimate closed-loop prediction.

The rollout is deliberately the pure Koopman one: psi is propagated *linearly*
and positions are read off its first two entries. Components are never
recomputed from the predicted state mid-rollout. Recomputing them (an EDMD-style
closure) would be a different, easier model and would not test the claim that a
linear map on this lift explains the motion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observable:
    """psi_t = [p_t, v_t, v_{t-1}, ..., v_{t-L+1}, (nonlinear), 1]."""

    n_delay: int = 1          # how many velocity lags, incl. v_t itself
    nonlin: bool = False      # append ||v_t||, ||v_t||^2, and v_t * ||v_t||
    name: str = ""

    @property
    def need(self) -> int:
        """Earliest index t at which psi_t is defined."""
        return self.n_delay

    @property
    def dim(self) -> int:
        return 2 + 2 * self.n_delay + (5 if self.nonlin else 0) + 1

    def lift(self, traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T,2) positions -> (psi (n,d), t_index (n,)) for every valid t."""
        v = np.diff(traj, axis=0)                       # v[i] = p_{i+1} - p_i
        T = len(traj)
        ts = np.arange(self.need, T)                    # v_t = v[t-1] exists
        if len(ts) == 0:
            return np.empty((0, self.dim)), ts
        cols = [traj[ts]]
        for lag in range(self.n_delay):
            cols.append(v[ts - 1 - lag])
        if self.nonlin:
            vt = v[ts - 1]
            sp = np.linalg.norm(vt, axis=1, keepdims=True)
            cols += [sp, sp ** 2, vt * sp]
        cols.append(np.ones((len(ts), 1)))
        return np.concatenate(cols, axis=1), ts


OBSERVABLES = [
    Observable(1, False, "base [p,v,1]"),
    Observable(2, False, "delay-2"),
    Observable(3, False, "delay-3"),
    Observable(4, False, "delay-4"),
    Observable(5, False, "delay-5"),
    Observable(6, False, "delay-6"),
    Observable(3, True, "delay-3 +nonlin"),
    Observable(5, True, "delay-5 +nonlin"),
]


def pairs(scenes, obs: Observable, min_len: int = 4):
    """Consecutive (psi_t, psi_{t+1}) pairs, never crossing a track boundary."""
    Xs, Ys = [], []
    for sc in scenes:
        for traj in sc.tracks(min_len=max(min_len, obs.need + 2)).values():
            psi, ts = obs.lift(traj)
            if len(psi) < 2:
                continue
            # ts are consecutive by construction, so psi[i+1] is psi_{t+1}
            Xs.append(psi[:-1])
            Ys.append(psi[1:])
    if not Xs:
        return np.empty((0, obs.dim)), np.empty((0, obs.dim))
    return np.concatenate(Xs), np.concatenate(Ys)


def rollout(scenes, obs: Observable, K_of, n_obs=8, n_pred=12, stride=1,
            clip=1e3):
    """Closed-loop ADE/FDE in metres.

    ``K_of(psi)`` returns the operator to apply to this state -- a constant
    function for a global K, a token/mixture lookup for switching. Diverging
    rollouts (an unstable K) are clipped so one blow-up cannot swamp the mean;
    the number of clipped windows is returned and must be reported.
    """
    ades, fdes, n_clip = [], [], 0
    L = n_obs + n_pred
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1, stride):
                w = traj[s:s + L]
                psi, ts = obs.lift(w[:n_obs])
                if len(psi) == 0:
                    continue
                x = psi[-1]                              # psi at last observed step
                pred = []
                for _ in range(n_pred):
                    x = K_of(x) @ x
                    pred.append(x[:2].copy())
                pred = np.asarray(pred)
                if not np.isfinite(pred).all() or np.abs(pred).max() > clip:
                    n_clip += 1
                    pred = np.clip(np.nan_to_num(pred, nan=0.0), -clip, clip)
                d = np.linalg.norm(pred - w[n_obs:], axis=-1)
                ades.append(d.mean())
                fdes.append(d[-1])
    return float(np.mean(ades)), float(np.mean(fdes)), n_clip, len(ades)


def const_vel(scenes, n_obs=8, n_pred=12, stride=1):
    """The baseline to beat."""
    ades, fdes = [], []
    for sc in scenes:
        for traj in sc.tracks(min_len=n_obs + n_pred).values():
            for s in range(0, len(traj) - (n_obs + n_pred) + 1, stride):
                w = traj[s:s + n_obs + n_pred]
                v = w[n_obs - 1] - w[n_obs - 2]
                pred = w[n_obs - 1] + v * np.arange(1, n_pred + 1)[:, None]
                d = np.linalg.norm(pred - w[n_obs:], axis=-1)
                ades.append(d.mean())
                fdes.append(d[-1])
    return float(np.mean(ades)), float(np.mean(fdes))
