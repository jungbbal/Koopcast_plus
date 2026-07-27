"""
Windowing -- turn canonical trajectories into (observation, prediction) pairs.

OpenTraj's "trajlet" idea specialised to prediction: fixed *frame counts*
(obs_len, pred_len) rather than seconds, because the literature speaks in steps
(standard ETH/UCY protocol = 8 observed -> 12 predicted).

Consumes only a ``TrajDataset`` and returns plain numpy: no disk round-trips,
no per-dataset special cases.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .trajdataset import TrajDataset


@dataclass
class AgentWindow:
    scene_id: int
    agent_id: int
    start_frame: int
    obs: np.ndarray     # (obs_len, 2) positions
    pred: np.ndarray    # (pred_len, 2) ground-truth future positions

    @property
    def full(self) -> np.ndarray:
        return np.concatenate([self.obs, self.pred], axis=0)


def agent_windows(ds: TrajDataset, *, obs_len: int = 8, pred_len: int = 12,
                  stride: int = 1, require_contiguous: bool = True) -> List[AgentWindow]:
    """Slide a fixed (obs_len + pred_len) window over each agent's trajectory.

    require_contiguous: keep only windows whose frame_ids are evenly spaced
    (no gaps from missing detections) -- the safe default for ADE/FDE.
    """
    win = obs_len + pred_len
    out: List[AgentWindow] = []
    for (scene_id, agent_id), tr in ds.get_trajectories():
        tr = tr.sort_values("frame_id")
        frames = tr["frame_id"].to_numpy()
        xy = tr[["pos_x", "pos_y"]].to_numpy()
        n = len(tr)
        if n < win:
            continue
        step = int(np.median(np.diff(frames))) if n > 1 else 1
        for s in range(0, n - win + 1, stride):
            sl = slice(s, s + win)
            if require_contiguous and not np.all(np.diff(frames[sl]) == step):
                continue
            out.append(AgentWindow(
                scene_id=int(scene_id), agent_id=int(agent_id),
                start_frame=int(frames[s]),
                obs=xy[s:s + obs_len].copy(),
                pred=xy[s + obs_len:s + win].copy(),
            ))
    return out


def stack_windows(windows: List[AgentWindow]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack windows into ``(obs, pred)`` arrays: (N,obs_len,2), (N,pred_len,2)."""
    if not windows:
        return np.empty((0, 0, 2)), np.empty((0, 0, 2))
    obs = np.stack([w.obs for w in windows])
    pred = np.stack([w.pred for w in windows])
    return obs, pred


# --------------------------------------------------------------------------- #
# Scene-level windows (neighbor-aware) -- the prerequisite for social models.
# --------------------------------------------------------------------------- #
@dataclass
class SceneWindow:
    """All agents observed up to instant ``t0`` in one scene, plus the subset
    that has a full future to score.

    obs:     {agent_id: (h, 2)} histories ending at t0, for every agent present
             at t0 -- this is the social context. h == obs_len by default;
             with full_history=True it extends backwards as far as the frames
             stay contiguous (all still at or before t0).
    targets: AgentWindows (full obs_len + pred_len) to be scored.
    """
    scene_id: int
    t0: int
    obs: Dict[int, np.ndarray] = field(default_factory=dict)
    targets: List[AgentWindow] = field(default_factory=list)


def scene_windows(ds: TrajDataset, *, obs_len: int = 8, pred_len: int = 12,
                  stride: int = 1, require_contiguous: bool = True,
                  full_history: bool = False) -> List[SceneWindow]:
    """Group the dataset into per-(scene, t0) windows carrying neighbor context.

    Targets use the SAME contiguity rule as ``agent_windows`` so that marginal
    and social predictors are scored on an identical set of (agent, t0) pairs.
    """
    win = obs_len + pred_len
    out: List[SceneWindow] = []

    for scene_id, sdf in ds.data.groupby("scene_id"):
        agents = {}
        for aid, tr in sdf.groupby("agent_id"):
            tr = tr.sort_values("frame_id")
            agents[int(aid)] = (tr["frame_id"].to_numpy(), tr[["pos_x", "pos_y"]].to_numpy())

        frames_all = np.sort(sdf["frame_id"].unique())
        step = int(np.median(np.diff(frames_all))) if len(frames_all) > 1 else 1

        # frame -> position lookup per agent
        pos_at = {aid: {int(f): xy for f, xy in zip(fr, XY)} for aid, (fr, XY) in agents.items()}

        # enumerate target windows, bucketed by their last observed frame t0
        targets_by_t0: Dict[int, List[AgentWindow]] = defaultdict(list)
        for aid, (fr, XY) in agents.items():
            n = len(fr)
            if n < win:
                continue
            for s in range(0, n - win + 1, stride):
                chunk = fr[s:s + win]
                if require_contiguous and not np.all(np.diff(chunk) == step):
                    continue
                t0 = int(fr[s + obs_len - 1])
                targets_by_t0[t0].append(AgentWindow(
                    scene_id=int(scene_id), agent_id=int(aid), start_frame=int(fr[s]),
                    obs=XY[s:s + obs_len].copy(), pred=XY[s + obs_len:s + win].copy(),
                ))

        for t0, targets in targets_by_t0.items():
            obs = {}
            for aid, pmap in pos_at.items():
                # match CANVAS get_scene: only agents with FULL contiguous
                # obs_len history ending at t0 are part of the social context.
                wanted = [t0 - k * step for k in range(obs_len - 1, -1, -1)]
                if not all(f in pmap for f in wanted):
                    continue
                if full_history:
                    # extend backwards past obs_len while frames stay contiguous.
                    # Everything here is at or before t0, so it is legitimately
                    # observed -- predictors that adapt online (KoopCast++ with
                    # eta > 0) can use it, while predictors that only read the
                    # last obs_len steps are unaffected.
                    k = obs_len
                    while (t0 - k * step) in pmap:
                        wanted.insert(0, t0 - k * step)
                        k += 1
                obs[aid] = np.asarray([pmap[f] for f in wanted], dtype=float)
            out.append(SceneWindow(scene_id=int(scene_id), t0=t0, obs=obs, targets=targets))

    return out
