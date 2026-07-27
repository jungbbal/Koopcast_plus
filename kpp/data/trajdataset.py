"""
Canonical trajectory representation -- the *narrow waist* of the data layer.

Every loader, no matter how ugly the raw format, produces one of these.
Everything downstream consumes only this and never needs to know which dataset
the data came from.

Canonical table (``self.data`` is a pandas DataFrame)::

    scene_id  frame_id  agent_id  pos_x  pos_y  vel_x  vel_y  timestamp  label

Cleaned/modernised port of OpenTraj's ``TrajDataset`` (MIT, J. Amirian): same
schema and semantics, pandas-2 friendly, trimmed to what this project needs.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

CRITICAL_COLUMNS = ["frame_id", "agent_id", "pos_x", "pos_y"]


class TrajDataset:
    def __init__(self) -> None:
        self.data = pd.DataFrame(columns=CRITICAL_COLUMNS)
        self.title: str = ""
        self.fps: float = -1.0

    # ------------------------------------------------------------------ #
    def postprocess(self, fps: float, sampling_rate: int = 1,
                    use_kalman: bool = False) -> "TrajDataset":
        """Normalise types, (optionally) downsample, fill timestamps, derive
        velocities, drop stubs.

        Shared work that turns a raw-but-tabular DataFrame into the canonical
        form every dataset shares. Velocities are always re-derived from
        positions so they are consistent across data sources.

        sampling_rate > 1 keeps every Nth frame (``frame_id % sampling_rate == 0``),
        used to bring high-rate sources down to the ~2.5 Hz prediction protocol.
        use_kalman is accepted for loader-call compatibility but is a no-op here
        (positions are kept raw; velocities are finite-differenced).
        """
        for col in CRITICAL_COLUMNS:
            if col not in self.data:
                raise ValueError(f"TrajDataset missing critical column '{col}'")
        if fps <= 0:
            raise ValueError(f"fps must be > 0 (got {fps})")

        df = self.data
        df["frame_id"] = df["frame_id"].astype(int)
        if len(df) and str(df["agent_id"].iloc[0]).replace(".", "", 1).isdigit():
            df["agent_id"] = df["agent_id"].astype(int)
        df["pos_x"] = df["pos_x"].astype(float)
        df["pos_y"] = df["pos_y"].astype(float)

        if "label" not in df:
            df["label"] = "pedestrian"
        df["label"] = df["label"].astype(str).str.lower()
        if "scene_id" not in df:
            df["scene_id"] = 0

        if sampling_rate and sampling_rate >= 2:
            df = df[(df["frame_id"] % int(sampling_rate)) == 0].reset_index(drop=True)

        self.fps = float(fps)
        df["timestamp"] = df["frame_id"] / self.fps

        df = df.sort_values(["scene_id", "agent_id", "frame_id"]).reset_index(drop=True)

        # drop single-sample trajectories (no velocity definable)
        sizes = df.groupby(["scene_id", "agent_id"])["frame_id"].transform("size")
        df = df[sizes >= 2].reset_index(drop=True)

        # derive velocities per agent (finite difference in time)
        g = df.groupby(["scene_id", "agent_id"])
        dt = g["timestamp"].diff()
        df["vel_x"] = g["pos_x"].diff() / dt
        df["vel_y"] = g["pos_y"].diff() / dt
        df["vel_x"] = df.groupby(["scene_id", "agent_id"])["vel_x"].transform(lambda s: s.bfill())
        df["vel_y"] = df.groupby(["scene_id", "agent_id"])["vel_y"].transform(lambda s: s.bfill())

        self.data = df
        return self

    # ------------------------------------------------------------------ #
    def get_agent_ids(self) -> np.ndarray:
        return pd.unique(self.data["agent_id"])

    def get_trajectories(self, label: str = ""):
        """Group the table into per-agent trajectories."""
        df = self.data
        if label:
            df = df[df["label"] == label.lower()]
        return df.groupby(["scene_id", "agent_id"])

    def get_frames(self, scene_id: int = 0) -> List[pd.DataFrame]:
        """Return per-frame slices (all agents present in each frame)."""
        df = self.data[self.data["scene_id"] == scene_id]
        return [g for _, g in df.groupby("frame_id")]

    def bbox(self) -> dict:
        xy = self.data[["pos_x", "pos_y"]].to_numpy()
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        return {"xmin": float(xy[:, 0].min()), "xmax": float(xy[:, 0].max()),
                "ymin": float(xy[:, 1].min()), "ymax": float(xy[:, 1].max())}

    def __repr__(self) -> str:
        n_agents = self.data["agent_id"].nunique() if len(self.data) else 0
        n_frames = self.data["frame_id"].nunique() if len(self.data) else 0
        return (f"TrajDataset(title='{self.title}', fps={self.fps}, "
                f"rows={len(self.data)}, agents={n_agents}, frames={n_frames})")


def from_long_dataframe(df: pd.DataFrame, *, title: str, fps: float,
                        sampling_rate: int = 1) -> TrajDataset:
    """Build a TrajDataset from a long-form DataFrame and run postprocess."""
    ds = TrajDataset()
    ds.title = title
    ds.data = df.copy()
    return ds.postprocess(fps=fps, sampling_rate=sampling_rate)
