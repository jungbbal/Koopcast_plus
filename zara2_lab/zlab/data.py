"""Standalone ETH/UCY loader for the zara2 sandbox.

Reads the repo's ``data/ethucy/zara2`` files directly, but imports nothing from
``kpp`` -- numpy only. The sandbox shares the repo's data (no duplicate copy)
while staying immune to churn in the main package's code.

Coordinates are world metres, sampled at 2.5 Hz (dt = 0.4 s, frame step 10).
The benchmark protocol is obs=8 (3.2 s) -> pred=12 (4.8 s).

Two things are called "zara2" and they are NOT the same set:

  * ``split="train"`` -- the leave-one-out *training* split for the zara2
    benchmark. Contains eth, hotel, zara1, zara3, students, uni_examples.
    It deliberately does NOT contain the zara2 scene.
  * ``split="test"``  -- crowds_zara02.txt, the held-out zara2 scene itself.

Pick with eyes open. ``scene_files()`` lets you take any single source file.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ethucy" / "zara2"
DT = 0.4  # seconds between consecutive samples

OBS_LEN = 8
PRED_LEN = 12


# --------------------------------------------------------------------------- #
# raw reading
# --------------------------------------------------------------------------- #
def read_txt(path) -> np.ndarray:
    """Read one ``frame_id agent_id x y`` file -> (N, 4) float array.

    The upstream distribution has a couple of cut-off trailing records
    (zara02, students001); lines that do not parse as exactly 4 floats are
    skipped, matching what every benchmark implementation does.
    """
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        p = line.split()
        if len(p) != 4:
            continue
        try:
            rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
        except ValueError:
            continue
    return np.asarray(rows, dtype=float)


def scene_files(split: str = "train") -> list[pathlib.Path]:
    """The source .txt files making up a split, sorted."""
    d = DATA / split
    files = sorted(d.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no txt files under {d}")
    return files


# --------------------------------------------------------------------------- #
# canonical containers
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    """One source file: a set of pedestrian tracks sharing a coordinate frame."""

    name: str
    frames: np.ndarray  # (N,) int   frame id
    agents: np.ndarray  # (N,) int   agent id
    xy: np.ndarray      # (N, 2)     world metres

    @property
    def n_agents(self) -> int:
        return len(np.unique(self.agents))

    def tracks(self, min_len: int = 1) -> dict[int, np.ndarray]:
        """agent_id -> (T_i, 2) positions, ordered by frame.

        Tracks are contiguous in ETH/UCY (no gaps within an agent), so a plain
        frame-sort is enough; ``assert_contiguous`` below checks that claim.
        """
        out = {}
        for a in np.unique(self.agents):
            m = self.agents == a
            order = np.argsort(self.frames[m], kind="stable")
            traj = self.xy[m][order]
            if len(traj) >= min_len:
                out[int(a)] = traj
        return out

    def frame_index(self) -> dict[int, np.ndarray]:
        """frame_id -> indices into this scene's rows (for neighbour lookups)."""
        idx = {}
        for f in np.unique(self.frames):
            idx[int(f)] = np.flatnonzero(self.frames == f)
        return idx


def load(split: str = "train") -> list[Scene]:
    """Every scene in a split, each kept separate.

    Keeping them separate matters: two source files have overlapping frame ids
    and unrelated coordinate origins, so pooling rows would invent neighbours
    that never shared a space.
    """
    scenes = []
    for f in scene_files(split):
        raw = read_txt(f)
        scenes.append(Scene(
            name=f.stem,
            frames=raw[:, 0].astype(int),
            agents=raw[:, 1].astype(int),
            xy=raw[:, 2:4],
        ))
    return scenes


def load_zara2_scene() -> Scene:
    """The zara2 scene itself (crowds_zara02) -- the held-out test file."""
    return load("test")[0]


# --------------------------------------------------------------------------- #
# windows: the obs/pred pairs everything is scored on
# --------------------------------------------------------------------------- #
def windows(
    scenes: list[Scene] | Scene,
    obs: int = OBS_LEN,
    pred: int = PRED_LEN,
    stride: int = 1,
) -> np.ndarray:
    """All (obs+pred)-long single-agent windows -> (W, obs+pred, 2).

    Sliding with ``stride=1`` is the benchmark convention and produces heavily
    overlapping windows; use a larger stride when you want roughly independent
    samples for a statistic.
    """
    if isinstance(scenes, Scene):
        scenes = [scenes]
    L = obs + pred
    out = []
    for sc in scenes:
        for traj in sc.tracks(min_len=L).values():
            for s in range(0, len(traj) - L + 1, stride):
                out.append(traj[s:s + L])
    if not out:
        return np.empty((0, L, 2))
    return np.stack(out)


def split_obs_pred(win: np.ndarray, obs: int = OBS_LEN):
    """(W, obs+pred, 2) -> ((W, obs, 2), (W, pred, 2))."""
    return win[:, :obs], win[:, obs:]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-window average displacement error, metres. Shapes (W, T, 2)."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def fde(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-window final displacement error, metres."""
    return np.linalg.norm(pred[:, -1] - gt[:, -1], axis=-1)


# --------------------------------------------------------------------------- #
# sanity checks -- cheap invariants worth asserting before trusting a result
# --------------------------------------------------------------------------- #
def assert_contiguous(sc: Scene) -> None:
    """Every agent's frames are an evenly spaced run (no holes)."""
    for a in np.unique(sc.agents):
        f = np.sort(sc.frames[sc.agents == a])
        if len(f) < 2:
            continue
        d = np.unique(np.diff(f))
        assert len(d) == 1, f"{sc.name}: agent {a} has irregular frame steps {d}"
