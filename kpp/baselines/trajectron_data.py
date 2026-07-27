"""
ETH/UCY  ->  Trajectron++ ``Environment``  converter.

CANVAS vendored Trajectron++'s model but not its data pipeline, so the raw-text
-> ``Environment`` step (upstream ``experiments/pedestrians/process_data.py``)
is reimplemented here. This is *our* glue: nothing under ``vendor/`` is touched.

It is the prerequisite for both
  * retraining  (``trajectron/train.py`` loads dill-pickled Environments), and
  * evaluation  (a scene carrying future ground truth satisfies the
    ``get_timesteps_data(min_future_timesteps=max_ft)`` guard that blocks the
    history-only live path -- see README).

Conventions follow upstream Trajectron++ for ETH/UCY so the shipped pretrained
weights stay compatible:
  * ``frame_id // 10`` -> integer timesteps, rebased to 0,
  * positions mean-centred per scene (the offset is added back at predict time),
  * velocity/acceleration from ``derivative_of`` at dt = 0.4 s,
  * the canonical position/velocity/acceleration standardization.
"""
from __future__ import annotations

import pathlib
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

_VENDOR = pathlib.Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from canvas.predictors.trajectron.environment import (  # noqa: E402
    Environment, Scene, Node, derivative_of,
)

#: identical to the block in vendor/.../trajectron_predictor.py, so Environments
#: built here match what the pretrained checkpoints expect.
STANDARDIZATION = {
    "PEDESTRIAN": {
        "position":     {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity":     {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}

DT = 0.4                  # 2.5 Hz prediction protocol
FRAME_STEP = 10           # ETH/UCY annotate every 10th video frame
ATTENTION_RADIUS = 3.0

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ETHUCY_ROOT = PROJECT_ROOT / "data" / "ethucy"

_COLS = pd.MultiIndex.from_product(
    [["position", "velocity", "acceleration"], ["x", "y"]])


def _scene_from_txt(path: pathlib.Path, node_type, dt: float = DT) -> Optional[Scene]:
    """One ``frame_id track_id x y`` file -> one ``Scene``."""
    data = pd.read_csv(path, sep=r"\s+", header=None,
                       names=["frame_id", "track_id", "pos_x", "pos_y"])
    # the official Social-STGCNN splits carry one truncated trailing record in
    # zara02 / students001 (same quirk kpp's own loaders drop)
    data = data.dropna()
    if data.empty:
        return None

    data["frame_id"] = (data["frame_id"].astype(float) // FRAME_STEP).astype(int)
    data["frame_id"] -= data["frame_id"].min()
    data["node_id"] = data["track_id"].astype(float).astype(int).astype(str)
    data = data.sort_values("frame_id")

    # mean-centre; the offset is restored when predictions are read back
    mean_x = float(data["pos_x"].mean())
    mean_y = float(data["pos_y"].mean())
    data["pos_x"] -= mean_x
    data["pos_y"] -= mean_y

    scene = Scene(timesteps=int(data["frame_id"].max()) + 1, dt=dt, name=path.stem)
    scene.pos_x_mean, scene.pos_y_mean = mean_x, mean_y

    for node_id in pd.unique(data["node_id"]):
        ndf = data[data["node_id"] == node_id].sort_values("frame_id")
        if len(ndf) < 2:
            continue
        x = ndf["pos_x"].to_numpy(dtype=float)
        y = ndf["pos_y"].to_numpy(dtype=float)
        vx, vy = derivative_of(x, dt), derivative_of(y, dt)
        ax, ay = derivative_of(vx, dt), derivative_of(vy, dt)
        node_data = pd.DataFrame(
            {("position", "x"): x, ("position", "y"): y,
             ("velocity", "x"): vx, ("velocity", "y"): vy,
             ("acceleration", "x"): ax, ("acceleration", "y"): ay},
            columns=_COLS)
        node = Node(node_type=node_type, node_id=node_id, data=node_data,
                    first_timestep=int(ndf["frame_id"].iloc[0]))
        scene.nodes.append(node)

    return scene if scene.nodes else None


def build_environment(txt_paths: List[pathlib.Path], dt: float = DT) -> Environment:
    """Build an ``Environment`` (one Scene per file) from ETH/UCY text files."""
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=STANDARDIZATION)
    ped = env.NodeType.PEDESTRIAN
    env.attention_radius = {(ped, ped): ATTENTION_RADIUS}

    scenes = []
    for p in txt_paths:
        sc = _scene_from_txt(pathlib.Path(p), ped, dt=dt)
        if sc is not None:
            scenes.append(sc)
    env.scenes = scenes
    return env


def load_split(scene: str, phase: str, dt: float = DT) -> Environment:
    """``load_split("zara1", "test")`` -> Environment over that split's files."""
    d = ETHUCY_ROOT / scene.strip().lower() / phase.strip().lower()
    if not d.is_dir():
        raise FileNotFoundError(f"no such split: {d}")
    paths = sorted(d.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt files under {d}")
    return build_environment(paths, dt=dt)


def dump_split(scene: str, phase: str, out_path: pathlib.Path, dt: float = DT) -> pathlib.Path:
    """Pickle a split as ``train.py`` expects (``dill``)."""
    import dill
    env = load_split(scene, phase, dt=dt)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        dill.dump(env, f, protocol=4)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="dump ETH/UCY splits as Trajectron++ Environments")
    ap.add_argument("--scenes", nargs="+",
                    default=["eth", "hotel", "univ", "zara1", "zara2"])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "trajectron"))
    args = ap.parse_args()

    out_root = pathlib.Path(args.out)
    for sc in args.scenes:
        for phase in ("train", "val", "test"):
            p = dump_split(sc, phase, out_root / sc / f"{sc}_{phase}.pkl")
            env = load_split(sc, phase)
            n_nodes = sum(len(s.nodes) for s in env.scenes)
            print(f"{sc:<7}{phase:<6} scenes={len(env.scenes):<3} nodes={n_nodes:<6} -> {p}")
