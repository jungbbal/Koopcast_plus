"""
Per-dataset loaders + a registry.

Following OpenTraj: *reading* is necessarily dataset-specific and lives here
(one reader per raw layout), but every loader returns the same canonical
``TrajDataset``. Downstream code never touches raw files again.

Readers currently implemented:
  * ``load_obsmat``   -- BIWI/UCY ``obsmat.txt`` (world metres, 8 columns).
                         Covers eth, hotel, zara1, zara2, univ. No homography
                         needed: these obsmat files are already in world coords.
  * ``load_xyf_txt``  -- simple ``frame agent x y`` whitespace txt (students001).
  * ``load_taa_npy``  -- dense ``(T, A, 2)`` array (custom SNU-ASRI dumps).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .trajdataset import TrajDataset, from_long_dataframe

# koopcast_plusplus/
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def _fps_from_frame_step(frame_ids: np.ndarray) -> float:
    """ETH/UCY are annotated at 2.5 Hz; the integer frame step encodes the video
    rate. fps = step * 2.5 makes dt = step/fps = 0.4 s for every set."""
    uniq = np.unique(frame_ids.astype(int))
    step = int(np.diff(uniq)[0]) if len(uniq) > 1 else 1
    return step * 2.5


# --------------------------------------------------------------------------- #
# Readers (one per raw layout)
# --------------------------------------------------------------------------- #
def load_obsmat(path, *, title: str, fps: Optional[float] = None) -> TrajDataset:
    """Read a BIWI/UCY ``obsmat.txt`` (8 cols, world metres).

    Columns: frame_id, agent_id, pos_x, pos_z, pos_y, vel_x, vel_z, vel_y.
    Only positions are kept; velocities are re-derived in postprocess.
    """
    raw = np.loadtxt(path)
    df = pd.DataFrame({
        "frame_id": raw[:, 0].astype(int),
        "agent_id": raw[:, 1].astype(int),
        "pos_x": raw[:, 2],
        "pos_y": raw[:, 4],
        "scene_id": 0,
        "label": "pedestrian",
    })
    if fps is None:
        fps = _fps_from_frame_step(df["frame_id"].to_numpy())
    return from_long_dataframe(df, title=title, fps=fps)


def load_xyf_txt(path, *, title: str, fps: float = 25.0) -> TrajDataset:
    """Read a simple ``frame_id agent_id pos_x pos_y`` whitespace txt."""
    raw = np.loadtxt(path)
    df = pd.DataFrame({
        "frame_id": raw[:, 0].astype(int),
        "agent_id": raw[:, 1].astype(int),
        "pos_x": raw[:, 2],
        "pos_y": raw[:, 3],
        "scene_id": 0,
        "label": "pedestrian",
    })
    return from_long_dataframe(df, title=title, fps=fps)


def load_taa_npy(path, *, title: str, fps: float = 2.5) -> TrajDataset:
    """Read a dense ``(T, A, 2)`` array (frame x agent x xy, NaN where absent)."""
    arr = np.load(path, allow_pickle=True).astype(float)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"{path}: expected (T, A, 2), got {arr.shape}")
    finite = np.all(np.isfinite(arr), axis=-1)
    frame_idx, agent_idx = np.nonzero(finite)
    xy = arr[frame_idx, agent_idx]
    df = pd.DataFrame({
        "frame_id": frame_idx.astype(int),
        "agent_id": agent_idx.astype(int),
        "pos_x": xy[:, 0],
        "pos_y": xy[:, 1],
        "scene_id": 0,
        "label": "pedestrian",
    })
    return from_long_dataframe(df, title=title, fps=fps)


ETHUCY_DIR = PROJECT_ROOT / "data" / "ethucy"
ETHUCY_SCENES = ("eth", "hotel", "univ", "zara1", "zara2")
ETHUCY_SPLITS = ("train", "val", "test")


def load_ethucy(scene: str, split: str = "train", *, fps: float = 25.0) -> TrajDataset:
    """Load the official Social-STGCNN / Social-GAN ETH-UCY leave-one-out split.

    These are the canonical ``frame_id  agent_id  pos_x  pos_y`` (tab/space) txt
    files everyone benchmarks on -- world metres, already at the 2.5 Hz protocol
    (frame step 10, fps 25 -> dt 0.4 s). For ``scene`` (the held-out test scene)
    the ``train``/``val`` dirs hold *other* scenes' splits and ``test`` holds the
    held-out scene itself.

    Each source ``.txt`` becomes its own ``scene_id`` so neighbour context never
    bleeds across physically distinct scenes. Malformed/truncated lines (the
    upstream files have one cut-off trailing record in zara02 and students001)
    are skipped.
    """
    scene = scene.strip().lower()
    split = split.strip().lower()
    if scene not in ETHUCY_SCENES:
        raise KeyError(f"unknown ETH-UCY scene '{scene}'; pick from {ETHUCY_SCENES}")
    if split not in ETHUCY_SPLITS:
        raise KeyError(f"unknown split '{split}'; pick from {ETHUCY_SPLITS}")
    d = ETHUCY_DIR / scene / split
    files = sorted(d.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no ETH-UCY txt files under {d}")
    parts = []
    for sid, f in enumerate(files):
        # robust read: keep only lines with exactly 4 parseable floats
        rows = []
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) != 4:
                continue
            try:
                rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                continue
        raw = np.asarray(rows, dtype=float)
        parts.append(pd.DataFrame({
            "frame_id": raw[:, 0].astype(int),
            "agent_id": raw[:, 1].astype(int),
            "pos_x": raw[:, 2],
            "pos_y": raw[:, 3],
            "scene_id": sid,
            "label": "pedestrian",
        }))
    df = pd.concat(parts, ignore_index=True)
    return from_long_dataframe(df, title=f"ethucy/{scene}/{split}", fps=fps)


from .loaders_opentraj import (load_gcs, load_town_center, load_edinburgh,
                               load_pets, load_wildtrack)

_READERS = {
    "obsmat": load_obsmat,
    "xyf_txt": load_xyf_txt,
    "taa_npy": load_taa_npy,
    "gcs": load_gcs,
    "town": load_town_center,
    "edinburgh": load_edinburgh,
    "pets": load_pets,
    "wildtrack": load_wildtrack,
}


# --------------------------------------------------------------------------- #
# Registry: dataset key -> raw path + reader (+ reader params)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    key: str
    relpath: str                       # file OR directory, relative to DATA_ROOT
    reader: str
    params: Dict[str, object] = field(default_factory=dict)  # extra reader kwargs

    @property
    def path(self) -> pathlib.Path:
        return DATA_ROOT / self.relpath


DATASETS: Dict[str, DatasetSpec] = {
    # --- ETH/UCY core: raw obsmat, world coords, fps auto-derived ---
    "eth":          DatasetSpec("eth",          "eth/obsmat.txt",   "obsmat"),
    "hotel":        DatasetSpec("hotel",        "hotel/obsmat.txt", "obsmat"),
    "univ":         DatasetSpec("univ",         "univ/obsmat.txt",  "obsmat"),
    "zara1":        DatasetSpec("zara1",        "zara1/obsmat.txt", "obsmat"),
    "zara2":        DatasetSpec("zara2",        "zara2/obsmat.txt", "obsmat"),
    "students001":  DatasetSpec("students001",  "students001/students001.txt", "xyf_txt", {"fps": 25.0}),
    # --- other OpenTraj pedestrian datasets ---
    "gc":           DatasetSpec("gc",           "gc",         "gcs",       {"fps": 25.0}),
    "town-centre":  DatasetSpec("town-centre",  "town-centre", "town",     {"fps": 25.0, "sampling_rate": 10}),
    "edinburgh":    DatasetSpec("edinburgh",    "edinburgh",  "edinburgh", {"fps": 9.0, "sampling_rate": 4}),
    "pets2009-s2l1": DatasetSpec("pets2009-s2l1", "pets",     "pets",      {"fps": 7.0, "sampling_rate": 2}),
    "wildtrack":    DatasetSpec("wildtrack",    "wildtrack",  "wildtrack", {"fps": 10.0}),
    # --- custom SNU-ASRI ---
    "snu-asri":     DatasetSpec("snu-asri",     "snu-asri/0.npy",            "taa_npy", {"fps": 2.5}),
    "snu-asri-ood": DatasetSpec("snu-asri-ood", "snu-asri/snu-asri-ood.npy", "taa_npy", {"fps": 2.5}),
}


def list_datasets() -> List[str]:
    return list(DATASETS.keys())


def load(key: str, *, fps: Optional[float] = None) -> TrajDataset:
    """Load a registered dataset by key into the canonical TrajDataset."""
    k = key.strip().lower()
    if k not in DATASETS:
        raise KeyError(f"Unknown dataset '{key}'. Known: {list_datasets()}")
    spec = DATASETS[k]
    if not spec.path.exists():
        raise FileNotFoundError(f"Dataset path missing: {spec.path}")
    reader = _READERS[spec.reader]
    kwargs = {"title": spec.key, **spec.params}
    if fps is not None:
        kwargs["fps"] = fps
    return reader(spec.path, **kwargs)
