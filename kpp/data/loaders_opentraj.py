"""
Heavier per-dataset readers ported from OpenTraj (MIT, J. Amirian).

These cover datasets whose raw layout needs real coordinate work (homography,
camera unprojection, custom parsing) rather than a plain table. Each reader
takes the dataset's raw *directory* and returns a canonical ``TrajDataset``,
so the rest of the pipeline stays dataset-agnostic.

Extra deps beyond numpy/pandas: scipy (interpolation, rotation) and opencv
(undistortion / homography). PETS additionally uses the vendored Tsai
calibration in ``_calib_tsai.py``.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import pathlib
import re
import xml.etree.ElementTree as et

import numpy as np
import pandas as pd

from .trajdataset import from_long_dataframe


def _image_to_world(pts: np.ndarray, H) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) image points -> (N,2) world points."""
    H = np.asarray(H, dtype=float)
    pp = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)      # (N,3)
    PP = (H @ pp.T).T
    return PP[:, :2] / PP[:, 2:3]


# --------------------------------------------------------------------------- #
# Grand Central (GCS)
# --------------------------------------------------------------------------- #
# Each <Annotation>/<id>.txt holds flat triples (py, px, frame). A >20-frame
# gap splits one file into separate tracks. Tracks are linearly interpolated to
# a 10-frame step, then mapped to world coords with a fixed homography (*0.8).
_GC_HOMOG = np.array([
    [4.97412897e-02, -4.24730883e-02, 7.25543911e+01],
    [1.45017874e-01, -3.35678711e-03, 7.97920970e+00],
    [1.36068797e-03, -4.98339188e-05, 1.00000000e+00],
])


def load_gcs(path, *, title: str = "gc", fps: float = 25.0) -> "from_long_dataframe":
    from scipy.interpolate import interp1d
    annot_dir = os.path.join(path, "Annotation")
    rows = []
    agent_inc = 0
    for fname in sorted(os.listdir(annot_dir)):
        with open(os.path.join(annot_dir, fname)) as f:
            vals = f.read().split()
        agent_inc += 1
        last = -1
        for i in range(len(vals) // 3):
            py = float(vals[3 * i]); px = float(vals[3 * i + 1]); fr = int(vals[3 * i + 2])
            if last > 0 and (fr - last) > 20:
                agent_inc += 1
            last = fr
            rows.append([fr, agent_inc, px, py])
    raw = pd.DataFrame(rows, columns=["frame_id", "agent_id", "pos_x", "pos_y"])

    parts = []
    for aid, tr in raw.groupby("agent_id"):
        if len(tr) < 2:
            continue
        f0, f1 = tr["frame_id"].iloc[0], tr["frame_id"].iloc[-1]
        if f1 - f0 < 10:
            continue
        fi = np.arange(f0, f1, 10).astype(int)
        xi = interp1d(tr["frame_id"], tr["pos_x"])(fi)
        yi = interp1d(tr["frame_id"], tr["pos_y"])(fi)
        parts.append(pd.DataFrame({"frame_id": fi, "agent_id": aid, "pos_x": xi, "pos_y": yi}))
    raw = pd.concat(parts, ignore_index=True)
    raw[["pos_x", "pos_y"]] = _image_to_world(raw[["pos_x", "pos_y"]].to_numpy(), _GC_HOMOG) * 0.8
    raw["scene_id"] = 0
    raw["label"] = "pedestrian"
    return from_long_dataframe(raw, title=title, fps=fps)


# --------------------------------------------------------------------------- #
# Oxford Town-Center
# --------------------------------------------------------------------------- #
def _town_unproject(pts: np.ndarray, calib_path: str) -> np.ndarray:
    import cv2
    from scipy.spatial.transform import Rotation
    param = {}
    with open(calib_path) as f:
        for line in f:
            name, _, value = line.split()
            param[name] = float(value)
    rvec = np.array([param["RotationX"], param["RotationY"], param["RotationZ"], param["RotationW"]])
    tvec = np.array([param["TranslationX"], param["TranslationY"], param["TranslationZ"]])
    cameraMatrix = np.array([[param["FocalLengthX"], param["Skew"], param["PrincipalPointX"]],
                            [0.0, param["FocalLengthY"], param["PrincipalPointY"]],
                            [0.0, 0.0, 1.0]])
    distCoeffs = np.array([param["DistortionK1"], param["DistortionK2"],
                          param["DistortionP1"], param["DistortionP2"]])
    n = pts.shape[0]
    und = cv2.undistortPoints(pts.reshape(n, 1, 2), cameraMatrix, distCoeffs).reshape(n, 2)
    und = np.concatenate([und, np.ones((n, 1))], axis=1)
    R = Rotation.from_quat(rvec).inv().as_matrix()
    normal = np.array([[0], [0], [1]])
    tvec = tvec.reshape(3, 1)
    rot = (R @ und.T).T
    num = -normal.T @ tvec
    denom = (normal.T @ rot.T).T
    return num / denom * rot + tvec.T


def load_town_center(path, *, title: str = "town-centre",
                     fps: float = 25.0, sampling_rate: int = 10):
    gt = os.path.join(path, "TownCentre-groundtruth-top.txt")
    calib = os.path.join(path, "TownCentre-calibration-ci.txt")
    cols = ["personNumber", "frameNumber", "headValid", "bodyValid",
            "headLeft", "headTop", "headRight", "headBottom",
            "bodyLeft", "bodyTop", "bodyRight", "bodyBottom"]
    raw = pd.read_csv(gt, sep=",", header=None, names=cols)
    body_x = (raw["bodyLeft"] + raw["bodyRight"]) / 2.0
    body_y = raw["bodyBottom"]
    world = _town_unproject(np.stack([body_x, body_y], axis=1), calib)
    raw["pos_x"], raw["pos_y"] = world[:, 0], world[:, 1]
    raw = raw[raw["bodyValid"] == 1]
    df = pd.DataFrame({
        "frame_id": raw["frameNumber"].astype(int),
        "agent_id": raw["personNumber"].astype(int),
        "pos_x": raw["pos_x"], "pos_y": raw["pos_y"],
        "scene_id": 0, "label": "pedestrian",
    })
    return from_long_dataframe(df, title=title, fps=fps, sampling_rate=sampling_rate)


# --------------------------------------------------------------------------- #
# Edinburgh Informatics Forum
# --------------------------------------------------------------------------- #
def _edinburgh_homog():
    import cv2
    img9 = [[155, 86.6], [350, 95.4], [539, 106], [149, 206], [345, 215],
            [537, 223], [144, 327], [341, 334], [533, 340]]
    img9 = [[x - img9[0][0], y - img9[0][1]] for x, y in img9]
    d_vert, d_hori = 2.97, 4.85
    world9 = [[d_hori * j, d_vert * i] for i in range(3) for j in range(3)]
    H, _ = cv2.findHomography(np.array(img9), np.array(world9))
    return H


# Each pedestrian is ONE "TRACK.Rn=[[x y f]; [x y f]; ...];" line.
_TRIPLE_RE = re.compile(r"\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]")


def load_edinburgh(path, *, title: str = "edinburgh",
                   fps: float = 9.0, sampling_rate: int = 4):
    files = sorted(glob.glob(os.path.join(path, "tracks.*.txt")))
    if not files:
        raise FileNotFoundError(f"no tracks.*.txt under {path}")
    H = _edinburgh_homog()
    rows = []
    aid = 0
    for scene_id, file in enumerate(files):
        with open(file) as f:
            for line in f:
                if not line.lstrip().startswith("TRACK"):
                    continue
                triples = _TRIPLE_RE.findall(line)
                if len(triples) >= 2:
                    for x, y, fr in triples:
                        rows.append([int(float(fr)), aid, float(x), float(y), scene_id])
                aid += 1
    raw = pd.DataFrame(rows, columns=["frame_id", "agent_id", "pos_x", "pos_y", "scene_id"])
    raw[["pos_x", "pos_y"]] = _image_to_world(raw[["pos_x", "pos_y"]].to_numpy(), H)
    raw["label"] = "pedestrian"
    return from_long_dataframe(raw, title=title, fps=fps, sampling_rate=sampling_rate)


# --------------------------------------------------------------------------- #
# PETS-2009 (S2L1)
# --------------------------------------------------------------------------- #
def load_pets(path, *, title: str = "pets2009-s2l1",
              fps: float = 7.0, sampling_rate: int = 2,
              annot: str = "PETS2009-S2L1.xml", calib: str = "View_001.xml"):
    from ._calib_tsai import (CameraParameters, CalibrationConstants,
                              image_coord_to_world_coord)
    annot_path = os.path.join(path, annot)
    calib_path = os.path.join(path, calib)

    cp, cc = CameraParameters(), CalibrationConstants()
    croot = et.parse(calib_path).getroot()
    g = croot.find("Geometry").attrib
    cp.Ncx, cp.Nfx = float(g["ncx"]), float(g["nfx"])
    cp.dx, cp.dy, cp.dpx, cp.dpy = float(g["dx"]), float(g["dy"]), float(g["dpx"]), float(g["dpy"])
    intr = croot.find("Intrinsic").attrib
    cc.f, cc.kappa1 = float(intr["focal"]), float(intr["kappa1"])
    cp.Cx, cp.Cy, cp.sx = float(intr["cx"]), float(intr["cy"]), float(intr["sx"])
    extr = croot.find("Extrinsic").attrib
    cc.Tx, cc.Ty, cc.Tz = float(extr["tx"]), float(extr["ty"]), float(extr["tz"])
    cc.Rx, cc.Ry, cc.Rz = float(extr["rx"]), float(extr["ry"]), float(extr["rz"])
    cc.calc_rr()

    rows = []
    for frame_node in et.parse(annot_path).getroot():
        fr = int(frame_node.attrib.get("number"))
        for obj in frame_node.find("objectlist").findall("object"):
            b = obj.find("box").attrib
            xc, yc, h = float(b["xc"]), float(b["yc"]), float(b["h"])
            pos_x, pos_y = image_coord_to_world_coord(xc, yc + h / 2, 0, cp, cc)
            rows.append([fr, int(obj.attrib["id"]), pos_x / 1000., pos_y / 1000.])
    raw = pd.DataFrame(rows, columns=["frame_id", "agent_id", "pos_x", "pos_y"])
    raw["scene_id"] = 0
    raw["label"] = "pedestrian"
    return from_long_dataframe(raw, title=title, fps=fps, sampling_rate=sampling_rate)


# --------------------------------------------------------------------------- #
# WILDTRACK
# --------------------------------------------------------------------------- #
def load_wildtrack(path, *, title: str = "wildtrack",
                   fps: float = 10.0, sampling_rate: int = 1):
    files = sorted(glob.glob(os.path.join(path, "*.json")))
    rows = []
    for file in files:
        fr = int(os.path.basename(file).replace(".json", ""))
        for annot in json.loads(pathlib.Path(file).read_text()):
            pid = annot["positionID"]
            X = -3.0 + 0.025 * (pid % 480)
            Y = -9.0 + 0.025 * (pid / 480)
            rows.append([fr, int(annot["personID"]), X, Y])
    raw = pd.DataFrame(rows, columns=["frame_id", "agent_id", "pos_x", "pos_y"])
    raw["scene_id"] = 0
    raw["label"] = "pedestrian"
    return from_long_dataframe(raw, title=title, fps=fps, sampling_rate=sampling_rate)
