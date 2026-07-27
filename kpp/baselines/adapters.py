"""
Adapters -- wrap the vendored CANVAS predictors in the kpp ``Predictor`` API.

The upstream code under ``vendor/canvas`` is byte-for-byte unmodified. All the
glue that makes those predictors "ours" lives here:

  * dataset-name -> weight-path mapping (per baseline),
  * feeding our ``{agent_id: (obs_len, 2)}`` scene dict straight through
    (CANVAS' native call signature is already ``dict -> dict``),
  * normalising the returned tensors to numpy ``(pred_len, 2)``.

Each CANVAS predictor's ``__call__`` takes ``{id: (T, 2)}`` history and returns
``{id: (P, 2)}`` future -- which is exactly kpp's ``predict_scene`` contract, so
the wrappers are thin. ``predict`` (marginal batch) falls back to a per-agent
dict round-trip.

Status (see ``README.md``):
  linear, stgcnn, socialvae, eigen -- working, weights vendored.
  gp                              -- blocked: ``george`` 0.4 API change.
  trajectron                      -- blocked: pretrained weights not shipped.
"""
from __future__ import annotations

import os
import sys
import pathlib
from typing import Dict

import numpy as np

from ..predictors.base import Predictor

# --------------------------------------------------------------------------- #
# make the vendored ``canvas`` package importable (path is *inside* this repo,
# so nothing is pulled from an external checkout).
# --------------------------------------------------------------------------- #
_VENDOR = pathlib.Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

_EIGEN_DIR = _VENDOR / "canvas" / "predictors" / "eigen"


# --------------------------------------------------------------------------- #
# dataset-name normalisation
# --------------------------------------------------------------------------- #
def _scene(dataset: str) -> str:
    """Canonical kpp scene key from any alias."""
    d = dataset.strip().lower()
    return {"zara01": "zara1", "zara02": "zara2", "uni": "univ"}.get(d, d)


# --------------------------------------------------------------------------- #
# common wrapper
# --------------------------------------------------------------------------- #
class _CanvasAdapter(Predictor):
    """Wrap one instantiated CANVAS predictor. Subclasses build ``self._m``."""
    social = True
    _label = "canvas"

    def __init__(self, dataset: str, *, pred_len: int = 12, history_len: int = 8,
                 dt: float = 0.4, device: str = "cpu") -> None:
        super().__init__(pred_len=pred_len)
        self.dataset = _scene(dataset)
        self.history_len = int(history_len)
        self.dt = float(dt)
        self.device = device
        self._m = self._build()

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    # ---- kpp Predictor API ------------------------------------------------- #
    def predict_scene(self, obs: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        if not obs:
            return {}
        clean = {a: np.asarray(o, dtype=float) for a, o in obs.items()
                 if np.asarray(o).shape[0] >= 2}
        if not clean:
            return {}
        out = self._m(clean)
        res: Dict[int, np.ndarray] = {}
        for a, p in out.items():
            p = np.asarray(p, dtype=float)
            res[a] = p[: self.pred_len]
        return res

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        d = self.predict_scene({i: obs[i] for i in range(len(obs))})
        return np.stack([d[i] for i in range(len(obs))])

    @property
    def name(self) -> str:
        return self._label


# --------------------------------------------------------------------------- #
# per-baseline adapters
# --------------------------------------------------------------------------- #
class LinearBaseline(_CanvasAdapter):
    """CANVAS curvature-aware constant-turn-rate predictor (no weights).

    Upstream only runs its model when ``len(traj) >= history_len + 2`` and
    otherwise silently holds the last position. Our scene windows carry exactly
    ``obs_len`` (=8) steps, so we configure the underlying predictor with
    ``history_len = obs_len - 2`` to keep it on the real code path. This is a
    constructor argument -- the vendored file is untouched.
    """
    social = False
    _label = "CANVAS-Linear"

    def _build(self):
        from canvas.predictors.linear_predictor import LinearPredictor
        return LinearPredictor(prediction_len=self.pred_len,
                               history_len=max(2, self.history_len - 2),
                               dt=self.dt, device=self.device)


class STGCNNBaseline(_CanvasAdapter):
    """Social-STGCNN (Mohamed et al., 2020)."""
    _label = "Social-STGCNN"
    _MAP = {"eth": "eth", "hotel": "hotel", "univ": "univ",
            "zara1": "zara1", "zara2": "zara2",
            "snu-asri": "lobby_data", "snu-asri-ood": "lobby_data"}

    def _build(self):
        from canvas.predictors.Social_STGCNN.STGCNN_live_test import STGCNN_Predictor
        folder = self._MAP[self.dataset]
        return STGCNN_Predictor(
            prediction_len=self.pred_len, history_len=self.history_len,
            dt=self.dt, device=self.device,
            model_dir=f"/checkpoint/social-stgcnn-{folder}")


class SocialVAEBaseline(_CanvasAdapter):
    """SocialVAE (Xu et al., 2022)."""
    _label = "SocialVAE"
    _MAP = {"eth": ("eth", "eth"), "hotel": ("hotel", "hotel"),
            "univ": ("univ", "univ"), "zara1": ("zara01", "zara01"),
            "zara2": ("zara02", "zara02"),
            "snu-asri": ("lobby_data", "lobby"), "snu-asri-ood": ("lobby_data", "lobby")}

    def _build(self):
        from canvas.predictors.SocialVAE.social_vae_runner import Social_VAE_Predictor
        cfg_name, model_name = self._MAP[self.dataset]
        return Social_VAE_Predictor(
            prediction_len=self.pred_len, history_len=self.history_len,
            dt=self.dt, device=self.device,
            cfg=f"/config/{cfg_name}.py", model_path=f"/models/{model_name}")


class EigenBaseline(_CanvasAdapter):
    """EigenTrajectory (Bae et al., 2023) over a Social-STGCNN predictor."""
    _label = "EigenTrajectory"
    _MAP = {"eth": ("eth", "eth"), "hotel": ("hotel", "hotel"),
            "univ": ("uni", "uni"), "zara1": ("zara01", "zara01"),
            "zara2": ("zara02", "zara02"),
            "snu-asri": ("lobby_data", "lobby_data"),
            "snu-asri-ood": ("lobby_data", "lobby_data")}

    def _build(self):
        from canvas.predictors.eigen.eigen_predictor_class import EigenTrajectoryPredictor
        model_sub, json_scene = self._MAP[self.dataset]
        cfg = str(_EIGEN_DIR / "json_files" / f"eigentrajectory-stgcnn-{json_scene}.json")
        model_path = str(_EIGEN_DIR / "models" / model_sub / "model_best.pth")
        return EigenTrajectoryPredictor(
            prediction_len=self.pred_len, history_len=self.history_len,
            dt=self.dt, device=self.device, cfg=cfg, model_path=model_path)


def _george_compat() -> None:
    """Restore the pre-0.4 ``george`` return arity, without editing vendored code.

    The vendored CANVAS GP does ``mu, _ = gp.predict(y, t, return_cov=False)``.
    george >=0.4 returns a bare ndarray for that call (a 2-tuple only when
    ``return_cov`` or ``return_var`` is set), so the unpack raises. We wrap
    ``george.GP.predict`` to hand back ``(mu, None)`` for exactly that case;
    every other call signature is passed through untouched. Idempotent.
    """
    import george
    if getattr(george.GP.predict, "_kpp_compat", False):
        return
    _orig = george.GP.predict

    def predict(self, y, t=None, **kw):
        out = _orig(self, y, t, **kw)
        want_cov = kw.get("return_cov", True)
        want_var = kw.get("return_var", False)
        if not want_cov and not want_var and not isinstance(out, tuple):
            return out, None
        return out

    predict._kpp_compat = True
    george.GP.predict = predict


class GPBaseline(_CanvasAdapter):
    """CANVAS Gaussian-process regressor (per-agent, george/HODLR).

    Needs ``_george_compat()`` for george >=0.4; note this predictor fits a GP
    per agent per call, so it is *far* slower than the neural baselines.

    Upstream requires ``len(traj) >= history_len + 1`` or it silently holds the
    last position, so we pass ``history_len = obs_len - 1`` (see LinearBaseline).
    """
    social = False
    _label = "CANVAS-GP"

    def _build(self):
        _george_compat()
        from canvas.predictors.gp_predictor import GaussianProcessPredictor
        return GaussianProcessPredictor(
            prediction_len=self.pred_len,
            history_len=max(2, self.history_len - 1),
            dt=self.dt, device=self.device)


class TrajectronBaseline(_CanvasAdapter):
    """Trajectron++ (Salzmann et al., 2020). BLOCKED: pretrained weights are not
    shipped with CANVAS -- download via ``assets/download_assets.sh`` and place
    under ``vendor/assets/models/trajectron/<scene>_vel_ar3`` first."""
    _label = "Trajectron++"
    _MAP = {"eth": "eth_vel_ar3", "hotel": "hotel_vel_ar3", "univ": "univ_vel_ar3",
            "zara1": "zara01_vel_ar3", "zara2": "zara02_vel_ar3",
            "snu-asri": "snu-asri_ar3", "snu-asri-ood": "snu-asri_ar3"}

    def _build(self):
        from canvas.predictors.trajectron_predictor import TrajectronPredictor
        model_dir = _VENDOR / "assets" / "models" / "trajectron" / self._MAP[self.dataset]
        return TrajectronPredictor(
            prediction_len=self.pred_len, history_len=self.history_len,
            dt=self.dt, device=self.device, model_dir=str(model_dir))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
BASELINES = {
    "linear": LinearBaseline,
    "stgcnn": STGCNNBaseline,
    "socialvae": SocialVAEBaseline,
    "eigen": EigenBaseline,
    "gp": GPBaseline,               # blocked (george version)
    "trajectron": TrajectronBaseline,  # blocked (weights)
}

#: baselines verified to load + run with the vendored weights
WORKING = ("linear", "stgcnn", "socialvae", "eigen")


def make_baseline(name: str, dataset: str, **kw) -> Predictor:
    """Factory: ``make_baseline("stgcnn", "zara1")`` -> ready Predictor."""
    key = name.strip().lower()
    if key not in BASELINES:
        raise ValueError(f"unknown baseline '{name}'. Available: {sorted(BASELINES)}")
    return BASELINES[key](dataset, **kw)
