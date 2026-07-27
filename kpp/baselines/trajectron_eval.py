"""
Trajectron++ evaluation on the ETH/UCY leave-one-out test splits.

Trajectron++ cannot go through kpp's ``predict_scene`` path: upstream
``get_timesteps_data`` forwards the *prediction horizon* where the caller's
``min_future_timesteps`` was intended, so every candidate node must already
carry ``ph`` steps of future ground truth (see ``README.md``). That never holds
for history-only inference, but it does hold for an offline ``Environment``
built by ``trajectron_data.py`` -- which is what this module scores.

Targets therefore match the standard protocol (>=7 history steps, 12 future
steps) rather than being routed through ``evaluate_scene``; ``n_samples`` is
reported so the target counts can be compared against the other baselines.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Optional

import numpy as np

from .trajectron_data import load_split

_VENDOR = pathlib.Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

DEFAULT_MODEL_ROOT = _VENDOR / "assets" / "models" / "trajectron"

#: scene key -> pretrained checkpoint directory name
MODEL_DIRS = {
    "eth": "eth_vel_ar3", "hotel": "hotel_vel_ar3", "univ": "univ_vel_ar3",
    "zara1": "zara01_vel_ar3", "zara2": "zara02_vel_ar3",
    "snu-asri": "snu-asri_ar3", "snu-asri-ood": "snu-asri_ar3",
}

_STATE = {"position": ["x", "y"]}


def _install_module_aliases() -> None:
    """Make the pretrained checkpoints unpicklable in our vendored namespace.

    The checkpoints were pickled from upstream Trajectron++, whose packages sit
    at the import root (``model.*``, ``environment.*``). Vendored here they live
    under ``canvas.predictors.trajectron.*``, so unpickling raises
    ``ModuleNotFoundError: No module named 'model'``. Aliasing the old names is
    the standard remedy and needs no edit to the pickles or the vendored code.
    """
    import importlib
    for alias, real in (
        ("model", "canvas.predictors.trajectron.model"),
        ("environment", "canvas.predictors.trajectron.environment"),
        ("utils", "canvas.predictors.trajectron.utils"),
    ):
        if alias not in sys.modules:
            try:
                sys.modules[alias] = importlib.import_module(real)
            except ImportError:
                pass


def load_trajectron(model_dir: str, env, ts: int = 100, device: str = "cpu"):
    """Load a Trajectron++ checkpoint with its weights actually bound.

    The vendored ``ModelRegistrar.load_models`` has its body commented out -- it
    only clears ``model_dict``, so every module comes back freshly initialised
    and the net predicts a near-stationary trajectory. We populate the registrar
    ourselves (what the commented-out code was meant to do) before
    ``set_environment`` wires the submodules up.
    """
    import json
    import torch
    from canvas.predictors.trajectron.model.model_registrar import ModelRegistrar
    from canvas.predictors.trajectron.model.trajectron import Trajectron

    _install_module_aliases()
    model_dir = str(model_dir)
    ckpt = pathlib.Path(model_dir) / f"model_registrar-{ts}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint {ckpt}")

    registrar = ModelRegistrar(model_dir, device)
    registrar.model_dict = torch.load(ckpt, map_location=device, weights_only=False)
    if len(registrar.model_dict) == 0:
        raise RuntimeError(f"checkpoint {ckpt} unpickled to an empty model_dict")

    with open(pathlib.Path(model_dir) / "config.json") as f:
        hyperparams = json.load(f)

    stg = Trajectron(registrar, hyperparams, None, device)
    stg.set_environment(env)
    stg.set_annealing_params()
    return stg, hyperparams


def evaluate_trajectron(scene_key: str, *, model_dir: Optional[str] = None,
                        ts: int = 100, ph: int = 12, min_history: int = 7,
                        device: str = "cpu"):
    """Score Trajectron++ on one held-out scene. Returns a dict of metrics."""
    scene_key = scene_key.strip().lower()
    if model_dir is None:
        model_dir = str(DEFAULT_MODEL_ROOT / MODEL_DIRS[scene_key])
    if not (pathlib.Path(model_dir) / "config.json").exists():
        raise FileNotFoundError(f"no Trajectron++ checkpoint at {model_dir}")

    env = load_split(scene_key, "test")
    stg, hyperparams = load_trajectron(model_dir, env, ts=ts, device=device)

    ades, fdes = [], []
    for scene in env.scenes:
        for t in range(scene.timesteps):
            preds = stg.predict(scene, np.array([t]), ph, num_samples=1,
                                min_history_timesteps=min_history,
                                min_future_timesteps=ph,
                                z_mode=True, gmm_mode=True, full_dist=False)
            if not preds:
                continue
            for _t, per_node in preds.items():
                for node, p in per_node.items():
                    p = np.asarray(p, dtype=float).reshape(-1, ph, 2)[0]
                    gt = np.asarray(node.get(np.array([t + 1, t + ph]), _STATE),
                                    dtype=float)
                    if gt.shape[0] != ph or not np.isfinite(gt).all():
                        continue
                    d = np.linalg.norm(p - gt, axis=-1)
                    ades.append(float(d.mean()))
                    fdes.append(float(d[-1]))

    if not ades:
        return None
    a, f = np.asarray(ades), np.asarray(fdes)
    return {
        "predictor": "Trajectron++", "dataset": f"ethucy/{scene_key}/test",
        "n_samples": len(ades),
        "ade_mean": float(a.mean()), "ade_std": float(a.std()),
        "fde_mean": float(f.mean()), "fde_std": float(f.std()),
    }
