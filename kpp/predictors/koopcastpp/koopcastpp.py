"""
KoopCast++ predictor (ours) -- neighbour-aware Koopman, inference.

Loads the per-scene artifacts produced by ``scripts/train_koopcastpp.py``
(a trained social encoder + the EDMD operator K) and rolls the lifted observable
forward. The observable carries the time-delayed state, so prediction is just
``psi <- K psi`` with the position read off the top block -- no decoder.

static   : eta = 0.0  (fixed global K)
adaptive : eta > 0.0  (online per-agent rank-1 K update over observed steps;
           needs >= 2 observed observables, see ``_core.adapt_K``)
"""
from __future__ import annotations

import pathlib
from typing import Dict, List

import numpy as np
import torch

from ..base import Predictor
from . import _core

_DATA = pathlib.Path(__file__).resolve().parent / "data"


def _artifact_path(dataset: str) -> pathlib.Path:
    return _DATA / f"koopcastpp_{dataset.strip().lower()}.pt"


def available_datasets() -> List[str]:
    return sorted(p.stem.replace("koopcastpp_", "") for p in _DATA.glob("koopcastpp_*.pt"))


class KoopCastPP(Predictor):
    social = True

    def __init__(self, dataset: str, pred_len: int = _core.PRED_LEN,
                 device: str = "cpu", *, eta: float = 0.0) -> None:
        super().__init__(pred_len=pred_len)
        path = _artifact_path(dataset)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained KoopCast++ artifact for '{dataset}' at {path}. "
                f"Train it first: python scripts/train_koopcastpp.py {dataset}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["cfg"]
        self.dataset = dataset.strip().lower()
        self.eta = float(eta)
        self.obs_len = int(cfg["obs_len"])
        self.radius = float(cfg["radius"])
        self.num_sectors = int(cfg["num_sectors"])
        self.K = np.asarray(ckpt["K"], dtype=float)
        self.enc = _core.SocialEncoder(num_sectors=self.num_sectors,
                                       social_dim=int(cfg["social_dim"]))
        self.enc.load_state_dict(ckpt["encoder"])
        self.enc.eval()
        self.device = device

    # ----- observable construction ----------------------------------------- #
    def _social_emb(self, ego: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
        g = _core.social_histogram(ego, neighbours, self.radius, self.num_sectors)
        with torch.no_grad():
            e = self.enc(torch.from_numpy(g).float()).numpy()
        return e

    def _psi(self, hist: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
        h = _core.history_block(hist, self.obs_len)              # (2*obs_len,)
        e = self._social_emb(hist[-1], neighbours)               # (social_dim,)
        return np.concatenate([h, e], axis=0)

    # ----- prediction ------------------------------------------------------- #
    def predict_scene(self, obs: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        if not obs:
            return {}
        out: Dict[int, np.ndarray] = {}
        last = {a: np.asarray(o, dtype=float)[-1] for a, o in obs.items() if len(o) >= 1}
        for aid, hist in obs.items():
            hist = np.asarray(hist, dtype=float)
            if hist.shape[0] < self.obs_len:
                continue
            neigh = np.asarray([last[a] for a in last if a != aid], dtype=float).reshape(-1, 2)
            psi0 = self._psi(hist, neigh)
            K = self.K
            if self.eta > 0 and hist.shape[0] > self.obs_len:
                K = self._adapt(aid, hist, obs)
            out[aid] = _core.rollout(K, psi0, self.pred_len)
        return out

    def _adapt(self, aid: int, hist: np.ndarray, obs: Dict[int, np.ndarray]) -> np.ndarray:
        """Build observed observables from the longer history and adapt K online."""
        n = hist.shape[0]
        seq = []
        for c in range(self.obs_len - 1, n):
            ego = hist[c]
            neigh = []
            for a, o in obs.items():
                o = np.asarray(o, dtype=float)
                if a != aid and o.shape[0] > c:
                    neigh.append(o[c])
            seq.append(self._psi(hist[:c + 1], np.asarray(neigh).reshape(-1, 2)))
        return _core.adapt_K(self.K, seq, self.eta)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Marginal fallback: each agent rolled out without neighbours."""
        obs = np.asarray(obs, dtype=float)
        empty = np.empty((0, 2))
        out = [_core.rollout(self.K, self._psi(obs[i], empty), self.pred_len)
               for i in range(obs.shape[0])]
        return np.stack(out)

    @property
    def name(self) -> str:
        return "KoopCast++" + ("(adaptive)" if self.eta > 0 else "(static)")
