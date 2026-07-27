"""
KoopCast++ core -- neighbour-aware Koopman trajectory predictor.

Design (see ``docs/koopcastpp_method.tex``):
  * observable  psi_t = [ time-delay history (2*obs_len) ; social_emb (d_s) ]
                where the raw positions live *inside* psi, so no decoder is
                needed: the next position is read off the top block after one
                Koopman step.
  * operator    K via Extended DMD (closed form, ridge-regularised).
  * learning    only the social encoder is trained; the loss is the plain
                one-step Koopman consistency  || Psi' - K Psi ||^2 .
  * social      R=2 m, 12-sector polar proximity histogram -> small MLP.

This module holds the pieces shared by training and inference; it is numpy +
torch and depends on nothing else in the repo.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

# defaults (match the design doc)
OBS_LEN = 8
PRED_LEN = 12
RADIUS = 2.0
NUM_SECTORS = 12
SOCIAL_DIM = 4
RIDGE = 1e-4


# --------------------------------------------------------------------------- #
# Social encoding
# --------------------------------------------------------------------------- #
def social_histogram(ego: np.ndarray, neighbours: np.ndarray,
                     radius: float = RADIUS, num_sectors: int = NUM_SECTORS) -> np.ndarray:
    """12-sector polar proximity histogram around ``ego``.

    ego        : (2,) position.
    neighbours : (M, 2) other agents' positions at the same instant.
    returns    : (num_sectors,) with max(1 - d/R) per angular sector (world frame).
    """
    vec = np.zeros(num_sectors, dtype=np.float64)
    if neighbours is None or len(neighbours) == 0:
        return vec
    d = np.asarray(neighbours, dtype=float) - np.asarray(ego, dtype=float)
    dist = np.hypot(d[:, 0], d[:, 1])
    keep = (dist > 1e-6) & (dist <= radius) & np.isfinite(dist)
    if not keep.any():
        return vec
    d, dist = d[keep], dist[keep]
    ang = (np.degrees(np.arctan2(d[:, 1], d[:, 0])) + 360.0) % 360.0
    sec = np.clip((ang // (360.0 / num_sectors)).astype(int), 0, num_sectors - 1)
    val = 1.0 - dist / radius
    for k, v in zip(sec, val):
        if v > vec[k]:
            vec[k] = v
    return vec


class SocialEncoder(nn.Module):
    """g (num_sectors) -> e (social_dim). The only learnable part of the model."""

    def __init__(self, num_sectors: int = NUM_SECTORS, social_dim: int = SOCIAL_DIM,
                 hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_sectors, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, social_dim),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


# --------------------------------------------------------------------------- #
# Koopman operator (Extended DMD, closed form)
# --------------------------------------------------------------------------- #
def edmd_operator(psi_t: torch.Tensor, psi_tp1: torch.Tensor,
                  ridge: float = RIDGE) -> torch.Tensor:
    """Closed-form EDMD operator K solving  psi_{t+1} ~ K psi_t.

    psi_t, psi_tp1 : (N, d) lifted snapshots (rows = samples).
    returns K      : (d, d), differentiable in the inputs (hence in the encoder).
        K = Phi' Phi^T (Phi Phi^T + ridge I)^{-1},  Phi = psi_t^T.
    """
    Phi = psi_t.T            # (d, N)
    Phi2 = psi_tp1.T         # (d, N)
    d = Phi.shape[0]
    eye = torch.eye(d, dtype=Phi.dtype, device=Phi.device)
    G = Phi @ Phi.T + ridge * eye          # (d, d)
    A = Phi2 @ Phi.T                        # (d, d)
    # K = A G^{-1}  (G symmetric PD) -> solve G^T X = A^T  =>  X = K^T
    K = torch.linalg.solve(G, A.T).T
    return K


def consistency_loss(K: torch.Tensor, psi_t: torch.Tensor,
                     psi_tp1: torch.Tensor) -> torch.Tensor:
    """Plain one-step Koopman consistency  mean || psi_{t+1} - K psi_t ||^2 ."""
    pred = psi_t @ K.T                      # (N, d)
    return ((psi_tp1 - pred) ** 2).sum(dim=1).mean()


# --------------------------------------------------------------------------- #
# Observable assembly
# --------------------------------------------------------------------------- #
def history_block(hist: np.ndarray, obs_len: int = OBS_LEN) -> np.ndarray:
    """Time-delay history block [p_t, p_{t-1}, ..., p_{t-obs_len+1}] flattened (2*obs_len,)."""
    h = np.asarray(hist, dtype=float)[-obs_len:]
    return h[::-1].reshape(-1)


def rollout(K: np.ndarray, psi0: np.ndarray, pred_len: int = PRED_LEN) -> np.ndarray:
    """Iterate psi <- K psi, reading the top position block each step. (pred_len, 2)."""
    out = np.empty((pred_len, 2), dtype=float)
    psi = psi0.astype(float).copy()
    for t in range(pred_len):
        psi = K @ psi
        out[t] = psi[:2]
    return out


def adapt_K(K0: np.ndarray, psi_seq: List[np.ndarray], eta: float) -> np.ndarray:
    """Online per-agent K adaptation over a sequence of *observed* lifted states.

    For each consecutive observed pair (psi_s, psi_{s+1}) apply the streaming
    rank-1 update
        K <- K + eta * (psi_{s+1} - K psi_s) psi_s^T / ||psi_s||^2 .
    Requires >= 2 observed observables; with the standard 8-step protocol only a
    single observable is available, so this reduces to the static K (no update).
    """
    K = K0.astype(float).copy()
    if eta <= 0 or len(psi_seq) < 2:
        return K
    for s in range(len(psi_seq) - 1):
        x, xn = psi_seq[s], psi_seq[s + 1]
        denom = float(x @ x) + 1e-8
        K = K + eta * np.outer(xn - K @ x, x) / denom
    return K
