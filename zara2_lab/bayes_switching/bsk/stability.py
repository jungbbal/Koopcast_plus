"""Stability-constrained operator fitting.

Two independent knobs, so the M-sweep can be read as a statement about regime
structure instead of about which M happens to fit stably:

  * ``fit_scaled_ridge`` -- lambda_j = lambda_0 * N / N_j, so a token with few
    samples is regularised harder.
  * ``project_spectrum`` -- pull eigenvalues into the unit disk.

Two cautions the naive version of each gets wrong:

1. STRUCTURAL UNIT EIGENVALUES. With psi = [p, v, ..., 1], the identity
   p_{t+1} = p_t + v_t and the constant channel force K to carry eigenvalue 1
   (in a Jordan block, so ||K^h|| grows linearly in h -- that IS the position
   integration, not an instability). Capping every eigenvalue at 1-eps damps
   position integration by (1-eps)^h: at eps=0.01 over 12 steps that is an 11%
   shrink toward the current position, which looks like "smoother" predictions
   but is a bias toward standing still. ``mode="excess"`` therefore caps only
   the genuinely unstable eigenvalues at exactly 1 and leaves the rest alone;
   ``mode="all"`` is the literal 1-eps rule, kept for comparison.

2. rho <= 1 DOES NOT MEAN BOUNDED OVER 12 STEPS. For a non-normal K the
   transient ||K^h|| can be large while every eigenvalue sits inside the disk.
   ``diagnose`` reports ||K^12||_2 alongside rho; a sweep that only controls rho
   has not necessarily controlled the rollout.
"""
from __future__ import annotations

import numpy as np


def fit_scaled_ridge(X, Y, lam0: float, n_total: int) -> np.ndarray:
    """argmin ||Y - X K^T||^2 + lambda_j ||K||^2 with lambda_j = lam0 * N/N_j."""
    d = X.shape[1]
    lam = lam0 * n_total / max(len(X), 1)
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y).T


def project_spectrum(K: np.ndarray, eps: float = 0.0, mode: str = "excess"):
    """Pull eigenvalues into the unit disk. Returns (K_projected, info).

    mode="excess": scale only |lambda| > 1 down to magnitude 1 - eps.
                   Structural lambda = 1 modes survive when eps = 0.
    mode="all":    the literal lambda / max(1, |lambda|/(1-eps)) rule, which
                   also shrinks the structural modes.

    ``info`` carries cond(V) -- the eigenvector conditioning. Eigen-decomposition
    of a strongly non-normal K is ill-conditioned, and a large cond(V) means the
    projected operator is not trustworthy; the caller should report it rather
    than silently accept the projection.
    """
    lam, V = np.linalg.eig(K)
    mag = np.abs(lam)
    if mode == "excess":
        over = mag > 1.0
        scale = np.where(over, (1.0 - eps) / np.maximum(mag, 1e-12), 1.0)
    elif mode == "all":
        scale = 1.0 / np.maximum(1.0, mag / (1.0 - eps))
    else:
        raise ValueError(mode)
    lam_new = lam * scale
    condV = float(np.linalg.cond(V))
    try:
        Kp = np.real(V @ np.diag(lam_new) @ np.linalg.inv(V))
    except np.linalg.LinAlgError:
        return K, {"cond_V": condV, "failed": True, "n_projected": int((mag > 1).sum())}
    return Kp, {"cond_V": condV, "failed": False,
                "n_projected": int((mag > 1).sum())}


def diagnose(K: np.ndarray, horizon: int = 12) -> dict:
    """rho(K) and the finite-horizon amplification ||K^h||_2 that rho misses."""
    rho = float(np.abs(np.linalg.eigvals(K)).max())
    Kh = np.linalg.matrix_power(K, horizon)
    with np.errstate(over="ignore", invalid="ignore"):
        amp = float(np.linalg.norm(Kh, 2)) if np.isfinite(Kh).all() else np.inf
    return {"rho": rho, "amp12": amp}
