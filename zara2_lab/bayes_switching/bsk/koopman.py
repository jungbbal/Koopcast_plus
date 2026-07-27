"""Least-squares Koopman fitting, global and switching.

Observable is deliberately minimal:  psi_t = [p_t, v_t, 1] in R^5,
with v_t = p_t - p_{t-1} (metres per 0.4 s step, NOT per second).

One thing to keep in mind when reading any error from this observable: because
v is *defined* as a position difference, p_{t+1} = p_t + v_t holds exactly, and
a K containing the rows [I I 0] reproduces the position block with zero error.
So the position component is not evidence of anything -- every fit gets it free.
The only quantity actually being modelled is v_{t+1}, i.e. acceleration.
``fit_lstsq``/``err`` therefore report the velocity block separately.
"""
from __future__ import annotations

import numpy as np

# token vocabulary for the handcrafted experiment
TOKENS = ["straight", "left", "right", "accel", "decel"]


# --------------------------------------------------------------------------- #
# observables and tokens
# --------------------------------------------------------------------------- #
def psi(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N,2),(N,2) -> (N,5) observable [px, py, vx, vy, 1]."""
    return np.concatenate([p, v, np.ones((len(p), 1))], axis=1)


def tokenize(
    v_prev: np.ndarray,
    v: np.ndarray,
    tau_theta: float = 0.15,
    tau_v: float = 0.04,
    static_eps: float = 0.04,
) -> np.ndarray:
    """Handcrafted motion tokens -> (N,) int index into TOKENS.

    The spec's five conditions overlap (a turning step is usually also
    changing speed), so they are applied in priority order: turn first, then
    speed change, else straight. Any other order is equally defensible -- it is
    reported alongside the result because it changes the partition.

    Below ``static_eps`` (0.04 m/step = 0.1 m/s) the heading of v is numerical
    noise, so Delta-theta is suppressed and the sample can only be a speed
    change or straight. Without this guard a standing pedestrian's jitter gets
    scattered uniformly across left/right.
    """
    sp_prev = np.linalg.norm(v_prev, axis=1)
    sp = np.linalg.norm(v, axis=1)
    dtheta = np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(v_prev[:, 1], v_prev[:, 0])
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi      # wrap to (-pi, pi]
    dtheta = np.where((sp < static_eps) | (sp_prev < static_eps), 0.0, dtheta)
    dsp = sp - sp_prev

    q = np.zeros(len(v), dtype=int)                      # default: straight
    q[dtheta > tau_theta] = TOKENS.index("left")
    q[dtheta < -tau_theta] = TOKENS.index("right")
    turn = np.abs(dtheta) > tau_theta
    q[~turn & (dsp > tau_v)] = TOKENS.index("accel")
    q[~turn & (dsp < -tau_v)] = TOKENS.index("decel")
    return q


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit_lstsq(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """argmin_K sum ||y - K x||^2 over rows of X, Y -> K of shape (d, d)."""
    d = X.shape[1]
    A = X.T @ X + ridge * np.eye(d)
    return np.linalg.solve(A, X.T @ Y).T


def fit_switching(X, Y, q, n_tokens, ridge=1e-6, min_count=30):
    """One K per token; tokens with too few samples fall back to the global K.

    Returns (Ks (M,d,d), counts (M,), n_fallback).
    """
    K_glob = fit_lstsq(X, Y, ridge)
    Ks, counts, n_fb = [], [], 0
    for j in range(n_tokens):
        m = q == j
        counts.append(int(m.sum()))
        if m.sum() < min_count:
            Ks.append(K_glob)
            n_fb += 1
        else:
            Ks.append(fit_lstsq(X[m], Y[m], ridge))
    return np.stack(Ks), np.array(counts), n_fb


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def err(X, Y, K=None, Ks=None, q=None):
    """Mean squared one-step residual, total and velocity-only.

    Pass either ``K`` (global) or ``Ks`` + ``q`` (switching).
    Returns dict with 'mse' (all 5 components) and 'mse_v' (velocity block),
    the latter being the only part that is not free -- see module docstring.
    """
    pred = X @ K.T if Ks is None else np.einsum("nij,nj->ni", Ks[q], X)
    r = pred - Y
    return {
        "mse": float((r ** 2).sum(axis=1).mean()),
        "mse_v": float((r[:, 2:4] ** 2).sum(axis=1).mean()),
    }


def transition_matrix(q: np.ndarray, seq_id: np.ndarray, n_tokens: int) -> np.ndarray:
    """P[j,k] = p(q_{t+1}=k | q_t=j), counting only within-track transitions."""
    P = np.zeros((n_tokens, n_tokens))
    same = seq_id[:-1] == seq_id[1:]
    for j, k in zip(q[:-1][same], q[1:][same]):
        P[j, k] += 1
    row = P.sum(axis=1, keepdims=True)
    return np.divide(P, row, out=np.zeros_like(P), where=row > 0)
