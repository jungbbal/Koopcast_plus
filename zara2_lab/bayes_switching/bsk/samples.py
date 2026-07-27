"""Turn Scenes into the (psi_t, psi_{t+1}, token) triples the fits consume."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import zlab

from .koopman import psi, tokenize


@dataclass
class Samples:
    X: np.ndarray        # (N, 5)  psi_t
    Y: np.ndarray        # (N, 5)  psi_{t+1}
    q: np.ndarray        # (N,)    handcrafted token
    seq: np.ndarray      # (N,)    track id -- transitions only valid within one
    hist: np.ndarray     # (N, 2*L) short velocity history, the k-means feature
    dtheta: np.ndarray   # (N,)    heading change into step t
    dspeed: np.ndarray   # (N,)    speed change into step t
    speed: np.ndarray    # (N,)    ||v_t||

    def __len__(self):
        return len(self.X)


def build(scenes, hist_len: int = 3, **tok_kw) -> Samples:
    """Every t inside a track with enough history for a token and a target.

    Needs p_{t-hist_len..t+1}: history for the k-means feature, p_{t-2..t} for
    the token, p_{t+1} for the target. Tracks shorter than that contribute
    nothing, and no sample ever straddles two tracks.
    """
    need = max(hist_len, 2)
    Xs, Ys, seqs, hists = [], [], [], []
    vprev_all, v_all = [], []
    sid = 0
    for sc in scenes:
        for traj in sc.tracks(min_len=need + 2).values():
            v = np.diff(traj, axis=0)                    # v[i] = p_{i+1} - p_i
            T = len(traj)
            for t in range(need, T - 1):
                # v_t = traj[t] - traj[t-1] = v[t-1]
                Xs.append(np.concatenate([traj[t], v[t - 1]]))
                Ys.append(np.concatenate([traj[t + 1], v[t]]))
                vprev_all.append(v[t - 2])
                v_all.append(v[t - 1])
                hists.append(v[t - hist_len:t].ravel())
                seqs.append(sid)
            sid += 1

    X = np.asarray(Xs)
    Y = np.asarray(Ys)
    vprev = np.asarray(vprev_all)
    vcur = np.asarray(v_all)

    sp, sp_prev = np.linalg.norm(vcur, axis=1), np.linalg.norm(vprev, axis=1)
    dth = np.arctan2(vcur[:, 1], vcur[:, 0]) - np.arctan2(vprev[:, 1], vprev[:, 0])
    dth = (dth + np.pi) % (2 * np.pi) - np.pi

    return Samples(
        X=psi(X[:, :2], X[:, 2:]),
        Y=psi(Y[:, :2], Y[:, 2:]),
        q=tokenize(vprev, vcur, **tok_kw),
        seq=np.asarray(seqs),
        hist=np.asarray(hists),
        dtheta=dth,
        dspeed=sp - sp_prev,
        speed=sp,
    )


def rollout(samples_scenes, Ks, q_fn, obs=8, pred=12, stride=1):
    """Closed-loop multi-step rollout -- the test the one-step MSE cannot do.

    Propagates psi through K, re-deriving the token at every step from the
    *predicted* state (the handcrafted token is a function of v_t and v_{t-1},
    both carried in the rollout), so nothing from the future leaks in.
    ``q_fn(v_prev, v) -> token index``; pass a constant-0 function for a global K.

    Returns (ADE, FDE) in metres over all windows.
    """
    ades, fdes = [], []
    for sc in samples_scenes:
        for traj in sc.tracks(min_len=obs + pred).values():
            for s in range(0, len(traj) - (obs + pred) + 1, stride):
                w = traj[s:s + obs + pred]
                p, v = w[obs - 1].copy(), w[obs - 1] - w[obs - 2]
                v_prev = w[obs - 2] - w[obs - 3] if obs >= 3 else v.copy()
                out = []
                for _ in range(pred):
                    j = int(q_fn(v_prev[None], v[None])[0])
                    nxt = Ks[j] @ np.array([p[0], p[1], v[0], v[1], 1.0])
                    v_prev, p, v = v, nxt[:2], nxt[2:4]
                    out.append(p.copy())
                out = np.asarray(out)
                d = np.linalg.norm(out - w[obs:], axis=-1)
                ades.append(d.mean())
                fdes.append(d[-1])
    return float(np.mean(ades)), float(np.mean(fdes))
