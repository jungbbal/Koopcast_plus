#!/usr/bin/env python3
"""
3-way comparison on the ETH/UCY leave-one-out test splits:

    ConstantVelocity   |   KoopCast++ (static K)   |   KoopCast++ (online K update)

All three are scored over the *same* neighbour-aware targets (same (agent, t0)
set per scene). The adaptive variant keeps the trained time-delay-8 model but,
before rolling out a target, updates K online from that agent's recent *observed*
transitions (real h_t -> h_{t+1}) ending at t0:

    K <- K + eta * (psi_{s+1} - K psi_s) psi_s^T / ||psi_s||^2 .

It uses up to ADAPT_STEPS observed steps before t0 (looked up from the full
trajectory); agents with no extra history fall back to the static K, so the
target set is identical to the static/CV runs.

Usage:
    python scripts/compare_koopcastpp.py [--eta 0.1] [scene ...]
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kpp.data import load_ethucy, ETHUCY_SCENES, scene_windows
from kpp.predictors import ConstantVelocity, KoopCastPP
from kpp.predictors.koopcastpp import available_datasets, _core
from kpp.eval import evaluate_scene

ADAPT_STEPS = 10   # max observed steps before t0 used for online adaptation


def _scene_index(ds):
    """Per-(scene_id, agent_id) sorted (frames, xy); per-scene frame map + step."""
    traj, fmap, step = {}, {}, {}
    for sid, sdf in ds.data.groupby("scene_id"):
        sid = int(sid)
        fm = {}
        for f, fdf in sdf.groupby("frame_id"):
            fm[int(f)] = (fdf["agent_id"].to_numpy(), fdf[["pos_x", "pos_y"]].to_numpy())
        fmap[sid] = fm
        diffs = np.diff(np.sort(sdf["frame_id"].unique()))
        step[sid] = int(np.median(diffs)) if len(diffs) else 1
        for aid, tr in sdf.groupby("agent_id"):
            tr = tr.sort_values("frame_id")
            traj[(sid, int(aid))] = (tr["frame_id"].to_numpy(),
                                     tr[["pos_x", "pos_y"]].to_numpy())
    return traj, fmap, step


def adaptive_eval(m: KoopCastPP, ds, *, eta: float, obs_len=8, pred_len=12):
    """Score KoopCast++ with online K adaptation over the standard scene targets."""
    traj, fmap, step = _scene_index(ds)
    cache = {}

    def psi_at(sid, aid, frames, xy, c):
        """lifted observable at trajectory index c (needs c-obs_len+1..c contiguous)."""
        ego = xy[c]
        f = int(frames[c])
        key = (sid, f, aid)
        neigh = cache.get(key)
        if neigh is None:
            aids, axy = fmap[sid][f]
            neigh = axy[aids != aid]
            cache[key] = neigh
        return m._psi(xy[c - obs_len + 1:c + 1], neigh)

    a_list, f_list = [], []
    for sw in scene_windows(ds, obs_len=obs_len, pred_len=pred_len):
        sid = sw.scene_id
        st = step[sid]
        for tgt in sw.targets:
            aid = tgt.agent_id
            frames, xy = traj[(sid, aid)]
            t0 = tgt.start_frame + (obs_len - 1) * st
            idx = np.where(frames == t0)[0]
            if len(idx) == 0:
                continue
            i0 = int(idx[0])
            # build the longest contiguous observable sequence ending at i0
            seq = []
            c = i0
            while c >= obs_len - 1 and len(seq) <= ADAPT_STEPS:
                win = frames[c - obs_len + 1:c + 1]
                if len(win) < obs_len or not np.all(np.diff(win) == st):
                    break
                seq.append(psi_at(sid, aid, frames, xy, c))
                if c - 1 >= 0 and frames[c] - frames[c - 1] == st:
                    c -= 1
                else:
                    break
            seq = seq[::-1]                       # chronological: ... -> psi(t0)
            K = _core.adapt_K(m.K, seq, eta) if (eta > 0 and len(seq) >= 2) else m.K
            pred = _core.rollout(K, seq[-1], pred_len)
            a_list.append(float(np.linalg.norm(pred - tgt.pred, axis=1).mean()))
            f_list.append(float(np.linalg.norm(pred[-1] - tgt.pred[-1])))
    return float(np.mean(a_list)), float(np.mean(f_list)), len(a_list)


def main():
    args = sys.argv[1:]
    eta = 0.1
    if "--eta" in args:
        i = args.index("--eta"); eta = float(args[i + 1]); del args[i:i + 2]
    want = [a.strip().lower() for a in args] or list(ETHUCY_SCENES)
    scenes = [s for s in want if s in set(available_datasets())]

    cv = ConstantVelocity(pred_len=12)
    print(f"\nonline-update eta = {eta},  adapt window <= {ADAPT_STEPS} steps\n")
    hdr = f"{'scene':<7}{'ADE_cv':>9}{'ADE_kpp':>9}{'ADE_kpp+':>10}   |{'FDE_cv':>9}{'FDE_kpp':>9}{'FDE_kpp+':>10}"
    print(hdr); print("-" * len(hdr))
    agg = []
    for sc in scenes:
        ds = load_ethucy(sc, "test")
        m = KoopCastPP(sc)
        r_cv = evaluate_scene(cv, ds)
        r_st = evaluate_scene(m, ds)
        a_ad, f_ad, n_ad = adaptive_eval(m, ds, eta=eta)
        agg.append((r_cv.ade_mean, r_st.ade_mean, a_ad, r_cv.fde_mean, r_st.fde_mean, f_ad))
        print(f"{sc:<7}{r_cv.ade_mean:>9.4f}{r_st.ade_mean:>9.4f}{a_ad:>10.4f}   |"
              f"{r_cv.fde_mean:>9.4f}{r_st.fde_mean:>9.4f}{f_ad:>10.4f}")
    if agg:
        m = np.mean(np.array(agg), axis=0)
        print("-" * len(hdr))
        print(f"{'AVG':<7}{m[0]:>9.4f}{m[1]:>9.4f}{m[2]:>10.4f}   |{m[3]:>9.4f}{m[4]:>9.4f}{m[5]:>10.4f}")
    print("\nlegend: cv=ConstantVelocity  kpp=KoopCast++ static  kpp+=KoopCast++ online-update")


if __name__ == "__main__":
    main()
