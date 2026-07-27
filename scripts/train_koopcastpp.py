#!/usr/bin/env python3
"""
Train KoopCast++ on the ETH/UCY leave-one-out splits.

Same model structure throughout (time-delay-8 observable + social encoder + EDMD
operator, no decoder); only the training loss differs:

  --loss multistep   (default) state-block multi-step prediction loss
        L = mean_samples  sum_{tau=1..P} || (K^tau psi_t)[:2] - p_{t+tau} ||^2
        with K = EDMD(Psi, Psi') fit (closed form) on the one-step pairs.
        Only positions enter the loss, so the social encoder is rewarded only
        for improving prediction -> it does NOT collapse to a constant.

  --loss consistency  plain one-step Koopman residual || Psi' - K Psi ||^2
        (the v1 objective; collapses the social encoder).

Artifacts (encoder + global K + cfg) -> kpp/predictors/koopcastpp/data/koopcastpp_<scene>.pt

Usage:
    python scripts/train_koopcastpp.py                       # all 5, multistep
    python scripts/train_koopcastpp.py --loss consistency eth
"""
import sys
import pathlib
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kpp.data import load_ethucy, load, ETHUCY_SCENES
from kpp.predictors.koopcastpp import _core

EPOCHS = 3000
LR = 1e-3
_ART = pathlib.Path(__file__).resolve().parents[1] / "kpp" / "predictors" / "koopcastpp" / "data"


def _frame_maps(ds):
    maps = {}
    for sid, sdf in ds.data.groupby("scene_id"):
        fm = {}
        for f, fdf in sdf.groupby("frame_id"):
            fm[int(f)] = (fdf["agent_id"].to_numpy(), fdf[["pos_x", "pos_y"]].to_numpy())
        maps[int(sid)] = fm
    return maps


def build_data(ds, obs_len=_core.OBS_LEN, pred_len=_core.PRED_LEN,
               radius=_core.RADIUS, num_sectors=_core.NUM_SECTORS):
    """Return one-step pairs (for EDMD K) and multi-step prediction samples.

    pairs   : H_t, G_t, H_tp1, G_tp1                (consecutive lifted transitions)
    predict : H0, G0, FUT(pred_len*2)              (observable at t0 + future GT)
    Social histograms cached per (scene, frame, agent).
    """
    fmaps = _frame_maps(ds)
    cache = {}

    def soc(sid, frame, aid, ego):
        key = (sid, frame, aid)
        v = cache.get(key)
        if v is None:
            aids, xy = fmaps[sid][frame]
            v = _core.social_histogram(ego, xy[aids != aid], radius, num_sectors)
            cache[key] = v
        return v

    Ht, Gt, Htp, Gtp = [], [], [], []
    H0, G0, FUT = [], [], []
    for (sid, aid), tr in ds.get_trajectories():
        tr = tr.sort_values("frame_id")
        fr = tr["frame_id"].to_numpy()
        xy = tr[["pos_x", "pos_y"]].to_numpy()
        n = len(fr)
        if n < obs_len + 1:
            continue
        step = int(np.median(np.diff(fr)))
        sid_i, aid_i = int(sid), int(aid)
        # one-step pairs
        for c in range(obs_len - 1, n - 1):
            if not np.all(np.diff(fr[c - obs_len + 1: c + 2]) == step):
                continue
            Ht.append(xy[c - obs_len + 1: c + 1][::-1].reshape(-1))
            Htp.append(xy[c - obs_len + 2: c + 2][::-1].reshape(-1))
            Gt.append(soc(sid_i, int(fr[c]), aid_i, xy[c]))
            Gtp.append(soc(sid_i, int(fr[c + 1]), aid_i, xy[c + 1]))
        # multi-step prediction samples (obs_len + pred_len contiguous)
        for c in range(obs_len - 1, n - pred_len):
            if not np.all(np.diff(fr[c - obs_len + 1: c + pred_len + 1]) == step):
                continue
            H0.append(xy[c - obs_len + 1: c + 1][::-1].reshape(-1))
            G0.append(soc(sid_i, int(fr[c]), aid_i, xy[c]))
            FUT.append(xy[c + 1: c + pred_len + 1].reshape(-1))
    f = lambda L: torch.tensor(np.asarray(L), dtype=torch.float32)
    return (f(Ht), f(Gt), f(Htp), f(Gtp)), (f(H0), f(G0), f(FUT))


#: snu-asri (lobby3) ships an official split: scenes 2..9 train, 1 val, 0 test.
#: ``data/raw/snu-asri`` holds only scene 0 -- i.e. the TEST scene -- so the
#: training scenes were copied to ``data/raw/snu-asri-train`` (see its
#: SOURCE.txt). The CANVAS lobby checkpoints follow the same split, so training
#: here puts KoopCast++ on equal footing with the vendored baselines.
SNU_TRAIN_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "snu-asri-train"
SNU_TRAIN_SCENES = [f"{i}.npy" for i in range(2, 10)]


def _load_snu_train():
    """Concatenate snu-asri training scenes 2..9 into one TrajDataset.

    Each file is an independent capture, so every scene gets its own
    ``scene_id`` -- neighbour context must not cross scenes.
    """
    import pandas as pd
    from kpp.data.loaders import load_taa_npy

    frames = []
    for sid, name in enumerate(SNU_TRAIN_SCENES):
        path = SNU_TRAIN_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}. snu-asri training scenes live in the lobby3 "
                f"split; see data/raw/snu-asri-train/SOURCE.txt")
        d = load_taa_npy(path, title=f"snu-asri-train/{name}", fps=2.5)
        df = d.data.copy()
        df["scene_id"] = sid
        frames.append(df)

    ds = load_taa_npy(SNU_TRAIN_DIR / SNU_TRAIN_SCENES[0],
                      title="snu-asri/train", fps=2.5)
    ds.data = pd.concat(frames, ignore_index=True)
    return ds


def load_train_split(scene):
    """Training data for a scene: ETH/UCY uses its official leave-one-out
    ``train`` split; snu-asri uses the lobby3 train scenes (2..9)."""
    scene = scene.strip().lower()
    if scene in ETHUCY_SCENES:
        return load_ethucy(scene, "train")
    if scene in ("snu-asri", "snu-asri-ood"):
        return _load_snu_train()
    return load(scene)


def train_scene(scene, loss_mode="multistep", epochs=EPOCHS, lr=LR, device="cpu",
                ridge=_core.RIDGE, verbose=True):
    t0 = time.time()
    ds = load_train_split(scene)
    (H_t, G_t, H_tp, G_tp), (H0, G0, FUT) = build_data(ds)
    P = _core.PRED_LEN
    FUT = FUT.view(-1, P, 2)
    enc = _core.SocialEncoder().to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)

    for ep in range(epochs):
        opt.zero_grad()
        psi_t = torch.cat([H_t, enc(G_t)], dim=1)
        psi_tp = torch.cat([H_tp, enc(G_tp)], dim=1)
        K = _core.edmd_operator(psi_t, psi_tp, ridge=ridge)   # closed-form EDMD (differentiable)
        if loss_mode == "consistency":
            loss = _core.consistency_loss(K, psi_t, psi_tp)
        else:                                           # multistep state-block prediction
            psi = torch.cat([H0, enc(G0)], dim=1)       # (M,d) observable at t0
            loss = 0.0
            for tau in range(P):
                psi = psi @ K.T
                loss = loss + ((psi[:, :2] - FUT[:, tau]) ** 2).sum(dim=1).mean()
            loss = loss / P
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
        opt.step()
        if verbose and (ep % 500 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:5d}  {loss_mode} loss = {loss.item():.6e}")

    with torch.no_grad():
        psi_t = torch.cat([H_t, enc(G_t)], dim=1)
        psi_tp = torch.cat([H_tp, enc(G_tp)], dim=1)
        K = _core.edmd_operator(psi_t, psi_tp, ridge=ridge).cpu().numpy()
        emb_std = enc(G_t).std(dim=0).cpu().numpy()     # collapse check

    _ART.mkdir(parents=True, exist_ok=True)
    out = _ART / f"koopcastpp_{scene}.pt"
    torch.save({
        "K": K,
        "encoder": {k: v.cpu() for k, v in enc.state_dict().items()},
        "cfg": {"obs_len": _core.OBS_LEN, "pred_len": _core.PRED_LEN,
                "radius": _core.RADIUS, "num_sectors": _core.NUM_SECTORS,
                "social_dim": _core.SOCIAL_DIM, "loss": loss_mode, "ridge": ridge},
    }, out)
    if verbose:
        eig = np.abs(np.linalg.eigvals(K)).max()
        print(f"  {scene:<6} |K|_spec={eig:.3f}  social_emb_std={np.round(emb_std,4)}  "
              f"-> {out.name}  ({time.time()-t0:.1f}s)")
    return out


#: EDMD ridge per dataset. ETH/UCY keeps the original 1e-4. snu-asri is a lobby
#: capture where people move ~4x less per 8 steps than in ETH/UCY, so its
#: history block is far more collinear (cond ~1.5e4 vs ~2e3) and 1e-4 yields an
#: unstable operator (|K|_spec 5.7 -> 12-step rollout explodes). 0.1 is the
#: smallest value on the scan that brings |K|_spec back to ~1.
RIDGE_BY_SCENE = {"snu-asri": 0.1, "snu-asri-ood": 0.1}


def main():
    args = sys.argv[1:]
    loss_mode = "multistep"
    if "--loss" in args:
        i = args.index("--loss"); loss_mode = args[i + 1]; del args[i:i + 2]
    ridge_override = None
    if "--ridge" in args:
        i = args.index("--ridge"); ridge_override = float(args[i + 1]); del args[i:i + 2]
    scenes = [a.strip().lower() for a in args] or list(ETHUCY_SCENES)
    print(f"training KoopCast++ ({loss_mode} loss) on: {scenes}")
    for sc in scenes:
        ridge = ridge_override if ridge_override is not None \
            else RIDGE_BY_SCENE.get(sc, _core.RIDGE)
        print(f"[{sc}]  ridge={ridge:g}")
        train_scene(sc, loss_mode=loss_mode, ridge=ridge)


if __name__ == "__main__":
    main()
