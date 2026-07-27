"""Q: did switching lose because local K_j don't exist, or because of domain shift?

Experiment 1 fit on the LOO train split (eth/hotel/zara1/zara3/students) and
scored on zara2 -- two different scenes. Switching lost there, but that result
confounds two explanations:

  (a) pedestrian motion really is one global linear map, or
  (b) per-token K_j are real but scene-specific, so they fail to transfer.

This script removes the shift: fit and test both inside zara2, split by AGENT
(never by time within a track, which would leak a track's own dynamics across
the split). If switching still loses here, (b) is dead and the negative result
in experiment 1 is about the hypothesis, not the protocol.

Also sweeps the token thresholds, because a partition that puts 87% of zara2 in
'straight' is barely a partition -- the null result could just be that.

Run: python exp/02_within_scene.py
"""
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np

import zlab
from bsk import TOKENS, err, fit_lstsq, fit_switching
from bsk.samples import build

RNG = np.random.default_rng(0)
OUT = _HERE.parents[1] / "out"
M = len(TOKENS)


def agent_split(sc, frac=0.5, seed=0):
    """Two Scenes holding disjoint halves of the agents."""
    ids = np.unique(sc.agents)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    a = set(perm[: int(len(perm) * frac)].tolist())
    masks = [np.array([x in a for x in sc.agents]),
             np.array([x not in a for x in sc.agents])]
    return [zlab.Scene(f"{sc.name}_{k}", sc.frames[m], sc.agents[m], sc.xy[m])
            for k, m in zip(("fit", "held"), masks)]


def compare(tr, te, label, results):
    K = fit_lstsq(tr.X, tr.Y)
    e_g = err(te.X, te.Y, K=K)["mse_v"]

    Ks_t, counts, _ = fit_switching(tr.X, tr.Y, tr.q, M)
    e_t = err(te.X, te.Y, Ks=Ks_t, q=te.q)["mse_v"]

    q_tr_r, q_te_r = RNG.permutation(tr.q), RNG.permutation(te.q)
    Ks_r, _, _ = fit_switching(tr.X, tr.Y, q_tr_r, M)
    e_r = err(te.X, te.Y, Ks=Ks_r, q=q_te_r)["mse_v"]

    frac_straight = float(np.mean(te.q == 0))
    print(f"  {label:22s} global={e_g:.6f}  random={e_r:.6f}  token={e_t:.6f}   "
          f"gain_vs_global={100*(e_g-e_t)/e_g:+6.2f}%  "
          f"gain_vs_random={100*(e_r-e_t)/e_r:+6.2f}%  straight={frac_straight:.2f}")
    results[label] = {"global": e_g, "random": e_r, "token": e_t,
                      "frac_straight": frac_straight, "counts": counts.tolist()}


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    zara2 = zlab.load_zara2_scene()
    fit_sc, held_sc = agent_split(zara2)

    print("=== within-zara2, agent-level split (no domain shift) ===")
    print("default thresholds tau_theta=0.15, tau_v=0.04")
    tr, te = build([fit_sc]), build([held_sc])
    print(f"  samples fit={len(tr)}  held={len(te)}")
    compare(tr, te, "within-zara2", results)

    print("\n=== threshold sweep: does a less lopsided partition help? ===")
    for tt, tv in [(0.05, 0.01), (0.10, 0.02), (0.15, 0.04), (0.30, 0.08)]:
        tr = build([fit_sc], tau_theta=tt, tau_v=tv)
        te = build([held_sc], tau_theta=tt, tau_v=tv)
        compare(tr, te, f"tau_th={tt} tau_v={tv}", results)

    print("\n=== seed sweep: is the sign of the effect stable? ===")
    for seed in range(5):
        f_sc, h_sc = agent_split(zara2, seed=seed)
        compare(build([f_sc]), build([h_sc]), f"seed={seed}", results)

    (OUT / "02_within_scene.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / '02_within_scene.json'}")


if __name__ == "__main__":
    main()
