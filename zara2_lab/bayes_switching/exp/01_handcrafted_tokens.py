"""Q: is pedestrian motion better explained by several local K_j than one K?

Experiment 1 of the switching-Koopman plan: handcrafted motion tokens, plain
least squares, no network. Fit on the ETH-UCY leave-one-out train split,
evaluate on the held-out zara2 scene.

Two controls the raw E_switch < E_global comparison needs, or it proves nothing:

  1. PARAMETER COUNT. Five K_j have 5x the parameters of one K, so they fit the
     training set better no matter how the partition is drawn. The `random`
     baseline -- the same token histogram, labels shuffled -- costs exactly as
     many parameters and carries zero motion information. Any honest claim is
     E_switch < E_random, not E_switch < E_global.
  2. FREE COMPONENTS. With psi = [p, v, 1] and v defined as a position
     difference, p_{t+1} = p_t + v_t holds identically; the position block is
     predicted exactly by every model. Total MSE is therefore mostly a constant.
     'mse_v' below is the part that is actually being modelled.

Run: python exp/01_handcrafted_tokens.py
"""
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))   # bayes_switching/  -> bsk
sys.path.insert(0, str(_HERE.parents[2]))   # zara2_lab/        -> zlab

import numpy as np
from sklearn.cluster import KMeans

import zlab
from bsk import TOKENS, err, fit_lstsq, fit_switching, tokenize, transition_matrix
from bsk.samples import build, rollout

RNG = np.random.default_rng(0)
OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
M = len(TOKENS)


def report_partition(name, tr_q, te_q, tr, te, results):
    """Fit one K per group on train, score on both splits."""
    Ks, counts, n_fb = fit_switching(tr.X, tr.Y, tr_q, M)
    e_tr = err(tr.X, tr.Y, Ks=Ks, q=tr_q)
    e_te = err(te.X, te.Y, Ks=Ks, q=te_q)
    results[name] = {
        "train": e_tr, "test": e_te,
        "counts": counts.tolist(), "n_fallback": n_fb,
    }
    print(f"  {name:10s} train mse_v={e_tr['mse_v']:.6f}   "
          f"test mse_v={e_te['mse_v']:.6f}   test mse={e_te['mse']:.6f}")
    return Ks


def main():
    OUT.mkdir(exist_ok=True)
    train_scenes = zlab.load("train")
    test_scenes = [zlab.load_zara2_scene()]

    tr = build(train_scenes)
    te = build(test_scenes)
    print(f"samples: train={len(tr)}  test(zara2)={len(te)}\n")

    print("token histogram (train / test):")
    for j, name in enumerate(TOKENS):
        print(f"  {name:9s} {np.mean(tr.q == j):6.3f} / {np.mean(te.q == j):6.3f}")

    results = {}

    # ---- global -----------------------------------------------------------
    print("\n=== one-step residuals (fit on train, scored on both) ===")
    K_glob = fit_lstsq(tr.X, tr.Y)
    e_tr, e_te = err(tr.X, tr.Y, K=K_glob), err(te.X, te.Y, K=K_glob)
    results["global"] = {"train": e_tr, "test": e_te}
    print(f"  {'global':10s} train mse_v={e_tr['mse_v']:.6f}   "
          f"test mse_v={e_te['mse_v']:.6f}   test mse={e_te['mse']:.6f}")

    # ---- random partition (parameter-matched control) ---------------------
    tr_rand = RNG.permutation(tr.q)
    te_rand = RNG.permutation(te.q)
    report_partition("random", tr_rand, te_rand, tr, te, results)

    # ---- k-means on short velocity history --------------------------------
    km = KMeans(n_clusters=M, n_init=10, random_state=0).fit(tr.hist)
    report_partition("kmeans", km.predict(tr.hist), km.predict(te.hist),
                     tr, te, results)

    # ---- handcrafted tokens ------------------------------------------------
    Ks_tok = report_partition("token", tr.q, te.q, tr, te, results)

    # ---- per-token error breakdown on the test scene ----------------------
    print("\n=== test mse_v by token: which motions does switching help? ===")
    print(f"  {'token':10s} {'n':>6s} {'global':>10s} {'switch':>10s} {'gain%':>7s}")
    per_token = {}
    for j, name in enumerate(TOKENS):
        m = te.q == j
        if m.sum() == 0:
            continue
        g = err(te.X[m], te.Y[m], K=K_glob)["mse_v"]
        s = err(te.X[m], te.Y[m], Ks=Ks_tok, q=te.q[m])["mse_v"]
        per_token[name] = {"n": int(m.sum()), "global": g, "switch": s}
        print(f"  {name:10s} {m.sum():6d} {g:10.6f} {s:10.6f} "
              f"{100*(g-s)/g:6.1f}%")
    results["per_token_test"] = per_token

    # ---- token motion statistics (experiment 3, handcrafted version) ------
    print("\n=== token motion statistics on zara2 (sanity: do labels mean it?) ===")
    print(f"  {'token':10s} {'||v||':>8s} {'dtheta':>8s} {'d||v||':>8s}")
    stats = {}
    for j, name in enumerate(TOKENS):
        m = te.q == j
        if m.sum() == 0:
            continue
        stats[name] = {"speed": float(te.speed[m].mean()),
                       "dtheta": float(te.dtheta[m].mean()),
                       "dspeed": float(te.dspeed[m].mean())}
        print(f"  {name:10s} {stats[name]['speed']:8.3f} "
              f"{stats[name]['dtheta']:8.3f} {stats[name]['dspeed']:8.3f}")
    results["token_stats_test"] = stats

    # ---- transition matrix -------------------------------------------------
    P = transition_matrix(te.q, te.seq, M)
    print("\n=== token transition matrix P[j,k] on zara2 ===")
    print(f"  {'from\\to':10s}" + "".join(f"{t:>9s}" for t in TOKENS))
    for j, name in enumerate(TOKENS):
        print(f"  {name:10s}" + "".join(f"{P[j,k]:9.3f}" for k in range(M)))
    results["transition_test"] = P.tolist()

    # ---- closed-loop rollout: the number that is comparable to the field ---
    print("\n=== 12-step closed-loop rollout on zara2 (metres) ===")
    ade_g, fde_g = rollout(test_scenes, np.stack([K_glob] * M),
                           lambda a, b: np.zeros(1, int))
    ade_s, fde_s = rollout(test_scenes, Ks_tok, tokenize)
    obs, gt = zlab.split_obs_pred(zlab.windows(test_scenes))
    v = obs[:, -1] - obs[:, -2]
    cv = obs[:, -1][:, None] + v[:, None] * np.arange(1, 13)[None, :, None]
    print(f"  {'const-vel':10s} ADE={zlab.ade(cv, gt).mean():.3f}  "
          f"FDE={zlab.fde(cv, gt).mean():.3f}")
    print(f"  {'global K':10s} ADE={ade_g:.3f}  FDE={fde_g:.3f}")
    print(f"  {'switching':10s} ADE={ade_s:.3f}  FDE={fde_s:.3f}")
    results["rollout_test"] = {
        "const_vel": {"ade": float(zlab.ade(cv, gt).mean()),
                      "fde": float(zlab.fde(cv, gt).mean())},
        "global": {"ade": ade_g, "fde": fde_g},
        "switching": {"ade": ade_s, "fde": fde_s},
    }

    (OUT / "01_handcrafted_tokens.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / '01_handcrafted_tokens.json'}")


if __name__ == "__main__":
    main()
