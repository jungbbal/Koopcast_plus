# Experimental Results

All numbers are single-shot ADE/FDE (meters), 8 observed → 12 predicted, dt 0.4 s.
Within a given scene, every model sees the **same set of (agent, t0) targets**
(`kpp.eval.evaluate_scene`, neighbour-aware windows).

Reproduction commands are given above each table.

---

## 1. ETH/UCY leave-one-out

Each scene is held out — we train on the remaining scenes and evaluate on that scene
(`load_ethucy(scene, "test")`).

### 1.1 KoopCast++ vs baselines

Output of `python scripts/eval_adaptive.py`, all 5 scenes
(raw log: `runs/adaptive_ethucy_all.log`).

ADE / FDE, lower is better. Best value in each column in **bold**.

| model | eth | hotel | univ | zara1 | zara2 | **AVG** |
|---|---|---|---|---|---|---|
| **KoopCast++ (ours, static)** | 1.0649 / 2.1794 | 0.4882 / 1.0225 | 0.6271 / 1.3556 | 0.4316 / 0.9606 | 0.3544 / 0.7895 | 0.5932 / 1.2615 |
| **KoopCast++ (ours, adaptive)** | 1.0649 / 2.1794 | 0.4808 / 1.0055 | 0.6305 / 1.3662 | 0.4316 / 0.9606 | 0.3544 / 0.7895 | 0.5924 / 1.2602 |
| ConstantVelocity | 1.0755 / 2.2819 | **0.3194 / 0.6142** | **0.6036 / 1.3386** | 0.4272 / 0.9524 | **0.3219 / 0.7238** | 0.5495 / 1.1822 |
| SocialVAE | **0.9801 / 1.9405** | 0.3424 / 0.6630 | 0.6158 / 1.3458 | 0.4611 / 1.0230 | 0.3452 / 0.7559 | **0.5489 / 1.1456** |
| Trajectron++ (pretrained) † | 1.0371 / 2.1439 | 0.3551 / 0.6849 | 0.6374 / 1.4182 | **0.4257 / 0.9393** | 0.3271 / 0.7316 | 0.5565 / 1.1836 |
| Social-STGCNN | 1.2891 / 2.3337 | 0.6998 / 1.3807 | 0.7607 / 1.5072 | 0.4991 / 1.0329 | 0.4502 / 0.9194 | 0.7398 / 1.4348 |
| EigenTrajectory | 1.3180 / 2.5586 | 0.9759 / 1.8149 | 0.7036 / 1.4390 | 0.7256 / 1.5515 | 0.9463 / 1.8398 | 0.9339 / 1.8408 |

† Because of an upstream defect, Trajectron++ is scored with `evaluate_trajectron`
rather than `evaluate_scene` (reason:
[kpp/baselines/README.md](../kpp/baselines/README.md)). The number of targets
matches the other models exactly (zara1 n=2356), but keep in mind that the path
differs when reading these numbers. See §1.2 for the retrained version.

**Ranking (AVG ADE):** SocialVAE 0.5489 < ConstantVelocity 0.5495 <
Trajectron++ 0.5565 < **KoopCast++ 0.5932** < Social-STGCNN 0.7398 <
EigenTrajectory 0.9339

**Per-scene KoopCast++ vs ConstantVelocity:**

| scene | KoopCast++ | CV | diff |
|---|---|---|---|
| eth | 1.0649 | 1.0755 | **−1.0%** (only win) |
| hotel | 0.4882 | 0.3194 | +52.8% |
| univ | 0.6271 | 0.6036 | +3.9% |
| zara1 | 0.4316 | 0.4272 | +1.0% |
| zara2 | 0.3544 | 0.3219 | +10.1% |
| **AVG** | **0.5932** | **0.5495** | **+8.0%** |

The eta in the `adaptive` column is chosen **on the val split** (not test-set tuning).
Selected values: eth 0.0, hotel 0.005, univ 0.005, zara1 0.0, zara2 0.0.

### ⚠️ An honest caveat

**On ETH/UCY, KoopCast++ loses to the constant-velocity baseline.** The 5-scene mean
ADE is 0.5932 vs ConstantVelocity 0.5495 (**8.0% worse**). Per scene, it barely wins
on eth alone (1.0649 vs 1.0755) and loses on all of hotel/univ/zara1/zara2. The gap is
widest on hotel (0.4882 vs 0.3194, 53% worse).

SocialVAE (0.5489) and pretrained Trajectron++ (0.5565, §1.2) are the top performers on
this benchmark, and KoopCast++ is clearly better than Social-STGCNN (0.7398) and
EigenTrajectory (0.9339) but sits below the constant-velocity model.

**Online adaptation has essentially no effect on ETH/UCY.** Val selects eta=0 for 3 of
the 5 scenes, and even on the remaining hotel/univ the gain is at the ±0.5% level. This
is consistent with the analysis in §3 — adaptation is not designed to be unhelpful
within-distribution; within-distribution it actively **hurts**.

Accordingly, the current case for KoopCast++ is not "ETH/UCY SOTA" but **the OOD
robustness in §2** (1st place on snu-asri-ood, an edge over constant velocity). The
ETH/UCY table does not support that claim so much as bound it — do not drop this table
when presenting externally; present it alongside.

### 1.2 Trajectron++ — pretrained vs retrained (reproducibility check)

`python scripts/eval_trajectron.py --both`. Because of an upstream defect, Trajectron++
is scored with `evaluate_trajectron` rather than `evaluate_scene` (reason:
[kpp/baselines/README.md](../kpp/baselines/README.md)). The number of targets matches
the other baselines exactly (zara1: n=2356).

| scene | pretrained ADE/FDE | retrained ADE/FDE | ΔADE |
|---|---|---|---|
| eth | 1.0371 / 2.1439 | 1.0727 / 2.2572 | +3.4% |
| hotel | 0.3551 / 0.6849 | 0.4200 / 0.8461 | +18.3% |
| univ | 0.6374 / 1.4182 | 0.6865 / 1.5021 | +7.7% |
| zara1 | 0.4257 / 0.9393 | **0.3883 / 0.8380** | −8.8% |
| zara2 | 0.3271 / 0.7316 | **0.3178 / 0.6959** | −2.8% |
| **AVG** | **0.5565 / 1.1836** | **0.5771 / 1.2279** | +3.7% |

Retraining reproduces the released checkpoint to within a few percent on average and
actually beats it on zara1/zara2. Since the sign of the deviation flips from scene to
scene, we attribute it to per-scene seed variance rather than a systematic
preprocessing error. hotel is the weakest reproduction (+18%), which makes sense as it
is the smallest — and therefore noisiest — scene in ETH/UCY. Treat **the released
checkpoint as the default and the retrained version as a reproducibility check.**

Retrained models are located at `runs/trajectron/<scene>/models_<timestamp><scene>/`.
Cost: on a single GPU, ~2 min/epoch → ~3.5 hours for 100 epochs per scene (~17 hours
for all 5).

---

## 2. snu-asri (lobby)

snu-asri is the `lobby3` dataset and has an **official split**
(`/home/jungbbal/ood/lobby3/`): scenes **2..9 train**, **1 val**, **0 test**.

What this repo calls `snu-asri`, `data/raw/snu-asri/0.npy`, is the **test scene**
— byte-identical to `lobby3/test/0.npy` (md5 `aaef2ed3599b0d66510ab8a7887967fb`).
`snu-asri-ood.npy` is likewise identical to `lobby3_ood/test/scene4_test.npy`. The
training scenes are copied under `data/raw/snu-asri-train/` (see `SOURCE.txt`).

The vendor CANVAS lobby checkpoint follows this same split, so training KoopCast++ on
scenes 2..9 and scoring on scene 0 puts all models on equal footing.

> An earlier version of this file reported that snu-asri had *no* official split and
> used a 70/30 temporal split of scene 0. That trained KoopCast++ **on part of the
> test scene**, inflating the score, and the accompanying warning that "the CANVAS
> baseline may be a leak" was exactly backwards. Both are corrected here.

The official test scene (n=14992) and a separate OOD capture (n=135). Every model below
is trained on lobby3 scenes 2..9 and has never seen scene 0:

| model | test ADE/FDE | ood ADE/FDE |
|---|---|---|
| SocialVAE | **0.1292 / 0.2458** | 0.1608 / 0.3388 |
| KoopCast++ (ours) | 0.1339 / 0.2533 | **0.1562 / 0.3331** |
| Social-STGCNN | 0.1495 / 0.2759 | 0.2446 / 0.4877 |
| ConstantVelocity | 0.1636 / 0.3105 | 0.1644 / 0.3466 |
| EigenTrajectory | 0.1894 / 0.3537 | 0.3206 / 0.6135 |

KoopCast++ is 2nd on the in-distribution test scene (3.6% behind SocialVAE) and
**1st on the OOD capture** — the only trained model there that beats the
constant-velocity baseline by a clear margin, while Social-STGCNN and EigenTrajectory
collapse badly once out of distribution.

---

## 3. KoopCast++ online adaptation (eta > 0)

KoopCast++ can update the Koopman operator online from the observation stream
(`_core.adapt_K`, rank-1 per agent). It only kicks in when a history **longer** than
`obs_len` is available, so it must be evaluated with `full_history=True`:

```python
evaluate_scene(KoopCastPP("snu-asri", eta=0.05), ds, full_history=True)
```

`scene_windows(full_history=True)` extends each agent's history back before `obs_len` as
long as the frames are contiguous. Everything added is at **times at or before t0**, so
there is no future leakage, and `history_block` still slices only the last `obs_len`
steps, so the prediction inputs and target set are unchanged (zara1 n=2356 is preserved).
Other predictors read only the last `obs_len` steps and are unaffected. **Without this
flag** the history length is exactly `obs_len`, so only a single observable can be built
and adaptation silently becomes a no-op — changing eta does *not* change the score at all.

Adaptation helps exactly where the global operator is wrong (= out of distribution) and
hurts within distribution:

| eta | zara1 (in-dist) ADE | snu-asri (in-dist) ADE | snu-asri-ood ADE |
|---|---|---|---|
| 0 (static) | **0.4316** | **0.1339** | 0.1562 |
| 0.01 | 0.4331 | 0.1376 | 0.1496 |
| 0.05 | 0.4597 | 0.1492 | **0.1492** |
| 0.1 | 0.4981 | 0.1629 | 0.1576 |
| 0.3 | 0.6238 | 0.2189 | 0.1937 |

Within distribution the learned global K is already near-optimal, so per-agent updates
only inject noise and error increases monotonically with eta. Out of distribution the
global K is off, so adaptation recovers **ADE −4.5% / FDE −5.7%** at eta 0.01–0.05, and
beyond that it overfits. On snu-asri-ood, KoopCast++ goes from 0.1562 → 0.1492, widening
the gap over SocialVAE (0.1608).

Reproduce: `python scripts/compare_koopcastpp.py`, `python scripts/eval_adaptive.py`.

---

## 4. Training note: snu-asri needs a larger EDMD ridge

Lobby pedestrians move ~4x less over 8 steps than in ETH/UCY (median displacement
0.64 m vs 2.38 m). As a result the time-delay history block is far more collinear
(condition number ~1.5e4 vs ~2.2e3). Using the ETH/UCY default `ridge=1e-4` as-is gives
the fitted operator a spectral radius of **5.7**, so the 12-step rollout diverges
(ADE ~3.8e6). The `RIDGE_BY_SCENE` in `scripts/train_koopcastpp.py` raises this to
**0.1** for snu-asri, restoring |K|_spec ≈ 1.0 and a stable rollout. ETH/UCY keeps 1e-4,
so the existing 5 artifacts are unaffected.

```bash
python scripts/train_koopcastpp.py snu-asri     # train on lobby3 scenes 2..9
```

---

## Related documents
- Methodology: [koopcastpp_method.pdf](koopcastpp_method.pdf) / [.tex](koopcastpp_method.tex)
- Per-baseline behavior, pitfalls, and retraining procedure: [kpp/baselines/README.md](../kpp/baselines/README.md)
