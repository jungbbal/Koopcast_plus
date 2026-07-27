# 실험 결과

모든 숫자는 single-shot ADE/FDE (meters), 8 observed → 12 predicted, dt 0.4 s.
같은 scene 안에서는 모든 모델이 **동일한 (agent, t0) 타깃 집합**을 봅니다
(`kpp.eval.evaluate_scene`, neighbour-aware windows).

재현 명령은 각 표 위에 적어 두었습니다.

---

## 1. ETH/UCY leave-one-out

각 scene 은 held-out — 그 scene 을 뺀 나머지로 학습하고 그 scene 으로 평가합니다
(`load_ethucy(scene, "test")`).

### 1.1 KoopCast++ vs baselines

`python scripts/eval_adaptive.py` 출력, 5개 scene 전부
(원본 로그: `runs/adaptive_ethucy_all.log`).

ADE / FDE, 낮을수록 좋음. 열 내 최고값 **굵게**.

| model | eth | hotel | univ | zara1 | zara2 | **AVG** |
|---|---|---|---|---|---|---|
| **KoopCast++ (ours, static)** | 1.0649 / 2.1794 | 0.4882 / 1.0225 | 0.6271 / 1.3556 | 0.4316 / 0.9606 | 0.3544 / 0.7895 | 0.5932 / 1.2615 |
| **KoopCast++ (ours, adaptive)** | 1.0649 / 2.1794 | 0.4808 / 1.0055 | 0.6305 / 1.3662 | 0.4316 / 0.9606 | 0.3544 / 0.7895 | 0.5924 / 1.2602 |
| ConstantVelocity | 1.0755 / 2.2819 | **0.3194 / 0.6142** | **0.6036 / 1.3386** | 0.4272 / 0.9524 | **0.3219 / 0.7238** | 0.5495 / 1.1822 |
| SocialVAE | **0.9801 / 1.9405** | 0.3424 / 0.6630 | 0.6158 / 1.3458 | 0.4611 / 1.0230 | 0.3452 / 0.7559 | **0.5489 / 1.1456** |
| Trajectron++ (pretrained) † | 1.0371 / 2.1439 | 0.3551 / 0.6849 | 0.6374 / 1.4182 | **0.4257 / 0.9393** | 0.3271 / 0.7316 | 0.5565 / 1.1836 |
| Social-STGCNN | 1.2891 / 2.3337 | 0.6998 / 1.3807 | 0.7607 / 1.5072 | 0.4991 / 1.0329 | 0.4502 / 0.9194 | 0.7398 / 1.4348 |
| EigenTrajectory | 1.3180 / 2.5586 | 0.9759 / 1.8149 | 0.7036 / 1.4390 | 0.7256 / 1.5515 | 0.9463 / 1.8398 | 0.9339 / 1.8408 |

† Trajectron++ 는 업스트림 결함 때문에 `evaluate_scene` 이 아닌
`evaluate_trajectron` 으로 채점합니다 (이유:
[kpp/baselines/README.md](../kpp/baselines/README.md)). 타깃 개수는 다른 모델과
정확히 일치하지만(zara1 n=2356) 경로가 다르다는 점은 감안해서 읽으세요.
재학습본은 §1.2 참조.

**순위 (AVG ADE):** SocialVAE 0.5489 < ConstantVelocity 0.5495 <
Trajectron++ 0.5565 < **KoopCast++ 0.5932** < Social-STGCNN 0.7398 <
EigenTrajectory 0.9339

**scene 별 KoopCast++ vs ConstantVelocity:**

| scene | KoopCast++ | CV | 차이 |
|---|---|---|---|
| eth | 1.0649 | 1.0755 | **−1.0%** (유일한 승) |
| hotel | 0.4882 | 0.3194 | +52.8% |
| univ | 0.6271 | 0.6036 | +3.9% |
| zara1 | 0.4316 | 0.4272 | +1.0% |
| zara2 | 0.3544 | 0.3219 | +10.1% |
| **AVG** | **0.5932** | **0.5495** | **+8.0%** |

`adaptive` 열의 eta 는 **val 스플릿에서** 고릅니다 (test-set tuning 아님).
선택된 값: eth 0.0, hotel 0.005, univ 0.005, zara1 0.0, zara2 0.0.

### ⚠️ 정직하게 읽어야 할 점

**ETH/UCY 에서 KoopCast++ 는 등속 baseline 에 지고 있습니다.** 5개 scene 평균
ADE 0.5932 vs ConstantVelocity 0.5495 (**8.0% 나쁨**). scene 별로도 eth 하나만
간신히 이기고 (1.0649 vs 1.0755) hotel/univ/zara1/zara2 는 전부 집니다. hotel 이
가장 크게 벌어집니다 (0.4882 vs 0.3194, 53% 나쁨).

SocialVAE(0.5489)와 Trajectron++ pretrained(0.5565, §1.2)가 이 벤치마크의 상위권이고
KoopCast++ 는 Social-STGCNN(0.7398)·EigenTrajectory(0.9339)보다는 확실히 낫지만
등속 모델 아래입니다.

**온라인 적응은 ETH/UCY 에서 사실상 무효과입니다.** val 이 5개 scene 중 3개에서
eta=0 을 고르고, 나머지 hotel/univ 에서도 gain 이 ±0.5% 수준입니다. 이는 §3 의
분석과 일관됩니다 — 적응은 분포 안에서 도움이 안 되도록 설계된 것이 아니라,
분포 안에서는 실제로 **해가 됩니다**.

따라서 현재 KoopCast++ 의 근거는 "ETH/UCY SOTA" 가 아니라 **§2 의 OOD 견고성**
입니다 (snu-asri-ood 에서 1위, 등속 대비 우위). ETH/UCY 표는 그 주장을 뒷받침하는
게 아니라 제약합니다 — 대외 발표 시 이 표를 빼지 말고 함께 제시해야 합니다.

### 1.2 Trajectron++ — pretrained vs 재학습 (재현성 확인)

`python scripts/eval_trajectron.py --both`. Trajectron++ 는 업스트림 결함 때문에
`evaluate_scene` 이 아닌 `evaluate_trajectron` 으로 채점합니다 (이유:
[kpp/baselines/README.md](../kpp/baselines/README.md)). 타깃 개수는 다른
baseline 과 정확히 일치합니다 (zara1: n=2356).

| scene | pretrained ADE/FDE | retrained ADE/FDE | ΔADE |
|---|---|---|---|
| eth | 1.0371 / 2.1439 | 1.0727 / 2.2572 | +3.4% |
| hotel | 0.3551 / 0.6849 | 0.4200 / 0.8461 | +18.3% |
| univ | 0.6374 / 1.4182 | 0.6865 / 1.5021 | +7.7% |
| zara1 | 0.4257 / 0.9393 | **0.3883 / 0.8380** | −8.8% |
| zara2 | 0.3271 / 0.7316 | **0.3178 / 0.6959** | −2.8% |
| **AVG** | **0.5565 / 1.1836** | **0.5771 / 1.2279** | +3.7% |

재학습이 배포된 체크포인트를 평균 몇 % 이내로 재현하고 zara1/zara2 에서는 오히려
이깁니다. 편차의 부호가 scene 마다 뒤집히므로 체계적인 전처리 오류가 아니라
scene 단위 seed variance 로 봅니다. hotel 이 가장 약한 재현(+18%)인데 ETH/UCY
에서 가장 작은 scene 이라 가장 noisy 합니다. **배포 체크포인트를 기본으로,
재학습본은 재현성 확인용으로** 취급하세요.

재학습본 위치: `runs/trajectron/<scene>/models_<timestamp><scene>/`.
비용: GPU 1장 기준 ~2분/epoch → scene 당 100 epoch 에 ~3.5시간 (5개 전부 ~17시간).

---

## 2. snu-asri (lobby)

snu-asri 는 `lobby3` 데이터셋으로 **공식 스플릿**이 있습니다
(`/home/jungbbal/ood/lobby3/`): scene **2..9 train**, **1 val**, **0 test**.

이 저장소가 `snu-asri` 라 부르는 `data/raw/snu-asri/0.npy` 는 **test scene** 입니다
— `lobby3/test/0.npy` 와 byte-identical (md5 `aaef2ed3599b0d66510ab8a7887967fb`).
`snu-asri-ood.npy` 도 `lobby3_ood/test/scene4_test.npy` 와 동일합니다. 학습 scene
들은 `data/raw/snu-asri-train/` 에 복사되어 있습니다 (`SOURCE.txt` 참고).

벤더 CANVAS lobby 체크포인트도 이 스플릿을 따르므로, KoopCast++ 를 scene 2..9 로
학습하고 scene 0 으로 채점하면 모든 모델이 동일 조건에 놓입니다.

> 이 파일의 이전 버전은 snu-asri 에 공식 스플릿이 *없다*고 보고 scene 0 을 70/30
> 시간 분할해서 썼습니다. 그건 KoopCast++ 를 **test scene 일부로 학습**시켜
> 점수를 부풀린 것이었고, 함께 붙어 있던 "CANVAS baseline 이 leak 일 수 있다"는
> 경고는 정확히 반대였습니다. 둘 다 여기서 바로잡았습니다.

official test scene (n=14992) 과 별도 OOD 캡처 (n=135). 아래 모든 모델은 lobby3
scene 2..9 로 학습되었고 scene 0 을 본 적이 없습니다:

| model | test ADE/FDE | ood ADE/FDE |
|---|---|---|
| SocialVAE | **0.1292 / 0.2458** | 0.1608 / 0.3388 |
| KoopCast++ (ours) | 0.1339 / 0.2533 | **0.1562 / 0.3331** |
| Social-STGCNN | 0.1495 / 0.2759 | 0.2446 / 0.4877 |
| ConstantVelocity | 0.1636 / 0.3105 | 0.1644 / 0.3466 |
| EigenTrajectory | 0.1894 / 0.3537 | 0.3206 / 0.6135 |

KoopCast++ 는 in-distribution test scene 에서 2위(SocialVAE 대비 3.6% 뒤)이고
**OOD 캡처에서 1위** 입니다 — 거기서 등속 baseline 을 뚜렷한 차이로 이기는 유일한
학습 모델이며, Social-STGCNN 과 EigenTrajectory 는 분포를 벗어나자 크게
무너집니다.

---

## 3. KoopCast++ 온라인 적응 (eta > 0)

KoopCast++ 는 관측 스트림으로부터 Koopman 연산자를 온라인 갱신할 수 있습니다
(`_core.adapt_K`, agent 당 rank-1). `obs_len` 보다 **긴** 히스토리가 들어올 때만
동작하므로 `full_history=True` 로 평가해야 합니다:

```python
evaluate_scene(KoopCastPP("snu-asri", eta=0.05), ds, full_history=True)
```

`scene_windows(full_history=True)` 는 프레임이 연속인 한 각 agent 의 히스토리를
`obs_len` 이전으로 확장합니다. 추가되는 것은 전부 **t0 이하 시점**이라 미래 누수가
없고, `history_block` 은 여전히 마지막 `obs_len` 스텝만 잘라 쓰므로 예측 입력과
타깃 집합은 그대로입니다 (zara1 n=2356 유지). 다른 predictor 는 마지막 `obs_len`
만 읽으므로 영향을 받지 않습니다. **이 플래그가 없으면** 히스토리 길이가 정확히
`obs_len` 이라 observable 을 하나밖에 못 만들고, 적응은 조용히 no-op 이 됩니다 —
eta 를 바꿔도 점수가 *전혀* 변하지 않습니다.

적응은 전역 연산자가 틀린 곳(=분포 밖)에서 정확히 도움이 되고, 분포 안에서는
해가 됩니다:

| eta | zara1 (in-dist) ADE | snu-asri (in-dist) ADE | snu-asri-ood ADE |
|---|---|---|---|
| 0 (static) | **0.4316** | **0.1339** | 0.1562 |
| 0.01 | 0.4331 | 0.1376 | 0.1496 |
| 0.05 | 0.4597 | 0.1492 | **0.1492** |
| 0.1 | 0.4981 | 0.1629 | 0.1576 |
| 0.3 | 0.6238 | 0.2189 | 0.1937 |

분포 안에서는 학습된 전역 K 가 이미 거의 최적이라 agent 별 갱신은 노이즈만 넣고
오차가 eta 에 대해 단조 증가합니다. 분포 밖에서는 전역 K 가 어긋나 있어 적응이
eta 0.01–0.05 에서 **ADE −4.5% / FDE −5.7%** 를 회복하고, 그 너머로는 과적응합니다.
snu-asri-ood 에서 KoopCast++ 는 0.1562 → 0.1492 가 되어 SocialVAE(0.1608) 와의
격차를 벌립니다.

재현: `python scripts/compare_koopcastpp.py`, `python scripts/eval_adaptive.py`.

---

## 4. 학습 시 주의: snu-asri 는 더 큰 EDMD ridge 가 필요

로비 보행자는 8 스텝 동안 ETH/UCY 대비 ~4배 덜 움직입니다 (median displacement
0.64 m vs 2.38 m). 그래서 time-delay history block 이 훨씬 더 공선적입니다
(condition number ~1.5e4 vs ~2.2e3). ETH/UCY 기본값 `ridge=1e-4` 를 그대로 쓰면
적합된 연산자의 spectral radius 가 **5.7** 이 되어 12-스텝 rollout 이 발산합니다
(ADE ~3.8e6). `scripts/train_koopcastpp.py` 의 `RIDGE_BY_SCENE` 이 snu-asri 에
대해 이를 **0.1** 로 올려 |K|_spec ≈ 1.0 과 안정적인 rollout 을 복원합니다.
ETH/UCY 는 1e-4 를 유지하므로 기존 5개 artifact 는 영향을 받지 않습니다.

```bash
python scripts/train_koopcastpp.py snu-asri     # lobby3 scene 2..9 로 학습
```

---

## 관련 문서
- 방법론: [koopcastpp_method.pdf](koopcastpp_method.pdf) / [.tex](koopcastpp_method.tex)
- baseline 별 동작·함정·재학습 절차: [kpp/baselines/README.md](../kpp/baselines/README.md)
