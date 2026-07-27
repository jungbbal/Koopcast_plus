# kpp/predictors/ — ★ 우리 모델

우리가 설계·학습한 예측기가 있는 곳. 외부 baseline 은 여기가 아니라
[`kpp/baselines/`](../baselines/) 에 있습니다.

| 파일 | 무엇 |
|---|---|
| `base.py` | `Predictor` 인터페이스 — **모든** 모델(우리 것 + baseline)의 공통 계약 |
| `koopcastpp/` | ★ **KoopCast++ (ours)** — neighbour-aware Koopman 예측기 |
| `constant_velocity.py` | 등속 baseline. 우리 기여물이 아니라 파이프라인 sanity check |

```python
from kpp.predictors import KoopCastPP
m = KoopCastPP("zara1")           # zara1 held-out 으로 학습된 artifact 로드
```

## koopcastpp/
- `koopcastpp.py` — `Predictor` 구현 (외부에서 보는 얼굴)
- `_core.py` — observable / EDMD 적합 / rollout / 온라인 갱신 `adapt_K`
- `data/*.pt` — 학습된 artifact. scene 이름 = **held-out scene**
  (`koopcastpp_zara1.pt` = zara1 빼고 학습, zara1 로 평가)

`.consistency_v1.pt` 는 one-step Koopman residual loss 로 학습한 변형이고,
접미사 없는 쪽이 기본(multi-step prediction loss)입니다 —
`scripts/train_koopcastpp.py --loss` 참고.

학습: `python scripts/train_koopcastpp.py <scene>`
방법론: [docs/koopcastpp_method.pdf](../../docs/koopcastpp_method.pdf)
결과: [docs/RESULTS.md](../../docs/RESULTS.md)

## 새 predictor 추가
`Predictor` 를 상속해 `predict(obs) -> pred` (batched numpy) 만 구현하면
평가/제어 루프는 손댈 것이 없습니다.
