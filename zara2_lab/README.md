# zara2_lab — 사고실험용 샌드박스

zara2 하나만 놓고 이것저것 굴려보는 공간. 저장소 안에 있지만 `kpp` 를 **import 하지 않습니다** —
데이터는 `../data/ethucy/zara2` 를 직접 읽고, 로더는 numpy만 쓰는 자체 구현.
본체 코드가 바뀌어도 여기 결과는 재현되고, 여기서 뭘 부수든 본체는 안전합니다.

```
zara2_lab/
  zlab/data.py             로더 · 윈도우 · ADE/FDE · sanity assert
  exp/                     실험 스크립트 (번호순, 하나당 하나의 질문)
  out/                     그림·표 출력물
```

데이터는 복사하지 않습니다. `zlab.DATA` 가 저장소의 `data/ethucy/zara2/{train,val,test}` 를
가리키므로, 그 경로가 움직이면 여기도 같이 고쳐야 합니다.

## "zara2 trainset" 은 두 가지를 뜻할 수 있습니다

leave-one-out 관례 때문에 이름이 헷갈립니다. 둘은 **다른 데이터**입니다:

| 호출 | 내용 | 언제 |
|---|---|---|
| `zlab.load("train")` | eth, hotel, zara1, **zara3**, students, uni — 즉 zara2를 제외한 나머지 | zara2 벤치마크 모델을 학습시킬 때 |
| `zlab.load_zara2_scene()` | `crowds_zara02.txt`, zara2 장면 그 자체 | zara2 라는 **장면의 성질**을 들여다볼 때 |

"표준적인 데이터셋이라 골랐다" 는 맥락이면 대개 후자(zara2 장면 자체)일 텐데,
확실치 않아 둘 다 열어뒀습니다. 실험마다 어느 쪽인지 명시해 두세요.

## 규격

세계 좌표 미터, 2.5 Hz (`dt = 0.4 s`). 벤치마크 프로토콜은 obs 8 (3.2초) → pred 12 (4.8초).
`windows()` 는 stride 1 슬라이딩이 기본 — 벤치마크 관례지만 윈도우가 심하게 겹칩니다.
통계량을 뽑을 땐 stride를 키워 대략 독립인 표본을 쓰세요.

## 현재 상태 (`python exp/00_sanity.py`)

| | rows | agents | windows | CV ADE | CV FDE |
|---|---|---|---|---|---|
| train (zara2 제외) | 32,208 | 1,233 | 12,702 | 0.546 | 1.210 |
| val | 11,819 | 475 | 4,262 | — | — |
| zara2 장면 | 6,541 | 149 | 3,789 | **0.322** | **0.724** |

CV = 마지막 관측 속도 등속 외삽. zara2 0.322 는 문헌값과 맞습니다 —
로더가 제대로 물려있다는 뜻이고, **앞으로 나오는 모든 숫자의 바닥선**입니다.

눈에 띄는 것 하나: zara2는 정지 프레임 비율이 28.8% 로 train 평균 9.3% 보다 훨씬 높습니다
(가게 앞에서 멈춰 서는 장면). 속도 분포도 훨씬 좁습니다 (p95 1.62 vs 2.58 m/s).
zara2가 "쉬운" 이유의 상당 부분이 여기 있을 가능성이 큽니다 — 사고실험 소재.

## 쓰는 법

```python
import zlab

sc = zlab.load_zara2_scene()          # Scene: frames, agents, xy
tracks = sc.tracks(min_len=20)        # agent_id -> (T, 2)
w = zlab.windows(sc, stride=4)        # (W, 20, 2)
obs, gt = zlab.split_obs_pred(w)      # (W,8,2), (W,12,2)
zlab.ade(pred, gt).mean()
```

새 실험은 `exp/NN_이름.py` 로, 맨 위에 **답하려는 질문 한 줄**을 적고 시작하세요.
