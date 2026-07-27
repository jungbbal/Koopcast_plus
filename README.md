# koopcast_plusplus

Pedestrian trajectory prediction: **KoopCast++** (ours) benchmarked against
vendored external baselines on the ETH/UCY leave-one-out splits and on the
SNU-ASRI lobby dataset.

The repo is self-contained — no external checkouts are needed to load data,
train, predict, or score. Everything runs in-process.

The data layer follows OpenTraj's *narrow waist*: heterogeneous raw files are
read by per-dataset loaders that all converge to **one** in-memory table
(`TrajDataset`), after which every downstream step is dataset-agnostic.

```
raw files ──[loaders]──► TrajDataset ──[windows]──► numpy ──[predict]──► ADE/FDE
 (per-dataset, messy)      (one schema)    (8→12 pairs)   (one interface)
```

---

## 어디에 무엇이 있나 (한눈에)

```
koopcast_plusplus/
│
├─ data/                    ★ 데이터셋 — 이 저장소가 읽는 모든 원본/스플릿
│   ├─ raw/                   OpenTraj 원본 (eth hotel univ zara1 zara2 students001
│   │                         gc town-centre edinburgh pets wildtrack snu-asri ...)
│   ├─ ethucy/                ETH/UCY 공식 leave-one-out 스플릿 (학습·평가 기준)
│   └─ trajectron/            Trajectron++ 재학습용 dill Environment (생성물)
│
├─ kpp/                     ★ 라이브러리 (import 하는 것은 전부 여기)
│   ├─ data/                  로더 + TrajDataset + 윈도잉
│   ├─ predictors/          ★★ 우리 모델 — KoopCast++ 가 여기 있음
│   │   ├─ base.py              Predictor 인터페이스 (모든 모델의 공통 계약)
│   │   ├─ constant_velocity.py 등속 baseline (파이프라인 sanity check)
│   │   └─ koopcastpp/        ← ★ 우리 설계. 코드 + 학습된 가중치(.pt)
│   ├─ baselines/           ★★ 외부 baseline — 남의 모델, vendor 로 격리
│   │   ├─ adapters.py          우리 glue (Predictor 래퍼 + 가중치 매핑)
│   │   └─ vendor/              업스트림 CANVAS 코드 — 한 글자도 수정 안 함
│   └─ eval/                  ADE/FDE
│
├─ scripts/                 ★ 실행 진입점 (train_* / eval_*)
├─ runs/                    학습 산출물 (체크포인트·로그). 코드 아님
├─ docs/                    방법론 문서 + 실험 결과표
└─ external/                참고용 업스트림 원본. 파이프라인은 여기 쓰지 않음
```

**핵심 구분 세 줄**

| 보고 싶은 것 | 갈 곳 |
|---|---|
| 우리가 설계한 모델 | [kpp/predictors/koopcastpp/](kpp/predictors/koopcastpp/) |
| 비교 대상 (남의 모델) | [kpp/baselines/](kpp/baselines/) — 상세는 [baselines/README.md](kpp/baselines/README.md) |
| 데이터셋 | [data/](data/) — 스펙은 아래 [Datasets](#datasets) |
| 실험 결과 숫자 | [docs/RESULTS.md](docs/RESULTS.md) |

`kpp/predictors/` 와 `kpp/baselines/` 는 같은 `Predictor` 인터페이스를 구현하므로,
평가 코드 입장에서는 우리 모델과 baseline이 완전히 교체 가능합니다.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke.py          # 모든 데이터셋 로드 + CV baseline 채점 (의존성 numpy/pandas만)
```

```python
from kpp.data import load_ethucy
from kpp.predictors import KoopCastPP, ConstantVelocity
from kpp.eval import evaluate_scene

test = load_ethucy("zara1", "test")                    # zara1 을 held-out 으로
print(evaluate_scene(KoopCastPP("zara1"), test))       # 우리 모델
print(evaluate_scene(ConstantVelocity(pred_len=12), test))
```

Baseline 을 같은 자리에 끼워 넣으면 그대로 비교가 됩니다:

```python
from kpp.baselines import make_baseline
print(evaluate_scene(make_baseline("socialvae", "zara1"), test))
```

## Scripts

| script | 하는 일 |
|---|---|
| `scripts/smoke.py` | 전 데이터셋 로드 + CV 채점, 파이프라인 sanity check |
| `scripts/train_koopcastpp.py` | KoopCast++ 학습 (ETH/UCY LOO, snu-asri) |
| `scripts/eval_koopcastpp.py` | KoopCast++ vs ConstantVelocity, 동일 타깃 |
| `scripts/eval_baselines.py` | KoopCast++ vs 벤더 baseline 전체 비교표 |
| `scripts/eval_adaptive.py` | 온라인 적응(eta) — val 에서 튜닝, test 에서 보고 |
| `scripts/compare_koopcastpp.py` | static K vs online-update K 3-way 비교 |
| `scripts/train_trajectron.py` | Trajectron++ 재학습 (재현성 확인용) — **optional, 추가 데이터 필요** |
| `scripts/eval_trajectron.py` | Trajectron++ 채점 (pretrained / retrained) — **optional, 추가 데이터 필요** |

모두 저장소 루트에서 실행합니다: `python scripts/<name>.py`.

## 모델

### KoopCast++ (ours)
Neighbour-aware Koopman 예측기. time-delay-8 observable + social encoder + EDMD
연산자(디코더 없음). ETH/UCY leave-one-out 스플릿과 snu-asri lobby3 스플릿에서
재학습되어 있고, 학습된 가중치는 `kpp/predictors/koopcastpp/data/*.pt` 에 함께
들어 있습니다. 방법론은 [docs/koopcastpp_method.pdf](docs/koopcastpp_method.pdf).

관측 스트림에서 Koopman 연산자를 온라인으로 갱신하는 변형(`eta > 0`)도 있습니다 —
OOD에서 도움이 되고 in-distribution 에서는 해가 됩니다.

**현재 위치 (숫자는 [docs/RESULTS.md](docs/RESULTS.md)):**
ETH/UCY 5개 scene 평균에서 KoopCast++ 는 **등속 baseline 에 8% 뒤집니다**
(ADE 0.5932 vs 0.5495) — Social-STGCNN·EigenTrajectory 보다는 낫지만
SocialVAE·Trajectron++ 아래입니다. 강점은 **분포 밖 견고성** 쪽입니다:
snu-asri OOD 캡처에서 1위이고, 등속 baseline 을 뚜렷하게 이기는 유일한 학습
모델입니다. ETH/UCY 표를 빼고 OOD 결과만 제시하면 안 됩니다.

### Baselines (vendored)
`make_baseline(name, dataset)` 로 생성: `stgcnn`, `socialvae`, `eigen`,
`linear`, `gp`. 이들은 vendor 트리(`kpp/baselines/vendor/`)만으로 self-contained
하게 동작합니다.

> **⚠️ Trajectron++ 는 선택사항(optional)입니다.** 업스트림 결함 때문에 별도 경로
> (`kpp.baselines.trajectron_eval`)로 채점하며, **재학습용 dill Environment
> (`data/trajectron/`)가 추가로 필요**합니다. 이 디렉터리는 `.gitignore` 대상이라
> **저장소에 포함되지 않습니다** — 즉 clone 직후에는 `train_trajectron.py` /
> `eval_trajectron.py` 가 바로 돌아가지 않습니다. KoopCast++ 와 나머지 baseline
> (`stgcnn`/`socialvae`/`eigen`/`linear`/`gp`)만 쓸 경우 무시해도 됩니다. 이유와
> 데이터 생성 방법은 [kpp/baselines/README.md](kpp/baselines/README.md) 참고.

업스트림 파일은 **수정하지 않습니다.** 모든 적응은 `adapters.py` 쪽에서 외부적으로
이루어지므로 vendor 트리는 CANVAS 와 `diff` 가 깨끗합니다.

---

## Datasets

| key | source | reader | coords / notes |
|---|---|---|---|
| `eth` `hotel` `univ` `zara1` `zara2` | OpenTraj ETH/UCY `obsmat.txt` | `obsmat` | world (m), no homography needed |
| `students001` | OpenTraj UCY | `xyf_txt` | world (m) |
| `gc` | OpenTraj Grand Central | `gcs` | pixel→world homography + interp |
| `town-centre` | OpenTraj Oxford Town-Center | `town` | camera undistort+unproject (high variance) |
| `edinburgh` | OpenTraj Edinburgh Forum | `edinburgh` | per-track parse + homography |
| `pets2009-s2l1` | OpenTraj PETS-2009 | `pets` | Tsai camera calibration |
| `wildtrack` | OpenTraj WILDTRACK | `wildtrack` | multi-camera grid positions |
| `snu-asri` `snu-asri-ood` | custom | `taa_npy` | world (m) |

모든 리더는 ~2.5 Hz 예측 프로토콜(dt ≈ 0.4 s)로 다운샘플합니다(`fps` +
`sampling_rate`). ETH/UCY `obsmat` 는 이미 world 좌표이고, 나머지는 OpenTraj 의
homography / 카메라 모델을 적용합니다. 무거운 리더
(`gcs`/`town`/`edinburgh`/`pets`/`wildtrack`)는 `kpp/data/loaders_opentraj.py`
에 있으며 `scipy` + `opencv-python` 이 필요합니다.
SNU-ASRI fps 는 2.5 로 가정합니다 — `load("snu-asri", fps=...)` 로 덮어쓸 수 있습니다.

### ETH/UCY leave-one-out — 학습·비교의 기준
공식 Social-STGCNN / Social-GAN 스플릿이 `data/ethucy/` 에 들어 있습니다(우리가
만든 npy 변환이 아님). held-out scene 마다 `train`/`val`/`test` 디렉터리가 있고,
`train`/`val` 은 *나머지* scene 들, `test` 는 held-out scene 입니다.

```python
from kpp.data import load_ethucy
train = load_ethucy("zara1", "train")   # zara1 빼고 전부로 학습
test  = load_ethucy("zara1", "test")    # held-out zara1 로 평가
```

각 원본 파일이 자기 `scene_id` 를 가지므로 이웃 컨텍스트가 scene 을 넘지 않습니다.
이미 2.5 Hz 프로토콜이라 추가 다운샘플은 없습니다.

### snu-asri (lobby)
공식 `lobby3` 스플릿을 따릅니다 — scene 2..9 학습, 1 val, 0 test.
`data/raw/snu-asri/0.npy` 가 **test scene** 이고 학습 scene 은
`data/raw/snu-asri-train/` 에 있습니다. 자세한 내용은
[docs/RESULTS.md](docs/RESULTS.md).

### 데이터셋 추가하기
원본 파일을 `data/raw/<key>/` 에 넣고 `kpp/data/loaders.py:DATASETS` 에
`DatasetSpec` 한 줄을 추가합니다. 원본 레이아웃이 처음 보는 형태면 `load_*` 리더를
하나 추가하세요 — 파이프라인을 분기시키지 마세요.

## Predictor 추가하기
`Predictor` 인터페이스(`predict(obs) -> pred`, batched numpy)를 구현하면 `evaluate`
에 그대로 꽂힙니다. 평가/제어 루프는 아무것도 바꿀 필요가 없습니다.

## Notes
- **clone 직후 바로 되는 것**: KoopCast++(학습된 `.pt` 동봉) + `stgcnn`/`socialvae`/
  `eigen`/`linear`/`gp` baseline + 모든 데이터 로딩/채점. Trajectron++ 만 추가 데이터가
  필요한 예외입니다(위 Baselines 참고).
- `runs/` 와 `external/` 은 코드가 아닙니다 — 각각 학습 산출물, 참고용 업스트림 원본.
- `external/CANVAS-main/` 은 vendor 트리의 출처입니다. 파이프라인은 이 폴더를
  참조하지 않으므로, 지워도 예측 코드는 동작합니다(제어 트랙 코드가 여기에만 있음).
- 재작성 이전 스냅샷: `../koopcast_plusplus_backup_*.tar.gz`.
