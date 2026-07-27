# data/ — 데이터셋 (코드 없음)

이 저장소가 읽는 모든 궤적 데이터. 스크립트는 여기 두지 않습니다.

| 폴더 | 내용 | 누가 씀 |
|---|---|---|
| `raw/` | OpenTraj 원본 파일 (`obsmat.txt`, `H.txt`, `*.npy` …) | `kpp/data/loaders.py` → `load(key)` |
| `ethucy/` | ETH/UCY 공식 leave-one-out 스플릿 (Social-STGCNN / Social-GAN 배포본) | `load_ethucy(scene, split)` |
| `trajectron/` | Trajectron++ 학습용 dill `Environment` **생성물** | `kpp/baselines/trajectron_data.py` 가 생성 |

`raw/` 와 `ethucy/` 는 소스, `trajectron/` 은 파생물입니다 —
`python -m kpp.baselines.trajectron_data --out data/trajectron` 로 언제든 재생성.

데이터셋 목록과 좌표계는 저장소 루트 [README.md](../README.md#datasets),
스플릿의 함의는 [docs/RESULTS.md](../docs/RESULTS.md).
