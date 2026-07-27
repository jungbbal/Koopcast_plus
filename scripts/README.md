# scripts/ — 실행 진입점

전부 저장소 루트에서 `python scripts/<name>.py` 로 실행. 각 파일 상단 docstring 에
정확한 usage 가 있습니다.

| script | 하는 일 |
|---|---|
| `smoke.py` | 전 데이터셋 로드 + CV 채점. 의존성 numpy/pandas 만 — **여기서 시작** |
| `train_koopcastpp.py` | KoopCast++ 학습 (ETH/UCY LOO, snu-asri) |
| `eval_koopcastpp.py` | KoopCast++ vs ConstantVelocity, 동일 타깃 |
| `eval_baselines.py` | KoopCast++ vs 벤더 baseline 전체 비교표 |
| `eval_adaptive.py` | 온라인 적응 eta — val 에서 튜닝, test 에서 보고 |
| `compare_koopcastpp.py` | static K vs online-update K 3-way |
| `train_trajectron.py` | Trajectron++ 재학습 (재현성 확인) |
| `eval_trajectron.py` | Trajectron++ 채점 (pretrained / retrained) |

산출물은 `runs/` 로, 결과 숫자는 [docs/RESULTS.md](../docs/RESULTS.md) 로 갑니다.
