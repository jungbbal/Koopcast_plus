# scripts/ — execution entry points

Run everything from the repo root with `python scripts/<name>.py`. The docstring
at the top of each file gives the exact usage.

| script | what it does |
|---|---|
| `smoke.py` | Load all datasets + CV scoring. numpy/pandas dependencies only — **start here** |
| `train_koopcastpp.py` | Train KoopCast++ (ETH/UCY LOO, snu-asri) |
| `eval_koopcastpp.py` | KoopCast++ vs ConstantVelocity, same targets |
| `eval_baselines.py` | KoopCast++ vs full vendored-baseline comparison table |
| `eval_adaptive.py` | Online adaptation eta — tuned on val, reported on test |
| `compare_koopcastpp.py` | static K vs online-update K, 3-way |
| `train_trajectron.py` | Retrain Trajectron++ (reproducibility check) |
| `eval_trajectron.py` | Score Trajectron++ (pretrained / retrained) |

Artifacts go to `runs/`, and result numbers go to [docs/RESULTS.md](../docs/RESULTS.md).
