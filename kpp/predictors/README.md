# kpp/predictors/ — ★ our models

Where the predictors we designed and trained live. External baselines are not
here but in [`kpp/baselines/`](../baselines/).

| file | what |
|---|---|
| `base.py` | `Predictor` interface — the shared contract for **every** model (ours + baselines) |
| `koopcastpp/` | ★ **KoopCast++ (ours)** — neighbour-aware Koopman predictor |
| `constant_velocity.py` | Constant-velocity baseline. Not our contribution but a pipeline sanity check |

```python
from kpp.predictors import KoopCastPP
m = KoopCastPP("zara1")           # load the artifact trained with zara1 held out
```

## koopcastpp/
- `koopcastpp.py` — `Predictor` implementation (the outward-facing entry point)
- `_core.py` — observables / EDMD fit / rollout / online update `adapt_K`
- `data/*.pt` — trained artifacts. scene name = **held-out scene**
  (`koopcastpp_zara1.pt` = trained with zara1 excluded, evaluated on zara1)

`.consistency_v1.pt` is a variant trained with the one-step Koopman residual loss,
while the suffix-less one is the default (multi-step prediction loss) —
see `scripts/train_koopcastpp.py --loss`.

Training: `python scripts/train_koopcastpp.py <scene>`
Methodology: [docs/koopcastpp_method.pdf](../../docs/koopcastpp_method.pdf)
Results: [docs/RESULTS.md](../../docs/RESULTS.md)

## Adding a new predictor
Subclass `Predictor` and implement just `predict(obs) -> pred` (batched numpy);
the evaluation/control loops need no changes.
