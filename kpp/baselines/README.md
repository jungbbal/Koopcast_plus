# kpp/baselines

External trajectory-prediction baselines, **vendored** into this repo so
KoopCast++ can be compared apples-to-apples without any external checkout.

```
baselines/
├─ adapters.py        # our glue: thin Predictor wrappers + dataset→weight maps
├─ __init__.py        # make_baseline(name, dataset) factory
└─ vendor/            # UPSTREAM CODE — byte-for-byte unmodified
   └─ canvas/predictors/
      ├─ linear_predictor.py         CANVAS constant-turn-rate baseline
      ├─ gp_predictor.py             CANVAS Gaussian-process baseline
      ├─ Social_STGCNN/  (+checkpoint/)   Social-STGCNN + weights
      ├─ SocialVAE/      (+config/ models/) SocialVAE + weights
      ├─ eigen/          (+models/ json_files/) EigenTrajectory + weights
      └─ trajectron/                 Trajectron++ (weights in ../assets/)
   └─ assets/models/trajectron/       Trajectron++ pretrained checkpoints
```

## Provenance
Source: `../CANVAS-main/canvas/predictors/` (local CANVAS checkout).
Copied on 2026-07-19 with `rsync -a --exclude=__pycache__`. The `koopcast`
predictor was intentionally dropped (superseded by our `kpp` KoopCast++).

**The upstream files are not edited.** Every model/algorithm file is identical
to CANVAS (`diff -q` clean). All adaptation is external:
- dataset→weight-path mapping lives in `adapters.py`, not in the vendored code;
- each vendored predictor already self-locates its weights via
  `os.path.dirname(__file__)`, so moving the tree here "just works";
- `vendor/` is put on `sys.path` from `adapters.py` — the path points *inside*
  this repo, so the "no external checkout" property still holds.

## Design fit
Every CANVAS predictor's native call signature is
`__call__({id: (T,2)}) -> {id: (P,2)}`, which is exactly kpp's
`Predictor.predict_scene`. Our `scene_windows` always hands over full
`(obs_len, 2)` histories, so the wrapper feeds the dict straight through.

## Status

| baseline | `make_baseline` key | weights | works | notes |
|---|---|---|---|---|
| Social-STGCNN | `stgcnn` | ✅ 6 scenes | ✅ | deterministic single-shot |
| SocialVAE | `socialvae` | ✅ 7 scenes | ✅ | single sample per agent |
| EigenTrajectory | `eigen` | ✅ 6 scenes | ✅ | over Social-STGCNN backbone |
| CANVAS-Linear | `linear` | — | ✅ | needs `history_len = obs_len-2` (see below) |
| CANVAS-GP | `gp` | — | ✅ | needs `_george_compat()` + `history_len = obs_len-1`; ~38 ms/agent |
| Trajectron++ | `trajectron` | ✅ 6 scenes | ✅* | *via `trajectron_eval`, not `make_baseline` (see below) |

### Silent-fallback trap (linear / gp)
Both upstream predictors run their model only when the history is *longer* than
`history_len` (`>= history_len+2` for linear, `>= history_len+1` for GP) and
otherwise **silently return the last position repeated**. Our scene windows carry
exactly `obs_len` steps, so the adapters pass a reduced `history_len` to keep them
on the real code path. Without this they score like a "hold position" predictor
(e.g. zara1 ADE 2.50 instead of 1.03) while looking like they ran fine.

### Trajectron++ — working, but only via the offline path
Use `kpp.baselines.trajectron_eval.evaluate_trajectron(scene)`, **not**
`make_baseline("trajectron", ...)`. Three upstream defects had to be worked
around, all from our side — no vendored file was edited:

1. **`ModelRegistrar.load_models` is a no-op.** Its body is commented out
   (`model_registrar.py:60-72`), so it only calls `model_dict.clear()` and every
   submodule comes back freshly initialised. The net then predicts a
   near-stationary trajectory — zara1 ADE **2.44** (≈ "hold position") instead of
   **0.43**. `trajectron_eval.load_trajectron()` populates the registrar itself.
2. **Checkpoints won't unpickle in the vendored namespace.** They were pickled
   when `model.*`/`environment.*` sat at the import root, so `torch.load` raises
   `ModuleNotFoundError: No module named 'model'`. `_install_module_aliases()`
   maps the old names onto `canvas.predictors.trajectron.*`.
3. **The live path can't be used at all.** `get_timesteps_data` forwards
   `min_future_timesteps=max_ft` (the prediction horizon) where the caller's
   `min_ft=0` was intended, so a node needs 12 steps of *future ground truth* to
   even be a candidate. That never holds for history-only inference — hence the
   offline `Environment` route via `trajectron_data.py`. CANVAS' glue also
   hardcodes `dt=0.1`, wrong for our 2.5 Hz protocol.

Because of (3) Trajectron++ is scored by `evaluate_trajectron` rather than
`evaluate_scene`; the target count matches the other baselines exactly
(zara1: n=2356), so the numbers stay comparable.

### Retraining Trajectron++
Working and verified end-to-end. `trajectron_data.py` dumps the `dill`
Environments `train.py` expects; `scripts/train_trajectron.py` drives training:

```bash
python -m kpp.baselines.trajectron_data --out data/trajectron   # once
python scripts/train_trajectron.py zara1 --epochs 100 --device cuda:0
```

Checkpoints land in `runs/trajectron/<scene>/models_<timestamp><scene>/` as
`model_registrar-<epoch>.pt` + `config.json`, which feeds straight back in:
```python
evaluate_trajectron("zara1", model_dir="runs/trajectron/zara1/models_...", ts=100)
```

**Full retraining result** — all five scenes were retrained from scratch for 100
epochs on our splits and scored with `scripts/eval_trajectron.py --both`.
Retraining reproduces the shipped checkpoints to within a few percent on average
(AVG ADE 0.5565 → 0.5771) and *beats* them on zara1/zara2. Treat the pretrained
checkpoints as the default and the retrained ones as the reproducibility check.
**Per-scene numbers: [docs/RESULTS.md](../../docs/RESULTS.md#12-trajectron--pretrained-vs-retrained-reproducibility-check).**

Retrained checkpoints live in `runs/trajectron/<scene>/models_<timestamp><scene>/`.

An earlier 1-epoch round-trip (zara1 ADE 0.688) validated the loop before the
full run.

Two more upstream defects had to be worked around in the wrapper (again without
editing vendored files): `trajectron/visualization` imports `cp.adaptive_cp`,
which **exists nowhere in CANVAS**, and `train.py` uses root-level imports
(`from model.trajectron import ...`) while the model files use subpackage
relative imports (`from ...utils import ...`) — mutually exclusive, so upstream
`train.py` cannot run as shipped. The wrapper stubs `cp` and aliases the
subpackages under their root-level names.

Cost: ~2 min/epoch on one GPU → ~3.5 h per scene at 100 epochs (~17 h for all
five). Rarely worth it for ETH/UCY, since the shipped checkpoints were trained
on these same leave-one-out splits; it matters for datasets with no released
weights.

Weights map to the ETH/UCY leave-one-out convention: the checkpoint named for a
scene is the model trained on the *other* scenes and tested on that held-out
scene — matching `load_ethucy(scene, "test")`.

## snu-asri (lobby) / KoopCast++ results

The description of the official snu-asri split, the snu-asri results table, the
KoopCast++ online adaptation (eta) results, and the snu-asri-specific EDMD ridge
issue have been moved to **[docs/RESULTS.md](../../docs/RESULTS.md)** — to gather
all experiment numbers in one place.

## Usage
```python
from kpp.baselines import make_baseline
from kpp.data import load_ethucy
from kpp.eval import evaluate_scene

m = make_baseline("socialvae", "zara1")
print(evaluate_scene(m, load_ethucy("zara1", "test")))
```
Or run the whole comparison table: `python scripts/eval_baselines.py`.

## Extra dependencies
Beyond the kpp core (numpy/pandas), these baselines need: `torch`,
`networkx` (STGCNN/Trajectron), `george` (GP only). They are imported lazily —
importing `kpp` core stays numpy/pandas-only; the heavy deps load only when a
baseline is instantiated.
