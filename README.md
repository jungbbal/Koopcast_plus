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

## Where everything lives (at a glance)

```
koopcast_plusplus/
│
├─ data/                    ★ datasets — every raw file / split this repo reads
│   ├─ raw/                   OpenTraj raw data (eth hotel univ zara1 zara2 students001
│   │                         gc town-centre edinburgh pets wildtrack snu-asri ...)
│   ├─ ethucy/                official ETH/UCY leave-one-out splits (train/eval reference)
│   └─ trajectron/            dill Environment for retraining Trajectron++ (generated)
│
├─ kpp/                     ★ the library (everything you import lives here)
│   ├─ data/                  loaders + TrajDataset + windowing
│   ├─ predictors/          ★★ our model — KoopCast++ is here
│   │   ├─ base.py              Predictor interface (the common contract for every model)
│   │   ├─ constant_velocity.py constant-velocity baseline (pipeline sanity check)
│   │   └─ koopcastpp/        ← ★ our design. code + trained weights (.pt)
│   ├─ baselines/           ★★ external baselines — other people's models, isolated under vendor/
│   │   ├─ adapters.py          our glue (Predictor wrapper + weight mapping)
│   │   └─ vendor/              upstream CANVAS code — not modified by a single character
│   └─ eval/                  ADE/FDE
│
├─ scripts/                 ★ entry points (train_* / eval_*)
├─ runs/                    training artifacts (checkpoints/logs). Not code
├─ docs/                    method writeup + results tables
└─ external/                upstream originals, for reference. The pipeline does not use this
```

**Three-line orientation**

| What you want to see | Where to go |
|---|---|
| The model we designed | [kpp/predictors/koopcastpp/](kpp/predictors/koopcastpp/) |
| The comparison targets (other people's models) | [kpp/baselines/](kpp/baselines/) — details in [baselines/README.md](kpp/baselines/README.md) |
| Datasets | [data/](data/) — spec in [Datasets](#datasets) below |
| The result numbers | [docs/RESULTS.md](docs/RESULTS.md) |

`kpp/predictors/` and `kpp/baselines/` implement the same `Predictor` interface,
so from the evaluation code's point of view our model and any baseline are fully
interchangeable.

---

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke.py          # load every dataset + score the CV baseline (only needs numpy/pandas)
```

```python
from kpp.data import load_ethucy
from kpp.predictors import KoopCastPP, ConstantVelocity
from kpp.eval import evaluate_scene

test = load_ethucy("zara1", "test")                    # zara1 held out
print(evaluate_scene(KoopCastPP("zara1"), test))       # our model
print(evaluate_scene(ConstantVelocity(pred_len=12), test))
```

Drop a baseline into the same slot and it compares directly:

```python
from kpp.baselines import make_baseline
print(evaluate_scene(make_baseline("socialvae", "zara1"), test))
```

## Scripts

| script | what it does |
|---|---|
| `scripts/smoke.py` | load every dataset + score CV; pipeline sanity check |
| `scripts/train_koopcastpp.py` | train KoopCast++ (ETH/UCY LOO, snu-asri) |
| `scripts/eval_koopcastpp.py` | KoopCast++ vs ConstantVelocity, same target |
| `scripts/eval_baselines.py` | full comparison table: KoopCast++ vs vendored baselines |
| `scripts/eval_adaptive.py` | online adaptation (eta) — tune on val, report on test |
| `scripts/compare_koopcastpp.py` | 3-way comparison: static K vs online-update K |
| `scripts/train_trajectron.py` | retrain Trajectron++ (reproducibility check) — **optional, needs extra data** |
| `scripts/eval_trajectron.py` | score Trajectron++ (pretrained / retrained) — **optional, needs extra data** |

Run all of them from the repo root: `python scripts/<name>.py`.

## Models

### KoopCast++ (ours)
A neighbour-aware Koopman predictor: time-delay-8 observable + social encoder +
EDMD operator (no decoder). It is retrained on the ETH/UCY leave-one-out splits
and on the snu-asri lobby3 split, and the trained weights ship alongside the code
in `kpp/predictors/koopcastpp/data/*.pt`. Method writeup:
[docs/koopcastpp_method.pdf](docs/koopcastpp_method.pdf).

There is also a variant that updates the Koopman operator online from the
observation stream (`eta > 0`) — it helps out-of-distribution and hurts
in-distribution.

**Where it currently stands (numbers in [docs/RESULTS.md](docs/RESULTS.md)):**
Averaged over the 5 ETH/UCY scenes, KoopCast++ is **8% behind the
constant-velocity baseline** (ADE 0.5932 vs 0.5495) — better than Social-STGCNN
and EigenTrajectory, but below SocialVAE and Trajectron++. Its strength is
**out-of-distribution robustness**: it ranks first on the snu-asri OOD capture and
is the only learned model that clearly beats the constant-velocity baseline there.
Do not present the OOD result while omitting the ETH/UCY table.

### Baselines (vendored)
Created via `make_baseline(name, dataset)`: `stgcnn`, `socialvae`, `eigen`,
`linear`, `gp`. These are self-contained on the vendor tree
(`kpp/baselines/vendor/`) alone.

> **⚠️ Trajectron++ is optional.** Because of an upstream defect it is scored
> through a separate path (`kpp.baselines.trajectron_eval`), and it additionally
> **needs the retraining dill Environment (`data/trajectron/`)**. That directory
> is covered by `.gitignore` and is **not included in the repo** — so right after
> a clone, `train_trajectron.py` / `eval_trajectron.py` will not run out of the
> box. If you only use KoopCast++ and the other baselines
> (`stgcnn`/`socialvae`/`eigen`/`linear`/`gp`) you can ignore it. See
> [kpp/baselines/README.md](kpp/baselines/README.md) for the reason and how to
> generate the data.

Upstream files are **not modified.** All adaptation happens externally in
`adapters.py`, so the vendor tree stays `diff`-clean against CANVAS.

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

Every reader downsamples to the ~2.5 Hz prediction protocol (dt ≈ 0.4 s) via
`fps` + `sampling_rate`. ETH/UCY `obsmat` is already in world coordinates; the
rest apply OpenTraj's homography / camera models. The heavy readers
(`gcs`/`town`/`edinburgh`/`pets`/`wildtrack`) live in
`kpp/data/loaders_opentraj.py` and require `scipy` + `opencv-python`.
SNU-ASRI fps is assumed to be 2.5 — override it with `load("snu-asri", fps=...)`.

> Note: the Grand Central (`gc`) raw data is **not shipped** with this repo — it
> is 12,000+ per-frame files and is not needed by the main benchmark. The `gc`
> loader is still present; drop the OpenTraj originals into `data/raw/gc/` to use
> it.

### ETH/UCY leave-one-out — the training/comparison reference
The official Social-STGCNN / Social-GAN splits are in `data/ethucy/` (not an npy
conversion we made). Each held-out scene has `train`/`val`/`test` directories,
where `train`/`val` are the *other* scenes and `test` is the held-out scene.

```python
from kpp.data import load_ethucy
train = load_ethucy("zara1", "train")   # train on everything except zara1
test  = load_ethucy("zara1", "test")    # evaluate on held-out zara1
```

Since each raw file carries its own `scene_id`, neighbour context never crosses
scenes. It is already on the 2.5 Hz protocol, so there is no extra downsampling.

### snu-asri (lobby)
Follows the official `lobby3` split — scenes 2..9 train, 1 val, 0 test.
`data/raw/snu-asri/0.npy` is the **test scene** and the training scenes are in
`data/raw/snu-asri-train/`. See [docs/RESULTS.md](docs/RESULTS.md) for details.

### Adding a dataset
Drop the raw files into `data/raw/<key>/` and add one `DatasetSpec` line to
`kpp/data/loaders.py:DATASETS`. If the raw layout is one you have not seen before,
add a `load_*` reader — do not branch the pipeline.

## Adding a predictor
Implement the `Predictor` interface (`predict(obs) -> pred`, batched numpy) and it
plugs straight into `evaluate`. The evaluation/control loops need no changes.

## Notes
- **Works right after a clone**: KoopCast++ (trained `.pt` included) +
  `stgcnn`/`socialvae`/`eigen`/`linear`/`gp` baselines + all data loading/scoring.
  Trajectron++ is the only exception that needs extra data (see Baselines above).
- `runs/` and `external/` are not code — training artifacts and reference upstream
  originals, respectively.
- `external/CANVAS-main/` is the source of the vendor tree. The pipeline does not
  reference this folder, so the prediction code works even if you delete it (only
  the control-track code lives there).
