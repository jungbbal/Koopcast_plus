# zara2_lab — a sandbox for thought experiments

A space for messing around with zara2 alone. It lives inside the repo but does **not import** `kpp` —
it reads the data directly from `../data/ethucy/zara2`, and the loader is a self-contained implementation that uses only numpy.
Results here stay reproducible even when the main codebase changes, and whatever you break here, the main codebase stays safe.

```
zara2_lab/
  zlab/data.py             loader · windows · ADE/FDE · sanity assert
  exp/                     experiment scripts (numbered, one question each)
  out/                     figure and table outputs
```

The data is not copied. `zlab.DATA` points to the repo's `data/ethucy/zara2/{train,val,test}`,
so if that path moves, this has to be fixed along with it.

## "zara2 trainset" can mean two things

The leave-one-out convention makes the name confusing. The two are **different data**:

| call | contents | when |
|---|---|---|
| `zlab.load("train")` | eth, hotel, zara1, **zara3**, students, uni — i.e. everything except zara2 | when training the zara2 benchmark model |
| `zlab.load_zara2_scene()` | `crowds_zara02.txt`, the zara2 scene itself | when examining the **properties of the scene** called zara2 |

If the context is "chosen because it's a standard dataset," it usually means the latter (the zara2 scene itself),
but since it isn't certain, both are left open. State which one each experiment uses.

## Specifications

World coordinates in metres, 2.5 Hz (`dt = 0.4 s`). The benchmark protocol is obs 8 (3.2 s) → pred 12 (4.8 s).
`windows()` defaults to stride-1 sliding — the benchmark convention, but the windows overlap heavily.
When computing statistics, increase the stride to use roughly independent samples.

## Current status (`python exp/00_sanity.py`)

| | rows | agents | windows | CV ADE | CV FDE |
|---|---|---|---|---|---|
| train (excluding zara2) | 32,208 | 1,233 | 12,702 | 0.546 | 1.210 |
| val | 11,819 | 475 | 4,262 | — | — |
| zara2 scene | 6,541 | 149 | 3,789 | **0.322** | **0.724** |

CV = constant-velocity extrapolation of the last observed velocity. zara2's 0.322 matches the literature —
meaning the loader is wired up correctly, and it is **the floor for every number that follows**.

One thing stands out: zara2's fraction of stationary frames is 28.8%, far higher than train's average of 9.3%
(people stopping in front of the shops). The speed distribution is also much narrower (p95 1.62 vs 2.58 m/s).
A large part of why zara2 is "easy" likely lies here — material for thought experiments.

## How to use

```python
import zlab

sc = zlab.load_zara2_scene()          # Scene: frames, agents, xy
tracks = sc.tracks(min_len=20)        # agent_id -> (T, 2)
w = zlab.windows(sc, stride=4)        # (W, 20, 2)
obs, gt = zlab.split_obs_pred(w)      # (W,8,2), (W,12,2)
zlab.ade(pred, gt).mean()
```

Start a new experiment as `exp/NN_name.py`, writing **the one-line question you're trying to answer** at the top.
