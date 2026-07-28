# bayes_switching — is pedestrian motion a single K, or several local K_j?

```
bayes_switching/
  bsk/koopman.py    observable, handcrafted token, least-squares fit, error/transition matrix
  bsk/samples.py    Scene -> (psi_t, psi_{t+1}, q_t) triples, closed-loop rollout
  bsk/observables.py             arbitrary observable (time-delay/nonlinear lift) + closed-loop rollout
  exp/01_handcrafted_tokens.py   experiment 1: LOO protocol (fit on train split -> zara2 test)
  exp/02_within_scene.py         experiment 1's confounder control: agent split within zara2
  exp/03_observable.py           gate: find an observable that beats CV with a single global K
  exp/04_M_sweep.py              M in {1,2,4,8,16,32} — discrete regimes or continuous?
  exp/05_seed_stability.py       verify whether the M=8 peak is seed luck + stability diagnostics
  out/*.json                     raw results
```

The observable is as planned: `psi_t = [p_t, v_t, 1] in R^5`, `v_t = p_t - p_{t-1}` (metre/step, not per second).

## Two things controlled for up front in the design

Following the plan literally, looking only at `E_switch < E_global` **almost certainly makes the hypothesis appear supported.** Both of these turned out to be real problems.

**1. Parameter count.** Five K_j have 5x the degrees of freedom of a single K, so however you draw the partition, the train error goes down.
So a `random` control was added — the token histogram is kept as is and only the labels are shuffled. Same parameter count, zero motion information.
The honest claim is not `E_switch < E_global` but **`E_switch < E_random`**.

**2. Components that fit for free.** Since `v` is *defined* as the position difference, `p_{t+1} = p_t + v_t` is an identity.
A K with an `[I I 0]` row reproduces the position block **with zero error** — for free, for any model.
So the total MSE is mostly constant, and what actually gets modeled is only `v_{t+1}` (= acceleration).
All the numbers below are `mse_v`, extracting only the velocity block.

## Experiment 1 — LOO protocol (fit on train split → zara2)

train 27,348 samples / zara2 5,945 samples.

| partition | train mse_v | **test mse_v** |
|---|---|---|
| global | 0.006452 | **0.001863** |
| random | 0.006446 | 0.001867 |
| k-means (velocity history) | 0.006358 | 0.001883 |
| token (handcrafted) | 0.006329 | **0.001925** |

**This is the opposite of the hypothesis.** On train, token is the lowest (−1.9%), but on test it is **the highest** (+3.3%).
The plan's prediction that "the difference will be large in the left/right/deceleration segments" also came out with the opposite sign —
in those segments switching is actually −5 to −38% worse. In other words, classic overfitting.

12-step closed-loop rollout (the token is recomputed every step from the predicted state — no future-information leakage):

| | ADE | FDE |
|---|---|---|
| const-velocity | **0.322** | **0.724** |
| global K | 0.339 | 0.765 |
| switching | 0.332 | 0.743 |

**All the Koopman variants lose to constant-velocity extrapolation.** switching is very slightly better than global, but both are worse than doing nothing.

## Experiment 2 — removing the confounder: split within zara2

Experiment 1 fit and tested on **different scenes**. This can't distinguish whether switching lost because (a) a single K was sufficient in the first place,
or (b) local K_j do exist but **differ per scene** so they don't transfer.
So it was re-measured by splitting zara2 in half at the agent level (within-track temporal splits are forbidden because the same track's dynamics would leak).

| | global | random | token | vs global | vs random |
|---|---|---|---|---|---|
| within-zara2 | 0.001897 | 0.001905 | 0.001853 | **+2.30%** | +2.73% |

**The sign flips.** Removing the scene shift, switching wins — and it beats the random control too, so it isn't a parameter-count effect.
In 4 of 5 seeds +1.5 to 2.3%, in 1 seed −0.04%. The direction is stable but **the magnitude is very small.**

The threshold sweep produced something more interesting:

| tau_theta / tau_v | straight fraction | vs global |
|---|---|---|
| 0.05 / 0.01 | 0.68 | **+6.36%** |
| 0.10 / 0.02 | 0.78 | +3.70% |
| 0.15 / 0.04 | 0.88 | +2.30% |
| 0.30 / 0.08 | 0.96 | −0.06% |

**The finer the partition, the larger the gain.** This looks less like the picture of "there are 5 discrete modes" and more like a picture
where the dynamics vary continuously and a piecewise-linear approximation improves as you add pieces.

## Conclusions so far

1. **Local K_j do exist but are scene-specific.** Within the same scene they beat the random control (+2 to 6%), and across scenes they lose. The plan's "if it succeeds, immediately repeat on Hotel and Univ" won't work as is for this reason — transferability itself is what needs to be validated.
2. **The effect size is too small to carry the narrative.** Even the best setting is +6%, and in closed-loop ADE it falls short of even constant-velocity extrapolation. As things stand, "the observable is too weak" fits better than "multiple K are needed" — with `psi=[p,v,1]`, all K_j can really learn is a per-token constant acceleration.
3. **The evidence for discrete modes is weak.** The tendency to improve with finer splits suggests continuous variation. The transition matrix's self-transitions (straight 0.89, the rest 0.13 to 0.19) also show that no mode other than straight is maintained.

## Experiment 3 — gate: does enlarging the observable let a single global K beat CV?

M=1 fixed, sweeping only the observable. The metric is not one-step MSE but **closed-loop 12-step ADE**
(one-step residuals are dominated by the structurally free components and say almost nothing about rollout quality).
ridge was swept as well — a K fit at ridge=1e-6 can have eigenvalues exceeding 1 and diverge within 12 steps, and that is a problem of the fit, not the observable.

| observable (dim) | within-zara2 vs CV | LOO vs CV |
|---|---|---|
| base `[p,v,1]` (5) | +0.6% | −6.1% |
| delay-2 (7) | +2.8% | −5.7% |
| delay-3 (9) | +2.6% | −5.9% |
| delay-5 (13) | +2.6% | −5.6% |
| delay-6 (15) | +2.2% | −6.0% |
| delay-3 +nonlin (14) | +2.6% | −1.6% |
| **delay-5 +nonlin (18)** | **+2.9%** | **+0.1%** |

**A narrow pass.** time-delay **already saturates at delay-2** — adding more lag gives no improvement.
This means the linear latent can't capture the "recent acceleration trend / curvature change" that the plan expected.
What actually contributed to beating CV in LOO was not the delay but the **nonlinear terms** (‖v‖, ‖v‖², v‖v‖), and even that is only a +0.1% tie.

## Experiments 4 & 5 — M-sweep: discrete regimes or continuous?

Observable fixed at delay-5+nonlin, partitioning by k-means on psi's velocity-delay block (handcrafted tokens cap at 5 and are 88% straight, so unsuitable).
Assignment during rollout uses only the **propagated psi**, so there is no future leakage. LOO protocol, 5 k-means seeds:

| M | ADE mean ± std | vs CV | max ρ(K_j) |
|---|---|---|---|
| 1 | 0.3217 ± 0.0000 | +0.0% | 1.000 |
| 2 | 0.3185 ± 0.0000 | +1.0% | 1.004 |
| 4 | 0.3267 ± 0.0001 | −1.5% | 1.124 |
| **8** | **0.3066 ± 0.0007** | **+4.7%** | 1.172 |
| 16 | 0.3257 ± 0.0127 | −1.2% | 1.925 |
| 32 | 1.1900 ± 0.0995 | −269.7% | 3.033 |

The M=8 peak is **stable across seeds** (std 0.0007). At +4.7% over CV under the standard LOO benchmark protocol, it is the best result so far.
Against the size-matched random control it is also +6.3%, so it isn't a parameter-count effect.

**But this curve must not be used to decide discrete vs. continuous.** The ρ(K_j) column is why —
as M grows, each K_j is fit from fewer samples and its **eigenvalues exceed 1** (3.03 at M=32), and the 12-step rollout amplifies this exponentially.
The non-monotonicity where M=4 is worse than M=2 and M=8 improves again is also better explained by this instability than by regime structure.
In other words, what the sweep measured is **not "how many regimes there are" but "how many can be stably fit."**

## Experiment 6 — M-sweep with stability controlled

`bsk/stability.py`. Two mechanisms were applied independently and compared (LOO, 3 k-means seeds, sharing the same partition).

**A structural trap that had to be noted before implementation.** In `psi = [p, v, ..., 1]`, because of the `p_{t+1} = p_t + v_t` identity and the constant channel,
K **necessarily has a Jordan block for eigenvalue 1**. That `‖K^h‖` grows linearly in h is not a pathology but **position integration itself**.
Hence applying `λ̃ = λ/max(1,|λ|/(1-ε))` to every eigenvalue damps position integration by `(1-ε)^h`.
So two variants were separated and compared: `proj/excess` (ε=0), which caps only the unstable eigenvalues at exactly 1, and `proj/all` (ε=0.01), the original proposal as is.

**ADE (LOO, CV=0.3219):**

| method | M=1 | M=2 | M=4 | **M=8** | M=16 | M=32 |
|---|---|---|---|---|---|---|
| plain (fixed ridge) | 0.3217 | 0.3185 | 0.3268 | 0.3064 | 0.3191 | 1.1313 |
| **scaled ridge** | 0.3217 | 0.3186 | 0.3249 | **0.3050** | 0.3076 | 0.3694 |
| proj/excess | 0.3217 | 0.3705 | 1.7688 | 1.0741 | 0.8986 | 0.8700 |
| proj/all | 0.7705 | 1.7266 | 1.6150 | 1.3444 | 1.5006 | 1.2607 |
| scaled+proj | 0.3217 | 0.3549 | 0.7237 | 1.0245 | 0.6554 | 0.5343 |

**Eigenvalue projection failed — and it did so while perfectly meeting the ρ target.** Three things were confirmed at once:

1. **ρ=1.000 was hit exactly, yet ADE is 5x worse** (proj/excess M=4: 1.7688). As feared, this is because **the eigendecomposition of a non-normal matrix is ill-conditioned** — `cond(V)` is 1.6e3 at M=4 and 2.6e3 at M=32. The `V Λ̃ V⁻¹` reconstruction corrupts the original operator in proportion to that condition number.
2. **`proj/all` collapses even at M=1, 0.3217 → 0.7705.** This directly confirms the structural λ=1 argument — shaving the position integration biases predictions toward standing still.
3. **ρ ≤ 1 does not prevent the 12 steps.** proj/excess has ρ=1.000 yet `‖K¹²‖` is still 94.6. Finite-horizon blow-up is produced by non-normality, not the spectrum.

**Size-based ridge succeeded.** With just `λ_j = λ₀·N/N_j`, ρ and `‖K¹²‖` are held far better (M=16: ρ 1.920→1.142, ‖K¹²‖ 7859→32), and the M=32 collapse (1.1313→0.3694) is largely recovered.

## So, discrete or continuous?

The stability-controlled curve (scaled ridge, ± is seed std):

| M | 1 | 2 | 4 | **8** | 16 | 32 |
|---|---|---|---|---|---|---|
| ADE | 0.3217 | 0.3186 | 0.3249 | **0.3050**±0.0008 | 0.3076±0.0021 | 0.3694±0.0113 |
| vs CV | +0.0% | +1.0% | −0.9% | **+5.2%** | +4.4% | −14.8% |

**It saturates around M≈8.** 8→16 is flat with no improvement (0.3050 vs 0.3076), and increasing M further makes it worse.
By the proposed decision criterion, this points to **"a small number of discrete regimes."** The tentative lean toward "continuous" from Experiment 2 is overturned —
what looked better with finer splits back then was a one-step metric, and it does not hold in closed-loop.

**Two caveats remain, however.**
- **M=4 is worse than M=2** (0.3249 vs 0.3186). It is non-monotonic rather than a clean saturation curve, and this part is unexplained. The claim "exactly 8 regimes" cannot yet be made.
- **M=32 still has uncontrolled stability** (ρ=1.959 even with scaled). The tail region is still measuring fit stability, not regime structure.

## Overall conclusions

| | result |
|---|---|
| one K vs. several K_j | within the same scene K_j wins significantly (+2 to 6% over the random control) |
| cross-scene transfer | handcrafted token fails. k-means partition + scaled ridge holds even under LOO |
| observable enrichment | delay saturates at 2, the nonlinear terms are the real contributor, global K ties CV |
| stability constraint | **eigenvalue projection fails** (cond(V)~1e3), **size-based ridge is the answer** |
| discrete vs. continuous | **saturates at M≈8 → leans to a few discrete regimes**, though the M=4 non-monotonicity is unexplained |
| best performance | LOO, delay-5+nonlin, M=8, scaled ridge → ADE **0.3050** vs CV 0.3219 (**+5.2%**) |

## Experiment 7 — hard vs soft mixture

`exp/07_hard_vs_soft.py`. Now that stability is controlled (scaled ridge, max ρ=1.093 at M=8), the improvement from
`sum_j pi_j K_j psi` can be read without confounding it with "the averaging effect of unstable K_j."

**The decisive control is `uniform`.** Even if soft beats hard, that alone does not establish "smooth regime transitions matter" —
a convex combination shrinks toward the mean operator, so it could be a variance reduction unrelated to regimes.
Only if soft beats `uniform = mean_j K_j` (the state-independent average) can we say that pi_t carries information.

ADE (LOO, CV=0.3219, 3 seeds):

| variant | M=4 | M=8 | M=16 | effective # of modes (M=8) |
|---|---|---|---|---|
| hard | 0.3249 | **0.3050** | 0.3076 | 1.00 |
| soft tau=0.05 | 0.3249 | 0.3051 | 0.3075 | 1.01 |
| soft tau=0.5 | 0.3222 | 0.3054 | 0.3074 | 1.15 |
| soft tau=1.0 | 0.3679 | 0.3129 | 0.3081 | 1.41 |
| soft tau=2.0 | 0.4639 | 0.3270 | 0.3099 | 1.99 |
| soft tau=4.0 | 0.5743 | 0.3951 | 0.3334 | 3.32 |
| **uniform (mean operator)** | 0.7127 | **1.0729** | 0.8430 | 8.00 |

**Hypothesis rejected.** But before running `exp/07b_weight_entropy.py`, this table was nearly misread.
soft tying hard at tau ≤ 0.5 is not a finding but a **tautology** —
the effective number of modes there is 1.01 to 1.15, so the mixture is effectively one-hot and **soft simply was hard.**
In the tau ≥ 1.0 range where the mixture truly becomes soft, it **worsens monotonically.**
So "the softer, the worse" is the accurate summary, and the only exception, +0.85% at M=4 tau=0.5, has 1.14 effective modes —
not a mixture effect but the effect of slightly smoothing the assignment boundary.

### Why the average collapses — the first hypothesis was wrong

Since ρ(K_unif)=0.984 < 1 yet ADE is 3.5x worse, the expectation was "a bias toward standing still from the broken λ=1 mode" (the same pathology as proj/all in Experiment 6).
Measuring the predicted stride length directly with `exp/07c_stall.py` showed **the opposite.**

predicted stride / last observed stride:

| | h=1 | h=6 | h=12 |
|---|---|---|---|
| ground truth | 1.041 | 1.445 | 2.172 |
| hard | 1.067 | 1.486 | 2.415 |
| soft tau=1 | 1.124 | 2.762 | 6.289 |
| **uniform** | **13.272** | 49.562 | 69.459 |

It doesn't stall — it **explodes. And already 13x at the first step.** If it blows up in a single step, it's not a spectrum or horizon problem.

`exp/07d_offdomain.py` pinpoints the real cause. Since `(sum_j pi_j K_j) psi = sum_j pi_j (K_j psi)`,
the soft mixture **inevitably evaluates K_j on states that cluster j has never seen.**

| M | in-domain RMSE | off-domain RMSE | ratio | global K |
|---|---|---|---|---|
| 4 | 0.0438 | 0.1025 | 2.3x | 0.0435 |
| 8 | 0.0425 | 0.1268 | **3.0x** | 0.0435 |
| 16 | 0.0429 | 0.1423 | 3.3x | 0.0435 |

**K_j is 2.3 to 3.3x worse outside its own domain, and worse as M grows.** A local operator is an affine map valid only locally,
so it must not be extrapolated, and therefore mixing them is itself an invalid operation.

Incidentally, this is also **evidence in favor of the regime hypothesis.** If the K_j were all just noise around the global K,
the off-domain error would have been similar to the in-domain one. The 3x difference means the K_j are **genuinely different maps.**

**But do not read beyond this experiment's range.** What was rejected here is only the *operator averaging* form of soft mixture.
The plan's branch-preserving mixture `sum_j pi_j N(K_j psi, Q_j)` does not average operators, so it is **neither rejected nor validated.**
The problem of averaging a left turn and a right turn into going straight must also be handled separately over there.

## Experiment 8 — what is a regime (+ a parameter-matched alternative partition)

`exp/08_regime_identity.py`. Merely naming cluster statistics is unfalsifiable —
**any** partition of velocity space produces a plausible table. So, at the same M, same ridge, and same rollout,
**a control that changes only the partition to a poor one** was run alongside.

| partition | M=4 | M=8 |
|---|---|---|
| **k-means (vel-delay)** | 0.3249 | **0.3050** |
| speed-quantile (speed only) | 0.3400 | 0.3303 |
| heading-octant (direction only) | 0.3431 | 0.3318 |
| speed × dtheta (handmade primitive axes) | 0.3395 | 0.3363 |
| random | 0.3249 | 0.3272 |
| (global K = 0.3217, CV = 0.3219) | | |

**Two things emerge at once.** The k-means partition clearly beats the poor partitions — i.e. a regime is neither a simple speed bin
nor a simple direction bin. Yet **a partition that has structure but is wrong is worse than random.**
In the random partition each K_j is an unbiased sample of the whole, so it effectively converges to the global K (0.3249 ≈ 0.3217), and is therefore safe.
Conversely, the speed/heading partitions produce mis-specialized operators and become **worse than the global K.**
And at M=4 even k-means ties random (0.3249 vs 0.3249). **The only setting that actually beats the global K is M=8 k-means.**

Identity of the M=8 regimes (train, sorted by speed):

| j | N | frac | E‖v‖ | E\|dθ\| | E Δ‖v‖ | heading concentration | top scene | share |
|---|---|---|---|---|---|---|---|---|
| 2 | 6726 | 0.269 | 0.105 | 0.195 | +0.005 | **0.31** | students001 | 0.37 |
| 4 | 4325 | 0.173 | 0.270 | 0.090 | −0.001 | 0.88 | students001 | 0.42 |
| 7 | 5012 | 0.200 | 0.465 | 0.054 | −0.000 | 0.95 | zara01 | 0.29 |
| 6 | 1548 | 0.062 | 0.506 | 0.106 | −0.007 | 0.95 | biwi_hotel | 0.65 |
| 5 | 1713 | 0.068 | 0.512 | 0.107 | +0.003 | 0.96 | biwi_hotel | 0.57 |
| 0 | 3712 | 0.148 | 0.521 | 0.043 | −0.004 | 0.97 | zara01 | 0.30 |
| 3 | 817 | 0.033 | 1.014 | 0.085 | +0.002 | 0.97 | **biwi_eth 0.95** | |
| 1 | 1166 | 0.047 | 1.027 | 0.070 | −0.007 | 0.98 | **biwi_eth 0.91** | |

**The motion primitives the plan expected did not appear.**

- **heading concentration is 0.88 to 0.98 in 7 of the 8.** A regime is effectively a **velocity-vector (speed × direction) bin.** The only isotropic one is the slowest regime 2, and that's because heading is noise when slow.
- **There is no acceleration/deceleration regime.** E[Δ‖v‖] is all within ±0.007. E\|dθ\| is merely inversely correlated with speed (larger when slower), not an independent turning mode.
- **The high-speed regimes 1 and 3 are effectively scene tokens** (biwi_eth 91 to 95%).

## Experiment 9 — is this even switching in the first place?

Experiment 8's transition matrix gave **an average self-transition of 0.965.** Over a 12-step rollout that means the token never changes.
So it was directly compared against a variant that fixes the token once at the last observation and **holds it constant across all 12 steps.**

| M | switching | frozen | difference | switches per 12 steps | windows that switched at least once |
|---|---|---|---|---|---|
| 4 | 0.3249 | 0.3246 | **−0.09%** | 0.09 | 0.07 |
| **8** | **0.3050** | **0.3063** | **+0.42%** | **0.11** | **0.10** |
| 16 | 0.3076 | 0.3125 | +1.57% | 0.22 | 0.19 |

**This is the key result. Of the +5.2% over CV, switching contributes only +0.42%.**
In 90% of windows the token never changes. The remaining 92% of the gain comes from **choosing the right local operator at the start.**

In other words, an honest description of the currently winning model is **not a switching Koopman but
a static mixture-of-experts over velocity features (= a piecewise-linear approximation of the velocity field).**

The regimes zara2 actually uses:

| regime | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| train occupancy | 0.148 | 0.047 | 0.269 | 0.033 | 0.173 | 0.068 | 0.062 | 0.200 |
| **zara2 occupancy** | 0.220 | **0.003** | 0.372 | **0.003** | 0.058 | 0.012 | 0.022 | 0.310 |

zara2 **barely uses** regimes 1 and 3, where biwi_eth made up 91 to 95% (0.003).
The reason the scene-specific regimes didn't ruin transfer is not that transfer works well but **that they simply aren't used.**
In practice zara2 runs on 3 of the 8 (2, 7, 0 = 90%).

## Experiment 10 — does the 18×18 operator earn its parameters?

`exp/10_complexity_ladder.py`. Since the token is effectively fixed in Experiment 9, the model is "pick one local operator from the velocity vector and apply it 12 times."
If that's all it is, a much smaller local model should yield the same gain. **The partition (M=8 k-means) was fixed and only the per-regime model class** was varied.

| model | params | ADE frozen | ADE switch | FDE | vs CV |
|---|---|---|---|---|---|
| const-velocity | 0 | 0.3219 | 0.3219 | 0.7238 | +0.0% |
| accel-global (v←v+a) | 2 | 0.3359 | 0.3359 | 0.7572 | −4.4% |
| accel-regime (v←v+a_j) | 16 | 0.3312 | 0.3313 | 0.7463 | −2.9% |
| vlin-regime (v←A_j v+b_j) | 48 | 0.3170 | 0.3164 | 0.7076 | +1.7% |
| vlin+hist (+v_{t−1}) | 80 | 0.3100 | 0.3096 | 0.6911 | +3.8% |
| **koopman-regime** | **2592** | 0.3063 | **0.3050** | 0.6769 | **+5.2%** |

**The explanation "each group has a different average acceleration" is rejected** — a per-regime constant acceleration is **worse** than CV (−2.9%).
The gain requires a **linear map** of the velocity (damping/rotation), and bias alone is not enough.
And **80 parameters capture +3.8% (73%) of the +5.2%.** The remaining +1.4% costs 32x more parameters.
The Koopman lift does earn its keep, but that value is mostly the residual left after a much simpler model has already taken the rest.

## Experiment 11 — is the regime partition distinguishable from "a better global model"?

Experiment 10 ran every rung at M=8 only, so it can't separate the two explanations.
(a) regimes matter, or (b) `v←Av+b` is just a better global model than CV and the M=8 column was measuring that.
**This is the control that decides the project's headline.** Each rung was re-run at M∈{1,2,4,8,16} — **read the rows.**

| model | M=1 | M=2 | M=4 | M=8 | M=16 | M=8 vs M=1 |
|---|---|---|---|---|---|---|
| accel | 0.3359 | 0.3630 | 0.3640 | 0.3313 | 0.3234 | +1.37% |
| vlin | 0.3308 | 0.3310 | 0.3147 | 0.3164 | 0.3173 | +4.36% |
| vlin+hist | 0.3319 | 0.3224 | 0.3235 | 0.3096 | 0.3174 | **+6.72%** |
| koopman | 0.3217 | 0.3186 | 0.3249 | **0.3050** | 0.3076 | +5.19% |

**(b) is rejected and (a) survives.** Every model class improves from M=1→M=8 (+1.4 to 6.7%).
The regime partition does not reduce to "a better global model."

**There is one more decisive observation at the same time. At M=1 no model beats CV** (koopman 0.3217 is the best, a tie).
In other words, the only path by which this project beats CV is the regime partition — and that total is 5.2%.
The M=2/M=4 non-monotonicity (accel worsens sharply at M=2, 4) is also still unexplained.

## Overall conclusions (incorporating Experiments 7–11)

| question | answer |
|---|---|
| hard vs soft | **hard.** operator averaging ends up evaluating K_j outside its own domain (3x off-domain error) and explodes 13x at the first step |
| regime = motion primitive? | **No.** 7 of 8 are **velocity-vector bins** with heading concentration 0.88 to 0.98. No acceleration/deceleration/turning mode |
| regime = scene ID? | **Partly yes.** high-speed regimes 1 and 3 are 91 to 95% biwi_eth, and zara2 uses those two only 0.003 |
| is this switching? | **No.** self-transition 0.965, the token is unchanged in 90% of windows. Of the +5.2%, switching contributes **+0.42%** |
| is the regime partition needed? | **Yes.** every model class improves from M=1→8, and at M=1 nothing beats CV |
| is the Koopman lift needed? | **Partly.** the 80-parameter model captures 73% of the gain |
| effect size | ADE 0.3219 → **0.3050 (+5.2%)** |

**The most honest single sentence:** what actually worked here is not switching Koopman dynamics but
**a local linearization of velocity-vector space (a static mixture-of-experts)**, and its gain is reproducible but at the 5% level.

Mapped onto the decision criteria, it is **neither A nor B, but a combination of C and E.**
A (a few discrete regimes) holds in the form of M≈8 saturation, but those regimes are velocity-vector bins rather than motion primitives, and
they do not switch over time, so **it is not A in the intended sense.**
D (scene-specific) is only partly right — scene-token regimes exist, but they were harmless because the test scene doesn't use them.

## Next steps

1. ~~hard vs soft mixture~~ → rejected in Experiment 7.
2. ~~regime interpretation~~ → Experiments 8 & 9. velocity-vector bins, almost no switching.
3. ~~complexity ladder~~ → Experiments 10 & 11. the partition is needed, but 73% of the gain is covered by 80 parameters.
4. **What remains.** (i) the branch-preserving mixture `sum_j pi_j N(K_j psi, Q_j)` — the only soft form that doesn't average operators, still unvalidated.
   (ii) the M=2/M=4 non-monotonicity. (iii) for social context, because of the Experiment 9 finding that switching almost never happens,
   the original entry point of "improving the regime transition prior" has **effectively vanished** — there's no reason to proceed without a redesign.
