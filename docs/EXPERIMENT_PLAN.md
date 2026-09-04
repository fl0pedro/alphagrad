# EXPERIMENT PLAN — the approximation campaign, 2026-08-28

Five waves of 3–4 arms, one battery at a time, on `pgi15-gpu15..18`. Every
launcher is generated from `tools/gen_fq_launchers.py`; editing an
`fq_*.sbatch` by hand is not supported and will be overwritten.

This document is the registry. Each experiment states the question, the exact
configuration, the **registered prediction** (recorded before the run, never
edited afterwards), the falsification criterion, the cost, and its
dependencies. The decision table at the end is the part that stops a battery
from producing data nobody acts on.

---

## 0. The settled configuration — every arm, no exceptions

Owner decision; not re-litigated here.

| | channel | how |
|---|---|---|
| **TRAINED** | real measured latency | `--measure-latency --cmp-type latency --lambda-cmp 1`, `--latency-inner-reps 50` |
| **TRAINED** | real measured peak memory | `--mem-type peak_memory --lambda-mem 1` |
| **TRAINED** | gradient cosine at init, K=1 | `--quality-metric grad_cosine`, `ALPHAGRAD_GRAD_COSINE_K=1` |
| **LOGGED** | sparsity (slot 10) | `--sparsity-log` (weight 0) |
| **LOGGED** | clipped relative Frobenius (slot 8) | automatic — `grad_cosine` materialises the exact reference slot 8 needs |
| **LOGGED** | legacy Jacobian cosine | `--cos-log-every 20` |
| **GUARD** | gradient coverage (slot 7) | `--reject-frozen-grads` (default ON, named anyway) — **removed 2026-09-03** (owner ruling 2026-09-03, ticket dsnn-3qm.15): no guard, slot 7 reserved |

Fixed across all arms:

```
--terminal-rewards-only --discount 1.0 --gae-lambda 1.0
--reward-mode additive --symlog-channels cost --advantage-norm none
--init-scheme classic --ppo-epochs 1 --num-envs 16 --minibatches 4
--walk-rotate
ALPHAGRAD_QUALITY_GATE_MIN=0.05   GRAPHAX_QUANT_PULLDOWN=0 (pullup)
```

Four of these need justifying, because a launcher that omits any of them looks
identical and behaves differently.

* **`--quality-metric grad_cosine` must be named.** `auto` still resolves to
  `loss_drop` (`env.py:3516`). An arm that forgets silently runs the channel
  that is 200× dearer for +0.011 Pearson and −0.040 Spearman.
* **`--discount 1.0 --gae-lambda 1.0` must be named.** `ppo.py`'s defaults are
  0.99/0.95 and **no campaign run v57–v66 ever set them**. Over ~95
  eliminations `(γλ)^95 = 0.0079`: the terminal reward reaches the first
  decision at 0.8 % strength. R2 (γ=λ=1) held at quality 0.885; R3, differing
  by exactly these two flags, drifted to destruction at ep131. This is the
  campaign's one confirmed causal result.
* **`--walk-rotate` must be named, and it is not only about the walk.**
  `env._walk_seed` is shared, so `--walk-rotate` rotates the **grad-cosine**
  probe batch too. Without it, `grad_cosine` scores every plan of every
  episode against one frozen batch — the exact defect `62414d0` was written to
  fix for `loss_drop`. Sampling variance in the quality signal is wanted, and
  this is the flag that supplies it.
* **`--symlog-channels cost`.** Symlog on the cost channels, never on quality.
  Slots 6/8/9/10 are structurally exempt anyway
  (`NO_SYMLOG_REWARD_INDICES`), so this flag governs slot 7.

### The measurement protocol, restated as a rule

No latency claim in this campaign counts unless it is: **candidate and its own
exact reference, same process, same GPU, back to back, warm**, reported as a
**ratio**, at `--latency-inner-reps 50`, against the instrument's own null of
**1.0007 ± 0.0008** — and, where a distribution is claimed, over **5 seeds**.
Absolute nanoseconds are recorded and are never the claim. Cross-actor
absolute comparisons carry a systematic ~1.3 % bias floor (identity latency
spans 1.30 % across four GPUs of one node against 0.07–0.18 % within-actor
noise; gpu1 is consistently fast, gpu3 consistently slow).

---

## Wave 0 — unblock (1 GPU node + 2 CPU jobs)

Nothing here consumes a battery slot. It exists because three later
experiments are blocked on facts nobody has, and one is blocked on code
nobody wrote.

### W0-A `fq_w0_cpu_gates.sbatch` — pgi15-cpu1, ~2 CPU-h, **0 node-h**

`tools/ratio_gates.sh`, then `tools/smoke.sh NeuralNetwork`, then
`tools/pool_liveness_gate.sh`. Helmholtz is deliberately not smoked
(`landscape_map`/`_callback` on Helmholtz carries a pre-existing `TypeError`
from the scalar-loss retarget `744fc3d`).

**`tools/pool_liveness_gate.sh` is the third gate**, added 2026-08-29, and it
pins something the other two do not: that a `--ray-measure` run **measures
something**. Neither of the others starts a measure pool — the smoke's
canonical config has no `--ray-measure` at all — which is how the `87cdc49`
regression (§Findings (b) below) made every pooled measurement a degenerate
sentinel for 19 hours without turning one gate red. It runs (1) a static
`ast` check of the pool → actor → server call chain, so a kwarg the pool
forwards that the actor cannot accept is named in ~10 ms, and (2) a real
2-actor / 4-env / 1-episode `--ray-measure` run on NeuralNetwork with
`--plan-log`, then fails if **zero terminal plans were measured inside a
measure actor**, if **any `[SENTINEL]` line** appeared, or if any recorded
terminal reward is the degenerate sentinel on all six cost channels. Both
parts always run: part 1 is a proxy and part 2 is the evidence, and skipping
the evidence because the proxy is green is the mistake the gate exists to
correct. It exports `ALPHAGRAD_BATCHED_CALLBACK=1` itself — without it
`--ray-measure` exits with `ValueError` — and reports that class of failure as
**exit 2, HARNESS MISCONFIGURED**, distinct from **exit 1, MEASUREMENT DEAD**.
Both are failures; the distinction exists because they call for opposite
responses.

**Prediction:** green — `ratio_gates` and `smoke` both passed at `949f1af`;
the liveness gate is **measured green at `87cdc49`/`2ddbeef` and measured red
with `87cdc49` reverted**, so it is demonstrated to fail on the bug it targets
rather than merely asserted to.
**Falsifier:** any gate red *or skipped* blocks every later wave. A skip is a
failure: a gate that did not run pins nothing.

### W0-B `fq_w0_probe.sbatch` — one Blackwell, ≤12 h, **12 node-h**

Three verifications that would each otherwise cost a training node.

**P1 — re-derive the candidate list and the best-face claim.** Singleton SKIP
sweep over every live face, paired. This is X1's *and* X2's prerequisite. The
campaign's headline "`k21/f1` is the best free single skip nobody found"
(ratio 0.5325, quality 0.9257) came from a sweep under a different
configuration, and **A3 found `k21/f1` is not a live face in this graph** — it
substituted `k14/f0` (vertex 81). Nothing naming `k21/f1` may be written up
until P1 replaces it.

**P2 — does DIAG ever apply on a live run?** DIAG applied **0 of 103**
requested rules at every budget under both pulldown polarities: "every 'diag'
result in the campaign is an identity plan wearing a diag label". A1
(`690971a`) could not reproduce the 0/103, but its harness draws uniformly
over legal ops rather than from a collapsed trained policy, so the two numbers
are not the same quantity and neither settles the other. P2 measures
applied/requested per kind on the production configuration under (a) neither
flag, (b) `--per-face-masks`, (c) `--per-face-masks --diag-per-face`, with
idempotent no-ops removed from the denominator (~83 % of DIAG's "failures" are
correct no-ops). The owner keeps DIAG in the action space but **its factor is
not an action**, so (c) is measured for attribution only and is not a
candidate for any later wave.

**P3 — re-baseline after `744fc3d`.** The scalar-loss retarget landed after
every quality number the campaign quotes. `LIF_SNN_SHD` and `ADALIF_SNN_SEQ`
moved (58→56 and 37→35 equations) and the seven analytic AD benchmarks went
back to their full multi-output Jacobian target. `LIF_SNN`/`ADALIF_SNN` had
never run under `--measure-grad` at all.

**Gotcha, encoded in the launcher:** `landscape_map`'s `--quality-metric`
choices are `loss_drop, cosine, none` — the tool **cannot name
`grad_cosine`**. On TransformerLM (a scalar-loss target) `cosine` is the
deprecated alias that resolves to `grad_cosine`, which is the settled channel.
Passing `loss_drop` here would measure a different channel from every training
arm. *Fix worth making:* widen `landscape_map`'s choice list.

### W0-C `fq_w0_x2_screen.sbatch` — pgi15-cpu2, ~4 CPU-h, **0 node-h** — BLOCKED ON A BUILD

X2, the coverage-constrained frontier. Gradient coverage is computable
**without the walk and without a latency measurement**, so thousands of
multi-skip subsets screen on a CPU node and only survivors reach a GPU. Greedy
and beam over the ~31 singleton candidates P1 re-derives.

It also answers a question that could reframe the thesis: **the number of
evaluations a beam needs to reach the best known plan**. If a few hundred
suffice, RL-as-search is settled on this target and the claim becomes
RL-as-amortization.

`tools/coverage_beam.py` does not exist. The launcher's pre-flight aborts with
**exit 66** naming it. It needs: read P1's `rows.csv`, keep quality > 0.9 and
ratio < 0.99; greedy + beam (width 16, depth 8) scoring subsets **only** by
`env._grad_coverage` against the same-order exact reference; emit survivors
plus a running count of coverage evaluations to first reaching the best known
plan.

### W0-D — **A6, "record the losing plans"** (build, no launcher)

`_dump_pareto` persists the **front**. Nothing persists the plans that lost.
`--plan-log` / `--record-all-plans` do not exist. **X3 is blocked on this and
X3 alone is why A6 is first.** A6 is one append-only JSONL per run recording,
for every terminal plan: the replayable seq+faces, all 11 reward slots, the
per-kind applied/requested/idempotent counts, the coverage census, and the
episode. It is a logging extension — it changes no reward and no action — so
it is gated by `ALPHAGRAD_EQ_DUMP` inertness with the flag off, and it then
rides along inside every wave-1..4 arm at no extra node cost.

**X3 itself is then an analysis, not a battery:** diff the lowered graphs of
the recorded losers against identity and attribute the regression. The
motivating evidence, over 64 archived points: `pearson(applied rules, latency
ratio) = −0.12` (no relationship at all) against `pearson(applied rules,
quality) = −0.44`; 12 one-face plans average ratio 0.598 at quality 0.915
while 18 plans with ≥20 faces average 0.607 at quality 0.662; COMPRESS applies
111/115 (~96 %) and is **net negative on latency at every budget above 1**
(+1.0 % to +3.3 %, all outside the drift floor).

---

## Wave 1 — CONTRAST × PRICE (4 nodes, ~4.5 h each, **18 node-h**)

**Question.** R2 held and did not learn. Its quality spread was 3×10⁻⁶ and its
latency spread 1.7 µs across 16 plans — the advantage was about zero. The R
battery brackets the space:

```
R1  random init (B=0), no gate   -> destroyed absorber, no contrast
R2  identity init (B=6), gamma=1 -> identity fixed point, no contrast
R3  identity init,      gamma<1  -> drifts to destruction
```

The campaign's own registered next step is *"the R2 configuration plus a
bounded exploration pressure, made safe by the gradient-coverage guard so that
exploring toward skips cannot be rewarded for freezing gradients."* Two knobs
supply it, and this wave sweeps them together because sweeping either alone
reproduces a failure already seen.

**Why these values.** A5's corrected arithmetic (the factory prints a
*per-slot* rate and each face carries 3 slots; the true expectation is
`118·3·3·e^−B`) puts **B=6 at 2.6 approximations per plan — the floor of the
useful band — B=5 at 7.2, B=4 at 19.5**. Every measured win in the campaign
lives at **1–15 approximations per plan**. Under `--advantage-norm none` the
realized pull is `λ·σ_q/σ_cost` with `σ_q = 0.15852` raw, `σ_lat = 2.119`
symlog: **λ=16 buys 1.20 : 1, not 16 : 1**, and matching the pricing PopArt
actually realized needs **λ ≈ 134–199**. R1–R3 ran λ=16, i.e. quality
under-priced ~10× — safe only because they had no contrast to spend.

| arm | node | B | λ_acc | isolates |
|---|---|---|---|---|
| `fq_w1a_bias6_lam170` | gpu15 | 6 | 170 | control; R2's bias at a correct price |
| `fq_w1b_bias5_lam170` | gpu16 | 5 | 170 | **the recommended operating point** |
| `fq_w1c_bias4_lam170` | gpu17 | 4 | 170 | the far end of the bias band |
| `fq_w1d_bias5_lam16` | gpu18 | 5 | 16 | the price, at fixed bias |

Comparison pairs: (a,b,c) is the bias sweep at fixed λ; (b,d) is the price at
fixed bias. `w1a` is a *bridge* to R2, not a one-knob comparison with it — it
changes both the quality channel and the price — and must not be reported as
one.

**Panels, in this order.** (1) quality spread across the 16 envs; below 1e-3
every other panel is uninterpretable. (2) `grad_cov/rejected_this_ep` against
`plan/n_live` — a run where they stay *equal* is not learning, it is being
refused. (3) `entropy/approx_head ÷ faces/mean_valid`, never the raw
numerator: the raw metric averages a structural zero for every step whose
vertex has no live face, and v64b's entropy fell 357× while `mean_valid` fell
91× — the quotient fell 4×. A fall that tracks `mean_valid` is a
graph-destruction signal, not an entropy collapse. (4) `approx_prob/none` and
per-plan ratios against the drift floor.

**Registered prediction.** B=6 reproduces R2 (parks at identity, spread <1e-3,
`none` > 0.99) — a repriced λ does not create contrast by itself. B=5 produces
contrast (spread > 1e-3 and ≥1 plan/episode outside the drift floor by ep50)
**and** non-zero `grad_cov/rejected_this_ep`. B=4 over-approximates: more
rules, no more speed. λ=16 drifts further than λ=170, with the coverage guard
rather than the reward doing the stopping.

**Falsifier.** If none of a/b/c reaches spread > 1e-3 by ep50, contrast is not
bias-limited, the NONE-bias hypothesis is dead, and **wave 3 is cancelled**.
If d is indistinguishable from b, λ is not a live knob and the {130,170,210}
relaunch is closed as answered, not completed.

---

## Wave 2 — THE ELIMINATION ORDER (4 nodes, ~16 h each, **64 node-h**)

**Question.** The order axis carries **45–80×** of range against at most ~2×
on the approximation axis, and it has never been scheduled — it was starved by
the very credit horizon R2/R3 proved causal. Under
`ALPHAGRAD_FORCE_REV_ORDER=1` exactly one vertex is legal at each step, so the
pointer head has taken **zero gradient across the entire v57–v66 campaign and
R1–R3** (`ve` entropy `-0.0e+00`, `max|dH/dlogits| = 0.0e+00`). Lifting the
pin is the first time that head is trained at all.

**It is also the only place the memory channel is alive.** Under a pinned
order the memory ratio is `1.0000 ± 0.0000` for every archived winner and
every ladder rung; the only plan that moves it is `skip@all`, at 0.0406 with
quality 0.0000 — destruction. Across *orders* the same channel spans ~80×.
The owner's requirement that real measured peak memory be a trained component
is satisfiable here and nowhere else on this target. **This wave runs
regardless of wave 1's outcome**, and if wave 1 finds no contrast it becomes
the campaign's main line.

| arm | node | `FORCE_REV` | extra | isolates |
|---|---|---|---|---|
| `fq_w2a_order_pinned` | gpu15 | 1 | — | in-battery control |
| `fq_w2b_order_free` | gpu16 | 0 | — | both levers |
| `fq_w2c_order_free_exact` | gpu17 | 0 | `--exact` | **the decisive arm** — order alone |
| `fq_w2d_order_free_edgemem` | gpu18 | 0 | `--face-edge-mem` | ⚠ owner ruling required |

The control is **re-run inside the battery** rather than compared against wave
1, because GPU state moves 18–20 % over a session and an unpaired A-then-B
once produced a "17.5 % beats reverse" that evaporated to ratio 1.00 when
paired.

**What lifting the pin costs.** Face count 115 → ~313 (≈2.7× the face work);
distinct edge keys 101 → 206–286; the face stratum flips from 1
both-primitive / 114 one-intermediate / 0 both-intermediate to 164/311/463, so
**both-intermediate becomes the majority** — precisely the stratum where the
chunk-mean read measures worst and where the edge-keyed memory was built to
help. Random orders measure ~500× above rev. `MAX_FACES=2538` remains a valid
bound (`derived_max_faces` is order-independent). Each distinct order pays its
own exact compile for the coverage guard (`_EXACT_LEAF_NORMS` is keyed on the
order): 8.9 % of the first period under an order-searching policy against
0.71 % at steady state under a pinned one. Hence 24 h wall, not 12.

**Registered prediction.** `w2c` finds ≥1 plan whose paired ratio against
exact reverse is below 0.95 and moves peak memory by more than 2×. `w2b`, with
both levers, does **no better than the better of its halves** — the
approximation wins are attached to specific faces of the *reverse*
elimination and those faces do not survive reordering. `w2a` reproduces its
wave-1 counterpart within the drift floor; if it does not, the wave-1 reading
was drift and must be re-stated.

**Falsifier.** If `w2c` never beats exact reverse outside the drift floor over
250 episodes, then on TLM reverse is optimal-or-unbeatable-by-this-policy, the
45–80× range is a property of *bad* orders rather than of reachable good ones,
and **the order axis is closed**. Say so; do not re-run it wider.

**⚠ `w2d` needs an owner ruling before launch.** `CLEAN_DESIGN_AUDIT.md` row
g2 marks `--face-edge-mem` "would CONFLICT" on the same grounds as row g1's
`--face-endpoint-read`: it widens the face head's input with memory rows,
which the owner's design forbids. It has been inert so far so the conflict has
never bitten; off the pin it stops being inert. If the ruling is no, run a
second seed of `w2b` on that node instead.

**One correctness note if anything later reaches for lagrangian mode:**
`--lag-causal-mask` becomes *incorrect* once the pin is lifted — its docstring
specifies `{approx actions} ∪ {vertex choices}` but the code implements only
the face half, so it would zero quality credit on every step whose only
decision was the vertex choice. These arms run `--reward-mode additive` and
are unaffected.

---

## Wave 3 — PRICE AND STABILITY (4 nodes, ~4.5 h each, **18 node-h**) — conditional

Runs only if wave 1 found contrast. If w1a/b/c all park at identity this wave
is **cancelled, not rescoped**.

Two registered items, both recorded and neither executed. **The price:** the
v66 abort rule said "kill and relaunch at λ ∈ {130,170,210}" and never fired.
Separately the design record calls the λ sweep "the single most consequential
number in the design" and shows {1,4,16} lands *entirely inside the failed
band* — λ=16 buys exactly v66b's 1.20 : 1, and v66b ended at quality +0.233
with 155 ops/plan and latency 12 % worse than exact. The informative sweep is
{16, 130, 400}. Wave 1 runs 16 and 170; this wave adds 130 and 400.
**The stability:** A5's KL-to-reference, whose own docstring says it bounds
drift and **cannot create contrast** — "the contrast knob is
`ALPHAGRAD_FACE_NONE_BIAS`. Sweep them together." This is the second half of
that sweep, run after wave 1 establishes there is drift worth bounding.

| arm | node | λ_acc | `--kl-ref-weight` |
|---|---|---|---|
| `fq_w3a_lam_star` | gpu15 | `$W1_LAM` | 0 |
| `fq_w3b_lam130` | gpu16 | 130 | 0 |
| `fq_w3c_lam400` | gpu17 | 400 | 0 |
| `fq_w3d_kl_ref` | gpu18 | `$W1_LAM` | 0.1 |

The quality gate stays at 0.05 in every arm: removing it uncaps the full SKIP
prize (`symlog(157/70) ≈ +0.81` for one action against `λ·0.885`), so SKIP
wins outright at λ=1 and is only priced out at 16.

**Prediction.** 130 and the wave-1 winner behave alike (flat in λ across
130–210). λ=400 over-prices and parks at identity, establishing the band's
**upper edge, which no run has ever located**. The KL arm reaches the same
endpoint by a smoother path and does **not** improve the best plan found.
**Falsifier.** If a/b/c are indistinguishable, price is not a live knob across
a 3× range and λ is retired from the backlog. If the KL arm finds a *better*
plan, A5's own docstring is wrong and that needs explaining before it is used.

---

## Wave 4 — THE FACE READ POINT (4 nodes, ~5 h each, **20 node-h**)

`--face-read` is the **cheapest unrun experiment in the backlog**: no extra
encode, no extra scan, no new parameters, no width change — the pooled rows
are already materialised and only the mask moves. All three modes are pinned
`rollout == replay` by `tests/face_read_point_test.py`, so the PPO ratio is
safe in every arm. Because the arms are cheap and independent, this wave is
the natural slot-in whenever an earlier battery ends early.

Face f's chunk is `[approx-echo(f−1) ‖ header+contraction(f)]`, so
`chunk-mean` produces the 94 logits from a token-count-weighted blend of the
*previous* face's approximation with *this* face's contraction, with no type
channel to separate them. `own-span-mean` masks the pooling at the echo prefix
so the head reads face f's own span; `last-row` reads the recurrence's state
after the whole chunk. The read point is already causally correct; only the
readout is wrong.

| arm | node | `--face-read` |
|---|---|---|
| `fq_w4a_read_lastrow` | gpu15 | `last-row` — the incumbent (R1–R3 and every arm above) |
| `fq_w4b_read_chunkmean` | gpu16 | `chunk-mean` — the shipped default, and the read the dossier indicts |
| `fq_w4c_read_ownspan` | gpu17 | `own-span-mean` — the documented minimal fix |
| `fq_w4d_read_ownspan_seed2` | gpu18 | `own-span-mean`, seed 970520 |

All four carry `--var-probe --var-probe-steps 16` (the full per-episode oracle
replay was 210 s of a 246 s host episode before the sampled-step flag existed).

**Prediction.** Offline, chunk-mean reads 0.82/0.86/0.89 ndim and
0.45/0.48/0.55 size-R² on lhs/rhs/res; **online** the same read rose to lhs
ndim 0.64 by ~ep17 then decayed to the 0.47 majority baseline by ~ep60 and
stayed, while the vertex probe held 1.00. Prediction: `own-span-mean` beats
`chunk-mean` on `probe/face/*/ndim_acc` **at ep150** by ≥0.05, `last-row`
lands between them, and it is the **decay** that separates them — the peak is
not the metric.

**Falsifier.** If all three decay to baseline by ep60, the read *point* is not
the binding constraint — the collapse is upstream in the shared palimpsa rows
and the next lever is row-collapse telemetry, not a fourth read variant. Width
is already excluded (lhs size-R² pinned at ~0 from E=32 to E=512, slope
+0.021/doubling; reaching 0.22 extrapolates to E ≈ 1e5), so a negative here
leaves only the learned-query decoder at E=256 — a build, not a launch.

---

## Wave 5 — CONFIRMATION (5 seeds, 10 runs over 3 rounds, **45 node-h**)

The single best configuration versus its paired control, **5 seeds each**, no
other difference. Generated by adding ten entries to `ARMS` with `--seed`
overridden once the winner is known; the generator is the only supported way.
Four nodes and ten runs means 3 rounds (4 + 4 + 2). **This is what a
distribution claim costs, and no headline number leaves the campaign without
it.** A flat best-so-far curve after an early hit is luck, not learning.

---

## Cost summary

| wave | arms | wall/arm | node-h | GPU-h | gate to the next |
|---|---|---|---|---|---|
| 0 | 1 GPU + 2 CPU | ≤12 h / ≤4 h | 12 (+6 CPU-h) | 48 | gates green; P1 list; A6 landed |
| 1 | 4 | ~4.5 h | 18 | 72 | contrast at some B, or not |
| 2 | 4 | ~16 h | 64 | 256 | order beats reverse, or closed |
| 3 | 4 | ~4.5 h | 18 | 72 | conditional on wave 1 |
| 4 | 4 | ~5 h | 20 | 80 | slot-in |
| 5 | 10 | ~4.5 h | 45 | 180 | the published claim |
| | | **total** | **177** | **708** | |

Basis: R1/R2/R3 completed 250 episodes at `--latency-inner-reps 50` on one
4-GPU node in **4:05–4:48**. Wave 2 is scaled ×3.5 for the order search.

---

## THE DECISION TABLE

What each result makes us do next. One row per experiment; no row says
"investigate further".

| # | experiment | result | next action |
|---|---|---|---|
| 1 | W0-A gates | green | launch wave 1 |
| 2 | | any gate red **or skipped** | fix the gate. No arm launches: the ratio contract is what makes every gradient in the campaign meaningful |
| 3 | W0-B P1 | ~31 candidates reproduce, `k21/f1` absent | feed the list to W0-C; correct every write-up naming `k21/f1` |
| 4 | | no face beats the drift floor at quality > 0.9 | **cancel X1 and X2.** The singleton-skip premise is dead; wave 2 becomes the whole campaign |
| 5 | W0-B P2 | DIAG moves off 0/103 under `--per-face-masks` alone | add `--per-face-masks` to the shared stack from wave 1 on; leave `--diag-per-face` off (the factor is not an action) |
| 6 | | DIAG still 0/103 under both flags | keep DIAG in the action space as a **documented no-op**; no later result may credit it; note it in the thesis as a negative |
| 7 | W0-B P3 | SNN + analytic families moved | re-state every pre-`744fc3d` quality number for those families before it is quoted |
| 8 | W0-C X2 | full-coverage frontier beats the drift floor but not 0.53 | **the honest Pareto front is the coverage-clean one.** Every headline speedup is re-quoted against it, not against the frozen-gradient plans |
| 9 | | beam reaches the best plan in ≲ a few hundred evaluations | **reframe: RL-as-search is settled here; the thesis question becomes RL-as-amortization.** Tell the owner before wave 3 — it changes what wave 5 should confirm |
| 10 | | beam needs > 10⁴ evaluations | search is expensive here; the RL framing survives on cost grounds. Proceed unchanged |
| 11 | A6 | landed and EQ_DUMP-inert with the flag off | enable in every wave-1..4 arm; **X3 becomes runnable** as an offline analysis |
| 12 | X3 (analysis) | COMPRESS's +1–3 % is attributable to specific lowered-graph regressions | remove those from the action space or fix the lowering; re-run wave 1's winner |
| 13 | | the regression is diffuse | drop COMPRESS from the action space on this target and say why |
| 14 | W1 | B=5 shows spread > 1e-3 and ≥1 plan outside the drift floor | **B=5 is the operating point.** Export `W1_BIAS=5`; run wave 2, then wave 3 |
| 15 | | B=4 also shows contrast but worse latency at worse quality | confirms the more-rules-no-more-speed effect; hand it to X3 as its cleanest instance |
| 16 | | no B shows contrast | **NONE-bias hypothesis dead.** Cancel wave 3. Wave 2 is the campaign. Do not sweep B wider — B=4 is already 19.5 approx/plan, past the band every win lives in |
| 17 | | `grad_cov/rejected_this_ep ≈ plan/n_live` for many episodes | the policy is being *refused*, not learning. Switch the rejected plan's score to its own order's exact floor (`_order_floor` exists) and re-run that arm alone |
| 18 | | λ=16 ≈ λ=170 | λ is closed; **cancel wave 3's price arms**, keep only the KL arm and reassign two nodes to wave 4 |
| 19 | W2 | `w2c` beats exact reverse below 0.95, paired | **the order axis is the campaign's result.** Go to wave 5 on it, 5 seeds. Memory becomes a reportable channel for the first time |
| 20 | | `w2c` inside the drift floor over 250 eps | **the order axis is closed.** Report the negative. Do not extend episodes — R-battery arms converged in 250 |
| 21 | | `w2b` beats `w2c` materially | order and approximation *do* compose; that contradicts the registered prediction and is the more interesting result — wave 5 confirms `w2b` |
| 22 | | `w2a` disagrees with its wave-1 counterpart beyond the drift floor | the wave-1 reading was drift. Re-state it; all cross-battery comparisons in this plan become invalid and only in-battery pairs are quoted |
| 23 | | `w2d` ≈ `w2b` | the edge-keyed memory is inert even off the pin — its only predicted-useful regime — **retire the flag** |
| 24 | W3 | λ=400 parks at identity, 130 ≈ winner | the usable band is bracketed; publish it as `[~130, <400]` and stop sweeping |
| 25 | | all three λ indistinguishable | price is not a live knob across 3×; **retire λ from the backlog** |
| 26 | | KL arm finds a *better* plan | A5's docstring claim is wrong. Do not use KL until that is explained |
| 27 | W4 | `own-span-mean` holds ndim ≥ baseline+0.05 at ep150 | make it the default `--face-read`; re-run the wave-2 winner on it before wave 5 |
| 28 | | all three decay to baseline by ep60 | the read point is not the constraint. **Do not build a fourth read variant.** The remaining option is the learned-query decoder at E=256 — an owner decision, not a launch |
| 29 | | seeds w4c and w4d disagree in sign | the effect is smaller than seed noise; report no effect rather than the better seed |
| 30 | W5 | winner beats control across 5 seeds, paired | publish, with the distribution, never the best-so-far |
| 31 | | overlapping distributions | **there is no result.** Say so. This is the row the whole plan exists to be able to reach honestly |

---

## Backlog items I judged not worth running, and why

* **`--face-endpoint-read` A/B.** `CLEAN_DESIGN_AUDIT.md` g1 marks it
  **CONFLICTS**: the head input becomes
  `[chunk_mean ‖ vmem_slot_i ‖ vmem_slot_j]`, routing the **vertex** memory
  into the **face** head — "precisely what the owner forbids". Omitting it
  costs no code. It needs a ruling, not a node. (Its offline evidence is the
  strongest of any read variant — `mean ‖ endpoint slots` scores
  0.99/1.00/1.00 ndim — which is exactly why it should be an explicit owner
  decision rather than something a battery quietly adopts.)
* **`--face-edge-mem` under the rev pin.** Predicted inert there by its own
  dossier: the lhs operands are all primitive under reverse, the regime the
  endpoint read already covers. Scheduling it against a pinned order would buy
  a guaranteed null. It appears once, in wave 2, where it is not inert — and
  even there it needs the same ruling.
* **Entropy-floor recalibration as its own arm.** §13 shows the floor is not a
  stable unit *within a run*: the same 0.05 means ~0.05 nats/face at
  `mean_valid = 1.2` and ~3.2 nats at `mean_valid = 0.016`, a 60× scale driven
  by the policy's own destruction. The fix — a ratio-of-sums aggregate behind
  `--face-entropy-agg {mean,valid-weighted}` — **does not exist in `ppo.py`**.
  Spending a node on a floor whose meaning moves 60× would produce
  uninterpretable data. It is demoted to (a) a one-flag build and (b) the
  post-hoc correction `entropy/approx_head ÷ faces/mean_valid`, read on every
  arm. §13's recalibration is also *already* served: floor 0.30 corresponds to
  a none-bias of 5.66 and 4.9–8.1 approx/plan, which **is** wave 1's B=5 arm
  by another route.
* **Budget-1 / top-k approximations.** Subsumed: it and the NONE-bias both set
  approximations-per-plan, and B is a knob that exists while top-k is a build.
  Run it only if wave 1 shows the win band is narrow *and* the policy cannot
  hold it — that is the only condition under which construction beats a prior.
* **A `loss_drop` arm retained for logging.** There is no log-but-don't-train
  wiring for quality — `--quality-metric` selects exactly one — so logging it
  costs a whole run. And at `--walk-steps 200` the walk **fully overfits**:
  identity's held-out score is −0.0009 ± 0.124, while at 20 steps it is
  +0.259 ± 0.071, "the only working discrimination in the entire sweep".
  Cheaper and better: recover `loss_drop` **offline** from A6's plan dump by
  replaying through `landscape_map --quality-metric loss_drop --walk-steps 20
  --walk-heldout`. Note this cannot rescue it as a guard either: held-out
  scoring **cannot detect the frozen-gradient hack at any horizon** (gaps
  +0.0035 ± 0.0255 at 200 steps, −0.0010 ± 0.0010 at 20 — sign-inconsistent,
  and at short horizons the plan freezing the most parameters scores
  *higher*).
* **Preference-conditioned `w` now.** `--preference-conditioned` is live on
  `ppo.py` and `pref_proj` is zero-init (a no-op until trained), so it is
  runnable — but with the memory ratio pinned at `1.0000 ± 0.0000` under the
  rev pin there are effectively **two** live objectives, and a
  preference-conditioned front over two channels with no spread has nothing to
  trade off. Revisit **after wave 2**: if the order axis makes memory move,
  preference conditioning acquires a real front to condition on.

---

## Conflicts between the owner's constraints and the findings

1. **"Real measured peak memory is a non-negotiable reward component" vs. the
   memory channel being dead.** Under the pinned order the memory ratio is
   `1.0000 ± 0.0000` for every archived winner and every ladder rung; the only
   plan that moves it is `skip@all` at 0.0406 with quality 0.0000. So in waves
   1, 3 and 4 memory is a trained channel carrying **no signal except through
   destruction**, and `--lambda-mem 1` prices a constant. The constraint is
   honoured (memory stays trained everywhere) and the conflict is resolved by
   wave 2: across *orders* the same channel spans ~80×. **This is an
   independent argument for running the order wave early**, and it is why wave
   2 is scheduled second and unconditionally.
2. **"Sampling variance in the quality signal is wanted" vs. the static
   objective.** `--walk-rotate` supplies the variance (it rotates the
   grad-cosine probe batch, not only the walk's), but it makes the objective
   non-stationary across episodes — exactly what v66's static-objective battery
   was built to remove after diagnosing the critic as the bottleneck. The plan
   follows the owner: rotation ON everywhere. The cost is that a plan no longer
   scores identically at ep 5 and ep 250, so **all quality comparisons must be
   within-episode**. That is fine and in fact tighter: all plans in an episode
   share the batch, so the paired difference has sd 5.5e-6 against a 1.5e-2
   marginal — 2700× tighter. Only cross-episode comparisons need repeats.
3. **"Keep DIAG in the action space" vs. DIAG applying 0 of 103.** Honoured —
   DIAG stays. But until W0-B P2 reports, DIAG is a no-op occupying a third of
   the op alphabet, and the owner's separate constraint that **its factor is
   not an action** rules out `--diag-per-face`, which is one of the two things
   A1 built to move it. If P2 shows only `--diag-per-face` moves DIAG, the two
   constraints are jointly unsatisfiable and the owner must pick.
4. **`k21/f1` is load-bearing in the campaign's headline and is not a live
   face.** Ratio 0.5325 at quality 0.9257 — "the best free single skip" — is
   quoted in the Pareto dossier, in the coverage forensics and in the X1
   candidate list. A3 substituted `k14/f0` (vertex 81). Until W0-B P1 lands,
   **no write-up may use it**.
5. **5 seeds vs. 4 nodes vs. one battery at a time.** A distribution claim over
   one configuration and its control is 10 runs = 3 rounds ≈ 45 node-h. It
   cannot be folded into a 4-arm battery. Wave 5 is scheduled as its own
   battery and costed accordingly; the earlier waves are explicitly **screens**
   and no distribution may be claimed from them.
6. **`--ppo-epochs 1` makes PPO's clip inert.** At one epoch the importance
   ratio is identically 1 and the clip never binds, so what runs underneath is
   REINFORCE with a baseline. That is R2's deliberate choice and this plan
   keeps it — but the campaign should stop describing these runs as PPO
   results.
7. **`ALPHAGRAD_NEW_SLOT_JOIN` polarity is contested in the record.**
   `UNBIASED_PARETO_AND_MEASUREMENT.md` §7(c) says `=1` emits a face form
   graphax rejects, so every res-slot plan dies **silently** in
   `_trace_truncate` with no counter and no log line; the R launchers'
   committed G4 note says the working tree accepts it (`e5fd46c`,
   `graphax tests/misc/test_face_two_op_form.py`) and set `=1`. The plan runs
   `=1` and the launchers **verify it in pre-flight**, aborting with exit 70
   rather than trusting either document. The unresolved consequence stands:
   "audit v57–v66 for `NEW_SLOT_JOIN=1` casualties" is still open, and any
   archive statistic from that era may be biased by silently-dropped plans.
8. **`landscape_map` cannot name `grad_cosine`.** Its `--quality-metric`
   choices are `loss_drop, cosine, none`. On a scalar-loss target `cosine`
   aliases to `grad_cosine`, so W0-B is measuring the right channel — but the
   alias is deprecated and warns, and the tool should be widened so the
   measurement instrument and the trainer name the same thing.

---

## Findings recorded after this plan was written (2026-08-29)

Appended, not merged. Nothing above is restated and **the decision table's
numbering is unchanged** — rows 1 and 2 ("W0-A gates green" / "any gate red
**or skipped**") already govern the new gate in (b) exactly as they govern the
two gates that were there before it.

### (a) Gradient coverage is NECESSARY but NOT SUFFICIENT

Measured on the W0-C stand-in run: `coverage_beam.py` at `7313978` over
`run_analysis/landscape/rows_qb_sweep.csv` with the quality/ratio pre-filter
opened up (`--quality-min -2 --ratio-max 2 --no-beam --max-depth 1`), output at
`~/dsnn/run_analysis/w0c_dev/singleton_census.json`.

> **PROVENANCE (added 2026-08-30, ticket 22).** This census reads
> `run_analysis/landscape/rows_qb_sweep.csv`, produced by a `landscape_map.py`
> stamped `tool=3cb4ceb2` which **matches no commit**; it survived only inside
> the mutable `.ag_pin_landscape/` directory and is now frozen at
> `refs/archive/landscape_map_3cb4ceb2` in `~/alphagrad.git`. Its face indices
> are on a **118**-live-face graph (`--seed-vertices` defaulted True, since
> removed by A4); the current tool enumerates **117** and every `k` is shifted
> by one, so **`k7/f0` names `v88/exp` only under that instrument** -- re-match
> by `(vertex, primitive)`, not by index. The `min_leaf_ratio 1.0`,
> `frac_zeroed 0.0`, `quality 0.0000` and `ratio 0.4711` figures themselves are
> unaffected, as is the finding. Its `quality` column is `loss_drop`.

Face **`k7/f0`** (label `v88/exp`, vertex 88) scores gradient coverage
**perfectly clean**: `min_leaf_ratio` exactly **1.0**, `frac_zeroed` **0.0**,
15 of 15 leaves counted, 0 uncountable — *bit for bit the identity plan's own
census*, which the tool's own guard reads as `min_leaf_ratio 1.0 /
frac_zeroed 0.0 / n_leaves 15`. Its **quality is 0.0000** (three paired
trials, `[0.0, 0.0, 0.0]`) at ratio **0.4711** (median of `[0.47113, 0.48407,
0.40290]`).

Every parameter still receives a nonzero gradient. **The gradient simply
points the wrong way.** Coverage can ask *"is any leaf frozen?"*; it cannot
ask *"is the direction right?"*, and a face that rotates or rescales every
leaf without zeroing one is invisible to it.

**The beam's quality constraint comes from outside the beam.**
`coverage_beam.py` scores subsets by coverage ALONE — `rank_key = (frac_zeroed
ASC, -min_leaf_ratio ASC, tiebreak)` and `is_clean = (n_zeroed == 0)`. Quality
enters at exactly one point, the CSV row filter `c.quality > quality_min`, so
the constraint is *carried in from P1's measurements* and never re-evaluated
anywhere in the search. That is why `7313978`'s validation looked sane:
`--quality-min 0.9` had already deleted the destroyers before the beam saw
them. With the pre-filter opened up, the unfiltered census duly **crowned a
quality-0 plan its best coverage-clean subset** — greedy's `final_subset` is
`["k7/f0"]`, `eval_counts.best_known_plan` is `k7/f0`, reached at evaluation 3
of 92. Six of the 92 live singleton candidates have quality exactly 0.0, and
they are the six **cheapest** on the board (ratios 0.4635 `v87/sub`, 0.4650
`v93/mul`, 0.4711 `v88/exp`, 0.4716 `v95/neg`, 0.4801 `v94/reduce_sum`, 0.4882
`v92/log`). Coverage rejects five of the six and misses one — and the one it
misses is the one that then wins.

**Two consequences to carry into W0-C, both stated as limits on what its
output is worth:**

1. **Composed multi-face subsets are NEVER quality-checked.** The pre-filter
   is a per-singleton fact read off a CSV. Nothing in the beam or the greedy
   walk measures the quality of a two-or-more-face subset, so a subset of
   individually-acceptable faces that is jointly destructive survives the
   screen with a clean coverage score.
2. **The survivors' predicted ratios are a product-of-singletons heuristic,
   not measurements.** `predict_ratio(..., how="prod")` multiplies the member
   faces' own singleton ratios; single-face latency ratios are not composable
   in general. The tool labels it NOT MEASURED in every output and it never
   decides survival — but it *is* the sort key of the survivor list, so "the
   best survivor" is picked by that heuristic.

**Therefore:** coverage stays the screen — it costs 0 node-h and it does
reject most destroyers — but its survivor list is a **candidate list, not a
result**. Nothing from it may be quoted until a **paired GPU pass** measures
the composed subset's quality *and* its latency ratio against the same-order
exact reference, back to back (§0's drift rule). Decision-table rows 8/9/10 are
unchanged; this bounds what feeds them.

### (b) `--ray-measure` measured NOTHING for 19 hours, and no gate saw it

**The bug.** `4c4d872` (2026-08-28 05:18, this session's pooled walk-rotation
work) gave `CpuApproxPool.evaluate` / `evaluate_batch` an `episode` field and
made both forward it to `actor.evaluate.remote(...)`.
`CpuApproximationActor.evaluate` — the single hop between the pool and
`CpuApproximationServer.evaluate`, which had accepted `episode` all along —
was never given the parameter. Every pooled dispatch therefore died with

```
TypeError: got an unexpected keyword argument 'episode'
```

the pool sentinelled that row and killed the actor, and the remaining slots
came back `[SENTINEL] batch pool-drained (actor died)`. Under `--ray-measure`
**nothing was measured at all**: every terminal reward of every env of every
episode was the degenerate sentinel (−1e10 on all six cost channels,
`grad_coverage` −1, quality 0). Fixed by `87cdc49` (2026-08-29 00:35), a
15-line restoration of the pass-through. **The window is 19 h 17 min.**

**Why it matters to this plan specifically.** Every wave-1..4 arm runs
`--ray-measure 3` (`gen_fq_launchers.py`, `SHARED_CLI`). This *is* the
production measurement path. A run in this state exits 0, prints health rows
and steps PPO: the **177 node-hours** of §"Cost summary" would have been 177
node-hours of pure sentinel that still looked like a training run.

**Why no gate saw it.** `ratio_gates.sh` pins the PPO importance ratio;
`smoke.sh` pins finite post-warm-up health rows; **neither starts a measure
pool**, and the smoke's canonical config contains no `--ray-measure`. The only
trace was `[SENTINEL]` lines that nothing reads. It was caught by accident,
and late: A6's plan log came back silently **empty** on the pooled path, which
is a symptom nobody would have looked for on an arm that was not building a
plan log.

**The R battery predates it and is unaffected — but two numbers in the note
that said so were wrong, and are corrected here.** R1/R2/R3 are Slurm jobs
`62411` `r1-trim` / `62412` `r2-trimplus` / `62413` `r3-credit`; `sacct` puts
all three on **2026-08-27**, `02:50:12 → 07:38:19`, `02:50:13 → 07:28:21` and
`02:50:13 → 06:56:09` on `pgi15-gpu15/16/17`, one 4-GPU node each, all
`COMPLETED` at `04:48:07` / `04:38:08` / `04:05:56` — which is where
§"Cost summary"'s 4:05–4:48 basis comes from, and it holds. They did **not**
run on 2026-08-12/13 (that date belongs to the unrelated `ord-r0..r3` ordering
jobs `60141–60144`, which are 2-GPU and two of which TIMEOUT).

The margin is therefore **19 hours, not two weeks**: the last R run ended
2026-08-27 07:38 and `4c4d872` landed 2026-08-28 05:18. It is still a clean
margin — no R run ever executed a line of the broken wire — but it is a
narrow one and worth stating precisely rather than comfortably.

Their sentinel count is **0, not 49–65**:

```
$ grep -c "\[SENTINEL\]" r{1_trim_62411,2_trimplus_62412,3_credit_62413}.log
0
0
0
```

A case-insensitive `grep -c SENTINEL` returns 2 per log, and **both hits are
the string `ALPHAGRAD_MULS_SENTINEL_CAP=5e12` inside the worker's env dump** —
a tunable's name, not an event. (That is the same false-positive shape as the
`truncated': 0` telemetry-key incident; the count is quoted here with its
matched text for that reason.) Zero is a real "no pool failures", not a
missing emitter: `cpu_approx_pool.py` prints `[SENTINEL]` from ten call sites
and these runs exercised the pool (`ALPHAGRAD_BATCHED_CALLBACK=1`, actor
output throughout).

None of this changes the conclusion, and it strengthens the contrast: the R
battery's real background rate is **0** sentinels over 250 episodes × 3 runs,
against **400 sentinels and 0 measurements in one 1-episode / 4-env
reproduction** of the bug. **The credit-horizon result stands** (γ=λ=1 held at
quality 0.885; γ<1 drifted to destruction at ep131).

*Unrelated caveat found while checking this, recorded because it bears on the
same wall times:* R2 and R3 both end with `nvlink fatal : Input file ... newer
than toolkit (129 vs 128)` and compile-fallback counters (`compile FALLBACK
#603` in R2, `#481` in R3); R1 has none. Those two ran a nontrivial share of
their steps under degraded fusion, so 4:05–4:48 is a *pessimistic* scaling
basis, not a clean one.

**The gate.** `tools/pool_liveness_gate.sh` + `tools/pool_liveness_check.py`,
added to W0-A above. It is demonstrated in both directions rather than
asserted: with `87cdc49` reverted in a scratch tree it reports

```
CONTRACT: RED -- the pool/actor/server wire contract is BROKEN
  ! cpu_approx_pool.py:493: forwards `episode=` to CpuApproximationActor.evaluate(), which does NOT accept it
VERDICT: [SENTINEL] lines = 400  by kind={'dispatch-error': 200, 'pool-drained': 200}
VERDICT: plan records = 0  measured in a measure actor = 0
POOLED-MEASUREMENT LIVENESS GATE: RED -- MEASUREMENT IS DEAD
```

while the pre-fix training run it is reading **exited 0** and dumped a Pareto
front of `Total Reward: -2.00e+10 | CMP(flops): 1.00e+10 |
Quality(loss_drop): 0.0000 | Mem(peak_memory): 1.00e+10` — the degenerate
sentinel, ten times over, presented as a front. That contrast is the whole
argument for the gate.

*One false positive was found by running it, and is recorded because it is a
fact about this campaign's configuration and not only about the gate.* The
first version failed at HEAD too: 16/16 plans came back `sentinelled` with all
six cost channels at −1e10, because `--reject-frozen-grads` (default ON, and
right to be) **returns early before the cost channels are measured**, and at
episode 0 an untrained face policy on the 25-vertex NeuralNetwork graph
samples SKIPs that freeze every trainable leaf. On the reward vector alone
that is indistinguishable from the dead pool. So: **a fully guard-rejected
episode and a dead measurement path write the same terminal reward**, and any
future check that reads only the reward vector will confuse them. The gate's
own run now passes `--no-reject-frozen-grads` so its verdict depends on the
transport rather than on what a random policy sampled, and its degeneracy
check demands *positive* evidence — at least one real number on a cost channel
— rather than the absence of a sentinel, which an empty log also satisfies.
