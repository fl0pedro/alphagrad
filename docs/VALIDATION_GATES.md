# Validation gates: which gate proves what

Four gates exist. They are **not** interchangeable, and the weakest one was for a
long time the only one anybody ran. Pick by what your change can break.

| Change | Gate to run |
|---|---|
| Anything at all (cheap pre-flight) | `tools/smoke.sh` |
| Policy / masking / head wiring / tokenisation | unit tests + `tests/policy_regression_gate.py` |
| "This flag is off, so nothing changed" | `ALPHAGRAD_EQ_DUMP` |
| "This is faster / uses less memory" | paired-ratio measurement |

---

## 1. Unit tests — pin the *contracts*

### The command is `pytest`. Not `pytest tests/`.

```
pytest                 # 1460 tests: BOTH declared roots
pytest tests/          # 1371 tests: the 89 in src/alphagrad/approx/tests are invisible
```

`pyproject.toml` declares `testpaths = ["tests", "src/alphagrad/approx/tests"]`
and says that naming both roots *"is what makes the suite a gate rather than a
sample"*. Until 2026-09-12 every number this campaign quoted came from `pytest
tests/`, which overrides that declaration — including the `14 pre-existing
failures` baseline. The 89 unseen tests include `test_env_callback.py` (the
measurement callback) and `test_jacobian_equals_grad.py` (the gradient oracle
the grad-cosine channel rests on). Measured 2026-09-12: all 89 pass on both
sides of the fix14 diff, in 48 s. Use `pytest`.

### `EXHAUSTIVE=1` — the suite's count is a DRAW, and this measures the spread

```
EXHAUSTIVE=1 pytest                       # repeats the measurement tests 50x, rest once
EXHAUSTIVE=1 EXHAUSTIVE_REPEATS=100 pytest tests/mem_channel_test.py \
                                          tests/paired_log_reward_test.py \
                                          tests/measure_instrument_test.py \
                                          tests/test_all_cost_channels.py
```

Some tests here assert on a **wall-clock latency**, on a **drift ratio between
two adjacent timings**, or on a **measured byte count**. Their result is a draw
from a distribution, so `0 failed` is one sample, not a fact, and two runs of
the same commit can legitimately differ. There is no enumerable sample space —
the randomness is wall-clock assertions, allocator state and compile-cache
state — so repetition is the only instrument.

`EXHAUSTIVE=1` repeats every test declared in
`_pytest_config_guard.MEASUREMENT_TESTS` `EXHAUSTIVE_REPEATS` times (default 50)
and prints a **per-test failure rate** plus the worst rate as the suite's noise
floor, so a landing decision can be stated as *"N always, plus these M
sometimes"* instead of a single integer. Everything else runs once, which is the
check that the remainder is deterministic.

`MEASUREMENT_TESTS` is a **declaration with a reason per entry**, derived by
reading every assertion in both roots. A test that starts asserting on a clock
belongs in it; one that stops should leave.


The one that matters most is **sampling == replay**, i.e. the PPO ratio is
exactly 1 at epoch 0. The stored old log-prob *is* the sampling log-prob, so any
divergence means the loss reconstructs the behaviour policy differently from how
it sampled. That bug ran the ratio to 2.3e23 and was invisible in a
batch-averaged KL.

It is pinned by, in rough order of directness:

* `tests/per_face_masks_test.py::test_sample_equals_evaluate_with_per_face_sizes_and_quant`
* `tests/per_face_sizes_live_test.py::test_sample_equals_replay_with_live_derived_sizes`
* `tests/live_vertex_mask_test.py::test_sample_and_evaluate_agree_under_the_same_masks`
* `tests/endpoint_read_test.py::test_flag_on_rollout_equals_replay`
* `tests/masked_extend_equivalence_test.py::test_valid_prefix_is_bitwise_identical`
  (a 1e-7 drift in `enc_x` becomes a ratio != 1 and a spurious first-epoch update)

The other contract a measurement run depends on is **the measured object is the
gradient a training run computes**, pinned by
`src/alphagrad/approx/tests/test_jacobian_equals_grad.py`:

* the REGISTERED TARGET IS MODEL + LOSS: `examples.get_fn(name)` returns a 0-d
  scalar for every trainable family in the registry, so `jacve` of the traced
  graph is a gradient with no flag involved. `--measure-grad` is a deprecated
  no-op (it used to switch the target to the raw per-element example, i.e. a
  per-class Jacobian); `--no-measure-grad`, which asked for that mode, is a
  hard error in `tools/landscape_map`;
* `graphax.jacve` of that graph equals `jax.grad` of the same loss, leaf by
  leaf, at `max|jacve-grad|/max|grad| <= 1e-5` AND graph-wide
  `||jacve-grad||/||grad|| <= 2e-6`, for all 18 trainable families (worst
  per-leaf 8.8e-6 TransformerLM3; `Encoder` 1.7e-5 and `EncoderDecoder` 8.2e-5
  carry a documented per-leaf relaxation because one of their leaves has a
  near-zero gradient -- their graph-wide residuals are 3.2e-7 and 1.5e-7);
* the reduction is the one a real training run optimizes and is written NEXT TO
  THE MODEL in `examples.get_fn`, not inferred from `ndim` and not looked up in
  a reduction table at wrap time. The seven analytic AD benchmarks (Simple,
  Lighthouse, Helmholtz, RobotArm_6DOF, RoeFlux_1d/3d, BlackScholes_Jacobian)
  have no training run and are EXEMPT by name (`has_scalar_loss`): their target
  is the full multi-output Jacobian, which is what they exist to measure;
* `downstream_train` — this repo's definition of "a real training run" —
  optimizes the SAME registered target (it fetches it by name), so the two
  cannot drift apart. `scalar_loss_fn` survives only as an assertion that its
  argument is 0-d; it adds no equation.

`tests/policy_regression_gate.py` is the broader tripwire: a seeded rollout
through the real policy path, compared **bit-exactly** against a committed
semantic golden (realized choices, available-vertex sets, mask fill values, live
per-face wires and legality masks, every log-prob/entropy/value as float32 bit
patterns, token-chunk counts and hashes). It is shape/padding independent on
purpose, so re-bucketing does not fail it but a moved mask sentinel does.

```
JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py
```

**These are the gate for the ratio.** The smoke does not check ratio == 1; it
only checks that the ratio statistic is finite (see below).

Note what the health line's `ratio/max_log` actually is: episode metrics are
reduced by `mean` over the `(ppo_epochs, minibatches)` scan axes, and
`--ppo-epochs` defaults to 2. Epoch 0 contributes exactly 0 by construction and
epoch 1 contributes a real, nonzero ratio, so a healthy run shows a small
nonzero number (~0.03-0.05 on the canonical smoke), **not** 0. A broken epoch-0
ratio is diluted by `ppo_epochs` but not hidden — the 2.3e23 regression would
still read ~1e23 — so treat this number as a gross-breakage tripwire, never as
proof that sampling == replay. The unit tests above are that proof.

## 2. `tools/smoke.sh` — proves the pipeline runs *and the numbers are finite*

```
srun -p pgi15-cpu -w pgi15-cpu1 --mem=64G -c 8 -t 2:00:00 tools/smoke.sh
tools/smoke.sh Helmholtz            # one example
```

Runs the canonical 2-episode CPU config (`--live-faces --grad-window 0 --wandb
disabled`, `--episodes 2 --num-envs 2 --seed 42`) on Helmholtz and
NeuralNetwork, and **exits non-zero** when, on any post-warm-up `[health ep..]`
row, `ppo` / `value` / `ent` / `ratio/max_log` is NaN, Inf, or missing — or when
no post-warm-up row was printed at all.

Runtime: a few minutes per example on CPU with a warm
`JAX_COMPILATION_CACHE_DIR`; the first, cold run is dominated by compile.

**Guarantees:** the pipeline runs end to end, and the loss, the value loss, the
entropy and the ratio tail statistic are finite on real gradient steps.
**Does NOT guarantee:** that the ratio is *correct* (only that it is finite),
that learning works, or that anything is fast. `rc=0` alone never meant more
than "it ran".

### Why warm-up rows print `n/a`

`--popart-init-episodes` (default **3**) runs random-plan rollouts with **no
gradient step**; the caller passes an all-NaN metrics tuple purely to keep the
tuple shape uniform. Those rows are labelled `[health warmup]` and print `n/a`
for every loss-derived field.

This used to be a trap. Warm-up rows consumed the health budget
(`ALPHAGRAD_HEALTH_EPISODES`, also default **3**), so the three health rows a run
printed were *always* the three warm-up rows, reading
`ppo=nan value=nan ent=nan ratio/max_log=nan`. The first real training episode
never printed one. Multiple agents read those expected NaNs as either
pre-existing breakage or as fine — and a genuine NaN would have been
indistinguishable. Warm-up rows no longer consume the budget, so **a `nan` on a
`[health ep..]` row is always real.**

## 3. `ALPHAGRAD_EQ_DUMP` — bit-identity between two builds

The gate for "my change is inert when its flag is off". For that specific claim
it is much stronger than the smoke: it compares *state*, not just finiteness.

But it is **not** strictly stronger, because it is a *differential* gate. It
compares two arms against each other and says nothing about either in absolute
terms — if both arms are broken in the same way it reports "bit-identical" and
passes. That is not hypothetical: an EQ_DUMP run proved 334 bit-identical leaves
per episode at the same time as every health row in both arms read
`ppo=nan value=nan ent=nan`. Both results were correct. Use EQ_DUMP for
inertness and the smoke for health; neither substitutes for the other.

```
ALPHAGRAD_EQ_DUMP=$R/eq_base  <run arm A>     # e.g. PYTHONPATH=<pre-patch src>
ALPHAGRAD_EQ_DUMP=$R/eq_off   <run arm B>     # patched, flags absent
# then diff leaf by leaf: eq_base.ep<N>.pkl vs eq_off.ep<N>.pkl
```

Writes one pickle **per training episode**, after the PPO update. Scope:

* **covers** — `params` (all agent arrays), `opt` (optimizer state), `metrics`,
  `actions` (the full action pack), `rewards` (per-env reward vectors),
  `step` (global step), `ret` (true scalar return). Compared with
  `np.array_equal(..., equal_nan=True)`; A5 got 334 bit-identical leaves per
  episode this way.
* **does NOT cover** — the PopArt warm-up episodes (dumped only from the main
  loop), per-step policy internals (masks, face wires, token streams — that is
  `policy_regression_gate.py`'s job), wall-clock or memory, and anything on a
  path the run does not exercise. Bit-identity also cannot tell you that
  identical behaviour is *correct* behaviour.

Runtime: two full smoke runs plus the comparison, so roughly **2x the smoke**.
Both arms must use the same seed and config; the dump is pure instrumentation
and nothing downstream reads it.

## 4. Paired-ratio measurement — the only gate for performance claims

GPU state drifts ~18-20% on its own. An unpaired A-then-B comparison once
produced a "17.5% beats reverse" that evaporated to a ratio of 1.00 when the
candidate and the reference were measured back to back.

So: measure candidate **and** reference in the same process, back to back, and
report the **ratio** plus a distribution over **5 seeds** — never a single
best-so-far number. A flat best-so-far curve after an early hit is luck, not
learning.

---

## 5. Running the ratio gates — **one process each**, and a skip is a failure

`tools/ratio_gates.sh` is the runnable form of section 1. It runs each
sampling-==-replay pin in **its own python process**:

```
tests/endpoint_read_test.py::test_flag_on_rollout_equals_replay
tests/per_face_masks_test.py::test_sample_equals_evaluate_with_per_face_sizes_and_quant
tests/per_face_sizes_live_test.py::test_sample_equals_replay_with_live_derived_sizes
tests/live_vertex_mask_test.py::test_sample_and_evaluate_agree_under_the_same_masks
tests/masked_extend_equivalence_test.py::test_valid_prefix_is_bitwise_identical
tests/policy_regression_gate_test.py
```

```
PY="uv run --no-sync python" tools/ratio_gates.sh
```

The process isolation is **load-bearing, not tidiness**.
`alphagrad.approx.env` freezes `MAX_DELTA_TOKENS` and the `MAX_FACES` default
into module constants at *its first import*, so a test module that asks for a
small test scale with `os.environ.setdefault` only gets it when it is the first
module in the process to import alphagrad. Batch these gates into one pytest
process and `tests/endpoint_read_test.py` reads `ALPHAGRAD_MAX_DELTA_TOKENS=128`
for its stand-in face chunks while `_face_loop` is still building a 32768-wide
chunk-stream buffer:

```
ValueError: Incompatible shapes for broadcasting: shapes=[(32768,), (128,), ()]
```

That is **scaffolding, not the mirror**. Production builds the face chunk
callback as `LiveFaceStream(..., window=MAX_DELTA_TOKENS)` from the very
constant `_face_loop` sizes the stream with, so the two cannot disagree — and a
shape error is a *trace-time crash*, never a silently wrong ratio.
`tests/_scale_guard.py` now turns the batched case into a **named skip** with
that explanation instead of a broadcast error 200 frames down, and this runner
**treats a skip as a failure**: a gate that did not run pins nothing.

---

## 6. `tools/pool_liveness_gate.sh` — does `--ray-measure` measure anything?

Sections 1–5 pin what the numbers *mean*. This one pins that there **are**
numbers. Nothing above it starts a measure pool: the ratio gates never leave
one process, and `tools/smoke.sh`'s canonical config has no `--ray-measure` in
it at all.

`87cdc49` is why that gap is not academic. `4c4d872` gave
`CpuApproxPool.evaluate` / `evaluate_batch` an `episode` field and made both
forward it to `actor.evaluate.remote(...)`, but `CpuApproximationActor.evaluate`
— the one hop to `CpuApproximationServer.evaluate`, which had accepted
`episode` all along — never got the parameter. Every pooled dispatch died with

```
TypeError: got an unexpected keyword argument 'episode'
```

the pool sentinelled the row and killed the actor, and under `--ray-measure`
**nothing was measured at all**: every terminal reward was the degenerate
sentinel (−1e10 on all six cost channels, `grad_coverage` −1, quality 0). For
19 hours the run still exited 0, still printed finite `[health ep..]` rows and
still stepped PPO. The only trace was `[SENTINEL]` lines no gate reads.

```
srun -p pgi15-cpu -w pgi15-cpu1 --mem=96G -c 24 -t 0:30:00 \
     tools/pool_liveness_gate.sh
```

Two parts, **both always run** — part 1 is a proxy, part 2 is the evidence,
and there is no fast-fail because skipping the evidence when the proxy looks
fine is the failure mode this gate is written against:

1. **CONTRACT** — an `ast` read of the pool → actor → server call chain. Every
   keyword the pool forwards through `.evaluate.remote(...)` must be a
   parameter the actor wrapper accepts, and every keyword the wrapper forwards
   to `self._impl.<m>(...)` must be one the server accepts. No ray, no jax,
   ~10 ms, and it names the offending kwarg and line.
2. **LIVE** — a real 2-actor / 4-env / 1-episode `--ray-measure` run on
   NeuralNetwork with `--plan-log`, then `tools/pool_liveness_check.py
   verdict`: the pool must have started, there must be **zero `[SENTINEL]`
   lines of any kind**, at least one terminal plan must carry an `actor` stamp
   (i.e. it was measured *inside* a measure actor, which is what
   `measure_pool.merge_pool_plan_records` records), and at least one recorded
   plan must carry a **real number on a cost channel**.

Three things about it are deliberate:

* **It exports `ALPHAGRAD_BATCHED_CALLBACK=1` itself.** Without it
  `--ray-measure` exits with `ValueError` before starting a pool, and a gate
  that reported "measurement dead" there would be lying: it never got to test
  one. That case is detected **by name** and reported as **exit 2, HARNESS
  MISCONFIGURED**, distinct from **exit 1, MEASUREMENT DEAD**. Both are
  failures — a skip is a failure — but they call for opposite responses.
* *(2026-09-03: the guard described in the next bullet was removed — owner ruling 2026-09-03, ticket dsnn-3qm.15;
  the gate no longer passes the flag.)*
* **It passes `--no-reject-frozen-grads`.** The frozen-gradient guard is
  default ON and right to be, but it returns early with `_SENTINEL_BAD_REWARD`
  *before* the cost channels are measured. At episode 0 an untrained face
  policy on a 25-vertex graph samples SKIPs that freeze every trainable leaf,
  so at HEAD with the guard armed 16/16 plans came back `sentinelled` with all
  six cost channels at −1e10 — on the reward vector alone, indistinguishable
  from the dead pool. Turning the guard off makes the verdict depend on the
  transport rather than on what a random policy happened to sample. (This was
  a real false positive, found by running the gate at HEAD, not a hypothetical.)
* **Check (4) demands positive evidence.** "No degenerate cost vector" is not
  enough — an empty log satisfies it. At least one plan must carry a real cost.

Demonstrated in both directions, which is the only thing that makes a gate a
gate: with `87cdc49` reverted in a scratch tree it goes **red** (contract names
all three `episode=` call sites; the live half reports 400 `[SENTINEL]` lines
and 0 plan records) while the pre-fix training run it is reading **exits 0**;
at `2ddbeef` it goes **green** (contract ok, 0 sentinels, 16/16 plans measured
across both actors). ~6 min per direction on `pgi15-cpu1`.
