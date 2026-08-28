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

* the traced target is a SCALAR loss with `--measure-grad` on *and* off (it used
  to be the raw per-element example without the flag, i.e. a per-class Jacobian);
* `graphax.jacve` of that graph equals `jax.grad` of the same loss, leaf by leaf,
  at `max|jacve-grad|/max|grad| <= 1e-5` (measured: 1.3e-7 NeuralNetwork,
  1.1e-7 VmappedNeuralNetwork, 6.4e-7 TransformerLM-shaped);
* the reduction is the one a real training run optimizes and is DECLARED per
  family in `examples.loss_reduction`, not inferred from `ndim`;
* `downstream_train` — this repo's definition of "a real training run" — calls
  the same `scalar_loss_fn`, so the two cannot drift apart.

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
