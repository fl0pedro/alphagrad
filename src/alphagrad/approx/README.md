# `alphagrad.approx` — micro-action masking, measurement, and current wiring

Notes on the changes made in the 2026-07-22/23 session. Written to be read by
someone about to develop here, so it says what is *actually* wired as well as
what exists.

## 1. Micro-action legality is a property of the LIVE edge

`graphax` fails loudly when a micro-action does not fit the tensor it lands on
(`TRANSFORM DID NOT FIT …`) and tells callers to mask invalid actions up front.
Nothing built that mask, so an exploring policy hit it constantly — ~59 failures
per episode in a real `ppo_ray` run, each one costing a sentinel penalty.

The mask lives in the **env** (`common/masks.py`), not in graphax: the action
space is the RL problem's concern, not the AD kernel's.

### The rules

`Diag(i, j, factor)` — `i`/`j` index the **concatenated** `out_dims + primal_dims`:

* both in range and distinct;
* **split across the out/primal boundary** — a diagonal ties one OUT axis to one
  PRIMAL axis, so out↔out and primal↔primal are meaningless;
* **neither side already spoken for** — a dim with `is_sparse` is paired via
  `other_id`, so its only legal partner is that partner; two dense dims pair
  freely;
* **neither side implicit** (`axis is None`) — no physical axis to diagonalise,
  e.g. after an earlier `Compress` dropped it;
* `factor` from `diag_pair_factor_space` (below).

`Compress(axes, kind)` — `axes` are PHYSICAL positions into `val`, so
`val.ndim - 1` is the highest legal axis. An edge with no materialised `val`
masks to all-`False`: graphax accepts it, but it is a guaranteed no-op and
spending a micro-action slot on nothing is never what the policy wants.

### Factors: `diag_pair_gcd` vs `diag_pair_factor_space`

They gate different fields of the same action.

| helper | question | feeds |
|---|---|---|
| `diag_valid_mask` | which `(i,j)` pairs exist at all | the i/j pointer heads |
| `diag_pair_factor_space` | given that pair, how finely may it be blocked | the prime-exponent factor head |

`diag_pair_factor_space(st, i, j) -> (base, span)`: legal factors are exactly
`base * d` for divisors `d` of `span` **with `d > 1`**.

* free pair → `(1, gcd(N_i, N_j))`
* already-coupled pair → `(meta, gcd // meta)` — a further Diag may only
  **subdivide**, never coarsen.

**Units trap.** `Diag.factor` is the ABSOLUTE resulting meta count (block sizes
come out as `N // factor`); the `d` in `base * d` is the RELATIVE subdivision.
Halving the blocks of a meta-2 pair is `d = 2`, i.e. `factor = 4`, not 2.
Worked example on sizes `(4, 8)`: `factor=2` → 2 blocks of `(2,4)`;
`factor=4` → 4 blocks of `(1,2)`.

`d = 1` is excluded because it is a pure no-op (`apply_diag` returns the very
same object; for a coupled pair `factor == meta` is likewise a defined no-op).
Consequently a pair whose `span == 1` affords **no legal action at all** and is
dropped from `diag_valid_mask` — otherwise the factor head gets an empty
support, and a softmax over all-masked logits is uniform, i.e. confident-looking
garbage.

`factor = -1` and `0` need no handling — `Diag.__post_init__` already rejects
them as the legacy `sparsity_map` sentinels they were.

### Nominal shapes are NOT enough

The per-vertex transform does **not** act on a pre-existing edge. `core.py`
applies it to `edge_outval` — the per-face contraction `J_out @ J_in`, merged
with parallel edges and drained: a join intermediate exactly like the per-face
`lhs`/`rhs`/`res` slots. A mask computed from the vertex's nominal
`(out_shape ++ primal_shape)` is therefore wrong. Measured on nn256: vertex 5 is
`tanh` with nominal `out=(16,63) primal=(16,63)`, but its transform actually
receives `out=(16,10) primal=(63,)` on one face and `out=(16,10) primal=(63,784)`
on another.

`LiveVertexMaskOracle` handles this: it keeps a structural elimination in step
with the episode and, per candidate vertex, replays that vertex on a graph copy
with a recording no-op in the transform slot. The mask is the intersection over
every face (one rule list must fit them all) and over both dispatch modes.

### Per-face slots need a chooser, not a precomputed mask

`lhs`/`rhs`/`res` are join intermediates that do not exist before `eliminate`
runs, so their structure is unknowable in advance. graphax lets a slot hold a
callable; as of core-v2 that callable may return the **micro-action it chose**,
which is then applied and recorded exactly like a literal one. (The older
tensor-returning callable path applied a transform without ever recording it —
invisible to the AOJ transform log.) `masked_micro_chooser(pick)` wraps
`pick(tensor, actions)` where `actions` is enumerated from the tensor itself, so
an illegal action is not unlikely — it is unrepresentable.

## 2. Policy head changes

* The DIAG block-structure pair rules are now **unconditional**;
  `ALPHAGRAD_MICRO_PAIR_MASKS` is gone. It defaulted OFF, which meant the shipped
  default disagreed with the executor: coupled axes were barred from DIAG
  entirely, making re-diagonalisation unreachable even though graphax supports
  it out of the box.
* `_compute_op_legality` is now **derived from** `_compute_axis_masks` instead of
  keeping its own copy of the rules. The copy had gone stale — it still carried
  `& ~in_diag`, so it declared DIAG illegal exactly when only coupled axes
  remained, which would have vetoed the action the axis masks had just enabled.
* Fixed a latent bug independent of any of the above: an `i` whose entire `j`-row
  was masked stayed DIAG-selectable. The j-head then received an all `-1e9`
  logit vector, and softmax of a constant vector is **uniform** — so instead of
  being rejected it produced a confident-looking sample over uniformly illegal
  partners. Reachable whenever exactly one axis is diag-eligible.
* The ray worker now defaults to **palimpsa**, not the quadratic transformer.
  `policy.py` documented palimpsa as the default all along, but the worker
  resolved `ALPHAGRAD_POLICY` with `"transformer"` and there is no CLI flag. At
  these sizes that is not a small difference: full attention materialised
  `f32[16, 16384, 16384]` = 16 GiB and OOM'd a 24 GB card during autotuning.

## 3. Vertex numbering must match the graph that is eliminated

`jacve` and the AOJ splice jit/pjit bodies into the parent jaxpr before
eliminating, which ADDS equations. Numbering vertices from the raw
`jax.make_jaxpr` output addresses a different graph and leaves the spliced-in
vertices un-eliminated. `_traced_inlined()` (in `cpu_approx_worker`, `ppo`, and
`ppo_ray_worker`) routes the trace through `graphax.inline_call_primitives`
first. Measured on a function with one nested jit: 3 eqns → 5, and an order over
the raw 3 reproduces the "left N intermediate vertices" error.

Note this changes the action space — the policy sees the inlined vertex count,
which is the count that can actually be eliminated.

## 4. Environment flags

| flag | default | meaning |
|---|---|---|
| `ALPHAGRAD_MEASURE_VIA_AOJ` | `0` | build the measured executable with `IncrementalJaxpr` instead of `jacve`. **Required for per-face approximation to affect the reward at all** — `jacve` takes per-vertex transforms only. |
| `ALPHAGRAD_POLICY` | `palimpsa` | encoder backbone: `palimpsa` / `palimpsa_bi` / `transformer`. Use plain `palimpsa`. |
| `ALPHAGRAD_LIVE_MASKS` | `1` | live per-vertex mask oracle; `0` restores the old (broken) nominal-shape screen for A/B. |
| `GRAPHAX_PRUNE` | `1` | `0` keeps the full graph so the policy can learn what to drop/approximate, instead of the eliminator silently removing non-argnum inputs and dead intermediates first. |
| `GRAPHAX_ALLOW_PARTIAL_ORDER` | `0` | accept an order that has not eliminated everything. Needed because the env measures partial/initial orders. |
| `GRAPHAX_KEEP_BLOCKDIAG` | `1` | allows re-diagonalising an already-coupled pair (subdivision). |

## 5. Trim

`32,039 → 29,809` lines in `approx/`.

Deleted (zero live references): `append_only_jaxpr.py` (a *second* tokenizer —
not the AOJ builder), `downstream_train.py`, `compare_baselines.py`,
`dtype_scan.py`.

`common/__init__.py` keeps its `_LAZY` / `__getattr__` table — it is
load-bearing, since the Ray driver must stay JAX-free until Ray spawns the GPU
actors or it hogs GPU memory the actors need (`ppo_ray.py` enforces this with
`_assert_jax_free()`). What it gained is a `TYPE_CHECKING` block so static tools
can see through it. That matters: an import-graph sweep previously reported 12
modules as unreachable, of which **four were false positives** created purely by
the indirection — including `common/masks.py` and `common/instrumentation.py`,
whose symbols `ppo.py` imports. Acting on that sweep would have deleted live
code. Post-fix the sweep reports 4, all genuine.

## 6. Known issues

* **`set_approx_active` inconsistency (unfixed).** `vertex_elimination_jaxpr`
  sets `dispatch.set_approx_active(...)` around its elimination;
  `IncrementalJaxpr.eliminate` never does. The flag gates the elemental
  composition layer, so the same order builds structurally different edges on
  the two paths `_callback` runs back to back (nn256 vertex 7: `val=(16,10,63)`
  with it on vs `(10,63)` with it off). The mask intersects over both modes so
  it is safe, but the inconsistency itself is real.
* **`Quant` is not masked** (~0.3% of observed failures). Chaining `Quant` onto a
  `val` already narrowed to float4/float8 raises `TypePromotionError` inside
  `apply_quant`. Needs a third mask channel carrying the live `val.dtype`.
* **Per-vertex DIAG is nearly a dead action.** Measured over 273
  `(prefix, vertex)` pairs on nn256: 1.75 legal COMPRESS axes per vertex but only
  **0.11** legal DIAG `(i,j)` entries. Not a masking artefact — one rule list
  must fit every face, and elementwise Jacobians arrive fully diagonal (span 1).
  DIAG only becomes useful on the per-face path.
* `src/alphagrad/approx/tests/test_env_callback.py::test_env_step_roundtrip_on_helmholtz`
  fails, and failed before all of this work.

## 7. What is built but NOT wired

Worth knowing before planning: several pieces of the intended architecture exist
and are not connected to the live rollout.

| piece | state |
|---|---|
| `eliminate_vertex_per_face` / `eliminate_order_per_face` (`env.py`) | the per-vertex → per-face → `sample_fn` loop. **No callers.** |
| `masked_micro_chooser` | proven end-to-end (17 faces, 35 actions applied+recorded, 0 raises), but `build_face_transforms` still emits literal `Diag`/`Compress` from the policy slots instead of using it. |
| `graphax.IncrementalPathTokenizer` | graphax's **PRIMARY** tokenizer. Zero uses here — alphagrad still calls the legacy `extract_jaxpr(...).tokenized()`, a one-shot whole-jaxpr tokenization. No graphax test covers the incremental one. |
| `incremental_encoder.py` | 2 weeks old, written for the deleted autoscheduler. Its three "test" files contain **zero** test functions between them and one errors at collection — the "PROVEN recurrence" header is not currently backed by anything running. |

So the policy is **not** fed incrementally today, and per-face approximation does
not reach the reward. Both are wiring jobs on existing pieces, not new builds.
