# Curriculum design — multi-trainer reference

This document is the single source of truth for the action-space
curriculum shared across PPO, MuZero, and (eventually) GFN.
Implementation lives in
[`alphagrad/src/alphagrad/approx/variants.py`](variants.py); per-trainer
wiring lives in each driver (`ppo_ray.py`, `mu0_ray.py`, `mu0.py`, `gfn.py`).

If you're adding a new trainer or porting an existing one to the
curriculum, read this once end-to-end and then mirror the PPO wiring
pattern.

---

## Stages

The curriculum has **seven stages**, expanding the policy's action
space monotonically from "nothing" (pure vertex elimination) to "full"
(every operator with every inner-choice palette).

| # | name | what's legal | rationale |
|---|---|---|---|
| 1 | `ve_only` | END (no rule emitted) | Bootstrap value head; the policy only picks vertex order. |
| 2 | `rot1_simple` | one simple op at a time (round-robin) | Introduce each *simple* operator in isolation. |
| 3 | `rot2_simple` | two simple ops at a time (round-robin paired) | Mix two simple operators; policy learns when to swap. |
| 4 | `all_simple` | all 3 simple ops concurrent | Full simple-operator palette. |
| 5 | `rot1_difficult` | one difficult op at a time (round-robin) | Introduce each *difficult* operator in isolation. |
| 6 | `rot2_difficult` | two difficult ops at a time (round-robin paired) | Mix two difficult operators. |
| 7 | `full` | full action footprint | Everything legal. |

### Simple vs difficult operators

The "simple" variant of an operator pins the operator's inner choices
to a single canonical value; the "difficult" variant exposes the full
inner-choice palette to the policy.

| operator | simple variant | difficult variant |
|---|---|---|
| DIAG    | `diag_gcd` (factor = gcd auto) | `diag_factor` (factor ∈ {2,3,4,8,16}) |
| COMPRESS | `compress_scalar` (kind = "mean", reduce to scalar) | `compress` (any of the 6 COMPRESS_KINDS) |
| QUANT   | `quant_smallest_float` (dtype = float4_e2m1fn) | `quantize` (any of the 28 QUANT_DTYPES) |

The "simple" family is `(diag_gcd, compress_scalar, quant_smallest_float)`
exported as `SIMPLE_OPERATORS`. The "difficult" family is
`(diag_factor, compress, quantize)` exported as `DIFFICULT_OPERATORS`.

### Rotation

Rotation stages cycle their constituent variants per-episode:

* `rot1_simple` slots: `[diag_gcd, compress_scalar, quant_smallest_float]`
* `rot2_simple` slots: `[diag_gcd+compress_scalar,
  diag_gcd+quant_smallest_float, compress_scalar+quant_smallest_float]`
* `rot1_difficult` slots: `[diag_factor, compress, quantize]`
* `rot2_difficult` slots: `[diag_factor+compress, diag_factor+quantize,
  compress+quantize]`

The slot for episode `e` (within the stage) is
`slots[e % len(slots)]`. Compound `"A+B"` strings produce the *union*
of A and B's masks (both operators legal).

`rotation_variant_at_episode(stage, ep_within_stage)` returns the
slot's variant name; `compute_union_variant_masks(name, ...)` accepts
both single-variant names (`"diag_gcd"`) and compound strings (`"A+B"`,
`"all_simple"`) and returns the merged mask.

---

## Pacing

Each stage's length is geometric in the base unit `N`:

| stage | multiplier of N |
|---|---|
| ve_only | 1 |
| rot1_simple | 2 |
| rot2_simple | 4 |
| all_simple | 8 |
| rot1_difficult | 16 |
| rot2_difficult | 32 |
| full | **256** (8× the previous step) |

Sum: 319 N episodes. For a 1000-episode budget, N ≈ 3.13; the final
stage (`full`) absorbs the rounding remainder so the totals add up.

### Per-trainer floors

To keep early stages trainable on smaller budgets, each trainer has
a per-stage **floor**:

| trainer | floor | rationale |
|---|---|---|
| ppo | 10 | episodes are cheap (~15s each) |
| mu0 | 20 | episodes are 10× slower; need more stage time to bootstrap |
| gfn | 10 | similar per-ep cost to PPO |

When `episodes / 319 < floor`, the early stages are clamped to the
floor and the final `full` stage absorbs the loss. If the total budget
can't satisfy floors × 7 stages, `compute_seven_stage_curriculum`
raises `ValueError`.

### Concrete numbers

| total ep | trainer | per-stage allocation |
|---|---|---|
| 1000 | ppo | 10 / 10 / 13 / 25 / 50 / 100 / **792** |
| 2000 | ppo | 10 / 13 / 25 / 50 / 100 / 201 / **1601** |
| 5000 | ppo | 16 / 31 / 63 / 125 / 251 / 502 / **4012** |
| 300 | mu0 | 20 / 20 / 20 / 20 / 20 / 30 / **170** |
| 1000 | mu0 | 20 / 20 / 25 / 50 / 100 / 200 / **585** |

---

## Variant masks (PPO contract)

PPO builds the agent with the **full action footprint** at init
(factor table = `(-1, 2, 3, 4, 8, 16)`, op_type ∈ {DIAG, COMPRESS,
QUANT, END}, full quant_dtype set). Per-stage restrictions are applied
by **masking the policy logits** in `_act_step` and `loss_fn`. This
keeps agent weights across stage transitions (no rebuild).

`compute_ppo_variant_masks(variant, factor_table, num_quant_dtypes)`
returns three boolean arrays:

* `op_type_mask: (4,)` — True for allowed op_types.
* `factor_mask: (F,)` — True for allowed factors (indices into
  the factor table).
* `quant_dtype_mask: (NUM_QUANT_DTYPES,)` — True for allowed dtypes.

`compute_union_variant_masks(variant_or_union, ...)` extends this to
accept compound `"A+B"` strings (returns the OR of A's and B's masks)
and the special `"all_simple"` string (OR of all three simple
operators).

The worker re-applies the masks at every act-step and loss-step via:
```python
op_l_curr = jnp.where(op_mask > 0.5, op_logits, -1e9)
f_l_curr  = jnp.where(factor_mask > 0.5, factor_logits, -1e9)
q_l_curr  = jnp.where(quant_mask > 0.5, quant_logits, -1e9)
```
so disallowed actions have ≈0 sampling probability and contribute ≈0
entropy.

When the curriculum advances, the driver calls
`actor.set_variant_masks.remote(variant_name)`, which updates the
worker's `self._current_op_mask_j / _current_factor_mask_j /
_current_quant_mask_j`. Mask values are *traced* JAX inputs to
`act_step` / `update_step`, so changing them does NOT trigger a re-jit.

---

## Driver wiring

### PPO (canonical)

```python
from alphagrad.approx.variants import (
    _parse_curriculum,
    compute_seven_stage_curriculum,
    compute_variant_at_episode,
)

if args.variant == "full_curriculum" and not args.curriculum:
    stages = compute_seven_stage_curriculum(args.episodes, "ppo")
    args.curriculum = ",".join(f"{n}:{k}" for n, k in stages)

curriculum_stages = _parse_curriculum(args.curriculum)
current_variant = None
for ep in range(args.episodes):
    stage, variant, within = compute_variant_at_episode(ep, curriculum_stages)
    if variant != current_variant:
        actor.set_variant_masks.remote(variant)
        current_variant = variant
    # ... run rollout ...
```

### MuZero (legacy 3-stage; rotation pending)

MuZero currently uses `_default_full_curriculum` (3-stage:
`diag_gcd → diag_factor → full`). The 7-stage round-robin requires
per-episode prior gating which MuZero's MCTS prior path doesn't yet
support. Migration is tracked alongside the **MuZero Phase 1+3
refactor** (per-component value head + PopArt + per-component
MinMaxStats):

1. Build MuZero's prior with the full action footprint.
2. Per-episode, gate prior probabilities for disallowed
   op_type / factor / quant choices via the same union-mask helper.
3. Then swap the MuZero curriculum default to
   `compute_seven_stage_curriculum(args.episodes, "mu0")`.

### GFN

GFN consumes variants through the same `_apply_variant_preset` path
MuZero uses. Until GFN's policy network supports per-episode mask
overrides, GFN should pin to a single `--variant` (e.g.
`--variant full`) rather than running the curriculum.

---

## Adding a new variant

1. Add to `VARIANT_PRESETS` with the right `factors`, `max_rules`,
   `pin_rules_to_exact` defaults.
2. Add a case to `compute_ppo_variant_masks` that produces its
   `op_type_mask` / `factor_mask` / `quant_dtype_mask`.
3. Add it to `SIMPLE_OPERATORS` or `DIFFICULT_OPERATORS` if it
   participates in rotation stages.
4. Update the per-stage tables in this doc.

Tests in `alphagrad/tests/test_curriculum_seven_stage.py` cover the
mask + schedule + rotation logic.

---

## Why these specific stages

The progression mirrors how a human would approach the optimization:

1. **`ve_only`** — pick the elimination order. Establishes the
   vertex-selection policy independent of operator choice.
2. **`rot1_simple`** — one operator at a time. The policy learns
   *when* to act (which vertex is amenable to DIAG vs COMPRESS vs
   QUANT) without yet juggling inner choices.
3. **`rot2_simple` + `all_simple`** — combine simple operators.
   The policy learns multi-operator strategies (e.g. DIAG then QUANT).
4. **`rot1_difficult` + `rot2_difficult`** — same progression but
   with the full inner-choice palette per operator. By now the
   vertex-selection policy is solid; the policy refines *how* to act.
5. **`full`** — every degree of freedom. The bulk of training time
   (8× the previous stage) so the policy has room to converge.

The geometric pacing means most of the wall-clock is in stage 7;
stages 1-6 are scaffolding that gives the policy good initial
gradients before the full search space opens up.
