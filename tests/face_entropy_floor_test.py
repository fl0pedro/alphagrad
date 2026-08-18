"""--face-entropy-floor + --face-logit-clamp: THE SATURATION GUARD. Pins it.

v61 (job 61610, wandb auw0vzlm) collapsed to 100% SKIP by ~ep90 with
approx-head entropy 8e-5: once deterministic every plan is identical, all
advantages are zero on every channel, PG is dead regardless of reward
shape, and the -entropy_weight*H bonus gradient vanishes under the
saturated softmax. The guard is two-part:

1. HINGE  ``w * relu(floor - H_face)**2`` added to the loss. Below the
   floor its gradient GROWS (2*w*(floor-H)) exactly while the bonus
   gradient vanishes -- that asymmetry is the point.
2. CLAMP  every face logit bounded to (-C, C) via C*tanh(z/C)
   (unified_face_head.LOGIT_CLAMP). Bounded logits keep dH/dparams from
   dying entirely and raw drift from running to +-inf, so the hinge always
   retains a restoring gradient. tanh, not hard clip: a hard clip has zero
   gradient OUTSIDE the bound, stranding a saturated head exactly where
   the guard must pull it back.

The adversarial test below pushes the head toward skip-determinism with a
PG-shaped force (CE toward one action -- the same (1-p)-vanishing gradient
PPO itself produces at saturation) and pins that with the guard the
entropy cannot drop below HALF the floor, while without the guard the same
push drives it far under.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import equinox as eqx                                           # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import optax                                                    # noqa: E402

from alphagrad.approx.unified_face_head import (                # noqa: E402
    FACE_SLOTS,
    LOGIT_CLAMP,
    MAX_PAIR_IDX,
    NUM_APPROX_OPS,
    NUM_REDUCE_AXES,
    FaceFields,
    UnifiedFaceHead,
    set_logit_clamp,
)
from alphagrad.approx.ppo import _face_entropy_floor_penalty    # noqa: E402

FLOOR = 0.3
WEIGHT = 10.0
EMBD = 16


def _mk_head(seed=0):
    return UnifiedFaceHead(EMBD, key=jrand.PRNGKey(seed))


def _masks():
    return dict(
        op_mask=jnp.ones((FACE_SLOTS, NUM_APPROX_OPS), jnp.float32),
        i_mask=jnp.ones((FACE_SLOTS, MAX_PAIR_IDX), jnp.float32),
        j_mask=jnp.ones((FACE_SLOTS, MAX_PAIR_IDX), jnp.float32),
        axis_mask=jnp.ones((FACE_SLOTS, NUM_REDUCE_AXES), jnp.float32),
    )


def _skip_fields():
    z3 = jnp.zeros((FACE_SLOTS,), jnp.int32)
    return FaceFields(skip=jnp.asarray(1, jnp.int32), op=z3, i=z3, j=z3,
                      axis=z3, reduce_fn=z3, dtype_idx=z3)


def _lp_ent(head, ctx):
    """(log_prob, arity-normalised entropy) of the skip=1 decision --
    the same H the loss hinges on (per-face f_ent / max(f_arity, 1))."""
    z = head.logits(ctx)
    lp, ent, arity = head.score(z, _skip_fields(), **_masks())
    return lp, ent / jnp.maximum(arity, 1.0)


# ---------------------------------------------------------------------------
# 1. the hinge itself
# ---------------------------------------------------------------------------

def test_penalty_zero_at_and_above_floor():
    assert float(_face_entropy_floor_penalty(FLOOR, FLOOR, WEIGHT)) == 0.0
    assert float(_face_entropy_floor_penalty(0.5, FLOOR, WEIGHT)) == 0.0
    assert float(_face_entropy_floor_penalty(2.0, FLOOR, WEIGHT)) == 0.0


def test_penalty_positive_and_increasing_below_floor():
    p2 = float(_face_entropy_floor_penalty(0.2, FLOOR, WEIGHT))
    p1 = float(_face_entropy_floor_penalty(0.1, FLOOR, WEIGHT))
    p0 = float(_face_entropy_floor_penalty(0.0, FLOOR, WEIGHT))
    assert 0.0 < p2 < p1 < p0
    np.testing.assert_allclose(p0, WEIGHT * FLOOR ** 2, rtol=1e-6)
    # gradient magnitude GROWS as H falls -- the anti-saturation asymmetry
    g = jax.grad(lambda h: _face_entropy_floor_penalty(h, FLOOR, WEIGHT))
    assert abs(float(g(0.1))) > abs(float(g(0.2))) > 0.0


# ---------------------------------------------------------------------------
# 2. the clamp
# ---------------------------------------------------------------------------

def test_logit_clamp_bounds_and_identity():
    head = _mk_head()
    ctx = 100.0 * jnp.ones((EMBD,))  # drive the MLP hard
    try:
        set_logit_clamp(0.0)
        z_raw = head.logits(ctx)
        set_logit_clamp(15.0)
        z_cl = head.logits(ctx)
        assert float(jnp.max(jnp.abs(z_cl))) <= 15.0
        # near-identity where |z| << C
        set_logit_clamp(0.0)
        z_small = head.logits(0.01 * jnp.ones((EMBD,)))
        set_logit_clamp(15.0)
        z_small_cl = head.logits(0.01 * jnp.ones((EMBD,)))
        np.testing.assert_allclose(np.asarray(z_small_cl),
                                   np.asarray(z_small), rtol=1e-3, atol=1e-4)
        # off = bitwise the raw projection
        set_logit_clamp(0.0)
        np.testing.assert_array_equal(np.asarray(head.logits(ctx)),
                                      np.asarray(z_raw))
    finally:
        set_logit_clamp(0.0)
    assert LOGIT_CLAMP[0] == 0.0


# ---------------------------------------------------------------------------
# 3. THE PINNED BOUND: pushed toward determinism, H stays >= floor/2
# ---------------------------------------------------------------------------

def _push(head, ctx, with_guard, steps=600, lr=3e-2):
    """Adam on CE-toward-skip (PG-shaped, (1-p)-vanishing adversary),
    optionally + the hinge. Returns the final normalised entropy."""
    opt = optax.adam(lr)
    params, static = eqx.partition(head, eqx.is_array)
    opt_state = opt.init(params)

    def loss_fn(p):
        h = eqx.combine(p, static)
        lp, ent = _lp_ent(h, ctx)
        loss = -lp  # push toward deterministic skip=1
        if with_guard:
            loss = loss + _face_entropy_floor_penalty(ent, FLOOR, WEIGHT)
        return loss

    @jax.jit
    def step(p, s):
        g = jax.grad(loss_fn)(p)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s

    for _ in range(steps):
        params, opt_state = step(params, opt_state)
    _, ent = _lp_ent(eqx.combine(params, static), ctx)
    return float(ent)


def test_determinism_push_cannot_break_half_floor_with_guard():
    ctx = jrand.normal(jrand.PRNGKey(7), (EMBD,))
    try:
        set_logit_clamp(15.0)
        h_guarded = _push(_mk_head(1), ctx, with_guard=True)
        h_bare = _push(_mk_head(1), ctx, with_guard=False)
    finally:
        set_logit_clamp(0.0)
    # without the guard the same push saturates the head far below half
    # the floor -- the guard is load-bearing, not decoration.
    assert h_bare < FLOOR / 2, h_bare
    # with hinge + clamp the head cannot be pushed below half the floor.
    assert h_guarded >= FLOOR / 2, h_guarded
