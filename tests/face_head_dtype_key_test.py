"""D8 (ticket dsnn-3qm.19, finding 56): the face head's dtype Bernoulli
draws from its OWN key, not from the reduce_fn categorical's key.

Before the fix ``sample()`` drew ``reduce_fn`` and ``dtype_idx`` from the same
``k[4]``. Under ``jax_threefry_partitionable`` a scalar ``uniform(k)`` and the
first Gumbel of ``categorical(k, .)`` read the SAME counter word of ``k``, so
the pair was deterministically coupled: measured P(bf16 | reduce_fn=mean) was
0.01-0.04 against 0.59-0.67 for every other reduce function, chi-square 19614
/ 19178 / 24280 on df=4 over 10^5 draws (job 63579). ``score()`` sums
``lp_fn + lp_dt`` as independent terms, so the sampler drew from a joint the
scorer does not describe.

What is pinned here:

1. INDEPENDENCE. 10^5 seeded draws with all-ones masks: the (reduce_fn,
   dtype_idx) contingency table of every slot passes a chi-square test at
   p = 0.001 (critical 18.47 on df = 4). Fails on the unfixed head by three
   orders of magnitude.
2. NOTHING ELSE MOVED. skip / op / i / j / axis / reduce_fn are recomputed
   from the documented key layout ``split(key, 1 + 5*S)`` and must equal what
   ``sample()`` returns, seed for seed. The dtype draw must NOT be the old
   ``uniform(k[4])`` draw.
3. score() IS UNCHANGED. It never took a key. Its values for fixed actions
   under a formula-built ``z`` are pinned to the numbers the base commit
   (0de8562, before the fix) produced; sample() and score() still agree.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.nn as jnn                                            # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402

from alphagrad.approx.common.masks import NUM_FACE_QUANT_DTYPES
from alphagrad.approx.unified_face_head import (                # noqa: E402
    CONTRACTION_LAYOUT, FACE_SLOTS, FaceFields, MAX_PAIR_IDX, NUM_APPROX_OPS,
    NUM_REDUCE_AXES, NUM_REDUCE_FNS, O_SKIP, OP_BLOCKDIAG, OP_NONE, OP_QUANT,
    OP_REDUCE, S_AXIS, S_DTYPE, S_I, S_J, S_OP, S_RFN, UnifiedFaceHead,
    _sample_cat, j_mask_given_i, slot_base)

S = FACE_SLOTS
E = 32
N_DRAWS = 100_000
#: chi-square critical values at p = 0.001, by degrees of freedom
#: (NUM_REDUCE_FNS - 1) * (NUM_FACE_QUANT_DTYPES - 1): 4 (two dtypes), 12 (four).
CHI2_CRIT_P001 = {4: 18.47, 12: 32.91}


def _masks():
    return dict(
        op_mask=jnp.ones((S, NUM_APPROX_OPS), jnp.float32),
        i_mask=jnp.ones((S, MAX_PAIR_IDX), jnp.float32),
        j_mask=jnp.ones((S, MAX_PAIR_IDX), jnp.float32),
        axis_mask=jnp.ones((S, NUM_REDUCE_AXES), jnp.float32),
    )


def _head_and_ctx():
    head = UnifiedFaceHead(E, key=jrand.PRNGKey(0))
    ctx = jrand.normal(jrand.PRNGKey(1), (E,))
    return head, ctx


def _chi2_independence(a, b, na, nb):
    table = np.zeros((na, nb), np.float64)
    np.add.at(table, (a, b), 1.0)
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    return float(((table - expected) ** 2 / expected).sum()), table


# --------------------------------------------------------------------------
# 1. independence
# --------------------------------------------------------------------------

def test_dtype_draw_is_independent_of_reduce_fn():
    head, ctx = _head_and_ctx()
    m = _masks()
    sample = jax.jit(jax.vmap(lambda k: head.sample(ctx, k, **m)[1]))
    fields = sample(jrand.split(jrand.PRNGKey(2), N_DRAWS))
    fn = np.asarray(fields.reduce_fn)
    dt = np.asarray(fields.dtype_idx)
    for s in range(S):
        chi2, table = _chi2_independence(fn[:, s], dt[:, s],
                                         NUM_REDUCE_FNS, NUM_FACE_QUANT_DTYPES)
        df = (NUM_REDUCE_FNS - 1) * (NUM_FACE_QUANT_DTYPES - 1)
        p_dtype_given_fn = table / table.sum(1, keepdims=True)
        assert chi2 < CHI2_CRIT_P001[df], (
            f"slot {s}: chi-square {chi2:.1f} on df={df} (critical "
            f"{CHI2_CRIT_P001[df]}); P(dtype | reduce_fn) = "
            f"{np.round(p_dtype_given_fn, 3).tolist()}")


# --------------------------------------------------------------------------
# 2. the other draws keep the documented key layout
# --------------------------------------------------------------------------

def test_other_draws_keep_the_documented_key_layout():
    head, ctx = _head_and_ctx()
    m = _masks()
    z = head.logits(ctx)
    n_dtype_equal_to_old_draw = 0
    for seed in range(64):
        key = jrand.PRNGKey(seed)
        _, f, *_ = head.sample(ctx, key, **m)
        # The layout sample() documents: one skip key, then five per slot.
        keys = jrand.split(key, 1 + 5 * S)
        skip = (jrand.uniform(keys[0]) < jnn.sigmoid(z[O_SKIP])).astype(
            jnp.int32)
        assert int(f.skip) == int(skip), seed
        for s in range(S):
            b = slot_base(s)
            k = keys[1 + 5 * s:1 + 5 * (s + 1)]
            op = _sample_cat(z[b + S_OP:b + S_I], m["op_mask"][s], k[0])
            i_idx = _sample_cat(z[b + S_I:b + S_J], m["i_mask"][s], k[1])
            jm = j_mask_given_i(i_idx, m["j_mask"][s])
            j_idx = _sample_cat(z[b + S_J:b + S_AXIS], jm, k[2])
            ax = _sample_cat(z[b + S_AXIS:b + S_RFN], m["axis_mask"][s],
                             k[3])
            fn = _sample_cat(z[b + S_RFN:b + S_DTYPE],
                             jnp.ones((NUM_REDUCE_FNS,), jnp.float32), k[4])
            got = (int(f.op[s]), int(f.i[s]), int(f.j[s]), int(f.axis[s]),
                   int(f.reduce_fn[s]))
            want = (int(op), int(i_idx), int(j_idx), int(ax), int(fn))
            assert got == want, (seed, s, got, want)
            old_dt = int(jrand.uniform(k[4]) < jnn.sigmoid(z[b + S_DTYPE]))
            n_dtype_equal_to_old_draw += int(int(f.dtype_idx[s]) == old_dt)
    # 64 seeds x 3 slots. The old draw agrees with an independent one half
    # the time; agreeing on ALL of them is the shared key.
    assert n_dtype_equal_to_old_draw < 64 * S, (
        "dtype_idx is still the uniform(k[4]) draw on every seed")


# --------------------------------------------------------------------------
# 3. score() is unchanged
# --------------------------------------------------------------------------

# (op, dtype_idx, reduce_fn) -> (log_prob, entropy, arity) of the fixed
# action below under the formula-built z, as computed on 0de8562 (before the
# key fix). score() takes no key, so these may not move.
_GOLDEN = {
    (OP_BLOCKDIAG, 0, 0): (-18.3910866, 12.9543247, 4.0),
    (OP_BLOCKDIAG, 0, 3): (-18.3910866, 12.9543247, 4.0),
    (OP_BLOCKDIAG, 1, 0): (-18.3910866, 12.9543247, 4.0),
    (OP_BLOCKDIAG, 1, 3): (-18.3910866, 12.9543247, 4.0),
    (OP_REDUCE, 0, 0): (-15.9752541, 14.0296965, 4.0),
    (OP_REDUCE, 0, 3): (-19.7695656, 14.0296965, 4.0),
    (OP_REDUCE, 1, 0): (-15.9752541, 14.0296965, 4.0),
    (OP_REDUCE, 1, 3): (-19.7695656, 14.0296965, 4.0),
    (OP_QUANT, 0, 0): (-5.8218083, 6.0022840, 4.0),
    (OP_QUANT, 0, 3): (-5.8218083, 6.0022840, 4.0),
    (OP_QUANT, 1, 0): (-8.9324436, 6.0022840, 4.0),
    (OP_QUANT, 1, 3): (-8.9324436, 6.0022840, 4.0),
    (OP_NONE, 0, 0): (-3.2549269, 4.4599562, 1.0),
    (OP_NONE, 0, 3): (-3.2549269, 4.4599562, 1.0),
    (OP_NONE, 1, 0): (-3.2549269, 4.4599562, 1.0),
    (OP_NONE, 1, 3): (-3.2549269, 4.4599562, 1.0),
}


def _fixed_fields(op_val, dt, fn):
    return FaceFields(
        skip=jnp.array(0, jnp.int32),
        op=jnp.full((S,), op_val, jnp.int32),
        i=jnp.array([0, 1, 2], jnp.int32), j=jnp.array([1, 2, 3], jnp.int32),
        axis=jnp.array([0, 4, 8], jnp.int32),
        reduce_fn=jnp.full((S,), fn, jnp.int32),
        dtype_idx=jnp.full((S,), dt, jnp.int32))


def test_score_of_a_fixed_action_is_unchanged():
    head, _ = _head_and_ctx()
    m = _masks()
    # The CONTRACTION-ONLY width (94): these goldens were computed on the
    # lossy/lossless head, which is what `UnifiedFaceHead(...)` builds when no
    # --approx-add is named. A wider value is a DIFFERENT head with different
    # parameters, so the goldens cannot be shared across widths.
    z = jnp.asarray(np.sin(np.arange(CONTRACTION_LAYOUT.width,
                                     dtype=np.float32) * 0.37) * 2.0)
    for (op_val, dt, fn), (lp_g, ent_g, ar_g) in _GOLDEN.items():
        lp, ent, ar = head.score(z, _fixed_fields(op_val, dt, fn), **m)
        assert abs(float(lp) - lp_g) < 1e-5, (op_val, dt, fn, float(lp), lp_g)
        assert abs(float(ent) - ent_g) < 1e-5, (op_val, dt, fn, float(ent),
                                                ent_g)
        assert float(ar) == ar_g, (op_val, dt, fn, float(ar), ar_g)


def test_sample_logp_still_equals_score():
    head, ctx = _head_and_ctx()
    m = _masks()
    for seed in range(100):
        z, f, lp, ent, _ = head.sample(ctx, jrand.PRNGKey(seed), **m)
        lp2, ent2, _ = head.score(z, f, **m)
        assert float(lp) == float(lp2), (seed, float(lp), float(lp2))
        assert float(ent) == float(ent2), (seed, float(ent), float(ent2))
