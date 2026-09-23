"""D8 (ticket dsnn-3qm.19, finding 56), carried to the per-face quant bit
(owner ruling 2026-09-23): every draw of the face head reads its OWN key.

The per-slot dtype Bernoulli once shared ``k[4]`` with the reduce_fn
categorical. Under ``jax_threefry_partitionable`` a scalar ``uniform(k)`` and
the first Gumbel of ``categorical(k, .)`` read the SAME counter word of ``k``,
so the pair was deterministically coupled: measured P(bf16 | reduce_fn=mean)
was 0.01-0.04 against 0.59-0.67 for every other reduce function, chi-square
19614 / 19178 / 24280 on df=4 over 10^5 draws (job 63579), while ``score()``
summed the two as independent terms. The dtype field is gone; the quant bit
that replaced it is one Bernoulli per face and must not inherit the fault.

What is pinned here:

1. INDEPENDENCE. 10^5 seeded draws with all-ones masks: the (reduce_fn,
   quant) contingency table of every slot passes a chi-square test at
   p = 0.001 (critical 18.47 on df = 4), and so does (op, quant) on the
   ``new`` slot, the one operand the bit does not force.
2. NOTHING ELSE MOVED. skip / op / i / j / axis / reduce_fn are recomputed
   from the documented key layout ``split(key, 2 + 5*S)`` and must equal what
   ``sample()`` returns, seed for seed, on the slots the bit leaves free; the
   bit's draw must NOT be any slot's ``uniform(k[4])`` draw.
3. sample() and score() agree, seed for seed, bit set or clear.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.nn as jnn                                            # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402

from alphagrad.approx.unified_face_head import (                # noqa: E402
    FACE_SLOTS, MAX_PAIR_IDX, NUM_APPROX_OPS, NUM_REDUCE_AXES,
    NUM_REDUCE_FNS, O_QUANT, O_SKIP, OP_NONE, QUANT_SLOTS, S_AXIS, S_I, S_J,
    SLOT_WIDTH, S_OP, S_RFN, UnifiedFaceHead, _sample_cat, j_mask_given_i,
    slot_base)

S = FACE_SLOTS
E = 32
N_DRAWS = 100_000
#: chi-square critical values at p = 0.001, by degrees of freedom.
CHI2_CRIT_P001 = {2: 13.82, 4: 18.47}


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

def test_the_quant_bit_is_independent_of_every_slot_draw():
    head, ctx = _head_and_ctx()
    m = _masks()
    sample = jax.jit(jax.vmap(lambda k: head.sample(ctx, k, **m)[1]))
    fields = sample(jrand.split(jrand.PRNGKey(2), N_DRAWS))
    fn = np.asarray(fields.reduce_fn)
    q = np.asarray(fields.quant)
    assert 0.2 < q.mean() < 0.8, q.mean()
    for s in range(S):
        chi2, table = _chi2_independence(fn[:, s], q, NUM_REDUCE_FNS, 2)
        df = (NUM_REDUCE_FNS - 1) * 1
        p_q_given_fn = table / table.sum(1, keepdims=True)
        assert chi2 < CHI2_CRIT_P001[df], (
            f"slot {s}: chi-square {chi2:.1f} on df={df} (critical "
            f"{CHI2_CRIT_P001[df]}); P(quant | reduce_fn) = "
            f"{np.round(p_q_given_fn, 3).tolist()}")
    # the new slot's op is free under the bit and must not lean on it
    new = S - 1
    assert new not in QUANT_SLOTS
    op = np.asarray(fields.op)[:, new]
    chi2, _ = _chi2_independence(op, q, NUM_APPROX_OPS, 2)
    assert chi2 < CHI2_CRIT_P001[NUM_APPROX_OPS - 1], chi2
    # and the operand slots read none exactly when the bit is set
    for s in QUANT_SLOTS:
        assert bool(np.all(np.asarray(fields.op)[q > 0, s] == OP_NONE))


# --------------------------------------------------------------------------
# 2. the other draws keep the documented key layout
# --------------------------------------------------------------------------

def test_other_draws_keep_the_documented_key_layout():
    head, ctx = _head_and_ctx()
    m = _masks()
    z = head.logits(ctx)
    n_bit_equal_to_a_slot_draw = 0
    for seed in range(64):
        key = jrand.PRNGKey(seed)
        _, f, *_ = head.sample(ctx, key, **m)
        # The layout sample() documents: one skip key, five per slot, then
        # the bit's own key.
        keys = jrand.split(key, 2 + 5 * S)
        skip = (jrand.uniform(keys[0]) < jnn.sigmoid(z[O_SKIP])).astype(
            jnp.int32)
        assert int(f.skip) == int(skip), seed
        quant = int(jrand.uniform(keys[1 + 5 * S]) < jnn.sigmoid(z[O_QUANT]))
        quant = quant * (1 - int(skip))
        assert int(f.quant) == quant, seed
        for s in range(S):
            b = slot_base(s)
            k = keys[1 + 5 * s:1 + 5 * (s + 1)]
            op = _sample_cat(z[b + S_OP:b + S_I], m["op_mask"][s], k[0])
            if s in QUANT_SLOTS and quant:
                op = OP_NONE
            i_idx = _sample_cat(z[b + S_I:b + S_J], m["i_mask"][s], k[1])
            jm = j_mask_given_i(i_idx, m["j_mask"][s])
            j_idx = _sample_cat(z[b + S_J:b + S_AXIS], jm, k[2])
            ax = _sample_cat(z[b + S_AXIS:b + S_RFN], m["axis_mask"][s],
                             k[3])
            fn = _sample_cat(z[b + S_RFN:b + SLOT_WIDTH],
                             jnp.ones((NUM_REDUCE_FNS,), jnp.float32), k[4])
            got = (int(f.op[s]), int(f.i[s]), int(f.j[s]), int(f.axis[s]),
                   int(f.reduce_fn[s]))
            want = (int(op), int(i_idx), int(j_idx), int(ax), int(fn))
            assert got == want, (seed, s, got, want)
            old_dt = int(jrand.uniform(k[4]) < jnn.sigmoid(z[O_QUANT]))
            n_bit_equal_to_a_slot_draw += int(int(f.quant) == old_dt)
    # 64 seeds x 3 slots. A shared key would agree on ALL of them.
    assert n_bit_equal_to_a_slot_draw < 64 * S, (
        "the quant bit is still a slot's uniform(k[4]) draw on every seed")


# --------------------------------------------------------------------------
# 3. sample() and score() agree
# --------------------------------------------------------------------------

def test_sample_logp_still_equals_score():
    head, ctx = _head_and_ctx()
    m = _masks()
    seen = {0: 0, 1: 0}
    for seed in range(100):
        z, f, lp, ent, _ = head.sample(ctx, jrand.PRNGKey(seed), **m)
        lp2, ent2, _ = head.score(z, f, **m)
        assert float(lp) == float(lp2), (seed, float(lp), float(lp2))
        assert float(ent) == float(ent2), (seed, float(ent), float(ent2))
        seen[int(f.quant)] += 1
    assert seen[0] > 0 and seen[1] > 0, seen
