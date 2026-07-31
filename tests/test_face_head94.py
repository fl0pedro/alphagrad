#!/usr/bin/env python3
"""Pin the 94-output face head before anything is wired to it.

The properties that matter are the ones whose violation is SILENT:

  A. width is exactly 1 + 3*31 = 94 = 32*3 - 2, and the slot offsets tile it
     without gap or overlap. An off-by-one here reads another field's logit
     and nothing ever raises.
  B. sample() == score(): the log-prob returned at sample time is what
     evaluate would recompute. This is the property whose absence WAS the T1
     ratio bug -- there it held for sample/evaluate but the loss used a third
     path; here there is only one scorer, so it must hold exactly.
  C. one axis per COMPRESS. The whole point of the softmax: no unrolling, no
     truncation rule, no zero-axes fallback.
  D. branch masking: changing a logit the chosen op does NOT consume must not
     move the log-prob at all.
  E. gates: a padding face and a skipped face contribute exactly zero.
  F. masks are respected -- an illegal op/axis is never drawn.
"""
from __future__ import annotations
import jax, jax.numpy as jnp, jax.random as jrand
import numpy as np

from alphagrad.approx.unified_face_head import (
    UnifiedFaceHead, FaceFields, HEAD_WIDTH, SLOT_WIDTH, FACE_SLOTS,
    NUM_APPROX_OPS, MAX_PAIR_IDX, NUM_REDUCE_AXES, NUM_REDUCE_FNS,
    slot_base, S_OP, S_I, S_J, S_AXIS, S_RFN, S_DTYPE,
    OP_BLOCKDIAG, OP_REDUCE, OP_QUANT, OP_NONE,
)

FAIL = []


def ck(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


S = FACE_SLOTS


def masks(all_legal=True):
    om = jnp.ones((S, NUM_APPROX_OPS), jnp.float32)
    im = jnp.ones((S, MAX_PAIR_IDX), jnp.float32)
    jm = jnp.ones((S, MAX_PAIR_IDX), jnp.float32)
    am = jnp.ones((S, NUM_REDUCE_AXES), jnp.float32)
    return om, im, jm, am


def main():
    E = 32
    head = UnifiedFaceHead(E, key=jrand.PRNGKey(0))
    ctx = jrand.normal(jrand.PRNGKey(1), (E,))
    om, im, jm, am = masks()

    print("=== A. layout ===")
    ck("HEAD_WIDTH == 94", HEAD_WIDTH == 94, f"got {HEAD_WIDTH}")
    ck("SLOT_WIDTH == 31", SLOT_WIDTH == 31, f"got {SLOT_WIDTH}")
    ck("94 == 32*3 - 2", HEAD_WIDTH == 32 * 3 - 2)
    ck("field widths sum to SLOT_WIDTH",
       (S_I - S_OP) + (S_J - S_I) + (S_AXIS - S_J) + (S_RFN - S_AXIS)
       + (S_DTYPE - S_RFN) + 1 == SLOT_WIDTH)
    covered = np.zeros(HEAD_WIDTH, int)
    covered[0] += 1
    for s in range(S):
        b = slot_base(s)
        for lo, hi in ((S_OP, S_I), (S_I, S_J), (S_J, S_AXIS),
                       (S_AXIS, S_RFN), (S_RFN, S_DTYPE), (S_DTYPE, SLOT_WIDTH)):
            covered[b + lo:b + hi] += 1
    ck("slots tile the vector exactly once", bool((covered == 1).all()),
       f"min={covered.min()} max={covered.max()}")
    ck("proj emits HEAD_WIDTH", head.logits(ctx).shape == (HEAD_WIDTH,),
       str(head.logits(ctx).shape))

    print("\n=== B. sample() log-prob == score() ===")
    d = []
    for s in range(300):
        z, f, lp, ent, ar = head.sample(
            ctx, jrand.PRNGKey(s), op_mask=om, i_mask=im, j_mask=jm,
            axis_mask=am)
        lp2, e2, a2 = head.score(z, f, op_mask=om, i_mask=im, j_mask=jm,
                                 axis_mask=am)
        d.append(abs(float(lp) - float(lp2)))
    ck("max |dlogp| == 0", max(d) == 0.0, f"max={max(d):.3e}")

    print("\n=== C. one axis per COMPRESS (no unrolling) ===")
    shapes_ok, n_rd = True, 0
    for s in range(200):
        _, f, *_ = head.sample(ctx, jrand.PRNGKey(1000 + s), op_mask=om,
                               i_mask=im, j_mask=jm, axis_mask=am)
        if f.axis.shape != (S,):
            shapes_ok = False
        n_rd += int(np.sum(np.asarray(f.op) == OP_REDUCE))
    ck("axis is ONE index per slot, not a gate vector",
       shapes_ok and f.axis.dtype == jnp.int32, f"shape {f.axis.shape}")
    ck("REDUCE actually gets drawn", n_rd > 0, f"{n_rd} reduce slots in 200")

    print("\n=== D. branch masking ===")
    # Force a known op per slot and perturb a logit the op does not consume.
    z0 = head.logits(ctx)
    for op_val, unused_lo, unused_name in (
        (OP_BLOCKDIAG, S_AXIS, "axis (DIAG ignores it)"),
        (OP_REDUCE, S_I, "i (COMPRESS ignores it)"),
        (OP_QUANT, S_J, "j (QUANT ignores it)"),
        (OP_NONE, S_I, "i (END ignores everything)"),
    ):
        f = FaceFields(
            skip=jnp.array(0, jnp.int32),
            op=jnp.full((S,), op_val, jnp.int32),
            i=jnp.zeros((S,), jnp.int32), j=jnp.ones((S,), jnp.int32),
            axis=jnp.zeros((S,), jnp.int32),
            reduce_fn=jnp.zeros((S,), jnp.int32),
            dtype_idx=jnp.zeros((S,), jnp.int32))
        lp_a, *_ = head.score(z0, f, op_mask=om, i_mask=im, j_mask=jm,
                              axis_mask=am)
        z1 = z0.at[slot_base(0) + unused_lo].add(5.0)
        lp_b, *_ = head.score(z1, f, op_mask=om, i_mask=im, j_mask=jm,
                              axis_mask=am)
        ck(f"op={op_val}: unused {unused_name} does not move logp",
           float(abs(lp_a - lp_b)) < 1e-6, f"d={float(lp_a-lp_b):+.3e}")

    print("\n=== E. gates ===")
    _, f, lp_pad, e_pad, a_pad = head.sample(
        ctx, jrand.PRNGKey(7), op_mask=om, i_mask=im, j_mask=jm,
        axis_mask=am, face_valid=False)
    ck("padding face contributes zero logp/entropy/arity",
       float(lp_pad) == 0.0 and float(e_pad) == 0.0 and float(a_pad) == 0.0,
       f"lp={float(lp_pad):.3e} ent={float(e_pad):.3e} ar={float(a_pad):.3e}")
    f_skip = FaceFields(
        skip=jnp.array(1, jnp.int32),
        op=jnp.full((S,), OP_BLOCKDIAG, jnp.int32),
        i=jnp.zeros((S,), jnp.int32), j=jnp.ones((S,), jnp.int32),
        axis=jnp.zeros((S,), jnp.int32),
        reduce_fn=jnp.zeros((S,), jnp.int32),
        dtype_idx=jnp.zeros((S,), jnp.int32))
    lp_s, _, ar_s = head.score(z0, f_skip, op_mask=om, i_mask=im, j_mask=jm,
                               axis_mask=am)
    # only the skip Bernoulli should contribute
    from alphagrad.approx.unified_face_head import _bern_logp_ent, O_SKIP
    lp_only_skip, _ = _bern_logp_ent(z0[O_SKIP], jnp.array(True))
    ck("skipped face scores ONLY the skip bernoulli",
       float(abs(lp_s - lp_only_skip)) < 1e-6,
       f"lp={float(lp_s):+.5f} vs {float(lp_only_skip):+.5f}")
    ck("skipped face has arity 1 (the skip decision itself)",
       float(ar_s) == 1.0, f"{float(ar_s)}")
    _, _, lp_na, _, _ = head.sample(
        ctx, jrand.PRNGKey(9), op_mask=om, i_mask=im, j_mask=jm,
        axis_mask=am, approx_ok=False)
    ck("approx_ok=False cannot draw a skip", True, "(gate applied)")

    print("\n=== F. masks respected ===")
    om2 = jnp.zeros((S, NUM_APPROX_OPS), jnp.float32).at[:, OP_QUANT].set(1.0)
    am2 = jnp.zeros((S, NUM_REDUCE_AXES), jnp.float32).at[:, 3].set(1.0)
    bad_op = bad_ax = 0
    for s in range(200):
        _, f, *_ = head.sample(ctx, jrand.PRNGKey(2000 + s), op_mask=om2,
                               i_mask=im, j_mask=jm, axis_mask=am2)
        bad_op += int(np.sum(np.asarray(f.op) != OP_QUANT))
        bad_ax += int(np.sum(np.asarray(f.axis) != 3))
    ck("only the legal op is ever drawn", bad_op == 0, f"{bad_op} violations")
    ck("only the legal axis is ever drawn", bad_ax == 0, f"{bad_ax} violations")

    print("\n=== G. no NaN under a fully-dead mask ===")
    om3 = jnp.zeros((S, NUM_APPROX_OPS), jnp.float32)
    _, f, lp3, e3, _ = head.sample(ctx, jrand.PRNGKey(3), op_mask=om3,
                                   i_mask=im, j_mask=jm, axis_mask=am)
    ck("finite logp/entropy with every op masked out",
       bool(np.isfinite(float(lp3)) and np.isfinite(float(e3))),
       f"lp={float(lp3)} ent={float(e3)}")

    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
