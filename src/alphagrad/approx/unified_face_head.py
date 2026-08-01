"""ONE approximation head per FACE. 94 outputs, one forward pass, one sample.

The per-vertex 32-output head, restructured for the per-path action space the
environment already consumes (``env.FACE_SLOTS = 3``, ``face_rows[v][f][s]``,
``face_skips[v][f]`` -> ``jacve(face_transforms=...)``).

LAYOUT (94 logits)
------------------
    [0:1)   skip        Bernoulli -- ONE for the whole face

    then slot s in (pre=lhs, post=rhs, new=res) at ``1 + SLOT_WIDTH*s``:
      +0 :+4    op          softmax {blockdiag, reduce, quant, none}
      +4 :+10   i           softmax over 1..6
      +10:+16   j           softmax over 1..6
      +16:+25   reduce axis softmax over 9
      +25:+30   reduce fn   softmax {mean, min, max, abs_min, abs_max}
      +30:+31   dtype       Bernoulli {float32, bfloat16}

    31 per slot, 3 slots = 93, plus the one shared skip = 94 = 32*3 - 2.

WHAT CHANGED, AND WHY
---------------------
**The skip is hoisted.** A skip drops the face's contraction outright
(``graphax.SKIP_FACE``), so it is a property of the FACE, not of an operand
slot. Three independent skip gates could never mean anything coherent.

**The reduce axes are a SOFTMAX, not nine gates.** The per-vertex head had nine
independent Bernoullis and emitted one COMPRESS sub-step per axis that fired,
which is why it needed ``max_substeps`` unrolling, a truncation rule, and a
zero-axes fallback (``_canonical_axes``). One softmax draw picks exactly ONE
axis, so a COMPRESS is exactly one rule row. Both failure modes stop existing
rather than being handled, and ``max_substeps`` is 1 by construction.

**One MLP call per face covers all three slots.** ``FacePathPolicy`` ran the
encoder F*(S+1) = 32 times per vertex and the head 24 times, conditioned by a
slot embedding. Here the encoder runs ONCE per vertex and the head once per
face; the slot is a position in the output vector, not a separate query.

BRANCH MASKING
--------------
Only the fields the chosen op CONSUMES contribute to log-prob and entropy:
DIAG reads (i, j), COMPRESS reads (axis, fn), QUANT reads (dtype), END reads
nothing. Unused sub-heads otherwise take gradient from rewards they had no part
in and drift. A padding face, and every slot behind ``skip == 1``, is forced to
a canonical no-op contributing exactly zero -- which is also what keeps
sample() and evaluate() scoring the same variable, and hence the PPO ratio at 1
on epoch 0.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
from typing import NamedTuple

OP_BLOCKDIAG, OP_REDUCE, OP_QUANT, OP_NONE = 0, 1, 2, 3
NUM_APPROX_OPS = 4
MAX_PAIR_IDX = 6
NUM_REDUCE_AXES = 9
REDUCE_FNS = ("mean", "min", "max", "abs_min", "abs_max")
NUM_REDUCE_FNS = len(REDUCE_FNS)
FACE_SLOTS = 3

# per-slot offsets, relative to the slot base
S_OP = 0
S_I = S_OP + NUM_APPROX_OPS                 # 4
S_J = S_I + MAX_PAIR_IDX                    # 10
S_AXIS = S_J + MAX_PAIR_IDX                 # 16
S_RFN = S_AXIS + NUM_REDUCE_AXES            # 25
S_DTYPE = S_RFN + NUM_REDUCE_FNS            # 30
SLOT_WIDTH = S_DTYPE + 1                    # 31

O_SKIP = 0
O_SLOT0 = 1
HEAD_WIDTH = O_SLOT0 + FACE_SLOTS * SLOT_WIDTH   # 94


def slot_base(s: int) -> int:
    return O_SLOT0 + SLOT_WIDTH * s


class FaceFields(NamedTuple):
    """One face's complete decision. Slot fields are (S,)."""
    skip: jax.Array          # () int32
    op: jax.Array            # (S,) int32
    i: jax.Array             # (S,) int32, 0-based 0..5
    j: jax.Array             # (S,) int32, 0-based 0..5
    axis: jax.Array          # (S,) int32, 0..8
    reduce_fn: jax.Array     # (S,) int32
    dtype_idx: jax.Array     # (S,) int32 {0: float32, 1: bfloat16}


def _cat_logp_ent(logits, mask, idx):
    """Masked categorical. -inf in the OUTPUT, never inside the softmax.

    Two constraints that pull against each other.

    (1) The masked-out log-prob must be -inf, NOT -1e9. That sentinel once
    collided with the set pointer's own -1e9 and produced uniform sampling
    over every vertex, which broke the order permutation and zeroed the
    Jacobians for six runs.

    (2) An -inf may never reach `log_softmax`. Its forward value is fine, but
    its DERIVATIVE at an -inf input is 0*inf = NaN, and `jnp.where` does not
    save you: the where's VJP still evaluates the cotangent of the branch it
    discarded, so a NaN there propagates through the zero selector. Measured:
    a finite loss whose gradient was non-finite in 119857 entries across 190
    leaves in the first minibatch, poisoning every parameter, after which
    every reported loss and entropy read nan.

    So the normalisation is done arithmetically against the mask -- the
    exponentials of masked entries are ZEROED rather than driven to zero by an
    -inf logit -- and the -inf appears only in the returned log-prob vector,
    where `jnp.where`'s zero cotangent is the whole story because the false
    branch is a constant. Entropy sums the FINITE shifted logits under the
    same mask, so nothing unsafe is differentiated.

    A fully masked head falls back to uniform; the caller's gate zeroes its
    contribution anyway.
    """
    m = (mask > 0.5)
    m = jnp.where(jnp.any(m), m, jnp.ones_like(m))
    mf = m.astype(logits.dtype)
    # Shift by the max over LIVE entries only (masked entries must not set it).
    zmax = jnp.max(jnp.where(m, logits, -jnp.finfo(logits.dtype).max))
    shifted = logits - jax.lax.stop_gradient(zmax)
    # SANITISE BEFORE THE UNSAFE OP, not after. zmax is the max over LIVE
    # entries, so a masked-out logit can sit far above it and overflow exp to
    # inf; `jnp.where` would hide that forward (it selects the 0) while the
    # VJP still evaluates ct * exp(x) = 0 * inf = NaN on the discarded branch.
    safe = jnp.where(m, shifted, 0.0)
    e = jnp.where(m, jnp.exp(safe), 0.0)
    Z = jnp.sum(e)
    logZ = jnp.log(Z)
    p = e / Z
    lp_live = safe - logZ                         # finite everywhere
    ent = -jnp.sum(jnp.where(m, p * lp_live, 0.0))
    logp_all = jnp.where(m, lp_live, -jnp.inf)
    return logp_all[idx], ent


def _bern_logp_ent(logit, x):
    """Bernoulli log-prob and entropy, finite at a SATURATED logit.

    p*log p is 0*(-inf) = NaN once sigmoid saturates to exactly 0 or 1, which
    a diverging logit reaches in float32 well before it reaches inf. The limit
    is 0 -- select it rather than computing it.
    """
    lp1 = jnn.log_sigmoid(logit)
    lp0 = jnn.log_sigmoid(-logit)
    p = jnn.sigmoid(logit)
    t1 = jnp.where(p > 0, p * lp1, 0.0)
    t0 = jnp.where(p < 1, (1.0 - p) * lp0, 0.0)
    ent = -(t1 + t0)
    return jnp.where(x, lp1, lp0), ent


def _sample_cat(logits, mask, key):
    logits = jnp.where(mask > 0.5, logits, -jnp.inf)
    dead = jnp.sum(mask > 0.5) == 0
    logits = jnp.where(dead, jnp.zeros_like(logits), logits)
    return jrand.categorical(key, logits).astype(jnp.int32)


def j_mask_given_i(i_idx, j_mask, pair_ok=None):
    """Legal ``j`` once ``i`` is fixed: not i itself, not coprime with it.

    gcd(N_i, N_j) == 1 admits only factor 1, a no-op block-diagonal, so those
    pairs would absorb probability mass for an action that cannot change
    anything. Falls back to the unrestricted mask if that leaves nothing.
    """
    m = j_mask * (1.0 - jnn.one_hot(i_idx, MAX_PAIR_IDX))
    if pair_ok is not None:
        m = m * pair_ok[i_idx]
    return jnp.where(jnp.sum(m) > 0, m, j_mask)


class UnifiedFaceHead(eqx.Module):
    """embd_dim -> 94 logits -> one complete per-face approximation decision."""

    proj: eqx.nn.MLP

    def __init__(self, embd_dim: int, *, hidden: int | None = None, key):
        self.proj = eqx.nn.MLP(embd_dim, HEAD_WIDTH, hidden or embd_dim,
                               depth=1, key=key)

    def logits(self, ctx):
        return self.proj(ctx)

    # ------------------------------------------------------------------ score
    def score(self, z, fields: FaceFields, *, op_mask, i_mask, j_mask,
              axis_mask, pair_ok=None, face_valid=True, approx_ok=True):
        """(log_prob, entropy, arity) of ``fields`` under logits ``z``.

        Masks are (S, ...) so each slot can carry its own legality; the caller
        passes the oracle's per-face masks. `face_valid` / `approx_ok` are the
        gates that force a padding face or a disallowed variant to contribute
        exactly zero -- sample() applies the SAME gates, which is what keeps
        the ratio at 1 before any update.
        """
        gate_face = jnp.asarray(face_valid, jnp.float32) * jnp.asarray(
            approx_ok, jnp.float32)

        # GATES SELECT, THEY DO NOT MULTIPLY. A gate of 0.0 times an -inf
        # log-prob is NaN, and -inf is the CORRECT log-prob for a masked
        # index -- so every gated-off face (padding, variant-disallowed) or
        # gated-off slot (skipped face) would poison the batch mean. That is
        # the `ent:nan` this head produced against FacePathPolicy's 2.613 on
        # the same config. Same trap the branch masks below already avoid.
        _on = gate_face > 0.5
        _z = jnp.zeros((), jnp.float32)

        lp_skip, e_skip = _bern_logp_ent(z[O_SKIP], fields.skip > 0)
        logp = jnp.where(_on, lp_skip, _z)
        ent = jnp.where(_on, e_skip, _z)
        arity = gate_face

        active = gate_face * (fields.skip == 0).astype(jnp.float32)
        _act = active > 0.5
        for s in range(FACE_SLOTS):
            b = slot_base(s)
            op = fields.op[s]
            lp_op, e_op = _cat_logp_ent(
                z[b + S_OP:b + S_I], op_mask[s], op)
            # Branch masks: exactly the fields this op consumes.
            is_bd = op == OP_BLOCKDIAG
            is_rd = op == OP_REDUCE
            is_qt = op == OP_QUANT

            lp_i, e_i = _cat_logp_ent(
                z[b + S_I:b + S_J], i_mask[s], fields.i[s])
            jm = j_mask_given_i(fields.i[s], j_mask[s],
                                None if pair_ok is None else pair_ok[s])
            lp_j, e_j = _cat_logp_ent(z[b + S_J:b + S_AXIS], jm, fields.j[s])
            lp_ax, e_ax = _cat_logp_ent(
                z[b + S_AXIS:b + S_RFN], axis_mask[s], fields.axis[s])
            lp_fn, e_fn = _cat_logp_ent(
                z[b + S_RFN:b + S_DTYPE],
                jnp.ones((NUM_REDUCE_FNS,), jnp.float32), fields.reduce_fn[s])
            lp_dt, e_dt = _bern_logp_ent(
                z[b + S_DTYPE], fields.dtype_idx[s] > 0)

            # SELECT, never multiply. A branch mask of 0.0 times a -inf
            # log-prob is NaN, and the unused branches genuinely are -inf:
            # _rows writes a canonical j = 0 for every non-DIAG slot, which
            # j_mask_given_i masks out (it removes i itself), so evaluate
            # scores an illegal index whose logit is -inf. sample() never hit
            # it because it drew j from the masked distribution. This is the
            # forward-value bug AND the classic 0*inf gradient trap.
            _z0 = jnp.zeros_like(lp_op)
            slot_lp = (lp_op
                       + jnp.where(is_bd, lp_i + lp_j, _z0)
                       + jnp.where(is_rd, lp_ax + lp_fn, _z0)
                       + jnp.where(is_qt, lp_dt, _z0))
            slot_e = (e_op
                      + jnp.where(is_bd, e_i + e_j, _z0)
                      + jnp.where(is_rd, e_ax + e_fn, _z0)
                      + jnp.where(is_qt, e_dt, _z0))
            logp = logp + jnp.where(_act, slot_lp, _z)
            ent = ent + jnp.where(_act, slot_e, _z)
            arity = arity + active * (op != OP_NONE).astype(jnp.float32)
        return logp, ent, arity

    # ----------------------------------------------------------------- sample
    def sample(self, ctx, key, *, op_mask, i_mask, j_mask, axis_mask,
               pair_ok=None, face_valid=True, approx_ok=True):
        """Draw one face decision. Returns ``(z, FaceFields, lp, ent, arity)``.

        Every field is drawn from the SINGLE forward pass ``z`` -- nothing is
        conditioned on a previously drawn slot, and nothing is unrolled.
        """
        z = self.logits(ctx)
        keys = jrand.split(key, 1 + FACE_SLOTS * 5)

        p_skip = jnn.sigmoid(z[O_SKIP])
        skip = (jrand.uniform(keys[0]) < p_skip).astype(jnp.int32)
        # A skip IS an approximation (it deletes the contraction), so a
        # variant that forbids approximations must not be able to draw one.
        skip = skip * jnp.asarray(approx_ok, jnp.int32) \
            * jnp.asarray(face_valid, jnp.int32)

        ops, iis, jjs, axs, fns, dts = [], [], [], [], [], []
        for s in range(FACE_SLOTS):
            b = slot_base(s)
            k = keys[1 + 5 * s:1 + 5 * (s + 1)]
            op = _sample_cat(z[b + S_OP:b + S_I], op_mask[s], k[0])
            i_idx = _sample_cat(z[b + S_I:b + S_J], i_mask[s], k[1])
            jm = j_mask_given_i(i_idx, j_mask[s],
                                None if pair_ok is None else pair_ok[s])
            j_idx = _sample_cat(z[b + S_J:b + S_AXIS], jm, k[2])
            ax = _sample_cat(z[b + S_AXIS:b + S_RFN], axis_mask[s], k[3])
            fn = _sample_cat(z[b + S_RFN:b + S_DTYPE],
                             jnp.ones((NUM_REDUCE_FNS,), jnp.float32), k[4])
            dt = (jrand.uniform(k[4]) < jnn.sigmoid(z[b + S_DTYPE])
                  ).astype(jnp.int32)
            ops.append(op); iis.append(i_idx); jjs.append(j_idx)
            axs.append(ax); fns.append(fn); dts.append(dt)

        fields = FaceFields(
            skip=skip,
            op=jnp.stack(ops), i=jnp.stack(iis), j=jnp.stack(jjs),
            axis=jnp.stack(axs), reduce_fn=jnp.stack(fns),
            dtype_idx=jnp.stack(dts),
        )
        lp, ent, arity = self.score(
            z, fields, op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
            axis_mask=axis_mask, pair_ok=pair_ok, face_valid=face_valid,
            approx_ok=approx_ok)
        return z, fields, lp, ent, arity
