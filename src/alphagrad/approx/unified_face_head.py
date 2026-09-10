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
# --- the #73 APPEND (ticket .56). Every field base above keeps its index. ---
# [0:94) unchanged; [94:95) the `choose` bit; [95:126) learned1, the OLD EDGE
# (graphax `res:jr`); [126:157) learned2, the SUMMED EDGE (`res:jres`).
#
# JOIN_SLOTS are slot blocks of the SAME 31-wide shape as the three
# contraction slots, so `slot_base` / `S_OP` / `score`'s per-slot loop cover
# them unchanged. They are APPENDED AFTER the choose bit, which is why
# `slot_base` needs a branch rather than one multiply: `O_SLOT0 + 31*3` is 94,
# the choose bit's own index. The alternative -- the bit last, at 156 -- would
# make `slot_base` a single expression, and it was rejected because the
# offsets above are the ones the review and the goldens are written against.
O_CHOOSE = O_SLOT0 + FACE_SLOTS * SLOT_WIDTH     # 94
JOIN_SLOTS = 2
O_JOIN0 = O_CHOOSE + 1                           # 95
N_HEAD_SLOTS = FACE_SLOTS + JOIN_SLOTS           # 5
HEAD_WIDTH = O_JOIN0 + JOIN_SLOTS * SLOT_WIDTH   # 157

#: ``choose`` encoding. 0 = ``lossy``, 1 = ``lossless``. 0 is the canonical
#: no-op value every forced field is written to, and ``lossy`` is the declared
#: default of ``--approx-add``, so a golden that zeroes this field reproduces
#: the default behaviour rather than a third thing.
JOIN_LOSSY, JOIN_LOSSLESS = 0, 1

# --face-logit-clamp (v62 saturation guard, ppo.py sets this right after
# argparse, BEFORE any jit trace -- UnifiedFaceHead.logits reads it at
# trace time). 0.0 = off, so az_gumbel and every existing test see the
# unclamped head. When C > 0 every head logit is bounded to (-C, C) via
# C*tanh(z/C): near-identity for |z| << C, but raw-parameter drift can no
# longer run the effective logits to +-inf, so the --face-entropy-floor
# hinge always retains a nonzero restoring gradient and recovery from a
# saturated head stays possible (a hard clip would have ZERO gradient
# outside the bound -- exactly where the guard must still pull back).
LOGIT_CLAMP = [0.0]


def set_logit_clamp(c: float) -> None:
    """Set the global face-logit bound; 0 disables. Call before tracing."""
    LOGIT_CLAMP[0] = float(c)


def slot_base(s: int) -> int:
    """Logit offset of slot ``s``'s 31-wide block.

    ``s < FACE_SLOTS`` -> the three contraction slots at 1, 32, 63, unchanged.
    ``s >= FACE_SLOTS`` -> the join slots at 95, 126, i.e. AFTER the choose bit
    at 94. The branch is the price of putting the bit at 94 (see the layout
    comment); it is not an accident.
    """
    if not (0 <= s < N_HEAD_SLOTS):
        raise IndexError(
            f"slot {s} out of range [0, {N_HEAD_SLOTS}); the head has "
            f"{FACE_SLOTS} contraction slots and {JOIN_SLOTS} join slots.")
    if s < FACE_SLOTS:
        return O_SLOT0 + SLOT_WIDTH * s
    return O_JOIN0 + SLOT_WIDTH * (s - FACE_SLOTS)


class FaceFields(NamedTuple):
    """One face's complete decision.

    Slot fields are ``(N_HEAD_SLOTS,)``: the three CONTRACTION slots
    (lhs, rhs, new) followed by the two JOIN slots (learned1 on the old edge,
    learned2 on the summed edge). ``join`` is the per-face ``choose`` bit.

    A field the running ``--approx-add`` value does not use is FORCED to its
    canonical no-op here (``join = JOIN_LOSSY``, a join slot's ``op =
    OP_NONE``) and contributes exactly zero log-prob and zero entropy, the
    same discipline a padding face and a slot behind ``skip == 1`` get. That
    is what keeps ``sample`` and ``score`` scoring the same variable, and so
    the PPO ratio at 1 on epoch 0.
    """
    skip: jax.Array          # () int32
    op: jax.Array            # (S,) int32
    i: jax.Array             # (S,) int32, 0-based 0..5
    j: jax.Array             # (S,) int32, 0-based 0..5
    axis: jax.Array          # (S,) int32, 0..8
    reduce_fn: jax.Array     # (S,) int32
    dtype_idx: jax.Array     # (S,) int32 {0: float32, 1: bfloat16}
    join: jax.Array = None   # () int32, JOIN_LOSSY / JOIN_LOSSLESS

    @property
    def join_bit(self) -> jax.Array:
        """``join``, or the canonical ``JOIN_LOSSY`` when absent.

        ``join`` defaults to ``None`` so every existing construction of
        ``FaceFields`` -- tests, goldens, the AZ tokenizer -- keeps working and
        means what it used to mean: no ``choose`` decision. Reading it through
        here is what stops a ``None`` reaching the scorer.
        """
        return (jnp.asarray(JOIN_LOSSY, jnp.int32) if self.join is None
                else self.join)


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

    def __init__(self, embd_dim: int, *, in_dim: int | None = None,
                 hidden: int | None = None, key):
        """``in_dim`` is the width of the face REPRESENTATION the head reads.

        It is 3*embd_dim under :class:`UnifiedFacePolicy`, whose input is
        ``[ctx_i || ctx_j || face_latent]`` -- the two endpoint vertices'
        contexts and the face's own palimpsa readout. It defaults to
        ``embd_dim`` so the single-vector callers (tests, the blind path)
        keep working unchanged.
        """
        self.proj = eqx.nn.MLP(in_dim or embd_dim, HEAD_WIDTH,
                               hidden or embd_dim, depth=1, key=key)

    def logits(self, ctx):
        z = self.proj(ctx)
        c = LOGIT_CLAMP[0]
        if c > 0.0:
            # tanh, not hard clip: see LOGIT_CLAMP. sample() and score()
            # both come through here, so behaviour and evaluation stay the
            # same parameterization and the PPO ratio is untouched.
            z = c * jnp.tanh(z / c)
        return z

    # ------------------------------------------------------------------ score
    @staticmethod
    def _check_mask_slots(*masks) -> int:
        """How many slots the caller's masks describe: FACE_SLOTS or
        N_HEAD_SLOTS, never anything else.

        A mask row that nobody computed must not be invented here. Padding a
        3-row mask up to 5 with all-ones would CLEAR every action on the two
        join slots -- on tensors the probe never looked at -- which is exactly
        the "mask admits it, hook refuses it" defect finding 72 exists to
        forbid. So a short mask means "the join slots are not in play", a full
        mask means "they are", and anything else raises.
        """
        rows = {int(m.shape[0]) for m in masks if m is not None}
        if rows - {FACE_SLOTS, N_HEAD_SLOTS}:
            raise ValueError(
                f"face masks must have {FACE_SLOTS} or {N_HEAD_SLOTS} slot "
                f"rows, got {sorted(rows)}. A padded row would clear actions "
                f"on a tensor no mask was computed from.")
        if len(rows) > 1:
            raise ValueError(
                f"face masks disagree on their slot count: {sorted(rows)}. "
                f"One decision, one slot count.")
        return rows.pop() if rows else FACE_SLOTS

    @staticmethod
    def _join_gates(join_slot_ok):
        """``(JOIN_SLOTS,)`` float gates; ``None`` means both join slots dead.

        Dead is the right default: every ``--approx-add`` value except the
        learned ones leaves those slots unused, and an unused field must
        contribute exactly zero rather than take gradient from a reward it had
        no part in.
        """
        if join_slot_ok is None:
            return jnp.zeros((JOIN_SLOTS,), jnp.float32)
        g = jnp.asarray(join_slot_ok, jnp.float32).reshape(-1)
        if g.shape[0] != JOIN_SLOTS:
            raise ValueError(
                f"join_slot_ok must have {JOIN_SLOTS} entries (learned1, "
                f"learned2), got {g.shape[0]}.")
        return g

    def score(self, z, fields: FaceFields, *, op_mask, i_mask, j_mask,
              axis_mask, dtype_mask=None, pair_ok=None, face_valid=True,
              approx_ok=True, choose_ok=False, join_slot_ok=None):
        """(log_prob, entropy, arity) of ``fields`` under logits ``z``.

        Masks are ``(S, ...)`` so each slot can carry its own legality; the
        caller passes the oracle's per-face masks. ``S`` may be
        ``FACE_SLOTS`` (the three contraction slots only) or
        ``N_HEAD_SLOTS``; anything else raises, because a silently padded mask
        row would clear actions on a tensor nobody computed a mask from.

        ``face_valid`` / ``approx_ok`` are the gates that force a padding face
        or a disallowed variant to contribute exactly zero. ``choose_ok`` and
        ``join_slot_ok`` are the same discipline for the #73 fields:

        ``choose_ok`` -- the ``choose`` bit is a LIVE decision (true only under
        ``--approx-add choose``). False forces it to the canonical
        ``JOIN_LOSSY`` with zero log-prob and zero entropy.

        ``join_slot_ok`` -- ``(JOIN_SLOTS,)``, per join slot, live only under
        the value that uses it (``learned1`` for slot 3, ``learned2`` for slot
        4). Default: both dead, which is what every value except the learned
        ones means.

        THERE IS NO LEGALITY MASK ON THE ``choose`` BIT, and that is measured,
        not assumed. Both arms are always formable: ``lossless`` is the plain
        sparse add, and ``lossy`` ends in
        ``graphax.sparse.ops.join.unify_containers``, which equalises the two
        addends' containers unconditionally -- 23 of 23 TLM merge faces formed
        ``lossy`` with 0 raises (finding 73). So the bit is free and a mask
        would be a mask of all-ones. If that ever stops being true the bit
        needs one, which is why
        ``tests/approx_add_test.py::test_the_choose_bit_needs_no_legality_mask``
        pins the claim on the engine rather than trusting this comment.

        sample() applies the SAME gates, which is what keeps the ratio at 1
        before any update.
        """
        n_slots = self._check_mask_slots(op_mask, i_mask, j_mask, axis_mask)
        join_slot_ok = self._join_gates(join_slot_ok)
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

        # THE `choose` BIT. Gated by `choose_ok` and by the face gate, and
        # SELECTED not multiplied, for the same reason every other gate here
        # is: a 0.0 multiplier on a log-prob that is legitimately -inf is NaN.
        _cho_on = _on & (jnp.asarray(choose_ok, jnp.float32) > 0.5)
        lp_cho, e_cho = _bern_logp_ent(
            z[O_CHOOSE], fields.join_bit > JOIN_LOSSY)
        logp = logp + jnp.where(_cho_on, lp_cho, _z)
        ent = ent + jnp.where(_cho_on, e_cho, _z)

        active = gate_face * (fields.skip == 0).astype(jnp.float32)
        _act = active > 0.5
        for s in range(n_slots):
            # A JOIN slot is live only under the value that uses it. Dead ->
            # the same zero contribution a slot behind `skip == 1` gets.
            slot_on = (_act if s < FACE_SLOTS
                       else _act & (join_slot_ok[s - FACE_SLOTS] > 0.5))
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
            dm = jnp.ones((2,), jnp.float32) if dtype_mask is None else dtype_mask[s]
            z_dt = jnp.stack([0.0, z[b + S_DTYPE]])
            lp_dt, e_dt = _cat_logp_ent(z_dt, dm, fields.dtype_idx[s])

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
            logp = logp + jnp.where(slot_on, slot_lp, _z)
            ent = ent + jnp.where(slot_on, slot_e, _z)
            arity = arity + (active * slot_on.astype(jnp.float32)
                             * (op != OP_NONE).astype(jnp.float32))
        return logp, ent, arity

    # ----------------------------------------------------------------- sample
    def sample(self, ctx, key, *, op_mask, i_mask, j_mask, axis_mask,
               dtype_mask=None, pair_ok=None, face_valid=True, approx_ok=True,
               choose_ok=False, join_slot_ok=None):
        """Draw one face decision. Returns ``(z, FaceFields, lp, ent, arity)``.

        Every field is drawn from the SINGLE forward pass ``z`` -- nothing is
        conditioned on a previously drawn slot, and nothing is unrolled.

        ``choose_ok`` / ``join_slot_ok`` gate the #73 fields exactly as
        :meth:`score` documents, and are applied HERE as well as there: a field
        that is forced in one and drawn in the other is the ratio bug.
        """
        z = self.logits(ctx)
        n_slots = self._check_mask_slots(op_mask, i_mask, j_mask, axis_mask)
        join_gate = self._join_gates(join_slot_ok)
        # One skip key, five per slot (op, i, j, axis, reduce_fn), then one
        # dtype key per slot. The dtype Bernoulli used to share k[4] with the
        # reduce_fn categorical: under threefry a scalar uniform and the
        # first Gumbel of a categorical read the same counter word of the
        # key, so the pair was coupled (P(bf16 | mean) 0.01-0.04 against
        # 0.6 for every other fn, finding 56 D8) while score() adds
        # lp_fn + lp_dt as independent terms. The dtype keys are APPENDED:
        # split(key, n)[i] does not depend on n under
        # jax_threefry_partitionable, so every other draw is the same as
        # before for the same seed.
        # The key budget is written in N_HEAD_SLOTS, not FACE_SLOTS: the two
        # join slots draw from their own keys. split(key, n)[i] does not depend
        # on n under jax_threefry_partitionable, so slots 0-2 and the skip draw
        # EXACTLY what they drew at 94 logits for the same seed -- the #73
        # append does not move any existing random draw. The choose bit's key
        # is appended last for the same reason.
        keys = jrand.split(key, 2 + N_HEAD_SLOTS * 6)
        dt_keys = keys[1 + N_HEAD_SLOTS * 5:]
        cho_key = keys[1 + N_HEAD_SLOTS * 6]

        p_skip = jnn.sigmoid(z[O_SKIP])
        skip = (jrand.uniform(keys[0]) < p_skip).astype(jnp.int32)
        # A skip IS an approximation (it deletes the contraction), so a
        # variant that forbids approximations must not be able to draw one.
        skip = skip * jnp.asarray(approx_ok, jnp.int32) \
            * jnp.asarray(face_valid, jnp.int32)

        # The `choose` bit. Forced to the canonical JOIN_LOSSY when the running
        # value does not use it, which is also what score() scores then.
        join = (jrand.uniform(cho_key) < jnn.sigmoid(z[O_CHOOSE])
                ).astype(jnp.int32)
        join = join * jnp.asarray(choose_ok, jnp.int32) \
            * jnp.asarray(face_valid, jnp.int32)

        ops, iis, jjs, axs, fns, dts = [], [], [], [], [], []
        for s in range(n_slots):
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
            dm = jnp.ones((2,), jnp.float32) if dtype_mask is None else dtype_mask[s]
            z_dt = jnp.stack([0.0, z[b + S_DTYPE]])
            dt = _sample_cat(z_dt, dm, dt_keys[s])
            if s >= FACE_SLOTS:
                # A dead join slot is forced to OP_NONE with every sub-field
                # at 0 -- the same canonical no-op a slot behind `skip == 1`
                # carries. Forcing the VALUE as well as zeroing the log-prob
                # matters: the wire is built from these fields, so a dead slot
                # must also emit no rule.
                live = join_gate[s - FACE_SLOTS] > 0.5
                op = jnp.where(live, op, jnp.asarray(OP_NONE, jnp.int32))
                i_idx = jnp.where(live, i_idx, 0)
                j_idx = jnp.where(live, j_idx, 0)
                ax = jnp.where(live, ax, 0)
                fn = jnp.where(live, fn, 0)
                dt = jnp.where(live, dt, 0)
            ops.append(op); iis.append(i_idx); jjs.append(j_idx)
            axs.append(ax); fns.append(fn); dts.append(dt)

        fields = FaceFields(
            skip=skip,
            op=jnp.stack(ops), i=jnp.stack(iis), j=jnp.stack(jjs),
            axis=jnp.stack(axs), reduce_fn=jnp.stack(fns),
            dtype_idx=jnp.stack(dts), join=join,
        )
        lp, ent, arity = self.score(
            z, fields, op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
            axis_mask=axis_mask, dtype_mask=dtype_mask, pair_ok=pair_ok,
            face_valid=face_valid, approx_ok=approx_ok,
            choose_ok=choose_ok, join_slot_ok=join_slot_ok)
        return z, fields, lp, ent, arity
