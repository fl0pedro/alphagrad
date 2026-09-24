"""ONE approximation head per FACE, one forward pass, one sample.

The per-vertex 32-output head, restructured for the per-path action space the
environment already consumes (``env.FACE_SLOTS = 3``, ``face_rows[v][f][s]``,
``face_skips[v][f]`` -> ``jacve(face_transforms=...)``).

THE WIDTH IS A FUNCTION OF ``--approx-add``
-------------------------------------------
Owner ruling 2026-09-11 (ticket dsnn-3qm.56). A field the running value does
not use is not gated off -- IT DOES NOT EXIST, so it cannot be indexed, cannot
hold a parameter and cannot take a gradient:

    --approx-add   width              contents           (W = SLOT_WIDTH)
    lossy          W*3 + 2            skip + quant + the three contraction slots
    lossless       W*3 + 2            skip + quant + the three contraction slots
    choose         W*3 + 3            + ONE Bernoulli, lossy vs lossless per face
    learned1       W*4 + 2            + slot 3: the OLD EDGE's own approximation
    learned2       W*5 + 2            + slot 4: the ADD OUTPUT's own approximation

``SLOT_WIDTH = 29`` since the owner ruling of 2026-09-23: the per-slot dtype
field is gone and the op softmax is {blockdiag, reduce, none}; the Quant is
ONE Bernoulli per face beside the skip (widths 89 / 89 / 90 / 118 / 147). It
was 31 with a per-slot two-dtype Bernoulli (94 / 94 / 95 / 125 / 156) and 34
with the four-float set (103 / 103 / 104 / 137 / 171).

READ THE ARITHMETIC: ``learned1`` and ``learned2`` are ``W*N + 2``, NOT
``+3``. THEY HAVE NO CHOOSE BIT. Under those values the model does not pick
lossy-or-lossless; it picks the old edge's (and, under ``learned2``, the sum's)
approximation DIRECTLY, and that pick is what answers the container question.

WHICH CONTAINER THE ADD THEN USES under ``learned1`` / ``learned2``: the UNION,
i.e. ``lossless`` (owner ruling, ``env.resolve_join_mode``). Both addends have
already been shaped by the model's own picks, so compressing further would
silently override a decision the model made, and the union is exact. Under
``learned2`` the learned output approximation then compresses the sum, which is
the model's decision and the right place for the loss.

LAYOUT
------
    [0:1)   skip        Bernoulli -- ONE for the whole face
    [1:2)   quant       Bernoulli -- ONE for the whole face: both contraction
                        operands (lhs AND rhs) in the narrow float

    then slot s at ``2 + SLOT_WIDTH*s`` -- ONE multiply, at EVERY width:
      +0 :+3    op          softmax {blockdiag, reduce, none}
      +3 :+9    i           softmax over 1..6
      +9 :+15   j           softmax over 1..6
      +15:+24   reduce axis softmax over 9
      +24:+29   reduce fn   softmax {mean, min, max, abs_min, abs_max}

    and, under ``choose`` ONLY, the join bit immediately after the last slot
    block, at ``2 + SLOT_WIDTH*n_slots``.

The slot blocks TILE from 2 upwards with no hole, so slot ``s``'s base is
``2 + SLOT_WIDTH*s`` whatever the width is -- slot 3 sits where ``choose``'s
bit sits under ``choose``. The two never coexist: ``choose`` has three slots
and ``learned1`` has no bit. An earlier layout put the bit after slot 2 under
EVERY value and started the join slots one later, which is why ``slot_base``
used to need a branch. That justification is GONE, and so is the branch.

THE QUANT BIT (owner ruling 2026-09-23). A face Quant is a narrow
CONTRACTION: both operands bfloat16, float32 sums, bfloat16 result, and graphax
refuses a Quant on one contraction slot only. So the decision is one bit per
face, and the wire writes it as a QUANT row on lhs AND rhs. A wire row holds
one rule, so under ``quant == 1`` the lhs and rhs structural picks are forced
to ``none`` and contribute exactly zero, the way every slot does behind
``skip == 1``; the ``new`` slot and the learned slots stay free.

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
DIAG reads (i, j), COMPRESS reads (axis, fn), END reads nothing. Unused
sub-heads otherwise take gradient from rewards they had no part
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
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

OP_BLOCKDIAG, OP_REDUCE, OP_NONE = 0, 1, 2
NUM_APPROX_OPS = 3
MAX_PAIR_IDX = 6
NUM_REDUCE_AXES = 9
REDUCE_FNS = ("mean", "min", "max", "abs_min", "abs_max")
NUM_REDUCE_FNS = len(REDUCE_FNS)
FACE_SLOTS = 3
#: The two CONTRACTION OPERAND slots (lhs, rhs). The face's Quant bit lands
#: on exactly these, and on both.
QUANT_SLOTS = (0, 1)

# per-slot offsets, relative to the slot base
S_OP = 0
S_I = S_OP + NUM_APPROX_OPS                 # 3
S_J = S_I + MAX_PAIR_IDX                    # 9
S_AXIS = S_J + MAX_PAIR_IDX                 # 15
S_RFN = S_AXIS + NUM_REDUCE_AXES            # 24
SLOT_WIDTH = S_RFN + NUM_REDUCE_FNS         # 29

O_SKIP = 0
O_QUANT = 1
O_SLOT0 = 2

#: ``choose`` encoding. 0 = ``lossy``, 1 = ``lossless``. 0 is the value a
#: golden that zeroes the field reproduces, and it is the only value the bit
#: can hold under a layout that has no bit -- i.e. nowhere, because such a
#: layout refuses the field entirely.
JOIN_LOSSY, JOIN_LOSSLESS = 0, 1

#: ``--approx-add`` value -> (number of SLOT_WIDTH-wide slot blocks, has a choose bit).
#: THE ONE TABLE the width, the slot count and the bit's presence come from;
#: :func:`head_layout` is the only reader. See the module docstring for the
#: arithmetic and for why the learned values carry no bit.
_LAYOUT_SPEC: dict[str, tuple[int, bool]] = {
    "lossy":    (FACE_SLOTS,     False),
    "lossless": (FACE_SLOTS,     False),
    "choose":   (FACE_SLOTS,     True),
    "learned1": (FACE_SLOTS + 1, False),
    "learned2": (FACE_SLOTS + 2, False),
}


@dataclass(frozen=True)
class FaceHeadLayout:
    """ONE ``--approx-add`` value's head geometry: the SINGLE SOURCE OF TRUTH.

    Everything that constructs the head, sizes a checkpoint or indexes a logit
    asks this object, so none of them can disagree -- and every question it
    cannot answer RAISES instead of returning a number that would silently
    slice into somebody else's field:

    * :meth:`slot_base` raises ``IndexError`` for a slot this width does not
      contain. Not a wrap-around, not a clamp: the block is not there.
    * :attr:`choose_index` raises ``IndexError`` unless the value is
      ``choose``. The learned values have no bit (module docstring), and a
      caller that asks for one has lost track of which value is running.
    """

    mode: str
    n_slots: int
    has_choose: bool

    @property
    def width(self) -> int:
        """The number of logits. ``2 + SLOT_WIDTH*n_slots (+1 under ``choose``)``."""
        return O_SLOT0 + SLOT_WIDTH * self.n_slots + int(self.has_choose)

    @property
    def n_join_slots(self) -> int:
        """Slots beyond the three contraction ones: 0, 1 (learned1) or 2."""
        return self.n_slots - FACE_SLOTS

    def slot_base(self, s: int) -> int:
        """Logit offset of slot ``s``'s ``SLOT_WIDTH``-wide block: ``2 + SLOT_WIDTH*s``.

        ONE MULTIPLY, at every width, because the slot blocks tile upwards from
        2 with no hole -- ``choose``'s bit sits AFTER the last block, not
        between blocks 2 and 3.
        """
        if not (0 <= int(s) < self.n_slots):
            raise IndexError(
                f"slot {s} does not exist in the {self.mode!r} head: it has "
                f"{self.n_slots} slots ({FACE_SLOTS} contraction + "
                f"{self.n_join_slots} join) and {self.width} logits. The "
                f"field is absent at this width, not gated off -- indexing it "
                f"would land in another field.")
        return O_SLOT0 + SLOT_WIDTH * int(s)

    @property
    def choose_index(self) -> int:
        """Logit index of the per-face ``lossy``/``lossless`` Bernoulli.

        ``2 + SLOT_WIDTH*n_slots``, i.e. immediately after the last slot block.
        Raises unless the running value is ``choose``.
        """
        if not self.has_choose:
            raise IndexError(
                f"--approx-add {self.mode!r} has NO choose bit: its head is "
                f"{self.width} logits = {SLOT_WIDTH}*{self.n_slots} + 2. Only 'choose' "
                f"carries one. Under the learned values the model picks the "
                f"old edge's approximation directly and the ADD reconciles "
                f"with the UNION (env.resolve_join_mode), so there is no bit "
                f"to read.")
        return O_SLOT0 + SLOT_WIDTH * self.n_slots


@lru_cache(maxsize=None)
def head_layout(mode: str) -> FaceHeadLayout:
    """The :class:`FaceHeadLayout` of one ``--approx-add`` value.

    ``mode`` is the value itself, PLUMBED in as an architecture parameter (it
    reaches the head through ``UnifiedFacePolicy(approx_add=...)`` from
    ``ppo._build_agent``), never read from the environment here: a width that
    depended on a global would make a checkpoint's shape depend on a variable
    nobody passed to the constructor.
    """
    try:
        n_slots, has_choose = _LAYOUT_SPEC[mode]
    except (KeyError, TypeError):
        raise ValueError(
            f"--approx-add {mode!r} has no head layout. Known values: "
            f"{sorted(_LAYOUT_SPEC)}. A value whose width is unknown must not "
            f"fall back to another value's width -- the checkpoint and every "
            f"logit index would silently describe a different head.") from None
    return FaceHeadLayout(mode=mode, n_slots=n_slots, has_choose=has_choose)


#: The layout of the CONTRACTION-ONLY head -- ``lossy`` / ``lossless``, 89
#: logits, three slots, no bit. It is the default for a caller that names no
#: value, and the bound :func:`slot_base` checks against.
CONTRACTION_LAYOUT = head_layout("lossless")

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


# THE PER-GRAPH FACE-INIT OFFSET (owner ruling 2026-09-22). A run may hold two
# graphs of one target, and the face head is SHARED, so its bias vector can
# carry only ONE (B, Bs). The normalized init is a function of the graph's own
# reference face count F, which differs (measured: 42 and 65), so each graph
# carries a fixed offset that restores its own (B_g, Bs_g) on top of the
# shared pair. The offset is a CONSTANT, never trained, and it is added where
# the init bias sits -- to the projection's output, BEFORE the clamp -- so the
# two are the same parameterization and nothing about a one-graph run moves
# (its offset is None).
#
# IT MUST BE THE SAME NUMBER IN THE SAMPLER AND IN THE REPLAY. That is
# dsnn-dfw.95 exactly: when the two disagreed by a bias, the PPO ratio of
# every plan that approximated went to 1e-4 and those plans left the policy
# gradient. `logits` is the ONE funnel both paths come through, which is why
# the offset is added here and nowhere else.
FACE_LOGIT_OFFSET = [None]


def set_face_logit_offset(vec) -> None:
    """Install the active graph's offset vector, or ``None`` for no offset.

    Call it before the episode's programs are traced. Every program that
    reaches the face head takes the env as an argument, and two graphs are two
    envs with two different treedefs, so the two graphs get two traces and
    each bakes its own offset.
    """
    FACE_LOGIT_OFFSET[0] = vec


def face_logit_offset():
    """The offset currently installed, or ``None``."""
    return FACE_LOGIT_OFFSET[0]


def slot_base(s: int, layout: FaceHeadLayout | None = None) -> int:
    """Logit offset of slot ``s``'s ``SLOT_WIDTH``-wide block: ``2 + SLOT_WIDTH*s``.

    ``layout`` defaults to :data:`CONTRACTION_LAYOUT`, so a bare
    ``slot_base(s)`` answers for the three contraction slots -- 2, 31, 60 --
    and raises ``IndexError`` for anything beyond them. That default is SAFE
    rather than convenient: the three contraction bases are the SAME at every
    width (the blocks tile from 2 with no hole), which is exactly what the
    2026-09-11 layout buys, so a caller looping ``range(FACE_SLOTS)`` is right
    under every value. A caller that wants a JOIN slot must say which layout it
    is indexing, because whether that slot exists at all is a property of the
    running ``--approx-add`` value.
    """
    return (CONTRACTION_LAYOUT if layout is None else layout).slot_base(s)


class FaceFields(NamedTuple):
    """One face's complete decision.

    Slot fields are ``(layout.n_slots,)``: the three CONTRACTION slots (lhs,
    rhs, new), then -- only at a width that HAS them -- learned1 on the old
    edge and learned2 on the summed edge.

    ``quant`` is the per-face Quant bit: 1 puts the narrow float on BOTH
    contraction operands, and forces the lhs and rhs structural picks to
    ``none`` (a wire row holds one rule). Present at every width.

    ``join`` is the per-face ``choose`` bit and is PRESENT IFF the running
    layout has one. ``None`` is not "the default value of the bit", it is "this
    decision has no bit", which is the only honest reading now that a layout
    without a bit has no logit to score it against: :meth:`UnifiedFaceHead.score`
    raises either way round (a bit under a layout that has none, a missing bit
    under ``choose``) rather than substituting a value the head never drew.
    """
    skip: jax.Array          # () int32
    quant: jax.Array         # () int32, 1 = narrow lhs AND rhs
    op: jax.Array            # (S,) int32
    i: jax.Array             # (S,) int32, 0-based 0..5
    j: jax.Array             # (S,) int32, 0-based 0..5
    axis: jax.Array          # (S,) int32, 0..8
    reduce_fn: jax.Array     # (S,) int32
    join: jax.Array = None   # () int32 under `choose`, else None


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
    """``embd_dim`` -> ``layout.width`` logits -> one complete per-face decision.

    The width is the layout's, and the layout is the running ``--approx-add``
    value's (module docstring). ``proj``'s output size, every logit index
    :meth:`score` reads and the number of mask rows it accepts all come from
    that ONE object, so a checkpoint built under one value cannot be read under
    another: the shapes differ and equinox refuses.
    """

    proj: eqx.nn.MLP
    layout: FaceHeadLayout = eqx.field(static=True)

    def __init__(self, embd_dim: int, *, in_dim: int | None = None,
                 hidden: int | None = None, key,
                 approx_add: str = CONTRACTION_LAYOUT.mode):
        """``in_dim`` is the width of the face REPRESENTATION the head reads.

        It is 3*embd_dim under :class:`UnifiedFacePolicy`, whose input is
        ``[ctx_i || ctx_j || face_latent]`` -- the two endpoint vertices'
        contexts and the face's own palimpsa readout. It defaults to
        ``embd_dim`` so the single-vector callers (tests, the blind path)
        keep working unchanged.

        ``approx_add`` is the ARCHITECTURE parameter that sets the OUTPUT
        width, exactly like ``value_dims`` sets a value head's. It is plumbed
        from ``ppo._build_agent``'s ``args.approx_add`` through
        ``UnifiedFacePolicy``; it is NOT read from the environment here,
        because a checkpoint's shape must not depend on a variable nobody
        passed to the constructor. ``ppo._build_agent`` cross-checks the value
        it plumbs against ``env.approx_add()`` so the two cannot drift.

        The default is the CONTRACTION-ONLY layout -- 94 logits, the
        ``lossy`` / ``lossless`` width, which is also ``APPROX_ADD_DEFAULT``
        and the width every caller built before the join slots existed.
        """
        self.layout = head_layout(approx_add)
        self.proj = eqx.nn.MLP(in_dim or embd_dim, self.layout.width,
                               hidden or embd_dim, depth=1, key=key)

    def logits(self, ctx):
        z = self.proj(ctx)
        # THE PER-GRAPH FACE-INIT OFFSET, added where the init bias sits: on
        # the projection's output, BEFORE the clamp. sample() and score()
        # both come through here, so the sampler and the replay read the same
        # number by construction (dsnn-dfw.95).
        _off = FACE_LOGIT_OFFSET[0]
        if _off is not None:
            if _off.shape != (self.layout.width,):
                raise ValueError(
                    f"the face-logit offset has shape {_off.shape} and this "
                    f"head is {self.layout.width} wide. An offset built for "
                    f"another layout would land on the wrong logits.")
            z = z + _off
        c = LOGIT_CLAMP[0]
        if c > 0.0:
            # tanh, not hard clip: see LOGIT_CLAMP. sample() and score()
            # both come through here, so behaviour and evaluation stay the
            # same parameterization and the PPO ratio is untouched.
            # dsnn-dfw.95: THE BARRIER IS LOAD-BEARING. Without it the loss
            # program fused the projection into the bound and scored
            # c*tanh(z) instead of c*tanh(z/c), so the +5.93 OP_NONE bias
            # became the +15 rail and the replay read an approximation slot
            # 9.2 nats below the sampler. The rollout scored the bound as
            # written, so the PPO ratio of every plan that approximated was
            # 1e-4 and those plans left the policy gradient.
            z = c * jnp.tanh(jax.lax.optimization_barrier(z) / c)
        return z

    # ------------------------------------------------------------------ score
    def _check_mask_slots(self, *masks) -> int:
        """Require EXACTLY ``layout.n_slots`` mask rows, and return that.

        A mask row that nobody computed must not be invented here, and a slot
        the width HAS must not be left unscored. Padding a 3-row mask up to 4
        with all-ones would clear every action on the learned slot -- on a
        tensor the probe never looked at -- which is the "mask admits it, hook
        refuses it" defect finding 72 exists to forbid; silently scoring only
        the first three of four slots is the same defect from the other side,
        with slot 3's logits taking no gradient while the engine applies its
        row. So both are refused here.

        THIS IS WHERE THE UNFINISHED TRAINER WIRE SURFACES. Under ``learned1``
        / ``learned2`` the head has 4 / 5 slots while the policy's per-slot
        features and the rollout wire still cover the three contraction slots
        (``face_driver.make_face_slot_callback`` narrows to them deliberately),
        so a trainer run under those values reaches here with 3 rows and RAISES
        -- which is the honest boundary, named in the message, rather than a
        head that trains three of its four slots.
        """
        rows = {int(m.shape[0]) for m in masks if m is not None}
        want = self.layout.n_slots
        if rows - {want}:
            raise ValueError(
                f"the {self.layout.mode!r} head has {want} slots "
                f"({self.layout.width} logits) and needs exactly {want} mask "
                f"rows, got {sorted(rows)}. A padded row would clear actions "
                f"on a tensor no mask was computed from; a short one would "
                f"leave a slot the width HAS unscored while the engine still "
                f"applies its wire row. If this is {want} > {FACE_SLOTS}: the "
                f"policy's per-slot features and the rollout wire still cover "
                f"the {FACE_SLOTS} contraction slots only -- the trainer wire "
                f"for the learned join slots is not built (finding 73 section "
                f"9b, ticket dsnn-3qm.56).")
        return want

    def _join_bit(self, fields: FaceFields):
        """The ``choose`` bit to score, RAISING on either mismatch.

        A bit under a layout that has none cannot be scored -- there is no
        logit for it -- and a missing bit under ``choose`` must not become
        ``JOIN_LOSSY``: the plan would then be measured under a join the policy
        did not pick while the log-prob the trainer stored scored the one it
        did. That is the silent action/reward mismatch of finding 72, and it is
        the same reason ``env.resolve_join_mode`` raises on a missing bit.
        """
        if not self.layout.has_choose:
            if fields.join is not None:
                raise ValueError(
                    f"FaceFields carries a join bit but --approx-add "
                    f"{self.layout.mode!r} has no choose logit to score it "
                    f"against ({self.layout.width} logits = {SLOT_WIDTH}*"
                    f"{self.layout.n_slots} + 2). Under the learned values the "
                    f"model picks the old edge's approximation directly and "
                    f"the ADD reconciles with the UNION; there is no bit.")
            return None
        if fields.join is None:
            raise ValueError(
                "--approx-add 'choose' decides the join PER FACE from the "
                "head's own bit, and this FaceFields has join=None. Defaulting "
                "to JOIN_LOSSY would score a decision the head never drew: "
                "sample() and score() would disagree and the PPO ratio would "
                "not be 1 at epoch 0. The bit rides the wire as face_join[f].")
        return fields.join

    def score(self, z, fields: FaceFields, *, op_mask, i_mask, j_mask,
              axis_mask, quant_mask=None, pair_ok=None, face_valid=True,
              approx_ok=True):
        """(log_prob, entropy, arity) of ``fields`` under logits ``z``.

        Masks are ``(S, ...)`` so each slot can carry its own legality; the
        caller passes the oracle's per-face masks. ``S`` must be exactly
        ``self.layout.n_slots`` -- see :meth:`_check_mask_slots`.
        ``quant_mask`` is the ONE per-face legality of the Quant bit (a
        scalar 0/1: the narrow float is a legal, non-idempotent cast on lhs
        or rhs, and the other operand holds it as an identity cast);
        ``None`` means legal.

        ``face_valid`` / ``approx_ok`` are the gates that force a padding face
        or a disallowed variant to contribute exactly zero. There are no gates
        for the ``choose`` bit or the learned slots any more, and that is the
        2026-09-11 correction rather than an omission: WHETHER THOSE FIELDS
        EXIST IS THE WIDTH'S ANSWER. A field this layout does not contain has
        no logit, no parameter and no index -- ``layout.slot_base`` and
        ``layout.choose_index`` raise ``IndexError`` for it -- so it cannot
        take a gradient from a reward it had no part in, which is what the old
        ``choose_ok`` / ``join_slot_ok`` gates were for.

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
        join_bit = self._join_bit(fields)
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

        # THE `choose` BIT, present iff this width has one. Gated by the face
        # gate and SELECTED not multiplied, for the same reason every other
        # gate here is: a 0.0 multiplier on a log-prob that is legitimately
        # -inf is NaN.
        if join_bit is not None:
            lp_cho, e_cho = _bern_logp_ent(
                z[self.layout.choose_index], join_bit > JOIN_LOSSY)
            logp = logp + jnp.where(_on, lp_cho, _z)
            ent = ent + jnp.where(_on, e_cho, _z)

        active = gate_face * (fields.skip == 0).astype(jnp.float32)
        _act = active > 0.5

        # THE QUANT BIT: one Bernoulli per face, scored behind the skip gate
        # (a skipped face has no contraction to narrow) and behind its own
        # legality. An illegal bit is forced to 0 by sample() and contributes
        # exactly zero here, so the two score the same variable.
        q_ok = (jnp.ones((), jnp.float32) if quant_mask is None
                else jnp.asarray(quant_mask, jnp.float32).reshape(()))
        _qact = _act & (q_ok > 0.5)
        lp_q, e_q = _bern_logp_ent(z[O_QUANT], fields.quant > 0)
        logp = logp + jnp.where(_qact, lp_q, _z)
        ent = ent + jnp.where(_qact, e_q, _z)
        quant_on = _act & (fields.quant > 0)
        arity = arity + jnp.where(quant_on, 1.0, 0.0)

        for s in range(n_slots):
            b = self.layout.slot_base(s)
            op = fields.op[s]
            lp_op, e_op = _cat_logp_ent(
                z[b + S_OP:b + S_I], op_mask[s], op)
            # Branch masks: exactly the fields this op consumes.
            is_bd = op == OP_BLOCKDIAG
            is_rd = op == OP_REDUCE

            lp_i, e_i = _cat_logp_ent(
                z[b + S_I:b + S_J], i_mask[s], fields.i[s])
            jm = j_mask_given_i(fields.i[s], j_mask[s],
                                None if pair_ok is None else pair_ok[s])
            lp_j, e_j = _cat_logp_ent(z[b + S_J:b + S_AXIS], jm, fields.j[s])
            lp_ax, e_ax = _cat_logp_ent(
                z[b + S_AXIS:b + S_RFN], axis_mask[s], fields.axis[s])
            lp_fn, e_fn = _cat_logp_ent(
                z[b + S_RFN:b + SLOT_WIDTH],
                jnp.ones((NUM_REDUCE_FNS,), jnp.float32), fields.reduce_fn[s])

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
                       + jnp.where(is_rd, lp_ax + lp_fn, _z0))
            slot_e = (e_op
                      + jnp.where(is_bd, e_i + e_j, _z0)
                      + jnp.where(is_rd, e_ax + e_fn, _z0))
            # A contraction operand slot behind the quant bit holds the QUANT
            # row, so its structural pick is forced to none and contributes
            # exactly zero, as every slot does behind the skip.
            _slot_act = (_act & ~quant_on) if s in QUANT_SLOTS else _act
            logp = logp + jnp.where(_slot_act, slot_lp, _z)
            ent = ent + jnp.where(_slot_act, slot_e, _z)
            arity = arity + jnp.where(
                _slot_act & (op != OP_NONE), 1.0, 0.0)
        return logp, ent, arity

    # ----------------------------------------------------------------- sample
    def sample(self, ctx, key, *, op_mask, i_mask, j_mask, axis_mask,
               quant_mask=None, pair_ok=None, face_valid=True, approx_ok=True):
        """Draw one face decision. Returns ``(z, FaceFields, lp, ent, arity)``.

        Every field is drawn from the SINGLE forward pass ``z`` -- nothing is
        conditioned on a previously drawn slot, and nothing is unrolled. The
        fields drawn are exactly the ones this width contains, which is what
        keeps :meth:`score` scoring the same variable.
        """
        z = self.logits(ctx)
        n_slots = self._check_mask_slots(op_mask, i_mask, j_mask, axis_mask)
        # One skip key, five per slot (op, i, j, axis, reduce_fn), then the
        # quant bit's key, then -- only under `choose` -- the join bit's.
        # Every draw has its own key: under threefry a scalar uniform and the
        # first Gumbel of a categorical read the same counter word of a shared
        # key, which once coupled the dtype draw to reduce_fn (finding 56 D8).
        #
        # THE BUDGET IS WIDTH-DEPENDENT AND THAT MOVES NO DRAW.
        # `split(key, n)[i]` does not depend on n under
        # jax_threefry_partitionable, so the skip and slots 0-2 draw EXACTLY
        # what they draw at every other width.
        keys = jrand.split(key, 2 + n_slots * 5 + int(self.layout.has_choose))
        q_key = keys[1 + n_slots * 5]

        p_skip = jnn.sigmoid(z[O_SKIP])
        skip = (jrand.uniform(keys[0]) < p_skip).astype(jnp.int32)
        # A skip IS an approximation (it deletes the contraction), so a
        # variant that forbids approximations must not be able to draw one.
        skip = skip * jnp.asarray(approx_ok, jnp.int32) \
            * jnp.asarray(face_valid, jnp.int32)

        # The `choose` bit, drawn iff this width has one. There is nothing to
        # force otherwise: a width without the bit has no logit and
        # `FaceFields.join` stays None, which is what score() then scores.
        join = None
        if self.layout.has_choose:
            join = (jrand.uniform(keys[2 + n_slots * 5])
                    < jnn.sigmoid(z[self.layout.choose_index])
                    ).astype(jnp.int32)
            join = join * jnp.asarray(face_valid, jnp.int32)

        # THE QUANT BIT, forced to 0 wherever score() gates it off: a padding
        # face, a forbidden variant, a skipped face, an illegal cast.
        q_ok = (jnp.ones((), jnp.float32) if quant_mask is None
                else jnp.asarray(quant_mask, jnp.float32).reshape(()))
        quant = (jrand.uniform(q_key) < jnn.sigmoid(z[O_QUANT])
                 ).astype(jnp.int32)
        quant = (quant * jnp.asarray(approx_ok, jnp.int32)
                 * jnp.asarray(face_valid, jnp.int32)
                 * (1 - skip) * (q_ok > 0.5).astype(jnp.int32))

        ops, iis, jjs, axs, fns = [], [], [], [], []
        for s in range(n_slots):
            b = self.layout.slot_base(s)
            k = keys[1 + 5 * s:1 + 5 * (s + 1)]
            op = _sample_cat(z[b + S_OP:b + S_I], op_mask[s], k[0])
            if s in QUANT_SLOTS:
                # The operand slot's row IS the QUANT row when the bit is set:
                # the structural pick is forced to none, and score() gates the
                # slot off, so the two agree on what was drawn.
                op = jnp.where(quant > 0, OP_NONE, op).astype(jnp.int32)
            i_idx = _sample_cat(z[b + S_I:b + S_J], i_mask[s], k[1])
            jm = j_mask_given_i(i_idx, j_mask[s],
                                None if pair_ok is None else pair_ok[s])
            j_idx = _sample_cat(z[b + S_J:b + S_AXIS], jm, k[2])
            ax = _sample_cat(z[b + S_AXIS:b + S_RFN], axis_mask[s], k[3])
            fn = _sample_cat(z[b + S_RFN:b + SLOT_WIDTH],
                             jnp.ones((NUM_REDUCE_FNS,), jnp.float32), k[4])
            ops.append(op); iis.append(i_idx); jjs.append(j_idx)
            axs.append(ax); fns.append(fn)

        fields = FaceFields(
            skip=skip, quant=quant,
            op=jnp.stack(ops), i=jnp.stack(iis), j=jnp.stack(jjs),
            axis=jnp.stack(axs), reduce_fn=jnp.stack(fns),
            join=join,
        )
        lp, ent, arity = self.score(
            z, fields, op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
            axis_mask=axis_mask, quant_mask=quant_mask, pair_ok=pair_ok,
            face_valid=face_valid, approx_ok=approx_ok)
        return z, fields, lp, ent, arity
