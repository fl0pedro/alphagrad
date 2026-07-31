"""``UnifiedApproxHead`` as a DROP-IN for :class:`MicroActionPolicy`.

Rather than refactoring the action pipeline (env decoder, Trajectory fields and
PPO loss all consume ``MicroAction``), this adapter reproduces
``MicroActionPolicy``'s exact ``sample`` / ``evaluate`` contract and emits the
same ``MicroAction`` sequence. Everything downstream is untouched.

TRANSLATION
-----------
* op codes already coincide: BLOCKDIAG/REDUCE/QUANT/NONE == DIAG/COMPRESS/
  QUANT/END == 0/1/2/3.
* ``i``/``j`` are 1..6 in the head, 0-based axis-token indices in MicroAction.
* FACTOR is not sampled. It is ``g = gcd(N_i, N_j)`` -- the largest legal
  factor, hence the smallest blocks: a square pair collapses to a PURE
  DIAGONAL, a 2 x 8 pair to two 1 x 4 blocks. ``exponents`` is g's full
  factorisation (``tables.max_exps[g]``), so ``prod(primes ** exps) == g`` and
  the emitted action is legal by construction.
* COPRIME PAIRS ARE MASKED. ``pair_ok[a, b] = gcd(N_a, N_b) > 1``; a pair with
  gcd 1 admits only factor 1, a no-op. If NO pair survives, BLOCKDIAG itself is
  masked out of the op categorical rather than emitting a wasted action.
* REDUCE with several axes becomes several COMPRESS sub-steps, one per selected
  axis -- which is what ``max_substeps`` was always for. They are emitted in
  DESCENDING axis order: compressing an axis removes it, so descending order
  leaves every not-yet-applied index still pointing at the axis it named.
* ``compress_kind`` maps the head's 5 reductions onto graphax's 6
  (``median`` is simply never emitted).

WHY ``evaluate`` CAN RECONSTRUCT THE FIELDS
-------------------------------------------
Everything the head scores is recoverable from the stored MicroAction: ``skip``
= "every sub-step is END" (NONE is masked out of the op set, so that mapping is
unambiguous), op/i/j from sub-step 0, the axis set from the COMPRESS sub-steps'
``i`` fields, and kind/dtype from theirs. With the prime gates gone there is no
longer any unrecoverable latent (the old Poisson-binomial trick existed purely
because the individual gate draws were not stored).
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from typing import NamedTuple

from alphagrad.approx.heads import (
    COMPRESS_KINDS, MAX_EXPONENT, MAX_PRIMES, MicroAction, NUM_COMPRESS_KINDS,
    NUM_OPS, NUM_QUANT_DTYPES, OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT,
    QUANT_DTYPES)
from alphagrad.approx.unified_head import (
    MAX_PAIR_IDX, NUM_APPROX_OPS, NUM_REDUCE_AXES, NUM_REDUCE_FNS,
    OP_BLOCKDIAG, REDUCE_FNS, UnifiedApproxHead, O_AXES, O_DTYPE, O_I, O_J,
    O_OP, O_RFN)

_KIND_MAP = jnp.asarray([COMPRESS_KINDS.index(k) for k in REDUCE_FNS],
                        dtype=jnp.int32)
_KIND_INV = jnp.asarray(
    [REDUCE_FNS.index(k) if k in REDUCE_FNS else 0 for k in COMPRESS_KINDS],
    dtype=jnp.int32)


def _dtype_slot(name, fallback):
    for i, d in enumerate(QUANT_DTYPES):
        if str(d) == name or getattr(d, "__name__", "") == name:
            return i
    return fallback


_F32_SLOT = _dtype_slot("float32", 0)
_BF16_SLOT = _dtype_slot("bfloat16", min(1, NUM_QUANT_DTYPES - 1))


class _Fields(NamedTuple):
    """The subset of ApproxAction that ``_emit`` consumes."""
    skip: jax.Array
    op: jax.Array
    i: jax.Array
    j: jax.Array
    axes: jax.Array
    reduce_fn: jax.Array
    dtype_idx: jax.Array


class UnifiedMicroPolicy(eqx.Module):
    """MicroActionPolicy-compatible facade over a single UnifiedApproxHead."""

    head: UnifiedApproxHead
    max_substeps: int = eqx.field(static=True)

    def __init__(self, *, embd_dim, max_substeps, key, **_ignored):
        self.head = UnifiedApproxHead(embd_dim, key=key)
        self.max_substeps = max_substeps

    # ------------------------------------------------------------- gcd tables
    def _pair_sizes(self, features, i_idx, j_idx):
        n = features.size.shape[0]
        gi = jnp.clip(i_idx, 0, n - 1)
        gj = jnp.clip(j_idx, 0, n - 1)
        return (features.size[gi].astype(jnp.int32),
                features.size[gj].astype(jnp.int32))

    def _gcd(self, tables, N_a, N_b):
        S0, S1 = tables.gcd.shape
        return tables.gcd[jnp.clip(N_a, 0, S0 - 1), jnp.clip(N_b, 0, S1 - 1)]

    def _pair_ok(self, features, tables, pair_valid):
        """(6, 6) mask of pairs a BLOCKDIAG can actually act on.

        Legal iff both axes are valid, a != b, and ``gcd(N_a, N_b) > 1``. A
        coprime pair only admits factor 1, i.e. a block-diagonal that leaves
        the tensor exactly as it was.
        """
        n = features.size.shape[0]
        m = min(MAX_PAIR_IDX, n)
        sz = jnp.zeros((MAX_PAIR_IDX,), jnp.int32).at[:m].set(
            features.size[:m].astype(jnp.int32))
        val = jnp.zeros((MAX_PAIR_IDX,)).at[:m].set(features.valid_mask[:m])
        g = self._gcd(tables, sz[:, None], sz[None, :])
        ok = (g > 1).astype(jnp.float32)
        ok = ok * val[:, None] * val[None, :]
        ok = ok * (1.0 - jnp.eye(MAX_PAIR_IDX))
        if pair_valid is not None and jnp.ndim(pair_valid) == 2:
            pv = jnp.zeros((MAX_PAIR_IDX, MAX_PAIR_IDX)).at[:m, :m].set(
                jnp.asarray(pair_valid, jnp.float32)[:m, :m])
            ok = ok * pv
        return ok

    def _masks(self, features, pair_valid, compress_valid, op_legality_override,
               tables):
        n = features.size.shape[0]
        valid = features.valid_mask
        m = min(MAX_PAIR_IDX, n)
        base = jnp.zeros((MAX_PAIR_IDX,)).at[:m].set(valid[:m])
        base = jnp.where(jnp.sum(base) > 0, base, jnp.ones((MAX_PAIR_IDX,)))

        pair_ok = self._pair_ok(features, tables, pair_valid)
        any_pair = jnp.sum(pair_ok) > 0
        # i is only worth choosing if it HAS a non-coprime partner.
        i_mask = jnp.where(any_pair,
                           (jnp.sum(pair_ok, axis=1) > 0).astype(jnp.float32),
                           base)
        j_mask = base

        if compress_valid is not None:
            cv = compress_valid.astype(jnp.float32)
            k = min(NUM_REDUCE_AXES, cv.shape[0])
            axis_mask = jnp.zeros((NUM_REDUCE_AXES,)).at[:k].set(cv[:k])
        else:
            k = min(NUM_REDUCE_AXES, n)
            axis_mask = jnp.zeros((NUM_REDUCE_AXES,)).at[:k].set(valid[:k])

        op_mask = jnp.ones((NUM_APPROX_OPS,))
        if op_legality_override is not None:
            op_mask = op_mask * op_legality_override.astype(jnp.float32)
        # NONE is MASKED OUT: it is redundant with the skip head and produces a
        # byte-identical all-END sequence, so evaluate could not tell the two
        # apart and the round-trip log-prob was off by ~10.9 nats. With NONE
        # unavailable, "every sub-step is END" means EXACTLY skip=True.
        op_mask = op_mask.at[3].set(0.0)
        # No non-coprime pair anywhere -> BLOCKDIAG can only be a no-op.
        op_mask = op_mask.at[OP_BLOCKDIAG].mul(any_pair.astype(jnp.float32))
        # If nothing is legal the categorical still needs a non-empty support
        # (an all -1e9 softmax gives NaN), but the ACTION must be `skip`, not a
        # substituted QUANT -- see module docstring.
        no_op_legal = jnp.sum(op_mask) <= 0
        op_mask = jnp.where(no_op_legal,
                            jnp.zeros((NUM_APPROX_OPS,)).at[OP_QUANT].set(1.0),
                            op_mask)
        return op_mask, i_mask, j_mask, axis_mask, pair_ok, no_op_legal

    def _canonical_axes(self, z, axes, axis_mask):
        """The axis set that will actually be EMITTED, so score() sees it too.

        Two ways the raw draw and the emitted sub-steps could disagree, each of
        which silently breaks the PPO ratio (evaluate reads the axis set back
        off the COMPRESS sub-steps):

        * ZERO axes selected under REDUCE -- no COMPRESS sub-step is emitted at
          all, so the sequence is all-END and evaluate reads it as ``skip``.
          Fall back to the single legal axis with the highest logit.
        * MORE than ``max_substeps`` axes selected -- the tail is dropped on
          emission. Truncate here, keeping the highest axis indices (emission
          order is descending).
        """
        S = self.max_substeps
        legal = axis_mask > 0.5
        best = jnp.argmax(jnp.where(legal, z[O_AXES:O_RFN], -jnp.inf))
        one = jnn.one_hot(best, NUM_REDUCE_AXES) > 0.5
        axes = axes & legal
        axes = jnp.where(jnp.sum(axes) == 0, one, axes)
        rev = axes[::-1]
        return (rev & (jnp.cumsum(rev.astype(jnp.int32)) <= S))[::-1]

    # ------------------------------------------------------------------ emit
    def _emit(self, act, tables, features):
        S = self.max_substeps
        i0 = act.i - 1
        j0 = act.j - 1
        N_i, N_j = self._pair_sizes(features, i0, j0)
        g = self._gcd(tables, N_i, N_j)
        # factor = g exactly: the largest legal factor = the smallest blocks.
        env_exps = tables.max_exps[g].astype(jnp.int32)
        env_primes = tables.primes[g]
        env_factor = jnp.maximum(
            jnp.prod(jnp.where(env_primes > 0, env_primes, 1) ** env_exps)
            .astype(jnp.int32), 1)

        slots = jnp.arange(S)
        sel = act.axes
        # Selected axes first, in DESCENDING axis order (see module docstring).
        ax = jnp.arange(NUM_REDUCE_AXES, dtype=jnp.int32)
        order = jnp.argsort(jnp.where(sel, -(ax + 1), NUM_REDUCE_AXES + ax))
        n_sel = jnp.sum(sel.astype(jnp.int32))
        axis_for_slot = order[jnp.clip(slots, 0, NUM_REDUCE_AXES - 1)]

        is_bd = (act.op == OP_DIAG) & (~act.skip)
        is_rd = (act.op == OP_COMPRESS) & (~act.skip)
        is_qt = (act.op == OP_QUANT) & (~act.skip)
        comp_here = is_rd & (slots < n_sel)
        first = slots == 0

        op_seq = jnp.where(comp_here, OP_COMPRESS,
                           jnp.where(first & is_bd, OP_DIAG,
                                     jnp.where(first & is_qt, OP_QUANT, OP_END)))
        i_seq = jnp.where(comp_here, axis_for_slot,
                          jnp.where(first & is_bd, i0, 0))
        j_seq = jnp.where(first & is_bd, j0, 0)
        exp_seq = jnp.where((first & is_bd)[:, None], env_exps[None, :], 0)
        fac_seq = jnp.where(first & is_bd, env_factor, 0)
        kind_seq = jnp.where(comp_here, _KIND_MAP[act.reduce_fn], 0)
        qdt_seq = jnp.where(first & is_qt,
                            jnp.where(act.dtype_idx > 0, _BF16_SLOT, _F32_SLOT), 0)

        actions = MicroAction(
            op_type=op_seq.astype(jnp.int32),
            i=i_seq.astype(jnp.int32),
            j=j_seq.astype(jnp.int32),
            exponents=exp_seq.astype(jnp.int32),
            factor=fac_seq.astype(jnp.int32),
            compress_kind=kind_seq.astype(jnp.int32),
            quant_dtype=qdt_seq.astype(jnp.int32),
            quant_scale_sign=jnp.ones((S,), jnp.int32),
            quant_scale_frac=jnp.zeros((S,), jnp.float32),
        )
        arity = jnp.where(is_rd, n_sel, jnp.where(is_bd | is_qt, 1, 0))
        return actions, arity.astype(jnp.int32)

    def _dists(self, z, features, actions):
        """Per-sub-step dists as POINT MASSES at the taken action.

        These feed `old_micro_log_prob_for_action`, which REBUILDS the old
        log-prob by summing log(dist[action]) over the sub-step heads. This
        head cannot be expressed that way -- there is no slot for the skip
        Bernoulli, the nine axis gates, the reduce-fn or the dtype bit -- so
        any partial reconstruction disagrees with `score()` and the PPO ratio
        stops being 1 at epoch 0 (measured: median 2.37, max 2.3e23).

        Emitting point masses makes every rebuilt term log(1) = 0, and the
        joint log-prob is carried intact in the `quant_logp` slot instead. The
        ratio is then exactly exp(score_new - score_old).
        """
        S, n = self.max_substeps, features.size.shape[0]
        ops = jnp.clip(actions.op_type, 0, NUM_OPS - 1)
        op_d = jnn.one_hot(ops, NUM_OPS)
        i_d = jnn.one_hot(jnp.clip(actions.i, 0, n - 1), n)
        j_d = jnn.one_hot(jnp.clip(actions.j, 0, n - 1), n)
        exp_d = jnn.one_hot(
            jnp.clip(actions.exponents, 0, MAX_EXPONENT), MAX_EXPONENT + 1)
        kind_d = jnn.one_hot(
            jnp.clip(actions.compress_kind, 0, NUM_COMPRESS_KINDS - 1),
            NUM_COMPRESS_KINDS)
        return op_d, i_d, j_d, exp_d, kind_d

    # ---------------------------------------------------------------- sample
    def sample(self, vertex_context, init_features, tables, key,
               pair_valid=None, compress_valid=None,
               quant_legality_mask=None, op_legality_override=None):
        op_mask, i_mask, j_mask, axis_mask, pair_ok, no_op_legal = self._masks(
            init_features, pair_valid, compress_valid, op_legality_override,
            tables)
        (z, skip, op, i_idx, j_idx, axes, rfn, dt) = self.head.sample_fields(
            vertex_context, key, op_mask=op_mask, i_mask=i_mask,
            j_mask=j_mask, axis_mask=axis_mask, pair_ok=pair_ok)
        # Canonicalise BEFORE scoring: the value scored must be the value
        # emitted, or sample and evaluate score different variables.
        axes = self._canonical_axes(z, axes, axis_mask)
        skip = jnp.logical_or(skip, no_op_legal)
        lp, ent = self.head.score(
            z, skip, op, i_idx, j_idx, axes, rfn, dt,
            op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
            axis_mask=axis_mask, pair_ok=pair_ok)
        act = _Fields(skip=skip, op=op, i=i_idx + 1, j=j_idx + 1, axes=axes,
                      reduce_fn=rfn, dtype_idx=dt.astype(jnp.int32))
        actions, arity = self._emit(act, tables, init_features)
        op_d, i_d, j_d, exp_d, kind_d = self._dists(z, init_features, actions)
        # The joint log-prob rides in the quant_logp slot: the loss's
        # reconstruction adds it verbatim, and the point-mass dists above
        # contribute 0, so old_log_probs == this exact value.
        return (actions, lp, ent, arity,
                op_d, i_d, j_d, exp_d, kind_d, lp)

    # -------------------------------------------------------------- evaluate
    def evaluate(self, vertex_context, init_features, tables, actions,
                 pair_valid=None, compress_valid=None,
                 quant_legality_mask=None, op_legality_override=None):
        op_mask, i_mask, j_mask, axis_mask, pair_ok, _ = self._masks(
            init_features, pair_valid, compress_valid, op_legality_override,
            tables)
        z = self.head.logits(vertex_context)

        ops = actions.op_type
        skip = jnp.all(ops == OP_END)
        head_op = ops[0]
        i_idx = jnp.clip(actions.i[0], 0, MAX_PAIR_IDX - 1)
        j_idx = jnp.clip(actions.j[0], 0, MAX_PAIR_IDX - 1)

        is_c = (ops == OP_COMPRESS)
        axes = jnp.any(
            (jnp.arange(NUM_REDUCE_AXES)[:, None] == actions.i[None, :])
            & is_c[None, :], axis=1)
        rfn = _KIND_INV[jnp.clip(actions.compress_kind[0], 0,
                                 NUM_COMPRESS_KINDS - 1)]
        dt = (actions.quant_dtype[0] == _BF16_SLOT)

        lp, ent = self.head.score(
            z, skip, head_op, i_idx, j_idx, axes, rfn, dt,
            op_mask=op_mask, i_mask=i_mask, j_mask=j_mask,
            axis_mask=axis_mask, pair_ok=pair_ok)
        arity = jnp.sum((ops != OP_END).astype(jnp.int32))
        op_d, i_d, j_d, exp_d, kind_d = self._dists(z, init_features, actions)
        return (lp, ent, arity, op_d, i_d, j_d, exp_d, kind_d, lp)
