"""Autoregressive sub-episode head for the post-pointer policy.

After the base pointer net picks a vertex ``v``, this module opens a
sub-episode at ``v`` that emits a variable-length sequence of typed
micro-actions until END:

    a_t ∈ { Diag(i, j, factor),  Compress(physical_axes),  End }

Three concrete heads compose:

* :class:`OpTypeHead` — categorical over ``{DIAG, COMPRESS, END}``.
* :class:`AxisPointerHead` — attention pointer over the current axis
  tokens. Used twice for DIAG (``i`` then ``j``); once for COMPRESS.
* :class:`PrimeExponentHead` — autoregressive sequence of small
  categoricals, one per prime factor of ``gcd(N_i, N_j)``. Only fired
  for DIAG. Prime list is dynamic per sub-step but padded to
  :data:`MAX_PRIMES` for JAX scan compatibility.

:class:`AxisSetEncoder` is a small transformer over the axis token set
that re-encodes after every micro-action (axis tags change as DIAGs pair
axes and COMPRESS marks axes as mean-compressed). The vertex context
from the base pointer net is concatenated in at every sub-step.

:class:`MicroActionHead` ties everything together for a single
micro-step: pool axis set → emit op_type → emit axis indices → emit
prime exponents → return joint log-prob + entropy + sampled action.

Integration
-----------
The expected ``Agent`` integration mirrors the existing
``AutoregRulePolicy.sample`` / ``.evaluate`` pattern but with typed
outputs:

  * ``sample(vertex_context, axis_tokens, axis_mask, key) -> (
        op_type_seq, i_seq, j_seq, prime_exponents_seq,
        op_dist_seq, i_dist_seq, j_dist_seq, prime_dist_seq, joint_logp)``
  * ``evaluate(vertex_context, axis_tokens, axis_mask, op_type_seq,
        i_seq, j_seq, prime_exponents_seq) -> (logp, entropy, ...)``

Each ``*_seq`` has a leading ``max_substeps`` dimension; entries past
the emitted END are masked out of the joint log-prob and entropy.

The micro-action sequence is mapped to graphax's
:class:`graphax.sparse.micro_actions.Diag` / ``Compress`` via
:func:`micro_actions_from_policy_emissions` and applied with
``apply_micro_actions`` inside the env callback.

This module is intentionally standalone: it does not import from
``ppo.py`` so it can be tested in isolation. The Agent class will need
to be extended (separate change) to construct a :class:`MicroActionHead`
and swap it in for ``rule_policy`` when ``--dynamic-substeps`` is set.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import distrax
import equinox as eqx
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand


# ---------------------------------------------------------------------------
# Constants and shape primitives
# ---------------------------------------------------------------------------

OP_DIAG = 0
OP_COMPRESS = 1
OP_END = 2
NUM_OPS = 3

# Any integer < 1e9 has ≤ 9 distinct prime factors. The prime-exponent head
# pads up to this for JAX scan compatibility; primes past the real count for
# a given gcd are masked out so they contribute 0 to log-prob and entropy.
MAX_PRIMES = 9

# Maximum exponent any single prime can have in a value we care about. log2 of
# 2^30 ≈ 1e9 bounds the 2-exponent; smaller primes contribute fewer. The head
# emits a categorical over [0, MAX_EXPONENT], masked to [0, e_k] per prime.
MAX_EXPONENT = 30

# Number of structural-tag bits per axis token in :class:`AxisTokenFeatures`.
# Layout: (is_logical, is_compressed, in_diag_group). group_id is encoded as
# a small embedding via :class:`AxisSetEncoder.group_embedding`.
AXIS_TAG_BITS = 3


class AxisTokenFeatures(NamedTuple):
    """Per-axis feature bundle consumed by :class:`AxisSetEncoder`.

    ``size`` carries the axis's logical size; ``log_size`` is the same
    quantity in log-space so the encoder can compare orders of magnitude
    without exploding scale. ``tag_bits`` is the structural one-hot from
    :data:`AXIS_TAG_BITS` (is_logical / is_compressed / in_diag_group),
    and ``group_id`` indexes into the encoder's group embedding table so
    axes that belong to the same diagonal block share representation.

    All fields have a leading ``num_axes`` dim; padding rows are flagged
    by ``valid_mask`` (used by every downstream attention layer).
    """

    size: jax.Array          # (num_axes,) int32 — logical size
    log_size: jax.Array      # (num_axes,) float32 — log(max(size, 1))
    tag_bits: jax.Array      # (num_axes, AXIS_TAG_BITS) float32
    group_id: jax.Array      # (num_axes,) int32 — -1 for ungrouped
    valid_mask: jax.Array    # (num_axes,) float32 — 1 for real axes


# ---------------------------------------------------------------------------
# Axis-set encoder
# ---------------------------------------------------------------------------


class AxisSetEncoder(eqx.Module):
    """Small transformer over a set of axis tokens.

    Re-encodes after every micro-action: the policy emits a Diag pairing
    two axes (updates their tags + group_id), or a Compress marking
    axes as compressed (sets ``is_compressed`` and zeros their ``size``
    pointer to ``val``). The encoder takes the updated
    :class:`AxisTokenFeatures` plus the per-episode vertex context and
    returns per-axis embeddings + a pooled summary.

    The transformer is 1–2 layers (small — axis counts are typically
    < 32 per vertex) and uses standard multi-head self-attention.
    """

    proj_in: eqx.nn.Linear
    group_embedding: eqx.nn.Embedding
    blocks: tuple
    pool_query: jax.Array
    output_proj: eqx.nn.Linear

    embd_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    max_groups: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, num_heads: int, num_layers: int = 1,
                 max_groups: int = 16, *, key):
        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.max_groups = max_groups

        keys = jrand.split(key, num_layers + 4)
        # In features: size, log_size, AXIS_TAG_BITS tag bits, group_emb.
        # We embed group_id separately and concatenate.
        feat_in = 1 + 1 + AXIS_TAG_BITS + embd_dim
        self.proj_in = eqx.nn.Linear(feat_in, embd_dim, key=keys[0])
        # `+1` slot: index 0 reserved for "no group" (group_id == -1 gets
        # remapped to 0 before the embedding lookup).
        self.group_embedding = eqx.nn.Embedding(
            max_groups + 1, embd_dim, key=keys[1],
        )

        self.blocks = tuple(
            _SelfAttentionBlock(embd_dim, num_heads, key=keys[2 + i])
            for i in range(num_layers)
        )

        self.pool_query = jrand.normal(keys[-2], (embd_dim,)) * 0.02
        self.output_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[-1])

    def __call__(self, features: AxisTokenFeatures, vertex_context: jax.Array):
        """Return ``(per_axis_tokens, pooled_summary)``.

        ``per_axis_tokens`` has shape ``(num_axes, embd_dim)``; pad rows
        carry zero output (the value will be masked out at every
        downstream attention layer anyway). ``pooled_summary`` is the
        attention-weighted sum used by :class:`OpTypeHead` and (for
        DIAG) :class:`PrimeExponentHead`.
        """
        # Build per-axis input feature vector.
        size_f = features.size.astype(jnp.float32)[..., None]              # (N, 1)
        log_size_f = features.log_size[..., None]                          # (N, 1)
        tag_f = features.tag_bits.astype(jnp.float32)                      # (N, T)
        group_slot = jnp.where(
            features.group_id >= 0, features.group_id + 1, 0,
        ).astype(jnp.int32)
        group_slot = jnp.clip(group_slot, 0, self.max_groups)
        group_emb = jax.vmap(self.group_embedding)(group_slot)             # (N, E)

        feats = jnp.concatenate([size_f, log_size_f, tag_f, group_emb], axis=-1)
        x = jax.vmap(self.proj_in)(feats)                                  # (N, E)
        # Add the vertex context at every axis token so the attention has
        # both axis-local and vertex-global signal at every layer.
        x = x + vertex_context[None, :]

        valid = features.valid_mask                                         # (N,)
        for block in self.blocks:
            x = block(x, valid)

        # Pool: learned query attends over per-axis tokens, masked by valid.
        q = self.pool_query
        scores = (x @ q) / jnp.sqrt(self.embd_dim)
        scores = jnp.where(valid > 0.5, scores, -1e9)
        attn = jnn.softmax(scores, axis=-1)
        pooled = jnp.sum(attn[:, None] * x, axis=0)

        return self.output_proj(x), self.output_proj(pooled)


class _SelfAttentionBlock(eqx.Module):
    """One multi-head self-attention + MLP block with mask support."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    mlp1: eqx.nn.Linear
    mlp2: eqx.nn.Linear
    embd_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, num_heads: int, *, key):
        if embd_dim % num_heads != 0:
            raise ValueError(
                f"embd_dim ({embd_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads

        keys = jrand.split(key, 6)
        self.q_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[0])
        self.k_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[1])
        self.v_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[2])
        self.out_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[3])
        self.mlp1 = eqx.nn.Linear(embd_dim, embd_dim * 4, key=keys[4])
        self.mlp2 = eqx.nn.Linear(embd_dim * 4, embd_dim, key=keys[5])

    def __call__(self, x: jax.Array, valid: jax.Array) -> jax.Array:
        """``x``: ``(N, E)`` per-axis tokens. ``valid``: ``(N,)`` mask."""
        N = x.shape[0]
        q = jax.vmap(self.q_proj)(x).reshape(N, self.num_heads, self.head_dim)
        k = jax.vmap(self.k_proj)(x).reshape(N, self.num_heads, self.head_dim)
        v = jax.vmap(self.v_proj)(x).reshape(N, self.num_heads, self.head_dim)

        # Attention scores: (heads, N, N).
        scores = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(self.head_dim)
        # Mask both query- and key-side invalid axes.
        mask2d = (valid[:, None] * valid[None, :]) > 0.5
        scores = jnp.where(mask2d[None, :, :], scores, -1e9)
        attn = jnn.softmax(scores, axis=-1)
        out = jnp.einsum("hij,jhd->ihd", attn, v).reshape(N, self.embd_dim)
        out = jax.vmap(self.out_proj)(out)
        x = x + out

        # Position-wise MLP.
        h = jax.vmap(self.mlp1)(x)
        h = jnn.gelu(h)
        h = jax.vmap(self.mlp2)(h)
        return x + h


# ---------------------------------------------------------------------------
# Op-type head: categorical over {DIAG, COMPRESS, END}
# ---------------------------------------------------------------------------


class OpTypeHead(eqx.Module):
    """Categorical over ``{DIAG, COMPRESS, END}``.

    Operates on the pooled axis-set summary plus vertex context (already
    folded into the summary by :class:`AxisSetEncoder`). The
    ``op_legality_mask`` argument is set by the rollout code based on
    the current axis state:

    * DIAG legal iff at least 2 distinct logical axes remain.
    * COMPRESS legal iff at least 1 physical axis remains.
    * END always legal.

    Returns the masked probability distribution; the caller samples
    from it and records the log-prob.
    """

    proj: eqx.nn.Linear

    def __init__(self, embd_dim: int, *, key):
        self.proj = eqx.nn.Linear(embd_dim, NUM_OPS, key=key)

    def __call__(self, summary: jax.Array, op_legality_mask: jax.Array):
        logits = self.proj(summary)
        masked = jnp.where(op_legality_mask > 0.5, logits, -1e9)
        return jnn.softmax(masked, axis=-1)


# ---------------------------------------------------------------------------
# Axis pointer head: attention over the current axis tokens
# ---------------------------------------------------------------------------


class AxisPointerHead(eqx.Module):
    """Single-head attention pointer over axis tokens.

    Used for the DIAG ``i`` / ``j`` selection and for the COMPRESS axis
    selection. For DIAG ``j``, the caller is expected to mask out
    ``j == i`` (passing ``axis_mask`` with the ``i``-th entry zeroed)
    so the categorical is over the legal sub-set.
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    embd_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, *, key):
        self.embd_dim = embd_dim
        keys = jrand.split(key, 2)
        self.query_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[0])
        self.key_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[1])

    def __call__(
        self,
        context: jax.Array,
        axis_tokens: jax.Array,
        axis_mask: jax.Array,
    ):
        q = self.query_proj(context)
        k = jax.vmap(self.key_proj)(axis_tokens)
        scores = (k @ q) / jnp.sqrt(self.embd_dim)
        scores = jnp.where(axis_mask > 0.5, scores, -1e9)
        return jnn.softmax(scores, axis=-1)


# ---------------------------------------------------------------------------
# Prime-exponent factor head
# ---------------------------------------------------------------------------


def factorize(n: int, max_primes: int = MAX_PRIMES) -> tuple[tuple[int, int], ...]:
    """Trial-division prime factorization, capped at ``max_primes`` distinct primes.

    Returns a tuple of ``(prime, exponent)`` pairs in ascending prime order.
    For ``n < 1e9`` this is well under a millisecond and the cap is never
    hit; the cap is just a safety net for adversarial inputs.
    """
    if n <= 0:
        return ()
    factors: list[tuple[int, int]] = []
    rem = n
    p = 2
    while p * p <= rem and len(factors) < max_primes:
        if rem % p == 0:
            e = 0
            while rem % p == 0:
                rem //= p
                e += 1
            factors.append((p, e))
        p += 1 if p == 2 else 2
    if rem > 1 and len(factors) < max_primes:
        factors.append((rem, 1))
    return tuple(factors)


def exponents_to_factor(
    primes: Sequence[int], exponents: Sequence[int],
) -> int:
    """Inverse of :func:`factorize` — multiply ``primes ** exponents``.

    Used by the env-side translator when converting a sampled prime-
    exponent tuple back into an integer factor for
    :class:`graphax.sparse.micro_actions.Diag`. Pads positions past the
    real prime count are ignored (exponent 0 contributes a factor of 1).
    """
    f = 1
    for p, e in zip(primes, exponents):
        f *= int(p) ** int(e)
    return f


class PrimeExponentHead(eqx.Module):
    """Autoregressive prime-exponent factor head.

    For each prime ``p_k`` in the factorization of ``g = gcd(N_i, N_j)``,
    in ascending prime order, emit a categorical over
    ``{0, 1, ..., e_k}`` (where ``e_k`` is the multiplicity of ``p_k``
    in ``g``). The chosen exponent ``c_k`` contributes ``p_k^c_k`` to
    the final factor; the full sample is ``f = ∏ p_k^c_k``.

    The head is autoregressive in the prime dimension — each prime's
    categorical is conditioned on the previously-chosen exponents
    (via a running hidden state) plus per-prime features (``p_k``,
    ``e_k``, running partial factor, log-ratios). The prime sequence
    is padded to :data:`MAX_PRIMES`; positions past the real prime
    count are masked so they contribute 0 to log-prob / entropy.

    The output categorical has fixed size ``MAX_EXPONENT + 1`` so the
    JAX scan stays shape-static; the per-prime mask
    ``e_k_mask[k] = arange(MAX_EXPONENT + 1) <= e_k`` zeros out logits
    above the legal exponent range.
    """

    prime_feat_proj: eqx.nn.Linear
    rnn_update: eqx.nn.Linear
    head_proj: eqx.nn.Linear

    embd_dim: int = eqx.field(static=True)
    max_exponent: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, max_exponent: int = MAX_EXPONENT, *, key):
        self.embd_dim = embd_dim
        self.max_exponent = max_exponent

        keys = jrand.split(key, 3)
        # Per-prime features fed into the recurrence:
        #   p_k, e_k, log(p_k), log(e_k+1), log(g), log(N_i), log(N_j),
        #   log(partial factor so far).
        feat_in = 8
        self.prime_feat_proj = eqx.nn.Linear(feat_in, embd_dim, key=keys[0])
        # RNN update: combines (prev_hidden, prime_features) -> new hidden.
        self.rnn_update = eqx.nn.Linear(embd_dim * 2, embd_dim, key=keys[1])
        # Output: hidden -> logits over [0, max_exponent].
        self.head_proj = eqx.nn.Linear(embd_dim, max_exponent + 1, key=keys[2])

    def _step_logits(
        self,
        prev_hidden: jax.Array,
        prime: jax.Array,
        max_exp: jax.Array,
        log_g: jax.Array,
        log_Ni: jax.Array,
        log_Nj: jax.Array,
        log_partial: jax.Array,
    ):
        feats = jnp.stack([
            prime.astype(jnp.float32),
            max_exp.astype(jnp.float32),
            jnp.log(jnp.maximum(prime.astype(jnp.float32), 1.0)),
            jnp.log(jnp.maximum(max_exp.astype(jnp.float32) + 1.0, 1.0)),
            log_g,
            log_Ni,
            log_Nj,
            log_partial,
        ])
        feat_emb = self.prime_feat_proj(feats)
        new_hidden = jnn.gelu(self.rnn_update(
            jnp.concatenate([prev_hidden, feat_emb], axis=-1)
        ))
        logits = self.head_proj(new_hidden)
        # Mask exponents above the prime's true multiplicity.
        legal = jnp.arange(self.max_exponent + 1) <= max_exp
        masked_logits = jnp.where(legal, logits, -1e9)
        return new_hidden, masked_logits

    def sample(
        self,
        init_hidden: jax.Array,
        primes: jax.Array,           # (MAX_PRIMES,) int32, padded with 0
        max_exps: jax.Array,         # (MAX_PRIMES,) int32, padded with 0
        prime_mask: jax.Array,       # (MAX_PRIMES,) float32, 1 for real
        N_i: jax.Array,
        N_j: jax.Array,
        g: jax.Array,
        key,
    ):
        """Autoregressive sample over the padded prime sequence."""
        log_g = jnp.log(jnp.maximum(g.astype(jnp.float32), 1.0))
        log_Ni = jnp.log(jnp.maximum(N_i.astype(jnp.float32), 1.0))
        log_Nj = jnp.log(jnp.maximum(N_j.astype(jnp.float32), 1.0))
        keys = jrand.split(key, MAX_PRIMES)

        def step(carry, inputs):
            hidden, log_partial = carry
            prime, max_exp, mask, k = inputs
            new_hidden, logits = self._step_logits(
                hidden, prime, max_exp, log_g, log_Ni, log_Nj, log_partial,
            )
            dist = jnn.softmax(logits, axis=-1)
            exponent = distrax.Categorical(probs=dist).sample(seed=k)
            # If this prime slot is padding, force exponent = 0 (no contribution)
            # and skip the hidden-state update so the recurrence stays clean.
            exponent_eff = jnp.where(mask > 0.5, exponent, 0).astype(jnp.int32)
            log_inc = exponent_eff.astype(jnp.float32) * jnp.log(
                jnp.maximum(prime.astype(jnp.float32), 1.0)
            )
            new_log_partial = log_partial + log_inc * mask
            hidden_eff = jnp.where(mask > 0.5, new_hidden, hidden)
            return (hidden_eff, new_log_partial), (exponent_eff, dist)

        init_carry = (init_hidden, jnp.array(0.0, dtype=jnp.float32))
        _, (exponents, dists) = lax.scan(
            step, init_carry, (primes, max_exps, prime_mask, keys),
        )
        return exponents, dists

    def evaluate(
        self,
        init_hidden: jax.Array,
        primes: jax.Array,
        max_exps: jax.Array,
        prime_mask: jax.Array,
        N_i: jax.Array,
        N_j: jax.Array,
        g: jax.Array,
        chosen_exponents: jax.Array,
    ):
        """Return (log_prob, entropy, per-prime dists) for ``chosen_exponents``."""
        log_g = jnp.log(jnp.maximum(g.astype(jnp.float32), 1.0))
        log_Ni = jnp.log(jnp.maximum(N_i.astype(jnp.float32), 1.0))
        log_Nj = jnp.log(jnp.maximum(N_j.astype(jnp.float32), 1.0))

        def step(carry, inputs):
            hidden, log_partial = carry
            prime, max_exp, mask, chosen = inputs
            new_hidden, logits = self._step_logits(
                hidden, prime, max_exp, log_g, log_Ni, log_Nj, log_partial,
            )
            dist = jnn.softmax(logits, axis=-1)
            log_p = jnp.log(dist[chosen] + 1e-8) * mask
            # Entropy on the masked distribution; padding contributes 0.
            ent = -jnp.sum(dist * jnp.log(dist + 1e-8), axis=-1) * mask
            log_inc = chosen.astype(jnp.float32) * jnp.log(
                jnp.maximum(prime.astype(jnp.float32), 1.0)
            )
            new_log_partial = log_partial + log_inc * mask
            hidden_eff = jnp.where(mask > 0.5, new_hidden, hidden)
            return (hidden_eff, new_log_partial), (log_p, ent, dist)

        init_carry = (init_hidden, jnp.array(0.0, dtype=jnp.float32))
        _, (log_ps, ents, dists) = lax.scan(
            step, init_carry,
            (primes, max_exps, prime_mask, chosen_exponents.astype(jnp.int32)),
        )
        return jnp.sum(log_ps), jnp.sum(ents), dists


# ---------------------------------------------------------------------------
# Composed micro-action head (one sub-step)
# ---------------------------------------------------------------------------


class MicroAction(NamedTuple):
    """Typed sample emitted by :meth:`MicroActionHead.sample` per sub-step.

    * ``op_type`` ∈ ``{OP_DIAG, OP_COMPRESS, OP_END}``.
    * ``i`` / ``j``: axis-token indices into the current axis set.
      ``j`` is unused for COMPRESS / END (filled with 0). For COMPRESS
      ``i`` is the physical axis being compressed.
    * ``exponents``: ``(MAX_PRIMES,)`` int32 prime exponents for DIAG;
      zeros for COMPRESS / END.

    The env-side translator (alphagrad.approx.env, future change) maps
    this tuple to a graphax ``Diag`` / ``Compress`` using the current
    axis state's ``(prime_list, max_exponents)`` for the chosen ``(i, j)``
    pair, then dispatches to ``apply_micro_actions``.
    """

    op_type: jax.Array
    i: jax.Array
    j: jax.Array
    exponents: jax.Array


class MicroActionHead(eqx.Module):
    """Per-sub-step head: op_type + axis pointers + prime exponents.

    The encoder is *not* re-run inside this module — the caller is
    expected to invoke :class:`AxisSetEncoder` once per sub-step (after
    updating the axis-token features to reflect the previous
    micro-action) and pass in ``axis_tokens`` and ``summary``.

    Per-sub-step legality is the caller's responsibility:

    * ``op_legality_mask`` reflects how many logical / physical axes
      remain (DIAG needs ≥ 2 logical, COMPRESS needs ≥ 1 physical).
    * For DIAG, ``i_mask`` and ``j_mask`` exclude axes already paired
      in this vertex's sub-episode (the policy is bipartite — each
      axis used in at most one DIAG pair per vertex).
    * For COMPRESS, the mask covers physical axes only.

    Apart from these structural masks, the head is unrestricted; the
    same module is used in sampling and evaluation.
    """

    op_head: OpTypeHead
    axis_i_head: AxisPointerHead
    axis_j_head: AxisPointerHead
    factor_head: PrimeExponentHead

    embd_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, *, key):
        self.embd_dim = embd_dim
        keys = jrand.split(key, 4)
        self.op_head = OpTypeHead(embd_dim, key=keys[0])
        self.axis_i_head = AxisPointerHead(embd_dim, key=keys[1])
        self.axis_j_head = AxisPointerHead(embd_dim, key=keys[2])
        self.factor_head = PrimeExponentHead(embd_dim, key=keys[3])

    # The sample / evaluate methods take per-sub-step state. They return
    # *flat* (single sub-step) outputs; the surrounding scan in
    # :class:`MicroActionPolicy` handles the variable-length sequence.

    def sample_step(
        self,
        summary: jax.Array,             # (E,) pooled axis-set summary
        axis_tokens: jax.Array,         # (N, E) per-axis embeddings
        op_legality_mask: jax.Array,    # (NUM_OPS,)
        i_mask: jax.Array,              # (N,) — valid axes for `i`
        j_mask_for_i: jax.Array,        # (N, N) — valid `j` axes given `i`
        primes: jax.Array,              # (MAX_PRIMES,) for the chosen (i, j)
        max_exps: jax.Array,            # (MAX_PRIMES,)
        prime_mask: jax.Array,          # (MAX_PRIMES,)
        N_i: jax.Array, N_j: jax.Array, g: jax.Array,
        key,
    ) -> tuple[MicroAction, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Sample one micro-action. Returns ``(action, op_dist, i_dist, j_dist, exp_dists)``.

        The caller is responsible for re-running the axis encoder after
        applying ``action`` and updating the axis-state features. The
        prime / max_exp / mask / N_i / N_j / g arrays are pre-computed
        host-side from the current axis state.
        """
        k_op, k_i, k_j, k_f = jrand.split(key, 4)

        op_dist = self.op_head(summary, op_legality_mask)
        op_type = distrax.Categorical(probs=op_dist).sample(seed=k_op)

        i_dist = self.axis_i_head(summary, axis_tokens, i_mask)
        i_idx = distrax.Categorical(probs=i_dist).sample(seed=k_i)

        # `j` legality depends on which `i` was chosen. ``j_mask_for_i`` is
        # the precomputed per-i row.
        j_mask = j_mask_for_i[i_idx]
        # Use the chosen-axis embedding as additional context for the j head
        # so the pointer can pick a complementary axis.
        j_context = summary + axis_tokens[i_idx]
        j_dist = self.axis_j_head(j_context, axis_tokens, j_mask)
        j_idx = distrax.Categorical(probs=j_dist).sample(seed=k_j)

        exponents, exp_dists = self.factor_head.sample(
            init_hidden=summary,
            primes=primes, max_exps=max_exps, prime_mask=prime_mask,
            N_i=N_i, N_j=N_j, g=g, key=k_f,
        )

        # Force i/j/exponents to 0 for non-DIAG ops so the recorded action
        # is unambiguous. The masking in `log_prob_step` mirrors this.
        is_diag = op_type == OP_DIAG
        is_compress = op_type == OP_COMPRESS

        i_out = jnp.where(is_diag | is_compress, i_idx, 0).astype(jnp.int32)
        j_out = jnp.where(is_diag, j_idx, 0).astype(jnp.int32)
        exp_out = jnp.where(is_diag, exponents, jnp.zeros_like(exponents))

        action = MicroAction(
            op_type=op_type.astype(jnp.int32),
            i=i_out, j=j_out, exponents=exp_out,
        )
        return action, op_dist, i_dist, j_dist, exp_dists

    def log_prob_step(
        self,
        action: MicroAction,
        summary: jax.Array,
        axis_tokens: jax.Array,
        op_legality_mask: jax.Array,
        i_mask: jax.Array,
        j_mask_for_i: jax.Array,
        primes: jax.Array,
        max_exps: jax.Array,
        prime_mask: jax.Array,
        N_i: jax.Array, N_j: jax.Array, g: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Joint log-prob + entropy for one sub-step (variable-arity gated by op_type)."""
        op_dist = self.op_head(summary, op_legality_mask)
        log_p_op = jnp.log(op_dist[action.op_type] + 1e-8)
        ent_op = -jnp.sum(op_dist * jnp.log(op_dist + 1e-8))

        is_diag = action.op_type == OP_DIAG
        is_compress = action.op_type == OP_COMPRESS

        # i: emitted for DIAG and COMPRESS.
        i_dist = self.axis_i_head(summary, axis_tokens, i_mask)
        log_p_i = jnp.log(i_dist[action.i] + 1e-8)
        ent_i = -jnp.sum(i_dist * jnp.log(i_dist + 1e-8))
        i_active = (is_diag | is_compress).astype(jnp.float32)

        # j: emitted only for DIAG.
        j_mask = j_mask_for_i[action.i]
        j_context = summary + axis_tokens[action.i]
        j_dist = self.axis_j_head(j_context, axis_tokens, j_mask)
        log_p_j = jnp.log(j_dist[action.j] + 1e-8)
        ent_j = -jnp.sum(j_dist * jnp.log(j_dist + 1e-8))
        j_active = is_diag.astype(jnp.float32)

        # Prime exponents: emitted only for DIAG.
        log_p_f, ent_f, _ = self.factor_head.evaluate(
            init_hidden=summary,
            primes=primes, max_exps=max_exps, prime_mask=prime_mask,
            N_i=N_i, N_j=N_j, g=g,
            chosen_exponents=action.exponents,
        )
        f_active = is_diag.astype(jnp.float32)

        log_p = log_p_op + log_p_i * i_active + log_p_j * j_active + log_p_f * f_active
        entropy = ent_op + ent_i * i_active + ent_j * j_active + ent_f * f_active

        # Sub-step "arity" = number of component categoricals actually emitted.
        # Used by the surrounding PPO loss to normalize the entropy bonus by
        # expected sub-episode length (mirroring the existing pair_active /
        # factor_active masking in AutoregRulePolicy.evaluate).
        arity = 1.0 + i_active + j_active + f_active
        return log_p, entropy, arity


# ---------------------------------------------------------------------------
# Sub-episode policy (encoder + head + scan over max_substeps)
# ---------------------------------------------------------------------------


class MicroActionPolicy(eqx.Module):
    """Full sub-episode: encoder + MicroActionHead, scanned to ``max_substeps``.

    Per sub-step the scan:

    1. Re-encodes the current axis-token set with :class:`AxisSetEncoder`
       (axis state changes after every action — paired DIAG axes carry
       the chosen factor as their new size, compressed axes are flagged
       and dropped from the valid set).
    2. Computes per-step legality masks: op-type (DIAG ≥ 2 logical axes,
       COMPRESS ≥ 1 physical axis), i / j axis pointers (j excludes the
       chosen i and any axis already paired in this sub-episode for the
       bipartite constraint).
    3. Samples one micro-action via :class:`MicroActionHead.sample_step`.
    4. Applies the action to the axis features for the next iteration.

    Iterations past the first END contribute 0 to the joint log-prob and
    entropy via a ``post_end_mask`` that gates the head's contribution.
    The axis state stops mutating after END (the update functions all
    no-op when ``ended == True``), so the scan stays shape-static.

    Open dependency (host-side preprocessing required)
    --------------------------------------------------
    Prime factorisations of the (per-pair) ``gcd(N_i, N_j)`` are needed
    by :class:`PrimeExponentHead`. Trial division inside a JAX scan is
    awkward; the practical answer is either:

    * A static lookup table built at env-init time, gathered inside the
      scan from ``axis_features.size[i]`` and ``axis_features.size[j]``.
      Works when axis sizes are bounded (which they are for any
      concrete jaxpr) — table size ``O(MAX_AXIS_SIZE)``.
    * A host-side ``io_callback`` per scan iteration. Slower but no
      bound on axis sizes.

    Neither is implemented in this module — the scan currently expects
    the caller to provide pre-computed ``(primes, max_exps, prime_mask,
    g)`` arrays keyed by sub-step index. The env-side change that wires
    :class:`MicroActionPolicy` into ``Agent.sample_action`` will own
    this preprocessing.
    """

    encoder: AxisSetEncoder
    head: MicroActionHead

    max_substeps: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)

    def __init__(
        self, embd_dim: int, num_heads: int, max_substeps: int,
        num_encoder_layers: int = 1, max_groups: int = 16, *, key,
    ):
        self.embd_dim = embd_dim
        self.max_substeps = max_substeps
        keys = jrand.split(key, 2)
        self.encoder = AxisSetEncoder(
            embd_dim, num_heads, num_layers=num_encoder_layers,
            max_groups=max_groups, key=keys[0],
        )
        self.head = MicroActionHead(embd_dim, key=keys[1])

    # The actual scan body needs (a) the axis-feature update function and
    # (b) the per-pair prime tables. Both depend on env-side choices we
    # haven't pinned down (lookup table size, how compressed-axis
    # sizes propagate, etc.). The skeleton below documents the expected
    # control flow; the env-side wiring will fill in the gaps.

    def sample(self, *args, **kwargs):  # pragma: no cover — scaffold only
        raise NotImplementedError(
            "MicroActionPolicy.sample is scaffolded but not wired. "
            "Implementation needs: (1) prime-factor lookup or io_callback "
            "for per-pair gcd factorisation, (2) axis-feature update "
            "function consistent with graphax.sparse.micro_actions semantics, "
            "(3) per-step legality mask computation. See module docstring."
        )

    def evaluate(self, *args, **kwargs):  # pragma: no cover — scaffold only
        raise NotImplementedError(
            "MicroActionPolicy.evaluate is scaffolded but not wired. "
            "Same dependencies as sample; needs to mirror the scan with "
            "fixed actions and return summed log-prob / entropy / arity."
        )


__all__ = [
    "OP_DIAG", "OP_COMPRESS", "OP_END", "NUM_OPS",
    "MAX_PRIMES", "MAX_EXPONENT", "AXIS_TAG_BITS",
    "AxisTokenFeatures",
    "AxisSetEncoder",
    "OpTypeHead",
    "AxisPointerHead",
    "PrimeExponentHead",
    "MicroAction",
    "MicroActionHead",
    "MicroActionPolicy",
    "factorize",
    "exponents_to_factor",
]
