"""Autoregressive sub-episode head for the post-pointer policy.

After the base pointer net picks a vertex ``v``, this module opens a
sub-episode at ``v`` that emits a variable-length sequence of typed
micro-actions until END:

    a_t ∈ { Diag(i, j, factor),  Compress(physical_axes),  Quant(dtype),  End }

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

Per-local-path (per-face) control
---------------------------------
Everything above is PER VERTEX: its rules apply uniformly to every local
path the elimination contracts. :class:`FacePathPolicy` (bottom of this
module) is the per-PATH layer built on top — for each face graphax's
``faces_of`` enumerates, it emits a SKIP decision and then, if the path is
to be approximated, one micro-action for each of the three slots
``lhs -> rhs -> res``, reusing the same head factorization. See the
"Per-LOCAL-PATH (per-face) approximation" banner there, in particular the
MIRROR INVARIANT note.

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
import numpy as np
import os

from graphax.sparse.micro_actions import NUM_QUANT_DTYPES, QUANT_DTYPES

# When ALPHAGRAD_SUBSTEP_NO_END=1, END is illegal in the op-type head, so every
# eliminated vertex emits exactly one real micro-action (with --max-substeps 1:
# one of {DIAG, COMPRESS, QUANT} — "always quant, else one of the other two";
# no exact/no-op vertices). Scoped via env var so the cmorl/mogfn stacks (which
# share heads.py) keep END legal by default.
# IMPORTANT: these two policy switches are read LAZILY (first use, at jit-trace
# time) — NOT at module import. In a Ray PPOActor the ALPHAGRAD_* env vars are
# delivered via runtime_env / set by init_worker AFTER this module is already
# imported, so an import-time read saw stock env and silently no-op'd (the bug
# that left the policy sampling all 28 quant dtypes + END enabled). Reading at
# first __call__ guarantees the actor's env is in place. Cached after first read.
_SUBSTEP_NO_END_CACHE = None


def _substep_no_end() -> bool:
    """ALPHAGRAD_SUBSTEP_NO_END=1 -> END illegal in the op-type head (every
    vertex emits exactly one real micro-action). Read lazily; see note above."""
    global _SUBSTEP_NO_END_CACHE
    if _SUBSTEP_NO_END_CACHE is None:
        _SUBSTEP_NO_END_CACHE = (
            os.environ.get("ALPHAGRAD_SUBSTEP_NO_END", "0") == "1"
        )
    return _SUBSTEP_NO_END_CACHE


# Restrict the QUANT dtype head to a configurable subset. ALPHAGRAD_QUANT_ALLOWED
# is a comma-list of QUANT_DTYPES names; disallowed dtypes are masked to -inf
# before the softmax (in sample AND log_prob, so PPO ratios stay consistent).
# Default (unset) = all dtypes legal. Used to drop dtypes JAX can't promote/cast
# (sub-byte int2/4, uint2/4, float4, exotic float8 *fnuz/e3m4/e8m0, complex) and
# no-op/unscaled ones (float32/64, plain int*) that otherwise crash or waste the
# forced per-vertex QUANT under the substeps=1 scheme. Read lazily (see note).
_QUANT_DTYPE_MASK_CACHE = None


def _quant_dtype_mask():
    """(NUM_QUANT_DTYPES,) float32 mask, 1.0 for allowed dtypes. Lazy; cached.

    Cached as NUMPY, not jnp: a jnp constant materialised during the FIRST
    jit trace (e.g. the rollout act_step) is a DynamicJaxprTracer of that
    trace — caching it and reusing it inside a LATER trace (the loss) raises
    UnexpectedTracerError. A numpy array is a fresh constant in every trace.
    """
    global _QUANT_DTYPE_MASK_CACHE
    if _QUANT_DTYPE_MASK_CACHE is None:
        _env = os.environ.get("ALPHAGRAD_QUANT_ALLOWED", "").strip()
        if _env:
            _allowed = {s.strip() for s in _env.split(",") if s.strip()}
            _QUANT_DTYPE_MASK_CACHE = np.array(
                [1.0 if d in _allowed else 0.0 for d in QUANT_DTYPES],
                dtype=np.float32,
            )
        else:
            _QUANT_DTYPE_MASK_CACHE = np.ones(NUM_QUANT_DTYPES, dtype=np.float32)
    return _QUANT_DTYPE_MASK_CACHE


# The DIAG axis-pair heads ALWAYS enforce the two block-diagonal legality rules
# (mirrored on the graphax execution side, whose GRAPHAX_KEEP_BLOCKDIAG is
# likewise on by default):
#   (1) FACTOR-DIVIDES-BLOCK: a further DIAG on an already-diagonalised axis may
#       only SUBDIVIDE its current block size -- the factor must be a multiple of
#       the current meta count that still divides both logical sizes. See
#       common.masks.diag_pair_factor_space, which returns exactly that space.
#   (2) AXIS-PAIR CONFLICT: an already-coupled axis i forces j == partner(i)
#       (j-head fast-skipped, its log-prob/entropy gated out of the PPO ratio);
#       a free axis i may only pair with another free axis.
# These were once behind ALPHAGRAD_MICRO_PAIR_MASKS, defaulting OFF, which meant
# the shipped default silently disagreed with the executor: coupled axes were
# excluded from DIAG entirely, so re-diagonalisation -- which graphax supports
# out of the box -- was unreachable, and i_coupled was pinned to zero so the
# j-head contributed log-prob for a choice it never really had. There is no
# reason to run without the rules the executor enforces, so the flag is gone.


# ---------------------------------------------------------------------------
# Constants and shape primitives
# ---------------------------------------------------------------------------

OP_DIAG = 0
OP_COMPRESS = 1
OP_QUANT = 2
OP_END = 3
NUM_OPS = 4

# Compress reductions emitted by :class:`CompressKindHead` and consumed by
# :class:`graphax.sparse.micro_actions.apply_compress`. Index alignment is
# the contract between the policy and the env-side translator — do not
# reorder without updating COMPRESS_KINDS in graphax (kept identical there).
COMPRESS_KINDS: tuple[str, ...] = (
    "mean", "min", "max", "median", "abs_min", "abs_max",
)
NUM_COMPRESS_KINDS = len(COMPRESS_KINDS)

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
# Indices into the tag_bits array — keep these in sync with the layout above
# and the env-side _AXIS_FEAT_* constants.
TAG_IS_LOGICAL = 0
TAG_IS_COMPRESSED = 1
TAG_IN_DIAG_GROUP = 2


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
    group_embedding: eqx.nn.Embedding | None
    blocks: tuple
    pool_query: jax.Array
    output_proj: eqx.nn.Linear

    embd_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    max_groups: int = eqx.field(static=True)
    use_group_embedding: bool = eqx.field(static=True)

    def __init__(self, embd_dim: int, num_heads: int, num_layers: int = 1,
                 max_groups: int = 16, *, key,
                 use_group_embedding: bool = False):
        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.max_groups = max_groups
        self.use_group_embedding = use_group_embedding

        keys = jrand.split(key, num_layers + 4)
        # In features: size, log_size, AXIS_TAG_BITS tag bits, plus an
        # optional group embedding when --axis-group-embedding is on.
        # Group membership is already represented in tag_bits[in_diag_group],
        # so the embedding is off by default — it adds an embedding table
        # (max_groups+1, embd_dim) and bloats proj_in's input by embd_dim.
        feat_in = 1 + 1 + AXIS_TAG_BITS + (embd_dim if use_group_embedding else 0)
        self.proj_in = eqx.nn.Linear(feat_in, embd_dim, key=keys[0])
        if use_group_embedding:
            # `+1` slot: index 0 reserved for "no group" (group_id == -1
            # gets remapped to 0 before the embedding lookup).
            self.group_embedding = eqx.nn.Embedding(
                max_groups + 1, embd_dim, key=keys[1],
            )
        else:
            self.group_embedding = None

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

        feat_list = [size_f, log_size_f, tag_f]
        if self.use_group_embedding:
            group_slot = jnp.where(
                features.group_id >= 0, features.group_id + 1, 0,
            ).astype(jnp.int32)
            group_slot = jnp.clip(group_slot, 0, self.max_groups)
            group_emb = jax.vmap(self.group_embedding)(group_slot)         # (N, E)
            feat_list.append(group_emb)
        feats = jnp.concatenate(feat_list, axis=-1)
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

        return jax.vmap(self.output_proj)(x), self.output_proj(pooled)


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
# Prime-factor lookup tables (env-static; gathered inside the policy scan)
# ---------------------------------------------------------------------------


class FactorTables(NamedTuple):
    """Static prime-factorization lookup tables consumed by the scan.

    Indexed by integer in ``[0, max_axis_size]`` (and ``[0, max_axis_size]
    × [0, max_axis_size]`` for the gcd table). Built once at env-init time
    from :func:`precompute_factor_tables` and passed into
    :class:`MicroActionPolicy` at every call; gathered inside the scan
    once we know which axes were picked.
    """

    gcd: jax.Array            # (S, S) int32; S = max_axis_size + 1
    primes: jax.Array         # (S, MAX_PRIMES) int32; padded with 0
    max_exps: jax.Array       # (S, MAX_PRIMES) int32; padded with 0
    prime_mask: jax.Array     # (S, MAX_PRIMES) float32; 1 for real, 0 for pad


def precompute_factor_tables(max_axis_size: int) -> FactorTables:
    """Build the static factor / gcd lookup tables.

    For an env whose axis sizes are bounded by ``max_axis_size`` (the max
    over the jaxpr's logical dim sizes), this returns the tables that
    let :class:`PrimeExponentHead` operate inside a JAX scan without
    trial-dividing integers at trace time.

    The tables are dense: ``(max_axis_size + 1)`` rows for the
    prime/exp tables, and ``(max_axis_size + 1)²`` entries for the gcd
    table. For ``max_axis_size = 1024`` that's ~1M int32 entries = 4 MiB
    in the gcd table; for typical jaxpr sizes (≤ 256) it's < 256 KiB.

    Index 0 represents "no axis" (the policy never samples it, but the
    table needs the row to keep static shape). ``gcd[0, *] = gcd[*, 0]
    = 0``; ``primes[0] = 0``; ``prime_mask[0] = 0``.
    """
    S = int(max_axis_size) + 1
    gcd = np.zeros((S, S), dtype=np.int32)
    primes = np.zeros((S, MAX_PRIMES), dtype=np.int32)
    exps = np.zeros((S, MAX_PRIMES), dtype=np.int32)
    prime_mask = np.zeros((S, MAX_PRIMES), dtype=np.float32)

    for n in range(1, S):
        factors = factorize(n, MAX_PRIMES)
        for k, (p, e) in enumerate(factors):
            primes[n, k] = p
            exps[n, k] = e
            prime_mask[n, k] = 1.0

    for i in range(S):
        for j in range(S):
            gcd[i, j] = math.gcd(i, j)

    return FactorTables(
        gcd=jnp.asarray(gcd),
        primes=jnp.asarray(primes),
        max_exps=jnp.asarray(exps),
        prime_mask=jnp.asarray(prime_mask),
    )


# ---------------------------------------------------------------------------
# Axis-state update helpers (used by MicroActionPolicy's scan carry)
# ---------------------------------------------------------------------------


def _features_after_diag(
    features: AxisTokenFeatures,
    i: jax.Array, j: jax.Array, factor: jax.Array, group_id: jax.Array,
) -> AxisTokenFeatures:
    """Mutate ``features`` to reflect a DIAG(i, j, factor) micro-action.

    Both axes ``i`` and ``j`` are tagged ``in_diag_group``, assigned the
    same ``group_id``, and have their size set to ``factor`` (the
    post-block-diagonalisation per-block index size). The block axes are
    not separately tracked in the policy's view — the structural
    consequence the policy cares about is "these two axes are paired and
    now have size ``factor``".
    """
    new_size = features.size.at[i].set(factor).at[j].set(factor)
    log_f = jnp.log(jnp.maximum(factor.astype(jnp.float32), 1.0))
    new_log_size = features.log_size.at[i].set(log_f).at[j].set(log_f)
    new_tag = features.tag_bits.at[i, TAG_IN_DIAG_GROUP].set(1.0).at[
        j, TAG_IN_DIAG_GROUP
    ].set(1.0)
    new_gid = features.group_id.at[i].set(group_id).at[j].set(group_id)
    return AxisTokenFeatures(
        size=new_size,
        log_size=new_log_size,
        tag_bits=new_tag,
        group_id=new_gid,
        valid_mask=features.valid_mask,
    )


def _features_after_compress(
    features: AxisTokenFeatures, i: jax.Array,
) -> AxisTokenFeatures:
    """Mutate ``features`` to reflect a COMPRESS(i) micro-action.

    Axis ``i`` is tagged ``is_compressed`` and removed from the active
    set via ``valid_mask[i] = 0`` (size collapses to 1, log_size to 0).
    The axis stays at its slot — JAX-static shape — but downstream
    attention masks zero it out at every layer.
    """
    new_tag = features.tag_bits.at[i, TAG_IS_COMPRESSED].set(1.0)
    new_valid = features.valid_mask.at[i].set(0.0)
    new_size = features.size.at[i].set(1)
    new_log_size = features.log_size.at[i].set(0.0)
    return AxisTokenFeatures(
        size=new_size,
        log_size=new_log_size,
        tag_bits=new_tag,
        group_id=features.group_id,
        valid_mask=new_valid,
    )


def _compute_op_legality(features: AxisTokenFeatures) -> jax.Array:
    """Op-type legality (NUM_OPS,) — DIAG / COMPRESS / QUANT / END."""
    is_compressed = features.tag_bits[:, TAG_IS_COMPRESSED] > 0.5
    in_diag = features.tag_bits[:, TAG_IN_DIAG_GROUP] > 0.5
    valid = features.valid_mask > 0.5

    diag_eligible = valid & ~is_compressed & ~in_diag
    compress_eligible = valid & ~is_compressed

    diag_legal = (jnp.sum(diag_eligible.astype(jnp.int32)) >= 2).astype(jnp.float32)
    compress_legal = (jnp.sum(compress_eligible.astype(jnp.int32)) >= 1).astype(
        jnp.float32
    )
    # QUANT is per-tensor, not per-axis — always legal at this layer. If the
    # SparseTensor has ``val is None`` at apply time, ``apply_quant`` returns
    # the tensor unchanged, so an emitted QUANT can't crash the env.
    quant_legal = jnp.array(1.0, dtype=jnp.float32)
    # END disabled under the substeps=1 "always-approximate" scheme: force one
    # real op (DIAG/COMPRESS/QUANT) per vertex. QUANT is always legal so there
    # is always >=1 legal op even when both structural ops are illegal.
    end_legal = jnp.array(0.0 if _substep_no_end() else 1.0, dtype=jnp.float32)
    return jnp.stack([diag_legal, compress_legal, quant_legal, end_legal])


def _compute_partner(features: AxisTokenFeatures) -> jax.Array:
    """Resolve, per axis, the index of its coupled DIAG partner (or -1).

    Two axes are partners iff they share a non-negative ``group_id`` (the
    id stamped by :func:`_features_after_diag` when they were block-
    diagonalised together). For an axis ``i`` with ``group_id[i] >= 0``
    the partner is the (unique) other valid axis ``k != i`` with
    ``group_id[k] == group_id[i]``. Free / ungrouped / compressed axes get
    partner ``-1``. Returned as ``(N,)`` int32.
    """
    gid = features.group_id
    valid = features.valid_mask > 0.5
    grouped = (gid >= 0) & valid                       # (N,)
    same_group = gid[:, None] == gid[None, :]          # (N, N)
    N = gid.shape[0]
    not_self = ~jnp.eye(N, dtype=jnp.bool_)
    # Candidate partners for i: valid, grouped, same gid, not i itself.
    cand = same_group & not_self & grouped[None, :] & grouped[:, None]
    # First matching column index per row; -1 if none.
    idx = jnp.arange(N)
    has_partner = jnp.any(cand, axis=-1)
    first = jnp.argmax(cand.astype(jnp.int32), axis=-1)
    return jnp.where(has_partner, first, -1).astype(jnp.int32)


def _finalise_axis_masks(diag_i_eligible, compress_eligible, j_mask_for_i,
                         i_coupled, pair_valid):
    """Shared tail of :func:`_compute_axis_masks` for both mask regimes."""
    if pair_valid is not None:
        j_mask_for_i = j_mask_for_i * pair_valid
    # An `i` whose entire j-row is masked would hand the j-head an all -1e9
    # logit vector, whose softmax is UNIFORM over illegal axes -- so a
    # perfectly legal-looking sample lands on an illegal pair. (This bites
    # without pair_valid too: when exactly one axis is diag-eligible its own
    # row is empty after the i != j term.) Such an `i` must not be selectable.
    has_partner = (jnp.sum(j_mask_for_i, axis=-1) > 0.0).astype(jnp.float32)
    return (diag_i_eligible * has_partner, compress_eligible, j_mask_for_i,
            i_coupled)


def _compute_axis_masks(
    features: AxisTokenFeatures,
    pair_valid: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-step axis legality.

    Returns ``(i_mask_diag, i_mask_compress, j_mask_for_i_diag, i_coupled)``.

    ``j_mask_for_i_diag[i]`` is the legal ``j`` set when DIAG is chosen
    with that particular ``i``; it enforces ``i != j`` and the
    block-structure pairing rules (rule #2): a coupled ``i`` forces ``j == partner(i)`` (its row is a
    single one-hot), a free ``i`` may only pair with another free axis.
    ``i_coupled[i]`` is 1.0 iff axis ``i`` already has a DIAG partner; the
    head uses it to fast-skip (gate out) the j-head when ``i`` is coupled.

    ``pair_valid`` is the OPTIONAL ``(N, N)`` per-edge legality mask computed
    by the env from the actual SparseTensor
    (:func:`alphagrad.approx.common.masks.diag_valid_mask`). The tag-bit
    reconstruction here is a proxy: it can see ``is_compressed`` and
    ``in_diag_group`` but NOT which side of the out/primal split an axis sits
    on, and a Diag must tie one OUT axis to one PRIMAL axis. When the env
    supplies ``pair_valid`` it is authoritative and is AND-ed in.
    """
    is_compressed = features.tag_bits[:, TAG_IS_COMPRESSED] > 0.5
    in_diag = features.tag_bits[:, TAG_IN_DIAG_GROUP] > 0.5
    valid = features.valid_mask > 0.5

    compress_eligible = (valid & ~is_compressed).astype(jnp.float32)
    N = valid.shape[0]
    eye = jnp.eye(N, dtype=jnp.float32)

    # --- Rule #2: coupled-vs-free pairing --------------------------------
    partner = _compute_partner(features)               # (N,) int32, -1 = free
    i_coupled = (partner >= 0).astype(jnp.float32)     # (N,)

    free = (valid & ~is_compressed & ~in_diag)         # uncoupled, DIAG-able
    free_f = free.astype(jnp.float32)

    # i may DIAG if it is a free axis (needs another free partner) OR it is
    # already coupled (re-diagonalise, forced partner + rule #1 subdivide).
    diag_i_eligible = (free | (partner >= 0)).astype(jnp.float32)

    # j-mask when i is FREE: only other free axes (never a coupled one).
    j_free = free_f[None, :] * (1.0 - eye)             # (N, N)
    # j-mask when i is COUPLED: the single forced partner (one-hot).
    partner_clip = jnp.clip(partner, 0, N - 1)
    partner_onehot = jnn.one_hot(partner_clip, N, dtype=jnp.float32)  # (N, N)
    partner_onehot = partner_onehot * i_coupled[:, None]

    j_mask_for_i = jnp.where(
        i_coupled[:, None] > 0.5, partner_onehot, j_free,
    )
    return _finalise_axis_masks(diag_i_eligible, compress_eligible,
                                j_mask_for_i, i_coupled, pair_valid)


# ---------------------------------------------------------------------------
# Composed micro-action head (one sub-step)
# ---------------------------------------------------------------------------


class MicroAction(NamedTuple):
    """Typed sample emitted by :meth:`MicroActionHead.sample` per sub-step.

    * ``op_type`` ∈ ``{OP_DIAG, OP_COMPRESS, OP_QUANT, OP_END}``.
    * ``i`` / ``j``: axis-token indices into the current axis set.
      ``j`` is unused for COMPRESS / QUANT / END (filled with 0). For
      COMPRESS ``i`` is the physical axis being compressed. ``i`` is
      unused for QUANT / END too.
    * ``exponents``: ``(MAX_PRIMES,)`` int32 prime exponents for DIAG;
      zeros for COMPRESS / QUANT / END.
    * ``factor``: derived integer factor for DIAG (``∏ p_k^c_k``); 0 for
      COMPRESS / QUANT / END. Stored explicitly because deriving it requires
      the prime table at sample time, and re-deriving inside evaluate would
      need the same gather — easier to keep the integer alongside its
      exponents.
    * ``compress_kind``: reduction-kind index ∈ ``[0, NUM_COMPRESS_KINDS)``
      consumed by :class:`graphax.sparse.micro_actions.Compress`. Meaningful
      only when ``op_type == OP_COMPRESS``; zero for DIAG / QUANT / END.
    * ``quant_dtype``: dtype index ∈ ``[0, NUM_QUANT_DTYPES)`` consumed by
      :class:`graphax.sparse.micro_actions.Quant`. Meaningful only when
      ``op_type == OP_QUANT``; zero for DIAG / COMPRESS / END.

    The env-side translator (alphagrad.approx.env) maps this tuple to a
    legacy rule_specs row via :func:`micro_actions_to_rule_specs_jax` or
    a graphax ``Diag`` / ``Compress`` / ``Quant`` via ``apply_micro_actions``
    for the future end-to-end path.
    """

    op_type: jax.Array
    i: jax.Array
    j: jax.Array
    exponents: jax.Array
    factor: jax.Array
    compress_kind: jax.Array
    quant_dtype: jax.Array


class CompressKindHead(eqx.Module):
    """Categorical over the six compress reductions in :data:`COMPRESS_KINDS`.

    Fires only when the op-type head picks ``OP_COMPRESS``; for DIAG / END
    sub-steps its sampled value is recorded but masked out of the joint
    log-prob and entropy (same gating used for the prime-exponent head on
    non-DIAG steps).
    """

    proj: eqx.nn.Linear

    def __init__(self, embd_dim: int, *, key):
        self.proj = eqx.nn.Linear(embd_dim, NUM_COMPRESS_KINDS, key=key)

    def __call__(self, summary: jax.Array, axis_token: jax.Array) -> jax.Array:
        logits = self.proj(summary + axis_token)
        return jnn.softmax(logits, axis=-1)


class QuantDtypeHead(eqx.Module):
    """Categorical over the JAX dtypes in :data:`QUANT_DTYPES`.

    Fires only when the op-type head picks ``OP_QUANT``; for non-QUANT
    sub-steps its sampled value is recorded but masked out of the joint
    log-prob, entropy, and arity (same gating used for
    :class:`CompressKindHead`). Reads only the pooled axis-set summary —
    Quant is per-tensor, not per-axis, so axis-token conditioning would
    be misleading.
    """

    proj: eqx.nn.Linear

    def __init__(self, embd_dim: int, *, key):
        self.proj = eqx.nn.Linear(embd_dim, NUM_QUANT_DTYPES, key=key)

    def __call__(self, summary: jax.Array) -> jax.Array:
        logits = self.proj(summary)
        # Mask disallowed quant dtypes to -inf (see _QUANT_DTYPE_MASK); applied
        # identically here for sampling and log-prob so PPO ratios are exact.
        logits = jnp.where(_quant_dtype_mask() > 0.5, logits, -1e9)
        return jnn.softmax(logits, axis=-1)


class MicroActionHead(eqx.Module):
    """Per-sub-step head: op_type + axis pointers + prime exponents + kind.

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
    compress_kind_head: CompressKindHead
    quant_dtype_head: QuantDtypeHead

    embd_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, *, key):
        self.embd_dim = embd_dim
        keys = jrand.split(key, 6)
        self.op_head = OpTypeHead(embd_dim, key=keys[0])
        self.axis_i_head = AxisPointerHead(embd_dim, key=keys[1])
        self.axis_j_head = AxisPointerHead(embd_dim, key=keys[2])
        self.factor_head = PrimeExponentHead(embd_dim, key=keys[3])
        self.compress_kind_head = CompressKindHead(embd_dim, key=keys[4])
        self.quant_dtype_head = QuantDtypeHead(embd_dim, key=keys[5])

    # The sample / evaluate methods take per-sub-step state. They return
    # *flat* (single sub-step) outputs; the surrounding scan in
    # :class:`MicroActionPolicy` handles the variable-length sequence.

    def sample_step(
        self,
        summary: jax.Array,             # (E,) pooled axis-set summary
        axis_tokens: jax.Array,         # (N, E) per-axis embeddings
        axis_sizes: jax.Array,          # (N,) int32 — current logical sizes
        op_legality_mask: jax.Array,    # (NUM_OPS,)
        i_mask_diag: jax.Array,         # (N,) — valid `i` for DIAG
        i_mask_compress: jax.Array,     # (N,) — valid `i` for COMPRESS
        j_mask_for_i_diag: jax.Array,   # (N, N) — valid `j` given `i`, DIAG only
        tables: FactorTables,           # precomputed gcd / prime tables
        key,
        i_coupled: jax.Array | None = None,  # (N,) — 1.0 if axis is coupled
    ):
        """Sample one micro-action.

        Returns ``(action, factor, op_dist, i_dist, j_dist, exp_dists,
        kind_dist)``. The factor is the *integer* derived from the sampled
        prime exponents — used by the surrounding scan to update axis sizes
        for the next sub-step. ``kind_dist`` is the categorical the
        :class:`CompressKindHead` sampled from at this step (constant
        across DIAG / END steps, used only on COMPRESS).

        The op-conditional ``i_mask`` selection happens inside this
        method: ``op_type`` is sampled first, then ``jnp.where`` picks
        between ``i_mask_diag`` and ``i_mask_compress``. END falls back
        to ``i_mask_diag`` (the sampled ``i`` is masked out downstream
        anyway).
        """
        k_op, k_i, k_j, k_f, k_kind, k_quant = jrand.split(key, 6)

        op_dist = self.op_head(summary, op_legality_mask)
        op_type = distrax.Categorical(probs=op_dist).sample(seed=k_op)

        is_diag = op_type == OP_DIAG
        is_compress = op_type == OP_COMPRESS
        is_quant = op_type == OP_QUANT

        # Op-conditional `i` mask. For END, falls back to the diag mask
        # (the resulting `i` is masked out via `i_active` downstream).
        i_mask = jnp.where(is_diag, i_mask_diag, i_mask_compress)
        i_dist = self.axis_i_head(summary, axis_tokens, i_mask)
        i_idx = distrax.Categorical(probs=i_dist).sample(seed=k_i)

        # `j` is sampled even for non-DIAG ops (its log-prob is masked
        # out in log_prob_step); using the DIAG j-mask keeps the
        # categorical well-defined.
        j_mask = j_mask_for_i_diag[i_idx]
        j_context = summary + axis_tokens[i_idx]
        j_dist = self.axis_j_head(j_context, axis_tokens, j_mask)
        j_idx = distrax.Categorical(probs=j_dist).sample(seed=k_j)

        # Rule #2 fast-skip: when the chosen `i` is already coupled, `j` is
        # DETERMINISTIC — the forced partner. `j_mask` above is already the
        # partner one-hot for a coupled `i`, so the sampled `j_idx` equals
        # the partner regardless of the RNG; we still overwrite it with the
        # explicit argmax so replay/telemetry is unambiguous. Its log-prob is
        # gated out of the joint (see `log_prob_step`), so this head makes no
        # PPO-ratio contribution for a coupled `i`.
        if i_coupled is not None:
            i_is_coupled = i_coupled[i_idx] > 0.5
            forced_j = jnp.argmax(j_mask).astype(j_idx.dtype)
            j_idx = jnp.where(i_is_coupled, forced_j, j_idx)

        # Per-pair prime-table gather. `axis_sizes` carries the current
        # logical sizes; `tables.gcd[N_i, N_j]` resolves the gcd at trace
        # time without a JAX-side trial division.
        N_i = axis_sizes[i_idx]
        N_j = axis_sizes[j_idx]
        g = tables.gcd[N_i, N_j]
        primes = tables.primes[g]
        max_exps = tables.max_exps[g]
        prime_mask = tables.prime_mask[g]

        exponents, exp_dists = self.factor_head.sample(
            init_hidden=summary,
            primes=primes, max_exps=max_exps, prime_mask=prime_mask,
            N_i=N_i, N_j=N_j, g=g, key=k_f,
        )

        # Derive integer factor from sampled exponents. ``primes ** 0 == 1``
        # for the padded entries, so the product is just ``∏ p_k^c_k`` over
        # the real primes.
        factor = jnp.prod(
            primes.astype(jnp.int32) ** exponents.astype(jnp.int32),
        ).astype(jnp.int32)

        # Compress-kind head: categorical over the six reductions. The
        # axis_token at the sampled `i` conditions the kind, so the
        # policy can pick e.g. abs_max for one axis and median for
        # another within the same sub-episode. Run unconditionally and
        # mask out the contribution for non-COMPRESS steps in log_prob_step.
        kind_dist = self.compress_kind_head(summary, axis_tokens[i_idx])
        kind_idx = distrax.Categorical(probs=kind_dist).sample(seed=k_kind)

        # Quant-dtype head: categorical over the JAX dtype catalog. Per-tensor
        # (no axis-token conditioning). Run unconditionally and mask out the
        # contribution for non-QUANT steps in log_prob_step.
        quant_dist = self.quant_dtype_head(summary)
        quant_idx = distrax.Categorical(probs=quant_dist).sample(seed=k_quant)

        # Force i/j/exponents/kind/quant to canonical values for non-emitting
        # ops so the recorded action is unambiguous. The masking in
        # `log_prob_step` mirrors this.
        i_out = jnp.where(is_diag | is_compress, i_idx, 0).astype(jnp.int32)
        j_out = jnp.where(is_diag, j_idx, 0).astype(jnp.int32)
        exp_out = jnp.where(is_diag, exponents, jnp.zeros_like(exponents))
        factor_out = jnp.where(is_diag, factor, jnp.array(0, dtype=jnp.int32))
        kind_out = jnp.where(is_compress, kind_idx, 0).astype(jnp.int32)
        quant_out = jnp.where(is_quant, quant_idx, 0).astype(jnp.int32)

        action = MicroAction(
            op_type=op_type.astype(jnp.int32),
            i=i_out, j=j_out, exponents=exp_out, factor=factor_out,
            compress_kind=kind_out, quant_dtype=quant_out,
        )
        return (
            action, factor_out, op_dist, i_dist, j_dist, exp_dists,
            kind_dist, quant_dist,
        )

    def log_prob_step(
        self,
        action: MicroAction,
        summary: jax.Array,
        axis_tokens: jax.Array,
        axis_sizes: jax.Array,
        op_legality_mask: jax.Array,
        i_mask_diag: jax.Array,
        i_mask_compress: jax.Array,
        j_mask_for_i_diag: jax.Array,
        tables: FactorTables,
        i_coupled: jax.Array | None = None,  # (N,) — 1.0 if axis is coupled
    ):
        """Joint log-prob + entropy + arity for one sub-step, plus the
        per-component distributions used downstream for KL tracking.

        Same per-op masking semantics as :meth:`sample_step`. The arity
        is the number of categoricals actually emitted (1 for END,
        ``2 + 1`` for COMPRESS — op + i + kind, and
        ``2 + sum(prime_mask)`` for DIAG — the prime sub-loop contributes
        one categorical per real prime in ``g``).

        Returns ``(log_p, entropy, arity, op_dist, i_dist, j_dist,
        exp_dists, kind_dist)`` so the caller can both train against the
        joint log-prob and compute per-component KL against stored old
        distributions.
        """
        op_dist = self.op_head(summary, op_legality_mask)
        log_p_op = jnp.log(op_dist[action.op_type] + 1e-8)
        ent_op = -jnp.sum(op_dist * jnp.log(op_dist + 1e-8))

        is_diag = action.op_type == OP_DIAG
        is_compress = action.op_type == OP_COMPRESS
        is_quant = action.op_type == OP_QUANT

        # i: emitted for DIAG and COMPRESS under the op-conditional mask.
        i_mask = jnp.where(is_diag, i_mask_diag, i_mask_compress)
        i_dist = self.axis_i_head(summary, axis_tokens, i_mask)
        log_p_i = jnp.log(i_dist[action.i] + 1e-8)
        ent_i = -jnp.sum(i_dist * jnp.log(i_dist + 1e-8))
        i_active = (is_diag | is_compress).astype(jnp.float32)

        # j: emitted only for DIAG. Under rule #2 the j-head is FAST-SKIPPED
        # (gated out of the joint log-prob AND entropy) when the chosen `i`
        # is already coupled — its `j` is the deterministic forced partner,
        # so it must not contribute to the PPO ratio, exactly like the other
        # inactive heads. `j_active` drops to 0 in that case.
        j_mask = j_mask_for_i_diag[action.i]
        j_context = summary + axis_tokens[action.i]
        j_dist = self.axis_j_head(j_context, axis_tokens, j_mask)
        log_p_j = jnp.log(j_dist[action.j] + 1e-8)
        ent_j = -jnp.sum(j_dist * jnp.log(j_dist + 1e-8))
        if i_coupled is not None:
            j_free = 1.0 - (i_coupled[action.i] > 0.5).astype(jnp.float32)
        else:
            j_free = 1.0
        j_active = is_diag.astype(jnp.float32) * j_free

        # Prime exponents: emitted only for DIAG. Tables gathered at the
        # stored (i, j); for non-DIAG actions the gather still runs but
        # the resulting log-prob / entropy are masked out.
        N_i = axis_sizes[action.i]
        N_j = axis_sizes[action.j]
        g = tables.gcd[N_i, N_j]
        primes = tables.primes[g]
        max_exps = tables.max_exps[g]
        prime_mask = tables.prime_mask[g]

        log_p_f, ent_f, exp_dists = self.factor_head.evaluate(
            init_hidden=summary,
            primes=primes, max_exps=max_exps, prime_mask=prime_mask,
            N_i=N_i, N_j=N_j, g=g,
            chosen_exponents=action.exponents,
        )
        f_active = is_diag.astype(jnp.float32)

        # Compress-kind: emitted only for COMPRESS. Conditioned on the
        # axis_token at action.i, matching the sample-time path.
        kind_dist = self.compress_kind_head(summary, axis_tokens[action.i])
        log_p_kind = jnp.log(kind_dist[action.compress_kind] + 1e-8)
        ent_kind = -jnp.sum(kind_dist * jnp.log(kind_dist + 1e-8))
        kind_active = is_compress.astype(jnp.float32)

        # Quant-dtype: emitted only for QUANT. Per-tensor; no axis-token
        # conditioning at sample time, so none here either.
        quant_dist = self.quant_dtype_head(summary)
        log_p_quant = jnp.log(quant_dist[action.quant_dtype] + 1e-8)
        ent_quant = -jnp.sum(quant_dist * jnp.log(quant_dist + 1e-8))
        quant_active = is_quant.astype(jnp.float32)

        log_p = (
            log_p_op + log_p_i * i_active + log_p_j * j_active
            + log_p_f * f_active + log_p_kind * kind_active
            + log_p_quant * quant_active
        )
        entropy = (
            ent_op + ent_i * i_active + ent_j * j_active
            + ent_f * f_active + ent_kind * kind_active
            + ent_quant * quant_active
        )

        # Arity counts the *components* actually emitted: 1 (op_type) + i +
        # j + the prime sub-loop + compress_kind + quant_dtype. The prime
        # sub-loop's contribution scales with the number of real primes in g;
        # we use ``prime_mask.sum()`` so DIAG sub-steps with more primes count
        # for more (matching the entropy term that already weights by
        # prime_mask). COMPRESS adds 1 for the kind categorical; QUANT adds
        # 1 for the dtype categorical.
        prime_arity = jnp.sum(prime_mask) * f_active
        arity = (
            1.0 + i_active + j_active + prime_arity + kind_active + quant_active
        )
        return (
            log_p, entropy, arity,
            op_dist, i_dist, j_dist, exp_dists, kind_dist, quant_dist,
        )


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

    Prime tables
    ------------
    The :class:`PrimeExponentHead` is gathered from the
    :class:`FactorTables` precomputed via :func:`precompute_factor_tables`
    at env init. The gather is over the *current* ``(N_i, N_j)`` axis
    sizes inside the scan, so post-DIAG size changes feed through
    naturally to the next sub-step's factor head.
    """

    encoder: AxisSetEncoder
    head: MicroActionHead

    max_substeps: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)

    def __init__(
        self, embd_dim: int, num_heads: int, max_substeps: int,
        num_encoder_layers: int = 1, max_groups: int = 16, *, key,
        use_group_embedding: bool = False,
    ):
        self.embd_dim = embd_dim
        self.max_substeps = max_substeps
        keys = jrand.split(key, 2)
        self.encoder = AxisSetEncoder(
            embd_dim, num_heads, num_layers=num_encoder_layers,
            max_groups=max_groups, key=keys[0],
            use_group_embedding=use_group_embedding,
        )
        self.head = MicroActionHead(embd_dim, key=keys[1])

    def _step_sample(
        self,
        carry,
        step_input,
        *,
        vertex_context: jax.Array,
        tables: FactorTables,
        cap: jax.Array,
    ):
        features, ended, next_gid, step_idx = carry
        key = step_input

        axis_tokens, summary = self.encoder(features, vertex_context)
        op_legal = _compute_op_legality(features)
        # Force END once the sub-episode has ended (sticky termination) or
        # the per-vertex hard cap (2 × num_axes) is reached — the latter is
        # a length bound, not a structural constraint.
        force_end = ended | (step_idx >= cap)
        # Order: DIAG, COMPRESS, QUANT, END — only END stays legal under
        # force_end (sticky termination).
        op_legal = jnp.where(
            force_end,
            jnp.array([0.0, 0.0, 0.0, 1.0], dtype=op_legal.dtype),
            op_legal,
        )
        i_diag, i_compress, j_diag, i_coupled = _compute_axis_masks(features)

        (
            action, factor, op_d, i_d, j_d, exp_d, kind_d, quant_d,
        ) = self.head.sample_step(
            summary, axis_tokens, features.size,
            op_legal, i_diag, i_compress, j_diag,
            tables, key, i_coupled=i_coupled,
        )

        # log_prob_step recomputes the dists internally but we discard
        # them here — the sample_step return values are the dists at the
        # rollout-time policy snapshot (the "old" dists for PPO), which
        # is exactly what the trajectory needs.
        log_p, ent, arity, *_ = self.head.log_prob_step(
            action, summary, axis_tokens, features.size,
            op_legal, i_diag, i_compress, j_diag, tables,
            i_coupled=i_coupled,
        )
        # Past-END contributions are zeroed so the joint log-prob /
        # entropy / arity reflect only the real sub-episode prefix.
        active = (1.0 - ended.astype(jnp.float32))
        log_p = log_p * active
        ent = ent * active
        arity = arity * active

        # Apply the action to the axis features — DIAG and COMPRESS get
        # the right update; END / past-END leave features untouched.
        is_diag = (action.op_type == OP_DIAG) & ~ended
        is_compress = (action.op_type == OP_COMPRESS) & ~ended

        diag_features = _features_after_diag(
            features, action.i, action.j, factor, next_gid,
        )
        compress_features = _features_after_compress(features, action.i)

        def _select(name):
            base = getattr(features, name)
            d = getattr(diag_features, name)
            c = getattr(compress_features, name)
            chosen_d = jnp.where(is_diag, d, base)
            return jnp.where(is_compress, c, chosen_d)

        new_features = AxisTokenFeatures(
            size=_select("size"),
            log_size=_select("log_size"),
            tag_bits=_select("tag_bits"),
            group_id=_select("group_id"),
            valid_mask=_select("valid_mask"),
        )

        new_gid = jnp.where(is_diag, next_gid + 1, next_gid)
        new_ended = ended | (action.op_type == OP_END)

        return (
            (new_features, new_ended, new_gid, step_idx + 1),
            (action, log_p, ent, arity, op_d, i_d, j_d, exp_d, kind_d, quant_d),
        )

    def sample(
        self,
        vertex_context: jax.Array,
        init_features: AxisTokenFeatures,
        tables: FactorTables,
        key,
    ):
        """Run the sub-episode autoregressively.

        Returns the per-step :class:`MicroAction` sequence (padded to
        ``max_substeps``) plus the joint log-prob, entropy, and total
        emitted-component count (used by the PPO loss to normalize the
        entropy bonus across variable-arity sub-episodes).
        """
        keys = jrand.split(key, self.max_substeps)
        # Per-vertex hard cap: 2 × number of real axes. The scan runs
        # `max_substeps` iterations regardless (JAX-static shape); once
        # `step_idx >= cap` the op-type head is forced to END so any
        # remaining iterations contribute zero log-prob / entropy.
        cap = 2 * jnp.sum(init_features.valid_mask.astype(jnp.int32))
        init_carry = (
            init_features,
            jnp.array(False, dtype=jnp.bool_),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )

        def step_fn(carry, k):
            return self._step_sample(
                carry, k,
                vertex_context=vertex_context, tables=tables, cap=cap,
            )

        (
            _,
            (actions, logps, ents, arities, op_dists, i_dists, j_dists,
             exp_dists, kind_dists, quant_dists),
        ) = lax.scan(step_fn, init_carry, keys)
        return (
            actions,
            jnp.sum(logps),
            jnp.sum(ents),
            jnp.sum(arities),
            op_dists,
            i_dists,
            j_dists,
            exp_dists,
            kind_dists,
            quant_dists,
        )

    def _step_evaluate(
        self,
        carry,
        step_input,
        *,
        vertex_context: jax.Array,
        tables: FactorTables,
        cap: jax.Array,
    ):
        features, ended, next_gid, step_idx = carry
        action: MicroAction = step_input

        axis_tokens, summary = self.encoder(features, vertex_context)
        op_legal = _compute_op_legality(features)
        force_end = ended | (step_idx >= cap)
        op_legal = jnp.where(
            force_end,
            jnp.array([0.0, 0.0, 0.0, 1.0], dtype=op_legal.dtype),
            op_legal,
        )
        i_diag, i_compress, j_diag, i_coupled = _compute_axis_masks(features)

        (
            log_p, ent, arity,
            op_dist, i_dist, j_dist, exp_dists, kind_dist, quant_dist,
        ) = self.head.log_prob_step(
            action, summary, axis_tokens, features.size,
            op_legal, i_diag, i_compress, j_diag, tables,
            i_coupled=i_coupled,
        )
        active = (1.0 - ended.astype(jnp.float32))
        log_p = log_p * active
        ent = ent * active
        arity = arity * active

        # Axis-state update uses the *stored* factor from the action.
        # This is the same integer sample_step computed via the prime
        # table; storing it avoids the re-derivation gather here and
        # makes the stored action self-contained for replay / debugging.
        factor = action.factor

        is_diag = (action.op_type == OP_DIAG) & ~ended
        is_compress = (action.op_type == OP_COMPRESS) & ~ended

        diag_features = _features_after_diag(
            features, action.i, action.j, factor, next_gid,
        )
        compress_features = _features_after_compress(features, action.i)

        def _select(name):
            base = getattr(features, name)
            d = getattr(diag_features, name)
            c = getattr(compress_features, name)
            chosen_d = jnp.where(is_diag, d, base)
            return jnp.where(is_compress, c, chosen_d)

        new_features = AxisTokenFeatures(
            size=_select("size"),
            log_size=_select("log_size"),
            tag_bits=_select("tag_bits"),
            group_id=_select("group_id"),
            valid_mask=_select("valid_mask"),
        )

        new_gid = jnp.where(is_diag, next_gid + 1, next_gid)
        new_ended = ended | (action.op_type == OP_END)

        return (
            (new_features, new_ended, new_gid, step_idx + 1),
            (
                log_p, ent, arity,
                op_dist, i_dist, j_dist, exp_dists, kind_dist, quant_dist,
            ),
        )

    def evaluate(
        self,
        vertex_context: jax.Array,
        init_features: AxisTokenFeatures,
        tables: FactorTables,
        actions: MicroAction,
    ):
        """Recompute joint log-prob / entropy / arity for a stored sequence,
        plus per-step distributions for KL tracking.

        ``actions`` is the per-step :class:`MicroAction` sequence emitted
        by :meth:`sample` (each field has a leading ``max_substeps`` dim).
        Returns ``(log_prob, entropy, arity, op_dists, i_dists, j_dists,
        exp_dists)`` — the scalars are summed across the sub-episode
        prefix (contributions past END masked out); the dists are the
        per-step distributions under the *current* policy, with shape
        ``(max_substeps, ...)``. Pair these with the stored old-policy
        dists in the trajectory to compute per-component KL.
        """
        cap = 2 * jnp.sum(init_features.valid_mask.astype(jnp.int32))
        init_carry = (
            init_features,
            jnp.array(False, dtype=jnp.bool_),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )

        def step_fn(carry, action_step):
            return self._step_evaluate(
                carry, action_step,
                vertex_context=vertex_context, tables=tables, cap=cap,
            )

        (
            _,
            (
                logps, ents, arities,
                op_dists, i_dists, j_dists, exp_dists, kind_dists, quant_dists,
            ),
        ) = lax.scan(step_fn, init_carry, actions)
        return (
            jnp.sum(logps), jnp.sum(ents), jnp.sum(arities),
            op_dists, i_dists, j_dists, exp_dists, kind_dists, quant_dists,
        )


# ---------------------------------------------------------------------------
# Per-LOCAL-PATH (per-face) approximation
# ---------------------------------------------------------------------------
#
# A vertex elimination contracts one FACE (local path) per
# ``in_edge -> central_var -> out_edge`` triple. The per-vertex micro-action
# sub-episode above applies its rules UNIFORMLY to every face of the vertex,
# which is exactly what a policy that wants to approximate one path and leave
# another exact cannot express.
#
# ``graphax.faces_of(graph, transpose_graph, vertex, jaxpr)`` returns the face
# keys ``(vidx[in_edge], vidx[out_edge])`` in the order the elimination will
# visit them, computable BEFORE the vertex is eliminated. The head group below
# consumes that ordered list and emits, PER FACE:
#
#   1. a SKIP decision (2 logits). ``skip == FACE_SKIP`` -> the path stays
#      EXACT: the env emits ``(None, None, None)`` for it and the three slot
#      heads are masked out of the joint log-prob, the entropy AND the arity.
#   2. otherwise three slot decisions in the order ``lhs -> rhs -> res``
#      (the local path is ``res = op(lhs, rhs)``: ``lhs`` transforms the
#      in-edge Jacobian, ``rhs`` the out-edge Jacobian, ``res`` the
#      contraction result). Each slot is ONE micro-action emitted by the
#      EXISTING :class:`MicroActionHead` — the deliberate
#      OpType/AxisPointer/PrimeExponent/CompressKind/QuantDtype factorization
#      is reused verbatim, not re-invented — and is autoregressive on the
#      previous slot's decision via :func:`_slot_descriptor`.
#
# The face loop is STATICALLY UNROLLED to ``max_faces`` with no early
# termination: faces past the real count are padding, flagged by
# ``FaceFeatures.valid_mask``, and contribute zero to log-prob / entropy /
# arity.
#
# MIRROR INVARIANT (the PPO ratio-correctness property)
# -----------------------------------------------------
# :meth:`FacePathPolicy.sample` and :meth:`FacePathPolicy.evaluate` are the
# SAME code path — :meth:`FacePathPolicy._run` — parameterised by a PYTHON
# bool ``sampling`` that is resolved at trace time. The masks, the gating, the
# conditioning contexts and the emission order are therefore identical by
# construction rather than by convention. Two further details make the two
# paths agree to floating-point identity, not merely to within a tolerance:
#
#   * every decision is CANONICALIZED (pad face -> FACE_SKIP; skipped /
#     padded slot -> the null OP_END micro-action) BEFORE its log-prob is
#     computed, so sample time scores exactly the action that evaluate later
#     reads back from the trajectory;
#   * the autoregressive descriptor fed to the next slot is derived from that
#     same canonical action, so the conditioning context cannot drift either.
#
# A mismatch here silently biases the PPO ratio — it is the exact bug class
# that made the legacy path unusable — so any change to one branch MUST be a
# change to the shared body, never to one side of the ``if sampling``.

# Slots of one local path, in emission order. ``res = op(lhs, rhs)``.
FACE_SLOT_LHS = 0
FACE_SLOT_RHS = 1
FACE_SLOT_RES = 2
NUM_FACE_SLOTS = 3
FACE_SLOT_NAMES: tuple[str, ...] = ("lhs", "rhs", "res")

# SKIP head outcomes.
FACE_APPROX = 0   # approximate this local path (the three slot heads fire)
FACE_SKIP = 1     # leave this local path EXACT -> (None, None, None)
NUM_FACE_DECISIONS = 2

# Width of the per-face numeric feature block the env supplies. Layout (kept
# in sync with ``alphagrad.approx.env.build_face_features``):
#   0  log1p(#elements of the in-edge Jacobian)
#   1  log1p(#elements of the out-edge Jacobian)
#   2  in-edge ndim
#   3  out-edge ndim
#   4  log1p(#elements in  x  #elements out)  — contraction-cost proxy
#   5  face position / max_faces              — the elimination visit order
#   6  1.0 if the in-edge var is a jaxpr invar
#   7  1.0 if the out-edge var is a jaxpr outvar
FACE_FEAT_DIM = 8

# Width of the autoregressive descriptor summarising one emitted slot.
FACE_SLOT_DESC_DIM = NUM_OPS + NUM_COMPRESS_KINDS + NUM_QUANT_DTYPES + 3


class FaceFeatures(NamedTuple):
    """The vertex's enumerated local paths, padded to a static cap.

    One row per face, in the order ``graphax.faces_of`` returns (which is the
    order the elimination visits them). ``in_edge`` / ``out_edge`` carry the
    stable var indices that form the graphax face key
    ``(vidx[in_edge], vidx[out_edge])``; they are passed through the policy
    untouched so the env can rebuild the ``face_transforms`` dict keys.
    """

    in_edge: jax.Array       # (F,) int32 — stable var index of the in-edge
    out_edge: jax.Array      # (F,) int32 — stable var index of the out-edge
    feats: jax.Array         # (F, FACE_FEAT_DIM) float32
    valid_mask: jax.Array    # (F,) float32 — 1.0 for a real face, 0.0 for pad


class FacePathAction(NamedTuple):
    """One vertex's full per-local-path action.

    ``skip`` is ``(F,)`` int32 over ``{FACE_APPROX, FACE_SKIP}``. ``slots``
    is a :class:`MicroAction` whose every field carries a leading
    ``(F, NUM_FACE_SLOTS)`` — one micro-action per (face, slot) pair, in the
    order lhs, rhs, res. Slots of a skipped or padded face are canonicalized
    to the null ``OP_END`` action.
    """

    skip: jax.Array          # (F,) int32
    slots: MicroAction       # fields (F, NUM_FACE_SLOTS, ...)


class FaceSkipHead(eqx.Module):
    """Binary categorical: approximate this local path, or keep it EXACT.

    Fires once per enumerated face, before the slot heads, conditioned on the
    pooled axis-set summary (which already carries the encoder's vertex
    context) plus the face's own embedded features. Padded faces are forced to
    :data:`FACE_SKIP` and gated out of the joint entirely, so the head is left
    unmasked — adding a mask here would be a second place for the sample and
    evaluate paths to drift.
    """

    proj: eqx.nn.Linear

    def __init__(self, embd_dim: int, *, key):
        self.proj = eqx.nn.Linear(embd_dim, NUM_FACE_DECISIONS, key=key)

    def __call__(self, context: jax.Array) -> jax.Array:
        return jnn.softmax(self.proj(context), axis=-1)


def _null_micro_action_like(action: MicroAction) -> MicroAction:
    """The canonical "this slot emits nothing" micro-action.

    ``op_type = OP_END`` (not 0 — that is ``OP_DIAG`` and would read as a real
    approximation in a stored trajectory), every payload field zeroed. Under
    ``ALPHAGRAD_SUBSTEP_NO_END`` END is an ILLEGAL op, so its masked
    probability is ~0 and its log-prob ~``log(1e-8)`` — finite, and multiplied
    by the zero gate in both the sample and the evaluate path, so it can
    neither NaN nor leak into the PPO ratio.
    """
    return MicroAction(
        op_type=jnp.full_like(action.op_type, OP_END),
        i=jnp.zeros_like(action.i),
        j=jnp.zeros_like(action.j),
        exponents=jnp.zeros_like(action.exponents),
        factor=jnp.zeros_like(action.factor),
        compress_kind=jnp.zeros_like(action.compress_kind),
        quant_dtype=jnp.zeros_like(action.quant_dtype),
    )


def _select_micro_action(
    emit: jax.Array, action: MicroAction, null: MicroAction,
) -> MicroAction:
    """``action`` where ``emit`` else ``null``, field by field."""
    return MicroAction(
        *(jnp.where(emit, a, n) for a, n in zip(action, null))
    )


def _slot_descriptor(action: MicroAction, num_axes: int) -> jax.Array:
    """``(FACE_SLOT_DESC_DIM,)`` summary of one emitted slot.

    Feeds the NEXT slot's conditioning context so ``rhs`` sees what ``lhs``
    did and ``res`` sees both. Derived from the CANONICAL action (post
    skip/pad canonicalization) so the sample and evaluate paths build the
    identical context. ``num_axes`` is static (the axis-token count), used
    only to normalise the pointer indices into [0, 1].
    """
    denom = float(max(int(num_axes), 1))
    scalars = jnp.stack([
        jnp.log1p(jnp.maximum(action.factor, 0).astype(jnp.float32)),
        action.i.astype(jnp.float32) / denom,
        action.j.astype(jnp.float32) / denom,
    ])
    return jnp.concatenate([
        jnn.one_hot(action.op_type, NUM_OPS, dtype=jnp.float32),
        jnn.one_hot(action.compress_kind, NUM_COMPRESS_KINDS, dtype=jnp.float32),
        jnn.one_hot(action.quant_dtype, NUM_QUANT_DTYPES, dtype=jnp.float32),
        scalars,
    ])


class FacePathHead(eqx.Module):
    """The per-face head group: SKIP + three autoregressive slot emitters.

    Deliberately thin — it owns only the face/slot CONDITIONING (projecting
    the face features, embedding the slot identity, projecting the previous
    slot's descriptor) and the SKIP categorical. The actual approximation
    choice for each of lhs / rhs / res is emitted by the shared
    :class:`MicroActionHead`, so the prime-exponent trick and every legality
    mask carry over unchanged.

    One ``MicroActionHead`` is shared across the three slots rather than three
    separate copies: the slots differ in WHICH tensor they transform, not in
    the space of transforms available, and the slot embedding gives the shared
    head that distinction with ``3 * embd_dim`` parameters instead of three
    full head stacks.
    """

    face_proj: eqx.nn.Linear
    slot_embedding: eqx.nn.Embedding
    prev_proj: eqx.nn.Linear
    skip_head: FaceSkipHead
    micro: MicroActionHead

    embd_dim: int = eqx.field(static=True)

    def __init__(self, embd_dim: int, *, key):
        self.embd_dim = embd_dim
        keys = jrand.split(key, 5)
        self.face_proj = eqx.nn.Linear(FACE_FEAT_DIM, embd_dim, key=keys[0])
        self.slot_embedding = eqx.nn.Embedding(
            NUM_FACE_SLOTS, embd_dim, key=keys[1],
        )
        self.prev_proj = eqx.nn.Linear(
            FACE_SLOT_DESC_DIM, embd_dim, key=keys[2],
        )
        self.skip_head = FaceSkipHead(embd_dim, key=keys[3])
        self.micro = MicroActionHead(embd_dim, key=keys[4])


class FacePathPolicy(eqx.Module):
    """Per-local-path policy over one vertex's faces.

    Usage (env side, after the pointer net picked ``vertex``)::

        keys  = graphax.faces_of(graph, tgraph, vertex, jaxpr)
        feats = build_face_features(...)          # padded to max_faces
        act, logp, ent, arity, *dists = policy.sample(
            vertex_context, axis_features, feats, tables, key)
        ft = build_face_transforms(keys, act)     # {face_key: (lhs, rhs, res)}
        incr.eliminate(vertex, face_transforms=ft)

    The axis-token encoder runs ONCE per vertex, outside the face loop: the
    axis features do not evolve between faces (each face transforms a
    DIFFERENT pair of edge Jacobians, not a single tensor being refined), so
    a per-face re-encode of the same input would be pure cost. See the
    module note on the per-path re-encode that is NOT done here.

    Axis-index convention: the ``i`` / ``j`` a slot's micro-action emits are
    PHYSICAL axis indices into that slot's edge SparseTensor, under the same
    ``(out axes..., primal axes...)`` layout the per-vertex axis tokens
    already use — so the env passes them straight through to
    ``Diag(i, j, factor)`` / ``Compress(axes=(i,))`` with no remapping.
    """

    encoder: AxisSetEncoder
    head: FacePathHead

    max_faces: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)

    def __init__(
        self, embd_dim: int, num_heads: int, max_faces: int,
        num_encoder_layers: int = 1, max_groups: int = 16, *, key,
        use_group_embedding: bool = False,
    ):
        self.embd_dim = embd_dim
        self.max_faces = int(max_faces)
        keys = jrand.split(key, 2)
        self.encoder = AxisSetEncoder(
            embd_dim, num_heads, num_layers=num_encoder_layers,
            max_groups=max_groups, key=keys[0],
            use_group_embedding=use_group_embedding,
        )
        self.head = FacePathHead(embd_dim, key=keys[1])

    # -- one slot (lhs / rhs / res) of one face ---------------------------

    def _slot_step(
        self, carry, slot_input, *,
        axis_tokens, summary, face_emb, features, masks, tables, emit,
        sampling: bool,
    ):
        """One slot. ``sampling`` is a PYTHON bool — the branch is resolved at
        trace time, so sampling and evaluation share every other line."""
        prev_desc, slot_idx = carry
        op_legal, i_diag, i_compress, j_diag, i_coupled = masks

        ctx = (
            summary
            + face_emb
            + self.head.slot_embedding(slot_idx)
            + self.head.prev_proj(prev_desc)
        )

        if sampling:
            sampled = self.head.micro.sample_step(
                ctx, axis_tokens, features.size,
                op_legal, i_diag, i_compress, j_diag,
                tables, slot_input, i_coupled=i_coupled,
            )[0]
            # Canonicalize BEFORE scoring: a skipped / padded slot records the
            # null action, and it is that action whose log-prob we accumulate,
            # so evaluate (which reads the null back) scores the same thing.
            action = _select_micro_action(
                emit, sampled, _null_micro_action_like(sampled),
            )
        else:
            action = slot_input

        (
            log_p, ent, arity,
            op_dist, i_dist, j_dist, exp_dists, kind_dist, quant_dist,
        ) = self.head.micro.log_prob_step(
            action, ctx, axis_tokens, features.size,
            op_legal, i_diag, i_compress, j_diag, tables,
            i_coupled=i_coupled,
        )

        new_desc = _slot_descriptor(action, features.size.shape[0])
        return (
            (new_desc, slot_idx + 1),
            (
                action, log_p, ent, arity,
                op_dist, i_dist, j_dist, exp_dists, kind_dist, quant_dist,
            ),
        )

    # -- one face (SKIP + three slots) ------------------------------------

    def _face_step(
        self, carry, face_input, *,
        axis_tokens, summary, features, masks, tables, sampling: bool,
    ):
        f_idx = carry
        feat_row, valid, extra = face_input

        face_emb = self.head.face_proj(feat_row)
        skip_dist = self.head.skip_head(summary + face_emb)

        if sampling:
            k_skip, k_slots = jrand.split(extra)
            sampled_skip = distrax.Categorical(probs=skip_dist).sample(
                seed=k_skip,
            ).astype(jnp.int32)
            # A padded face is canonically SKIP, decided before scoring so the
            # recorded action and its log-prob agree with evaluate's readback.
            skip = jnp.where(
                valid > 0.5, sampled_skip, jnp.int32(FACE_SKIP),
            ).astype(jnp.int32)
            slot_inputs = jrand.split(k_slots, NUM_FACE_SLOTS)
        else:
            skip = extra.skip.astype(jnp.int32)
            slot_inputs = extra.slots

        log_p_skip = jnp.log(skip_dist[skip] + 1e-8)
        ent_skip = -jnp.sum(skip_dist * jnp.log(skip_dist + 1e-8))

        # The slot heads fire iff this face is real AND the policy chose to
        # approximate it. Same expression on both paths.
        emit = (skip == FACE_APPROX) & (valid > 0.5)

        def slot_fn(c, si):
            return self._slot_step(
                c, si,
                axis_tokens=axis_tokens, summary=summary, face_emb=face_emb,
                features=features, masks=masks, tables=tables, emit=emit,
                sampling=sampling,
            )

        (
            _,
            (
                slot_actions, slot_logps, slot_ents, slot_arities,
                op_dists, i_dists, j_dists, exp_dists, kind_dists, quant_dists,
            ),
        ) = lax.scan(
            slot_fn,
            (jnp.zeros(FACE_SLOT_DESC_DIM, dtype=jnp.float32),
             jnp.array(0, dtype=jnp.int32)),
            slot_inputs,
        )

        face_active = (valid > 0.5).astype(jnp.float32)
        emit_f = emit.astype(jnp.float32)

        log_p = log_p_skip * face_active + jnp.sum(slot_logps) * emit_f
        entropy = ent_skip * face_active + jnp.sum(slot_ents) * emit_f
        # ARITY: the number of categoricals this face actually emitted. The
        # SKIP decision counts 1 for every REAL face (it is emitted whether or
        # not the path is then approximated); the three slots contribute their
        # own arities — which already count op + i + j + one per real prime +
        # kind + dtype — only when they fire. Without folding SKIP and the
        # slots in here, a face that approximates would collect a larger
        # entropy bonus than one that skips purely for emitting more
        # categoricals, and a vertex with more local paths would out-earn a
        # smaller one for the same reason.
        arity = face_active + jnp.sum(slot_arities) * emit_f

        action = FacePathAction(skip=skip, slots=slot_actions)
        return (
            f_idx + 1,
            (
                action, log_p, entropy, arity,
                skip_dist, op_dists, i_dists, j_dists, exp_dists,
                kind_dists, quant_dists,
            ),
        )

    # -- shared driver -----------------------------------------------------

    def _run(
        self, vertex_context, features: AxisTokenFeatures,
        face_features: FaceFeatures, tables: FactorTables,
        *, key=None, actions: FacePathAction | None = None,
    ):
        """THE single implementation behind :meth:`sample` and
        :meth:`evaluate`. See the module-level MIRROR INVARIANT note."""
        sampling = actions is None
        if sampling and key is None:
            raise ValueError("sample() requires a PRNG key.")
        if not sampling and key is not None:
            raise ValueError("evaluate() must not be given a PRNG key.")
        if face_features.valid_mask.shape[0] != self.max_faces:
            raise ValueError(
                f"face_features has {face_features.valid_mask.shape[0]} rows "
                f"but this policy is built for max_faces={self.max_faces}; "
                "the face loop is statically unrolled, so the caller must pad "
                "to exactly that many rows."
            )

        # ONE encode per vertex — the axis features are constant across the
        # face loop (see the class docstring / the re-encode caveat).
        axis_tokens, summary = self.encoder(features, vertex_context)
        masks = (
            _compute_op_legality(features),
            *_compute_axis_masks(features),
        )

        if sampling:
            extra = jrand.split(key, self.max_faces)
        else:
            extra = actions
        xs = (face_features.feats, face_features.valid_mask, extra)

        def step_fn(carry, x):
            return self._face_step(
                carry, x,
                axis_tokens=axis_tokens, summary=summary, features=features,
                masks=masks, tables=tables, sampling=sampling,
            )

        _, out = lax.scan(step_fn, jnp.array(0, dtype=jnp.int32), xs)
        (
            face_actions, logps, ents, arities,
            skip_dists, op_dists, i_dists, j_dists, exp_dists,
            kind_dists, quant_dists,
        ) = out
        return (
            face_actions,
            jnp.sum(logps), jnp.sum(ents), jnp.sum(arities),
            skip_dists, op_dists, i_dists, j_dists, exp_dists,
            kind_dists, quant_dists,
        )

    def sample(
        self, vertex_context, init_features: AxisTokenFeatures,
        face_features: FaceFeatures, tables: FactorTables, key,
    ):
        """Emit a full per-local-path action for one vertex.

        Returns ``(action, log_prob, entropy, arity, skip_dists, op_dists,
        i_dists, j_dists, exp_dists, kind_dists, quant_dists)``. The dists are
        the rollout-time ("old") policy snapshot for the PPO ratio / KL, with
        a leading ``(max_faces,)`` and — for the slot heads — a second
        ``(NUM_FACE_SLOTS,)``.
        """
        return self._run(
            vertex_context, init_features, face_features, tables, key=key,
        )

    def evaluate(
        self, vertex_context, init_features: AxisTokenFeatures,
        face_features: FaceFeatures, tables: FactorTables,
        actions: FacePathAction,
    ):
        """Re-score a stored :class:`FacePathAction` under the CURRENT policy.

        Returns ``(log_prob, entropy, arity, skip_dists, op_dists, i_dists,
        j_dists, exp_dists, kind_dists, quant_dists)`` — the same quantities
        :meth:`sample` returns, minus the action. For an unchanged policy the
        log-prob is bit-identical to the one ``sample`` returned; that
        equality is what the PPO ratio is built on.
        """
        _, log_p, ent, arity, *dists = self._run(
            vertex_context, init_features, face_features, tables,
            actions=actions,
        )
        return (log_p, ent, arity, *dists)


__all__ = [
    "OP_DIAG", "OP_COMPRESS", "OP_QUANT", "OP_END", "NUM_OPS",
    "MAX_PRIMES", "MAX_EXPONENT",
    "COMPRESS_KINDS", "NUM_COMPRESS_KINDS",
    "QUANT_DTYPES", "NUM_QUANT_DTYPES",
    "AXIS_TAG_BITS", "TAG_IS_LOGICAL", "TAG_IS_COMPRESSED", "TAG_IN_DIAG_GROUP",
    "AxisTokenFeatures",
    "FactorTables",
    "precompute_factor_tables",
    "AxisSetEncoder",
    "OpTypeHead",
    "AxisPointerHead",
    "PrimeExponentHead",
    "CompressKindHead",
    "QuantDtypeHead",
    "MicroAction",
    "MicroActionHead",
    "MicroActionPolicy",
    "factorize",
    "exponents_to_factor",
    # per-local-path (per-face) approximation
    "FACE_SLOT_LHS", "FACE_SLOT_RHS", "FACE_SLOT_RES",
    "NUM_FACE_SLOTS", "FACE_SLOT_NAMES",
    "FACE_APPROX", "FACE_SKIP", "NUM_FACE_DECISIONS",
    "FACE_FEAT_DIM", "FACE_SLOT_DESC_DIM",
    "FaceFeatures",
    "FacePathAction",
    "FaceSkipHead",
    "FacePathHead",
    "FacePathPolicy",
]
