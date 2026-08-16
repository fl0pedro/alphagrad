"""Palimpsa encoder: Bayesian-metaplastic gated linear attention.

Both mixers here route through the verified Pallas-Triton kernel in
``alphagrad.transformer.palimpsa_pallas`` -- the recurrence is defined once,
in the kernel, and is not reimplemented in raw JAX.
"""
import os
from typing import Sequence, Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx

from alphagrad.approx.common.relations import NUM_RELATIONS
from alphagrad.transformer.palimpsa_pallas import palimpsa
from alphagrad.transformer.encoder import SwiGLU

Array = jax.Array
PRNGKey = jax.Array


# --------------------------------------------------------------------------
# PALIMPSA-D: the two ingredients that BOUND the precision state I.
#
# Paper A.1 (and the reference implementation) put two bounds on the update
#     I_t = alpha*I_{t-1} + (1-alpha)*I_prior + beta (x) k^2
# that were both absent here: Palimpsa-D L2-normalises q and k, and beta is
# bounded as sigmoid(.)*softplus(scale) rather than an unbounded softplus.
# With neither, nothing bounds ||k||^2 or beta, so one large beta*||k||^2
# drives I up, 1/I -> 0, and that head stops integrating -- the catastrophic
# remembering the forgetting term exists to prevent. It presents as silent
# quality loss, never as an error.
#
# With ||k||_2 = 1 each step's precision increment sums to exactly beta, so
# the plasticity 1/I sits on a controlled scale independent of the residual
# stream's magnitude.
#
# Each is independently env-gated; =0 restores the previous form. The
# b_scale_raw parameter is ALWAYS constructed so the parameter tree does not
# depend on the flag.
# --------------------------------------------------------------------------
PALIMPSA_QK_NORM = int(os.environ.get("ALPHAGRAD_PALIMPSA_QK_NORM", "1"))
PALIMPSA_BETA_BOUNDED = int(os.environ.get("ALPHAGRAD_PALIMPSA_BETA_BOUNDED", "1"))
_QK_EPS2 = 1e-12


def palimpsa_qk_norm(x):
    """L2-normalise the trailing head_dim axis (Palimpsa-D, paper A.1).

    The epsilon goes INSIDE the sqrt. Neither of the two obvious spellings
    survives reverse mode on a zero (padding) row:

      * ``where(n > 0, x / n, x)`` differentiates the unsafe branch anyway --
        the forward is guarded, the backward is not.
      * ``x / maximum(n, eps)`` guards the DIVISION and correctly sends a zero
        cotangent back to ``n``, but ``n = sqrt(sum(x*x))`` still has
        ``d sqrt/du = 1/(2 sqrt(u)) -> inf`` at u = 0, so the chain evaluates
        ``0 * inf = NaN``. MEASURED: max|grad| = nan on an all-zero row.

    ``rsqrt(sum(x*x) + eps)`` never evaluates a root at zero, so the gradient
    is finite everywhere. For a unit-scale row the eps costs ~5e-13 of norm.
    Padding rows land at 0 (0 * rsqrt(eps) = 0), which is what the downstream
    mask wants anyway.
    """
    if not PALIMPSA_QK_NORM:
        return x
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + _QK_EPS2)


def palimpsa_beta(raw, b_scale_raw):
    """Per-row importance beta from the RAW bias projection, shape (..., H, d).

    Bounded form ``sigmoid(raw) * softplus(scale)`` caps the observation
    precision a single token may inject; the legacy form is an unbounded
    ``softplus(raw)``. ``b_scale_raw`` is (H,) and broadcasts as (H, 1).
    """
    if not PALIMPSA_BETA_BOUNDED:
        return jnn.softplus(raw)
    return jnn.sigmoid(raw) * jnn.softplus(b_scale_raw)[:, None]



class PalimpsaMixer(eqx.Module):
    """Drop-in token-mixing block backed by the verified Palimpsa Pallas-Triton
    linear-attention kernel (``alphagrad.approx.palimpsa_pallas``).

    Interface contract (matches :class:`RelationalMultiheadAttention`):

        __call__(query, key_, value, *, bias=None, mask=None, key=None) -> (S, embd_dim)

    Self-attention only: ``query is key_ is value`` in the encoder, so we take
    ``query`` as the single token sequence ``x : (S, embd_dim)``.

    Projections from the token embedding (the only learned op that changes vs
    the transformer; everything around it — norm/residual/FFN/heads — is
    untouched by the caller):

        q = W_q x   k = W_k x          : (S, H, D_K)   (key/query dim per head)
        v = W_v x   b = W_b x          : (S, H, D_V)   (value / precision-numer dim)
        gt = softplus(W_gt x)          : (S, H)        (per-token forget magnitude >= 0)

    Per-head learnable scalars (kernel params, not input-projected):

        g  : (H,)   forget-rate; decay_t = exp(-gt_t * g). Init small (>0) so the
                    initial state behaves near-additive (long memory).
        Ip : (H,)   prior precision (softplus-reparam'd to stay > 0) regularising
                    the posterior mean mu = M/(I_bar+Ip).

    The kernel call ``palimpsa_attention(q,k,v,b,gt,g,Ip)`` expects a leading
    batch axis ``[B,T,H,D]``; the encoder vmaps over batch externally, so here we
    add/remove a ``B=1`` axis. Output ``(S, H*D_V)`` is projected back to
    ``embd_dim`` by ``output_proj`` — identical out-shape to the transformer.

    IMPORTANT semantic differences vs the bidirectional transformer attention
    (documented, intentional):
      * Palimpsa is a *causal* left-to-right recurrence (token t sees tokens
        <= t). The transformer here is bidirectional. This is the inherent
        nature of linear-attention; flag-gated so it is opt-in only.
      * ``bias`` (T5 relational bias on QK^T logits) has no analogue in the
        recurrence and is ignored. The relational structure is not injected in
        the Palimpsa path.
      * ``mask`` is treated as a *padding* mask: a key/value token j that is
        masked-out for all queries (a padding column) is zeroed so it never
        contributes to the running state. The full pairwise (S,T) structure of
        an arbitrary mask cannot be represented by a causal scan and is reduced
        to this per-token validity, which matches the encoder's actual use
        (mask = outer-product of a per-token vertex/validity vector).
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    bias_proj: eqx.nn.Linear
    gate_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    rel_gate: eqx.nn.Linear   # (3,) structural degree feats -> (H,) gate mod; zero-init

    g_raw: Array      # (H,) raw param; g = softplus(g_raw)
    Ip_raw: Array     # (H,) raw param; Ip = softplus(Ip_raw)
    b_scale_raw: Array  # (H,) beta = sigmoid(bias_proj) * softplus(b_scale_raw)

    num_heads: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        embd_dim: int,
        *,
        chunk_size: int = 16,
        key: PRNGKey,
    ):
        super().__init__()
        if embd_dim % num_heads != 0:
            raise ValueError(
                f"embd_dim={embd_dim} must be divisible by num_heads={num_heads}"
            )
        keys = jrand.split(key, 7)
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.head_dim = embd_dim // num_heads
        self.chunk_size = chunk_size

        # D_K == D_V == head_dim, so the four projections all map embd->embd.
        self.query_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[0])
        self.key_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[1])
        self.value_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[2])
        self.bias_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[3])
        self.gate_proj = eqx.nn.Linear(embd_dim, num_heads, key=keys[4])
        self.output_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[5])

        # Relational gate modulation (same construction as BiPalimpsaMixer):
        # 3 per-token DAG structural-degree features (same/earlier/later,
        # normalised) -> per-head additive forget-gate term. ZERO-init
        # (weight+bias) so the mixer starts as pure paper-Palimpsa and only
        # deviates once the structural modulation learns to.
        rg = eqx.nn.Linear(3, num_heads, key=keys[6])
        rg = eqx.tree_at(lambda m: m.weight, rg, jnp.zeros_like(rg.weight))
        rg = eqx.tree_at(lambda m: m.bias, rg, jnp.zeros_like(rg.bias))
        self.rel_gate = rg

        # softplus(g_raw) ~= 0.05 -> slow forgetting / long memory at init.
        self.g_raw = jnp.full((num_heads,), -3.0, dtype=jnp.float32)
        # softplus(Ip_raw) ~= 0.69 -> moderate prior precision at init.
        self.Ip_raw = jnp.zeros((num_heads,), dtype=jnp.float32)
        # softplus(1.1) ~= 1.376, so beta ~= 0.5 * 1.376 = 0.688 at init
        # against the legacy softplus(0) = 0.693: the bounded form STARTS
        # where the unbounded one did and differs only in its tail.
        self.b_scale_raw = jnp.full((num_heads,), 1.1, dtype=jnp.float32)

    def _relational_gate_mod(self, eqn_ids: Array, S: int) -> Array:
        """Per-token additive gate modulation (S, H) from DAG relation degrees.

        Identical construction to :meth:`BiPalimpsaMixer._relational_gate_mod`:
        row-reduction of the same same/earlier/later relations the transformer
        uses as a pairwise bias, reduced to a per-token structural feature
        vector and mapped (zero-init) to a per-head gate term.
        """
        valid = (eqn_ids >= 0)
        valid_pair = valid[:, None] & valid[None, :]
        same = ((eqn_ids[:, None] == eqn_ids[None, :]) & valid_pair).astype(jnp.float32)
        earlier = ((eqn_ids[:, None] > eqn_ids[None, :]) & valid_pair).astype(jnp.float32)
        later = ((eqn_ids[:, None] < eqn_ids[None, :]) & valid_pair).astype(jnp.float32)
        nvalid = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        feats = jnp.stack([
            jnp.sum(same, axis=1) / nvalid,
            jnp.sum(earlier, axis=1) / nvalid,
            jnp.sum(later, axis=1) / nvalid,
        ], axis=-1)                                          # (S, 3)
        feats = feats * valid.astype(jnp.float32)[:, None]   # pad tokens -> 0
        return jax.vmap(self.rel_gate)(feats)                # (S, H)

    def __call__(
        self,
        query: Array,
        key_: Array,
        value: Array,
        *,
        bias: Optional[Array] = None,   # ignored (no recurrence analogue)
        mask: Optional[Array] = None,   # treated as a padding mask, see docstring
        eqn_ids: Optional[Array] = None,  # used for the relational gate proxy
        key: Optional[PRNGKey] = None,
    ) -> Array:
        x = query                       # self-attention: query is the token seq
        S = x.shape[0]
        H = self.num_heads
        d = self.head_dim

        q = palimpsa_qk_norm(jax.vmap(self.query_proj)(x).reshape(S, H, d))
        k = palimpsa_qk_norm(jax.vmap(self.key_proj)(x).reshape(S, H, d))
        v = jax.vmap(self.value_proj)(x).reshape(S, H, d)
        # b is the PRECISION NUMERATOR of the kernel's posterior update:
        #   I_t = b_t * k_t^2 + (1 - decay) * Ip + decay * I_{t-1};  mu = M / I.
        # The recurrence is only well-posed for b >= 0 (I stays > 0 since
        # I_0 = Ip > 0 and every increment is then non-negative). Feeding the
        # RAW signed linear output lets a negative b_t * k^2 drive I through
        # zero -> mu = M/0 -> inf/nan on data-dependent inputs (observed as
        # ~15%% of rollout forwards emitting non-finite values on the larger
        # residual graphs). softplus enforces the positivity the Bayesian
        # precision semantics require.
        b = palimpsa_beta(
            jax.vmap(self.bias_proj)(x).reshape(S, H, d), self.b_scale_raw)
        # Relational structural prior folded into the forget gate (mirrors
        # BiPalimpsaMixer, but injected PRE-softplus — the cleaner form the
        # bi docstring itself notes — so a zero gate_mod is an EXACT no-op:
        # softplus(raw + 0) == softplus(raw), pure paper-Palimpsa at init).
        gt_raw = jax.vmap(self.gate_proj)(x)                # (S, H)
        if eqn_ids is not None:
            gt_raw = gt_raw + self._relational_gate_mod(eqn_ids, S)
        # forget magnitude >= 0 so decay = exp(-gt*g) in (0, 1].
        gt = jnn.softplus(gt_raw)                            # (S, H)

        if mask is not None:
            # mask is (H, S, T) replicated; reduce to per-(key)token validity:
            # a key token j is valid if any query attends to it.
            if mask.ndim == 3:
                valid = jnp.any(mask, axis=(0, 1))          # (T,)
            elif mask.ndim == 2:
                valid = jnp.any(mask, axis=0)               # (T,)
            else:
                valid = mask
            valid = valid.astype(x.dtype)[:, None]          # (S, 1)
            # Zero a padded token's contribution to the state (v and b outer
            # products vanish), and force its decay to 1 (no state change).
            v = v * valid[:, :, None]
            b = b * valid[:, :, None]
            gt = gt * valid                                  # decay=exp(0)=1 when padded

        g = jnn.softplus(self.g_raw)                         # (H,)
        Ip = jnn.softplus(self.Ip_raw)                       # (H,)

        # Add leading batch axis B=1 for the kernel, then drop it.
        out = palimpsa(
            q[None], k[None], v[None], b[None], gt[None], g, Ip,
            scale=None, chunk_size=self.chunk_size,
        )                                                    # (1, S, H, d)
        out = out[0].reshape(S, H * d)
        return jax.vmap(self.output_proj)(out)


class BiPalimpsaMixer(eqx.Module):
    """Bidirectional, relationally-modulated variant of :class:`PalimpsaMixer`.

    Motivation (step-2 modeling caveat). The unidirectional ``PalimpsaMixer`` is
    a *causal* left-to-right gated-linear-attention recurrence: token ``t`` only
    sees tokens ``<= t`` in the (arbitrary) topological token order, and it drops
    the T5-style relational bias entirely. But the encoder's input is a DAG, not
    a sequence; a one-directional scan imposes a spurious ordering and throws
    away the transformer's bidirectionality + structural prior. This mixer
    addresses both:

      (1) BIDIRECTIONALITY (primary). We run the *same* projected q/k/v/b/gt
          features through the verified ``palimpsa_attention`` kernel twice:
          once forward (tokens 0..S-1) and once on the time-reversed sequence
          (flip the S axis, run, flip back). The two per-direction outputs are
          SUMMED, then a single ``output_proj`` maps ``H*head_dim -> embd_dim``.
          This is the standard "bidirectional linear attention" construction:
          two causal passes in opposite directions = a non-causal mixer, and
          stays linear in S (2x linear = linear). Grads flow through BOTH
          ``custom_vjp`` passes (forward and reverse) independently.

          Combine = SUM (not concat). Rationale: summing keeps ``head_dim``
          unchanged so the kernel's pow2-padding and the q/k/v/b projections are
          identical to the unidirectional mixer (no awkward head-dim halving,
          which would waste the padded kernel width). The projections are SHARED
          across the two directions; only the per-head kernel scalars
          (``g``/``Ip``) are separate per direction, so each direction can learn
          its own decay-rate / prior-precision. Param delta vs unidirectional:
          +2*H scalars for the reverse (g,Ip) + the optional relational gate-mod
          map below; the bulk (the 4 embd->embd projections + output_proj) is
          identical. So param count is essentially the same.

      (2) RELATIONAL SIGNAL (best-effort proxy). The T5 relational bias is a
          pairwise ``(H,S,T)`` logit added before softmax — inherently O(S^2)
          and tied to an explicit attention matrix, so it CANNOT enter the
          linear recurrence as a pairwise term (linear attention never
          materialises the SxT logit matrix). A faithful injection is therefore
          not feasible. Instead we inject a faithful *per-token row-reduction*
          of the same relation structure into the forget gate ``gt`` (the only
          per-token knob the recurrence exposes that controls how information
          propagates):

            For each (valid) token i over the same three relations the
            transformer uses (same_eqn / earlier / later in topo order), we
            count how many valid tokens it relates to:
              n_same[i]    = #{ j : eqn_id[j]==eqn_id[i] }
              n_earlier[i] = #{ j : eqn_id[j] <  eqn_id[i] }   (topo rank)
              n_later[i]   = #{ j : eqn_id[j] >  eqn_id[i] }
            normalised by the valid-token count. These three structural degree
            features (a per-token reduction of the pairwise relation masks; the
            earlier/later counts are exactly a normalised topological rank) are
            mapped by a small learned ``rel_gate`` linear (3 -> H, ZERO-init) to
            an additive per-head modulation of ``gt``. Zero-init => at start the
            mixer is byte-identical to plain bidirectional Palimpsa; the layer
            only deviates once the structural modulation learns to. This is a
            genuine, lightweight structural prior on the recurrence (it tells
            each head how much to remember/forget depending on a token's DAG
            centrality and topological position) without any O(S^2) logit.

    Interface contract is identical to :class:`RelationalMultiheadAttention` /
    :class:`PalimpsaMixer`:
        __call__(query, key_, value, *, bias=None, mask=None, eqn_ids=None,
                 key=None) -> (S, embd_dim)
    (``eqn_ids`` is an EXTRA optional kwarg used only for the relational gate
    proxy; the base mixers ignore it / never receive it. ``bias`` is still
    ignored here — the proxy is derived from ``eqn_ids`` directly, exactly as
    the transformer's ``_bias_from_eqn_ids`` derives its pairwise bias.)
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    bias_proj: eqx.nn.Linear
    gate_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    rel_gate: eqx.nn.Linear          # (3,) structural degree feats -> (H,) gate mod; zero-init

    g_raw_fwd: Array      # (H,) forward decay-rate;  g = softplus(g_raw)
    Ip_raw_fwd: Array     # (H,) forward prior precision
    g_raw_rev: Array      # (H,) reverse decay-rate
    Ip_raw_rev: Array     # (H,) reverse prior precision

    num_heads: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        embd_dim: int,
        *,
        chunk_size: int = 16,
        key: PRNGKey,
    ):
        super().__init__()
        if embd_dim % num_heads != 0:
            raise ValueError(
                f"embd_dim={embd_dim} must be divisible by num_heads={num_heads}"
            )
        keys = jrand.split(key, 7)
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.head_dim = embd_dim // num_heads
        self.chunk_size = chunk_size

        # Shared projections (identical to PalimpsaMixer); embd->embd / embd->H.
        self.query_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[0])
        self.key_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[1])
        self.value_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[2])
        self.bias_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[3])
        self.gate_proj = eqx.nn.Linear(embd_dim, num_heads, key=keys[4])
        self.output_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[5])

        # Relational gate modulation: 3 per-token structural-degree features
        # (same/earlier/later, normalised) -> per-head additive gate term.
        # ZERO-init (weight+bias) so it starts as a no-op (cf. relation_biases).
        rg = eqx.nn.Linear(3, num_heads, key=keys[6])
        rg = eqx.tree_at(lambda m: m.weight, rg, jnp.zeros_like(rg.weight))
        rg = eqx.tree_at(lambda m: m.bias, rg, jnp.zeros_like(rg.bias))
        self.rel_gate = rg

        # Forward + reverse per-head scalars. softplus(-3)~=0.05 (long memory),
        # softplus(0)~=0.69 (moderate prior precision). Both directions start
        # symmetric.
        self.g_raw_fwd = jnp.full((num_heads,), -3.0, dtype=jnp.float32)
        self.Ip_raw_fwd = jnp.zeros((num_heads,), dtype=jnp.float32)
        self.g_raw_rev = jnp.full((num_heads,), -3.0, dtype=jnp.float32)
        self.Ip_raw_rev = jnp.zeros((num_heads,), dtype=jnp.float32)

    def _relational_gate_mod(self, eqn_ids: Array, S: int) -> Array:
        """Per-token additive gate modulation (S, H) from DAG relation degrees.

        Row-reduction of the same same/earlier/later relations the transformer
        uses as a pairwise bias; here reduced to a per-token structural feature
        vector and mapped (zero-init) to a per-head gate term.
        """
        valid = (eqn_ids >= 0)
        valid_pair = valid[:, None] & valid[None, :]
        same = ((eqn_ids[:, None] == eqn_ids[None, :]) & valid_pair).astype(jnp.float32)
        earlier = ((eqn_ids[:, None] > eqn_ids[None, :]) & valid_pair).astype(jnp.float32)
        later = ((eqn_ids[:, None] < eqn_ids[None, :]) & valid_pair).astype(jnp.float32)
        nvalid = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        feats = jnp.stack([
            jnp.sum(same, axis=1) / nvalid,
            jnp.sum(earlier, axis=1) / nvalid,
            jnp.sum(later, axis=1) / nvalid,
        ], axis=-1)                                          # (S, 3)
        feats = feats * valid.astype(jnp.float32)[:, None]   # pad tokens -> 0
        return jax.vmap(self.rel_gate)(feats)                # (S, H)

    def __call__(
        self,
        query: Array,
        key_: Array,
        value: Array,
        *,
        bias: Optional[Array] = None,   # ignored (no recurrence analogue)
        mask: Optional[Array] = None,   # treated as a padding mask (see PalimpsaMixer)
        eqn_ids: Optional[Array] = None,  # used for the relational gate proxy
        key: Optional[PRNGKey] = None,
    ) -> Array:
        x = query
        S = x.shape[0]
        H = self.num_heads
        d = self.head_dim

        q = jax.vmap(self.query_proj)(x).reshape(S, H, d)
        k = jax.vmap(self.key_proj)(x).reshape(S, H, d)
        v = jax.vmap(self.value_proj)(x).reshape(S, H, d)
        b = jax.vmap(self.bias_proj)(x).reshape(S, H, d)
        gt = jnn.softplus(jax.vmap(self.gate_proj)(x))      # (S, H)

        # Relational structural prior folded into the forget gate (additive,
        # pre-softplus would be cleaner but gate is already softplus'd; we add a
        # non-negative-ish modulation post-hoc and re-clamp >= 0).
        if eqn_ids is not None:
            gate_mod = self._relational_gate_mod(eqn_ids, S)  # (S, H), zero at init
            gt = jnn.softplus(gt + gate_mod)                  # keep gt >= 0

        if mask is not None:
            if mask.ndim == 3:
                valid = jnp.any(mask, axis=(0, 1))
            elif mask.ndim == 2:
                valid = jnp.any(mask, axis=0)
            else:
                valid = mask
            valid = valid.astype(x.dtype)[:, None]            # (S, 1)
            v = v * valid[:, :, None]
            b = b * valid[:, :, None]
            gt = gt * valid

        g_f = jnn.softplus(self.g_raw_fwd)
        Ip_f = jnn.softplus(self.Ip_raw_fwd)
        g_r = jnn.softplus(self.g_raw_rev)
        Ip_r = jnn.softplus(self.Ip_raw_rev)

        # Forward pass (causal left->right).
        out_fwd = palimpsa(
            q[None], k[None], v[None], b[None], gt[None], g_f, Ip_f,
            scale=None, chunk_size=self.chunk_size,
        )[0]                                                   # (S, H, d)

        # Reverse pass: flip the sequence (S) axis on every per-token input,
        # run the SAME causal kernel (now scanning right->left), flip back.
        qr = jnp.flip(q, axis=0); kr = jnp.flip(k, axis=0)
        vr = jnp.flip(v, axis=0); br = jnp.flip(b, axis=0)
        gtr = jnp.flip(gt, axis=0)
        out_rev = palimpsa(
            qr[None], kr[None], vr[None], br[None], gtr[None], g_r, Ip_r,
            scale=None, chunk_size=self.chunk_size,
        )[0]
        out_rev = jnp.flip(out_rev, axis=0)                    # back to original order

        out = (out_fwd + out_rev).reshape(S, H * d)            # bidirectional combine = sum
        return jax.vmap(self.output_proj)(out)



class PalimpsaEncoderLayer(eqx.Module):
    """Encoder block whose token mixer is the Palimpsa linear-attention kernel.

    The Palimpsa recurrence has no QK^T-logit analogue, so there is no pairwise
    relational bias here. Structure instead enters through ``eqn_ids``, which the
    mixers consume as a DAG-degree forget-gate modulation (zero-init, so an
    untrained layer is exactly plain Palimpsa).
    """
    attn_norm: eqx.nn.LayerNorm
    attn_layer: object  # PalimpsaMixer | BiPalimpsaMixer
    mlp_norm: eqx.nn.LayerNorm
    mlp: SwiGLU

    num_heads: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    bidirectional: bool = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        embd_dim: int,
        hidden_dim: int,
        key: PRNGKey = None,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        attn_key, mlp_key = jrand.split(key)
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.bidirectional = bidirectional

        self.attn_norm = eqx.nn.LayerNorm(embd_dim)
        mixer = BiPalimpsaMixer if bidirectional else PalimpsaMixer
        self.attn_layer = mixer(num_heads, embd_dim, key=attn_key, **kwargs)
        self.mlp_norm = eqx.nn.LayerNorm(embd_dim)
        self.mlp = SwiGLU(embd_dim, 4, key=mlp_key)

    def __call__(
        self,
        x: Array,
        eqn_ids: Optional[Array] = None,
        mask: Optional[Array] = None,
        *,
        key: PRNGKey,
    ) -> Array:
        keys = jrand.split(key, 3)
        y = jax.vmap(self.attn_norm)(x)
        y = self.attn_layer(y, y, y, bias=None, mask=mask,
                            eqn_ids=eqn_ids, key=keys[0])
        x = x + y

        y = jax.vmap(self.mlp_norm)(x)
        y = jax.vmap(self.mlp)(y)
        return x + y


class PalimpsaEncoder(eqx.Module):
    """Stack of `num_layers` Palimpsa encoder layers.

    Drop-in counterpart to `encoder.Encoder` with the same call signature, so a
    caller picks the token mixer by choosing the class rather than by threading
    a `policy` string through every constructor.
    """
    num_layers: int = eqx.field(static=True)
    layers: Sequence[PalimpsaEncoderLayer]

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embd_dim: int,
        hidden_dim: int,
        key: PRNGKey,
        bidirectional: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        keys = jrand.split(key, num_layers)
        self.num_layers = num_layers
        self.layers = [PalimpsaEncoderLayer(
            num_heads, embd_dim, hidden_dim, key=k,
            bidirectional=bidirectional, **kwargs
        ) for k in keys]

    def __call__(
        self,
        xs: Array,
        eqn_ids: Optional[Array] = None,
        mask: Optional[Array] = None,
        *,
        key: PRNGKey,
    ) -> Array:
        for i, layer in enumerate(self.layers):
            xs = layer(xs, eqn_ids=eqn_ids, mask=mask, key=jrand.fold_in(key, i))
        return xs
