from typing import Sequence, Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand

import equinox as eqx

from alphagrad.approx.common.relations import NUM_RELATIONS

Array = jax.Array
PRNGKey = jax.Array


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


class SwiGLU(eqx.Module):
    """
    Implementation of SwiGLU MLP as proposed in Llama paper
    """
    gate_up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear
    
    def __init__(self, 
        hidden_size: int,
        expansion: float,
        key: PRNGKey = None,
    ):
        super().__init__()
        inner_dim = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        up_key, down_key = jrand.split(key)
        self.gate_up_proj = eqx.nn.Linear(
            hidden_size, inner_dim * 2, use_bias=False, key=up_key
        )
        self.down_proj = eqx.nn.Linear(
            inner_dim, hidden_size, use_bias=False, key=down_key
        )

    def __call__(self, x: Array) -> Array:
        out = self.gate_up_proj(x)
        gate, up = jnp.split(out, 2, axis=-1)
        return self.down_proj(jnn.silu(gate) * up)
        
class RelationalMultiheadAttention(eqx.Module):
    """Multi-head self-attention with an additive bias on the QK^T logits.

    Functionally equivalent to ``eqx.nn.MultiheadAttention`` (same Q/K/V/output
    projections, same scaling) but exposes a ``bias`` argument of shape
    ``(num_heads, S, T)`` that is added to the attention logits before
    softmax. This is what the relational-bias encoder uses to inject DAG
    structural priors via T5-style learned per-relation per-head scalars.

    When ``bias`` is None and ``mask`` is None, behaves like a vanilla MHA.
    """

    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    num_heads: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, num_heads: int, embd_dim: int, *, key: PRNGKey):
        super().__init__()
        if embd_dim % num_heads != 0:
            raise ValueError(
                f"embd_dim={embd_dim} must be divisible by num_heads={num_heads}"
            )
        keys = jrand.split(key, 4)
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.head_dim = embd_dim // num_heads
        self.query_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[0])
        self.key_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[1])
        self.value_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[2])
        self.output_proj = eqx.nn.Linear(embd_dim, embd_dim, key=keys[3])

    def __call__(
        self,
        query: Array,
        key_: Array,
        value: Array,
        *,
        bias: Optional[Array] = None,
        mask: Optional[Array] = None,
        key: Optional[PRNGKey] = None,
    ) -> Array:
        S = query.shape[0]
        T = key_.shape[0]
        H = self.num_heads
        d = self.head_dim

        Q = jax.vmap(self.query_proj)(query).reshape(S, H, d)
        K = jax.vmap(self.key_proj)(key_).reshape(T, H, d)
        V = jax.vmap(self.value_proj)(value).reshape(T, H, d)

        # Per-head logits: (H, S, T) = Q @ K^T / sqrt(d)
        logits = jnp.einsum("shd,thd->hst", Q, K) / jnp.sqrt(d)

        if bias is not None:
            logits = logits + bias

        if mask is not None:
            logits = jnp.where(mask[None, :, :], logits, -1e9)

        attn = jnn.softmax(logits, axis=-1)
        out = jnp.einsum("hst,thd->shd", attn, V).reshape(S, H * d)
        return jax.vmap(self.output_proj)(out)

class EncoderLayer(eqx.Module):
    """Transformer encoder block with optional relational bias on attention.

    Attention follows ``RelationalMultiheadAttention`` and has its own learned
    ``relation_biases`` of shape ``(NUM_RELATIONS, num_heads)`` (T5-style
    scalar per-head per-relation). When the layer is called without ``eqn_ids``
    the bias is skipped and the layer reduces to a vanilla post-LN transformer.
    """
    attn_norm: eqx.nn.LayerNorm
    attn_layer: RelationalMultiheadAttention
    mlp_norm: eqx.nn.LayerNorm
    mlp: SwiGLU
    relation_biases: Array  # (NUM_RELATIONS, num_heads); zero-initialised

    num_heads: int = eqx.field(static=True)
    embd_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        embd_dim: int,
        hidden_dim: int,
        key: PRNGKey = None,
        **kwargs,
    ) -> None:
        super().__init__()
        attn_key, mlp_key = jrand.split(key)
        self.num_heads = num_heads
        self.embd_dim = embd_dim

        self.attn_norm = eqx.nn.LayerNorm(embd_dim)
        self.attn_layer = RelationalMultiheadAttention(
            num_heads, embd_dim, key=attn_key, **kwargs
        )
        self.mlp_norm = eqx.nn.LayerNorm(embd_dim)
        self.mlp = SwiGLU(embd_dim, 4, key=mlp_key)

        # Zero-init: the layer behaves like a vanilla transformer until the
        # relation biases learn to deviate from zero. Avoids destabilising
        # the initial policy distribution.
        self.relation_biases = jnp.zeros((NUM_RELATIONS, num_heads), dtype=jnp.float32)

    def _bias_from_eqn_ids(self, eqn_ids: Array) -> Array:
        """Lift the eqn-level relations to a per-token ``(H, T, T)`` bias.

        Three relations are derived directly from the per-token equation ID
        array (matches `alphagrad.approx.common.relations.RELATION_NAMES`):

        * ``same_eqn``  — both tokens in the same equation.
        * ``earlier``   — the key token's equation index is smaller (it
                          comes earlier in topological order).
        * ``later``     — symmetrical case, key token comes later.

        Padding tokens (``eqn_ids[i] == -1``) contribute zero bias regardless
        of which relation pair they fall in, so the layer can attend across
        them freely as in the vanilla path.
        """
        valid = (eqn_ids >= 0)
        valid_pair = valid[:, None] & valid[None, :]
        same = (eqn_ids[:, None] == eqn_ids[None, :]) & valid_pair
        earlier = (eqn_ids[:, None] > eqn_ids[None, :]) & valid_pair
        later = (eqn_ids[:, None] < eqn_ids[None, :]) & valid_pair
        rel_masks = jnp.stack([same, earlier, later], axis=0).astype(jnp.float32)
        return jnp.einsum("rh,rij->hij", self.relation_biases, rel_masks)

    def __call__(
        self,
        x: Array,
        eqn_ids: Optional[Array] = None,
        mask: Optional[Array] = None,
        *,
        key: PRNGKey,
    ) -> Array:
        keys = jrand.split(key, 3)
        bias = None if eqn_ids is None else self._bias_from_eqn_ids(eqn_ids)

        y = jax.vmap(self.attn_norm)(x)
        y = self.attn_layer(y, y, y, bias=bias, mask=mask, key=keys[0])
        x = x + y

        y = jax.vmap(self.mlp_norm)(x)
        y = jax.vmap(self.mlp)(y)
        return x + y


class Encoder(eqx.Module):
    """Stack of `num_layers` relational-bias-aware transformer encoder layers."""
    num_layers: int
    layers: Sequence[EncoderLayer]

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embd_dim: int,
        hidden_dim: int,
        key: PRNGKey,
        **kwargs,
    ) -> None:
        super().__init__()
        keys = jrand.split(key, num_layers)

        self.num_layers = num_layers
        self.layers = [EncoderLayer(
            num_heads, embd_dim, hidden_dim, key=k, **kwargs
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
            layer_key = jrand.fold_in(key, i)
            xs = layer(xs, eqn_ids=eqn_ids, mask=mask, key=layer_key)
        return xs

