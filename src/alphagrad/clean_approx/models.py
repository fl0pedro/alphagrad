import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Sequence, Optional

class LinearAttention(nn.Module):
    """Minimal Linear Attention to serve as the Palimpsa Encoder core."""
    dim: int
    heads: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        B, N, D = x.shape
        head_dim = self.dim // self.heads
        
        q = nn.Dense(self.dim, use_bias=False)(x).reshape((B, N, self.heads, head_dim))
        k = nn.Dense(self.dim, use_bias=False)(x).reshape((B, N, self.heads, head_dim))
        v = nn.Dense(self.dim, use_bias=False)(x).reshape((B, N, self.heads, head_dim))
        
        q = jax.nn.elu(q) + 1.0
        k = jax.nn.elu(k) + 1.0
        
        # Linear attention: Q @ (K^T @ V)
        kv = jnp.einsum('bnhd,bnhe->bhde', k, v)
        out = jnp.einsum('bnhd,bhde->bnhe', q, kv)
        normalizer = jnp.einsum('bnhd,bnhd->bnh', q, k) + 1e-6
        
        out = out / jnp.expand_dims(normalizer, -1)
        out = out.reshape((B, N, self.dim))
        
        return nn.Dense(self.dim)(out)

class PalimpsaEncoder(nn.Module):
    """Encodes the incrementally tokenized JAXPR."""
    vocab_size: int
    embed_dim: int
    heads: int
    layers: int
    
    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(tokens)
        for _ in range(self.layers):
            x = x + LinearAttention(dim=self.embed_dim, heads=self.heads)(nn.LayerNorm()(x))
            ffn = nn.Dense(self.embed_dim)(jax.nn.relu(nn.Dense(self.embed_dim * 4)(nn.LayerNorm()(x))))
            x = x + ffn
        return x

class PointerNetwork(nn.Module):
    """Masked pointer network for vertex elimination."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, query: jax.Array, keys: jax.Array, mask: jax.Array) -> jax.Array:
        # query: (B, D)
        # keys: (B, N, D)
        # mask: (B, N) where 1.0 is available, 0.0 is masked
        
        q_proj = nn.Dense(self.hidden_dim)(query)
        k_proj = nn.Dense(self.hidden_dim)(keys)
        
        logits = jnp.einsum('bd,bnd->bn', q_proj, k_proj)
        
        # Apply mask
        inf_mask = jnp.where(mask == 1.0, 0.0, -1e9)
        return logits + inf_mask

class ApproxHeads(nn.Module):
    """Specialized heads for approximations."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x: jax.Array) -> dict:
        # x: (B, D) summary of the context
        
        # Categorical head (skip, diag, compress, quantize)
        action_logits = nn.Dense(4, name='action_head')(x)
        
        # Parameter categorical head (for compress agg and quantize dtype)
        # Assuming 3 agg types and 4 dtypes
        param_logits = nn.Dense(7, name='param_head')(x)
        
        # Pointer head for axes (i, j)
        # Needs to point into the axis representations, which we mock as a fixed size for now
        axis_i_logits = nn.Dense(16, name='axis_i_head')(x)
        axis_j_logits = nn.Dense(16, name='axis_j_head')(x)
        
        # Prime exponent head for factor
        # E.g. exponent for base 2, 3, 5
        prime_exp_logits = nn.Dense(3, name='prime_exp_head')(x)
        
        return {
            'action': action_logits,
            'param': param_logits,
            'axis_i': axis_i_logits,
            'axis_j': axis_j_logits,
            'prime_exp': prime_exp_logits
        }

class BasePolicy(nn.Module):
    vocab_size: int
    embed_dim: int
    num_vertices: int
    
    def setup(self):
        self.encoder = PalimpsaEncoder(vocab_size=self.vocab_size, embed_dim=self.embed_dim, heads=4, layers=2)
        self.elim_head = PointerNetwork(hidden_dim=self.embed_dim)
        self.skip_head = nn.Dense(2, name='skip_head')
        self.approx_heads = ApproxHeads(hidden_dim=self.embed_dim)

class NNPolicy(BasePolicy):
    """Standard Feed Forward Policy"""
    @nn.compact
    def __call__(self, tokens: jax.Array, avail_mask: jax.Array, step_type: jax.Array):
        encoded = self.encoder(tokens)
        context = encoded.mean(axis=1) # (B, D)
        
        # Depending on step_type, we would normally route to different heads.
        # For a clean unified return, we compute all.
        elim_logits = self.elim_head(context, encoded[:, :self.num_vertices, :], avail_mask)
        skip_logits = self.skip_head(context)
        approx_dict = self.approx_heads(context)
        
        return elim_logits, skip_logits, approx_dict

class ALIFSNNPolicy(BasePolicy):
    """ALIF Spiking Neural Network Policy"""
    @nn.compact
    def __call__(self, tokens: jax.Array, avail_mask: jax.Array, step_type: jax.Array, v_mem: jax.Array):
        # Spiking recurrence: v_mem = decay * v_mem + input - spikes
        # Not fully implemented. We provide the structure.
        encoded = self.encoder(tokens)
        context = encoded.mean(axis=1)
        
        # Dummy ALIF layer
        decay = 0.9
        threshold = 1.0
        v_mem = decay * v_mem + nn.Dense(self.embed_dim)(context)
        spikes = (v_mem >= threshold).astype(jnp.float32)
        v_mem = jnp.where(spikes > 0, v_mem - threshold, v_mem)
        
        elim_logits = self.elim_head(spikes, encoded[:, :self.num_vertices, :], avail_mask)
        skip_logits = self.skip_head(spikes)
        approx_dict = self.approx_heads(spikes)
        
        return elim_logits, skip_logits, approx_dict, v_mem
