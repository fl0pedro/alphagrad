import jax
import jax.numpy as jnp
import equinox as eqx

# Basic definitions for the RL State
class State:
    def __init__(self, jaxpr_tokens: jnp.ndarray, eliminated_mask: jnp.ndarray, current_vertex: int = -1):
        self.jaxpr_tokens = jaxpr_tokens
        self.eliminated_mask = eliminated_mask
        self.current_vertex = current_vertex

class PalimpsaEncoder(eqx.Module):
    """
    Linear attention based encoder for incrementally tokenized jaxprs.
    """
    embd_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    
    embedding: eqx.nn.Embedding
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    
    def __init__(self, embd_dim: int, num_heads: int, key):
        self.embd_dim = embd_dim
        self.num_heads = num_heads
        
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.embedding = eqx.nn.Embedding(num_embeddings=10000, embedding_size=embd_dim, key=k1)
        self.q_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k2)
        self.k_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k3)
        self.v_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k4)
        self.out_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k5)

    def __call__(self, tokens: jnp.ndarray, mask: jnp.ndarray = None):
        # Embed tokens (b, n) -> (b, n, d)
        x = jax.vmap(jax.vmap(self.embedding))(tokens)
        
        # Linear attention mechanism
        q = jax.vmap(jax.vmap(self.q_proj))(x)
        k = jax.vmap(jax.vmap(self.k_proj))(x)
        v = jax.vmap(jax.vmap(self.v_proj))(x)
        
        q = jax.nn.elu(q) + 1.0
        k = jax.nn.elu(k) + 1.0
        
        # Linear attention: (Q K^T V) approx via Q (K^T V)
        kv = jnp.einsum('bnd,bne->bde', k, v)
        out = jnp.einsum('bnd,bde->bne', q, kv)
        
        # Normalize by denominator
        denom = jnp.einsum('bnd,bd->bn', q, jnp.sum(k, axis=1))
        out = out / (denom[..., None] + 1e-5)
        
        return jax.vmap(jax.vmap(self.out_proj))(out)
