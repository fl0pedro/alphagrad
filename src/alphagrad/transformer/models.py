from typing import Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
from jax import Array
from jax.random import PRNGKey

from alphagrad.transformer import MLP, Encoder, PositionalEncoder


class GraphEmbedding(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array

    def __init__(
        self,
        graph_shape: Sequence[int],
        embd_dim: int,
        key: PRNGKey = None,
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key = jrand.split(key, 3)
        kernel_size, stride = 1, 1
        self.embedding = eqx.nn.Conv2d(
            num_vo, num_vo, (5, kernel_size), stride=(1, stride), key=embed_key
        )
        conv_size = (num_i + num_vo - kernel_size) // stride + 1
        self.projection = jrand.normal(proj_key, (conv_size, embd_dim))
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))

    def __call__(self, graph: Array, key: PRNGKey = None) -> Tuple[Array, Array]:
        output_mask = graph.at[2, 0, :].get()
        vertex_mask = graph.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(
            vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1)
        )

        # output_token_mask = jnp.where(graph.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = graph.at[:, 1:, :].get()  #  + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)

        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(
            embeddings, self.projection
        )
        return embeddings.T, attn_mask.T


class SequentialTransformer(eqx.Module):
    num_heads: int
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_enc: Encoder
    policy_head: MLP
    value_head: MLP
    # global_token: Array
    # global_token_mask_x: Array = static_field()
    # global_token_mask_y: Array = static_field()

    def __init__(
        self,
        embd_dim: int,
        seq_len: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int = 1024,
        num_policy_layers: int = 2,
        policy_hidden_dims: Sequence[int] = [512, 256],
        value_hidden_dims: Sequence[int] = [1024, 512],
        key: PRNGKey = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        e_key, p_key, pe_key, v_key, t_key = jrand.split(key, 5)

        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=e_key,
        )

        self.policy_enc = Encoder(
            num_layers=num_policy_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=pe_key,
        )

        # self.global_token = jrand.normal(t_key, (embd_dim, 1))
        # self.global_token_mask_x = jnp.ones((seq_len, 1))
        # self.global_token_mask_y = jnp.ones((1, seq_len+1))
        self.policy_head = MLP(embd_dim, 1, policy_hidden_dims, key=p_key)
        self.value_head = MLP(embd_dim, 1, value_hidden_dims, key=v_key)

    def __call__(self, xs: Array, mask: Array = None, key: PRNGKey = None) -> Array:
        e_key, p_key = jrand.split(key, 2)

        # Add global token to input
        # xs = jnp.concatenate((self.global_token, xs), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_x, mask), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_y, mask), axis=-2)

        # Transpose inputs for equinox attention mechanism
        xs = self.pos_enc(xs).T

        # Replicate mask and apply encoder
        if mask is not None:
            mask = mask.T  #  TODO fix this weird thing here
            mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
            xs = self.encoder(xs, mask=mask, key=e_key)
        else:
            xs = self.encoder(xs, mask=None, key=e_key)
        # global_token_xs = xs[0]
        values = jax.vmap(self.value_head)(xs)

        # similar to Global Average Pooling in ViTs
        value = jnp.mean(values)

        policy = jax.vmap(self.policy_head)(xs)
        return jnp.concatenate((jnp.array([value]), policy.squeeze()))


class PPOModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    transformer: SequentialTransformer

    def __init__(
        self,
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        key: PRNGKey = None,
        **kwargs,
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        embed_key, token_key, proj_key, tf_key = jrand.split(key, 4)
        self.embedding = eqx.nn.Conv2d(num_vo, num_vo, (5, 1), key=embed_key)
        self.projection = eqx.nn.Linear(num_i + num_vo, embd_dim, key=proj_key)
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
        self.transformer = SequentialTransformer(
            embd_dim, num_vo, num_layers, num_heads, key=tf_key, **kwargs
        )

    def __call__(self, xs: Array, key: PRNGKey = None) -> Array:
        output_mask = xs.at[2, 0, :].get()
        vertex_mask = xs.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(
            vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1)
        )

        # output_token_mask = jnp.where(xs.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = xs.at[:, 1:, :].get()  # + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)

        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        # embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(embeddings, self.projection)
        embeddings = jax.vmap(self.projection, in_axes=0)(embeddings)
        return self.transformer(embeddings.T, mask=attn_mask, key=key)


class AlphaZeroModel(eqx.Module):
    embedding: eqx.nn.Conv2d
    projection: Array
    # output_token: Array
    transformer: SequentialTransformer

    def __init__(
        self,
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        key: PRNGKey = None,
        **kwargs,
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        num_i, num_vo = int(num_i), int(num_vo)
        embed_key, token_key, proj_key, tf_key = jrand.split(key, 4)
        kernel_size, stride = 3, 2
        self.embedding = eqx.nn.Conv2d(
            num_vo, num_vo, (5, kernel_size), stride=(1, stride), key=embed_key
        )
        conv_size = (num_i + num_vo - kernel_size) // stride + 1
        self.projection = jrand.normal(proj_key, (conv_size, embd_dim))
        # self.output_token = jrand.normal(token_key, (num_i+num_vo, 1))
        self.transformer = SequentialTransformer(
            embd_dim, num_vo, num_layers, num_heads, key=tf_key, **kwargs
        )

    def __call__(self, xs: Array, key: PRNGKey = None) -> Array:
        output_mask = xs.at[2, 0, :].get()
        vertex_mask = xs.at[1, 0, :].get() - output_mask
        attn_mask = jnp.logical_or(
            vertex_mask.reshape(1, -1), vertex_mask.reshape(-1, 1)
        )

        # output_token_mask = jnp.where(xs.at[2, 0, :].get() > 0, self.output_token, 0.)
        edges = xs.at[:, 1:, :].get()  #  + output_token_mask[jnp.newaxis, :, :]
        edges = edges.astype(jnp.float32)

        embeddings = self.embedding(edges.transpose(2, 0, 1)).squeeze()
        embeddings = jax.vmap(jnp.matmul, in_axes=(0, None))(
            embeddings, self.projection
        )
        return self.transformer(embeddings.T, mask=attn_mask, key=key)


class PolicyNet(eqx.Module):
    num_heads: int
    embedding: GraphEmbedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    head: MLP

    def __init__(
        self,
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int = 1024,
        mlp_dims: Sequence[int] = [512, 256],
        key: PRNGKey = None,
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        self.num_heads = num_heads
        encoder_key, embed_key, key = jrand.split(key, 3)
        self.embedding = GraphEmbedding(graph_shape, embd_dim, key=embed_key)

        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(embd_dim, num_vo)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=encoder_key,
        )

        self.head = MLP(embd_dim, 1, mlp_dims, key=key)

    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:
        # Embed the input graph
        embeddings, mask = self.embedding(graph)

        # Transpose inputs for equinox attention mechanism
        embeddings = self.pos_enc(embeddings).T

        # Replicate mask and apply encoder
        mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(embeddings, mask=mask, key=key)

        policy = jax.vmap(self.head)(xs)
        return policy.squeeze()


class ValueNet(eqx.Module):
    num_heads: int
    embedding: GraphEmbedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    head: MLP
    # global_token: Array
    # global_token_mask_x: Array = static_field()
    # global_token_mask_y: Array = static_field()

    def __init__(
        self,
        graph_shape: Sequence[int],
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int = 1024,
        mlp_dims: Sequence[int] = [1024, 512],
        key: PRNGKey = None,
    ) -> None:
        super().__init__()
        num_i, num_vo, num_o = graph_shape
        self.num_heads = num_heads
        embedding_key, encoder_key, token_key, key = jrand.split(key, 4)
        self.embedding = GraphEmbedding(graph_shape, embd_dim, key=embedding_key)

        # Here we use seq_len + 1 because of the global class token
        self.pos_enc = PositionalEncoder(embd_dim, num_vo)

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=encoder_key,
        )

        # self.global_token = jrand.normal(token_key, (embd_dim, 1))
        # self.global_token_mask_x = jnp.ones((num_vo, 1))
        # self.global_token_mask_y = jnp.ones((1, num_vo+1))
        self.head = MLP(embd_dim, 1, mlp_dims, key=key)

    def __call__(self, graph: Array, key: PRNGKey = None) -> Array:
        # Embed the input graph
        embeddings, mask = self.embedding(graph)

        # Add global token to input
        # embeddings = jnp.concatenate((self.global_token, embeddings), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_x, mask), axis=-1)
        # mask = jnp.concatenate((self.global_token_mask_y, mask), axis=-2)

        # Transpose inputs for equinox attention mechanism
        embeddings = self.pos_enc(embeddings).T

        # Replicate mask and apply encoder
        mask = jnp.tile(mask[jnp.newaxis, :, :], (self.num_heads, 1, 1))
        xs = self.encoder(embeddings, mask=mask, key=key)
        values = jax.vmap(self.head)(xs)

        return jnp.mean(values)


class SwiGLU(eqx.Module):
    gate_up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, embd_dim: int, hidden_dim: int, key: PRNGKey):
        super().__init__()

        up_key, down_key = jrand.split(key)
        self.gate_up_proj = eqx.nn.Linear(
            embd_dim, hidden_dim * 2, use_bias=False, key=up_key
        )
        self.down_proj = eqx.nn.Linear(
            hidden_dim, embd_dim, use_bias=False, key=down_key
        )

    def __call__(self, x: Array) -> Array:
        out = self.gate_up_proj(x)
        gate, up = jnp.split(out, 2, axis=-1)
        return self.down_proj(jnn.silu(gate) * up)


class TransformerBlock(eqx.Module):
    attn_norm: eqx.nn.LayerNorm
    attn_layer: eqx.nn.MultiheadAttention
    mlp_norm: eqx.nn.LayerNorm
    mlp: SwiGLU

    def __init__(self, embd_dim: int, num_heads: int, hidden_dim: int, key: PRNGKey):
        super().__init__()
        attn_key, mlp_key = jrand.split(key)
        self.attn_norm = eqx.nn.LayerNorm(embd_dim)
        self.attn_layer = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embd_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=attn_key,
        )
        self.mlp_norm = eqx.nn.LayerNorm(embd_dim)
        self.mlp = SwiGLU(embd_dim, hidden_dim, key=mlp_key)

    def __call__(
        self, x: Array, mask: Optional[Array] = None, *, key: Optional[PRNGKey] = None
    ) -> Array:
        norm_x = jax.vmap(self.attn_norm)(x)

        attn_out = self.attn_layer(norm_x, norm_x, norm_x, mask=mask, key=key)
        x = x + attn_out

        norm_x = jax.vmap(self.mlp_norm)(x)
        mlp_out = jax.vmap(self.mlp)(norm_x)
        x = x + mlp_out
        return x


class BaseSSM(eqx.Module):
    A_log: Array
    B: Array
    C: Array
    D: Array
    dt: Array

    state_dim: int
    embd_dim: int

    def __init__(self, embd_dim: int, state_dim: int = 16, key: PRNGKey = None):
        super().__init__()
        key_A, key_B, key_C, key_D, key_dt = jrand.split(key, 5)
        self.state_dim = state_dim
        self.embd_dim = embd_dim

        self.A_log = jrand.uniform(
            key_A, (embd_dim, state_dim), minval=-3.0, maxval=0.0
        )

        self.B = jrand.normal(key_B, (embd_dim, state_dim)) * 0.02
        self.C = jrand.normal(key_C, (embd_dim, state_dim)) * 0.02
        self.D = jrand.normal(key_D, (embd_dim,))

        self.dt = jrand.uniform(key_dt, (embd_dim,), minval=-3.0, maxval=-1.0)

    def __call__(self, x: Array) -> Array:
        dt = jnp.exp(self.dt)[:, None]
        A = -jnp.exp(self.A_log)

        A_bar = jnp.exp(A * dt)
        B_bar = (A_bar - 1.0) / A * self.B

        bu = jax.vmap(lambda u: u[:, None] * B_bar)(x)

        def scan_fn(h_prev, bu_curr):
            h_new = A_bar * h_prev + bu_curr
            return h_new, h_new

        h0 = jnp.zeros((self.embd_dim, self.state_dim))
        _, hs = jax.lax.scan(scan_fn, h0, bu)
        y_ssm = jnp.sum(hs * self.C[None, :, :], axis=-1)
        y_skip = x * self.D[None, :]
        return y_ssm + y_skip


class SSMBlock(eqx.Module):
    norm: eqx.nn.LayerNorm
    ssm: BaseSSM
    mlp_norm: eqx.nn.LayerNorm
    mlp: SwiGLU

    def __init__(self, embd_dim: int, hidden_dim: int, key: PRNGKey):
        super().__init__()
        ssm_key, mlp_key = jrand.split(key)
        self.norm = eqx.nn.LayerNorm(embd_dim)
        self.ssm = BaseSSM(embd_dim, key=ssm_key)
        self.mlp_norm = eqx.nn.LayerNorm(embd_dim)
        self.mlp = SwiGLU(embd_dim, hidden_dim, key=mlp_key)

    def __call__(self, x: Array) -> Array:
        # Pre-norm architecture
        norm_x = jax.vmap(self.norm)(x)
        x = x + self.ssm(norm_x)

        norm_x = jax.vmap(self.mlp_norm)(x)
        x = x + jax.vmap(self.mlp)(norm_x)
        return x


class TransformerClassifier(eqx.Module):
    embedding: eqx.nn.Embedding
    layers: list[TransformerBlock]
    head: eqx.nn.Linear

    def __init__(
        self,
        vocab_size: int = 256,
        embd_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 6,
        key: PRNGKey = None,
    ):
        super().__init__()

        keys = jrand.split(key, num_layers + 2)
        embed_key, head_key = keys[0], keys[1]
        layer_keys = keys[2:]

        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=embed_key)
        self.layers = [
            TransformerBlock(embd_dim, num_heads, hidden_dim, key=k) for k in layer_keys
        ]
        self.head = eqx.nn.Linear(embd_dim, 2, key=head_key)

    def __call__(self, x: Array, key: Optional[PRNGKey] = None) -> Array:
        x = jax.vmap(self.embedding)(x)

        for layer in self.layers:
            x = layer(x, mask=None, key=key)

        x = jnp.mean(x, axis=0)
        return self.head(x)


class SSMClassifier(eqx.Module):
    embedding: eqx.nn.Embedding
    layers: list[SSMBlock]
    head: eqx.nn.Linear

    def __init__(
        self,
        vocab_size: int = 256,
        embd_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        key: PRNGKey = None,
    ):
        super().__init__()

        keys = jrand.split(key, num_layers + 2)
        embed_key, head_key = keys[0], keys[1]
        layer_keys = keys[2:]

        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=embed_key)
        self.layers = [SSMBlock(embd_dim, hidden_dim, key=k) for k in layer_keys]
        self.head = eqx.nn.Linear(embd_dim, 2, key=head_key)

    def __call__(self, x: Array, key: Optional[PRNGKey] = None) -> Array:
        x = jax.vmap(self.embedding)(x)

        for layer in self.layers:
            x = layer(x)

        x = jnp.mean(x, axis=0)
        return self.head(x)
