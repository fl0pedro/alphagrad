"""Pytree-friendly minibatch shuffling."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrand


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch(tree, minibatches: int, key):
    """Reshape a `(num_envs, rollout_length, ...)` pytree into `(minibatches, size, ...)`.

    Trailing samples that don't fit into a full minibatch are dropped. Every leaf
    of `tree` is sliced through the same shared random index so corresponding
    fields stay aligned across the batch.
    """
    leaves, _ = jax.tree_util.tree_flatten(tree)
    num_envs, rollout_length = leaves[0].shape[:2]
    size = num_envs * rollout_length // minibatches
    valid_samples = size * minibatches

    indices = jrand.permutation(key, jnp.arange(num_envs * rollout_length))
    indices = indices[:valid_samples].reshape(minibatches, size)

    def _process(x):
        x = x.reshape(-1, *x.shape[2:])
        return x[indices]

    return jax.tree_util.tree_map(_process, tree)
