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


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch_by_trajectory(tree, minibatches: int, key):
    """Per-trajectory minibatching for the B.4.next "encode once per episode" path.

    Reshapes a ``(num_envs, rollout_length, ...)`` pytree into
    ``(minibatches, envs_per_minibatch, rollout_length, ...)`` — the env axis
    is shuffled but the trajectory dimension is preserved so the loss can
    encode each trajectory's tokens once and reuse the result for all
    ``rollout_length`` steps.

    Trailing trajectories that don't fit into a full minibatch are dropped.
    """
    leaves, _ = jax.tree_util.tree_flatten(tree)
    num_envs, rollout_length = leaves[0].shape[:2]
    envs_per_mb = num_envs // minibatches
    valid_envs = envs_per_mb * minibatches

    perm = jrand.permutation(key, jnp.arange(num_envs))[:valid_envs]
    perm = perm.reshape(minibatches, envs_per_mb)

    def _process(x):
        # x: (num_envs, rollout_length, ...) → (minibatches, envs_per_mb, K, ...)
        return x[perm]

    return jax.tree_util.tree_map(_process, tree)
