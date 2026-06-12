"""Circular replay buffer over trajectory pytrees, with optional
priority-weighted sampling and disk checkpointing.

Used by alpha0 / mu0 / gfn to retain past rollouts and sample from them at
training time — the SEED-style "rollout actors push, learner pulls" pattern,
collapsed into a single process where each episode adds its fresh
trajectories to the buffer and the gradient step pulls a (potentially
larger) batch of trajectories to train on.

Storage is a pytree where every leaf has leading shape ``(capacity,
*traj_leaf_shape)``. The buffer itself is an `eqx.Module` so it can sit
inside JIT-compiled training-loop functions; ``replay_add_batch`` and
``replay_sample`` are pure functions that return updated buffers / sampled
pytrees.

Priorities
----------
Every slot has a scalar ``priority`` (default 1.0 = uniform). At sample
time the per-slot weight is ``priority**alpha`` — so:

  * ``alpha == 0``  → uniform sampling regardless of priorities.
  * ``alpha == 1``  → fully proportional ("greedy" prioritisation).
  * ``0 < alpha < 1``  → soft prioritisation (Schaul et al. 2016 default).

Importance-sampling correction (the ``β`` parameter from the prioritised-
replay paper) is *not* applied here — the trainers don't currently keep a
log of the policy version each trajectory came from, so the IS weight
isn't well-defined. If you later add policy-version stamping you can apply
the standard ``(N · p)^(-β)`` reweighting per-sample.

Checkpointing
-------------
``save_replay_buffer`` / ``load_replay_buffer`` round-trip the buffer
through ``eqx.tree_serialise_leaves`` so a long experiment can resume
mid-run with the buffer's content intact.
"""

from __future__ import annotations

import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand


class ReplayBuffer(eqx.Module):
    """Pytree-friendly circular buffer.

    Fields
    ------
    storage : pytree
        Each leaf has leading shape ``(capacity, *leaf_shape)``.
    priorities : float32 array of shape ``(capacity,)``
        Per-slot priority scalar; 0 for unused slots, > 0 for filled ones.
        Defaults to 1.0 when ``replay_add_batch`` is called without
        explicit priorities (i.e., uniform).
    cursor : int32 scalar
        Next slot to write into.
    size : int32 scalar
        Number of valid slots (≤ capacity).
    capacity : int (static)
        Maximum number of stored trajectories.
    """

    storage: object  # pytree of jax.Arrays
    priorities: jax.Array
    cursor: jax.Array
    size: jax.Array
    capacity: int = eqx.field(static=True)


def init_replay_buffer(sample_traj, capacity: int) -> ReplayBuffer:
    """Allocate a buffer matching ``sample_traj``'s pytree structure.

    ``sample_traj`` is a single trajectory (no leading batch dim) — every
    leaf must already have the per-trajectory shape the buffer should
    store. Pass one slice of a batched rollout (e.g. ``traj[0]``).
    """
    storage = jax.tree_util.tree_map(
        lambda x: jnp.zeros((capacity,) + x.shape, dtype=x.dtype),
        sample_traj,
    )
    return ReplayBuffer(
        storage=storage,
        priorities=jnp.zeros((capacity,), dtype=jnp.float32),
        cursor=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        capacity=capacity,
    )


def replay_add_batch(
    buffer: ReplayBuffer,
    batch,
    priorities: "jax.Array | None" = None,
) -> ReplayBuffer:
    """Insert a batch of trajectories with optional per-trajectory priorities.

    Each leaf in ``batch`` has leading shape ``(B, *leaf_shape)`` where
    ``B`` is typically ``num_envs``. ``priorities``, if given, has shape
    ``(B,)`` and is stored alongside the corresponding slot — used at
    sample time when ``replay_sample`` receives ``alpha > 0``.

    Slots are written in cursor-major order with wrap-around; ``size`` is
    clipped at ``capacity``. Safe for ``B > capacity`` (the latest
    ``capacity`` trajectories survive).
    """
    leaves = jax.tree_util.tree_leaves(batch)
    if not leaves:
        return buffer
    B = leaves[0].shape[0]
    indices = (buffer.cursor + jnp.arange(B)) % buffer.capacity
    new_storage = jax.tree_util.tree_map(
        lambda s, b: s.at[indices].set(b), buffer.storage, batch,
    )
    if priorities is None:
        # Uniform default — every slot gets weight 1.0 so prioritised
        # sampling reduces to uniform when no caller supplies priorities.
        priorities = jnp.ones((B,), dtype=jnp.float32)
    else:
        priorities = jnp.asarray(priorities, dtype=jnp.float32)
    new_priorities = buffer.priorities.at[indices].set(priorities)
    new_cursor = (buffer.cursor + B) % buffer.capacity
    new_size = jnp.minimum(buffer.size + B, buffer.capacity)
    return ReplayBuffer(
        storage=new_storage,
        priorities=new_priorities,
        cursor=new_cursor,
        size=new_size,
        capacity=buffer.capacity,
    )


def replay_sample(
    buffer: ReplayBuffer,
    num_samples: int,
    key,
    alpha: float = 0.0,
) -> object:
    """Priority-weighted sampling-with-replacement from the valid slots.

    Per-slot weight is ``max(priority, 1e-8)**alpha``; ``alpha == 0`` is
    uniform regardless of stored priorities, so callers that don't use
    priorities can leave the default and get the same behaviour as the
    pre-priority API.

    Returns a pytree mirroring ``buffer.storage`` with each leaf reshaped
    to ``(num_samples, *leaf_shape)``. Caller is responsible for ensuring
    ``buffer.size >= 1``.
    """
    valid_mask = (jnp.arange(buffer.capacity) < buffer.size).astype(jnp.float32)
    safe_priorities = jnp.maximum(buffer.priorities, 1e-8)
    weights = jnp.where(valid_mask > 0.5, safe_priorities ** float(alpha), 0.0)
    probs = weights / jnp.maximum(jnp.sum(weights), 1e-8)
    indices = jrand.choice(
        key, buffer.capacity, shape=(num_samples,), p=probs, replace=True,
    )
    return jax.tree_util.tree_map(lambda s: s[indices], buffer.storage)


# ---------------------------------------------------------------------------
# Disk checkpointing
# ---------------------------------------------------------------------------


def save_replay_buffer(buffer: ReplayBuffer, path: str) -> None:
    """Serialise the buffer's leaves to ``path``.

    ``path``'s parent directory is created if needed. Use
    :func:`load_replay_buffer` with a freshly-initialised buffer of
    matching capacity + spec to read it back.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    eqx.tree_serialise_leaves(path, buffer)


def load_replay_buffer(path: str, like: ReplayBuffer) -> ReplayBuffer:
    """Deserialise a buffer from ``path`` using ``like`` as the pytree
    spec (capacity / leaf shapes / dtypes must match what was saved).

    Typical usage: initialise an empty buffer for the current run, then
    call this to overwrite it with the contents of a prior checkpoint.
    """
    return eqx.tree_deserialise_leaves(path, like)
