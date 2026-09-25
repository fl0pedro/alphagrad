"""Helpers for refreshing eval-time argument samples between rollouts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrand


def generate_eval_samples(env_obj, key, num_samples: int = 10):
    """Build a stacked batch of fresh args (data + random weights) for the given env.

    Mirrors the behaviour of `VertexEliminationEnv.from_jaxpr`: data inputs are
    refreshed via `config.data_gen`, and any `argnums` slots that aren't covered
    by the data generator get a fresh `jrand.normal` draw with the same shape
    and dtype as the existing arg.

    WHICH SLOTS THE GENERATOR COVERS is the generator's own statement, read off
    its ``data_slots`` attribute. Without one the slots are ``0`` to
    ``len(data) - 1``, which is what every image / token generator does and is
    byte-identical to the positional loop this replaces. The recurrent SHD
    generator fills ten slots and then a block further along, so the
    positional assumption would have scattered its carried state over the
    weights.
    """
    config = env_obj.config
    args = env_obj.args
    # The full rollout's samples are rollout tuples (owner ruling 2026-09-25 Q24 a).
    _base = getattr(config.data_gen, "measure_base", None)
    if _base is not None:
        args = tuple(_base(tuple(args)))

    def _slots_of(data):
        slots = getattr(config.data_gen, "data_slots", None)
        if slots is None:
            return tuple(range(len(data)))
        slots = tuple(int(i) for i in slots)
        if len(slots) != len(data):
            raise ValueError(
                f"the data generator declares {len(slots)} slots {slots} and "
                f"returned {len(data)} arrays. A generator's `data_slots` is "
                f"the contract every refresher reads; a mismatch would put "
                f"one of its arrays in the wrong argument.")
        return slots

    def get_one_sample(k):
        dk, wk = jrand.split(k)
        e_args = list(args)
        covered: tuple = ()
        if config.data_gen is not None:
            data = config.data_gen(jrand.split(dk, 5))
            covered = _slots_of(data)
            for i, d in zip(covered, data):
                e_args[i] = d
        if config.argnums:
            w_keys = jrand.split(wk, len(config.argnums))
            for i, arg_idx in enumerate(config.argnums):
                if config.data_gen is not None and arg_idx in covered:
                    continue
                curr_val = e_args[arg_idx]
                # Seed-vertex tangent seed (a 0-d scalar appended as the last
                # differentiated arg by seed_loss_fn) must stay at its injected
                # value t=0 — the gradient/JVP is the linearization at t=0.
                # Randomizing it to N(0,1) shifts every weight by t*ones, which
                # saturates the net into a dead-gradient region (||grad||=0 ->
                # cosine 0/0 -> 0). Leave 0-d argnums untouched.
                if getattr(curr_val, "ndim", 0) == 0:
                    continue
                e_args[arg_idx] = jrand.normal(
                    w_keys[i], curr_val.shape, curr_val.dtype
                )
        return tuple(e_args)

    keys = jrand.split(key, num_samples)
    if getattr(config.data_gen, "host_draw", False):
        # A draw that runs a compiled program from the host cannot be vmapped.
        samples = [get_one_sample(k) for k in keys]
        return tuple(jnp.stack([s[i] for s in samples])
                     for i in range(len(samples[0])))
    return jax.vmap(get_one_sample)(keys)
