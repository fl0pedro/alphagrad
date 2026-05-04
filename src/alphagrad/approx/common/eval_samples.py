"""Helpers for refreshing eval-time argument samples between rollouts."""

from __future__ import annotations

import jax
import jax.random as jrand


def generate_eval_samples(env_obj, key, num_samples: int = 10):
    """Build a stacked batch of fresh args (data + random weights) for the given env.

    Mirrors the behaviour of `VertexEliminationEnv.from_jaxpr`: data inputs are
    refreshed via `config.data_gen`, and any `argnums` slots that aren't covered
    by the data generator get a fresh `jrand.normal` draw with the same shape
    and dtype as the existing arg.
    """
    config = env_obj.config
    args = env_obj.args

    def get_one_sample(k):
        dk, wk = jrand.split(k)
        e_args = list(args)
        if config.data_gen is not None:
            data = config.data_gen(jrand.split(dk, 5))
            for i, d in enumerate(data):
                e_args[i] = d
        if config.argnums:
            w_keys = jrand.split(wk, len(config.argnums))
            for i, arg_idx in enumerate(config.argnums):
                if config.data_gen is not None and arg_idx < len(data):
                    continue
                curr_val = e_args[arg_idx]
                e_args[arg_idx] = jrand.normal(
                    w_keys[i], curr_val.shape, curr_val.dtype
                )
        return tuple(e_args)

    keys = jrand.split(key, num_samples)
    return jax.vmap(get_one_sample)(keys)
