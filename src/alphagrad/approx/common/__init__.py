"""Shared utilities for the approx-RL trainers.

The trainers in `alphagrad.approx` (PPO, GDPO, AlphaZero, MuZero, GFN) share a lot
of plumbing around the `VertexEliminationEnv` — example lookup, sparsity masks,
GAE computation, weight initialisation, batching. This package collects those
pieces in one place so the trainer files contain only the bits that are
actually specific to the RL algorithm.
"""

from alphagrad.approx.common.batching import shuffle_and_batch
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.common.examples import (
    data_gen,
    get_args,
    get_fn,
    infer_argnums,
)
from alphagrad.approx.common.gae import (
    get_advantages,
    get_num_clipping_triggers,
    inverse_reward_normalization_fn,
    reward_normalization_fn,
)
from alphagrad.approx.common.init import (
    init_linear_weights,
    scale_module_weight,
)
from alphagrad.approx.common.masks import (
    build_legacy_sp_valid_mask,
    build_pair_valid_mask,
    build_vertex_valid_static,
    vertex_avail_at_step,
    vertex_axis_dims,
)

__all__ = [
    "build_legacy_sp_valid_mask",
    "build_pair_valid_mask",
    "build_vertex_valid_static",
    "data_gen",
    "generate_eval_samples",
    "get_advantages",
    "get_args",
    "get_fn",
    "get_num_clipping_triggers",
    "infer_argnums",
    "init_linear_weights",
    "inverse_reward_normalization_fn",
    "reward_normalization_fn",
    "scale_module_weight",
    "shuffle_and_batch",
    "vertex_avail_at_step",
    "vertex_axis_dims",
]
