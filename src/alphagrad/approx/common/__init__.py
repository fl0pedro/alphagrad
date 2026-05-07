"""Shared utilities for the approx-RL trainers.

The trainers in `alphagrad.approx` (PPO, GDPO, AlphaZero, MuZero, GFN) share a lot
of plumbing around the `VertexEliminationEnv` — example lookup, sparsity masks,
GAE computation, weight initialisation, batching. This package collects those
pieces in one place so the trainer files contain only the bits that are
actually specific to the RL algorithm.
"""

from alphagrad.approx.common.batching import (
    shuffle_and_batch,
    shuffle_and_batch_by_trajectory,
)
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
    build_pair_factor_valid_mask,
    build_pair_valid_mask,
    build_vertex_valid_static,
    vertex_avail_at_step,
    vertex_axis_dims,
)
from alphagrad.approx.common.instrumentation import (
    NUM_VERTEX_FEATURES,
    OP_TYPE_VOCAB_SIZE,
    VERTEX_FEATURE_NAMES,
    compute_per_sample_vertex_features,
    compute_vertex_features,
)
from alphagrad.approx.common.relations import (
    NUM_RELATIONS,
    RELATION_NAMES,
    compute_eqn_ids_from_tokens,
)

__all__ = [
    "NUM_RELATIONS",
    "NUM_VERTEX_FEATURES",
    "OP_TYPE_VOCAB_SIZE",
    "RELATION_NAMES",
    "VERTEX_FEATURE_NAMES",
    "build_legacy_sp_valid_mask",
    "build_pair_factor_valid_mask",
    "build_pair_valid_mask",
    "build_vertex_valid_static",
    "compute_eqn_ids_from_tokens",
    "compute_per_sample_vertex_features",
    "compute_vertex_features",
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
    "shuffle_and_batch_by_trajectory",
    "vertex_avail_at_step",
    "vertex_axis_dims",
]
