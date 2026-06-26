"""Shared utilities for the approx-RL trainers.

The trainers in `alphagrad.approx` (PPO, GDPO, AlphaZero, MuZero, GFN) share a lot
of plumbing around the `VertexEliminationEnv` — example lookup, sparsity masks,
GAE computation, weight initialisation, batching. This package collects those
pieces in one place so the trainer files contain only the bits that are
actually specific to the RL algorithm.

**JAX hygiene.** The Ray-driver entry points in ``ppo_ray.py`` and ``mu0_ray.py``
must stay JAX-free until Ray spawns the GPU actors (otherwise the driver
process hogs CUDA memory the actors need). Earlier this ``__init__`` eagerly
imported the JAX-using helpers (``batching``, ``mcts``, ``masks``, …) — which
meant ``from alphagrad.approx.common.cache import ...`` from the driver
also pulled JAX in via the parent package's ``__init__``, breaking the
``_assert_jax_free`` guard at PPO startup. The fix below moves the JAX-using
re-exports behind a PEP 562 ``__getattr__`` so they're loaded lazily; the
JAX-free helpers (``cache``, ``ray_runtime``, ``reward_scaling``,
``calibration``, ``checkpoint``) are still imported eagerly so callers
get a useful ``from alphagrad.approx.common import build_reward_weights``
even from JAX-free contexts.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# JAX-free re-exports — safe to load at package import. Driver processes
# (which assert no JAX module is present) rely on these working without
# triggering the lazy imports below.
# ---------------------------------------------------------------------------
from alphagrad.approx.common.cache import (
    SENTINEL_REWARD_VALUE,
    setup_jax_compile_cache,
)
from alphagrad.approx.common.calibration import run_calibration
from alphagrad.approx.common.ray_runtime import (
    _args_from_dict,
    _assert_jax_free,
    _disable_ray_uv_autodetect,
    add_common_ray_args,
)
from alphagrad.approx.common.reward_scaling import (
    COSINE_SIM_IDX,
    FROB_RESIDUAL_IDX,
    NO_SYMLOG_MASK_NP,
    NO_SYMLOG_REWARD_INDICES,
    SPARSE_TERMINAL_INDICES,
    SPARSE_TERMINAL_MASK_NP,
    aggregate_per_channel_stats,
    best_sequences_snapshot,
    build_best_sequences_wandb_payload,
    build_best_sequences_wandb_table,
    build_reward_weights,
    build_wandb_log_dict,
    dump_best_sequences_json,
    filter_sentinel_mask,
    format_milestone_line,
    init_running_bests,
    parse_reward_conditions,
    symlog_np,
    update_running_bests,
)
# ``NUM_REWARDS`` / ``REWARD_NAMES`` / ``REWARD_INDEX`` are also
# re-exported by alphagrad.approx.env (where they originate alongside
# the JAX env). Most callers import from env directly; we still expose
# them here for back-compat with the JAX-free shape constants.
from alphagrad.approx.common.reward_scaling import (
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
)


# ---------------------------------------------------------------------------
# Lazy proxies for JAX-using submodules. ``__getattr__`` (PEP 562) fires
# when an attribute isn't found via normal package lookup, so the JAX
# import is deferred until first access — which is well past the driver's
# ``_assert_jax_free`` guard. Each entry maps ``attr_name -> (submodule,
# attr_in_submodule)``.
# ---------------------------------------------------------------------------
_LAZY: dict[str, tuple[str, str]] = {
    # batching
    "shuffle_and_batch": ("alphagrad.approx.common.batching", "shuffle_and_batch"),
    "shuffle_and_batch_by_trajectory": (
        "alphagrad.approx.common.batching", "shuffle_and_batch_by_trajectory",
    ),
    # eval_samples
    "generate_eval_samples": (
        "alphagrad.approx.common.eval_samples", "generate_eval_samples",
    ),
    # examples
    "data_gen": ("alphagrad.approx.common.examples", "data_gen"),
    "get_args": ("alphagrad.approx.common.examples", "get_args"),
    "get_fn": ("alphagrad.approx.common.examples", "get_fn"),
    "scalar_loss_fn": ("alphagrad.approx.common.examples", "scalar_loss_fn"),
    "seed_loss_fn": ("alphagrad.approx.common.examples", "seed_loss_fn"),
    "grad_target_setup": ("alphagrad.approx.common.examples", "grad_target_setup"),
    "grad_target_fn": ("alphagrad.approx.common.examples", "grad_target_fn"),
    "infer_argnums": ("alphagrad.approx.common.examples", "infer_argnums"),
    # gae
    "get_advantages": ("alphagrad.approx.common.gae", "get_advantages"),
    "get_num_clipping_triggers": (
        "alphagrad.approx.common.gae", "get_num_clipping_triggers",
    ),
    "inverse_reward_normalization_fn": (
        "alphagrad.approx.common.gae", "inverse_reward_normalization_fn",
    ),
    "reward_normalization_fn": (
        "alphagrad.approx.common.gae", "reward_normalization_fn",
    ),
    # init
    "init_linear_weights": ("alphagrad.approx.common.init", "init_linear_weights"),
    "scale_module_weight": ("alphagrad.approx.common.init", "scale_module_weight"),
    # masks
    "build_legacy_sp_valid_mask": (
        "alphagrad.approx.common.masks", "build_legacy_sp_valid_mask",
    ),
    "build_pair_valid_mask": (
        "alphagrad.approx.common.masks", "build_pair_valid_mask",
    ),
    "build_vertex_valid_static": (
        "alphagrad.approx.common.masks", "build_vertex_valid_static",
    ),
    "vertex_avail_at_step": (
        "alphagrad.approx.common.masks", "vertex_avail_at_step",
    ),
    "vertex_axis_dims": (
        "alphagrad.approx.common.masks", "vertex_axis_dims",
    ),
    # instrumentation
    "NUM_VERTEX_FEATURES": (
        "alphagrad.approx.common.instrumentation", "NUM_VERTEX_FEATURES",
    ),
    "OP_TYPE_VOCAB_SIZE": (
        "alphagrad.approx.common.instrumentation", "OP_TYPE_VOCAB_SIZE",
    ),
    "VERTEX_FEATURE_NAMES": (
        "alphagrad.approx.common.instrumentation", "VERTEX_FEATURE_NAMES",
    ),
    "compute_per_sample_vertex_features": (
        "alphagrad.approx.common.instrumentation",
        "compute_per_sample_vertex_features",
    ),
    "compute_vertex_features": (
        "alphagrad.approx.common.instrumentation", "compute_vertex_features",
    ),
    # relations
    "NUM_RELATIONS": ("alphagrad.approx.common.relations", "NUM_RELATIONS"),
    "RELATION_NAMES": ("alphagrad.approx.common.relations", "RELATION_NAMES"),
    "compute_eqn_ids_from_tokens": (
        "alphagrad.approx.common.relations", "compute_eqn_ids_from_tokens",
    ),
    # mcts
    "extract_path_visits": (
        "alphagrad.approx.common.mcts", "extract_path_visits",
    ),
    # preferences
    "sample_preferences": (
        "alphagrad.approx.common.preferences", "sample_preferences",
    ),
    # replay
    "ReplayBuffer": ("alphagrad.approx.common.replay", "ReplayBuffer"),
    "init_replay_buffer": (
        "alphagrad.approx.common.replay", "init_replay_buffer",
    ),
    "load_replay_buffer": (
        "alphagrad.approx.common.replay", "load_replay_buffer",
    ),
    "replay_add_batch": (
        "alphagrad.approx.common.replay", "replay_add_batch",
    ),
    "replay_sample": (
        "alphagrad.approx.common.replay", "replay_sample",
    ),
    "save_replay_buffer": (
        "alphagrad.approx.common.replay", "save_replay_buffer",
    ),
    # schedules
    "SCHEDULES": ("alphagrad.approx.common.schedules", "SCHEDULES"),
    "schedule_at": ("alphagrad.approx.common.schedules", "schedule_at"),
}


def __getattr__(name: str):
    """PEP 562 lazy lookup — loads the JAX-using submodule only when
    one of its symbols is actually accessed."""
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(
            f"module 'alphagrad.approx.common' has no attribute {name!r}"
        )
    import importlib
    modname, attr = spec
    mod = importlib.import_module(modname)
    value = getattr(mod, attr)
    # Cache on the package module so subsequent lookups skip the
    # ``__getattr__`` round-trip.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Make ``dir(alphagrad.approx.common)`` discoverable even for the
    lazy attrs. Doesn't trigger any imports."""
    eager = [k for k in globals() if not k.startswith("_")]
    return sorted(set(eager) | set(_LAZY.keys()))


# ``__all__`` is the union of eager + lazy public names — keeps
# wildcard ``from ... import *`` callers behaving as before. We can't
# include the imports themselves in ``__all__`` until they're loaded
# (otherwise wildcard would trigger them), so this is just the keys.
__all__ = sorted(
    {n for n in globals() if not n.startswith("_") and n != "annotations"}
    | set(_LAZY.keys())
)
