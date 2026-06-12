"""JAX-free helpers shared by the Ray driver entry points.

The drivers in `ppo_ray.py` and `mu0_ray.py` MUST stay JAX-free so the
Ray actors are the only place CUDA/JAX initialises. This module
collects helpers that those drivers need (and was previously
duplicated across them) while *never* importing JAX.

If you add a new helper here and find yourself wanting `import jax`,
put it in `reward_scaling.py` (numpy-only consumers) or
`calibration.py` (driver-side ray.get glue) instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace


def _disable_ray_uv_autodetect() -> None:
    """Ray's `uv` autodetect re-installs project deps in every actor's
    runtime env — wasteful on our pre-baked /tmp venvs. Same as the
    `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` exports in the sbatch scripts;
    setting it here too means local runs (without the sbatch wrapper)
    don't pay the cost either."""
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"


def _assert_jax_free() -> None:
    """Bail loudly if some helper leaked an `import jax` into the
    driver process. JAX in the driver hogs GPU memory the actors need
    and silently breaks the no-preallocate contract."""
    leaked = sorted(m for m in sys.modules if m == "jax" or m.startswith("jax."))
    if leaked:
        raise RuntimeError(
            f"jax has leaked into the driver process: {leaked[:5]}..."
        )


def _args_from_dict(args_dict: dict) -> SimpleNamespace:
    """Lift a JSON-able args dict back into the dotted-attribute shape
    the worker classes expect. Both `ppo_ray_worker.py` and
    `mu0_ray_worker.py` had identical copies of this two-liner."""
    return SimpleNamespace(**args_dict)


def _maybe_add(p: argparse.ArgumentParser, *names: str, **kwargs) -> None:
    """``parser.add_argument`` that no-ops when any of ``names`` is
    already registered. Lets driver-side helpers be order-insensitive
    when the algorithm-specific `make_argparser` already declares an
    overlapping flag (e.g. ``--calibrate-steps`` in `mu0_args.py`)."""
    existing_options: set[str] = set()
    for action in p._actions:
        existing_options.update(action.option_strings)
    if any(n in existing_options for n in names):
        return
    p.add_argument(*names, **kwargs)


def add_common_ray_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Driver knobs shared by `ppo_ray.py` and `mu0_ray.py`.

    Algorithm-specific knobs (`--spmd-gpus`, `--actor-num-gpus`,
    `--variant-sweep`, `--learner-train-every`) stay in each driver's
    own `_extend_argparser`.

    Idempotent and order-insensitive — each flag is only added if no
    parser action already owns that option string, so callers can
    invoke before or after their algorithm-specific arg parser
    without risking ``ArgumentError``.
    """
    _maybe_add(
        p, "--num-cpu-workers",
        type=int,
        default=4,
        help="Ray CpuApproximationActor pool size for tokenisation / "
             "approximation. Typically set 1:1 with --num-envs.",
    )
    _maybe_add(
        p, "--ray-address",
        type=str,
        default="",
        help="Address of an existing Ray cluster (passed to ray.init). "
             "Empty = local cluster.",
    )
    _maybe_add(
        p, "--wandb-project",
        type=str,
        default="dsnn-vertex",
        help="wandb project name.",
    )
    _maybe_add(
        p, "--wandb-entity",
        type=str,
        default="",
        help="wandb entity (team / user namespace). Empty = wandb default "
             "(your personal namespace). Set to e.g. 'dll-streetview' to "
             "land the run in a team's project.",
    )
    _maybe_add(
        p, "--cpu-callback-timeout",
        type=float,
        default=600.0,
        help="Per-call (warm-cache) timeout (seconds) for ray.get on the "
             "CPU-approx pool. When exceeded, the actor is killed, a "
             "replacement is respawned, and the env step receives a "
             "sentinel reward so the rollout continues. Set to 0 (or "
             "any non-positive value) to DISABLE the timeout — useful "
             "when investigating zero-reward signals to rule out the "
             "sentinel path. With timeout disabled, only actor crashes "
             "produce sentinels; slow compiles just block until done.",
    )
    _maybe_add(
        p, "--cpu-callback-initial-timeout",
        type=float,
        default=1800.0,
        help="Per-call timeout applied to the FIRST few calls per actor "
             "before its on-disk cache is warm. After this many calls "
             "(see --cpu-callback-warm-after), the regular "
             "--cpu-callback-timeout kicks in. The cold path can take "
             "10x longer than warm on novel elimination orders, so 1800s "
             "(30 min) is the conservative default. Set to 0 to disable.",
    )
    _maybe_add(
        p, "--cpu-callback-warm-after",
        type=int,
        default=3,
        help="Number of successful calls per actor before the warm-cache "
             "timeout (--cpu-callback-timeout) replaces the initial "
             "timeout (--cpu-callback-initial-timeout).",
    )
    _maybe_add(
        p, "--cpu-worker-recycle-every",
        type=int,
        default=50,
        help="Recycle (ray.kill + respawn) every actor in the CPU-approx "
             "pool after this many episodes. Bounds the per-actor "
             "cost_analysis() C++ residual (see scratch/leak_investigation/"
             "REPORT.md §6). 0 disables.",
    )
    _maybe_add(
        p, "--calibrate-steps",
        type=int,
        default=0,
        help="Pre-training reward-scale calibration: run K rollouts of "
             "the un-trained agent with zero preference, measure mean "
             "|symlog(reward)| per channel (sentinels filtered), and "
             "rescale reward_weights by 1/mean_abs so wide-magnitude "
             "channels contribute on a comparable scale. 0 disables "
             "(default — the per-rollout per-channel reward EMA in "
             "ppo_ray_worker.py covers the same need without the "
             "16-rollout up-front cost).",
    )
    _maybe_add(
        p, "--quant-dtypes",
        type=str,
        default="",
        help="Comma-separated dtype list to use for the QUANT micro-action "
             "(`bf16,f16,f8_e4m3,...`). Empty (default) = use graphax's "
             "built-in QUANT_DTYPES list. Only consumed when the variant "
             "exposes QUANT in its op-legality mask.",
    )
    _maybe_add(
        p, "--compile-cache-size",
        type=int,
        default=512,
        help="Max entries in the cluster-wide compile cache "
             "(``alphagrad.approx.common.compile_cache.CompileCacheCoordinator``). "
             "Each entry holds an ObjectRef to a serialised XLA "
             "Executable (~5-50 KB); the actual executable bytes live "
             "in Ray's plasma object store and are evicted by Ray's "
             "own LRU once it spills. 512 covers most rollouts; bump "
             "if the eviction rate stays >10% after warm-up.",
    )
    _maybe_add(
        p, "--checkpoint-path",
        type=str,
        default="",
        help="Directory for periodic agent + opt_state checkpoints. Empty "
             "disables checkpointing. On startup, a previous checkpoint at "
             "this path is auto-loaded so SLURM timeouts can resume. The "
             "SIGTERM handler also saves a final checkpoint before the "
             "process is killed.",
    )
    _maybe_add(
        p, "--checkpoint-every",
        type=int,
        default=25,
        help="Episodes between automatic checkpoints. 0 disables periodic "
             "saves (only the SIGTERM/exit hook fires).",
    )
    return p
