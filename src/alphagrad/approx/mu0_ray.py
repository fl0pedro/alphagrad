from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

from alphagrad.approx.mu0_args import make_argparser
from alphagrad.approx.variants import (
    VARIANT_PRESETS,
    _apply_variant_preset,
    _default_full_curriculum,
)

_DEFAULT_VARIANT_SWEEP = "ve_only,diag_gcd,diag_factor,compress,full,full_curriculum"

_REWARD_NAMES = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "cosine_sim",
    "frob_residual",
)
_NUM_REWARDS = len(_REWARD_NAMES)
_COSINE_SIM_IDX = _REWARD_NAMES.index("cosine_sim")


def _symlog_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _extend_argparser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument(
        "--spmd-gpus",
        type=float,
        default=1.0,
        help="Number of GPUs reserved for the centralized SPMD actor. "
        "Set to match total available GPUs.",
    )
    p.add_argument(
        "--num-cpu-workers",
        type=int,
        default=4,
        help="Number of asynchronous CPU workers for dataset/approximation tasks.",
    )
    p.add_argument(
        "--learner-train-every",
        type=int,
        default=1,
        help="Train-step count to trigger per rollout batch generated locally by the SPMD actor.",
    )
    p.add_argument(
        "--ray-address",
        type=str,
        default="",
        help="Address of an existing Ray cluster (passed to ray.init). Empty = local cluster.",
    )
    p.add_argument(
        "--variant-sweep",
        type=str,
        default=_DEFAULT_VARIANT_SWEEP,
        help="Comma-separated variant list to train sequentially. Empty string = honour --variant.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="dsnn-vertex",
        help="wandb project name.",
    )
    p.add_argument(
        "--strict-config",
        action="store_true",
        help="Disable auto-tuning of num_envs and minibatches for GPU divisibility.",
    )
    return p


def _disable_ray_uv_autodetect() -> None:
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"


def _assert_jax_free() -> None:
    leaked = sorted(m for m in sys.modules if m == "jax" or m.startswith("jax."))
    if leaked:
        raise RuntimeError(f"jax has leaked into the driver process: {leaked[:5]}...")


def _ensure_curriculum_for_variant(args, variant: str) -> None:
    if variant == "full_curriculum" and not args.curriculum.strip():
        stages = _default_full_curriculum(args.episodes)
        args.curriculum = ",".join(f"{name}:{n}" for name, n in stages)
        total = sum(n for _, n in stages)
        args.episodes = total
        print(f"  [{variant}] auto-curriculum: ... (total {total} episodes)")


def _run_calibration(args, spmd_actor, num_rollouts):
    print(f"  [calibration] {num_rollouts} zero-pref rollouts on SPMD actor...")
    import ray

    mean_vec = ray.get(
        spmd_actor.reward_vec_means.remote(
            rng_seed=int(args.seed) + 7, num_rollouts=num_rollouts
        )
    )
    mean_vec = np.asarray(mean_vec, dtype=np.float32)
    mean_abs = np.abs(_symlog_np(mean_vec))
    mean_abs[_COSINE_SIM_IDX] = 1.0
    scaling = 1.0 / np.maximum(mean_abs, 1e-3)

    w = np.zeros((_NUM_REWARDS,), dtype=np.float32)
    _CMP = {"graphax": "muls_adds_fmas", "flops": "flops", "latency": "latency_ns"}
    _MEM = {
        "graphax": "max_io_sum",
        "bytes_accessed": "bytes_accessed",
        "peak_memory": "peak_memory",
    }
    if "cmp" in args.rewards:
        w[_REWARD_NAMES.index(_CMP[args.cmp_type])] = args.lambda_cmp
    if "mem" in args.rewards:
        w[_REWARD_NAMES.index(_MEM[args.mem_type])] = args.lambda_mem
    if "acc" in args.rewards:
        w[_COSINE_SIM_IDX] = 1.0
    if args.lambda_frob != 0.0:
        w[_REWARD_NAMES.index("frob_residual")] = args.lambda_frob

    abs_weights = w * scaling
    ray.get(spmd_actor.set_reward_weights.remote(abs_weights))
    print("  [calibration] applied scaling.")


def _run_one_variant(args, variant: str) -> None:
    import ray

    import wandb
    from alphagrad.approx.mu0_ray_actors import CPUApproximationActor, SPMDActor

    variant_args = copy.deepcopy(args)
    variant_args.variant = variant
    _apply_variant_preset(variant_args)
    _ensure_curriculum_for_variant(variant_args, variant)

    args_dict = vars(variant_args)
    wandb.init(
        project=args.wandb_project,
        name=f"{args.name}-{variant}",
        config=args_dict,
        mode="disabled" if args.wandb == "disabled" else args.wandb,
        reinit=True,
    )

    print(f"\n========== variant: {variant} ==========")
    spmd_kwargs = {"num_gpus": args.spmd_gpus} if args.spmd_gpus > 0 else {}
    spmd_actor = SPMDActor.options(**spmd_kwargs).remote(
        args_dict, variant, int(variant_args.seed)
    )

    cpu_workers = [
        CPUApproximationActor.options(
            num_cpus=1, 
            num_gpus=0, 
            runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}}
        ).remote(args_dict, variant, i)
        for i in range(args.num_cpu_workers)
    ]

    ray.get([spmd_actor.ready.remote()] + [c.ready.remote() for c in cpu_workers])
    print(f"  actors spawned and JIT-warm in {time.time() - args.t_start:.1f}s")

    if variant_args.calibrate_steps > 0:
        _run_calibration(variant_args, spmd_actor, variant_args.calibrate_steps)

    # Launch asynchronous CPU tasks
    cpu_tasks = [c.compile_approximations.remote() for c in cpu_workers]

    best_global_return = -float("inf")
    best_global_seq = None
    seed_counter = int(variant_args.seed) + 100

    for ep in range(variant_args.episodes):
        seed_counter += 1
        stats = ray.get(
            spmd_actor.run_rollout_and_train.remote(
                rng_seed=seed_counter,
                preference_np=None,
                pin_rules=None,
                reset_env=True,
                train_steps=args.learner_train_every,
            )
        )

        ep_best = stats.get("best_return", -float("inf"))
        if ep_best > best_global_return:
            best_global_return = ep_best
            best_global_seq = stats.get("best_seq")

        log_dict = {
            "episode": ep,
            "best_return": best_global_return,
            "best_return_this_ep": ep_best,
        }
        log_dict.update(
            {
                k: v
                for k, v in stats.items()
                if k not in ["best_return", "best_seq", "per_reward_means"]
            }
        )
        for name, val in stats.get("per_reward_means", {}).items():
            log_dict[f"reward/{name}"] = val

        wandb.log(log_dict)

        if (ep + 1) % 25 == 0 or ep == variant_args.episodes - 1:
            print(
                f"  [{variant}] ep={ep + 1}/{variant_args.episodes} best={best_global_return:+.4g} step={stats.get('train_step', 0)}"
            )

    if variant_args.replay_checkpoint_path:
        ray.get(
            spmd_actor.checkpoint_replay.remote(variant_args.replay_checkpoint_path)
        )

    wandb.log({"best_global_return": best_global_return})
    wandb.finish()

    ray.kill(spmd_actor, no_restart=True)
    for c in cpu_workers:
        ray.kill(c, no_restart=True)


def main() -> int:
    _disable_ray_uv_autodetect()
    p = make_argparser()
    p = _extend_argparser(p)
    args = p.parse_args()
    args.t_start = time.time()
    _assert_jax_free()

    import ray

    init_kwargs = {"address": args.ray_address} if args.ray_address else {}
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    ray.init(**init_kwargs, ignore_reinit_error=True)

    sweep = (
        [v.strip() for v in args.variant_sweep.split(",") if v.strip()]
        if args.variant_sweep.strip()
        else [args.variant]
    )
    if args.mcts_mode != "gumbel":
        args.mcts_mode = "gumbel"
    if args.replay_buffer_size <= 0:
        args.replay_buffer_size = 1024

    for variant in sweep:
        _assert_jax_free()
        _run_one_variant(args, variant)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())