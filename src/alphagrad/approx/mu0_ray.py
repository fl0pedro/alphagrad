from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Use a plain threading lock so tqdm doesn't leak a named POSIX
# semaphore when the driver is signal-killed (matches mu0.py).
import threading as _threading
tqdm.set_lock(_threading.RLock())

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
    # CPU approximation pool — see alphagrad/src/alphagrad/approx/cpu_approx_pool.py
    p.add_argument(
        "--cpu-callback-timeout",
        type=float,
        default=60.0,
        help="Per-call timeout (seconds) for ray.get on the CPU-approx "
             "pool. When exceeded, the actor is killed, a replacement is "
             "respawned, and the env step receives a sentinel "
             "(zeros / -1e10 reward) so the rollout continues. Default 60s "
             "= ~70x normal per-callback wall (~0.9s baseline).",
    )
    p.add_argument(
        "--cpu-worker-recycle-every",
        type=int,
        default=50,
        help="Recycle (ray.kill + respawn) every actor in the CPU-approx "
             "pool after this many episodes. Bounds the per-actor "
             "cost_analysis() C++ residual (REPORT.md §6). 0 disables.",
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
    # JAX must be told NOT to preallocate before it imports — otherwise it
    # grabs ~90% of every visible GPU at startup, leaving no headroom for
    # the ~6-10 GiB transient buffers that mctx's tree-search + SPMD
    # collectives need at runtime. Ray's runtime_env env_vars fires in the
    # actor process before any `import jax`, so this is the right knob.
    spmd_env_vars = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",  # CUDA allocator, on-demand
    }
    spmd_kwargs = (
        {"num_gpus": args.spmd_gpus, "runtime_env": {"env_vars": spmd_env_vars}}
        if args.spmd_gpus > 0
        else {"runtime_env": {"env_vars": spmd_env_vars}}
    )

    spmd_actor = SPMDActor.options(**spmd_kwargs).remote(
        args_dict, variant, int(variant_args.seed)
    )

    # ``cpu_actor_options`` captures the same .options(...) kwargs the
    # initial pool was spawned with, so the SPMD actor can use them to
    # respawn killed actors after a callback timeout. Keep these in
    # sync with the .options(...) call right below.
    cpu_actor_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {
            "env_vars": {
                "JAX_PLATFORMS": "cpu",
                # Same on-demand allocation policy; CPU JAX shouldn't
                # preallocate but the var is harmless and keeps actor
                # env consistent with the GPU side.
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        },
    }
    # One actor per CPU worker. The previous ``* 8`` over-provisioning was
    # a holdover from an older design where actors served concurrent
    # io_callbacks; with the current ``num_envs=4``-driven dispatch we
    # only ever have ``num_envs`` simultaneous outstanding calls, so 32
    # actors meant 28 idle ones each paying the full JAX cold-cache
    # compile cost on first invocation. Dropping to ``num_cpu_workers``
    # (default 4) cuts calibration-phase wall time roughly 8x and saves
    # ~28 GB of actor-process RAM. Bump ``--num-cpu-workers`` only if
    # dispatch concurrency grows (e.g., bigger ``num_envs``).
    initial_pool_size = max(args.num_cpu_workers, 1)
    cpu_workers = [
        CPUApproximationActor.options(**cpu_actor_options).remote(
            args_dict, variant, i,
        )
        for i in range(initial_pool_size)
    ]

    ray.get(
        spmd_actor.init_server.remote(
            cpu_workers,
            callback_timeout_s=args.cpu_callback_timeout,
            recycle_every=args.cpu_worker_recycle_every,
            cpu_actor_options=cpu_actor_options,
            starting_actor_id=initial_pool_size,
        )
    )
    ray.get([spmd_actor.ready.remote()] + [c.ready.remote() for c in cpu_workers])
    print(f"  actors spawned and JIT-warm in {time.time() - args.t_start:.1f}s")

    if variant_args.calibrate_steps > 0:
        _run_calibration(variant_args, spmd_actor, variant_args.calibrate_steps)

    cpu_tasks = [c.compile_approximations.remote() for c in cpu_workers]

    best_global_return = -float("inf")
    best_global_seq = None
    # Per-channel raw rewards of the overall-best trajectory we've seen
    # (i.e., the cross-section of the trajectory that won the scalar
    # weighted-sum — *not* the per-channel argmaxes which can differ).
    best_global_rewards: dict[str, float] = {}
    best_global_weighted_split: dict[str, float] = {}
    best_global_ep = -1
    # Per-channel running bests: {reward_name: {"raw_value", "weighted_value",
    # "weighted_total", "ep", "seq"}}. Populated as we see new bests from
    # the actor each episode. Restricted to reward channels with non-zero
    # weight on the driver side (the actor already filters).
    best_per_reward: dict[str, dict] = {}
    seed_counter = int(variant_args.seed) + 100

    pbar = tqdm(
        total=variant_args.episodes,
        desc=variant,
        # Show the bar even when stderr isn't a TTY (nohup'd runs). We
        # still print explicit milestone lines via tqdm.write below so
        # a tailed log file has greppable content too.
        disable=False,
        leave=True,
        ncols=180,
    )

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
        ep_mean = stats.get("mean_return", float("nan"))
        ent_mean = stats.get("entropy_mean", float("nan"))
        ent_root = stats.get("entropy_root", float("nan"))
        ploss = stats.get("policy_loss", float("nan"))
        vloss = stats.get("value_loss", float("nan"))
        rloss = stats.get("reward_loss", float("nan"))
        bsize = stats.get("buffer_size", 0)
        tstep = stats.get("train_step", 0)

        if ep_best > best_global_return:
            best_global_return = ep_best
            best_global_seq = stats.get("best_seq")
            best_global_rewards = dict(stats.get("best_overall_rewards", {}))
            best_global_weighted_split = dict(stats.get("best_overall_weighted", {}))
            best_global_ep = ep

        # Update running per-channel bests.
        for name, info in stats.get("best_per_reward", {}).items():
            prev = best_per_reward.get(name)
            if prev is None or info["raw_value"] > prev["raw_value"]:
                best_per_reward[name] = {**info, "ep": ep}

        # ---- Progress bar (one line, updated in place) ----
        pbar.update(1)
        pbar.set_description(
            f"{variant} "
            f"best:{best_global_return:+.3g} "
            f"ep_best:{ep_best:+.3g} "
            f"mean:{ep_mean:+.3g} "
            f"ent:{ent_mean:.3f}({ent_root:.3f}) "
            f"loss(p/v/r):{ploss:.2g}/{vloss:.2g}/{rloss:.2g} "
            f"buf:{bsize} step:{tstep}"
        )

        # ---- Milestone line (greppable in log files) ----
        if (ep + 1) % 5 == 0 or ep == variant_args.episodes - 1:
            per_ch_str = " ".join(
                f"{n[:4]}={info['raw_value']:+.2g}"
                for n, info in best_per_reward.items()
            )
            mean_str = " ".join(
                f"{n[:4]}={v:+.2g}"
                for n, v in stats.get("per_reward_means", {}).items()
                if v != 0.0
            )
            best_overall_str = " ".join(
                f"{n[:4]}={v:+.2g}"
                for n, v in best_global_rewards.items()
                if v != 0.0
            )
            tqdm.write(
                f"  [{variant}] ep={ep + 1:>4}/{variant_args.episodes} "
                f"best={best_global_return:+.4g}(ep{best_global_ep}) mean={ep_mean:+.4g} "
                f"ent={ent_mean:.3f} buf={bsize} step={tstep}\n"
                f"            best-overall-traj-rewards: {best_overall_str}\n"
                f"            per-channel-best:          {per_ch_str}\n"
                f"            mean-per-channel:          {mean_str}"
            )

        # ---- wandb log ----
        log_dict = {
            "episode": ep,
            "best_return": best_global_return,
            "best_return_this_ep": ep_best,
            "mean_return": ep_mean,
            "entropy_mean": ent_mean,
            "entropy_root": ent_root,
            "policy_loss": ploss,
            "value_loss": vloss,
            "reward_loss": rloss,
            "total_loss": stats.get("total_loss", float("nan")),
            "buffer_size": bsize,
            "train_step": tstep,
        }
        for name, val in stats.get("per_reward_means", {}).items():
            log_dict[f"reward_mean/{name}"] = val
        # Running per-channel best raw values — one wandb scalar per
        # tuned channel, e.g. ``best_per_channel/flops``. The companion
        # sequences are too large to log every step; we dump them once
        # at the end below.
        for name, info in best_per_reward.items():
            log_dict[f"best_per_channel/{name}"] = info["raw_value"]
            log_dict[f"best_per_channel_weighted_total/{name}"] = info[
                "weighted_total"
            ]
        # Reward breakdown of the overall-best trajectory (running). Lets
        # the user see e.g. "the trajectory that won the scalar got
        # cosine_sim=4.5, flops=-7e10, peak_memory=-8e8" without having
        # to mine the FINAL block.
        for name, v in best_global_rewards.items():
            log_dict[f"best_overall_reward/{name}"] = v
        for name, v in best_global_weighted_split.items():
            log_dict[f"best_overall_weighted/{name}"] = v
        wandb.log(log_dict)

    pbar.close()

    # ---- Final per-variant summary (lands at the end of the log) ----
    tqdm.write(f"\n========== [{variant}] FINAL ==========")
    tqdm.write(f"  overall best weighted return: {best_global_return:+.6g}  (found at ep {best_global_ep})")
    if best_global_rewards:
        tqdm.write(f"  best-overall-trajectory per-channel rewards (raw):")
        for name in sorted(best_global_rewards):
            raw = best_global_rewards[name]
            wt = best_global_weighted_split.get(name, 0.0)
            tqdm.write(f"    {name:<18s}  raw={raw:+.4g}   weighted={wt:+.4g}")
    tqdm.write(f"  best-overall sequence: {best_global_seq}")
    if best_per_reward:
        tqdm.write(f"  best-per-channel (argmax over channel-sum per env, across all episodes):")
        for name in sorted(best_per_reward):
            info = best_per_reward[name]
            tqdm.write(
                f"    {name:<18s}  raw={info['raw_value']:+.4g}  "
                f"weighted={info['weighted_value']:+.4g}  "
                f"weighted_total_of_traj={info['weighted_total']:+.4g}  "
                f"(found at ep {info['ep']})\n"
                f"      seq={info['seq']}"
            )

    # Final wandb summary entries (single scalars + per-channel best
    # sequences as a string).
    summary: dict = {
        "best_global_return": best_global_return,
        "best_global_ep": best_global_ep,
    }
    for name, info in best_per_reward.items():
        summary[f"final_best_per_channel/{name}_raw"] = info["raw_value"]
        summary[f"final_best_per_channel/{name}_weighted"] = info["weighted_value"]
        summary[f"final_best_per_channel/{name}_seq"] = str(info["seq"])
    for name, v in best_global_rewards.items():
        summary[f"final_best_overall/{name}_raw"] = v
    for name, v in best_global_weighted_split.items():
        summary[f"final_best_overall/{name}_weighted"] = v
    if best_global_seq is not None:
        summary["final_best_overall/sequence"] = str(best_global_seq)
    wandb.log(summary)

    if variant_args.replay_checkpoint_path:
        ray.get(
            spmd_actor.checkpoint_replay.remote(variant_args.replay_checkpoint_path)
        )

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
