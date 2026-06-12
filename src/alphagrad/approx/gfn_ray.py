"""SEED-style off-policy GFlowNet trainer — Ray driver entry point.

This driver mirrors :mod:`alphagrad.approx.mu0_ray` (the MuZero Ray
driver) end-to-end. The structural reasons it picks the MuZero pattern
rather than the PPO one:

* TB is naturally off-policy: the loss re-evaluates ``log π_F(a | s)``
  under the *current* policy at every gradient step, so a replay buffer
  fits without needing importance sampling correction.
* The CPU approx pool + ``ray.put(eval_samples)`` + compile-cache
  coordinator + per-channel calibration + variant sweep are all
  algorithm-agnostic; we re-use them verbatim.

The driver stays JAX-free until Ray spawns the GPU actor (see
:func:`alphagrad.approx.common.ray_runtime._assert_jax_free` for the
guard). The GFNSPMDActor process is the only place JAX initialises.

Free-win optimizations inherited from commit b6d366a:
  #1 CPU pool 32→4 (size driven by ``--num-cpu-workers`` default 4).
  #2 ``ray.put(eval_samples)`` once per variant — wired in
     :class:`alphagrad.approx.cpu_approx_pool.CpuApproxPool.set_eval_samples`.
  #3 Pre-converted ``factor_table_j`` hoisted at worker init.
  #5 ``--terminal-rewards-only`` default ON — TB only reads
     ``traj.reward[:, -1, :]`` so per-step jacve compile is wasted work.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

# Use a plain threading lock so tqdm doesn't leak a named POSIX
# semaphore when the driver is signal-killed (matches gfn.py / mu0_ray).
import threading as _threading
tqdm.set_lock(_threading.RLock())

from alphagrad.approx.gfn_args import make_argparser
from alphagrad.approx.variants import (
    VARIANT_PRESETS,
    _apply_variant_preset,
    _default_full_curriculum,
)
from alphagrad.approx.common.calibration import run_calibration
from alphagrad.approx.common.compile_cache import (
    kill_coordinator as _kill_compile_cache,
    spawn_coordinator as _spawn_compile_cache,
)
from alphagrad.approx.common.ray_runtime import (
    _assert_jax_free,
    _disable_ray_uv_autodetect,
    add_common_ray_args,
)
from alphagrad.approx.common.reward_scaling import (
    NUM_REWARDS as _NUM_REWARDS,
    REWARD_NAMES as _REWARD_NAMES,
    COSINE_SIM_IDX as _COSINE_SIM_IDX,
    build_best_sequences_wandb_payload as _build_best_seq_wandb_payload,
    build_best_sequences_wandb_table as _build_best_seq_wandb_table,
    dump_best_sequences_json as _dump_best_sequences_json,
    symlog_np as _symlog_np,
)


_DEFAULT_VARIANT_SWEEP = (
    "ve_only,diag_gcd,diag_factor,compress,quantize,full,full_curriculum"
)


def _extend_argparser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_ray_args(p)
    p.add_argument(
        "--spmd-gpus",
        type=float,
        default=1.0,
        help="Number of GPUs reserved for the centralized SPMD actor. "
        "Set to match total available GPUs.",
    )
    p.add_argument(
        "--learner-train-every",
        type=int,
        default=1,
        help="Outer-loop train-step count per rollout. Total per-episode "
             "updates = --learner-train-every × --gradient-steps.",
    )
    p.add_argument(
        "--variant-sweep",
        type=str,
        default=_DEFAULT_VARIANT_SWEEP,
        help="Comma-separated variant list to train sequentially. Empty "
             "string = honour --variant.",
    )
    p.add_argument(
        "--strict-config",
        action="store_true",
        help="Disable auto-tuning of num_envs / minibatches for GPU divisibility.",
    )
    p.add_argument(
        "--cpu-cores-shared", action="store_true",
        help="Claim disjoint CPU-core slices from the cluster-wide allocator so "
             "this run's measurement actors don't oversubscribe cores shared "
             "with another job on the same CPU node (mirrors ppo_ray).",
    )
    p.add_argument(
        "--cpu-cores-per-actor", type=int, default=0,
        help="Force each CpuApproximationActor to pin to exactly N cores. "
             "1 = single-core-per-actor (cleanest latency CV, single-threaded "
             "exec — set --num-cpu-workers ≈ #cores). 0 = auto slice.",
    )
    return p


def _ensure_curriculum_for_variant(args, variant: str) -> None:
    if variant == "full_curriculum" and not args.curriculum.strip():
        stages = _default_full_curriculum(args.episodes)
        args.curriculum = ",".join(f"{name}:{n}" for name, n in stages)
        total = sum(n for _, n in stages)
        args.episodes = total
        print(f"  [{variant}] auto-curriculum: {args.curriculum} (total {total} episodes)")


# ---------------------------------------------------------------------------
# Pareto-front helpers (host-side, numpy) — same routines as cmorl_ray so the
# MOGFN front is recorded in the identical format for comparison.
# ---------------------------------------------------------------------------
# Shared single source for Pareto/HV math (see common.pareto_archive) — keeps
# gfn_ray, cmorl_ray and ppo_ray from drifting (the reviewer's 3-copy finding).
from alphagrad.approx.common.pareto_archive import (  # noqa: E402
    ParetoArchive,
    pareto_mask as _pareto_mask,
    hypervolume as _hypervolume,
)


def _run_one_variant(args, variant: str) -> None:
    import ray

    import wandb
    from alphagrad.approx.gfn_ray_actors import (
        CPUApproximationActor,
        GFNSPMDActor,
    )

    variant_args = copy.deepcopy(args)
    variant_args.variant = variant
    _apply_variant_preset(variant_args, variant)
    _ensure_curriculum_for_variant(variant_args, variant)

    args_dict = vars(variant_args)
    wandb.init(
        project=args.wandb_project,
        entity=getattr(args, "wandb_entity", None) or None,
        name=f"{args.name}-{variant}",
        config=args_dict,
        mode="disabled" if args.wandb == "disabled" else args.wandb,
        reinit=True,
    )

    print(f"\n========== variant: {variant} ==========")
    # JAX must be told NOT to preallocate before it imports — otherwise it
    # grabs ~90% of every visible GPU at startup, leaving no headroom for
    # transient compile/exec buffers. Ray's runtime_env env_vars fires in
    # the actor process before any `import jax`, so this is the right knob.
    spmd_env_vars = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    }
    for k, v in os.environ.items():
        if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_"):
            spmd_env_vars[k] = v
    spmd_kwargs = (
        # num_cpus=0 so the GPU trainer reserves no CPU slots — lets a
        # --num-cpus=0 GPU node physically exclude the num_cpus=1 measurement
        # actors, forcing all measurement onto the CPU node (mirrors ppo_ray).
        {"num_cpus": 0, "num_gpus": args.spmd_gpus,
         "runtime_env": {"env_vars": spmd_env_vars}}
        if args.spmd_gpus > 0
        else {"runtime_env": {"env_vars": spmd_env_vars}}
    )

    spmd_actor = GFNSPMDActor.options(**spmd_kwargs).remote(
        args_dict, variant, int(variant_args.seed),
    )

    cpu_env_vars = {
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for k, v in os.environ.items():
        if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_"):
            cpu_env_vars[k] = v
    cpu_actor_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": cpu_env_vars},
    }
    # One actor per requested CPU worker. The previous mu0_ray over-
    # provisioning (* 8) was a holdover from a design where actors served
    # concurrent io_callbacks; with num_envs=4 dispatch we only ever have
    # num_envs simultaneous outstanding calls. Free-win optimization #1
    # from commit b6d366a — 32→4 cuts calibration wall time ~8x.
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
            initial_timeout_s=args.cpu_callback_initial_timeout,
            warm_after=args.cpu_callback_warm_after,
            recycle_every=args.cpu_worker_recycle_every,
            cpu_actor_options=cpu_actor_options,
            starting_actor_id=initial_pool_size,
        )
    )
    ray.get([spmd_actor.ready.remote()] + [c.ready.remote() for c in cpu_workers])
    print(f"  actors spawned and JIT-warm in {time.time() - args.t_start:.1f}s")

    # Warm the CPU-approx pool's JIT cache BEFORE calibration so the
    # zero-pref rollouts don't time out and poison `reward_vec_means`
    # with sentinel rewards.
    cpu_tasks = [c.compile_approximations.remote() for c in cpu_workers]
    ray.get(cpu_tasks)
    print(f"  CPU pool JIT-warm in {time.time() - args.t_start:.1f}s")

    if variant_args.calibrate_steps > 0:
        run_calibration(
            spmd_actor,
            variant_args,
            variant_args.calibrate_steps,
            after_warmup=True,
        )

    # ------------------------------------------------------------------
    # Per-variant best-trajectory bookkeeping (mirrors mu0_ray.py)
    # ------------------------------------------------------------------
    best_global_return = -float("inf")
    best_global_seq = None
    best_global_rewards: dict[str, float] = {}
    best_global_weighted_split: dict[str, float] = {}
    best_global_ep = -1
    best_per_reward: dict[str, dict] = {}
    seed_counter = int(variant_args.seed) + 100

    _best_seq_path = getattr(variant_args, "best_sequences_json", "") or ""
    if _best_seq_path:
        root, ext = os.path.splitext(_best_seq_path)
        _best_seq_path = f"{root}.{variant}{ext or '.json'}"
    else:
        wb_dir = getattr(getattr(wandb, "run", None), "dir", None)
        if wb_dir:
            _best_seq_path = os.path.join(wb_dir, "best_sequences.json")
        else:
            _best_seq_path = os.path.abspath(f"best_sequences-{variant}.json")
    _best_seq_every = int(getattr(variant_args, "best_sequences_every", 0) or 0)

    # ------------------------------------------------------------------
    # MOGFN Pareto archive: non-dominated set of sampled terminal solutions
    # over the preference-channel objectives, each kept with the (vertex, pair,
    # factor) sequence that produced it. Dumped to mogfn_pareto_front.json so
    # the front is recoverable in the same format as cmorl_pareto_front.json.
    # ------------------------------------------------------------------
    from alphagrad.approx.common.reward_scaling import REWARD_INDEX as _RIDX
    _obj_names = [
        s.strip()
        for s in str(getattr(variant_args, "preference_channels", "")).split(",")
        if s.strip()
    ]
    _obj_idx = [_RIDX[n] for n in _obj_names]
    # Shared Pareto-front archive (single implementation in common.pareto_archive,
    # reused by cmorl_ray/ppo_ray).
    _archive = ParetoArchive(_obj_names, _obj_idx)
    _ep_box = [0]
    _pareto_extra = {
        "seq_format": (
            "[vertex, [<call>, ...]] per eliminated vertex, where <call> "
            "is one of diag(i, j, factor) / compress('kind', axis) / "
            "quant('dtype') — the full typed micro-action sub-episode"
        ),
    }
    _pareto_json = os.path.join(
        os.path.dirname(_best_seq_path) or os.getcwd(),
        f"mogfn_pareto_front.{variant}.json",
    )
    _all_cand_json = os.path.join(
        os.path.dirname(_best_seq_path) or os.getcwd(),
        f"mogfn_all_front_candidates.{variant}.json",
    )

    def _arch_add(sols) -> None:
        _archive.add_many(
            [(sol.get("obj"), sol.get("seq")) for sol in (sols or [])],
            _ep_box[0],
        )

    def _arch_hv() -> float:
        return _archive.hypervolume()

    def _dump_pareto() -> None:
        try:
            os.makedirs(os.path.dirname(_pareto_json) or ".", exist_ok=True)
            _archive.dump_front(_pareto_json, extra=_pareto_extra)
        except Exception as _exc:
            tqdm.write(f"  [MOGFN] pareto dump failed: {_exc}")

    def _dump_all_candidates() -> None:
        try:
            os.makedirs(os.path.dirname(_all_cand_json) or ".", exist_ok=True)
            _archive.dump_all_candidates(_all_cand_json)
        except Exception as _exc:
            tqdm.write(f"  [MOGFN] all-candidate dump failed: {_exc}")

    def _build_state() -> dict:
        return {
            "best_global_return": best_global_return,
            "best_global_ep": best_global_ep,
            "best_global_seq": best_global_seq,
            "best_global_rewards": dict(best_global_rewards),
            "best_global_weighted_split": dict(best_global_weighted_split),
            "best_per_reward": dict(best_per_reward),
        }

    pbar = tqdm(
        total=variant_args.episodes,
        desc=variant,
        disable=False,
        leave=True,
        ncols=180,
    )

    import time as _time_mod
    _t_train_start = _time_mod.time()
    _max_wall = float(getattr(args, "max_wall_seconds", 0.0) or 0.0)
    for ep in range(variant_args.episodes):
        # Wall-clock budget (--max-wall-seconds, 0 = unlimited): stop cleanly;
        # the final + per-episode dumps preserve everything explored so far.
        if _max_wall > 0 and (_time_mod.time() - _t_train_start) > _max_wall:
            tqdm.write(
                f"  [MOGFN] wall-clock budget ({_max_wall:.0f}s) reached at "
                f"ep {ep} — stopping; dumping final archive."
            )
            break
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
        tb_loss = stats.get("tb_loss", float("nan"))
        logZ_val = stats.get("logZ", float("nan"))
        bsize = stats.get("buffer_size", 0)
        tstep = stats.get("train_step", 0)

        if ep_best > best_global_return:
            best_global_return = ep_best
            best_global_seq = stats.get("best_seq")
            best_global_rewards = dict(stats.get("best_overall_rewards", {}))
            best_global_weighted_split = dict(stats.get("best_overall_weighted", {}))
            best_global_ep = ep

        for name, info in stats.get("best_per_reward", {}).items():
            prev = best_per_reward.get(name)
            if prev is None or info["raw_value"] > prev["raw_value"]:
                best_per_reward[name] = {**info, "ep": ep}

        # Update + persist the MOGFN Pareto front (points + sequences).
        _ep_box[0] = ep
        _arch_add(stats.get("terminal_solutions", []))
        _dump_pareto()
        _dump_all_candidates()

        # ---- Progress bar (one line, updated in place) ----
        pbar.update(1)
        pbar.set_description(
            f"{variant} "
            f"best:{best_global_return:+.3g} "
            f"ep_best:{ep_best:+.3g} "
            f"mean:{ep_mean:+.3g} "
            f"ent:{ent_mean:.3f} "
            f"tb:{tb_loss:.3g} logZ:{logZ_val:+.3g} "
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
                f"best={best_global_return:+.4g}(ep{best_global_ep}) "
                f"mean={ep_mean:+.4g} "
                f"tb={tb_loss:.3g} logZ={logZ_val:+.3g} "
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
            "tb_loss": tb_loss,
            "logZ": logZ_val,
            "mean_log_pf": stats.get("mean_log_pf", float("nan")),
            "mean_log_pb": stats.get("mean_log_pb", float("nan")),
            "mean_log_R": stats.get("mean_log_R", float("nan")),
            "mean_terminal_reward": stats.get("mean_terminal_reward", float("nan")),
            "total_loss": stats.get("total_loss", float("nan")),
            "beta": stats.get("beta", float("nan")),
            "buffer_size": bsize,
            "train_step": tstep,
        }
        for name, val in stats.get("per_reward_means", {}).items():
            log_dict[f"reward_mean/{name}"] = val
        for name, info in best_per_reward.items():
            log_dict[f"best_per_channel/{name}"] = info["raw_value"]
            log_dict[f"best_per_channel_weighted_total/{name}"] = info[
                "weighted_total"
            ]
        for name, v in best_global_rewards.items():
            log_dict[f"best_overall_reward/{name}"] = v
        for name, v in best_global_weighted_split.items():
            log_dict[f"best_overall_weighted/{name}"] = v
        # Forward pool / tokenization telemetry verbatim.
        for k, v in stats.items():
            if k.startswith("pool/") or k.startswith("tokenization/"):
                log_dict[k] = v
        wandb.log(log_dict)

        # Periodic best-sequences JSON snapshot.
        if _best_seq_every > 0 and (
            (ep + 1) % _best_seq_every == 0 or ep == variant_args.episodes - 1
        ):
            _state_now = _build_state()
            _dump_best_sequences_json(_state_now, _best_seq_path)
            wandb.log(_build_best_seq_wandb_payload(_state_now, ep=ep))

    pbar.close()
    _dump_pareto()
    _dump_all_candidates()

    # ---- Final per-variant summary -----------------------------------
    tqdm.write(f"\n========== [{variant}] FINAL ==========")
    tqdm.write(
        f"  MOGFN Pareto front: {len(_archive.pts)} points, hv={_arch_hv():.6g} "
        f"-> {_pareto_json}"
    )
    tqdm.write(
        f"  MOGFN all-time front candidates: {len(_archive.all_candidates)} "
        f"-> {_all_cand_json}"
    )
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

    # Final wandb summary entries (mirror mu0_ray exactly).
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

    _final_state = _build_state()
    _final_path = _dump_best_sequences_json(_final_state, _best_seq_path)
    if _final_path:
        tqdm.write(f"  best-sequences JSON written: {_final_path}")
        try:
            from alphagrad.approx.common.render_sequence import (
                render_best_sequences_json,
            )
            for k, v in render_best_sequences_json(_final_path).items():
                summary[f"final/{k}/repr"] = v
        except Exception as exc:
            tqdm.write(f"  [render] best-sequence repr failed: {exc}")
    try:
        _final_table = _build_best_seq_wandb_table(_final_state)
        if _final_table is not None:
            summary["final/best_sequences_table"] = _final_table
    except Exception as exc:
        tqdm.write(
            f"  [wandb] best-sequences Table construction failed: {exc}"
        )
    summary.update(
        _build_best_seq_wandb_payload(
            _final_state, ep=variant_args.episodes - 1,
        )
    )
    try:
        wandb.log(summary)
    except Exception as exc:
        tqdm.write(
            f"  [wandb] final wandb.log failed: {exc}; retrying without Table."
        )
        summary.pop("final/best_sequences_table", None)
        try:
            wandb.log(summary)
        except Exception as exc2:
            tqdm.write(f"  [wandb] retry also failed: {exc2}")

    if variant_args.replay_checkpoint_path:
        ray.get(
            spmd_actor.checkpoint_replay.remote(variant_args.replay_checkpoint_path)
        )

    try:
        wandb.finish()
    except Exception as exc:
        tqdm.write(f"  [wandb] finish failed: {exc}")

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

    # Shared compile cache — spawned once for the whole variant sweep
    # so the cache persists across variants and warms up faster on the
    # second+ variant. Driver kills it on exit.
    _spawn_compile_cache(
        max_size=int(getattr(args, "compile_cache_size", 512)),
    )

    sweep = (
        [v.strip() for v in args.variant_sweep.split(",") if v.strip()]
        if args.variant_sweep.strip()
        else [args.variant]
    )
    # SEED-style off-policy needs a buffer; if the user left it disabled
    # default to 1024 (same convention as mu0_ray.py:509-510).
    if args.replay_buffer_size <= 0:
        args.replay_buffer_size = 1024

    for variant in sweep:
        _assert_jax_free()
        _run_one_variant(args, variant)

    _kill_compile_cache()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
