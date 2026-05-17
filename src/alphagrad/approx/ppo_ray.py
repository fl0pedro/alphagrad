"""JAX-free driver for the Ray-actor PPO trainer.

First-cut counterpart to `mu0_ray.py`. The driver:

* spawns one ``PPOActor`` (the JAX trainer; single-GPU for now),
* spawns N ``CpuApproximationActor``s (the tokenizer / approximation
  pool),
* hands the CPU pool to the trainer,
* loops ``--episodes`` × ``run_rollout_and_train.remote(...)`` calls,
* logs to wandb and tqdm.

Driver process stays JAX-free so the Ray actors are the only place
JAX initialises a device — same hygiene as `mu0_ray.py`.

Compared to `mu0_ray.py` this driver is intentionally sparse: no
variant sweep, no calibration phase, no per-channel best tracking, no
replay-checkpoint plumbing. Those features can land later; for the
first cut we want a minimal flight check that the rollout + update
loop runs end-to-end against the external tokenizer pool.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

# Plain threading lock so tqdm doesn't leak a named POSIX semaphore on
# signal-kill — mirrors `mu0_ray.py`.
import threading as _threading
tqdm.set_lock(_threading.RLock())

from alphagrad.approx.ppo_args import make_argparser  # noqa: E402
from alphagrad.approx.common.calibration import run_calibration  # noqa: E402
from alphagrad.approx.common.compile_cache import (  # noqa: E402
    kill_coordinator as _kill_compile_cache,
    spawn_coordinator as _spawn_compile_cache,
)
from alphagrad.approx.common.ray_runtime import (  # noqa: E402
    _assert_jax_free,
    _disable_ray_uv_autodetect,
    add_common_ray_args,
)
from alphagrad.approx.common.reward_scaling import (  # noqa: E402
    build_best_sequences_wandb_payload,
    build_best_sequences_wandb_table,
    build_wandb_log_dict,
    dump_best_sequences_json,
    format_milestone_line,
    init_running_bests,
    update_running_bests,
)


def _extend_argparser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Driver-only knobs not present in `ppo.make_argparser`."""
    add_common_ray_args(p)
    p.add_argument(
        "--actor-num-gpus",
        type=float,
        default=1.0,
        help="GPUs reserved for the PPO actor (1 for the first cut; raise "
        "when SPMD sharding lands).",
    )
    p.add_argument(
        "--lagrangian-warmup-eps",
        type=int,
        default=20,
        help="Episodes at the start of training where the Lagrangian "
             "violations are computed for logging but the penalty does NOT "
             "flow into the advantage and the multipliers are not updated. "
             "Without this warm-up, a cold cosine_sim constraint (mean ~0.05) "
             "with a 0.5 threshold pushes the multiplier into the tens within "
             "100 episodes, killing exploration via overwhelming penalty.",
    )
    p.add_argument(
        "--lagrangian-multiplier-max",
        type=float,
        default=1.0,
        help="Upper clip on each Lagrangian multiplier. Stops dual ascent "
             "from dominating the PPO objective when a constraint is "
             "structurally hard to satisfy in the early policy.",
    )
    return p


def _run(args) -> int:
    import ray
    import wandb
    from alphagrad.approx.ppo_ray_actors import PPOActor
    from alphagrad.approx.cpu_approx_actors import CpuApproximationActor

    args_dict = vars(args)
    wandb.init(
        project=args.wandb_project,
        name=args.name,
        config=args_dict,
        mode="disabled" if args.wandb == "disabled" else args.wandb,
        reinit=True,
    )

    # Disable XLA preallocation in actor processes — the GPU actor needs
    # headroom for transient compile/exec buffers, and the CPU actors
    # don't benefit from preallocation at all.
    actor_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    }
    # Propagate any ``ALPHAGRAD_*`` env vars (debug switches like
    # ``ALPHAGRAD_DEBUG_QUALITY``, ``ALPHAGRAD_LEAK_PROFILE``,
    # ``ALPHAGRAD_SKIP_COST_ANALYSIS``) so a single ``sbatch --export``
    # reaches the CpuApproximationActor processes too. Without this
    # the actors run with stock env and the driver-side toggles
    # silently no-op.
    for k, v in os.environ.items():
        if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_"):
            actor_env[k] = v
    actor_kwargs = {
        "num_gpus": args.actor_num_gpus,
        "runtime_env": {"env_vars": actor_env},
    } if args.actor_num_gpus > 0 else {
        "runtime_env": {"env_vars": actor_env},
    }

    # Cluster-wide shared compile cache. Spawned BEFORE any CPU worker
    # so when those workers' env._callback runs cached_compile() on
    # the very first step, the coordinator is already registered.
    # Named actor, lifetime=detached — the driver kills it on exit.
    _spawn_compile_cache(
        max_size=int(getattr(args, "compile_cache_size", 512)),
    )

    actor = PPOActor.options(**actor_kwargs).remote(
        args_dict, int(args.seed),
    )

    # Mirror mu0_ray.py: pass the cpu actor's `.options(...)` kwargs to
    # `init_server` so the SPMD/PPO actor can respawn killed workers
    # via the same factory pattern after a timeout. Keep these in sync
    # with the .options(...) below.
    cpu_actor_env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    # Pass ALPHAGRAD_* / JAX_COMPILATION_* debug toggles through to the
    # CPU actor processes — same rationale as ``actor_env`` above.
    for k, v in os.environ.items():
        if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_"):
            cpu_actor_env[k] = v
    cpu_actor_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": cpu_actor_env},
    }
    cpu_workers = [
        CpuApproximationActor.options(**cpu_actor_options).remote(
            args_dict, variant=None, actor_id=i,
        )
        for i in range(args.num_cpu_workers)
    ]

    # Construct the timeout-bounded pool. `init_server` also internally
    # calls `init_worker` if the worker hasn't been built yet, so we
    # don't need a separate `init_worker` call. The pool kicks in
    # automatically — `_fan_out_tokenize` routes through it.
    ray.get(actor.init_server.remote(
        cpu_workers,
        callback_timeout_s=float(args.cpu_callback_timeout),
        initial_timeout_s=float(args.cpu_callback_initial_timeout),
        warm_after=int(args.cpu_callback_warm_after),
        recycle_every=int(args.cpu_worker_recycle_every),
        cpu_actor_options=cpu_actor_options,
        starting_actor_id=int(args.num_cpu_workers),
    ))
    ray.get(
        [actor.ready.remote()] + [c.ready.remote() for c in cpu_workers]
    )
    # Phase 4c: warm the CPU pool BEFORE calibration so the zero-pref
    # rollouts don't time out and poison `reward_vec_means`. Same
    # ordering as `mu0_ray.py` post-refactor.
    if cpu_workers:
        ray.get([c.compile_approximations.remote() for c in cpu_workers])
    print(
        f"  actors spawned + JIT-warm in {time.time() - args.t_start:.1f}s"
    )

    # Calibration is the pre-training pass that computed
    # ``1/|symlog(mean)|`` per-channel scaling weights from a random-
    # policy rollout. With ``--advantage-norm gdpo`` (the new default),
    # per-minibatch z-scoring inside the loss subsumes this magnitude
    # rescaling, so calibration is a no-op — skip it to save the
    # warm-up time. The legacy ``--advantage-norm scalar`` path still
    # benefits from calibrated weights (its global advantage z-score
    # operates on the weighted-sum reward, which depends on
    # cross-channel magnitude).
    if (
        getattr(args, "advantage_norm", "gdpo") == "scalar"
        and getattr(args, "calibrate_steps", 0) > 0
    ):
        run_calibration(actor, args, args.calibrate_steps, after_warmup=True)
    elif getattr(args, "calibrate_steps", 0) > 0:
        print(
            "  [calibration] skipped under --advantage-norm gdpo "
            "(per-channel z-scoring is self-calibrating)."
        )

    seed_counter = int(args.seed) + 100
    state = init_running_bests()

    # Curriculum sequencing. ``--variant full_curriculum`` with empty
    # ``--curriculum`` auto-expands into the 7-stage schedule
    # ``compute_seven_stage_curriculum`` returns:
    #   ve_only → rot1_simple → rot2_simple → all_simple →
    #   rot1_difficult → rot2_difficult → full
    # with geometric pacing (1:2:4:8:16:32:256 × N) and per-trainer
    # floors. See CURRICULUM.md for the full design.
    #
    # Manual ``--curriculum stage1:N1,...`` specs override the
    # auto-expansion (back-compat with the legacy 3-stage default and
    # any custom schedule).
    #
    # Per-episode round-robin: rotation stages (rot1_*, rot2_*) emit a
    # different concrete variant per episode (cycling through their
    # rotation slots). Non-rotation stages emit the same variant for
    # all their episodes.
    from alphagrad.approx.variants import (
        _parse_curriculum,
        compute_seven_stage_curriculum,
        compute_variant_at_episode,
    )
    curriculum_spec = getattr(args, "curriculum", "") or ""
    variant_name = getattr(args, "variant", "custom")
    if variant_name == "full_curriculum" and not curriculum_spec.strip():
        _stages = compute_seven_stage_curriculum(args.episodes, "ppo")
        curriculum_spec = ",".join(f"{n}:{k}" for n, k in _stages)
        print(
            f"  [curriculum] auto-expanded full_curriculum (7-stage): "
            f"{curriculum_spec}"
        )
    curriculum_stages = _parse_curriculum(curriculum_spec)
    if curriculum_stages:
        total_stage_eps = sum(n for _, n in curriculum_stages)
        if total_stage_eps != args.episodes:
            print(
                f"  [curriculum] WARNING: sum of stage episodes "
                f"({total_stage_eps}) != --episodes ({args.episodes}); "
                f"final stage will absorb the remainder."
            )
    current_variant_name: str | None = None
    current_stage_name: str | None = None

    # Resolve the best-sequences JSON path. Empty (default) → place it
    # next to the wandb run files so a single run dir holds both the
    # event log and the running per-channel bests. Falls back to
    # ``./best_sequences.json`` if wandb is disabled (no run dir).
    best_seq_json_path = getattr(args, "best_sequences_json", "") or ""
    if not best_seq_json_path:
        wb_dir = getattr(getattr(wandb, "run", None), "dir", None)
        if wb_dir:
            best_seq_json_path = os.path.join(wb_dir, "best_sequences.json")
        else:
            best_seq_json_path = os.path.abspath("best_sequences.json")
    best_seq_every = int(getattr(args, "best_sequences_every", 0) or 0)

    pbar = tqdm(
        total=args.episodes,
        desc="ppo_ray",
        disable=False,
        leave=True,
        ncols=180,
    )

    for ep in range(args.episodes):
        seed_counter += 1

        # Curriculum stage / variant transition. ``compute_variant_at_episode``
        # returns the current stage + the concrete variant for this
        # episode (rotation stages cycle through their slots per
        # episode). When EITHER the stage OR the variant changes we
        # tell the worker to update its masks. The agent weights
        # survive transitions (only masks change).
        if curriculum_stages:
            stage_now, variant_now, _within = compute_variant_at_episode(
                ep, curriculum_stages,
            )
            if stage_now != current_stage_name:
                tqdm.write(
                    f"  [curriculum] ep={ep + 1}: entering stage "
                    f"'{stage_now}'"
                )
                current_stage_name = stage_now
            if variant_now != current_variant_name:
                stage_info = ray.get(
                    actor.set_variant_masks.remote(variant_now)
                )
                tqdm.write(
                    f"  [curriculum]   ep={ep + 1} variant='{variant_now}' "
                    f"op={stage_info['op_type_legal_count']}/4 "
                    f"factor={stage_info['factor_legal_count']}/{len(stage_info)} "
                    f"quant={stage_info.get('quant_dtype_legal_count', '-')}"
                )
                current_variant_name = variant_now

        stats = ray.get(actor.run_rollout_and_train.remote(seed_counter))

        # `mean_return` / `best_return` come straight from the worker's
        # raw weighted per-env sum (Phase 2 fix at ppo_ray_worker.py).
        ep_mean = stats.get("mean_return", float("nan"))
        ploss = stats.get("ppo_loss", float("nan"))
        vloss = stats.get("value_loss", float("nan"))
        ent = stats.get("entropy", float("nan"))

        update_running_bests(state, stats, ep)

        pbar.update(1)
        pbar.set_description(
            f"ppo_ray "
            f"best:{state['best_global_return']:+.3g}(ep{state['best_global_ep']}) "
            f"mean:{ep_mean:+.3g} "
            f"loss(p/v):{ploss:.2g}/{vloss:.2g} "
            f"ent:{ent:.3f} "
            f"nan_skip:{int(stats.get('nan_skip_count', 0))}"
        )

        if (ep + 1) % 5 == 0 or ep == args.episodes - 1:
            tqdm.write(
                format_milestone_line("ppo_ray", ep, args.episodes, stats, state)
            )

        log_dict = build_wandb_log_dict(stats, state, ep)
        # Carry over the legacy `entropy` key for plot continuity with
        # earlier runs (build_wandb_log_dict uses `entropy_mean`).
        log_dict.setdefault("entropy", ent)
        wandb.log(log_dict)

        # Periodic JSON snapshot + wandb scalar payload of the
        # running per-channel + overall bests. The full
        # ``wandb.Table`` lands once at run-end (logging tables
        # every-N-eps clutters the run UI). ``best_seq_every == 0``
        # disables the periodic write — final snapshot still happens
        # in the FINAL block below.
        if best_seq_every > 0 and (
            (ep + 1) % best_seq_every == 0 or ep == args.episodes - 1
        ):
            dump_best_sequences_json(state, best_seq_json_path)
            wandb.log(
                build_best_sequences_wandb_payload(state, ep=ep)
            )

    pbar.close()

    tqdm.write("\n========== FINAL ==========")
    tqdm.write(
        f"  best return: {state['best_global_return']:+.6g}  "
        f"(ep {state['best_global_ep']})"
    )
    if state["best_global_rewards"]:
        tqdm.write("  best-overall-trajectory per-channel rewards (raw):")
        for name in sorted(state["best_global_rewards"]):
            raw = state["best_global_rewards"][name]
            wt = state["best_global_weighted_split"].get(name, 0.0)
            tqdm.write(f"    {name:<18s}  raw={raw:+.4g}   weighted={wt:+.4g}")
    # Final JSON + wandb dump of the running bests. The JSON file is
    # the canonical record (durable, easy to diff across runs); the
    # wandb Table is a UI nicety so the bests show up in the run page
    # without having to download the file.
    final_path = dump_best_sequences_json(state, best_seq_json_path)
    if final_path:
        tqdm.write(f"  best-sequences JSON written: {final_path}")
    final_payload = build_best_sequences_wandb_payload(
        state, ep=args.episodes - 1
    )
    final_payload["final/best_return"] = state["best_global_return"]
    final_payload["final/best_ep"] = state["best_global_ep"]
    # The Table goes through wandb's artifact upload path which on some
    # wandb configurations (`base_url='redacted'` in
    # ~/.config/wandb/settings) hits a pydantic-v2 URL validation
    # error and crashes the driver AFTER all training has completed
    # (see job 45278 in slurm/logs). The JSON dump above is the
    # canonical record; the Table is decorative. Wrap in try/except so
    # the wandb side-effect can never kill the training run.
    try:
        final_table = build_best_sequences_wandb_table(state)
        if final_table is not None:
            final_payload["final/best_sequences_table"] = final_table
    except Exception as exc:
        tqdm.write(
            f"  [wandb] best-sequences Table construction failed: {exc}; "
            f"falling back to scalar-only payload (JSON dump on disk "
            f"still has the full data)."
        )
    try:
        wandb.log(final_payload)
    except Exception as exc:
        # If wandb.log itself throws (e.g., from the Table → artifact
        # → pydantic URL validation chain), log the error and continue
        # to wandb.finish() so the run still closes cleanly.
        tqdm.write(
            f"  [wandb] final wandb.log failed: {exc}; continuing to "
            f"wandb.finish() to close the run."
        )
        # Retry without the Table (most common failure mode).
        final_payload.pop("final/best_sequences_table", None)
        try:
            wandb.log(final_payload)
        except Exception as exc2:
            tqdm.write(f"  [wandb] retry without Table also failed: {exc2}")
    try:
        wandb.finish()
    except Exception as exc:
        tqdm.write(f"  [wandb] finish failed: {exc}")

    ray.kill(actor, no_restart=True)
    for c in cpu_workers:
        ray.kill(c, no_restart=True)
    # Tear down the shared compile cache last so any in-flight
    # ``cached_compile`` calls finish before the coordinator dies.
    _kill_compile_cache()
    return 0


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

    rc = _run(args)
    ray.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
