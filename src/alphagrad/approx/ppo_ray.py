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

import numpy as np

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
    REWARD_NAMES,
    build_best_sequences_wandb_payload,
    build_best_sequences_wandb_table,
    build_reward_weights,
    build_wandb_log_dict,
    dump_best_sequences_json,
    format_milestone_line,
    init_running_bests,
    update_running_bests,
)
from alphagrad.approx.common.pareto_archive import ParetoArchive  # noqa: E402


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
        "--use-placement-group", action="store_true",
        help="Reserve a Ray placement group per run: one {GPU: actor_num_gpus} "
        "bundle for the (same-node) SPMD PPO trainer + one {GPU: "
        "cpu_actor_num_gpus, CPU:1} bundle per measure actor, PACK strategy. "
        "Makes N concurrent multi-GPU runs tile deterministically across mixed "
        "GPU nodes (e.g. 4+8) instead of Ray's greedy placement stranding GPUs.",
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
    # Thread ALPHAGRAD_* policy switches (e.g. ALPHAGRAD_QUANT_ALLOWED,
    # ALPHAGRAD_SUBSTEP_NO_END) to the PPOActor explicitly via args_dict, which
    # Ray serializes reliably. runtime_env env_vars do NOT reliably reach the
    # actor's heads module at the time it samples the policy, so the import/lazy
    # env read saw stock env and the switches silently no-op'd. init_worker
    # re-applies these to os.environ before the policy is built.
    args_dict["_alphagrad_env"] = {
        k: v for k, v in os.environ.items() if k.startswith("ALPHAGRAD_")
    }
    wandb.init(
        project=args.wandb_project,
        entity=getattr(args, "wandb_entity", None) or None,
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
    # Optional per-run placement group: one bundle of {GPU: actor_num_gpus}
    # for the PPO trainer (a single bundle => same node, required for the SPMD
    # mesh / NCCL) + one {CPU:1, GPU: cpu_actor_num_gpus} bundle per measure
    # actor. PACK co-locates the run's GPUs and lets N runs tile mixed GPU
    # nodes deterministically (no greedy fragmentation stranding GPUs).
    _pg = None
    if getattr(args, "use_placement_group", False):
        from ray.util.placement_group import placement_group
        _gg = float(getattr(args, "cpu_actor_num_gpus", 0.0) or 0.0)
        _bundles = [{"GPU": float(args.actor_num_gpus)}]
        for _ in range(int(args.num_cpu_workers)):
            _b = {"CPU": 1.0}
            if _gg > 0:
                _b["GPU"] = _gg
            _bundles.append(_b)
        _pg = placement_group(_bundles, strategy="PACK")
        ray.get(_pg.ready())

    actor_kwargs = {
        # GPU-bound trainer: reserve 0 CPU slots so the GPU node can be
        # declared --num-cpus=0, which physically excludes the num_cpus=1
        # measurement actors from it -> all measurement lands on the single
        # CPU node (consistent cores = clean latency reward, no node-mix).
        "num_cpus": 0,
        "num_gpus": args.actor_num_gpus,
        "runtime_env": {"env_vars": actor_env},
    } if args.actor_num_gpus > 0 else {
        "runtime_env": {"env_vars": actor_env},
    }
    if _pg is not None:
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        actor_kwargs["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
            _pg, placement_group_bundle_index=0,
        )

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
        # GPU measurement (--exec-on-gpu) needs the actor to SEE cuda;
        # otherwise pin to CPU JAX so measurement runs on the CPU pool.
        "JAX_PLATFORMS": "cuda" if getattr(args, "exec_on_gpu", False) else "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        # NOTE: do NOT force single-thread here. The approx-Jacobian exec
        # scales ~linearly with cores (0.45s@64-core → ~80s@1-core), so
        # single-thread starves it. Each actor is pinned to a disjoint
        # core SLICE (see CpuApproximationActor.__init__) sized to
        # cores//num_workers, giving multi-core execs without cross-actor
        # oversubscription. Profiled 2026-06-06.
    }
    # Pass ALPHAGRAD_* / JAX_COMPILATION_* debug toggles through to the
    # CPU actor processes — same rationale as ``actor_env`` above.
    for k, v in os.environ.items():
        if k.startswith("ALPHAGRAD_") or k.startswith("JAX_COMPILATION_"):
            cpu_actor_env[k] = v
    # Single-core measurement (--cpu-cores-per-actor 1): also kill the math-lib
    # and XLA-Eigen threadpools so the pinned core runs TRULY single-threaded
    # (no intra-op threads contending on one core) — cleanest per-reading CV for
    # the CPU reward signal. The legacy "don't single-thread" note above was for
    # the full Jacobian (~80s single-core); in grad-mode the gradient exec is
    # ~160x cheaper, so single-thread no longer starves it.
    if int(getattr(args, "cpu_cores_per_actor", 0) or 0) == 1:
        cpu_actor_env.update({
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "XLA_FLAGS": (os.environ.get("XLA_FLAGS", "")
                         + " --xla_cpu_multi_thread_eigen=false").strip(),
        })
    cpu_actor_options = {
        "num_cpus": 1,
        "num_gpus": float(getattr(args, "cpu_actor_num_gpus", 0.0) or 0.0),
        "runtime_env": {"env_vars": cpu_actor_env},
    }
    # SPREAD the measurement actors across ALL nodes in the cluster
    # (e.g. the 2-node pooled setup: GPU node's CPUs + cpu1's CPUs)
    # rather than letting Ray bin-pack them onto a single node. Combined
    # with the per-node core allocator (--cpu-cores-shared), this uses
    # every core across both nodes for measurement. Opt-in via
    # ``--spread-cpu-actors`` so single-node runs keep the default
    # (locality-friendly) packing.
    if getattr(args, "spread_cpu_actors", False) and _pg is None:
        cpu_actor_options["scheduling_strategy"] = "SPREAD"
    cpu_workers = []
    for i in range(args.num_cpu_workers):
        _opts = dict(cpu_actor_options)
        if _pg is not None:
            # Bundle i+1 (bundle 0 is the PPO trainer). Co-locates this measure
            # actor with the run's reserved GPU.
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            _opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                _pg, placement_group_bundle_index=1 + i,
            )
        cpu_workers.append(
            CpuApproximationActor.options(**_opts).remote(
                args_dict, variant=None, actor_id=i,
            )
        )

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

    # Pareto archive — the multi-objective frontier this single-objective run
    # sweeps through. Objectives = the channels with nonzero scalarising weight
    # (i.e. exactly what --rewards/--cmp-type/--mem-type select; on CPU the mem
    # channel is the deterministic xla_peak_memory — see build_reward_weights).
    _rw = build_reward_weights(args)
    _obj_idx = [int(i) for i in np.nonzero(_rw)[0]]
    _obj_names = [REWARD_NAMES[i] for i in _obj_idx]
    pareto = ParetoArchive(_obj_names, _obj_idx) if _obj_idx else None
    _arch_dir = os.path.dirname(best_seq_json_path) or os.getcwd()
    pareto_front_path = os.path.join(_arch_dir, "ppo_pareto_front.json")
    pareto_all_path = os.path.join(_arch_dir, "ppo_all_front_candidates.json")
    _arch_extra = {"dynamic_substeps": bool(getattr(args, "dynamic_substeps", False))}

    def _feed_pareto(stats: dict, ep: int) -> None:
        """Admit this episode's candidate terminals (overall-best env + each
        per-channel best env) to the Pareto archive — each carries its full
        reward_vec (dict name->raw) and elimination sequence."""
        if pareto is None:
            return

        def _vec(d):  # name->value dict -> full reward_vec (NaN-filled)
            return [float(d.get(REWARD_NAMES[i], float("nan"))) for i in range(len(REWARD_NAMES))]

        sols = []
        bor = stats.get("best_overall_rewards")
        # ``is not None`` (not truthiness): an empty seq is a valid pure-VE
        # terminal — don't silently drop the scalar-best from the archive.
        if isinstance(bor, dict) and stats.get("best_seq") is not None:
            sols.append((_vec(bor), stats["best_seq"]))
        for entry in (stats.get("best_per_reward") or {}).values():
            if isinstance(entry, dict) and entry.get("all_raw") is not None and entry.get("seq") is not None:
                sols.append((_vec(entry["all_raw"]), entry["seq"]))
        pareto.add_many(sols, ep)

    def _dump_pareto() -> None:
        if pareto is None:
            return
        try:
            pareto.dump_front(pareto_front_path, extra=_arch_extra)
            pareto.dump_all_candidates(pareto_all_path, extra=_arch_extra)
        except Exception as _exc:
            tqdm.write(f"  [ppo_ray] pareto dump failed: {_exc}")

    pbar = tqdm(
        total=args.episodes,
        desc="ppo_ray",
        disable=False,
        leave=True,
        ncols=180,
    )

    for ep in range(args.episodes):
        seed_counter += 1

        # Wall-clock budget: stop cleanly once --max-wall-seconds has elapsed
        # (the final Pareto + best-sequence archives are dumped after the loop).
        # 0 = run to --episodes. Granularity = one episode.
        _mw = float(getattr(args, "max_wall_seconds", 0.0) or 0.0)
        if _mw > 0.0 and (time.time() - args.t_start) > _mw:
            tqdm.write(
                f"  [max-wall] {_mw:.0f}s budget reached at ep={ep} "
                f"(elapsed {time.time() - args.t_start:.0f}s) — stopping."
            )
            break

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
        _feed_pareto(stats, ep)

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
            _dump_pareto()
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
    # Loud CPU-pool-sentinel summary. If ``total_timeouts == 0`` across a
    # full training run, the sentinel-callback path in
    # ``ppo_ray_worker._fan_out_tokenize`` is dead code and can be
    # deleted (see CpuApproxPool.fetch_timeout_delta docstring). We log
    # the running total here as the canonical "did the sentinel fire
    # this run" signal so the cleanup decision is one line away.
    try:
        pool_stats = ray.get(actor.pool_stats.remote()) if hasattr(actor, "pool_stats") else None
    except Exception:
        pool_stats = None
    if pool_stats is not None:
        n_t = int(pool_stats.get("timeouts", 0))
        n_c = int(pool_stats.get("calls", 0))
        tag = "(sentinel path is dead code — can be dropped)" if n_t == 0 else ""
        tqdm.write(f"  cpu-pool timeouts total: {n_t} / {n_c} calls  {tag}")
        final_payload_extra = {
            "final/cpu_pool_timeouts_total": n_t,
            "final/cpu_pool_calls_total": n_c,
        }
    else:
        final_payload_extra = {}
    # Final JSON + wandb dump of the running bests. The JSON file is
    # the canonical record (durable, easy to diff across runs); the
    # wandb Table is a UI nicety so the bests show up in the run page
    # without having to download the file.
    final_path = dump_best_sequences_json(state, best_seq_json_path)
    if final_path:
        tqdm.write(f"  best-sequences JSON written: {final_path}")
    _dump_pareto()
    if pareto is not None:
        tqdm.write(
            f"  pareto archive: {len(pareto.pts)} front pts / "
            f"{len(pareto.all_candidates)} candidates over "
            f"{_obj_names} -> {pareto_front_path}"
        )
    final_payload = build_best_sequences_wandb_payload(
        state, ep=args.episodes - 1
    )
    final_payload["final/best_return"] = state["best_global_return"]
    final_payload["final/best_ep"] = state["best_global_ep"]
    final_payload.update(final_payload_extra)
    if final_path:
        try:
            from alphagrad.approx.common.render_sequence import (
                render_best_sequences_json,
            )
            for k, v in render_best_sequences_json(final_path).items():
                final_payload[f"final/{k}/repr"] = v
        except Exception as exc:
            tqdm.write(f"  [render] best-sequence repr failed: {exc}")
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
    # Cap Ray's LOGICAL cpu count for a local cluster. Ray prestarts one
    # python worker per logical CPU; on a high-core node (e.g. the
    # 384-core pgi15-cpu nodes) the default auto-detect spawns hundreds
    # of workers — a fork storm that hangs init (profiled 2026-06-06:
    # do_wait block + num_prestart_python_workers=384). We only need
    # enough logical CPUs to schedule the driver + num_cpu_workers actors
    # (+ headroom). The PHYSICAL cores stay fully available to the actors
    # via sched_setaffinity (set in CpuApproximationActor.__init__) — Ray
    # logical CPUs are a scheduling abstraction, not a core binding. Only
    # applied for a locally-started cluster (not when joining via
    # --ray-address, where the cluster sized itself).
    if not args.ray_address:
        ray_cpus = int(getattr(args, "num_cpu_workers", 4)) + 8
        init_kwargs["num_cpus"] = ray_cpus
    ray.init(**init_kwargs, ignore_reinit_error=True)

    rc = _run(args)
    ray.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
