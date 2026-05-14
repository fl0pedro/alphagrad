"""Ray driver for off-policy Gumbel MuZero with a per-variant sweep.

JAX-free by construction — only imports :mod:`argparse`, :mod:`copy`,
:mod:`numpy`, :mod:`ray`, :mod:`wandb`, plus the JAX-free helpers from
:mod:`alphagrad.approx.variants` / :mod:`alphagrad.approx.mu0_args` and
the Ray actor *shims* in :mod:`alphagrad.approx.mu0_ray_actors`. The
shim defers the actual JAX-side worker import to inside each actor's
``__init__``, so JAX never enters this process.

Run with::

    python -m alphagrad.approx.mu0_ray --variant-sweep ve_only,diag_gcd,...

The default sweep covers the five non-``custom`` presets plus
``full_curriculum``; pass ``--variant-sweep <single>`` for a one-variant
run or ``--variant-sweep ""`` to honour ``--variant`` for a single
training run.

Per-variant flow:

    1. Apply variant preset, expand ``full_curriculum`` to its 3-stage
       curriculum if ``--curriculum`` is empty.
    2. Spawn one :class:`LearnerActor` (claims one GPU by default) and
       N :class:`RolloutActor` (CPU by default, fractional GPU via
       ``--rollout-actor-gpus``).
    3. Broadcast initial params to all rollout actors.
    4. (Optional) Calibration: each rollout actor runs a few zero-pref
       rollouts; the driver gathers the per-channel mean reward vec and
       rescales ``reward_weights`` to ``1 / mean_abs_symlog(reward)``
       so wide-magnitude reward channels contribute on a comparable
       scale.
    5. Training loop: pending rollouts are continually re-queued
       round-robin. As each completes, the driver pushes the trajectory
       to the learner and triggers ``--learner-train-every`` training
       steps. Every ``--actor-sync-every`` episodes the driver refreshes
       all rollout actors with the learner's latest params.
    6. Tear down actors and move to the next variant.

Each variant gets its own wandb run via ``wandb.init(reinit=True)``;
losses and rollout stats are merged on the driver and ``wandb.log``-ed
once per episode.
"""

from __future__ import annotations

import argparse
import copy
import math
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


# Default variants to sweep: everything except `custom` (the no-op
# placeholder). `full_curriculum` is included at the end so the run that
# spans the longest curriculum lands last.
_DEFAULT_VARIANT_SWEEP = (
    "ve_only,diag_gcd,diag_factor,compress,full,full_curriculum"
)


# Reward indices the calibration step refuses to symlog (mirrors mu0).
# Hardcoded here so the driver doesn't have to import the JAX-side
# constant from env.py. Kept in sync by hand — if the env adds reward
# channels, update both this constant and env.py.
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
        "--num-rollout-actors", type=int, default=4,
        help="Number of asynchronous rollout actors per training run.",
    )
    p.add_argument(
        "--rollout-actor-gpus", type=float, default=0.0,
        help="Fractional GPU per rollout actor (0 = CPU). Ray manages "
             "CUDA_VISIBLE_DEVICES per actor.",
    )
    p.add_argument(
        "--learner-gpus", type=float, default=1.0,
        help="Number of GPUs reserved for the single learner actor. "
             "Set to 0 to run the learner on CPU (slow).",
    )
    p.add_argument(
        "--learner-train-every", type=int, default=1,
        help="Train-step count to trigger per received rollout trajectory. "
             "Higher = more learner updates per fresh rollout (more "
             "off-policy reuse).",
    )
    p.add_argument(
        "--actor-sync-every", type=int, default=4,
        help="Refresh each rollout actor's params from the learner every N "
             "completed episodes.",
    )
    p.add_argument(
        "--ray-address", type=str, default="",
        help="Address of an existing Ray cluster (passed to ray.init). "
             "Empty = local cluster.",
    )
    p.add_argument(
        "--variant-sweep", type=str, default=_DEFAULT_VARIANT_SWEEP,
        help="Comma-separated variant list to train sequentially. Empty "
             "string = honour --variant for a single run.",
    )
    p.add_argument(
        "--wandb-project", type=str, default="dsnn-vertex",
        help="wandb project name. Each variant becomes its own run with "
             "name ``<--name>-<variant>``.",
    )
    return p


def _assert_jax_free() -> None:
    """Fail fast if anything in this process has imported JAX.

    Catches accidental imports through indirect dependency chains. The
    driver only ever holds numpy arrays / pickled pytrees; JAX device
    interaction must happen exclusively in actor processes.
    """
    leaked = sorted(
        m for m in sys.modules if m == "jax" or m.startswith("jax.")
    )
    if leaked:
        raise RuntimeError(
            "jax has leaked into the driver process — this will deadlock "
            "Ray actor spawn on platforms that use fork. Leaked modules: "
            f"{leaked[:5]}..."
        )


def _ensure_curriculum_for_variant(args, variant: str) -> None:
    """Expand ``full_curriculum`` into an explicit ``--curriculum`` spec.

    Mirrors mu0.main()'s auto-curriculum logic (mu0.py:789-810) but
    done driver-side so the actors get a fully-resolved curriculum
    string. For other variants this is a no-op.
    """
    if variant == "full_curriculum" and not args.curriculum.strip():
        stages = _default_full_curriculum(args.episodes)
        args.curriculum = ",".join(f"{name}:{n}" for name, n in stages)
        total = sum(n for _, n in stages)
        args.episodes = total
        print(
            f"  [{variant}] auto-curriculum: "
            + " → ".join(f"{name}:{n}" for name, n in stages)
            + f"  (total {total} episodes)"
        )


def _run_calibration(args, rollouts, num_rollouts_per_actor):
    """Driver-side reward-scale calibration.

    Asks each rollout actor for ``num_rollouts_per_actor`` zero-pref
    rollouts under the untrained policy, gathers the per-channel mean
    reward vectors on the host, computes a scaling vector
    ``1 / max(mean_abs_symlog, eps)`` per channel (cosine_sim excluded,
    matches mu0.py:1551-1610), then broadcasts the new weights.
    """
    print(f"  [calibration] {num_rollouts_per_actor} rollouts per actor across {len(rollouts)} actors...")
    abs_sum = np.zeros((_NUM_REWARDS,), dtype=np.float32)
    n_total = 0
    refs = [
        a.reward_vec_means.remote(
            rng_seed=int(args.seed) + 7 * (i + 1),
            num_rollouts=num_rollouts_per_actor,
        )
        for i, a in enumerate(rollouts)
    ]
    import ray
    means_per_actor = ray.get(refs)
    for mean_vec in means_per_actor:
        mean_vec = np.asarray(mean_vec, dtype=np.float32)
        abs_sum += np.abs(_symlog_np(mean_vec))
        n_total += 1
    if n_total == 0:
        return
    mean_abs = abs_sum / float(n_total)
    # Don't rescale cosine_sim (it's already bounded).
    mean_abs[_COSINE_SIM_IDX] = 1.0
    # Build the new reward-weight vector by scaling the current lambdas.
    # We don't actually have direct access to reward_weights on the
    # driver (it's a JAX array inside each actor), but we know the
    # init mapping from --lambda-cmp / --lambda-mem / --lambda-frob.
    # Simpler approach: compute the calibration scaling and ship it via
    # set_reward_weights — each actor multiplies its current
    # reward_weights by this scaling.
    scaling = 1.0 / np.maximum(mean_abs, 1e-3)
    # The actors' set_reward_weights expects an absolute weights vector,
    # not a scaling. Reconstruct the unscaled weights from args.
    abs_weights = _initial_reward_weights(args) * scaling
    for a in rollouts:
        a.set_reward_weights.remote(abs_weights)
    print(
        "  [calibration] scaling per channel: "
        + ", ".join(
            f"{_REWARD_NAMES[i]}={scaling[i]:.3g}"
            for i in range(_NUM_REWARDS)
            if abs(scaling[i] - 1.0) > 1e-6
        )
    )


# Maps args.cmp_type / args.mem_type onto the canonical 8-vec channel
# names. Duplicated from mu0.py:179-189 so the driver doesn't import.
_CMP_TYPE_TO_REWARD = {
    "graphax": "muls_adds_fmas",
    "flops": "flops",
    "latency": "latency_ns",
}
_MEM_TYPE_TO_REWARD = {
    "graphax": "max_io_sum",
    "bytes_accessed": "bytes_accessed",
    "peak_memory": "peak_memory",
}


def _initial_reward_weights(args) -> np.ndarray:
    """Reconstruct the un-calibrated initial reward_weights on the host."""
    w = np.zeros((_NUM_REWARDS,), dtype=np.float32)
    if "cmp" in args.rewards:
        w[_REWARD_NAMES.index(_CMP_TYPE_TO_REWARD[args.cmp_type])] = (
            args.lambda_cmp
        )
    if "mem" in args.rewards:
        w[_REWARD_NAMES.index(_MEM_TYPE_TO_REWARD[args.mem_type])] = (
            args.lambda_mem
        )
    if "acc" in args.rewards:
        w[_COSINE_SIM_IDX] = 1.0
    if args.lambda_frob != 0.0:
        w[_REWARD_NAMES.index("frob_residual")] = args.lambda_frob
    return w


def _run_one_variant(args, variant: str) -> None:
    """Train a single variant: spawn actors, run loop, tear down."""
    import ray
    import wandb
    from alphagrad.approx.mu0_ray_actors import LearnerActor, RolloutActor

    variant_args = copy.deepcopy(args)
    variant_args.variant = variant
    _apply_variant_preset(variant_args)
    _ensure_curriculum_for_variant(variant_args, variant)

    args_dict = vars(variant_args)
    run_name = f"{args.name}-{variant}"
    wandb_mode = (
        "disabled" if args.wandb == "disabled" else args.wandb
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=args_dict,
        mode=wandb_mode,
        reinit=True,
    )

    print(f"\n========== variant: {variant} ==========")
    print(
        f"  factors={variant_args.factors}, max_rules={variant_args.max_rules}, "
        f"pin_rules={variant_args.pin_rules_to_exact}, "
        f"episodes={variant_args.episodes}, "
        f"num_actors={args.num_rollout_actors}"
    )

    learner_kwargs = {"num_gpus": args.learner_gpus} if args.learner_gpus > 0 else {}
    rollout_kwargs = (
        {"num_gpus": args.rollout_actor_gpus}
        if args.rollout_actor_gpus > 0 else {}
    )
    learner = LearnerActor.options(**learner_kwargs).remote(
        args_dict, variant, int(variant_args.seed),
    )
    rollouts = [
        RolloutActor.options(**rollout_kwargs).remote(
            args_dict, variant, i,
        )
        for i in range(args.num_rollout_actors)
    ]

    # Wait for all actors to finish their JIT warm-up by hitting the
    # cheap ``ready`` endpoint. This blocks until every actor has
    # finished __init__, which is when the first .remote() call would
    # otherwise stall.
    ready_refs = [learner.ready.remote()] + [
        a.ready.remote() for a in rollouts
    ]
    ray.get(ready_refs)
    print(f"  actors spawned and JIT-warm in {time.time() - args.t_start:.1f}s (cumulative)")

    # Initial param broadcast (sync, so the first rollout doesn't run
    # the random init).
    params_ref = learner.get_params_numpy.remote()
    ray.get([a.set_params_numpy.remote(params_ref) for a in rollouts])

    # Optional calibration phase.
    if variant_args.calibrate_steps > 0:
        per_actor = max(
            variant_args.calibrate_steps // max(len(rollouts), 1), 1,
        )
        _run_calibration(variant_args, rollouts, per_actor)

    # Steady-state training loop.
    pending: dict = {}
    for i, actor in enumerate(rollouts):
        seed = int(variant_args.seed) + 11 * (i + 1)
        ref = actor.rollout_one.remote(
            rng_seed=seed,
            preference_np=None,
            pin_rules=None,
            reset_env=True,
        )
        pending[ref] = (actor, seed)

    best_global_return = -float("inf")
    best_global_seq = None
    seed_counter = int(variant_args.seed) + 100 * args.num_rollout_actors
    total_episodes = variant_args.episodes

    for ep in range(total_episodes):
        if not pending:
            break
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        done = ready[0]
        actor, _prev_seed = pending.pop(done)
        try:
            traj_np, stats = ray.get(done)
        except Exception as exc:
            print(f"  rollout {ep} failed on actor: {exc}; skipping", flush=True)
            seed_counter += 1
            new_ref = actor.rollout_one.remote(
                rng_seed=seed_counter, preference_np=None,
                pin_rules=None, reset_env=True,
            )
            pending[new_ref] = (actor, seed_counter)
            continue

        # Push trajectory to learner (fire-and-forget).
        learner.add_trajectories.remote(traj_np)

        # Trigger learner train steps. We only block on the last
        # train_step so the wandb log has fresh numbers; the earlier
        # ones run in parallel with the next rollout.
        train_refs = [
            learner.train_step.remote()
            for _ in range(max(args.learner_train_every, 1))
        ]
        train_metrics = ray.get(train_refs[-1])

        # Re-queue this actor with a fresh seed.
        seed_counter += 1
        new_ref = actor.rollout_one.remote(
            rng_seed=seed_counter, preference_np=None,
            pin_rules=None, reset_env=True,
        )
        pending[new_ref] = (actor, seed_counter)

        # Periodic param resync (fire-and-forget — staleness is OK).
        if (ep + 1) % max(args.actor_sync_every, 1) == 0:
            params_ref = learner.get_params_numpy.remote()
            for a in rollouts:
                a.set_params_numpy.remote(params_ref)

        # Best-so-far bookkeeping.
        ep_best = stats.get("best_return", -float("inf"))
        if ep_best > best_global_return:
            best_global_return = ep_best
            best_global_seq = stats.get("best_seq")

        log_dict = {
            "episode": ep,
            "actor_id": stats.get("actor_id", -1),
            "best_return": best_global_return,
            "best_return_this_ep": ep_best,
            "mean_return": stats.get("mean_return", float("nan")),
            "buffer_size": train_metrics.get("buffer_size", 0),
            "train_step": train_metrics.get("train_step", 0),
        }
        if not train_metrics.get("skipped", False):
            for k in ("policy_loss", "value_loss", "reward_loss", "total_loss"):
                if k in train_metrics:
                    log_dict[k] = train_metrics[k]
        for name, val in stats.get("per_reward_means", {}).items():
            log_dict[f"reward/{name}"] = val
        wandb.log(log_dict)

        if (ep + 1) % 25 == 0 or ep == total_episodes - 1:
            print(
                f"  [{variant}] ep={ep + 1}/{total_episodes} "
                f"best={best_global_return:+.4g} "
                f"buf={train_metrics.get('buffer_size', 0)} "
                f"step={train_metrics.get('train_step', 0)}",
                flush=True,
            )

    # Final checkpoint of the buffer (if requested) and teardown.
    if variant_args.replay_checkpoint_path:
        try:
            ray.get(
                learner.checkpoint_replay.remote(
                    variant_args.replay_checkpoint_path,
                )
            )
        except Exception as exc:
            print(f"  replay checkpoint failed: {exc}")

    print(
        f"  [{variant}] done. best_return={best_global_return:+.4g}; "
        f"best_seq={best_global_seq}"
    )
    wandb.log({"best_global_return": best_global_return})
    wandb.finish()

    for a in rollouts:
        ray.kill(a, no_restart=True)
    ray.kill(learner, no_restart=True)


def main() -> int:
    p = make_argparser()
    p = _extend_argparser(p)
    args = p.parse_args()
    args.t_start = time.time()

    _assert_jax_free()

    # ---- Ray init ----------------------------------------------------------
    # Two things go wrong without the explicit setup below when this driver
    # is launched via ``uv run`` from a project that has editable installs
    # (alphagrad, graphax) in the active venv:
    #
    #   1. Ray auto-packages the working directory because it detects
    #      editable installs, ships a runtime_env zip to each worker, and
    #      the worker process re-runs ``uv sync`` in a fresh venv that
    #      doesn't have ``ray`` (because ``ray`` is in pyproject's deps
    #      only via our imperative ``uv pip install ray`` in
    #      ``sync_pgi15.sh`` — uv-lock-driven sync doesn't see it).
    #   2. ``uv run`` sets ``VIRTUAL_ENV`` to an absolute path but the
    #      worker subprocesses are spawned with a different cwd, so uv
    #      complains "VIRTUAL_ENV=... does not match the project env path
    #      `.venv`" and falls back to a different env that lacks ray.
    #
    # Fix: explicit empty ``runtime_env`` to disable the auto-packaging,
    # and clear ``VIRTUAL_ENV`` / ``UV_PROJECT_ENVIRONMENT`` from the
    # worker env so uv stops trying to be clever. Workers inherit
    # ``sys.executable`` from the driver, which is already the venv's
    # python — they'll find ray on its site-packages naturally.
    import ray
    init_kwargs = {}
    if args.ray_address:
        init_kwargs["address"] = args.ray_address
    os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    # Keep workers from inheriting the uv-shim-confusing env vars. Setting
    # the value to an empty string in runtime_env's env_vars clears the var
    # in the worker process (Ray's documented behaviour for env_vars).
    worker_env_vars = {
        "VIRTUAL_ENV": "",
        "UV_PROJECT_ENVIRONMENT": "",
        # Make sure the worker python doesn't accidentally use a uv shim.
        # Ray's default Python search uses sys.executable, which already
        # points at the venv's python; we just remove the path-prepended
        # uv shim if any.
    }
    ray.init(
        **init_kwargs,
        ignore_reinit_error=True,
        # Empty runtime_env disables auto-detection of editable installs
        # → no working_dir packaging → workers use the existing venv's
        # site-packages directly.
        runtime_env={"env_vars": worker_env_vars},
    )

    sweep = args.variant_sweep.strip()
    if sweep:
        variants_to_run = [v.strip() for v in sweep.split(",") if v.strip()]
        unknown = [v for v in variants_to_run if v not in VARIANT_PRESETS]
        if unknown:
            raise ValueError(
                f"Unknown variants in --variant-sweep: {unknown}. "
                f"Known: {list(VARIANT_PRESETS)}"
            )
    else:
        variants_to_run = [args.variant]

    print(
        f"mu0_ray driver: sweeping {len(variants_to_run)} variant(s) "
        f"sequentially: {variants_to_run}"
    )
    print(
        f"  num_rollout_actors={args.num_rollout_actors}, "
        f"learner_gpus={args.learner_gpus}, "
        f"rollout_actor_gpus={args.rollout_actor_gpus}, "
        f"train_every={args.learner_train_every}, "
        f"sync_every={args.actor_sync_every}"
    )

    # Force the new script to actually run Gumbel MuZero — that's the
    # whole point of this entry point. The arg is still accepted because
    # we share mu0_args.make_argparser, but we override it here so the
    # actor's rollout_fn unambiguously uses mctx.gumbel_muzero_policy.
    if args.mcts_mode != "gumbel":
        print(
            f"  note: --mcts-mode was '{args.mcts_mode}', overriding to 'gumbel' "
            "(mu0_ray only supports Gumbel MuZero)"
        )
        args.mcts_mode = "gumbel"

    # Off-policy replay is the whole point of this driver — if the user
    # didn't pick a buffer size, default to something proportional to the
    # fan-in capacity. mu0_args defaults to 0 because mu0.py runs on-policy
    # by default; here we flip that.
    if args.replay_buffer_size <= 0:
        # Aim for ~32 episodes worth of fresh rollouts before we wrap.
        default_size = max(32 * max(args.num_rollout_actors, 1), 1024)
        print(
            f"  note: --replay-buffer-size was {args.replay_buffer_size}, "
            f"defaulting to {default_size} (off-policy replay is mandatory)"
        )
        args.replay_buffer_size = default_size

    for variant in variants_to_run:
        _assert_jax_free()
        _run_one_variant(args, variant)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
