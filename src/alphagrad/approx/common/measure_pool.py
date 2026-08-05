"""Shared GPU-pinned measurement pool — one implementation for every trainer.

WHY (2026-08-04 measurement audit): ``peak_bytes_in_use`` is a DEVICE-WIDE
counter, so a valid reading requires that nothing else allocates on the
measurement GPU for the duration of the clear -> exec -> read window. env.py
states the cost of violating it in its own comments: coefficient of variation
0.0000% clean vs **49.7% under a noisy neighbour**.

PPO satisfies this by giving every measure actor its own process with a single
pinned GPU (``CUDA_VISIBLE_DEVICES=idx+1``, ``num_gpus=0`` +
``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1``, ``ALPHAGRAD_MEASURE_ACTOR=1``)
and letting Ray serialize one task per actor. Measured over 245 episodes:
peak 4.17GB +/-0.6%, latency +/-1%, zero allocator warnings.

az_gumbel measured IN-PROCESS on the trainer's own GPU and got 8.35GB peak
(2x inflated) with a 3.63/20.0/5.05 ms latency spread, 210 allocator OOM
warnings and eventually a SIGKILL. This module lets it use PPO's mechanism
verbatim instead of maintaining a second, weaker one.
"""

from __future__ import annotations

import os
import sys


def spawn_measure_pool(args_dict: dict, *, n_actors: int, exec_on_gpu: bool,
                       timeout_s: float, max_tokens: int, num_rewards: int,
                       cosine_sim_idx: int, frob_residual_idx: int,
                       first_gpu: int = 1):
    """Create ``n_actors`` GPU-pinned measure actors and wrap them in a pool.

    ``first_gpu`` is the first device index handed to an actor; device 0 is
    reserved for the trainer (PPO's convention — letting Ray allocate handed
    the first actor the trainer's own device, which both hung the pool and
    destroyed the timing isolation).

    ``args_dict`` must describe the SAME measurement the caller's own env
    performs (``measure_grad``, ``per_face``, ``num_data_points``,
    ``reps_per_point``, ``latency_inner_reps``, target/dataset): the actor
    rebuilds its env from this dict, so a mismatch means the actor measures a
    different graph than the search acts on — silently.
    """
    import ray as _ray

    from alphagrad.approx.cpu_approx_actors import CpuApproximationActor
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    if not _ray.is_initialized():
        _ray.init(ignore_reinit_error=True, include_dashboard=False)

    def _actor_opts(idx: int) -> dict:
        rt = {"py_executable": sys.executable}
        if exec_on_gpu:
            rt["env_vars"] = {
                "CUDA_VISIBLE_DEVICES": str(first_gpu + idx),
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "ALPHAGRAD_MEASURE_ACTOR": "1",
            }
        return {"runtime_env": rt, "num_gpus": 0}

    d = dict(args_dict)
    # XLA sizes its Eigen pool from this BEFORE importing jax; leaving it at 1
    # makes every actor claim all cores (~48s/exec vs 0.45s).
    d["num_cpu_workers"] = int(n_actors)

    _next = [0]

    def _spawn(slot: int | None = None):
        _next[0] += 1
        s = (_next[0] - 1 if slot is None else int(slot)) % max(n_actors, 1)
        return CpuApproximationActor.options(**_actor_opts(s)).remote(
            d, variant=None, actor_id=_next[0])

    actors = [_spawn(i) for i in range(n_actors)]
    return CpuApproxPool(
        actors,
        timeout_s=float(timeout_s),
        initial_timeout_s=float(timeout_s) * 4.0,
        warm_after=3,
        respawn_factory=_spawn,
        max_tokens=int(max_tokens),
        num_rewards=int(num_rewards),
        cosine_sim_idx=int(cosine_sim_idx),
        frob_residual_idx=int(frob_residual_idx),
    )


def merge_pool_face_stats(pool, pf: dict | None = None) -> dict:
    """Merge the measure actors' per-face apply counters into ``pf``.

    WHY: ``env._PER_FACE_STATS`` is a plain module global written by the
    per-face legality hook inside the measurement ``_callback``. When the
    Ray measure pool is active that callback runs in the ACTOR processes,
    so the trainer's own dict is always empty, the ``if applied or
    skipped`` guard never fires and the ``approx_applied/*`` histogram is
    silently never logged.

    Polls every live actor, sums the additive counters, then RECOMPUTES
    ``applied_fraction`` -- the per-actor fractions are not additive.

    Best-effort by construction: a missing pool, a dead actor, or an old
    actor without ``consume_face_stats`` contributes nothing and never
    raises. Used by both ppo.py and az_gumbel.py so the two trainers emit
    identical key names.
    """
    out = dict(pf or {})
    try:
        import ray as _ray
        actors = list(pool.live_actors()) if pool is not None else []
    except Exception:
        return out
    for _h in actors:
        try:
            _s = _ray.get(_h.consume_face_stats.remote(), timeout=10)
        except Exception:
            continue
        for _k, _v in (_s or {}).items():
            if _k == "applied_fraction":
                continue
            if isinstance(_v, bool) or not isinstance(_v, (int, float)):
                continue
            out[_k] = out.get(_k, 0) + _v
    _tot = (out.get("applied", 0) + out.get("skipped", 0)
            + out.get("skipped_raised", 0)) or 1
    out["applied_fraction"] = out.get("applied", 0) / _tot
    return out


def measure_one_plan(pool, order, specs, face_specs, face_skips, step,
                     *, eval_samples=None, init: bool = False):
    """Measure a SINGLE plan through the pool, face wires included.

    Deliberately routed through ``evaluate_batch`` with a batch of one: the
    unbatched ``pool.evaluate`` path predates face actions and DROPS the face
    wires ("they are dropped here; the pool's own env measures per-vertex"),
    which would silently measure an unapproximated plan for any trainer using
    per-face actions.
    """
    tokens, eqn_ids, rewards, sentinel = pool.evaluate_batch(
        [order], [specs], [int(step)],
        eval_samples=eval_samples,
        init=init,
        face_specs_batch=None if face_specs is None else [face_specs],
        face_skips_batch=None if face_skips is None else [face_skips],
    )
    import numpy as np
    return (np.asarray(tokens)[0], np.asarray(eqn_ids)[0],
            np.asarray(rewards)[0], np.asarray(sentinel)[0])
