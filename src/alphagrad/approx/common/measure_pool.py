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
                       fidelity_idx: int | None = None,
                       sparsity_idx: int | None = None,
                       first_gpu: int = 1,
                       token_dtype=None, eqn_dtype=None):
    """Create ``n_actors`` GPU-pinned measure actors and wrap them in a pool.

    ``first_gpu`` is the first device index handed to an actor; device 0 is
    reserved for the trainer (PPO's convention — letting Ray allocate handed
    the first actor the trainer's own device, which both hung the pool and
    destroyed the timing isolation).

    ``args_dict`` must describe the SAME measurement the caller's own env
    performs (``measure_grad``, ``per_face``, ``num_data_points``,
    ``reps_per_point``, ``ref_num_data_points``, ``ref_reps_per_point``,
    ``latency_inner_reps``, target/dataset): the actor
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
                # Measurement semantics must not depend on the TRAINER process's
                # allocator: an inherited XLA_PYTHON_CLIENT_ALLOCATOR=platform puts
                # raw cudaMalloc/cudaFree in the timed region (154us -> 379us, 2.46x
                # flat, probe jobs 59598/59599) and breaks clear_memory_stats() so
                # peak_memory silently becomes the STATIC estimate. Pin the default
                # (BFC) allocator in every measure actor.
                "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
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
        # Reward slot 8 (A2). None => the sentinel writer leaves the slot at the
        # generic value, which is what callers built before the channel existed
        # already get.
        fidelity_idx=(None if fidelity_idx is None else int(fidelity_idx)),
        # Reward slot 10, same optionality and the same reason.
        sparsity_idx=(None if sparsity_idx is None else int(sparsity_idx)),
        # THE WIRE DTYPES (env.wire_token_dtype / env.wire_eqn_dtype). None
        # keeps the pool's int32 default, which is the legacy full-stream
        # wire -- every caller that does not run ``delta_obs``.
        **({} if token_dtype is None else {"token_dtype": token_dtype}),
        **({} if eqn_dtype is None else {"eqn_dtype": eqn_dtype}),
    )


def merge_pool_collapse_stats(pool, cs: dict | None = None) -> dict:
    """Merge the measure actors' truncation / collapse counters (#81, #96).

    Sibling of :func:`merge_pool_face_stats`, for the counters behind
    ``tokenization/*`` and ``collapse/*``. Those are module globals written
    inside the measurement ``_callback``; with the Ray pool active that
    runs in the ACTOR processes, so the trainer's own globals never move
    and the panels read 0 while clipping is happening.

    All counters here are ADDITIVE except ``trunc_max_observed_len``,
    which is a MAX -- summing it would report a length no step ever had.

    Best-effort: no pool, a dead actor, or an actor without the method
    contributes nothing and never raises.
    """
    out = dict(cs or {})
    try:
        import ray as _ray
        actors = list(pool.live_actors()) if pool is not None else []
    except Exception:
        return out
    for _h in actors:
        try:
            _s = _ray.get(_h.consume_collapse_stats.remote(), timeout=10)
        except Exception:
            continue
        for _k, _v in (_s or {}).items():
            if isinstance(_v, bool) or not isinstance(_v, (int, float)):
                continue
            if _k == "trunc_max_observed_len":
                out[_k] = max(out.get(_k, 0), _v)
            else:
                out[_k] = out.get(_k, 0) + _v
    return out


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


def merge_pool_plan_records(pool) -> dict:
    """Drain every live measure actor's A6 plan-log records.

    Sibling of :func:`merge_pool_face_stats`, for the per-terminal-plan
    records ``env._record_terminal_plan`` appends. With the Ray pool active
    the measurement callback runs in the ACTOR processes, so the trainer's
    own list only ever holds the terminals it measured itself (the
    ``ALPHAGRAD_POOL_TERMINAL_LOCAL`` rows) -- every other plan would be
    missing from
    a log the whole point of which is that nothing is missing.

    Each record is stamped with the ``actor`` it came from before it is
    handed back, so a plan can be attributed to the process that timed it.

    Returns ``{"records": [...], "dropped": int, "actors_polled": int,
    "actors_failed": int}``. Best-effort like its siblings -- a dead actor
    or one too old to have the method contributes nothing and never raises
    -- but a failed poll is COUNTED, because a silently unpolled actor is a
    silently incomplete log.
    """
    out = {"records": [], "dropped": 0, "actors_polled": 0,
           "actors_failed": 0, "have_pool": pool is not None,
           "actors_seen": 0, "error": None, "terminals": 0,
           "actors_disabled": 0,
           # Measure toolchain telemetry (finding 03): summed over actors;
           # ``toolchain_ok`` is False if ANY polled actor's node failed.
           "compile_fallbacks": 0, "compile_fallbacks_total": 0,
           "toolchain_ok": True,
           # Memory parity (ticket .49): the actors' (temp, watermark)
           # records and the counts `env.check_mem_parity_complete` compares.
           "mem_parity": {"records": [], "measured": 0, "dropped": 0},
           # The paired rev-exact reference (ticket .9): one record per
           # reference measurement the actors took.
           "paired_ref": {"records": [], "dropped": 0}}
    try:
        import ray as _ray
        actors = list(pool.live_actors()) if pool is not None else []
    except Exception as _exc:
        out["error"] = f"{type(_exc).__name__}: {_exc}"
        return out
    out["actors_seen"] = len(actors)
    for _h in actors:
        try:
            _s = _ray.get(_h.consume_plan_records.remote(), timeout=30)
        except Exception as _exc:
            # An actor whose memory parity is incomplete raises
            # env.MemChannelFault from its drain; that is an apparatus
            # fault and must not be counted as a failed poll.
            from alphagrad.approx.cpu_approx_pool import _is_toolchain_fault
            if _is_toolchain_fault(_exc):
                raise
            out["actors_failed"] += 1
            continue
        out["actors_polled"] += 1
        out["terminals"] += int((_s or {}).get("terminals", 0))
        _mp = (_s or {}).get("mem_parity") or {}
        out["mem_parity"]["records"].extend(_mp.get("records", ()))
        out["mem_parity"]["measured"] += int(_mp.get("measured", 0))
        out["mem_parity"]["dropped"] += int(_mp.get("dropped", 0))
        _pr = (_s or {}).get("paired_ref") or {}
        out["paired_ref"]["records"].extend(_pr.get("records", ()))
        out["paired_ref"]["dropped"] += int(_pr.get("dropped", 0))
        out["compile_fallbacks"] += int(
            (_s or {}).get("compile_fallbacks", 0))
        out["compile_fallbacks_total"] += int(
            (_s or {}).get("compile_fallbacks_total", 0))
        out["toolchain_ok"] = out["toolchain_ok"] and bool(
            (_s or {}).get("toolchain_ok", True))
        if not (_s or {}).get("enabled", True):
            out["actors_disabled"] += 1
        _aid = (_s or {}).get("actor_id")
        for _r in (_s or {}).get("records", ()):
            if isinstance(_r, dict):
                _r["actor"] = _aid
            out["records"].append(_r)
        out["dropped"] += int((_s or {}).get("dropped", 0))
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
