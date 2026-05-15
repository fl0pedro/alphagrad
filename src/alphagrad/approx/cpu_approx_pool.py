"""Pool of CPU-side approximation actors that the env's ``io_callback``
delegates to.

Lives on the SPMD actor process. The JIT compile / ``jacve`` execution
that the env's reward harness previously ran inline inside
``io_callback`` (see ``env.py::_callback``) now happens out-of-process
inside a Ray actor pool. This buys us two things:

1. **Compile work runs off the GPU-owning SPMD process.** The
   per-step ``jax.jit(jacve(...)).lower().compile()`` for novel
   elimination orders used to stall the SPMD actor's Python thread
   (and indirectly the GPU's launch queue) for minutes-to-hours. With
   the pool, the SPMD actor blocks only on ``ray.get`` — which we
   bound with a timeout.

2. **Cancellable compiles.** XLA's compile is uninterruptible C++; we
   can't kill it from inside the SPMD process. We can, however,
   ``ray.kill`` the actor that's stuck inside the compile and respawn
   a fresh one. ``CpuApproxPool.evaluate`` does exactly that on
   ``ray.exceptions.GetTimeoutError``.

The pool is deliberately **JAX-free** — it imports only ``numpy``,
``ray``, and stdlib. Importing this module is safe from the driver
process too (though the driver doesn't currently use it; the SPMD
worker is the sole consumer).

See ``scratch/leak_investigation/REPORT.md`` §6 for the root cause
analysis that motivated this design, and ``mu0_ray.py``'s
``--cpu-callback-timeout`` / ``--cpu-worker-recycle-every`` flags for
the user-facing knobs.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np


# Sentinel reward magnitude when the actor times out or errors. Matches
# the error path inside ``CpuApproximationServer.evaluate`` (see
# ``cpu_approx_worker.py:160-169``): treat "too slow to evaluate" /
# "actor crashed" identically to "graphax raised on a bad transform" —
# return zero observation tokens and a very negative scalar reward so
# the policy gradient is pushed away from the offending action.
_SENTINEL_REWARD_VALUE = -1e10


def _sentinel_callback_output(
    max_tokens: int,
    num_rewards: int,
    cosine_sim_idx: int,
    frob_residual_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tokens, eqn_ids, reward)`` matching the env's
    ``_callback_shape`` for a timeout / actor-death.

    ``cosine_sim`` is set to 0 rather than the sentinel so the
    accuracy reward head doesn't see a wildly negative signal when
    the channel happens to have non-zero weight (``cosine_sim``'s
    natural range is ``[0, 1]``, with higher better).
    """
    tokens = np.zeros((max_tokens,), dtype=np.int32)
    eqn_ids = np.zeros((max_tokens,), dtype=np.int32)
    reward = np.full((num_rewards,), _SENTINEL_REWARD_VALUE, dtype=np.float32)
    reward[cosine_sim_idx] = 0.0
    reward[frob_residual_idx] = _SENTINEL_REWARD_VALUE
    return tokens, eqn_ids, reward


class CpuApproxPool:
    """Round-robin pool of CPU approximation actors with cancel-on-timeout.

    Thread-safe: ``evaluate`` is called from inside JAX's
    ``io_callback`` machinery, which dispatches the host fn on a JAX
    worker thread. With ``num_envs=4`` rollouts run sequentially per
    SPMD actor, but we keep the lock to be safe under future
    concurrent expansion.

    Parameters
    ----------
    actor_handles
        Initial pool of ``ray.actor.ActorHandle`` for actors that
        expose an ``evaluate(order_np, specs_np, step, eval_samples,
        init)`` method returning ``(tokens, eqn_ids, reward)`` numpy
        triples. Order doesn't matter — pool consumes them
        round-robin.
    timeout_s
        Per-call timeout for ``ray.get``. When exceeded, the future
        is cancelled, the actor is killed, a replacement is requested
        from ``respawn_factory`` (in a daemon thread so the
        ``evaluate`` call returns quickly), and the sentinel is
        returned. Default 60s — REPORT.md baselines run ~0.9 s per
        callback, so 60 s is ~70× normal.
    respawn_factory
        Zero-arg callable that returns a fresh actor handle. Called
        from a daemon thread on the SPMD process. The factory is
        expected to capture ``CPUApproximationActor.options(...)`` +
        the ``args_dict`` / ``variant`` / fresh ``actor_id`` needed
        to construct a replacement; see ``mu0_ray._run_one_variant``
        for the canonical implementation.
    max_tokens, num_rewards, cosine_sim_idx, frob_residual_idx
        Shape / index parameters for ``_sentinel_callback_output``;
        wired through from ``env.py``'s constants so this module
        stays JAX-free (importing ``env.py`` would pull JAX).
    """

    def __init__(
        self,
        actor_handles: Sequence[Any],
        *,
        timeout_s: float,
        respawn_factory: Callable[[], Any] | None,
        max_tokens: int,
        num_rewards: int,
        cosine_sim_idx: int,
        frob_residual_idx: int,
    ):
        self._alive: collections.deque = collections.deque(actor_handles)
        self._timeout_s = float(timeout_s)
        self._respawn_factory = respawn_factory
        self._lock = threading.Lock()
        self._closed = False
        # Shape constants captured so we don't import jax here.
        self._max_tokens = max_tokens
        self._num_rewards = num_rewards
        self._cosine_sim_idx = cosine_sim_idx
        self._frob_residual_idx = frob_residual_idx
        # Cached eval-samples ObjectRef. ``set_eval_samples`` does the
        # ``ray.put`` once; ``evaluate`` then passes the ref in place of
        # the per-call tuple, so Ray re-uses the deserialised value on
        # each actor instead of re-serialising / re-shipping the same
        # multi-MB payload on every io_callback. None means "no cached
        # ref; pass through the per-call ``eval_samples`` argument."
        self._eval_samples_ref = None
        # Telemetry counters — driver-side code can poll via ``stats()``.
        self._n_calls = 0
        self._n_timeouts = 0
        self._n_actor_errors = 0
        self._n_other_errors = 0
        self._n_respawn_requested = 0

    # ------------------------------------------------------------------
    # Actor pick / put-back
    # ------------------------------------------------------------------
    def _pick(self) -> Any | None:
        with self._lock:
            if self._closed or not self._alive:
                return None
            actor = self._alive.popleft()
            return actor

    def _put_back(self, actor: Any) -> None:
        with self._lock:
            if self._closed:
                # Pool was closed while this call was in flight — let
                # the actor handle drop and be GC'd / killed by Ray
                # when the variant tears down.
                return
            self._alive.append(actor)

    # ------------------------------------------------------------------
    # Cancellation + respawn
    # ------------------------------------------------------------------
    def _poison(self, actor: Any, *, future: Any | None) -> None:
        """Kill ``actor`` and queue a respawn.

        Called when ``ray.get(future, timeout=...)`` raises
        ``GetTimeoutError`` (actor stuck inside an XLA compile) or
        ``RayActorError`` (actor crashed for another reason). The
        actor is removed from the rotation, the in-flight future is
        cancelled (which on Ray 2.55+ propagates to ``ray.kill`` the
        actor), and a daemon thread requests a replacement from
        ``self._respawn_factory``.

        Returns immediately — the ``evaluate`` caller doesn't block on
        respawn. Until a replacement lands in ``self._alive``, the
        pool runs at reduced capacity (degraded but never below
        zero — if the pool empties entirely, ``evaluate`` returns
        sentinels for every call).
        """
        import ray

        if future is not None:
            try:
                ray.cancel(future, force=True)
            except Exception:
                pass
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass

        if self._respawn_factory is None or self._closed:
            return

        self._n_respawn_requested += 1

        def _respawn_in_background() -> None:
            try:
                new_handle = self._respawn_factory()
            except Exception:
                # Respawn failed — log and accept reduced pool size.
                # Driver's periodic ``stats()`` poll will surface
                # the deficit.
                return
            with self._lock:
                if self._closed:
                    # Pool was closed while we were respawning —
                    # immediately kill the new handle.
                    try:
                        ray.kill(new_handle, no_restart=True)
                    except Exception:
                        pass
                    return
                self._alive.append(new_handle)

        t = threading.Thread(target=_respawn_in_background, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # eval_samples object-store sharing
    # ------------------------------------------------------------------
    def set_eval_samples(self, eval_samples) -> None:
        """``ray.put`` the eval_samples tuple once and cache the
        ObjectRef.

        After this, every ``evaluate`` call ships only the (tiny)
        ObjectRef instead of re-serialising the same multi-MB tuple of
        per-sample model inputs on every io_callback. Ray dedups by
        ref-id on the actor side — the deserialised value lives in
        each actor's object store cache after first deref.

        Call once at SPMD-actor init (after ``generate_eval_samples``).
        Subsequent training is amortised: with ``rollout_length=50 ×
        num_envs=4 × 200 callbacks/ep`` and typical eval_samples size
        ~2 MB, this is ~400 MB/ep of pure Ray-serialisation traffic
        avoided. Setting to ``None`` re-enables the per-call passthrough
        path.
        """
        import ray

        if eval_samples is None:
            with self._lock:
                self._eval_samples_ref = None
            return
        ref = ray.put(tuple(eval_samples))
        with self._lock:
            self._eval_samples_ref = ref

    # ------------------------------------------------------------------
    # Main dispatch — called from inside ``env.tokenize()``'s closure,
    # which is itself called from JAX's ``io_callback`` machinery on
    # the SPMD actor process.
    # ------------------------------------------------------------------
    def evaluate(
        self,
        order_np: Any,
        specs_np: Any,
        step: int,
        eval_samples: Any,
        *,
        init: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dispatch one ``(order, specs, step)`` request to a pool
        actor and return ``(tokens, eqn_ids, reward)`` as numpy arrays.

        Always returns valid arrays of the correct shape, even when
        the actor times out / dies / there's no actor available.
        Sentinel arrays mirror the in-process error path in
        ``CpuApproximationServer.evaluate`` (zero tokens, ``-1e10``
        reward except ``cosine_sim=0``).
        """
        import ray
        from ray.exceptions import GetTimeoutError, RayActorError

        self._n_calls += 1
        actor = self._pick()
        if actor is None:
            # Pool empty (all actors timed out and respawns haven't
            # landed yet, or pool is closed). Return sentinel rather
            # than block — the rollout will continue with this step
            # treated as a "bad action".
            return _sentinel_callback_output(
                self._max_tokens,
                self._num_rewards,
                self._cosine_sim_idx,
                self._frob_residual_idx,
            )

        future = None
        try:
            # Prefer the cached ObjectRef when one is available — Ray
            # sees ``ObjectRef`` and skips re-serialising the per-call
            # ``eval_samples`` tuple (the actor reads the cached value
            # from its local object store after the first fetch).
            with self._lock:
                samples_arg = self._eval_samples_ref
            if samples_arg is None:
                samples_arg = (
                    tuple(eval_samples) if eval_samples is not None else None
                )
            future = actor.evaluate.remote(
                np.asarray(order_np),
                np.asarray(specs_np),
                int(step),
                eval_samples=samples_arg,
                init=bool(init),
            )
            result = ray.get(future, timeout=self._timeout_s)
            tokens, eqn_ids, reward = result
            self._put_back(actor)
            return (
                np.asarray(tokens, dtype=np.int32),
                np.asarray(eqn_ids, dtype=np.int32),
                np.asarray(reward, dtype=np.float32),
            )
        except GetTimeoutError:
            self._n_timeouts += 1
            self._poison(actor, future=future)
            return _sentinel_callback_output(
                self._max_tokens,
                self._num_rewards,
                self._cosine_sim_idx,
                self._frob_residual_idx,
            )
        except RayActorError:
            self._n_actor_errors += 1
            self._poison(actor, future=future)
            return _sentinel_callback_output(
                self._max_tokens,
                self._num_rewards,
                self._cosine_sim_idx,
                self._frob_residual_idx,
            )
        except Exception:
            # Catch-all: anything else (serialization issue, malformed
            # return, etc.) is treated like a transient actor failure.
            # We don't ``raise`` because the io_callback caller can't
            # do anything useful with an exception and JAX would
            # propagate it as a NaN-poisoned trajectory.
            self._n_other_errors += 1
            self._poison(actor, future=future)
            return _sentinel_callback_output(
                self._max_tokens,
                self._num_rewards,
                self._cosine_sim_idx,
                self._frob_residual_idx,
            )

    # ------------------------------------------------------------------
    # Maintenance / introspection
    # ------------------------------------------------------------------
    def size(self) -> int:
        """Current live actor count."""
        with self._lock:
            return len(self._alive)

    def stats(self) -> dict:
        """Snapshot counters for driver-side wandb / stderr logging."""
        with self._lock:
            return {
                "pool_size": len(self._alive),
                "calls": self._n_calls,
                "timeouts": self._n_timeouts,
                "actor_errors": self._n_actor_errors,
                "other_errors": self._n_other_errors,
                "respawn_requested": self._n_respawn_requested,
            }

    def recycle(self) -> int:
        """Kill every current actor and replace it via ``respawn_factory``.

        Periodic recycling bounds the C++ ``cost_analysis()`` residual
        each actor accumulates over many evaluations (see
        ``scratch/leak_investigation/REPORT.md`` §6 — ~9 MB per call
        leaks even after the LRU cache was removed, attributable to
        ``HloCostAnalysis`` state held by the XLA backend). Drop the
        whole pool every ``--cpu-worker-recycle-every`` episodes to
        return that allocation to the OS.

        Returns the number of actors that were recycled. If there's
        no ``respawn_factory`` the pool is closed instead and we
        return 0.
        """
        import ray

        with self._lock:
            old = list(self._alive)
            self._alive.clear()
            n_old = len(old)
            if self._respawn_factory is None or self._closed:
                # No factory — kill and stay empty. Pool will return
                # sentinels for subsequent calls until next variant.
                for a in old:
                    try:
                        ray.kill(a, no_restart=True)
                    except Exception:
                        pass
                return 0

        # Outside the lock so respawn dispatch doesn't serialize with
        # in-flight ``_pick`` calls (there shouldn't be any during
        # recycle, but be defensive).
        for a in old:
            try:
                ray.kill(a, no_restart=True)
            except Exception:
                pass

        # Spawn replacements synchronously — caller (the SPMD worker)
        # is between episodes so a brief block here is fine.
        new_actors = []
        for _ in range(n_old):
            try:
                new_actors.append(self._respawn_factory())
            except Exception:
                # Stop on first failure; we'll run with the partial
                # pool we managed to build.
                break

        with self._lock:
            self._alive.extend(new_actors)
        return len(new_actors)

    def close(self) -> None:
        """Tear down the pool — kill all actors, refuse further calls."""
        import ray

        with self._lock:
            self._closed = True
            old = list(self._alive)
            self._alive.clear()
        for a in old:
            try:
                ray.kill(a, no_restart=True)
            except Exception:
                pass


# TODO: predictability follow-up — to answer "can we predict which
# compiles will be slow?" the actor's ``evaluate`` (in
# ``cpu_approx_worker.py``) could time and emit
# ``(len(jaxpr.eqns_after_jacve), wall_compile_s)`` per call. The
# pool would aggregate to ``stats()`` and the driver would log to
# wandb. After collecting a few hundred such pairs, the user can
# pick a "skip compile if len(jaxpr.eqns) > X" threshold and short-
# circuit pathological orders at the actor instead of relying on the
# timeout. Out of scope for the initial unblock.
