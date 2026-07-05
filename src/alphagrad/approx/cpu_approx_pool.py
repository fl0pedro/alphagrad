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
import os
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
        initial_timeout_s: float | None = None,
        warm_after: int = 3,
    ):
        self._alive: collections.deque = collections.deque(actor_handles)
        self._timeout_s = float(timeout_s)
        # Phase 4f: cold-cache budget. The first ``warm_after`` calls
        # per actor use ``initial_timeout_s`` (usually much larger than
        # the warm timeout); after that each actor switches to the
        # regular ``timeout_s``. We track the warm-state per actor in
        # ``_call_counts``; brand-new actors (initial spawn or respawn)
        # start at zero. ``None`` disables the cold-budget and uses
        # ``timeout_s`` from the first call.
        self._initial_timeout_s = (
            float(initial_timeout_s) if initial_timeout_s is not None
            else float(timeout_s)
        )
        self._warm_after = int(warm_after)
        self._call_counts: dict[int, int] = {
            id(a): 0 for a in actor_handles
        }
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
        # ---- recycle+retry-on-OOM (ALPHAGRAD_RECYCLE_RETRY_ON_OOM=1) ----
        # PRIMARY fix for the un-freeable per-measure XLA compile leak: the
        # memory config was PROVEN unable to bound it (LRU / clear_caches all
        # climb identically to the cap); only PROCESS TEARDOWN frees the
        # XLA-internal executable retention. So when a measure OOMs we RECYCLE
        # the offending actor (kill+respawn -> fresh process, memory freed)
        # and RETRY that measure ONCE on the fresh actor, keeping the run
        # progressing with REAL measurements instead of hanging / sentinel-
        # storming. Exactly one recycle+retry per OOM'd measure; a retry that
        # also fails returns the sentinel (a neutral no-op downstream).
        self._recycle_retry_on_oom = (
            os.environ.get("ALPHAGRAD_RECYCLE_RETRY_ON_OOM", "0") == "1"
        )
        # Optional PROACTIVE recycle: recycle an actor before it reaches the
        # ~500-measure OOM point, so the OOM never happens. A/B alternative to
        # the reactive path; 0 = off (reactive-only, the required deliverable).
        try:
            self._proactive_recycle_every = int(
                os.environ.get("ALPHAGRAD_PROACTIVE_RECYCLE_EVERY", "0") or "0"
            )
        except ValueError:
            self._proactive_recycle_every = 0
        # Per-actor measures-since-last-recycle counter (keyed by id(handle)).
        self._measures_since_recycle: dict[int, int] = {
            id(a): 0 for a in actor_handles
        }
        self._n_oom_recycles = 0
        self._n_oom_retries = 0
        self._n_oom_retry_success = 0
        self._n_proactive_recycles = 0

    def _timeout_for(self, actor: Any) -> float:
        """Cold vs warm timeout for ``actor``. Returns 0 when the
        user has disabled timeouts (``--cpu-callback-timeout 0``),
        which the call site interprets as ``ray.get`` without a
        timeout — i.e. block until the actor returns or dies.

        Reads the call count under the lock. A brand-new actor (never
        seen, ``warm_after`` not yet exhausted) gets
        ``initial_timeout_s``; otherwise the regular ``timeout_s``.
        """
        # Explicit disable: any zero or negative timeout means "no
        # timeout" for that phase. Cold/warm semantics still apply
        # independently — set both to 0 to disable globally.
        if self._timeout_s <= 0 and self._initial_timeout_s <= 0:
            return 0.0
        with self._lock:
            n = self._call_counts.get(id(actor), 0)
            if n < self._warm_after:
                return self._initial_timeout_s
            return self._timeout_s

    def _mark_call(self, actor: Any) -> None:
        """Increment the per-actor successful-call counter."""
        with self._lock:
            self._call_counts[id(actor)] = self._call_counts.get(id(actor), 0) + 1

    # ------------------------------------------------------------------
    # Actor pick / put-back
    # ------------------------------------------------------------------
    def _pick(self) -> Any | None:
        """Pop an actor from the round-robin queue (or ``None`` when
        the pool is empty / closed).

        Sticky/key-based routing was tried and reverted: per-actor
        LRU at 15% hit rate didn't beat the explicit-cache overhead.
        The shared compile cache (``common/compile_cache.py``) lives
        in a Ray named actor and is consulted from inside
        ``env._callback``, so any actor can serve any compile target —
        routing locality is no longer load-bearing.
        """
        with self._lock:
            if self._closed or not self._alive:
                return None
            return self._alive.popleft()

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

    def _recycle_actor(self, actor: Any) -> Any | None:
        """Synchronously kill ``actor`` and return a FRESH replacement
        handle (or ``None`` if respawn is impossible / failed).

        Unlike :meth:`_poison` (async, fire-and-forget, drops the actor from
        the rotation and repopulates the pool later) this is the SYNCHRONOUS
        recycle used on the OOM path: process teardown is the ONLY thing that
        frees the leaked XLA executables, and we need the fresh actor RIGHT
        NOW to retry the OOM'd measure. The caller owns ``actor`` (it was
        popped from the pool and is not in ``self._alive``), so we don't touch
        the alive-queue here — the caller decides whether to put the fresh
        handle back. Mirrors ``recycle_one``'s kill+factory dance but for a
        specific held handle.
        """
        import ray

        if self._respawn_factory is None or self._closed:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
            return None
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass
        # Drop the dead handle's per-actor bookkeeping.
        with self._lock:
            self._call_counts.pop(id(actor), None)
            self._measures_since_recycle.pop(id(actor), None)
        try:
            new_handle = self._respawn_factory()
        except Exception as _exc:
            print(
                f"[POOL] recycle respawn failed: {type(_exc).__name__}: "
                f"{str(_exc)[:120]}",
                flush=True,
            )
            return None
        with self._lock:
            if self._closed:
                try:
                    ray.kill(new_handle, no_restart=True)
                except Exception:
                    pass
                return None
            self._call_counts.setdefault(id(new_handle), 0)
            self._measures_since_recycle[id(new_handle)] = 0
        return new_handle

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
            print(f"[SENTINEL] pool-drained (no actor) step={int(step)}", flush=True)
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
            # Per-actor cold/warm timeout. ``timeout_for`` returns 0
            # when the user requested no-timeout (``--cpu-callback-timeout 0``);
            # in that case we ``ray.get`` without a timeout so a slow
            # compile never gets sentinel-poisoned. This is the
            # debugging escape hatch: useful when the rollout's
            # reward signal looks suspiciously zero and we want to
            # rule out the sentinel path. The pool-recycle still
            # bounds long-term memory growth.
            timeout = self._timeout_for(actor)
            result = ray.get(future) if timeout <= 0 else ray.get(future, timeout=timeout)
            tokens, eqn_ids, reward = result
            self._mark_call(actor)
            self._put_back(actor)
            return (
                np.asarray(tokens, dtype=np.int32),
                np.asarray(eqn_ids, dtype=np.int32),
                np.asarray(reward, dtype=np.float32),
            )
        except GetTimeoutError:
            self._n_timeouts += 1
            print(
                f"[SENTINEL] pool-timeout after {timeout:.0f}s step={int(step)} "
                f"(n_timeouts={self._n_timeouts})",
                flush=True,
            )
            self._poison(actor, future=future)
            return _sentinel_callback_output(
                self._max_tokens,
                self._num_rewards,
                self._cosine_sim_idx,
                self._frob_residual_idx,
            )
        except RayActorError:
            self._n_actor_errors += 1
            print(
                f"[SENTINEL] pool-actor-error RayActorError step={int(step)} "
                f"(n_actor_errors={self._n_actor_errors})",
                flush=True,
            )
            self._poison(actor, future=future)
            return _sentinel_callback_output(
                self._max_tokens,
                self._num_rewards,
                self._cosine_sim_idx,
                self._frob_residual_idx,
            )
        except Exception as _exc:
            # Catch-all: anything else (serialization issue, malformed
            # return, etc.) is treated like a transient actor failure.
            # We don't ``raise`` because the io_callback caller can't
            # do anything useful with an exception and JAX would
            # propagate it as a NaN-poisoned trajectory.
            self._n_other_errors += 1
            print(
                f"[SENTINEL] pool-other-error step={int(step)}: "
                f"{type(_exc).__name__}: {str(_exc)[:120]} "
                f"(n_other_errors={self._n_other_errors})",
                flush=True,
            )
            self._poison(actor, future=future)
            return _sentinel_callback_output(
                self._max_tokens,
                self._num_rewards,
                self._cosine_sim_idx,
                self._frob_residual_idx,
            )

    # ------------------------------------------------------------------
    # Batched dispatch — fan out N futures concurrently with a single
    # wall-clock timeout. Used by the PPO-ray rollout where ``num_envs``
    # simultaneous calls would otherwise serialise through ``_pick``'s
    # lock. The MuZero SPMD path is single-call at a time so it keeps
    # using ``evaluate`` directly.
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        order_batch: Sequence[Any],
        specs_batch: Sequence[Any],
        step_batch: Sequence[int],
        *,
        eval_samples: Any = None,
        init: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Phase-2 memory-mitigation probe #8: optional batch dedup.

        When ``ALPHAGRAD_MEASURE_DEDUP_BATCH=1`` and the batch contains
        duplicate ``(order, specs, step)`` requests, each UNIQUE config is
        measured once and its result is broadcast to the duplicate slots — so
        identical configs never compile/execute (or accumulate a distinct
        executable) twice in one step. Default OFF delegates straight to the
        full-fan-out impl (byte-identical legacy path). Payoff is small unless
        the policy emits repeated identical specs within a rollout batch.
        """
        if os.environ.get("ALPHAGRAD_MEASURE_DEDUP_BATCH", "0") != "1":
            return self._evaluate_batch_impl(
                order_batch, specs_batch, step_batch,
                eval_samples=eval_samples, init=init,
            )
        N = len(order_batch)
        # Build canonical key per slot; first-seen slot is the representative.
        rep_of: dict = {}
        uniq_idx: list = []
        slot_to_rep: list = [0] * N
        for i in range(N):
            key = (
                np.asarray(order_batch[i], dtype=np.int32).tobytes(),
                np.asarray(specs_batch[i], dtype=np.int32).tobytes(),
                int(step_batch[i]),
            )
            if key in rep_of:
                slot_to_rep[i] = rep_of[key]
            else:
                rep_of[key] = i
                slot_to_rep[i] = i
                uniq_idx.append(i)
        if len(uniq_idx) == N:
            # No duplicates — nothing to save; run the plain impl.
            return self._evaluate_batch_impl(
                order_batch, specs_batch, step_batch,
                eval_samples=eval_samples, init=init,
            )
        t_u, e_u, r_u, s_u = self._evaluate_batch_impl(
            [order_batch[i] for i in uniq_idx],
            [specs_batch[i] for i in uniq_idx],
            [step_batch[i] for i in uniq_idx],
            eval_samples=eval_samples, init=init,
        )
        pos = {orig: k for k, orig in enumerate(uniq_idx)}
        tokens_out = np.zeros((N, self._max_tokens), dtype=np.int32)
        eqn_ids_out = np.zeros((N, self._max_tokens), dtype=np.int32)
        rewards_out = np.zeros((N, self._num_rewards), dtype=np.float32)
        sentinel_mask = np.zeros((N,), dtype=bool)
        for i in range(N):
            k = pos[slot_to_rep[i]]
            tokens_out[i] = t_u[k]
            eqn_ids_out[i] = e_u[k]
            rewards_out[i] = r_u[k]
            sentinel_mask[i] = s_u[k]
        return tokens_out, eqn_ids_out, rewards_out, sentinel_mask

    def _evaluate_batch_impl(
        self,
        order_batch: Sequence[Any],
        specs_batch: Sequence[Any],
        step_batch: Sequence[int],
        *,
        eval_samples: Any = None,
        init: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dispatch N requests concurrently. Returns
        ``(tokens_stack, eqn_ids_stack, rewards_stack, sentinel_mask)``
        of shapes ``(N, max_tokens)``, ``(N, max_tokens)``,
        ``(N, num_rewards)``, ``(N,) bool``.

        Picks up to ``N`` actors from the pool (each gets its own per-
        actor cold/warm timeout), fires the futures in parallel, and
        ``ray.wait``s them with the MAX per-actor timeout as the wall
        clock. Per-call timeouts are then enforced by ``ray.get(...,
        timeout=0)`` on already-ready futures and via a final
        ``GetTimeoutError`` sweep on the still-pending ones.

        If the pool has fewer actors than N, the surplus slots get
        sentinel rows immediately (rather than blocking on respawn).
        """
        import ray
        from ray.exceptions import GetTimeoutError, RayActorError

        N = len(order_batch)
        assert len(specs_batch) == N and len(step_batch) == N

        # Pre-allocate output buffers + sentinel mask.
        tokens_out = np.zeros((N, self._max_tokens), dtype=np.int32)
        eqn_ids_out = np.zeros((N, self._max_tokens), dtype=np.int32)
        rewards_out = np.zeros((N, self._num_rewards), dtype=np.float32)
        sentinel_mask = np.zeros((N,), dtype=bool)

        with self._lock:
            samples_arg = self._eval_samples_ref
        if samples_arg is None:
            samples_arg = (
                tuple(eval_samples) if eval_samples is not None else None
            )

        def _sentinel_slot(i):
            tokens_out[i], eqn_ids_out[i], rewards_out[i] = (
                _sentinel_callback_output(
                    self._max_tokens, self._num_rewards,
                    self._cosine_sim_idx, self._frob_residual_idx,
                )
            )
            sentinel_mask[i] = True

        # Acquire as many actors as the pool has, up to N, then serve the N
        # slots in WAVES of size M = len(held). With fewer actors than slots
        # (e.g. one GPU measure actor for num_envs=4) each actor handles
        # multiple slots SEQUENTIALLY across waves, instead of leaving the
        # surplus slots "pool-drained" — the old per-slot _pick() sentineled
        # 3 of every 4 envs every step. When the pool has >= N actors this is
        # a single wave == the previous all-concurrent behavior. Slots only
        # sentinel when the pool is genuinely empty or an actor dies.
        held: list[Any] = []
        for _ in range(N):
            a = self._pick()
            if a is None:
                break
            held.append(a)
        M = len(held)

        if M == 0:
            for i in range(N):
                print(
                    f"[SENTINEL] batch pool-drained slot={i} "
                    f"step={int(step_batch[i])} (pool empty)",
                    flush=True,
                )
                _sentinel_slot(i)
            return tokens_out, eqn_ids_out, rewards_out, sentinel_mask

        # Map wave-local actor index j -> list of slots that OOM'd on it,
        # accumulated across waves. Drives the single post-loop recycle+retry.
        oom_by_actor: dict[int, list] = {}

        # Optional PROACTIVE recycle: swap out any held actor that has served
        # >= ALPHAGRAD_PROACTIVE_RECYCLE_EVERY measures since its last recycle,
        # BEFORE it reaches the OOM point. A/B alternative to the reactive path.
        if self._recycle_retry_on_oom and self._proactive_recycle_every > 0:
            for j in range(len(held)):
                a = held[j]
                if a is None:
                    continue
                with self._lock:
                    n_since = self._measures_since_recycle.get(id(a), 0)
                if n_since >= self._proactive_recycle_every:
                    fresh = self._recycle_actor(a)
                    self._n_proactive_recycles += 1
                    print(
                        f"[POOL] proactive-recycle actor#{j} after {n_since} "
                        f"measures (n_proactive={self._n_proactive_recycles})",
                        flush=True,
                    )
                    held[j] = fresh  # may be None (respawn failed)

        # held[j] is reused across waves; set to None when poisoned (dead).
        for wave_start in range(0, N, M):
            wave = list(range(wave_start, min(wave_start + M, N)))
            futures: dict[int, Any] = {}
            f_timeouts: dict[int, float] = {}
            for j, i in enumerate(wave):
                actor = held[j]
                if actor is None:
                    print(
                        f"[SENTINEL] batch pool-drained slot={i} "
                        f"step={int(step_batch[i])} (actor died)",
                        flush=True,
                    )
                    _sentinel_slot(i)
                    continue
                self._n_calls += 1
                try:
                    futures[i] = actor.evaluate.remote(
                        np.asarray(order_batch[i]),
                        np.asarray(specs_batch[i]),
                        int(step_batch[i]),
                        eval_samples=samples_arg,
                        init=bool(init),
                    )
                    f_timeouts[i] = self._timeout_for(actor)
                except Exception as _exc:
                    self._n_other_errors += 1
                    print(
                        f"[SENTINEL] batch dispatch-error slot={i} "
                        f"step={int(step_batch[i])}: {type(_exc).__name__}: "
                        f"{str(_exc)[:120]} (n_other_errors={self._n_other_errors})",
                        flush=True,
                    )
                    self._poison(actor, future=None)
                    held[j] = None
                    _sentinel_slot(i)

            # Per-wave wall clock = max per-actor timeout in this wave.
            wave_to = max((t for t in f_timeouts.values() if t > 0.0), default=0.0)
            wave_no_timeout = wave_to <= 0.0
            live = list(futures.values())
            if live and not wave_no_timeout:
                try:
                    ray.wait(live, num_returns=len(live), timeout=wave_to)
                except Exception:
                    pass

            for j, i in enumerate(wave):
                if sentinel_mask[i] or i not in futures:
                    continue
                actor = held[j]
                future = futures[i]
                try:
                    if wave_no_timeout:
                        tokens, eqn_ids, reward = ray.get(future)
                    else:
                        tokens, eqn_ids, reward = ray.get(future, timeout=0)
                    self._mark_call(actor)
                    tokens_out[i] = np.asarray(tokens, dtype=np.int32)
                    eqn_ids_out[i] = np.asarray(eqn_ids, dtype=np.int32)
                    rewards_out[i] = np.asarray(reward, dtype=np.float32)
                    with self._lock:
                        self._measures_since_recycle[id(actor)] = (
                            self._measures_since_recycle.get(id(actor), 0) + 1
                        )
                    # OOM sentinel? The worker returns the SAME sentinel arrays
                    # for a device OOM as for a benign graphax shape error, so
                    # the reward magnitude alone can't tell them apart. Query
                    # the actor's one-shot pop_oom_flag: True => this sentinel
                    # was the un-freeable measure-GPU leak filling up; recycle
                    # this actor and retry the slot on a fresh process. False =>
                    # benign; a retry would just re-sentinel, so leave it.
                    if (
                        self._recycle_retry_on_oom
                        and self._reward_is_sentinel(reward)
                    ):
                        try:
                            _was_oom = bool(ray.get(
                                actor.pop_oom_flag.remote(), timeout=10.0
                            ))
                        except Exception:
                            _was_oom = False
                        if _was_oom:
                            # Mark the slot as a sentinel NOW (the OOM came back
                            # through the success branch, so the mask wasn't set)
                            # — a failed retry then correctly leaves it masked;
                            # a successful retry clears it below.
                            sentinel_mask[i] = True
                            oom_by_actor.setdefault(j, []).append(i)
                except GetTimeoutError:
                    self._n_timeouts += 1
                    print(
                        f"[SENTINEL] batch timeout slot={i} step={int(step_batch[i])} "
                        f"after {f_timeouts.get(i, 0.0):.0f}s (n_timeouts={self._n_timeouts})",
                        flush=True,
                    )
                    self._poison(actor, future=future)
                    held[j] = None
                    _sentinel_slot(i)
                except RayActorError:
                    self._n_actor_errors += 1
                    print(
                        f"[SENTINEL] batch actor-error slot={i} "
                        f"step={int(step_batch[i])} (n_actor_errors={self._n_actor_errors})",
                        flush=True,
                    )
                    self._poison(actor, future=future)
                    held[j] = None
                    _sentinel_slot(i)
                except Exception as _exc:
                    self._n_other_errors += 1
                    print(
                        f"[SENTINEL] batch other-error slot={i} "
                        f"step={int(step_batch[i])}: {type(_exc).__name__}: "
                        f"{str(_exc)[:120]} (n_other_errors={self._n_other_errors})",
                        flush=True,
                    )
                    self._poison(actor, future=future)
                    held[j] = None
                    _sentinel_slot(i)

        # ---- reactive recycle+retry-on-OOM pass ----
        # For each held actor that OOM'd one or more slots: recycle it ONCE
        # (process teardown frees the leaked XLA memory) and retry that
        # actor's OOM'd slots ONCE on the fresh handle. Recycle-once-per-actor
        # (not once-per-slot) so a capped actor that OOM'd all its slots is
        # torn down a single time and the whole batch retried. A retry that
        # ALSO fails keeps the sentinel already written to the output buffers.
        if self._recycle_retry_on_oom and oom_by_actor:
            for j, slots in oom_by_actor.items():
                old_actor = held[j]
                if old_actor is None:
                    continue
                fresh = self._recycle_actor(old_actor)
                self._n_oom_recycles += 1
                held[j] = fresh  # replace in-place; may be None on failure
                if fresh is None:
                    print(
                        f"[POOL] oom-recycle actor#{j} slots={slots}: respawn "
                        f"failed, keeping sentinels "
                        f"(n_oom_recycles={self._n_oom_recycles})",
                        flush=True,
                    )
                    continue
                print(
                    f"[POOL] oom-recycle actor#{j} slots={slots}: fresh actor, "
                    f"retrying (n_oom_recycles={self._n_oom_recycles})",
                    flush=True,
                )
                for i in slots:
                    self._n_oom_retries += 1
                    try:
                        rf = fresh.evaluate.remote(
                            np.asarray(order_batch[i]),
                            np.asarray(specs_batch[i]),
                            int(step_batch[i]),
                            eval_samples=samples_arg,
                            init=bool(init),
                        )
                        rto = self._timeout_for(fresh)
                        if rto <= 0.0:
                            tk, eq, rw = ray.get(rf)
                        else:
                            tk, eq, rw = ray.get(rf, timeout=rto)
                    except Exception as _rexc:
                        print(
                            f"[POOL] oom-retry slot={i} FAILED on fresh actor: "
                            f"{type(_rexc).__name__}: {str(_rexc)[:120]} "
                            f"(keeping sentinel)",
                            flush=True,
                        )
                        continue
                    self._mark_call(fresh)
                    with self._lock:
                        self._measures_since_recycle[id(fresh)] = (
                            self._measures_since_recycle.get(id(fresh), 0) + 1
                        )
                    if self._reward_is_sentinel(rw):
                        # Retry ALSO sentineled (re-OOM on a fresh process, or a
                        # genuinely bad action). Do NOT recycle+retry again —
                        # exactly one recycle+retry per measure. Keep sentinel.
                        # Drain the fresh actor's OOM flag so it doesn't leak
                        # into the next batch's detection.
                        try:
                            ray.get(fresh.pop_oom_flag.remote(), timeout=10.0)
                        except Exception:
                            pass
                        print(
                            f"[POOL] oom-retry slot={i} re-sentineled on fresh "
                            f"actor (keeping sentinel)",
                            flush=True,
                        )
                        continue
                    # Retry succeeded — overwrite the sentinel with the REAL
                    # measurement and clear the sentinel mask for this slot.
                    tokens_out[i] = np.asarray(tk, dtype=np.int32)
                    eqn_ids_out[i] = np.asarray(eq, dtype=np.int32)
                    rewards_out[i] = np.asarray(rw, dtype=np.float32)
                    sentinel_mask[i] = False
                    self._n_oom_retry_success += 1

        # Return still-alive actors to the pool.
        for a in held:
            if a is not None:
                self._put_back(a)

        return tokens_out, eqn_ids_out, rewards_out, sentinel_mask

    # ------------------------------------------------------------------
    # Maintenance / introspection
    # ------------------------------------------------------------------
    def _reward_is_sentinel(self, reward) -> bool:
        """True if ``reward`` is a sentinel vector (frob_residual channel ==
        the sentinel magnitude). Matches ``_sentinel_callback_output`` and the
        worker's error path; used to spot an OOM sentinel that came back
        through the normal ``ray.get`` success branch."""
        try:
            return bool(
                float(reward[self._frob_residual_idx]) <= _SENTINEL_REWARD_VALUE / 2.0
            )
        except Exception:
            return False

    def size(self) -> int:
        """Current live actor count."""
        with self._lock:
            return len(self._alive)

    def live_actors(self) -> list:
        """Snapshot of the live actor handles. Used by the driver's
        per-point terminal fan-out (``ray.util.ActorPool``) to dynamically
        balance 16-env × n-point measurement tasks across the pool for
        full core utilisation. Read-only snapshot; the pool keeps owning
        recycling/timeouts via the normal ``evaluate_batch`` path."""
        with self._lock:
            return list(self._alive)

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
                "oom_recycles": self._n_oom_recycles,
                "oom_retries": self._n_oom_retries,
                "oom_retry_success": self._n_oom_retry_success,
                "proactive_recycles": self._n_proactive_recycles,
            }

    def fetch_timeout_delta(self) -> int:
        """Per-call (episode-delta) timeout count, AND-reset.

        ``self._n_timeouts`` is the cumulative counter exposed via
        :meth:`stats` — useful for `pool/timeouts` running-total wandb
        plots. Drivers also want the **per-episode delta** to spot when
        sentinel-replacement starts firing (it shouldn't, post-pool-size
        fix on 2026-05-23). This method returns the count since the last
        invocation and resets the per-episode bookkeeping; safe to call
        every episode from the trainer's per-rollout log emission.

        If this delta stays at 0 across the most complex training run
        (full --variant + dynamic-substeps + max-rules), the entire
        sentinel-callback handling in ``ppo_ray_worker._fan_out_tokenize``
        is dead code and can be deleted.
        """
        with self._lock:
            delta = self._n_timeouts - getattr(self, "_timeouts_last_seen", 0)
            self._timeouts_last_seen = self._n_timeouts
            return int(delta)

    def fetch_tokenization_truncation_stats(self) -> dict:
        """Aggregate per-actor jaxpr-truncation telemetry across the
        pool for the current period (= since the last poll, which
        callers invoke once per rollout, so values are per-episode).

        Each actor exposes ``consume_tokenization_truncation_stats``
        which returns ``{count, max_observed_len, overflow_sum}`` and
        resets its per-process counter. We ``ray.get`` from every live
        actor in parallel, then:
          * sum ``count`` and ``overflow_sum`` (per-episode totals);
          * take the ``max`` of ``max_observed_len`` (largest single
            jaxpr seen across the pool).
        Best-effort: individual actor failures contribute zero and are
        not raised.
        """
        import ray
        with self._lock:
            actors = list(self._alive)
        empty = {
            "count": 0, "max_observed_len": 0, "overflow_sum": 0,
            "raw_len_sum": 0, "raw_len_count": 0,
            "raw_len_max": 0, "raw_len_min": 0,
        }
        if not actors:
            return empty
        futures = [
            a.consume_tokenization_truncation_stats.remote() for a in actors
        ]
        try:
            results = ray.get(futures)
        except Exception:
            # Fall back to per-actor with-timeout reads — a single
            # dead/stuck actor shouldn't kill the diagnostic.
            results = []
            for f in futures:
                try:
                    results.append(ray.get(f, timeout=2.0))
                except Exception:
                    results.append(dict(empty))
        total = 0
        max_len = 0
        overflow_sum = 0
        rl_sum = 0
        rl_count = 0
        rl_max = 0
        rl_min = 0
        for r in results:
            if not r:
                continue
            total += int(r.get("count", 0))
            ml = int(r.get("max_observed_len", 0))
            if ml > max_len:
                max_len = ml
            overflow_sum += int(r.get("overflow_sum", 0))
            rl_sum += int(r.get("raw_len_sum", 0))
            rl_count += int(r.get("raw_len_count", 0))
            rlm = int(r.get("raw_len_max", 0))
            if rlm > rl_max:
                rl_max = rlm
            rlmin = int(r.get("raw_len_min", 0))
            if rlmin > 0 and (rl_min == 0 or rlmin < rl_min):
                rl_min = rlmin
        return {
            "count": total,
            "max_observed_len": max_len,
            "overflow_sum": overflow_sum,
            "raw_len_sum": rl_sum,
            "raw_len_count": rl_count,
            "raw_len_max": rl_max,
            "raw_len_min": rl_min,
        }

    def recycle_one(self) -> int:
        """Kill+respawn the OLDEST live actor (FIFO from the front of
        the alive queue). Returns the pool size after the swap; ``-1``
        when the recycle was skipped (no factory, empty pool, closed).

        Cascading recycle: callers invoke this every
        ``recycle_every / N`` episodes so a full ``recycle_every``
        window rotates the entire pool but only ONE actor is mid-
        respawn at any moment. Smears the memory spike + dead-pool
        window that ``recycle()`` produces, without changing the
        average per-actor lifetime.
        """
        import ray

        with self._lock:
            if self._closed or not self._alive:
                return -1
            if self._respawn_factory is None:
                return -1
            actor = self._alive.popleft()

        # Kill outside the lock — ``ray.kill`` synchronously waits for
        # the actor to die; holding the pool lock while waiting would
        # block in-flight ``_pick`` calls unnecessarily.
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass

        try:
            new_handle = self._respawn_factory()
        except Exception:
            # Respawn failed — accept the smaller pool. The driver's
            # next recycle attempt will try again.
            with self._lock:
                return len(self._alive)

        with self._lock:
            if self._closed:
                # Pool was closed between our pop and respawn; kill
                # the new handle immediately.
                try:
                    ray.kill(new_handle, no_restart=True)
                except Exception:
                    pass
                return 0
            self._alive.append(new_handle)
            return len(self._alive)

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
