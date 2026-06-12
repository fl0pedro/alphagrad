"""Cluster-wide shared compile cache for the env's per-step
``jit(jacve(...)).lower().compile()``.

Design:

* A single Ray *named* actor (``CompileCacheCoordinator``) owns a
  ``{cache_key -> ObjectRef}`` table. The values are ``ObjectRef``\\s
  to serialized XLA executables that were ``ray.put``\\'d by whichever
  CPU actor produced them. The coordinator never holds the
  executables itself — just the references — so its own memory
  stays bounded by ``max_size`` × (8 byte ref + key bytes).
* Producer side (``cached_compile``): consult the coordinator. On
  miss, run the caller's compile function, serialise via
  ``jax.experimental.serialize_executable.serialize``, ``ray.put``
  the blob, and register the resulting ref. On hit, ``ray.get``
  the blob and ``deserialize_and_load`` it locally.
* Failure mode: any Ray error (no cluster, coordinator not
  registered, network blip) silently falls back to the caller's
  uncached path — the caller's ``compile_fn`` is the single source
  of truth. Cache is best-effort.

Lookup is via Ray's named-actor registry (``ray.get_actor("...")``)
so the actor processes don't need an explicit handle passed in via
their args dict — they discover the coordinator lazily on first
compile. The driver is responsible for spawning the coordinator
before any worker actor tries to use it.

Stats: every per-process call increments local counters; the
coordinator's ``stats()`` method returns cluster-wide aggregates.
Both are surfaced in the leak-profile log so we can confirm the
hit rate empirically.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Named-actor handle resolution (lazy).
# ---------------------------------------------------------------------------

_COORDINATOR_NAME = os.environ.get(
    "ALPHAGRAD_COMPILE_CACHE_NAME", "alphagrad_compile_cache_coordinator"
)

_actor_handle: Any | None = None
_actor_handle_lookup_done = False
_actor_handle_lock = threading.Lock()

# Per-process counters — surfaced via ``local_stats`` for the leak
# profiler's per-actor cache=... column.
_local_hits = 0
_local_misses = 0
_local_errors = 0


def _get_actor_handle() -> Any | None:
    """Resolve the coordinator handle (lazy, cached). Returns ``None``
    when Ray isn't initialised or no coordinator was spawned — callers
    fall back to compiling locally without caching."""
    global _actor_handle, _actor_handle_lookup_done
    with _actor_handle_lock:
        if _actor_handle_lookup_done:
            return _actor_handle
        _actor_handle_lookup_done = True
        try:
            import ray
            if not ray.is_initialized():
                return None
            _actor_handle = ray.get_actor(_COORDINATOR_NAME)
        except Exception:
            _actor_handle = None
        return _actor_handle


def reset_actor_handle() -> None:
    """Force re-lookup on next ``cached_compile`` (e.g. after an
    actor respawn). Used by tests, not by the rollout hot path."""
    global _actor_handle, _actor_handle_lookup_done
    with _actor_handle_lock:
        _actor_handle = None
        _actor_handle_lookup_done = False


def local_stats() -> dict:
    """Per-process counters (hits / misses / errors). Cheap snapshot."""
    return {
        "local_hits": _local_hits,
        "local_misses": _local_misses,
        "local_errors": _local_errors,
    }


# ---------------------------------------------------------------------------
# Producer-side: cached compile.
# ---------------------------------------------------------------------------

def cached_compile(
    cache_key: bytes,
    compile_fn: Callable[[], Any],
) -> Any:
    """Return ``compile_fn()`` output, consulting the shared
    coordinator first.

    ``compile_fn`` must produce a SINGLE ``jax.stages.Compiled``
    object (single-jit case). For the two-compile case in
    ``env._callback`` (compiled_approx + compiled_exact), call
    ``cached_compile`` twice with different cache keys, or use
    :func:`cached_compile_pair` below.

    ``cache_key`` is intended to be a bytes value derived from the
    canonical compile inputs — typically
    ``(order.tobytes(), specs.tobytes(), step)`` hashed via blake2 or
    similar. The coordinator's hash table is keyed on this exact
    bytes value; identity must be deterministic across actors.
    """
    global _local_hits, _local_misses, _local_errors
    handle = _get_actor_handle()
    if handle is None:
        # No coordinator — fall through to uncached compile.
        return compile_fn()

    import ray

    # 1. Cache lookup — get the ObjectRef if present.
    try:
        ref = ray.get(handle.get.remote(cache_key))
    except Exception:
        _local_errors += 1
        ref = None

    if ref is not None:
        # 2. Cache hit — fetch + deserialise locally.
        try:
            blob = ray.get(ref)
            compiled = _deserialize(blob)
            _local_hits += 1
            return compiled
        except Exception:
            # Stale ref or deserialise failure — fall through to
            # recompile + reinsert.
            _local_errors += 1

    # 3. Cache miss — compile fresh, serialise, ray.put, register.
    _local_misses += 1
    compiled = compile_fn()
    try:
        blob = _serialize(compiled)
        new_ref = ray.put(blob)
        # Fire-and-forget put: don't ray.get the void return.
        handle.put.remote(cache_key, new_ref)
    except Exception:
        _local_errors += 1
    return compiled


# ---------------------------------------------------------------------------
# JAX-side serialization glue.
# ---------------------------------------------------------------------------

def _serialize(compiled) -> bytes:
    """Serialise a ``jax.stages.Compiled`` to a single bytes blob
    (containing the StableHLO + the in/out pytree shapes pickled
    together). Round-trippable on the same JAX/XLA version."""
    import pickle
    from jax.experimental.serialize_executable import serialize
    ser, in_tree, out_tree = serialize(compiled)
    return pickle.dumps((ser, in_tree, out_tree), protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize(blob: bytes):
    """Inverse of :func:`_serialize`. Returns the restored
    ``jax.stages.Compiled``."""
    import pickle
    from jax.experimental.serialize_executable import deserialize_and_load
    ser, in_tree, out_tree = pickle.loads(blob)
    return deserialize_and_load(ser, in_tree, out_tree)


# ---------------------------------------------------------------------------
# Coordinator actor — spawned by the driver as a named actor.
# ---------------------------------------------------------------------------

def make_coordinator_class():
    """Return the ``@ray.remote``-decorated coordinator class.

    Wrapped in a factory so the JAX-free driver can import this
    module without pulling ``ray`` at module-load time (matches the
    pattern in ``cpu_approx_pool.py``)."""
    import ray

    @ray.remote(num_cpus=0.1, num_gpus=0)
    class CompileCacheCoordinator:
        """Cluster-wide compile cache. Holds ``key -> ObjectRef``.

        Hot methods are intentionally short — Ray's actor dispatch
        overhead is ~ms per call, so we keep the work minimal and
        let serialisation / deserialisation happen at the call site.

        LRU eviction: ``dict`` preserves insertion order (Python
        3.7+), so when at capacity we pop the oldest entry. A hit
        on ``get`` moves the key to the end (LRU bump).
        """

        def __init__(self, max_size: int = 512):
            self._cache: dict[bytes, Any] = {}
            self._max_size = int(max_size)
            self._hits = 0
            self._misses = 0
            self._puts = 0
            self._evictions = 0

        def get(self, key: bytes):
            """Return the cached ``ObjectRef`` for ``key`` or
            ``None``. LRU-bumps the key on hit."""
            ref = self._cache.get(key)
            if ref is None:
                self._misses += 1
                return None
            self._hits += 1
            # Move-to-end so this key isn't the next eviction target.
            del self._cache[key]
            self._cache[key] = ref
            return ref

        def put(self, key: bytes, ref) -> None:
            """Insert (or refresh) the ref. Evicts the oldest entry
            when the cache is full."""
            if key in self._cache:
                # Race: another actor put first. Refresh recency
                # but don't overwrite (the existing ref is valid).
                del self._cache[key]
                self._cache[key] = self._cache.get(key, ref)
                return
            if len(self._cache) >= self._max_size:
                self._cache.pop(next(iter(self._cache)))
                self._evictions += 1
            self._cache[key] = ref
            self._puts += 1

        def stats(self) -> dict:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "puts": self._puts,
                "evictions": self._evictions,
            }

        def ready(self) -> bool:
            return True

    return CompileCacheCoordinator


def spawn_coordinator(max_size: int = 512):
    """Create the named coordinator actor (idempotent — returns the
    existing handle if already spawned). Driver-side helper."""
    import ray

    try:
        return ray.get_actor(_COORDINATOR_NAME)
    except (ValueError, Exception):
        pass

    cls = make_coordinator_class()
    handle = cls.options(
        name=_COORDINATOR_NAME,
        lifetime="detached",  # survives caller's exit; tied to the cluster
        max_concurrency=64,   # many actors may call concurrently
    ).remote(max_size=int(max_size))
    # Block on init so the caller knows the actor is ready before
    # any worker tries to ray.get_actor() it.
    ray.get(handle.ready.remote())
    return handle


def kill_coordinator() -> None:
    """Tear down the named coordinator. Driver-side cleanup."""
    import ray

    try:
        handle = ray.get_actor(_COORDINATOR_NAME)
        ray.kill(handle, no_restart=True)
    except Exception:
        pass
