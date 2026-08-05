"""Smoke tests for ``alphagrad.approx.common.compile_cache``.

Covers two regimes:

1. **No-Ray fallback**: ``cached_compile`` must transparently call
   ``compile_fn`` when the coordinator isn't reachable. The rollout
   keeps working in unit-test / local-smoke contexts where Ray isn't
   initialised.
2. **Serialize / deserialize roundtrip**: the internal ``_serialize``
   / ``_deserialize`` pair must round-trip a real
   ``jax.stages.Compiled`` so the on-the-wire blob is restored to a
   functionally-identical callable. This is the precondition for the
   coordinator-mediated path; the actual coordinator is exercised in
   the sbatch profile run, not here (the test would need a real Ray
   cluster).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_no_ray_fallback_calls_compile_fn():
    """When ``ray.is_initialized()`` is False, ``cached_compile``
    should just return ``compile_fn()`` without raising."""
    from alphagrad.approx.common.compile_cache import (
        cached_compile, local_stats, reset_actor_handle,
    )

    reset_actor_handle()
    called = []

    def fake_compile():
        called.append(1)
        return "sentinel-compiled"

    # The counters are process-global and monotone; other tests in the same
    # session may already have compiled through the cache. Measure the delta,
    # and use a key nothing else can have inserted into the local memo.
    before = local_stats()
    out = cached_compile(b"test_no_ray_fallback_calls_compile_fn", fake_compile)
    assert out == "sentinel-compiled"
    assert len(called) == 1
    # No Ray ⇒ the counters are untouched (the function bails before
    # incrementing either of them).
    s = local_stats()
    assert s["local_hits"] - before["local_hits"] == 0
    assert s["local_misses"] - before["local_misses"] == 0


def test_serialize_deserialize_roundtrip_jit():
    """A simple ``@jax.jit`` Compiled must round-trip via
    ``_serialize`` / ``_deserialize`` and produce identical output."""
    from alphagrad.approx.common.compile_cache import _deserialize, _serialize

    @jax.jit
    def f(x, y):
        return jnp.sin(x) + jnp.cos(y) * 2.0

    x = jnp.arange(8.0)
    y = jnp.arange(8.0) + 0.5
    original = f.lower(x, y).compile()
    expected = original(x, y)

    blob = _serialize(original)
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0

    restored = _deserialize(blob)
    got = restored(x, y)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected))


def test_serialize_blob_is_self_contained():
    """The blob shouldn't depend on any per-process state beyond
    JAX itself — we should be able to deserialize after dropping
    the original compiled object."""
    import gc
    from alphagrad.approx.common.compile_cache import _deserialize, _serialize

    @jax.jit
    def g(x):
        return x * x + 1.0

    x = jnp.arange(6.0)
    compiled = g.lower(x).compile()
    blob = _serialize(compiled)

    expected = compiled(x)
    del compiled
    gc.collect()

    restored = _deserialize(blob)
    np.testing.assert_allclose(
        np.asarray(restored(x)), np.asarray(expected),
    )


def test_local_stats_shape():
    """``local_stats`` returns a dict with the three expected keys
    (used by the leak-profile log line)."""
    from alphagrad.approx.common.compile_cache import local_stats
    s = local_stats()
    assert set(s.keys()) == {"local_hits", "local_misses", "local_errors"}
    for v in s.values():
        assert isinstance(v, int)
