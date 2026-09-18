"""The gradient oracle's float64 scope belongs to the ORACLE, not the process.

Oracle A (``--grad-oracle``) compares the exact elimination gradient with
``jax.grad`` in float64. Until this change it did that by flipping the
PROCESS-GLOBAL switch::

    jax.config.update("jax_enable_x64", True)
    ...
    jax.config.update("jax_enable_x64", prev_x64)

In a measure actor that process is also the one that traces, compiles and
TIMES the plan, so while the switch is on the measured program is a different
program: JAX bakes dtypes at TRACE time, and a trace taken anywhere in the
process during that window comes out in float64. Host callbacks run on the
THREAD THAT DISPATCHED their program, and the trainer dispatches from more
than one thread, so "anywhere in the process" is not hypothetical.

``env._x64_scope()`` sets the same setting on the CURRENT THREAD only. These
tests pin the three things that makes true:

1. The global flag never moves, during or after.
2. A trace taken on ANOTHER THREAD while the scope is open is float32, and
   its StableHLO is IDENTICAL to the one taken with no scope open at all --
   the same program, compiled the same way, whether the oracle is due or not.
3. Inside the scope float64 is really available, or the oracle would be
   comparing two float32 evaluations and proving nothing.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import threading                                                # noqa: E402

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

import alphagrad.approx.env as env                              # noqa: E402


def _measurement_like(x):
    """Stands in for the measured program: a trace whose dtypes are decided
    by the x64 setting that is live when it is TAKEN."""
    y = jnp.dot(x, x.T)
    return jnp.sum(y * jnp.asarray(2.0)) + jnp.sum(jnp.asarray([1.5]) * x)


def _fingerprint(x):
    """The lowered StableHLO of that program -- the fingerprint asked for.

    Lowering, not compiling: dtypes are baked at trace time, which is exactly
    where a global x64 toggle changes the program, and lowering needs no
    backend-specific autotuning to be comparable.
    """
    return jax.jit(_measurement_like).lower(x).as_text()


X32 = jnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4))


def test_the_scope_never_moves_the_global_flag():
    before = bool(jax.config.jax_enable_x64)
    with env._x64_scope():
        during_global = bool(jax.config.jax_enable_x64)
        assert during_global is True, (
            "the scope must make float64 visible to THIS thread")
    assert bool(jax.config.jax_enable_x64) is before


def test_float64_is_really_available_inside_the_scope():
    with env._x64_scope():
        a = jnp.asarray(np.ones((2,), dtype=np.float64))
        assert a.dtype == jnp.float64
    b = jnp.asarray(np.ones((2,), dtype=np.float64))
    assert b.dtype == jnp.float32, (
        "outside the scope float64 must still be unavailable, or the scope "
        "leaked")


def test_a_trace_on_another_thread_is_unchanged_while_the_scope_is_open():
    """THE PROOF. One plan, lowered with the oracle's float64 work in flight
    and with none in flight, gives the same fingerprint."""
    baseline = _fingerprint(X32)
    assert "f64" not in baseline

    seen = {}
    opened = threading.Event()
    done = threading.Event()

    def _other_thread():
        opened.wait(timeout=30)
        try:
            seen["hlo"] = _fingerprint(X32)
            seen["x64"] = bool(jax.config.jax_enable_x64)
        except BaseException as exc:            # noqa: BLE001 - reported below
            seen["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            done.set()

    t = threading.Thread(target=_other_thread, name="measure-like")
    t.start()
    try:
        with env._x64_scope():
            # The oracle's own float64 work, inside the scope, as it runs.
            a64 = jnp.asarray(np.arange(12, dtype=np.float64).reshape(3, 4))
            assert a64.dtype == jnp.float64
            _ = jax.jit(_measurement_like).lower(a64).as_text()
            opened.set()
            assert done.wait(timeout=120), "the other thread never finished"
    finally:
        opened.set()
        t.join(timeout=120)

    assert "error" not in seen, seen.get("error")
    assert seen["x64"] is False, (
        "the other thread saw float64: the scope is not thread-local")
    assert seen["hlo"] == baseline, (
        "the measured program changed while the oracle held its float64 "
        "scope")


def test_the_fingerprint_after_the_scope_matches_the_one_before():
    before = _fingerprint(X32)
    with env._x64_scope():
        _ = jnp.asarray(np.ones((2,), dtype=np.float64))
    after = _fingerprint(X32)
    assert after == before
    assert "f64" not in after
