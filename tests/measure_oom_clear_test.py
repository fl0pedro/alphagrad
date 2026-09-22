"""dsnn-dfw.100 -- EVERY TRUNCATED MEASURE OOM CLEARS THE ACTOR'S CACHES ONCE.

A device OOM inside ``_callback`` is a refusal, not a plan: ``_oom_truncate``
counts it and returns a truncated reward. It used to clear the JAX caches
itself and tell nobody, so the measure actor never set ``_last_was_oom``, the
pool's ``pop_oom_flag`` stayed False and the clear was invisible: job 67410
logged 86 ``[trunc] OOM`` lines against 5 actor clear lines.

The truncation now RECORDS the OOM (``note_measure_oom``). The actor drains
the record after every callback (``_settle_after_callback``), clears once per
OOM and raises the flag. Pinned here: one clear per OOM, no clear without
one, the flag, the refusal count, and the fallback clear when no actor is
registered (a single-process run must still reclaim).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402
from alphagrad.approx.cpu_approx_worker import (                # noqa: E402
    CpuApproximationServer,
)

# What jaxlib raises on a full device: a generic error carrying the text.
_OOM_TEXT = ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
             "2306867200 bytes.")


class _XlaRuntimeError(RuntimeError):
    pass


@pytest.fixture
def clears(monkeypatch):
    """Count jax.clear_caches() calls, wherever they are made from."""
    seen = []
    monkeypatch.setattr(jax, "clear_caches", lambda: seen.append(1))
    return seen


@pytest.fixture
def actor():
    """A bare measure actor: the drain path touches only these four fields."""
    a = object.__new__(CpuApproximationServer)
    a._n_calls = 0
    a._n_oom = 0
    a._last_was_oom = False
    a._cache_clear_every = 0
    env_mod.register_measure_oom_consumer()
    yield a
    env_mod.register_measure_oom_consumer(False)
    env_mod.pop_measure_oom()


def test_a_truncated_oom_clears_once_and_raises_the_flag(actor, clears):
    env_mod.note_measure_oom("measurement", _XlaRuntimeError(_OOM_TEXT))
    assert clears == []                     # the actor owns the reclaim now
    actor._settle_after_callback()
    assert len(clears) == 1
    assert actor.pop_oom_flag() is True
    assert actor._n_oom == 1


def test_one_clear_per_oom_and_none_without_one(actor, clears):
    for _ in range(3):
        env_mod.note_measure_oom("measurement", _XlaRuntimeError(_OOM_TEXT))
        actor._settle_after_callback()
        assert actor.pop_oom_flag() is True
    assert len(clears) == 3
    actor._settle_after_callback()          # a measurement that did not OOM
    assert len(clears) == 3
    assert actor.pop_oom_flag() is False


def test_the_cadence_does_not_add_a_second_clear(actor, clears):
    """With a cadence due on the same call the OOM still costs ONE clear."""
    actor._cache_clear_every = 1
    actor._n_calls = 4
    env_mod.note_measure_oom("measurement", _XlaRuntimeError(_OOM_TEXT))
    actor._settle_after_callback()
    assert len(clears) == 1


def test_without_an_actor_the_env_still_clears(clears):
    env_mod.register_measure_oom_consumer(False)
    env_mod.pop_measure_oom()
    env_mod.note_measure_oom("approx compile", _XlaRuntimeError(_OOM_TEXT))
    assert len(clears) == 1
    n, txt = env_mod.pop_measure_oom()
    assert n == 1
    assert "RESOURCE_EXHAUSTED" in txt


# ---------------------------------------------------------------------------
# THE REAL CALLBACK PATH: a RESOURCE_EXHAUSTED injected into the compile the
# measurement makes. The plan must come back REFUSED (counted, never scored)
# and the actor must clear exactly once.
# ---------------------------------------------------------------------------
def _toy_env():
    """A 16-wide scalar loss on the CPU, terminal rewards only."""
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    closed = jax.make_jaxpr(toy)(x)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[x], argnums=(0,), num_envs=0, target_fun=toy,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)


def _plan_arrays(n):
    """An all-exact plan of `n` vertices: no rule, no face action."""
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    return jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips)


def test_an_oom_in_the_callback_is_refused_and_clears_once(
        actor, clears, monkeypatch):
    from alphagrad.approx.common import compile_cache as cc

    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COUNT_OPS", "1")
    monkeypatch.delenv("ALPHAGRAD_PLAN_LOG", raising=False)

    def _oom(*a, **kw):
        raise _XlaRuntimeError(_OOM_TEXT)

    monkeypatch.setattr(cc, "cached_compile", _oom)

    env = _toy_env()
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs, faces, skips = _plan_arrays(len(order))
    samples = (jnp.asarray(
        np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)])),)

    env_mod.consume_refused_counts()
    n_trunc = int(env_mod._TRUNCATED_PLANS[0])
    out = env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order), specs,
        faces, skips, len(order), *samples)
    env_mod.consume_plan_records()

    # Refused: counted by kind, and counted as a resource truncation.
    counts = env_mod.consume_refused_counts()
    assert counts.get("oom", 0) == 1, counts
    assert counts.get("total", 0) == 1, counts
    assert int(env_mod._TRUNCATED_PLANS[0]) - n_trunc == 1

    # The reward is the truncation's, not a measurement of the plan.
    reward = np.asarray(out[-1], dtype=np.float32)
    assert reward.shape == (env_mod.NUM_REWARDS,)
    assert float(reward[env_mod.REWARD_INDEX["latency_ns"]]) < 0.0

    # The OOM reached the actor, and cost exactly one clear.
    assert clears == []
    actor._settle_after_callback()
    assert len(clears) == 1
    assert actor.pop_oom_flag() is True
