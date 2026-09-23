"""dsnn-dfw.127 -- a shared-memory kernel failure at compile is not an OOM.

XLA raises RESOURCE_EXHAUSTED when one fusion asks for more shared memory
than the SM has (job 67639 on a Blackwell: requested 131072, available
101376). The degraded-fusion fallback retries it once. When the retry fails
the same way, the plan is refused as raised, never as oom: no truncation
count, no measure-OOM record for the actor, no cache clear.
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

_SHMEM_TEXT = ("RESOURCE_EXHAUSTED: Shared memory size limit exceeded: "
               "requested 131072, available: 101376, context: [Fusion: "
               "f32[32,128,32,128]]")
_OOM_TEXT = ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
             "2306867200 bytes.")


class XlaRuntimeError(RuntimeError):
    pass


def test_is_oom_rejects_the_shared_memory_text():
    assert env_mod._is_oom(XlaRuntimeError(_SHMEM_TEXT)) is False


def test_is_oom_still_matches_a_device_oom():
    assert env_mod._is_oom(XlaRuntimeError(_OOM_TEXT)) is True
    assert env_mod._is_oom(MemoryError()) is True


def _toy_env():
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
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    return jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips)


def test_a_shared_memory_compile_failure_is_refused_as_raised(monkeypatch):
    from alphagrad.approx.common import compile_cache as cc

    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.delenv("ALPHAGRAD_MEASURE_COMPILE_FALLBACK", raising=False)
    monkeypatch.setitem(env_mod._MEASURE_TOOLCHAIN, "checked", True)
    monkeypatch.setattr(cc, "cached_compile", lambda key, fn: fn())
    compiles = []

    def _compile(self, compiler_options=None):
        compiles.append(compiler_options)
        raise XlaRuntimeError(_SHMEM_TEXT)

    monkeypatch.setattr(jax.stages.Lowered, "compile", _compile)
    clears = []
    monkeypatch.setattr(jax, "clear_caches", lambda: clears.append(1))

    env = _toy_env()
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs, faces, skips = _plan_arrays(len(order))
    samples = (jnp.asarray(
        np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)])),)

    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()
    n_trunc = int(env_mod._TRUNCATED_PLANS[0])
    n_fallback = int(env_mod._MEASURE_COMPILE_FALLBACKS["n"])
    with pytest.raises(XlaRuntimeError, match="Shared memory size limit"):
        env_mod._callback(
            env.config, env.args, env.consts, jnp.asarray(order), specs,
            faces, skips, len(order), *samples)

    assert len(compiles) == 2
    assert int(env_mod._MEASURE_COMPILE_FALLBACKS["n"]) == n_fallback + 1

    counts = env_mod.consume_refused_counts()
    assert counts.get("raised", 0) == 1, counts
    assert counts.get("oom", 0) == 0, counts
    assert counts.get("total", 0) == 1, counts
    rec = env_mod.consume_plan_records()["records"][-1]
    assert rec["refused"] == "raised:XlaRuntimeError"
    assert rec["sentinelled"] is True

    assert int(env_mod._TRUNCATED_PLANS[0]) == n_trunc
    assert env_mod.pop_measure_oom() == (0, "")
    assert clears == []
