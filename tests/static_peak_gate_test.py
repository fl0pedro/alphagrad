"""dsnn-dfw.121 -- the static peak gate refuses a candidate before it runs.

After the candidate's compile and before its first execution the callback
reads ``memory_analysis()`` (temp + argument + output bytes) and refuses the
plan above one quarter of the device's ``bytes_limit``: reason
``oom-static``, counted as refused, the sentinel reward, both numbers in the
refusal line and in the plan-log record. The paired reference is not gated.
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

LIMIT = 16_000_000_000
SWELL = 10_000_000_000


class _SwollenAnalysis:
    def __init__(self, inner, extra):
        self._inner = inner
        self.temp_size_in_bytes = int(inner.temp_size_in_bytes) + int(extra)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Swollen:
    def __init__(self, inner, extra):
        self._inner = inner
        self._extra = extra
        self.calls = 0

    def memory_analysis(self):
        return _SwollenAnalysis(self._inner.memory_analysis(), self._extra)

    def __call__(self, *a, **kw):
        self.calls += 1
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


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


@pytest.fixture
def measure(monkeypatch):
    from alphagrad.approx.common import compile_cache as cc

    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    real = cc.cached_compile
    wrapped = {}

    def run(swell):
        def fake(key, fn, *a, **kw):
            out = real(key, fn, *a, **kw)
            for prefix, extra in swell.items():
                if bytes(key).startswith(prefix):
                    out = _Swollen(out, extra)
                    wrapped[prefix] = out
            return out

        monkeypatch.setattr(cc, "cached_compile", fake)
        env = _toy_env()
        order = sorted(int(v) for v in np.asarray(env.valid_vertices))
        specs, faces, skips = _plan_arrays(len(order))
        samples = (jnp.asarray(
            np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)])),)
        env_mod.consume_refused_counts()
        env_mod.consume_plan_records()
        out = env_mod._callback(
            env.config, env.args, env.consts, jnp.asarray(order), specs,
            faces, skips, len(order), *samples)
        counts = env_mod.consume_refused_counts()
        recs = env_mod.consume_plan_records()["records"]
        return np.asarray(out[-1], dtype=np.float32), counts, recs, wrapped

    return run


def _is_sentinel(reward):
    return bool(np.all(reward[list(env_mod.COMPUTE_REWARD_INDICES)]
                       <= env_mod.SENTINEL_COST * 0.99))


def test_a_candidate_above_a_quarter_of_the_device_is_refused(
        measure, capsys):
    reward, counts, recs, wrapped = measure({b"approx:": SWELL})
    assert counts.get("oom-static", 0) == 1, counts
    assert counts.get("total", 0) == 1, counts
    assert _is_sentinel(reward)
    assert wrapped[b"approx:"].calls == 0
    rec = recs[-1]
    assert rec["refused"] == "oom-static"
    assert rec["sentinelled"] is True
    assert rec["static_peak_bytes"] > SWELL
    assert rec["static_peak_limit_bytes"] == LIMIT / 4
    assert rec["device_bytes_limit"] == LIMIT
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if "oom-static" in ln]
    assert line, "no refusal line"
    assert f"limit_bytes={LIMIT // 4}" in line[0]
    assert f"static_peak_bytes={int(rec['static_peak_bytes'])}" in line[0]


def test_a_candidate_below_the_gate_is_measured(measure):
    reward, counts, recs, wrapped = measure({b"approx:": 0})
    assert counts.get("total", 0) == 0, counts
    assert not _is_sentinel(reward)
    assert wrapped[b"approx:"].calls > 0
    assert "refused" not in recs[-1]


def test_the_reference_is_not_gated(measure):
    reward, counts, recs, wrapped = measure({b"paired-ref:": SWELL})
    assert b"paired-ref:" in wrapped
    assert counts.get("total", 0) == 0, counts
    assert not _is_sentinel(reward)
    assert "refused" not in recs[-1]


class _FakeDevice:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        return self._stats

    def __repr__(self):
        return f"_FakeDevice({self._stats!r})"


def test_the_limit_is_read_from_the_device_memory_stats():
    assert env_mod._device_bytes_limit(
        _FakeDevice({"bytes_limit": 12345, "bytes_in_use": 7})) == 12345
    assert env_mod._device_bytes_limit(_FakeDevice(None)) is None
