"""dsnn-dfw.121 -- the static peak gate refuses a candidate before it runs.

After the candidate's compile and before its first execution the callback
reads ``memory_analysis()`` (temp + argument + output bytes) and refuses the
plan above the device's ``bytes_limit`` (three fourths of the card): reason
``gate``, counted as refused, both numbers in the refusal line and in the
plan-log record. Since dsnn-4eq (2026-09-24) the refusal is SCORED at the
timeout, not sentinelled at -1e10 (tests/refusal_sentinel_test.py pins the
values). The paired reference is not gated.
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
SWELL = 20_000_000_000
BETWEEN = 10_000_000_000


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
    env_mod.set_measure_timeout_s(120.0)
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

    yield run
    env_mod.set_measure_timeout_s(None)


def _is_sentinel(reward):
    return bool(np.all(reward[list(env_mod.COMPUTE_REWARD_INDICES)]
                       <= env_mod.SENTINEL_COST * 0.99))


def test_a_candidate_above_the_bytes_limit_is_refused(
        measure, capsys):
    reward, counts, recs, wrapped = measure({b"approx:": SWELL})
    assert counts.get("gate", 0) == 1, counts
    assert counts.get("total", 0) == 1, counts
    # scored at the timeout, never the -1e10 sentinel (dsnn-4eq)
    assert not _is_sentinel(reward)
    assert np.isfinite(reward).all()
    assert reward[env_mod.REWARD_INDEX["latency_ns"]] < -10.0
    assert wrapped[b"approx:"].calls == 0
    rec = recs[-1]
    assert rec["refused"] == "gate"
    assert rec["sentinelled"] is True
    assert rec["static_peak_bytes"] > SWELL
    assert rec["static_peak_limit_bytes"] == LIMIT
    assert rec["device_bytes_limit"] == LIMIT
    assert rec["refusal_timeout_s"] == 120.0
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[refused] gate ")]
    assert line, "no refusal line"
    assert f"limit_bytes={LIMIT}" in line[0]
    assert "1/4 of" not in line[0]
    assert f"static_peak_bytes={int(rec['static_peak_bytes'])}" in line[0]


def test_a_candidate_below_the_gate_is_measured(measure):
    reward, counts, recs, wrapped = measure({b"approx:": 0})
    assert counts.get("total", 0) == 0, counts
    assert not _is_sentinel(reward)
    assert wrapped[b"approx:"].calls > 0
    assert "refused" not in recs[-1]


def test_a_candidate_between_a_quarter_and_the_limit_is_measured(measure):
    reward, counts, recs, wrapped = measure({b"approx:": BETWEEN})
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
    def __init__(self, stats, platform=None, id=0, name="dev"):
        self._stats = stats
        self.platform = platform
        self.id = id
        self._name = name

    def memory_stats(self):
        return self._stats

    def __repr__(self):
        return f"_FakeDevice({self._name}, {self._stats!r})"


MIB = 2 ** 20
CARD = 97887 * MIB


def test_the_limit_is_the_cards_three_fourths_whatever_the_allocator_holds(
        monkeypatch, capsys):
    """dsnn-dfw.238: the allocator's bytes_limit is 75 percent of the memory
    free at the process's JAX init, so a measure actor that started while
    the card was held read 7.65 GB and kept it. The gate's limit is three
    fourths of the card from NVML, read once per device."""
    monkeypatch.setattr(env_mod, "_DEVICE_BYTES_LIMIT", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls = []

    def rows():
        calls.append(1)
        return [["0", "GPU-aaaa", "97887"], ["1", "GPU-bbbb", "97887"]]

    monkeypatch.setattr(env_mod, "_nvidia_smi_rows", rows)
    held = _FakeDevice({"bytes_limit": 7651249356, "bytes_in_use": 7},
                       platform="gpu", id=0, name="held")
    assert env_mod._device_bytes_limit(held) == int(CARD * 0.75)
    assert env_mod._device_bytes_limit(held) == int(CARD * 0.75)
    assert len(calls) == 1
    assert env_mod.allocator_bytes_limit(held) == (int(CARD * 0.75), "device")
    out = capsys.readouterr().out
    assert f"static peak gate limit {int(CARD * 0.75)} B" in out
    # The logical device maps to its physical card through CUDA_VISIBLE_DEVICES.
    monkeypatch.setattr(env_mod, "_DEVICE_BYTES_LIMIT", {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb,1")
    monkeypatch.setattr(env_mod, "_nvidia_smi_rows", lambda: [
        ["0", "GPU-aaaa", "97887"], ["1", "GPU-bbbb", "48000"]])
    by_uuid = _FakeDevice({"bytes_limit": 1}, platform="gpu", id=0,
                          name="by-uuid")
    assert env_mod._device_bytes_limit(by_uuid) == int(48000 * MIB * 0.75)
    by_index = _FakeDevice({"bytes_limit": 1}, platform="gpu", id=1,
                           name="by-index")
    assert env_mod._device_bytes_limit(by_index) == int(48000 * MIB * 0.75)
    # A device that is not a card has no gate.
    assert env_mod._device_bytes_limit(
        _FakeDevice({"bytes_limit": 12345}, platform="cpu", name="cpu")
    ) is None
    assert env_mod._device_bytes_limit(_FakeDevice(None, name="bare")) is None
    # A card NVML does not list, or a logical device outside
    # CUDA_VISIBLE_DEVICES, raises instead of taking the allocator's number.
    monkeypatch.setattr(env_mod, "_DEVICE_BYTES_LIMIT", {})
    monkeypatch.setattr(env_mod, "_nvidia_smi_rows", lambda: [])
    with pytest.raises(RuntimeError, match="nvidia-smi lists no card"):
        env_mod._device_bytes_limit(
            _FakeDevice({"bytes_limit": 1}, platform="gpu", id=0,
                        name="unlisted"))
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        env_mod._device_bytes_limit(
            _FakeDevice({"bytes_limit": 1}, platform="gpu", id=2,
                        name="outside"))
