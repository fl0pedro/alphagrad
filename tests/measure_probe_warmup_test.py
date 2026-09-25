# dsnn-dfw.223: the warm-up probe of a measurement stops after one execution when that reading
# already fixes the window rule's counts; the second execution existed only to size them.
from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx.env import (                              # noqa: E402
    MEASURE_INNER_MIN, probe_at_floor, resolve_measure_inner,
    resolve_measure_windows)

WINDOW, BUDGET, HI, CAP = 0.05, 1.0, 50, 20


def test_the_campaign_rule_is_at_its_floor_past_ten_times_the_threshold():
    # windows(t) reads 1 for t > BUDGET / (MEASURE_INNER_MIN * 1.5); a tenth of 1.4 s is past it.
    assert resolve_measure_windows(0.14, MEASURE_INNER_MIN, BUDGET, CAP) == 1
    assert resolve_measure_windows(0.12, MEASURE_INNER_MIN, BUDGET, CAP) == 2
    assert probe_at_floor(10.0, WINDOW, BUDGET, HI, CAP)
    assert probe_at_floor(1.4, WINDOW, BUDGET, HI, CAP)
    assert not probe_at_floor(1.2, WINDOW, BUDGET, HI, CAP)
    assert not probe_at_floor(0.5, WINDOW, BUDGET, HI, CAP)
    assert not probe_at_floor(1e-3, WINDOW, BUDGET, HI, CAP)  # the ceiling, not the floor
    assert not probe_at_floor(0.0, WINDOW, BUDGET, HI, CAP)
    assert not probe_at_floor(float("nan"), WINDOW, BUDGET, HI, CAP)


def test_the_floor_is_the_counts_at_a_tenth_of_the_reading():
    for t in (0.2, 1.0, 1.4, 3.0, 40.0):
        inner = resolve_measure_inner(t / 10, WINDOW, HI)
        assert probe_at_floor(t, WINDOW, BUDGET, HI, CAP) == (
            inner == MEASURE_INNER_MIN
            and resolve_measure_windows(t / 10, inner, BUDGET, CAP) == 1)


class _Slow:
    def __init__(self, exe, secs, calls):
        self._exe, self._secs, self._calls = exe, secs, calls

    def __call__(self, *a, **k):
        self._calls.append(1)
        time.sleep(self._secs)
        return self._exe(*a, **k)

    def __getattr__(self, name):
        return getattr(self._exe, name)


@pytest.fixture
def server(monkeypatch):
    import jax
    import jax.numpy as jnp

    from alphagrad.approx import env as env_mod
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer
    from alphagrad.approx.env import VertexEliminationEnv

    monkeypatch.setenv("ALPHAGRAD_DISABLE_JIT_DISK_CACHE", "1")
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.delenv("ALPHAGRAD_PLAN_LOG", raising=False)
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    # The count threshold is max(window / 5, budget / 7.5) = 13.3 ms; the slow executable
    # reads 150 ms, past ten times that, so its one reading fixes (inner 5, windows 1).
    env = VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(toy)(x), args=[x], argnums=(0,), num_envs=0,
        target_fun=toy, measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=50, measure_budget_secs=0.1, measure_window_secs=0.005)
    calls: list = []
    real = env_mod._cached_measure_compile

    def slow(key, fn):
        exe, note = real(key, fn)
        if bytes(key).startswith(b"approx:"):
            return _Slow(exe, 0.15, calls), note
        return exe, note

    monkeypatch.setattr(env_mod, "_cached_measure_compile", slow)
    srv = CpuApproximationServer.from_env(env)
    env_mod.pop_measure_oom()
    env_mod.consume_refused_counts()
    yield srv, env, calls
    env_mod.pop_measure_oom()
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def test_a_slow_program_is_probed_once_and_timed_five_times(server):
    from alphagrad.approx.env import MAX_RULES_PER_VERTEX

    srv, env, calls = server
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs = np.full((len(order), MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    samples = (np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)]),)
    out = srv.evaluate(np.asarray(order, np.int32), specs, len(order),
                       eval_samples=samples, timeout_s=300.0)
    assert float(np.asarray(out[-1]).min()) > -1e9
    assert len(calls) == 1 + MEASURE_INNER_MIN, calls
