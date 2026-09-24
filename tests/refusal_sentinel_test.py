"""dsnn-4eq (owner ruling 2026-09-24) -- a refused plan is scored, not dropped.

Three kinds take a finite sentinel on each channel's own convention, strictly
worse than every plan that ran: ``gate`` (static memory above bytes_limit
after the compile), ``timeout`` (the pool killed the call) and ``compile``
(the compile raised). Latency is log(timeout / t_ref) with THE timeout the
pool kills at (10 x for a compile failure); memory is the candidate's real
static ratios for a gate refusal and bytes_limit against the reference temp
otherwise, on slot 5 and on slot 11; quality is 0. The reason rides the
plan-log record and the refusal counters.
"""
from __future__ import annotations

import math
import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402
from alphagrad.approx.common import compile_cache as _cc        # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    NUM_REWARDS,
    REWARD_INDEX,
    mem_objective,
    paired_log_costs,
)

# captured at import: a test that runs two plans patches the same attribute
# twice, and the second fake must wrap the real cache, not the first fake
_REAL_CACHED_COMPILE = _cc.cached_compile

LIMIT = 16_000_000_000
SWELL = 20_000_000_000
TIMEOUT = 120.0
LAT = int(REWARD_INDEX["latency_ns"])
MEM = int(REWARD_INDEX["peak_memory"])
QUAL = int(REWARD_INDEX["quality"])
MOBJ = int(REWARD_INDEX["mem_objective"])
TOK = 8


class XlaRuntimeError(RuntimeError):
    pass


_TRITON = ("INTERNAL: Failed to compile Triton kernel. Context: [Fusion: "
           "fusion.35 = f32[32,1024,32]{2,1,0}]")


# ----------------------------------------------------------------- the toy
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
def paired(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    env_mod.set_measure_timeout_s(TIMEOUT)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    yield
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def _run(monkeypatch, *, swell=None, fail_approx=False):
    real = _REAL_CACHED_COMPILE
    wrapped = {}

    def fake(key, fn, *a, **kw):
        if fail_approx and bytes(key).startswith(b"approx:"):
            try:
                raise XlaRuntimeError(_TRITON)
            except XlaRuntimeError as _e:
                raise env_mod.MeasureCompileFailure(_TRITON) from _e
        out = real(key, fn, *a, **kw)
        if swell is not None and bytes(key).startswith(b"approx:"):
            out = _Swollen(out, swell)
            wrapped[b"approx:"] = out
        return out

    monkeypatch.setattr(_cc, "cached_compile", fake)
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


def _is_hard_sentinel(reward):
    return bool(np.all(reward[list(env_mod.COMPUTE_REWARD_INDICES)]
                       <= env_mod.SENTINEL_COST * 0.99))


# ------------------------------------------------------------- gate refusal
def test_a_gate_refusal_is_scored_at_the_timeout_with_its_real_memory(
        paired, monkeypatch, capsys):
    reward, counts, recs, wrapped = _run(monkeypatch, swell=SWELL)
    assert counts == {"gate": 1, "total": 1}, counts
    assert not _is_hard_sentinel(reward)
    assert np.isfinite(reward).all()
    assert wrapped[b"approx:"].calls == 0, "a gated candidate must never run"
    rec = recs[-1]
    assert rec["refused"] == "gate"
    assert rec["sentinelled"] is True and rec["replayable"] is False
    assert rec["device_bytes_limit"] == LIMIT
    assert rec["static_peak_bytes"] > SWELL
    assert rec["refusal_timeout_s"] == TIMEOUT
    assert rec["refusal_timeout_factor"] == 1.0
    assert rec["refusal_latency_ns"] == TIMEOUT * 1e9
    # THE REAL MEMORY of the refused executable, read after its compile.
    ma = wrapped[b"approx:"].memory_analysis()
    cand = (float(ma.temp_size_in_bytes), float(ma.output_size_in_bytes),
            float(ma.argument_size_in_bytes))
    assert rec["mem_temp_bytes"] == cand[0] > SWELL
    assert rec["mem_output_bytes"] == cand[1]
    assert rec["mem_args_bytes"] == cand[2]
    assert rec["mem_peak_source"] == "not_measured"
    # THE REFERENCE was measured beside it, with its own window count.
    ref_lat = rec["ref_latency_ns"]
    ref = (rec["ref_temp_bytes"], rec["ref_output_bytes"],
           rec["ref_args_bytes"])
    assert ref_lat > 0.0 and ref[0] > 0.0
    assert rec["ref_measure_windows"] >= 1 and rec["ref_measure_inner"] >= 1
    assert rec["ref_measure_secs"] > 0.0
    assert rec["measure_windows"] == 0 and rec["measure_secs"] == 0.0
    # LATENCY = log(timeout / t_ref) on the channel's own convention.
    d_lat, d_mem, _ = paired_log_costs(TIMEOUT * 1e9, cand[0], ref_lat,
                                       ref[0])
    assert rec["rewards"][LAT] == -d_lat
    assert reward[LAT] == np.float32(-d_lat)
    assert d_lat > 10.0
    # SLOT 5 = the real static temp ratio; SLOT 11 = the ruling's three.
    assert rec["rewards"][MEM] == -d_mem
    assert d_mem == pytest.approx(math.log(cand[0] / ref[0]))
    want_obj, want_rec = mem_objective(cand, ref)
    assert rec["rewards"][MOBJ] == want_obj
    assert rec["mem_ratios"] == want_rec["ratios"]
    assert rec["mem_ratios"]["temp"] == pytest.approx(cand[0] / ref[0])
    assert rec["mem_ratios"]["out"] == pytest.approx(cand[1] / ref[1])
    assert rec["mem_ratios"]["args"] == pytest.approx(cand[2] / ref[2])
    # QUALITY 0, the bounded channels at their floors, the reserved at 0.
    assert reward[QUAL] == 0.0
    assert reward[REWARD_INDEX["fidelity"]] == -1.0
    assert reward[REWARD_INDEX["sparsity"]] == -1.0
    assert reward[REWARD_INDEX["grad_coverage"]] == 0.0
    assert reward[REWARD_INDEX["bkstep_acc"]] == 0.0
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[refused] gate")]
    assert line and f"limit_bytes={LIMIT}" in line[0]
    assert any("scored" in ln for ln in line)


def test_a_gate_refusal_scores_below_a_plan_that_ran(paired, monkeypatch):
    refused, counts, recs, _ = _run(monkeypatch, swell=SWELL)
    assert counts.get("gate") == 1
    measured, counts, recs, wrapped = _run(monkeypatch, swell=0)
    assert counts == {}, counts
    assert wrapped[b"approx:"].calls > 0
    assert "refused" not in recs[-1]
    for slot in (LAT, MEM, MOBJ):
        assert measured[slot] > refused[slot], (slot, measured, refused)


def test_the_sentinel_follows_the_configured_timeout(paired, monkeypatch):
    env_mod.set_measure_timeout_s(30.0)
    _, _, recs_a, _ = _run(monkeypatch, swell=SWELL)
    env_mod.set_measure_timeout_s(300.0)
    _, _, recs_b, _ = _run(monkeypatch, swell=SWELL)
    a, b = recs_a[-1], recs_b[-1]
    assert a["refusal_timeout_s"] == 30.0 and b["refusal_timeout_s"] == 300.0
    for rec, t in ((a, 30.0), (b, 300.0)):
        d_lat, _, _ = paired_log_costs(
            t * 1e9, rec["mem_temp_bytes"], rec["ref_latency_ns"],
            rec["ref_temp_bytes"])
        assert rec["rewards"][LAT] == -d_lat
    # the two references differ by timer noise only; the timeouts by 10x
    drift = math.log(b["ref_latency_ns"] / a["ref_latency_ns"])
    assert (b["rewards"][LAT] - a["rewards"][LAT]) == pytest.approx(
        -math.log(10.0) + drift, abs=1e-9)


def test_without_a_timeout_a_refusal_cannot_be_scored(paired, monkeypatch):
    env_mod.set_measure_timeout_s(None)
    with pytest.raises(RuntimeError, match="enforces no timeout"):
        _run(monkeypatch, swell=SWELL)


# ---------------------------------------------------------- compile failure
def test_a_compile_failure_is_scored_at_ten_times_the_timeout(
        paired, monkeypatch, capsys):
    reward, counts, recs, _ = _run(monkeypatch, fail_approx=True)
    assert counts == {"compile": 1, "total": 1}, counts
    assert not _is_hard_sentinel(reward)
    rec = recs[-1]
    assert rec["refused"] == "compile:XlaRuntimeError"
    assert rec["refusal_where"] == "approx compile"
    assert "Triton" in rec["refusal_error"]
    assert rec["refusal_timeout_factor"] == env_mod.COMPILE_REFUSAL_FACTOR
    assert rec["refusal_latency_ns"] == 10.0 * TIMEOUT * 1e9
    assert rec["refusal_bytes_limit"] == LIMIT
    assert rec["mem_temp_bytes"] is None
    ref_lat, ref_temp = rec["ref_latency_ns"], rec["ref_temp_bytes"]
    d_lat, d_mem, _ = paired_log_costs(10.0 * TIMEOUT * 1e9, LIMIT, ref_lat,
                                       ref_temp)
    assert rec["rewards"][LAT] == -d_lat
    assert reward[LAT] == np.float32(-d_lat)
    assert rec["rewards"][MEM] == -d_mem
    assert d_mem == pytest.approx(math.log(LIMIT / ref_temp))
    # bytes_limit against the reference temp on slot 11 too: one term
    assert rec["rewards"][MOBJ] == pytest.approx(-math.log(LIMIT / ref_temp))
    assert rec["mem_ratios"]["temp"] == pytest.approx(LIMIT / ref_temp)
    assert rec["mem_ratios"]["out"] == 1.0
    assert rec["mem_ratios"]["args"] == 1.0
    assert reward[QUAL] == 0.0
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[refused] compile")]
    assert line and "XlaRuntimeError" in line[0]


def test_a_compile_failure_scores_below_a_gate_refusal(paired, monkeypatch):
    compiled, _, _, _ = _run(monkeypatch, fail_approx=True)
    gated, _, _, _ = _run(monkeypatch, swell=SWELL)
    assert compiled[LAT] < gated[LAT]


def test_compile_measure_names_the_failure_and_keeps_the_cause(monkeypatch):
    monkeypatch.setitem(env_mod._MEASURE_TOOLCHAIN, "checked", True)

    class _Lowered:
        def compile(self, compiler_options=None):
            raise XlaRuntimeError("INVALID_ARGUMENT: no such option")

    with pytest.raises(env_mod.MeasureCompileFailure) as ei:
        env_mod._compile_measure(_Lowered())
    assert isinstance(ei.value.__cause__, XlaRuntimeError)
    assert isinstance(ei.value, RuntimeError)


# ------------------------------------------------------------- the timeout
@pytest.fixture
def fake_ray(monkeypatch):
    class _Future:
        def __init__(self, fn, *a, delay=0.0, **k):
            self._fn, self._a, self._k, self._delay = fn, a, k, float(delay)

        def result(self):
            return self._fn(*self._a, **self._k)

    class GetTimeoutError(Exception):
        pass

    class RayActorError(Exception):
        pass

    def _get(fut, timeout=None):
        if isinstance(fut, _Future):
            if timeout is not None and fut._delay > float(timeout):
                raise GetTimeoutError(f"{fut._delay} > {timeout}")
            return fut.result()
        return fut

    def _wait(futures, num_returns=None, timeout=None):
        ready = [f for f in futures
                 if not (timeout is not None and f._delay > float(timeout))]
        return ready, [f for f in futures if f not in ready]

    def _kill(actor, no_restart=False):
        actor.killed = True

    exc = types.SimpleNamespace(GetTimeoutError=GetTimeoutError,
                                RayActorError=RayActorError)
    ray = types.SimpleNamespace(get=_get, wait=_wait, kill=_kill,
                                cancel=lambda *a, **k: None, exceptions=exc)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return ray, _Future


class _Remote:
    def __init__(self, fn, future, delay=0.0, on_dispatch=None):
        self._fn, self._future, self._delay = fn, future, delay
        self._on_dispatch = on_dispatch

    def remote(self, *a, **k):
        if self._on_dispatch is not None:
            self._on_dispatch(k)
        return self._future(self._fn, *a, delay=self._delay, **k)


REF = {"latency_ns": 2.0e5, "memory_bytes": 4096.0,
       "static": (4096.0, 512.0, 1024.0), "temp_bytes": 4096.0,
       "output_bytes": 512.0, "args_bytes": 1024.0, "watermark_bytes": 0.0,
       "bytes_limit": LIMIT, "bytes_limit_source": "device",
       "measure_latency": True, "mem_channel": "temp",
       "cost_form": "paired-log", "wall_time": 0.0}


class _Actor:
    def __init__(self, future, *, delay, reference):
        # the timeout every dispatch carried, whether or not the call returned
        self.seen: list = []
        self.killed = False
        self._reference = reference
        self.evaluate = _Remote(
            self._evaluate, future, delay,
            on_dispatch=lambda k: self.seen.append(k.get("timeout_s")))
        self.last_reference = _Remote(self._last_reference, future)

    def _evaluate(self, order, specs, step, **kw):
        tokens = np.full((TOK,), 7, np.int32)
        eqn = np.full((TOK,), 3, np.int32)
        reward = np.full((NUM_REWARDS,), -100.0, np.float32)
        return tokens, eqn, reward

    def _last_reference(self, rule=None):
        return self._reference


def _pool(actors, **kw):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    return CpuApproxPool(
        actors, respawn_factory=None, max_tokens=TOK,
        num_rewards=NUM_REWARDS,
        cosine_sim_idx=int(REWARD_INDEX["cosine_sim"]),
        frob_residual_idx=int(REWARD_INDEX["frob_residual"]),
        fidelity_idx=int(REWARD_INDEX["fidelity"]),
        sparsity_idx=int(REWARD_INDEX["sparsity"]), **kw)


@pytest.fixture
def pool_env(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    yield
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def _batch(pool, n=2, step=3):
    orders = [np.arange(3, dtype=np.int32)] * n
    specs = [np.full((3, 1, 3), -1, np.int32)] * n
    return pool.evaluate_batch(orders, specs, [step] * n)


def test_a_timeout_is_scored_at_the_kill_timeout_against_the_reference(
        fake_ray, pool_env, capsys):
    ray, future = fake_ray
    slow = _Actor(future, delay=0.5, reference=None)
    fast = _Actor(future, delay=0.0, reference=REF)
    t = 0.01
    pool = _pool([slow, fast], timeout_s=t, initial_timeout_s=4 * t,
                 warm_after=3)
    tokens, eqn_ids, rewards, mask = _batch(pool)
    assert mask.tolist() == [True, False]
    assert slow.killed and not fast.killed
    # ONE NUMBER: the cold timeout the pool killed at is what it dispatched.
    assert slow.seen == [4 * t] and fast.seen == [4 * t]
    assert pool.stats()["timeouts"] == 1
    d_lat, d_mem, _ = paired_log_costs(4 * t * 1e9, LIMIT, REF["latency_ns"],
                                       REF["memory_bytes"])
    assert rewards[0, LAT] == np.float32(-d_lat)
    assert rewards[0, MEM] == np.float32(-d_mem)
    assert d_mem == pytest.approx(math.log(LIMIT / REF["memory_bytes"]))
    assert rewards[0, MOBJ] == pytest.approx(
        -math.log(LIMIT / REF["temp_bytes"]), rel=1e-6)
    assert rewards[0, QUAL] == 0.0
    assert not _is_hard_sentinel(rewards[0])
    assert (tokens[0] == 0).all()
    assert rewards[1, 0] == -100.0
    # counted and recorded in THIS process, the trainer's
    assert env_mod.consume_refused_counts() == {"timeout": 1, "total": 1}
    recs = env_mod.consume_plan_records()["records"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec["refused"] == "timeout"
    assert rec["sentinelled"] is True and rec["replayable"] is False
    assert rec["refusal_timeout_s"] == 4 * t
    assert rec["refusal_latency_ns"] == 4 * t * 1e9
    assert rec["refusal_bytes_limit"] == LIMIT
    assert rec["ref_latency_ns"] == REF["latency_ns"]
    assert rec["ref_temp_bytes"] == REF["temp_bytes"]
    assert rec["mem_ratios"]["temp"] == pytest.approx(LIMIT / REF["temp_bytes"])
    assert rec["rewards"][LAT] == -d_lat
    assert rec["order"] == [0, 1, 2]
    out = capsys.readouterr().out
    assert "[SENTINEL] batch timeout slot=0" in out
    assert "[refused] timeout scored step=3" in out


def test_the_timeout_sentinel_follows_the_pool_configuration(
        fake_ray, pool_env):
    ray, future = fake_ray
    rows = {}
    for t in (0.01, 0.02):
        slow = _Actor(future, delay=0.5, reference=None)
        fast = _Actor(future, delay=0.0, reference=REF)
        pool = _pool([slow, fast], timeout_s=t)
        rows[t] = _batch(pool)[2][0]
        assert slow.seen == [t]
    assert float(rows[0.02][LAT]) - float(rows[0.01][LAT]) == pytest.approx(
        -math.log(2.0), rel=1e-5)


def test_a_timeout_before_any_reference_raises(fake_ray, pool_env):
    ray, future = fake_ray
    slow = _Actor(future, delay=0.5, reference=None)
    fast = _Actor(future, delay=0.0, reference=None)
    pool = _pool([slow, fast], timeout_s=0.01)
    with pytest.raises(RuntimeError, match="paired reference"):
        _batch(pool)


def test_the_reference_is_remembered_for_a_lone_actor(fake_ray, pool_env):
    ray, future = fake_ray
    slow = _Actor(future, delay=0.5, reference=None)
    fast = _Actor(future, delay=0.0, reference=REF)
    pool = _pool([slow, fast], timeout_s=0.01)
    _batch(pool)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    # the fast actor dies too; the pool has only what it remembered
    fast_dead = _Actor(future, delay=0.5, reference=None)
    pool._alive.clear()
    pool._alive.append(fast_dead)
    rewards = _batch(pool, n=1)[2]
    d_lat, _, _ = paired_log_costs(0.01 * 1e9, LIMIT, REF["latency_ns"],
                                   REF["memory_bytes"])
    assert rewards[0, LAT] == np.float32(-d_lat)
    assert env_mod.consume_refused_counts() == {"timeout": 1, "total": 1}


def test_a_non_terminal_timeout_keeps_the_hard_sentinel(fake_ray, pool_env):
    ray, future = fake_ray
    slow = _Actor(future, delay=0.5, reference=None)
    fast = _Actor(future, delay=0.0, reference=REF)
    pool = _pool([slow, fast], timeout_s=0.01)
    rewards = _batch(pool, step=1)[2]
    assert _is_hard_sentinel(rewards[0])
    assert env_mod.consume_refused_counts() == {}
    assert env_mod.consume_plan_records()["records"] == []


def test_a_timeout_under_the_absolute_form_is_the_timeout_itself(
        fake_ray, pool_env, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "absolute")
    ray, future = fake_ray
    slow = _Actor(future, delay=0.5, reference=None)
    fast = _Actor(future, delay=0.0,
                  reference=dict(REF, cost_form="absolute"))
    pool = _pool([slow, fast], timeout_s=0.01)
    rewards = _batch(pool)[2]
    assert rewards[0, LAT] == np.float32(-0.01 * 1e9)
    assert rewards[0, MEM] == np.float32(-LIMIT)
    assert rewards[0, MOBJ] == 0.0
    assert rewards[0, QUAL] == 0.0


def test_the_single_dispatch_scores_a_terminal_timeout(fake_ray, pool_env):
    ray, future = fake_ray
    slow = _Actor(future, delay=0.5, reference=None)
    fast = _Actor(future, delay=0.0, reference=REF)
    pool = _pool([slow, fast], timeout_s=0.01)
    out = pool.evaluate(np.arange(3, dtype=np.int32),
                        np.full((3, 1, 3), -1, np.int32), 3, None)
    d_lat, _, _ = paired_log_costs(0.01 * 1e9, LIMIT, REF["latency_ns"],
                                   REF["memory_bytes"])
    assert out[-1][LAT] == np.float32(-d_lat)
    assert slow.seen == [0.01]
    assert env_mod.consume_refused_counts() == {"timeout": 1, "total": 1}


# ------------------------------------------------------------- the formula
def test_refused_reward_rejects_a_missing_timeout_or_limit():
    with pytest.raises(RuntimeError, match="no timeout"):
        env_mod.refused_reward("timeout", timeout_s=None, reference=REF)
    with pytest.raises(RuntimeError, match="no timeout"):
        env_mod.refused_reward("gate", timeout_s=0.0, reference=REF,
                               candidate_static=(1.0, 1.0, 1.0))
    with pytest.raises(RuntimeError, match="bytes_limit"):
        env_mod.refused_reward("compile", timeout_s=1.0,
                               reference=dict(REF, bytes_limit=None))
    with pytest.raises(ValueError):
        env_mod.refused_reward("oom", timeout_s=1.0, reference=REF)


def test_refused_reward_is_strictly_worse_than_any_gated_plan(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    # a measured plan: 0.9 ms against a 1 ms timeout, temp at the limit's edge
    cand_lat, cand_temp = 0.9e6, float(LIMIT) - 1.0
    d_lat, d_mem, _ = paired_log_costs(cand_lat, cand_temp,
                                       REF["latency_ns"], REF["memory_bytes"])
    measured = (-d_lat, -d_mem)
    for kind in ("timeout", "compile"):
        slots, _ = env_mod.refused_reward(kind, timeout_s=0.001,
                                          reference=REF)
        assert slots[LAT] < measured[0]
        assert slots[MEM] < measured[1]
    slots, _ = env_mod.refused_reward(
        "gate", timeout_s=0.001, reference=REF,
        candidate_static=(cand_temp + 1.0, 512.0, 1024.0))
    assert slots[LAT] < measured[0]
    assert slots[MEM] < measured[1]
