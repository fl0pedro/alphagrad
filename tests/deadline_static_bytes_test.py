# dsnn-dfw.302 (the Failed plan rule as written): a killed call whose compile had ended is scored
# on the real static bytes of its program. The measure actor hands them to a board outside itself
# when the compile ends, before the gate and the timed runs, so they outlive the kill. The pool
# takes them after the kill and sends them with the rescore. A kill before the compile ended keeps
# the full card. Until then every killed plan was scored on the full card.
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
    NUM_REWARDS, REWARD_INDEX, mem_objective)

_REAL_CACHED_COMPILE = _cc.cached_compile
_REAL_GATE = env_mod._static_peak_gate
DEADLINE = 0.25
LIMIT = 16_000_000_000
QUAL = int(REWARD_INDEX["quality"])
LAT = int(REWARD_INDEX["latency_ns"])
MOBJ = int(REWARD_INDEX["mem_objective"])
_SAMPLES = (np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)]),)


class _KilledHere(BaseException):
    # Where the test's actor stops, as a process the deadline kills stops. Not an Exception, so nothing scores it.
    pass


def _toy_env():
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    return VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(toy)(x), args=[x], argnums=(0,), num_envs=0,
        target_fun=toy, measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)


def _plan(env):
    from alphagrad.approx.env import MAX_RULES_PER_VERTEX

    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs = np.full((len(order), MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    return np.asarray(order, np.int32), specs


@pytest.fixture
def paired(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setenv("ALPHAGRAD_DISABLE_JIT_DISK_CACHE", "1")
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    state = {"seen": [], "kill": None}

    def fake(key, fn, *a, **kw):
        head = bytes(key).split(b":", 1)[0]
        state["seen"].append(head)
        if state["kill"] == "compile" and head == b"approx":
            raise _KilledHere()
        return _REAL_CACHED_COMPILE(key, fn, *a, **kw)

    def gate(*a, **kw):
        if state["kill"] == "after compile":
            raise _KilledHere()
        return _REAL_GATE(*a, **kw)

    monkeypatch.setattr(_cc, "cached_compile", fake)
    monkeypatch.setattr(env_mod, "_static_peak_gate", gate)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()
    yield state
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()


class _Future:
    def __init__(self, fn, *a, **k):
        self._fn, self._a, self._k = fn, a, k

    def result(self):
        return self._fn(*self._a, **self._k)


class _Stuck:
    pass


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return _Future(self._fn, *a, **k)


@pytest.fixture
def fake_ray(monkeypatch):
    class GetTimeoutError(Exception):
        pass

    class RayActorError(Exception):
        pass

    def _get(fut, timeout=None):
        if isinstance(fut, _Stuck):
            raise GetTimeoutError("the call is still running at its deadline")
        return fut.result() if isinstance(fut, _Future) else fut

    def _wait(futures, num_returns=1, timeout=None):
        ready = [f for f in futures if not isinstance(f, _Stuck)][:num_returns]
        return ready, [f for f in futures if f not in ready]

    def _kill(actor, no_restart=False):
        actor.killed = True

    def _ray_remote(*_a, **_k):
        # ray.remote(num_cpus=0)(cls).remote(): an actor whose methods answer through .remote().
        def deco(cls):
            def _start(*ca, **ck):
                inst = cls(*ca, **ck)
                return types.SimpleNamespace(**{
                    n: _Remote(getattr(inst, n)) for n in ("put", "take")})
            return types.SimpleNamespace(remote=_start)
        return deco

    exc = types.SimpleNamespace(GetTimeoutError=GetTimeoutError,
                                RayActorError=RayActorError)
    ray = types.SimpleNamespace(get=_get, wait=_wait, kill=_kill, remote=_ray_remote,
                                is_initialized=lambda: True,
                                cancel=lambda *a, **k: None, exceptions=exc)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return ray


class _KilledActor:
    # A real server that stops where `paired["kill"]` says, and loses what its process held, as a killed actor does.
    def __init__(self, server, state, where):
        self.killed = False
        self.reported: list = []
        self._server, self._state, self._where = server, state, where
        self.evaluate = types.SimpleNamespace(remote=self._evaluate)
        self.pop_oom_flag = _Remote(lambda: False)
        self.ready = _Remote(lambda: True)
        self.consume_call_telemetry = _Remote(lambda: {})

    def _evaluate(self, *a, rule=None, **k):
        sink = k.get("static_to")
        if sink is not None:
            def _tee(triple):
                self.reported.append(tuple(float(x) for x in triple))
                sink(triple)
            k["static_to"] = _tee
        self._state["kill"] = self._where
        try:
            self._server.evaluate(*a, **k)
        except _KilledHere:
            env_mod.consume_refused_counts()
            env_mod.consume_plan_records()
            return _Stuck()
        finally:
            self._state["kill"] = None
        raise AssertionError("the test's actor was meant to be killed")


class _ServerActor:
    def __init__(self, server):
        self.killed = False
        self.refused: list = []
        self._server = server
        self.evaluate = _Remote(self._evaluate)
        self.pop_oom_flag = _Remote(server.pop_oom_flag)
        self.ready = _Remote(lambda: True)
        self.consume_call_telemetry = _Remote(lambda: {})

    def _evaluate(self, *a, rule=None, **k):
        self.refused.append((k.get("refuse"), k.get("refuse_static")))
        return self._server.evaluate(*a, **k)


def _pool(env, state, where):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    killed = _KilledActor(CpuApproximationServer.from_env(env), state, where)
    scorer = _ServerActor(CpuApproximationServer.from_env(env))
    pool = CpuApproxPool(
        [killed, scorer], timeout_s=DEADLINE, respawn_factory=None,
        max_tokens=int(env.obs_width), token_dtype=env.wire_token_dtype,
        emit_eqn_ids=not env.config.delta_obs, num_rewards=NUM_REWARDS,
        cosine_sim_idx=int(REWARD_INDEX["cosine_sim"]),
        frob_residual_idx=int(REWARD_INDEX["frob_residual"]),
        fidelity_idx=int(REWARD_INDEX["fidelity"]),
        sparsity_idx=int(REWARD_INDEX["sparsity"]))
    return pool, killed, scorer


def _measured_static(env):
    # The real static bytes of the plan's program, from a measurement that nothing kills.
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    order, specs = _plan(env)
    CpuApproximationServer.from_env(env).evaluate(
        order, specs, len(order), eval_samples=_SAMPLES, timeout_s=300.0)
    rec = env_mod.consume_plan_records()["records"][-1]
    env_mod.consume_refused_counts()
    assert not rec.get("refused"), rec.get("refused")
    return (rec["mem_temp_bytes"], rec["mem_output_bytes"], rec["mem_args_bytes"])


def _dispatch(pool, env, single):
    order, specs = _plan(env)
    if single:
        return np.asarray(pool.evaluate(order, specs, len(order), _SAMPLES)[-1],
                          dtype=np.float32)
    out = pool.evaluate_batch([order], [specs], [len(order)], eval_samples=_SAMPLES)
    assert out[-1].tolist() == [False], "the failed plan trains"
    return np.asarray(out[-2][0], dtype=np.float32)


def _failed_plan_record(reward):
    counts = env_mod.consume_refused_counts()
    assert counts == {"deadline": 1, "total": 1, "scored": 1}, counts
    rec = env_mod.consume_plan_records()["records"][-1]
    assert rec["refused"] == "deadline" and rec["refusal_where"] == "the pool's deadline"
    assert rec["refusal_timeout_s"] == DEADLINE
    assert np.isfinite(reward).all() and reward[QUAL] == 0.0
    assert reward[LAT] == pytest.approx(
        -math.log(DEADLINE * 1e9 / rec["ref_latency_ns"]), rel=1e-5)
    return rec


@pytest.mark.parametrize("single", [False, True])
def test_a_kill_after_the_compile_scores_with_the_real_static_bytes(paired, fake_ray, single):
    env = _toy_env()
    real = _measured_static(env)
    pool, killed, scorer = _pool(env, paired, "after compile")
    paired["seen"].clear()
    reward = _dispatch(pool, env, single)
    assert killed.killed and killed.reported == [real], "reported when the compile ended"
    assert scorer.refused == [("deadline", list(real))]
    rec = _failed_plan_record(reward)
    assert rec["refusal_compiled"] is True and rec["refusal_program"] is True
    assert rec["refusal_sentinel"] is None, "no full card"
    assert (rec["mem_temp_bytes"], rec["mem_output_bytes"], rec["mem_args_bytes"]) == real
    ref = (rec["ref_temp_bytes"], rec["ref_output_bytes"], rec["ref_args_bytes"])
    assert reward[MOBJ] == np.float32(mem_objective(real, ref)[0])
    assert paired["seen"].count(b"approx") == 1, "the rescore compiles nothing again"
    stats = pool.stats()
    assert (stats["deadline_scored"], stats["deadline_static"]) == (1, 1)


@pytest.mark.parametrize("single", [False, True])
def test_a_kill_during_the_compile_scores_with_the_full_card(paired, fake_ray, single):
    env = _toy_env()
    pool, killed, scorer = _pool(env, paired, "compile")
    reward = _dispatch(pool, env, single)
    assert killed.killed and killed.reported == [], "the compile never ended"
    assert scorer.refused == [("deadline", None)]
    rec = _failed_plan_record(reward)
    assert rec["refusal_compiled"] is False and rec["refusal_program"] is False
    assert rec["refusal_sentinel"]["branch"] in ("fill", "split")
    assert rec["mem_temp_bytes"] is None
    stats = pool.stats()
    assert (stats["deadline_scored"], stats["deadline_static"]) == (1, 0)


def test_the_server_hands_out_the_static_bytes_before_the_gate(paired, monkeypatch):
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    env = _toy_env()
    order, specs = _plan(env)
    events: list = []

    def gate(*a, **kw):
        events.append("gate")
        return _REAL_GATE(*a, **kw)

    monkeypatch.setattr(env_mod, "_static_peak_gate", gate)
    CpuApproximationServer.from_env(env).evaluate(
        order, specs, len(order), eval_samples=_SAMPLES, timeout_s=300.0,
        static_to=lambda triple: events.append(tuple(float(x) for x in triple)))
    rec = env_mod.consume_plan_records()["records"][-1]
    real = (rec["mem_temp_bytes"], rec["mem_output_bytes"], rec["mem_args_bytes"])
    assert events == [real, "gate"], "once, with the real bytes, before the gate"
    assert env_mod._COMPILE_END_SINK[0] is None, "the sink belongs to one call"


def test_the_static_bytes_travel_only_with_a_forced_refusal(paired):
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    with pytest.raises(ValueError, match="temp, output"):
        env_mod.forced_refusal("deadline", DEADLINE, static=(1.0, 2.0))
    with pytest.raises(ValueError, match="not negative"):
        env_mod.forced_refusal("deadline", DEADLINE, static=(1.0, -2.0, 3.0))
    env = _toy_env()
    order, specs = _plan(env)
    with pytest.raises(ValueError, match="only with the forced refusal"):
        CpuApproximationServer.from_env(env).evaluate(
            order, specs, len(order), eval_samples=_SAMPLES, timeout_s=DEADLINE,
            refuse_static=[1.0, 2.0, 3.0])


def test_the_board_keeps_the_newest_entries_and_gives_each_one_once():
    from alphagrad.approx.cpu_approx_pool import _STATIC_BOARD_CAP, _StaticBoard

    board = _StaticBoard()
    for k in range(_STATIC_BOARD_CAP + 3):
        board.put(f"t{k}", (k, 1, 2))
    assert board.take("t0") is None, "the oldest entry went first"
    assert board.take(f"t{_STATIC_BOARD_CAP + 2}") == [_STATIC_BOARD_CAP + 2.0, 1.0, 2.0]
    assert board.take(f"t{_STATIC_BOARD_CAP + 2}") is None
