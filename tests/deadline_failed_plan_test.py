# dsnn-dfw.292 (owner ruling 2026-09-26, Q9b c): a measurement the deadline kills is a failed
# plan. The pool sends it to a free actor, whose real server scores it with the sentinel and the
# reason "deadline" before any compile. The row enters the update and its record carries the
# reason. Before this ruling the killed call was a masked row with an excluded "timeout" record.
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
from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX      # noqa: E402

_REAL_CACHED_COMPILE = _cc.cached_compile
DEADLINE = 0.25
LIMIT = 16_000_000_000
QUAL = int(REWARD_INDEX["quality"])
LAT = int(REWARD_INDEX["latency_ns"])
_SAMPLES = (np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)]),)


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
    seen: list = []

    def fake(key, fn, *a, **kw):
        seen.append(bytes(key).split(b":", 1)[0])
        return _REAL_CACHED_COMPILE(key, fn, *a, **kw)

    monkeypatch.setattr(_cc, "cached_compile", fake)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()
    yield seen
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

    exc = types.SimpleNamespace(GetTimeoutError=GetTimeoutError,
                                RayActorError=RayActorError)
    ray = types.SimpleNamespace(get=_get, wait=_wait, kill=_kill,
                                cancel=lambda *a, **k: None, exceptions=exc)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return ray


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return _Future(self._fn, *a, **k)


class _StuckActor:
    # Its measurement outlasts every deadline, as a compile that never returns does.
    def __init__(self):
        self.killed = False
        self.evaluate = types.SimpleNamespace(remote=lambda *a, **k: _Stuck())
        self.pop_oom_flag = _Remote(lambda: False)
        self.ready = _Remote(lambda: True)
        self.consume_call_telemetry = _Remote(lambda: {})


class _ServerActor:
    # The Ray wrapper's surface around the real server of this process.
    def __init__(self, server):
        self.killed = False
        self.refused: list = []
        self._server = server
        self.evaluate = _Remote(self._evaluate)
        self.pop_oom_flag = _Remote(server.pop_oom_flag)
        self.ready = _Remote(lambda: True)
        # The records stay in this process, where the checks read them.
        self.consume_call_telemetry = _Remote(lambda: {})

    def _evaluate(self, *a, rule=None, **k):
        self.refused.append(k.get("refuse"))
        return self._server.evaluate(*a, **k)


def _pool(env):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    stuck = _StuckActor()
    server = _ServerActor(CpuApproximationServer.from_env(env))
    pool = CpuApproxPool(
        [stuck, server], timeout_s=DEADLINE, respawn_factory=None,
        max_tokens=int(env.obs_width), token_dtype=env.wire_token_dtype,
        emit_eqn_ids=not env.config.delta_obs, num_rewards=NUM_REWARDS,
        cosine_sim_idx=int(REWARD_INDEX["cosine_sim"]),
        frob_residual_idx=int(REWARD_INDEX["frob_residual"]),
        fidelity_idx=int(REWARD_INDEX["fidelity"]),
        sparsity_idx=int(REWARD_INDEX["sparsity"]))
    return pool, stuck, server


def _is_hard_sentinel(reward):
    return bool(np.all(np.asarray(reward)[list(env_mod.COMPUTE_REWARD_INDICES)]
                       <= env_mod.SENTINEL_COST * 0.99))


def _check_failed_plan(reward, seen):
    counts = env_mod.consume_refused_counts()
    assert counts == {"deadline": 1, "total": 1, "scored": 1}, counts
    rec = env_mod.consume_plan_records()["records"][-1]
    assert rec["refused"] == "deadline" and rec["sentinelled"] is True
    assert rec["refusal_where"] == "the pool's deadline"
    assert rec["refusal_timeout_s"] == DEADLINE
    assert rec["refusal_latency_ns"] == DEADLINE * 1e9
    assert rec["refusal_program"] is False, "the program died with the killed actor"
    assert rec["refusal_bytes_limit"] == LIMIT
    assert b"approx" not in seen, "a killed plan is not compiled again"
    assert np.isfinite(reward).all() and not _is_hard_sentinel(reward)
    assert reward[QUAL] == 0.0
    assert reward[LAT] == pytest.approx(
        -math.log(DEADLINE * 1e9 / rec["ref_latency_ns"]), rel=1e-5)
    assert env_mod._FORCED_REFUSAL[0] is None, "the forced refusal belongs to one call"


def test_a_killed_call_enters_the_update_with_the_sentinel(paired, fake_ray):
    env = _toy_env()
    pool, stuck, server = _pool(env)
    order, specs = _plan(env)
    out = pool.evaluate_batch([order], [specs], [len(order)],
                              eval_samples=_SAMPLES)
    reward, mask = np.asarray(out[-2][0], dtype=np.float32), out[-1]
    assert stuck.killed and server.refused == ["deadline"]
    assert mask.tolist() == [False], "the failed plan trains"
    _check_failed_plan(reward, paired)
    stats = pool.stats()
    assert (stats["timeouts"], stats["deadline_scored"],
            stats["deadline_unscored"]) == (1, 1, 0)


def test_the_single_dispatch_scores_a_killed_call_the_same_way(paired, fake_ray):
    env = _toy_env()
    pool, stuck, server = _pool(env)
    order, specs = _plan(env)
    out = pool.evaluate(order, specs, len(order), _SAMPLES)
    assert stuck.killed and server.refused == ["deadline"]
    _check_failed_plan(np.asarray(out[-1], dtype=np.float32), paired)
    assert pool.stats()["deadline_scored"] == 1


def test_the_server_refuses_to_force_any_other_kind(paired):
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    env = _toy_env()
    server = CpuApproximationServer.from_env(env)
    order, specs = _plan(env)
    was = (env_mod._FORCED_REFUSAL[0], env_mod._MEASURE_TIMEOUT_S[0], env_mod._ENV_SLOT[0])
    with pytest.raises(ValueError, match="only a call the deadline killed"):
        server.evaluate(order, specs, len(order), eval_samples=_SAMPLES,
                        env_row=3, timeout_s=DEADLINE, refuse="gate")
    assert (env_mod._FORCED_REFUSAL[0], env_mod._MEASURE_TIMEOUT_S[0],
            env_mod._ENV_SLOT[0]) == was, "a refused request leaves the actor as it was"
