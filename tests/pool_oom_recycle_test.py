# dsnn-cl0, owner ruling 2026-09-24 Q52 (dsnn-dfw.123, dsnn-nnx): the pool reads
# the out-of-memory flag after every row and recycles on a fresh one only.
from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

NUM_REWARDS = 8
COS = 6
FROB = 7
TOK = 8
CLEAN, OOM, SENTINEL = 0, 1, 2
# A scored refusal: finite on every channel, frob_residual 0.0.
SCORED = np.array([-0.2, -0.1, -13.9, -0.3, -0.4, -3.2, 0.0, 0.0],
                  np.float32)
MEASURED = np.array([-0.2, -0.1, -0.5, -0.3, -0.4, -0.1, 0.9, 0.0],
                    np.float32)
HARD = np.full((NUM_REWARDS,), -1e10, np.float32)
HARD[COS] = 0.0


class _Future:
    def __init__(self, fn, *a, **k):
        self._fn, self._a, self._k = fn, a, k

    def result(self):
        return self._fn(*self._a, **self._k)


@pytest.fixture
def fake_ray(monkeypatch):
    class GetTimeoutError(Exception):
        pass

    class RayActorError(Exception):
        pass

    def _get(fut, timeout=None):
        return fut.result() if isinstance(fut, _Future) else fut

    def _wait(futures, num_returns=None, timeout=None):
        return list(futures), []

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


class _Actor:
    # order[0] is the row kind; an OOM flag stays set until it is popped.
    def __init__(self, name):
        self.name = name
        self.killed = False
        self.kinds: list = []
        self.pops = 0
        self._flag = False
        self.evaluate = _Remote(self._evaluate)
        self.pop_oom_flag = _Remote(self._pop)
        self.ready = _Remote(lambda: True)

    def _evaluate(self, order, specs, step, **kw):
        kind = int(np.asarray(order).reshape(-1)[0])
        self.kinds.append(kind)
        if kind == SENTINEL:
            return (np.zeros((TOK,), np.int32), np.zeros((TOK,), np.int32),
                    HARD.copy())
        if kind == OOM:
            self._flag = True
        row = SCORED if kind == OOM else MEASURED
        return (np.full((TOK,), 7, np.int32), np.full((TOK,), 3, np.int32),
                row.copy())

    def _pop(self):
        self.pops += 1
        was, self._flag = self._flag, False
        return was


def _pool(n_actors=1):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    first = [_Actor(f"a{i}") for i in range(n_actors)]
    fresh: list = []

    def factory():
        fresh.append(_Actor(f"fresh{len(fresh)}"))
        return fresh[-1]

    pool = CpuApproxPool(
        first, timeout_s=300.0, respawn_factory=factory, max_tokens=TOK,
        num_rewards=NUM_REWARDS, cosine_sim_idx=COS, frob_residual_idx=FROB)
    return pool, first, fresh


def _order(kind):
    return np.array([kind, 1, 2], np.int32)


def _batch(pool, kinds):
    n = len(kinds)
    return pool.evaluate_batch([_order(k) for k in kinds],
                               [np.full((3, 1, 3), -1, np.int32)] * n,
                               [3] * n)


def test_an_oom_scored_row_recycles_its_actor_and_a_clean_row_after_it_does_not(
        fake_ray):
    pool, (actor,), fresh = _pool()
    _tok, _eqn, rewards, mask = _batch(pool, [OOM])
    assert actor.killed, "the actor whose call ran out of memory must go"
    assert len(fresh) == 1 and pool.live_actors() == [fresh[0]]
    assert pool.stats()["oom_recycles"] == 1
    assert mask.tolist() == [False], "a scored row stays in the update"
    np.testing.assert_array_equal(rewards[0], SCORED)
    assert fresh[0].kinds == [], "the row is not measured again"

    _tok, _eqn, rewards, mask = _batch(pool, [CLEAN])
    assert fresh[0].kinds == [CLEAN] and not fresh[0].killed
    assert len(fresh) == 1 and pool.stats()["oom_recycles"] == 1
    assert mask.tolist() == [False]
    np.testing.assert_array_equal(rewards[0], MEASURED)


def test_a_stale_flag_never_recycles_a_later_call(fake_ray, capsys):
    # Two waves on one actor: an OOM row, then a benign -1e10 row.
    pool, (actor,), fresh = _pool()
    _tok, _eqn, rewards, mask = _batch(pool, [OOM, SENTINEL])
    assert pool.stats()["oom_recycles"] == 1
    assert mask.tolist() == [False, False]
    np.testing.assert_array_equal(rewards[0], SCORED)
    np.testing.assert_array_equal(rewards[1], HARD)
    assert actor.kinds == [OOM], "the recycle follows the wave of the OOM"
    assert fresh[0].kinds == [SENTINEL] and not fresh[0].killed
    assert "[POOL] oom-recycle actor#0 slots=[0]" in capsys.readouterr().out


def test_the_flag_is_read_after_every_row(fake_ray):
    pool, actors, fresh = _pool(n_actors=3)
    _batch(pool, [CLEAN, CLEAN, CLEAN, CLEAN])
    assert sorted(a.pops for a in actors) == [1, 1, 2]
    assert fresh == [] and pool.stats()["oom_recycles"] == 0


def test_an_oom_hard_sentinel_row_stays_excluded(fake_ray):
    # The server's -1e10 row of an OOM that escaped the callback.
    pool, (actor,), fresh = _pool()
    actor._flag = False

    def _escaped(order, specs, step, **kw):
        actor._flag = True
        return (np.zeros((TOK,), np.int32), np.zeros((TOK,), np.int32),
                HARD.copy())
    actor.evaluate = _Remote(_escaped)
    _tok, _eqn, rewards, mask = _batch(pool, [CLEAN])
    assert mask.tolist() == [True] and actor.killed and len(fresh) == 1


def test_the_single_dispatch_recycles_on_an_oom_row_and_not_after_it(
        fake_ray):
    pool, (actor,), fresh = _pool()
    out = pool.evaluate(_order(OOM), np.full((3, 1, 3), -1, np.int32), 3,
                        None)
    np.testing.assert_array_equal(out[-1], SCORED)
    assert actor.killed and actor.pops == 1
    assert len(fresh) == 1 and pool.live_actors() == [fresh[0]]
    assert pool.stats()["oom_recycles"] == 1
    out = pool.evaluate(_order(CLEAN), np.full((3, 1, 3), -1, np.int32), 3,
                        None)
    np.testing.assert_array_equal(out[-1], MEASURED)
    assert fresh[0].pops == 1 and not fresh[0].killed
    assert len(fresh) == 1 and pool.stats()["oom_recycles"] == 1
    assert pool.live_actors() == [fresh[0]]


# ---------------------------------------------------------- the real worker
_OOM_TEXT = ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
             "17179869184 bytes.")


class _XlaRuntimeError(RuntimeError):
    pass


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

    env = VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(toy)(x), args=[x], argnums=(0,), num_envs=0,
        target_fun=toy, measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)
    srv = CpuApproximationServer.from_env(env)
    env_mod.pop_measure_oom()
    env_mod.consume_refused_counts()
    yield srv, env
    env_mod.register_measure_oom_consumer(False)
    env_mod.pop_measure_oom()
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def _terminal(srv, env, timeout_s):
    from alphagrad.approx.env import MAX_RULES_PER_VERTEX

    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs = np.full((len(order), MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    samples = (np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)]),)
    return srv.evaluate(np.asarray(order, np.int32), specs, len(order),
                        eval_samples=samples, timeout_s=timeout_s)


def test_the_worker_flag_belongs_to_the_call_that_ran_out_of_memory(
        server, monkeypatch):
    # No deadline: the scorer raises after the callback recorded the OOM.
    from alphagrad.approx.common import compile_cache as cc

    srv, env = server
    real = cc.cached_compile

    def oom(key, fn, *a, **kw):
        if bytes(key).startswith(b"approx:"):
            raise _XlaRuntimeError(_OOM_TEXT)
        return real(key, fn, *a, **kw)

    monkeypatch.setattr(cc, "cached_compile", oom)
    out = _terminal(srv, env, None)
    assert float(np.asarray(out[-1]).min()) <= -1e9
    assert srv.pop_oom_flag() is True
    monkeypatch.setattr(cc, "cached_compile", real)
    out = _terminal(srv, env, 300.0)
    assert float(np.asarray(out[-1]).min()) > -1e9
    assert srv.pop_oom_flag() is False


def test_a_flag_nobody_read_does_not_survive_the_next_call(server):
    srv, env = server
    srv._last_was_oom = True
    _terminal(srv, env, 300.0)
    assert srv.pop_oom_flag() is False
