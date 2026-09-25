# dsnn-dfw.223: a replacement actor takes the dead actor's slot (its device and its cores). The
# respawn used to take the next slot of a running counter, which put the replacement on a device a
# live actor was measuring on, and both then ran out of memory (jobs 68194, 68196).
from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest

NUM_REWARDS = 8
COS = 6
FROB = 7
TOK = 8
CLEAN, OOM = 0, 1
SCORED = np.array([-0.2, -0.1, -13.9, -0.3, -0.4, -3.2, 0.0, 0.0], np.float32)
MEASURED = np.array([-0.2, -0.1, -0.5, -0.3, -0.4, -0.1, 0.9, 0.0], np.float32)


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
    def __init__(self, name):
        self.name = name
        self.killed = False
        self.kinds: list = []
        self._flag = False
        self.evaluate = _Remote(self._evaluate)
        self.pop_oom_flag = _Remote(self._pop)
        self.ready = _Remote(lambda: True)

    def _evaluate(self, order, specs, step, **kw):
        kind = int(np.asarray(order).reshape(-1)[0])
        self.kinds.append(kind)
        if kind == OOM:
            self._flag = True
        row = SCORED if kind == OOM else MEASURED
        return (np.full((TOK,), 7, np.int32), np.full((TOK,), 3, np.int32), row.copy())

    def _pop(self):
        was, self._flag = self._flag, False
        return was


def _pool(n_actors, timeout_s=300.0):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    first = [_Actor(f"a{i}") for i in range(n_actors)]
    calls: list = []
    fresh: list = []

    def factory(slot=None):
        calls.append(slot)
        fresh.append(_Actor(f"fresh{len(fresh)}@{slot}"))
        return fresh[-1]

    pool = CpuApproxPool(
        first, timeout_s=timeout_s, respawn_factory=factory, max_tokens=TOK,
        num_rewards=NUM_REWARDS, cosine_sim_idx=COS, frob_residual_idx=FROB)
    return pool, first, fresh, calls


def _batch(pool, kinds):
    n = len(kinds)
    return pool.evaluate_batch([np.array([k, 1, 2], np.int32) for k in kinds],
                               [np.full((3, 1, 3), -1, np.int32)] * n, [3] * n)


def test_an_oom_recycle_respawns_into_the_dead_actors_slot(fake_ray):
    pool, first, fresh, calls = _pool(3)
    _batch(pool, [CLEAN, OOM, CLEAN])
    assert first[1].killed and not first[0].killed and not first[2].killed
    assert calls == [1], "the replacement takes slot 1, the slot of the actor that ran out of memory"
    assert pool.live_actors() == [first[0], fresh[0], first[2]]
    _batch(pool, [CLEAN, CLEAN, OOM])
    assert calls == [1, 2]
    _batch(pool, [CLEAN, OOM, CLEAN])
    assert calls == [1, 2, 1], "a replacement's own replacement keeps the slot"


def test_a_deadline_kill_respawns_into_the_dead_actors_slot(fake_ray):
    pool, first, fresh, calls = _pool(2, timeout_s=0.01)
    stuck = _Future(lambda: None)
    first[1].evaluate = types.SimpleNamespace(remote=lambda *a, **k: stuck)
    real_get = fake_ray.get

    def _get(fut, timeout=None):
        if fut is stuck:
            raise fake_ray.exceptions.GetTimeoutError()
        return real_get(fut, timeout)

    def _wait(futs, num_returns=1, timeout=None):
        return [f for f in futs if f is not stuck], [f for f in futs if f is stuck]
    fake_ray.get, fake_ray.wait = _get, _wait
    _batch(pool, [CLEAN, CLEAN])
    assert first[1].killed and pool.stats()["timeouts"] == 1
    stop = time.time() + 10.0
    while pool.size() < 2 and time.time() < stop:
        time.sleep(0.01)
    assert calls == [1]
    assert fresh and pool.size() == 2


def test_a_factory_without_a_slot_keyword_is_called_as_before(fake_ray):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    first = [_Actor("a0")]
    fresh: list = []

    def factory():
        fresh.append(_Actor("fresh"))
        return fresh[-1]

    pool = CpuApproxPool(first, timeout_s=300.0, respawn_factory=factory, max_tokens=TOK,
                         num_rewards=NUM_REWARDS, cosine_sim_idx=COS, frob_residual_idx=FROB)
    _batch(pool, [OOM])
    assert first[0].killed and pool.live_actors() == [fresh[0]]
