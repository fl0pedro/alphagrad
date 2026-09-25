# dsnn-dfw.223: the measurement pool serves a batch as a work queue, not in waves of M. A slot
# goes to the first free actor, so a fast plan does not hold its actor until the slowest plan of
# its wave is back, and a deadline kill takes only the killed slot with it.
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


class _Clock:
    t = 0.0


class _Future:
    def __init__(self, fn, dur, clock, *a, **k):
        self._fn, self._a, self._k = fn, a, k
        self.done_at = clock.t + float(dur)

    def result(self):
        return self._fn(*self._a, **self._k)


@pytest.fixture
def sim_ray(monkeypatch):
    clock = _Clock()

    class GetTimeoutError(Exception):
        pass

    class RayActorError(Exception):
        pass

    def _get(fut, timeout=None):
        if not isinstance(fut, _Future):
            return fut
        if fut.done_at > clock.t:
            if timeout is not None and fut.done_at - clock.t > float(timeout):
                raise GetTimeoutError(f"done at {fut.done_at}, now {clock.t}")
            clock.t = fut.done_at
        return fut.result()

    def _wait(futures, num_returns=1, timeout=None):
        futs = sorted(futures, key=lambda f: f.done_at)
        want = futs[:num_returns]
        latest = max(f.done_at for f in want)
        if timeout is not None and latest - clock.t > float(timeout):
            clock.t += float(timeout)
            ready = [f for f in futs if f.done_at <= clock.t]
            return ready, [f for f in futs if f not in ready]
        clock.t = max(clock.t, latest)
        return want, [f for f in futs if f not in want]

    def _kill(actor, no_restart=False):
        actor.killed = True

    exc = types.SimpleNamespace(GetTimeoutError=GetTimeoutError,
                                RayActorError=RayActorError)
    ray = types.SimpleNamespace(get=_get, wait=_wait, kill=_kill,
                                cancel=lambda *a, **k: None, exceptions=exc)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return ray, clock


class _Remote:
    def __init__(self, fn, dur, clock, on_dispatch=None):
        self._fn, self._dur, self._clock, self._on = fn, dur, clock, on_dispatch

    def remote(self, *a, **k):
        if self._on is not None:
            self._on(*a)
        return _Future(self._fn, self._dur(*a) if callable(self._dur) else self._dur,
                       self._clock, *a, **k)


class _Actor:
    # order[0] is the plan's duration in simulated seconds, order[1] its slot; a slot is
    # recorded at dispatch, so a killed slot names the actor it was sent to.
    def __init__(self, name, clock):
        self.name = name
        self.killed = False
        self.slots: list = []
        self.evaluate = _Remote(self._evaluate, lambda order, *a: float(order[0]), clock,
                                on_dispatch=lambda order, *a: self.slots.append(int(order[1])))
        self.pop_oom_flag = _Remote(lambda: False, 0.0, clock)
        self.ready = _Remote(lambda: True, 0.0, clock)

    def _evaluate(self, order, specs, step, **kw):
        reward = np.full((NUM_REWARDS,), -100.0, np.float32)
        return (np.full((TOK,), 7, np.int32), np.full((TOK,), 3, np.int32), reward)


def _pool(actors, timeout_s):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool
    return CpuApproxPool(
        actors, timeout_s=timeout_s, respawn_factory=None, max_tokens=TOK,
        num_rewards=NUM_REWARDS, cosine_sim_idx=COS, frob_residual_idx=FROB)


def _batch(pool, durations):
    n = len(durations)
    orders = [np.array([d, i, 2], np.float32) for i, d in enumerate(durations)]
    return pool.evaluate_batch(orders, [np.full((3, 1, 3), -1, np.int32)] * n, [3] * n)


def test_a_free_actor_takes_the_next_slot_instead_of_waiting_for_its_wave(sim_ray):
    _ray, clock = sim_ray
    actors = [_Actor("a0", clock), _Actor("a1", clock)]
    pool = _pool(actors, timeout_s=300.0)
    _tok, _eqn, rewards, mask = _batch(pool, [10, 1, 1, 1, 1, 1])
    assert mask.tolist() == [False] * 6
    assert (rewards[:, 0] == -100.0).all()
    assert actors[0].slots == [0]
    assert actors[1].slots == [1, 2, 3, 4, 5]
    assert clock.t == 10.0, "the batch takes as long as its longest plan, not 12"
    assert pool.live_actors() == actors


def test_a_deadline_kill_takes_only_the_killed_slot(sim_ray):
    _ray, clock = sim_ray
    t = 0.02
    actors = [_Actor("a0", clock), _Actor("a1", clock)]
    pool = _pool(actors, timeout_s=t)
    t0 = time.time()
    _tok, _eqn, rewards, mask = _batch(pool, [100 * t, 0.0, 0.0, 0.0])
    assert time.time() - t0 < 5.0
    assert mask.tolist() == [True, False, False, False]
    assert actors[0].killed and not actors[1].killed
    assert actors[0].slots == [0]
    assert actors[1].slots == [1, 2, 3]
    assert pool.stats()["timeouts"] == 1
    assert pool.live_actors() == [actors[1]]
