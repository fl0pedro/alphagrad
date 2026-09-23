"""dsnn-dfw.120 -- an OOM'd slot is not retried on a fresh actor.

The actor whose measure OOMs is recycled once; its slot stays refused. The
retry on the fresh actor and its switch ALPHAGRAD_RECYCLE_RETRY_ON_OOM are
gone: 40 of 47 retries failed again (jobs 67489, 67490, 67491).
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

NUM_REWARDS = 8
FROB = 7
MAX_TOKENS = 8


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
    def __init__(self, oom: bool):
        self.oom = oom
        self.calls = 0
        self.killed = False
        self._flag = False
        self.evaluate = _Remote(self._evaluate)
        self.pop_oom_flag = _Remote(self._pop)

    def _evaluate(self, order, specs, step, **kw):
        self.calls += 1
        tokens = np.full((MAX_TOKENS,), 7, np.int32)
        eqn = np.full((MAX_TOKENS,), 3, np.int32)
        reward = np.full((NUM_REWARDS,), -100.0, np.float32)
        if self.oom:
            reward[:] = -1e10
            reward[6] = 0.0
            self._flag = True
        return tokens, eqn, reward

    def _pop(self):
        was, self._flag = self._flag, False
        return was


def _run(n_slots=2):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    oom_actor, ok_actor = _Actor(oom=True), _Actor(oom=False)
    fresh = []

    def factory():
        fresh.append(_Actor(oom=False))
        return fresh[-1]

    pool = CpuApproxPool(
        [oom_actor, ok_actor], timeout_s=0.0, respawn_factory=factory,
        max_tokens=MAX_TOKENS, num_rewards=NUM_REWARDS,
        cosine_sim_idx=6, frob_residual_idx=FROB)
    orders = [np.arange(3, dtype=np.int32)] * n_slots
    specs = [np.zeros((3, 1, 3), np.int32)] * n_slots
    out = pool.evaluate_batch(orders, specs, [3] * n_slots)
    return pool, oom_actor, ok_actor, fresh, out


@pytest.mark.parametrize("legacy_switch", [None, "1"])
def test_an_oom_slot_is_recycled_and_not_retried(
        fake_ray, monkeypatch, legacy_switch):
    if legacy_switch is None:
        monkeypatch.delenv("ALPHAGRAD_RECYCLE_RETRY_ON_OOM", raising=False)
    else:
        monkeypatch.setenv("ALPHAGRAD_RECYCLE_RETRY_ON_OOM", legacy_switch)
    pool, oom_actor, ok_actor, fresh, out = _run()
    tokens, eqn_ids, rewards, mask = out
    assert oom_actor.killed
    assert len(fresh) == 1, "the OOM'd actor must be recycled exactly once"
    assert fresh[0].calls == 0, "the OOM'd slot was retried on the fresh actor"
    assert mask.tolist() == [True, False]
    assert rewards[0, FROB] <= -1e9
    assert rewards[1, FROB] == -100.0
    st = pool.stats()
    assert st["oom_recycles"] == 1
    assert "oom_retries" not in st and "oom_retry_success" not in st
    assert fresh[0] in pool.live_actors()
    assert not ok_actor.killed


def test_the_switch_is_gone_from_the_pool_source():
    import inspect

    from alphagrad.approx import cpu_approx_pool

    assert "RECYCLE_RETRY_ON_OOM" not in inspect.getsource(cpu_approx_pool)
