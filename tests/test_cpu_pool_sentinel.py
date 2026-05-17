"""Smoke tests for ``CpuApproxPool`` sentinel + cold-cache budget paths.

Uses a tiny synchronous fake Ray API so we can exercise the timeout /
sentinel / cold-vs-warm logic without spinning up a real cluster.
"""

from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest


@pytest.fixture
def _ray_fake(monkeypatch):
    """Synchronous Ray substitute. Calls are immediate; futures are
    just thin wrappers around the call result (or a configured delay).
    """

    class _Future:
        def __init__(self, fn, *args, delay=0.0):
            self._fn = fn
            self._args = args
            self._delay = float(delay)
            self._called = False

        def _resolve(self):
            if not self._called:
                if self._delay > 0:
                    time.sleep(self._delay)
                self._value = self._fn(*self._args)
                self._called = True

    class GetTimeoutError(Exception):
        pass

    class RayActorError(Exception):
        pass

    def _get(fut, timeout=None):
        # If we have a configured delay greater than the timeout,
        # raise; otherwise resolve.
        if isinstance(fut, _Future):
            if timeout is not None and fut._delay > float(timeout):
                raise GetTimeoutError(f"delay {fut._delay} > timeout {timeout}")
            fut._resolve()
            return fut._value
        return fut

    def _wait(futures, num_returns=None, timeout=None):
        # Synchronous fakes — return all "ready" if they fit within
        # the timeout.
        ready = []
        not_ready = []
        for f in futures:
            if isinstance(f, _Future) and timeout is not None and f._delay > float(timeout):
                not_ready.append(f)
            else:
                if isinstance(f, _Future):
                    f._resolve()
                ready.append(f)
        return ready, not_ready

    def _cancel(fut, force=False):
        pass

    def _kill(actor, no_restart=False):
        actor._killed = True

    fake_ray = types.SimpleNamespace(
        get=_get,
        wait=_wait,
        cancel=_cancel,
        kill=_kill,
        exceptions=types.SimpleNamespace(
            GetTimeoutError=GetTimeoutError,
            RayActorError=RayActorError,
        ),
    )
    # The module under test does `from ray.exceptions import ...` and
    # `import ray` — both forms need to resolve.
    fake_exc = types.SimpleNamespace(
        GetTimeoutError=GetTimeoutError,
        RayActorError=RayActorError,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", fake_exc)
    yield fake_ray, _Future


class _FakeActor:
    """Tiny actor with `.evaluate.remote(...)` returning sentinel-shaped data."""

    def __init__(self, fake_ray, _Future, max_tokens=8, num_rewards=8, delay=0.0):
        self._killed = False
        self._delay = delay
        self._max_tokens = max_tokens
        self._num_rewards = num_rewards
        # Build a tiny "evaluate.remote" surface that wraps a future.

        def _ev(order, specs, step, eval_samples=None, init=False):
            tokens = np.full((max_tokens,), 7, dtype=np.int32)
            eqn_ids = np.full((max_tokens,), 3, dtype=np.int32)
            reward = np.full((num_rewards,), -100.0, dtype=np.float32)
            return tokens, eqn_ids, reward

        class _EvalProxy:
            def __init__(self, fn, delay):
                self._fn = fn
                self._delay = delay

            def remote(self, *a, **k):
                return _Future(self._fn, *a, delay=self._delay)

        self.evaluate = _EvalProxy(_ev, self._delay)


def test_pool_returns_value_when_under_timeout(_ray_fake):
    fake_ray, _Future = _ray_fake
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    actors = [
        _FakeActor(fake_ray, _Future, delay=0.0) for _ in range(2)
    ]
    pool = CpuApproxPool(
        actors,
        timeout_s=1.0,
        respawn_factory=None,
        max_tokens=8, num_rewards=8,
        cosine_sim_idx=6, frob_residual_idx=7,
    )
    out = pool.evaluate(
        np.zeros(3, np.int32), np.zeros((1, 3), np.int32), 0, None,
    )
    tokens, eqn_ids, reward = out
    assert (tokens == 7).all()
    assert reward[0] == -100.0
    assert pool.stats()["timeouts"] == 0


def test_pool_returns_sentinel_on_timeout(_ray_fake):
    fake_ray, _Future = _ray_fake
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool, _SENTINEL_REWARD_VALUE

    # delay > timeout => GetTimeoutError => poison + sentinel.
    actors = [
        _FakeActor(fake_ray, _Future, delay=0.5) for _ in range(2)
    ]
    pool = CpuApproxPool(
        actors,
        timeout_s=0.001,
        respawn_factory=None,
        max_tokens=8, num_rewards=8,
        cosine_sim_idx=6, frob_residual_idx=7,
    )
    out = pool.evaluate(
        np.zeros(3, np.int32), np.zeros((1, 3), np.int32), 0, None,
    )
    tokens, eqn_ids, reward = out
    # Tokens are sentinel-shaped zeros (NOT the actor's 7s).
    assert (tokens == 0).all()
    # Cost channels are -1e10; cosine_sim is 0.
    assert reward[0] == _SENTINEL_REWARD_VALUE
    assert reward[6] == 0.0  # cosine_sim
    assert pool.stats()["timeouts"] == 1


def test_cold_warm_timeout_budget(_ray_fake):
    """First ``warm_after`` calls per actor get ``initial_timeout_s``;
    after that, the regular ``timeout_s`` applies."""
    fake_ray, _Future = _ray_fake
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    # Use a delay that fits within initial_timeout_s (0.05s) but NOT
    # within the warm timeout (0.001s). Each actor should succeed on
    # the first warm_after calls and start timing out after that.
    actors = [_FakeActor(fake_ray, _Future, delay=0.05)]
    pool = CpuApproxPool(
        actors,
        timeout_s=0.001,
        initial_timeout_s=0.5,
        warm_after=2,
        respawn_factory=None,
        max_tokens=8, num_rewards=8,
        cosine_sim_idx=6, frob_residual_idx=7,
    )

    # First call: cold, succeeds.
    out1 = pool.evaluate(
        np.zeros(3, np.int32), np.zeros((1, 3), np.int32), 0, None,
    )
    assert pool.stats()["timeouts"] == 0
    assert out1[2][0] == -100.0  # success reward

    # Second call: still cold (warm_after=2), succeeds.
    out2 = pool.evaluate(
        np.zeros(3, np.int32), np.zeros((1, 3), np.int32), 0, None,
    )
    assert pool.stats()["timeouts"] == 0

    # Third call: warm now — 0.001s timeout < 0.05s delay → sentinel.
    # The actor is poisoned so the pool is then empty.
    out3 = pool.evaluate(
        np.zeros(3, np.int32), np.zeros((1, 3), np.int32), 0, None,
    )
    assert pool.stats()["timeouts"] == 1
    from alphagrad.approx.cpu_approx_pool import _SENTINEL_REWARD_VALUE
    assert out3[2][0] == _SENTINEL_REWARD_VALUE


def test_evaluate_batch_emits_sentinel_mask(_ray_fake):
    fake_ray, _Future = _ray_fake
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    # One actor with a long delay (will time out), one without.
    actors = [
        _FakeActor(fake_ray, _Future, delay=0.5),
        _FakeActor(fake_ray, _Future, delay=0.0),
    ]
    pool = CpuApproxPool(
        actors,
        timeout_s=0.01,
        respawn_factory=None,
        max_tokens=8, num_rewards=8,
        cosine_sim_idx=6, frob_residual_idx=7,
    )
    N = 2
    tokens, eqn_ids, rewards, mask = pool.evaluate_batch(
        [np.zeros(3, np.int32)] * N,
        [np.zeros((1, 3), np.int32)] * N,
        [0] * N,
    )
    assert tokens.shape == (N, 8)
    assert rewards.shape == (N, 8)
    assert mask.shape == (N,)
    # Exactly one slot was sentinel.
    assert mask.sum() == 1
