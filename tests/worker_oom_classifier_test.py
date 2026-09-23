"""dsnn-dfw.127 -- the measure actor classifies an OOM the same way env does.

The actor's except branch in ``CpuApproximationServer.evaluate`` sets the
one-shot flag the batched pool reads (``pop_oom_flag``) to recycle the actor.
It used to set the flag for any RESOURCE_EXHAUSTED text and for any exception
type named XlaRuntimeError. A fusion over the SM's shared-memory budget and a
bare XlaRuntimeError therefore recycled the actor as if the device were full.
Pinned here: shared-memory limit False, bare XlaRuntimeError False, device
OOM True, and a cache clear only for the device OOM.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

from types import SimpleNamespace                               # noqa: E402

import jax                                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402
from alphagrad.approx.cpu_approx_worker import (                # noqa: E402
    CpuApproximationServer,
)

_SHMEM_TEXT = ("RESOURCE_EXHAUSTED: Shared memory size limit exceeded: "
               "requested 131072, available: 101376")
_BARE_TEXT = "INTERNAL: Failed to launch CUDA kernel: fusion_17"
_OOM_TEXT = ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
             "2306867200 bytes.")


class XlaRuntimeError(RuntimeError):
    pass


@pytest.fixture
def clears(monkeypatch):
    seen = []
    monkeypatch.setattr(jax, "clear_caches", lambda: seen.append(1))
    return seen


@pytest.fixture
def actor():
    a = object.__new__(CpuApproximationServer)
    a._config = SimpleNamespace(delta_obs=True)
    a._args = ()
    a._consts = ()
    a._eval_samples = ()
    a._leak_profile = None
    a._n_calls = 0
    a._n_oom = 0
    a._last_was_oom = False
    a._cache_clear_every = 0
    return a


def _raise_through_evaluate(actor, monkeypatch, exc):
    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(env_mod, "_callback", _boom)
    n = 3
    specs = np.full((n, env_mod.MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    out = actor.evaluate(np.arange(1, n + 1, dtype=np.int32), specs, n)
    assert float(np.asarray(out[-1]).min()) <= -1e9
    return actor.pop_oom_flag()


@pytest.mark.parametrize("text, is_oom, n_clears", [
    (_SHMEM_TEXT, False, 0),
    (_BARE_TEXT, False, 0),
    (_OOM_TEXT, True, 1),
], ids=["shared-memory", "bare-xla-runtime-error", "device-oom"])
def test_the_worker_flag_follows_env_is_oom(
        actor, clears, monkeypatch, text, is_oom, n_clears):
    assert _raise_through_evaluate(
        actor, monkeypatch, XlaRuntimeError(text)) is is_oom
    assert actor._n_oom == int(is_oom)
    assert len(clears) == n_clears
