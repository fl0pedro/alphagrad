# dsnn-cl0, owner ruling 2026-09-24 Q53: only a failure of the plan's own program
# is a refusal. An error while the candidate is built, traced, compiled or run
# is scored and never raises. A failure of the reference raises ReferenceFault,
# the memory check of Q44 raises MemoryBoundFault, and a tokenization error of
# the terminal step raises in-process and is excluded and counted in the pool.
# One test per case, in-process and through the pool with the real server.
from __future__ import annotations

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
DEADLINE = 300.0
LIMIT = 16_000_000_000
_OOM = ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
        "17179869184 bytes.")
_TRITON = ("INTERNAL: Failed to compile Triton kernel. Context: [Fusion: "
           "fusion.35 = f32[32,1024,32]{2,1,0}]")


class XlaRuntimeError(RuntimeError):
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


_SAMPLES = (np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)]),)


class _Analysis:
    def __init__(self, inner, out=0):
        self._inner = inner
        self.temp_size_in_bytes = int(inner.temp_size_in_bytes)
        self.output_size_in_bytes = int(inner.output_size_in_bytes) + int(out)
        self.argument_size_in_bytes = int(inner.argument_size_in_bytes)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Wrapped:
    def __init__(self, inner, *, out=0, fail=None):
        self._inner, self._out, self._fail = inner, out, fail

    def memory_analysis(self):
        return _Analysis(self._inner.memory_analysis(), self._out)

    def __call__(self, *a, **kw):
        if self._fail is not None:
            raise self._fail
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _raise(exc):
    raise exc


def _compile_failure(make):
    try:
        raise XlaRuntimeError(_TRITON)
    except XlaRuntimeError as _e:
        raise env_mod.MeasureCompileFailure(_TRITON) from _e


def _patch_jacve_build(monkeypatch):
    def broken(*a, **kw):
        raise ValueError("injected build error")
    monkeypatch.setattr(env_mod, "jacve", broken)


def _patch_trace(monkeypatch):
    import graphax.core as gcore

    def broken(*a, **kw):
        raise ValueError("injected trace error")
    monkeypatch.setattr(gcore, "vertex_elimination_jaxpr", broken)


def _patch_count_pass(exc):
    def patch(monkeypatch):
        monkeypatch.setenv("ALPHAGRAD_SKIP_COUNT_OPS", "0")

        def broken(*a, **kw):
            raise exc
        monkeypatch.setattr(env_mod, "vertex_elimination_jaxpr", broken)
    return patch


def _patch_carry(monkeypatch):
    def broken(*a, **kw):
        raise ValueError("injected carry container error")
    monkeypatch.setattr(env_mod._carry, "armed", lambda config=None: True)
    monkeypatch.setattr(env_mod._carry, "container_for_plan", broken)


# (patch of the process, fake of the candidate's compile, reason, where)
CANDIDATE_ERRORS = {
    "build": (_patch_jacve_build, None, "raised:ValueError",
              "measurement"),
    "trace": (_patch_trace, None, "untraceable:approx compile",
              "approx compile"),
    "compile": (None, _compile_failure, "compile:XlaRuntimeError",
                "approx compile"),
    "compile-oom": (None, lambda make: _raise(XlaRuntimeError(_OOM)),
                    "oom:approx compile", "approx compile"),
    "execution": (None, lambda make: _Wrapped(
        make(), fail=ValueError("injected execution error")),
        "raised:ValueError", "measurement"),
    "execution-oom": (None, lambda make: _Wrapped(
        make(), fail=XlaRuntimeError(_OOM)),
        "oom:measurement", "measurement"),
    "count-pass": (_patch_count_pass(ValueError("injected count error")),
                   None, "raised:ValueError", "count pass"),
    "count-pass-oom": (_patch_count_pass(XlaRuntimeError(_OOM)), None,
                       "oom:count pass", "count pass"),
    "carry-container": (_patch_carry, None, "raised:ValueError",
                        "carry container"),
}


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
    env_mod.set_measure_timeout_s(DEADLINE)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()
    yield
    env_mod.register_measure_oom_consumer(False)
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()


def _fake_compile(monkeypatch, approx=None, ref=None):
    real = _REAL_CACHED_COMPILE

    def fake(key, fn, *a, **kw):
        head = bytes(key).split(b":", 1)[0]
        if head == b"approx":
            # Built and traced anew on every call: an executable from the
            # process memo would skip the candidate's build and trace.
            return fn() if approx is None else approx(fn)
        make = (lambda: real(key, fn, *a, **kw))
        if ref is not None and head == b"paired-ref":
            return ref(make)
        return make()

    monkeypatch.setattr(_cc, "cached_compile", fake)


def _plan(env):
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    n = len(order)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    return np.asarray(order, np.int32), specs, faces, skips


def _in_process(env):
    order, specs, faces, skips = _plan(env)
    out = env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order),
        jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips),
        len(order), *[jnp.asarray(s) for s in _SAMPLES])
    return np.asarray(out[-1], dtype=np.float32)


# --------------------------------------------------------------- the pool
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

    def _kill(actor, no_restart=False):
        actor.killed = True

    exc = types.SimpleNamespace(GetTimeoutError=GetTimeoutError,
                                RayActorError=RayActorError)
    ray = types.SimpleNamespace(
        get=_get, wait=lambda f, num_returns=None, timeout=None: (list(f), []),
        kill=_kill, cancel=lambda *a, **k: None, exceptions=exc)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return ray


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return _Future(self._fn, *a, **k)


class _ServerActor:
    # The Ray wrapper's surface around the real server of this process.
    def __init__(self, server):
        self.killed = False
        self.evaluate = _Remote(
            lambda *a, rule=None, **k: server.evaluate(*a, **k))
        self.pop_oom_flag = _Remote(server.pop_oom_flag)
        self.ready = _Remote(lambda: True)


def _pool(env):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

    actor = _ServerActor(CpuApproximationServer.from_env(env))
    return CpuApproxPool(
        [actor], timeout_s=DEADLINE, respawn_factory=None,
        max_tokens=int(env.obs_width), token_dtype=env.wire_token_dtype,
        emit_eqn_ids=not env.config.delta_obs, num_rewards=NUM_REWARDS,
        cosine_sim_idx=int(REWARD_INDEX["cosine_sim"]),
        frob_residual_idx=int(REWARD_INDEX["frob_residual"]),
        fidelity_idx=int(REWARD_INDEX["fidelity"]),
        sparsity_idx=int(REWARD_INDEX["sparsity"])), actor


def _pooled(env, single=False):
    pool, actor = _pool(env)
    order, specs, _f, _s = _plan(env)
    if single:
        out = pool.evaluate(order, specs, len(order), _SAMPLES)
        return np.asarray(out[-1], dtype=np.float32), None, actor
    out = pool.evaluate_batch([order], [specs], [len(order)],
                              eval_samples=_SAMPLES)
    return np.asarray(out[-2][0], dtype=np.float32), out[-1], actor


def _is_hard_sentinel(reward):
    return bool(np.all(np.asarray(reward)[list(env_mod.COMPUTE_REWARD_INDICES)]
                       <= env_mod.SENTINEL_COST * 0.99))


def _check_scored(reward, reason, where):
    kind = reason.split(":", 1)[0]
    counts = env_mod.consume_refused_counts()
    assert counts == {kind: 1, "total": 1, "scored": 1}, counts
    rec = env_mod.consume_plan_records()["records"][-1]
    assert rec["refused"] == reason
    assert rec["refusal_where"] == where
    assert rec["refusal_timeout_s"] == DEADLINE
    assert np.isfinite(reward).all() and not _is_hard_sentinel(reward)
    assert reward[REWARD_INDEX["quality"]] == 0.0


# ------------------------------------------- 1. the candidate never raises
@pytest.mark.parametrize("case", sorted(CANDIDATE_ERRORS))
def test_a_candidate_error_is_scored_in_process(paired, monkeypatch, case):
    patch, approx, reason, where = CANDIDATE_ERRORS[case]
    env = _toy_env()
    if patch is not None:
        patch(monkeypatch)
    _fake_compile(monkeypatch, approx=approx)
    reward = _in_process(env)
    _check_scored(reward, reason, where)
    if reason.startswith("oom:"):
        assert env_mod.pop_measure_oom()[0] == 1


@pytest.mark.parametrize("case", sorted(CANDIDATE_ERRORS))
def test_a_candidate_error_is_scored_in_the_pool(paired, monkeypatch,
                                                 fake_ray, case):
    patch, approx, reason, where = CANDIDATE_ERRORS[case]
    env = _toy_env()
    if patch is not None:
        patch(monkeypatch)
    _fake_compile(monkeypatch, approx=approx)
    reward, mask, actor = _pooled(env)
    assert mask.tolist() == [False]
    _check_scored(reward, reason, where)
    assert actor.killed is reason.startswith("oom:"), (
        "an out-of-memory error recycles the actor, no other error does")


# --------------------------------------------- 2. the reference raises
def _reference_compile_fails(monkeypatch):
    _fake_compile(monkeypatch, ref=lambda make: _raise(
        ValueError("injected reference compile error")))


def _reference_run_fails(monkeypatch):
    # A refused plan (its execution fails) is scored against the reference,
    # and the reference cannot run.
    _fake_compile(
        monkeypatch,
        approx=lambda make: _Wrapped(make(), fail=ValueError("candidate")),
        ref=lambda make: _Wrapped(make(), fail=ValueError(
            "injected reference run error")))


def _memory_bound_violated(monkeypatch):
    _fake_compile(monkeypatch, approx=lambda make: _Wrapped(make(), out=1))


FAULTS = {
    "reference-compile": (_reference_compile_fails,
                          lambda: env_mod.ReferenceFault),
    "reference-run": (_reference_run_fails, lambda: env_mod.ReferenceFault),
    "memory-bound": (_memory_bound_violated,
                     lambda: env_mod.MemoryBoundFault),
}


@pytest.mark.parametrize("case", sorted(FAULTS))
def test_a_fault_of_the_apparatus_raises_in_process(paired, monkeypatch,
                                                    case):
    setup, fault = FAULTS[case]
    env = _toy_env()
    setup(monkeypatch)
    with pytest.raises(fault()):
        _in_process(env)
    assert env_mod.consume_refused_counts().get("scored", 0) == 0


@pytest.mark.parametrize("single", [False, True], ids=["batch", "single"])
@pytest.mark.parametrize("case", sorted(FAULTS))
def test_a_fault_of_the_apparatus_raises_in_the_pool(paired, monkeypatch,
                                                     fake_ray, case, single):
    setup, fault = FAULTS[case]
    env = _toy_env()
    setup(monkeypatch)
    with pytest.raises(fault()):
        _pooled(env, single=single)
    assert env_mod.consume_refused_counts().get("scored", 0) == 0


# ------------------------------------------ 3. the terminal tokenization
def _tokenizer_fails(monkeypatch):
    def broken(*a, **kw):
        raise ValueError("injected tokenization error")
    monkeypatch.setattr(env_mod, "_incremental_stream_tokens", broken)


def test_a_tokenization_error_raises_in_process_and_is_counted(paired,
                                                               monkeypatch):
    env = _toy_env()
    _tokenizer_fails(monkeypatch)
    with pytest.raises(ValueError, match="injected tokenization error"):
        _in_process(env)
    assert env_mod.consume_refused_counts() == {
        "raised": 1, "total": 1, "excluded": 1}


@pytest.mark.parametrize("single", [False, True], ids=["batch", "single"])
def test_a_tokenization_error_is_excluded_and_counted_in_the_pool(
        paired, monkeypatch, fake_ray, single):
    env = _toy_env()
    _tokenizer_fails(monkeypatch)
    reward, mask, actor = _pooled(env, single=single)
    assert _is_hard_sentinel(reward)
    if mask is not None:
        assert mask.tolist() == [False]
    assert not actor.killed
    assert env_mod.consume_refused_counts() == {
        "raised": 1, "total": 1, "excluded": 1}
    rec = env_mod.consume_plan_records()["records"][-1]
    assert rec["refused"] == "raised:ValueError"
