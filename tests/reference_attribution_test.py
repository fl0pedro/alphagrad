# dsnn-0rh, owner ruling 2026-09-24 Q55: a candidate error is its scored
# refusal; a reference error raises ReferenceFault (dsnn-w85, -63j, -goy).
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
SAMPLE0 = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
ONE_POINT = (SAMPLE0[None],)
TWO_POINTS = (np.stack([SAMPLE0, SAMPLE0[::-1]]),)
_W = jnp.asarray(np.random.default_rng(0).standard_normal(
    (16, 16), dtype=np.float32) / 4.0)


class XlaRuntimeError(RuntimeError):
    pass


ERRORS = {
    "oom": lambda: XlaRuntimeError(_OOM),
    "error": lambda: ValueError("injected error"),
}


def _toy(v):
    return jnp.sum(jnp.tanh(_W @ v) ** 2)


def _normal_draw(keys):
    return (jax.random.normal(keys[0], (16,)),)


def _zero_draw(keys):
    return (jnp.zeros((16,), jnp.float32),)


def _toy_env(draw=_normal_draw, measure_latency=True):
    from alphagrad.approx.env import VertexEliminationEnv

    x = jnp.asarray(SAMPLE0)
    return VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(_toy)(x), args=[x], argnums=(0,), num_envs=0,
        target_fun=_toy, data_gen=draw, scalar_target=True,
        measure_latency=measure_latency, terminal_rewards_only=True,
        latency_inner_reps=1)


# ---------------------------------------------------------- the injection
def _site(frame):
    # The quality channels call the candidate through dense_measured_program's wrapper (ALPHAGRAD_MEASURE_SPARSE=1, the default): the site is its caller.
    while frame.f_code.co_qualname == "dense_measured_program.<locals>.run":
        frame = frame.f_back
    return frame.f_code.co_name


class _Wrapped:
    def __init__(self, inner, fail):
        self._inner, self._fail = inner, fail

    def __call__(self, *a, **kw):
        exc = self._fail(a, _site(sys._getframe(1)))
        if exc is not None:
            raise exc
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _wrap(fail):
    return lambda make: _Wrapped(make(), fail)


def _once_from(site, exc):
    # Fails the first call made from the function named `site`, and no other.
    fired = []

    def fail(args, caller):
        if caller == site and not fired:
            fired.append(1)
            return exc
        return None
    return fail


def _raise(exc):
    raise exc


def _compile_failure(make):
    try:
        raise XlaRuntimeError(_TRITON)
    except XlaRuntimeError as _e:
        raise env_mod.MeasureCompileFailure(_TRITON) from _e


def _fake_compile(monkeypatch, approx=None, ref=None, rev=None):
    def fake(key, fn, *a, **kw):
        head = bytes(key).split(b":", 1)[0]
        if head == b"approx":
            # Built anew: a memo hit would skip the candidate's build.
            return fn() if approx is None else approx(fn)
        hook = {b"paired-ref": ref, b"rev-exact": rev}.get(head)
        make = (lambda: _REAL_CACHED_COMPILE(key, fn, *a, **kw))
        return make() if hook is None else hook(make)

    monkeypatch.setattr(_cc, "cached_compile", fake)


def _reset():
    _cc._LOCAL_CACHE.clear()
    env_mod._COSINE_REF.clear()
    env_mod._PROBE_BATCH.clear()
    env_mod._PROBE_META.clear()
    env_mod._REV_TELEMETRY["episode"] = None
    env_mod._REV_TELEMETRY["keys"] = set()
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()


@pytest.fixture
def measured(monkeypatch):
    for name, value in {
            "ALPHAGRAD_COST_FORM": "paired-log",
            "ALPHAGRAD_QUALITY_METRIC": "none",
            "ALPHAGRAD_DIRECT_MEASURE": "1",
            "ALPHAGRAD_PLAN_LOG": "1",
            "ALPHAGRAD_MEM_CHANNEL": "temp",
            "ALPHAGRAD_MEASURE_DEDUPE": "0",
            "ALPHAGRAD_SKIP_COUNT_OPS": "1",
            "ALPHAGRAD_DISABLE_JIT_DISK_CACHE": "1"}.items():
        monkeypatch.setenv(name, value)
    for name in ("ALPHAGRAD_REV_EXACT_TELEMETRY", "ALPHAGRAD_GRAD_COSINE_K"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    env_mod.set_measure_timeout_s(DEADLINE)
    _reset()
    yield
    env_mod.register_measure_oom_consumer(False)
    env_mod.set_measure_timeout_s(None)
    _reset()


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


def _in_process(env, samples=ONE_POINT):
    order, specs, faces, skips = _plan(env)
    out = env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order),
        jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips),
        len(order), *[jnp.asarray(s) for s in samples])
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
    def __init__(self, server):
        self.killed = False
        self.evaluate = _Remote(
            lambda *a, rule=None, **k: server.evaluate(*a, **k))
        self.pop_oom_flag = _Remote(server.pop_oom_flag)
        self.ready = _Remote(lambda: True)
        # The records stay in this process, where the checks read them.
        self.consume_call_telemetry = _Remote(lambda: {})


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


def _pooled(pool, env, samples=ONE_POINT):
    order, specs, _f, _s = _plan(env)
    out = pool.evaluate_batch([order], [specs], [len(order)],
                              eval_samples=samples)
    return np.asarray(out[-2][0], dtype=np.float32), out[-1]


# ------------------------------------------------------------- the checks
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


def _check_reference_fault(fault, cause, site):
    assert isinstance(fault.__cause__, cause), repr(fault.__cause__)
    assert site in str(fault), str(fault)
    assert env_mod.consume_refused_counts().get("scored", 0) == 0


# ------------------------------------ 1. dsnn-w85: the gradient cosine
COSINE_CANDIDATE = {
    "oom": (ERRORS["oom"], "oom:measurement"),
    "error": (ERRORS["error"], "raised:ValueError"),
}
COSINE_SITE = "probe batch 0 of the gradient cosine"


@pytest.mark.parametrize("case", sorted(COSINE_CANDIDATE))
def test_a_candidate_error_in_the_cosine_is_scored(measured, monkeypatch,
                                                   case):
    make_exc, reason = COSINE_CANDIDATE[case]
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    env = _toy_env()
    _fake_compile(monkeypatch, approx=_wrap(
        _once_from("_grad_cosine_quality", make_exc())))
    reward = _in_process(env)
    _check_scored(reward, reason, "measurement")
    assert env_mod.pop_measure_oom()[0] == (1 if case == "oom" else 0)


def test_a_candidate_oom_in_the_cosine_recycles_the_actor(
        measured, monkeypatch, fake_ray):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    env = _toy_env()
    _fake_compile(monkeypatch, approx=_wrap(
        _once_from("_grad_cosine_quality", ERRORS["oom"]())))
    pool, actor = _pool(env)
    reward, mask = _pooled(pool, env)
    assert mask.tolist() == [False]
    _check_scored(reward, "oom:measurement", "measurement")
    assert actor.killed


@pytest.mark.parametrize("error", sorted(ERRORS))
def test_a_reference_error_in_the_cosine_raises(measured, monkeypatch,
                                                error):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    exc = ERRORS[error]()
    env = _toy_env()
    _fake_compile(monkeypatch, ref=_wrap(_once_from("_cosine_reference", exc)))
    with pytest.raises(env_mod.ReferenceFault) as ei:
        _in_process(env)
    _check_reference_fault(ei.value, type(exc), COSINE_SITE)


def test_a_reference_draw_error_in_the_cosine_raises(measured, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")

    def draw(keys):
        return _normal_draw(keys)

    def broken(keys):
        raise ValueError("injected reference draw error")
    draw.reference_draw = broken
    env = _toy_env(draw)
    _fake_compile(monkeypatch)
    with pytest.raises(env_mod.ReferenceFault) as ei:
        _in_process(env)
    _check_reference_fault(ei.value, ValueError, COSINE_SITE)


def test_only_a_zero_reference_gradient_leaves_the_quality_undefined(
        measured, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "grad_cosine")
    env = _toy_env(_zero_draw)
    _fake_compile(monkeypatch)
    reward = _in_process(env)
    assert env_mod.consume_refused_counts() == {
        "quality-undefined": 1, "total": 1, "excluded": 1}
    rec = env_mod.consume_plan_records()["records"][-1]
    assert rec["refused"] == "quality-undefined:grad_cosine"
    assert _is_hard_sentinel(reward)


# --------------------------------- 2. dsnn-63j: the timing loop
# (the function the reference fails once from, samples, latency, the site)
# Every execution is timed (owner ruling 2026-09-25): with the latency on the
# first `_time_one_rep` call is the reference's first execution, with it off
# there is none and the first call is an interleaved window.
TIMING = {
    "first": ("_time_one_rep", ONE_POINT, True, "in its first execution"),
    "window": ("_time_one_rep", ONE_POINT, False,
               "in its interleaved windows"),
    "window-two-points": ("_time_one_rep", TWO_POINTS, False,
                          "in its interleaved windows"),
}


@pytest.mark.parametrize("error", sorted(ERRORS))
@pytest.mark.parametrize("site", sorted(TIMING))
def test_a_reference_error_in_the_timing_loop_raises(measured, monkeypatch,
                                                     site, error):
    caller, samples, latency, text = TIMING[site]
    exc = ERRORS[error]()
    env = _toy_env(measure_latency=latency)
    _fake_compile(monkeypatch, ref=_wrap(_once_from(caller, exc)))
    with pytest.raises(env_mod.ReferenceFault) as ei:
        _in_process(env, samples)
    _check_reference_fault(ei.value, type(exc), text)


def _telemetry_reference_window(exc):
    # The reference's first window after the rev-exact executable has run.
    state = {"rev": False, "fired": False}

    def rev_seen(args, caller):
        state["rev"] = True
        return None

    def ref_fails(args, caller):
        if state["rev"] and caller == "_time_one_rep" and not state["fired"]:
            state["fired"] = True
            return exc
        return None
    return {"rev": _wrap(rev_seen), "ref": _wrap(ref_fails)}


TELEMETRY = {
    "compile": lambda exc: {"rev": lambda make: _raise(exc)},
    "rev-exact-window": lambda exc: {
        "rev": _wrap(_once_from("_time_one_rep", exc))},
    "reference-window": _telemetry_reference_window,
}


@pytest.mark.parametrize("error", sorted(ERRORS))
@pytest.mark.parametrize("case", sorted(TELEMETRY))
def test_a_rev_exact_telemetry_error_raises(measured, monkeypatch, case,
                                            error):
    monkeypatch.setenv("ALPHAGRAD_REV_EXACT_TELEMETRY", "1")
    exc = ERRORS[error]()
    env = _toy_env()
    _fake_compile(monkeypatch, **TELEMETRY[case](exc))
    with pytest.raises(env_mod.ReferenceFault) as ei:
        _in_process(env)
    _check_reference_fault(ei.value, type(exc), "the rev-exact telemetry")


def test_a_reference_oom_in_the_timing_loop_raises_in_the_pool(
        measured, monkeypatch, fake_ray):
    env = _toy_env()
    _fake_compile(monkeypatch, ref=_wrap(
        _once_from("_time_one_rep", ERRORS["oom"]())))
    pool, actor = _pool(env)
    with pytest.raises(env_mod.ReferenceFault):
        _pooled(pool, env)
    assert not actor.killed
    assert env_mod.consume_refused_counts().get("scored", 0) == 0


def test_a_candidate_error_in_a_timing_window_is_still_scored(measured,
                                                              monkeypatch):
    env = _toy_env()
    _fake_compile(monkeypatch, approx=_wrap(
        _once_from("_time_one_rep", ERRORS["error"]())))
    reward = _in_process(env)
    _check_scored(reward, "raised:ValueError", "measurement")


# ----------------- 3. dsnn-goy: a refused plan on a carry variant
T_PIN = 7


def _rtrl_env():
    import alphagrad.approx.tools.landscape_map as lm

    argv = ["--example", "RSNN_SHD", "--dataset", "none",
            "--temporal-rule", "rtrl", "--step-position", str(T_PIN),
            "--num-eval-samples", "1", "--num-data-points", "1",
            "--reps-per-point", "1", "--latency-inner-reps", "1",
            "--out-dir", "/tmp/reference_attribution_test"]
    env, eval_samples, _cj = lm.build_env(lm.make_argparser().parse_args(argv))
    return lm, env, eval_samples


def _carry_first_order(env):
    from alphagrad.approx.common import carry_plan as CP

    valid = sorted(int(v) for v in env.valid_vertices)
    mask = CP.carry_scope_mask(env.config.jaxpr)
    return ([v for v in valid if mask[v - 1]]
            + sorted((v for v in valid if not mask[v - 1]), reverse=True))


def _diag_on_the_carry(lm, env, order):
    from alphagrad.approx.common import carry_plan as CP

    jx = env.config.jaxpr
    mask = CP.carry_scope_mask(jx)
    picked = [e for e in lm.face_inventory(env, np.asarray(order, np.int32))
              if mask[int(e["vertex"]) - 1]
              and jx.eqns[int(e["vertex"]) - 1].primitive.name
              == "dot_general"]
    assert picked, "the dense carry block contracts with a dot_general"
    return [{"k": int(e["k"]), "f": int(e["f"]), "slot": 0,
             "row": [0, 0, -1], "kind": "X"} for e in picked]


def test_a_refused_plan_on_a_carry_variant_records_the_policys_order(
        measured, monkeypatch):
    from alphagrad.approx.common import carry_plan as CP

    monkeypatch.setattr(env_mod, "MAX_FACES", env_mod.MAX_FACES)
    try:
        lm, env, eval_samples = _rtrl_env()
        order = _carry_first_order(env)
        plan = {"specs": None, "face_specs": None, "face_skips": None,
                "wires": _diag_on_the_carry(lm, env, order)}
        specs, faces, skips = lm.get_plan_arrays(plan, len(order))
        assert CP.container_for_plan(env.config, order, faces, skips,
                                     specs) == "diag"
        moved = CP.transport_order(order, CP.measurement_env("diag",
                                                             env.config))
        assert moved != order, (
            "the variant numbers the order as the policy does, so this test "
            "cannot see the defect")
        _fake_compile(monkeypatch, approx=_compile_failure)
        env_mod._callback(
            env.config, env.args, env.consts, jnp.asarray(order),
            jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips),
            len(order), *eval_samples)
        rec = env_mod.consume_plan_records()["records"][-1]
    finally:
        CP.reset()
    assert rec["refused"] == "compile:XlaRuntimeError"
    assert rec["carry_container"] == "diag"
    assert rec["order"] == order, (
        f"the record carries the variant's order {rec['order'][:8]}..., "
        f"not the policy's {order[:8]}...")
