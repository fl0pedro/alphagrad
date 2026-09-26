# dsnn-dkz, owner rulings 2026-09-24 Q42-Q48: one 300 s deadline without a cold
# budget and a readiness wait, every refusal scored in the pool and in-process,
# the (c') memory sentinel with its two checks, eps = 2^-10 x the smallest
# nonzero reference value on slot 11. Since 2026-09-26 (Q9b c, dsnn-dfw.292) a
# call the deadline kills is a failed plan too: a free actor scores it as
# "deadline", and only a kill that no actor can score is excluded.
from __future__ import annotations

import csv
import inspect
import json
import math
import os
import sys
import time
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
    NUM_REWARDS,
    REWARD_INDEX,
    mem_objective,
    paired_log_costs,
)


# Resolved per call, so a tree without it fails the one test that needs it.
def memory_sentinel(reference, bytes_limit):
    return env_mod.memory_sentinel(reference, bytes_limit)


_REAL_CACHED_COMPILE = _cc.cached_compile

DEADLINE = 300.0
LIMIT = 16_000_000_000
SWELL = 20_000_000_000
LAT = int(REWARD_INDEX["latency_ns"])
MEM = int(REWARD_INDEX["peak_memory"])
QUAL = int(REWARD_INDEX["quality"])
MOBJ = int(REWARD_INDEX["mem_objective"])
MULS = int(REWARD_INDEX["muls_adds_fmas"])
IO = int(REWARD_INDEX["max_io_sum"])
TOK = 8
_OOM = ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate "
        "17179869184 bytes.")
_TRITON = ("INTERNAL: Failed to compile Triton kernel. Context: [Fusion: "
           "fusion.35 = f32[32,1024,32]{2,1,0}]")


class XlaRuntimeError(RuntimeError):
    pass


# ------------------------------------------------------------- the toy plans
def _toy_env(two_leaves=False, sparse=True):
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))
    if two_leaves:
        def toy(w, v):
            return jnp.sum(jnp.tanh(w @ v) ** 2)
        args, argnums = [W, x], (0, 1)
    else:
        def toy(v):
            return jnp.sum(jnp.tanh(W @ v) ** 2)
        args, argnums = [x], (0,)
    closed = jax.make_jaxpr(toy)(*args)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=args, argnums=argnums, num_envs=0, target_fun=toy,
        sparse=sparse,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)
    samples = tuple(jnp.asarray(np.stack([np.asarray(a)])) for a in args)
    return env, samples


def _plan_arrays(n):
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    return jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips)


class _Analysis:
    def __init__(self, inner, temp=0, out=0, args=0):
        self._inner = inner
        self.temp_size_in_bytes = int(inner.temp_size_in_bytes) + int(temp)
        self.output_size_in_bytes = int(inner.output_size_in_bytes) + int(out)
        self.argument_size_in_bytes = (int(inner.argument_size_in_bytes)
                                       + int(args))

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Wrapped:
    def __init__(self, inner, *, temp=0, out=0, args=0, fail=None):
        self._inner, self._fail = inner, fail
        self._extra = (temp, out, args)
        self.calls = 0

    def memory_analysis(self):
        return _Analysis(self._inner.memory_analysis(), *self._extra)

    def __call__(self, *a, **kw):
        self.calls += 1
        if self._fail is not None:
            raise self._fail
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _triple(ex):
    ma = ex.memory_analysis()
    return (float(ma.temp_size_in_bytes), float(ma.output_size_in_bytes),
            float(ma.argument_size_in_bytes))


@pytest.fixture
def paired(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    env_mod.set_measure_timeout_s(DEADLINE)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()
    yield
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.pop_measure_oom()


def _run(monkeypatch, *, approx=None, two_leaves=False, reverse=False,
         sparse=True):
    real = _REAL_CACHED_COMPILE
    seen = {}

    def fake(key, fn, *a, **kw):
        if approx is not None and bytes(key).startswith(b"approx:"):
            out = approx(lambda: real(key, fn, *a, **kw))
        else:
            out = real(key, fn, *a, **kw)
        seen[bytes(key).split(b":", 1)[0]] = out
        return out

    monkeypatch.setattr(_cc, "cached_compile", fake)
    env, samples = _toy_env(two_leaves, sparse)
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    if reverse:
        order = order[::-1]
    specs, faces, skips = _plan_arrays(len(order))
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    out = env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order), specs,
        faces, skips, len(order), *samples)
    counts = env_mod.consume_refused_counts()
    recs = env_mod.consume_plan_records()["records"]
    return np.asarray(out[-1], dtype=np.float32), counts, recs, seen


def _is_hard_sentinel(reward):
    return bool(np.all(np.asarray(reward)[list(env_mod.COMPUTE_REWARD_INDICES)]
                       <= env_mod.SENTINEL_COST * 0.99))


def _raise(exc):
    raise exc


# ----------------------------------------- 3. every scored kind, every channel
def _check_scored(reward, rec, counts, *, reason, program):
    kind = reason.split(":", 1)[0]
    assert counts == {kind: 1, "total": 1, "scored": 1}, counts
    assert rec["refused"] == reason
    assert rec["sentinelled"] is True and rec["replayable"] is True
    assert rec["refusal_timeout_s"] == DEADLINE
    assert rec["refusal_latency_ns"] == DEADLINE * 1e9
    assert rec["refusal_bytes_limit"] == LIMIT
    ref = (rec["ref_temp_bytes"], rec["ref_output_bytes"],
           rec["ref_args_bytes"])
    ref_lat = rec["ref_latency_ns"]
    assert ref_lat > 0.0 and ref[1] > 0.0 and ref[2] > 0.0
    if program is None:
        assert rec["refusal_program"] is False
        cand, sent = memory_sentinel(ref, LIMIT)
        assert rec["refusal_sentinel"] == sent
        assert sent["branch"] == "fill"
        assert cand == (LIMIT - ref[2] - ref[1], ref[1], ref[2])
        assert rec["mem_temp_bytes"] is None
    else:
        assert rec["refusal_program"] is True
        assert rec["refusal_sentinel"] is None
        cand = program
        assert (rec["mem_temp_bytes"], rec["mem_output_bytes"],
                rec["mem_args_bytes"]) == cand
    d_lat, d_mem, _ = paired_log_costs(DEADLINE * 1e9, cand[0], ref_lat,
                                       ref[0])
    obj, obj_rec = mem_objective(cand, ref)
    assert rec["rewards"][LAT] == -d_lat
    assert d_lat == pytest.approx(math.log(DEADLINE * 1e9 / ref_lat))
    assert rec["rewards"][MEM] == -d_mem
    assert rec["rewards"][MOBJ] == obj
    assert rec["mem_ratios"] == obj_rec["ratios"]
    assert rec["mem_objective_eps"] == obj_rec["eps"] == 2.0 ** -10 * min(
        x for x in ref if x > 0.0)
    assert reward[LAT] == np.float32(-d_lat)
    assert reward[MEM] == np.float32(-d_mem)
    assert reward[MOBJ] == np.float32(obj)
    assert reward[QUAL] == 0.0
    assert reward[REWARD_INDEX["fidelity"]] == -1.0
    assert reward[REWARD_INDEX["sparsity"]] == -1.0
    assert reward[REWARD_INDEX["grad_coverage"]] == 0.0
    assert reward[REWARD_INDEX["bkstep_acc"]] == 0.0
    assert not _is_hard_sentinel(reward) and np.isfinite(reward).all()
    return ref


def test_a_gate_refusal_is_scored_on_its_real_static_memory(paired,
                                                            monkeypatch):
    reward, counts, recs, seen = _run(
        monkeypatch, approx=lambda make: _Wrapped(make(), temp=SWELL))
    assert seen[b"approx"].calls == 0
    _check_scored(reward, recs[-1], counts, reason="gate",
                  program=_triple(seen[b"approx"]))
    assert reward[MULS] == 0.0 and reward[IO] == 0.0


def test_a_compile_failure_is_scored_at_the_deadline_with_the_sentinel(
        paired, monkeypatch):
    def fail(make):
        try:
            raise XlaRuntimeError(_TRITON)
        except XlaRuntimeError as _e:
            raise env_mod.MeasureCompileFailure(_TRITON) from _e
    reward, counts, recs, _ = _run(monkeypatch, approx=fail)
    _check_scored(reward, recs[-1], counts, reason="compile:XlaRuntimeError",
                  program=None)


def test_an_oom_in_the_compile_is_scored_with_the_sentinel(paired,
                                                           monkeypatch):
    reward, counts, recs, _ = _run(
        monkeypatch, approx=lambda make: _raise(XlaRuntimeError(_OOM)))
    _check_scored(reward, recs[-1], counts, reason="oom:approx compile",
                  program=None)
    assert env_mod.pop_measure_oom()[0] == 1


def test_an_oom_in_the_execution_is_scored_on_its_real_static_memory(
        paired, monkeypatch):
    reward, counts, recs, seen = _run(
        monkeypatch, approx=lambda make: _Wrapped(
            make(), fail=XlaRuntimeError(_OOM)))
    assert seen[b"approx"].calls == 1
    _check_scored(reward, recs[-1], counts, reason="oom:measurement",
                  program=_triple(seen[b"approx"]))


def test_an_untraceable_plan_is_scored_with_the_sentinel(paired,
                                                         monkeypatch):
    from graphax import jacve

    def untraceable(make):
        jacve(lambda v: v, [1], jaxpr=object())
    reward, counts, recs, _ = _run(monkeypatch, approx=untraceable)
    _check_scored(reward, recs[-1], counts,
                  reason="untraceable:approx compile", program=None)


def test_an_op_count_refusal_is_scored_with_the_sentinel(paired, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_SKIP_COUNT_OPS", "0")
    monkeypatch.setenv("ALPHAGRAD_MULS_SENTINEL_CAP", "-1")
    reward, counts, recs, seen = _run(monkeypatch)
    assert b"approx" not in seen, "a capped plan must never be compiled"
    rec = recs[-1]
    _check_scored(reward, rec, counts, reason="muls-cap", program=None)
    assert rec["refusal_muls"] > 0.0
    assert rec["rewards"][MULS] == -rec["refusal_muls"]


def test_a_raise_before_the_program_exists_is_scored_with_the_sentinel(
        paired, monkeypatch):
    reward, counts, recs, _ = _run(
        monkeypatch, approx=lambda make: _raise(ValueError("injected")))
    rec = recs[-1]
    _check_scored(reward, rec, counts, reason="raised:ValueError",
                  program=None)
    assert "injected" in rec["refusal_error"]


def test_a_raise_after_the_program_exists_is_scored_on_its_memory(
        paired, monkeypatch):
    def broken_gate(*a, **kw):
        raise KeyError("injected after the compile")
    monkeypatch.setattr(env_mod, "_static_peak_gate", broken_gate)
    reward, counts, recs, seen = _run(monkeypatch)
    _check_scored(reward, recs[-1], counts, reason="raised:KeyError",
                  program=_triple(seen[b"approx"]))


def test_a_call_the_deadline_killed_is_scored_as_a_failed_plan(paired,
                                                               monkeypatch):
    # dsnn-dfw.292: the pool sends the killed plan to a free actor with this forced refusal.
    monkeypatch.setattr(env_mod, "_FORCED_REFUSAL",
                        [env_mod.forced_refusal("deadline", DEADLINE)])
    reward, counts, recs, seen = _run(monkeypatch)
    assert b"approx" not in seen, "a killed plan is not compiled again"
    rec = recs[-1]
    _check_scored(reward, rec, counts, reason="deadline", program=None)
    assert rec["refusal_where"] == "the pool's deadline"


def test_an_apparatus_fault_still_stops_the_measurement(paired, monkeypatch):
    def faulty(make):
        raise env_mod.MeasureToolchainFault("TOOLCHAIN FAULT injected")
    with pytest.raises(env_mod.MeasureToolchainFault):
        _run(monkeypatch, approx=faulty)
    assert env_mod.consume_refused_counts() == {
        "raised": 1, "total": 1, "excluded": 1}


# ------------------------------------------------ 4. the (c') memory sentinel
REF3 = (4096.0, 512.0, 1024.0)


def test_the_sentinel_fills_the_card_when_twice_the_args_fit():
    t, o, a = REF3
    limit = 1_000_000.0
    assert 2 * a + o <= limit
    triple, rec = memory_sentinel(REF3, limit)
    assert rec["branch"] == "fill" and triple == (limit - a - o, o, a)
    eps = 2.0 ** -10 * min(REF3)
    # One total since 2026-09-26 (Q1 a): the sentinel's total is the card.
    want = -math.log((limit + eps) / (t + a + o + eps))
    assert mem_objective(triple, REF3)[0] == pytest.approx(want, abs=1e-12)


def test_the_sentinel_splits_the_card_when_they_do_not():
    t, o, a = REF3
    limit = 2 * a + o - 2.0
    triple, rec = memory_sentinel(REF3, limit)
    half = (limit - o) / 2.0
    assert rec["branch"] == "split" and triple == (half, o, half)
    eps = 2.0 ** -10 * min(REF3)
    want = -math.log((2 * half + o + eps) / (t + a + o + eps))
    assert mem_objective(triple, REF3)[0] == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("limit", [1_000_000.0, 2 * 1024.0 + 512.0 - 2.0])
def test_no_plan_that_passes_the_gate_scores_below_the_sentinel(limit):
    t, o, a = REF3
    sent = mem_objective(memory_sentinel(REF3, limit)[0], REF3)[0]
    rng = np.random.default_rng(0)
    for _ in range(2000):
        pa = float(rng.uniform(0.0, a))
        po = float(rng.uniform(0.0, o))
        pt = float(rng.uniform(0.0, max(limit - pa - po, 0.0)))
        assert mem_objective((pt, po, pa), REF3)[0] >= sent - 1e-9
    worst = min(mem_objective((limit - x - o, o, x), REF3)[0]
                for x in np.linspace(0.0, min(a, limit - o), 4001))
    assert worst == pytest.approx(sent, abs=1e-6)


def test_the_sentinel_raises_when_the_reference_output_exceeds_the_card():
    with pytest.raises(ValueError, match="exceeds the device limit"):
        memory_sentinel(REF3, 100.0)


def test_refused_reward_puts_the_sentinel_on_slot_5_and_slot_11(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    ref = {"latency_ns": 2.0e5, "memory_bytes": REF3[0], "static": REF3,
           "bytes_limit": 1_000_000, "bytes_limit_source": "device",
           "measure_latency": True}
    slots, info = env_mod.refused_reward("compile", timeout_s=DEADLINE,
                                         reference=ref)
    triple, sent = memory_sentinel(REF3, 1_000_000)
    d_lat, d_mem, _ = paired_log_costs(DEADLINE * 1e9, triple[0], 2.0e5,
                                       REF3[0])
    assert slots[LAT] == -d_lat and slots[MEM] == -d_mem
    assert slots[MOBJ] == mem_objective(triple, REF3)[0]
    assert info["refusal_sentinel"] == sent
    assert info["refusal_program"] is False
    # A call the deadline killed is scored like any refusal without a program (Q9b c).
    killed, _info = env_mod.refused_reward("deadline", timeout_s=DEADLINE,
                                           reference=ref)
    assert killed == slots


# ------------------------------------------------------- 4a. O == O* exactly
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("two_leaves", [False, True])
def test_an_exact_plan_returns_the_reference_output_to_the_byte(
        paired, monkeypatch, two_leaves, sparse):
    reward, counts, recs, seen = _run(monkeypatch, two_leaves=two_leaves,
                                      sparse=sparse,
                                      reverse=True)
    assert counts == {}, counts
    rec = recs[-1]
    print(f"[check-a] leaves={2 if two_leaves else 1} sparse={sparse} "
          f"out {rec['mem_output_bytes']:.0f} B vs reference "
          f"{rec['ref_output_bytes']:.0f} B, args {rec['mem_args_bytes']:.0f}"
          f" B vs {rec['ref_args_bytes']:.0f} B")
    assert rec["mem_output_bytes"] == rec["ref_output_bytes"]
    assert rec["mem_args_bytes"] == rec["ref_args_bytes"]
    leaves = jax.tree_util.tree_leaves(
        seen[b"approx"](*[jnp.asarray(a) for a in _toy_env(two_leaves)[0]
                          .args]))
    assert len(leaves) == (2 if two_leaves else 1)


# ------------------------------------------------------------ 4b. the checks
def test_check_b_raises_on_either_violation():
    env_mod.check_memory_bounds(REF3, REF3, "equal")
    with pytest.raises(env_mod.MemoryBoundFault, match="argument bytes"):
        env_mod.check_memory_bounds((1.0, 512.0, 1025.0), REF3, "args")
    with pytest.raises(env_mod.MemoryBoundFault, match="output bytes"):
        env_mod.check_memory_bounds((1.0, 513.0, 1024.0), REF3, "out")
    assert issubclass(env_mod.MemoryBoundFault,
                      env_mod.MeasureToolchainFault)


@pytest.mark.parametrize("extra", [{"out": 1}, {"args": 1}])
def test_a_measured_plan_above_the_reference_stops_the_measurement(
        paired, monkeypatch, extra):
    with pytest.raises(env_mod.MemoryBoundFault):
        _run(monkeypatch, approx=lambda make: _Wrapped(make(), **extra))


def test_a_refused_plan_above_the_reference_stops_the_measurement(
        paired, monkeypatch):
    with pytest.raises(env_mod.MemoryBoundFault):
        _run(monkeypatch, approx=lambda make: _Wrapped(
            make(), temp=SWELL, args=1))


# ------------------------------------------------------------ 5. eps on slot 11
def test_eps_stays_and_a_zero_reference_temp_is_part_of_the_total():
    # Since the one total (owner ruling 2026-09-26, Q1 a) a halved output moves slot 11 by its
    # share of the total, and the logged temp ratio stays finite through eps.
    ref = (0.0, 1024.0, 4096.0)
    value, rec = mem_objective((0.0, 512.0, 4096.0), ref)
    assert rec["eps"] == 2.0 ** -10 * 1024.0
    assert rec["ratios"]["temp"] == 1.0 and rec["ratios"]["args"] == 1.0
    assert value == -math.log((4096.0 + 512.0 + 1.0) / (4096.0 + 1024.0 + 1.0))
    value, rec = mem_objective((64.0, 1024.0, 4096.0), ref)
    assert value == -math.log((64.0 + 4096.0 + 1024.0 + 1.0) / (4096.0 + 1024.0 + 1.0))
    assert rec["ratios"]["temp"] == (64.0 + 1.0) / (0.0 + 1.0)


def test_eps_is_two_to_the_minus_ten_of_the_smallest_nonzero_reference():
    assert env_mod.mem_objective_eps((0.0, 3072.0, 2048.0)) == 2.0
    with pytest.raises(ValueError, match="all zero"):
        env_mod.mem_objective_eps((0.0, 0.0, 0.0))


# ------------------------------------------------ 1. one deadline, readiness
@pytest.fixture
def fake_ray(monkeypatch):
    class _Future:
        def __init__(self, fn, a, k, delay):
            self._fn, self._a, self._k = fn, a, k
            self._delay = float(delay)

        def result(self):
            return self._fn(*self._a, **self._k)

    class GetTimeoutError(Exception):
        pass

    class RayActorError(Exception):
        pass

    def _get(fut, timeout=None):
        if isinstance(fut, _Future):
            if timeout is not None and fut._delay > float(timeout):
                raise GetTimeoutError(f"{fut._delay} > {timeout}")
            return fut.result()
        return fut

    def _wait(futures, num_returns=None, timeout=None):
        ready = [f for f in futures
                 if not (timeout is not None and f._delay > float(timeout))]
        return ready, [f for f in futures if f not in ready]

    def _kill(actor, no_restart=False):
        actor.killed = True

    exc = types.SimpleNamespace(GetTimeoutError=GetTimeoutError,
                                RayActorError=RayActorError)
    ray = types.SimpleNamespace(get=_get, wait=_wait, kill=_kill,
                                cancel=lambda *a, **k: None, exceptions=exc)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return _Future


class _Method:
    def __init__(self, future, fn, delay, seen=None):
        self._future, self._fn, self._delay = future, fn, delay
        self._seen = seen

    def remote(self, *a, **k):
        if self._seen is not None:
            self._seen.append(k.get("timeout_s"))
        return self._future(self._fn, a, k, self._delay())


class _Actor:
    # Construction takes `build` simulated seconds and an evaluate `run`;
    # an evaluate dispatched before `ready` resolved waits for the rest of
    # the construction first, as a Ray actor's first call does.
    def __init__(self, future, *, build=0.0, run=0.0):
        self.built = False
        self.killed = False
        self.timeouts: list = []
        self.order: list = []
        self.refuse_seen: list = []
        self.ready = _Method(future, self._ready, lambda: build)
        self.pop_oom_flag = _Method(future, lambda: False, lambda: 0.0)
        self.consume_call_telemetry = _Method(future, lambda: {}, lambda: 0.0)
        self.evaluate = _Method(
            future, self._evaluate,
            lambda: run + (0.0 if self.built else build), self.timeouts)

    def _ready(self):
        self.order.append("ready")
        self.built = True
        return True

    def _evaluate(self, order, specs, step, **kw):
        self.order.append("evaluate")
        self.refuse_seen.append(kw.get("refuse"))
        tokens = np.full((TOK,), 7, np.int32)
        eqn = np.full((TOK,), 3, np.int32)
        reward = np.full((NUM_REWARDS,), -100.0, np.float32)
        return tokens, eqn, reward


def _pool(actors, **kw):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool
    kw.setdefault("respawn_factory", None)
    return CpuApproxPool(
        actors, max_tokens=TOK, num_rewards=NUM_REWARDS,
        cosine_sim_idx=int(REWARD_INDEX["cosine_sim"]),
        frob_residual_idx=int(REWARD_INDEX["frob_residual"]),
        fidelity_idx=int(REWARD_INDEX["fidelity"]),
        sparsity_idx=int(REWARD_INDEX["sparsity"]), **kw)


@pytest.fixture
def pool_env(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    yield
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def _batch(pool, n=1, step=3):
    orders = [np.arange(3, dtype=np.int32)] * n
    specs = [np.full((3, 1, 3), -1, np.int32)] * n
    return pool.evaluate_batch(orders, specs, [step] * n)


def test_an_actor_whose_construction_outlasts_the_deadline_is_not_killed(
        fake_ray, pool_env):
    t = 0.01
    slow = _Actor(fake_ray, build=100 * t)
    pool = _pool([slow], timeout_s=t)
    tokens, _eqn, rewards, mask = _batch(pool)
    assert not slow.killed
    assert slow.order == ["ready", "evaluate"]
    assert mask.tolist() == [False] and rewards[0, 0] == -100.0
    assert slow.timeouts == [t]
    assert pool.stats()["timeouts"] == 0


def test_a_respawned_actor_is_ready_before_it_gets_a_plan(fake_ray,
                                                          pool_env):
    t = 0.01
    fresh: list = []

    def factory():
        fresh.append(_Actor(fake_ray, build=100 * t))
        return fresh[-1]

    first = _Actor(fake_ray)
    pool = _pool([first], timeout_s=t, respawn_factory=factory)
    swapped = pool._recycle_actor(first)
    assert first.killed and swapped is fresh[-1] and swapped.built
    held = _Actor(fake_ray, run=100 * t)
    pool._alive.clear()
    pool._alive.append(held)
    _batch(pool)
    assert held.killed
    stop = time.time() + 10.0
    while pool.size() == 0 and time.time() < stop:
        time.sleep(0.01)
    assert pool.size() == 1
    # The respawned actor is ready first, then scores the killed plan as "deadline" (Q9b c).
    assert fresh[-1].built and fresh[-1].order == ["ready", "evaluate"]
    assert fresh[-1].refuse_seen == ["deadline"]
    _batch(pool)
    assert fresh[-1].order == ["ready", "evaluate", "evaluate"]
    assert not fresh[-1].killed


def test_there_is_no_cold_budget(fake_ray, pool_env):
    from alphagrad.approx import ppo
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool
    params = inspect.signature(CpuApproxPool.__init__).parameters
    assert "initial_timeout_s" not in params and "warm_after" not in params
    with pytest.raises(TypeError):
        _pool([_Actor(fake_ray)], timeout_s=1.0, initial_timeout_s=4.0)
    assert "initial_timeout_s" not in inspect.getsource(ppo)
    assert ppo.make_argparser().parse_args([]).ray_measure_timeout == 300.0
    t = 0.01
    actor = _Actor(fake_ray, run=2 * t)
    _batch(_pool([actor], timeout_s=t))
    assert actor.killed and actor.timeouts == [t]


# ---------------------------------------------- 2. a killed call is a failed plan
def test_a_killed_call_goes_to_a_free_actor_to_be_scored_as_deadline(
        fake_ray, pool_env, capsys):
    t = 0.01
    slow, fast = _Actor(fake_ray, run=2 * t), _Actor(fake_ray)
    pool = _pool([slow, fast], timeout_s=t)
    tokens, _eqn, rewards, mask = _batch(pool, n=2)
    assert slow.killed and not fast.killed
    assert fast.refuse_seen == [None, "deadline"]
    assert mask.tolist() == [False, False], "a failed plan enters the update"
    assert rewards[0, 0] == -100.0 and rewards[1, 0] == -100.0
    assert pool.stats()["timeouts"] == 1
    assert pool.stats()["deadline_scored"] == 1
    # The scoring actor counts and records the plan; nothing is left in this process.
    assert env_mod.consume_refused_counts() == {}
    assert env_mod.consume_plan_records()["records"] == []
    assert "[SENTINEL] batch timeout slot=0" in capsys.readouterr().out


def test_the_single_dispatch_sends_a_killed_call_to_a_free_actor(
        fake_ray, pool_env):
    t = 0.01
    slow, fast = _Actor(fake_ray, run=2 * t), _Actor(fake_ray)
    pool = _pool([slow, fast], timeout_s=t)
    out = pool.evaluate(np.arange(3, dtype=np.int32),
                        np.full((3, 1, 3), -1, np.int32), 3, None)
    assert slow.killed and fast.refuse_seen == ["deadline"]
    assert not _is_hard_sentinel(out[-1]) and out[-1][0] == -100.0
    assert pool.stats()["deadline_scored"] == 1


def test_a_kill_that_no_actor_can_score_is_excluded_and_recorded(
        fake_ray, pool_env, capsys):
    t = 0.01
    pool = _pool([_Actor(fake_ray, run=2 * t)], timeout_s=t)
    out = pool.evaluate(np.arange(3, dtype=np.int32),
                        np.full((3, 1, 3), -1, np.int32), 3, None)
    assert _is_hard_sentinel(out[-1])
    assert env_mod.consume_refused_counts() == {
        "deadline": 1, "total": 1, "excluded": 1}
    rec = env_mod.consume_plan_records()["records"][-1]
    assert rec["refused"] == "deadline" and rec["refusal_where"] == "no live actor"
    assert rec["refusal_timeout_s"] == t
    assert pool.stats()["deadline_unscored"] == 1
    assert "could not be scored (no live actor)" in capsys.readouterr().out


def test_a_killed_tokenization_step_is_no_plan(fake_ray, pool_env):
    t = 0.01
    pool = _pool([_Actor(fake_ray, run=2 * t)], timeout_s=t)
    rewards = _batch(pool, step=1)[2]
    assert _is_hard_sentinel(rewards[0])
    assert env_mod.consume_refused_counts() == {}
    assert env_mod.consume_plan_records()["records"] == []


# ------------------------------------------------------------ 6. in-process
def test_the_trainer_scores_a_refused_plan_at_its_own_deadline(
        paired, monkeypatch):
    from alphagrad.approx import ppo
    env_mod.set_measure_timeout_s(None)
    ns = ppo.make_argparser().parse_args([])
    assert ppo.configure_measure_timeout(ns) == 300.0
    reward, counts, recs, seen = _run(
        monkeypatch, approx=lambda make: _Wrapped(make(), temp=SWELL))
    _check_scored(reward, recs[-1], counts, reason="gate",
                  program=_triple(seen[b"approx"]))
    ns = ppo.make_argparser().parse_args(["--ray-measure-timeout", "30"])
    ppo.configure_measure_timeout(ns)
    reward, _c, recs, _s = _run(
        monkeypatch, approx=lambda make: _Wrapped(make(), temp=SWELL))
    assert recs[-1]["refusal_timeout_s"] == 30.0


def test_the_sweep_writes_a_row_for_a_refused_plan(tmp_path, monkeypatch):
    # The sweep rewrites the face width and exports its knobs at import; both
    # are put back so no later module of this worker inherits them.
    saved_env = dict(os.environ)
    monkeypatch.setattr(env_mod, "MAX_FACES", env_mod.MAX_FACES)
    import alphagrad.approx.tools.landscape_map as lm
    monkeypatch.delenv("ALPHAGRAD_COST_FORM", raising=False)
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: 1)
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    args = lm.make_argparser().parse_args([
        "--example", "Helmholtz", "--dataset", "none",
        "--out-dir", str(tmp_path), "--order", "reverse",
        "--ladder", "", "--no-all-rung", "--no-skip-plan",
        "--reps", "1", "--warmup-trials", "0",
        "--noise-floor-reps", "0", "--no-figure"])
    assert args.ray_measure_timeout == 300.0
    monkeypatch.setattr(lm, "ARGS", args)
    try:
        lm.main()
    finally:
        env_mod.set_measure_timeout_s(None)
        env_mod.consume_refused_counts()
        env_mod.consume_plan_records()
        for key in set(os.environ) - set(saved_env):
            del os.environ[key]
        os.environ.update(saved_env)
    with open(tmp_path / "rows.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["role"] for r in rows} == {"reference", "candidate"}
    for r in rows:
        assert r["refused"] == "gate", r
        assert float(r["latency_ns"]) == float(np.float32(DEADLINE * 1e9))
        assert float(r["quality"]) == 0.0
        detail = json.loads(r["refusal"])
        assert detail["refusal_timeout_s"] == DEADLINE
        assert detail["refusal_program"] is True
        assert detail["static_peak_limit_bytes"] == 1.0
