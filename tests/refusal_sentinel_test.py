"""dsnn-4eq and dsnn-dkz (owner rulings 2026-09-24) -- a refused plan is
scored, not dropped.

A refused plan takes a finite sentinel on each channel's own convention:
latency is log(timeout / t_ref) with THE configured deadline, for every kind;
memory is the candidate's real static ratios when a compiled program exists
(a ``gate`` refusal) and the (c') sentinel otherwise (a ``compile`` refusal),
on slot 5 and on slot 11; quality is 0. The reason rides the plan-log record
and the refusal counters. A call the deadline killed is excluded instead
(refusal_rules_test.py).
"""
from __future__ import annotations

import math
import os

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
    REWARD_INDEX,
    mem_objective,
    paired_log_costs,
)


# Resolved per call, so a tree without it fails the one test that needs it.
def memory_sentinel(reference, bytes_limit):
    return env_mod.memory_sentinel(reference, bytes_limit)


# captured at import: a test that runs two plans patches the same attribute
# twice, and the second fake must wrap the real cache, not the first fake
_REAL_CACHED_COMPILE = _cc.cached_compile

LIMIT = 16_000_000_000
SWELL = 20_000_000_000
TIMEOUT = 120.0
LAT = int(REWARD_INDEX["latency_ns"])
MEM = int(REWARD_INDEX["peak_memory"])
QUAL = int(REWARD_INDEX["quality"])
MOBJ = int(REWARD_INDEX["mem_objective"])


class XlaRuntimeError(RuntimeError):
    pass


_TRITON = ("INTERNAL: Failed to compile Triton kernel. Context: [Fusion: "
           "fusion.35 = f32[32,1024,32]{2,1,0}]")


# ----------------------------------------------------------------- the toy
class _SwollenAnalysis:
    def __init__(self, inner, extra):
        self._inner = inner
        self.temp_size_in_bytes = int(inner.temp_size_in_bytes) + int(extra)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Swollen:
    def __init__(self, inner, extra):
        self._inner = inner
        self._extra = extra
        self.calls = 0

    def memory_analysis(self):
        return _SwollenAnalysis(self._inner.memory_analysis(), self._extra)

    def __call__(self, *a, **kw):
        self.calls += 1
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _toy_env():
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    closed = jax.make_jaxpr(toy)(x)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[x], argnums=(0,), num_envs=0, target_fun=toy,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)


def _plan_arrays(n):
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    return jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips)


@pytest.fixture
def paired(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    env_mod.set_measure_timeout_s(TIMEOUT)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    yield
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def _run(monkeypatch, *, swell=None, fail_approx=False):
    real = _REAL_CACHED_COMPILE
    wrapped = {}

    def fake(key, fn, *a, **kw):
        if fail_approx and bytes(key).startswith(b"approx:"):
            try:
                raise XlaRuntimeError(_TRITON)
            except XlaRuntimeError as _e:
                raise env_mod.MeasureCompileFailure(_TRITON) from _e
        out = real(key, fn, *a, **kw)
        if swell is not None and bytes(key).startswith(b"approx:"):
            out = _Swollen(out, swell)
            wrapped[b"approx:"] = out
        return out

    monkeypatch.setattr(_cc, "cached_compile", fake)
    env = _toy_env()
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs, faces, skips = _plan_arrays(len(order))
    samples = (jnp.asarray(
        np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)])),)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    out = env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(order), specs,
        faces, skips, len(order), *samples)
    counts = env_mod.consume_refused_counts()
    recs = env_mod.consume_plan_records()["records"]
    return np.asarray(out[-1], dtype=np.float32), counts, recs, wrapped


def _is_hard_sentinel(reward):
    return bool(np.all(reward[list(env_mod.COMPUTE_REWARD_INDICES)]
                       <= env_mod.SENTINEL_COST * 0.99))


# ------------------------------------------------------------- gate refusal
def test_a_gate_refusal_is_scored_at_the_timeout_with_its_real_memory(
        paired, monkeypatch, capsys):
    reward, counts, recs, wrapped = _run(monkeypatch, swell=SWELL)
    assert counts == {"gate": 1, "total": 1, "scored": 1}, counts
    assert not _is_hard_sentinel(reward)
    assert np.isfinite(reward).all()
    assert wrapped[b"approx:"].calls == 0, "a gated candidate must never run"
    rec = recs[-1]
    assert rec["refused"] == "gate"
    assert rec["sentinelled"] is True and rec["replayable"] is True
    assert rec["device_bytes_limit"] == LIMIT
    assert rec["static_peak_bytes"] > SWELL
    assert rec["refusal_timeout_s"] == TIMEOUT
    assert rec["refusal_latency_ns"] == TIMEOUT * 1e9
    assert rec["refusal_program"] is True
    assert rec["refusal_sentinel"] is None
    # THE REAL MEMORY of the refused executable, read after its compile.
    ma = wrapped[b"approx:"].memory_analysis()
    cand = (float(ma.temp_size_in_bytes), float(ma.output_size_in_bytes),
            float(ma.argument_size_in_bytes))
    assert rec["mem_temp_bytes"] == cand[0] > SWELL
    assert rec["mem_output_bytes"] == cand[1]
    assert rec["mem_args_bytes"] == cand[2]
    assert rec["mem_peak_source"] == "not_measured"
    # THE REFERENCE was measured beside it, with its own window count.
    ref_lat = rec["ref_latency_ns"]
    ref = (rec["ref_temp_bytes"], rec["ref_output_bytes"],
           rec["ref_args_bytes"])
    assert ref_lat > 0.0 and ref[1] > 0.0
    assert rec["ref_measure_windows"] >= 1 and rec["ref_measure_inner"] >= 1
    assert rec["ref_measure_secs"] > 0.0
    assert rec["measure_windows"] == 0 and rec["measure_secs"] == 0.0
    # LATENCY = log(timeout / t_ref) on the channel's own convention.
    d_lat, d_mem, _ = paired_log_costs(TIMEOUT * 1e9, cand[0], ref_lat,
                                       ref[0])
    assert rec["rewards"][LAT] == -d_lat
    assert reward[LAT] == np.float32(-d_lat)
    assert d_lat > 10.0
    # SLOT 5 = the real static temp ratio; SLOT 11 = the ruling's three.
    assert rec["rewards"][MEM] == -d_mem
    assert d_mem == pytest.approx(math.log(cand[0] / max(ref[0], 1.0)))
    want_obj, want_rec = mem_objective(cand, ref)
    eps = 2.0 ** -10 * min(x for x in ref if x > 0.0)
    assert rec["rewards"][MOBJ] == want_obj
    assert rec["mem_ratios"] == want_rec["ratios"]
    assert rec["mem_objective_eps"] == eps
    assert rec["mem_ratios"]["temp"] == (cand[0] + eps) / (ref[0] + eps)
    assert rec["mem_ratios"]["out"] == (cand[1] + eps) / (ref[1] + eps)
    assert rec["mem_ratios"]["args"] == (cand[2] + eps) / (ref[2] + eps)
    # QUALITY 0, the bounded channels at their floors, the reserved at 0.
    assert reward[QUAL] == 0.0
    assert reward[REWARD_INDEX["fidelity"]] == -1.0
    assert reward[REWARD_INDEX["sparsity"]] == -1.0
    assert reward[REWARD_INDEX["grad_coverage"]] == 0.0
    assert reward[REWARD_INDEX["bkstep_acc"]] == 0.0
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[refused] gate")]
    assert line and f"limit_bytes={LIMIT}" in line[0]
    assert any("scored" in ln for ln in line)


def test_a_gate_refusal_scores_below_a_plan_that_ran(paired, monkeypatch):
    refused, counts, recs, _ = _run(monkeypatch, swell=SWELL)
    assert counts.get("gate") == 1
    measured, counts, recs, wrapped = _run(monkeypatch, swell=0)
    assert counts == {}, counts
    assert wrapped[b"approx:"].calls > 0
    assert "refused" not in recs[-1]
    for slot in (LAT, MEM, MOBJ):
        assert measured[slot] > refused[slot], (slot, measured, refused)


def test_the_sentinel_follows_the_configured_timeout(paired, monkeypatch):
    env_mod.set_measure_timeout_s(30.0)
    _, _, recs_a, _ = _run(monkeypatch, swell=SWELL)
    env_mod.set_measure_timeout_s(300.0)
    _, _, recs_b, _ = _run(monkeypatch, swell=SWELL)
    a, b = recs_a[-1], recs_b[-1]
    assert a["refusal_timeout_s"] == 30.0 and b["refusal_timeout_s"] == 300.0
    for rec, t in ((a, 30.0), (b, 300.0)):
        d_lat, _, _ = paired_log_costs(
            t * 1e9, rec["mem_temp_bytes"], rec["ref_latency_ns"],
            rec["ref_temp_bytes"])
        assert rec["rewards"][LAT] == -d_lat
    # the two references differ by timer noise only; the timeouts by 10x
    drift = math.log(b["ref_latency_ns"] / a["ref_latency_ns"])
    assert (b["rewards"][LAT] - a["rewards"][LAT]) == pytest.approx(
        -math.log(10.0) + drift, abs=1e-9)


def test_without_a_timeout_a_refusal_cannot_be_scored(paired, monkeypatch):
    env_mod.set_measure_timeout_s(None)
    with pytest.raises(RuntimeError, match="enforces no timeout"):
        _run(monkeypatch, swell=SWELL)


# ---------------------------------------------------------- compile failure
def test_a_compile_failure_is_scored_at_the_timeout_with_the_sentinel(
        paired, monkeypatch, capsys):
    reward, counts, recs, _ = _run(monkeypatch, fail_approx=True)
    assert counts == {"compile": 1, "total": 1, "scored": 1}, counts
    assert not _is_hard_sentinel(reward)
    rec = recs[-1]
    assert rec["refused"] == "compile:XlaRuntimeError"
    assert rec["refusal_where"] == "approx compile"
    assert "Triton" in rec["refusal_error"]
    assert "refusal_timeout_factor" not in rec
    assert rec["refusal_latency_ns"] == TIMEOUT * 1e9
    assert rec["refusal_bytes_limit"] == LIMIT
    assert rec["refusal_program"] is False
    assert rec["mem_temp_bytes"] is None
    ref_lat = rec["ref_latency_ns"]
    ref = (rec["ref_temp_bytes"], rec["ref_output_bytes"],
           rec["ref_args_bytes"])
    # THE (c') SENTINEL: the reference's inputs and outputs, temporaries for
    # the rest of the card.
    cand, sent = memory_sentinel(ref, LIMIT)
    assert sent["branch"] == "fill" and rec["refusal_sentinel"] == sent
    assert cand == (LIMIT - ref[2] - ref[1], ref[1], ref[2])
    d_lat, d_mem, _ = paired_log_costs(TIMEOUT * 1e9, cand[0], ref_lat,
                                       ref[0])
    assert rec["rewards"][LAT] == -d_lat
    assert reward[LAT] == np.float32(-d_lat)
    assert rec["rewards"][MEM] == -d_mem
    eps = 2.0 ** -10 * min(x for x in ref if x > 0.0)
    assert rec["rewards"][MOBJ] == -math.log(
        (cand[0] + eps) / (ref[0] + eps))
    assert rec["mem_ratios"]["out"] == 1.0
    assert rec["mem_ratios"]["args"] == 1.0
    assert reward[QUAL] == 0.0
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[refused] compile")]
    assert line and "XlaRuntimeError" in line[0]


def test_a_compile_failure_and_a_gate_refusal_share_one_latency_sentinel(
        paired, monkeypatch):
    _, _, recs_c, _ = _run(monkeypatch, fail_approx=True)
    _, _, recs_g, _ = _run(monkeypatch, swell=SWELL)
    assert (recs_c[-1]["refusal_latency_ns"]
            == recs_g[-1]["refusal_latency_ns"] == TIMEOUT * 1e9)


def test_compile_measure_names_the_failure_and_keeps_the_cause(monkeypatch):
    monkeypatch.setitem(env_mod._MEASURE_TOOLCHAIN, "checked", True)

    class _Lowered:
        def compile(self, compiler_options=None):
            raise XlaRuntimeError("INVALID_ARGUMENT: no such option")

    with pytest.raises(env_mod.MeasureCompileFailure) as ei:
        env_mod._compile_measure(_Lowered())
    assert isinstance(ei.value.__cause__, XlaRuntimeError)
    assert isinstance(ei.value, RuntimeError)


# ------------------------------------------------------------- the formula
REF = {"latency_ns": 2.0e5, "memory_bytes": 4096.0,
       "static": (4096.0, 512.0, 1024.0), "bytes_limit": LIMIT,
       "bytes_limit_source": "device", "measure_latency": True}


def test_refused_reward_rejects_a_missing_timeout_or_limit(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    with pytest.raises(RuntimeError, match="no timeout"):
        env_mod.refused_reward("oom", timeout_s=None, reference=REF)
    with pytest.raises(RuntimeError, match="no timeout"):
        env_mod.refused_reward("gate", timeout_s=0.0, reference=REF,
                               candidate_static=(1.0, 1.0, 1.0))
    with pytest.raises(RuntimeError, match="bytes_limit"):
        env_mod.refused_reward("compile", timeout_s=1.0,
                               reference=dict(REF, bytes_limit=None))
    with pytest.raises(ValueError, match="never scored"):
        env_mod.refused_reward("timeout", timeout_s=1.0, reference=REF)


def test_refused_reward_is_never_better_than_a_plan_that_passed_the_gate(
        monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    t_ref, o_ref, a_ref = REF["static"]
    # the plan at the gate's edge: every byte the card has, inputs and
    # outputs no larger than the reference's, 0.9 ms against a 1 ms timeout
    edge = (float(LIMIT) - a_ref - o_ref, o_ref, a_ref)
    d_lat, d_mem, _ = paired_log_costs(0.9e6, edge[0], REF["latency_ns"],
                                       REF["memory_bytes"])
    measured = (-d_lat, -d_mem, mem_objective(edge, REF["static"])[0])
    for kind in ("compile", "oom", "untraceable", "muls-cap", "raised"):
        slots, _ = env_mod.refused_reward(kind, timeout_s=0.001,
                                          reference=REF)
        assert slots[LAT] < measured[0]
        assert slots[MEM] <= measured[1]
        assert slots[MOBJ] <= measured[2]
    slots, _ = env_mod.refused_reward(
        "gate", timeout_s=0.001, reference=REF,
        candidate_static=(edge[0] + 1.0, o_ref, a_ref))
    assert slots[LAT] < measured[0]
    assert slots[MEM] < measured[1]
    assert slots[MOBJ] < measured[2]
