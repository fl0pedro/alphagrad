from __future__ import annotations

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
from alphagrad.approx.env import REWARD_INDEX                   # noqa: E402

_REAL_CACHED_COMPILE = _cc.cached_compile

LIMIT = 16_000_000_000
SWELL = 20_000_000_000
LAT = int(REWARD_INDEX["latency_ns"])
MEM = int(REWARD_INDEX["peak_memory"])
REF_POINTS = 2
REF_REPS = 3


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

    def memory_analysis(self):
        return _SwollenAnalysis(self._inner.memory_analysis(), self._extra)

    def __call__(self, *a, **kw):
        return self._inner(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _toy_env(**kw):
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    closed = jax.make_jaxpr(toy)(x)
    kw.setdefault("ref_num_data_points", REF_POINTS)
    kw.setdefault("ref_reps_per_point", REF_REPS)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[x], argnums=(0,), num_envs=0, target_fun=toy,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1, num_data_points=2, reps_per_point=2, **kw)


def _plan_arrays(n):
    from alphagrad.approx.env import (
        FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX)
    specs = np.full((n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, MAX_FACES), np.int32)
    return jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips)


def _samples(n):
    rows = [np.linspace(-1.0, 1.0, 16, dtype=np.float32) * (0.5 + 0.25 * k)
            for k in range(n)]
    return (jnp.asarray(np.stack(rows)),)


class _Instrument:
    def __init__(self, monkeypatch):
        self.refs: list = []
        self.windows: list = []
        self.episode_ref_secs: list = []
        self.swell = False
        real_rep = env_mod._time_one_rep

        def rep(ex, eval_args, devices, inner):
            _l, _p, _s, _o = real_rep(ex, eval_args, devices, inner)
            n = len(self.windows)
            lat, peak = 1000.0 + 10.0 * n, 4096.0 + 64.0 * n
            self.windows.append((ex, lat, peak))
            return lat, peak, _s, _o

        def compile_(key, fn, *a, **kw):
            out = _REAL_CACHED_COMPILE(key, fn, *a, **kw)
            if bytes(key).startswith(b"paired-ref:"):
                self.refs.append(out)
            if self.swell and bytes(key).startswith(b"approx:"):
                out = _Swollen(out, SWELL)
            return out

        monkeypatch.setattr(env_mod, "_time_one_rep", rep)
        monkeypatch.setattr(_cc, "cached_compile", compile_)

    def _is_ref(self, ex):
        return any(ex is r for r in self.refs)

    def measure(self, env, order, samples):
        start = len(self.windows)
        specs, faces, skips = _plan_arrays(len(order))
        env_mod._callback(
            env.config, env.args, env.consts, jnp.asarray(order), specs,
            faces, skips, len(order), *samples)
        self.episode_ref_secs.append(
            list(env_mod._MEASURE_EPISODE["ref_secs"]))
        rec = env_mod.consume_plan_records()["records"][-1]
        mine = self.windows[start:]
        ref = [(lat, peak) for ex, lat, peak in mine if self._is_ref(ex)]
        cand = [(lat, peak) for ex, lat, peak in mine if not self._is_ref(ex)]
        # The first timed execution of each half is its cold reading and the
        # second its warm run, both outside the sample (no quality metric
        # here). The first window and the windows after it are the sample
        # (owner, 2026-09-26, dsnn-ep8v).
        self.first = (cand[:1], ref[:1])
        self.warm = (cand[1:2], ref[1:2])
        return rec, ref[2:], cand[2:]


def _median(values):
    return float(env_mod._aggregate_samples(list(values),
                                            want_top_quartile=True))


@pytest.fixture
def paired(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "watermark")
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setattr(env_mod, "_device_bytes_limit", lambda d: LIMIT)
    env_mod.set_measure_timeout_s(120.0)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.flush_measure_episode()
    yield _Instrument(monkeypatch)
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    env_mod.flush_measure_episode()


def _orders(env):
    vs = sorted(int(v) for v in np.asarray(env.valid_vertices))
    return vs, vs[::-1]


def test_the_second_plan_runs_no_reference_window_and_pairs_against_the_first(
        paired):
    env = _toy_env()
    fwd, rev = _orders(env)
    samples = _samples(2)
    rec_a, ref_a, cand_a = paired.measure(env, fwd, samples)
    cand_first_a, ref_first_a = paired.first
    cand_warm_a, ref_warm_a = paired.warm
    assert len(ref_a) == REF_POINTS * REF_REPS, ref_a
    assert cand_a and len(cand_first_a) == 1 and len(ref_first_a) == 1
    assert rec_a["measure_warm_s"] == cand_warm_a[0][0] / 1e9
    assert rec_a["ref_measure_warm_s"] == ref_warm_a[0][0] / 1e9
    rec_b, ref_b, cand_b = paired.measure(env, rev, samples)
    cand_first_b, ref_first_b = paired.first
    assert ref_b == [] and ref_first_b == [], (
        "the second plan timed the reference again")
    assert cand_b
    ref_lat = _median(lat for lat, _ in ref_a)
    ref_mem = _median(peak for _, peak in ref_a)
    cand_lat = _median(lat for lat, _ in cand_b)
    cand_mem = _median(peak for _, peak in cand_b)
    assert rec_a["ref_timing"] == "timed"
    assert rec_b["ref_timing"] == "reused"
    assert rec_a["ref_latency_ns"] == rec_b["ref_latency_ns"] == ref_lat
    assert rec_a["ref_watermark_bytes"] == ref_mem
    assert rec_b["ref_watermark_bytes"] == ref_mem
    assert rec_b["ref_temp_bytes"] == rec_a["ref_temp_bytes"]
    assert rec_b["candidate_latency_ns"] == cand_lat
    assert rec_b["candidate_memory_bytes"] == cand_mem
    d_lat, d_mem, _ = env_mod.paired_log_costs(cand_lat, cand_mem, ref_lat,
                                               ref_mem)
    assert rec_b["rewards"][LAT] == -d_lat
    assert rec_b["rewards"][MEM] == -d_mem
    want = env_mod.paired_window_log_ratios(
        [lat for lat, _ in cand_b], [lat for lat, _ in ref_a],
        env_mod._LAT_FLOOR_NS, 0.0)
    assert rec_b["ratio_log"]["latency"]["windows"] == [float(w) for w in want]
    for k in ("ref_measure_inner", "ref_measure_windows", "ref_measure_secs",
              "ref_measure_first_s"):
        assert rec_b[k] == rec_a[k], k
    assert rec_b["ref_measure_windows"] == REF_POINTS * REF_REPS
    assert rec_b["measure_first_s"] == cand_first_b[0][0] / 1e9
    assert rec_a["ref_measure_first_s"] == ref_first_a[0][0] / 1e9
    assert rec_a["measure_windows"] == len(cand_a)
    secs_a, secs_b = paired.episode_ref_secs
    assert len(secs_a) == 1 and secs_a[0] > 0.0
    assert secs_b == [0.0]


def test_a_refused_plan_pairs_against_the_reference_of_the_plan_before(
        paired):
    env = _toy_env()
    fwd, rev = _orders(env)
    samples = _samples(2)
    rec_a, ref_a, _ = paired.measure(env, fwd, samples)
    assert len(ref_a) == REF_POINTS * REF_REPS
    paired.swell = True
    rec_b, ref_b, cand_b = paired.measure(env, rev, samples)
    assert rec_b["refused"] == "gate"
    assert ref_b == [] and cand_b == []
    assert rec_b["ref_timing"] == "reused"
    assert rec_b["ref_latency_ns"] == _median(lat for lat, _ in ref_a)
    assert rec_b["ref_watermark_bytes"] == _median(pk for _, pk in ref_a)
    assert rec_b["ref_measure_windows"] == REF_POINTS * REF_REPS
    assert rec_b["ref_measure_secs"] == rec_a["ref_measure_secs"]


def test_a_refusal_times_the_reference_for_the_plans_after_it(paired):
    env = _toy_env()
    fwd, rev = _orders(env)
    samples = _samples(2)
    paired.swell = True
    rec_a, ref_a, cand_a = paired.measure(env, fwd, samples)
    assert rec_a["refused"] == "gate"
    assert len(ref_a) == REF_POINTS * REF_REPS and cand_a == []
    assert rec_a["ref_timing"] == "timed"
    paired.swell = False
    rec_b, ref_b, cand_b = paired.measure(env, rev, samples)
    assert "refused" not in rec_b
    assert ref_b == [] and cand_b
    assert rec_b["ref_timing"] == "reused"
    assert rec_b["ref_latency_ns"] == rec_a["ref_latency_ns"]
    assert rec_b["ref_watermark_bytes"] == rec_a["ref_watermark_bytes"]


def test_other_eval_sample_shapes_time_their_own_reference(paired):
    env = _toy_env(ref_num_data_points=3)
    fwd, rev = _orders(env)
    rec_a, ref_a, _ = paired.measure(env, fwd, _samples(2))
    rec_b, ref_b, _ = paired.measure(env, rev, _samples(3))
    assert len(ref_a) == 2 * REF_REPS
    assert len(ref_b) == 3 * REF_REPS
    assert rec_a["ref_timing"] == rec_b["ref_timing"] == "timed"
    rec_c, ref_c, _ = paired.measure(env, fwd, _samples(3))
    assert ref_c == [] and rec_c["ref_timing"] == "reused"
    assert rec_c["ref_latency_ns"] == _median(lat for lat, _ in ref_b)


def test_another_reference_protocol_times_its_own_reference(paired):
    env_a = _toy_env()
    env_b = _toy_env(ref_reps_per_point=REF_REPS + 1)
    fwd, rev = _orders(env_a)
    samples = _samples(2)
    rec_a, ref_a, _ = paired.measure(env_a, fwd, samples)
    rec_b, ref_b, _ = paired.measure(env_b, rev, samples)
    assert len(ref_a) == REF_POINTS * REF_REPS
    assert len(ref_b) == REF_POINTS * (REF_REPS + 1)
    assert rec_b["ref_timing"] == "timed"


def test_a_plan_refused_after_its_reference_windows_records_them_as_timed(
        paired, monkeypatch):
    env = _toy_env()
    fwd, rev = _orders(env)
    samples = _samples(2)
    real = env_mod._prof_add

    def boom(key, dt):
        if key == "cb.exec_measure":
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory")
        return real(key, dt)

    monkeypatch.setattr(env_mod, "_prof_add", boom)
    rec_a, ref_a, _ = paired.measure(env, fwd, samples)
    monkeypatch.setattr(env_mod, "_prof_add", real)
    env_mod.pop_measure_oom()
    assert rec_a["refused"] == "oom:measurement"
    assert len(ref_a) == REF_POINTS * REF_REPS
    assert rec_a["ref_timing"] == "timed"
    assert rec_a["ref_latency_ns"] == _median(lat for lat, _ in ref_a)
    rec_b, ref_b, _ = paired.measure(env, rev, samples)
    assert ref_b == [] and rec_b["ref_timing"] == "reused"
    assert rec_b["ref_latency_ns"] == rec_a["ref_latency_ns"]
