# THE PAIRED REFERENCE IS jax.grad OF THE TARGET FUNCTION (dsnn-xta, owner
# rulings 2026-09-24 Q39-Q41): reverse-mode autodiff of cfg.target_fun with
# respect to cfg.argnums on the base program's inputs, compiled with the
# measurement's compiler options, once per (process, episode, target). It is
# the quality reference gradient, the paired latency reference and the static
# memory reference of slot 11. The graphax rev-exact stays as telemetry
# behind ALPHAGRAD_REV_EXACT_TELEMETRY, one paired ratio per episode, off by
# default.
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
from graphax import jacve                                       # noqa: E402

import alphagrad.approx.env as envmod                            # noqa: E402
from alphagrad.approx.common import compile_cache as _cc         # noqa: E402
from alphagrad.approx.env import (                               # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    REWARD_INDEX,
    StepAction,
    VertexEliminationEnv,
)

_LAT = int(REWARD_INDEX["latency_ns"])
_MEM = int(REWARD_INDEX["peak_memory"])
_MSLOT = int(REWARD_INDEX["mem_objective"])

_N = 64
_rng = np.random.default_rng(0)
_W1 = jnp.asarray(_rng.standard_normal((_N, _N), dtype=np.float32) / 8.0)
_W2 = jnp.asarray(_rng.standard_normal((_N, _N), dtype=np.float32) / 8.0)
_X = jnp.asarray(np.linspace(-1.0, 1.0, _N, dtype=np.float32))


def _toy(x):
    h = jnp.tanh(_W1 @ x)
    y = jnp.tanh(_W2 @ h)
    return jnp.sum(y * y)


def _make_env(**kw):
    closed = jax.make_jaxpr(_toy)(_X)
    kw.setdefault("measure_latency", True)
    kw.setdefault("num_data_points", 2)
    kw.setdefault("reps_per_point", 2)
    kw.setdefault("latency_inner_reps", 2)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X], argnums=(0,), num_envs=0, target_fun=_toy, **kw,
    )


def _run_plan(env, order):
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    for v in order:
        face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
        face_skip = jnp.zeros((MAX_FACES,), jnp.int32)
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        ).state
    return np.asarray(state.reward)


def _rev_order(env):
    return sorted(int(x) for x in np.asarray(env.valid_vertices))[::-1]


def _drain():
    out = envmod.consume_plan_records()
    assert out["records"], "no terminal plan was recorded"
    assert out["paired_ref"]["records"], "no reference was recorded"
    return out


def _triple(ex):
    ma = ex.memory_analysis()
    return (float(ma.temp_size_in_bytes), float(ma.output_size_in_bytes),
            float(ma.argument_size_in_bytes))


def _compile_reference(env):
    return envmod._compile_measure(
        jax.jit(envmod.reference_program(env.config), keep_unused=True)
        .lower(*env.args))


def _spy_executables(monkeypatch, env=None):
    """Every executable the callback asks the compile cache for, by prefix.
    With ``env`` the candidate (``approx:``) is REPLACED by the reference
    program, so the callback scores the reference against itself."""
    seen = {"approx": [], "paired-ref": [], "rev-exact": []}
    real = _cc.cached_compile

    def _spy(key, fn):
        key = bytes(key)
        if env is not None and key.startswith(b"approx:"):
            fn = lambda: _compile_reference(env)               # noqa: E731
        out = real(key, fn)
        for prefix in seen:
            if key.startswith(prefix.encode() + b":"):
                seen[prefix].append(out)
        return out
    monkeypatch.setattr(_cc, "cached_compile", _spy)
    return seen


@pytest.fixture(autouse=True)
def _paired_log_with_the_plan_log(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "temp")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.delenv("ALPHAGRAD_REV_EXACT_TELEMETRY", raising=False)
    monkeypatch.delenv("ALPHAGRAD_WALK_EPISODE", raising=False)
    # The process-local executable memo is keyed on the plan, not on what
    # was compiled under it, so a substituted candidate must not outlive
    # its test.
    _cc._LOCAL_CACHE.clear()
    envmod._REV_TELEMETRY["episode"] = None
    envmod._REV_TELEMETRY["keys"] = set()
    envmod.consume_plan_records()
    yield
    _cc._LOCAL_CACHE.clear()
    envmod.consume_plan_records()


# ------------------------------------------------- 1. the reference program
def test_the_paired_reference_is_jax_grad_of_the_target(monkeypatch):
    seen = _spy_executables(monkeypatch)
    env = _make_env()
    assert envmod.reference_kind(env.config) == "jax.grad"
    _run_plan(env, _rev_order(env))
    assert len(seen["paired-ref"]) == 1 and not seen["rev-exact"]
    ref_ex = seen["paired-ref"][0]
    out = ref_ex(*env.args)
    leaves = jax.tree_util.tree_leaves(out)
    g = jax.grad(_toy)(_X)
    assert len(leaves) == 1 and leaves[0].shape == g.shape
    assert np.allclose(np.asarray(leaves[0]), np.asarray(g),
                       rtol=1e-6, atol=1e-7)
    # the same program `reference_program` builds, compiled the same way
    assert _triple(_compile_reference(env)) == _triple(ref_ex)
    pr = _drain()["paired_ref"]["records"][-1]
    assert pr["reference"] == "jax.grad"
    assert pr["rev_exact"] is None
    assert "order" not in pr


def test_the_reference_program_needs_a_target_function():
    closed = jax.make_jaxpr(_toy)(_X)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=[_X], argnums=(0,), num_envs=0, target_fun=None)
    with pytest.raises(ValueError, match="target_fun"):
        envmod.reference_program(env.config)


# ------------------- 2. the reference gradient IS the rev-exact gradient
_TLM_ENV = {"ALPHAGRAD_TLM_SEQ": "32", "ALPHAGRAD_TLM_DMODEL": "128",
            "ALPHAGRAD_TLM_VOCAB": "1024"}


def _lm_env(monkeypatch, example, dataset, rule):
    import alphagrad.approx.tools.landscape_map as lm
    from alphagrad.approx.common import examples as _ex
    for k, v in _TLM_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(_ex, "_EQ_NN_HIDDEN", 256)
    argv = ["--example", example, "--dataset", dataset, "--seed", "250197",
            "--num-eval-samples", "1", "--num-data-points", "1",
            "--reps-per-point", "1", "--latency-inner-reps", "1",
            "--out-dir", "/tmp/grad_reference_test"]
    if rule is not None:
        argv += ["--temporal-rule", rule, "--step-position", "7"]
    a = lm.make_argparser().parse_args(argv)
    env, _eval, _cj = lm.build_env(a)
    return env


def _rev_exact_and_reference(env):
    cfg = env.config
    rev = _rev_order(env)
    fn_rev = jacve(cfg.target_fun, rev, argnums=cfg.argnums,
                   has_aux=cfg.has_aux, sparse_representation=cfg.sparse,
                   jaxpr=cfg.jaxpr, consts=list(env.consts),
                   transforms=[], face_transforms=None)
    args = tuple(env.args)
    out_rev = jax.jit(fn_rev, keep_unused=True)(*args)
    out_ref = jax.jit(envmod.reference_program(cfg), keep_unused=True)(*args)
    return out_rev, out_ref


def _leaf_cosines(out_ref, out_rev):
    e = envmod._gradient_leaves(out_ref)
    a = envmod._gradient_leaves(out_rev)
    assert len(e) == len(a) and len(e) > 0
    cos = []
    for x, y in zip(e, a):
        assert envmod._leaf_shape(x) == envmod._leaf_shape(y)
        y = y.dense() if envmod._is_sparse_tensor(y) else y
        x64 = np.asarray(x, dtype=np.float64).ravel()
        y64 = np.asarray(y, dtype=np.float64).ravel()
        cos.append(float(np.dot(x64, y64)
                         / max(np.linalg.norm(x64) * np.linalg.norm(y64),
                               1e-300)))
    return cos


def test_on_the_toy_the_reference_gradient_is_the_rev_exact_gradient():
    env = _make_env()
    out_rev, out_ref = _rev_exact_and_reference(env)
    cos = _leaf_cosines(out_ref, out_rev)
    assert min(cos) > 1.0 - 1e-6
    # and the production comparator accepts the pair as it is
    c, _rel = envmod._quality_metrics(out_ref, out_rev)
    assert float(c) > 1.0 - 1e-6


@pytest.mark.parametrize("example,dataset,rule", [
    ("NeuralNetwork", "mnist", None),
    ("TransformerLM", "wikitext2", None),
    ("RSNN_SHD", "none", "rtrl"),
    ("RSNN_SHD", "none", "bptt"),
])
def test_the_reference_gradient_is_the_rev_exact_gradient(
        monkeypatch, example, dataset, rule):
    env = _lm_env(monkeypatch, example, dataset, rule)
    assert envmod.reference_kind(env.config) == "jax.grad"
    out_rev, out_ref = _rev_exact_and_reference(env)
    cos = _leaf_cosines(out_ref, out_rev)
    print(f"[grad-reference] {example} {rule or ''}: {len(cos)} leaves, "
          f"min cosine {min(cos):.9f}")
    assert min(cos) > 1.0 - 1e-6
    c, _rel = envmod._quality_metrics(out_ref, out_rev)
    assert float(c) > 1.0 - 1e-6


# --------------------------------- 3. the reference scores exactly 0
def test_the_reference_scores_exactly_zero_on_latency_and_slot_11(
        monkeypatch):
    real = envmod._time_one_rep

    def _const_rep(ex, eval_args, devices, inner):
        _l, _p, _s, out = real(ex, eval_args, devices, inner)
        return 123_456.0, _p, _s, out

    monkeypatch.setattr(envmod, "_time_one_rep", _const_rep)
    env = _make_env()
    seen = _spy_executables(monkeypatch, env)
    r = _run_plan(env, _rev_order(env))
    assert seen["approx"] and seen["paired-ref"]
    assert _triple(seen["approx"][-1]) == _triple(seen["paired-ref"][-1])
    assert float(r[_LAT]) == 0.0
    assert float(r[_MEM]) == 0.0
    assert float(r[_MSLOT]) == 0.0
    d = _drain()
    rec = d["records"][-1]
    pr = d["paired_ref"]["records"][-1]
    assert rec["rewards"][_MSLOT] == 0.0
    assert rec["mem_ratios"] == {"temp": 1.0, "args": 1.0, "out": 1.0}
    assert (rec["mem_temp_bytes"], rec["mem_output_bytes"],
            rec["mem_args_bytes"]) == (
        rec["ref_temp_bytes"], rec["ref_output_bytes"], rec["ref_args_bytes"])
    assert pr["delta_latency"] == 0.0 and pr["delta_memory"] == 0.0
    assert pr["latency_ns"] == pr["candidate_latency_ns"] == 123_456.0
    assert pr["reference"] == "jax.grad"


def test_the_rev_exact_plan_is_scored_against_the_reference(monkeypatch):
    seen = _spy_executables(monkeypatch)
    env = _make_env()
    r = _run_plan(env, _rev_order(env))
    cand = _triple(seen["approx"][-1])
    ref = _triple(seen["paired-ref"][-1])
    rec = _drain()["records"][-1]
    floor = envmod._MEM_LOG_FLOOR_BYTES
    expect = -(math.log(max(cand[0], floor) / max(ref[0], floor))
               + math.log(max(cand[2], floor) / max(ref[2], floor))
               + math.log(max(cand[1], floor) / max(ref[1], floor)))
    assert rec["rewards"][_MSLOT] == expect
    assert float(r[_MSLOT]) == float(np.float32(expect))
    print(f"[grad-reference] rev-exact plan against jax.grad on the toy: "
          f"cand={cand} ref={ref} r_mem={expect:+.6f}")


# ------------------------------------------ 4. the rev-exact as telemetry
def test_rev_exact_telemetry_is_off_by_default(monkeypatch):
    seen = _spy_executables(monkeypatch)
    env = _make_env()
    _run_plan(env, _rev_order(env))
    assert not seen["rev-exact"]
    assert _drain()["paired_ref"]["records"][-1]["rev_exact"] is None


def test_rev_exact_telemetry_is_one_paired_ratio_per_episode(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_REV_EXACT_TELEMETRY", "1")
    monkeypatch.setenv("ALPHAGRAD_WALK_EPISODE", "0")
    seen = _spy_executables(monkeypatch)
    env = _make_env()
    _run_plan(env, _rev_order(env))
    tel = _drain()["paired_ref"]["records"][-1]["rev_exact"]
    assert len(seen["rev-exact"]) == 1
    assert tel is not None
    assert tel["windows"] == (env.config.ref_num_data_points
                              * env.config.ref_reps_per_point)
    assert tel["latency_ratio"] > 0.0 and math.isfinite(tel["latency_ratio"])
    assert tel["latency_ratio"] == tel["latency_ns"] / tel["reference_latency_ns"]
    rev_temp = _triple(seen["rev-exact"][0])[0]
    ref_temp = _triple(seen["paired-ref"][0])[0]
    assert tel["temp_bytes"] == rev_temp
    assert tel["temp_ratio"] == rev_temp / ref_temp
    print(f"[rev-exact] toy: latency ratio {tel['latency_ratio']:.3f}, "
          f"temp ratio {tel['temp_ratio']:.3f} ({rev_temp:.0f} B vs "
          f"{ref_temp:.0f} B)")
    # the second plan of the episode carries none
    _run_plan(env, _rev_order(env)[::-1])
    assert _drain()["paired_ref"]["records"][-1]["rev_exact"] is None
    assert len(seen["rev-exact"]) == 1
    # a new episode takes one more
    monkeypatch.setenv("ALPHAGRAD_WALK_EPISODE", "1")
    _run_plan(env, _rev_order(env))
    assert _drain()["paired_ref"]["records"][-1]["rev_exact"] is not None
    assert len(seen["rev-exact"]) == 2
