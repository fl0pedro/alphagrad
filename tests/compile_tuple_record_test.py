# The 0/1 compile tuple of a measured plan (owner ruling 2026-09-25, dsnn-dfw.212): it lands on
# the plan record and in the repro bundle, a cache hit names it, and a replay compiles with it.
# CPU only: the live compile of the candidate is made to fail the way the recorded Blackwell plans fail.
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as E                                # noqa: E402
import alphagrad.approx.tools.landscape_map as LM               # noqa: E402
from alphagrad.approx.common import compile_cache as CC         # noqa: E402
from alphagrad.approx.common import plan_log as plog            # noqa: E402
from alphagrad.approx.common import repro_bundle as RB          # noqa: E402

_VERIFY = "INTERNAL: Failed to verify Triton module for fusion: the class of job 67490"
_LIVE = [int(b) for b in E.measure_compile_live_tuple()]
_FIRST = [int(b) for b in E.MEASURE_COMPILE_TRY_ORDER[0]]


def _toy_env():
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    closed = jax.make_jaxpr(toy)(x)
    return E.VertexEliminationEnv.from_jaxpr(
        closed, args=[x], argnums=(0,), num_envs=0, target_fun=toy,
        measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)


def _wires(env):
    order = np.asarray(sorted(int(v) for v in np.asarray(env.valid_vertices)),
                       np.int32)
    n = len(order)
    specs = np.full((n, E.MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, E.MAX_FACES, E.FACE_SLOTS, 3), -1, np.int32)
    skips = np.zeros((n, E.MAX_FACES), np.int32)
    return order, specs, faces, skips


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
    monkeypatch.setenv("ALPHAGRAD_SKIP_COUNT_OPS", "1")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.delenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", raising=False)
    monkeypatch.delenv("ALPHAGRAD_MEASURE_COMPILE_FALLBACK", raising=False)
    monkeypatch.setattr(E, "_MEASURE_TOOLCHAIN", dict(
        E._MEASURE_TOOLCHAIN, checked=False, ok=True, detail="", mode="",
        link_faults=0))
    monkeypatch.setattr(E, "_toolchain_probe_compile", lambda: None)
    monkeypatch.setattr(CC, "_LOCAL_CACHE", {})
    monkeypatch.setattr(E, "_COMPILE_NOTES", {})
    # compile_fallbacks is a delta since the previous record of this process.
    monkeypatch.setattr(E, "_MEASURE_FALLBACKS_AT_LAST_RECORD",
                        [int(E._MEASURE_COMPILE_FALLBACKS["n"])])
    E.set_measure_timeout_s(120.0)
    real = jax.stages.Lowered.compile
    st = {"fail": 0, "seen": []}

    def compile(self, compiler_options=None, **kw):
        st["seen"].append(compiler_options)
        if st["fail"] > 0:
            st["fail"] -= 1
            raise RuntimeError(_VERIFY)
        return real(self, compiler_options=compiler_options, **kw)

    monkeypatch.setattr(jax.stages.Lowered, "compile", compile)
    E.consume_plan_records()
    yield st
    E.set_measure_timeout_s(None)
    E.consume_plan_records()


_SAMPLES = (jnp.asarray(np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)])),)


def _measure(env, wires, **kw):
    order, specs, faces, skips = wires
    E._callback(env.config, env.args, env.consts, order, specs, faces, skips,
                len(order), *_SAMPLES, **kw)
    recs = E.consume_plan_records()["records"]
    assert len(recs) == 1, recs
    return json.loads(json.dumps(plog.jsonable(recs[0]), allow_nan=False))


def _bundle(tmp_path, monkeypatch, rec):
    monkeypatch.setattr(RB, "job_log", lambda: None)
    run = {"seed": 250197, "commits": {}, "flags": {}, "wandb_id": None,
           "run_dir": str(tmp_path)}
    path = RB.write(run, source="test", episode=0, env_index=0, plan=rec,
                    exception={"class": "none", "message": None,
                               "traceback": None})
    with open(path) as fh:
        return path, json.load(fh)


def test_a_fallback_tuple_lands_on_the_record_the_bundle_and_a_replay(
        stub, tmp_path, monkeypatch):
    env = _toy_env()
    wires = _wires(env)
    stub["fail"] = 1
    rec = _measure(env, wires)
    assert "refused" not in rec
    assert rec["compile_option_layout"] == E.measure_compile_layout()
    assert rec["compile_options"] == _FIRST
    assert rec["compile_options_tried"] == [_LIVE, _FIRST]
    assert rec["ref_compile_options"] == _LIVE
    assert rec["compile_fallbacks"] == 1
    # The candidate: the live options, then the first tuple. The reference: the live options.
    assert stub["seen"] == [E._measure_compiler_options(),
                            E.measure_compile_options(_FIRST),
                            E._measure_compiler_options()]

    # A cache hit compiles nothing and names the tuple of the compile that made it.
    stub["seen"].clear()
    again = _measure(env, wires)
    assert stub["seen"] == []
    assert again["compile_options"] == _FIRST
    assert again["compile_options_tried"] == [_LIVE, _FIRST]

    path, bundle = _bundle(tmp_path, monkeypatch, rec)
    assert bundle["compile_options"] == {
        "layout": E.measure_compile_layout(), "candidate": _FIRST,
        "tried": [_LIVE, _FIRST], "reference": _LIVE, "from": "plan record"}
    assert bundle["plan"]["compile_options"] == _FIRST

    # The replay reads the tuple from the bundle and compiles with it alone.
    back = LM.record_of(path)
    tup = LM.record_compile_tuple(back)
    assert list(tup) == _FIRST
    order, plan = LM.record_plan(env, back)
    stub["seen"].clear()
    m = LM.measure(env, _SAMPLES, order, plan, compile_tuple=tup)
    assert stub["seen"][0] == E.measure_compile_options(_FIRST)
    assert m["compile_options"] == ",".join(str(b) for b in _FIRST)
    assert m["refused"] == ""
    assert E.last_measure_compile()["compile_options_tried"] == [_FIRST]


def test_a_plan_no_tuple_compiles_records_every_tuple_it_tried(
        stub, tmp_path, monkeypatch):
    env = _toy_env()
    stub["fail"] = 1 + len(E.MEASURE_COMPILE_TRY_ORDER)
    rec = _measure(env, _wires(env))
    tried = [_LIVE] + [list(t) for t in E.MEASURE_COMPILE_TRY_ORDER]
    assert rec["refused"] == "compile:RuntimeError"
    assert rec["compile_options"] is None
    assert rec["compile_options_tried"] == tried
    assert rec["ref_compile_options"] == _LIVE
    _path, bundle = _bundle(tmp_path, monkeypatch, rec)
    assert bundle["compile_options"]["candidate"] is None
    assert bundle["compile_options"]["tried"] == tried
    with pytest.raises(ValueError, match="names no executable"):
        LM.record_compile_tuple(rec)


def test_a_live_plan_records_the_live_tuple(stub):
    env = _toy_env()
    rec = _measure(env, _wires(env))
    assert rec["compile_options"] == _LIVE
    assert rec["compile_options_tried"] == [_LIVE]
    assert rec["compile_fallbacks"] == 0
    assert list(LM.record_compile_tuple(rec)) == _LIVE


def test_a_record_older_than_the_allowed_list():
    assert LM.record_compile_tuple({"compile_fallbacks": 0}) == (
        E.measure_compile_live_tuple())
    with pytest.raises(ValueError, match="degraded-fusion"):
        LM.record_compile_tuple({"compile_fallbacks": 1})
    with pytest.raises(ValueError, match="not a prefix"):
        LM.record_compile_tuple({"compile_option_layout": ["x=1"],
                                 "compile_options": [1]})
