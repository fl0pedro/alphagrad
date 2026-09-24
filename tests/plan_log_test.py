"""A6 -- THE PLAN LOG: the wire round trip, the record, and REPLAYABILITY.

What is pinned here:

1. THE ROUND TRIP IS EXACT. ``encode_wires`` -> JSON -> ``decode_wires``
   returns the four integer buffers ``_callback`` consumed, bit for bit.
   This is the property 264 archived Pareto points did not have (907c231):
   their vertex column was the 1-based jaxpr vertex id while the replay
   helper documented it as a 0-based action index, so a replay either raised
   ``IndexError`` or silently shifted the order by one and measured a
   garbage Jacobian while reporting a healthy number. The record removes the
   interpretation step -- it stores the wire, not a rendering of it -- and
   NAMES the convention in a field so nobody has to infer it again.

2. A TRUNCATED RECORD IS REFUSED, not silently replayed as a different plan.

3. THE RECORD IS PRODUCED BY A REAL MEASUREMENT and replaying it reproduces
   the reward vector. The plan replayed here is a SKIPPED FACE -- i.e. a
   loser, the exact kind of plan the Pareto front drops and X3 needs.

4. FLAG OFF RECORDS NOTHING. (The trained path's inertness is the
   ALPHAGRAD_EQ_DUMP gate's job; this is the library-level half of it.)
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.pop("ALPHAGRAD_PLAN_LOG", None)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.common import plan_log as plog            # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    COMPRESS_SENTINEL,
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    NUM_REWARDS,
    QUANT_SENTINEL,
    REWARD_INDEX,
    StepAction,
    VertexEliminationEnv,
)

_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)
_Q = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 17.0 + 0.3)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _square(x):
    e = _M @ x
    return _P @ e, _Q @ e


def _make_env():
    closed = jax.make_jaxpr(_square)(_X4)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X4], argnums=(0,), num_envs=0, target_fun=_square,
    )


def _run_episode(env, skip_face_of_vertex=None):
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
        face_skip = jnp.zeros((MAX_FACES,), jnp.int32)
        if skip_face_of_vertex is not None and v == skip_face_of_vertex:
            face_skip = face_skip.at[0].set(1)
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        ).state
    return np.asarray(state.reward)


# --------------------------------------------------------------------------
# 1/2. the wire round trip, in isolation
# --------------------------------------------------------------------------

def _synthetic_wires(n=4, max_rules=3, max_faces=5, face_slots=3, seed=0):
    rng = np.random.default_rng(seed)
    order = np.arange(1, n + 1, dtype=np.int32)
    rules = np.full((n, max_rules, 3), -1, np.int32)
    rules[:, :, 2] = 0
    faces = np.full((n, max_faces, face_slots, 3), -1, np.int32)
    skips = np.zeros((n, max_faces), np.int32)
    # A DIAG, a COMPRESS and a QUANT on the vertex wire...
    rules[0, 0] = (2, 1, 3)
    rules[1, 1] = (COMPRESS_SENTINEL, 0, 0)
    rules[2, 0] = (QUANT_SENTINEL, 4, 0)
    # ...and the same three plus two skips on the face wire.
    faces[0, 1, 0] = (1, 0, 2)
    faces[1, 3, 2] = (COMPRESS_SENTINEL, 2, 0)
    faces[3, 0, 1] = (QUANT_SENTINEL, 1, 0)
    skips[2, 4] = 1
    skips[3, 0] = 1                 # a face carrying BOTH a rule and a skip
    del rng
    return order, rules, faces, skips


def test_wire_roundtrip_is_bit_exact():
    order, rules, faces, skips = _synthetic_wires()
    rec = plog.encode_wires(order, rules, faces, skips,
                            compress_sentinel=COMPRESS_SENTINEL,
                            quant_sentinel=QUANT_SENTINEL)
    # Through real JSON, because that is what the log actually stores.
    rec = json.loads(json.dumps(plog.jsonable(rec), allow_nan=False))
    o2, r2, f2, k2 = plog.decode_wires(rec)
    assert np.array_equal(o2, order), (o2, order)
    assert np.array_equal(r2, rules)
    assert np.array_equal(f2, faces)
    assert np.array_equal(k2, skips)
    assert rec["vertex_convention"] == "jaxpr-vertex-id-1-based"
    assert rec["replayable"] is True


def test_requested_counts_are_read_off_the_wire():
    order, rules, faces, skips = _synthetic_wires()
    rec = plog.encode_wires(order, rules, faces, skips,
                            compress_sentinel=COMPRESS_SENTINEL,
                            quant_sentinel=QUANT_SENTINEL)
    assert rec["requested_vertex"] == {"diag": 1, "compress": 1, "quant": 1,
                                       "other": 0, "total": 3}
    assert rec["requested_face"]["diag"] == 1
    assert rec["requested_face"]["compress"] == 1
    assert rec["requested_face"]["quant"] == 1
    assert rec["requested_face"]["skip"] == 2
    assert rec["requested"]["total"] == 6
    assert rec["requested"]["skip"] == 2
    # 3 faces carry a rule + 2 carry a skip, one of which carries both.
    assert rec["n_live_faces"] == 4


def test_a_truncated_record_is_refused_rather_than_replayed():
    order, rules, faces, skips = _synthetic_wires()
    rec = plog.encode_wires(order, rules, faces, skips,
                            max_faces_recorded=2,
                            compress_sentinel=COMPRESS_SENTINEL,
                            quant_sentinel=QUANT_SENTINEL)
    assert rec["replayable"] is False
    assert rec["faces_truncated"] == 2
    with pytest.raises(ValueError, match="NOT replayable"):
        plog.decode_wires(rec)


def test_non_finite_floats_survive_json():
    rec = {"a": float("nan"), "b": float("inf"), "c": float("-inf"),
           "d": [1.0, float("nan")]}
    txt = json.dumps(plog.jsonable(rec), allow_nan=False)
    back = json.loads(txt)
    assert np.isnan(plog.unjson_float(back["a"]))
    assert plog.unjson_float(back["b"]) == float("inf")
    assert plog.unjson_float(back["c"]) == float("-inf")
    assert np.isnan(plog.unjson_float(back["d"][1]))


# --------------------------------------------------------------------------
# 3/4. the real measurement
# --------------------------------------------------------------------------

def test_flag_off_records_nothing():
    os.environ["ALPHAGRAD_PLAN_LOG"] = "0"
    envmod.consume_plan_records()
    env = _make_env()
    _run_episode(env, skip_face_of_vertex=1)
    out = envmod.consume_plan_records()
    assert out["records"] == [], out["records"]
    assert out["dropped"] == 0


def test_terminal_plan_is_recorded_and_replays_to_the_same_reward():
    os.environ["ALPHAGRAD_PLAN_LOG"] = "1"
    try:
        envmod.consume_plan_records()
        env = _make_env()
        reward = _run_episode(env, skip_face_of_vertex=1)
        recs = envmod.consume_plan_records()["records"]
        # ONE record per TERMINAL plan -- the non-terminal steps of the same
        # episode must not appear.
        assert len(recs) == 1, [r["order"] for r in recs]
        rec = recs[0]
        assert rec["schema"] == plog.SCHEMA
        assert len(rec["rewards"]) == NUM_REWARDS == 12
        assert rec["requested"]["skip"] == 1
        assert rec["requested"]["total"] == 0     # a skip is not a rule
        # The recorded slots ARE the emitted reward vector.
        assert np.allclose(np.asarray(rec["rewards"], np.float64),
                           reward.astype(np.float64), rtol=0, atol=0,
                           equal_nan=True)

        # ---- REPLAY. Straight back into the callback, no interpretation.
        rec = json.loads(json.dumps(plog.jsonable(rec), allow_nan=False))
        order, rules, faces, skips = plog.decode_wires(rec)
        _t, _e, replay = envmod._callback(
            env.config, env.args, env.consts, order, rules, faces, skips,
            len(order))
        replay = np.asarray(replay)
        # Every channel except the wall-clock one, which is a timing and is
        # not expected to repeat (it is not measured in this config anyway).
        lat = REWARD_INDEX["latency_ns"]
        keep = [i for i in range(NUM_REWARDS) if i != lat]
        assert np.array_equal(replay[keep].astype(np.float64),
                              reward[keep].astype(np.float64)), (
            f"replay {replay} != recorded {reward}")
    finally:
        os.environ["ALPHAGRAD_PLAN_LOG"] = "0"
        envmod.consume_plan_records()


def test_record_carries_the_per_kind_fields_and_no_census():
    os.environ["ALPHAGRAD_PLAN_LOG"] = "1"
    try:
        envmod.consume_plan_records()
        env = _make_env()
        _run_episode(env, skip_face_of_vertex=1)
        rec = envmod.consume_plan_records()["records"][0]
        for bucket in ("applied", "skipped", "idempotent_noop", "repaired"):
            assert bucket in rec, bucket
            for kind in ("diag", "compress", "quant"):
                assert isinstance(rec[bucket][kind], int)
        assert isinstance(rec["counts_from_trace"], bool)
        # The gradient-coverage census and the guard's ``sentinelled`` flag
        # left the record with the guard (owner ruling 2026-09-03, ticket dsnn-3qm.15). Slot 7 keeps its NAME so
        # archived logs still decode against the same table.
        assert "coverage" not in rec
        assert "sentinelled" not in rec
        # The record names its own slots: slot 6 is spelled "quality" in
        # REWARD_NAMES (``cosine_sim`` is the back-compat alias in
        # REWARD_INDEX, not the canonical name) and holds whichever metric
        # env.quality_metric() selected.
        assert rec["reward_names"] == list(envmod.REWARD_NAMES)
        assert rec["reward_names"][6] == "quality"
        assert rec["reward_names"][7] == "grad_coverage"
        assert rec["reward_names"][10] == "sparsity"
    finally:
        os.environ["ALPHAGRAD_PLAN_LOG"] = "0"
        envmod.consume_plan_records()


def test_the_jsonl_is_append_only(tmp_path):
    p = str(tmp_path / "plan_log.jsonl")
    assert plog.append_records(p, [{"a": 1}]) == 1
    assert plog.append_records(p, [{"a": 2}, {"a": 3}]) == 2
    recs = plog.read_records(p)
    assert [r["a"] for r in recs] == [1, 2, 3]


# --------------------------------------------------------------------------
# 5. the measure toolchain telemetry (finding 03, ticket dsnn-3qm.21)
# --------------------------------------------------------------------------

def test_record_and_drain_carry_the_measure_toolchain_telemetry():
    os.environ["ALPHAGRAD_PLAN_LOG"] = "1"
    try:
        envmod.consume_plan_records()
        env = _make_env()
        _run_episode(env, skip_face_of_vertex=1)
        out = envmod.consume_plan_records()
        rec = out["records"][0]
        # Per plan: the degraded-fusion compiles taken WHILE THIS PLAN was
        # measured, the process total, and whether this node's toolchain
        # passed the gate. Nothing degraded here, so 0 / 0 / True.
        assert rec["compile_fallbacks"] == 0
        assert isinstance(rec["compile_fallbacks"], int)
        assert rec["compile_fallbacks_total"] == (
            envmod._MEASURE_COMPILE_FALLBACKS["n"])
        assert rec["toolchain_ok"] is True
        # The drain carries the live counter from THIS process -- the one
        # that compiles -- so the trainer never reads its own zero.
        for k in ("compile_fallbacks", "compile_fallbacks_total",
                  "toolchain_ok", "toolchain_host"):
            assert k in out, k
        assert out["toolchain_ok"] is True
        # A fallback taken between two drains is reported ONCE, as a delta.
        envmod._MEASURE_COMPILE_FALLBACKS["n"] += 1
        try:
            assert envmod.consume_plan_records()["compile_fallbacks"] == 1
            assert envmod.consume_plan_records()["compile_fallbacks"] == 0
        finally:
            envmod._MEASURE_COMPILE_FALLBACKS["n"] -= 1
    finally:
        os.environ["ALPHAGRAD_PLAN_LOG"] = "0"
        envmod.consume_plan_records()


def test_gate_mode_and_plan_log_leave_the_reward_bit_identical(monkeypatch):
    """The library-level half of the flag-off gate: neither the toolchain
    gate's mode nor the plan-log flag may move a reward slot. (The
    trajectory-level half is the ALPHAGRAD_EQ_DUMP run in ppo.py.)"""
    rewards = []
    for mode, log in (("abort", "0"), ("off", "0"), ("warn", "0"),
                      ("abort", "1")):
        monkeypatch.setenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", mode)
        monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", log)
        monkeypatch.setattr(envmod, "_MEASURE_TOOLCHAIN", dict(
            envmod._MEASURE_TOOLCHAIN, checked=False, ok=True))
        envmod.consume_plan_records()
        rewards.append(_run_episode(_make_env(), skip_face_of_vertex=1))
    envmod.consume_plan_records()
    lat = REWARD_INDEX["latency_ns"]
    keep = [i for i in range(NUM_REWARDS) if i != lat]
    for r in rewards[1:]:
        assert np.array_equal(r[keep].astype(np.float64),
                              rewards[0][keep].astype(np.float64)), (
            r, rewards[0])


# --------------------------------------------------------------------------
# A REFUSED PLAN IS A RECORD (canary job 65319, 2026-09-13)
#
# The terminal counter fires at the top of `env._callback` and the record is
# written at the bottom. Four `return`s in between -- the op-count cap, a
# missing target function, an untraceable graph and an OOM -- used to drop
# the record while the counter had already fired. The trainer then printed
# `pool_terminals=16 ... wrote=0 total=0` for every episode of a 250-episode
# campaign arm, and the whole run left no replayable evidence.
# --------------------------------------------------------------------------

def test_a_plan_refused_by_the_op_count_cap_is_still_recorded(monkeypatch):
    """The muls cap is the reachable refusal: set it to 0 and every plan
    trips it. The record must exist, carry the reason, and be marked refused
    and sentinelled -- its rewards are a sentinel, not a measurement. Its wire
    is complete, so it stays replayable (dsnn-sf5)."""
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    # -1, so any counted op count trips it. This file turns the count pass
    # OFF at import (line 31), so the test has to turn it back on -- which
    # works because `env.skip_count_ops()` reads the variable per call.
    monkeypatch.setenv("ALPHAGRAD_MULS_SENTINEL_CAP", "-1")
    monkeypatch.setattr(envmod, "_MEASURE_TIMEOUT_S", [300.0])
    monkeypatch.setenv("ALPHAGRAD_SKIP_COUNT_OPS", "0")
    envmod.consume_plan_records()
    env = _make_env()
    _run_episode(env, skip_face_of_vertex=1)
    out = envmod.consume_plan_records()
    assert out["terminals"] == 1
    assert len(out["records"]) == out["terminals"], (
        f"{out['terminals']} terminal(s) counted, {len(out['records'])} "
        f"record(s) written: a refused plan was dropped")
    rec = out["records"][0]
    assert rec["refused"] == "muls-cap"
    assert rec["sentinelled"] is True
    assert rec["replayable"] is True
    assert len(rec["rewards"]) == NUM_REWARDS
    # the wire is still there, so the refusal can be attributed to a plan
    assert rec["order"] and "faces" in rec


def test_every_terminal_is_a_record_under_the_normal_path(monkeypatch):
    """The control: with no cap, terminals and records still agree. This is
    the invariant the refusal paths broke, stated once for both."""
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.delenv("ALPHAGRAD_MULS_SENTINEL_CAP", raising=False)
    envmod.consume_plan_records()
    env = _make_env()
    _run_episode(env, skip_face_of_vertex=1)
    out = envmod.consume_plan_records()
    assert out["terminals"] == len(out["records"]) == 1
    assert out["records"][0].get("refused") is None
    assert out["records"][0].get("sentinelled") is not True


def test_a_plan_that_raises_is_still_recorded_and_the_error_still_reaches_the_caller(monkeypatch):
    """A counted terminal is a record, whatever killed the measurement.

    The observation-delta overflow is the reachable raise: it fires inside
    the tokenizer, long after the terminal counter and long before the
    record. Job 65339 lost three of four plans to it. The record must
    appear, name the exception, and the exception must still propagate --
    the raise is the apparatus telling the operator to raise the budget.
    """
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    envmod.consume_plan_records()
    # MAX_DELTA_TOKENS sizes static wire shapes and is read once at import,
    # so it cannot be shrunk per test. The raise is injected instead at the
    # tokenizer's length hook, which `_callback_measured` reaches AFTER it
    # has counted the terminal (the counter sits at the top of the call, the
    # hook inside the tokenization block) -- the same point in the call at
    # which the real overflow raised in job 65339. Only the terminal call
    # raises, so the episode reaches it.
    _orig = envmod._record_token_length

    def _overflow_at_the_terminal(raw_len):
        _orig(raw_len)
        if int(envmod._PLAN_LOG_TERMINALS[0]) > 0:
            raise ValueError("[alphagrad.approx.env] token DELTA truncated: "
                             "injected overflow at the tokenizer")

    monkeypatch.setattr(envmod, "_record_token_length",
                        _overflow_at_the_terminal)
    env = _make_env()
    # The raise crosses `jax.io_callback`, which re-raises it as a
    # `JaxRuntimeError` whose message quotes the original. Match the text.
    with pytest.raises(Exception, match="token DELTA truncated"):
        _run_episode(env, skip_face_of_vertex=1)
    out = envmod.consume_plan_records()
    assert out["terminals"] == 1, (
        f"the injected raise fired {out['terminals']} terminal(s) in: the "
        "hook must run after the counter, once, at the terminal step")
    assert len(out["records"]) == out["terminals"], (
        f"{out['terminals']} terminal(s) counted, {len(out['records'])} "
        f"record(s): a crashed plan left no trace")
    rec = out["records"][-1]
    assert rec["refused"].startswith("raised:ValueError")
    assert rec["sentinelled"] is True and rec["replayable"] is True
