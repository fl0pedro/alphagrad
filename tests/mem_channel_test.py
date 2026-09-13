"""THE MEMORY CHANNEL (ticket dsnn-3qm.49, ruling .29).

Reward slot 5 (``peak_memory``, stored negated) holds the XLA static temp
bytes of the plan's own timed executable; the runtime watermark is recorded
beside it per measurement and drained with the plan records. Pinned here,
on a toy scalar-loss target (no TLM, no data generator, CPU):

1. A SKIP-EVERYTHING plan reports a temp ratio FAR BELOW 1 against rev-exact
   (TLM reference from finding 49 / ticket .24: 0.0406). Dead-code
   elimination removes the gradient graph, so the temp goes to (near) zero.

2. ``watermark - temp`` is a near-constant across plans (finding 41: 2.104 MB
   on TLM, R^2 = 1.000000): its spread is small relative to the temp spread.
   THE INSTRUMENT IS PINNED to the spec-native primitives
   (ALPHAGRAD_DIRECT_MEASURE=1), under which a CPU backend has no allocator
   statistics and the "watermark" is the in-place static substitution (temp
   + output + argument bytes): this half tests the plumbing, not the
   allocator. It has to be pinned: with the ResourceMonitor instrument a CPU
   node reports a FLAT device peak of 256-260 B for every plan (job 63632:
   temps 512 / 0 / 33024 B against watermarks 260 / 256 / 260 B), i.e. the
   sigma = 0 of finding 49 in miniature, and the gap is not a constant. The
   GPU landing test of the real watermark is ticket .24's.

3. FLAG OFF (``--mem-channel watermark``) is the pre-.49 channel: slot 5 is
   the recorded watermark, and every other slot is identical under both
   channels. (The trajectory-level half is the ALPHAGRAD_EQ_DUMP run.)

4. The parity drain rides ``consume_plan_records`` and every measured plan
   leaves a record (``records + dropped == measured``); a missing record is a
   ``MemChannelFault``, which rides the toolchain-fault escalation.

5. The plan-log record carries ``mem_channel`` / ``mem_temp_bytes`` /
   ``mem_watermark_bytes``. (Until 2026-09-04 this item also pinned that the
   quality gate refused to clamp a static temp with a watermark floor; the
   gate was deleted, ticket dsnn-3qm.9.)

This module measures under the ABSOLUTE cost form (the default when
ALPHAGRAD_COST_FORM is absent): slot 5 IS the temp. The paired-log form is
tests/paired_log_reward_test.py's.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
# THE MEASUREMENT CONFIGURATION IS DECLARED IN A FIXTURE, NOT HERE. See
# ``_the_configuration_this_module_measures_under`` below; this comment records
# why, because the old form cost the campaign three red tests.
#
# "One process per module (finding 47), so this module owns its configuration"
# used to stand here above four ``os.environ[...]`` assignments. It is not true
# of the suite: ``pytest tests/`` IMPORTS EVERY TEST MODULE during collection
# and only then runs the first test, so a module-level assignment is a
# COLLECTION-TIME mutation of one shared process and the LAST module collected
# wins. This module popped ALPHAGRAD_COST_FORM at its own import; four files
# later in collection order ``tests/paired_log_reward_test.py`` (m < p) set it
# to "paired-log" at ITS import. Every test below then ran under the PAIRED-LOG
# cost form while this module's docstring claims the ABSOLUTE one and reads slot
# 5 as the temp -- so slot 5 held a log-difference against rev-exact instead.
# Run alone, nothing re-set it afterwards and the module passed.

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    NUM_REWARDS,
    REWARD_INDEX,
    MemChannelFault,
    StepAction,
    VertexEliminationEnv,
)

_MEM = REWARD_INDEX["peak_memory"]
_LAT = REWARD_INDEX["latency_ns"]

# A 2-layer, 64-wide scalar loss: big enough that reverse mode carries real
# temporaries (512 B on this toy) and a skip-everything plan visibly drops
# them, small enough to compile in well under a minute on a CPU.
_N = 64
_rng = np.random.default_rng(0)
_W1 = jnp.asarray(_rng.standard_normal((_N, _N), dtype=np.float32) / 8.0)
_W2 = jnp.asarray(_rng.standard_normal((_N, _N), dtype=np.float32) / 8.0)
_X = jnp.asarray(np.linspace(-1.0, 1.0, _N, dtype=np.float32))


def _toy(x):
    h = jnp.tanh(_W1 @ x)
    y = jnp.tanh(_W2 @ h)
    return jnp.sum(y * y)


def _make_env():
    closed = jax.make_jaxpr(_toy)(_X)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X], argnums=(0,), num_envs=0, target_fun=_toy,
    )


def _run_plan(env, order, skip_everything=False):
    """One episode: ``order`` with no rules; every face of every vertex
    SKIPPED when ``skip_everything``. Returns the terminal reward vector."""
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    for v in order:
        face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
        face_skip = (jnp.ones((MAX_FACES,), jnp.int32) if skip_everything
                     else jnp.zeros((MAX_FACES,), jnp.int32))
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        ).state
    return np.asarray(state.reward)


def _rev_order(env):
    return sorted(int(x) for x in np.asarray(env.valid_vertices))[::-1]


def _terminal_records(mp):
    return [r for r in mp["records"] if r["terminal"]]


@pytest.fixture(autouse=True)
def _the_configuration_this_module_measures_under(monkeypatch):
    """The process-global knobs this module's claims are made under, in force
    for the duration of each test and restored afterwards.

    * ALPHAGRAD_QUALITY_METRIC=none -- the toy has no data generator and the
      memory channel is what is under test.
    * ALPHAGRAD_DIRECT_MEASURE=1 -- the one instrument this module makes claims
      about (see the module docstring: under the ResourceMonitor instrument a
      CPU node reports a flat 256-260 B device peak for every plan).
    * no ALPHAGRAD_COST_FORM -- the ABSOLUTE form, under which slot 5 IS the
      temp. The paired-log form's slot 5 is a log-difference and is
      tests/paired_log_reward_test.py's subject.
    * no ALPHAGRAD_MEM_CHANNEL / ALPHAGRAD_PLAN_LOG -- the default channel and
      no plan log; the tests that want them set them themselves.

    autouse so it also covers tests that take no fixture argument, and
    ``monkeypatch`` so the values do not outlive the module the way the
    module-scope assignments they replaced did.
    """
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.delenv("ALPHAGRAD_PLAN_LOG", raising=False)
    monkeypatch.delenv("ALPHAGRAD_MEM_CHANNEL", raising=False)
    monkeypatch.delenv("ALPHAGRAD_COST_FORM", raising=False)


@pytest.fixture
def channel(monkeypatch):
    def _set(name):
        monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", name)
        envmod.consume_mem_parity()
    return _set


# --------------------------------------------------------------------------
# 1. skip-everything reads a temp ratio far below 1 against rev-exact
# --------------------------------------------------------------------------

def test_skip_everything_temp_ratio_is_far_below_one(channel):
    channel("temp")
    env = _make_env()
    order = _rev_order(env)
    r_exact = _run_plan(env, order)
    r_skip = _run_plan(env, order, skip_everything=True)
    temp_exact = -float(r_exact[_MEM])
    temp_skip = -float(r_skip[_MEM])
    assert temp_exact > 0.0, "rev-exact carries no temporaries on this toy"
    ratio = temp_skip / temp_exact
    # Measured on this toy: rev-exact temp 512 B, skip-everything 0 B
    # (ratio 0.0). TLM reference 0.0406. The bound is loose on purpose: the
    # claim is "far below 1", the value is printed for the record.
    print(f"[mem-channel] temp rev-exact={temp_exact:.0f} B "
          f"skip-everything={temp_skip:.0f} B ratio={ratio:.4f}")
    assert ratio < 0.5, (temp_exact, temp_skip, ratio)
    # The channel IS the parity record's temp, for both plans.
    recs = _terminal_records(envmod.consume_mem_parity())
    assert [r["static_temp_bytes"] for r in recs] == [temp_exact, temp_skip]
    assert all(r["channel"] == "temp" for r in recs)


# --------------------------------------------------------------------------
# 2. watermark - temp is a near-constant across plans
# --------------------------------------------------------------------------

def test_watermark_minus_temp_is_near_constant_across_plans(channel):
    channel("temp")
    env = _make_env()
    rev = _rev_order(env)
    _run_plan(env, rev)
    _run_plan(env, rev, skip_everything=True)
    _run_plan(env, rev[::-1])                  # forward mode: much more temp
    recs = _terminal_records(envmod.consume_mem_parity())
    assert len(recs) == 3
    # Name the instrument before claiming anything about the gap.
    assert [r["peak_source"] for r in recs] == ["static_fallback"] * 3, recs
    temps = np.asarray([r["static_temp_bytes"] for r in recs], np.float64)
    marks = np.asarray([r["runtime_peak_bytes"] for r in recs], np.float64)
    assert np.all(np.isfinite(temps)) and np.all(np.isfinite(marks))
    gaps = marks - temps
    temp_spread = float(temps.max() - temps.min())
    gap_spread = float(gaps.max() - gaps.min())
    print(f"[mem-channel] temps={temps.tolist()} watermarks={marks.tolist()} "
          f"gap={gaps.tolist()} spread temp={temp_spread:.0f} "
          f"gap={gap_spread:.0f}")
    assert temp_spread > 0.0
    # Finding 41: 72 B of gap spread over 58,309x of temp on TLM. Under the
    # static fallback the watermark IS temp + output + argument bytes
    # (`_memory_analysis_bytes`), so the gap is exactly the output and
    # argument bytes the record carries, and it moves only when XLA folds an
    # output to a constant (the skip-everything plan). The old bound (5% of
    # the temp spread) assumed forward mode densified a 33 kB temp on this
    # toy; the lazy engine (dsnn-3qm.28.2) leaves 768 B, which made the
    # bound tighter than the instrument it was checking. Name the identity.
    outs = np.asarray([r["static_output_bytes"] for r in recs], np.float64)
    argb = np.asarray([r["static_argument_bytes"] for r in recs], np.float64)
    assert np.array_equal(gaps, outs + argb), (gaps, outs, argb)
    assert gaps[0] == gaps[2], gaps            # rev and forward: same buffers
    assert gaps[1] <= gaps[0], gaps            # the folded plan only shrinks
    summary = envmod.mem_parity_summary(recs)
    assert summary["n"] == 3 and summary["n_paired"] == 3
    assert summary["gap_max_bytes"] == float(gaps.max())
    assert summary["gap_min_bytes"] == float(gaps.min())
    assert summary["gap_mean_bytes"] == pytest.approx(float(gaps.mean()))
    assert summary["static_fallbacks"] == 3          # CPU: every reading


# --------------------------------------------------------------------------
# 3. flag off = the pre-.49 channel, every other slot identical
# --------------------------------------------------------------------------

def test_watermark_channel_is_the_recorded_watermark_and_nothing_else_moves(
        channel):
    env = _make_env()
    rev = _rev_order(env)
    out = {}
    for name in ("watermark", "temp"):
        channel(name)
        r = _run_plan(env, rev)
        rec = _terminal_records(envmod.consume_mem_parity())[-1]
        out[name] = (r, rec)
    r_w, rec_w = out["watermark"]
    r_t, rec_t = out["temp"]
    # Slot 5 under each channel IS that channel's number in the record.
    assert -float(r_w[_MEM]) == rec_w["runtime_peak_bytes"]
    assert -float(r_t[_MEM]) == rec_t["static_temp_bytes"]
    assert rec_w["channel"] == "watermark" and rec_t["channel"] == "temp"
    # The same executable was analysed both times.
    assert rec_w["static_temp_bytes"] == rec_t["static_temp_bytes"]
    assert rec_w["runtime_peak_bytes"] == rec_t["runtime_peak_bytes"]
    # The two channels differ on this plan (under the pinned instrument the
    # substituted watermark carries the output + argument bytes the temp
    # does not), and nothing else moves.
    assert rec_w["peak_source"] == "static_fallback"
    assert -float(r_w[_MEM]) > -float(r_t[_MEM])
    keep = [i for i in range(NUM_REWARDS) if i not in (_MEM, _LAT)]
    assert np.array_equal(r_w[keep].astype(np.float64),
                          r_t[keep].astype(np.float64)), (r_w, r_t)


def test_mem_channel_reader_rejects_a_hand_edit(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_MEM_CHANNEL", raising=False)
    assert envmod.mem_channel() == "temp"
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "Watermark ")
    assert envmod.mem_channel() == "watermark"
    monkeypatch.setenv("ALPHAGRAD_MEM_CHANNEL", "static")
    with pytest.raises(ValueError, match="mem-channel"):
        envmod.mem_channel()


# --------------------------------------------------------------------------
# 4. the drain and the completeness assertion
# --------------------------------------------------------------------------

def test_parity_rides_the_plan_record_drain_and_is_complete(channel):
    channel("temp")
    envmod.consume_plan_records()
    env = _make_env()
    rev = _rev_order(env)
    _run_plan(env, rev)
    _run_plan(env, rev, skip_everything=True)
    out = envmod.consume_plan_records()
    mp = out["mem_parity"]
    # Every step of both episodes measured (terminal_rewards_only is off
    # on this env), and every measurement left a record.
    assert mp["measured"] == 2 * len(rev)
    assert len(mp["records"]) == mp["measured"]
    assert mp["dropped"] == 0
    envmod.check_mem_parity_complete(mp, "test")          # does not raise
    assert sum(r["terminal"] for r in mp["records"]) == 2
    # Drained: a second drain is empty and still complete.
    mp2 = envmod.consume_plan_records()["mem_parity"]
    assert mp2 == {"records": [], "measured": 0, "dropped": 0}
    envmod.check_mem_parity_complete(mp2, "test")


def test_a_measured_plan_without_a_record_is_a_fault():
    with pytest.raises(MemChannelFault, match="1 plan"):
        envmod.check_mem_parity_complete(
            {"records": [{}], "measured": 2, "dropped": 0}, "actor 7")
    # Records refused at the cap still count as accounted for.
    envmod.check_mem_parity_complete(
        {"records": [{}], "measured": 2, "dropped": 1}, "actor 7")
    # The fault rides the toolchain-fault escalation.
    assert issubclass(MemChannelFault, envmod.MeasureToolchainFault)


# --------------------------------------------------------------------------
# 5. the plan-log record
# --------------------------------------------------------------------------

def test_plan_log_record_carries_both_memory_numbers(channel, monkeypatch):
    channel("temp")
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    envmod.consume_plan_records()
    env = _make_env()
    r = _run_plan(env, _rev_order(env))
    out = envmod.consume_plan_records()
    rec = out["records"][0]
    assert rec["mem_channel"] == "temp"
    assert rec["mem_temp_bytes"] == -float(r[_MEM])
    assert rec["mem_peak_source"] == "static_fallback"  # the pinned instrument
    assert rec["mem_watermark_bytes"] > rec["mem_temp_bytes"]
    assert rec["rewards"][_MEM] == float(r[_MEM])
    # ...and the drain's own parity record agrees with the plan record.
    term = _terminal_records(out["mem_parity"])
    assert len(term) == 1
    assert term[0]["static_temp_bytes"] == rec["mem_temp_bytes"]
    assert term[0]["runtime_peak_bytes"] == rec["mem_watermark_bytes"]
    # Absolute form: no reference was measured, and the record says so.
    assert rec["cost_form"] == "absolute"
    assert rec["ref_temp_bytes"] is None
    assert out["paired_ref"] == {"records": [], "dropped": 0}
