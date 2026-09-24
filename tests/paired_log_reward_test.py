"""THE PAIRED LOG-DIFFERENCE COST CHANNELS (ticket dsnn-3qm.9).

Under ``--cost-form paired-log`` reward slots 2 (latency_ns) and 5
(peak_memory) carry ``-(log cost(candidate) - log cost(reference))``, with
the reference -- jax.grad of the target (dsnn-xta; the graphax rev-exact
until 2026-09-24) -- measured in the SAME callback, interleaved with the
candidate, through the same executable path and the same instrument. Pinned
here on the 64-wide toy scalar loss of tests/mem_channel_test.py (no TLM, no
data generator, CPU, the spec-native instrument ALPHAGRAD_DIRECT_MEASURE=1):

1. THE REFERENCE SCORES EXACTLY 0 on the memory channel (the reference
   program measured as the candidate: same program, same static temp) and
   within the drift floor on latency (two back-to-back timings of one
   program). With the per-rep timer stubbed to a constant both channels are
   exactly 0.
2. A CHEAPER PLAN scores a NEGATIVE Delta (a positive slot): the
   skip-everything plan, whose temp is exactly 0 and takes the one-byte
   floor; a COSTLIER plan (forward order) scores a positive Delta.
3. THE REFERENCE IS RE-MEASURED EVERY EPISODE: one `_campaign_measure_cost`
   call per terminal callback, none for non-terminal steps.
4. Non-terminal steps carry 0.0 in both slots under this form.
5. The reference rides the drain (``consume_plan_records()["paired_ref"]``)
   and the plan-log record, in positive units.
6. FLAG OFF (``absolute``, the reader's default) is the measured number,
   negated -- the library-level half of the flag-off bit-identity gate; the
   trajectory-level half is the ALPHAGRAD_EQ_DUMP run.
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
# THE MEASUREMENT CONFIGURATION IS DECLARED IN A FIXTURE, NOT HERE -- see
# ``_the_configuration_this_module_measures_under`` below.
#
# It used to be five ``os.environ[...]`` statements at module scope, and
# ALPHAGRAD_COST_FORM="paired-log" among them was a process-wide mutation
# performed AT COLLECTION TIME: ``pytest tests/`` imports every test module
# before it runs the first test, so this line put the whole run into the
# paired-log cost form. ``tests/mem_channel_test.py`` (collected four files
# earlier, m < p) and ``tests/test_all_cost_channels.py`` both measure under
# the ABSOLUTE form and both read slot 5 directly; they went red for this, and
# passed the moment they were run without this module in the same process.

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.common import compile_cache as _cc        # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    NUM_REWARDS,
    REWARD_INDEX,
    StepAction,
    VertexEliminationEnv,
)

_MEM = REWARD_INDEX["peak_memory"]
_LAT = REWARD_INDEX["latency_ns"]
_FLOOR = envmod._MEM_LOG_FLOOR_BYTES

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


def _run_plan(env, order, skip_everything=False, stop_after=None):
    """One episode: ``order`` with no rules; every face of every vertex
    SKIPPED when ``skip_everything``. Returns the terminal reward vector, or
    the reward after ``stop_after`` steps when given."""
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    for k, v in enumerate(order):
        face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
        face_skip = (jnp.ones((MAX_FACES,), jnp.int32) if skip_everything
                     else jnp.zeros((MAX_FACES,), jnp.int32))
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        ).state
        if stop_after is not None and k + 1 == stop_after:
            return np.asarray(state.reward)
    return np.asarray(state.reward)


def _rev_order(env):
    return sorted(int(x) for x in np.asarray(env.valid_vertices))[::-1]


def _the_reference_as_the_candidate(monkeypatch, env):
    """Every ``approx:`` compile of the callback becomes the reference
    program (jax.grad of the target), so the callback scores the reference
    against itself. The process-local executable memo is keyed on the plan,
    so it is cleared on both sides of the substitution."""
    real = _cc.cached_compile
    _cc._LOCAL_CACHE.clear()

    def _substitute(key, fn):
        if bytes(key).startswith(b"approx:"):
            fn = lambda: envmod._compile_measure(                # noqa: E731
                jax.jit(envmod.reference_program(env.config),
                        keep_unused=True).lower(*env.args))
        return real(key, fn)
    monkeypatch.setattr(_cc, "cached_compile", _substitute)


@pytest.fixture(autouse=True)
def _no_substituted_candidate_outlives_its_test():
    yield
    _cc._LOCAL_CACHE.clear()


@pytest.fixture(autouse=True)
def _the_configuration_this_module_measures_under(monkeypatch):
    """THE PAIRED-LOG COST FORM, in force for this module's tests only.

    ``ALPHAGRAD_COST_FORM`` is re-read on every measurement, so a fixture is
    enough -- and a module-scope assignment is actively wrong: it puts the whole
    pytest process into the paired-log form from COLLECTION onwards, and the
    modules that measure the ABSOLUTE form (tests/mem_channel_test.py,
    tests/test_all_cost_channels.py) then read slot 5 -- which they document as
    the temp -- as a log-difference against rev-exact.
    """
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    # The floor policy this module's numbers were computed under. The
    # reference floor has its own tests at the bottom of the file.
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    monkeypatch.delenv("ALPHAGRAD_PLAN_LOG", raising=False)
    monkeypatch.delenv("ALPHAGRAD_MEM_CHANNEL", raising=False)


@pytest.fixture(autouse=True)
def _drain():
    envmod.consume_plan_records()
    yield
    envmod.consume_plan_records()


# --------------------------------------------------------------------------
# 0. the pure function and the reader
# --------------------------------------------------------------------------

def test_paired_log_costs_pure():
    # identity: exactly 0 on both channels
    assert envmod.paired_log_costs(1.5e5, 512.0, 1.5e5, 512.0) == (0.0, 0.0, 0)
    # cheaper on both: negative Delta
    d_lat, d_mem, n = envmod.paired_log_costs(7.5e4, 256.0, 1.5e5, 512.0)
    assert d_lat == pytest.approx(math.log(0.5))
    assert d_mem == pytest.approx(math.log(0.5))
    assert n == 0
    # latency 0.0 = NOT MEASURED passes through as 0.0; memory still pairs
    d_lat, d_mem, n = envmod.paired_log_costs(0.0, 1024.0, 0.0, 512.0)
    assert d_lat == 0.0 and d_mem == pytest.approx(math.log(2.0))
    # THE FLOOR: a zero temp reads as one byte, once, and is counted
    d_lat, d_mem, n = envmod.paired_log_costs(1.0e5, 0.0, 1.0e5, 512.0)
    assert d_mem == pytest.approx(-math.log(512.0 / _FLOOR))
    assert n == 1
    assert _FLOOR == 1.0
    # both sides at zero: exactly 0, counted twice
    assert envmod.paired_log_costs(1.0e5, 0.0, 1.0e5, 0.0) == (0.0, 0.0, 2)


def test_cost_form_reader(monkeypatch):
    assert envmod.cost_form() == "paired-log"          # this module's setting
    monkeypatch.delenv("ALPHAGRAD_COST_FORM", raising=False)
    assert envmod.cost_form() == "absolute"            # absent = absolute
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", " Paired-Log ")
    assert envmod.cost_form() == "paired-log"
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "ratio")
    with pytest.raises(ValueError, match="cost-form"):
        envmod.cost_form()


# --------------------------------------------------------------------------
# 1. the reference scores 0
# --------------------------------------------------------------------------

def test_the_reference_scores_exactly_zero_on_memory_and_inside_drift_on_latency(
        monkeypatch):
    env = _make_env()
    _the_reference_as_the_candidate(monkeypatch, env)
    r = _run_plan(env, _rev_order(env))
    recs = envmod.consume_plan_records()["paired_ref"]["records"]
    assert len(recs) == 1
    rec = recs[0]
    # Same program on both sides: the static temp is the same number.
    assert rec["temp_bytes"] == rec["candidate_memory_bytes"] > 0.0
    assert float(r[_MEM]) == 0.0
    # Two back-to-back timings of one program: a DRIFT FLOOR sample. A
    # shared CPU box moves well under 2x between adjacent windows.
    d_lat = -float(r[_LAT])
    print(f"[paired-log] the reference vs itself: Delta_lat={d_lat:+.4f} "
          f"(ratio {math.exp(d_lat):.3f}); temp={rec['temp_bytes']:.0f} B "
          f"lat={rec['latency_ns']/1e3:.1f} us")
    assert abs(d_lat) < math.log(2.0)
    assert rec["latency_ns"] > 0.0 and rec["mem_floored"] == 0
    assert rec["reference"] == "jax.grad"


def test_the_reference_scores_exactly_zero_with_a_deterministic_instrument(
        monkeypatch):
    """With the per-rep timer stubbed to a constant, the pair is exact on
    BOTH channels: the reward form has no offset of its own."""
    real = envmod._time_one_rep

    def _const_rep(ex, eval_args, devices, inner):
        _l, _p, _s, out = real(ex, eval_args, devices, inner)
        return 123_456.0, _p, _s, out

    monkeypatch.setattr(envmod, "_time_one_rep", _const_rep)
    env = _make_env()
    _the_reference_as_the_candidate(monkeypatch, env)
    r = _run_plan(env, _rev_order(env))
    assert float(r[_LAT]) == 0.0
    assert float(r[_MEM]) == 0.0


# --------------------------------------------------------------------------
# 2. cheaper scores negative Delta, costlier scores positive Delta
# --------------------------------------------------------------------------

def test_cheaper_plan_scores_negative_delta_and_takes_the_floor():
    env = _make_env()
    rev = _rev_order(env)
    r_skip = _run_plan(env, rev, skip_everything=True)
    rec = envmod.consume_plan_records()["paired_ref"]["records"][-1]
    temp_ref = rec["temp_bytes"]
    temp_skip = rec["candidate_memory_bytes"]
    # Measured on this toy (job 63632): rev-exact 512 B, skip-everything 0 B.
    assert temp_ref > 0.0
    assert temp_skip == 0.0
    d_mem = math.log(max(temp_skip, _FLOOR)) - math.log(temp_ref)
    assert -float(r_skip[_MEM]) == pytest.approx(d_mem)
    assert float(r_skip[_MEM]) > 0.0                    # cheaper -> above 0
    assert -float(r_skip[_MEM]) == pytest.approx(-math.log(temp_ref))
    assert rec["mem_floored"] == 1
    print(f"[paired-log] skip-everything: Delta_mem={d_mem:+.3f} "
          f"(temp {temp_skip:.0f} -> floor {_FLOOR:.0f} B vs ref "
          f"{temp_ref:.0f} B) Delta_lat={-float(r_skip[_LAT]):+.3f}")


def test_costlier_plan_scores_positive_delta():
    env = _make_env()
    rev = _rev_order(env)
    r_fwd = _run_plan(env, rev[::-1])                   # forward mode
    rec = envmod.consume_plan_records()["paired_ref"]["records"][-1]
    # The reference is jax.grad of the target REGARDLESS of the candidate's
    # order.
    assert rec["reference"] == "jax.grad"
    assert rec["candidate_memory_bytes"] > rec["temp_bytes"]
    assert float(r_fwd[_MEM]) < 0.0                     # costlier -> below 0
    assert -float(r_fwd[_MEM]) == pytest.approx(
        math.log(rec["candidate_memory_bytes"]) - math.log(rec["temp_bytes"]))
    assert rec["mem_floored"] == 0


# --------------------------------------------------------------------------
# 3./4. one reference per terminal callback; non-terminal steps carry 0
# --------------------------------------------------------------------------

def test_reference_is_measured_once_per_terminal_callback(monkeypatch):
    """ONE full reference measurement per terminal callback, re-taken for a
    plan that was already measured, and none at a non-terminal step.

    COUNTED ON THE WINDOWS SINCE 2026-09-14. The reference used to be a
    single call to `_campaign_measure_cost` placed after the candidate's
    loop; the owner's ruling interleaves it with the candidate window by
    window inside that loop, so there is no such call to count any more.
    The property is unchanged and is pinned here on the thing that does the
    measuring: the reference executable takes exactly
    ``ref_num_data_points x ref_reps_per_point`` timed windows, once per
    terminal callback.
    """
    from alphagrad.approx.common import compile_cache as _cc

    # EVERY executable the cache ever hands back under a paired-ref key, not
    # just the last one. `compile_cache._LOCAL_CACHE` is an LRU capped at 32
    # entries, so in a full-suite run the reference can be evicted and
    # recompiled BETWEEN two measurements of the same plan -- a second, equal
    # executable at a different address. Counting only the newest one then
    # misses every window the older one took, which is what this test read
    # alone as a pass and in the suite as a failure (job 65528). The strong
    # references in `ref_ex` are what makes the id set safe: an id can only be
    # reused after its object is collected.
    #
    # THE CANDIDATE'S EXECUTABLES ARE COLLECTED TOO, and the two sets are
    # asserted DISJOINT before anything is counted. Counting windows by
    # ``id(ex)`` is only sound while the two halves are different objects,
    # and they need not be: `env._callback`'s own comment records that an
    # identity plan and its rev-exact reference are the SAME program, and
    # `jax.jit(...).lower(...).compile()` hands back the SAME `Compiled`
    # object for two equal lowerings. This test ran the REVERSE order as the
    # candidate, so the two halves aliased whenever the candidate's entry had
    # been evicted and recompiled; the callback then charged the candidate's
    # four windows to the reference and the count read 484 where 480 was
    # expected. Whether that happens depends on how much OTHER work has
    # passed through a 32-entry cache first, which is why it surfaced as
    # "passes alone, fails in the suite" a second time (job 65733, after five
    # tests were added to `measure_instrument_test.py`). The candidate now
    # runs the FORWARD order -- the same remedy, and the same wording,
    # `measure_instrument_test.test_the_reference_runs_its_own_points_and_reps`
    # already uses -- and the disjointness assertion makes a future alias a
    # loud failure instead of a silent miscount.
    ref_ex = []
    ref_ids = set()
    cand_ex = []
    cand_ids = set()
    real_cc = _cc.cached_compile

    def _spy_compile(key, fn):
        out = real_cc(key, fn)
        if bytes(key).startswith(b"paired-ref:"):
            ref_ex.append(out)
            ref_ids.add(id(out))
        elif bytes(key).startswith(b"approx:"):
            cand_ex.append(out)
            cand_ids.add(id(out))
        return out

    seen = []
    real_rep = envmod._time_one_rep

    def _counting(ex, *a, **k):
        seen.append(id(ex))
        return real_rep(ex, *a, **k)

    monkeypatch.setattr(_cc, "cached_compile", _spy_compile)
    monkeypatch.setattr(envmod, "_time_one_rep", _counting)
    env = _make_env()
    # FORWARD, so the candidate's executable is never the rev-exact one.
    fwd = sorted(int(x) for x in np.asarray(env.valid_vertices))
    per_cb = int(env.config.ref_num_data_points) * int(
        env.config.ref_reps_per_point)

    def _ref_windows():
        assert ref_ids, "the paired reference was never compiled"
        assert ref_ids.isdisjoint(cand_ids), (
            "the candidate and the reference are the SAME executable object, "
            "so counting timed windows by id(ex) charges the candidate's "
            "windows to the reference -- run a candidate order the rev-exact "
            "reference cannot equal")
        return sum(1 for i in seen if i in ref_ids)

    _run_plan(env, fwd)
    assert _ref_windows() == per_cb                     # one episode, one ref
    _run_plan(env, fwd)                                 # same plan again...
    assert _ref_windows() == 2 * per_cb                 # ...measured anew
    _run_plan(env, fwd, skip_everything=True)
    assert _ref_windows() == 3 * per_cb
    out = envmod.consume_plan_records()
    assert len(out["paired_ref"]["records"]) == 3
    assert out["paired_ref"]["dropped"] == 0
    # ...and the candidate was measured on every step (terminal_rewards_only
    # is off on this env), yet the reference only at the terminal one.
    assert out["mem_parity"]["measured"] == 3 * len(fwd)


def test_non_terminal_steps_carry_zero_costs():
    env = _make_env()
    rev = _rev_order(env)
    assert len(rev) >= 2
    r_mid = _run_plan(env, rev, stop_after=1)
    assert float(r_mid[_LAT]) == 0.0 and float(r_mid[_MEM]) == 0.0
    assert envmod.consume_plan_records()["paired_ref"]["records"] == []


# --------------------------------------------------------------------------
# 5. the drain and the plan-log record
# --------------------------------------------------------------------------

def test_summary_and_plan_log_record_carry_the_reference(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    env = _make_env()
    rev = _rev_order(env)
    r = _run_plan(env, rev)
    _run_plan(env, rev, skip_everything=True)
    out = envmod.consume_plan_records()
    refs = out["paired_ref"]["records"]
    s = envmod.paired_ref_summary(refs)
    assert s["n"] == 2 and s["mem_floored"] == 1
    assert s["latency_ns"] > 0.0 and s["temp_bytes"] > 0.0
    assert s["temp_bytes"] == pytest.approx(
        np.mean([x["temp_bytes"] for x in refs]))
    rec = out["records"][0]
    assert rec["cost_form"] == "paired-log"
    assert rec["ref_temp_bytes"] == refs[0]["temp_bytes"]
    assert rec["ref_latency_ns"] == refs[0]["latency_ns"]
    assert rec["ref_watermark_bytes"] == refs[0]["watermark_bytes"]
    assert rec["candidate_memory_bytes"] == rec["mem_temp_bytes"]
    assert rec["mem_log_floored"] == 0
    assert rec["rewards"][_MEM] == float(r[_MEM]) == 0.0
    assert out["records"][1]["mem_log_floored"] == 1
    assert envmod.consume_plan_records()["paired_ref"] == {
        "records": [], "dropped": 0}


# --------------------------------------------------------------------------
# 6. flag off = the measured number, negated
# --------------------------------------------------------------------------

def test_absolute_form_is_the_measured_number(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "absolute")
    env = _make_env()
    r = _run_plan(env, _rev_order(env))
    out = envmod.consume_plan_records()
    term = [x for x in out["mem_parity"]["records"] if x["terminal"]]
    assert -float(r[_MEM]) == term[-1]["static_temp_bytes"] > 0.0
    assert -float(r[_LAT]) >= envmod._LAT_FLOOR_NS
    assert out["paired_ref"] == {"records": [], "dropped": 0}
    _mobj = REWARD_INDEX["mem_objective"]
    assert float(r[_mobj]) == 0.0
    keep = [i for i in range(NUM_REWARDS) if i not in (_MEM, _LAT, _mobj)]
    monkeypatch.setenv("ALPHAGRAD_COST_FORM", "paired-log")
    r2 = _run_plan(env, _rev_order(env))
    # Nothing but the three paired slots moves between the forms (slot 11
    # is a paired quantity by definition: 0.0 under the absolute form).
    assert np.array_equal(r[keep].astype(np.float64),
                          r2[keep].astype(np.float64)), (r, r2)


# --------------------------------------------------------------------------
# 6. THE PAIRED-COST FLOOR (ticket .9, owner decision 2026-09-13)
#
# The absorber -- the plan that skips every face -- allocates nothing and
# runs in microseconds. Under the one-byte floor it collected
# log(ref_temp / 1 B) nats of memory credit: 17.3 nats on TLM, 29.7 over
# both channels on the Markowitz order (finding 63, job 65308). The
# reference floor prices that at 0 without touching any plan that costs
# more than the reference, which on the Markowitz order is all of them.
# --------------------------------------------------------------------------

def test_floor_reader_defaults_to_reference_and_refuses_a_typo(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_PAIRED_COST_FLOOR", raising=False)
    assert envmod.paired_cost_floor() == "reference"
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", " Reference ")
    assert envmod.paired_cost_floor() == "reference"
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    assert envmod.paired_cost_floor() == "byte"
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "none")
    with pytest.raises(ValueError, match="PAIRED_COST_FLOOR"):
        envmod.paired_cost_floor()


def test_reference_floor_prices_the_absorber_at_zero(monkeypatch):
    """The measured TLM absorber against the measured rev-exact reference."""
    ref_lat, ref_mem = 2.32e5, 3.362e7       # rev-exact on TLM (job 65308)
    abs_lat, abs_mem = 1.4932e4, 0.0         # skip@all: 15 us, no temp

    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    d_lat, d_mem, n = envmod.paired_log_costs(abs_lat, abs_mem, ref_lat, ref_mem)
    assert d_mem == pytest.approx(-math.log(ref_mem / 1.0))
    assert -d_mem > 17.0, "the one-byte floor hands the absorber 17+ nats"
    assert -d_lat > 2.0
    assert n == 1

    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "reference")
    d_lat, d_mem, n = envmod.paired_log_costs(abs_lat, abs_mem, ref_lat, ref_mem)
    assert d_mem == 0.0 and d_lat == 0.0, "no credit below the reference"
    assert n == 1, "the floored reading is still counted"


def test_reference_floor_leaves_an_honest_markowitz_plan_alone(monkeypatch):
    """Every Markowitz plan costs MORE than rev-exact (temp 32x to 56x,
    finding 63), so the floor never touches one, and the signal between the
    identity and the best memory saver survives intact."""
    ref_lat, ref_mem = 2.32e5, 3.362e7
    ident_lat, ident_mem = 7.7192e7, 1.930e9          # Markowitz identity
    best_lat, best_mem = 0.246 * ident_lat, 0.587 * ident_mem

    out = {}
    for policy in ("byte", "reference"):
        monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", policy)
        out[policy] = (envmod.paired_log_costs(ident_lat, ident_mem, ref_lat, ref_mem),
                       envmod.paired_log_costs(best_lat, best_mem, ref_lat, ref_mem))
    assert out["byte"] == out["reference"]
    (id_lat, id_mem, _), (bs_lat, bs_mem, _) = out["reference"]
    assert id_mem == pytest.approx(math.log(ident_mem / ref_mem))
    assert bs_mem < id_mem and (id_mem - bs_mem) == pytest.approx(-math.log(0.587))
    assert (id_lat - bs_lat) == pytest.approx(-math.log(0.246))


def test_reference_floor_costs_a_sub_reference_plan_its_difference(monkeypatch):
    """The one case the floor does change: a plan cheaper than the reference.
    On the reverse order the best float8 quant sits at 0.86x rev-exact temp
    and forfeits those 0.15 nats. Recorded so the trade is not a surprise."""
    ref_lat, ref_mem = 2.32e5, 3.362e7
    cand_mem = 0.859 * ref_mem
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "byte")
    _, d_byte, _ = envmod.paired_log_costs(ref_lat, cand_mem, ref_lat, ref_mem)
    monkeypatch.setenv("ALPHAGRAD_PAIRED_COST_FLOOR", "reference")
    _, d_ref, n = envmod.paired_log_costs(ref_lat, cand_mem, ref_lat, ref_mem)
    assert d_byte == pytest.approx(math.log(0.859))
    assert abs(d_byte) == pytest.approx(0.152, abs=0.002)
    assert d_ref == 0.0 and n == 1
