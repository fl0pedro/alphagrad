"""THE COLLECTION-TIME CONFIGURATION GUARD (ticket dsnn-3qm.75).

WHAT WENT WRONG, because the whole file is one lesson.

``pytest tests/`` IMPORTS EVERY TEST MODULE FIRST, and only then runs the first
test. So a module-level ``os.environ["ALPHAGRAD_..."] = ...`` is not "this
module's configuration": it is a mutation of ONE SHARED PROCESS performed during
collection, and by the time any test runs, the value in force is whatever the
LAST-COLLECTED module left there. Eight tests of this suite were red for a whole
campaign because of exactly that, each with a module docstring asserting the
opposite ("one process per module (finding 47), so this module owns its
configuration"):

  * ``tests/paired_log_reward_test.py`` set ALPHAGRAD_COST_FORM=paired-log at
    import, so ``tests/mem_channel_test.py`` (collected earlier, m < p) and
    ``tests/test_all_cost_channels.py`` read slot 5 -- documented as the XLA
    temp -- as a log-difference against rev-exact. Three + two failures.
  * ``tests/delta_obs_emission_test.py`` set ALPHAGRAD_MAX_DELTA_TOKENS=1024 at
    import. ``env.py`` had already frozen 32768 (a module collected earlier
    imported it first), so the pin was a DEAD WRITE and the module's own
    "this graph no longer exercises the clip" guard fired; and the variable
    stayed set, so ``tests/per_face_apply_test.py``'s subprocess probe for
    "the default is 32768" inherited a pinned value (4096 by the end of
    collection, ``policy_regression_gate.py`` having written over the 1024)
    and failed too. One + one failures.
  * ``tests/policy_regression_gate_test.py`` imported a module that pins ten
    ALPHAGRAD_* variables at import, then ran a BIT-IDENTICAL comparison under
    a configuration those pins had failed to establish. The gate's own diff
    names the field: ``window: golden=4096 live=32768``, which IS
    ALPHAGRAD_MAX_DELTA_TOKENS. One failure.
  * ``tests/face_actions_env_test.py`` (2): both of its assertions read the
    quality slot, and ``mem_channel_test`` / ``paired_log_reward_test`` set
    ALPHAGRAD_QUALITY_METRIC=none at module scope (f < m < p), so the channel
    was off and the slot read 0.0.

WHAT THIS GUARD DOES NOT CATCH, stated so nobody trusts it further than it goes:
a UNILATERAL claim. ALPHAGRAD_QUALITY_METRIC=none was written by two modules with
the SAME value, so there is no conflict to see, and the module it broke
(face_actions_env_test) never declared that it wanted the default. Flagging every
collection-time write instead would flag the ~20 modules that use
``os.environ.setdefault`` as a legitimate "I need this if nobody has spoken"
before importing env. The remedy for the unilateral case is the other half of
this change: a module whose measurement depends on a knob DECLARES it in a
fixture, and face_actions_env_test now does.

Every one of those modules passed when run alone. The aggregate suite number
never said why.

TWO KINDS OF HAZARD, and this file treats them differently because they are not
equally decidable:

1. A CONFLICT -- two collected modules leave DIFFERENT values for the same
   ALPHAGRAD_* / GRAPHAX_* variable. There is no legitimate reason for this: the
   two modules cannot both get what they asked for, and which one loses depends
   on filename order. This FAILS THE RUN -- after every test has run, so the
   results are still there to read -- naming both modules and the variable.
   The fix is always the same: declare the configuration in an autouse fixture
   (it is then in force while that module's tests run, and restored after) or,
   if the value must be frozen before ``env.py`` is imported, run the case in a
   fresh interpreter. ``tests/mem_channel_test.py``,
   ``tests/paired_log_reward_test.py``, ``tests/test_all_cost_channels.py`` and
   ``tests/delta_obs_emission_test.py`` are the worked examples of each form.

2. A DEAD WRITE -- a collection-time write to a variable ``env.py`` has already
   frozen into a module constant, disagreeing with the value it froze. This is
   REPORTED, not failed, and the report is printed on every run. It is not a
   failure yet because several modules use ``os.environ.setdefault`` as an
   honest "small scale please, if I happen to be the first module in this
   process" request and then handle losing the race by name
   (``tests/_scale_guard.py``'s ``request_scale``, and
   ``tests/face_read_point_test.py``, which reads the live constants instead).
   Telling those apart from a real mistake needs each module to declare its
   request the way ``request_scale`` does; until they all do, printing the list
   is the honest gate. Anything on this list is a line of code with no effect.

Scope: ALPHAGRAD_* and GRAPHAX_* only -- the project's own knobs, which is the
blast radius above. JAX_*, XLA_* and CUDA_VISIBLE_DEVICES are deliberately NOT
watched, and the reason is checked rather than assumed: they are set with
``setdefault`` before the jax import by nearly every module, and the only three
that ASSIGN conflicting CUDA_VISIBLE_DEVICES values --
environment_interaction_test (2), runtime_game_test (0,1,2,3) and
seq_transformer_test (3) -- all call ``pytest.skip(allow_module_level=True)`` on
LINE 3, before the ``os.environ`` line is ever reached, so those three writes
never execute at all. If a live module ever starts assigning a device list, add
the prefix here.


-----------------------------------------------------------------------------
TWO TEST ROOTS, ONE GUARD. ``pyproject`` declares ``testpaths = ["tests",
"src/alphagrad/approx/tests"]``. This logic lives in
``_pytest_config_guard.py`` AT THE REPOSITORY ROOT and is loaded by BOTH
``conftest.py`` (root, covering both roots) and ``tests/conftest.py`` (kept, so
``pytest tests/`` from inside a subtree still gets the guard). The two shims
share this module rather than duplicating the hook bodies, and the module is
IDEMPOTENT: the first conftest to load it becomes the primary and the other's
hooks are no-ops, so nothing is counted twice when both are in the chain.

Measured 2026-09-12: with the guard at the root only, nothing moves (89/89 in
the second root, 1372 collected under ``tests/``, 1461 across both, no import
errors) and the second root's collection-time write to
``ALPHAGRAD_SKIP_COUNT_OPS`` becomes visible for the first time. The hazard
``prepend`` import mode was supposed to create -- the repository root on
``sys.path`` shadowing a top-level module -- did not materialise. The copy is
deliberate (owner ruling 2026-09-12: "COPY, do not move. Divergence does not
have to be strictly prohibited.").

-----------------------------------------------------------------------------
``EXHAUSTIVE=1`` -- THE NOISE FLOOR AS A NUMBER (ticket dsnn-3qm.75).

Several tests in this suite assert on a WALL-CLOCK latency, a drift ratio
between two adjacent timings, or a measured byte count. Their result is a DRAW
from a distribution, not a fact, and every suite count this campaign has quoted
-- including "0 failed" and the "14 pre-existing failures" baseline -- is ONE
such draw. There is no enumerable sample space (the randomness is wall-clock
assertions, allocator state and compile-cache state), so repetition is the only
instrument.

``EXHAUSTIVE=1 pytest <paths>`` repeats every test in
:data:`MEASUREMENT_TESTS` ``EXHAUSTIVE_REPEATS`` times (default 50) and prints a
PER-TEST FAILURE RATE in the terminal summary. Everything else runs once, so
one command gives both halves of the question: the rate for the measurement
tests, and a pass/fail for the deterministic remainder.
:data:`MEASUREMENT_TESTS` is a DECLARATION with a reason per entry -- a test
that starts asserting on a clock belongs in it, and a test that stops should
leave.
"""
from __future__ import annotations

import os
import pathlib
import sys


_WATCHED = ("ALPHAGRAD_", "GRAPHAX_")

# ---------------------------------------------------------------------------
# IDEMPOTENCE. Both conftests load THIS module, under this name, so they share
# its state. Only the first to register acts; the other's hooks return early.
# ---------------------------------------------------------------------------
_PRIMARY: str | None = None


def register(conftest_path: str) -> bool:
    """``True`` for the first conftest to register -- the one whose hooks run."""
    global _PRIMARY
    if _PRIMARY is None:
        _PRIMARY = str(conftest_path)
    return _PRIMARY == str(conftest_path)


# ---------------------------------------------------------------------------
# THE MEASUREMENT TESTS (ticket dsnn-3qm.75). A DECLARATION, with the reason.
#
# Every entry asserts on something the machine decides at run time: a
# wall-clock latency, a ratio between two adjacent timings, or a measured byte
# count. Identified by reading every assertion in both test roots, not by
# watching which ones happened to go red.
#
# Classes, as reported in the noise-floor measurement:
#   B  a RATIO or DRIFT between two measured timings
#   A  an absolute wall-clock / latency reading
#   C  a measured memory byte count (static temp, runtime watermark)
# ---------------------------------------------------------------------------
MEASUREMENT_TESTS: dict[str, str] = {
    # --- B: two adjacent timings of the same executable ---------------------
    "tests/paired_log_reward_test.py::"
    "test_rev_exact_scores_exactly_zero_on_memory_and_inside_drift_on_latency":
        "B -- asserts abs(Delta_lat) < log(2) on two back-to-back windows of "
        "ONE executable at ~20-37 us (ticket .75)",
    "tests/measure_instrument_test.py::"
    "test_reference_matches_the_campaign_measurement_of_the_same_executable":
        "B -- asserts ref_lat/plan_lat == approx(1.0, rel=0.5) and "
        "ref_peak == approx(plan_peak, rel=0.5) (ticket .75)",
    # --- A: an absolute wall-clock reading ---------------------------------
    "tests/paired_log_reward_test.py::test_absolute_form_is_the_measured_number":
        "A -- asserts the latency slot is >= _LAT_FLOOR_NS",
    "tests/paired_log_reward_test.py::"
    "test_summary_and_plan_log_record_carry_the_reference":
        "A -- asserts latency_ns > 0 and temp_bytes > 0",
    "tests/test_all_cost_channels.py::"
    "test_all_six_cost_channels_populate_when_measure_latency_on":
        "A -- fails any cost channel that reads 0.0, latency_ns included",
    "tests/test_all_cost_channels.py::test_latency_is_zero_when_measure_latency_off":
        "A/C -- the inverse pin, on the same measured channels",
    "tests/test_all_cost_channels.py::"
    "test_target_fun_none_early_return_zeros_jit_channels":
        "A/C -- asserts the measured flops/latency/bytes/peak channels are all "
        "0.0 on the early return; shares its module's measured configuration",
    "tests/landscape_map_sweep_test.py::test_measure_singleton_and_stacks":
        "A -- measure() wraps a real _callback in perf_counter and the test "
        "asserts its latency_ns is finite",
    # --- C: measured bytes -------------------------------------------------
    "tests/mem_channel_test.py": (
        "C -- every test in the module reads real XLA static-temp and runtime "
        "watermark bytes through env._callback"),
    "tests/paired_log_reward_test.py::"
    "test_cheaper_plan_scores_negative_delta_and_takes_the_floor":
        "C -- asserts on measured temp bytes and the floor flag",
    "tests/paired_log_reward_test.py::test_costlier_plan_scores_positive_delta":
        "C -- asserts on measured candidate vs reference bytes",
}

# Whole modules may be declared (no "::"); a node id under one matches.
_EXHAUSTIVE = os.environ.get("EXHAUSTIVE", "0").strip().lower() in ("1", "true", "yes")


def _repeats() -> int:
    try:
        return max(1, int(os.environ.get("EXHAUSTIVE_REPEATS", "50")))
    except ValueError:
        return 50


def _measurement_reason(nodeid: str) -> str | None:
    """The declared reason this node is a measurement test, or None."""
    nodeid = nodeid.replace("\\", "/")
    for key, reason in MEASUREMENT_TESTS.items():
        if "::" in key:
            if nodeid == key or nodeid.startswith(key + "["):
                return reason
        else:
            if nodeid.startswith(key + "::"):
                return reason
    return None


_REP_PARAM = "__exhaustive_rep"
# base nodeid -> {repeat index: outcome} for the call phase of every repeat.
# The INDEX is kept, not just the count, because repeat 0 is the only COLD one
# (first execution in the process: cold compile cache, cold allocator, first
# trace) and a cold-start effect would otherwise be averaged away. Whether a
# given flake prefers repeat 0 is itself MACHINE-DEPENDENT and was measured both
# ways on 2026-09-12:
#   pgi15-cpu2  (job 65008): measure_instrument 2/60, paired_log drift 1/60, and
#               the drift firing was on repeat 0.
#   pgi15-gpu17 (job 65019, same commits, same command, CPU backend, load 0.21):
#               measure_instrument 7/60 -- repeats 18, 22, 39, 42, 45, 46, 51,
#               NOT repeat 0 -- and the drift test 0/60.
# So: 3.3% vs 11.7% for the SAME test on the SAME code, a 3.5x difference from
# the machine alone, and the cold-start reading holds on one node and not the
# other. A rate reported without its node name means nothing.
_rep_outcomes: dict[str, dict[int, str]] = {}


def generate_tests(metafunc):
    """Under ``EXHAUSTIVE=1``, a declared measurement test is parametrized over
    ``EXHAUSTIVE_REPEATS`` repeats, so each repeat gets its own node id and its
    own report instead of a single draw."""
    if not _EXHAUSTIVE:
        return
    if _measurement_reason(metafunc.definition.nodeid) is None:
        return
    if _REP_PARAM in metafunc.fixturenames:            # already parametrized
        return
    metafunc.fixturenames.append(_REP_PARAM)
    metafunc.parametrize(_REP_PARAM, range(_repeats()))


def _split_nodeid(nodeid: str) -> tuple[str, int]:
    """``(base nodeid, repeat index)``; index -1 when not one of our repeats."""
    if "[" not in nodeid or not nodeid.endswith("]"):
        return nodeid, -1
    base, _, tail = nodeid.partition("[")
    try:
        return base, int(tail[:-1])
    except ValueError:
        return base, -1


def runtest_logreport(report):
    if not _EXHAUSTIVE:
        return
    if _measurement_reason(report.nodeid) is None:
        return
    if report.when == "call" or (report.when == "setup"
                                and report.outcome == "failed"):
        base, rep = _split_nodeid(report.nodeid)
        slot = _rep_outcomes.setdefault(base, {})
        key = rep if rep >= 0 else -(len(slot) + 1)
        if slot.get(key) != "failed":        # a setup failure wins over a skip
            slot[key] = "failed" if report.outcome == "failed" else report.outcome


def _exhaustive_summary(write_line, section):
    if not _EXHAUSTIVE:
        return
    section("EXHAUSTIVE -- PER-TEST FAILURE RATE")
    if not _rep_outcomes:
        write_line("no declared measurement test was selected by this run; "
                   "MEASUREMENT_TESTS names %d entries" % len(MEASUREMENT_TESTS))
        return
    write_line("EXHAUSTIVE_REPEATS=%d. A rate, not a pass/fail: these tests "
               "assert on a clock or on measured bytes, so one run is one "
               "draw (ticket dsnn-3qm.75)." % _repeats())
    worst_noise = 0.0
    deterministic = []
    for nodeid in sorted(_rep_outcomes):
        outs = _rep_outcomes[nodeid]
        fails = sorted(k for k, v in outs.items() if v == "failed")
        rate = len(fails) / max(len(outs), 1)
        if len(fails) == len(outs) and len(outs) > 1:
            deterministic.append(nodeid)
            tag = "ALWAYS"
        else:
            worst_noise = max(worst_noise, rate)
            tag = "%5.1f%%" % (100.0 * rate)
        write_line("  %-7s %3d/%-3d failed   %s"
                   % (tag, len(fails), len(outs), nodeid))
        write_line("           reason declared: %s" % _measurement_reason(nodeid))
        if fails and len(fails) != len(outs):
            write_line("           failing repeats: %s%s"
                       % (fails[:12], "" if 0 not in fails else
                          "   <-- includes repeat 0, the only COLD one"))
    write_line("")
    if deterministic:
        write_line("NOT NOISE -- these failed EVERY repeat, so they are "
                   "deterministic for this selection, not random:")
        for nodeid in deterministic:
            write_line("    %s" % nodeid)
        write_line("  A test that fails every repeat of a SUBSET and passes in "
                   "the full suite is order- or configuration-dependent, which "
                   "is a different bug from noise. Re-run it under the full "
                   "suite before calling it broken.")
        write_line("")
    write_line("NOISE FLOOR: the worst NON-deterministic per-test failure rate "
               "above is %.1f%%. A suite count is reproducible only to within "
               "the tests on this list." % (100.0 * worst_noise))
    write_line("CAVEAT 1, measured: THIS RATE IS A PROPERTY OF THIS MACHINE. "
               "The same test, same commit, same command gave 3.3%% on "
               "pgi15-cpu2 and 11.7%% on pgi15-gpu17's CPUs (jobs 65008 / "
               "65019, 2026-09-12). Report the node with the rate.")
    write_line("CAVEAT 2: repeat 0 is the only COLD execution (cold compile "
               "cache, cold allocator, first trace) and a suite run executes "
               "each test exactly once, always cold -- so a warm-sensitive "
               "test's rate here is a LOWER BOUND. Whether a flake prefers "
               "repeat 0 is itself machine-dependent (it did on pgi15-cpu2, "
               "it did not on pgi15-gpu17), so the indices above are printed "
               "rather than summarised.")





def _frozen_at_env_import() -> frozenset[str]:
    """The env variables ``alphagrad.approx.env`` reads ONCE, at MODULE SCOPE,
    into a constant -- derived from the source with ``ast`` rather than listed
    here, so the set cannot rot as env.py gains or loses a knob. Every name in
    it is a value that the FIRST module to import env decides for the whole
    process; a later write to one is dead.

    Parsing failure is not fatal: the conflict gate (the part that fails a run)
    does not depend on this set, only the dead-write report does.
    """
    import ast
    import importlib.util

    # find_spec LOCATES the module without EXECUTING it. Importing env here
    # would be a bug with teeth: pytest_configure runs before any test module
    # is imported, so an import here would make THIS FILE the first importer
    # and freeze MAX_DELTA_TOKENS / MAX_FACES from the bare job environment,
    # taking the race away from the modules that currently win it.
    #
    # find_spec does import the PARENT packages in order to read their
    # __path__, so it is only safe because of two facts about this tree, both
    # checked: ``src/alphagrad/__init__.py`` imports nothing eagerly (its
    # vertexgame exports are behind a module __getattr__, deliberately, so that
    # importing alphagrad does not pull in JAX), and ``src/alphagrad/approx/``
    # has NO ``__init__.py`` at all -- it is a namespace package, so resolving
    # ``alphagrad.approx`` executes no code. If either ever changes, this must
    # become a plain path lookup.
    try:
        spec = importlib.util.find_spec("alphagrad.approx.env")
        path = pathlib.Path(spec.origin)
    except Exception:
        path = (pathlib.Path(__file__).resolve().parent.parent
                / "src" / "alphagrad" / "approx" / "env.py")
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return frozenset()
    names: set[str] = set()
    # MODULE SCOPE ONLY: descend through module-level if/try/with/for, but NEVER
    # into a function or class body -- a read inside a function happens per call
    # and is exactly the kind that is NOT frozen.
    _opaque = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _opaque):
            continue
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "getenv")
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value)
        elif (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            names.add(node.slice.value)
        stack.extend(ast.iter_child_nodes(node))
    return frozenset(n for n in names if n.startswith(_WATCHED))


_ENV_MODULE = "alphagrad.approx.env"

FROZEN_AT_ENV_IMPORT: frozenset[str] = frozenset()

# nodeid -> (env-was-already-imported, watched-variables) as of collect start.
_before: dict[str, tuple[bool, dict[str, str]]] = {}
# variable -> list of (nodeid, value) -- one entry per module that set it.
_claims: dict[str, list[tuple[str, str | None]]] = {}
# (nodeid, variable, old, new) for writes made after env.py froze the constant.
_dead_writes: list[tuple[str, str, str | None, str | None]] = []


def _watched_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(_WATCHED)}


def configure(config):
    global FROZEN_AT_ENV_IMPORT
    FROZEN_AT_ENV_IMPORT = _frozen_at_env_import()


def collectstart(collector):
    _before[collector.nodeid] = (_ENV_MODULE in sys.modules, _watched_env())


def collectreport(report):
    pre = _before.pop(report.nodeid, None)
    if pre is None or not report.nodeid.endswith(".py"):
        return
    env_was_imported, before = pre
    after = _watched_env()
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        _claims.setdefault(key, []).append((report.nodeid, new))
        if env_was_imported and key in FROZEN_AT_ENV_IMPORT:
            _dead_writes.append((report.nodeid, key, old, new))


def _frozen_values() -> dict[str, object]:
    """What ``env.py`` actually holds, for the dead-write report. Empty when the
    module was never imported (a collection that touches no env test)."""
    mod = sys.modules.get(_ENV_MODULE)
    if mod is None:
        return {}
    return {
        "ALPHAGRAD_MAX_DELTA_TOKENS": getattr(mod, "MAX_DELTA_TOKENS", "?"),
        "ALPHAGRAD_MAX_FACES": getattr(mod, "MAX_FACES", "?"),
        "ALPHAGRAD_MAX_BASE_TOKENS": getattr(mod, "MAX_BASE_TOKENS", "?"),
        "ALPHAGRAD_DELTA_OVERFLOW": getattr(mod, "_DELTA_OVERFLOW", "?"),
        "ALPHAGRAD_SKIP_COUNT_OPS": getattr(mod, "_SKIP_COUNT_OPS", "?"),
    }


_conflict_report: list[str] = []

# test nodeid -> {variable: (value before the test, value after it)} for every
# ALPHAGRAD_*/GRAPHAX_* variable a test changed and did not put back.
_run_phase_leaks: dict[str, dict[str, tuple[str | None, str | None]]] = {}


def run_phase_leak(request):
    """REPORT -- deliberately not repair -- any ALPHAGRAD_*/GRAPHAX_* variable a
    test changes and does not put back.

    The collection-time half of this hazard is gated above. The run-phase half is
    the same hazard one phase later: a test that writes ``os.environ`` directly
    instead of through ``monkeypatch`` changes the configuration of every test
    that runs after it. Nothing in the fourteen failures this file documents was
    caused by a run-phase leak, so this half only measures: restoring the
    environment here would quietly change the meaning of any test that (wrongly)
    relies on a predecessor's write, and that is a change to make deliberately
    with the list in hand, not as a side effect. The list is printed in the
    terminal summary.
    """
    before = _watched_env()
    yield
    after = _watched_env()
    changed = {
        key: (before.get(key), after.get(key))
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }
    if changed:
        _run_phase_leaks[request.node.nodeid] = changed


def collection_finish(session):
    if _dead_writes:
        frozen = _frozen_values()
        print("\n[config] DEAD WRITES AT COLLECTION TIME -- these lines had no "
              "effect, because alphagrad.approx.env had already frozen the "
              "constant when the module ran:")
        for nodeid, key, old, new in _dead_writes:
            held = frozen.get(key)
            held = "" if held is None else f"; env.py holds {held!r}"
            print(f"    {nodeid}: {key} {old!r} -> {new!r}{held}")
        print("[config] A module that needs one of these frozen before the "
              "import must run in its own interpreter (see "
              "tests/delta_obs_emission_test.py) or accept what it was given "
              "by name (see tests/_scale_guard.py).")

    conflicts = {
        key: claims for key, claims in _claims.items()
        if len({value for _nodeid, value in claims}) > 1
    }
    if not conflicts:
        return
    lines = [
        "CROSS-MODULE CONFIGURATION CONFLICT AT COLLECTION TIME.",
        "",
        "pytest imports every test module before running the first test, so a",
        "module-level `os.environ[...] = ...` configures the WHOLE RUN and the",
        "last module collected wins. These variables were left at different",
        "values by different modules, so at least one module below is measuring",
        "under a configuration it did not choose:",
        "",
    ]
    for key in sorted(conflicts):
        lines.append(f"  {key}")
        for nodeid, value in conflicts[key]:
            lines.append(f"      {nodeid} left it {value!r}")
    lines += [
        "",
        "THE FIX: declare the configuration in an autouse fixture, so it is in",
        "force while that module's tests run and is restored afterwards --",
        "tests/mem_channel_test.py and tests/paired_log_reward_test.py are the",
        "worked examples. If the value must be frozen before env.py is imported",
        "(MAX_DELTA_TOKENS, MAX_FACES, ...), a fixture cannot help: run the case",
        "in a fresh interpreter, as tests/delta_obs_emission_test.py and",
        "tests/policy_regression_gate_test.py now do.",
    ]
    _conflict_report.extend(lines)
    # Printed here as well as in the summary: a 1400-test run scrolls, and this
    # has to be visible at the point it is detected.
    print("\n" + "\n".join("[config] " + ln for ln in lines))


def terminal_summary(terminalreporter, exitstatus, config):
    if _run_phase_leaks:
        terminalreporter.section("CONFIGURATION LEFT CHANGED BY A TEST")
        terminalreporter.write_line(
            "These tests changed an ALPHAGRAD_*/GRAPHAX_* variable and did not "
            "put it back, so every test after them ran under the new value. "
            "Use monkeypatch.setenv / delenv instead of os.environ.")
        shown = 0
        for nodeid, changed in _run_phase_leaks.items():
            for key, (old, new) in sorted(changed.items()):
                if shown >= 40:
                    terminalreporter.write_line(
                        f"    ... and more, {len(_run_phase_leaks)} tests in "
                        f"total left something changed")
                    return
                terminalreporter.write_line(
                    f"    {nodeid}: {key} {old!r} -> {new!r}")
                shown += 1
    if _conflict_report:
        terminalreporter.section("CONFIGURATION CONFLICT", red=True, bold=True)
        for line in _conflict_report:
            terminalreporter.write_line(line)
    _exhaustive_summary(terminalreporter.write_line,
                        lambda t: terminalreporter.section(t, bold=True))


def sessionfinish(session, exitstatus):
    """A configuration conflict FAILS THE RUN -- but only after every test has
    run. Aborting collection instead would hide the test results behind the
    guard, and the results are how the next person sees what the conflict did;
    a run whose configuration is ambiguous must not be able to report success.
    """
    if _conflict_report and exitstatus == 0:
        session.exitstatus = 1
