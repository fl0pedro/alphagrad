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
    "the default is 32768" inherited 1024 and failed too. One + one failures.
  * ``tests/policy_regression_gate_test.py`` imported a module that pins ten
    ALPHAGRAD_* variables at import, then ran a BIT-IDENTICAL comparison under
    a configuration those pins had failed to establish. One failure.

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
watched: they are set before the jax import by nearly every module, and the
three modules that do assign conflicting CUDA_VISIBLE_DEVICES values
(environment_interaction_test, runtime_game_test, seq_transformer_test) collect
ZERO tests today, which is ticket dsnn-3qm.78's subject and not this guard's.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

_WATCHED = ("ALPHAGRAD_", "GRAPHAX_")


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


def pytest_configure(config):
    global FROZEN_AT_ENV_IMPORT
    FROZEN_AT_ENV_IMPORT = _frozen_at_env_import()


def pytest_collectstart(collector):
    _before[collector.nodeid] = (_ENV_MODULE in sys.modules, _watched_env())


def pytest_collectreport(report):
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


@pytest.fixture(autouse=True)
def _report_run_phase_configuration_leaks(request):
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


def pytest_collection_finish(session):
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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
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


def pytest_sessionfinish(session, exitstatus):
    """A configuration conflict FAILS THE RUN -- but only after every test has
    run. Aborting collection instead would hide the test results behind the
    guard, and the results are how the next person sees what the conflict did;
    a run whose configuration is ambiguous must not be able to report success.
    """
    if _conflict_report and exitstatus == 0:
        session.exitstatus = 1
