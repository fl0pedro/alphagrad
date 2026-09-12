"""The collection-time configuration guard and the ``EXHAUSTIVE=1`` noise-floor
mode are CONTRACTS, not conveniences (tickets dsnn-3qm.75, .78).

Three things can rot silently and each has cost this campaign real time:

1. ``_pytest_config_guard.MEASUREMENT_TESTS`` names node ids by hand. A rename
   or a deletion turns an entry into a no-op and ``EXHAUSTIVE=1`` then reports
   a rate for a test that does not exist -- or, worse, reports nothing and
   looks clean. So every declared node id must RESOLVE.
2. The two conftests must share one implementation. A divergence is allowed
   (owner ruling 2026-09-12: "COPY, do not move") but a SECOND COPY OF THE
   HOOK BODIES is not: that is how the root root and the ``tests/`` root would
   start disagreeing about what a conflict is.
3. Only one of the two may be primary, or every count the guard reports is
   doubled.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_GUARD = _ROOT / "_pytest_config_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("_guard_under_test", _GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_guard_lives_in_one_place_and_both_conftests_are_shims():
    assert _GUARD.is_file(), f"{_GUARD} is the shared implementation"
    for shim in (_ROOT / "conftest.py", _ROOT / "tests" / "conftest.py"):
        src = shim.read_text()
        assert "_pytest_config_guard.py" in src, f"{shim} must load the shared guard"
        # the hook BODIES must not be duplicated: the shim forwards and nothing more
        assert "_claims" not in src and "_dead_writes" not in src, (
            f"{shim} carries guard state of its own -- the two roots will "
            f"diverge on what a conflict is")
        assert "PRIMARY = _G.register(__file__)" in src, (
            f"{shim} must take part in the primacy handshake")


def test_only_the_first_conftest_is_primary():
    g = _load_guard()
    assert g.register("/a/conftest.py") is True
    assert g.register("/b/conftest.py") is False
    assert g.register("/a/conftest.py") is True      # idempotent for the owner


@pytest.mark.parametrize("nodeid", sorted(_load_guard().MEASUREMENT_TESTS))
def test_every_declared_measurement_test_resolves(nodeid):
    """A declared node id that pytest cannot collect is a dead declaration."""
    target = nodeid.split("::")[0]
    assert (_ROOT / target).is_file(), f"{target} does not exist"
    if "::" not in nodeid:
        return
    # EXHAUSTIVE must be OFF in the child: with it on, the guard parametrizes
    # this very node into EXHAUSTIVE_REPEATS copies and the count is 60, not 1.
    # (Measured: this test failed for all ten declared ids under
    # `EXHAUSTIVE=1 pytest`, job 65019, before the child's environment was
    # scrubbed.)
    env = dict(os.environ)
    env.pop("EXHAUSTIVE", None)
    env.pop("EXHAUSTIVE_REPEATS", None)
    out = subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "--collect-only", "-q",
         "-p", "no:randomly", "--no-header"],
        cwd=_ROOT, capture_output=True, text=True, timeout=600, env=env)
    assert "1 test collected" in out.stdout or "1 tests collected" in out.stdout, (
        f"{nodeid} collected nothing:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")


def test_every_declaration_carries_a_reason_naming_its_class():
    g = _load_guard()
    assert g.MEASUREMENT_TESTS, "the declaration must not be empty"
    for nodeid, reason in g.MEASUREMENT_TESTS.items():
        assert reason.split()[0].rstrip("-").strip() in ("A", "B", "C", "A/C", "B/C"), (
            f"{nodeid}: the reason must open with the class (A/B/C): {reason!r}")
        assert len(reason) > 30, f"{nodeid}: say WHAT it asserts on: {reason!r}"


def test_exhaustive_is_read_from_the_environment_and_is_opt_in():
    """The repeat mode must be OPT-IN: the guard's flag follows ``EXHAUSTIVE``
    and nothing else. Asserted against the live environment rather than against
    ``False``, because this test is itself run under ``EXHAUSTIVE=1`` by the very
    campaign the flag exists for -- the first version asserted ``False`` and
    failed exactly there (job 65019)."""
    g = _load_guard()
    want = os.environ.get("EXHAUSTIVE", "0").strip().lower() in ("1", "true", "yes")
    assert g._EXHAUSTIVE is want, (
        f"EXHAUSTIVE={os.environ.get('EXHAUSTIVE')!r} but the guard reads "
        f"{g._EXHAUSTIVE!r}")
    if not want:
        assert g._repeats() == 50, "the default repeat count is 50"


def test_a_declared_node_is_recognised_and_an_undeclared_one_is_not():
    g = _load_guard()
    declared = next(k for k in g.MEASUREMENT_TESTS if "::" in k)
    assert g._measurement_reason(declared) is not None
    assert g._measurement_reason(declared + "[7]") is not None, (
        "a repeat's parametrized node id must still match its declaration")
    assert g._measurement_reason("tests/gradient_structure_test.py::test_exact_match_is_perfect") is None
