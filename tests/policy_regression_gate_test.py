"""pytest entry point for the bit-identical policy-path regression gate.

The gate itself, its rationale and its one-liner live in
``tests/policy_regression_gate.py``. This file exists only so the gate is
collected by a plain ``pytest tests/`` run alongside everything else.

Standalone (this is the form to use after every Phase 1-2 edit)::

    JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py

THE GATE RUNS IN ITS OWN INTERPRETER, AND HAS TO.

This file used to ``exec_module`` the gate at COLLECTION time and call
``G.check()`` in-process. Both halves of that were wrong, and it is why
``test_policy_path_is_bit_identical_to_golden`` failed in every full-suite run
of this campaign while passing on its own:

  * The gate's own header says it plainly -- "Every one of these must be set
    BEFORE alphagrad/graphax import: env.py reads MAX_FACES /
    MAX_DELTA_TOKENS at module scope". In a shared pytest process alphagrad has
    already been imported, during the collection of an alphabetically earlier
    module, so the gate's ten pins are DEAD WRITES. The gate then traced a
    policy path configured by somebody else and compared it, bit for bit,
    against a golden recorded under the pins. A bit-identical gate cannot be
    run under a configuration it does not control.
  * The pins were not merely dead, they LEAKED: ten ALPHAGRAD_* variables
    (ALPHAGRAD_POLICY, ALPHAGRAD_FORCE_REV, ALPHAGRAD_MAX_DELTA_TOKENS,
    ALPHAGRAD_FACE_ENUM_CACHE, ...) were written into the shared process at
    collection time and stayed there for every module collected or run
    afterwards.

A child process gets the pins it asks for and leaves this one alone, so the
gate is a real gate inside the suite again rather than a green that depended on
being run by itself. The cost is one interpreter start-up plus the gate's own
trace -- what ``python tests/policy_regression_gate.py`` has always cost.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_GATE = Path(__file__).with_name("policy_regression_gate.py")


def _run_gate(*argv: str) -> subprocess.CompletedProcess:
    """Run the gate script in a fresh interpreter with NO inherited
    ALPHAGRAD_* configuration, so the gate's own module-scope pins ARE the
    whole configuration -- the only state the golden was recorded under.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env.setdefault("JAX_PLATFORMS", "cpu")
    return subprocess.run([sys.executable, str(_GATE), *argv],
                          env=env, capture_output=True, text=True,
                          timeout=3600)


def _report(what: str, r: subprocess.CompletedProcess) -> str:
    return (f"{what} (rc={r.returncode})\n"
            f"--- the gate's stdout ---\n{r.stdout}\n"
            f"--- the gate's stderr ---\n{r.stderr}")


def test_policy_path_is_bit_identical_to_golden():
    """Fails with the gate's own readable diff, not a bare assert."""
    r = _run_gate()
    assert r.returncode == 0, _report("POLICY-PATH REGRESSION GATE FAILED", r)
    assert "[gate] PASS" in r.stdout, _report(
        "the gate exited 0 without reporting a PASS", r)


def test_the_golden_is_not_trivial():
    """A golden that pinned the fail-soft path would be green forever while
    the thing it protects was dead. Re-checked against the RECORDED golden,
    not only against the live trace, so a future re-record cannot quietly
    downgrade the fixture.

    In the child for one reason: importing the gate module executes its
    module-scope ALPHAGRAD_* pins, and this process must not inherit them.
    """
    r = _run_gate("--assert-golden-nontrivial")
    assert r.returncode == 0, _report(
        "the RECORDED golden is trivial (or unreadable)", r)
    # rc == 0 alone would also be what a gate that IGNORED the flag returns
    # (it would run the whole trace and report a PASS), so the mode has to
    # identify itself or this test could pass without checking the golden.
    assert "the recorded golden" in r.stdout, _report(
        "--assert-golden-nontrivial did not run the golden check", r)
