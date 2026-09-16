"""pytest entry point for the per-step host-callback census.

The census itself, its rationale and its one-liner live in
``tests/face_callback_census.py``. This file exists only so it is collected by
a plain ``pytest tests/`` run alongside everything else.

IN A CHILD INTERPRETER, for the reason ``policy_regression_gate_test.py``
spells out at length: the census drives the gate's own rollout, and importing
the gate executes ten module-scope ``ALPHAGRAD_*`` pins. ``env.py`` freezes
``MAX_FACES`` and ``MAX_DELTA_TOKENS`` into module constants at its first
import, so in a shared pytest process those pins are dead writes and the case
is built at a scale nobody chose -- measured, before this file was moved to a
child: ``env.step`` raised ``dot_general requires contracting dimensions to
have the same shape, got (32,) and (16,)`` from inside the target's own jaxpr.
And the pins LEAK into every module collected afterwards.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_CENSUS = Path(__file__).with_name("face_callback_census.py")


def _run_census(*argv: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env.setdefault("JAX_PLATFORMS", "cpu")
    return subprocess.run([sys.executable, str(_CENSUS), *argv],
                          env=env, capture_output=True, text=True,
                          timeout=3600)


def test_one_host_call_per_step_serves_the_count_and_the_legality():
    """One merged round trip per step, the count callback never fires, and the
    gate's whole semantic trace is identical on both routes."""
    r = _run_census()
    report = (f"the per-step callback census failed (rc={r.returncode})\n"
              f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert r.returncode == 0, report
    assert "[census] PASS" in r.stdout, report
