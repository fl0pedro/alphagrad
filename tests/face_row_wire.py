#!/usr/bin/env python3
"""THE ROW WIRE IS EXACT (owner ruling 2026-09-15, item 2) -- the harness.
``face_row_wire_test.py`` is the pytest entry.

The rollout's face callbacks and the env step callback used to ride with the
whole elimination-prefix face history, ``(N, W, FACE_SLOTS, 3)`` int32 plus
``(N, W)`` skips, on every one of about four calls per step. The rows below
``step_count`` are the same bytes every time. The device now sends ONE ROW --
the last row of the call's own prefix -- and ``env.face_prefix_step`` keeps
the rest.

This runs ``policy_regression_gate.run_trace`` under both wires and compares
the gate's whole semantic trace, which is the same record the policy-path
golden is made of. ``ALPHAGRAD_FACE_ROW_WIRE`` picks the wire, and the harness
reports which one it ran so a flag that stopped being read cannot pass as an
agreement.

IN ITS OWN INTERPRETER, twice, and it has to be: the gate pins ten
``ALPHAGRAD_*`` variables at module scope and ``env.py`` freezes MAX_FACES and
MAX_DELTA_TOKENS at its first import, AND ``face_driver`` resolves the wire
once per process (the bind that slices the row and the host body that expands
it have to agree). Same reason ``policy_regression_gate_test.py`` runs the gate
in a child.

Usage::

    ALPHAGRAD_FACE_ROW_WIRE=1 python tests/face_row_wire.py    # prints a sha
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import policy_regression_gate as _gate                            # noqa: E402

from alphagrad.approx.env import (                                # noqa: E402
    face_prefix_stats, face_row_wire)


def main() -> int:
    wire = face_row_wire()
    trace = _gate.run_trace()
    blob = json.dumps(trace["steps"], sort_keys=True)
    stats = face_prefix_stats()
    print(f"[row-wire] ALPHAGRAD_FACE_ROW_WIRE -> {int(wire)}")
    print(f"[row-wire] host prefix {stats}")
    print(f"[row-wire] steps={len(trace['steps'])}")
    print(f"[row-wire] sha={hashlib.sha256(blob.encode()).hexdigest()}")
    # A row wire that never reached the host store would agree with the full
    # history for the wrong reason.
    if wire and (stats["append"] + stats["verify"]) == 0:
        print("[row-wire] FAIL: the wire is on and the host prefix served "
              "nothing -- the callbacks are still carrying the history")
        return 1
    if not wire and (stats["append"] + stats["verify"]) != 0:
        print("[row-wire] FAIL: the wire is off and the host prefix was "
              "extended anyway")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
