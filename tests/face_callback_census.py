#!/usr/bin/env python3
"""HOW MANY HOST ROUND TRIPS ONE ROLLOUT STEP MAKES (owner ruling 2026-09-15,
item 1) -- the harness. ``face_callback_census_test.py`` is the pytest entry.

``face_driver`` keeps a census: one counter per host body, bumped once per
``pure_callback`` INVOCATION. Every face body is dispatched with
``vmap_method="broadcast_all"``, so one invocation serves the whole batch and
the census IS the number of device-to-host round trips the face path makes.

THE ROLLOUT IS ``policy_regression_gate.run_trace``, UNCHANGED. That module is
where agent / env / live-face-stream construction and the step loop live, and a
second hand-copy of either is exactly what this repository already paid for
once. This file swaps ONE object into the case -- the slot-legality callback --
runs the gate's rollout twice, and compares the census and the trace.

IN ITS OWN INTERPRETER, and it has to be: the gate pins ten ALPHAGRAD_*
variables at module scope, and ``env.py`` freezes MAX_FACES and
MAX_DELTA_TOKENS into module constants at its first import. A shared pytest
process has already imported alphagrad, so the pins would be dead writes and
the case would be built at a scale nobody chose. Same reason
``policy_regression_gate_test.py`` runs the gate in a child.

WHAT IT CHECKS.

1. THE MERGE. ``make_face_slot_legality_callback(with_count=True)`` serves the
   face COUNT and the per-slot MASKS in ONE call. Over a whole episode the
   merged route fires ``faces.count_legality`` once per step and
   ``faces.live_count`` NEVER; the unmerged route fires ``faces.live_count``
   and ``faces.live_slot_legality`` once each per step. One round trip per step
   is removed, and the operand set those two shared -- the prefix order, the
   rule specs and the two face-history wires -- crosses the bus once instead of
   twice.

2. THE MERGE IS EXACT. The gate's whole semantic trace, which is what the
   policy-path golden is recorded from, is identical on both routes. The count
   is :meth:`LiveFaceStream.n_faces` on both, which is why: the merged call
   does not read ``face_slot_legality``'s own ``min(len(keys), F)``, which
   clamps and soft-fails to zero where ``n_faces`` raises.

3. THE REST OF THE CENSUS, WHICH IS NOT ONE. The ruling asked for a rollout
   step to make exactly one ``pure_callback`` for all host work. It cannot, and
   this says so in numbers rather than in a comment. Per step the face path
   fires ONE ``faces.count_legality`` before the draw; ONE
   ``faces.live_chunk`` PER LIVE FACE, from inside ``UnifiedPolicy._face_loop``'s
   ``lax.while_loop``, because face f's chunk is read on the graph faces
   0..f-1 of the SAME vertex have already been approximated on; and ONE
   ``faces.vertex_decide`` after that loop, whose operands are the loop's own
   output. Plus the env step callback, which runs after the action is complete.
   The moments are separated by device-side draws, so merging them means moving
   the neural face draw to the host.

Usage::

    python tests/face_callback_census.py          # prints a report, rc 0 = pass
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import policy_regression_gate as _gate                            # noqa: E402

from alphagrad.approx.common.face_driver import (                 # noqa: E402
    consume_callback_census,
    make_face_slot_legality_callback,
)

STEPS = None      # None => the gate's own default (num_valid - 1)


def _run(with_count):
    """``(trace, census)`` for one gate rollout on the chosen route."""
    case = _gate.build_case()
    case["sizes_cb"] = make_face_slot_legality_callback(
        case["stream"], max_faces=case["max_faces"],
        max_axes=case["max_axes"], with_count=with_count)
    consume_callback_census()
    trace = _gate.run_trace(case, steps=STEPS)
    return trace, consume_callback_census()


def _n_live(trace):
    return [0 if s["face"] is None else int(s["face"]["n_live"])
            for s in trace["steps"]]


def check() -> int:
    problems = []

    merged, cen_m = _run(True)
    plain, cen_p = _run(False)

    steps = len(merged["steps"])
    live = _n_live(merged)
    if steps < 2:
        problems.append(f"the rollout ran {steps} steps; the census is vacuous")
    if sum(live) == 0:
        problems.append("no vertex in this episode had a live face; the "
                        "per-face chunk count is vacuous")

    print(f"[census] steps={steps}  live faces per step={live}")
    print(f"[census] merged   {dict(sorted(cen_m.items()))}")
    print(f"[census] unmerged {dict(sorted(cen_p.items()))}")

    # 1. the merge
    if cen_m.get("faces.count_legality", 0) != steps:
        problems.append(
            f"merged: faces.count_legality={cen_m.get('faces.count_legality', 0)}"
            f" over {steps} steps, want one per step")
    if cen_m.get("faces.live_count", 0) != 0:
        problems.append(
            f"merged: the separate face-count callback fired "
            f"{cen_m['faces.live_count']} times; the count rides out of the "
            f"legality call")
    if cen_m.get("faces.live_slot_legality", 0) != 0:
        problems.append(
            "merged: the unmerged legality key was bumped on the merged route")

    # the control: two round trips without it
    if cen_p.get("faces.live_count", 0) != steps:
        problems.append(
            f"unmerged: faces.live_count={cen_p.get('faces.live_count', 0)} "
            f"over {steps} steps")
    if cen_p.get("faces.live_slot_legality", 0) != steps:
        problems.append(
            f"unmerged: faces.live_slot_legality="
            f"{cen_p.get('faces.live_slot_legality', 0)} over {steps} steps")
    if cen_p.get("faces.count_legality", 0) != 0:
        problems.append("unmerged: the merged key was bumped")

    # 3. the rest of the census
    if set(cen_m) - {"faces.count_legality", "faces.live_chunk",
                     "faces.vertex_decide"}:
        problems.append(f"an unexpected host body fired: {sorted(cen_m)}")
    if cen_m.get("faces.vertex_decide", 0) != steps:
        problems.append(
            f"faces.vertex_decide={cen_m.get('faces.vertex_decide', 0)} over "
            f"{steps} steps, want one per step")
    if cen_m.get("faces.live_chunk", 0) != sum(live):
        problems.append(
            f"faces.live_chunk={cen_m.get('faces.live_chunk', 0)} for "
            f"{sum(live)} live faces over the episode, want one per live face")

    # 2. exactness: the gate's own semantic trace, both routes
    a = json.dumps(merged["steps"], sort_keys=True)
    b = json.dumps(plain["steps"], sort_keys=True)
    if a != b:
        for t, (x, y) in enumerate(zip(merged["steps"], plain["steps"])):
            if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
                for k in sorted(set(x) | set(y)):
                    if x.get(k) != y.get(k):
                        problems.append(
                            f"step {t}: the merge moved {k!r}: "
                            f"{x.get(k)!r} != {y.get(k)!r}")
                break
        else:
            problems.append("the two traces differ outside the step list")

    if problems:
        print("[census] FAIL")
        for p in problems:
            print("  - " + p)
        return 1
    print("[census] PASS: one merged count+legality call per step, the count "
          "callback never fires, and the gate's trace is identical on both "
          "routes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
