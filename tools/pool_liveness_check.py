#!/usr/bin/env python
"""THE POOLED-MEASUREMENT LIVENESS GATE -- the checker half.

WHY THIS EXISTS.  ``87cdc49`` fixed a fifteen-line regression that had made
``--ray-measure`` -- the production measurement path of EVERY wave-1..4 arm of
``docs/EXPERIMENT_PLAN.md`` -- measure NOTHING AT ALL.  ``4c4d872`` gave
``CpuApproxPool.evaluate`` / ``evaluate_batch`` an ``episode`` field and both
forward it to ``actor.evaluate.remote(...)``, but ``CpuApproximationActor.
evaluate`` -- the ONE hop between the pool and ``CpuApproximationServer.
evaluate``, which had accepted ``episode`` all along -- was never given the
parameter.  Every pooled dispatch therefore died with

    TypeError: got an unexpected keyword argument 'episode'

the pool sentinelled the row, killed the actor, and the rest of the wave came
back ``pool-drained (actor died)``.  Every terminal reward of every env of
every episode was the degenerate sentinel (-1e10 on all six cost channels,
grad_coverage -1, quality 0) and the run still LOOKED like a training run: it
exited 0, it printed health rows, it stepped PPO.

THE POINT IS THAT NO EXISTING GATE SAW IT.  ``tools/ratio_gates.sh`` pins the
importance ratio, ``tools/smoke.sh`` pins finite health metrics -- neither
runs the pool at all, and the smoke's config has no ``--ray-measure``.  The
only trace was ``[SENTINEL]`` lines that nothing reads.  ~177 node-hours of
pure sentinel were one launch away.

WHAT THIS CHECKS.  Two independent parts, either of which is red on the
pre-fix tree:

  ``contract``   A STATIC read of the pool -> actor -> server call chain: every
                 keyword the pool forwards through ``.evaluate.remote(...)``
                 must be a parameter the actor wrapper accepts, and every
                 keyword the wrapper forwards to ``self._impl.<m>(...)`` must
                 be a parameter the server accepts.  Pure ``ast``: no ray, no
                 jax, no import of the package, ~10 ms.  This is the exact
                 shape of the bug and it names the offending kwarg.

  ``verdict``    The LIVE read of a short real ``--ray-measure`` run:
                   (1) the pool actually started            (else MISCONFIGURED)
                   (2) ZERO [SENTINEL] lines of any kind     (else DEAD)
                   (3) >0 terminal plans recorded, and >0 of them carrying an
                       ``actor`` stamp, i.e. measured INSIDE a measure actor
                       (``measure_pool.merge_pool_plan_records`` stamps it)
                   (4) at least one recorded plan carrying a REAL number on a
                       cost channel, and no plan whose six cost channels are
                       all the degenerate sentinel without the frozen-gradient
                       guard having refused it

FALSE POSITIVE ALREADY FOUND AND CLOSED.  The first version of check (4) failed
at HEAD: 16/16 plans came back `sentinelled` with all six cost channels at
-1e10, because the frozen-gradient guard (DEFAULT ON, and right to be) returns
early BEFORE measuring, and an untrained face policy on a 25-vertex graph
samples SKIPs that freeze every trainable leaf at episode 0.  On the reward
vector alone that is indistinguishable from a dead pool.  The gate's own run
therefore passes ``--no-reject-frozen-grads`` so the verdict depends on the
TRANSPORT and not on what a random policy sampled, and (4) now demands
POSITIVE evidence (a real cost number) rather than absence of the sentinel.

A SKIP IS A FAILURE (116c540).  A gate that did not run pins nothing, so the
"could not run" verdict exits non-zero too -- but with its OWN exit code and
its own headline, because "the harness is misconfigured" and "the measurement
is dead" call for opposite responses and conflating them is how a red gate
gets waved through.

Exit codes:
    0   GREEN   -- the pooled measurement path is live
    1   RED     -- MEASUREMENT DEAD (or the wire contract is broken)
    2   RED     -- GATE COULD NOT RUN (harness misconfigured); still a failure
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

# env.SENTINEL_COST is -1e10 and env._SENTINEL_BAD_REWARD stamps it into every
# unbounded cost slot; cpu_approx_pool._SENTINEL_REWARD_VALUE is the same
# number.  Half of it is a threshold no real measurement can reach (a real
# latency_ns / peak_memory / flops reading is >= 0 and the symlog is applied
# downstream of the reward vector).
SENTINEL_THRESHOLD = -5e9

# The six UNBOUNDED cost channels of env.REWARD_NAMES.  Slots 6..10 (quality,
# grad_coverage, fidelity, bkstep_acc, sparsity) are BOUNDED and take their own
# floor on a sentinel, so they are not part of the degeneracy test.
COST_CHANNELS = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
)

EXIT_GREEN = 0
EXIT_DEAD = 1
EXIT_MISCONFIGURED = 2


# ---------------------------------------------------------------------------
# PART 1 -- the static wire contract
# ---------------------------------------------------------------------------

def _methods_of(path: str) -> dict:
    """{class_name: {method_name: (positional, kwonly, has_var_kw)}} by ast."""
    with open(path, "r") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out: dict = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {}
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = item.args
            pos = [p.arg for p in (list(a.posonlyargs) + list(a.args))]
            if pos and pos[0] in ("self", "cls"):
                pos = pos[1:]
            kwonly = [p.arg for p in a.kwonlyargs]
            methods[item.name] = (pos, kwonly, a.kwarg is not None)
        out[node.name] = methods
    return out


def _remote_calls(path: str) -> list:
    """Every ``<anything>.<method>.remote(...)`` call: (method, kwargs, npos, line)."""
    with open(path, "r") as fh:
        tree = ast.parse(fh.read(), filename=path)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "remote"):
            continue
        inner = f.value
        if not isinstance(inner, ast.Attribute):
            continue
        kws = [k.arg for k in node.keywords]
        found.append((inner.attr, kws, len(node.args), node.lineno))
    return found


def _impl_calls(path: str) -> list:
    """Every ``self._impl.<method>(...)`` call: (method, kwargs, npos, line)."""
    with open(path, "r") as fh:
        tree = ast.parse(fh.read(), filename=path)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        v = f.value
        if not (isinstance(v, ast.Attribute) and v.attr == "_impl"
                and isinstance(v.value, ast.Name) and v.value.id == "self"):
            continue
        kws = [k.arg for k in node.keywords]
        found.append((f.attr, kws, len(node.args), node.lineno))
    return found


def _check_hop(calls, callee_methods, caller_label, callee_label):
    """Every forwarded kwarg must be a parameter the callee accepts."""
    problems = []
    checked = 0
    for method, kws, npos, line in calls:
        sig = callee_methods.get(method)
        if sig is None:
            problems.append(
                f"{caller_label}:{line}: calls `{method}(...)` but "
                f"{callee_label} has NO method of that name")
            continue
        pos, kwonly, has_var_kw = sig
        accepted = set(pos) | set(kwonly)
        checked += 1
        if any(k is None for k in kws):
            # ``**something`` -- the names are not statically knowable.
            continue
        for k in kws:
            if k not in accepted and not has_var_kw:
                problems.append(
                    f"{caller_label}:{line}: forwards `{k}=` to "
                    f"{callee_label}.{method}(), which does NOT accept it -- "
                    f"every such dispatch dies with "
                    f"TypeError: got an unexpected keyword argument '{k}' "
                    f"(accepted: {sorted(accepted)})")
        if npos > len(pos):
            problems.append(
                f"{caller_label}:{line}: passes {npos} positional args to "
                f"{callee_label}.{method}(), which takes {len(pos)}")
    return checked, problems


def cmd_contract(args) -> int:
    root = args.repo
    pool_py = os.path.join(root, "src/alphagrad/approx/cpu_approx_pool.py")
    actors_py = os.path.join(root, "src/alphagrad/approx/cpu_approx_actors.py")
    worker_py = os.path.join(root, "src/alphagrad/approx/cpu_approx_worker.py")
    for p in (pool_py, actors_py, worker_py):
        if not os.path.exists(p):
            print(f"CONTRACT: CANNOT RUN -- {p} does not exist")
            return EXIT_MISCONFIGURED

    actor_methods = _methods_of(actors_py).get("CpuApproximationActor")
    server_methods = _methods_of(worker_py).get("CpuApproximationServer")
    if not actor_methods or not server_methods:
        print("CONTRACT: CANNOT RUN -- CpuApproximationActor or "
              "CpuApproximationServer not found by ast")
        return EXIT_MISCONFIGURED

    n1, p1 = _check_hop(_remote_calls(pool_py), actor_methods,
                        "cpu_approx_pool.py", "CpuApproximationActor")
    n2, p2 = _check_hop(_impl_calls(actors_py), server_methods,
                        "cpu_approx_actors.py", "CpuApproximationServer")

    print(f"CONTRACT: pool -> actor      {n1} .remote() call site(s) checked")
    print(f"CONTRACT: actor -> server    {n2} self._impl call site(s) checked")
    if n1 == 0 or n2 == 0:
        print("CONTRACT: CANNOT RUN -- one of the two hops has NO call sites; "
              "the ast pattern no longer matches the code it is meant to pin")
        return EXIT_MISCONFIGURED
    if p1 or p2:
        print("CONTRACT: RED -- the pool/actor/server wire contract is BROKEN")
        for line in p1 + p2:
            print(f"  ! {line}")
        return EXIT_DEAD
    print("CONTRACT: ok -- every forwarded kwarg is accepted by its callee")
    return EXIT_GREEN


# ---------------------------------------------------------------------------
# PART 2 -- the live verdict
# ---------------------------------------------------------------------------

_SENTINEL_RE = re.compile(r"\[SENTINEL[^\]]*\]")
_RAY_MEASURE_RE = re.compile(r"\[ray-measure\]\s+(\d+)\s+actors")
_PLANLOG_RE = re.compile(r"\[plan-log ep(\d+)\]\s+(.*)")
_MISCONFIG_MARKERS = (
    "--ray-measure needs ALPHAGRAD_BATCHED_CALLBACK=1",
    "The current node timed out during startup",
    "RAY_TMPDIR",
)


def _read_text(path: str) -> str:
    with open(path, "r", errors="replace") as fh:
        return fh.read()


def _plan_records(path: str) -> list:
    recs = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def cmd_verdict(args) -> int:
    if not os.path.exists(args.log):
        print(f"VERDICT: CANNOT RUN -- run log {args.log} does not exist; "
              f"the gate's own training run never produced output")
        return EXIT_MISCONFIGURED
    log = _read_text(args.log)

    # ---- (0) did the harness itself work? ---------------------------------
    if "--ray-measure needs ALPHAGRAD_BATCHED_CALLBACK=1" in log:
        print("VERDICT: HARNESS MISCONFIGURED -- the run refused to start:")
        print("         --ray-measure needs ALPHAGRAD_BATCHED_CALLBACK=1")
        print("         Set ALPHAGRAD_BATCHED_CALLBACK=1 before ppo.py. "
              "NOTHING about the measurement path was tested.")
        return EXIT_MISCONFIGURED

    m = _RAY_MEASURE_RE.search(log)
    if m is None:
        print("VERDICT: HARNESS MISCONFIGURED -- no `[ray-measure] N actors` "
              "line in the run log: the measure pool never started, so this "
              "run tested NOTHING about pooled measurement.")
        for marker in _MISCONFIG_MARKERS[1:]:
            for line in log.splitlines():
                if marker in line:
                    print(f"         possible cause: {line.strip()[:160]}")
                    break
        print(f"         run rc was {args.rc}")
        return EXIT_MISCONFIGURED
    n_actors = int(m.group(1))
    print(f"VERDICT: pool started -- {m.group(0)}")

    planlog_lines = _PLANLOG_RE.findall(log)
    if not planlog_lines:
        print("VERDICT: HARNESS MISCONFIGURED -- the run printed no "
              "`[plan-log epN]` drain line, so --plan-log was not on and this "
              "gate has no liveness evidence to read.")
        return EXIT_MISCONFIGURED

    rc = EXIT_GREEN
    fatal = []

    # ---- (1) did the run itself survive? ----------------------------------
    if int(args.rc) != 0:
        fatal.append(f"the training run exited {args.rc} (not 0)")

    # ---- (2) the sentinel rate must be exactly zero ------------------------
    sent_lines = [ln for ln in log.splitlines() if _SENTINEL_RE.search(ln)]
    kinds: dict = {}
    for ln in sent_lines:
        for kind in ("dispatch-error", "pool-drained", "pool-timeout",
                     "batch timeout", "actor-error", "other-error",
                     "measure-exception"):
            if kind in ln:
                kinds[kind] = kinds.get(kind, 0) + 1
                break
        else:
            kinds["other"] = kinds.get("other", 0) + 1
    print(f"VERDICT: [SENTINEL] lines = {len(sent_lines)}  by kind={kinds}")
    if sent_lines:
        fatal.append(
            f"{len(sent_lines)} [SENTINEL] line(s) -- a nonzero dispatch-error "
            f"/ pool-drained rate means rows were NOT measured; by kind {kinds}")
        for ln in sent_lines[:6]:
            # Quote from the tag onward: a tqdm bar or an ANSI actor prefix
            # shares the line and the matched text is what matters.
            mm = _SENTINEL_RE.search(ln)
            print(f"  ! {ln[mm.start():].strip()[:200]}")
        if len(sent_lines) > 6:
            print(f"  ! ... and {len(sent_lines) - 6} more")

    # ---- (3) terminal plans measured INSIDE a measure actor ---------------
    for ep, rest in planlog_lines:
        print(f"VERDICT: [plan-log ep{ep}] {rest.strip()[:200]}")
    pooled_total = 0
    for _ep, rest in planlog_lines:
        mm = re.search(r"pooled=(-?\d+)", rest)
        if mm:
            pooled_total += int(mm.group(1))

    if not os.path.exists(args.plan_log):
        fatal.append(
            f"the plan log {args.plan_log} was never written: ZERO terminal "
            f"plans were recorded, i.e. the measurement callback never ran")
        recs = []
    else:
        try:
            recs = _plan_records(args.plan_log)
        except Exception as exc:
            print(f"VERDICT: CANNOT RUN -- plan log unreadable: {exc!r}")
            return EXIT_MISCONFIGURED

    n_from_actor = sum(1 for r in recs if r.get("actor") is not None)
    actors_seen = sorted({r.get("actor") for r in recs
                          if r.get("actor") is not None})
    n_sentinelled = sum(1 for r in recs if r.get("sentinelled"))
    print(f"VERDICT: plan records = {len(recs)}  "
          f"measured in a measure actor = {n_from_actor}  "
          f"actor ids = {actors_seen}  "
          f"drained as pooled = {pooled_total}  "
          f"sentinelled = {n_sentinelled}")

    if not recs:
        fatal.append(
            "ZERO terminal plans were recorded -- under --ray-measure the "
            "measurement callback runs inside the actors, so an empty plan "
            "log means NOTHING WAS MEASURED (this is exactly the 87cdc49 "
            "failure mode)")
    elif n_from_actor == 0:
        fatal.append(
            f"{len(recs)} terminal plan(s) recorded but NONE carries an "
            f"`actor` stamp: not one of them was measured inside a measure "
            f"actor, so the pooled path is dead even though the trainer "
            f"measured locally")
    elif len(actors_seen) < n_actors:
        print(f"  (note) only {len(actors_seen)} of {n_actors} actors "
              f"contributed a record; not fatal at this size, but a "
              f"persistently silent actor is worth a look")

    # ---- (4) degenerate terminal rewards ----------------------------------
    # POSITIVE EVIDENCE, not merely absence of the sentinel: at least one
    # recorded plan must carry a real number on at least one cost channel.
    # A record is DEGENERATE when all six unbounded cost channels sit at the
    # sentinel; that is what a dead pool writes, and it is ALSO what the
    # frozen-gradient guard writes when it refuses a plan before measuring it
    # -- which is why the gate's own run turns that guard off (see
    # tools/pool_liveness_gate.sh). A guard-sentinelled record is reported
    # separately and never counted as evidence that a cost was measured.
    n_degenerate = 0
    n_degenerate_live = 0
    n_measured = 0
    n_scored = 0
    first_degenerate = None
    first_measured = None
    for r in recs:
        names = list(r.get("reward_names") or ())
        vals = list(r.get("rewards") or ())
        if not names or len(names) != len(vals):
            continue
        idx = {n: i for i, n in enumerate(names)}
        cost = [vals[idx[c]] for c in COST_CHANNELS if c in idx]
        if not cost:
            continue
        n_scored += 1
        if all(float(v) <= SENTINEL_THRESHOLD for v in cost):
            n_degenerate += 1
            if not r.get("sentinelled"):
                n_degenerate_live += 1
                if first_degenerate is None:
                    first_degenerate = dict(zip(COST_CHANNELS, cost))
        else:
            n_measured += 1
            if first_measured is None:
                first_measured = dict(zip(COST_CHANNELS, cost))
    print(f"VERDICT: cost vectors: {n_measured} REAL / {n_degenerate} "
          f"degenerate (of {n_scored} scored; {n_degenerate_live} degenerate "
          f"WITHOUT being guard-sentinelled)")
    if first_measured is not None:
        print(f"  (sample measured cost vector) {first_measured}")
    if n_degenerate_live:
        fatal.append(
            f"{n_degenerate_live} recorded plan(s) carry the DEGENERATE "
            f"SENTINEL on every cost channel without being guard-sentinelled: "
            f"{first_degenerate}")
    if recs and n_measured == 0:
        if n_sentinelled == len(recs):
            fatal.append(
                "EVERY recorded plan was refused by the frozen-gradient guard "
                "before its cost was measured, so NOT ONE real cost vector "
                "exists. If this gate's own run passed --no-reject-frozen-"
                "grads, that guard should be off and this is a real failure; "
                "if a caller re-armed it, the run is misconfigured for this "
                "gate rather than proving the pool dead")
        else:
            fatal.append(
                f"{len(recs)} plan(s) recorded but NOT ONE carries a real "
                f"number on any of the {len(COST_CHANNELS)} cost channels")

    if fatal:
        print()
        print("POOLED-MEASUREMENT LIVENESS GATE: RED -- MEASUREMENT IS DEAD")
        for f in fatal:
            print(f"  FAIL: {f}")
        print("  Under --ray-measure this is the production measurement path "
              "of every wave-1..4 arm; a run in this state still exits 0, "
              "still prints health rows and measures NOTHING.")
        rc = EXIT_DEAD
    else:
        print()
        print("POOLED-MEASUREMENT LIVENESS GATE: GREEN -- "
              f"{n_from_actor}/{len(recs)} terminal plans measured inside a "
              f"measure actor, 0 sentinels, 0 degenerate cost vectors")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("contract", help="static pool->actor->server kwarg check")
    c.add_argument("--repo", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    c.set_defaults(fn=cmd_contract)

    v = sub.add_parser("verdict", help="read a short --ray-measure run")
    v.add_argument("--log", required=True, help="the run's stdout+stderr log")
    v.add_argument("--plan-log", required=True, help="the run's --plan-log JSONL")
    v.add_argument("--rc", default="0", help="the run's exit code")
    v.set_defaults(fn=cmd_verdict)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
