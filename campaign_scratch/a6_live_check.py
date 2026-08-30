#!/usr/bin/env python3
"""A6: read a LIVE run's plan log and REPLAY its losers.

Two claims are checked against a JSONL written by a real training run:

  (1) STRUCTURE -- every record carries the schema's fields, one record per
      terminal plan per episode, and every record decodes back to the four
      integer buffers `_callback` consumes.
  (2) REPLAY -- the recorded wire, fed straight back into `env._callback`,
      reproduces the DETERMINISTIC reward channels the record stores. The
      wall-clock channel is excluded (a timing does not repeat) and named.

The plans replayed are chosen to be LOSERS: the worst quality first, plus
every sentinelled (frozen-gradient) record, because those are exactly the
rows the Pareto front drops and the ones X3 exists to attribute.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import numpy as np                                              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--example", default="NeuralNetwork")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replay", type=int, default=3)
    a = ap.parse_args()

    from alphagrad.approx.common import plan_log as plog
    recs = plog.read_records(a.log)
    print(f"[check] {len(recs)} record(s) in {a.log}")
    if not recs:
        print("FAIL: the plan log is EMPTY")
        return 1

    # ---- (1) structure ---------------------------------------------------
    need = ("schema", "episode", "plan_index", "order", "shape", "rules",
            "faces", "rewards", "reward_names", "requested", "applied",
            "skipped", "idempotent_noop", "repaired", "coverage",
            "sentinelled", "counts_from_trace", "vertex_convention",
            "replayable", "n_live_faces")
    bad = 0
    for r in recs:
        miss = [k for k in need if k not in r]
        if miss:
            print(f"FAIL: record {r.get('plan_index')} missing {miss}")
            bad += 1
        if len(r.get("rewards", ())) != 11:
            print(f"FAIL: record {r.get('plan_index')} has "
                  f"{len(r.get('rewards', ()))} reward slots, want 11")
            bad += 1
        if not r.get("replayable", False):
            print(f"FAIL: record {r.get('plan_index')} is not replayable")
            bad += 1
    eps = sorted({int(r["episode"]) for r in recs})
    per_ep = {e: sum(1 for r in recs if int(r["episode"]) == e) for e in eps}
    print(f"[check] episodes={eps} records/episode={per_ep}")
    print(f"[check] sentinelled={sum(1 for r in recs if r['sentinelled'])} "
          f"counts_from_trace={sum(1 for r in recs if r['counts_from_trace'])}"
          f"/{len(recs)}")
    n_cov = sum(1 for r in recs if r["coverage"].get("measured"))
    print(f"[check] coverage census measured on {n_cov}/{len(recs)}")
    r0 = recs[0]
    print(f"[check] example record: order_len={len(r0['order'])} "
          f"n_live_faces={r0['n_live_faces']} "
          f"requested={r0['requested']} applied={r0['applied']} "
          f"noop={r0['idempotent_noop']}")
    print(f"[check] coverage[0]={ {k: v for k, v in r0['coverage'].items() if not isinstance(v, list)} }")
    if bad:
        print(f"STRUCTURE: {bad} FAILURE(S)")
        return 1
    print("STRUCTURE: ok")

    # ---- (2) replay ------------------------------------------------------
    # IMPORT SIDE EFFECT, NOT A STYLE CHOICE. landscape_map.py line 278 is
    #     ARGS = make_argparser().parse_args()
    # at MODULE level, so importing it parses THIS script's argv and exits 2
    # on the first flag it does not recognise. sys.argv is therefore replaced
    # with what landscape_map should see BEFORE the import, and its own
    # module-level ARGS is the env configuration.
    import jax                                                 # noqa: F401
    sys.argv = [sys.argv[0],
                "--example", a.example, "--seed", str(a.seed),
                "--dataset", "none", "--cmp-type", "flops",
                "--mem-type", "peak_memory", "--out-dir", "/tmp/a6_replay"]
    from alphagrad.approx.tools import landscape_map as LM
    import alphagrad.approx.env as envmod
    from alphagrad.approx.env import REWARD_INDEX

    env, eval_samples, _cj = LM.build_env(LM.ARGS)
    print(f"[replay] env built: {len(env.valid_vertices)} valid vertices")

    qi = REWARD_INDEX["cosine_sim"]
    order_bad = sorted(range(len(recs)),
                       key=lambda i: (not recs[i]["sentinelled"],
                                      plog.unjson_float(recs[i]["rewards"][qi])))
    # Channels that are NOT a wall-clock reading. latency_ns is a timing and
    # is not expected to repeat; peak_memory is a runtime high-water mark and
    # is only compared when it was actually measured identically.
    skip = {REWARD_INDEX["latency_ns"]}
    fails = 0
    for i in order_bad[: a.replay]:
        r = recs[i]
        o, rules, faces, skips = plog.decode_wires(r)
        _t, _e, rep = envmod._callback(
            env.config, env.args, env.consts, o, rules, faces, skips, len(o),
            *(eval_samples or ()))
        rep = np.asarray(rep, dtype=np.float64)
        rec_r = np.asarray([plog.unjson_float(x) for x in r["rewards"]],
                           dtype=np.float64)
        diff = [(j, rec_r[j], rep[j]) for j in range(len(rec_r))
                if j not in skip and not (
                    rec_r[j] == rep[j]
                    or (np.isnan(rec_r[j]) and np.isnan(rep[j])))]
        tag = "SENTINELLED" if r["sentinelled"] else "loser"
        if diff:
            fails += 1
            print(f"[replay] ep{r['episode']} #{r['plan_index']} ({tag}) "
                  f"MISMATCH on {[(r['reward_names'][j], rc, rp) for j, rc, rp in diff]}")
        else:
            print(f"[replay] ep{r['episode']} #{r['plan_index']} ({tag}) "
                  f"q={rec_r[qi]:.6g} order={list(o)[:6]}... "
                  f"faces={r['n_live_faces']} -> EXACT MATCH on all "
                  f"{len(rec_r) - len(skip)} non-timing channels")
    print(f"REPLAY: {a.replay - fails}/{a.replay} exact")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
