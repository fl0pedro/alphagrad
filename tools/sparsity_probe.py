"""A7: what does the SPARSITY channel actually say, and does it discriminate?

The channel (reward slot 10) is ``clip(1 - stored_bytes(approx) /
stored_bytes(exact), -1, +1)`` over the accumulated Jacobians of the SAME
elimination order. Before anybody trains on it, three questions have to be
answered with numbers on the real measurement path:

  1. IS THE IDENTITY CASE EXACTLY 1.0? It must be, by construction; a
     measured 1.0 is what proves the two arms really are the same order.
  2. DOES IT DISCRIMINATE? If DIAG lands on ~0 faces on TLM and COMPRESS is
     the only structural lever, the ratio may be nearly constant -- in which
     case the channel carries no signal and that is worth knowing FIRST.
  3. WHAT DOES IT REWARD? Sparsity is maximised by DELETING computation. The
     all-SKIP plan and the known gradient-freezing single-face skips are
     measured HERE, so the failure
     mode is a number in a table rather than a discovery six weeks into a
     campaign.

It also reports the PEARSON CORRELATION between the sparsity ratio and the
latency ratio: if they move together, sparsity adds nothing the cost channels
do not already carry, and the honest recommendation is to log it, not train
on it.

THE GRADIENT-COVERAGE GUARD WAS REMOVED 2026-09-03 (owner ruling 2026-09-03, ticket dsnn-3qm.15). Until then
this probe disarmed it to see what sparsity would have paid the plans it
refused; nothing refuses them now, so the sparsity column IS the verdict.
THIS IS A DIAGNOSTIC CONFIGURATION AND NOT A TRAINING ONE.

Measurement is the trainer's own ``env._callback`` on plans built by
``landscape_map``'s own builders -- there is no second implementation of
either here.

USAGE (CPU validation first, always)::

    JAX_PLATFORMS=cpu uv run --no-sync python tools/sparsity_probe.py \
      --example Helmholtz --dataset none --out /tmp/a7_helm.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# ARGUMENTS, IN TWO PASSES, AND THE ORDER IS LOAD-BEARING.
#
# `landscape_map` parses sys.argv AT MODULE LEVEL (its `ARGS = make_argparser()
# .parse_args()`) precisely so it can export the import-time env knobs
# alphagrad.approx.env freezes at ITS first import. So this probe cannot add
# options to that parser after the fact -- by the time the import returns, the
# parse has already happened and rejected them. Instead: consume THIS script's
# own options first, hand landscape_map the remainder, and let it parse and
# configure exactly as it does when run directly.
#
# THIS SCRIPT'S OWN OPTIONS (they do not appear in --help, which is
# landscape_map's):
#   --out PATH            CSV for the rows
#   --budgets 1,5,all     ladder budgets per op
#   --single-skip K:F     one single-face SKIP plan; repeatable. These are the
#                         known gradient-freezing plans and they are the point
#                         of the table.
#   --repeat-identity N   re-measure identity N times; the spread is the drift
#                         floor every other ratio is read against.
# ---------------------------------------------------------------------------
_MY = argparse.ArgumentParser(add_help=False)
_MY.add_argument("--out", default="")
_MY.add_argument("--budgets", default="5,all")
_MY.add_argument("--single-skip", action="append", default=[])
_MY.add_argument("--repeat-identity", type=int, default=2)
MINE, _REST = _MY.parse_known_args()
sys.argv = [sys.argv[0]] + _REST

# Exported BEFORE landscape_map is imported, for the same reason it exports
# its own: env.py reads these at import.
# setdefault, not assignment: an A/B run that wants the channel OFF sets it
# in the launcher and wins.
os.environ.setdefault("ALPHAGRAD_SPARSITY", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                            # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import alphagrad.approx.env as envmod                         # noqa: E402
from alphagrad.approx.env import (                            # noqa: E402
    REWARD_INDEX, consume_per_face_stats, consume_sparsity_stats,
)
# THIS IMPORT PARSES sys.argv and sets the import-time knobs.
from alphagrad.approx.tools import landscape_map as LM        # noqa: E402


ROW_FIELDS = [
    "plan_id", "op", "budget", "n_faces_approx", "total_live_faces",
    "sparsity_ratio", "sparsity_cells_ratio", "sparsity_channel",
    "stored_bytes_approx", "stored_bytes_exact",
    "stored_edges_approx", "stored_edges_exact",
    "latency_ns", "latency_ratio", "peak_memory", "peak_ratio",
    # DETERMINISTIC cost channels. At CPU scale the latency ratio of
    # identity-against-itself is 0.33-0.47, i.e. the drift floor is a
    # factor of 2-3 and any correlation against latency is noise. flops
    # and bytes_accessed come from XLA cost analysis and are exactly
    # reproducible, so they are the honest thing to correlate against.
    "flops", "flops_ratio", "bytes_accessed", "bytes_ratio",
    "quality",
    "applied", "skipped", "wall_s", "fallback_traces",
]


def main() -> int:
    args = LM.ARGS
    env, eval_samples, _ = LM.build_env(args)
    order = LM.rev_order(env)
    print(f"[a7] example={args.example} vertices={len(order)} "
          f"max_faces={envmod.MAX_FACES}", flush=True)

    # ---- the plan set -------------------------------------------------
    plans: list[tuple[str, str, str, dict]] = []
    plans.append(("identity", "identity", "0",
                  dict(zip(("specs", "face_specs", "face_skips"),
                           LM.empty_plan(len(order))),
                       n_faces_approx=0, total_live_faces=-1)))
    budgets = [b.strip() for b in MINE.budgets.split(",") if b.strip()]
    for op in ("diag", "compress", "quant"):
        for b in budgets:
            bb = "all" if b == "all" else int(b)
            try:
                plans.append((f"{op}@{b}", op, b,
                              LM.build_ladder_plan(env, order, op, bb, args)))
            except Exception as exc:                       # noqa: BLE001
                print(f"[a7] plan {op}@{b} could not be BUILT "
                      f"({type(exc).__name__}: {str(exc)[:120]}) -- recorded "
                      f"as missing, never as a bad score", flush=True)
    plans.append(("skip@all", "skip", "all",
                  LM.build_ladder_plan(env, order, "skip", "all", args)))
    for spec in MINE.single_skip:
        k, f = spec.replace("/", ":").replace("k", "").replace("f", "").split(":")
        plans.append((f"skip@k{k}/f{f}", "skip", "1",
                      LM.build_skip_only_plan(env, order, [(int(k), int(f))])))
    plans.extend([("identity", "identity", "0",
                   dict(zip(("specs", "face_specs", "face_skips"),
                            LM.empty_plan(len(order))),
                        n_faces_approx=0, total_live_faces=-1))
                  for _ in range(max(0, MINE.repeat_identity - 1))])

    # ---- measure ------------------------------------------------------
    rows = []
    for plan_id, op, budget, plan in plans:
        consume_per_face_stats()
        consume_sparsity_stats()
        t0 = time.perf_counter()
        try:
            _, _, reward = envmod._callback(
                env.config, env.args, env.consts,
                jnp.asarray(order),
                jnp.asarray(plan["specs"]),
                jnp.asarray(plan["face_specs"]),
                jnp.asarray(plan["face_skips"]),
                int(len(order)), *eval_samples)
        except Exception as exc:                           # noqa: BLE001
            print(f"[a7] {plan_id}: MEASUREMENT FAILED "
                  f"({type(exc).__name__}: {str(exc)[:160]}) -- dropped as "
                  f"missing data, never scored worst", flush=True)
            continue
        wall = time.perf_counter() - t0
        r = np.asarray(reward, dtype=np.float64)
        sp = consume_sparsity_stats()
        st = consume_per_face_stats()
        last = sp.get("last") or {}
        rows.append({
            "plan_id": plan_id, "op": op, "budget": budget,
            "n_faces_approx": int(plan.get("n_faces_approx", 0)),
            "total_live_faces": int(plan.get("total_live_faces", -1)),
            "sparsity_ratio": sp["ratio_mean"],
            "sparsity_cells_ratio": sp["cells_ratio_mean"],
            "sparsity_channel": float(r[REWARD_INDEX["sparsity"]]),
            "stored_bytes_approx": last.get("approx_bytes"),
            "stored_bytes_exact": last.get("exact_bytes"),
            "stored_edges_approx": last.get("approx_edges"),
            "stored_edges_exact": last.get("exact_edges"),
            "latency_ns": float(-r[REWARD_INDEX["latency_ns"]]),
            "latency_ratio": float("nan"),
            "peak_memory": float(-r[REWARD_INDEX["peak_memory"]]),
            "peak_ratio": float("nan"),
            "flops": float(-r[REWARD_INDEX["flops"]]),
            "flops_ratio": float("nan"),
            "bytes_accessed": float(-r[REWARD_INDEX["bytes_accessed"]]),
            "bytes_ratio": float("nan"),
            "quality": float(r[REWARD_INDEX["quality"]]),
            "applied": int(st.get("applied", 0)),
            "skipped": int(st.get("skipped", 0)),
            "wall_s": wall,
            "fallback_traces": int(sp.get("fallback_traces", 0)),
        })
        print(f"[a7] {plan_id:22s} stored_ratio="
              f"{rows[-1]['sparsity_ratio']:.6f} "
              f"cells={rows[-1]['sparsity_cells_ratio']:.6f} "
              f"chan={rows[-1]['sparsity_channel']:+.4f} "
              f"lat={rows[-1]['latency_ns']:.4g} "
              f"wall={wall:.1f}s", flush=True)

    # ---- ratios against the FIRST identity row, paired in-process ------
    ident = [r for r in rows if r["plan_id"] == "identity"]
    if ident:
        lat0 = ident[0]["latency_ns"] or float("nan")
        pk0 = ident[0]["peak_memory"] or float("nan")
        fl0 = ident[0]["flops"] or float("nan")
        by0 = ident[0]["bytes_accessed"] or float("nan")
        for r in rows:
            r["latency_ratio"] = r["latency_ns"] / lat0 if lat0 else float("nan")
            r["peak_ratio"] = r["peak_memory"] / pk0 if pk0 else float("nan")
            r["flops_ratio"] = r["flops"] / fl0 if fl0 else float("nan")
            r["bytes_ratio"] = (r["bytes_accessed"] / by0 if by0
                                else float("nan"))
        # THE DRIFT FLOOR, stated as a number: identity measured against
        # itself. Every latency ratio below must be read against THIS.
        if len(ident) > 1:
            print(f"[a7] DRIFT FLOOR: identity-vs-identity latency ratio = "
                  f"{ident[-1]['latency_ns'] / lat0:.4f} "
                  f"(sparsity ratio {ident[-1]['sparsity_ratio']!r} -- the "
                  f"sparsity tally is STATIC and repeats exactly)", flush=True)

    # ---- does it discriminate, and is it redundant? --------------------
    sr = np.array([r["sparsity_ratio"] for r in rows], dtype=np.float64)
    lr = np.array([r["latency_ratio"] for r in rows], dtype=np.float64)
    ok = np.isfinite(sr) & np.isfinite(lr)
    print("\n================ A7 SPARSITY SUMMARY ================")
    if ok.sum() >= 2:
        spread = float(np.nanmax(sr[ok]) - np.nanmin(sr[ok]))
        print(f"stored-byte ratio: min={np.nanmin(sr[ok]):.6f} "
              f"max={np.nanmax(sr[ok]):.6f} spread={spread:.6f} "
              f"n={int(ok.sum())}")
        print("DISCRIMINATES" if spread > 1e-3 else
              "DOES NOT DISCRIMINATE (flat to <1e-3 across every plan tested)")
        for nm in ("latency_ratio", "flops_ratio", "bytes_ratio",
                   "peak_ratio"):
            other = np.array([r[nm] for r in rows], dtype=np.float64)
            m = ok & np.isfinite(other)
            if m.sum() >= 3 and np.nanstd(sr[m]) > 0 and np.nanstd(other[m]) > 0:
                rho = float(np.corrcoef(sr[m], other[m])[0, 1])
                note = ("  <-- REDUNDANT at this correlation: log it, do not "
                        "train on it" if abs(rho) > 0.9 else "")
                print(f"pearson(sparsity_ratio, {nm:14s}) = {rho:+.4f} "
                      f"over {int(m.sum())} plans{note}")
    ident_rows = [r for r in rows if r["plan_id"] == "identity"]
    for r in ident_rows:
        print(f"identity stored ratio = {r['sparsity_ratio']!r} "
              f"(MUST be exactly 1.0), channel = {r['sparsity_channel']!r} "
              f"(MUST be exactly 0.0)")
    if rows:
        best = max(rows, key=lambda r: (r["sparsity_channel"]
                                        if np.isfinite(r["sparsity_channel"])
                                        else -9))
        print(f"\nHIGHEST SPARSITY SCORE: {best['plan_id']} "
              f"= {best['sparsity_channel']:+.4f}  (nothing refuses it: the "
              f"coverage guard was removed 2026-09-03)")

    if MINE.out:
        with open(MINE.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in ROW_FIELDS})
        print(f"\nrows -> {MINE.out}")
    print(json.dumps({"n_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
