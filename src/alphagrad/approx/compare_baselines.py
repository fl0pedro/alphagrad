"""Static-metric comparison: jax.jacfwd/rev, graphax fwd/rev, and each
recorded best sequence from a list of wandb runs.

Produces a single unified table with columns:

    source             | seq_len | flops | bytes_accessed | peak_mem
                       | latency_ns | cossim | frob_residual | seq_repr
                       | in_corridor

For each recorded sequence we also report the rendered human-readable
form (via :mod:`alphagrad.approx.common.render_sequence`) and an
``in_corridor`` flag (cossim ∈ [0.8, 0.9]) so the degenerate
``cossim=1.0`` rows are visually demoted vs genuine approximations.

Unlike :mod:`alphagrad.approx.downstream_train`, this does NOT train
anything — it's the cheap static-metrics sanity table the research
plan calls "Infra 5". Pair with ``alphagrad/run_compare.sh`` for an
sbatched bulk comparison.

Usage:
    uv run alphagrad/src/alphagrad/approx/compare_baselines.py \\
        --example VmappedNeuralNetwork \\
        --wandb-runs ~/dsnn/wandb/run-20260517_073107-j2yl2wn7 \\
                     ~/dsnn/wandb/run-20260517_072719-7lt7oiwm \\
                     ~/dsnn/wandb/run-20260517_054709-k8qzjo23 \\
        --output-json /tmp/compare.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
from graphax import jacve

from alphagrad.approx.common.examples import get_args, get_fn, infer_argnums
from alphagrad.approx.common.render_sequence import render_best_sequences_json
from alphagrad.approx.common.seq_replay import (
    load_best_sequence,
    parse_recorded_seq,
)


_DEFAULT_RUNS = [
    "/Users/assmuth/dsnn/wandb/run-20260517_073107-j2yl2wn7",  # PPO
    "/Users/assmuth/dsnn/wandb/run-20260517_072719-7lt7oiwm",  # MuZero
    "/Users/assmuth/dsnn/wandb/run-20260517_054709-k8qzjo23",  # GFN
]
_DEFAULT_CHANNELS = [
    "best_overall",
    "best_per_channel/flops",
    "best_per_channel/peak_memory",
    "best_per_channel/cosine_sim",
    "best_per_channel/frob_residual",
]
_DEFAULT_BASELINES = ["jax_jacfwd", "jax_jacrev", "graphax_fwd", "graphax_rev"]


def _cossim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two flattened vectors. Returns NaN when either
    vector has zero norm — a zero-norm Jacobian is a degenerate
    elimination path, not a "perfect match" with anything, so we don't
    want the table to read ``cossim = 0`` for both genuine zero error
    and undefined comparisons.
    """
    af, bf = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    na, nb = float(np.linalg.norm(af)), float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def _frob_residual(exact: np.ndarray, approx: np.ndarray) -> float:
    """Relative Frobenius residual ``||exact - approx|| / ||exact||``.
    Returns NaN when ``exact`` has zero norm (the residual is undefined,
    not zero).
    """
    diff = (exact - approx).reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(exact.reshape(-1).astype(np.float64)))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(diff) / denom)


def _flatten_jacobian(jac) -> np.ndarray:
    """Concat all leaves of a Jacobian pytree into a 1-D numpy array."""
    if isinstance(jac, (tuple, list)):
        leaves = [np.asarray(j).reshape(-1) for j in jac]
    else:
        leaves = [np.asarray(jac).reshape(-1)]
    return np.concatenate(leaves)


def _measure(
    jac_fn: Callable,
    args: tuple,
    *,
    measure_latency: bool,
    n_warmup: int = 1,
    n_repeat: int = 10,
) -> dict:
    """Compile ``jac_fn`` against ``args`` and extract static metrics.

    Returns a dict with ``flops``, ``bytes_accessed``, ``peak_memory``,
    ``latency_ns``, plus the raw Jacobian (as a 1-D numpy array) under
    ``"jacobian_flat"`` for cossim / frob computation against the exact
    reference.

    Variance / measurement notes:

    * ``flops`` / ``bytes_accessed`` come from XLA's static
      ``cost_analysis()`` and are deterministic given the compiled
      executable.
    * ``peak_memory`` is the max over ``n_repeat`` runs (or 1 if
      ``measure_latency`` is False). HBM allocation can vary slightly
      across runs on shared GPUs; expect ±5% noise.
    * ``latency_ns`` is the top-quartile mean over ``n_repeat`` samples,
      dropping the slowest 25% (typically the first run still amortising
      JIT warmup despite ``n_warmup``). Coefficient-of-variation is
      ~10–30% on a shared 4-GPU node; multi-run aggregation upstream is
      advised before comparing latency across rows.
    * When ``jax_memory_monitor`` is unavailable both ``peak_memory`` and
      ``latency_ns`` return NaN, not 0.
    """
    compiled = jax.jit(jac_fn).lower(*args).compile()
    cost = compiled.cost_analysis() or {}
    flops = float(cost.get("flops", 0))
    bytes_accessed = float(cost.get("bytes accessed", 0))

    peak_memory = 0.0
    latency_ns = 0.0
    try:
        from jax_memory_monitor import ResourceMonitor  # type: ignore
        for _ in range(n_warmup):
            out = compiled(*args)
            jax.block_until_ready(out)
        samples_t: list[float] = []
        samples_m: list[float] = []
        repeats = n_repeat if measure_latency else 1
        for _ in range(repeats):
            with ResourceMonitor() as monitor:
                out = compiled(*args)
            samples_t.append(float(monitor.stats.get("time", 0.0)))
            samples_m.append(float(monitor.stats.get("memory", 0.0)))
        if measure_latency and samples_t:
            # Top-quartile mean — drops the JIT-warmup outliers; ``repeats``
            # is always >= 1 here so the slice is always non-empty.
            samples_t.sort()
            cutoff = max(1, len(samples_t) // 4)
            top_q = samples_t[cutoff - 1 :]  # smallest top-quartile slice has 1 elem
            latency_ns = (sum(top_q) / len(top_q)) * 1e9
        peak_memory = float(max(samples_m)) if samples_m else float("nan")
    except ImportError:
        # jax_memory_monitor unavailable → can't measure peak HBM /
        # latency. Emit NaN so summary tables don't conflate "unmeasured"
        # with "zero".
        out = compiled(*args)
        jax.block_until_ready(out)
        peak_memory = float("nan")
        latency_ns = float("nan")

    jac_np = _flatten_jacobian(out)
    return {
        "flops": flops,
        "bytes_accessed": bytes_accessed,
        "peak_memory": peak_memory,
        "latency_ns": latency_ns,
        "jacobian_flat": jac_np,
    }


def _build_baseline_fn(
    name: str, target_fn: Callable, argnums: Sequence[int]
) -> Callable:
    if name == "jax_jacfwd":
        return jax.jacfwd(target_fn, argnums=tuple(argnums))
    if name == "jax_jacrev":
        return jax.jacrev(target_fn, argnums=tuple(argnums))
    if name == "graphax_fwd":
        return jacve(target_fn, order="fwd", argnums=tuple(argnums))
    if name == "graphax_rev":
        return jacve(target_fn, order="rev", argnums=tuple(argnums))
    raise ValueError(f"Unknown baseline {name!r}")


def _build_replay_fn(
    run_dir: str,
    channel: str,
    target_fn: Callable,
    argnums: Sequence[int],
    sample_args: tuple,
) -> tuple[Callable, int]:
    """Construct ``jacve(target_fn, order=..., transforms=...)`` for a
    recorded best sequence. Returns (callable, seq_len)."""
    seq = load_best_sequence(run_dir, channel)
    try:
        jaxpr = jax.make_jaxpr(target_fn)(*sample_args)
        axis_sizes: list[int] = []
        for eqn in jaxpr.jaxpr.eqns:
            for out in eqn.outvars:
                if hasattr(out, "aval") and hasattr(out.aval, "shape"):
                    axis_sizes.extend(int(s) for s in out.aval.shape)
    except Exception:
        axis_sizes = []
    order, transforms = parse_recorded_seq(
        seq, axis_sizes=axis_sizes, skip_low_precision_quant=True,
    )
    fn = jacve(
        target_fn, order=order, transforms=transforms,
        argnums=tuple(argnums),
    )
    return fn, len(seq)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example", default="VmappedNeuralNetwork")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--seed", type=int, default=250197)
    p.add_argument(
        "--wandb-runs", nargs="*", default=_DEFAULT_RUNS,
        help="wandb run directories to replay. Default = 3 latest "
        "(PPO j2yl2wn7, MuZero 7lt7oiwm, GFN k8qzjo23).",
    )
    p.add_argument(
        "--channels", nargs="+", default=_DEFAULT_CHANNELS,
        help="best_sequences.json channels to replay per run.",
    )
    p.add_argument(
        "--baselines", nargs="+", default=_DEFAULT_BASELINES,
        help="Baseline gradient sources (no recorded sequence).",
    )
    p.add_argument(
        "--corridor-low", type=float, default=0.8,
        help="Lower bound of the 'in_corridor' flag.",
    )
    p.add_argument(
        "--corridor-high", type=float, default=0.9,
        help="Upper bound of the 'in_corridor' flag.",
    )
    p.add_argument("--measure-latency", action="store_true")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument(
        "--wandb", choices=("online", "offline", "disabled"),
        default="offline",
        help="wandb mode for the static-metrics table.",
    )
    p.add_argument(
        "--wandb-project", default="dsnn-compare-baselines",
        help="wandb project name when --wandb != disabled.",
    )
    p.add_argument(
        "--wandb-entity", default="",
        help="wandb entity (team / user namespace). Empty = personal default.",
    )
    p.add_argument(
        "--name", default=None,
        help="wandb run name (default: 'compare_<example>_<dataset>').",
    )
    args = p.parse_args()

    # Locked: this comparison is only meaningful when the replayed
    # sequences match the model+data they were recorded on. See the
    # equivalent guard in downstream_train.py for the full reasoning.
    if args.example != "VmappedNeuralNetwork" or args.dataset != "mnist":
        raise SystemExit(
            f"compare_baselines is locked to --example VmappedNeuralNetwork "
            f"--dataset mnist (got example={args.example!r}, "
            f"dataset={args.dataset!r}). Replaying recorded best sequences "
            "against a different jaxpr would silently mismatch vertex "
            "indices and produce zero gradients."
        )

    target_fn = get_fn(args.example)
    argnums = tuple(infer_argnums(args.example))
    key = jrand.PRNGKey(args.seed)
    init_args = get_args(args.example, key, dataset=args.dataset)
    weights = list(init_args[2:])
    # Sample a single (x, y) so the lowered Jacobian fns see concrete
    # shapes. The metrics depend on this shape; we report it for clarity.
    from alphagrad.approx.common.examples import data_gen
    sampler = data_gen(args.example, dataset=args.dataset)
    sample_x, sample_y = sampler(jrand.split(key, 4))
    sample_args = (sample_x, sample_y, *weights)

    rows: list[dict] = []

    # Establish the exact reference via jax.jacrev for cossim / frob.
    print("[compare] computing exact reference (jax.jacrev) ...")
    ref_fn = _build_baseline_fn("jax_jacrev", target_fn, argnums)
    ref = _measure(ref_fn, sample_args, measure_latency=args.measure_latency)
    ref_jac = ref["jacobian_flat"]

    def _record(label: str, m: dict, seq_len: int, seq_repr: str | None):
        cs = _cossim(ref_jac, m["jacobian_flat"])
        fr = _frob_residual(ref_jac, m["jacobian_flat"])
        rows.append({
            "source": label,
            "seq_len": seq_len,
            "flops": m["flops"],
            "bytes_accessed": m["bytes_accessed"],
            "peak_memory": m["peak_memory"],
            "latency_ns": m["latency_ns"],
            "cossim": cs,
            "frob_residual": fr,
            "in_corridor": bool(args.corridor_low <= cs <= args.corridor_high),
            "seq_repr": seq_repr or "",
        })

    # Baselines.
    for bname in args.baselines:
        print(f"[compare] baseline {bname} ...")
        try:
            fn = _build_baseline_fn(bname, target_fn, argnums)
            m = _measure(fn, sample_args, measure_latency=args.measure_latency)
            _record(bname, m, seq_len=0, seq_repr=None)
        except Exception as exc:
            print(f"  ! {bname} failed: {exc}")

    # Recorded sequences.
    for run_dir in args.wandb_runs:
        run_name = Path(run_dir).name
        run_tag = run_name.split("-", 1)[-1] if "-" in run_name else run_name
        try:
            reprs = render_best_sequences_json(
                Path(run_dir) / "files" / "best_sequences.json"
            )
        except Exception:
            reprs = {}
        for channel in args.channels:
            label = f"{run_tag}/{channel}"
            print(f"[compare] replay {label} ...")
            try:
                fn, seq_len = _build_replay_fn(
                    run_dir, channel, target_fn, argnums, sample_args,
                )
                m = _measure(
                    fn, sample_args, measure_latency=args.measure_latency,
                )
                _record(label, m, seq_len=seq_len, seq_repr=reprs.get(channel, ""))
            except Exception as exc:
                print(f"  ! {label} failed: {type(exc).__name__}: {exc}")

    # Print table.
    print()
    print(
        f"{'source':<60s} {'seq':>4s}  {'flops':>12s}  "
        f"{'bytes':>12s}  {'peak_mem':>10s}  {'cossim':>8s}  "
        f"{'frob_r':>8s}  {'band':>5s}"
    )
    print("-" * 130)
    for r in rows:
        print(
            f"{r['source']:<60s} {r['seq_len']:>4d}  "
            f"{r['flops']:>12.4g}  {r['bytes_accessed']:>12.4g}  "
            f"{r['peak_memory']:>10.4g}  {r['cossim']:>8.4f}  "
            f"{r['frob_residual']:>8.4f}  "
            f"{'YES' if r['in_corridor'] else '.':>5s}"
        )

    # Drop the raw jacobian_flat key before serialising.
    for r in rows:
        r.pop("jacobian_flat", None)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[compare] wrote {args.output_json}")
    # Log the comparison table to wandb as a Table artifact + scalar
    # summary keys, so the master research-run dashboard shows it.
    if args.wandb != "disabled" and rows:
        import os
        os.environ["WANDB_MODE"] = args.wandb
        import wandb as _wandb
        run_name = args.name or f"compare_{args.example}_{args.dataset}"
        wb = _wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=run_name,
            config={
                "example": args.example,
                "dataset": args.dataset,
                "seed": args.seed,
                "baselines": args.baselines,
                "channels": args.channels,
                "n_rows": len(rows),
            },
            reinit=True,
        )
        # One wandb scalar per (source, metric) pair so the
        # comparison shows up directly in the run summary.
        for r in rows:
            src = r["source"].replace("/", "__")
            for col in (
                "flops", "bytes_accessed", "peak_memory", "latency_ns",
                "cossim", "frob_residual", "seq_len",
            ):
                wb.summary[f"by_source/{src}/{col}"] = r.get(col, float("nan"))
            wb.summary[f"by_source/{src}/in_corridor"] = int(bool(r["in_corridor"]))
        # The full table as a wandb.Table for the dashboard.
        try:
            cols = ["source", "seq_len", "flops", "bytes_accessed",
                    "peak_memory", "latency_ns", "cossim",
                    "frob_residual", "in_corridor", "seq_repr"]
            tbl = _wandb.Table(columns=cols)
            for r in rows:
                tbl.add_data(*[r.get(c) for c in cols])
            wb.log({"compare/table": tbl})
        except Exception as exc:
            print(f"[compare] wandb Table upload failed: {exc}")
        try:
            wb.finish()
        except Exception as exc:
            print(f"[compare] wandb.finish() failed: {exc}")
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[compare] wrote {args.output_csv}")


if __name__ == "__main__":
    main()
