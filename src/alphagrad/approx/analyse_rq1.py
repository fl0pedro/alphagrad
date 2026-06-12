"""Aggregate RQ1 sanity-check results.

After ``run_rq1_ve_only.sh`` runs the 3×3×3 sweep, this script:

1. Loads each variant's ``best_sequences.json`` from its wandb dir.
2. Compares the best-found FMA count against the ``graphax.jacve(order="rev")``
   baseline computed locally for the same example.
3. Per (example, cmp-type), computes the median best-FMA across seeds
   — RQ1's success criterion is that this median ≤ graphax_rev for
   every (example, cmp-type) pair.
4. Computes the top-5 elimination-order overlap (Jaccard index) across
   cmp-types within each example — the second success criterion
   is that the policies converge to a similar optimum regardless of
   *which* cost proxy drove the reward.

Outputs ``rq1_summary.csv`` and prints a Markdown-style table on stdout.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


# Locked to the model the rest of the research plan trains against
# (RQ2..RQ5, downstream MNIST eval). Do not add other examples here.
_EXAMPLES = ("VmappedNeuralNetwork",)
_DATASET = "mnist"
_CMP_TYPES = ("graphax", "flops", "latency")
_REWARD_TO_CMP = {
    "graphax": "muls_adds_fmas",
    "flops": "flops",
    "latency": "latency_ns",
}


def _baseline_fma(example: str, dataset: str | None = None) -> float | None:
    """Compute the graphax.jacve(order="rev") FMA count for an example.

    Returns None on failure (missing example, JAX import error, etc.).
    """
    try:
        import jax
        import jax.random as jrand
        from graphax import jacve
        from alphagrad.approx.common.examples import (
            get_args,
            get_fn,
            infer_argnums,
        )

        target_fn = get_fn(example)
        argnums = tuple(infer_argnums(example))
        key = jrand.PRNGKey(0)
        args = get_args(example, key, dataset=dataset)
        # ``count_ops=True`` returns int counters that on large graphs
        # overflow int32 (jax's default for int literals). Force x64.
        from jax import config as _jax_config
        _jax_config.update("jax_enable_x64", True)
        compiled = jax.jit(
            jacve(target_fn, order="rev", argnums=argnums, count_ops=True)
        )
        _, aux = compiled(*args)
        return float(
            int(aux.get("adds", 0)) + int(aux.get("muls", 0))
            + int(aux.get("fmas", 0))
        )
    except Exception as exc:
        print(f"  [warn] baseline FMA for {example!r} failed: {exc}")
        return None


def _load_best_sequence(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _parse_tag(name: str) -> tuple[str, str, int] | None:
    """Extract ``(example, cmp_type, seed)`` from a wandb run dir name."""
    m = re.search(r"rq1_(?P<ex>[^_]+(?:_\w+)*?)_(?P<ct>graphax|flops|latency)_seed(?P<sd>\d+)", name)
    if not m:
        return None
    return m.group("ex"), m.group("ct"), int(m.group("sd"))


def _flatten_order(seq: Sequence) -> tuple[int, ...]:
    """Sequence of vertex IDs the agent eliminated, in order."""
    out: list[int] = []
    seen: set[int] = set()
    for row in seq:
        if isinstance(row, int):
            v = row
        else:
            try:
                v = int(row[0])
            except (TypeError, IndexError, ValueError):
                continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return tuple(out)


def _jaccard_top_k(a: list[tuple[int, ...]], b: list[tuple[int, ...]]) -> float:
    """Top-k order-similarity proxy: set-Jaccard of the tuples."""
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wandb-dir", default="~/dsnn/wandb",
        help="Root wandb directory containing the rq1_* run dirs.",
    )
    p.add_argument(
        "--log-dir", default="~/dsnn/logs_rq1_ve_only",
        help="Where the sbatch wrote per-variant stdout (just for reference).",
    )
    p.add_argument(
        "--dataset", default=_DATASET,
        help=f"Dataset to use when computing the graphax_rev baseline FMA. "
             f"Default {_DATASET!r} — matches the RQ1 sbatch's locked target.",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Top-k orders per (example, cmp_type) to compare across cmp-types.",
    )
    p.add_argument(
        "--output-csv", default="out/rq1_summary.csv",
    )
    args = p.parse_args()

    wandb_root = Path(args.wandb_dir).expanduser()
    # `--name rq1_<tag>` doesn't always make it into the directory
    # stem (online runs use run-<8charid>-<name> but offline runs use
    # offline-run-<timestamp>-<8charid>, no name). Scan EVERY
    # ``(offline-)run-*`` dir and read wandb-metadata.json to filter.
    all_run_dirs = list(wandb_root.glob("run-*")) + list(
        wandb_root.glob("offline-run-*")
    )
    candidates = []
    for run_dir in all_run_dirs:
        cfg = run_dir / "files" / "wandb-metadata.json"
        if not cfg.exists():
            continue
        text = cfg.read_text()
        if '"rq1_' in text or '--name=rq1_' in text:
            candidates.append(run_dir)
    print(
        f"Found {len(candidates)} candidate wandb runs "
        f"(out of {len(all_run_dirs)} total under {wandb_root})."
    )

    # Bucket by (example, cmp_type) → list of (seed, best_fma, top_orders).
    bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run_dir in candidates:
        meta_p = run_dir / "files" / "wandb-metadata.json"
        if not meta_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        # Search the args list for --example, --cmp-type, --seed.
        argv = meta.get("args", [])
        ex = ct = sd = None
        for i, a in enumerate(argv):
            if a == "--example" and i + 1 < len(argv):
                ex = argv[i + 1]
            elif a == "--cmp-type" and i + 1 < len(argv):
                ct = argv[i + 1]
            elif a == "--seed" and i + 1 < len(argv):
                try:
                    sd = int(argv[i + 1])
                except ValueError:
                    pass
        if ex is None or ct is None or sd is None:
            continue
        # Defensive: skip runs whose cmp-type isn't one of the sweep's
        # canonical {graphax, flops, latency}. A new value (e.g. a
        # hand-edited config) shouldn't crash the analyser.
        if ct not in _REWARD_TO_CMP:
            print(f"  [warn] unknown cmp_type {ct!r} in {run_dir.name}; skipping")
            continue

        bsj = run_dir / "files" / "best_sequences.json"
        if not bsj.exists():
            continue
        data = _load_best_sequence(bsj)
        if data is None:
            continue
        overall = data.get("best_overall", {})
        seq = overall.get("seq", [])
        order = _flatten_order(seq)
        rewards_raw = overall.get("rewards_raw", {})
        cmp_channel = _REWARD_TO_CMP[ct]
        best_signed = float(rewards_raw.get(cmp_channel, float("nan")))
        # Trainer stores negative cost (reward = -fma); flip back.
        best_fma = -best_signed if not np.isnan(best_signed) else float("nan")
        bucket[(ex, ct)].append(
            {
                "seed": sd,
                "best_fma": best_fma,
                "order": order,
            }
        )

    print()
    print(
        f"{'example':<24s} {'cmp-type':<10s} {'n_seeds':>7s} "
        f"{'median_fma':>14s}  {'graphax_rev_fma':>16s}  "
        f"{'beat_baseline':>14s}"
    )
    print("-" * 100)

    rows: list[dict] = []
    baselines: dict[str, float | None] = {
        ex: _baseline_fma(ex, dataset=args.dataset) for ex in _EXAMPLES
    }

    for ex in _EXAMPLES:
        for ct in _CMP_TYPES:
            runs = bucket.get((ex, ct), [])
            if not runs:
                continue
            best_fmas = [r["best_fma"] for r in runs if not np.isnan(r["best_fma"])]
            median = float(np.median(best_fmas)) if best_fmas else float("nan")
            baseline = baselines.get(ex)
            beat = (
                "YES" if baseline is not None and median <= baseline else (
                    "no" if baseline is not None else "?"
                )
            )
            print(
                f"{ex:<24s} {ct:<10s} {len(runs):>7d} {median:>14.4g}  "
                f"{(baseline if baseline is not None else float('nan')):>16.4g}  "
                f"{beat:>14s}"
            )
            rows.append(
                {
                    "example": ex,
                    "cmp_type": ct,
                    "n_seeds": len(runs),
                    "median_fma": median,
                    "graphax_rev_fma": baseline,
                    "beat_baseline": beat,
                }
            )

    # Cross-cmp-type top-k order overlap (per example).
    print()
    print("Top-k order overlap across cmp-types (Jaccard, per example)")
    print("-" * 70)
    for ex in _EXAMPLES:
        # Top-k orders per cmp-type: take the unique orders found across
        # seeds, sorted by best_fma ascending, keep top-k.
        per_ct: dict[str, list[tuple[int, ...]]] = {}
        for ct in _CMP_TYPES:
            runs = bucket.get((ex, ct), [])
            runs = sorted(
                runs, key=lambda r: (
                    r["best_fma"] if not np.isnan(r["best_fma"]) else float("inf")
                ),
            )
            unique_orders: list[tuple[int, ...]] = []
            seen: set[tuple[int, ...]] = set()
            for r in runs:
                if r["order"] in seen:
                    continue
                seen.add(r["order"])
                unique_orders.append(r["order"])
                if len(unique_orders) >= args.top_k:
                    break
            per_ct[ct] = unique_orders
        if all(len(v) == 0 for v in per_ct.values()):
            continue
        print(f"{ex}:")
        for i, ct_a in enumerate(_CMP_TYPES):
            for ct_b in _CMP_TYPES[i + 1 :]:
                jac = _jaccard_top_k(per_ct.get(ct_a, []), per_ct.get(ct_b, []))
                print(f"   {ct_a:<10s} vs {ct_b:<10s} = {jac:.3f}")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
