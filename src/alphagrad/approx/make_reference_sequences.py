"""Write synthetic reference best_sequences.json files for nn_mnist.py.

For a given (example, dataset), emit two JSONs into the target dir:

* ``graphax_fwd.json`` — forward-mode elimination order [1, 2, ..., n]
* ``graphax_rev.json`` — reverse-mode elimination order [n, n-1, ..., 1]

The vertex count ``n`` is the number of eliminable vertices in the
target function's jaxpr (output_vars filtered out per the env's
``valid_vertices`` rule, matching how the RL trainer numbers them).

These references give ``nn_mnist.py`` something to compare the RL
sequences against in the same MNIST-training framework. NB:
``jax.jacrev`` / ``jax.jacfwd`` produce mathematically identical
gradients to ``graphax.jacve(order="rev"|"fwd")`` on a fully-
eliminated order, modulo XLA scheduling differences — the
``graphax_*`` reference covers both for the test-accuracy metric.

Usage:
    uv run alphagrad/src/alphagrad/approx/make_reference_sequences.py \\
        --example VmappedNeuralNetwork --dataset mnist \\
        --output-dir ~/dsnn/results/rq1_seq_eval/sequences
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# Op type sentinel matching alphagrad.approx.heads.OP_END.
OP_END = 3


def _count_eliminable_vertices(example: str, dataset: str) -> int:
    """Mirror ``alphagrad.approx.env.VertexEliminationEnv``'s rule for
    ``valid_vertices``: enumerate the jaxpr's equations 1-indexed and
    keep the ones whose outvar isn't a graph output (or that ARE an
    output but in ``vo_vertices``).
    """
    # Defer JAX import — this helper is intended to run on a GPU node
    # but JAX needs the env vars set before import.
    import jax  # noqa: F401
    import jax.random as jrand
    from graphax.core import _build_graph
    from alphagrad.approx.common.examples import (
        get_args,
        get_fn,
        infer_argnums,
    )

    target_fn = get_fn(example)
    argnums = tuple(infer_argnums(example))
    key = jrand.PRNGKey(0)
    args = get_args(example, key, dataset=dataset)
    closed_jaxpr = jax.make_jaxpr(target_fn)(*args)
    _, _, _, vo_vertices = _build_graph(
        closed_jaxpr.jaxpr, args, [], argnums,
    )
    valid = []
    for i, eqn in enumerate(closed_jaxpr.jaxpr.eqns, 1):
        if eqn.outvars[0] not in closed_jaxpr.jaxpr.outvars or i in vo_vertices:
            valid.append(i)
    return len(valid)


def _make_reference_payload(order: list[int]) -> dict:
    """Wrap an explicit elimination order in the best_sequences.json
    schema nn_mnist.py expects.

    The seq is a list of 6-tuples ``[vertex, op, i, j, factor, q]``.
    For a pure-VE reference, every op is OP_END (no DIAG / COMPRESS /
    QUANT applied). The trainer's recording uses 0-indexed vertex IDs;
    nn_mnist.py's decoder accepts either convention (its
    ``_decode_record`` reads the first slot as the literal vertex
    index without offset).
    """
    seq = [[int(v), OP_END, 0, 0, 0, 0] for v in order]
    # Provide zeros for the reward metadata that nn_mnist.py's
    # _make() reads but doesn't strictly require — keeps the JSON
    # round-trippable with the trainers' richer format.
    return {
        "best_overall": {
            "return": 0.0,
            "ep": 0,
            "rewards_raw": {
                "muls_adds_fmas": 0.0,
                "flops": 0.0,
                "latency_ns": 0.0,
                "max_io_sum": 0.0,
                "bytes_accessed": 0.0,
                "peak_memory": 0.0,
                "cosine_sim": 1.0,
                "frob_residual": 0.0,
            },
            "rewards_weighted": {},
            "seq": seq,
        },
        "best_per_channel": {},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example", default="VmappedNeuralNetwork")
    p.add_argument("--dataset", default="mnist")
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write graphax_fwd.json and graphax_rev.json into.",
    )
    p.add_argument(
        "--zero-indexed", action="store_true",
        help="Emit 0-indexed vertex IDs (matching the trainer's "
        "recording convention). Default is 1-indexed (matching "
        "graphax.jacve's internal numbering).",
    )
    args = p.parse_args()

    n = _count_eliminable_vertices(args.example, args.dataset)
    print(f"[ref] {args.example}+{args.dataset}: {n} eliminable vertices")

    offset = 0 if args.zero_indexed else 1
    fwd_order = list(range(offset, n + offset))
    rev_order = list(reversed(fwd_order))

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, order in (("graphax_fwd", fwd_order), ("graphax_rev", rev_order)):
        payload = _make_reference_payload(order)
        path = out_dir / f"{tag}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[ref] wrote {path} (order len={len(order)}, first={order[0]}, last={order[-1]})")


if __name__ == "__main__":
    main()
