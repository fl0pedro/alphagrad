"""COMPRESS `is_last` gate: is it a SEMANTIC constraint or an implementation shortcut?

Part A -- position sweep. For every vertex of a FULL elimination order, decode the
same COMPRESS wire row with `is_last=True` (so it passes the env's own legality
filter) and apply it at that vertex's ACTUAL position. Raw rule AND
make_live_masked_hook-wrapped (what the env really installs). Record: raise? cos
vs exact AD? Answers "does a mid-plan COMPRESS blow up".

Part B -- the observation/application mismatch. Show that a COMPRESS the env
APPLIED at step k has silently vanished from the TERMINAL measurement (the
decode drops it once the vertex is no longer last), i.e. the terminal reward is
bit-identical to exact AD while the policy was told it approximated.
"""
import os
import sys
import time
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax
import jax.numpy as jnp
import numpy as np

from graphax import jacve
from graphax.core import _build_graph
from graphax.sparse.micro_actions import Compress

from alphagrad.approx.env import (
    COMPRESS_SENTINEL, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs,
)
from alphagrad.approx.common.masks import make_live_masked_hook
from alphagrad.approx.common.examples import get_fn, get_args


def _valid_vertices(jaxpr, args, consts, argnums):
    _, _, _, vo = _build_graph(jaxpr, args, consts, argnums)
    return tuple(
        i for i, eqn in enumerate(jaxpr.eqns, 1)
        if eqn.outvars[0] not in jaxpr.outvars or i in vo
    )


def _flat(t):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(t)])


def _cos(a, b):
    a = _flat(a).astype(jnp.float64)
    b = _flat(b).astype(jnp.float64)
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(jnp.dot(a, b) / (na * nb))


def _rows(axis, kind=0):
    r = [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]
    r[0] = [COMPRESS_SENTINEL, axis, kind]
    return r


def probe(name, max_positions=None):
    print(f"\n{'='*78}\nTARGET {name}\n{'='*78}", flush=True)
    key = jax.random.PRNGKey(0)
    fn = get_fn(name)
    args = get_args(name, key)
    argnums = tuple(range(len(args)))
    cj = jax.make_jaxpr(fn)(*args)
    jaxpr, consts = cj.jaxpr, cj.literals
    vv = _valid_vertices(jaxpr, args, consts, argnums)
    order = list(reversed(vv))  # a reverse-mode-like complete order
    print(f"eqns={len(jaxpr.eqns)} eliminable={len(vv)} order_len={len(order)}",
          flush=True)

    t0 = time.time()
    ref = jax.jit(jacve(fn, order, argnums=argnums))(*args)
    print(f"exact AD ok ({time.time()-t0:.1f}s)", flush=True)

    positions = list(range(len(order)))
    if max_positions and len(positions) > max_positions:
        # first / middle / last band -- keep the run affordable on big graphs
        positions = (positions[:max_positions // 3]
                     + positions[len(positions) // 2:
                                 len(positions) // 2 + max_positions // 3]
                     + positions[-max_positions // 3:])

    stats = {"raw_ok": 0, "raw_raise": 0, "hook_ok": 0, "hook_raise": 0,
             "identity": 0, "changed": 0, "tried": 0}
    rows_out = []
    for idx in positions:
        v = int(order[idx])
        is_last = (idx == len(order) - 1)
        # pick the first axis for which the env's own decoder emits a rule
        rules = ()
        chosen_axis = None
        for axis in range(6):
            r = decode_vertex_rule_specs(jaxpr, v, _rows(axis), is_last=True)
            if r:
                rules, chosen_axis = r, axis
                break
        if not rules:
            continue
        stats["tried"] += 1

        rec = {"idx": idx, "v": v, "last": is_last, "axis": chosen_axis,
               "rule": repr(rules[0])}
        for tag, tr in (("raw", tuple(rules)),
                        ("hook", (make_live_masked_hook(tuple(rules)),))):
            try:
                out = jax.jit(jacve(fn, order, argnums=argnums,
                                    transforms=[(v, tr)]))(*args)
                c = _cos(out, ref)
                rec[tag] = f"ok cos={c:.6f}"
                rec[tag + "_cos"] = c
                stats[tag + "_ok"] += 1
                if tag == "hook":
                    if abs(c - 1.0) < 1e-12:
                        stats["identity"] += 1
                    else:
                        stats["changed"] += 1
            except Exception as e:  # noqa: BLE001
                rec[tag] = f"RAISE {type(e).__name__}: {str(e)[:160]}"
                stats[tag + "_raise"] += 1
        rows_out.append(rec)
        print(f"  pos {idx:3d}/{len(order)-1} v={v:3d} last={int(is_last)} "
              f"axis={chosen_axis} | raw: {rec.get('raw')} | "
              f"hook: {rec.get('hook')}", flush=True)

    print(f"\nSUMMARY {name}: {stats}", flush=True)
    return rows_out, stats, dict(fn=fn, args=args, argnums=argnums,
                                 jaxpr=jaxpr, order=order, ref=ref)


def part_b(name, ctx):
    """The mismatch: what the env DECODES for the same plan at successive
    prefix lengths, and what the TERMINAL measurement therefore contains."""
    print(f"\n--- PART B ({name}): decode of the SAME plan vs prefix length ---",
          flush=True)
    jaxpr, order = ctx["jaxpr"], ctx["order"]
    fn, args, argnums, ref = ctx["fn"], ctx["args"], ctx["argnums"], ctx["ref"]
    # plant a COMPRESS on the vertex at position j (a mid vertex)
    j = max(0, len(order) // 2)
    v = int(order[j])
    rules = ()
    for axis in range(6):
        r = decode_vertex_rule_specs(jaxpr, v, _rows(axis), is_last=True)
        if r:
            rules = r
            break
    if not rules:
        print("  no legal COMPRESS at the mid vertex; skipping part B")
        return
    print(f"  planted {rules[0]!r} on vertex {v} (position {j} of "
          f"{len(order)-1})", flush=True)
    for plen in (j + 1, j + 2, len(order)):
        if plen > len(order):
            continue
        last_idx = plen - 1
        dec = decode_vertex_rule_specs(
            jaxpr, v, _rows(axis), is_last=(j == last_idx))
        print(f"  prefix length {plen:3d}: vertex {v} decodes to {dec!r}"
              f"   <-- {'APPLIED' if dec else 'DROPPED'}", flush=True)

    # numeric consequence at the TERMINAL step
    trs_gate_on = []  # what the env builds today: is_last only at the end
    trs_gate_off = [(v, (make_live_masked_hook(tuple(rules)),))]
    out_on = jax.jit(jacve(fn, order, argnums=argnums,
                           transforms=trs_gate_on))(*args)
    print(f"  TERMINAL, gate ON  (today): transforms={trs_gate_on} "
          f"cos_vs_exact={_cos(out_on, ref):.12f}", flush=True)
    try:
        out_off = jax.jit(jacve(fn, order, argnums=argnums,
                                transforms=trs_gate_off))(*args)
        print(f"  TERMINAL, gate OFF        : cos_vs_exact="
              f"{_cos(out_off, ref):.12f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  TERMINAL, gate OFF        : RAISE {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    targets = sys.argv[1:] or ["NeuralNetwork"]
    for t in targets:
        try:
            rows, stats, ctx = probe(t, max_positions=45)
            part_b(t, ctx)
        except Exception:  # noqa: BLE001
            print(f"TARGET {t} FAILED", flush=True)
            traceback.print_exc()
    print("\nDONE", flush=True)
