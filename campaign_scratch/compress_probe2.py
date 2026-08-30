"""COMPRESS is_last gate, round 2.

(1) Sweep mid-plan COMPRESS on the CAMPAIGN target (TransformerLM) and a few
    others: does graphax raise, raw and hook-wrapped?
(2) The observation/application MISMATCH, on a vertex where COMPRESS actually
    bites: what the policy is credited with at decision time vs what the
    TERMINAL measurement contains.
(3) The prefix property of the token stream, with the gate ON (today) and with
    the gate OFF (is_last=True everywhere).
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

from graphax import jacve, IncrementalPathTokenizer
from graphax.core import _build_graph

from alphagrad.approx.env import (
    COMPRESS_SENTINEL, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs,
)
from alphagrad.approx.common.masks import make_live_masked_hook
from alphagrad.approx.common.examples import get_fn, get_args


def _valid_vertices(jaxpr, args, consts, argnums):
    _, _, _, vo = _build_graph(jaxpr, args, consts, argnums)
    return tuple(i for i, eqn in enumerate(jaxpr.eqns, 1)
                 if eqn.outvars[0] not in jaxpr.outvars or i in vo)


def _flat(t):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(t)])


def _cos(a, b):
    a = np.asarray(_flat(a), dtype=np.float64)
    b = np.asarray(_flat(b), dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _rows(axis, kind=0):
    r = [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]
    r[0] = [COMPRESS_SENTINEL, axis, kind]
    return r


def _ctx(name):
    key = jax.random.PRNGKey(0)
    fn = get_fn(name)
    args = get_args(name, key)
    argnums = tuple(range(len(args)))
    cj = jax.make_jaxpr(fn)(*args)
    vv = _valid_vertices(cj.jaxpr, args, cj.literals, argnums)
    return dict(fn=fn, args=args, argnums=argnums, jaxpr=cj.jaxpr,
                consts=cj.literals, order=list(reversed(vv)))


def sweep(name, n_pos=12):
    print(f"\n{'='*78}\n(1) POSITION SWEEP -- {name}\n{'='*78}", flush=True)
    c = _ctx(name)
    fn, args, argnums, jaxpr, order = (c["fn"], c["args"], c["argnums"],
                                       c["jaxpr"], c["order"])
    print(f"eqns={len(jaxpr.eqns)} order_len={len(order)}", flush=True)
    t0 = time.time()
    ref = jax.jit(jacve(fn, order, argnums=argnums))(*args)
    print(f"exact AD ok ({time.time()-t0:.1f}s)", flush=True)

    T = len(order)
    idxs = sorted(set(
        [0, 1, 2] + [T // 4, T // 3, T // 2, 2 * T // 3, 3 * T // 4]
        + [T - 3, T - 2, T - 1]))[:n_pos]
    stats = {"raw_raise": 0, "hook_raise": 0, "ok": 0, "eff": 0}
    best = None
    for idx in idxs:
        v = int(order[idx])
        rules, ax = (), None
        for axis in range(6):
            r = decode_vertex_rule_specs(jaxpr, v, _rows(axis), is_last=True)
            if r:
                rules, ax = r, axis
                break
        if not rules:
            print(f"  pos {idx:3d} v={v:3d}: no legal COMPRESS row", flush=True)
            continue
        line = f"  pos {idx:3d}/{T-1} v={v:3d} last={int(idx==T-1)} axis={ax}"
        cos_hook = None
        for tag, tr in (("raw", tuple(rules)),
                        ("hook", (make_live_masked_hook(tuple(rules)),))):
            try:
                out = jax.jit(jacve(fn, order, argnums=argnums,
                                    transforms=[(v, tr)]))(*args)
                cc = _cos(out, ref)
                line += f" | {tag}: ok cos={cc:.6f}"
                if tag == "hook":
                    cos_hook = cc
            except Exception as e:  # noqa: BLE001
                line += f" | {tag}: RAISE {type(e).__name__}: {str(e)[:120]}"
                stats[tag + "_raise"] += 1
        print(line, flush=True)
        stats["ok"] += 1
        if cos_hook is not None and cos_hook < 0.999:
            stats["eff"] += 1
            if idx != T - 1 and (best is None or cos_hook < best[2]):
                best = (idx, v, cos_hook, ax, rules)
    print(f"SWEEP {name}: {stats}", flush=True)
    return c, ref, best


def mismatch(name, c, ref, best):
    print(f"\n{'='*78}\n(2) OBSERVATION/APPLICATION MISMATCH -- {name}\n{'='*78}",
          flush=True)
    if best is None:
        print("  no effective mid-plan COMPRESS found; skipped", flush=True)
        return
    idx, v, cos_hook, ax, rules = best
    fn, args, argnums, jaxpr, order = (c["fn"], c["args"], c["argnums"],
                                       c["jaxpr"], c["order"])
    T = len(order)
    print(f"  policy plants {rules[0]!r} on vertex {v} at position {idx} of "
          f"{T-1}", flush=True)
    for plen in (idx + 1, idx + 2, T):
        if plen > T:
            continue
        dec = decode_vertex_rule_specs(jaxpr, v, _rows(ax),
                                       is_last=(idx == plen - 1))
        print(f"    prefix length {plen:3d}: vertex {v} -> {dec!r}  "
              f"[{'APPLIED' if dec else 'DROPPED'}]", flush=True)

    # what the DECISION step measured (prefix of length idx+1, compress honored)
    pre = order[: idx + 1]
    ref_pre = jax.jit(jacve(fn, pre, argnums=argnums))(*args)
    out_pre = jax.jit(jacve(
        fn, pre, argnums=argnums,
        transforms=[(v, (make_live_masked_hook(tuple(rules)),))]))(*args)
    print(f"  DECISION step (len {idx+1}) cos(approx, exact) = "
          f"{_cos(out_pre, ref_pre):.6f}   <- what the policy was credited with",
          flush=True)

    # what the TERMINAL step measures, gate ON (today) vs gate OFF
    out_on = jax.jit(jacve(fn, order, argnums=argnums, transforms=[]))(*args)
    out_off = jax.jit(jacve(
        fn, order, argnums=argnums,
        transforms=[(v, (make_live_masked_hook(tuple(rules)),))]))(*args)
    print(f"  TERMINAL gate ON  (today) cos vs exact = {_cos(out_on, ref):.6f}"
          f"   <- the COMPRESS silently vanished", flush=True)
    print(f"  TERMINAL gate OFF (fixed) cos vs exact = {_cos(out_off, ref):.6f}"
          f"   <- the COMPRESS is actually applied", flush=True)


def prefix_property(name, c, n=6):
    print(f"\n{'='*78}\n(3) TOKEN-STREAM PREFIX PROPERTY -- {name}\n{'='*78}",
          flush=True)
    jaxpr, consts, args, argnums, order = (c["jaxpr"], c["consts"], c["args"],
                                           c["argnums"], c["order"])
    prefix = order[:n]

    def stream_for(pl, gate_on):
        tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                      vocab_size=512)
        s = [int(t) for t in tk.base_tokens()]
        last = pl - 1
        for k, v in enumerate(prefix[:pl]):
            rules = decode_vertex_rule_specs(
                jaxpr, int(v), _rows(0),
                is_last=((k == last) if gate_on else True))
            hooks = (make_live_masked_hook(tuple(rules)),) if rules else ()
            s += [int(t) for t in tk.eliminate(int(v), hooks)]
        return s

    for gate_on, label in ((True, "gate ON  (today)"),
                           (False, "gate OFF (is_last removed)")):
        holds_all = True
        first_bad = None
        for pl in range(1, len(prefix)):
            short = stream_for(pl, gate_on)
            longs = stream_for(pl + 1, gate_on)
            ok = longs[: len(short)] == short
            if not ok and first_bad is None:
                i = next((i for i, (a, b) in enumerate(zip(short, longs))
                          if a != b), len(short))
                first_bad = (pl, i, len(short))
            holds_all &= ok
        print(f"  COMPRESS rows, {label}: prefix property "
              f"{'HOLDS' if holds_all else 'VIOLATED'}"
              + ("" if holds_all else
                 f" (first divergence at prefix len {first_bad[0]}, token "
                 f"{first_bad[1]} of {first_bad[2]})"), flush=True)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["NeuralNetwork"]
    for t in targets:
        try:
            c, ref, best = sweep(t)
            mismatch(t, c, ref, best)
            prefix_property(t, c)
        except Exception:  # noqa: BLE001
            print(f"TARGET {t} FAILED", flush=True)
            traceback.print_exc()
    print("\nDONE", flush=True)
