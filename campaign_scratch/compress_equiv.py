"""Byte-equivalence probe: is anything OTHER than COMPRESS affected?

Dumps the full token stream for every prefix length of a fixed plan, for three
plans: exact, DIAG-only, and COMPRESS-carrying. Run once against the HEAD
snapshot's `src` and once against the patched tree, then diff the JSONs.

Expected: exact and DIAG-only streams byte-identical between the two trees;
the COMPRESS plan differs from prefix length (compress_position + 2) onward.
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

import alphagrad.approx.env as E
from alphagrad.approx.common.masks import make_live_masked_hook


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)


def _decode(jaxpr, v, rows, is_last):
    try:
        return E.decode_vertex_rule_specs(jaxpr, v, rows, is_last=is_last)
    except TypeError:                       # patched tree: no is_last
        return E.decode_vertex_rule_specs(jaxpr, v, rows)


def _plan(kind, T, compress_at):
    specs = []
    for i in range(T):
        rows = [[-1, -1, 0] for _ in range(E.MAX_RULES_PER_VERTEX)]
        if kind == "diag" and i % 2 == 0:
            rows[0] = [0, 0, 2]
        if kind == "compress" and i == compress_at:
            rows[0] = [E.COMPRESS_SENTINEL, 0, 0]
        specs.append(rows)
    return specs


def main(out):
    cj = jax.make_jaxpr(_fn)(*ARGS)
    cfg = SimpleNamespace(jaxpr=cj.jaxpr, argnums=(0, 1), per_face=False)
    consts, args = list(cj.literals), list(ARGS)
    T = len(cj.jaxpr.eqns)
    order = list(range(1, T + 1))
    res = {}
    for kind in ("exact", "diag", "compress"):
        specs = _plan(kind, T, compress_at=1)
        per_len = []
        for t in range(1, T + 1):
            E._INCR_STREAM_CACHE.clear()          # always COLD -> no cache effects
            last = t - 1
            tok = {}
            for i, v in enumerate(order[:t]):
                r = _decode(cj.jaxpr, int(v), specs[i], i == last)
                if r:
                    tok[int(v)] = (make_live_masked_hook(tuple(r)),)
            stream, seg, _ft, ls = E._incremental_stream_tokens(
                cfg, consts, args, order[:t], specs[:t], tok,
                ft_by_vertex=None, face_key=None)
            per_len.append({"len": len(stream), "stream": list(map(int, stream)),
                            "last_start": int(ls)})
        res[kind] = per_len
    json.dump(res, open(out, "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
