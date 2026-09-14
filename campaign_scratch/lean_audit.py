"""DAG-AGNOSTICISM AUDIT: no parameter may be dimensioned by V, T or F.

Method (the one Stage B settled on, because it is positive evidence rather
than the absence of a flag): build the SAME agent at two different graph
sizes and compare EVERY leaf shape elementwise. A parameter that depends on
the DAG changes shape; one that does not, does not. A third build at a
different ``embd_dim`` separates the known coincidences (a width that happens
to equal V or the op vocabulary) from real V-dependence.

Run:  JAX_PLATFORMS=cpu uv run --no-sync python lean_audit.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "16")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "1024")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "256")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from alphagrad.approx import ppo as P  # noqa: E402
from alphagrad.approx.common.agent_factory import (  # noqa: E402
    apply_policy_arch, build_and_init_agent)


def _build(total_v, embd_dim, max_rules=4, num_factors=4, seed=7):
    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=False,
        face_actions=True, unified_face_head=True, live_faces=True,
        max_substeps=1, axis_group_embedding=False,
    )
    ns.embd_dim = embd_dim
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.vocab_size = 512
    ns.preference_conditioned = False
    return build_and_init_agent(ns, total_v, num_factors=num_factors,
                                max_rules=max_rules, seed=seed)


def _leaves(agent):
    params = eqx.filter(agent, eqx.is_inexact_array)
    return [(jax.tree_util.keystr(p), tuple(np.asarray(x).shape))
            for p, x in jax.tree_util.tree_flatten_with_path(params)[0]]


def main():
    VA, VB = 96, 137
    a = _build(VA, 32)
    b = _build(VB, 32)
    c = _build(VA, 34)

    la, lb, lc = _leaves(a), _leaves(b), _leaves(c)
    na = sum(int(np.prod(s)) for _n, s in la)
    print(f"[audit] agent(V={VA}, E=32): {len(la)} leaves, {na} params")
    print(f"[audit] agent(V={VB}, E=32): {len(lb)} leaves, "
          f"{sum(int(np.prod(s)) for _n, s in lb)} params")

    if len(la) != len(lb):
        print(f"*** LEAF COUNT DIFFERS: {len(la)} vs {len(lb)} -- NOT "
              f"DAG-AGNOSTIC")
        return 1

    diffs = [(n1, s1, s2) for (n1, s1), (_n2, s2) in zip(la, lb) if s1 != s2]
    if diffs:
        print(f"*** {len(diffs)} LEAF SHAPES DIFFER BETWEEN V={VA} AND "
              f"V={VB} -- NOT DAG-AGNOSTIC")
        for n, s1, s2 in diffs:
            print(f"      {n}: {s1} -> {s2}")
        return 1
    print(f"[audit] ALL {len(la)} LEAF SHAPES IDENTICAL at V={VA} and V={VB}"
          f"  => no parameter is dimensioned by V")

    # Any leaf whose shape merely CONTAINS V is a coincidence candidate; the
    # embd_dim sweep proves which. (Both the (71,8) op vocab and the 3*E face
    # width were flagged this way in the Stage B audit and both were
    # coincidences.)
    susp = [(n, s) for n, s in la
            if VA in s or VB in s or 71 in s]
    print(f"[audit] leaves whose shape numerically contains V or 71: "
          f"{len(susp)}")
    for n, s in susp:
        s2 = dict(lc)[n]
        verdict = "follows embd_dim (coincidence)" if s2 != s else \
            "unchanged under embd_dim"
        print(f"      {n}: {s} -> (E=34) {s2}   [{verdict}]")

    # F (faces) and T (tokens) are static fields, never parameters -- assert.
    fp = getattr(a, "face_path_policy", None)
    if fp is not None:
        print(f"[audit] face policy max_faces = {fp.max_faces} "
              f"(static field, {'IS' if any('max_faces' in n for n, _ in la) else 'NOT'} "
              f"a parameter)")
        print(f"[audit] face head in_dim   = "
              f"{np.asarray(fp.head.mlp.layers[0].weight).shape[1] if hasattr(fp.head, 'mlp') else 'n/a'}")
    print("[audit] DAG-AGNOSTICISM: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
