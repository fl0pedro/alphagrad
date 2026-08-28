#!/usr/bin/env python3
"""applied/requested per approximation kind, BEFORE and AFTER --per-face-masks.

WHAT THIS MEASURES, AND WHAT IT DOES NOT.

The quantity the flag is supposed to move is the fraction of the policy's
approximation REQUESTS that survive the live-operand legality check --
`approx_applied/*` over `approx_applied/* + approx_skipped/*`, the same
counters `env._PER_FACE_STATS` accumulates in a real run. That check is
`masks.make_live_masked_hook`, and it is a PURE FUNCTION of (live operand,
requested rule). So this harness reproduces it exactly:

  * `LiveVertexMaskOracle.probe_faces` gives the live operand of every face of
    every vertex, in visit order -- the same tensors the real run's per-vertex
    hook receives (`graphax.core._eliminate_vertex` applies the list to
    `edge_outval`, i.e. the face's `res` slot);
  * the rule is drawn the way the 94-slot face head draws it, from the oracle's
    own per-face masks: op uniform over the LEGAL ops, (i, j) uniform over the
    admitted pairs, factor = gcd(N_i, N_j) exactly as `_rows` hardcodes it,
    COMPRESS axis uniform over the admitted axes, QUANT dtype uniform over the
    two the head can emit;
  * the rule is then handed to the real `make_live_masked_hook` with the flag
    in the arm's state, and the arm's stats dict is the answer.

BEFORE = the mask and sizes the head sees today (nominal per-vertex sizes,
`face_masks`' nominal-gcd screen, QUANT unmasked). AFTER = the same with
`--per-face-masks`. Same seed, same order, same vertices: a PAIRED comparison.

NOT measured here: the reward, the latency, or the policy's learned
preferences. The elimination prefix is advanced EXACTLY in both arms (the
probes are read-only), so the graphs the two arms see are identical by
construction -- which is what makes the comparison paired, and also means this
does not model how an early approximation changes later structure. It is the
masking layer's number, not an end-to-end training number.

Usage:  python pfm_measure.py [example ...]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math

import jax
import jax.numpy as jnp
import numpy as np

from graphax import inline_call_primitives
from graphax.sparse.micro_actions import Compress, Diag, Quant

from alphagrad.approx.common import masks as M
from alphagrad.approx.common.masks import (
    FACE_QUANT_DTYPES, LiveVertexMaskOracle, make_live_masked_hook,
    rule_is_idempotent_noop, set_per_face_masks)
from alphagrad.approx.common.examples import get_args, get_fn, infer_argnums

try:
    from jax.extend.core import ClosedJaxpr
except ImportError:                                        # pragma: no cover
    from jax._src.core import ClosedJaxpr

N_AX = 8
MAX_F = 32
KINDS = ("diag", "compress", "quant")


def _closed(example):
    fn = get_fn(example)
    xs = get_args(example, jax.random.PRNGKey(0))
    argnums = infer_argnums(example)
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)
    return closed, list(xs), tuple(argnums)


def _nominal_sizes(jaxpr, v):
    """The head's TODAY sizes: out_shape ++ first-invar shape, N-padded.

    `compute_static_axis_state` uses the first invar as the primal proxy, and
    `diag_row_to_pair` puts primal axis `b` at index `out_len + b` -- so this
    is the vector `_rows` indexes with `fields.i` / `fields.j`.
    """
    eqn = jaxpr.eqns[v - 1]
    out = tuple(eqn.outvars[0].aval.shape)
    ins = [tuple(iv.aval.shape) for iv in eqn.invars if hasattr(iv, "aval")]
    dims = list(out) + list(ins[0] if ins else ())
    z = np.zeros((N_AX,), np.int32)
    z[: min(len(dims), N_AX)] = np.asarray(dims[:N_AX], np.int32)
    return z


def _draw(rng, fp_k, fc_k, sizes_k, quant_k, per_face):
    """One face's request, drawn the way the 94-slot head draws it."""
    pairs = [(i, j) for i in range(N_AX) for j in range(N_AX) if fp_k[i, j]]
    axes = [a for a in range(N_AX) if fc_k[a]]
    ops = []
    if pairs:
        ops.append("diag")
    if axes:
        ops.append("compress")
    # QUANT: unmasked today, per-face under the flag.
    if (not per_face) or bool(quant_k):
        ops.append("quant")
    if not ops:
        return None
    op = ops[int(rng.integers(len(ops)))]
    if op == "diag":
        i, j = pairs[int(rng.integers(len(pairs)))]
        g = math.gcd(int(sizes_k[i]), int(sizes_k[j]))
        return Diag(i, j, max(g, 1))
    if op == "compress":
        return Compress((axes[int(rng.integers(len(axes)))],), "mean")
    return Quant(dtype=FACE_QUANT_DTYPES[int(rng.integers(2))])


def _kind_of(rule):
    return {Diag: "diag", Compress: "compress", Quant: "quant"}.get(
        type(rule), "other")


def _audited(rule, stats):
    """The production hook, wrapped so BOTH arms report the honest denominator.

    ``skipped_K_noop`` is emitted by the hook only when the per-face flag is
    on, so it cannot be read off the BEFORE arm -- and without it the two arms'
    ``applied / requested`` are not the same statistic (BEFORE counts an inert
    ``Quant(float32)`` on a float32 operand as applied; AFTER does not). This
    wrapper sees the SAME live operand the hook sees, so it can classify every
    request identically in both arms WITHOUT changing what the hook does.
    """
    inner = make_live_masked_hook((rule,), max_dims=N_AX, max_axes=N_AX,
                                  stats=stats)

    def _h(st):
        k = _kind_of(rule)
        noop = rule_is_idempotent_noop(st, rule, max_dims=N_AX, max_axes=N_AX)
        n0 = stats.get("applied", 0)
        out = inner(st)
        if noop:
            stats[f"audit_noop_{k}"] = stats.get(f"audit_noop_{k}", 0) + 1
            if stats.get("applied", 0) > n0:
                # Applied, but inert. This is the count that makes BEFORE's
                # `applied` an overstatement -- and it is invisible to the
                # production counters, which is the whole reason QUANT read
                # 36/36.
                key = f"audit_noop_applied_{k}"
                stats[key] = stats.get(key, 0) + 1
        return out

    return _h


def _arm(closed, xs, argnums, per_face, seed):
    """One arm: drive a REAL elimination with the drawn rule in all 3 slots.

    THE THREE SLOTS ARE THE POINT. The oracle's per-face mask is computed
    against a PROBE of the face -- the tensor the per-vertex `transforms`
    callable receives, i.e. the `res` slot. The rule is then applied to lhs
    (the in-edge Jacobian), rhs (the out-edge Jacobian) AND res, whose index
    structures differ from the probe's and from each other
    (`masks.py`'s "WHY THIS EXISTS" note). Planting the mask-admitted rule in
    all three is exactly the forensics setup that measured 155 live rejections
    on NeuralNetwork, and it is what a harness that only touched the probe
    would miss entirely.

    The MASKS come from a `LiveVertexMaskOracle` advanced EXACTLY, while the
    application runs on its own `IncrementalJaxpr` that carries the
    approximations. That divergence is real and is shared by both arms, which
    is what keeps the comparison paired.
    """
    from graphax.incremental import IncrementalJaxpr

    jaxpr = closed.jaxpr
    o = LiveVertexMaskOracle(jaxpr, list(closed.literals), list(xs), argnums,
                             max_axes=N_AX)
    incr = IncrementalJaxpr(jaxpr, argnums, list(closed.literals), list(xs))
    rng = np.random.default_rng(seed)
    stats: dict = {}
    n_req = 0
    set_per_face_masks(bool(per_face))
    try:
        # REVERSE order: the cheapest exact order and the campaign's reference.
        for v in range(len(jaxpr.eqns), 0, -1):
            try:
                fp, fc, sizes, quant, n = o.face_masks_and_sizes(
                    v, MAX_F, per_face=per_face)
            except Exception:
                n = 0
                fp = fc = sizes = quant = None
            ft = {}
            if n:
                nom = _nominal_sizes(jaxpr, v)
                try:
                    keys = list(incr.faces(int(v)))
                except Exception:
                    keys = []
                for k in range(min(n, len(keys))):
                    rule = _draw(rng, fp[k], fc[k],
                                 sizes[k] if per_face else nom,
                                 quant[k] if per_face else 1, per_face)
                    if rule is None:
                        continue
                    n_req += 3          # one request per slot
                    ft[keys[k]] = tuple(
                        _audited(rule, stats) for _ in range(3))
            try:
                incr.eliminate(int(v), face_transforms=ft or None)
            except Exception as exc:
                print(f"  [warn] eliminate v{v} failed: "
                      f"{type(exc).__name__}: {str(exc)[:120]}")
                break
            try:
                o.advance(v, rules=())
            except Exception:
                break
    finally:
        set_per_face_masks(False)
    stats["_requests"] = n_req
    return stats


def _row(stats, kind):
    """``(applied, real_applied, skipped, noop, req, app/req, real/(req-noop))``.

    ``noop`` is the AUDIT count -- classified identically in both arms.
    ``real_applied`` subtracts the requests that were applied but were
    idempotent (only the BEFORE arm has any: the AFTER arm masks them out),
    so the last column is the same statistic on both rows: of the requests
    that COULD have changed the operand, how many did.
    """
    ap = stats.get(f"applied_{kind}", 0)
    sk = stats.get(f"skipped_{kind}", 0)
    noop = stats.get(f"audit_noop_{kind}", 0)
    req = ap + sk
    real = ap - stats.get(f"audit_noop_applied_{kind}", 0)
    frac = (ap / req) if req else float("nan")
    denom = req - noop
    hfrac = (real / denom) if denom > 0 else float("nan")
    return ap, real, sk, noop, req, frac, hfrac


def report(example, seed=0):
    closed, xs, argnums = _closed(example)
    print(f"\n=== {example}  ({len(closed.jaxpr.eqns)} eqns) ===")
    before = _arm(closed, xs, argnums, False, seed)
    after = _arm(closed, xs, argnums, True, seed)
    print(f"{'kind':9} | {'arm':6} | {'appl':>5} {'real':>5} {'skip':>5} "
          f"{'noop':>5} {'req':>5} | {'app/req':>8} {'real/(req-noop)':>16}")
    print("-" * 84)
    for kind in KINDS:
        for name, st in (("before", before), ("after", after)):
            ap, real, sk, noop, req, frac, hfrac = _row(st, kind)
            print(f"{kind:9} | {name:6} | {ap:5d} {real:5d} {sk:5d} "
                  f"{noop:5d} {req:5d} | {frac:8.3f} {hfrac:16.3f}")
    for name, st in (("before", before), ("after", after)):
        print(f"[{name}] requests={st['_requests']} "
              f"applied={st.get('applied', 0)} "
              f"skipped={st.get('skipped', 0)} "
              f"skipped_raised={st.get('skipped_raised', 0)} "
              f"repaired_diag={st.get('repaired_diag', 0)} "
              f"repaired_compress={st.get('repaired_compress', 0)}")
    assert before.get("skipped_raised", 0) == 0
    assert after.get("skipped_raised", 0) == 0


if __name__ == "__main__":
    for ex in (sys.argv[1:] or ["NeuralNetwork"]):
        try:
            report(ex)
        except Exception as exc:
            import traceback
            print(f"\n=== {ex}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()
