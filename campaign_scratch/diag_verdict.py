"""DIAG verdict probe: WHY does rule_is_legal reject ~99% of requested DIAGs?

Part A -- static/logic verdict on the stated hypothesis (`d = factor//base == 1`
by construction).
Part B -- live per-face verdict on a real small target: for every vertex/face,
what does the head's hardcoded `factor = gcd(N_i,N_j)` (NOMINAL jaxpr sizes) do
against `diag_pair_factor_space` of THAT face's LIVE tensor?
Part C -- end-to-end applied fraction with planted DIAG rows, per slot.
"""
import os
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_QUALITY_METRIC", "cosine")

import math
import collections
import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.env import (
    FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX, REWARD_INDEX,
    StepAction, VertexEliminationEnv, consume_per_face_stats,
    diag_row_to_pair,
)
from alphagrad.approx.common.masks import (
    LiveVertexMaskOracle, diag_pair_factor_space, diag_valid_mask,
    rule_is_legal,
)
from alphagrad.approx.common.examples import get_fn, get_args
from graphax import inline_call_primitives
from graphax.sparse.micro_actions import Diag
try:
    from jax.extend.core import ClosedJaxpr
except ImportError:
    from jax._src.core import ClosedJaxpr

COS = REWARD_INDEX["cosine_sim"]
TARGET = os.environ.get("VERDICT_TARGET", "NeuralNetwork")
N_AX = 8

print("=" * 78, flush=True)
print("PART A: is `d = factor // base == 1` true BY CONSTRUCTION?", flush=True)
print("=" * 78, flush=True)
print("diag_pair_factor_space returns (base, span):", flush=True)
print("  FREE pair    -> base = 1,          span = gcd(N_i, N_j)", flush=True)
print("  COUPLED pair -> base = meta m > 1, span = gcd(N_i, N_j) // m", flush=True)
print("head emits factor = gcd(N_i, N_j) over NOMINAL jaxpr sizes.", flush=True)
for (Ni, Nj, meta) in [(64, 64, None), (64, 16, None), (63, 16, None),
                       (64, 64, 4), (64, 16, 4), (64, 64, 64)]:
    g = math.gcd(Ni, Nj)
    if meta is None:
        base, span = 1, g
    else:
        base, span = (meta, g // meta) if g % meta == 0 else (0, 0)
    if base <= 0 or span <= 1:
        verdict = "pair MASKED OUT (span<=1 / bad base)"
    elif g % base:
        verdict = "REJECT (factor %% base != 0)"
    else:
        d = g // base
        verdict = ("APPLY  d=%d" % d) if (d > 1 and span % d == 0) \
            else ("REJECT d=%d" % d)
    print(f"  N_i={Ni:3d} N_j={Nj:3d} meta={str(meta):>4s} g={g:3d} "
          f"base={base:3d} span={span:3d} -> {verdict}", flush=True)

# ------------------------------------------------------------------ setup
key = jax.random.PRNGKey(0)
fn = get_fn(TARGET)
xs = get_args(TARGET, key)
argnums = tuple(range(len(xs)))
cj = jax.make_jaxpr(fn)(*xs)
jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)
env = VertexEliminationEnv.from_jaxpr(closed, args=xs, argnums=argnums,
                                      num_envs=0, target_fun=fn, per_face=True)
order = [int(x) for x in np.asarray(env.valid_vertices)][::-1]
jaxpr = env.config.jaxpr
print(f"\ntarget={TARGET} order={order}", flush=True)


def _divisors_above_one(n):
    return [d for d in range(2, int(n) + 1) if n % d == 0]


print("\n" + "=" * 78, flush=True)
print("PART B: head factor (nominal gcd) vs each FACE's own legal set", flush=True)
print("=" * 78, flush=True)
oracle = LiveVertexMaskOracle(jaxpr, list(closed.literals), list(xs), argnums,
                              max_axes=N_AX)
tot = collections.Counter()
rows = []
for v in order:
    eqn = jaxpr.eqns[v - 1]
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        oracle.advance(v, rules=())
        continue
    out_shape = tuple(eqn.outvars[0].aval.shape)
    out_len = len(out_shape)
    invars = [iv for iv in eqn.invars if hasattr(iv, "aval")]
    if not invars:
        oracle.advance(v, rules=())
        continue
    proxy = tuple(invars[0].aval.shape)          # the head's primal proxy
    sizes = list(out_shape) + list(proxy)        # == compute_static_axis_state
    try:
        fp, fc, nf = oracle.face_masks(v, MAX_FACES)
        faces = oracle.probe_faces(v, approx=True)
    except Exception as exc:
        print(f"  v={v} probe failed: {type(exc).__name__}: {exc}", flush=True)
        oracle.advance(v, rules=())
        continue
    for k in range(int(nf)):
        if k >= len(faces):
            break
        st = faces[k]
        live = [int(d.logical_size) for d in
                (tuple(st.out_dims) + tuple(st.primal_dims))]
        for i in range(N_AX):
            for j in range(N_AX):
                if not fp[k, i, j]:
                    continue
                # only look at the canonical (out, primal) orientation the
                # wire format can actually address
                if not (i < out_len and j >= out_len):
                    continue
                if (j - out_len) >= len(proxy):
                    continue
                base, span = diag_pair_factor_space(st, i, j)
                g_head = math.gcd(int(sizes[i]), int(sizes[j]))
                ok_head = bool(g_head > 1) and rule_is_legal(
                    st, Diag(i, j, g_head), max_dims=N_AX)
                divs = _divisors_above_one(span) if span > 1 else []
                best = base * divs[-1] if divs else 0
                ok_face = bool(divs) and rule_is_legal(
                    st, Diag(i, j, best), max_dims=N_AX)
                tot["pairs"] += 1
                tot["head_ok"] += int(ok_head)
                tot["face_ok"] += int(ok_face)
                tot["coupled"] += int(base > 1)
                if not ok_head:
                    # classify the failing clause
                    if not diag_valid_mask(st, N_AX)[i, j]:
                        why = "valid_mask"
                    elif span <= 1 or base <= 0:
                        why = "span<=1"
                    elif g_head % base:
                        why = "factor%base"
                    else:
                        d = g_head // base
                        why = "d==1" if d <= 1 else "span%d"
                    tot["why_" + why] += 1
                rows.append((v, k, i, j, sizes[i], sizes[j], g_head,
                             base, span, best, ok_head, ok_face,
                             tuple(live)))
    oracle.advance(v, rules=())

print(f"\nadmitted (vertex,face,i,j) pairs examined: {tot['pairs']}", flush=True)
print(f"  head factor = nominal gcd  LEGAL on the live face: "
      f"{tot['head_ok']} ({tot['head_ok']/max(tot['pairs'],1):.3f})", flush=True)
print(f"  per-face legal factor      LEGAL on the live face: "
      f"{tot['face_ok']} ({tot['face_ok']/max(tot['pairs'],1):.3f})", flush=True)
print(f"  already-COUPLED (base>1) faces: {tot['coupled']}", flush=True)
for k2 in sorted(k3 for k3 in tot if k3.startswith("why_")):
    print(f"  reject reason {k2[4:]:>12s}: {tot[k2]}", flush=True)
print("\nfirst 40 rows  (v, face, i, j, Nnom_i, Nnom_j, g_head, base, span, "
      "best_face_factor, head_ok, face_ok, live_logical_sizes)", flush=True)
for r in rows[:40]:
    print("   ", r, flush=True)

print("\n" + "=" * 78, flush=True)
print("PART C: end-to-end applied fraction with PLANTED DIAG rows", flush=True)
print("=" * 78, flush=True)
nr = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)


def episode(plants):
    """plants: {pos: (F, S, 3) int array}"""
    consume_per_face_stats()
    st = env.reset()
    fs = np.zeros((MAX_FACES,), np.int32)
    for pos, v in enumerate(order):
        fr = plants.get(pos)
        if fr is None:
            fr = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
        st = env.step(st, StepAction(jnp.asarray(int(v), jnp.int32), nr,
                                     jnp.asarray(fr), jnp.asarray(fs))).state
    return np.asarray(st.reward, np.float64), consume_per_face_stats()


def build_plan(mode, slots):
    """mode: 'head' (nominal gcd) | 'face' (largest legal per-face d)."""
    o2 = LiveVertexMaskOracle(jaxpr, list(closed.literals), list(xs), argnums,
                              max_axes=N_AX)
    plants = {}
    for pos, v in enumerate(order):
        fr = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
        eqn = jaxpr.eqns[v - 1]
        ok_eqn = bool(eqn.outvars) and hasattr(eqn.outvars[0], "aval")
        if ok_eqn:
            out_len = len(eqn.outvars[0].aval.shape)
            invars = [iv for iv in eqn.invars if hasattr(iv, "aval")]
            sizes = list(eqn.outvars[0].aval.shape) + (
                list(invars[0].aval.shape) if invars else [])
            try:
                fp, fc, nf = o2.face_masks(v, MAX_FACES)
                faces = o2.probe_faces(v, approx=True)
            except Exception:
                nf, faces = 0, []
            for k in range(int(nf)):
                if k >= len(faces):
                    break
                st = faces[k]
                pick = None
                for i in range(N_AX):
                    for j in range(N_AX):
                        if not fp[k, i, j]:
                            continue
                        if not (i < out_len and j >= out_len):
                            continue
                        if j >= len(sizes):
                            continue
                        base, span = diag_pair_factor_space(st, i, j)
                        if mode == "head":
                            f = math.gcd(int(sizes[i]), int(sizes[j]))
                        else:
                            divs = _divisors_above_one(span)
                            if not divs:
                                continue
                            f = base * divs[-1]
                        if f <= 1:
                            continue
                        pick = (i, j - out_len, f)
                        break
                    if pick:
                        break
                if pick:
                    for s in slots:
                        fr[k, s, :] = np.asarray(pick, np.int32)
        plants[pos] = fr
        o2.advance(v, rules=())
    return plants


r_ex, s_ex = episode({})
print(f"exact              cos={r_ex[COS]:.8f}  stats={s_ex}", flush=True)
for mode in ("head", "face"):
    for slots in ([0], [1], [2], [0, 1, 2]):
        pl = build_plan(mode, slots)
        r, s = episode(pl)
        a, sk = s.get("applied_diag", 0), s.get("skipped_diag", 0)
        frac = a / max(a + sk, 1)
        print(f"mode={mode:4s} slots={str(slots):<10s} cos={r[COS]:.8f} "
              f"applied_diag={a:5d} skipped_diag={sk:5d} frac={frac:.4f} "
              f"raised={s.get('skipped_raised', 0)}", flush=True)
print("DONE", flush=True)
