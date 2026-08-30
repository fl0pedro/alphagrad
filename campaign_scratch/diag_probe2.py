"""Per-invocation autopsy of every DIAG rejection on the LIVE measured trace."""
import os
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_QUALITY_METRIC", "cosine")

import math
import collections
import warnings
warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
import numpy as np

import alphagrad.approx.common.masks as M
from alphagrad.approx.env import (
    FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX, REWARD_INDEX,
    StepAction, VertexEliminationEnv, consume_per_face_stats,
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
LOG = []
_orig = M.rule_is_legal


def _spy(st, rule, *, max_dims=8, max_axes=8):
    ok = _orig(st, rule, max_dims=max_dims, max_axes=max_axes)
    if isinstance(rule, Diag):
        dims = tuple(st.out_dims) + tuple(st.primal_dims)
        vm = M.diag_valid_mask(st, max_dims)
        base, span = M.diag_pair_factor_space(st, rule.i, rule.j)
        if ok:
            why = "OK"
        elif not (0 <= rule.i < max_dims and 0 <= rule.j < max_dims):
            why = "range"
        elif not vm[rule.i, rule.j]:
            if rule.i >= len(dims) or rule.j >= len(dims):
                why = "vm:out_of_rank"
            elif (rule.i < len(st.out_dims)) == (rule.j < len(st.out_dims)):
                why = "vm:same_side"
            elif span <= 1:
                why = "vm:span<=1(gcd)"
            else:
                why = "vm:already_paired"
        elif span <= 1 or base <= 0:
            why = "span<=1"
        elif rule.factor % base:
            why = "factor%base"
        elif rule.factor // base <= 1:
            why = "d==1"
        else:
            why = "span%d"
        LOG.append((why, int(rule.i), int(rule.j), int(rule.factor),
                    int(base), int(span), len(st.out_dims), len(dims),
                    tuple(int(d.logical_size) for d in dims),
                    tuple(bool(d.is_sparse) for d in dims)))
    return ok


M.rule_is_legal = _spy

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
nr = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)


def _divs(n):
    return [d for d in range(2, int(n) + 1) if n % d == 0]


def build_plan(slots):
    o2 = M.LiveVertexMaskOracle(jaxpr, list(closed.literals), list(xs),
                                argnums, max_axes=N_AX)
    plants = {}
    for pos, v in enumerate(order):
        fr = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
        eqn = jaxpr.eqns[v - 1]
        if eqn.outvars and hasattr(eqn.outvars[0], "aval"):
            out_len = len(eqn.outvars[0].aval.shape)
            iv = [a for a in eqn.invars if hasattr(a, "aval")]
            sizes = list(eqn.outvars[0].aval.shape) + (
                list(iv[0].aval.shape) if iv else [])
            try:
                fp, fc, nf = o2.face_masks(v, MAX_FACES)
                faces = o2.probe_faces(v, approx=True)
            except Exception:
                nf, faces = 0, []
            for k in range(int(nf)):
                if k >= len(faces):
                    break
                st = faces[k]
                for i in range(N_AX):
                    done = False
                    for j in range(N_AX):
                        if not fp[k, i, j] or not (i < out_len and j >= out_len):
                            continue
                        if j >= len(sizes):
                            continue
                        base, span = M.diag_pair_factor_space(st, i, j)
                        d = _divs(span)
                        if not d:
                            continue
                        for s in slots:
                            fr[k, s, :] = np.asarray(
                                (i, j - out_len, base * d[-1]), np.int32)
                        done = True
                        break
                    if done:
                        break
        plants[pos] = fr
        o2.advance(v, rules=())
    return plants


def episode(plants):
    consume_per_face_stats()
    LOG.clear()
    st = env.reset()
    fs = np.zeros((MAX_FACES,), np.int32)
    for pos, v in enumerate(order):
        fr = plants.get(pos, np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32))
        st = env.step(st, StepAction(jnp.asarray(int(v), jnp.int32), nr,
                                     jnp.asarray(fr), jnp.asarray(fs))).state
    return np.asarray(st.reward, np.float64), consume_per_face_stats()


for slots in ([2], [0, 1, 2]):
    pl = build_plan(slots)
    r, s = episode(pl)
    c = collections.Counter(x[0] for x in LOG)
    print(f"\n### slots={slots} cos={r[COS]:.6f} applied={s.get('applied_diag',0)} "
          f"skipped={s.get('skipped_diag',0)} raised={s.get('skipped_raised',0)}",
          flush=True)
    print("   reason histogram (all rule_is_legal calls, incl. unarmed replays):",
          flush=True)
    for w, n in c.most_common():
        print(f"     {w:>18s} {n}", flush=True)
    seen = set()
    print("   sample rows (why,i,j,factor,base,span,n_out,n_dims,logical,sparse):",
          flush=True)
    for row in LOG:
        if row[0] in seen:
            continue
        seen.add(row[0])
        print("    ", row, flush=True)
print("DONE", flush=True)
