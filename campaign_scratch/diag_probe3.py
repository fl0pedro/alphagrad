"""Prototype of PER-FACE DIAG masking: reproject an illegal request onto the
LIVE operand's own legal (pair, factor) set. Measures applied fraction, cosine
and op counts against today's behaviour."""
import os
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
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

print("REWARD_INDEX:", REWARD_INDEX, flush=True)
COS = REWARD_INDEX["cosine_sim"]
N_AX = 8
MODE = [os.environ.get("DIAG_MODE", "off")]   # off | factor | pair
RULE = [os.environ.get("DIAG_RULE", "largest")]
_orig = M.rule_is_legal


def _divs(n):
    return [d for d in range(2, int(n) + 1) if n % d == 0]


def _pick_factor(base, span, want):
    ds = _divs(span)
    if not ds:
        return None
    if RULE[0] == "largest":
        d = ds[-1]
    elif RULE[0] == "smallest":
        d = ds[0]
    else:                                    # nearest to the requested factor
        d = min(ds, key=lambda x: abs(base * x - int(want)))
    return base * d


def reproject(st, rule):
    """Return a Diag legal on THIS operand, or None."""
    if MODE[0] == "off":
        return None
    vm = M.diag_valid_mask(st, N_AX)
    if 0 <= rule.i < N_AX and 0 <= rule.j < N_AX and vm[rule.i, rule.j]:
        base, span = M.diag_pair_factor_space(st, rule.i, rule.j)
        f = _pick_factor(base, span, rule.factor)
        return None if f is None else Diag(rule.i, rule.j, f)
    if MODE[0] != "pair":
        return None
    # Pair illegal here: fall back to a legal pair on this operand. Prefer the
    # one sharing the requested OUT index, then lowest index order (stable).
    cands = [(i, j) for i in range(N_AX) for j in range(N_AX) if vm[i, j]]
    if not cands:
        return None
    cands.sort(key=lambda ij: (ij[0] != rule.i, ij[1] != rule.j, ij))
    i, j = cands[0]
    base, span = M.diag_pair_factor_space(st, i, j)
    f = _pick_factor(base, span, rule.factor)
    return None if f is None else Diag(i, j, f)


_STATS = collections.Counter()


def _spy(st, rule, *, max_dims=8, max_axes=8):
    return _orig(st, rule, max_dims=max_dims, max_axes=max_axes)


def patched_hook(rules, *, max_dims=8, max_axes=8, stats=None, gated=False):
    from graphax.sparse.micro_actions import (
        apply_compress, apply_diag, apply_quant, Compress, Quant)

    def _bump(k):
        if stats is None or (gated and not M.face_counts_armed()):
            return
        stats[k] = stats.get(k, 0) + 1

    coupled, _ = M.couple_quant_rules(rules)

    def _hook(st):
        cur = st
        for rule in coupled:
            kind = ("diag" if isinstance(rule, Diag) else
                    "compress" if isinstance(rule, Compress) else
                    "quant" if isinstance(rule, Quant) else "other")
            r = rule
            if not M.rule_is_legal(cur, r, max_dims=max_dims,
                                   max_axes=max_axes):
                if isinstance(rule, Diag):
                    alt = reproject(cur, rule)
                    if alt is not None and M.rule_is_legal(
                            cur, alt, max_dims=max_dims, max_axes=max_axes):
                        r = alt
                        _STATS["repaired"] += 1
                    else:
                        b, sp = M.diag_pair_factor_space(cur, rule.i, rule.j)
                        if b > 1 and sp == 1:
                            _STATS["noop"] += 1
                        _bump("skipped")
                        _bump(f"skipped_{kind}")
                        continue
                else:
                    _bump("skipped")
                    _bump(f"skipped_{kind}")
                    continue
            try:
                if isinstance(r, Diag):
                    cur = apply_diag(cur, r)
                elif isinstance(r, Compress):
                    cur = apply_compress(cur, r)
                elif isinstance(r, Quant):
                    cur = apply_quant(cur, r)
                else:
                    _bump("skipped")
                    _bump(f"skipped_{kind}")
                    continue
                _bump("applied")
                _bump(f"applied_{kind}")
            except ValueError:
                _bump("skipped_raised")
                _bump(f"skipped_{kind}")
        return cur

    return _hook


M.make_live_masked_hook = patched_hook
import alphagrad.approx.env as E
E.make_live_masked_hook = patched_hook


def run_target(TARGET):
    key = jax.random.PRNGKey(0)
    fn = get_fn(TARGET)
    xs = get_args(TARGET, key)
    argnums = tuple(range(len(xs)))
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)
    env = VertexEliminationEnv.from_jaxpr(closed, args=xs, argnums=argnums,
                                          num_envs=0, target_fun=fn,
                                          per_face=True)
    order = [int(x) for x in np.asarray(env.valid_vertices)][::-1]
    jaxpr = env.config.jaxpr
    nr = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)

    def build_plan():
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
                except Exception:
                    nf = 0
                for k in range(int(nf)):
                    done = False
                    for i in range(N_AX):
                        for j in range(N_AX):
                            if not fp[k, i, j] or not (i < out_len
                                                       and j >= out_len):
                                continue
                            if j >= len(sizes):
                                continue
                            g = math.gcd(int(sizes[i]), int(sizes[j]))
                            if g <= 1:
                                continue
                            for s in range(FACE_SLOTS):
                                fr[k, s, :] = np.asarray(
                                    (i, j - out_len, g), np.int32)
                            done = True
                            break
                        if done:
                            break
            plants[pos] = fr
            o2.advance(v, rules=())
        return plants

    plan = build_plan()

    def episode(plants):
        consume_per_face_stats()
        _STATS.clear()
        st = env.reset()
        fs = np.zeros((MAX_FACES,), np.int32)
        for pos, v in enumerate(order):
            fr = plants.get(pos, np.full((MAX_FACES, FACE_SLOTS, 3), -1,
                                         np.int32))
            st = env.step(st, StepAction(jnp.asarray(int(v), jnp.int32), nr,
                                         jnp.asarray(fr),
                                         jnp.asarray(fs))).state
        return np.asarray(st.reward, np.float64), consume_per_face_stats()

    mode, rule = MODE[0], RULE[0]
    r, s = episode(plan)
    a, sk = s.get("applied_diag", 0), s.get("skipped_diag", 0)
    print(f"RESULT target={TARGET} |V|={len(order)} mode={mode:<7s} "
          f"rule={rule:<9s} applied={a:5d} skipped={sk:5d} "
          f"frac={a/max(a+sk,1):.4f} noop_skips={_STATS['noop']:5d} "
          f"repaired={_STATS['repaired']:5d} raised={s.get('skipped_raised',0)} "
          f"cos={r[COS]:.6f} muls={r[0]:.0f} flops={r[1]:.0f} "
          f"bytes={r[4]:.0f} peak={r[5]:.0f}", flush=True)


for t in os.environ.get("VERDICT_TARGETS", "NeuralNetwork,Helmholtz").split(","):
    try:
        run_target(t.strip())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  {t}: FAILED {type(exc).__name__}: {exc}", flush=True)
print("DONE", flush=True)
