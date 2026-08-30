"""Applied-fraction of DIAG, before vs after --diag-per-face, on the REAL
implementation (no monkeypatching). One process per arm: a compile-cache HIT
skips the traced elimination, so the per-face counters only describe the FIRST
measurement of a given plan."""
import os
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_QUALITY_METRIC", "cosine")

import math
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
try:
    from jax.extend.core import ClosedJaxpr
except ImportError:
    from jax._src.core import ClosedJaxpr

COS = REWARD_INDEX["cosine_sim"]
LAT = REWARD_INDEX["latency_ns"]
N_AX = 8
TARGET = os.environ.get("VERDICT_TARGET", "NeuralNetwork")
ON = os.environ.get("ARM", "off") == "on"
RULE = os.environ.get("DIAG_RULE", "largest")
REPAIR = os.environ.get("REPAIR", "1") != "0"

M.set_diag_per_face(ON, rule=RULE, repair_pair=REPAIR)

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


def build_plan():
    """The plan the LIVE head would emit: pair from face_masks, factor from the
    vertex's NOMINAL sizes (unified_face_policy._rows hardcodes gcd(N_i,N_j))."""
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
                        if not fp[k, i, j] or not (i < out_len and j >= out_len):
                            continue
                        if j >= len(sizes):
                            continue
                        g = math.gcd(int(sizes[i]), int(sizes[j]))
                        if g <= 1:
                            continue
                        for s in range(FACE_SLOTS):
                            fr[k, s, :] = np.asarray((i, j - out_len, g),
                                                     np.int32)
                        done = True
                        break
                    if done:
                        break
        plants[pos] = fr
        o2.advance(v, rules=())
    return plants


def episode(plants):
    consume_per_face_stats()
    st = env.reset()
    fs = np.zeros((MAX_FACES,), np.int32)
    for pos, v in enumerate(order):
        fr = plants.get(pos, np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32))
        st = env.step(st, StepAction(jnp.asarray(int(v), jnp.int32), nr,
                                     jnp.asarray(fr), jnp.asarray(fs))).state
    return np.asarray(st.reward, np.float64), consume_per_face_stats()


r, s = episode(build_plan())
a = s.get("applied_diag", 0)
sk = s.get("skipped_diag", 0)
noop = s.get("skipped_diag_noop", 0)
print(f"ARM target={TARGET} diag_per_face={'ON ' if ON else 'OFF'} "
      f"rule={RULE} repair_pair={int(REPAIR)} | "
      f"applied_diag={a:5d} skipped_diag={sk:5d} "
      f"frac={a/max(a+sk,1):.4f} noop={noop:5d} "
      f"frac_excl_noop={a/max(a+sk-noop,1):.4f} "
      f"repaired={s.get('repaired_diag',0):5d} "
      f"raised={s.get('skipped_raised',0)} | "
      f"cos={r[COS]:.6f} muls={r[0]:.0f} flops={r[1]:.0f} "
      f"bytes={r[4]:.0f} peak={r[5]:.0f} lat={r[LAT]:.0f}", flush=True)
