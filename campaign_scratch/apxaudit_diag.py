"""Why did the face path do nothing? Three checks, cheapest first."""
import os
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_QUALITY_METRIC", "cosine")

import jax
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.env import (
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX,
    QUANT_SENTINEL, REWARD_INDEX, StepAction, VertexEliminationEnv,
    consume_per_face_stats, decode_vertex_rule_specs, _face_dict_for_vertex,
)
from alphagrad.approx.common.examples import get_fn, get_args

COS = REWARD_INDEX["cosine_sim"]

# ---------------------------------------------------------------- toy control
_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)
_Q = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 17.0 + 0.3)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _square(x):
    e = _M @ x
    return _P @ e, _Q @ e


def episode(env, order, plants):
    consume_per_face_stats()
    st = env.reset()
    nr = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    for k, v in enumerate(order):
        fr = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
        fs = np.zeros((MAX_FACES,), np.int32)
        p = plants.get(k)
        if p:
            for slot, row in p.items():
                if slot == "skip":
                    fs[:] = 1
                else:
                    fr[:, int(slot), :] = np.asarray(row, np.int32)
        st = env.step(st, StepAction(jnp.asarray(int(v), jnp.int32), nr,
                                     jnp.asarray(fr), jnp.asarray(fs))).state
    return np.asarray(st.reward, np.float64), consume_per_face_stats()


print("=" * 70, "\nCHECK 1: the TOY graph the passing test uses\n", "=" * 70,
      flush=True)
cj = jax.make_jaxpr(_square)(_X4)
env = VertexEliminationEnv.from_jaxpr(cj, args=[_X4], argnums=(0,), num_envs=0,
                                      target_fun=_square, per_face=True)
order = [int(x) for x in np.asarray(env.valid_vertices)]
print("toy order", order, flush=True)
r0, s0 = episode(env, order, {})
print(f"  exact          cos={r0[COS]:.8f} stats={s0}", flush=True)
for k in range(len(order)):
    rs, ss = episode(env, order, {k: {"skip": True}})
    print(f"  SKIP at pos {k} (v={order[k]}) cos={rs[COS]:.8f} stats={ss}",
          flush=True)

print("\n" + "=" * 70, "\nCHECK 2: faces per vertex on NeuralNetwork\n",
      "=" * 70, flush=True)
from graphax import inline_call_primitives, faces_of
from graphax.incremental import IncrementalJaxpr
try:
    from jax.extend.core import ClosedJaxpr
except ImportError:
    from jax._src.core import ClosedJaxpr

key = jax.random.PRNGKey(0)
fn = get_fn("NeuralNetwork")
xs = get_args("NeuralNetwork", key)
argnums = tuple(range(len(xs)))
cjn = jax.make_jaxpr(fn)(*xs)
jx, consts = inline_call_primitives(cjn.jaxpr, cjn.literals)
closed = cjn if jx is cjn.jaxpr else ClosedJaxpr(jx, consts)
envn = VertexEliminationEnv.from_jaxpr(closed, args=xs, argnums=argnums,
                                       num_envs=0, target_fun=fn, per_face=True)
ordn = [int(x) for x in np.asarray(envn.valid_vertices)][::-1]
print("NN plan", ordn, flush=True)
ij = IncrementalJaxpr(envn.config.jaxpr, tuple(envn.config.argnums),
                      list(closed.literals), list(xs), track_faces=False)
zero_row = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
zero_skip = np.zeros((MAX_FACES,), np.int32)
for k, v in enumerate(ordn):
    keys = faces_of(ij.graph, ij.tgraph, int(v), envn.config.jaxpr)
    rows = {}
    for ax in range(9):
        if decode_vertex_rule_specs(envn.config.jaxpr, v,
                                    [[COMPRESS_SENTINEL, ax, 0]]
                                    + [[-1, -1, 0]] * (MAX_RULES_PER_VERTEX-1)):
            rows["compress"] = ax
            break
    eqn = envn.config.jaxpr.eqns[v - 1]
    oshape = eqn.outvars[0].aval.shape if eqn.outvars else ()
    pshapes = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
    print(f"  pos{k:3d} v={v:3d} nfaces={len(keys)} prim={eqn.primitive.name:>16s}"
          f" out={oshape} in={pshapes} compress_ax={rows.get('compress')}",
          flush=True)
    ij.eliminate(int(v), (), None)

print("\n" + "=" * 70, "\nCHECK 3: NN, SKIP at every position\n", "=" * 70,
      flush=True)
r0, s0 = episode(envn, ordn, {})
print(f"  exact cos={r0[COS]:.8f} stats={s0}", flush=True)
for k in range(0, len(ordn)):
    rs, ss = episode(envn, ordn, {k: {"skip": True}})
    flag = "" if abs(rs[COS] - r0[COS]) > 1e-7 else "   <-- NO EFFECT"
    print(f"  SKIP pos{k:3d} (v={ordn[k]:3d}) cos={rs[COS]:.8f} "
          f"applied={ss.get('applied',0)} skipped={ss.get('skipped',0)}{flag}",
          flush=True)
print("DONE", flush=True)
