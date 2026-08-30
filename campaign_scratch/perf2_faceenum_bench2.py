"""perf2_faceenum_bench2: cold rebuild vs prefix cache vs LIVE STATE.

Drives `env._face_transforms_for_order` exactly the way `_callback` does --
prefix lengths 1..T with `n_envs` chains INTERLEAVED per step, which is what
`pure_callback(vmap_method="sequential")` does under --num-envs N -- for three
arms:

  cold   stateless rebuild + full replay every step   (O(T^2) per episode)
  cache  ALPHAGRAD_FACE_ENUM_CACHE=1 pop-and-extend dict
  live   ALPHAGRAD_FACE_LIVE_STATE=1 (default) live chain, advance by one

Reports per-step wall and the CUMULATIVE ELIMINATION COUNT, which is the
asymptotic quantity: T*n_envs is O(T), T(T+1)/2*n_envs is O(T^2). All three
arms' enumerations are compared entry by entry.

usage: perf2_faceenum_bench2.py TARGET N_ENVS TMAX COMPRESS_FRAC OUT.json
"""
import os, sys, json, time
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np
import jax

TARGET = sys.argv[1] if len(sys.argv) > 1 else "TransformerLM"
N_ENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CFRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
OUT = sys.argv[5] if len(sys.argv) > 5 else "perf2_faceenum2.json"

from alphagrad.approx.common.examples import (
    get_fn, get_args, infer_argnums, grad_target_setup)
from graphax.core import _build_graph
from alphagrad.approx import env as E


def traced_inlined(fn, xs):
    from graphax import inline_call_primitives
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    try:
        from jax.extend.core import ClosedJaxpr
    except ImportError:
        from jax.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


key = jax.random.PRNGKey(0)
fn0 = get_fn(TARGET)
xs0 = get_args(TARGET, key, dataset="wikitext2")
argnums0 = infer_argnums(TARGET)
fn, xs, argnums = grad_target_setup(
    SimpleNamespace(measure_grad=True, seed_vertices=False), fn0, xs0, TARGET)
cj = traced_inlined(fn, xs)
jaxpr, consts = cj.jaxpr, cj.literals

_, _, _, vo = _build_graph(jaxpr, tuple(xs), consts, argnums)
valid = [i for i, eq in enumerate(jaxpr.eqns, 1)
         if eq.outvars[0] not in jaxpr.outvars or i in vo]

B = E.derived_max_faces(jaxpr, argnums, consts, xs)
E.configure_max_faces(B)
F = E.MAX_FACES
print(f"[perf2] target={TARGET} eqns={len(jaxpr.eqns)} elim={len(valid)} "
      f"derived_max_faces={B} MAX_FACES={F}", flush=True)

config = E.EnvConfig(jaxpr=jaxpr, argnums=tuple(argnums), has_aux=False,
                     sparse=False, cmp_type="latency", mem_type="peak_memory",
                     per_face=True)

T = len(valid) if TMAX <= 0 else min(TMAX, len(valid))
rng = np.random.default_rng(0)

orders, faces, skips, specs = [], [], [], []
for e in range(N_ENVS):
    o = np.array(rng.permutation(valid)[:T], dtype=np.int64)
    fa = np.full((T, F, E.FACE_SLOTS, 3), -1, dtype=np.int64)
    sk = np.zeros((T, F), dtype=np.int64)
    sp = np.full((T, E.MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int64)
    for k in range(T):
        for f in range(1 + int(rng.random() < 0.24)):
            sk[k, f] = 1
            if rng.random() < CFRAC:
                fa[k, f, int(rng.integers(0, E.FACE_SLOTS))] = [
                    E.COMPRESS_SENTINEL, 0, 1]
    orders.append(o); faces.append(fa); skips.append(sk); specs.append(sp)


def canon(out):
    from graphax import SKIP_FACE
    return {str(v): {str(k): ("SKIP" if val is SKIP_FACE else
                              "T%d" % len(val) if isinstance(val, tuple)
                              else type(val).__name__)
                     for k, val in per.items()}
            for v, per in out.items()}


ARMS = {"cold": (0, 0), "cache": (1, 0), "live": (0, 1)}


def run(arm):
    cache, live = ARMS[arm]
    os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = str(cache)
    E._FACE_LIVE_STATE = bool(live)
    E._FACE_ENUM_CACHE.clear()
    E._LIVE_CHAINS.clear()
    E.consume_live_chain_stats()
    E._FACE_ENUM_STATS.update(ext=0, cold=0, compress=0, elims=0, calls=0,
                              build=0)
    E.consume_profile()
    per_step, sigs = [], []
    for k in range(1, T + 1):
        t_step = 0.0
        for e in range(N_ENVS):
            o_list = [int(x) for x in orders[e][:k]]
            t0 = time.perf_counter()
            out = E._face_transforms_for_order(
                config, consts, xs, o_list, specs[e][:k],
                faces[e][:k], skips[e][:k])
            t_step += time.perf_counter() - t0
            sigs.append(canon(out))
        ne = (E._LIVE_CHAIN_STATS["elims"] if live
              else E._FACE_ENUM_STATS["elims"])
        per_step.append(dict(k=k, enum_s=t_step, elims=ne))
    return (per_step, sigs,
            dict(E._FACE_ENUM_STATS), dict(E._LIVE_CHAIN_STATS),
            E.consume_profile())


res, sig_by_arm = {}, {}
for arm in ("cold", "cache", "live"):
    t0 = time.perf_counter()
    per_step, sigs, fst, lst, prof = run(arm)
    wall = time.perf_counter() - t0
    sig_by_arm[arm] = sigs
    tot = sum(r["enum_s"] for r in per_step)
    res[arm] = dict(wall_s=wall, enum_s=tot, face_enum_stats=fst,
                    live_stats=lst, prof=prof, per_step=per_step)
    print(f"\n[perf2] === {arm} === wall={wall:.1f}s enum={tot:.1f}s", flush=True)
    print(f"[perf2]   face_enum_stats={fst}", flush=True)
    print(f"[perf2]   live_stats={lst}", flush=True)
    print(f"[perf2]   prof={ {k: round(v,3) for k,v in prof.items()} }",
          flush=True)
    print(f"[perf2]   elims={per_step[-1]['elims']}  "
          f"O(T)={T*N_ENVS}  O(T^2)={T*(T+1)//2*N_ENVS}", flush=True)
    step = max(1, T // 12)
    print("[perf2]      k    enum_ms/step   cum_elims")
    for r in per_step[::step]:
        print(f"[perf2]   {r['k']:4d}  {r['enum_s']*1e3:12.1f}  "
              f"{r['elims']:10d}", flush=True)

ok_cache = sig_by_arm["cold"] == sig_by_arm["cache"]
ok_live = sig_by_arm["cold"] == sig_by_arm["live"]
n = len(sig_by_arm["cold"])
print(f"\n[perf2] EQUIVALENCE vs cold rebuild ({n} enumerations): "
      f"cache={ok_cache}  live={ok_live}", flush=True)
res["equiv_cache"], res["equiv_live"] = bool(ok_cache), bool(ok_live)
res["meta"] = dict(target=TARGET, n_envs=N_ENVS, T=T, max_faces=F,
                   compress_frac=CFRAC, eqns=len(jaxpr.eqns))
a, b, c = (res["cold"]["enum_s"], res["cache"]["enum_s"], res["live"]["enum_s"])
print(f"[perf2] SPEEDUP enum-only vs cold: cache {a/max(b,1e-9):.2f}x   "
      f"live {a/max(c,1e-9):.2f}x   (cold {a:.1f}s / cache {b:.1f}s / "
      f"live {c:.1f}s)", flush=True)
json.dump(res, open(OUT, "w"))
print("[perf2] wrote", OUT, flush=True)
