"""perf2_faceenum_bench: is the face-enum prefix cache ACTUALLY incremental?

Drives `env._face_transforms_for_order` exactly the way `_callback` does --
prefix lengths 1..T, `n_envs` independent chains INTERLEAVED per step (which
is what `pure_callback(vmap_method="sequential")` does under --num-envs N) --
and reports, per step:

  * wall inside the enum call
  * the KEY-construction wall (`cb.face_enum_key`, the O(prefix x MAX_FACES)
    `np.asarray(python_nested_list).tobytes()` loop)
  * the caller's dense `.tolist()` wall (`cb.face_tolist`), which is paid
    whether or not the cache is on
  * `ext` / `cold` / `compress` / `elims` -- elims is the ONLY asymptotic
    quantity that matters: O(T) total means incremental, O(T^2) means not.

Cache OFF and ON are run BACK TO BACK in one process and their face
ENUMERATIONS are compared entry by entry (equivalence is non-negotiable).

usage: perf2_faceenum_bench.py TARGET N_ENVS TMAX COMPRESS_FRAC OUT.json
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
OUT = sys.argv[5] if len(sys.argv) > 5 else "perf2_faceenum.json"

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

# --- per-env wire arrays, shaped exactly like the env's -------------------
orders, faces, skips, specs = [], [], [], []
for e in range(N_ENVS):
    o = np.array(rng.permutation(valid)[:T], dtype=np.int64)
    fa = np.full((T, F, E.FACE_SLOTS, 3), -1, dtype=np.int64)
    sk = np.zeros((T, F), dtype=np.int64)
    sp = np.full((T, E.MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int64)
    for k in range(T):
        # measured live-face occupancy is ~1.24 faces/vertex: touch face 0
        # (and sometimes face 1) only -- everything else stays padding.
        nf = 1 + int(rng.random() < 0.24)
        for f in range(nf):
            # SKIP rows only. `_face_dict_for_vertex` short-circuits a
            # skipped face BEFORE `decode_vertex_rule_specs`, so a synthetic
            # rule can never raise on a vertex whose operand it does not fit
            # -- while the expensive parts we are measuring (`faces_of`
            # enumeration, the elimination replay, and the cache KEY over
            # non-trivial wire arrays) are exercised identically.
            sk[k, f] = 1
            if rng.random() < CFRAC:
                # COMPRESS in the wire is what `_last_compress` reads; on a
                # skipped face it is never decoded.
                slot = int(rng.integers(0, E.FACE_SLOTS))
                fa[k, f, slot] = [E.COMPRESS_SENTINEL, 0, 1]
    orders.append(o); faces.append(fa); skips.append(sk); specs.append(sp)


def canon(out):
    """Comparable description of ONE enumeration: {vertex: {face_key: kind}}"""
    from graphax import SKIP_FACE
    d = {}
    for v, per in out.items():
        d[str(v)] = {str(k): ("SKIP" if val is SKIP_FACE else
                              "T%d" % len(val) if isinstance(val, tuple)
                              else type(val).__name__)
                     for k, val in per.items()}
    return d


def run(cache):
    os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "1" if cache else "0"
    E._FACE_ENUM_CACHE.clear()
    E._FACE_ENUM_STATS.update(ext=0, cold=0, compress=0, elims=0, calls=0,
                              build=0)
    E.consume_profile()
    per_step, sigs = [], []
    for k in range(1, T + 1):
        t_step = 0.0
        t_tol = 0.0
        for e in range(N_ENVS):
            o_list = [int(x) for x in orders[e][:k]]
            sp_list = specs[e][:k].tolist()
            t0 = time.perf_counter()
            fr = faces[e][:k].tolist()
            fs = skips[e][:k].tolist()
            t1 = time.perf_counter()
            out = E._face_transforms_for_order(
                config, consts, xs, o_list, sp_list, fr, fs,
                honor_last_compress=True)
            t2 = time.perf_counter()
            t_tol += t1 - t0
            t_step += t2 - t1
            sigs.append(canon(out))
        per_step.append(dict(k=k, enum_s=t_step, tolist_s=t_tol,
                             elims=E._FACE_ENUM_STATS["elims"]))
    prof = E.consume_profile()
    st = dict(E._FACE_ENUM_STATS)
    return per_step, st, prof, sigs


res = {}
for cache in (0, 1):
    t0 = time.perf_counter()
    per_step, st, prof, sigs = run(cache)
    wall = time.perf_counter() - t0
    res["cache%d" % cache] = dict(
        wall_s=wall, stats=st, prof=prof, per_step=per_step)
    res["sig%d" % cache] = sigs
    print(f"\n[perf2] === cache={cache} === wall={wall:.1f}s  stats={st}",
          flush=True)
    print(f"[perf2]     prof={ {k: round(v,3) for k,v in prof.items()} }",
          flush=True)
    tot_enum = sum(r["enum_s"] for r in per_step)
    tot_tol = sum(r["tolist_s"] for r in per_step)
    print(f"[perf2]     enum_total={tot_enum:.1f}s  tolist_total={tot_tol:.1f}s"
          f"  elims={st['elims']} (O(T) would be {T*N_ENVS}, "
          f"O(T^2) would be {T*(T+1)//2*N_ENVS})", flush=True)
    step = max(1, T // 12)
    print("[perf2]     k     enum_s/step  tolist_s/step  cum_elims")
    for r in per_step[::step]:
        print(f"[perf2]   {r['k']:4d}  {r['enum_s']*1e3:10.1f}ms "
              f"{r['tolist_s']*1e3:12.1f}ms  {r['elims']:8d}", flush=True)

ok = res["sig0"] == res["sig1"]
n = len(res["sig0"])
bad = [i for i in range(n) if res["sig0"][i] != res["sig1"][i]]
print(f"\n[perf2] EQUIVALENCE cached==uncached: {ok}  "
      f"({n - len(bad)}/{n} identical)", flush=True)
res["equivalent"] = bool(ok)
res["n_mismatch"] = len(bad)
res["meta"] = dict(target=TARGET, n_envs=N_ENVS, T=T, max_faces=F,
                   compress_frac=CFRAC, eqns=len(jaxpr.eqns))
a = sum(r["enum_s"] for r in res["cache0"]["per_step"])
b = sum(r["enum_s"] for r in res["cache1"]["per_step"])
print(f"[perf2] SPEEDUP enum-only: {a/max(b,1e-9):.2f}x  ({a:.1f}s -> {b:.1f}s)",
      flush=True)
for k in ("sig0", "sig1"):
    res.pop(k)
json.dump(res, open(OUT, "w"))
print("[perf2] wrote", OUT, flush=True)
