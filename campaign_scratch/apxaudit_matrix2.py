"""Audit: does EVERY approximation type and combination DO SOMETHING at
INTERMEDIATE (non-final) eliminations?

Drives the REAL chain (StepAction.face_rows/face_skip -> EnvState -> _callback
-> _face_transforms_for_order -> make_live_masked_hook -> jacve face_transforms
-> terminal measurement), exactly like tests/face_actions_env_test.py.

The env computes the Jacobian cosine ONLY at the terminal step
(env.py:3729 ``if is_terminal and _qmetric == "cosine"``), so the decisive
number is the TERMINAL cos of a rule planted at an INTERMEDIATE position.

Columns per (position, config):
  (a) decode_survives -- _face_transforms_for_order at EVERY prefix length
                         k = p+1..T still yields a live hook for vertex v
  (b) applied         -- _PER_FACE_STATS applied_<kind>, counted ONLY inside
                         env's armed measurement scope (_do_compile_approx)
  (c) cos_env         -- terminal reward slot 6 vs exact AD, same order
      cos_oracle      -- INDEPENDENT check: the same decoded rules applied
                         through jacve's per-vertex ``transforms`` at the same
                         vertex.  oracle != 1 but env == 1  =>  the face chain
                         lost it.  both == 1  =>  genuinely degenerate here.
  (d)(e) latency / peak memory, paired back-to-back against a control.
"""
import argparse
import json
import os
import time
import traceback

os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_QUALITY_METRIC", "cosine")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from alphagrad.approx.env import (  # noqa: E402
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX,
    QUANT_SENTINEL, REWARD_INDEX, StepAction, VertexEliminationEnv,
    consume_per_face_stats, decode_vertex_rule_specs,
    _face_transforms_for_order,
)
from alphagrad.approx.common.examples import get_fn, get_args  # noqa: E402
from alphagrad.approx.common.masks import make_live_masked_hook  # noqa: E402
from graphax.sparse.micro_actions import QUANT_DTYPES  # noqa: E402
from graphax import jacve, faces_of, inline_call_primitives  # noqa: E402
from graphax.incremental import IncrementalJaxpr  # noqa: E402

COS = REWARD_INDEX["cosine_sim"]
LAT = REWARD_INDEX["latency_ns"]
MEM = REWARD_INDEX["peak_memory"]
SLOT_NAME = ("pre", "post", "new")
EPS = 1e-6


def _dt(name):
    for i, d in enumerate(QUANT_DTYPES):
        if str(d) == name:
            return i
    return None


DT = {"f32": _dt("float32"), "bf16": _dt("bfloat16"), "f16": _dt("float16"),
      "i8": _dt("int8")}
DT["narrowfp"] = next((_dt(n) for n in ("float8_e4m3fn", "float8_e4m3",
                                        "float8_e5m2") if _dt(n) is not None),
                      None)


def _pad(row):
    r = [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]
    r[0] = list(row)
    return r


def _flat(t):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(t)])


def _cos(a, b):
    a = np.asarray(_flat(a), np.float64)
    b = np.asarray(_flat(b), np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ------------------------------------------------------------------- target
def build(name, measure_latency, seed=0):
    try:
        from jax.extend.core import ClosedJaxpr
    except ImportError:
        from jax._src.core import ClosedJaxpr
    fn = get_fn(name)
    xs = get_args(name, jax.random.PRNGKey(seed))
    argnums = tuple(range(len(xs)))
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=argnums, num_envs=0, target_fun=fn,
        cmp_type="latency" if measure_latency else "flops",
        mem_type="peak_memory", measure_latency=measure_latency, per_face=True,
        terminal_rewards_only=True, num_data_points=3, reps_per_point=3,
        latency_inner_reps=50)
    return env, closed, xs, argnums, fn


def rows_for(jaxpr, v):
    """One DECODING wire row of each kind for vertex ``v``."""
    out = {}
    eqn = jaxpr.eqns[v - 1]
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        return out
    out_len = len(eqn.outvars[0].aval.shape)
    prim = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
    maxp = max((len(p) for p in prim), default=0)
    for bi1 in range(max(out_len, 1)):
        for bi2 in range(max(maxp, 1)):
            for f in (-1, 2, 4, 8):
                if decode_vertex_rule_specs(jaxpr, v, _pad([bi1, bi2, f])):
                    out["diag"] = [bi1, bi2, f]
                    break
            if "diag" in out:
                break
        if "diag" in out:
            break
    for ax in range(9):
        if decode_vertex_rule_specs(jaxpr, v, _pad([COMPRESS_SENTINEL, ax, 0])):
            out["compress"] = [COMPRESS_SENTINEL, ax, 0]
            break
    for tag, di in DT.items():
        if di is not None:
            out["quant_" + tag] = [QUANT_SENTINEL, di, 0]
    return out


def face_counts(env, closed, xs, order):
    ij = IncrementalJaxpr(env.config.jaxpr, tuple(env.config.argnums),
                          list(closed.literals), list(xs), track_faces=False)
    n = []
    for v in order:
        n.append(len(faces_of(ij.graph, ij.tgraph, int(v), env.config.jaxpr)))
        ij.eliminate(int(v), (), None)
    return n


def make_configs(rows):
    c = [("SKIP", {"skip": True})]
    for kind, tag in (("diag", "DIAG"), ("compress", "COMPRESS"),
                      ("quant_bf16", "QUANT_bf16")):
        if kind in rows:
            for s in range(FACE_SLOTS):
                c.append((f"{tag}@{SLOT_NAME[s]}", {s: rows[kind]}))
    for tag in ("f32", "f16", "i8", "narrowfp"):
        k = "quant_" + tag
        if k in rows:
            lbl = "QUANT_f32(degen-ctl)" if tag == "f32" else f"QUANT_{tag}"
            c.append((f"{lbl}@new", {2: rows[k]}))

    def has(*ks):
        return all(k in rows for k in ks)
    if has("diag", "compress"):
        c.append(("DIAG@pre+COMPRESS@post", {0: rows["diag"],
                                             1: rows["compress"]}))
    if has("diag", "quant_bf16"):
        c.append(("DIAG@pre+QUANT@post", {0: rows["diag"],
                                          1: rows["quant_bf16"]}))
    if has("compress", "quant_bf16"):
        c.append(("COMPRESS@pre+QUANT@post", {0: rows["compress"],
                                              1: rows["quant_bf16"]}))
    if has("diag", "compress", "quant_bf16"):
        c.append(("DIAG@pre+COMPRESS@post+QUANT@new",
                  {0: rows["diag"], 1: rows["compress"],
                   2: rows["quant_bf16"]}))
    if has("compress"):
        c.append(("COMPRESS@pre+COMPRESS@post", {0: rows["compress"],
                                                 1: rows["compress"]}))
    if has("diag"):
        c.append(("SKIP+DIAG@pre", {"skip": True, 0: rows["diag"]}))
    return c


def _wire(plants, T):
    frs, fss = [], []
    for k in range(T):
        fr = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
        fs = np.zeros((MAX_FACES,), np.int32)
        for slot, row in plants.get(k, {}).items():
            if slot == "skip":
                fs[:] = 1
            else:
                fr[:, int(slot), :] = np.asarray(row, np.int32)
        frs.append(fr)
        fss.append(fs)
    return frs, fss


def episode(env, order, plants):
    consume_per_face_stats()
    st = env.reset()
    nr = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    frs, fss = _wire(plants, len(order))
    for k, v in enumerate(order):
        st = env.step(st, StepAction(jnp.asarray(int(v), jnp.int32), nr,
                                     jnp.asarray(frs[k]),
                                     jnp.asarray(fss[k]))).state
    return np.asarray(st.reward, np.float64), consume_per_face_stats()


def decode_survives(env, closed, xs, order, plants):
    T = len(order)
    specs = [[[-1, -1, 0]] * MAX_RULES_PER_VERTEX for _ in range(T)]
    frs, fss = _wire(plants, T)
    frs = [f.tolist() for f in frs]
    fss = [f.tolist() for f in fss]
    p0 = min(plants)
    bad = []
    for k in range(p0 + 1, T + 1):
        ft = _face_transforms_for_order(
            env.config, list(closed.literals), list(xs),
            [int(x) for x in order[:k]], specs[:k], frs[:k], fss[:k])
        v = int(order[p0])
        d = ft.get(v)
        live = 0
        if d:
            for val in d.values():
                if val is None:
                    continue
                if isinstance(val, tuple):
                    slots = val[0] if (val and isinstance(val[0], tuple)) else val
                    live += sum(1 for s in slots if s is not None)
                else:
                    live += 1
        if live == 0:
            bad.append(k)
    return (not bad), (f"lost at prefix lengths {bad[:5]}" if bad else "")


def oracle_cos(fn, xs, argnums, jaxpr, order, v, plant, ref):
    """Same decoded rules, applied through jacve's PER-VERTEX transforms."""
    rules = []
    for slot, row in plant.items():
        if slot == "skip":
            return None, "skip has no per-vertex analogue"
        rules += list(decode_vertex_rule_specs(jaxpr, v, _pad(row)))
    if not rules:
        return None, "no rules decoded"
    try:
        out = jax.jit(jacve(fn, list(order), argnums=argnums,
                            transforms=[(int(v),
                                         (make_live_masked_hook(tuple(rules)),))]
                            ))(*xs)
        return _cos(out, ref), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="NeuralNetwork")
    ap.add_argument("--phase", default="correctness",
                    choices=["correctness", "cost"])
    ap.add_argument("--npos", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    print(f"[cfg] target={a.target} phase={a.phase} "
          f"backend={jax.default_backend()}", flush=True)

    env, closed, xs, argnums, fn = build(a.target, a.phase == "cost")
    order = [int(x) for x in np.asarray(env.valid_vertices)][::-1]
    T = len(order)
    jaxpr = closed.jaxpr
    nf = face_counts(env, closed, xs, order)
    print(f"[cfg] eqns={len(jaxpr.eqns)} plan_len={T}  faces/vertex "
          f"nonzero at {sum(1 for x in nf if x)} of {T}", flush=True)

    ref = jax.jit(jacve(fn, list(order), argnums=argnums))(*xs)

    # eligible INTERMEDIATE positions: has faces AND has decodable rules
    elig = []
    for k in range(T - 1):          # exclude the terminal position
        if nf[k] == 0:
            continue
        r = rows_for(jaxpr, order[k])
        if not r:
            continue
        score = ("diag" in r) + ("compress" in r)
        elig.append((k, r, score))
    elig.sort(key=lambda t: t[0])
    if len(elig) <= a.npos:
        chosen = list(elig)
    else:
        idx = [round(i * (len(elig) - 1) / (a.npos - 1)) for i in range(a.npos)]
        chosen = [elig[i] for i in sorted(set(idx))]
    # positive control: the LAST position (where COMPRESS always worked)
    last_rows = rows_for(jaxpr, order[T - 1])
    if nf[T - 1] and last_rows:
        chosen.append((T - 1, last_rows, 0))
    print(f"[cfg] eligible intermediate positions {[e[0] for e in elig]}",
          flush=True)
    print(f"[cfg] AUDITING {[c[0] for c in chosen]} "
          f"(last={T-1} is the positive control)", flush=True)

    base, bstats = episode(env, order, {})
    print(f"[control] cos={base[COS]:.8f} lat={base[LAT]:.6g} "
          f"mem={base[MEM]:.6g}", flush=True)

    results = []
    for pos, rows, _ in chosen:
        v = order[pos]
        is_last = (pos == T - 1)
        print(f"\n{'='*92}\nPOS {pos}/{T-1} v={v} nfaces={nf[pos]} "
              f"{'(TERMINAL control)' if is_last else 'INTERMEDIATE'} "
              f"rows={sorted(rows)}\n{'='*92}", flush=True)
        for label, plant in make_configs(rows):
            rec = {"target": a.target, "pos": pos, "vertex": v,
                   "terminal_position": is_last, "config": label,
                   "nfaces": nf[pos],
                   "plant": {str(k): val for k, val in plant.items()}}
            try:
                ok_a, why = decode_survives(env, closed, xs, order,
                                            {pos: plant})
                rec["decode_survives"] = bool(ok_a)
                rec["decode_note"] = why
                r, stats = episode(env, order, {pos: plant})
                rec["cos_env"] = float(r[COS])
                rec["lat"] = float(r[LAT])
                rec["mem"] = float(r[MEM])
                rec["applied"] = int(stats.get("applied", 0))
                rec["skipped"] = int(stats.get("skipped", 0))
                rec["skipped_raised"] = int(stats.get("skipped_raised", 0))
                for kk in ("diag", "compress", "quant"):
                    rec[f"applied_{kk}"] = int(stats.get(f"applied_{kk}", 0))
                    rec[f"skipped_{kk}"] = int(stats.get(f"skipped_{kk}", 0))
                oc, onote = oracle_cos(fn, xs, argnums, jaxpr, order, v,
                                       plant, ref)
                rec["cos_oracle"] = oc
                rec["oracle_note"] = onote
                moved = abs(r[COS] - base[COS]) > EPS
                rec["moved_jacobian"] = bool(moved)
                omoved = (oc is not None and abs(oc - 1.0) > EPS)
                if moved:
                    _vd = "ACTS"
                elif rec["applied"] > 0:
                    _vd = "APPLIED_NO_EFFECT"
                elif rec["skipped"] or rec["skipped_raised"]:
                    _vd = "MASKED_ON_SLOT" if omoved else "MASKED_ON_SLOT_degen"
                elif omoved:
                    _vd = "SILENTLY_DISCARDED"
                else:
                    _vd = "DEGENERATE"
                rec["verdict"] = _vd
                print(f"  {label:34s} dec_ok={int(ok_a)} "
                      f"cos_env={r[COS]:.6f} "
                      f"cos_oracle={'  n/a  ' if oc is None else f'{oc:.6f}'} "
                      f"app={rec['applied']}(d{rec['applied_diag']}"
                      f"/c{rec['applied_compress']}/q{rec['applied_quant']}) "
                      f"skp={rec['skipped']} -> {rec['verdict']}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  {label:34s} ERROR {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)
                traceback.print_exc()
            results.append(rec)

    if a.phase == "cost":
        print(f"\n{'='*92}\nPAIRED COST x{a.pairs}\n{'='*92}", flush=True)
        for rec in [r for r in results if r.get("moved_jacobian")]:
            plant = {(k if k == "skip" else int(k)): val
                     for k, val in rec["plant"].items()}
            lr, mr = [], []
            try:
                for _ in range(a.pairs):
                    c, _ = episode(env, order, {})
                    x, _ = episode(env, order, {rec["pos"]: plant})
                    if c[LAT] and x[LAT]:
                        lr.append(abs(x[LAT]) / abs(c[LAT]))
                    if c[MEM] and x[MEM]:
                        mr.append(abs(x[MEM]) / abs(c[MEM]))
                rec["lat_ratio"] = float(np.median(lr)) if lr else None
                rec["mem_ratio"] = float(np.median(mr)) if mr else None
                rec["lat_ratios"], rec["mem_ratios"] = lr, mr
                print(f"  pos{rec['pos']:3d} {rec['config']:34s} "
                      f"lat x{rec['lat_ratio']} mem x{rec['mem_ratio']}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                rec["cost_error"] = str(e)
                print(f"  pos{rec['pos']} {rec['config']} COST ERR {e}",
                      flush=True)

    out = a.out or f"apxaudit_out/apxaudit_{a.target}_{a.phase}.json"
    with open(out, "w") as f:
        json.dump({"target": a.target, "phase": a.phase, "plan_len": T,
                   "control": {"cos": float(base[COS]), "lat": float(base[LAT]),
                               "mem": float(base[MEM])},
                   "faces_per_pos": nf, "rows": results}, f, indent=1)
    print(f"\nWROTE {out} ({len(results)} rows, {time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
