"""Does EVERY approximation type (and combination) DO SOMETHING at INTERMEDIATE
eliminations?

Drives the REAL chain -- StepAction.face_rows/face_skip -> EnvState histories ->
_callback -> _face_transforms_for_order -> make_live_masked_hook -> jacve
face_transforms -> the terminal measurement -- exactly like
tests/face_actions_env_test.py, but with the rule planted at an INTERMEDIATE
position of the plan and the reward read at EVERY prefix length.

The COMPRESS-bug fingerprint is a cos series that moves off 1.0 at the step the
rule is taken and SNAPS BACK to 1.000000 at every later step.  Any rule type
that shows it is being silently discarded downstream of its own decision.

Columns produced per (config, position):
  (a) decode-survives  -- _face_transforms_for_order at EVERY prefix length
                          k = p+1 .. T still yields a hook for vertex v
  (b) applied          -- _PER_FACE_STATS applied_<kind> from the ARMED
                          measurement scope (env._do_compile_approx)
  (c) cos != 1         -- terminal reward slot 6 vs exact AD at the same order
  (d) latency delta    -- terminal reward slot 2, paired against a control
  (e) memory delta     -- terminal reward slot 5, paired against a control
"""
import argparse
import json
import os
import sys
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
from graphax.sparse.micro_actions import QUANT_DTYPES  # noqa: E402

COS = REWARD_INDEX["cosine_sim"]
LAT = REWARD_INDEX["latency_ns"]
MEM = REWARD_INDEX["peak_memory"]
FROB = REWARD_INDEX["frob_residual"]

SLOT_NAME = ("pre", "post", "new")


def _dtype_idx(name):
    for i, d in enumerate(QUANT_DTYPES):
        if str(d) == name:
            return i
    return None


DT_F32 = _dtype_idx("float32")
DT_BF16 = _dtype_idx("bfloat16")
DT_F16 = _dtype_idx("float16")
DT_I8 = _dtype_idx("int8")
DT_NARROW = next((_dtype_idx(n) for n in
                  ("float8_e4m3fn", "float8_e4m3", "float8_e5m2")
                  if _dtype_idx(n) is not None), None)


# --------------------------------------------------------------------- target
def build_env(name, measure_latency, seed=0):
    from graphax import inline_call_primitives
    try:
        from jax.extend.core import ClosedJaxpr
    except ImportError:
        from jax._src.core import ClosedJaxpr

    key = jax.random.PRNGKey(seed)
    fn = get_fn(name)
    xs = get_args(name, key)
    argnums = tuple(range(len(xs)))
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    closed = cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=argnums, num_envs=0, target_fun=fn,
        cmp_type="latency" if measure_latency else "flops",
        mem_type="peak_memory", measure_latency=measure_latency,
        per_face=True, terminal_rewards_only=False,
        num_data_points=3, reps_per_point=3, latency_inner_reps=50,
    )
    return env, closed, xs, argnums, fn


# --------------------------------------------------------------- legal rows
def _pad(row):
    r = [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]
    r[0] = list(row)
    return r


def find_rows(jaxpr, v):
    """A DECODING row of each kind for vertex ``v`` (None if the geometry
    admits none -- that is DEGENERATE, not a drop)."""
    out = {}
    eqn = jaxpr.eqns[v - 1]
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        return out
    out_len = len(eqn.outvars[0].aval.shape)
    prim = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
    max_p = max((len(p) for p in prim), default=0)

    for bi1 in range(max(out_len, 1)):
        for bi2 in range(max(max_p, 1)):
            for f in (-1, 2, 4, 8):
                row = [bi1, bi2, f]
                if decode_vertex_rule_specs(jaxpr, v, _pad(row)):
                    out["diag"] = row
                    break
            if "diag" in out:
                break
        if "diag" in out:
            break

    for ax in range(9):
        row = [COMPRESS_SENTINEL, ax, 0]
        if decode_vertex_rule_specs(jaxpr, v, _pad(row)):
            out["compress"] = row
            break

    for tag, di in (("quant_bf16", DT_BF16), ("quant_f32", DT_F32),
                    ("quant_f16", DT_F16), ("quant_i8", DT_I8),
                    ("quant_narrow", DT_NARROW)):
        if di is None:
            continue
        row = [QUANT_SENTINEL, di, 0]
        if decode_vertex_rule_specs(jaxpr, v, _pad(row)):
            out[tag] = row
    return out


# ----------------------------------------------------------------- the matrix
def make_configs(rows):
    """(label, plant) where plant is {slot: row} plus an optional 'skip'."""
    cfgs = [("NONE(control)", {})]
    if rows:
        cfgs.append(("SKIP", {"skip": True}))
    for kind, tag in (("diag", "DIAG"), ("compress", "COMPRESS"),
                      ("quant_bf16", "QUANT_bf16")):
        if kind in rows:
            for s in range(FACE_SLOTS):
                cfgs.append((f"{tag}@{SLOT_NAME[s]}", {s: rows[kind]}))
    # degenerate / out-of-head-reach QUANT controls, one slot only
    for kind, tag in (("quant_f32", "QUANT_f32(degen-control)"),
                      ("quant_f16", "QUANT_f16"),
                      ("quant_i8", "QUANT_int8"),
                      ("quant_narrow", "QUANT_narrowfp")):
        if kind in rows:
            cfgs.append((f"{tag}@new", {2: rows[kind]}))
    # combinations WITHIN one face (multi-slot)
    def has(*ks):
        return all(k in rows for k in ks)
    if has("diag", "compress"):
        cfgs.append(("DIAG@pre+COMPRESS@post",
                     {0: rows["diag"], 1: rows["compress"]}))
    if has("diag", "quant_bf16"):
        cfgs.append(("DIAG@pre+QUANT@post",
                     {0: rows["diag"], 1: rows["quant_bf16"]}))
    if has("compress", "quant_bf16"):
        cfgs.append(("COMPRESS@pre+QUANT@post",
                     {0: rows["compress"], 1: rows["quant_bf16"]}))
    if has("diag", "compress", "quant_bf16"):
        cfgs.append(("DIAG@pre+COMPRESS@post+QUANT@new",
                     {0: rows["diag"], 1: rows["compress"],
                      2: rows["quant_bf16"]}))
    if has("compress"):
        cfgs.append(("COMPRESS@pre+COMPRESS@post",
                     {0: rows["compress"], 1: rows["compress"]}))
    if has("diag"):
        cfgs.append(("SKIP+DIAG@pre", {"skip": True, 0: rows["diag"]}))
    return cfgs


# --------------------------------------------------------------- episode run
def run_episode(env, order, plants, want_series=True):
    """``plants`` = {position_index: {slot|'skip': row|True}}.  Returns the
    per-step reward matrix (T, NUM_REWARDS) and the per-face stats."""
    consume_per_face_stats()
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    rewards = []
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
        out = env.step(state, StepAction(
            jnp.asarray(int(v), jnp.int32), no_rules,
            jnp.asarray(fr), jnp.asarray(fs)))
        state = out.state
        rewards.append(np.asarray(state.reward, dtype=np.float64).copy())
    return np.stack(rewards), consume_per_face_stats()


# ------------------------------------------------- (a) decode survival check
def decode_survives(env, closed, xs, order, plants):
    """Rebuild the graphax face_transforms dict at EVERY prefix length and
    check the planted vertex still carries a non-None hook."""
    cfg = env.config
    T = len(order)
    specs = [[[-1, -1, 0]] * MAX_RULES_PER_VERTEX for _ in range(T)]
    frs, fss = [], []
    for k in range(T):
        fr = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
        fs = np.zeros((MAX_FACES,), np.int32)
        p = plants.get(k)
        if p:
            for slot, row in p.items():
                if slot == "skip":
                    fs[:] = 1
                else:
                    fr[:, int(slot), :] = np.asarray(row, np.int32)
        frs.append(fr.tolist())
        fss.append(fs.tolist())
    firsts = sorted(plants)
    if not firsts:
        return True, ""
    bad = []
    for k in range(firsts[0] + 1, T + 1):
        ft = _face_transforms_for_order(
            cfg, list(closed.literals), list(xs), [int(x) for x in order[:k]],
            specs[:k], frs[:k], fss[:k])
        for pos in firsts:
            if pos >= k:
                continue
            v = int(order[pos])
            d = ft.get(v)
            if not d:
                bad.append((k, pos, v, "vertex absent"))
                continue
            live = 0
            for key, val in d.items():
                if val is None:
                    continue
                if isinstance(val, tuple):
                    slots = val[0] if (val and isinstance(val[0], tuple)) else val
                    live += sum(1 for s in slots if s is not None)
                else:
                    live += 1   # SKIP_FACE sentinel
            if live == 0:
                bad.append((k, pos, v, "all slots None"))
    if bad:
        return False, f"{len(bad)} losses, first={bad[0]}"
    return True, ""


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="NeuralNetwork")
    ap.add_argument("--phase", default="correctness",
                    choices=["correctness", "cost"])
    ap.add_argument("--fracs", default="0.25,0.5,0.75")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print(f"[cfg] target={a.target} phase={a.phase} "
          f"platform={jax.default_backend()} devices={jax.devices()}",
          flush=True)
    print(f"[cfg] QUANT_DTYPES f32={DT_F32} bf16={DT_BF16} f16={DT_F16} "
          f"i8={DT_I8} narrow={DT_NARROW} n={len(QUANT_DTYPES)}", flush=True)

    t0 = time.time()
    env, closed, xs, argnums, fn = build_env(
        a.target, measure_latency=(a.phase == "cost"))
    order = [int(x) for x in np.asarray(env.valid_vertices)][::-1]
    T = len(order)
    jaxpr = closed.jaxpr
    print(f"[cfg] eqns={len(jaxpr.eqns)} plan_len={T} "
          f"({time.time()-t0:.1f}s)", flush=True)

    fracs = [float(x) for x in a.fracs.split(",")]
    positions = sorted({max(0, min(T - 2, int(f * T))) for f in fracs})
    print(f"[cfg] INTERMEDIATE positions {positions} of 0..{T-1} "
          f"(last={T-1} excluded from the intermediate claim)", flush=True)

    results = []

    # --- baseline / control -------------------------------------------------
    base_r, base_stats = run_episode(env, order, {})
    base_cos = base_r[-1, COS]
    print(f"[control] terminal cos={base_cos:.8f} "
          f"lat={base_r[-1, LAT]:.4g} mem={base_r[-1, MEM]:.4g} "
          f"stats={base_stats}", flush=True)

    for pos in positions:
        v = order[pos]
        rows = find_rows(jaxpr, v)
        print(f"\n{'='*78}\nPOSITION {pos}/{T-1}  vertex={v}  "
              f"decodable rows: {sorted(rows)}\n{'='*78}", flush=True)
        cfgs = make_configs(rows)
        for label, plant in cfgs:
            if not plant:
                continue
            rec = {"target": a.target, "pos": pos, "vertex": v,
                   "config": label, "plant": {str(k): val for k, val
                                              in plant.items()}}
            try:
                ok_a, why_a = decode_survives(
                    env, closed, xs, order, {pos: plant})
                rec["decode_survives"] = bool(ok_a)
                rec["decode_note"] = why_a

                r, stats = run_episode(env, order, {pos: plant})
                series = r[:, COS]
                rec["cos_at_decision"] = float(series[pos])
                rec["cos_terminal"] = float(series[-1])
                rec["frob_terminal"] = float(r[-1, FROB])
                rec["applied"] = int(stats.get("applied", 0))
                rec["skipped"] = int(stats.get("skipped", 0))
                rec["skipped_raised"] = int(stats.get("skipped_raised", 0))
                for k_ in ("diag", "compress", "quant"):
                    rec[f"applied_{k_}"] = int(stats.get(f"applied_{k_}", 0))
                    rec[f"skipped_{k_}"] = int(stats.get(f"skipped_{k_}", 0))
                # The COMPRESS fingerprint: bit off 1.0 then snapped back.
                bit = abs(series[pos] - 1.0) > 1e-7
                back = abs(series[-1] - 1.0) <= 1e-7
                rec["silent_drop_fingerprint"] = bool(bit and back)
                tail = [float(x) for x in series[pos:]]
                rec["cos_tail"] = tail[:8]
                print(f"  {label:38s} decode_ok={int(ok_a)} "
                      f"cos@dec={series[pos]:.6f} cos@term={series[-1]:.6f} "
                      f"applied={rec['applied']}/"
                      f"skipped={rec['skipped']}"
                      + ("  <<< SILENT DROP" if rec["silent_drop_fingerprint"]
                         else ""), flush=True)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  {label:38s} ERROR {type(e).__name__}: "
                      f"{str(e)[:200]}", flush=True)
                traceback.print_exc()
            results.append(rec)

    # --- cost phase: paired latency / memory --------------------------------
    if a.phase == "cost":
        print(f"\n{'='*78}\nPAIRED COST (candidate vs control, "
              f"back-to-back x{a.pairs})\n{'='*78}", flush=True)
        live = [r for r in results
                if r.get("cos_terminal") is not None
                and abs(r["cos_terminal"] - 1.0) > 1e-7]
        seen = set()
        for rec in live:
            key = (rec["pos"], rec["config"])
            if key in seen:
                continue
            seen.add(key)
            plant = {}
            for k_, val in rec["plant"].items():
                plant[k_ if k_ == "skip" else int(k_)] = val
            lat_r, mem_r = [], []
            try:
                for _ in range(a.pairs):
                    c, _ = run_episode(env, order, {})
                    x, _ = run_episode(env, order, {rec["pos"]: plant})
                    if c[-1, LAT] and x[-1, LAT]:
                        lat_r.append(abs(x[-1, LAT]) / abs(c[-1, LAT]))
                    if c[-1, MEM] and x[-1, MEM]:
                        mem_r.append(abs(x[-1, MEM]) / abs(c[-1, MEM]))
                rec["lat_ratio_median"] = (float(np.median(lat_r))
                                           if lat_r else None)
                rec["mem_ratio_median"] = (float(np.median(mem_r))
                                           if mem_r else None)
                rec["lat_ratios"] = lat_r
                rec["mem_ratios"] = mem_r
                print(f"  pos{rec['pos']} {rec['config']:38s} "
                      f"lat x{rec['lat_ratio_median']} "
                      f"mem x{rec['mem_ratio_median']}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec["cost_error"] = f"{type(e).__name__}: {e}"
                print(f"  pos{rec['pos']} {rec['config']} COST ERROR {e}",
                      flush=True)

    out = a.out or f"apxaudit_{a.target}_{a.phase}.json"
    with open(out, "w") as f:
        json.dump({"target": a.target, "phase": a.phase, "plan_len": T,
                   "positions": positions, "control_cos": float(base_cos),
                   "quant_dtypes": list(QUANT_DTYPES), "rows": results},
                  f, indent=1)
    print(f"\nWROTE {out}  ({len(results)} rows, {time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
