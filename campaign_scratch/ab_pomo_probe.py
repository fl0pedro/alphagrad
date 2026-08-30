"""ab_ SCRATCH probe (no tracked source modified): splits POMO's per-measurement
`compile_s` into (a) target rebuild, (b) jacve-rev REFERENCE compile repeated on
the SAME plan (task #123: is it recompiled every call?), (c) the full worker call
the trainer actually pays, for the reference and for fresh candidate plans.
"""
import json
import random
import time

import jax

from alphagrad.elimrl.measure_worker import (
    _resolve_target, _build_jac_callable, _compile)

TARGET = {"builder": "alphagrad.elimrl.baselines:tlm_target",
          "kwargs": {"seq": 64, "dmodel": 128, "vocab": 256}}
OUT = {}
print("backend", jax.default_backend(), flush=True)


def t(f):
    s = time.perf_counter()
    r = f()
    return time.perf_counter() - s, r


# (a) target rebuild alone (wikitext load + arg construction + tracing setup)
res = []
for _ in range(6):
    dt, _ = t(lambda: _resolve_target(TARGET))
    res.append(dt)
OUT["resolve_target_s"] = res
print("resolve_target_s", res, flush=True)

fn, args, argnums = _resolve_target(TARGET)

# (b) the SAME reference plan compiled repeatedly, target already resolved
ref = []
for _ in range(5):
    dt, _ = t(lambda: _compile(
        jax.jit(_build_jac_callable(
            {"method": "jacve", "order": "rev"}, fn, tuple(argnums))), args))
    ref.append(dt)
OUT["jacve_rev_recompile_s"] = ref
print("jacve_rev_recompile_s", ref, flush=True)

# (c) exactly what the worker does per REFERENCE call
full = []
for _ in range(3):
    s = time.perf_counter()
    fn2, args2, an2 = _resolve_target(TARGET)
    f = _build_jac_callable({"method": "jacve", "order": "rev"}, fn2, tuple(an2))
    _compile(jax.jit(f), args2)
    full.append(time.perf_counter() - s)
OUT["worker_full_ref_call_s"] = full
print("worker_full_ref_call_s", full, flush=True)

# (d) exactly what the worker does per CANDIDATE call (fresh random orders)
from alphagrad.elimrl.env import ElimEnv  # noqa: E402

senv = ElimEnv(fn, args, argnums, vertex_only=True, symbolic=True)
OUT["eliminable"] = len(senv.jacve_vertices)
print("eliminable", OUT["eliminable"], flush=True)
rng = random.Random(0)
cand, cand_build = [], []
for _ in range(4):
    senv.reset()
    order = []
    while not senv.done:
        st = senv.state()
        if not st.legal_vertices:
            break
        j = rng.choice(list(st.legal_vertices))
        senv.apply(("V", int(j)))
        order.append(int(j))
    s = time.perf_counter()
    fn2, args2, an2 = _resolve_target(TARGET)
    f = _build_jac_callable(
        {"method": "elim_plan", "plan": [["V", j] for j in order],
         "vertex_only": True}, fn2, tuple(an2))
    t0 = time.perf_counter()
    _compile(jax.jit(f), args2)
    cand.append(time.perf_counter() - s)
    cand_build.append(time.perf_counter() - t0)
OUT["worker_full_candidate_call_s"] = cand
OUT["candidate_lower_compile_only_s"] = cand_build
print("worker_full_candidate_call_s", cand, flush=True)
print("candidate_lower_compile_only_s", cand_build, flush=True)

print("AB_PROBE_JSON " + json.dumps(OUT), flush=True)
