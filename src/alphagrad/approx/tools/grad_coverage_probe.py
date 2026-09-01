"""GRADIENT COVERAGE -- the validation probe.

Answers, on the live TransformerLM target and THROUGH THE SHIPPED
IMPLEMENTATION (``env._leaf_norms`` / ``env._grad_coverage`` / ``env._callback``),
the five questions the feature has to pass:

  (a) do the four faces of ``run_analysis/landscape/face_forensics.json``
      (k24/f0, k22/f0, k19/f0, k13/f1) reproduce -- SAME zeroed-leaf sets;
  (b) is the identity plan exactly ``min_leaf_ratio == 1.0`` and
      ``frac_zeroed == 0``;
  (c) what does k21/f1 do -- the v74 ``dot_general (32,128)x(128,128)`` face at
      ratio 0.5325 / q 0.9257 that no run ever archived and that was NOT in the
      forensics batch;
  (d) [pinned in tests/grad_coverage_test.py, not here] flag-off bit-identity;
  (e) what does coverage COST per plan measurement.

Phase G additionally drives the HARD GUARD end to end through ``_callback``
and checks that a frozen plan comes back as the exact degenerate sentinel and
lands on the dedicated counter.

Usage (one CPU node, one job)::

    PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \\
      python -m alphagrad.approx.tools.grad_coverage_probe [OUTDIR]
"""
import os
import sys
import json
import time

os.environ.setdefault("ALPHAGRAD_TLM_SEQ", "32")
os.environ.setdefault("ALPHAGRAD_TLM_DMODEL", "128")
os.environ.setdefault("ALPHAGRAD_TLM_VOCAB", "1024")
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "2538")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "32768")
os.environ.setdefault("GRAPHAX_PLANNER_EXACT", "1")
os.environ.setdefault("GRAPHAX_DEMAND_EMIT", "1")
os.environ.setdefault("ALPHAGRAD_NEW_SLOT_JOIN", "0")
os.environ.setdefault("ALPHAGRAD_MAX_EQNS", "512")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
# The probe drives the guard EXPLICITLY per phase; never inherit it.
os.environ["ALPHAGRAD_GRAD_COVERAGE"] = "0"
os.environ["ALPHAGRAD_REJECT_FROZEN_GRADS"] = "0"

import numpy as np                                                # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
import jax.random as jrand                                       # noqa: E402
import alphagrad.approx.env as envmod                            # noqa: E402
from alphagrad.approx.env import (                               # noqa: E402
    VertexEliminationEnv, MAX_RULES_PER_VERTEX, FACE_SLOTS,
    REWARD_INDEX, SENTINEL_COST, COMPUTE_REWARD_INDICES,
    _grad_coverage, _leaf_norms,
    consume_frozen_grad_plan_count, consume_grad_coverage_stats,
)
from alphagrad.approx.common.examples import (                    # noqa: E402
    get_fn, get_args, data_gen, grad_target_setup)
from types import SimpleNamespace as NS                           # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/assmuth/dsnn/run_analysis/landscape"
FORENSICS = os.path.join(OUT, "face_forensics.json")
# The forensics batch, then the face the campaign never found.
FACES = [(24, 0), (22, 0), (19, 0), (13, 1), (21, 1)]

a = NS(measure_grad=True, seed_vertices=True)
key = jrand.PRNGKey(250197)
key, ak = jrand.split(key)
fn0 = get_fn("TransformerLM")
xs = get_args("TransformerLM", ak, dataset="wikitext2")
gen = data_gen("TransformerLM", dataset="wikitext2", dataset_size=-1)
fn0, xs, argnums = grad_target_setup(a, fn0, xs, "TransformerLM")
from graphax import inline_call_primitives, jacve                 # noqa: E402
cj = jax.make_jaxpr(fn0)(*xs)
jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
try:
    from jax.extend.core import ClosedJaxpr
except ImportError:
    from jax._src.core import ClosedJaxpr
cj = ClosedJaxpr(jx, consts) if jx is not cj.jaxpr else cj

env = VertexEliminationEnv.from_jaxpr(
    cj, args=xs, argnums=argnums, num_envs=0, data_gen=gen, target_fun=fn0,
    cmp_type="latency", mem_type="peak_memory", measure_latency=True,
    per_face=True, measure_grad=True, delta_obs=True)
envmod.configure_max_faces(
    envmod.derived_max_faces(cj.jaxpr, argnums, cj.literals, xs))
cfg = env.config
order = np.array(sorted((int(v) for v in env.valid_vertices), reverse=True),
                 dtype=np.int32)
o_list = [int(v) for v in order]
MF = envmod.MAX_FACES
print(f"order len {len(o_list)}  MAX_FACES {MF}", flush=True)


def _wire(skips):
    """The (specs, face_specs, face_skips) wire for a pure-SKIP plan."""
    specs = np.full((len(o_list), MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    face_specs = np.full((len(o_list), MF, FACE_SLOTS, 3), -1, dtype=np.int32)
    face_skips = np.zeros((len(o_list), MF), dtype=np.int32)
    for (k, f) in skips:
        face_skips[k, f] = 1
    return specs, face_specs, face_skips


def build_fn(skips):
    """The MEASUREMENT PATH's own approx gradient fn for these skipped faces.

    Byte-for-byte the construction ``ls_face_forensics.py`` used, so a
    disagreement with the forensics is a disagreement about COVERAGE, never
    about which function was measured.
    """
    specs, face_specs, face_skips = _wire(skips)
    ft = None
    if skips:
        ft = envmod._face_transforms_for_order(
            cfg, list(env.consts), list(env.args), o_list, specs.tolist(),
            face_specs, face_skips)
    return jacve(cfg.target_fun, list(o_list), argnums=cfg.argnums,
                 has_aux=cfg.has_aux, sparse_representation=cfg.sparse,
                 transforms=[], face_transforms=ft)


# =========================================================== PHASE A/B/C
print("\n=== PHASE A/B/C: per-leaf gradient coverage ===", flush=True)
_t0 = time.perf_counter()
exact_norms = _leaf_norms(jax.jit(build_fn([]))(*env.args), cfg.has_aux)
_t_exact = time.perf_counter() - _t0
print(f"exact reference: {len(exact_norms)} leaves, "
      f"compile+exec {_t_exact:.2f}s", flush=True)

results = {"_exact_norms": exact_norms}

# (b) identity: the exact function against itself.
cov_id = _grad_coverage(exact_norms, exact_norms)
results["identity"] = cov_id
print(f"\n[identity] min_leaf_ratio={cov_id['min_leaf_ratio']!r} "
      f"frac_zeroed={cov_id['frac_zeroed']!r} "
      f"counted={cov_id['n_counted']} uncounted={cov_id['n_uncounted']} "
      f"channel={cov_id['channel']!r}", flush=True)
assert cov_id["min_leaf_ratio"] == 1.0, cov_id
assert cov_id["frac_zeroed"] == 0.0, cov_id
print("[identity] PASS: min_leaf_ratio == 1.0 and frac_zeroed == 0 exactly",
      flush=True)

for (k, f) in FACES:
    _t0 = time.perf_counter()
    an = _leaf_norms(jax.jit(build_fn([(k, f)]))(*env.args), cfg.has_aux)
    cov = _grad_coverage(an, exact_norms)
    cov["_wall_s"] = time.perf_counter() - _t0
    results[f"skip_k{k}f{f}"] = cov
    print(f"\n[k{k}/f{f}] min_leaf_ratio={cov['min_leaf_ratio']:.6g} "
          f"frac_zeroed={cov['frac_zeroed']:.6g} "
          f"zeroed={cov['n_zeroed']}/{cov['n_counted']} "
          f"uncounted={cov['n_uncounted']} channel={cov['channel']:.6g}",
          flush=True)
    print(f"          zeroed leaves {cov['zeroed']}", flush=True)

# ---- (a) reproduce the forensics -----------------------------------------
print("\n=== (a) forensics cross-check ===", flush=True)
if os.path.exists(FORENSICS):
    fx = json.load(open(FORENSICS))
    fex = fx["exact"]["norms"]
    ok = True
    for name, rec in fx.items():
        if name == "exact":
            continue
        want = sorted(i for i, (na, ne) in enumerate(zip(rec["norms"], fex))
                      if ne > 0 and na == 0.0)
        got = results.get(name)
        if got is None:
            print(f"  {name}: NOT MEASURED HERE (skipped)")
            continue
        got_z = sorted(got["zeroed"])
        same = (got_z == want)
        ok = ok and same
        print(f"  {name}: forensics zeroed={len(want)} {want}")
        print(f"  {name}: this impl zeroed={len(got_z)} {got_z}  "
              f"-> {'MATCH' if same else 'MISMATCH'}")
    _verdict = "REPRODUCED" if ok else (
        "DISAGREES -- the implementation is wrong, "
        "do not adjust the expectation")
    print(f"\n(a) VERDICT: {_verdict}", flush=True)
    results["_forensics_reproduced"] = bool(ok)
else:
    print(f"  {FORENSICS} not found; (a) not checked", flush=True)

# ---- (c) k21/f1 verdict ---------------------------------------------------
c = results.get("skip_k21f1")
if c is not None:
    print("\n=== (c) k21/f1 (v74 dot_general (32,128)x(128,128), "
          "ratio 0.5325, q 0.9257) ===", flush=True)
    print(f"  min_leaf_ratio={c['min_leaf_ratio']:.6g}  "
          f"frac_zeroed={c['frac_zeroed']:.6g}  "
          f"zeroed {c['n_zeroed']} of {c['n_counted']} counted leaves",
          flush=True)
    print("  PREDICTION (it also freezes leaves): "
          f"{'CONFIRMED' if c['n_zeroed'] > 0 else 'REFUTED'}", flush=True)
    print(f"  per-leaf ratios: "
          f"{['%.4g' % r for r in c['ratios']]}", flush=True)

# ============================================================ PHASE G/E
# End-to-end through `_callback`: the guard, and the cost.
def _cb(skips, cov_on, guard_on):
    specs, face_specs, face_skips = _wire(skips)
    os.environ["ALPHAGRAD_GRAD_COVERAGE"] = "1" if cov_on else "0"
    os.environ["ALPHAGRAD_REJECT_FROZEN_GRADS"] = "1" if guard_on else "0"
    t0 = time.perf_counter()
    tk, ei, rw = envmod._callback(
        cfg, list(env.args), list(env.consts), order, specs,
        face_specs, face_skips, len(o_list))
    return np.asarray(rw), time.perf_counter() - t0


print("\n=== PHASE G: the hard guard, end to end through _callback ===",
      flush=True)
consume_frozen_grad_plan_count()
consume_grad_coverage_stats()
rw, _ = _cb([(24, 0)], cov_on=True, guard_on=True)
n_rej = consume_frozen_grad_plan_count()
_sent = bool(np.all(rw[list(COMPUTE_REWARD_INDICES)] <= SENTINEL_COST * 0.99))
print(f"  k24/f0 with guard ON: reward={rw.tolist()}")
print(f"  is the degenerate sentinel (what ppo._is_degen recognises): {_sent}")
print(f"  rejections counted: {n_rej}")
assert _sent and n_rej == 1, (rw, n_rej)
rw2, _ = _cb([(24, 0)], cov_on=True, guard_on=False)
print(f"  k24/f0 with guard OFF: grad_coverage slot = "
      f"{rw2[REWARD_INDEX['grad_coverage']]:.6g} (negative = -frac_zeroed)")
consume_frozen_grad_plan_count()
consume_grad_coverage_stats()
print("  PHASE G PASS", flush=True)

print("\n=== (e) COST: added wall per plan measurement ===", flush=True)
REPS = int(os.environ.get("ALPHAGRAD_GC_COST_REPS", "3"))
off, on = [], []
for _ in range(REPS):                       # warm both paths first
    _cb([], cov_on=False, guard_on=False)
    _cb([], cov_on=True, guard_on=False)
consume_grad_coverage_stats()
for _ in range(REPS):
    off.append(_cb([], cov_on=False, guard_on=False)[1])
    on.append(_cb([], cov_on=True, guard_on=False)[1])
gcs = consume_grad_coverage_stats()
m_off, m_on = float(np.median(off)), float(np.median(on))
print(f"  identity plan, _callback wall: coverage OFF {m_off * 1e3:.1f} ms, "
      f"ON {m_on * 1e3:.1f} ms  -> +{100.0 * (m_on / m_off - 1.0):.2f}%")
print(f"  self-reported grad_cov/wall_frac = "
      f"{100.0 * gcs['wall_frac']:.2f}%  "
      f"(wall {gcs['wall_s'] * 1e3:.1f} ms over {gcs['count']} plans)")
print(f"  paired samples off={['%.3f' % x for x in off]} "
      f"on={['%.3f' % x for x in on]}")
results["_cost"] = {"off_s": off, "on_s": on, "median_off_s": m_off,
                    "median_on_s": m_on,
                    "pct_added": 100.0 * (m_on / m_off - 1.0),
                    "self_reported_wall_frac": gcs["wall_frac"],
                    "exact_ref_compile_exec_s": _t_exact}

_p = os.path.join(OUT, "grad_coverage_probe.json")
with open(_p, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nwrote {_p}", flush=True)
