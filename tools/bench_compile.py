"""Isolated AOT compile benchmark: does LLVM parallel compilation pay?

THE QUESTION. Measurement compile is 70-95% of a measure actor's host time
on TLM (20-90 s/plan, read from cb.xla_compile in three wandb runs). XLA's
parallel LLVM-module compilation is OFF in every run to date (it is gated
behind a persistent-cache setting nobody enables). If turning it on -- with
a full node's CPUs available -- collapses that number, no dispatch queue is
needed. If it does nothing, the time is not in LLVM and the queue discussion
resumes with that fact.

DESIGN, deliberately minimal:
  * near-pure JAX: build the fn once, then per plan exactly three calls --
        lowered   = jax.jit(jacve(fn, order, ...)).lower(*xs)
        exe_a     = lowered.compile(BASE_OPTS)
        exe_b     = lowered.compile(BASE_OPTS + LLVM parallelism)
    Everything else (env, Ray, tokenizer, oracle) is absent by construction.
  * PAIRED per plan: both arms compile the SAME lowered object, so graph
    structure, process warmth and node state cancel. Arm order alternates by
    plan parity so warm-up cannot systematically favour one arm.
  * lower/compile timed separately: .lower() is graphax's Python elimination
    + tracing + MLIR emission (single-threaded, no flag can touch it);
    .compile() is the XLA backend (where the flag acts). If lower dominates,
    THAT is the finding -- cores cannot help and the fix is elsewhere.
  * BASE_OPTS mirrors the production measure path (autotune 0, no Triton
    GEMM -- env._measure_compiler_options), so times are representative of
    what training actually pays, not of a default-flags strawman.
  * plans = random elimination orders; half also carry random QUANT-bf16
    approximations on ~8 vertices via jacve's per-vertex `transforms`
    (the same construction apxaudit_matrix2.py uses). QUANT is used because
    it is legal on effectively every vertex, so "random approximation"
    needs no oracle.
  * no persistent cache: JAX_COMPILATION_CACHE_DIR is left unset by the
    sbatch, so every compile is a real compile.

Output: one line per (plan, arm) --
  plan=K approx=0|1 lower=S hlo_bytes=N compile[default]=S compile[parallel]=S
then medians. Read the medians; per-plan lines exist for spread.
"""

import argparse
import os
import time

import jax
import numpy as np

from alphagrad.approx.common.examples import get_args, get_fn
from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX, QUANT_SENTINEL, decode_vertex_rule_specs,
)
from alphagrad.approx.common.masks import make_live_masked_hook
from graphax import jacve
from graphax.sparse.micro_actions import QUANT_DTYPES

BASE_OPTS = {
    "xla_gpu_autotune_level": 0,
    "xla_gpu_enable_triton_gemm": False,
}
PAR_OPTS = dict(BASE_OPTS)
PAR_OPTS["xla_gpu_enable_llvm_module_compilation_parallelism"] = True

_BF16 = next(i for i, d in enumerate(QUANT_DTYPES) if "bfloat16" in str(d))


def _pad(row):
    # decode_vertex_rule_specs walks ALL MAX_RULES_PER_VERTEX rows; -1 rows
    # are the inert filler (same construction as apxaudit_matrix2._pad).
    out = [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]
    out[0] = list(row)
    return out


def random_transforms(jaxpr, order, rng, n_approx):
    """QUANT-bf16 on up to n_approx random vertices of the order.

    Skips vertices where the rule does not decode instead of consulting an
    oracle -- the point is a REPRESENTATIVE approx compile, not a legal plan
    for training.
    """
    out = []
    for v in rng.permutation(order):
        if len(out) >= n_approx:
            break
        rules = decode_vertex_rule_specs(
            jaxpr, int(v), _pad([QUANT_SENTINEL, _BF16, 0]))
        if rules:
            out.append((int(v), (make_live_masked_hook(tuple(rules)),)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="TransformerLM")
    ap.add_argument("--plans", type=int, default=12)
    ap.add_argument("--approx-frac", type=float, default=0.5)
    ap.add_argument("--n-approx", type=int, default=8)
    ap.add_argument("--seed", type=int, default=250197)
    a = ap.parse_args()

    fn = get_fn(a.example)
    xs = get_args(a.example, jax.random.PRNGKey(a.seed))
    argnums = tuple(range(len(xs)))   # all-args, same as apxaudit_matrix2
    jaxpr = jax.make_jaxpr(fn)(*xs)
    n_v = len(jaxpr.eqns)
    rng = np.random.default_rng(a.seed)

    dev = jax.devices()[0]
    print(f"example={a.example} vertices={n_v} device={dev.device_kind} "
          f"cpus={os.cpu_count()} affinity={len(os.sched_getaffinity(0))}",
          flush=True)

    rows = []
    for k in range(a.plans):
        order = [int(v) for v in rng.permutation(np.arange(1, n_v + 1))]
        use_approx = (k % 2 == 1) if a.approx_frac >= 0.5 else False
        tf = (random_transforms(jaxpr, order, rng, a.n_approx)
              if use_approx else [])

        t0 = time.perf_counter()
        lowered = jax.jit(
            jacve(fn, order, argnums=argnums,
                  transforms=tf if tf else None)
        ).lower(*xs)
        t_lower = time.perf_counter() - t0
        hlo_bytes = len(lowered.as_text())

        # Alternate arm order by parity so warmth cannot favour one arm.
        arms = (("default", BASE_OPTS), ("parallel", PAR_OPTS))
        if k % 2 == 1:
            arms = arms[::-1]
        t = {}
        for name, opts in arms:
            t1 = time.perf_counter()
            lowered.compile(compiler_options=opts)
            t[name] = time.perf_counter() - t1

        rows.append((k, int(bool(tf)), t_lower, hlo_bytes,
                     t["default"], t["parallel"]))
        print(f"plan={k:2d} approx={int(bool(tf))} lower={t_lower:7.2f}s "
              f"hlo_bytes={hlo_bytes} compile[default]={t['default']:7.2f}s "
              f"compile[parallel]={t['parallel']:7.2f}s", flush=True)

    r = np.array([(x[2], x[4], x[5]) for x in rows])
    med = np.median(r, axis=0)
    print("\n=== MEDIANS over "
          f"{len(rows)} plans (example={a.example}) ===", flush=True)
    print(f"lower             : {med[0]:7.2f}s")
    print(f"compile[default]  : {med[1]:7.2f}s")
    print(f"compile[parallel] : {med[2]:7.2f}s")
    print(f"parallel/default  : {med[2] / max(med[1], 1e-9):7.3f}")
    print(f"lower share of (lower+default) : "
          f"{med[0] / max(med[0] + med[1], 1e-9):6.1%}")


if __name__ == "__main__":
    main()
