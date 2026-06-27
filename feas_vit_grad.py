"""Feasibility gate: can VmappedViT run in GRAD MODE (value_and_grad of the
scalar ViT-MNIST loss -> the loss gradient) under the prevalidate-gate + local
tokenize, doing one TERMINAL measure of a simple (default sequential, dense)
elimination order? Reports FEASIBLE or BLOCKED with the error.

Mirrors grad_diag.build_env / cpu_approx_worker grad wiring exactly:
  grad_target_setup(--measure-grad) -> scalar_loss_fn(ViT) ; argnums = vision weights.
"""
import os, sys, time, traceback
import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "grad"   # grad | jacve
EXAMPLE = "VmappedViT"

# Match the launcher env that makes the ViT measure feasible.
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_LOCAL_TOKENIZE", "1")
os.environ.setdefault("GRAPHAX_STATE_TOKENS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_MAX_MEASURE_TOKENS", "0")
os.environ.setdefault("ALPHAGRAD_MAX_MEASURE_MEM_GIB", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if MODE == "grad":
    os.environ.setdefault("ALPHAGRAD_QUALITY_PROXY", "trajectory")
    os.environ.setdefault("ALPHAGRAD_TRAJ_STEPS", "3")
    os.environ.setdefault("ALPHAGRAD_TRAJ_LR", "0.1")
    os.environ.setdefault("ALPHAGRAD_DEBUG_QUALITY", "1")
    os.environ.setdefault("ALPHAGRAD_REWARD_MODE", "mult")

import jax, jax.numpy as jnp, equinox as eqx
import alphagrad.approx.common.datasets as ds
from alphagrad.approx.env import VertexEliminationEnv, _callback, MAX_RULES_PER_VERTEX
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, grad_target_setup, infer_argnums,
)
from alphagrad.approx.common.eval_samples import generate_eval_samples


class A:  # argparse-like for grad_target_setup
    measure_grad = (MODE == "grad")
    seed_vertices = False


def main():
    t0 = time.time()
    print(f"[feas] platform={jax.default_backend()} devices={jax.devices()}", flush=True)
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    base_fn = get_fn(EXAMPLE)
    xs = get_args(EXAMPLE, ak, dataset=None)            # ViT does not use the mnist dataset arg path
    gen = data_gen(EXAMPLE, dataset=None, dataset_size=16)
    target_fn, xs, argnums = grad_target_setup(A, base_fn, xs, EXAMPLE)
    print(f"[feas] MODE={MODE} measure_grad={A.measure_grad} argnums={argnums} "
          f"n_args={len(xs)}", flush=True)

    print("[feas] make_jaxpr ...", flush=True)
    tj = time.time()
    closed = jax.make_jaxpr(target_fn)(*xs)
    neqns = len(closed.eqns)
    print(f"[feas] jaxpr OK eqns={neqns} ({time.time()-tj:.1f}s)", flush=True)

    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=argnums, num_envs=0, data_gen=gen,
        target_fun=target_fn, cmp_type="latency", mem_type="peak_memory",
        measure_latency=False, num_data_points=1, reps_per_point=1,
        percentile_keep=0.6, slow_exec_cutoff_seconds=0.0,
        flop_gate_threshold=0.0, measure_grad=A.measure_grad,
        latency_timer="perf_counter",
    )
    nvalid = len(env.valid_vertices)
    print(f"[feas] env built; valid_vertices={nvalid}", flush=True)
    ev = generate_eval_samples(env, ek, 1)
    env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)

    # Simple terminal order = default sequential valid_vertices, NO micro-actions
    # (empty specs -> plain dense elimination). This is the cheapest real order.
    order = np.array(env.valid_vertices, dtype=np.int32)
    specs = np.full((order.shape[0], MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    print(f"[feas] terminal measure: order_len={len(order)} (dense, no micro-actions) ...",
          flush=True)
    tm = time.time()
    out = _callback(
        env.config, env.args, env.consts,
        jnp.asarray(order), jnp.asarray(specs), len(order), *ev,
    )
    jax.block_until_ready(out)
    dt = time.time() - tm
    rvec = out[0] if isinstance(out,(tuple,list)) else out
    print(f"[feas] TERMINAL MEASURE OK ({dt:.1f}s) reward_vec={np.asarray(rvec)}",
          flush=True)
    print(f"[feas] RESULT: grad-mode-ViT FEASIBLE (mode={MODE})  total={time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[feas] RESULT: BLOCKED", flush=True)
        print(f"[feas] ERROR_TYPE={type(e).__name__}", flush=True)
        print(f"[feas] ERROR_MSG={e}", flush=True)
        traceback.print_exc()
        sys.exit(3)
