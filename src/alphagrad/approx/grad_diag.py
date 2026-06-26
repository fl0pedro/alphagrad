"""Explain cos≈0 → high-accuracy: for selected front rules, compute the approx
gradient's cosine/frobenius vs exact, BOTH front-style (flatten as-is) AND after
aligning grads to the param layout (transpose-back). Self-contained.

  uv run grad_diag.py --source cmorl --idxs 3,5,8,9
"""
import os, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="cmorl")
ap.add_argument("--idxs", default="3,5,8,9")
A = ap.parse_args()

import alphagrad.approx.common.datasets as ds
ds.NN_VMAP_BATCH = 16
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback
import alphagrad.approx.common.compile_cache as ccmod
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

ARGN = infer_argnums("VmappedNeuralNetwork")


def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    tfn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=16)
    closed = jax.make_jaxpr(tfn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=tfn,
        cmp_type="latency", mem_type="peak_memory", measure_latency=False,
        num_data_points=1, reps_per_point=1, percentile_keep=0.6,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter")
    ev = generate_eval_samples(env, ek, 1)
    env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)
    return env, ev


def capture_gfn(env, ev, order, specs):
    cap = {}; orig = ccmod.cached_compile
    def cc(key, fn):
        out = orig(key, fn)
        if isinstance(key, (bytes, bytearray)) and key.startswith(b"approx:"):
            cap["fn"] = out
        return out
    ccmod.cached_compile = cc
    try:
        _callback(env.config, env.args, env.consts, jnp.asarray(order),
                  jnp.asarray(specs), len(order), *ev)
    finally:
        ccmod.cached_compile = orig
    return cap["fn"]


def init_weights(seed, in_dim=784, hid=256, out=10):
    k = jax.random.PRNGKey(seed); k1, k2 = jax.random.split(k)
    return (jax.random.normal(k1, (hid, in_dim)) / np.sqrt(in_dim), jnp.zeros(hid),
            jax.random.normal(k2, (out, hid)) / np.sqrt(hid), jnp.zeros(out))


def align(g, p):
    if g.shape == p.shape: return g
    if g.ndim == 2 and g.shape == p.shape[::-1]: return g.T
    return g.reshape(p.shape)


env, ev = build_env()
front = json.load(open(os.path.expanduser(f"~/dsnn/morl_fronts/{A.source}_front.json")))["front"]
loss_fn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
x, y = ev[0][0], ev[1][0]
W = init_weights(0)
_, ge = jax.value_and_grad(loss_fn, argnums=ARGN)(x, y, *W)
ge = [np.asarray(g) for g in ge]
fe = np.concatenate([g.ravel() for g in ge])


def cosf(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def frobf(e, a): return float(np.linalg.norm(e - a) / (np.linalg.norm(e) + 1e-30))


print(f"{'idx':>4} {'front_cos':>9} {'raw_cos':>8} {'raw_frob':>9} {'algn_cos':>8} {'algn_frob':>9}  transposed?")
for idx in [int(s) for s in A.idxs.split(",")]:
    p = front[idx]
    order, specs, _ = build_order_specs(p["seq"], env)
    _, ga = capture_gfn(env, ev, order, specs)(x, y, *W)
    ga = [np.asarray(g) for g in ga]
    same = all(a.shape == e.shape for a, e in zip(ga, ge))
    raw = np.concatenate([g.ravel() for g in ga]) if same else None
    al = [np.asarray(align(jnp.asarray(g), jnp.asarray(p_))) for g, p_ in zip(ga, ge)]
    fal = np.concatenate([g.ravel() for g in al])
    transp = [tuple(a.shape) != tuple(b.shape) for a, b in zip(ga, al)]
    rc = cosf(raw, fe) if same else float("nan")
    rf = frobf(fe, raw) if same else float("nan")
    print(f"{idx:>4} {p['obj']['cosine_sim']:>9.4f} {rc:>8.4f} {rf:>9.3f} {cosf(fal,fe):>8.4f} {frobf(fe,fal):>9.3f}  {transp}")
