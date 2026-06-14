"""Train the VmappedNeuralNetwork on MNIST with each Pareto-front 'learning
rule' (elimination order + micro-actions = an approximate gradient), measure
test accuracy + per-step latency over N seeds, and compare to reverse-mode.

Run one (source, gpu-shard) per process:
  CUDA_VISIBLE_DEVICES=k uv run train_rules_experiment.py \
      --source cmorl --front ~/dsnn/morl_fronts/cmorl_front.json \
      --shard k --nshards 4 --out ~/dsnn/train_exp/cmorl_shard{k}.json
"""
import os, re, sys, json, time, argparse
import numpy as np

argp = argparse.ArgumentParser()
argp.add_argument("--source", required=True)
argp.add_argument("--front", required=True)
argp.add_argument("--out", required=True)
argp.add_argument("--shard", type=int, default=0)
argp.add_argument("--nshards", type=int, default=1)
argp.add_argument("--seeds", type=int, default=20)
argp.add_argument("--batch", type=int, default=16)
argp.add_argument("--lr", type=float, default=1e-3)
argp.add_argument("--max-steps", type=int, default=5000)
argp.add_argument("--eval-every", type=int, default=300)
argp.add_argument("--patience", type=int, default=5)
argp.add_argument("--lat-reps", type=int, default=30)
argp.add_argument("--smoke", action="store_true")
a = argp.parse_args()
if a.smoke:
    a.seeds, a.max_steps, a.eval_every, a.patience = 2, 3000, 500, 4

# Training batch == env vmap batch (so the captured fn matches MNIST batches).
import alphagrad.approx.common.datasets as ds
ds.NN_VMAP_BATCH = a.batch

import jax, jax.numpy as jnp, equinox as eqx, optax
import alphagrad.approx.env as envmod
from alphagrad.approx.env import VertexEliminationEnv, _callback, MAX_RULES_PER_VERTEX, REWARD_INDEX
import alphagrad.approx.common.compile_cache as ccmod
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.common.datasets import load_dataset
from alphagrad.approx.verify_pareto_solution import build_order_specs

ARGN = infer_argnums("VmappedNeuralNetwork")  # (2,3,4,5) = W1,b1,W2,b2


def build_env():
    key = jax.random.PRNGKey(0)
    args_key, eval_key = jax.random.split(key)
    # Scalar-loss target (mean MSE) — matches the front's grad-mode jaxpr
    # (14 vertices); the vector-output model jaxpr would be 13 and mis-index.
    tfn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
    xs = get_args("VmappedNeuralNetwork", args_key, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=a.batch)
    closed = jax.make_jaxpr(tfn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=tfn,
        cmp_type="latency", mem_type="peak_memory",
        measure_latency=False, num_data_points=1, reps_per_point=1,
        percentile_keep=0.60, slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter",
    )
    eval_samples = generate_eval_samples(env, eval_key, 1)
    env = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)
    return env, eval_samples


def capture_gfn(env, eval_samples, order, specs):
    """Run _callback once, capturing the compiled value_and_grad executable."""
    cap = {}
    orig = ccmod.cached_compile
    def cc(key, fn):
        out = orig(key, fn)
        if isinstance(key, (bytes, bytearray)) and key.startswith(b"approx:"):
            cap["fn"] = out
        return out
    ccmod.cached_compile = cc
    try:
        _callback(env.config, env.args, env.consts,
                  jnp.asarray(order), jnp.asarray(specs), len(order),
                  *eval_samples)
    finally:
        ccmod.cached_compile = orig
    return cap.get("fn")


def init_weights(seed, in_dim=784, hid=256, out=10):
    k = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(k)
    s1 = 1.0 / np.sqrt(in_dim); s2 = 1.0 / np.sqrt(hid)
    W1 = jax.random.normal(k1, (hid, in_dim)) * s1
    b1 = jnp.zeros(hid)
    W2 = jax.random.normal(k2, (out, hid)) * s2
    b2 = jnp.zeros(out)
    return (W1, b1, W2, b2)


@jax.jit
def predict(x, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    return jnp.tanh(a1 @ W2.T + b2)


def accuracy(W, x, y):
    # batch the test set through predict
    preds = []
    for i in range(0, x.shape[0], 1000):
        preds.append(jnp.argmax(predict(x[i:i+1000], *W), -1))
    pred = jnp.concatenate(preds)
    return float(jnp.mean(pred == jnp.argmax(y, -1)))


def _align(g, p):
    """Map a returned grad to its param's layout. graphax can return a weight
    grad in the transposed (∂L/∂Wᵀ) layout for some elimination orders; the
    transpose recovers the true ∂L/∂W. Returns None if unalignable."""
    if g.shape == p.shape:
        return g
    if g.ndim == 2 and g.shape == p.shape[::-1]:
        return g.T
    if g.size == p.size:
        return g.reshape(p.shape)
    return None


def _align_grads(grads, W):
    out = []
    for g, p in zip(grads, W):
        ag = _align(g, p)
        if ag is None:
            raise ValueError(f"grad shape {g.shape} not alignable to param {p.shape}")
        out.append(ag)
    return tuple(out)


def train_one(gfn, seed, xtr, ytr, xte, yte):
    """Train with the (approx) grad fn until train-loss early-stops; return
    final test accuracy + steps taken."""
    W = init_weights(seed)
    opt = optax.adam(a.lr)
    ostate = opt.init(W)
    rng = np.random.default_rng(seed)
    n = xtr.shape[0]
    best, bad, last_eval = 1e9, 0, 0.0
    step = 0
    while step < a.max_steps:
        idx = rng.integers(0, n, size=a.batch)
        xb, yb = xtr[idx], ytr[idx]
        val, grads = gfn(xb, yb, *W)
        # grads is the tuple for argnums (W1,b1,W2,b2); align to param layout
        grads = _align_grads(grads, W)
        updates, ostate = opt.update(grads, ostate, W)
        W = optax.apply_updates(W, updates)
        step += 1
        if step % a.eval_every == 0:
            cur = float(val)
            if cur < best - 1e-4:
                best, bad = cur, 0
            else:
                bad += 1
                if bad >= a.patience:
                    break
    return accuracy(W, xte, yte), step


def measure_latency(gfn, xb, yb, W):
    for _ in range(3):
        jax.block_until_ready(gfn(xb, yb, *W))
    ts = []
    for _ in range(a.lat_reps):
        t0 = time.perf_counter()
        out = gfn(xb, yb, *W)
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) * 1e9)
    ts = np.array(ts)
    lo, hi = np.percentile(ts, [10, 90])
    w = ts[(ts >= lo) & (ts <= hi)]
    return float(np.mean(w)), float(np.std(ts))


def run_rule(label, gfn, xtr, ytr, xte, yte, seeds):
    accs, steps = [], []
    for s in range(seeds):
        acc, st = train_one(gfn, 1000 + s, xtr, ytr, xte, yte)
        accs.append(acc); steps.append(st)
    W0 = init_weights(0)
    lat_mean, lat_std = measure_latency(gfn, xtr[:a.batch], ytr[:a.batch], W0)
    return {
        "label": label,
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "accs": [float(x) for x in accs],
        "steps_mean": float(np.mean(steps)),
        "lat_ns_mean": lat_mean, "lat_ns_std": lat_std,
    }


def main():
    xtr, ytr = load_dataset("mnist", None, "train")
    xte, yte = load_dataset("mnist", None, "test")
    xtr, ytr, xte, yte = (jnp.asarray(z) for z in (xtr, ytr, xte, yte))
    print(f"[data] train {xtr.shape} test {xte.shape}", flush=True)

    env, eval_samples = build_env()
    d = json.load(open(os.path.expanduser(a.front)))
    front = d["front"]
    results = []

    # Reverse-mode baseline (exact) only on shard 0.
    if a.shard == 0:
        loss_fn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
        rev = jax.jit(jax.value_and_grad(loss_fn, argnums=ARGN))
        print("[reverse-mode] training baseline...", flush=True)
        results.append(run_rule("reverse_mode", rev, xtr, ytr, xte, yte, a.seeds))
        print("  reverse:", results[-1]["acc_mean"], results[-1]["lat_ns_mean"], flush=True)

    mine = [(i, p) for i, p in enumerate(front) if i % a.nshards == a.shard]
    for i, p in mine:
        seq = p["seq"]
        order, specs, nr = build_order_specs(seq, env)
        t0 = time.time()
        try:
            gfn = capture_gfn(env, eval_samples, order, specs)
            if gfn is None:
                raise RuntimeError("no compiled fn captured")
            r = run_rule(f"{a.source}#{i}", gfn, xtr, ytr, xte, yte, a.seeds)
        except Exception as exc:
            r = {"label": f"{a.source}#{i}", "error": f"{type(exc).__name__}: {exc}"[:200]}
        r["idx"] = i
        r["front_obj"] = p["obj"]
        r["wall_s"] = round(time.time() - t0, 1)
        results.append(r)
        print(f"  [{i}] {r.get('acc_mean','ERR')} acc  lat={r.get('lat_ns_mean',0)/1e3:.0f}µs  ({r['wall_s']}s)", flush=True)
        os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True)
        json.dump({"source": a.source, "shard": a.shard, "results": results},
                  open(os.path.expanduser(a.out), "w"), indent=2)
    print(f"[done] {len(results)} results -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
