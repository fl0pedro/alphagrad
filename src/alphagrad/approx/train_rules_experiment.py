"""Train the VmappedNeuralNetwork on MNIST with each Pareto-front 'learning
rule' (elimination order + micro-actions = an approximate gradient), measure
test accuracy + per-step latency over N seeds, and compare to reverse-mode.

Run one (source, gpu-shard) per process:
  CUDA_VISIBLE_DEVICES=k uv run train_rules_experiment.py \
      --source cmorl --front ~/dsnn/morl_fronts/cmorl_front.json \
      --shard k --nshards 4 --out ~/dsnn/train_exp/cmorl_shard{k}.json

INSTRUMENTED PARALLEL MODE: pass --rule-index I to run EXACTLY one rule per
process (I>=0 => front[I]; I==-2 => reverse_mode baseline). Each process then
writes its own out JSON. This lets a single sbatch fan out 33 taskset-pinned
processes over disjoint core sets. Extra instrumentation captured per rule:
  * learning_curve: [(step, test_acc, wall_s)] at each --eval-every
  * lat_ns_samples: full raw per-step latency sample list
  * ru_maxrss_mb: peak RSS of this process (true memory for the whole run)
  * xla_peak_memory: deterministic per-grad XLA peak (cross-check)
  * rec_cosine (recorded) + replay_cosine (replayed/true) per seq
"""
import os, re, sys, json, time, argparse, resource
import numpy as np

argp = argparse.ArgumentParser()
argp.add_argument("--source", required=True)
argp.add_argument("--front", required=True)
argp.add_argument("--out", required=True)
argp.add_argument("--shard", type=int, default=0)
argp.add_argument("--nshards", type=int, default=1)
argp.add_argument("--rule-index", type=int, default=-1,
                  help=">=0: run only front[I]; -2: run only reverse_mode; -1: legacy shard mode")
argp.add_argument("--seeds", type=int, default=20)
argp.add_argument("--seed-offset", type=int, default=0,
                  help="seed index of the FIRST seed (seeds run 1000+offset .. 1000+offset+seeds-1); "
                       "lets a parallel sbatch run 1 seed per process with a distinct init/data order")
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
    """Run _callback once, capturing the compiled value_and_grad executable
    AND the raw_sink (replayed cosine + deterministic xla_peak_memory)."""
    cap = {}
    raw_sink = {}
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
                  *eval_samples, raw_sink=raw_sink)
    finally:
        ccmod.cached_compile = orig
    return cap.get("fn"), raw_sink


def init_weights(seed, in_dim=784, hid=63, out=10):  # hid MUST match env NN (_EQ_NN_HIDDEN=63); the AOT approx gfn is compiled for this shape
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


def train_one(gfn, seed, xtr, ytr, xte, yte, t0=None, learning_curve=None):
    """Train with the (approx) grad fn until train-loss early-stops; return
    final test accuracy + steps taken. If learning_curve is not None, append
    (step, test_acc, wall_s) at every eval (uses t0 for wall clock)."""
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
            if learning_curve is not None:
                acc_now = accuracy(W, xte, yte)
                wall = (time.time() - t0) if t0 is not None else 0.0
                learning_curve.append([int(step), float(acc_now), float(wall)])
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
    return float(np.mean(w)), float(np.std(ts)), [float(x) for x in ts]


def run_rule(label, gfn, xtr, ytr, xte, yte, seeds):
    accs, steps = [], []
    learning_curves = []  # per-seed [(step, test_acc, wall_s)] for std bands
    seed_ids = []
    for s in range(a.seed_offset, a.seed_offset + seeds):
        t0 = time.time()
        lc = []
        acc, st = train_one(gfn, 1000 + s, xtr, ytr, xte, yte, t0=t0, learning_curve=lc)
        accs.append(acc); steps.append(st)
        learning_curves.append(lc); seed_ids.append(1000 + s)
    learning_curve = learning_curves[0] if learning_curves else []  # back-compat: first-seed curve
    W0 = init_weights(0)
    lat_mean, lat_std, lat_samples = measure_latency(gfn, xtr[:a.batch], ytr[:a.batch], W0)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux KB -> MB
    return {
        "label": label,
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "accs": [float(x) for x in accs],
        "seed_ids": seed_ids,
        "steps_mean": float(np.mean(steps)),
        "lat_ns_mean": lat_mean, "lat_ns_std": lat_std,
        "lat_ns_samples": lat_samples,
        "learning_curve": learning_curve,
        "learning_curves": learning_curves,
        "ru_maxrss_mb": float(ru),
    }


def run_reverse(xtr, ytr, xte, yte):
    loss_fn = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
    rev = jax.jit(jax.value_and_grad(loss_fn, argnums=ARGN))
    print("[reverse-mode] training baseline...", flush=True)
    r = run_rule("reverse_mode", rev, xtr, ytr, xte, yte, a.seeds)
    print("  reverse:", r["acc_mean"], r["lat_ns_mean"], flush=True)
    return r


def run_front_index(i, p, env, eval_samples, xtr, ytr, xte, yte):
    seq = p["seq"]
    order, specs, nr = build_order_specs(seq, env)
    t0 = time.time()
    try:
        gfn, raw_sink = capture_gfn(env, eval_samples, order, specs)
        if gfn is None:
            raise RuntimeError("no compiled fn captured")
        r = run_rule(f"{a.source}#{i}", gfn, xtr, ytr, xte, yte, a.seeds)
        replay_cos = raw_sink.get("cosine_sim_per_point") or []
        r["replay_cosine"] = float(replay_cos[0]) if replay_cos else None
        r["replay_cosine_per_point"] = replay_cos
        r["xla_peak_memory"] = float(raw_sink.get("xla_peak_memory", 0.0))
        r["replay_frob"] = (raw_sink.get("frob_residual_per_point") or [None])[0]
    except Exception as exc:
        r = {"label": f"{a.source}#{i}", "error": f"{type(exc).__name__}: {exc}"[:200]}
    r["idx"] = i
    r["front_obj"] = p["obj"]
    r["seq_label"] = p.get("label")
    r["rec_cosine"] = p.get("rec_cosine")
    r["rec_latency_ns"] = p.get("rec_latency_ns")
    r["rec_xla_peak"] = p.get("rec_xla_peak")
    r["wall_s"] = round(time.time() - t0, 1)
    print(f"  [{i}] {r.get('acc_mean','ERR')} acc  lat={r.get('lat_ns_mean',0)/1e3:.0f}us  "
          f"rec_cos={r.get('rec_cosine')} replay_cos={r.get('replay_cosine')}  ({r['wall_s']}s)", flush=True)
    return r


def main():
    xtr, ytr = load_dataset("mnist", None, "train")
    xte, yte = load_dataset("mnist", None, "test")
    xtr, ytr, xte, yte = (jnp.asarray(z) for z in (xtr, ytr, xte, yte))
    print(f"[data] train {xtr.shape} test {xte.shape}", flush=True)

    d = json.load(open(os.path.expanduser(a.front)))
    front = d["front"]
    results = []

    # --- single-rule mode (parallel fan-out) ---
    if a.rule_index == -2:
        results.append(run_reverse(xtr, ytr, xte, yte))
    elif a.rule_index >= 0:
        env, eval_samples = build_env()
        i = a.rule_index
        results.append(run_front_index(i, front[i], env, eval_samples, xtr, ytr, xte, yte))
    else:
        # --- legacy shard mode ---
        env, eval_samples = build_env()
        if a.shard == 0:
            results.append(run_reverse(xtr, ytr, xte, yte))
        mine = [(i, p) for i, p in enumerate(front) if i % a.nshards == a.shard]
        for i, p in mine:
            results.append(run_front_index(i, p, env, eval_samples, xtr, ytr, xte, yte))

    os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True)
    json.dump({"source": a.source, "shard": a.shard, "rule_index": a.rule_index,
               "results": results},
              open(os.path.expanduser(a.out), "w"), indent=2)
    print(f"[done] {len(results)} results -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
