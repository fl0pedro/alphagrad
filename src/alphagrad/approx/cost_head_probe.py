"""Phase-1 learned-cost-model GO/NO-GO probe (offline, single GPU).

Adds a cost_head to a MicroPPOAgent (mirrors value_head: attention-pool the
encoder output via its OWN pool query, then MLP(embd_dim -> NUM_REWARDS)),
FREEZES the encoder, generates a supervised dataset of
(order+micro-seq -> measured 4-tuple) via the env grad-measure raw_sink, trains
the cost_head (Huber in symlog space, held-out split), and reports per-channel
Spearman + MSE + a GO/NO-GO read.

Encoder snapshot: FRESH-INIT (no trained policy checkpoint on disk; the runs
did not serialise agent.eqx). This is a WEAKER test than a trained encoder --
flagged in the report. Does NOT touch anything the live run 52572 writes.
"""
import os, sys, argparse, time, json
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_SYNC", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_ABSOLUTE", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("ALPHAGRAD_QUANT_ALLOWED", "int8,int16,bfloat16,float16")  # promotion-safe (avoid float8 fatal)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax, jax.numpy as jnp, equinox as eqx

ap = argparse.ArgumentParser()
ap.add_argument("--example", default="VmappedNeuralNetwork")
ap.add_argument("--nn-hidden", type=int, default=256)
ap.add_argument("--n-orders", type=int, default=600)
ap.add_argument("--ndata", type=int, default=3)
ap.add_argument("--reps-per-point", type=int, default=1)
ap.add_argument("--latency-inner-reps", type=int, default=50)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.expanduser("~/dsnn/cost_head_probe_out"))
A = ap.parse_args()
os.environ["ALPHAGRAD_NN_HIDDEN"] = str(A.nn_hidden)
os.makedirs(A.out, exist_ok=True)

from alphagrad.approx.env import (
    VertexEliminationEnv, _callback, REWARD_INDEX, MAX_TOKENS,
)
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn,
)
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent, NUM_REWARDS
from alphagrad.transformer import MLP

TARGET_CHANNELS = ["latency_ns", "peak_memory", "flops", "cosine_sim"]
TIDX = {c: REWARD_INDEX[c] for c in TARGET_CHANNELS}
print(f"[probe] node={os.environ.get('SLURMD_NODENAME','?')} device={jax.devices()[0]} "
      f"example={A.example} nn_hidden={A.nn_hidden}", flush=True)

# ---------------------------------------------------------------- env
LOSS = scalar_loss_fn(get_fn(A.example))
ARGN = infer_argnums(A.example)
def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args(A.example, ak, dataset="mnist")
    gen = data_gen(A.example, dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory",
        exec_on_gpu=True, measure_latency=True,
        num_data_points=A.ndata, reps_per_point=A.reps_per_point,
        percentile_keep=0.60, slow_exec_cutoff_seconds=0.0,
        flop_gate_threshold=0.0, measure_grad=True,
        latency_inner_reps=A.latency_inner_reps, latency_timer="perf_counter",
    )
    ev = generate_eval_samples(env, ek, A.ndata)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev
env, ev = build_env()
valid = np.asarray(env.valid_vertices, dtype=np.int32)
nvtx = len(valid)
print(f"[probe] nvtx={nvtx}", flush=True)

# ---------------------------------------------------------------- agent (frozen encoder) + cost_head
# Match the runs' agent config exactly: vocab 512, embd 128, 4 layers/heads,
# hidden 256, palimpsa backbone. num_vertices = jaxpr eqn count (worker total_v).
key = jax.random.PRNGKey(A.seed)
kA, kC = jax.random.split(key)
EMBD = 128
_total_v = len(jax.make_jaxpr(LOSS)(*get_args(A.example, jax.random.PRNGKey(0), dataset="mnist")).jaxpr.eqns)
agent = MicroPPOAgent(
    vocab_size=512,
    embd_dim=EMBD, num_layers=4, num_heads=4, hidden_dim=256,
    num_vertices=_total_v, value_dims=(128, 128), key=kA,
    max_substeps=16, policy="palimpsa",
)
# cost_head mirrors value_head EXACTLY: own attention pool query + MLP->NUM_REWARDS.
cost_pool_query = jax.random.normal(kC, (EMBD,)) * 0.02
cost_head = MLP(EMBD, NUM_REWARDS, (128, 128), key=jax.random.split(kC)[0])
print("[probe] ENCODER = FRESH-INIT (no trained checkpoint on disk) -> WEAKER TEST", flush=True)

def encode_ctx(tokens_j, eqn_ids_j):
    """Frozen-encoder attention-pooled context (mirrors value_from_encoding)."""
    enc_x, tok_mask = agent.encode_tokens(tokens_j, key=jax.random.PRNGKey(0), eqn_ids=eqn_ids_j)
    scores = (enc_x @ cost_pool_query) / jnp.sqrt(jnp.float32(EMBD))
    scores = jnp.where(tok_mask, scores, -1e9)
    attn = jax.nn.softmax(scores, axis=-1)
    pooled = jnp.sum(attn[:, None] * enc_x, axis=0)
    return pooled  # (EMBD,)
encode_ctx_j = eqx.filter_jit(encode_ctx)

# ---------------------------------------------------------------- order generators (MIX)
def reverse_order():
    return list(reversed(range(nvtx)))
def forward_order():
    return list(range(nvtx))
def random_order(rng):
    o = list(range(nvtx)); rng.shuffle(o); return o
def two_swap(base, rng):
    o = list(base); a, b = rng.integers(0, nvtx, 2); o[a], o[b] = o[b], o[a]; return o

QD = ["int8", "int16", "bfloat16", "float16"]  # promotion-safe only
def make_seq(order, rng, budget):
    """order = list of 0-based action idx; budget micro-ops per vertex (0..).

    COMPRESS is EXCLUDED: on the NN/ViT grad graph a COMPRESS densify triggers
    an UNCATCHABLE Triton mixed-dtype LLVM fatal (kills the process; same class
    seen in sigma calibration). QUANT gives the needed cosine variance
    (quant -> lower fidelity, cosine < 1) without the fatal; budget=0 = pure
    elimination (cosine=1). So the cosine target still spans [~low, 1.0].
    """
    seq = []
    for v in order:
        ops = [f"quant('{rng.choice(QD)}')" for _ in range(int(budget))]
        seq.append((v, ops))
    return seq

def gen_configs(n, seed=0):
    rng = np.random.default_rng(seed)
    cfgs = []
    rev = reverse_order(); fwd = forward_order()
    # budgets spread 0..3 so cosine has variance; budget 0 = pure elimination.
    while len(cfgs) < n:
        pick = rng.random()
        if pick < 0.20:
            order = rev
        elif pick < 0.35:
            order = fwd
        elif pick < 0.60:
            order = two_swap(rev, rng)
        else:
            order = random_order(rng)
        budget = int(rng.choice([0, 0, 1, 1, 2, 3]))  # skew to small (measurable)
        cfgs.append((make_seq(order, rng, budget), budget))
    return cfgs

# ---------------------------------------------------------------- dataset
X, Y = [], []   # X: pooled ctx (EMBD,), Y: raw 4-tuple (measured)
budgets_seen = []
t0 = time.time()
cfgs = gen_configs(A.n_orders, seed=A.seed)
n_ok = n_fail = 0
for idx, (seq, budget) in enumerate(cfgs):
    try:
        order, specs, _ = build_order_specs(seq, env)
        rs = {}
        tokens, eqn_ids, _rew = _callback(
            env.config, env.args, env.consts,
            jnp.asarray(order), jnp.asarray(specs), len(order), *ev, raw_sink=rs,
        )
        lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
        flops = float(rs.get("flops", np.nan))
        cos_pp = rs.get("cosine_sim_per_point", [])
        lat = float(np.mean(lat_s)) if lat_s else np.nan
        # LEAK-FREE PEAK (Phase 1b): use the analytic xla_peak_memory (deterministic
        # XLA temp+output+args estimate) instead of the RM runtime peak. The RM
        # peak tracked the measure-GPU executable leak (peak vs collection-index
        # Spearman 0.999) not per-order memory; xla_peak_memory is order-intrinsic
        # and leak-immune.
        peak = float(rs.get("xla_peak_memory", np.nan))
        cos = float(np.mean(cos_pp)) if cos_pp else np.nan
        if not (np.isfinite(lat) and np.isfinite(peak) and np.isfinite(flops) and np.isfinite(cos)):
            n_fail += 1; continue
        ctx = np.asarray(encode_ctx_j(jnp.asarray(tokens), jnp.asarray(eqn_ids)))
        if not np.all(np.isfinite(ctx)):
            n_fail += 1; continue
        X.append(ctx)
        Y.append([lat, peak, flops, cos])
        budgets_seen.append(budget)
        n_ok += 1
        # Mitigate the measure-GPU executable leak so collection reaches larger N:
        # drop the in-process XLA compile cache every 64 measures (frees retained
        # executables; on-disk cache survives -> recurring shapes reload cheap).
        if (idx + 1) % 64 == 0:
            jax.clear_caches()
            import gc as _gc; _gc.collect()
        if n_ok % 50 == 0:
            print(f"[probe] collected {n_ok} ok / {n_fail} fail / {idx+1} tried "
                  f"({time.time()-t0:.0f}s)", flush=True)
            # incremental checkpoint so an uncatchable Triton fatal mid-run
            # doesn't lose the dataset (eval can rerun on the saved .npz).
            np.savez(os.path.join(A.out, "dataset_partial.npz"),
                     X=np.asarray(X, dtype=np.float64),
                     Y=np.asarray(Y, dtype=np.float64),
                     channels=TARGET_CHANNELS, budgets=np.asarray(budgets_seen))
    except Exception as e:
        n_fail += 1
        if n_fail <= 5:
            print(f"[probe] cfg {idx} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)

X = np.asarray(X, dtype=np.float64)
Y = np.asarray(Y, dtype=np.float64)  # (N, 4) raw
print(f"[probe] DATASET: N={len(X)} ok, {n_fail} fail, budgets={np.bincount(budgets_seen)} "
      f"({time.time()-t0:.0f}s)", flush=True)
np.savez(os.path.join(A.out, "dataset.npz"), X=X, Y=Y, channels=TARGET_CHANNELS,
         budgets=np.asarray(budgets_seen))

if len(X) < 40:
    print("[probe] INSUFFICIENT DATA — abort", flush=True); sys.exit(1)

# ---------------------------------------------------------------- symlog targets + split
def symlog(v): return np.sign(v) * np.log1p(np.abs(v))
Ys = symlog(Y)  # symlog space regression targets
# per-channel standardize (for training stability); invert only affects MSE scale
mu = Ys.mean(0); sd = Ys.std(0) + 1e-8
Yn = (Ys - mu) / sd
rng = np.random.default_rng(A.seed)
perm = rng.permutation(len(X))
n_te = max(20, int(0.2 * len(X)))
te_idx, tr_idx = perm[:n_te], perm[n_te:]
Xtr, Ytr = jnp.asarray(X[tr_idx]), jnp.asarray(Yn[tr_idx])
Xte, Yte = jnp.asarray(X[te_idx]), jnp.asarray(Yn[te_idx])
print(f"[probe] split: train={len(tr_idx)} test={len(te_idx)}", flush=True)

# ---------------------------------------------------------------- train cost_head (encoder FROZEN)
import optax
def huber_loss(head, xb, yb):
    pred = jax.vmap(head)(xb)              # (B, NUM_REWARDS)
    pred4 = pred[:, [TIDX[c] for c in TARGET_CHANNELS]]  # select the 4 target channels
    return jnp.mean(optax.huber_loss(pred4, yb, delta=1.0))

head = cost_head  # ONLY the cost_head trains; encoder + pool query are frozen (not in optimizer)
opt = optax.adam(1e-3)
opt_state = opt.init(eqx.filter(head, eqx.is_array))
@eqx.filter_jit
def step(head, opt_state, xb, yb):
    loss, grads = eqx.filter_value_and_grad(huber_loss)(head, xb, yb)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(head, eqx.is_array))
    head = eqx.apply_updates(head, updates)
    return head, opt_state, loss
for ep in range(A.epochs):
    head, opt_state, loss = step(head, opt_state, Xtr, Ytr)
    if ep % 50 == 0 or ep == A.epochs - 1:
        print(f"[probe] train ep{ep} huber={float(loss):.4f}", flush=True)

# ---------------------------------------------------------------- held-out eval
from scipy.stats import spearmanr
pred_te = np.asarray(jax.vmap(head)(Xte))[:, [TIDX[c] for c in TARGET_CHANNELS]]  # (n_te, 4) normalized
Yte_np = np.asarray(Yte)
print("\n[probe] ===== HELD-OUT PER-CHANNEL (symlog-normalized space) =====", flush=True)
results = {}
for k, ch in enumerate(TARGET_CHANNELS):
    p = pred_te[:, k]; t = Yte_np[:, k]
    if np.std(t) < 1e-9:
        rho, mse = float("nan"), float(np.mean((p - t) ** 2))
        note = "(target constant on held-out -> Spearman undefined)"
    else:
        rho = float(spearmanr(p, t).correlation)
        mse = float(np.mean((p - t) ** 2))
        note = ""
    results[ch] = {"spearman": rho, "mse": mse}
    print(f"[RESULT] {ch:12s} Spearman={rho:+.3f}  MSE(norm)={mse:.4f}  {note}", flush=True)

go_lat = results["latency_ns"]["spearman"]
go_peak = results["peak_memory"]["spearman"]
GO = (np.isfinite(go_lat) and go_lat > 0.7) and (np.isfinite(go_peak) and go_peak > 0.7)
print(f"\n[GO-NO-GO] latency Spearman={go_lat:+.3f}, peak Spearman={go_peak:+.3f} "
      f"(threshold >0.7 both) -> {'GO' if GO else 'NO-GO'}", flush=True)
print(f"[GO-NO-GO] ENCODER=fresh-init (weaker test); measured on "
      f"{os.environ.get('SLURMD_NODENAME','?')}", flush=True)
json.dump({"results": results, "GO": bool(GO), "encoder": "fresh-init",
           "N": len(X), "node": os.environ.get("SLURMD_NODENAME", "?")},
          open(os.path.join(A.out, "results.json"), "w"), indent=2)
print("[probe] DONE", flush=True)
