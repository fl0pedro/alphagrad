"""Offline validation of trajectory multi-step cosine proxy.
Builds rules directly as action-index sequences (0..n_valid-1):
  EXACT        : full order, no micro-actions -> cosine~1, stable along trajectory
  QUANT_bf16   : every vertex quantized to bfloat16 -> mild fidelity loss
  QUANT_f16    : every vertex quantized to float16 -> stronger fidelity loss/degrade
Discriminative check: trajectory cosine separates stable from degrading fidelity."""
import os, json
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

MODEL = "VmappedNeuralNetwork"
LOSS = scalar_loss_fn(get_fn(MODEL))
ARGN = infer_argnums(MODEL)
CI = REWARD_INDEX["cosine_sim"]
NDATA = 3

def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args(MODEL, ak, dataset="mnist")
    gen = data_gen(MODEL, dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory", measure_latency=False,
        num_data_points=NDATA, reps_per_point=1, percentile_keep=0.60,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter")
    ev = generate_eval_samples(env, ek, NDATA)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev

env, ev = build_env()
NV = len(env.valid_vertices)

def make_seq(calls_per_vertex):
    # full elimination order over all action indices 0..NV-1, given calls list per step
    return [[v, list(calls_per_vertex)] for v in range(NV)]

def make_seq_compress(kind):
    return [[v, [f"compress('{kind}', 1)"]] for v in range(NV)]

RULES = {
    "EXACT_noapprox": make_seq([]),
    "QUANT_bf16_all": make_seq(["quant('bfloat16')"]),
    "QUANT_f16_all":  make_seq(["quant('float16')"]),
    "COMPRESS_mean":  make_seq_compress("mean"),
    "COMPRESS_max":   make_seq_compress("max"),
}

def measure(seq, proxy):
    if proxy: os.environ["ALPHAGRAD_QUALITY_PROXY"] = "trajectory"
    else: os.environ.pop("ALPHAGRAD_QUALITY_PROXY", None)
    order, specs, _ = build_order_specs(seq, env)
    _, _, reward = _callback(env.config, env.args, env.consts,
                             jnp.asarray(order), jnp.asarray(specs), len(order),
                             *ev, raw_sink={})
    jax.clear_caches()
    return float(np.asarray(reward)[CI])

print(f"{'rule':<18} {'single_cos':>12} {'traj_cos':>12} {'delta':>10}  finite")
results = {}
for label, seq in RULES.items():
    try:
        sp = measure(seq, proxy=False)
    except Exception as e:
        sp = float("nan"); print(f"  [single err {label}]: {e}")
    try:
        tp = measure(seq, proxy=True)
    except Exception as e:
        tp = float("nan"); print(f"  [traj err {label}]: {e}")
    fin = np.isfinite(sp) and np.isfinite(tp)
    print(f"{label:<18} {sp:>12.4f} {tp:>12.4f} {tp-sp:>10.4f}  {fin}")
    results[label] = dict(single=sp, traj=tp)

print("\n=== VALIDATION ===")
ex = results["EXACT_noapprox"]
# pick the rule with the LOWEST single-point cosine as the "degrading" exemplar
deg_label = min(results, key=lambda k: results[k]['single'] if np.isfinite(results[k]['single']) else 1e9)
deg = results[deg_label]
def _fin(r): return bool(np.isfinite(float(r['single'])) and np.isfinite(float(r['traj'])))
allf = all(_fin(r) for r in results.values())
print(f"[1] all finite: {allf}")
print(f"[2] traj in [0,1]: {all(0.0 <= float(r['traj']) <= 1.0 for r in results.values() if np.isfinite(float(r['traj'])))}")
print(f"[3] EXACT traj stays high (>0.9): {ex['traj'] > 0.9}  (traj={ex['traj']:.4f})")
print(f"[4] DISCRIMINATIVE: EXACT traj ({ex['traj']:.4f}) > degrading '{deg_label}' traj ({deg['traj']:.4f}): {ex['traj'] > deg['traj']}")
print(f"[5] degrading rule '{deg_label}' single={deg['single']:.4f} traj={deg['traj']:.4f} (traj reflects fidelity over trajectory)")
print("\nsummary:")
for k, r in results.items():
    print(f"  {k:<18} single={r['single']:+.4f} traj={r['traj']:+.4f} delta={r['traj']-r['single']:+.4f}")
