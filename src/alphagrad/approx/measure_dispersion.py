"""Phase 0 (dispersion gate) — measurement. Re-measure a stratified sample of MORL
candidate configs with the CURRENT aligned env, capturing the per-data-point cosine
VECTOR (not collapsed) plus deterministic xla_peak_memory and single-core latency.
The downstream gate (analyze_cosine_dispersion.py) decides whether cosine's
within-config tail spread is large enough vs between-config gaps to justify a
distributional (CVaR) reward layer.

Measurement discipline (per plan): cosine layout-aligned (env _align_jac, on by
default), peak memory = xla_peak_memory (measure_grad=True), latency on an ISOLATED
single core (this process pins its own affinity; launcher sets OMP=MKL=1). Sharded,
resumable, jax.clear_caches() per config to bound RSS.

  uv run python src/alphagrad/approx/measure_dispersion.py --shard 0 --nshards 48 \
      --nsample 400 --ndata 16 --out ~/dsnn/train_exp/dispersion/d_0.jsonl
"""
import os, glob, json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--nsample", type=int, default=400, help="total configs sampled across both sources")
ap.add_argument("--ndata", type=int, default=16, help="data points per config => cosine vector length")
ap.add_argument("--reps", type=int, default=3, help="latency reps per point")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

# --- pin this process to a single physical-ish core for clean latency (no threading) ---
try:
    avail = sorted(os.sched_getaffinity(0))
    core = avail[A.shard % len(avail)]
    os.sched_setaffinity(0, {core})
    _PINNED = [core]
except (AttributeError, OSError, ValueError):
    _PINNED = None
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
ARGN = infer_argnums("VmappedNeuralNetwork")
CI, LI, XI = REWARD_INDEX["cosine_sim"], REWARD_INDEX["latency_ns"], REWARD_INDEX["xla_peak_memory"]


def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory", measure_latency=True,
        num_data_points=A.ndata, reps_per_point=A.reps, percentile_keep=0.60,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="rm")
    ev = generate_eval_samples(env, ek, A.ndata)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev


def stratified_sample():
    """Deterministic stratified sample across the stored-cosine range, both sources."""
    pool = []
    for src in ("cmorl", "mogfn"):
        f = os.path.expanduser(f"~/dsnn/morl_fronts/{src}_allcand.json")
        for c in json.load(open(f))["candidates"]:
            pool.append((src, c["seq"], float(c["obj"]["cosine_sim"])))
    pool.sort(key=lambda t: t[2])                       # by stored cosine
    n = min(A.nsample, len(pool))
    idxs = np.linspace(0, len(pool) - 1, n).round().astype(int)
    return [(int(i), pool[i][0], pool[i][1]) for i in idxs]  # (global_id, src, seq)


def measure(seq):
    order, specs, _ = build_order_specs(seq, env)
    rs = {}
    _, _, reward = _callback(env.config, env.args, env.consts,
                             jnp.asarray(order), jnp.asarray(specs), len(order), *ev, raw_sink=rs)
    reward = np.asarray(reward)
    return rs.get("cosine_sim_per_point", []), float(reward[LI]), float(reward[XI]), float(reward[CI])


env, ev = build_env()
sample = stratified_sample()
mine = [(gid, src, seq) for k, (gid, src, seq) in enumerate(sample) if k % A.nshards == A.shard]
os.makedirs(os.path.dirname(A.out), exist_ok=True)
done = set()
for f in glob.glob(os.path.join(os.path.dirname(A.out), "d_*.jsonl")):
    for line in open(f):
        try:
            done.add(json.loads(line)["gid"])
        except Exception:
            pass
with open(A.out, "a") as fh:
    for gid, src, seq in mine:
        if gid in done:
            continue
        try:
            cpp, lat, xla, cos_collapsed = measure(seq)
        except Exception as e:
            cpp, lat, xla, cos_collapsed = [], float("nan"), float("nan"), float("nan")
        fh.write(json.dumps(dict(gid=gid, src=src, cos_per_point=[float(x) for x in cpp],
                                 latency_ns=lat, xla_peak_memory=xla,
                                 cos_collapsed=cos_collapsed)) + "\n")
        fh.flush()
        jax.clear_caches()
print(f"done shard {A.shard}/{A.nshards} pinned={_PINNED} measured={len(mine)}")
