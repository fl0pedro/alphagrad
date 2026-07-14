"""Task1: measure the cossim edge C (fracred vs cossim) + live-rule cossim.
CPU-only. Enumerates diverse rules on the 256-NN grad graph, records
(cossim, fracred=bkstep_acc) for each, and replays the live best.json rules.
"""
import os, json, argparse, itertools, random
os.environ.setdefault("ALPHAGRAD_BKSTEP", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP_K", "40")
os.environ.setdefault("ALPHAGRAD_BKSTEP_SEEDS", "2")
os.environ.setdefault("ALPHAGRAD_BKSTEP_SIGNAL", "fracred")
os.environ.setdefault("ALPHAGRAD_NN_HIDDEN", "256")
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs

def _op_to_str(o):
    if isinstance(o, str):
        return o
    kind_op = o.get("op")
    if kind_op == "Compress":
        ax = o.get("axes", [0])[0]
        return f"compress('{o.get('kind','mean')}', {int(ax)})"
    if kind_op == "Quant":
        return f"quant('{o.get('dtype','int4')}')"
    if kind_op == "Diag":
        return f"diag({int(o.get('i',0))}, {int(o.get('j',0))}, {int(o.get('factor',-1))})"
    raise ValueError(f"bad op {o!r}")

def _norm_seq(seq):
    out = []
    for s in seq:
        if isinstance(s, dict):
            v = s["vertex"]; ops = s.get("ops", [])
        else:
            v, ops = s
        out.append((v, [_op_to_str(o) for o in ops]))
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--best", default=os.path.expanduser("~/dsnn/campaign_full_nn256_grad/nn256/best.json"))
ap.add_argument("--nrandom", type=int, default=24)
ap.add_argument("--ndata", type=int, default=5)
A = ap.parse_args()

LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
ARGN = infer_argnums("VmappedNeuralNetwork")
CI = REWARD_INDEX["cosine_sim"]
BI = REWARD_INDEX["bkstep_acc"]

def build_env():
    k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
    xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
    gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
    closed = jax.make_jaxpr(LOSS)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
        cmp_type="latency", mem_type="peak_memory", measure_latency=False,
        num_data_points=A.ndata, reps_per_point=1, percentile_keep=0.60,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        measure_grad=True, latency_timer="perf_counter")
    ev = generate_eval_samples(env, ek, A.ndata)
    return eqx.tree_at(lambda e: e.eval_args_samples, env, ev), ev

env, ev = build_env()
valid = np.asarray(env.valid_vertices, dtype=np.int32)
nvtx = len(valid)
print(f"[edge] nvtx={nvtx} valid={valid.tolist()}", flush=True)

def measure_seq(seq, label):
    try:
        seq = _norm_seq(seq)
        order, specs, nr = build_order_specs(seq, env)
        rs = {}
        _callback(env.config, env.args, env.consts,
                  jnp.asarray(order), jnp.asarray(specs), len(order), *ev, raw_sink=rs)
        cos_pp = rs.get("cosine_sim_per_point", [])
        cos_mean = float(np.mean(cos_pp)) if cos_pp else float("nan")
        cos_p60 = float(np.percentile(cos_pp, 60)) if cos_pp else float("nan")
        bk = float(rs.get("bkstep_acc", float("nan")))
        print(f"[edge] {label:28s} nrules={nr:2d} cos_mean={cos_mean:+.4f} cos_p60={cos_p60:+.4f} fracred={bk:.4f}", flush=True)
        return cos_mean, cos_p60, bk
    except Exception as e:
        print(f"[edge] {label}: FAIL {type(e).__name__}: {e}", flush=True)
        return None

# --- live rules from best.json ---
results = []
try:
    bj = json.load(open(A.best))
    live = {}
    if "best_overall" in bj and "seq" in bj["best_overall"]:
        live["live_best_overall"] = bj["best_overall"]["seq"]
    for ch, info in bj.get("best_per_channel", {}).items():
        if isinstance(info, dict) and info.get("seq"):
            live[f"live_{ch}"] = info["seq"]
    for name, seq in live.items():
        r = measure_seq(seq, name)
        if r: results.append(("live", *r))
except Exception as e:
    print(f"[edge] live load fail: {e}", flush=True)

# --- enumerate diverse rules: vary #COMPRESS from 0..nvtx to span cossim range ---
# A rule = per-vertex op list in a fixed identity order (0..nvtx-1 action idx).
KINDS = ["min", "mean"]
def make_seq(ncompress, nquant, quant_dtype="int4", ckind="mean"):
    idxs = list(range(nvtx))
    seq = []
    for i in idxs:
        ops = []
        if i < ncompress:
            ops = [f"compress('{ckind}', 0)"]
        elif i < ncompress + nquant:
            ops = [f"quant('{quant_dtype}')"]
        seq.append((i, ops))
    return seq

# Sweep: mostly-identity (few ops -> high cossim) down to all-compress (low cossim)
configs = []
for nc in range(0, nvtx+1, max(1, nvtx//8)):
    configs.append((f"c{nc}q0_mean", make_seq(nc, 0, ckind="mean")))
for nq in [1,2,4]:
    configs.append((f"c0q{nq}_int8", make_seq(0, nq, "int8")))
    configs.append((f"c0q{nq}_int4", make_seq(0, nq, "int4")))
# random mixes for coverage near the edge
rng = random.Random(0)
for r in range(A.nrandom):
    seq=[]
    for i in range(nvtx):
        c = rng.random()
        if c < 0.35:
            seq.append((i, [f"compress('{rng.choice(KINDS)}', {rng.choice([0,1])})"]))
        elif c < 0.55:
            _qd = rng.choice(["int8", "int4", "int2", "uint8"])
            seq.append((i, [f"quant('{_qd}')"]))
        else:
            seq.append((i, []))
    configs.append((f"rand{r}", seq))

for name, seq in configs:
    r = measure_seq(seq, name)
    if r: results.append(("sweep", *r))

# --- edge analysis ---
sweep = [(c,b) for tag,cm,c,b in results if not (np.isnan(c) or np.isnan(b))]
if sweep:
    arr = np.array(sweep)  # cos_p60, fracred
    order = arr[arr[:,0].argsort()]
    plateau = float(np.percentile(arr[:,1], 90))  # robust plateau estimate
    thr = 0.9 * plateau
    # smallest cossim where fracred >= 0.9*plateau (monotone-ish)
    above = order[order[:,1] >= thr]
    C = float(above[0,0]) if len(above) else float("nan")
    print(f"\n[edge] === SUMMARY ===")
    print(f"[edge] plateau(fracred p90)={plateau:.4f}  0.9*plateau={thr:.4f}")
    print(f"[edge] EDGE CAP C (min cossim reaching 0.9*plateau) = {C:.4f}")
    print(f"[edge] sorted (cos_p60, fracred):")
    for c,b in order:
        print(f"[edge]   cos={c:+.4f}  fracred={b:.4f}")
