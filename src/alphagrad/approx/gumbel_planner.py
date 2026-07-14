"""Phase-2 MVP: Gumbel-AlphaZero planner over elimination ORDERS (NN grad graph),
leaf-evaluated by the Phase-1b cost model. Pure ORDER only (no micro-actions).

Acid test: MEASURE the planned order's real grad cost vs reverse-mode,
Markowitz-greedy, and forward orders (same grad-measure harness). Also report
cost-model-predicted vs measured for the planned order.

Node: single GPU (arg). Does NOT touch the live run.
"""
import os, sys, math, argparse, json, time
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_SYNC", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
import optax

ap = argparse.ArgumentParser()
ap.add_argument("--nn-hidden", type=int, default=256)
ap.add_argument("--dataset", default=os.path.expanduser("~/dsnn/cost_head_probe_out_v2/dataset_partial.npz"))
ap.add_argument("--ndata", type=int, default=3)
ap.add_argument("--latency-inner-reps", type=int, default=50)
ap.add_argument("--n-candidates", type=int, default=8)   # Gumbel top-m at root
ap.add_argument("--sim-budget", type=int, default=32)    # sequential-halving sims
ap.add_argument("--n-plan-restarts", type=int, default=6)
ap.add_argument("--train-epochs", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
os.environ["ALPHAGRAD_NN_HIDDEN"] = str(A.nn_hidden)
node = os.environ.get("SLURMD_NODENAME", "?")
print(f"[plan] node={node} device={jax.devices()[0]} nn_hidden={A.nn_hidden}", flush=True)

from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX, MAX_TOKENS
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent, NUM_REWARDS
from alphagrad.transformer import MLP
from graphax.core import _build_graph, _prune_graph, _eliminate_vertex
MAX_RULES = 16  # MAX_RULES_PER_VERTEX

LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork"))
ARGN = infer_argnums("VmappedNeuralNetwork")
CHANNELS = ["latency_ns", "peak_memory", "flops", "cosine_sim"]

# --------------------------------------------------------------- env + jaxpr graph
k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
closed = jax.make_jaxpr(LOSS)(*xs)
env = VertexEliminationEnv.from_jaxpr(
    closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
    cmp_type="latency", mem_type="peak_memory", exec_on_gpu=True, measure_latency=True,
    num_data_points=A.ndata, reps_per_point=1, percentile_keep=0.60,
    slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0, measure_grad=True,
    latency_inner_reps=A.latency_inner_reps, latency_timer="perf_counter",
)
ev = generate_eval_samples(env, ek, A.ndata)
env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)
VALID = list(np.asarray(env.valid_vertices, dtype=np.int32))   # 1-based ids, authoritative
NV = len(VALID)
print(f"[plan] valid_vertices (1-based) = {VALID}  NV={NV}", flush=True)

jaxpr = closed.jaxpr
# graphax connectivity for Markowitz + legality (pure planning transition)
_env_g, GRAPH0, TG0, VO = _build_graph(jaxpr, xs, closed.literals, ARGN)
_prune_graph(GRAPH0, TG0, jaxpr, ARGN)

def copy_g(g): return {kk: dict(vv) for kk, vv in g.items()}

def outvar(i):  # 1-based vertex -> central outvar
    return jaxpr.eqns[i - 1].outvars[0]

def legal_set(graph):
    out = []
    for i in VALID:
        ov = outvar(i)
        if ov in graph:
            out.append(i)
    return out

def markowitz_degrees(graph, tg, cand):
    d = {}
    for i in cand:
        ov = outvar(i)
        d[i] = max(1, len(tg.get(ov, {}))) * max(1, len(graph.get(ov, {})))
    return d

# --------------------------------------------------------------- cost model
d = np.load(A.dataset, allow_pickle=True)
X = np.asarray(d["X"], np.float64); Y = np.asarray(d["Y"], np.float64)
print(f"[plan] cost dataset N={len(X)}", flush=True)
def symlog(v): return np.sign(v) * np.log1p(np.abs(v))
Ys = symlog(Y); ymu, ysd = Ys.mean(0), Ys.std(0) + 1e-8
Yn = (Ys - ymu) / ysd
xmu, xsd = X.mean(0), X.std(0) + 1e-8
Xn = (X - xmu) / xsd

EMBD = 128
kA, kC = jax.random.split(jax.random.PRNGKey(A.seed))
# SAME fresh-init encoder + cost pool as the Phase-1b probe (seed 0) so the
# cost_head (trained on the probe's X) applies to fresh encodings we compute here.
agent = MicroPPOAgent(vocab_size=512, embd_dim=EMBD, num_layers=4, num_heads=4,
                      hidden_dim=256, num_vertices=len(jaxpr.eqns),
                      value_dims=(128, 128), key=kA, max_substeps=16, policy="palimpsa")
cost_pool_query = jax.random.normal(kC, (EMBD,)) * 0.02
cost_head = MLP(EMBD, NUM_REWARDS, (128, 128), key=jax.random.split(kC)[0])

# train cost_head (encoder frozen) on the precomputed contexts
TIDX = [REWARD_INDEX[c] for c in CHANNELS]
def huber(head, xb, yb):
    p = jax.vmap(head)(xb)[:, TIDX]
    return jnp.mean(optax.huber_loss(p, yb, delta=1.0))
opt = optax.adam(1e-3); ostate = opt.init(eqx.filter(cost_head, eqx.is_array))
Xtr = jnp.asarray(Xn); Ytr = jnp.asarray(Yn)
@eqx.filter_jit
def tstep(h, o, xb, yb):
    l, g = eqx.filter_value_and_grad(huber)(h, xb, yb)
    u, o = opt.update(g, o, eqx.filter(h, eqx.is_array)); return eqx.apply_updates(h, u), o, l
for e in range(A.train_epochs):
    cost_head, ostate, l = tstep(cost_head, ostate, Xtr, Ytr)
    if e % 1000 == 0 or e == A.train_epochs - 1:
        print(f"[plan] cost_head train ep{e} huber={float(l):.4f}", flush=True)

@eqx.filter_jit
def _encode(tokens_j, eqn_ids_j):
    enc_x, tok_mask = agent.encode_tokens(tokens_j, key=jax.random.PRNGKey(0), eqn_ids=eqn_ids_j)
    sc = (enc_x @ cost_pool_query) / jnp.sqrt(jnp.float32(EMBD))
    sc = jnp.where(tok_mask, sc, -1e9)
    at = jax.nn.softmax(sc, axis=-1)
    return jnp.sum(at[:, None] * enc_x, axis=0)

def order_tokens(order_ids):
    """Tokens+eqn_ids for a (partial or full) order via the env _callback path."""
    order = jnp.asarray(list(order_ids), dtype=jnp.int32)
    specs = jnp.full((len(order_ids), MAX_RULES, 3), -1, dtype=jnp.int32).at[:, :, 2].set(0)
    tokens, eqn_ids, _ = _callback(env.config, env.args, env.consts,
                                   order, specs, len(order_ids), *ev)
    return np.asarray(tokens), np.asarray(eqn_ids)

def cost_value(order_ids):
    """Scalarized search value (higher=better) from the cost model for a (partial
    or full) order. costs negated, cosine positive."""
    if len(order_ids) == 0:
        return 0.0
    tok, eqn = order_tokens(order_ids)
    ctx = np.asarray(_encode(jnp.asarray(tok), jnp.asarray(eqn)))
    ctxn = (ctx - xmu) / xsd
    pn = np.asarray(jax.vmap(cost_head)(jnp.asarray(ctxn[None])))[0][TIDX]  # norm symlog
    sl = pn * ysd + ymu
    raw = np.sign(sl) * np.expm1(np.abs(sl))  # [lat, peak, flops, cos]
    lat, peak, flops, cos = raw
    return float(-0.06 * lat - 0.06 * peak - 0.06 * flops + 1.0 * cos), raw

# --------------------------------------------------------------- Gumbel AZ move
def plan_order(rng):
    """Build a full order move-by-move via 1-ply Gumbel-AZ + sequential halving,
    cost-model leaf value at each candidate's resulting partial order."""
    graph, tg = copy_g(GRAPH0), copy_g(TG0)
    chosen = []
    while True:
        legal = [i for i in legal_set(graph) if i not in chosen]
        if not legal:
            break
        if len(legal) == 1:
            v = legal[0]
        else:
            # PRIOR: softmax over -Markowitz (low degree preferred)
            mk = markowitz_degrees(graph, tg, legal)
            logits = np.array([-float(mk[i]) for i in legal], dtype=np.float64)
            logits -= logits.max()
            # Gumbel top-m candidates
            m = min(A.n_candidates, len(legal))
            g = rng.gumbel(size=len(legal))
            topm = np.argsort(-(logits + g))[:m]
            cand = [legal[t] for t in topm]
            # SEQUENTIAL HALVING over the m candidates using cost-model value of
            # the resulting partial order (1-step lookahead: chosen+[v]).
            survivors = list(cand)
            phases = max(1, int(math.ceil(math.log2(max(m, 2)))))
            while len(survivors) > 1:
                scored = []
                for v in survivors:
                    val, _ = cost_value(chosen + [v])
                    # combine prior (gumbel-perturbed logit) + value (Gumbel-AZ
                    # completed-Q form: g + logit + sigma(qhat))
                    li = logits[legal.index(v)]
                    gi = g[legal.index(v)]
                    scored.append((gi + li + val, v))
                scored.sort(reverse=True)
                keep = max(1, len(survivors) // 2)
                survivors = [v for _, v in scored[:keep]]
            v = survivors[0]
        chosen.append(v)
        _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
    return chosen

# --------------------------------------------------------------- baseline orders
def reverse_order():
    return list(reversed(VALID))
def forward_order():
    return list(VALID)
def markowitz_greedy():
    graph, tg = copy_g(GRAPH0), copy_g(TG0); chosen = []
    while True:
        legal = [i for i in legal_set(graph) if i not in chosen]
        if not legal: break
        mk = markowitz_degrees(graph, tg, legal)
        v = min(legal, key=lambda i: mk[i])   # greedy min-degree
        chosen.append(v)
        _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
    return chosen

# --------------------------------------------------------------- measure (acid test)
def measure(order_ids, label):
    order = jnp.asarray(list(order_ids), dtype=jnp.int32)
    specs = jnp.full((len(order_ids), MAX_RULES, 3), -1, dtype=jnp.int32).at[:, :, 2].set(0)
    rs = {}
    _callback(env.config, env.args, env.consts, order, specs, len(order_ids), *ev, raw_sink=rs)
    lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
    lat = float(np.mean(lat_s)) if lat_s else float("nan")
    peak = float(rs.get("xla_peak_memory", float("nan")))
    flops = float(rs.get("flops", float("nan")))
    cos_pp = rs.get("cosine_sim_per_point", [])
    cos = float(np.mean(cos_pp)) if cos_pp else float("nan")
    print(f"[MEASURE] {label:16s} order={list(order_ids)} "
          f"lat_us={lat/1e3:.2f} peak_MB={peak/1e6:.2f} flops={flops:.4g} cos={cos:+.3f}",
          flush=True)
    return dict(order=list(map(int, order_ids)), lat_us=lat/1e3, peak_MB=peak/1e6, flops=flops, cos=cos)

# --------------------------------------------------------------- run
rng = np.random.default_rng(A.seed)
print(f"\n[plan] === GUMBEL-AZ PLANNING (m={A.n_candidates} candidates, "
      f"sim_budget={A.sim_budget}, {A.n_plan_restarts} restarts, seq-halving) ===", flush=True)
plans = []
for r in range(A.n_plan_restarts):
    po = plan_order(np.random.default_rng(A.seed + r))
    pv, _ = cost_value(po)
    plans.append((pv, po))
    print(f"[plan] restart {r}: cost-model value={pv:+.4f} order={po}", flush=True)
plans.sort(reverse=True)
planned = plans[0][1]
print(f"[plan] BEST planned order (by cost-model value): {planned}", flush=True)

# cost-model-predicted vs measured for the planned order
_, pred_raw = cost_value(planned)
print(f"[plan] planned cost-model PREDICTED (raw): lat_us={pred_raw[0]/1e3:.2f} "
      f"peak_MB={pred_raw[1]/1e6:.2f} flops={pred_raw[2]:.4g} cos={pred_raw[3]:+.3f}", flush=True)

print("\n[plan] === ACID TEST: MEASURED real grad cost ===", flush=True)
results = {}
results["planned"] = measure(planned, "planned(Gumbel)")
results["reverse"] = measure(reverse_order(), "reverse-mode")
results["markowitz"] = measure(markowitz_greedy(), "markowitz-greedy")
results["forward"] = measure(forward_order(), "forward")
results["_planned_predicted"] = {"lat_us": pred_raw[0]/1e3, "peak_MB": pred_raw[1]/1e6,
                                 "flops": float(pred_raw[2]), "cos": float(pred_raw[3])}
results["_node"] = node

print("\n[plan] ===== BENCHMARK (measured) =====", flush=True)
for k2 in ["planned", "reverse", "markowitz", "forward"]:
    rr = results[k2]
    print(f"[TABLE] {k2:16s} lat_us={rr['lat_us']:8.2f} peak_MB={rr['peak_MB']:9.2f} "
          f"flops={rr['flops']:.4g} cos={rr['cos']:+.3f}", flush=True)
# verdict: is planned competitive/better on measured latency+peak+flops?
def score(r): return (r["lat_us"], r["peak_MB"], r["flops"])
pl = results["planned"]
print("\n[VERDICT]", flush=True)
for base in ["reverse", "markowitz", "forward"]:
    b = results[base]
    better_lat = pl["lat_us"] <= b["lat_us"]
    better_peak = pl["peak_MB"] <= b["peak_MB"]
    better_flops = pl["flops"] <= b["flops"]
    print(f"[VERDICT] planned vs {base}: lat {'<=' if better_lat else '>'} "
          f"({pl['lat_us']:.1f} vs {b['lat_us']:.1f}), "
          f"peak {'<=' if better_peak else '>'} ({pl['peak_MB']:.1f} vs {b['peak_MB']:.1f}), "
          f"flops {'<=' if better_flops else '>'} ({pl['flops']:.3g} vs {b['flops']:.3g})", flush=True)

json.dump(results, open(os.path.expanduser("~/dsnn/gumbel_planner_results.json"), "w"), indent=2, default=float)
print("[plan] DONE", flush=True)
