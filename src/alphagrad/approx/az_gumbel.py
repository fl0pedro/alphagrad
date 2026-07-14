"""Sampled + Gumbel AlphaZero over vertex elimination, built from the PPO components.

CLEAN implementation (ignores autoscheduler_loop's surrogate-Gumbel search and mu0):
  * KNOWN dynamics: symbolic graph elimination (_eliminate_vertex) — no learned model,
    no mctx (elimination is not jittable). Python MCTS.
  * Value + prior = the PPO MicroPPOAgent (palimpsa encoder + PointerVertexPolicy +
    per-channel value head). Leaf evaluation = value net; simulations NEVER measure.
  * GUMBEL (Danihelka 2022): root Gumbel-top-m without replacement over the prior
    logits, SEQUENTIAL HALVING of the simulation budget, action chosen by
    argmax(g + logits + sigma(q)), policy trained by CE to the COMPLETED-Q improved
    target softmax(logits + sigma(completed_q)) over the legal set.
  * SAMPLED (Hubert 2021, pragmatic): with ALPHAGRAD_GAZ_MICRO=1 each root candidate
    is (vertex, micro-action) with the micro drawn from an explicit-range proposal
    (quant/diag/compress, sparse); the search Q decides which survive. The learned
    heads stay vertex-level in v1 (micro-head learning = follow-up).
  * Real measurements ONLY at episode terminals (budget = --total-measurements).
  * Objective identical to the E2 campaign: equal-weight z-scored
    {cosine_sim, latency_ns, xla_peak_memory}; flops unrewarded.

Run:  python -m alphagrad.approx.az_gumbel --seed 7 --total-measurements 150
"""
import os, sys, json, time, math, argparse

os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--task", default="VmappedNeuralNetwork")   # e.g. VmappedViT
ap.add_argument("--dataset", default="mnist")
ap.add_argument("--total-measurements", type=int, default=150)
ap.add_argument("--n-candidates", type=int, default=8)      # Gumbel top-m at the root
ap.add_argument("--num-simulations", type=int, default=16)  # sequential-halving budget
ap.add_argument("--rollout-depth", type=int, default=3)     # greedy descent depth per sim
ap.add_argument("--train-epochs", type=int, default=4)
ap.add_argument("--replay-episodes", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--nn-hidden", type=int, default=256)
ap.add_argument("--ndata", type=int, default=5)
ap.add_argument("--latency-inner-reps", type=int, default=50)
ap.add_argument("--out", default=os.path.expanduser("~/dsnn/az_gumbel_out"))
ap.add_argument("--wandb", action="store_true")
ap.add_argument("--wandb-name", default="az_gumbel")
ap.add_argument("--wandb-project", default="dsnn-jac-gpu")
ap.add_argument("--wandb-entity", default="dll-streetview")
A = ap.parse_args()
os.environ["ALPHAGRAD_NN_HIDDEN"] = str(A.nn_hidden)
# propagate task to the measure-server child (inherits our env at spawn)
os.environ["ALPHAGRAD_MS_TASK"] = A.task
os.environ["ALPHAGRAD_MS_DATASET"] = A.dataset
os.environ.setdefault("ALPHAGRAD_MS_NDATA", str(A.ndata))
os.environ.setdefault("ALPHAGRAD_MS_INNER_REPS", str(A.latency_inner_reps))

import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import optax

from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn)
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent, NUM_REWARDS
from graphax.core import _build_graph, _prune_graph, _eliminate_vertex
from graphax.sparse.micro_actions import COMPRESS_KINDS

# ---------------------------------------------------------------- env (measure_worker pattern)
TASK = A.task; DSET = A.dataset
LOSS = scalar_loss_fn(get_fn(TASK))
ARGN = infer_argnums(TASK)
k0 = jax.random.PRNGKey(0); ak, ek = jax.random.split(k0)
xs = get_args(TASK, ak, dataset=DSET)
gen = data_gen(TASK, dataset=DSET, dataset_size=128)
closed = jax.make_jaxpr(LOSS)(*xs)
env = VertexEliminationEnv.from_jaxpr(
    closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
    sparse=(os.environ.get("ALPHAGRAD_SPARSE", "0") == "1"),
    cmp_type="latency", mem_type="peak_memory", exec_on_gpu=True, measure_latency=True,
    num_data_points=A.ndata, reps_per_point=1, percentile_keep=0.60,
    slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0, measure_grad=True,
    latency_inner_reps=A.latency_inner_reps, latency_timer="perf_counter")
ev = generate_eval_samples(env, ek, A.ndata)
env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)
jaxpr = closed.jaxpr
VALID = list(np.asarray(env.valid_vertices, dtype=np.int32)); NV = len(VALID)
_eg, GRAPH0, TG0, VO = _build_graph(jaxpr, xs, closed.literals, ARGN)
_prune_graph(GRAPH0, TG0, jaxpr, ARGN)
def copy_g(g): return {kk: dict(vv) for kk, vv in g.items()}
def outvar(i): return jaxpr.eqns[i - 1].outvars[0]
def legal_set(graph): return [i for i in VALID if outvar(i) in graph]

# ---------------------------------------------------------------- objective (matches campaign)
CH = ["latency_ns", "xla_peak_memory", "flops", "cosine_sim"]
TIDX = np.array([REWARD_INDEX[c] for c in CH], dtype=np.int32)
W4 = np.array([-1.0, -1.0, 0.0, 1.0], dtype=np.float64)   # equal-weight, flops dropped
MU4 = np.zeros(4); SD4 = np.ones(4)
_measured = []
def _refresh_norm():
    global MU4, SD4
    Y = np.asarray(_measured, dtype=np.float64)
    if Y.ndim == 2 and len(Y) >= 8:
        MU4, SD4 = Y.mean(0), Y.std(0) + 1e-8
def scalarize(raw4):
    r = np.asarray(raw4, dtype=np.float64)
    return float(np.sum(W4 * (r - MU4) / SD4))

# ---------------------------------------------------------------- state <-> tokens
QD = [d for d in os.environ.get(
    "ALPHAGRAD_QUANT_ALLOWED", "int8,int16,bfloat16,float16").split(",") if d]
GAZ_MICRO = os.environ.get("ALPHAGRAD_GAZ_MICRO", "0") == "1"
MICRO_P = float(os.environ.get("ALPHAGRAD_GAZ_MICRO_P", "0.25"))
MAXAX = int(os.environ.get("ALPHAGRAD_GAZ_MAX_AX", "2"))
FACS = [int(x) for x in os.environ.get("ALPHAGRAD_GAZ_FACTORS", "2,3,4").split(",")]

def rand_micro(rng, force=False):
    """SAMPLED proposal for a candidate's micro-action (explicit ranges).
    force=True always returns a non-None micro (root candidate variants)."""
    if not GAZ_MICRO or (not force and rng.random() >= MICRO_P):
        return None
    op = rng.integers(3)
    if op == 0:
        return ("q", QD[int(rng.integers(len(QD)))])
    if op == 1:
        i = int(rng.integers(MAXAX)); j = (i + 1) % max(MAXAX, 2)
        return ("d", i, j, FACS[int(rng.integers(len(FACS)))])
    return ("c", int(rng.integers(MAXAX)), int(rng.integers(len(COMPRESS_KINDS))))

def micro_str(m):
    if m is None: return []
    if m[0] == "q": return ["quant('%s')" % m[1]]
    if m[0] == "d": return ["diag(%d,%d,%d)" % (m[1], m[2], m[3])]
    return ["compress('%s',%d)" % (COMPRESS_KINDS[m[2]], m[1])]

def _rules_of(m):
    """micro tuple -> graphax rule objects (for the append-only micro tokens)."""
    from graphax.sparse.micro_actions import Diag as _D, Compress as _C, Quant as _Q
    if m is None: return ()
    if m[0] == "q": return (_Q(dtype=m[1]),)
    if m[0] == "d": return (_D(i=int(m[1]), j=int(m[2]), factor=int(m[3])),)
    return (_C(axes=(int(m[1]),), kind=COMPRESS_KINDS[m[2]]),)

def seq_of(state):
    """state = list of (action_idx, micro-or-None) -> build_order_specs seq."""
    return [(int(a), micro_str(m)) for a, m in state]

# ---- tokenization optimizations (ported from the surrogate-search + PPO work) ----
#  * ALPHAGRAD_GAZ_AOJ=1 (default): append-only jaxpr tokens (ProposerTokenizer —
#    tokens-only template, persistent vocab, NO Python re-trace per child, ~17x
#    shorter stream). Pure-order states only; micro states fall back to _callback.
#  * fixed-cap PADDING (ALPHAGRAD_GAZ_TOKCAP): every stream padded to one length
#    (pad id 0 = masked via tok>0 downstream) -> the encoder compiles ONCE.
AOJ = os.environ.get("ALPHAGRAD_GAZ_AOJ", "1") == "1"
_TK = None
if AOJ:
    from alphagrad.approx.append_only_jaxpr import ProposerTokenizer
    _TK = ProposerTokenizer(LOSS, xs, ARGN, VALID)
    _full_len = len(_TK.order_token_ids(list(range(NV))[::-1]))
    # MICRO-AWARE cap: micro blocks add ~10-15 tokens/vertex (compress-all on
    # NN-256 measures 490 vs 268 order-only) — size the cap from the measured
    # worst single-micro-per-vertex stream, else micro-heavy states TRUNCATE.
    _cap_src = _full_len
    if GAZ_MICRO:
        from graphax.sparse.micro_actions import Compress as _Cw
        _worst = [(a, (_Cw(axes=(0,), kind="mean"),)) for a in list(range(NV))[::-1]]
        _cap_src = max(_cap_src, len(_TK.order_token_ids_micro(_worst)))
    TOKCAP = int(os.environ.get("ALPHAGRAD_GAZ_TOKCAP", str(int(_cap_src * 1.3) + 8)))
    print(f"[gaz] AOJ tokenizer ON: base={len(_TK.base_token_ids)} "
          f"full_order={_full_len} worst_micro={_cap_src} TOKCAP={TOKCAP}", flush=True)
else:
    TOKCAP = int(os.environ.get("ALPHAGRAD_GAZ_TOKCAP", "16384"))

def _padcap(ids):
    if len(ids) > TOKCAP:
        print(f"[gaz][WARN] token stream {len(ids)} > TOKCAP {TOKCAP} — "
              f"TRUNCATED (value net blind past the cap)", flush=True)
    a = np.zeros(TOKCAP, dtype=np.int32)
    n = min(len(ids), TOKCAP)
    a[:n] = np.asarray(ids[:n], dtype=np.int32)
    return a

_tok_cache = {}
_tok_miss = [0]
# AOJ=0 tokenizes via _callback(init=True), which LEAKS an XLA executable per
# distinct (order,specs) — the documented measure-actor leak (~16MB/state on ViT,
# fills a 24GB card by ~decision 25). Periodic clear_caches()+gc bounds it. Only
# needed on the _callback path (AOJ off); the append-only tokenizer never leaks.
_TOK_CLEAR_EVERY = int(os.environ.get("ALPHAGRAD_GAZ_TOK_CLEAR_EVERY",
                                      "0" if AOJ else "40"))
def tokens_of(state):
    key = tuple((int(a), tuple(m) if m else None) for a, m in state)
    hit = _tok_cache.get(key)
    if hit is not None:
        return hit
    if AOJ and all(m is None for _, m in state):
        ids = _TK.order_token_ids([int(a) for a, _ in state])
        tok = _padcap(ids)
        eqn = np.zeros_like(tok)
    elif AOJ:
        # MICRO-bearing state: append-only micro blocks (env 0dfe7ef pattern) —
        # still pure Python, no re-trace.
        ids = _TK.order_token_ids_micro(
            [(int(a), _rules_of(m)) for a, m in state])
        tok = _padcap(ids)
        eqn = np.zeros_like(tok)
    else:
        order, specs, _ = build_order_specs(seq_of(state), env)
        tok_r, eqn_r, _ = _callback(env.config, env.args, env.consts,
                                    jnp.asarray(order), jnp.asarray(specs),
                                    len(order), *ev, init=True)
        tok = _padcap(np.asarray(tok_r)); eqn = _padcap(np.asarray(eqn_r))
    # cache on the HOST (numpy). Caching jnp (device) arrays leaked ~16MB of GPU
    # per distinct ViT state (the array + its referenced XLA buffer stayed live),
    # growing linearly with the search until _net_fwd_b OOMs. Host arrays are
    # ~64KB each and moved to device only transiently inside _net_fwd_b.
    out = (np.asarray(tok, dtype=np.int32), np.asarray(eqn, dtype=np.int32))
    if len(_tok_cache) < 8192:
        _tok_cache[key] = out
    _tok_miss[0] += 1
    if _TOK_CLEAR_EVERY and _tok_miss[0] % _TOK_CLEAR_EVERY == 0:
        import gc
        jax.clear_caches(); gc.collect()   # free leaked _callback executables
    return out

_MS = os.environ.get("ALPHAGRAD_MEASURE_SERVER", "0") == "1"
_ms_client = None
def measure(state):
    """Terminal REAL measurement -> [lat, xla_peak, flops, cos] or None.
    ALPHAGRAD_MEASURE_SERVER=1 -> measure in an isolated SUBPROCESS (a CUDA-
    poisoned kernel kills the child, not this training process)."""
    global _ms_client
    if _MS:
        if _ms_client is None:
            from alphagrad.approx.measure_client import MeasureClient
            _ms_client = MeasureClient()
        return _ms_client.measure_seq(seq_of(state))
    try:
        order, specs, _ = build_order_specs(seq_of(state), env)
        rs = {}
        _callback(env.config, env.args, env.consts, jnp.asarray(order),
                  jnp.asarray(specs), len(order), *ev, raw_sink=rs)
    except BaseException:
        try:
            jax.clear_caches(); import gc; gc.collect()
        except Exception:
            pass
        return None
    lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
    lat = float(np.mean(lat_s)) if lat_s else float("nan")
    peak = float(rs.get("xla_peak_memory", float("nan")))
    flops = float(rs.get("flops", float("nan")))
    cos_pp = rs.get("cosine_sim_per_point", [])
    cos = float(np.mean(cos_pp)) if cos_pp else float("nan")
    r = np.array([lat, peak, flops, cos])
    return r if np.all(np.isfinite(r)) else None

# ---------------------------------------------------------------- agent (PPO components)
EMBD = 128
kA = jax.random.PRNGKey(A.seed)
agent = MicroPPOAgent(vocab_size=512, embd_dim=EMBD, num_layers=4, num_heads=4,
                      hidden_dim=256, num_vertices=len(jaxpr.eqns),
                      value_dims=(128, 128), key=kA, max_substeps=1,
                      policy="palimpsa")
opt = optax.adam(A.lr)
opt_state = opt.init(eqx.filter(agent, eqx.is_array))

@eqx.filter_jit
def _net_fwd_b(agent, toks, eqns):
    """BATCHED fixed-shape forward: vmap over a stack of TOKCAP-padded states.
    One GPU dispatch evaluates every candidate/rollout frontier at once; only a
    handful of batch sizes occur (<= n_candidates) so compiles are bounded."""
    def one(tok, eqn):
        enc_x, tm = agent.encode_tokens(tok, key=jax.random.PRNGKey(0), eqn_ids=eqn)
        vlog, _ = agent.vertex_policy(enc_x, tm)
        v10 = agent.value_from_encoding(enc_x, tm)
        return vlog, v10
    return jax.vmap(one)(toks, eqns)

_eval_cache = {}                       # state-key -> (vlog, v10); cleared on net update
def _skey(state):
    return tuple((int(a), tuple(m) if m else None) for a, m in state)

def batch_eval(states):
    """Evaluate a LIST of states in one batched forward, with per-state caching
    (within a search, siblings share prefixes and halving re-visits survivors)."""
    miss = [i for i, s in enumerate(states) if _skey(s) not in _eval_cache]
    if miss:
        # chunk the batched forward: long ViT contexts (16k tokens) OOM the
        # encoder if the whole candidate batch is vmapped at once.
        chunk = int(os.environ.get("ALPHAGRAD_GAZ_EVAL_CHUNK", "0")) or len(miss)
        for c0 in range(0, len(miss), chunk):
            grp = miss[c0:c0 + chunk]
            toks = jnp.stack([tokens_of(states[i])[0] for i in grp])
            eqns = jnp.stack([tokens_of(states[i])[1] for i in grp])
            vlogs, v10s = _net_fwd_b(agent, toks, eqns)
            vlogs = np.asarray(vlogs, dtype=np.float64)
            v10s = np.asarray(v10s, dtype=np.float64)
            for k, i in enumerate(grp):
                _eval_cache[_skey(states[i])] = (vlogs[k], v10s[k])
    return [_eval_cache[_skey(s)] for s in states]

def net_eval(agent, state, legal):
    """(prior logits over legal action idxs, scalar value in z-space)."""
    vlog, v10 = batch_eval([state])[0]
    la = np.array([VALID.index(v) for v in legal], dtype=np.int32)
    logits = vlog[la] - vlog[la].max()
    vz = float(np.sum(W4 * v10[TIDX]))     # value head trained in z-space -> scalarize
    return logits, vz, la

# ---------------------------------------------------------------- known dynamics
def step_state(graph, tg, state, vertex, micro):
    _eliminate_vertex(vertex, jaxpr, graph, tg, VO, count_ops=False, transforms=())
    state.append((VALID.index(vertex), micro))

def lockstep_rollout_values(entries, depth):
    """Advance ALL rollouts in LOCKSTEP: one batched forward per depth level
    (instead of one dispatch per rollout per step). entries = [(state, graph,
    tg)] already stepped into each candidate. Returns per-entry scalar value in
    z-space, or None where the rollout reached terminal in-search."""
    live = [{"st": list(s), "g": g, "t": t, "done": False} for s, g, t in entries]
    for _ in range(depth):
        idx = []
        for i, e in enumerate(live):
            if not e["done"] and not legal_set(e["g"]):
                e["done"] = True
            if not e["done"]:
                idx.append(i)
        if not idx:
            break
        evs = batch_eval([live[i]["st"] for i in idx])     # ONE dispatch for the level
        for k, i in enumerate(idx):
            vlog = evs[k][0]
            legal = legal_set(live[i]["g"])
            la = [VALID.index(v) for v in legal]
            v = legal[int(np.argmax(vlog[la]))]
            step_state(live[i]["g"], live[i]["t"], live[i]["st"], v, None)
    fin = [i for i, e in enumerate(live) if legal_set(e["g"])]
    evs = batch_eval([live[i]["st"] for i in fin]) if fin else []
    vmap_ = {i: evs[k] for k, i in enumerate(fin)}
    out = []
    for i, e in enumerate(live):
        if i in vmap_:
            out.append(float(np.sum(W4 * vmap_[i][1][TIDX])))
        else:
            out.append(None)                               # terminal reached in-search
    return out

def sigma(q, cs=float(os.environ.get("ALPHAGRAD_GAZ_CSCALE", "1.0"))):
    q = np.asarray(q, dtype=np.float64)
    s = q.std() + 1e-8
    return cs * (q - q.mean()) / s

# ---------------------------------------------------------------- Gumbel root search
def gumbel_search(state, graph, tg, rng):
    legal = legal_set(graph)
    logits, v_root, la = net_eval(agent, state, legal)
    m = min(A.n_candidates, len(legal))
    g = rng.gumbel(size=len(legal))
    order_idx = np.argsort(-(logits + g))[:m]
    # SAMPLED micro variants as first-class root candidates: each Gumbel-selected
    # vertex enters as (v, None) plus K_MICRO sampled micro variants sharing the
    # vertex's (g + logit); the search Q decides which variant survives halving.
    K_MICRO = int(os.environ.get("ALPHAGRAD_GAZ_K_MICRO", "2"))
    cands = []
    for ci in order_idx:
        variants = [None]
        if GAZ_MICRO:
            variants += [rand_micro(rng, force=True) for _ in range(K_MICRO)]
        for micro in variants:
            cands.append({"li": int(ci), "v": legal[int(ci)], "micro": micro,
                          "q": [], "g": float(g[int(ci)]),
                          "logit": float(logits[int(ci)])})
    # SEQUENTIAL HALVING with PROGRESSIVE DEEPENING (deterministic dynamics +
    # deterministic value net => repeated sims of a candidate are IDENTICAL, so
    # instead of re-simulating, each halving phase gives the SURVIVORS a 2x
    # deeper lockstep rollout — the budget buys more accurate Q, not duplicates).
    surv = list(cands)
    depth = A.rollout_depth
    while True:
        entries = []
        for c in surv:
            st2 = list(state); g2, t2 = copy_g(graph), copy_g(tg)
            step_state(g2, t2, st2, c["v"], c["micro"])
            entries.append((st2, g2, t2))
        vals = lockstep_rollout_values(entries, depth)     # batched per level
        for c, vz in zip(surv, vals):
            c["q"].append(v_root if vz is None else vz)
        if len(surv) <= 1:
            break
        qbar = np.array([np.mean(c["q"]) for c in surv])
        sc = np.array([c["g"] + c["logit"] for c in surv]) + sigma(qbar)
        keep = np.argsort(-sc)[:max(1, len(surv) // 2)]
        surv = [surv[i] for i in keep]
        depth = min(depth * 2, NV)                         # deepen survivors
    chosen = surv[0]
    # completed-Q improved policy target over the FULL legal set. A vertex's Q =
    # MAX over its evaluated micro variants (the vertex is as good as its best
    # variant); unvisited vertices complete with v_root (Danihelka completed-Q).
    comp_q = np.full(len(legal), v_root, dtype=np.float64)
    _seen = set()
    for c in cands:
        if c["q"]:
            qv = float(np.mean(c["q"]))
            if c["li"] not in _seen or qv > comp_q[c["li"]]:
                comp_q[c["li"]] = qv
            _seen.add(c["li"])
    pi = logits + sigma(comp_q)
    pi = np.exp(pi - pi.max()); pi = pi / pi.sum()
    return chosen, pi, la, legal

# ---------------------------------------------------------------- training
def loss_fn(agent, toks, eqns, la_pad, la_mask, pi_pad, vtgt, vmask):
    def per(tok, eqn, la, lam, pi, vt, vm):
        enc_x, tm = agent.encode_tokens(tok, key=jax.random.PRNGKey(0), eqn_ids=eqn)
        vlog, _ = agent.vertex_policy(enc_x, tm)
        lg = vlog[la]
        lg = jnp.where(lam > 0.5, lg, -1e9)
        logp = jax.nn.log_softmax(lg)
        ce = -jnp.sum(jnp.where(lam > 0.5, pi * logp, 0.0))
        v10 = agent.value_from_encoding(enc_x, tm)
        vl = jnp.sum(vm * (v10[jnp.asarray(TIDX)] - vt) ** 2)
        return ce + 0.5 * vl
    return jnp.mean(jax.vmap(per)(toks, eqns, la_pad, la_mask, pi_pad, vtgt, vmask))

@eqx.filter_jit
def train_step(agent, opt_state, batch):
    l, gr = eqx.filter_value_and_grad(loss_fn)(agent, *batch)
    up, opt_state = opt.update(gr, opt_state, eqx.filter(agent, eqx.is_array))
    return eqx.apply_updates(agent, up), opt_state, l

# ---------------------------------------------------------------- main loop
os.makedirs(A.out, exist_ok=True)
rng = np.random.default_rng(A.seed)
replay = []                                    # episodes of (tok, eqn, la, pi) + ztarget
_solutions = []                                # every (raw4, state, n) — re-ranked under current norm
best = {"scalar": -1e18, "raw": None, "state": None, "at": 0}
n_meas = 0; ep = 0
wb = None
if A.wandb:
    try:
        import wandb
        wb = wandb.init(project=A.wandb_project, entity=A.wandb_entity,
                        name=A.wandb_name, config=vars(A))
    except Exception:
        wb = None
print(f"[gaz] NV={NV} budget={A.total_measurements} m={A.n_candidates} "
      f"sims={A.num_simulations} depth={A.rollout_depth} micro={GAZ_MICRO}", flush=True)

MAXTOK = 0
while n_meas < A.total_measurements:
    ep += 1
    state = []; graph, tg = copy_g(GRAPH0), copy_g(TG0)
    steps = []
    _memlog = os.environ.get("ALPHAGRAD_GAZ_MEMLOG", "0") == "1"
    _dstep = 0
    while True:
        legal = legal_set(graph)
        if not legal:
            break
        chosen, pi, la, legal = gumbel_search(state, graph, tg, rng)
        tok, eqn = tokens_of(state)
        steps.append({"tok": np.asarray(tok), "eqn": np.asarray(eqn),
                      "la": la.copy(), "pi": pi.copy()})
        step_state(graph, tg, state, chosen["v"], chosen["micro"])
        _dstep += 1
        if _memlog and _dstep % 5 == 0:
            try:
                ms = jax.devices()[0].memory_stats()
                print(f"[gaz][mem] ep={ep} decision={_dstep}/{len(legal)+_dstep} "
                      f"peak={ms.get('peak_bytes_in_use',0)/1e9:.2f}GB "
                      f"curr={ms.get('bytes_in_use',0)/1e9:.2f}GB "
                      f"tokcache={len(_tok_cache)} evalcache={len(_eval_cache)}", flush=True)
            except Exception:
                pass
    raw = measure(state)
    n_meas += 1
    if raw is None:
        print(f"[gaz] ep={ep} measure FAILED (n={n_meas})", flush=True)
        continue
    _measured.append(raw); _refresh_norm()
    # best-tracker: scores from different normalizer epochs are NOT comparable
    # (pre-warmup raw-scale ~-1e6 vs z-scored O(1) let a worse order overwrite a
    # better one at n=8). Keep every (raw, state) and re-argmax under the CURRENT
    # normalizer each episode.
    _solutions.append((raw.copy(),
                       [(int(a), list(m) if m else None) for a, m in state], n_meas))
    bi = int(np.argmax([scalarize(r) for r, _, _ in _solutions]))
    braw, bstate, bat = _solutions[bi]
    best.update(scalar=scalarize(braw), raw=braw.tolist(), state=bstate, at=bat)
    for s in steps:
        s["raw4"] = raw.copy()                 # store RAW; z-normalize at TRAIN time
    replay.append(steps)
    replay = replay[-A.replay_episodes:]
    # ---- train on the replay (only once the normalizer is warm: raw latency/peak
    # magnitudes ~1e4-1e6 would explode the value loss before MU/SD are set) ----
    flat = [s for epi in replay for s in epi]
    if len(flat) >= 8 and len(_measured) >= 8:
        # fixed shapes (tok already TOKCAP-padded; legal set <= NV) -> train_step
        # compiles ONCE instead of re-jitting as episode lengths vary
        MAXTOK = TOKCAP
        MAXLA = NV
        def pad(x, n, v=0):
            return np.pad(x, (0, n - len(x)), constant_values=v)
        toks = jnp.asarray([pad(s["tok"], MAXTOK) for s in flat])
        eqns = jnp.asarray([pad(s["eqn"], MAXTOK) for s in flat])
        la_p = jnp.asarray([pad(s["la"], MAXLA) for s in flat])
        la_m = jnp.asarray([pad(np.ones(len(s["la"])), MAXLA) for s in flat])
        pi_p = jnp.asarray([pad(s["pi"], MAXLA) for s in flat])
        vt = jnp.asarray([(s["raw4"] - MU4) / SD4 for s in flat])  # CURRENT normalizer
        vm = jnp.asarray(np.broadcast_to(np.abs(W4) > 0, (len(flat), 4)).astype(np.float32))
        idx = rng.permutation(len(flat))[:64]
        batch = tuple(x[jnp.asarray(idx)] for x in (toks, eqns, la_p, la_m, pi_p, vt, vm))
        for _ in range(A.train_epochs):
            agent, opt_state, L = train_step(agent, opt_state, batch)
        L = float(L)
        _eval_cache.clear()          # net changed -> cached (prior, value) stale
    else:
        L = float("nan")
    b = best["raw"]
    print(f"[gaz] ep={ep} n={n_meas}/{A.total_measurements} this(lat={raw[0]/1e3:.1f}us "
          f"cos={raw[3]:+.3f}) best(lat={b[0]/1e3:.1f}us peak={b[1]/1e6:.2f}MB "
          f"cos={b[3]:+.4f} at={best['at']}) loss={L:.4f}", flush=True)
    # persist EVERY measured solution each episode (crash-safe): the per-run
    # PARETO FRONT over {lat, xla_peak, cos} is computed offline from this —
    # any weighting re-analyzable without re-running.
    json.dump({"best": best, "n_measured": n_meas, "config": vars(A),
               "micro": GAZ_MICRO,
               "solutions": [{"n": nn, "raw": r.tolist(), "state": st}
                             for r, st, nn in _solutions]},
              open(os.path.join(A.out, "gaz_result.json"), "w"),
              indent=1, default=float)
    if wb is not None:
        try:
            wb.log({"ep": ep, "n_meas": n_meas, "loss": L, "best_scalar": best["scalar"],
                    "best_lat_us": b[0] / 1e3, "best_cos": b[3], "this_lat_us": raw[0] / 1e3})
        except Exception:
            pass

json.dump({"best": best, "n_measured": n_meas, "config": vars(A),
           "micro": GAZ_MICRO}, open(os.path.join(A.out, "gaz_result.json"), "w"),
          indent=2, default=float)
print(f"[gaz] DONE best={best['raw']} at n={best['at']}", flush=True)
if wb is not None:
    try: wb.finish()
    except Exception: pass
