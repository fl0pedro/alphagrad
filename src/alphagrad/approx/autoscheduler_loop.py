"""Phase-3 autoscheduler CLOSED LOOP (NN-256, pure-order): model proposes,
hardware decides, retrain online. Reuses the MVP planner components.

Loop (N_ROUNDS): (1) PROPOSE a pool of ~POOL diverse candidate orders via the
Gumbel planner (multi-restart + temperature) ranked by the current cost head;
(2) SELECT top-K by predicted scalar cost (+ fidelity>=thresh hook); (3) MEASURE
top-K for real (grad-measure, xla_peak, inner-reps 50); (4) RETRAIN the cost head
with a PAIRWISE RANKING loss (Offline-RaM style) on the growing buffer; (5) LOG
per round: ranking Spearman(pred,meas) on a held-out set, best-real-found,
predicted-vs-measured GAP on the round's top-K (the over-optimization alarm),
buffer size. Track best real order across rounds.

Cost model = cost head on the FROZEN fresh-init encoder (Phase 1b, seed 0), so a
candidate order's pooled-encoder context is comparable to the seed dataset. Buffer
seeded from ~/dsnn/cost_head_probe_out_v2 (N~800: X=ctx, Y=measured 4-tuple).
"""
import os, sys, math, argparse, json, time
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_PEAK_MEMORY_SYNC", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("ALPHAGRAD_QUANT_ALLOWED", "int8,int16,bfloat16,float16")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Ray measure-actor backend (--measure-backend ray): disable uv-run working-dir
# upload (~/dsnn 8.5GB > Ray 512MB cap) + keep Ray from clobbering our manual
# per-actor CUDA_VISIBLE_DEVICES pinning. Must precede any import ray.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
# INCREMENTAL causal palimpsa encode for the AZ policy prior. The
# append-only token stream (static graph prefix | order ; micro-actions)
# is ONLY append-only when graphax's state-tokenizer is on, so force it.
os.environ.setdefault("GRAPHAX_STATE_TOKENS", "1")
# Default ON; ALPHAGRAD_AZ_INCREMENTAL=0 restores legacy full-reencode.
AZ_INCREMENTAL = os.environ.get("ALPHAGRAD_AZ_INCREMENTAL", "1") == "1"

import numpy as np
import jax, jax.numpy as jnp, equinox as eqx
import optax
from scipy.stats import spearmanr
# Force true-float32 matmul accumulation so the INCREMENTAL (unbatched
# matvec) and FULL (batched GEMM) encode paths match bit-for-bit on
# Blackwell tensor-cores (proven in incremental_encoder_test.py).
if AZ_INCREMENTAL:
    jax.config.update("jax_default_matmul_precision", "highest")
from alphagrad.approx import incremental_encoder as _ie
from alphagrad.approx.append_only_jaxpr import ProposerTokenizer
# GENUINE append-only jaxpr proposer state (ALPHAGRAD_APPEND_ONLY_JAXPR=1):
# the search STATE is the literal per-action jaxpr stream (base value jaxpr +
# reserved Jacobian outputs + eliminate/COMPRESS/QUANT blocks), tokenized by
# our append_only_jaxpr tokenizer and fed to the CAUSAL palimpsa incremental
# encoder -- recompile-free by construction (pure-Python emission, 0 jit(_loss)).
APPEND_ONLY_JAXPR = os.environ.get("ALPHAGRAD_APPEND_ONLY_JAXPR", "0") == "1"
# Append-only jaxpr streams are VARIABLE length (per order) -> pad every
# batched token stream to a FIXED cap (pad id 0; encoder masks tok>0, CE masks
# illegal actions) so np.stack is uniform AND jit sees a STABLE shape (no
# recompiles from a varying batch-max). Cap >> longest full-order stream.
AOJ_TOK_CAP = int(os.environ.get("ALPHAGRAD_AOJ_TOK_CAP", "1024"))
def _pad_tok_1d(a, cap=None):
    import numpy as _np
    cap = AOJ_TOK_CAP if cap is None else cap
    a = _np.asarray(a).reshape(-1)
    if a.shape[0] >= cap:
        return a[:cap].astype(_np.int32)
    return _np.pad(a.astype(_np.int32), (0, cap - a.shape[0]))

ap = argparse.ArgumentParser()
ap.add_argument("--nn-hidden", type=int, default=256)
ap.add_argument("--seed-dataset", default=os.path.expanduser("~/dsnn/cost_head_probe_out_v2/dataset_partial.npz"))
ap.add_argument("--ndata", type=int, default=3)
ap.add_argument("--latency-inner-reps", type=int, default=50)
ap.add_argument("--rounds", type=int, default=15)
ap.add_argument("--pool", type=int, default=64)      # candidate orders proposed per round
ap.add_argument("--topk", type=int, default=10)      # measured per round
ap.add_argument("--n-candidates", type=int, default=8)   # Gumbel top-m at each move
ap.add_argument("--retrain-epochs", type=int, default=1500)
ap.add_argument("--micro-budget", type=int, default=0)   # 0 = pure order; 1-2 = with headroom
ap.add_argument("--full-search", action="store_true")    # per-step cost-head lookahead (slow)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.expanduser("~/dsnn/autoscheduler_out"))
ap.add_argument("--total-measurements", type=int, default=0)  # if >0, rounds = ceil(total/topk)
ap.add_argument("--measure-workers", type=int, default=0)     # >0 = parallel measure over N GPUs
ap.add_argument("--measure-gpu-base", type=int, default=1)    # first GPU idx for measure workers
ap.add_argument("--measure-backend", default="serial", choices=["serial", "subprocess", "ray"])
ap.add_argument("--num-measure-actors", type=int, default=7)
ap.add_argument("--clear-caches-every", type=int, default=1)  # jax.clear_caches every N rounds
ap.add_argument("--policy-prior", default="markowitz", choices=["markowitz", "trained"])
ap.add_argument("--policy-epochs", type=int, default=200)   # policy CE epochs/round
ap.add_argument("--az-sigma-scale", type=float, default=1.0)  # sigma(Q) scale (Danihelka c_scale)
ap.add_argument("--wandb", action="store_true")           # live per-round logging
ap.add_argument("--wandb-project", default="dsnn-jac-gpu")
ap.add_argument("--wandb-entity", default="dll-streetview")
ap.add_argument("--wandb-name", default="")
A = ap.parse_args()
os.environ["ALPHAGRAD_NN_HIDDEN"] = str(A.nn_hidden)
os.makedirs(A.out, exist_ok=True)
node = os.environ.get("SLURMD_NODENAME", "?")
if A.total_measurements > 0:
    import math as _m
    A.rounds = int(_m.ceil(A.total_measurements / max(A.topk, 1)))
print(f"[loop] node={node} device={jax.devices()[0]} nn_hidden={A.nn_hidden} "
      f"rounds={A.rounds} pool={A.pool} topk={A.topk} micro_budget={A.micro_budget} "
      f"measure_workers={A.measure_workers} total_meas={A.total_measurements}", flush=True)
_wb_run = None
if A.wandb:
    try:
        import wandb
        _wb_name = A.wandb_name or f"autoloop_micro{A.micro_budget}"
        _wb_run = wandb.init(project=A.wandb_project, entity=A.wandb_entity,
                             name=_wb_name, config=vars(A))
        print(f"[loop] wandb: {_wb_run.get_url()}", flush=True)
    except Exception as _we:
        print(f"[loop] wandb init failed ({_we}); continuing local-only", flush=True)
        _wb_run = None

from alphagrad.approx.env import VertexEliminationEnv, _callback, REWARD_INDEX, MAX_TOKENS
from alphagrad.approx.common.examples import get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn
from alphagrad.approx.common.eval_samples import generate_eval_samples
from alphagrad.approx.verify_pareto_solution import build_order_specs
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent, NUM_REWARDS, _scale_micro_policy_heads, _rezero_encoder_rel_gates
from alphagrad.transformer import MLP
from graphax.core import _build_graph, _prune_graph, _eliminate_vertex
MAX_RULES = 16
CHANNELS = ["latency_ns", "peak_memory", "flops", "cosine_sim"]  # measured tuple order
TIDX = [REWARD_INDEX[c] for c in CHANNELS]

# --------------------------------------------------------------- env + jaxpr graph
LOSS = scalar_loss_fn(get_fn("VmappedNeuralNetwork")); ARGN = infer_argnums("VmappedNeuralNetwork")
k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
xs = get_args("VmappedNeuralNetwork", ak, dataset="mnist")
gen = data_gen("VmappedNeuralNetwork", dataset="mnist", dataset_size=128)
closed = jax.make_jaxpr(LOSS)(*xs)
env = VertexEliminationEnv.from_jaxpr(
    closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
    cmp_type="latency", mem_type="peak_memory", exec_on_gpu=True, measure_latency=True,
    num_data_points=A.ndata, reps_per_point=1, percentile_keep=0.60,
    slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0, measure_grad=True,
    latency_inner_reps=A.latency_inner_reps, latency_timer="perf_counter")
ev = generate_eval_samples(env, ek, A.ndata)
env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)
VALID = list(np.asarray(env.valid_vertices, dtype=np.int32)); NV = len(VALID)
jaxpr = closed.jaxpr
_eg, GRAPH0, TG0, VO = _build_graph(jaxpr, xs, closed.literals, ARGN); _prune_graph(GRAPH0, TG0, jaxpr, ARGN)
print(f"[loop] NV={NV} valid={VALID}", flush=True)
_AOJ_TK = None
if APPEND_ONLY_JAXPR:
    _AOJ_TK = ProposerTokenizer(LOSS, xs, ARGN, VALID)
    print(f"[loop] APPEND-ONLY JAXPR proposer ON: base_tokens="
          f"{len(_AOJ_TK.base_token_ids)} vocab0={_AOJ_TK.vocab.size}", flush=True)
    _demo_ids = _AOJ_TK.order_tokens([VALID.index(v) for v in reversed(VALID)][:3])
    print("[loop] APPEND-ONLY JAXPR stream (base + 3 blocks, strings):", flush=True)
    print("   " + " ".join(_demo_ids[:120]), flush=True)
def copy_g(g): return {kk: dict(vv) for kk, vv in g.items()}
def outvar(i): return jaxpr.eqns[i - 1].outvars[0]
def legal_set(graph): return [i for i in VALID if outvar(i) in graph]
def markowitz(graph, tg, cand):
    return {i: max(1, len(tg.get(outvar(i), {}))) * max(1, len(graph.get(outvar(i), {}))) for i in cand}

# --------------------------------------------------------------- cost model (frozen encoder + head)
EMBD = 128
kA, kC = jax.random.split(jax.random.PRNGKey(A.seed))
agent = MicroPPOAgent(vocab_size=512, embd_dim=EMBD, num_layers=4, num_heads=4,
                      hidden_dim=256, num_vertices=len(jaxpr.eqns), value_dims=(128, 128),
                      key=kA, max_substeps=16, policy="palimpsa")
cost_pool_query = jax.random.normal(kC, (EMBD,)) * 0.02
cost_head = MLP(EMBD, NUM_REWARDS, (128, 128), key=jax.random.split(kC)[0])

# ----- AlphaZero POLICY (fresh-init, separate from the cost encoder) -----
import optax as _optax
_USE_TRAINED_PRIOR = (A.policy_prior == "trained")
if _USE_TRAINED_PRIOR:
    _kp = jax.random.split(jax.random.PRNGKey(A.seed + 777), 1)[0]
    policy_agent = MicroPPOAgent(vocab_size=512, embd_dim=EMBD, num_layers=4,
                                 num_heads=4, hidden_dim=256,
                                 num_vertices=len(jaxpr.eqns), value_dims=(128, 128),
                                 key=_kp, max_substeps=16, policy="palimpsa")
    # palimpsa init details (match the PPO policy): near-uniform initial
    # policy via head-init-scale 0.1 + re-zeroed encoder rel-gates.
    policy_agent = _scale_micro_policy_heads(policy_agent, 0.1)
    policy_agent = _rezero_encoder_rel_gates(policy_agent)
    _pol_opt = _optax.adam(3e-4)
    _pol_ostate = _pol_opt.init(eqx.filter(policy_agent, eqx.is_array))
    print("[loop] POLICY PRIOR = TRAINED (fresh-init AlphaZero policy agent)", flush=True)
else:
    policy_agent = None
    print("[loop] POLICY PRIOR = markowitz (heuristic)", flush=True)

@eqx.filter_jit
def _policy_vertex_logits(agent, tokens_j, eqn_ids_j):
    """(num_vertices,) prior logits from the policy over the partial-order
    state tokens. Indexing matches the vertex/action-idx space."""
    enc_x, tok_mask = agent.encode_tokens(tokens_j, key=jax.random.PRNGKey(0), eqn_ids=eqn_ids_j)
    vlogits, _vctx = agent.vertex_policy(enc_x, tok_mask)
    return vlogits  # (num_vertices,)


@eqx.filter_jit
def _vertex_policy_from_enc(agent, enc_x, tok_mask):
    """Pointer logits from a PRE-COMPUTED (S,E) enc_x + (S,) mask. Used by the
    incremental path (enc_x built causally, eqn_ids=None)."""
    vlogits, _vctx = agent.vertex_policy(enc_x, tok_mask)
    return vlogits


# --- INCREMENTAL encoder state: static-graph prefix is encoded ONCE at init.
#     eqn_ids=None throughout the AZ policy encode (relational gate OFF =>
#     exactly causal, so appending the order-delta tokens reproduces the full
#     re-encode). The vertex pointer head cross-attends over the whole enc_x
#     sequence, but enc_x is reproduced bit-for-bit, so the logits match.
_IE_STATIC = {"state": None, "n": 0}

def _init_incremental_prefix():
    """Encode the empty-order (static-graph) token prefix once, snapshot the
    per-layer palimpsa carry. Returns nothing; fills _IE_STATIC."""
    if APPEND_ONLY_JAXPR:
        # append-only jaxpr BASE = value jaxpr + reserved Jacobian outputs,
        # encoded ONCE into the causal palimpsa carry.
        real = list(_AOJ_TK.base_token_ids)
    else:
        tok, _eqn, _ = _callback(env.config, env.args, env.consts,
                                 np.zeros((0,), np.int32),
                                 np.zeros((0, MAX_RULES, 3), np.int32), 0, *ev, init=True)
        tok = np.asarray(tok)
        real = [int(t) for t in tok[tok > 0]]
    st = _ie.init_state(policy_agent)
    _ie.extend(policy_agent, st, [int(t) for t in real])
    _IE_STATIC["state"] = st
    _IE_STATIC["n"] = int(len(real))
    print(f"[loop] incremental static-prefix real_tokens={int(len(real))}", flush=True)


def _incremental_full_tokens(chosen_a):
    """Full REAL (non-pad) token stream for the given order via _callback."""
    if len(chosen_a) == 0:
        tok, eqn, _ = _callback(env.config, env.args, env.consts,
                                np.zeros((0,), np.int32),
                                np.zeros((0, MAX_RULES, 3), np.int32), 0, *ev, init=True)
    else:
        seq = _seq_from_order(chosen_a, 0)
        order, specs, _ = build_order_specs(seq, env)
        tok, eqn, _ = _callback(env.config, env.args, env.consts,
                                jnp.asarray(order), jnp.asarray(specs), len(order), *ev, init=True)
    tok = np.asarray(tok); eqn = np.asarray(eqn)
    return tok, eqn


def policy_prior_logits(chosen_a, legal_vids):
    """Policy prior logits over the LEGAL vertices for the current partial
    order (chosen_a = 0-based action idxs already eliminated). Returns a np
    array aligned to legal_vids (1-based vertex ids). Also returns the
    (tokens, eqn_ids, legal_action_idxs) needed to reconstruct the state for
    the AZ training target.

    INCREMENTAL path (default): extend the static-prefix palimpsa carry by the
    order-delta REAL tokens (O(delta)), build enc_x causally, run the pointer
    head with eqn_ids=None. FULL path (ALPHAGRAD_AZ_INCREMENTAL=0): re-encode.
    The returned (tok, eqn) are always the FULL _callback stream so the AZ
    training target is byte-identical to legacy."""
    _dbg = os.environ.get("ALPHAGRAD_IE_DEBUG", "0") == "1"
    if _dbg:
        print(f"[ppl] START ndecided={len(chosen_a)} nlegal={len(legal_vids)}", flush=True)
    if APPEND_ONLY_JAXPR:
        # GENUINE append-only jaxpr state: base (already in the static carry) +
        # one literal jaxpr block per eliminated vertex. The delta appended to
        # the causal encoder is exactly the concatenation of the chosen blocks.
        base_ids, block_deltas = _AOJ_TK.block_token_ids(chosen_a)
        _full_ids = list(base_ids)
        for _d in block_deltas:
            _full_ids += _d
        tok = np.asarray(_full_ids, dtype=np.int32)
        eqn = np.zeros_like(tok)  # eqn_ids unused on the causal (eqn_ids=None) path
    else:
        tok, eqn = _incremental_full_tokens(chosen_a)
    if _dbg:
        print(f"[ppl] tokens done nreal={int((tok>0).sum())}", flush=True)
    if AZ_INCREMENTAL:
        real = tok[tok > 0] if not APPEND_ONLY_JAXPR else tok
        st = _IE_STATIC["state"].copy()
        # append only the NEW tokens past the static (base) prefix.
        delta = [int(t) for t in real[_IE_STATIC["n"]:]]
        if _dbg:
            print(f"[ppl] delta_tokens={len(delta)} calling extend", flush=True)
        _ie.extend(policy_agent, st, delta)
        enc_x = _ie.enc_x(st)                        # (S_real, E)
        tok_mask = jnp.ones((enc_x.shape[0],), dtype=bool)
        if _dbg:
            print(f"[ppl] extend done enc_x={enc_x.shape} -> vertex_policy", flush=True)
            import time as _tt; _vt = _tt.time()
        vlog = np.asarray(_vertex_policy_from_enc(policy_agent, enc_x, tok_mask))
        if _dbg:
            print(f"[ppl] vertex_policy done in {_tt.time()-_vt:.3f}s", flush=True)
    else:
        vlog = np.asarray(_policy_vertex_logits(policy_agent, jnp.asarray(tok), jnp.asarray(eqn)))
    legal_aidx = [VALID.index(v) for v in legal_vids]
    prior = vlog[legal_aidx]  # logits over legal, aligned to legal_vids
    return prior, (np.asarray(tok), np.asarray(eqn), legal_aidx)

def az_improved_target(prior_logits, child_Q):
    """Gumbel-AZ improved policy over legal actions (Danihelka 2022):
    improved = softmax(prior_logits + sigma(completed_Q)), where completed_Q
    are the cost-model values of the legal children (higher=better) and sigma
    is a monotone scale. We standardize Q (so the 1e6-scale scalar doesnt
    saturate softmax) then scale by --az-sigma-scale. Uses the child VALUES
    (cost-model evaluations), NOT visit counts."""
    q = np.asarray(child_Q, dtype=np.float64)
    qz = (q - q.mean()) / (q.std() + 1e-8)
    logits = np.asarray(prior_logits, dtype=np.float64) + A.az_sigma_scale * qz
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()  # improved policy over legal actions

_az_targets = []  # list of (tokens, eqn_ids, legal_aidx, improved_target) per round

@eqx.filter_jit
def _encode(tokens_j, eqn_ids_j):
    enc_x, tm = agent.encode_tokens(tokens_j, key=jax.random.PRNGKey(0), eqn_ids=eqn_ids_j)
    sc = jnp.where(tm, (enc_x @ cost_pool_query) / jnp.sqrt(jnp.float32(EMBD)), -1e9)
    at = jax.nn.softmax(sc, axis=-1)
    return jnp.sum(at[:, None] * enc_x, axis=0)

def _seq_from_order(order_ids, micro_budget, rng=None):
    """order_ids (0-based action idx) -> seq [(v, ops)]; quant micro-actions when
    micro_budget>0 (headroom), else empty (pure order). DTYPES ARE DETERMINISTIC
    per order (seeded from the order tuple) so the SAME candidate is ENCODED
    (ranking) and MEASURED (real) with identical ops -> a micro-action order is
    one well-defined candidate. rng ignored for the dtype (signature compat)."""
    QD = ["int8", "int16", "bfloat16", "float16"]
    seq = []
    _dr = (np.random.default_rng(abs(hash(tuple(int(x) for x in order_ids))) % (2 ** 31))
           if micro_budget > 0 else None)
    for v in order_ids:
        ops = ([f"quant('{QD[int(_dr.integers(0, len(QD)))]}')" for _ in range(micro_budget)]
               if micro_budget > 0 else [])
        seq.append((int(v), ops))
    return seq

_ctx_fallback = np.zeros((EMBD,), dtype=np.float64)
_ctx_cache = {}
def order_ctx(order_ids, micro_budget=0, rng=None):
    _ck = (tuple(int(x) for x in order_ids), int(micro_budget))
    _cv = _ctx_cache.get(_ck)
    if _cv is not None:
        return _cv
    if APPEND_ONLY_JAXPR:
        # cost-head input = the append-only jaxpr token stream for this order.
        ids = _AOJ_TK.order_token_ids(list(order_ids))
        _tok = jnp.asarray(ids, dtype=jnp.int32)
        _eqn = jnp.zeros_like(_tok)
        _res = np.asarray(_encode(_tok, _eqn))
        _ctx_cache[_ck] = _res
        return _res
    seq = _seq_from_order(order_ids, micro_budget, rng)
    order, specs, _ = build_order_specs(seq, env)
    for _attempt in range(2):
        try:
            tok, eqn, _ = _callback(env.config, env.args, env.consts,
                                    jnp.asarray(order), jnp.asarray(specs), len(order), *ev, init=True)
            _res = np.asarray(_encode(jnp.asarray(tok), jnp.asarray(eqn)))
            _ctx_cache[_ck] = _res
            return _res  # (EMBD,)
        except RuntimeError as _e:
            if ("mem-gate" in str(_e) or "RESOURCE" in str(_e)) and _attempt == 0:
                jax.clear_caches(); import gc; gc.collect()
                continue
            # double-failure -> return a neutral fallback ctx so the proposer
            # survives (that candidate just gets a poor/neutral score).
            return _ctx_fallback.copy()

def measure_order(order_ids, micro_budget=0, rng=None):
    seq = _seq_from_order(order_ids, micro_budget, rng)
    order, specs, _ = build_order_specs(seq, env)
    rs = {}
    try:
        _callback(env.config, env.args, env.consts, jnp.asarray(order),
                  jnp.asarray(specs), len(order), *ev, raw_sink=rs)
    except RuntimeError as _e:
        # measure-GPU leak / mem-gate skip -> clear caches + retry ONCE on the
        # freed device, else return NaN (dropped by the caller).
        if "mem-gate" in str(_e) or "RESOURCE" in str(_e):
            jax.clear_caches(); import gc; gc.collect()
            try:
                _callback(env.config, env.args, env.consts, jnp.asarray(order),
                          jnp.asarray(specs), len(order), *ev, raw_sink=rs)
            except Exception:
                return np.array([np.nan] * 4, dtype=np.float64)
        else:
            return np.array([np.nan] * 4, dtype=np.float64)
    lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
    lat = float(np.mean(lat_s)) if lat_s else np.nan
    peak = float(rs.get("xla_peak_memory", np.nan))
    flops = float(rs.get("flops", np.nan))
    cos_pp = rs.get("cosine_sim_per_point", [])
    cos = float(np.mean(cos_pp)) if cos_pp else np.nan
    return np.array([lat, peak, flops, cos], dtype=np.float64)

def symlog(v): return np.sign(v) * np.log1p(np.abs(v))

# scalarized cost (higher=better): costs negated, cosine positive. This is the
# search/selection objective and the ranking-loss target.
def scalarize(raw4):
    lat, peak, flops, cos = raw4
    return float(-0.06 * lat - 0.06 * peak - 0.06 * flops + 1.0 * cos)

# --------------------------------------------------------------- buffer (X=ctx, Y=measured, S=scalar)
d = np.load(A.seed_dataset, allow_pickle=True)
bufX = list(np.asarray(d["X"], np.float64))            # pooled encoder ctx
bufY = list(np.asarray(d["Y"], np.float64))            # measured 4-tuple
bufS = [scalarize(y) for y in bufY]                    # scalarized cost (rank target)
print(f"[loop] seeded buffer N={len(bufX)} from {A.seed_dataset}", flush=True)

# held-out set for ranking-Spearman eval: a fixed slice of the seed buffer
rng0 = np.random.default_rng(A.seed)
_ho = rng0.permutation(len(bufX))[:150]
HOX = np.array([bufX[i] for i in _ho]); HOS = np.array([bufS[i] for i in _ho])

# --------------------------------------------------------------- ranking-loss trainer
def predict_scalar_batch(head, Xn):
    """Predicted SCALARIZED score for a batch of standardized ctx (higher=better).
    Invert the head's symlog-normalized 4-tuple prediction -> raw -> scalarize."""
    pn = jax.vmap(head)(Xn)[:, TIDX]                    # (B,4) normalized symlog
    return pn  # keep normalized; scalarize in numpy after standardization inverse

def _pred_scalar_np(head, X, xmu, xsd, ymu4, ysd4):
    # ymu4/ysd4 are the symlog-target stats for the 4 measured channels in the
    # [lat,peak,flops,cos] order — the SAME order head(x)[:, TIDX] returns.
    Xn = (X - xmu) / xsd
    pn = np.asarray(jax.vmap(head)(jnp.asarray(Xn)))[:, TIDX]  # (B,4) normalized symlog
    sl = pn * ysd4 + ymu4
    raw = np.sign(sl) * np.expm1(np.abs(sl))
    return np.array([scalarize(r) for r in raw])

def train_ranking(head, X, S, epochs):
    """PAIRWISE logistic ranking loss (Offline-RaM style): for random pairs,
    P(score_i > score_j) = sigmoid(f_i - f_j); target = 1 if S_i>S_j. f = the
    head's scalarized prediction (differentiable via the normalized 4-tuple).
    Standardize ctx + symlog-target space consistently."""
    X = np.asarray(X); S = np.asarray(S)
    xmu, xsd = X.mean(0), X.std(0) + 1e-8
    # bufY is the (N,4) measured tuple [lat,peak,flops,cos]; its symlog stats are
    # in that 4-order (NOT the 10-channel REWARD_INDEX order). head(x)[:, TIDX]
    # returns the same [lat,peak,flops,cos] order, so no TIDX re-index on ymu.
    Ys = symlog(np.array(bufY)); ymu, ysd = Ys.mean(0), Ys.std(0) + 1e-8  # shape (4,)
    Xn = jnp.asarray((X - xmu) / xsd)
    Sj = jnp.asarray(S)
    W = jnp.asarray(np.array([-0.06, -0.06, -0.06, 1.0]))  # scalarize weights over [lat,peak,flops,cos] raw
    ymu_j, ysd_j = jnp.asarray(ymu), jnp.asarray(ysd)
    def scalar_pred(head, xn):
        pn = jax.vmap(head)(xn)[:, jnp.asarray(TIDX)]     # (B,4) norm symlog
        sl = pn * ysd_j + ymu_j
        raw = jnp.sign(sl) * jnp.expm1(jnp.abs(sl))       # (B,4) raw
        return jnp.sum(raw * W, axis=-1)                  # (B,) scalar higher=better
    def loss(head, xn, s, ki, kj):
        f = scalar_pred(head, xn)                          # (B,)
        fi, fj = f[ki], f[kj]; si, sj = s[ki], s[kj]
        tgt = (si > sj).astype(jnp.float32)                # 1 if i better
        # standardize the scalar diff so sigmoid isn't saturated by 1e6 costs
        dz = (fi - fj) / (jnp.std(f) + 1e-6)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(dz, tgt))
    opt = optax.adam(1e-3); ostate = opt.init(eqx.filter(head, eqx.is_array))
    B = Xn.shape[0]
    rng = np.random.default_rng(A.seed + len(bufX))
    @eqx.filter_jit
    def step(head, ostate, xn, s, ki, kj):
        l, g = eqx.filter_value_and_grad(loss)(head, xn, s, ki, kj)
        u, ostate = opt.update(g, ostate, eqx.filter(head, eqx.is_array))
        return eqx.apply_updates(head, u), ostate, l
    for e in range(epochs):
        ki = jnp.asarray(rng.integers(0, B, min(512, B)))
        kj = jnp.asarray(rng.integers(0, B, min(512, B)))
        head, ostate, l = step(head, ostate, Xn, Sj, ki, kj)
    return head, (xmu, xsd, ymu, ysd)

# --------------------------------------------------------------- Gumbel proposer (pool)
def _build_gumbel_order(temp, r):
    """Construct ONE complete order greedily via a Gumbel-perturbed
    -Markowitz/temp prior (NO cost-head lookahead -> fast). Returns 0-based
    action-idx list."""
    graph, tg = copy_g(GRAPH0), copy_g(TG0)
    chosen_v, chosen_a = [], []
    while True:
        legal = [i for i in legal_set(graph) if i not in chosen_v]
        if not legal: break
        if len(legal) == 1:
            v = legal[0]
        else:
            mk = markowitz(graph, tg, legal)
            logits = np.array([-float(mk[i]) / max(temp, 1e-3) for i in legal])
            g = r.gumbel(size=len(legal))
            v = legal[int(np.argmax(logits + g))]
        chosen_v.append(v); chosen_a.append(VALID.index(v))
        _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
    return chosen_a

def propose_pool_fast(head, stds, n_pool, micro_budget, temp, rng):
    """FAST proposer (Offline-RaM style): sample n_pool diverse COMPLETE orders
    from the Gumbel-perturbed prior, then rank each ONCE by the cost head (1
    encode/order instead of per-step lookahead). Returns [(order_ids, pred)]."""
    xmu, xsd, ymu, ysd = stds
    pool = {}
    tries = 0
    while len(pool) < n_pool and tries < n_pool * 6:
        tries += 1
        r = np.random.default_rng(rng.integers(1 << 30))
        aidx = tuple(_build_gumbel_order(temp, r))
        if aidx in pool:
            continue
        ctx = order_ctx(list(aidx), micro_budget)
        ps = _pred_scalar_np(head, ctx[None], xmu, xsd, ymu, ysd)[0]
        pool[aidx] = ps
    return [(list(k), v) for k, v in pool.items()]

def propose_pool(head, stds, n_pool, micro_budget, temp, rng):
    """Propose n_pool diverse full orders via Gumbel search with temperature, each
    scored by the current cost head. Returns list of (order_ids, pred_scalar)."""
    xmu, xsd, ymu, ysd = stds
    pool = {}
    tries = 0
    while len(pool) < n_pool and tries < n_pool * 4:
        tries += 1
        graph, tg = copy_g(GRAPH0), copy_g(TG0)
        chosen_v = []        # 1-based vertex ids (for graphax elimination)
        chosen_a = []        # 0-based action idx (for build_order_specs / encode)
        r = np.random.default_rng(rng.integers(1 << 30))
        while True:
            legal = [i for i in legal_set(graph) if i not in chosen_v]
            if not legal: break
            if len(legal) == 1:
                v = legal[0]
            else:
                # PRIOR: trained policy logits (AlphaZero) or -Markowitz.
                if _USE_TRAINED_PRIOR:
                    prior_np, _state_info = policy_prior_logits(chosen_a, legal)
                    logits = np.asarray(prior_np, dtype=np.float64) / max(temp, 1e-3)
                else:
                    mk = markowitz(graph, tg, legal)
                    logits = np.array([-float(mk[i]) / max(temp, 1e-3) for i in legal])
                    _state_info = None
                logits = logits - logits.max()
                m = min(A.n_candidates, len(legal))
                g = r.gumbel(size=len(legal))
                cand = [legal[t] for t in np.argsort(-(logits + g))[:m]]
                # Evaluate EACH candidate child's cost-model Q ONCE (needed for
                # both the sequential-halving pick AND the AZ improved target).
                # VMAP child scoring: build the (K, EMBD) context batch (one
                # order_ctx encode per candidate -- distinct tokens) then run
                # the cost head ONCE over the whole batch (jax.vmap inside
                # _pred_scalar_np) instead of K serial (1,EMBD) head calls.
                _cand_ctx = np.stack([order_ctx(chosen_a + [VALID.index(vv)], micro_budget)
                                      for vv in cand], axis=0)   # (K, EMBD)
                _cand_q = _pred_scalar_np(head, _cand_ctx, xmu, xsd, ymu, ysd)  # (K,)
                qmap = {vv: float(_cand_q[i]) for i, vv in enumerate(cand)}
                # AZ improved-policy TARGET over the LEGAL set (trained prior
                # only): improved = softmax(prior + sigma(completed_Q)); legal
                # actions not in `cand` keep completed_Q = mean(evaluated Q)
                # (Danihelka completed-Q: unvisited -> the value estimate).
                if _USE_TRAINED_PRIOR and _state_info is not None:
                    _qmean = float(np.mean(list(qmap.values())))
                    child_Q = np.array([qmap.get(lv, _qmean) for lv in legal], dtype=np.float64)
                    _improved = az_improved_target(prior_np, child_Q)
                    _tok, _eqn, _legal_aidx = _state_info
                    _az_targets.append((_tok, _eqn, _legal_aidx, _improved))
                # sequential halving pick using the (already-computed) Q.
                surv = list(cand)
                while len(surv) > 1:
                    surv.sort(key=lambda vv: -qmap[vv])
                    surv = surv[:max(1, len(surv) // 2)]
                v = surv[0]
            chosen_v.append(v); chosen_a.append(VALID.index(v))
            _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
        aidx = tuple(chosen_a)
        if aidx not in pool:
            ctx = order_ctx(list(aidx), micro_budget)
            ps = _pred_scalar_np(head, ctx[None], xmu, xsd, ymu, ysd)[0]
            pool[aidx] = ps
    return [(list(k), v) for k, v in pool.items()]

# --------------------------------------------------------------- THE LOOP
import subprocess as _sp, tempfile as _tf
_WORKER = os.path.join(os.path.dirname(__file__), "measure_worker.py")
def measure_topk_parallel(order_list, micro_budget, seed, n_workers, gpu_base):
    """Measure a list of orders across n_workers GPU subprocesses (each pinned
    to a distinct GPU, builds env once, measures its chunk, EXITS -> frees the
    measure-GPU leak). Returns list of [lat,peak,flops,cos] or None, aligned."""
    import numpy as _np
    n = len(order_list)
    if n == 0:
        return []
    chunks = [order_list[i::n_workers] for i in range(n_workers)]
    idx_of = [list(range(i, n, n_workers)) for i in range(n_workers)]
    procs, outs = [], []
    _d = _tf.mkdtemp(prefix="mw_")
    for w, ch in enumerate(chunks):
        if not ch:
            procs.append(None); outs.append(None); continue
        jf = os.path.join(_d, f"job{w}.json"); of = os.path.join(_d, f"out{w}.json")
        json.dump({"orders": [list(map(int, o)) for o in ch],
                   "micro_budget": int(micro_budget), "seed": int(seed)}, open(jf, "w"))
        env2 = dict(os.environ)
        env2["CUDA_VISIBLE_DEVICES"] = str(gpu_base + w)
        p = _sp.Popen([sys.executable, _WORKER, jf, of, str(A.nn_hidden),
                       str(A.ndata), str(A.latency_inner_reps)], env=env2,
                      stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        procs.append(p); outs.append(of)
    results = [None] * n
    for w, p in enumerate(procs):
        if p is None:
            continue
        p.wait()
        try:
            rr = json.load(open(outs[w]))["results"]
            for k2, ridx in enumerate(idx_of[w]):
                results[ridx] = (_np.array(rr[k2], dtype=_np.float64)
                                 if rr[k2] is not None else None)
        except Exception:
            pass
    import shutil as _sh; _sh.rmtree(_d, ignore_errors=True)
    _ok = sum(1 for r in results if r is not None)
    print(f"[measure] parallel {_ok}/{n} measured across {sum(1 for c in chunks if c)} "
          f"GPU workers (base={gpu_base})", flush=True)
    return results

def train_policy(targets, epochs):
    """Train the fresh policy agent by cross-entropy to the AZ improved
    targets over legal actions. targets = list of (tokens, eqn_ids,
    legal_aidx, improved_prob). Returns (policy_ce, policy_entropy)."""
    global policy_agent, _pol_ostate
    if not targets:
        return float('nan'), float('nan')
    # group by legal-length so we can batch same-shaped states (the pointer
    # head returns all num_vertices logits; we gather legal per sample).
    import numpy as _np
    _AZ_MAX_STATES = 64
    if len(targets) > _AZ_MAX_STATES:
        _sel = _np.random.default_rng(A.seed + len(bufX)).choice(
            len(targets), _AZ_MAX_STATES, replace=False)
        targets = [targets[int(i)] for i in _sel]
    # pad each (possibly variable-length, append-only) token/eqn stream to a
    # FIXED cap before stacking -> uniform shape + stable jit shape.
    _cap = AOJ_TOK_CAP if APPEND_ONLY_JAXPR else int(max(len(_np.asarray(t[0]).reshape(-1)) for t in targets))
    toks = _np.stack([_pad_tok_1d(t[0], _cap) for t in targets])
    eqns = _np.stack([_pad_tok_1d(t[1], _cap) for t in targets])
    # legal masks + target over the full num_vertices space (0 on illegal).
    NVv = len(jaxpr.eqns)
    tgt_full = _np.zeros((len(targets), NVv), dtype=_np.float32)
    legal_mask = _np.zeros((len(targets), NVv), dtype=_np.float32)
    for i, (_, _, laidx, imp) in enumerate(targets):
        for k2, ai in enumerate(laidx):
            tgt_full[i, ai] = imp[k2]; legal_mask[i, ai] = 1.0
    toks_j = jnp.asarray(toks); eqns_j = jnp.asarray(eqns)
    tgt_j = jnp.asarray(tgt_full); mask_j = jnp.asarray(legal_mask)
    def _ce(agent, tj, ej, tg_, mk_):
        def per(t, e, tgt, mk):
            enc_x, tm = agent.encode_tokens(t, key=jax.random.PRNGKey(0), eqn_ids=e)
            vlog, _ = agent.vertex_policy(enc_x, tm)
            vlog = jnp.where(mk > 0.5, vlog, -1e9)
            logp = jax.nn.log_softmax(vlog)
            ce = -jnp.sum(tgt * logp)                  # CE over legal
            p = jnp.exp(logp) * mk
            ent = -jnp.sum(jnp.where(mk > 0.5, p * logp, 0.0))
            return ce, ent
        ce, ent = jax.vmap(per)(tj, ej, tg_, mk_)
        return jnp.mean(ce), jnp.mean(ent)
    def _loss(agent, tj, ej, tg_, mk_):
        ce, ent = _ce(agent, tj, ej, tg_, mk_); return ce
    @eqx.filter_jit
    def _step(agent, ost, tj, ej, tg_, mk_):
        l, gr = eqx.filter_value_and_grad(_loss)(agent, tj, ej, tg_, mk_)
        up, ost = _pol_opt.update(gr, ost, eqx.filter(agent, eqx.is_array))
        return eqx.apply_updates(agent, up), ost, l
    # MINIBATCH the states (each is a MAX_TOKENS encoder pass; vmapping ALL
    # of them at once OOMs — a full round has ~pool*|V| states). Chunk of
    # _POL_MB states per step.
    _POL_MB = 4
    _N = toks_j.shape[0]
    last_ce = float('nan')
    _rngp = np.random.default_rng(A.seed + len(bufX))
    for _e in range(epochs):
        _perm = _rngp.permutation(_N)
        for _i in range(0, _N, _POL_MB):
            _ix = _perm[_i:_i + _POL_MB]
            policy_agent, _pol_ostate, last_ce = _step(
                policy_agent, _pol_ostate,
                toks_j[_ix], eqns_j[_ix], tgt_j[_ix], mask_j[_ix])
    # eval CE/entropy on a small held slice (avoid the full-batch OOM).
    _ev_ix = jnp.asarray(np.arange(min(_N, 8)))
    _ce_v, _ent_v = _ce(policy_agent, toks_j[_ev_ix], eqns_j[_ev_ix],
                        tgt_j[_ev_ix], mask_j[_ev_ix])
    return float(_ce_v), float(_ent_v)

stds = None
# warm-start the head with a quick ranking fit on the seed buffer
cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)
# Encode the static-graph token prefix ONCE (incremental AZ policy prior).
if _USE_TRAINED_PRIOR and AZ_INCREMENTAL:
    _init_incremental_prefix()  # static-prefix carry snapshot
best_real = {"scalar": -1e18, "raw": None, "order": None, "round": -1}
# ---- Pareto-front archive (like ppo_ray, ParetoArchive) over the 4 measured
# channels. maximize-oriented vector = [-lat, -peak, -flops, +cos].
from alphagrad.approx.common.pareto_archive import ParetoArchive
_pareto = ParetoArchive(["neg_latency_ns", "neg_xla_peak", "neg_flops", "cosine_sim"],
                        [0, 1, 2, 3])
_pareto_path = os.path.join(A.out, "ppo_pareto_front.json")
def _pareto_vec(raw4):
    # raw4 = [lat, peak, flops, cos] -> maximize orientation.
    return [-float(raw4[0]), -float(raw4[1]), -float(raw4[2]), float(raw4[3])]
def _feed_pareto(order_ids, raw4, rnd):
    # drop sentinels / all-zero (failed measures) — a real measure has
    # nonzero lat/peak/flops.
    if raw4 is None or (not np.all(np.isfinite(raw4))):
        return
    if not np.any(np.asarray(raw4[:3]) != 0.0):
        return
    _pareto.add_many([(_pareto_vec(raw4), [int(v) for v in order_ids])], rnd)
# ---- RAY measure-actor pool (--measure-backend ray) -----------------------
# Each top-k candidate is measured in its OWN isolated Ray CpuApproximationActor
# (the SAME actor class PPO uses), pinned to its own GPU -> fair, contention-free
# per-measurement grad-measure distributed across GPUs {base..base+N-1}. The
# main-process trainer keeps GPU 0. Reuses env._callback inside each actor; the
# raw 4-tuple [lat, xla_peak, flops, cos] is recovered from the negated reward
# vector (byte-consistent with measure_order: peak/flops/cos exact, lat in noise).
_RAY_ACTORS = None
_RAY_EV = None
def _init_ray_measure():
    global _RAY_ACTORS, _RAY_EV
    import ray
    ray.init(num_cpus=A.num_measure_actors + 8, ignore_reinit_error=True,
             include_dashboard=False, _temp_dir=f"/tmp/rayaz{os.getpid()}")
    from alphagrad.approx.cpu_approx_actors import CpuApproximationActor
    # args_dict that _build_env_from_args reads -> builds the IDENTICAL grad
    # graph (grad_target_setup(measure_grad=True) == scalar_loss_fn, argnums
    # (2,3,4,5)) + measure knobs matching the loop's own env.
    args_dict = dict(
        example="VmappedNeuralNetwork", dataset="mnist", dataset_size=128,
        seed=A.seed, measure_grad=True, seed_vertices=False,
        cmp_type="latency", mem_type="peak_memory", exec_on_gpu=True,
        measure_latency=True, latency_samples=1, num_data_points=A.ndata,
        reps_per_point=1, percentile_keep=0.60,
        latency_inner_reps=A.latency_inner_reps, latency_warmup=0,
        latency_winsor=0.0, latency_timer="perf_counter", quant_once=False,
        slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0,
        intermediate_rewards=True, num_eval_samples=A.ndata,
        cost_pipeline_schedule="always_full", rewards="latency",
        num_cpu_workers=A.num_measure_actors, cpu_cores_per_actor=0,
    )
    base_env = {
        "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "ALPHAGRAD_NN_HIDDEN": str(A.nn_hidden),
        # cosine_sim rewarded -> actor computes the EXACT reference Jacobian
        # (else cos channel = 0). Only affects the (unused) scalar-reward weight.
        "ALPHAGRAD_REWARD_CHANNELS": "latency_ns,xla_peak_memory,flops,cosine_sim",
    }
    acts = []
    for i in range(A.num_measure_actors):
        aenv = dict(base_env)
        aenv["CUDA_VISIBLE_DEVICES"] = str(A.measure_gpu_base + i)
        opts = {"num_cpus": 1, "num_gpus": 0, "runtime_env": {"env_vars": aenv}}
        acts.append(CpuApproximationActor.options(**opts).remote(
            args_dict, variant=None, actor_id=i))
    _RAY_ACTORS = acts
    _RAY_EV = [np.asarray(x) for x in ev]
    print(f"[loop] RAY measure pool: {len(acts)} actors on GPUs "
          f"{A.measure_gpu_base}..{A.measure_gpu_base + A.num_measure_actors - 1} "
          f"(trainer GPU0)", flush=True)

def measure_topk_ray(order_list, micro_budget, rng=None):
    """Measure order_list across the Ray actor pool (one per GPU, concurrent).
    Returns [lat, peak, flops, cos] or None per order, aligned."""
    import ray
    from alphagrad.approx.env import REWARD_INDEX
    LAT = REWARD_INDEX["latency_ns"]; XP = REWARD_INDEX["xla_peak_memory"]
    FL = REWARD_INDEX["flops"]; CO = REWARD_INDEX["cosine_sim"]
    futs = []
    for i, o in enumerate(order_list):
        seq = _seq_from_order(o, micro_budget, rng)
        order, specs, _ = build_order_specs(seq, env)
        a = _RAY_ACTORS[i % len(_RAY_ACTORS)]
        futs.append(a.evaluate.remote(
            np.asarray(order, np.int32), np.asarray(specs, np.int32),
            len(order), eval_samples=_RAY_EV, init=False))
    out = []
    for f in futs:
        try:
            _tok, _eqn, reward = ray.get(f, timeout=1200)
            r = np.asarray(reward)
            # sentinel (failed measure) = all -1e10 cost channels
            if float(r[FL]) <= -1e9 or not np.all(np.isfinite(r)):
                out.append(None)
            else:
                out.append(np.array([-r[LAT], -r[XP], -r[FL], r[CO]],
                                    dtype=np.float64))
        except Exception as _e:
            print(f"[measure-ray] get failed: {type(_e).__name__}: {_e}", flush=True)
            out.append(None)
    _ok = sum(1 for x in out if x is not None)
    print(f"[measure] ray {_ok}/{len(out)} measured across "
          f"{len(_RAY_ACTORS)} GPU actors", flush=True)
    return out

measured_hist = []   # (ctx, meas, scalar) accumulated this session for held-out Spearman
rows = []
rng = np.random.default_rng(A.seed)
if A.measure_backend == "ray":
    _init_ray_measure()
for rnd in range(A.rounds):
    t0 = time.time()
    # Bound the leak on the MAIN process (encoding). The parallel measure
    # workers free their own leak by process teardown each round.
    if rnd % max(A.clear_caches_every, 1) == 0:
        jax.clear_caches()
        import gc as _gc; _gc.collect()
    temp = max(0.3, 1.0 - rnd * 0.05)   # anneal exploration temperature
    if _USE_TRAINED_PRIOR:
        _az_targets.clear()
    # trained prior needs the per-step lookahead search (records AZ targets);
    # force the full search when the policy is on.
    _proposer = propose_pool if (A.full_search or _USE_TRAINED_PRIOR) else propose_pool_fast
    print(f"[loop] round {rnd} START temp={temp:.3f} calling proposer={_proposer.__name__}", flush=True)
    pool = _proposer(cost_head, stds, A.pool, A.micro_budget, temp, rng)
    print(f"[loop] round {rnd} proposer done pool={len(pool)}", flush=True)
    pool.sort(key=lambda x: -x[1])       # rank by predicted scalar (higher=better)
    topk = pool[:A.topk]
    # MEASURE top-k for real
    meas_raw, pred_scalar, meas_scalar = [], [], []
    _orders_topk = [o for o, _ in topk]
    if A.measure_backend == "ray":
        _raws = measure_topk_ray(_orders_topk, A.micro_budget,
                                 np.random.default_rng(A.seed + rnd))
    elif A.measure_workers > 0:
        _raws = measure_topk_parallel(_orders_topk, A.micro_budget, A.seed + rnd,
                                      A.measure_workers, A.measure_gpu_base)
    else:
        _raws = [measure_order(o, A.micro_budget,
                               np.random.default_rng(A.seed + rnd * 100 + len(o)))
                 for o in _orders_topk]
    for (order_ids, ps), raw in zip(topk, _raws):
        if raw is None or not np.all(np.isfinite(raw)):
            continue
        ctx = order_ctx(order_ids, A.micro_budget,
                        np.random.default_rng(A.seed + rnd * 100 + len(order_ids)))
        ms = scalarize(raw)
        meas_raw.append(raw); pred_scalar.append(ps); meas_scalar.append(ms)
        bufX.append(ctx); bufY.append(raw); bufS.append(ms)
        _feed_pareto(order_ids, raw, rnd)
        measured_hist.append((ctx, raw, ms))
        if ms > best_real["scalar"]:
            best_real = {"scalar": ms, "raw": raw.tolist(), "order": list(map(int, order_ids)), "round": rnd}
    # RETRAIN on the growing buffer (ranking loss)
    cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)
    # TRAIN the AZ policy on this round's improved targets (from-scratch).
    _pol_ce, _pol_ent = (train_policy(list(_az_targets), A.policy_epochs)
                         if _USE_TRAINED_PRIOR else (float('nan'), float('nan')))
    # LOG: ranking Spearman on held-out (seed) set + on the accumulated measured set
    ho_pred = _pred_scalar_np(cost_head, HOX, *stds)
    sp_ho = float(spearmanr(ho_pred, HOS).correlation)
    if len(measured_hist) >= 5:
        MX = np.array([m[0] for m in measured_hist]); MS = np.array([m[2] for m in measured_hist])
        mpred = _pred_scalar_np(cost_head, MX, *stds)
        sp_meas = float(spearmanr(mpred, MS).correlation)
    else:
        sp_meas = float("nan")
    # predicted-vs-measured GAP on this round's top-k (the over-optimization alarm)
    if pred_scalar:
        ps_a, ms_a = np.array(pred_scalar), np.array(meas_scalar)
        gap_mag = float(np.mean(ps_a - ms_a))               # signed: >0 = model over-optimistic
        gap_rank = float(spearmanr(ps_a, ms_a).correlation) if len(ps_a) > 2 else float("nan")
    else:
        gap_mag, gap_rank = float("nan"), float("nan")
    # best real found so far -> latency/peak
    br = best_real["raw"]
    row = dict(round=rnd, buffer=len(bufX), spearman_holdout=sp_ho, spearman_measured=sp_meas,
               gap_mag=gap_mag, gap_rank=gap_rank,
               best_lat_us=(br[0] / 1e3 if br else float("nan")),
               best_peak_MB=(br[1] / 1e6 if br else float("nan")),
               best_scalar=best_real["scalar"], n_measured=len(meas_raw), temp=temp,
               secs=time.time() - t0)
    row["policy_ce"] = _pol_ce if _USE_TRAINED_PRIOR else float('nan')
    row["policy_entropy"] = _pol_ent if _USE_TRAINED_PRIOR else float('nan')
    row["n_az_targets"] = len(_az_targets) if _USE_TRAINED_PRIOR else 0
    row["round_sec"] = float(time.time() - t0)
    rows.append(row)
    # dump the Pareto front each round (ppo_pareto_front.json style) + metrics.
    try:
        _hv = _pareto.hypervolume()
        _pareto.dump_front(_pareto_path, extra={"round": rnd,
                           "hypervolume": (float(_hv) if np.isfinite(_hv) else None)})
        row["pareto_hypervolume"] = float(_hv) if np.isfinite(_hv) else float("nan")
        row["pareto_archive_size"] = int(len(_pareto.pts))
    except Exception as _pe:
        row["pareto_hypervolume"] = float("nan"); row["pareto_archive_size"] = 0
        print(f"[loop] pareto dump failed: {_pe}", flush=True)
    if _wb_run is not None:
        try:
            _wb_run.log({
                "round": rnd, "buffer_size": len(bufX),
                "sp_holdout": sp_ho, "sp_measured": sp_meas,
                "gap_rank": gap_rank, "gap_mag": gap_mag,
                "best_lat_us": row["best_lat_us"], "best_peak_mb": row["best_peak_MB"],
                "best_cos": (best_real["raw"][3] if best_real["raw"] else float("nan")),
                "best_scalar": best_real["scalar"], "n_measured": len(meas_raw),
                "temp": temp,
                "policy_ce": row["policy_ce"], "policy_entropy": row["policy_entropy"],
                "n_az_targets": row["n_az_targets"],
                "round_sec": row["round_sec"],
                "pareto/hypervolume": row.get("pareto_hypervolume", float("nan")),
                "pareto/archive_size": row.get("pareto_archive_size", 0),
            })
        except Exception:
            pass
    print(f"[ROUND {rnd:2d}] buf={len(bufX)} sp_ho={sp_ho:+.3f} sp_meas={sp_meas:+.3f} "
          f"gap_mag={gap_mag:+.4g} gap_rank={gap_rank:+.3f} "
          f"best_real=(lat={row['best_lat_us']:.1f}us,peak={row['best_peak_MB']:.2f}MB) "
          f"n_meas={len(meas_raw)} temp={temp:.2f}"
          + (f" pol_ce={_pol_ce:.3f} pol_ent={_pol_ent:.3f}" if _USE_TRAINED_PRIOR else "")
          + f" ({row['secs']:.0f}s)", flush=True)
    json.dump({"rows": rows, "best_real": best_real}, open(os.path.join(A.out, f"loop_micro{A.micro_budget}.json"), "w"), indent=2, default=float)

# --------------------------------------------------------------- final benchmark
def rev(): return [VALID.index(c) for c in reversed(VALID)]
def markowitz_greedy_order():
    graph, tg = copy_g(GRAPH0), copy_g(TG0); chosen = []
    while True:
        legal = [i for i in legal_set(graph) if i not in chosen]
        if not legal: break
        mk = markowitz(graph, tg, legal); v = min(legal, key=lambda i: mk[i])
        chosen.append(v); _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
    return [VALID.index(c) for c in chosen]
print("\n[loop] === FINAL BENCHMARK (real measured, micro_budget=%d) ===" % A.micro_budget, flush=True)
bench = {}
for name, order in [("loop_best", best_real["order"]), ("reverse", rev()), ("markowitz", markowitz_greedy_order())]:
    if order is None: continue
    raw = measure_order(order, A.micro_budget)
    bench[name] = {"lat_us": raw[0] / 1e3, "peak_MB": raw[1] / 1e6, "flops": float(raw[2]), "cos": float(raw[3]), "order": list(map(int, order))}
    print(f"[BENCH] {name:12s} lat_us={raw[0]/1e3:.2f} peak_MB={raw[1]/1e6:.2f} flops={raw[2]:.4g} cos={raw[3]:+.3f}", flush=True)
json.dump({"rows": rows, "best_real": best_real, "bench": bench, "node": node},
          open(os.path.join(A.out, f"loop_micro{A.micro_budget}_final.json"), "w"), indent=2, default=float)
if _wb_run is not None:
    try:
        _wb_run.summary["best_lat_us"] = (best_real["raw"][0] / 1e3 if best_real["raw"] else float("nan"))
        _wb_run.summary["best_peak_mb"] = (best_real["raw"][1] / 1e6 if best_real["raw"] else float("nan"))
        _wb_run.summary["best_cos"] = (best_real["raw"][3] if best_real["raw"] else float("nan"))
        for _bn, _bv in bench.items():
            _wb_run.summary[f"bench/{_bn}/lat_us"] = _bv["lat_us"]
            _wb_run.summary[f"bench/{_bn}/peak_mb"] = _bv["peak_MB"]
            _wb_run.summary[f"bench/{_bn}/cos"] = _bv["cos"]
        _wb_run.finish()
    except Exception:
        pass
print("[loop] DONE", flush=True)
