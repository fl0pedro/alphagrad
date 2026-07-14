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
ap.add_argument("--example", default="VmappedNeuralNetwork")   # target fn (Feature A)
ap.add_argument("--dataset", default="mnist")                  # dataset for get_args/data_gen
ap.add_argument("--argnums", default="")       # "" = infer_argnums; else comma-sep ints override
ap.add_argument("--fixed-orders", default="")  # "" = RL loop; else comma-sep names / custom:a-b-c (Feature B, measure-only)
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
ap.add_argument("--clear-caches-every", type=int, default=50)  # jax.clear_caches every N rounds
ap.add_argument("--policy-prior", default="markowitz", choices=["markowitz", "trained"])
ap.add_argument("--optimizer", default="none", choices=["none", "random", "sa", "ga", "bo"],
                help="model-free blackbox order search (no policy/surrogate); none=AZ/BO loop")
ap.add_argument("--policy-epochs", type=int, default=200)   # policy CE epochs/round
ap.add_argument("--az-sigma-scale", type=float, default=1.0)  # sigma(Q) scale (Danihelka c_scale)
ap.add_argument("--wandb", action="store_true")           # live per-round logging
ap.add_argument("--wandb-project", default="dsnn-jac-gpu")
ap.add_argument("--wandb-entity", default="dll-streetview")
ap.add_argument("--wandb-name", default="")
# --------------------------------------------------------------- FIXED-ORDER APPROX SEARCH (default OFF)
ap.add_argument("--approx-search", default="")   # OFF. reverse/forward/custom:a-b-c -> fixed vertex order, search DIAG/COMPRESS/QUANT micro-actions
ap.add_argument("--approx-pool", type=int, default=32)     # micro-action candidates sampled per round
ap.add_argument("--approx-max-sub", type=int, default=2)   # max micro-actions per vertex
ap.add_argument("--approx-p-op", default="0.4,0.4,0.2")    # per-slot sampling prob DIAG,COMPRESS,QUANT
ap.add_argument("--approx-quant-dtypes", default="int8,int16,bfloat16,float16,float8_e4m3,float8_e5m2")
# LEARNED MicroActionPolicy vs uniform random micro-action sampling.
ap.add_argument("--approx-policy", default="random", choices=["random", "learned"])
ap.add_argument("--approx-policy-lr", type=float, default=3e-4)
ap.add_argument("--approx-ent-coef", type=float, default=0.01)
ap.add_argument("--approx-policy-epochs", type=int, default=4)   # REINFORCE grad steps/round on the round batch

A = ap.parse_args()
os.environ["ALPHAGRAD_NN_HIDDEN"] = str(A.nn_hidden)
os.makedirs(A.out, exist_ok=True)
# Feature A: optional argnums override (else infer_argnums per example).
_ARGN_OVERRIDE = (tuple(int(x) for x in A.argnums.replace(" ", "").split(",") if x != "")
                  if A.argnums.strip() else None)
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
from graphax.core import _build_graph, _prune_graph, _eliminate_vertex, vertex_elimination_jaxpr
MAX_RULES = 16
CHANNELS = ["latency_ns", "peak_memory", "flops", "cosine_sim"]  # measured tuple order
TIDX = [REWARD_INDEX[c] for c in CHANNELS]

# --------------------------------------------------------------- env + jaxpr graph
# LOSS: default = mean over all outputs (scalar_loss_fn). For tuple-output
# targets (e.g. LIF_SNN returns 7 outputs), ALPHAGRAD_LOSS_OUT_INDEX>=0 selects
# a single output (0 = the squared-error training loss) so mean() is valid.
_LOSS_OUT = int(os.environ.get("ALPHAGRAD_LOSS_OUT_INDEX", "-1"))
if _LOSS_OUT >= 0:
    _basefn = get_fn(A.example)
    LOSS = (lambda *a, _f=_basefn, _i=_LOSS_OUT: jnp.mean(_f(*a)[_i]))
else:
    LOSS = scalar_loss_fn(get_fn(A.example))
ARGN = (_ARGN_OVERRIDE if _ARGN_OVERRIDE is not None else infer_argnums(A.example))
k = jax.random.PRNGKey(0); ak, ek = jax.random.split(k)
xs = get_args(A.example, ak, dataset=A.dataset)
gen = data_gen(A.example, dataset=A.dataset, dataset_size=128)
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
_DEEP = os.environ.get("ALPHAGRAD_PROFILE_DEEP", "0") == "1"
_PROF = {"ppl_t": 0.0, "ppl_n": 0, "octx_t": 0.0, "octx_n": 0,
         "pol_compile": 0.0, "pol_steps": 0, "pol_states": 0, "pol_total": 0.0}
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

# ---- E2 OBJECTIVE ALIGNMENT (user-directed 2026-07-11): PopArt per-channel
# normalisation + DROP FLOPS so AZ optimises the SAME objective as PPO
# {cosine_sim:1.0, latency:0.06, peak:0.06}. Flag-guarded: default OFF =
# byte-identical legacy raw scalarize (MU=0, SD=1, W=[-.06,-.06,-.06,1.0]).
_AZ_ALIGN = os.environ.get("ALPHAGRAD_AZ_ALIGN", "0") == "1"
_AZ_W4 = (np.array([-1.0, -1.0, 0.0, 1.0], dtype=np.float64) if _AZ_ALIGN
          else np.array([-0.06, -0.06, -0.06, 1.0], dtype=np.float64))  # aligned=EQUAL weights (PopArt-normed)
_AZ_MU4 = np.zeros(4, dtype=np.float64)   # PopArt per-channel mean over [lat,peak,flops,cos]
_AZ_SD4 = np.ones(4, dtype=np.float64)    # PopArt per-channel std (1 => legacy raw scale)
def _refresh_az_norm(_bufY):
    """Recompute PopArt (mu,sigma) over the measured 4-tuple buffer. No-op unless
    ALPHAGRAD_AZ_ALIGN=1 (keeps MU=0,SD=1 => scalarize == legacy raw)."""
    global _AZ_MU4, _AZ_SD4
    if not _AZ_ALIGN:
        return
    Y = np.asarray(_bufY, dtype=np.float64)
    if Y.ndim == 2 and len(Y):
        Y = Y[np.all(np.isfinite(Y), axis=1)]
    if Y.ndim == 2 and len(Y) >= 8:
        _AZ_MU4 = Y.mean(0)
        _AZ_SD4 = Y.std(0) + 1e-8

# scalarized cost (higher=better): costs negated, cosine positive. This is the
# search/selection objective and the ranking-loss target.
def scalarize(raw4):
    # higher=better. Legacy(AZ_ALIGN off): -0.06*(lat+peak+flops)+1.0*cos.
    # Aligned: PopArt z-score, flops weight 0, W=[-.06,-.06,0,1.0].
    r = np.asarray(raw4, dtype=np.float64)
    return float(np.sum(_AZ_W4 * (r - _AZ_MU4) / _AZ_SD4))

# --- adds/muls/fmas SEPARATE op-counts (log-only). Symbolic graphax counters;
#     INVARIANT to quant micro-actions (precision, not op-count) so transforms=()
#     is exact for pure-order + quant sweeps (COMPRESS would change counts).
_amf_cache = {}
def _count_amf(order_ids, micro_budget=0):
    _ck = tuple(int(x) for x in order_ids)
    if _ck in _amf_cache:
        return _amf_cache[_ck]
    try:
        seq = _seq_from_order(order_ids, micro_budget)
        order, specs, _ = build_order_specs(seq, env)
        _, aux = vertex_elimination_jaxpr(
            jaxpr, list(order), closed.literals, *xs, argnums=ARGN,
            count_ops=True, sparse_representation=env.config.sparse, transforms=())
        res = (float(aux["adds"]), float(aux["muls"]), float(aux["fmas"]))
    except Exception as _e:
        res = (float("nan"), float("nan"), float("nan"))
    _amf_cache[_ck] = res
    return res

# --------------------------------------------------------------- FIXED-ORDERS measure-only mode (Feature B)
if A.fixed_orders.strip():
    # Resolve each name to a 0-based action-idx order, MEASURE it once at the
    # run's --seed/--micro-budget via the SAME in-process measure path the loop
    # uses (measure_order), record adds/muls/fmas + the 4-tuple, then exit. The
    # 5-seed replication is handled by the CALLER launching 5 seeds.
    def _rev_order(): return [VALID.index(c) for c in reversed(VALID)]
    def _markowitz_order():
        graph, tg = copy_g(GRAPH0), copy_g(TG0); chosen = []
        while True:
            legal = [i for i in legal_set(graph) if i not in chosen]
            if not legal: break
            mk = markowitz(graph, tg, legal); v = min(legal, key=lambda i: mk[i])
            chosen.append(v); _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
        return [VALID.index(c) for c in chosen]
    _BEST_ORDER = [1, 9, 10, 2, 5, 7, 12, 4, 13, 11, 8, 0, 6, 3]  # discovered 27us order
    def _resolve_fixed(nm):
        nm = nm.strip()
        if nm == "reverse": return _rev_order()
        if nm == "forward": return [VALID.index(c) for c in VALID]  # ascending = fwd-mode order
        if nm == "markowitz": return _markowitz_order()
        if nm == "best": return list(_BEST_ORDER)
        if nm.startswith("custom:"):
            return [int(x) for x in nm.split(":", 1)[1].split("-") if x != ""]
        raise ValueError("unknown fixed-order name: %r" % (nm,))
    _fixed_names = [s for s in A.fixed_orders.split(",") if s.strip()]
    print("[loop] FIXED-ORDERS mode: %s micro_budget=%d seed=%d" % (
          _fixed_names, A.micro_budget, A.seed), flush=True)
    _fixed_records = []
    for _nm in _fixed_names:
        _order = _resolve_fixed(_nm)
        _raw = measure_order(_order, A.micro_budget, np.random.default_rng(A.seed))
        _amf = _count_amf(_order, A.micro_budget)
        _rec = {"name": _nm.strip(), "order": list(map(int, _order)), "seed": int(A.seed),
                "adds": float(_amf[0]), "muls": float(_amf[1]), "fmas": float(_amf[2]),
                "lat_ns": float(_raw[0]), "peak": float(_raw[1]),
                "flops": float(_raw[2]), "cos": float(_raw[3])}
        _fixed_records.append(_rec)
        print("[FIXED %s] lat=%.2fus peak=%.2fMB cos=%+.4f adds=%.0f muls=%.0f fmas=%.0f" % (
              _nm.strip(), _raw[0] / 1e3, _raw[1] / 1e6, _raw[3],
              _amf[0], _amf[1], _amf[2]), flush=True)
    json.dump(_fixed_records, open(os.path.join(A.out, "fixed_orders_measured.json"), "w"),
              indent=2, default=float)
    print("[loop] FIXED-ORDERS DONE", flush=True)
    if _wb_run is not None:
        try: _wb_run.finish()
        except Exception: pass
    sys.exit(0)

# --------------------------------------------------------------- buffer (X=ctx, Y=measured, S=scalar)
_AZ_NO_SEED = os.environ.get("ALPHAGRAD_AZ_NO_SEED", "0") == "1"
if _AZ_NO_SEED:
    bufX, bufY = [], []            # COLD START (fair vs cold PPO): no offline cost-head seed
    print("[loop] AZ_NO_SEED=1: COLD START (empty buffer, no offline seed)", flush=True)
else:
    d = np.load(A.seed_dataset, allow_pickle=True)
    bufX = list(np.asarray(d["X"], np.float64))        # pooled encoder ctx
    bufY = list(np.asarray(d["Y"], np.float64))        # measured 4-tuple
_refresh_az_norm(bufY)                                 # PopArt stats from seed buffer (no-op unless AZ_ALIGN)
bufS = [scalarize(y) for y in bufY]                    # scalarized cost (rank target)
print(f"[loop] seeded buffer N={len(bufX)} from {A.seed_dataset}", flush=True)

# held-out set for ranking-Spearman eval: a fixed slice of the seed buffer
rng0 = np.random.default_rng(A.seed)
_ho = rng0.permutation(len(bufX))[:150]
HOX = np.array([bufX[i] for i in _ho]) if len(_ho) else np.zeros((0, EMBD))
HOS = np.array([bufS[i] for i in _ho]) if len(_ho) else np.zeros((0,))

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
    W = jnp.asarray(_AZ_W4)                                 # aligned: flops weight 0; legacy: [-.06]*3+[1]
    _MUj = jnp.asarray(_AZ_MU4); _SDj = jnp.asarray(_AZ_SD4)  # PopArt (0,1 => legacy raw)
    ymu_j, ysd_j = jnp.asarray(ymu), jnp.asarray(ysd)
    def scalar_pred(head, xn):
        pn = jax.vmap(head)(xn)[:, jnp.asarray(TIDX)]     # (B,4) norm symlog
        sl = pn * ysd_j + ymu_j
        raw = jnp.sign(sl) * jnp.expm1(jnp.abs(sl))       # (B,4) raw
        return jnp.sum(W * (raw - _MUj) / _SDj, axis=-1)  # (B,) scalar higher=better (PopArt when aligned)
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
                    _pt0 = time.time()
                    prior_np, _state_info = policy_prior_logits(chosen_a, legal)
                    if _DEEP: _PROF["ppl_t"] += time.time() - _pt0; _PROF["ppl_n"] += 1
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
                _oc0 = time.time()
                _cand_ctx = np.stack([order_ctx(chosen_a + [VALID.index(vv)], micro_budget)
                                      for vv in cand], axis=0)   # (K, EMBD)
                if _DEEP: _PROF["octx_t"] += time.time() - _oc0; _PROF["octx_n"] += len(cand)
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
    _mwdbg0 = os.environ.get("ALPHAGRAD_MW_DEBUG", "0") == "1"
    _mwdir = os.path.expanduser("~/dsnn/mw_err") if _mwdbg0 else None
    if _mwdir: os.makedirs(_mwdir, exist_ok=True)
    _d = _tf.mkdtemp(prefix="mw_", dir=_mwdir)
    for w, ch in enumerate(chunks):
        if not ch:
            procs.append(None); outs.append(None); continue
        jf = os.path.join(_d, f"job{w}.json"); of = os.path.join(_d, f"out{w}.json")
        json.dump({"orders": [list(map(int, o)) for o in ch],
                   "micro_budget": int(micro_budget), "seed": int(seed)}, open(jf, "w"))
        env2 = dict(os.environ)
        env2["CUDA_VISIBLE_DEVICES"] = str(gpu_base + w)
        _mwdbg = os.environ.get("ALPHAGRAD_MW_DEBUG", "0") == "1"
        _errf = open(of + ".err", "w") if _mwdbg else _sp.DEVNULL
        p = _sp.Popen([sys.executable, _WORKER, jf, of, str(A.nn_hidden),
                       str(A.ndata), str(A.latency_inner_reps)], env=env2,
                      stdout=_sp.DEVNULL, stderr=_errf)
        procs.append(p); outs.append(of)
        import time as _tmw; _tmw.sleep(float(os.environ.get("ALPHAGRAD_MW_STAGGER", "4")))  # avoid concurrent CUDA-init CPU-fallback
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
    if os.environ.get("ALPHAGRAD_MW_DEBUG", "0") == "1":
        print(f"[mw-debug] worker temp dir kept: {_d}", flush=True)
    else:
        import shutil as _sh; _sh.rmtree(_d, ignore_errors=True)
    _ok = sum(1 for r in results if r is not None)
    print(f"[measure] parallel {_ok}/{n} measured across {sum(1 for c in chunks if c)} "
          f"GPU workers (base={gpu_base})", flush=True)
    return results

# ============ MODEL-FREE OPTIMIZERS (--optimizer random|sa|ga|bo) ============
# Genome = (order, per-vertex micro-action). ALPHAGRAD_MF_APPROX=1 searches the
# FULL micro-action space {none, quant(dtype), diag(i,j,factor), compress(axis,
# kind)}; params drawn from EXPLICIT ranges (ALPHAGRAD_MF_MAX_AX / _FACTORS) and
# REDRAWN on an invalid measure (bounded retries). Two-level focus: an order can
# be revisited with fresh random approximations. =0 keeps pure order.
if getattr(A, "optimizer", "none") != "none":
    import os as _os_mf
    from graphax.sparse.micro_actions import COMPRESS_KINDS as _CKS
    _os_mf.makedirs(A.out, exist_ok=True)
    _opt = A.optimizer
    _budget = A.total_measurements if A.total_measurements > 0 else (A.rounds * A.topk)
    _APPROX = _os_mf.environ.get("ALPHAGRAD_MF_APPROX", "0") == "1"
    _QDr = [d.strip() for d in _os_mf.environ.get(
        "ALPHAGRAD_QUANT_ALLOWED", "int8,int16,float8_e4m3fn,float8_e5m2,bfloat16,float16"
        ).split(",") if d.strip()]
    _MAXAX = int(_os_mf.environ.get("ALPHAGRAD_MF_MAX_AX", "2"))
    _FACS = [int(x) for x in _os_mf.environ.get("ALPHAGRAD_MF_FACTORS", "2,3,4").split(",") if x]
    _OPS = [o.strip() for o in _os_mf.environ.get(
        "ALPHAGRAD_MF_MICRO_OPS", "none,quant,diag,compress").split(",") if o.strip()]
    _REDRAW = int(_os_mf.environ.get("ALPHAGRAD_MF_REDRAW", "3"))       # retries on invalid measure
    _orng = np.random.default_rng(A.seed)
    _mf_bufY = []
    _mf_all = []          # every measured (n, raw, order, approx) -> offline Pareto front
    _mf_best = {"scalar": -1e18, "raw": None, "order": None, "approx": None, "at": 0}
    _mf_hist = []
    _n = [0]
    print(f"[MF] optimizer={_opt} approx={_APPROX} budget={_budget} NV={NV} "
          f"ops={_OPS} maxax={_MAXAX} facs={_FACS}", flush=True)

    def _mf_reverse():
        return list(range(NV))[::-1]
    def _mf_random_order(rng):
        graph, tg = copy_g(GRAPH0), copy_g(TG0); ca = []
        while True:
            legal = legal_set(graph)
            if not legal:
                break
            v = legal[int(rng.integers(len(legal)))]
            ca.append(VALID.index(v))
            _eliminate_vertex(v, jaxpr, graph, tg, VO, count_ops=False, transforms=())
        return ca
    _MICRO_P = float(_os_mf.environ.get("ALPHAGRAD_MF_MICRO_P", "0.25"))
    _OPS_NN = [o for o in _OPS if o != "none"] or ["quant"]
    def _rand_micro(rng):
        if (not _APPROX) or rng.random() >= _MICRO_P:
            return None
        op = _OPS_NN[int(rng.integers(len(_OPS_NN)))]
        if op == "none":
            return None
        if op == "quant":
            return ("q", _QDr[int(rng.integers(len(_QDr)))])
        if op == "diag":
            i = int(rng.integers(_MAXAX)); j = int(rng.integers(_MAXAX))
            if j == i: j = (i + 1) % max(_MAXAX, 2)
            return ("d", i, j, _FACS[int(rng.integers(len(_FACS)))])
        if op == "compress":
            return ("c", int(rng.integers(_MAXAX)), int(rng.integers(len(_CKS))))
        return None
    def _rand_approx(rng, n):
        return [_rand_micro(rng) for _ in range(n)] if _APPROX else None
    def _mf_rand_cand(rng):
        o = _mf_random_order(rng)
        return (o, _rand_approx(rng, len(o)))
    def _micro_str(m):
        if m is None: return []
        if m[0] == "q": return ["quant('%s')" % m[1]]
        if m[0] == "d": return ["diag(%d,%d,%d)" % (m[1], m[2], m[3])]
        if m[0] == "c": return ["compress('%s',%d)" % (_CKS[m[2]], m[1])]
        return []
    def _mf_seq(cand):
        o, ap = cand
        if ap is None:
            return [(int(a), []) for a in o]
        return [(int(a), _micro_str(ap[k])) for k, a in enumerate(o)]
    def _mf_feat(cand):
        o, ap = cand
        pos = np.zeros(NV, dtype=np.float64)
        for rank, a in enumerate(o):
            pos[int(a)] = rank / max(NV - 1, 1)
        opv = np.zeros(NV, dtype=np.float64)  # per-vertex micro-op class (0 none..3 compress)
        if ap is not None:
            _m = {None: 0}
            for k, a in enumerate(o):
                m = ap[k]
                opv[int(a)] = {None: 0, "q": 1, "d": 2, "c": 3}.get(m[0] if m else None, 0)
        return np.concatenate([pos, opv / 3.0])
    _MS_ON = _os_mf.environ.get("ALPHAGRAD_MEASURE_SERVER", "0") == "1"
    _MS_CLIENT = [None]
    def _measure1(cand):
        if _MS_ON:
            if _MS_CLIENT[0] is None:
                from alphagrad.approx.measure_client import MeasureClient
                _MS_CLIENT[0] = MeasureClient()
            return _MS_CLIENT[0].measure_seq(_mf_seq(cand))
        return _measure1_inproc(cand)
    def _measure1_inproc(cand):
        try:
            order_arr, specs, _ = build_order_specs(_mf_seq(cand), env)
            rs = {}
            _callback(env.config, env.args, env.consts, jnp.asarray(order_arr),
                      jnp.asarray(specs), len(order_arr), *ev, raw_sink=rs)
        except BaseException:
            try:
                jax.clear_caches(); import gc as _gcm; _gcm.collect()
            except Exception:
                pass
            return None
        lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
        lat = float(np.mean(lat_s)) if lat_s else float("nan")
        peak = float(rs.get("xla_peak_memory", float("nan")))
        flops = float(rs.get("flops", float("nan")))
        cos_pp = rs.get("cosine_sim_per_point", [])
        cos = float(np.mean(cos_pp)) if cos_pp else float("nan")
        r = np.array([lat, peak, flops, cos], dtype=np.float64)
        return r if np.all(np.isfinite(r)) else None
    def _mf_measure(cands):
        # measure each candidate; REDRAW its approximations on invalid (bounded).
        out = []
        for cand in cands:
            r = _measure1(cand); tries = 0
            while r is None and _APPROX and cand[1] is not None and tries < _REDRAW:
                cand = (cand[0], _rand_approx(_orng, len(cand[0]))); r = _measure1(cand); tries += 1
            if r is None:
                out.append((cand, None)); continue
            _n[0] += 1  # only VALID measurements count toward the budget
            _mf_bufY.append(r); out.append((cand, r))
            _mf_all.append({"n": _n[0], "raw": r.tolist(),
                            "order": list(map(int, cand[0])),
                            "approx": ([list(m) if m else None for m in cand[1]]
                                       if cand[1] is not None else None)})
        _refresh_az_norm(_mf_bufY)
        for cand, r in out:
            if r is None:
                continue
            sc = scalarize(r)
            if sc > _mf_best["scalar"]:
                _mf_best.update(scalar=float(sc), raw=r.tolist(),
                                order=list(map(int, cand[0])),
                                approx=([list(m) if m else None for m in cand[1]] if cand[1] else None),
                                at=_n[0])
        if _mf_best["raw"] is not None:
            b = _mf_best["raw"]
            _mf_hist.append((_n[0], b[0] / 1e3, b[1] / 1e6, b[3], _mf_best["scalar"]))
        return out
    def _mf_log(tag):
        b = _mf_best["raw"]
        if b is not None:
            na = 0 if not _mf_best["approx"] else sum(1 for x in _mf_best["approx"] if x)
            print(f"[MF-{_opt}] {tag} n={_n[0]}/{_budget} best_lat={b[0]/1e3:.1f}us "
                  f"best_peak={b[1]/1e6:.2f}MB best_cos={b[3]:+.4f} napprox={na} "
                  f"(at n={_mf_best['at']})", flush=True)

    _P_REUSE = float(_os_mf.environ.get("ALPHAGRAD_MF_REUSE", "0.5"))  # focus: revisit an order
    _seen_orders = []
    def _pick_order(rng):
        if _APPROX and _seen_orders and rng.random() < _P_REUSE:
            return list(_seen_orders[int(rng.integers(len(_seen_orders)))])
        o = _mf_random_order(rng); _seen_orders.append(o); return o

    if _opt == "random":
        while _n[0] < _budget:
            k = min(max(A.topk, 1), max(_budget - _n[0], 1))
            cands = [(_pick_order(_orng), _rand_approx(_orng, NV)) for _ in range(k)]
            _mf_measure(cands); _mf_log("iter")
    elif _opt == "sa":
        cur = (_mf_reverse(), _rand_approx(_orng, NV))
        cr = _mf_measure([cur]); craw = cr[0][1] if cr and cr[0][1] is not None else None
        if craw is None:
            cur = _mf_rand_cand(_orng); craw = _mf_measure([cur])[0][1]
        T = float(_os_mf.environ.get("ALPHAGRAD_SA_T0", "1.0"))
        cool = float(_os_mf.environ.get("ALPHAGRAD_SA_COOL", "0.98"))
        while _n[0] < _budget:
            if _APPROX and _orng.random() < 0.5:
                nxt = (list(cur[0]), _rand_approx(_orng, NV))     # FOCUS: same order, redraw approx
            else:
                o2 = list(cur[0]); _L = max(len(o2), 2)
                a, b = int(_orng.integers(_L)), int(_orng.integers(_L))
                o2[a], o2[b] = o2[b], o2[a]; nxt = (o2, _rand_approx(_orng, NV))
            mr = _mf_measure([nxt]); nraw = mr[0][1] if mr else None
            if nraw is None:
                T *= cool; continue
            csc = scalarize(craw); nsc = scalarize(nraw)
            if nsc > csc or _orng.random() < np.exp((nsc - csc) / max(T, 1e-6)):
                cur, craw = mr[0][0], nraw
            T *= cool; _mf_log(f"T={T:.3f}")
    elif _opt == "ga":
        P = int(_os_mf.environ.get("ALPHAGRAD_GA_POP", "10"))
        pop0 = [_mf_rand_cand(_orng) for _ in range(P)]
        mr = _mf_measure(pop0)
        pop = [c for c, r in mr if r is not None]; praws = [r for c, r in mr if r is not None]
        _ga_re = 0
        while len(pop) < 2 and _n[0] < _budget and _ga_re < 20:
            _mr2 = _mf_measure([_mf_rand_cand(_orng) for _ in range(P)])
            pop += [c for c, r in _mr2 if r is not None]; praws += [r for c, r in _mr2 if r is not None]
            _ga_re += 1
        def _ox(p1, p2, rng):
            n = len(p1); i, j = sorted(int(rng.integers(0, n)) for _ in range(2))
            child = [-1] * n; child[i:j + 1] = p1[i:j + 1]
            fill = [x for x in p2 if x not in child]; k = 0
            for t in range(n):
                if child[t] == -1:
                    child[t] = fill[k]; k += 1
            return child
        def _tourn(rng):
            a, b = int(rng.integers(0, len(pop))), int(rng.integers(0, len(pop)))
            return pop[a] if scalarize(praws[a]) >= scalarize(praws[b]) else pop[b]
        while _n[0] < _budget and len(pop) >= 1:
            kids = []
            for _ in range(min(P, max(_budget - _n[0], 1))):
                pa, pb = _tourn(_orng), _tourn(_orng)
                co = _ox(pa[0], pb[0], _orng)
                ap = _rand_approx(_orng, NV) if _APPROX else None    # mutate = redraw approx
                if _orng.random() < 0.3 and len(co) >= 2:
                    _L = len(co)
                    a, b = int(_orng.integers(_L)), int(_orng.integers(_L)); co[a], co[b] = co[b], co[a]
                kids.append((co, ap))
            mr = _mf_measure(kids)
            allc = list(zip(pop, praws)) + [(c, r) for c, r in mr if r is not None]
            allc.sort(key=lambda t: -scalarize(t[1])); allc = allc[:P]
            pop = [c for c, _ in allc]; praws = [r for _, r in allc]; _mf_log("gen")
    if _opt == "bo":
        _bo = []   # (feat, cand, raw)
        init = [_mf_rand_cand(_orng) for _ in range(max(A.topk, 8))]
        for c, r in _mf_measure(init):
            if r is not None: _bo.append((_mf_feat(c), c, r))
        _beta = float(_os_mf.environ.get("ALPHAGRAD_BO_BETA", "0.3"))
        while _n[0] < _budget and _bo:
            Xa = np.asarray([x[0] for x in _bo]); ya = np.asarray([scalarize(x[2]) for x in _bo])
            cands = [(_pick_order(_orng), _rand_approx(_orng, NV)) for _ in range(max(A.pool, A.topk))]
            scr = []
            for c in cands:
                d = np.linalg.norm(Xa - _mf_feat(c), axis=1); idx = np.argsort(d)[:min(5, len(d))]
                w = 1.0 / (d[idx] + 1e-6); scr.append(float(np.sum(w * ya[idx]) / np.sum(w)) + _beta * float(d[idx].min()))
            top = [cands[i] for i in np.argsort(scr)[::-1][:A.topk]]
            for c, r in _mf_measure(top):
                if r is not None: _bo.append((_mf_feat(c), c, r))
            _mf_log("bo")

    import json as _json_mf
    _json_mf.dump({"optimizer": _opt, "approx": _APPROX, "seed": int(A.seed),
                   "budget": int(_budget), "best": _mf_best, "history": _mf_hist,
                   "all_measurements": _mf_all},
                  open(_os_mf.path.join(A.out, f"mf_{_opt}.json"), "w"), indent=2, default=float)
    _mf_log("DONE")
    _wbr = globals().get("_wb_run", None)
    if _wbr is not None:
        try: _wbr.finish()
        except Exception: pass
    sys.exit(0)

# HOISTED to module scope so @eqx.filter_jit compiles ONCE. Previously these
# were defined INSIDE train_policy -> a fresh jit object each round -> ~8s
# recompile every round (profiled 2026-07-11 job 52955 compile1=8.4s/round).
def _tp_ce(agent, tj, ej, tg_, mk_):
    def per(t, e, tgt, mk):
        enc_x, tm = agent.encode_tokens(t, key=jax.random.PRNGKey(0), eqn_ids=e)
        vlog, _ = agent.vertex_policy(enc_x, tm)
        vlog = jnp.where(mk > 0.5, vlog, -1e9)
        logp = jax.nn.log_softmax(vlog)
        ce = -jnp.sum(tgt * logp)
        p = jnp.exp(logp) * mk
        ent = -jnp.sum(jnp.where(mk > 0.5, p * logp, 0.0))
        return ce, ent
    ce, ent = jax.vmap(per)(tj, ej, tg_, mk_)
    return jnp.mean(ce), jnp.mean(ent)
def _tp_loss(agent, tj, ej, tg_, mk_):
    ce, ent = _tp_ce(agent, tj, ej, tg_, mk_); return ce
@eqx.filter_jit
def _tp_step(agent, ost, tj, ej, tg_, mk_):
    l, gr = eqx.filter_value_and_grad(_tp_loss)(agent, tj, ej, tg_, mk_)
    up, ost = _pol_opt.update(gr, ost, eqx.filter(agent, eqx.is_array))
    return eqx.apply_updates(agent, up), ost, l

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
    # _tp_ce/_tp_loss/_tp_step are module-level (compile-once).
    # MINIBATCH the states (each is a MAX_TOKENS encoder pass; vmapping ALL
    # of them at once OOMs — a full round has ~pool*|V| states). Chunk of
    # _POL_MB states per step.
    _POL_MB = 4
    _N = toks_j.shape[0]
    last_ce = float('nan')
    _rngp = np.random.default_rng(A.seed + len(bufX))
    _pol_t0 = time.time(); _pol_steps = 0; _pol_first = 0.0
    for _e in range(epochs):
        _perm = _rngp.permutation(_N)
        for _i in range(0, _N, _POL_MB):
            _ix = _perm[_i:_i + _POL_MB]
            _s0 = time.time()
            policy_agent, _pol_ostate, last_ce = _tp_step(
                policy_agent, _pol_ostate,
                toks_j[_ix], eqns_j[_ix], tgt_j[_ix], mask_j[_ix])
            if _DEEP and _pol_steps == 0:
                float(last_ce); _pol_first = time.time() - _s0
            _pol_steps += 1
    if _DEEP:
        float(last_ce)
        _PROF["pol_compile"] = _pol_first
        _PROF["pol_steps"] = _pol_steps
        _PROF["pol_states"] = int(_N)
        _PROF["pol_total"] = time.time() - _pol_t0
    # eval CE/entropy on a small held slice (avoid the full-batch OOM).
    _ev_ix = jnp.asarray(np.arange(min(_N, 8)))
    _ce_v, _ent_v = _tp_ce(policy_agent, toks_j[_ev_ix], eqns_j[_ev_ix],
                        tgt_j[_ev_ix], mask_j[_ev_ix])
    return float(_ce_v), float(_ent_v)

stds = (np.zeros(EMBD), np.ones(EMBD), np.zeros(4), np.ones(4))  # default (cold) standardization
# warm-start the head with a quick ranking fit on the seed buffer (skip if cold/empty)
if len(bufX) >= 2:
    cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)
# Encode the static-graph token prefix ONCE (incremental AZ policy prior).
if _USE_TRAINED_PRIOR and AZ_INCREMENTAL:
    _init_incremental_prefix()  # static-prefix carry snapshot
best_real = {"scalar": -1e18, "raw": None, "order": None, "round": -1}
# ---- Pareto-front archive (like ppo_ray, ParetoArchive) over the 4 measured
# channels. maximize-oriented vector = [-lat, -peak, -flops, +cos].
from alphagrad.approx.common.pareto_archive import ParetoArchive
_pareto = ParetoArchive(["neg_latency_ns", "neg_xla_peak", "cosine_sim"],
                        [0, 1, 2])   # FLOPS DROPPED (logged only, not a reward)
_pareto_path = os.path.join(A.out, "ppo_pareto_front.json")
def _pareto_vec(raw4):
    # raw4 = [lat, peak, flops, cos] -> maximize orientation. 3 rewards only
    # (latency, memory, cosine); flops dropped from the reward/front.
    return [-float(raw4[0]), -float(raw4[1]), float(raw4[3])]
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
        example=A.example, dataset=A.dataset, dataset_size=128,
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

# --------------------------------------------------------------- FIXED-ORDER APPROX MICRO-ACTION SEARCH (Feature C, default OFF)
if A.approx_search.strip():
    # Pin the vertex elimination ORDER (default reverse) and SEARCH over the
    # per-vertex DIAG / COMPRESS / QUANT approximation micro-actions. Valid
    # micro-actions are sampled from the env's OWN static per-vertex axis state
    # (axis_state_static / axis_valid_static) + graphax's validity rules
    # (Diag.factor a >1 divisor of gcd(N_i,N_j) with distinct out/primal axes;
    # Compress on a physical axis; Quant dtype) -> reuses build_order_specs +
    # micro_actions_to_rule_specs (NOT a reimplementation of validity). Each
    # candidate measured through the SAME Ray pool / _callback grad-measure path,
    # ranked by scalarize(), pareto-archived, amf-logged. Then exit.
    import math as _math
    from graphax.sparse.micro_actions import COMPRESS_KINDS as _CK

    _nm = A.approx_search.strip()
    if _nm == "reverse":
        _order_ids = [VALID.index(c) for c in reversed(VALID)]
    elif _nm == "forward":
        _order_ids = [VALID.index(c) for c in VALID]
    elif _nm.startswith("custom:"):
        _order_ids = [int(x) for x in _nm.split(":", 1)[1].split("-") if x != ""]
    else:
        raise ValueError("unknown --approx-search %r (reverse/forward/custom:a-b-c)" % (_nm,))
    _resolved = [int(VALID[int(v)]) for v in _order_ids]      # jaxpr vertex ids in measured order
    _rev_ref = [int(c) for c in reversed(VALID)]
    _is_rev = (_resolved == _rev_ref)
    print("[approx] FIXED-ORDER APPROX SEARCH mode=%s pool=%d max_sub=%d rounds=%d backend=%s"
          % (_nm, A.approx_pool, A.approx_max_sub, A.rounds, A.measure_backend), flush=True)
    print("[approx] order_action_idx = %s" % (_order_ids,), flush=True)
    print("[approx] measured jaxpr-vertex order = %s" % (_resolved,), flush=True)
    print("[approx] reverse(VALID)               = %s" % (_rev_ref,), flush=True)
    print("[approx] ORDER_IS_REVERSE = %s" % (_is_rev,), flush=True)

    _pp = [float(x) for x in A.approx_p_op.split(",")]
    assert len(_pp) == 3, "--approx-p-op needs 3 comma values (diag,compress,quant)"
    _ps = sum(_pp); _pp = [x / _ps for x in _pp]
    _qd = [x.strip() for x in A.approx_quant_dtypes.split(",") if x.strip()]
    _AXST = np.asarray(env.axis_state_static)     # (V, MAX_AXES, 4): size,is_output,is_compressed,group_id
    _AXVA = np.asarray(env.axis_valid_static)     # (V, MAX_AXES)

    def _approx_axis_info(vid):
        ast = _AXST[vid - 1]; val = _AXVA[vid - 1]
        outs, prims, allax, sizes = [], [], [], {}
        for t in range(val.shape[0]):
            if val[t] > 0.5:
                allax.append(t); sizes[t] = int(ast[t, 0])
                (outs if ast[t, 1] == 1 else prims).append(t)
        return outs, prims, allax, sizes

    def _divisors_gt1(n):
        if n <= 1 or n > 200000:
            return []
        return [d for d in range(2, n + 1) if n % d == 0]

    def _gen_micro_seq(order_ids, rng):
        seq = []
        for v in order_ids:
            vid = int(VALID[int(v)])
            outs, prims, allax, sizes = _approx_axis_info(vid)
            ops = []
            nsub = int(rng.integers(0, A.approx_max_sub + 1))
            for _ in range(nsub):
                choice = str(rng.choice(["diag", "compress", "quant"], p=_pp))
                if choice == "diag" and outs and prims:
                    i = int(rng.choice(outs)); j = int(rng.choice(prims))
                    divs = _divisors_gt1(_math.gcd(sizes[i], sizes[j]))
                    if not divs:
                        continue
                    ops.append("diag(%d,%d,%d)" % (i, j, int(rng.choice(divs))))
                elif choice == "compress" and allax:
                    ax = int(rng.choice(allax))
                    kind = _CK[int(rng.integers(0, len(_CK)))]
                    ops.append("compress('%s', %d)" % (kind, ax))
                elif choice == "quant" and _qd:
                    ops.append("quant('%s')" % (_qd[int(rng.integers(0, len(_qd)))],))
            seq.append((int(v), ops))
        return seq

    def _meas_serial(order, specs):
        rs = {}
        for _attempt in range(2):
            try:
                _callback(env.config, env.args, env.consts, jnp.asarray(order),
                          jnp.asarray(specs), len(order), *ev, raw_sink=rs)
                break
            except RuntimeError as _e:
                if ("mem-gate" in str(_e) or "RESOURCE" in str(_e)) and _attempt == 0:
                    jax.clear_caches(); import gc as _g; _g.collect(); continue
                return None
            except Exception:
                return None
        lat_s = [x for x in rs.get("latency_ns_samples", []) if x > 0 and np.isfinite(x)]
        lat = float(np.mean(lat_s)) if lat_s else np.nan
        peak = float(rs.get("xla_peak_memory", np.nan))
        flops = float(rs.get("flops", np.nan))
        cos_pp = rs.get("cosine_sim_per_point", [])
        cos = float(np.mean(cos_pp)) if cos_pp else np.nan
        raw = np.array([lat, peak, flops, cos], dtype=np.float64)
        return raw if np.all(np.isfinite(raw)) else None

    def _meas_ray(order, specs_list):
        import ray
        from alphagrad.approx.env import REWARD_INDEX as _RI
        LAT = _RI["latency_ns"]; XP = _RI["xla_peak_memory"]; FL = _RI["flops"]; CO = _RI["cosine_sim"]
        futs = []
        for i, specs in enumerate(specs_list):
            a = _RAY_ACTORS[i % len(_RAY_ACTORS)]
            futs.append(a.evaluate.remote(np.asarray(order, np.int32),
                        np.asarray(specs, np.int32), len(order),
                        eval_samples=_RAY_EV, init=False))
        out = []
        for f in futs:
            try:
                _t, _e, reward = ray.get(f, timeout=1200); r = np.asarray(reward)
                if float(r[FL]) <= -1e9 or not np.all(np.isfinite(r)):
                    out.append(None)
                else:
                    out.append(np.array([-r[LAT], -r[XP], -r[FL], r[CO]], dtype=np.float64))
            except Exception as _ex:
                print("[approx-measure-ray] get failed: %s: %s" % (type(_ex).__name__, _ex), flush=True)
                out.append(None)
        return out

    if A.measure_backend == "ray":
        _init_ray_measure()
    _rng = np.random.default_rng(A.seed)
    # ============================== LEARNED MicroActionPolicy setup ==============================
    # Order is PINNED to reverse; the policy searches ONLY over per-vertex
    # DIAG/COMPRESS/QUANT micro-actions via the autoregressive typed heads in
    # heads.py (op_type -> i -> j -> prime-exponent factor -> compress-kind ->
    # quant-dtype), conditioned on MicroPPOAgent's palimpsa encoder over the
    # initial graph state. Trained per round by REINFORCE (baseline = mean) on
    # the per-channel z-scored, equal-weight (lat,mem,cos; flops dropped) reward.
    _LEARNED = (A.approx_policy == "learned")
    learned_agent = None
    if _LEARNED:
        assert _is_rev, "--approx-policy learned requires reverse-pinned order (--approx-search reverse)"
        from alphagrad.approx.ppo import _axis_features_from_state as _axfeat
        from alphagrad.approx.heads import (precompute_factor_tables as _pft,
            OP_DIAG as OP_DIAG, OP_COMPRESS as OP_COMPRESS, OP_QUANT as OP_QUANT, OP_END as OP_END)
        from alphagrad.approx.env import micro_actions_to_rule_specs_jax as _m2rs
        from graphax.sparse.micro_actions import COMPRESS_KINDS as _CK2, QUANT_DTYPES as _QDT
        _pk = jax.random.PRNGKey(A.seed + 4242)
        learned_agent = MicroPPOAgent(vocab_size=512, embd_dim=EMBD, num_layers=4,
            num_heads=4, hidden_dim=256, num_vertices=len(jaxpr.eqns),
            value_dims=(128, 128), key=_pk, max_substeps=16, policy="palimpsa")
        # Match the PPO policy init: near-uniform heads (scale 0.1) + re-zeroed
        # encoder rel-gates so the initial policy explores all op-types.
        learned_agent = _scale_micro_policy_heads(learned_agent, 0.1)
        learned_agent = _rezero_encoder_rel_gates(learned_agent)
        _lopt = _optax.adam(A.approx_policy_lr)
        _lostate = _lopt.init(eqx.filter(learned_agent, eqx.is_array))
        _axsz = np.asarray(env.axis_state_static)[..., 0]
        _tbl_size = min(4096, max(64, int(_axsz.max()) if _axsz.size else 1))
        _ftables = _pft(_tbl_size)   # prime tables => head emits ANY divisor of gcd(N_i,N_j) (no {2,4,8} clamp)
        _REV_AIDX = np.asarray([int(v) - 1 for v in _resolved], np.int32)   # 0-based vertex idx, reverse order
        _REV_AIDX_J = jnp.asarray(_REV_AIDX)
        _AXST_rev = jnp.asarray(np.asarray(env.axis_state_static)[_REV_AIDX])   # (NV, MA, FD)
        _AXVA_rev = jnp.asarray(np.asarray(env.axis_valid_static)[_REV_AIDX])   # (NV, MA)
        _feats_rev = jax.vmap(_axfeat)(_AXST_rev, _AXVA_rev)                    # AxisTokenFeatures batched over NV
        _tok0, _eqn0, _ = _callback(env.config, env.args, env.consts,
            np.zeros((0,), np.int32), np.zeros((0, MAX_RULES, 3), np.int32), 0, *ev, init=True)
        _TOK0 = jnp.asarray(np.asarray(_tok0)); _EQN0 = jnp.asarray(np.asarray(_eqn0))
        print("[approx] LEARNED policy ON: MicroActionPolicy(max_substeps=16) reverse-pinned NV=%d "
              "factor_table<=%d p_lr=%g ent=%g epochs=%d" % (
              NV, _tbl_size, A.approx_policy_lr, A.approx_ent_coef, A.approx_policy_epochs), flush=True)

        @eqx.filter_jit
        def _vcontexts(agent, tok, eqn):
            enc_x, tm = agent.encode_tokens(tok, key=jax.random.PRNGKey(0), eqn_ids=eqn)
            _vl, _vc = agent.vertex_policy(enc_x, tm)
            return _vc   # (num_vertices, E)

        @eqx.filter_jit
        def _sample_round(agent, vctx_rev, feats_rev, keys_pnv):
            def per_cand(keys_nv):
                def per_vertex(vc, feat_i, k):
                    a, lp, ent, ar, *_d = agent.micro_action_policy.sample(vc, feat_i, _ftables, k)
                    return a, lp
                return jax.vmap(per_vertex)(vctx_rev, feats_rev, keys_nv)
            return jax.vmap(per_cand)(keys_pnv)

        @eqx.filter_jit
        def _specs_round(op, i, j, f, kind, q):
            def per_cand(op_c, i_c, j_c, f_c, k_c, q_c):
                def per_vertex(op_v, i_v, j_v, f_v, k_v, q_v, axst_v):
                    return _m2rs(op_v, i_v, j_v, f_v, axst_v, compress_kinds=k_v, quant_dtypes=q_v)
                return jax.vmap(per_vertex)(op_c, i_c, j_c, f_c, k_c, q_c, _AXST_rev)
            return jax.vmap(per_cand)(op, i, j, f, kind, q)   # (P, NV, MAX_RULES, 3)

        def _loss_fn(agent, tok, eqn, actions_b, adv, valid, ent_coef):
            enc_x, tm = agent.encode_tokens(tok, key=jax.random.PRNGKey(0), eqn_ids=eqn)
            _vl, vctx = agent.vertex_policy(enc_x, tm)
            vctx_rev = vctx[_REV_AIDX_J]
            def per_cand(act_c, adv_c, val_c):
                def per_vertex(vc, feat_i, act_v):
                    lp, ent, ar, *_d = agent.micro_action_policy.evaluate(vc, feat_i, _ftables, act_v)
                    return lp, ent
                lps, ents = jax.vmap(per_vertex)(vctx_rev, _feats_rev, act_c)
                logp = jnp.sum(lps); ent = jnp.sum(ents)
                # REINFORCE with baseline: minimise -adv*logp - ent_coef*entropy.
                return val_c * (-(adv_c * logp) - ent_coef * ent), val_c * ent
            terms, ents = jax.vmap(per_cand)(actions_b, adv, valid)
            _n = jnp.maximum(jnp.sum(valid), 1.0)
            return jnp.sum(terms) / _n, jnp.sum(ents) / _n

        _loss_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(_loss_fn, has_aux=True))
    # ============================================================================================
    _recs = []; _amf_rows_ap = []; _best = {"scalar": -1e18, "raw": None, "ops": None, "round": -1}
    _seen_diag = _seen_comp = _seen_quant = 0
    # best-per-cost-metric, EXCLUDING zero values (0 = sentinel/degenerate);
    # cossim is quality (tracked as best_cos, a MAX) not a min-cost metric.
    _best_lat_m = (float("inf"), None); _best_peak_m = (float("inf"), None)
    _best_flops_m = (float("inf"), None); _best_cos_m = (-1.0, None)
    for rnd in range(A.rounds):
        jax.clear_caches(); import gc as _g; _g.collect()
        _round_scalars = []
        _round_factors = []       # every DIAG factor sampled this round (full-dynamic-factor proof)
        _train_actions = None
        if _LEARNED:
            _vctx = _vcontexts(learned_agent, _TOK0, _EQN0)           # (num_vertices, E)
            _vctx_rev = _vctx[_REV_AIDX_J]                            # (NV, E) reverse order
            _kk = jax.random.PRNGKey(A.seed + 100003 + rnd)
            _keys_pnv = jax.random.split(_kk, A.approx_pool * NV).reshape(A.approx_pool, NV, 2)
            _acts, _lp = _sample_round(learned_agent, _vctx_rev, _feats_rev, _keys_pnv)
            _train_actions = _acts                                   # MicroAction (P, NV, S, ...)
            _op_np = np.asarray(_acts.op_type); _i_np = np.asarray(_acts.i)
            _j_np = np.asarray(_acts.j); _f_np = np.asarray(_acts.factor)
            _kind_np = np.asarray(_acts.compress_kind); _q_np = np.asarray(_acts.quant_dtype)
            _specs_all = np.asarray(_specs_round(jnp.asarray(_op_np), jnp.asarray(_i_np),
                jnp.asarray(_j_np), jnp.asarray(_f_np), jnp.asarray(_kind_np), jnp.asarray(_q_np)))
            specs_list = [np.ascontiguousarray(_specs_all[p]).astype(np.int32) for p in range(A.approx_pool)]
            orders = [np.asarray(_resolved, np.int32) for _ in range(A.approx_pool)]
            _S = _op_np.shape[2]
            _ops_per = []; _flags_per = []; _seq_per = []
            for p in range(A.approx_pool):
                _ops = []
                for _vi in range(NV):
                    for _si in range(_S):
                        _o = int(_op_np[p, _vi, _si])
                        if _o == OP_END:
                            break
                        if _o == OP_DIAG:
                            _fac = int(_f_np[p, _vi, _si])
                            _ops.append("diag(%d,%d,%d)" % (int(_i_np[p, _vi, _si]), int(_j_np[p, _vi, _si]), _fac))
                            if _fac > 1:
                                _round_factors.append(_fac)
                        elif _o == OP_COMPRESS:
                            _ops.append("compress('%s',%d)" % (_CK2[int(_kind_np[p, _vi, _si]) % len(_CK2)], int(_i_np[p, _vi, _si])))
                        elif _o == OP_QUANT:
                            _ops.append("quant('%s')" % (_QDT[int(_q_np[p, _vi, _si]) % len(_QDT)],))
                _ops_per.append(_ops)
                _vops = [[] for _ in range(NV)]
                for _vi in range(NV):
                    for _si in range(_S):
                        _o = int(_op_np[p, _vi, _si])
                        if _o == OP_END: break
                        if _o == OP_DIAG: _vops[_vi].append("diag(%d,%d,%d)" % (int(_i_np[p,_vi,_si]),int(_j_np[p,_vi,_si]),int(_f_np[p,_vi,_si])))
                        elif _o == OP_COMPRESS: _vops[_vi].append("compress('%s',%d)" % (_CK2[int(_kind_np[p,_vi,_si])%len(_CK2)],int(_i_np[p,_vi,_si])))
                        elif _o == OP_QUANT: _vops[_vi].append("quant('%s')" % (_QDT[int(_q_np[p,_vi,_si])%len(_QDT)],))
                _seq_per.append(_vops)
                _flags_per.append((any(c.startswith("diag") for c in _ops),
                                   any(c.startswith("compress") for c in _ops),
                                   any(c.startswith("quant") for c in _ops)))
        else:
            seqs = [_gen_micro_seq(_order_ids, _rng) for _ in range(A.approx_pool)]
            specs_list, orders = [], []
            _ops_per = []; _flags_per = []
            for seq in seqs:
                order, specs, _ = build_order_specs(seq, env)
                orders.append(order); specs_list.append(specs)
                _ops = [c for _, cs in seq for c in cs]
                _ops_per.append(_ops)
                _flags_per.append((any(c.startswith("diag") for c in _ops),
                                   any(c.startswith("compress") for c in _ops),
                                   any(c.startswith("quant") for c in _ops)))
        # HARD VERIFY: every candidate's measured order == the pinned reverse order.
        assert all(list(map(int, o)) == _resolved for o in orders), "order drift!"
        assert (not _LEARNED) or _is_rev, "learned approx-search requires reverse-pinned order"
        if A.measure_backend == "ray":
            raws = _meas_ray(orders[0], specs_list)
        else:
            raws = [_meas_serial(o, s) for o, s in zip(orders, specs_list)]
        _nok = _nd = _ncp = _nq = 0
        _valid_idx = []
        for _ci, raw in enumerate(raws):
            _ops = _ops_per[_ci]
            _hd, _hc, _hq = _flags_per[_ci]
            if raw is None:
                continue
            _valid_idx.append(_ci)
            _nok += 1; _nd += _hd; _ncp += _hc; _nq += _hq
            _seen_diag += _hd; _seen_comp += _hc; _seen_quant += _hq
            _feed_pareto(_order_ids, raw, rnd)
            _amf = _count_amf(_order_ids)
            _sc = scalarize(raw)
            _rec = {"round": rnd, "mode": _nm, "order": list(map(int, _resolved)),
                    "seq_per_vertex": (_seq_per[_ci] if _ci < len(_seq_per) else None),
                    "ops": _ops, "n_ops": len(_ops), "has_diag": bool(_hd),
                    "has_compress": bool(_hc), "has_quant": bool(_hq),
                    "lat_ns": float(raw[0]), "peak": float(raw[1]),
                    "flops": float(raw[2]), "cos": float(raw[3]), "scalar": float(_sc),
                    "adds": float(_amf[0]), "muls": float(_amf[1]), "fmas": float(_amf[2])}
            _recs.append(_rec)
            _amf_rows_ap.append({"round": rnd, "order": list(map(int, _order_ids)),
                    "ops": _ops, "adds": float(_amf[0]), "muls": float(_amf[1]),
                    "fmas": float(_amf[2]), "maf_sum": float(_amf[0] + _amf[1] + _amf[2]),
                    "lat_ns": float(raw[0]), "peak": float(raw[1]),
                    "flops": float(raw[2]), "cos": float(raw[3])})
            if _sc > _best["scalar"]:
                _best = {"scalar": float(_sc), "raw": raw.tolist(), "ops": _ops, "round": rnd}
            _round_scalars.append(float(_sc))
            _lr, _pr, _fr, _cr = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
            if _lr > 0 and _lr < _best_lat_m[0]:  _best_lat_m  = (_lr, raw.tolist())
            if _pr > 0 and _pr < _best_peak_m[0]: _best_peak_m = (_pr, raw.tolist())
            if _fr > 0 and _fr < _best_flops_m[0]:_best_flops_m= (_fr, raw.tolist())
            if _cr > _best_cos_m[0]:              _best_cos_m  = (_cr, raw.tolist())
        # ------------------------- LEARNED policy: REINFORCE update this round -------------------------
        _pol_loss = float("nan"); _pol_ent = float("nan")
        if _LEARNED and len(_valid_idx) >= 2:
            _rr = np.array([raws[i] for i in _valid_idx], dtype=np.float64)   # (nv,4): lat,peak,flops,cos
            _zf2 = lambda x: (x - x.mean()) / (x.std() + 1e-9)
            # per-channel z-score, EQUAL WEIGHT over the 3 rewards (lat,mem,cos); flops dropped.
            _nrew = (_zf2(-_rr[:, 0]) + _zf2(-_rr[:, 1]) + _zf2(_rr[:, 3])) / 3.0
            _adv_full = np.zeros((A.approx_pool,), np.float64)
            _val_full = np.zeros((A.approx_pool,), np.float64)
            _baseline = float(_nrew.mean())
            for _k, _i in enumerate(_valid_idx):
                _adv_full[_i] = float(_nrew[_k]) - _baseline   # REINFORCE advantage = reward - baseline
                _val_full[_i] = 1.0
            _advj = jnp.asarray(_adv_full); _valj = jnp.asarray(_val_full)
            for _ep in range(A.approx_policy_epochs):
                (_pl, _pe), _grads = _loss_and_grad(learned_agent, _TOK0, _EQN0,
                    _train_actions, _advj, _valj, float(A.approx_ent_coef))
                _upd, _lostate = _lopt.update(_grads, _lostate, eqx.filter(learned_agent, eqx.is_array))
                learned_agent = eqx.apply_updates(learned_agent, _upd)
            _pol_loss = float(_pl); _pol_ent = float(_pe)
            print("[approx][round %2d] LEARNED update pol_loss=%+.4f pol_ent=%.4f "
                  "nrew(mean=%.3f max=%.3f) nvalid=%d" % (rnd, _pol_loss, _pol_ent,
                  float(_nrew.mean()), float(_nrew.max()), len(_valid_idx)), flush=True)
        if _LEARNED:
            _fset = sorted(set(int(x) for x in _round_factors if int(x) > 1))
            _np2 = [x for x in _fset if (x & (x - 1)) != 0]   # non-power-of-2 factors (proves full dynamic factor)
            print("[approx][round %2d] LEARNED factors_sampled(distinct>1)=%s NON_POW2=%s" % (
                  rnd, _fset, _np2), flush=True)
        # ----------------------------------------------------------------------------------------------
        json.dump(_amf_rows_ap, open(os.path.join(A.out, "amf_signals.json"), "w"), indent=2, default=float)
        json.dump({"mode": _nm, "order": list(map(int, _resolved)),
                   "order_is_reverse": bool(_is_rev), "n_records": len(_recs),
                   "seen_diag": int(_seen_diag), "seen_compress": int(_seen_comp),
                   "seen_quant": int(_seen_quant), "best": _best, "records": _recs},
                  open(os.path.join(A.out, "approx_search_records.json"), "w"), indent=2, default=float)
        try:
            _hv = _pareto.hypervolume()
            _pareto.dump_front(_pareto_path, extra={"round": rnd, "mode": _nm,
                               "order_is_reverse": bool(_is_rev),
                               "hypervolume": (float(_hv) if np.isfinite(_hv) else None)})
        except Exception as _pe:
            print("[approx] pareto dump failed: %s" % (_pe,), flush=True); _hv = float("nan")
        _braw = _best.get("raw")
        _mean_r = float(np.mean(_round_scalars)) if _round_scalars else float("nan")
        # PopArt-style per-channel z-score, EQUAL WEIGHT over the 3 rewards
        # (latency, memory, cosine); flops dropped. Balances scales so cos is
        # not dwarfed by raw ns/bytes -> no cos=0 degenerate wins.
        if _recs:
            _L = np.array([r["lat_ns"] for r in _recs]); _P = np.array([r["peak"] for r in _recs])
            _C = np.array([r["cos"] for r in _recs])
            _zf = lambda x: (x - x.mean()) / (x.std() + 1e-9)
            _nsc = (_zf(-_L) + _zf(-_P) + _zf(_C)) / 3.0
            _bi = int(np.argmax(_nsc)); _bn = _recs[_bi]
        else:
            _bi = -1; _nsc = np.array([float("nan")]); _bn = {"lat_ns": float("nan"), "peak": float("nan"), "cos": float("nan")}
        def _mv(t, idx, sc): return (t[1][idx] / sc) if t[1] else float("nan")
        _bl_us  = _best_lat_m[0]/1e3 if _best_lat_m[1] else float("nan")
        _bp_mb  = _best_peak_m[0]/1e6 if _best_peak_m[1] else float("nan")
        _bf_m   = _best_flops_m[0]/1e6 if _best_flops_m[1] else float("nan")
        print("[approx][round %2d] mean_reward=%+.4f best_scalar(unnorm)=%+.4f | "
              "best_lat=%.2fus(cos%+.3f) best_peak=%.3fMB(cos%+.3f) best_flops=%.2fM(cos%+.3f) "
              "best_cos=%+.4f" % (rnd, _mean_r, _best["scalar"], _bl_us, _mv(_best_lat_m,3,1),
              _bp_mb, _mv(_best_peak_m,3,1), _bf_m, _mv(_best_flops_m,3,1), _best_cos_m[0]), flush=True)
        print("[approx][round %2d] NORM-EQ-WEIGHT best (3 rewards, flops dropped): "
              "cos=%+.4f lat=%.2fus peak=%.3fMB score=%+.3f" % (rnd, _bn["cos"],
              (_bn["lat_ns"]/1e3 if _bi>=0 else float("nan")),
              (_bn["peak"]/1e6 if _bi>=0 else float("nan")),
              (float(_nsc[_bi]) if _bi>=0 else float("nan"))), flush=True)
        print("[approx][round %2d] measured=%d/%d  w/diag=%d w/compress=%d w/quant=%d  "
              "cum(diag=%d compress=%d quant=%d)  best(lat_us=%.2f peak_MB=%.2f cos=%+.4f)  "
              "pareto_hv=%.4g size=%d" % (
              rnd, _nok, len(specs_list), _nd, _ncp, _nq, _seen_diag, _seen_comp, _seen_quant,
              (_braw[0] / 1e3 if _braw else float("nan")),
              (_braw[1] / 1e6 if _braw else float("nan")),
              (_braw[3] if _braw else float("nan")),
              (_hv if np.isfinite(_hv) else float("nan")), len(_pareto.pts)), flush=True)
        if _best.get("ops"):
            print("[approx][round %2d] best ops: %s" % (rnd, _best["ops"]), flush=True)
        if _wb_run is not None:
            try:
                _wb_run.log({"round": rnd, "n_measured": _nok, "w_diag": _nd,
                             "policy_loss": _pol_loss, "policy_entropy": _pol_ent,
                             "learned": (1.0 if _LEARNED else 0.0),
                             "mean_terminal_reward": _mean_r,
                             "norm_best_cos": _bn["cos"],
                             "norm_best_lat_us": (_bn["lat_ns"]/1e3 if _bi>=0 else float("nan")),
                             "norm_best_peak_mb": (_bn["peak"]/1e6 if _bi>=0 else float("nan")),
                             "norm_best_score": (float(_nsc[_bi]) if _bi>=0 else float("nan")),
                             "best_scalar_unnorm": _best["scalar"],
                             "best_lat_us": _bl_us, "best_lat_cos": _mv(_best_lat_m,3,1),
                             "best_peak_mb": _bp_mb, "best_peak_cos": _mv(_best_peak_m,3,1),
                             "best_flops_m": _bf_m, "best_flops_cos": _mv(_best_flops_m,3,1),
                             "best_cos_overall": _best_cos_m[0],
                             "w_compress": _ncp, "w_quant": _nq,
                             "cum_diag": _seen_diag, "cum_compress": _seen_comp,
                             "cum_quant": _seen_quant, "best_scalar": _best["scalar"],
                             "best_cos": (_braw[3] if _braw else float("nan")),
                             "pareto_hv": (_hv if np.isfinite(_hv) else float("nan")),
                             "pareto_size": len(_pareto.pts)})
            except Exception:
                pass
    print("[approx] SEARCH DONE order_is_reverse=%s seen_diag=%d seen_compress=%d seen_quant=%d best=%s"
          % (_is_rev, _seen_diag, _seen_comp, _seen_quant, json.dumps(_best, default=float)), flush=True)
    if _wb_run is not None:
        try:
            _wb_run.finish()
        except Exception:
            pass
    sys.exit(0)

measured_hist = []   # (ctx, meas, scalar) accumulated this session for held-out Spearman
_amf_rows = []       # per-measured-order adds/muls/fmas side-channel (log-only)
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
    # FAIRNESS (2026-07-11): flag-guarded temp floor/anneal. Legacy floor 0.3 +
    # anneal 0.05 collapsed Gumbel diversity by ~round14 -> duplicate orders ->
    # topk under-fills -> AZ starved of its measurement budget. Higher floor +
    # slower anneal keeps proposals diverse so each round measures a full topk.
    _t_floor = float(os.environ.get("ALPHAGRAD_AZ_TEMP_FLOOR", "0.3"))
    _t_anneal = float(os.environ.get("ALPHAGRAD_AZ_TEMP_ANNEAL", "0.05"))
    temp = max(_t_floor, 1.0 - rnd * _t_anneal)   # anneal exploration temperature
    if _USE_TRAINED_PRIOR:
        _az_targets.clear()
    # trained prior needs the per-step lookahead search (records AZ targets);
    # force the full search when the policy is on.
    _proposer = propose_pool if (A.full_search or _USE_TRAINED_PRIOR) else propose_pool_fast
    print(f"[loop] round {rnd} START temp={temp:.3f} calling proposer={_proposer.__name__}", flush=True)
    _tp0 = time.time()
    pool = _proposer(cost_head, stds, A.pool, A.micro_budget, temp, rng)
    _t_prop = time.time() - _tp0
    print(f"[loop] round {rnd} proposer done pool={len(pool)}", flush=True)
    pool.sort(key=lambda x: -x[1])       # rank by predicted scalar (higher=better)
    topk = pool[:A.topk]
    # MEASURE top-k for real
    meas_raw, pred_scalar, meas_scalar = [], [], []
    _orders_topk = [o for o, _ in topk]
    _tm0 = time.time()
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
    _t_meas = time.time() - _tm0
    for (order_ids, ps), raw in zip(topk, _raws):
        if raw is None or not np.all(np.isfinite(raw)):
            continue
        ctx = order_ctx(order_ids, A.micro_budget,
                        np.random.default_rng(A.seed + rnd * 100 + len(order_ids)))
        ms = scalarize(raw)
        meas_raw.append(raw); pred_scalar.append(ps); meas_scalar.append(ms)
        bufX.append(ctx); bufY.append(raw); bufS.append(ms)
        _feed_pareto(order_ids, raw, rnd)
        _amf = _count_amf(order_ids, A.micro_budget)
        _amf_rows.append({"round": rnd, "order": list(map(int, order_ids)),
                          "adds": _amf[0], "muls": _amf[1], "fmas": _amf[2],
                          "maf_sum": float(_amf[0] + _amf[1] + _amf[2]),
                          "lat_ns": float(raw[0]), "peak": float(raw[1]),
                          "flops": float(raw[2]), "cos": float(raw[3])})
        measured_hist.append((ctx, raw, ms))
        if ms > best_real["scalar"]:
            best_real = {"scalar": ms, "raw": raw.tolist(), "order": list(map(int, order_ids)), "round": rnd}
    # E2-ALIGN: refresh PopArt (mu,sigma) on the grown buffer + recompute rank
    # targets so train_ranking sees a consistent normalisation (no-op when OFF:
    # MU=0,SD=1 => scalarize unchanged => bufS recompute is value-identical).
    _refresh_az_norm(bufY)
    bufS[:] = [scalarize(y) for y in bufY]
    # RETRAIN on the growing buffer (ranking loss)
    _tr0 = time.time()
    if len(bufX) >= 2:
        cost_head, stds = train_ranking(cost_head, np.array(bufX), np.array(bufS), A.retrain_epochs)
    _t_rank = time.time() - _tr0
    # TRAIN the AZ policy on this round's improved targets (from-scratch).
    _tpt0 = time.time()
    _pol_ce, _pol_ent = (train_policy(list(_az_targets), A.policy_epochs)
                         if _USE_TRAINED_PRIOR else (float('nan'), float('nan')))
    _t_poltrain = time.time() - _tpt0
    # LOG: ranking Spearman on held-out (seed) set + on the accumulated measured set
    if len(HOX) > 2:
        ho_pred = _pred_scalar_np(cost_head, HOX, *stds)
        sp_ho = float(spearmanr(ho_pred, HOS).correlation)
    else:
        sp_ho = float("nan")
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
    _b_amf = _count_amf(best_real["order"], A.micro_budget) if best_real.get("order") else (float('nan'),)*3
    row["best_adds"], row["best_muls"], row["best_fmas"] = _b_amf
    try:
        json.dump(_amf_rows, open(os.path.join(A.out, "amf_signals.json"), "w"), indent=2, default=float)
    except Exception as _ae:
        print(f"[loop] amf dump failed: {_ae}", flush=True)
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
                "amf/best_adds": row["best_adds"], "amf/best_muls": row["best_muls"],
                "amf/best_fmas": row["best_fmas"],
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
    print(f"[TIMING {rnd:2d}] prop={_t_prop:.1f} meas={_t_meas:.1f} rank={_t_rank:.1f} "
          f"poltrain={_t_poltrain:.1f} total={row['round_sec']:.1f} clearN={A.clear_caches_every}", flush=True)
    if _DEEP:
        _p = _PROF
        print(f"[DEEP {rnd:2d}] prop_split: policy_prior={_p['ppl_t']:.1f}s/{_p['ppl_n']} "
              f"order_ctx={_p['octx_t']:.1f}s/{_p['octx_n']} | poltrain_split: total={_p['pol_total']:.1f}s "
              f"compile1={_p['pol_compile']:.1f}s steps={_p['pol_steps']} states={_p['pol_states']} "
              f"cap={AOJ_TOK_CAP if APPEND_ONLY_JAXPR else 0}", flush=True)
        for _k in _PROF: _PROF[_k] = type(_PROF[_k])(0)
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
