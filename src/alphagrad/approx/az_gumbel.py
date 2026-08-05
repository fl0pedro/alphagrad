"""Sampled + Gumbel AlphaZero over vertex elimination, built from the PPO components.

CLEAN implementation (ignores autoscheduler_loop's surrogate-Gumbel search and mu0):
  * KNOWN dynamics: symbolic graph elimination (_eliminate_vertex) — no learned model,
    no mctx (elimination is not jittable). Python MCTS.
  * Value + prior = the PPO MicroPPOAgent (palimpsa encoder + PointerVertexPolicy +
    per-channel value head). Leaf evaluation = value net; simulations NEVER measure.
  * GUMBEL (Danihelka 2022): root Gumbel-top-m without replacement over the prior
    logits, candidate-set halving with PROGRESSIVE DEEPENING (survivors get a
    2x deeper lockstep rollout each phase; total work ~= n_candidates x
    rollout_depth x ceil(log2 m) — there is no separate simulation budget
    knob), action chosen by
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
import os, sys, json, time, math, argparse, collections

os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "0")
os.environ.setdefault("ALPHAGRAD_PREVALIDATE_MEASURE", "1")
os.environ.setdefault("ALPHAGRAD_BKSTEP", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from alphagrad.approx.az_args import make_argparser  # noqa: E402
A = make_argparser().parse_args()
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

from alphagrad.approx.env import (
    FACE_SLOTS, MAX_FACES as ENV_MAX_FACES, SENTINEL_COST,
    VertexEliminationEnv, _callback, REWARD_INDEX, REWARD_NAMES,
    consume_per_face_stats)

# Full 8-channel reward of the most recent measurement (PPO REWARD_NAMES
# layout) — logged as ``mean_<name>`` for wandb parity with the PPO arms.
LAST_FULL_REWARD = np.zeros(len(REWARD_NAMES), dtype=np.float64)
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn)
from alphagrad.approx.common.eval_samples import generate_eval_samples
# M6 (search dynamics must apply the approximation) needs this at module
# level: step_state runs outside the functions that imported it locally, so
# the approx arm crashed with NameError on its first micro action while the
# exact arm (micro always None) never reached the branch.
from alphagrad.approx.common.masks import make_live_masked_hook
from alphagrad.approx.common.order_specs import build_order_specs
from alphagrad.approx.common.popart import PopArtStats
from alphagrad.approx.common.pareto_archive import ParetoArchive
from graphax.core import _build_graph, _prune_graph, _eliminate_vertex
from graphax.sparse.micro_actions import COMPRESS_KINDS

# ------------------------------------------------------- 1. config/env (measure_worker pattern)
TASK = A.task; DSET = A.dataset
# ALPHAGRAD_GAZ_MEASURE_GRAD=0: measure the RAW JACOBIAN (parity with the
# v45 PPO arms, which run without --measure-grad); default 1 keeps the
# legacy gradient-pipeline target for existing scripts.
_GAZ_MGRAD = os.environ.get("ALPHAGRAD_GAZ_MEASURE_GRAD", "1") == "1"
LOSS = scalar_loss_fn(get_fn(TASK)) if _GAZ_MGRAD else get_fn(TASK)
ARGN = infer_argnums(TASK)
k0 = jax.random.PRNGKey(0); ak, ek = jax.random.split(k0)
xs = get_args(TASK, ak, dataset=DSET)
gen = data_gen(TASK, dataset=DSET, dataset_size=128)
closed = jax.make_jaxpr(LOSS)(*xs)
env = VertexEliminationEnv.from_jaxpr(
    closed, args=xs, argnums=ARGN, num_envs=0, data_gen=gen, target_fun=LOSS,
    sparse=(os.environ.get("ALPHAGRAD_SPARSE", "0") == "1"),
    cmp_type="latency", mem_type="peak_memory",
    # GPU by default; ALPHAGRAD_GAZ_EXEC_ON_GPU=0 for CPU smokes (the GPU
    # path requires >= 2 devices for the measure rotation).
    exec_on_gpu=os.environ.get("ALPHAGRAD_GAZ_EXEC_ON_GPU", "1") == "1",
    # Per-face application (parity with PPO's --per-face): a proposed rule
    # lands only where it is legal on that face's live operand instead of
    # raising TRANSFORM DID NOT FIT and voiding the whole measurement.
    per_face=os.environ.get("ALPHAGRAD_GAZ_PER_FACE", "1") == "1",
    measure_latency=True,
    # W3: reps_per_point was HARDCODED to 1, so AZ medianed ndata samples
    # (2 at the campaign default) against PPO's num_data_points x reps_per_point
    # = 20. ALPHAGRAD_GAZ_REPS matches PPO's 4.
    num_data_points=A.ndata,
    reps_per_point=int(os.environ.get("ALPHAGRAD_GAZ_REPS", "4")),
    percentile_keep=0.60,
    slow_exec_cutoff_seconds=0.0, flop_gate_threshold=0.0, measure_grad=_GAZ_MGRAD,
    latency_inner_reps=A.latency_inner_reps, latency_timer="perf_counter")
ev = generate_eval_samples(env, ek, A.ndata)

# ---- W3-durable: optional GPU-pinned Ray measure pool --------------------
# ALPHAGRAD_GAZ_RAY_MEASURE=N spawns N measure actors the way ppo.py does, so
# AZ gets the SAME isolation guarantees its measurements are compared against:
# separate process, exactly one pinned GPU, one timed execution at a time.
# Default 0 keeps the in-process path (with the clear_caches cadence below).
_GAZ_RAY_N = int(os.environ.get("ALPHAGRAD_GAZ_RAY_MEASURE", "0") or "0")
_MEASURE_POOL = None
if _GAZ_RAY_N > 0:
    from alphagrad.approx.common.measure_pool import (
        spawn_measure_pool, measure_one_plan)
    from alphagrad.approx.env import MAX_TOKENS as _MT, NUM_REWARDS as _NR

    # The actor rebuilds its env from this dict; every field that changes WHAT
    # is measured must match AZ's own env or the actor measures a different
    # graph than the search acts on.
    _pool_args = {
        "example": TASK,
        "dataset": DSET if DSET else "none",
        "dataset_size": 128,
        "seed": int(A.seed),
        "rewards": ["cmp", "mem", "acc"],
        "cmp_type": "latency",
        "mem_type": "peak_memory",
        "exec_on_gpu": True,
        "measure_latency": True,
        "measure_grad": bool(_GAZ_MGRAD),
        "per_face": bool(os.environ.get("ALPHAGRAD_GAZ_PER_FACE", "1") != "0"),
        "face_actions": False,
        "num_data_points": int(A.ndata),
        "reps_per_point": int(os.environ.get("ALPHAGRAD_GAZ_REPS", "4")),
        "latency_inner_reps": int(A.latency_inner_reps),
        "latency_samples": 1,
        "latency_warmup": 0,
        "latency_winsor": 0.0,
        "percentile_keep": 0.60,
        "terminal_rewards_only": False,
        "num_eval_samples": int(A.ndata),
        "hidden_dim": 256,
        "num_cpu_workers": _GAZ_RAY_N,
    }
    try:
        _MEASURE_POOL = spawn_measure_pool(
            _pool_args, n_actors=_GAZ_RAY_N, exec_on_gpu=True,
            timeout_s=float(os.environ.get("ALPHAGRAD_GAZ_RAY_TIMEOUT", "600")),
            max_tokens=int(_MT), num_rewards=int(_NR),
            cosine_sim_idx=int(REWARD_INDEX["cosine_sim"]),
            frob_residual_idx=int(REWARD_INDEX["frob_residual"]))
        print(f"[gaz] ray-measure pool: {_GAZ_RAY_N} actors on gpus "
              f"{list(range(1, _GAZ_RAY_N + 1))} (trainer keeps gpu 0)",
              flush=True)
    except Exception as _pexc:
        print(f"[gaz] ray-measure pool FAILED to start ({type(_pexc).__name__}: "
              f"{_pexc}); falling back to in-process measurement", flush=True)
        _MEASURE_POOL = None
env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)
jaxpr = closed.jaxpr
VALID = list(np.asarray(env.valid_vertices, dtype=np.int32)); NV = len(VALID)
# Two vertex-indexing conventions coexist: net_eval/rollouts use
# ``VALID.index(v)`` while the micro head indexes contexts by ``v - 1``. They
# agree only while VALID is contiguous 1..NV. Fail loudly rather than silently
# scoring the wrong vertex on a task whose jaxpr has an early output eqn.
assert VALID == list(range(1, NV + 1)), (
    f"VALID must be contiguous 1..{NV} for the vertex-index conventions to "
    f"agree; got {VALID[:8]}... (see az_gumbel learned_micro vs net_eval)")
_eg, GRAPH0, TG0, VO = _build_graph(jaxpr, xs, closed.literals, ARGN)
_prune_graph(GRAPH0, TG0, jaxpr, ARGN)
def copy_g(g): return {kk: dict(vv) for kk, vv in g.items()}
def outvar(i): return jaxpr.eqns[i - 1].outvars[0]
def legal_set(graph): return [i for i in VALID if outvar(i) in graph]

# --------------------------- 4. normalisation (PopArt) + Pareto archive / objective (campaign)
# Objective channels. ``xla_peak_memory`` was a 10-channel-era name; this env
# emits 8 and reports the deterministic XLA estimate through the host-side
# side-channel (env.consume_xla_memory_stats) instead of as a reward slot, so
# the measured RM peak is the right stand-in here. Resolve by name with an
# explicit alias table and fail loudly (listing what IS available) rather than
# dying on a bare KeyError deep in module import.
_CH_ALIASES = {"xla_peak_memory": "peak_memory", "bkstep_acc": "cosine_sim"}
CH = ["latency_ns", "xla_peak_memory", "flops", "cosine_sim"]
CH = [_CH_ALIASES.get(c, c) if c not in REWARD_INDEX else c for c in CH]
_missing = [c for c in CH if c not in REWARD_INDEX]
if _missing:
    raise KeyError(
        f"az_gumbel objective channels {_missing} are not emitted by this env. "
        f"Available: {sorted(REWARD_INDEX)}"
    )
TIDX = np.array([REWARD_INDEX[c] for c in CH], dtype=np.int32)
# W2: the objective comes from the SAME helper PPO uses, so the two arms cannot
# optimize different functions again (PPO weighted cosine x2 while AZ used x1).
# Overridable per run with ALPHAGRAD_LAMBDA_{CMP,MEM,ACC}.
class _WNS:
    rewards = ("cmp", "mem", "acc")
    lambda_cmp = float(os.environ.get("ALPHAGRAD_LAMBDA_CMP", "1.0"))
    lambda_mem = float(os.environ.get("ALPHAGRAD_LAMBDA_MEM", "1.0"))
    lambda_acc = float(os.environ.get("ALPHAGRAD_LAMBDA_ACC", "1.0"))


W4 = None   # set after the factory import below (needs az_w4)
# PopArt value-target normalisation over the TIDX focus channels (van Hasselt
# 2016; multi-channel IMPALA form) replaces the batch mean/std z-score: a slow
# debiased-EMA per-channel (mu, sigma) that a homogeneous batch cannot amplify
# (sigma is floored), PLUS the "Art" -- an OUTPUT-PRESERVING rescale of the
# value head's final linear layer on every stats step (applied in the measure
# loop). K = len(TIDX) = the 4 channels the value head is read out at [lat,
# peak, flops, cos]; flops has W4=0 so it is inert in every scalarization,
# tracked only so (mu, sigma) line up 1:1 with v10[TIDX] and the head rows.
# PopArt config MIRRORS ppo_ray_worker (verified): raw cost channels span
# ~1e9, so sigma_max MUST admit the true scale -- the default 1e6 CLAMPS sigma
# and O(100)-under-normalises the cost advantage (ppo observed this ep1-4).
# cosine_sim is a QUALITY channel that converges to ~const (var->0), so it
# gets a higher sigma floor. az feeds M=1 per update, so the robust winsor
# pass is inert here, but the config is kept at parity with ppo_ray_worker.
_sig_min_base = float(os.environ.get("ALPHAGRAD_POPART_SIGMA_MIN", "0.1"))
_sig_min_qual = float(os.environ.get("ALPHAGRAD_POPART_SIGMA_MIN_QUALITY", "0.2"))
_sig_min_vec = np.full(len(TIDX), _sig_min_base, dtype=np.float64)
for _qi, _c in enumerate(CH):
    if _c in ("cosine_sim", "bkstep_acc", "frob_residual"):
        _sig_min_vec[_qi] = max(_sig_min_base, _sig_min_qual)
popart = PopArtStats(
    len(TIDX),
    beta=float(os.environ.get("ALPHAGRAD_POPART_BETA", "0.01")),
    sigma_min=_sig_min_vec,
    sigma_max=float(os.environ.get("ALPHAGRAD_POPART_SIGMA_MAX", "1e12")),
    robust_std=os.environ.get("ALPHAGRAD_POPART_ROBUST_STD", "1") == "1",
    winsor_k=float(os.environ.get("ALPHAGRAD_POPART_WINSOR_K", "5.0")),
)
def scalarize(raw4):
    r = np.asarray(raw4, dtype=np.float64)
    return float(np.sum(W4 * (r - popart.mu) / popart.sigma))

# Pareto archive over the 3 focus OBJECTIVES {latency, xla_peak, cosine} in RAW
# native units (kept SEPARATE from PopArt -- never the normalized values). The
# archive MAXIMISES, so feed a sign-oriented raw vector (negate the cost
# channels, keep cosine) and select the lat/peak/cos slots via obj_idx (flops
# excluded). Exactly 3 objectives => hypervolume() uses the exact 2/3-D sweep,
# not the >=4-D normalized Monte-Carlo estimate.
_PSGN = np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float64)   # over [lat,peak,flops,cos]
_pareto = ParetoArchive(["latency_ns", "xla_peak_memory", "cosine_sim"], [0, 1, 3])

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

# ---- tokenization: graphax append-only STATE tokenizer ----
# VEJaxpr(base_jaxpr, elim_order, transforms) emits
#     <original-graph tokens> | <vertex [: micro-actions] ;> ...
# pure Python over the UNTRACED base jaxpr — no re-trace, no XLA compile, no
# executable leak. The base block is invariant and the suffix grows a few
# tokens per action, so streams stay short and prefix-stable step-to-step.
#  * fixed-cap PADDING (ALPHAGRAD_GAZ_TOKCAP): every stream padded to one length
#    (pad id 0 = masked via tok>0 downstream) -> the encoder compiles ONCE.
from graphax.jaxpr import VEJaxpr

def _state_ids(vertices, transforms=()):
    """(1-based vertex order, ((vertex, rules), ...)) -> token id list."""
    ve = VEJaxpr(jaxpr, elim_order=list(vertices), transforms=tuple(transforms))
    return [int(t) for t in ve.tokenized()]

_full_order = [int(VALID[a]) for a in range(NV)][::-1]
_full_len = len(_state_ids(_full_order))
# MICRO-AWARE cap: micro sub-blocks add a few tokens per vertex — size the cap
# from the worst single-micro-per-vertex stream, else micro states TRUNCATE.
_cap_src = _full_len
if GAZ_MICRO:
    from graphax.sparse.micro_actions import Compress as _Cw
    _worst = tuple((v, (_Cw(axes=(0,), kind="mean"),)) for v in _full_order)
    _cap_src = max(_cap_src, len(_state_ids(_full_order, _worst)))
TOKCAP = int(os.environ.get("ALPHAGRAD_GAZ_TOKCAP", str(int(_cap_src * 1.3) + 8)))
print(f"[gaz] graphax state tokenizer: base={len(_state_ids([]))} "
      f"full_order={_full_len} worst_micro={_cap_src} TOKCAP={TOKCAP}", flush=True)

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
# Retained as an escape hatch; the VEJaxpr state tokenizer never touches XLA,
# so cache clearing is off by default.
_TOK_CLEAR_EVERY = int(os.environ.get("ALPHAGRAD_GAZ_TOK_CLEAR_EVERY", "0"))
def tokens_of(state):
    key = tuple((int(a), tuple(m) if m else None) for a, m in state)
    hit = _tok_cache.get(key)
    if hit is not None:
        return hit
    vertices = [int(VALID[int(a)]) for a, _ in state]
    transforms = tuple((int(VALID[int(a)]), _rules_of(m))
                       for a, m in state if m is not None)
    ids = _state_ids(vertices, transforms)
    tok = _padcap(ids)
    eqn = np.zeros_like(tok)
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
_MEASURE_FAILS: dict = {}
_MEASURE_CALLS: list = [0]
_MEASURE_CLEAR_EVERY = int(
    os.environ.get("ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY", "16") or "0")


def _clear_measure_caches(force: bool = False) -> None:
    """Bound the executable population on the measure device.

    AZ measures IN-PROCESS, so the mitigation ``cpu_approx_worker`` applies for
    the Ray actors (clear every N evaluates, always on OOM) never reached it:
    the distinct-executable population is effectively unbounded and the device
    fills until even a small allocation OOMs (observed on job 58274: 74GB
    resident, 210 allocator warnings, then SIGKILL).
    """
    _MEASURE_CALLS[0] += 1
    _due = _MEASURE_CLEAR_EVERY > 0 and _MEASURE_CALLS[0] % _MEASURE_CLEAR_EVERY == 0
    if not (force or _due):
        return
    try:
        jax.clear_caches()
        import gc
        gc.collect()
    except Exception:
        pass


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
        # Current 8-channel _callback: no raw_sink (that was the 9-channel-era
        # API — passing it raised TypeError, the blanket except returned None,
        # and every "measurement" silently failed). The reward VECTOR carries
        # the winsorized aggregates; costs are stored NEGATED (higher=better).
        n = len(order)
        _zface = (
            jnp.full((n, ENV_MAX_FACES, FACE_SLOTS, 3), -1, dtype=jnp.int32),
            jnp.zeros((n, ENV_MAX_FACES), dtype=jnp.int32),
        )
        if _MEASURE_POOL is not None:
            # Batch of ONE through evaluate_batch: the unbatched pool.evaluate
            # path predates face actions and silently DROPS the face wires.
            _, _, reward, _sent = measure_one_plan(
                _MEASURE_POOL, np.asarray(order), np.asarray(specs),
                np.asarray(_zface[0]), np.asarray(_zface[1]), n,
                eval_samples=ev, init=False)
            reward = np.asarray(reward, dtype=np.float64)
        else:
            _, _, reward = _callback(
                env.config, env.args, env.consts, jnp.asarray(order),
                jnp.asarray(specs), *_zface, n, *ev,
            )
            reward = np.asarray(reward, dtype=np.float64)
    except (KeyboardInterrupt, SystemExit):
        raise                       # W3: never swallow an interrupt
    except BaseException as _exc:
        # W3: distinguish an apparatus OOM (expected, truncate) from a real bug
        # (must be seen). The blanket catch is how the dead ``raw_sink`` kwarg
        # hid: every measurement silently returned None and the run looked
        # merely slow.
        _txt = f"{type(_exc).__name__}: {_exc}".upper()
        _oom = any(k in _txt for k in (
            "RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OUT_OF_MEMORY",
            "OOM WHEN ALLOCATING", "CUDA_ERROR_OUT_OF_MEMORY"))
        _MEASURE_FAILS["oom" if _oom else "other"] = (
            _MEASURE_FAILS.get("oom" if _oom else "other", 0) + 1)
        if not _oom:
            print(f"[gaz] measure FAILED (not OOM): {type(_exc).__name__}: "
                  f"{str(_exc)[:200]}", flush=True)
        _clear_measure_caches(force=True)
        return None
    if reward[REWARD_INDEX["latency_ns"]] <= SENTINEL_COST + 1.0:
        return None  # sentinelled measurement
    lat = -float(reward[REWARD_INDEX["latency_ns"]])
    peak = -float(reward[REWARD_INDEX["peak_memory"]])
    flops = -float(reward[REWARD_INDEX["flops"]])
    cos = float(reward[REWARD_INDEX["cosine_sim"]])
    _clear_measure_caches()          # W3: periodic bound, not only on failure
    r = np.array([lat, peak, flops, cos])
    if not np.all(np.isfinite(r)):
        return None
    # Stash the full 8-channel vector (PPO's REWARD_NAMES layout, PPO sign
    # convention) so the run can log ``mean_<channel>`` on the SAME wandb
    # panels as the PPO arms; the returned 4-vector stays the objective.
    LAST_FULL_REWARD[:] = reward
    return r

# --------------------- 2. policy — ppo.py's MAINLINE Agent (stage-2 parity)
# The prior + value come from the same architecture PPO trains: palimpsa
# encoder (ALPHAGRAD_POLICY), PointerVertexPolicy, and the FOUR single-output
# value heads (latency, mem, cos, frob). MicroPPOAgent (the deprecated Ray
# line's policy) is gone from this file. Head<->objective mapping: GAZ's CH
# order is [latency, peak, flops, cos]; Agent heads are [lat, mem, cos, frob]
# — flops has W4 == 0 (never scalarized) and no head, frob is unused here.
from alphagrad.approx.ppo import (
    _build_agent as _ppo_build_agent,
    _build_factor_table as _ppo_build_factor_table,
    _popart_rescale_heads as _ppo_popart_rescale_heads,
    make_argparser as _ppo_make_argparser,
)

_HEAD_PICK = np.array([0, 1, 2], dtype=np.int32)   # heads for CH [lat, peak, cos]
_CH_ACTIVE = jnp.array([1.0, 1.0, 0.0, 1.0])       # flops row inert (no head)
# W1 (2026-08-04 parity audit): build the SHARED architecture through the shared
# factory instead of ppo's bare argparser defaults. Previously this namespace
# came from ``parse_args([])`` + 5 overrides while the PPO launcher overrode 5
# DIFFERENT fields, so embd_dim/num_heads/num_layers/pointer-class all diverged
# silently (161,916 vs 1,171,693 params) and neither ``init_linear_weights`` nor
# ``_scale_output_heads`` ran here — leaving AZ's initial vertex logits at full
# scale with non-zero biases, i.e. a BIASED Gumbel root prior.
from alphagrad.approx.common.agent_factory import (      # noqa: E402
    apply_policy_arch, build_and_init_agent, az_w4, ALGO_HEAD_FIELDS)

_ns = _ppo_make_argparser().parse_args([])
apply_policy_arch(
    _ns,
    # The approximation-head surface is the ONLY legitimate divergence, and it
    # is set explicitly here rather than inherited from a default. AZ acts per
    # VERTEX today; W5 moves it onto PPO's per-face streamed action space, at
    # which point these match PPO's too.
    dynamic_substeps=True,
    unified_head=False,
    no_approx_head=False,
    face_actions=False,
    unified_face_head=False,
    live_faces=False,
    max_substeps=1,
    axis_group_embedding=False,
)
_ns.seed = A.seed
_ft_table, _ft_py, _n_factors, _max_rules = _ppo_build_factor_table(_ns)
agent = build_and_init_agent(
    _ns, len(jaxpr.eqns), _n_factors, _max_rules, seed=A.seed)
EMBD = _ns.embd_dim
W4 = az_w4(_WNS)          # W2: [-w_cmp, -w_mem, 0, +w_acc]


def _agent_fwd(agent, tok, eqn):
    """(masked-later vertex logits, per-head value (4,)) from the mainline
    Agent's single encode pass."""
    vlog, _ctx, v4 = agent.encode(tok, eqn_ids=eqn, key=jax.random.PRNGKey(0))
    return vlog, v4


# ---------------- stage-3: LEARNED Sampled-AZ micro proposals ----------------
# Root candidate micros drawn from the mainline MicroActionPolicy under the
# LIVE oracle masks (per-state replay), instead of rand_micro's blind
# explicit-range draw — proposals are legal-by-construction on the vertex's
# live edge, and improve as the heads train. ALPHAGRAD_GAZ_MICRO_LEARNED=0
# restores the blind draw.
GAZ_MICRO_LEARNED = os.environ.get("ALPHAGRAD_GAZ_MICRO_LEARNED", "1") == "1"
from alphagrad.approx.common.masks import LiveVertexMaskOracle as _LVMO
from alphagrad.approx.heads import precompute_factor_tables as _pft
from alphagrad.approx.env import (
    decode_vertex_rule_specs as _decode_rows,
    micro_actions_to_rule_specs_jax as _micro_to_rows,
)
from alphagrad.approx.ppo import (
    _axis_features_from_state as _axis_feats,
)
from graphax.sparse.micro_actions import Compress as _GxC, Diag as _GxD, Quant as _GxQ

_LM_TABLES = None
_lm_oracle_cache: dict = {}


@eqx.filter_jit
def _ctx_fwd(agent, tok, eqn):
    _vlog, ctxs, _v4 = agent.encode(tok, eqn_ids=eqn, key=jax.random.PRNGKey(0))
    return ctxs


def _rule_to_tuple(rule):
    if isinstance(rule, _GxD):
        return ("d", int(rule.i), int(rule.j), int(rule.factor))
    if isinstance(rule, _GxC):
        kind = rule.kind if isinstance(rule.kind, str) else str(rule.kind)
        return ("c", int(rule.axes[0]), COMPRESS_KINDS.index(kind))
    if isinstance(rule, _GxQ):
        try:
            name = np.dtype(rule.dtype).name
        except Exception:
            name = str(rule.dtype)
        return ("q", name)
    return None


def learned_micro(state, vertex, key):
    """One masked draw from the mainline micro policy for ``vertex`` at the
    graph produced by ``state``; None on END or any mask/replay failure (the
    candidate then enters plain, exactly like rand_micro's None)."""
    global _LM_TABLES
    if agent.micro_action_policy is None:
        return None
    N = int(env.axis_state_static.shape[1])
    if _LM_TABLES is None:
        _LM_TABLES = _pft(max(8, int(np.asarray(env.axis_state_static)[..., 0].max())))
    k = ("lm",) + _skey(state)
    o = _lm_oracle_cache.get(k)
    if o is None:
        try:
            o = _LVMO(jaxpr, list(closed.literals), list(xs), tuple(ARGN),
                      max_axes=N)
            for a, m in state:
                o.advance(VALID[int(a)], rules=_rules_of(m))
        except Exception:
            return None
        if len(_lm_oracle_cache) > 256:
            _lm_oracle_cache.clear()
        _lm_oracle_cache[k] = o
    try:
        pair, comp = o.vertex_mask(int(vertex))
    except Exception:
        return None
    tok, eqn = tokens_of(state)
    ctxs = _ctx_fwd(agent, jnp.asarray(tok), jnp.asarray(eqn))
    v_idx = int(vertex) - 1
    feats = _axis_feats(env.axis_state_static[v_idx], env.axis_valid_static[v_idx])
    acts, *_r = agent.micro_action_policy.sample(
        ctxs[v_idx], feats, _LM_TABLES, key,
        pair_valid=jnp.asarray(pair, jnp.float32),
        compress_valid=jnp.asarray(comp, jnp.float32),
    )
    rows = _micro_to_rows(
        acts.op_type, acts.i, acts.j, acts.factor,
        env.axis_state_static[v_idx],
        compress_kinds=acts.compress_kind, quant_dtypes=acts.quant_dtype,
        quant_scale_signs=acts.quant_scale_sign,
        quant_scale_fracs=acts.quant_scale_frac,
    )
    try:
        rules = _decode_rows(jaxpr, int(vertex), np.asarray(rows).tolist(),
                             is_last=(len(state) == NV - 1))
    except Exception:
        return None
    if not rules:
        return None
    return _rule_to_tuple(rules[0])

# ------------------------------------------------------------------ 3. optimizer
opt = optax.adam(A.lr)
opt_state = opt.init(eqx.filter(agent, eqx.is_array))

@eqx.filter_jit
def _net_fwd_b(agent, toks, eqns):
    """BATCHED fixed-shape forward: vmap over a stack of TOKCAP-padded states.
    One GPU dispatch evaluates every candidate/rollout frontier at once; only a
    handful of batch sizes occur (<= n_candidates) so compiles are bounded."""
    return jax.vmap(lambda tok, eqn: _agent_fwd(agent, tok, eqn))(toks, eqns)

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
    vlog, v4 = batch_eval([state])[0]
    la = np.array([VALID.index(v) for v in legal], dtype=np.int32)
    logits = vlog[la] - vlog[la].max()
    # Heads [lat, mem, cos] scalarized with the CH weights (flops has no head
    # and W4[2] == 0); trained in PopArt-normalised space like the targets.
    vz = float(W4[0] * v4[0] + W4[1] * v4[1] + W4[3] * v4[2])
    return logits, vz, la

# ---------------------------------------------------------------- known dynamics
def step_state(graph, tg, state, vertex, micro):
    # M6: the search's graph model should apply the SAME approximation the
    # measurement will -- with ``transforms=()`` the search plans on an exact
    # graph and then measures an approximated one, i.e. two different MDPs.
    #
    # GATED OFF BY DEFAULT. The search eliminates in-process on the trainer
    # GPU, and applying the hooks on deep elimination states hits the graphax
    # densify wall: job 58352 produced ZERO episodes in 35 min, looping
    # "GPU_0_bfc ran out of memory trying to allocate 3.83GiB" and finally
    # "byte size of input/output arguments (362538860544) exceeds the base
    # limit (76479332352)" -- a 362 GB program on a 76 GB GPU. The exact arm
    # and the pre-M6 approx arm both ran fine, so the hooks are the trigger.
    # W5 (drive the single authoritative tokenizer graph) removes the
    # trade-off; until then this is an explicit, logged limitation.
    _hooks = ()
    if micro is not None and os.environ.get(
            "ALPHAGRAD_GAZ_SEARCH_HOOKS", "0") == "1":
        _rules = _rules_of(micro)
        _hooks = ((make_live_masked_hook(tuple(_rules)),) if _rules else ())
    _eliminate_vertex(vertex, jaxpr, graph, tg, VO, count_ops=False,
                      transforms=_hooks)
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
    # C1: evaluate EVERY rollout, terminal or not. The old code filtered
    # terminals out and the caller substituted the ROOT value for them, so at
    # the deciding halvings (where most rollouts have terminated) every
    # candidate's Q was the same constant: halving degenerated to
    # argmax(g + logit) and the CE target to softmax(logits) — cross-entropy
    # against itself, i.e. zero learning signal exactly where the search
    # matters most. A terminal state's value head reads its own tokens fine.
    evs = batch_eval([e["st"] for e in live])
    out = []
    for k in range(len(live)):
        v4 = evs[k][1]  # mainline Agent heads [lat, mem, cos]
        out.append(float(W4[0] * v4[0] + W4[1] * v4[1] + W4[3] * v4[2]))
    return out

CVISIT = float(os.environ.get("ALPHAGRAD_GAZ_CVISIT", "50.0"))

def sigma(q, max_n=1, cs=float(os.environ.get("ALPHAGRAD_GAZ_CSCALE", "0.1"))):
    """Danihelka et al. 2022 monotone Q-transform (mctx qtransform form):
    min-max-normalize Q over the candidate set, then scale by
    (c_visit + max_N) * c_scale, so evaluated Q outweighs the prior+Gumbel
    and increasingly so as the search deepens. max_N = evaluation rounds of
    the most-evaluated candidate (progressive-deepening analogue of the
    paper's max visit count). Replaces a per-set z-score that capped every
    Q-gap at ~1 sigma and erased magnitudes (2026-07-16 review, Fix 1)."""
    q = np.asarray(q, dtype=np.float64)
    lo, hi = float(q.min()), float(q.max())
    qn = (q - lo) / max(hi - lo, 1e-8)
    return (CVISIT + float(max_n)) * cs * qn

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
    _lm_key = jax.random.PRNGKey(int(rng.integers(2**31)))
    for ci in order_idx:
        variants = [None]
        if GAZ_MICRO:
            n_rand = K_MICRO
            if GAZ_MICRO_LEARNED:
                lm = learned_micro(state, legal[int(ci)],
                                   jax.random.fold_in(_lm_key, int(ci)))
                if lm is not None:
                    variants.append(lm)
                    n_rand = max(0, K_MICRO - 1)
            variants += [rand_micro(rng, force=True) for _ in range(n_rand)]
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
            c["q"].append(vz)          # C1: never a v_root stand-in
        if len(surv) <= 1:
            break
        # M2: each phase evaluates at a DEEPER rollout, so the entries of
        # ``q`` estimate different quantities; averaging them dilutes the
        # deepest (most informative) one. Use the latest.
        qbar = np.array([c["q"][-1] for c in surv])
        sc = np.array([c["g"] + c["logit"] for c in surv]) + sigma(
            qbar, max_n=max(len(c["q"]) for c in surv))
        # M3: sequential halving keeps CEIL(n/2) (Karnin 2013; Danihelka
        # Alg. 1). ``floor`` collapses 3 -> 1 and skips a comparison phase.
        _keep_n = max(1, -(-len(surv) // 2))
        if _keep_n >= len(surv):        # guard: must strictly shrink
            _keep_n = len(surv) - 1
        keep = np.argsort(-sc)[:_keep_n]
        surv = [surv[i] for i in keep]
        if len(surv) <= 1:
            break                      # M3: before paying for another rollout
        depth = min(depth * 2, NV)                         # deepen survivors
    chosen = surv[0]
    # completed-Q improved policy target over the FULL legal set. A vertex's Q =
    # MAX over its evaluated micro variants (the vertex is as good as its best
    # variant); unvisited vertices complete with v_root (Danihelka completed-Q).
    # C2: complete UNVISITED actions with Danihelka's v_mix (eq. 8-9), not the
    # bare root value. v_mix is the prior-weighted mixture over the VISITED
    # actions blended with v_hat, which keeps the completion inside the range
    # of the evaluated Qs. With the raw v_root fill, any state where
    # v_root > max_a q(a) — about half of them, since both are noisy estimates
    # of the same terminal outcome — gave every unsearched action the top of
    # the sigma range and the executed action the bottom, i.e. a target
    # ANTI-correlated with the search.
    _prior = np.exp(logits - logits.max())
    _prior = _prior / max(_prior.sum(), 1e-12)
    _qv, _nv = {}, {}
    for c in cands:
        if not c["q"]:
            continue
        qv = float(c["q"][-1])          # deepest estimate, as in the halving
        li = c["li"]
        if li not in _qv or qv > _qv[li]:
            _qv[li] = qv
        _nv[li] = max(_nv.get(li, 0), len(c["q"]))
    if _qv:
        _Nsum = float(sum(_nv.values()))
        _den = float(sum(_prior[li] for li in _qv)) or 1e-12
        _num = float(sum(_prior[li] * _qv[li] for li in _qv))
        v_mix = (v_root + _Nsum * (_num / _den)) / (1.0 + _Nsum)
    else:
        v_mix = v_root
    comp_q = np.full(len(legal), v_mix, dtype=np.float64)
    for li, qv in _qv.items():
        comp_q[li] = qv
    pi = logits + sigma(comp_q, max_n=max([len(c["q"]) for c in cands] + [1]))
    pi = np.exp(pi - pi.max()); pi = pi / pi.sum()
    return chosen, pi, la, legal

# ---------------------------------------------------------------- training
def loss_fn(agent, toks, eqns, la_pad, la_mask, pi_pad, vtgt, vmask):
    def per(tok, eqn, la, lam, pi, vt, vm):
        vlog, v4 = _agent_fwd(agent, tok, eqn)
        lg = vlog[la]
        lg = jnp.where(lam > 0.5, lg, -jnp.inf)   # -1e9 collided with a sentinel
        logp = jax.nn.log_softmax(lg)
        ce = -jnp.sum(jnp.where(lam > 0.5, pi * logp, 0.0))
        # Predicted CH vector from the 4 heads: [lat, mem, 0 (flops: no
        # head, _CH_ACTIVE masks it), cos]; targets vt are PopArt-normalised.
        pred = jnp.stack([v4[0], v4[1], jnp.zeros_like(v4[0]), v4[2]])
        vl = jnp.sum(vm * _CH_ACTIVE * (pred - vt) ** 2)
        return ce + 0.5 * vl
    return jnp.mean(jax.vmap(per)(toks, eqns, la_pad, la_mask, pi_pad, vtgt, vmask))

@eqx.filter_jit
def train_step(agent, opt_state, batch):
    l, gr = eqx.filter_value_and_grad(loss_fn)(agent, *batch)
    up, opt_state = opt.update(gr, opt_state, eqx.filter(agent, eqx.is_array))
    return eqx.apply_updates(agent, up), opt_state, l

# ---------------------------------------------------------------- 5.-6. main loop
def _run(args) -> int:
    """The training loop: search -> measure -> PopArt -> Pareto -> train.

    Mirrors `ppo_ray._run`, with one deliberate asymmetry: az's SETUP (env,
    agent, opt, popart, _pareto and the TASK/VALID/NV/TIDX/W4 constants) stays
    at MODULE level, because the search functions (`gumbel_search`, `net_eval`,
    `batch_eval`, `measure`, `scalarize`) read it through module globals —
    moving it into a function would silently turn those reads stale. For the
    same reason `gumbel_search` still reads `A.n_candidates` / `A.rollout_depth`
    off the module-level `A`, not off `args` (same values; see `main`).

    `global agent, opt_state` is LOAD-BEARING, not decoration: this loop REBINDS
    `agent` (PopArt head rescale, then `train_step`) and `batch_eval` must see
    the rebound agent. Without it both names would become `_run` locals and the
    net would either go stale in the search or raise UnboundLocalError.
    """
    global agent, opt_state
    # Phase map, shared with `ppo_ray._run` so the two trainers read in the same
    # order. Phases 1-4 run at IMPORT time here (see the banners above), so only
    # 5 and 6 are in this function:
    #   1. config/env                      -> module level, "1. config/env"
    #   2. policy (build_policy)           -> module level, "2. policy"
    #   3. optimizer                       -> module level, "3. optimizer"
    #   4. normalisation (PopArt) + Pareto -> module level, "4. normalisation"
    #   5. loop / 6. logging+dump          -> below
    # (module order is 1, 4, 2, 3 — the objective constants sit next to PopArt,
    #  which reads them; nothing else depends on that ordering.)

    # --- 1. config/env (run-local tail: out dir, rng, replay, wandb) ---
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    replay = []                                    # episodes of (tok, eqn, la, pi) + ztarget
    _solutions = []                                # every (raw4, state, n) — re-ranked under current norm
    best = {"scalar": -1e18, "raw": None, "state": None, "at": 0}
    n_meas = 0; ep = 0
    _t_start = time.time(); _t_prev = _t_start   # time/* parity with PPO
    # Realized choice counts over this episode's decisions -> approx_prob/*
    _MICRO_CHOICES = collections.Counter()
    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                            name=args.wandb_name, config=vars(args))
        except Exception:
            wb = None
    print(f"[gaz] NV={NV} budget={args.total_measurements} m={args.n_candidates} "
          f"depth={args.rollout_depth} micro={GAZ_MICRO}", flush=True)

    MAXTOK = 0
    # --- 5. loop: act/search -> measure -> popart -> pareto -> train ---
    while n_meas < args.total_measurements:
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
            _mk = chosen.get("micro")
            _MICRO_CHOICES["none" if _mk is None else
                           {"q": "quant", "d": "diag",
                            "c": "compress"}.get(_mk[0], "other")] += 1
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
        # PopArt: one debiased-EMA step on the measured focus channels, then the
        # output-preserving rescale of the value head's TIDX rows (identity on the
        # rest); the critic keeps predicting PopArt-normalised values and its
        # existing predictions stay consistent across the stats jump.
        _o_mu, _o_sig, _n_mu, _n_sig = popart.update(raw[None, :])
        # Mainline Agent: three single-output heads, all CH-mapped
        # ([lat, mem, cos] <- CH rows 0, 1, 3).
        def _stats4(mu, sig):
            m = np.zeros(4, dtype=np.float32)
            s = np.ones(4, dtype=np.float32)
            m[[0, 1, 2]] = np.asarray(mu, np.float32)[[0, 1, 3]]
            s[[0, 1, 2]] = np.asarray(sig, np.float32)[[0, 1, 3]]
            return m, s
        agent = _ppo_popart_rescale_heads(
            agent, *_stats4(_o_mu, _o_sig), *_stats4(_n_mu, _n_sig))
        _eval_cache.clear()          # value head rescaled -> cached (prior, value) stale
        # Pareto front over RAW {lat, xla_peak, cos} (sign-oriented; archive maximises)
        _pareto.add(_PSGN * raw,
                    [(int(a), list(m) if m else None) for a, m in state], ep)
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
        replay = replay[-args.replay_episodes:]
        # ---- train on the replay (only once the normalizer is warm: raw latency/peak
        # magnitudes ~1e4-1e6 would explode the value loss before MU/SD are set) ----
        flat = [s for epi in replay for s in epi]
        if len(flat) >= 8 and popart.n_updates >= 8:
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
            vt = jnp.asarray([(s["raw4"] - popart.mu) / popart.sigma for s in flat])  # PopArt-normalised
            vm = jnp.asarray(np.broadcast_to(np.abs(W4) > 0, (len(flat), 4)).astype(np.float32))
            # M5: resample the minibatch EVERY epoch. Taking one fixed 64-sample
            # draw and hitting it ``train_epochs`` times overfits that draw and
            # discards the rest of the replay for this update.
            for _ in range(args.train_epochs):
                idx = rng.permutation(len(flat))[:64]
                batch = tuple(x[jnp.asarray(idx)]
                              for x in (toks, eqns, la_p, la_m, pi_p, vt, vm))
                agent, opt_state, L = train_step(agent, opt_state, batch)
            L = float(L)
            _eval_cache.clear()          # net changed -> cached (prior, value) stale
        else:
            L = float("nan")
        b = best["raw"]
        try:
            _scal = float(scalarize(raw))
        except Exception:
            _scal = float("nan")
        print(f"[gaz] ep={ep} n={n_meas}/{args.total_measurements} this(lat={raw[0]/1e3:.1f}us "
              f"cos={raw[3]:+.3f}) best(lat={b[0]/1e3:.1f}us peak={b[1]/1e6:.2f}MB "
              f"cos={b[3]:+.4f} at={best['at']}) loss={L:.4f}", flush=True)
        # persist EVERY measured solution each episode (crash-safe): the per-run
        # PARETO FRONT over {lat, xla_peak, cos} is computed offline from this —
        # any weighting re-analyzable without re-running.
        json.dump({"best": best, "n_measured": n_meas, "config": vars(args),
                   "micro": GAZ_MICRO,
                   "solutions": [{"n": nn, "raw": r.tolist(), "state": st}
                                 for r, st, nn in _solutions]},
                  open(os.path.join(args.out, "gaz_result.json"), "w"),
                  indent=1, default=float)
        # Pareto frontier (crash-safe each episode; mirrors ppo_ray): live front +
        # every admitted candidate, plus hypervolume vs a fixed nadir (3-obj exact).
        try:
            _pex = {"episode": ep, "n_measured": n_meas, "task": args.task, "seed": args.seed}
            _pareto.dump_front(os.path.join(args.out, "gaz_pareto_front.json"), extra=_pex)
            _pareto.dump_all_candidates(
                os.path.join(args.out, "gaz_all_front_candidates.json"), extra=_pex)
        except Exception as _pexc:
            print(f"[gaz] pareto dump failed: {_pexc}", flush=True)
        if wb is not None:
            try:
                _now = time.time()
                _log = {
                    # --- AZ-native ---
                    "ep": ep, "n_meas": n_meas, "loss": L,
                    "best_scalar": best["scalar"],
                    "best_lat_us": b[0] / 1e3, "best_cos": b[3],
                    "this_lat_us": raw[0] / 1e3,
                    # --- PPO-parity keys (same names => same panels) ---
                    "pareto/hypervolume": _pareto.hypervolume(),
                    "pareto/archive_size": len(_pareto.pts),
                    "pareto/size": len(_pareto.pts),   # legacy AZ key
                    "Charts/weighted_mean_return": float(_scal),
                    "measure/xla_peak_memory": raw[1] / 1e6,
                    "time/episode": ep,
                    "time/sec_per_episode": _now - _t_prev,
                    "time/wall_seconds": _now - _t_start,
                    "time/wall_minutes": (_now - _t_start) / 60.0,
                }
                for _j, _nm in enumerate(REWARD_NAMES):
                    _log[f"mean_{_nm}"] = float(LAST_FULL_REWARD[_j])
                # --- approximation telemetry (identical keys to the PPO runs)
                _tot = sum(_MICRO_CHOICES.values()) or 1
                for _nm in ("none", "diag", "compress", "quant"):
                    _log[f"approx_prob/{_nm}"] = _MICRO_CHOICES[_nm] / _tot
                # AZ has no per-face SKIP action (skipping a face is a PPO
                # live-faces gate); "eliminate but approximate nothing" is
                # the 'none' class above. Logged as 0.0 so the panel exists
                # on both runs and is honestly empty here.
                _log["approx_prob/skip"] = 0.0
                _pf = consume_per_face_stats()
                # Same story as PPO: the per-face hooks run inside the Ray
                # measure actors, so this process' counters are always empty
                # and every approx_applied/* key would log as a flat 0.
                try:
                    from alphagrad.approx.common.measure_pool import (
                        merge_pool_face_stats as _merge_pf)
                    _pf = _merge_pf(_MEASURE_POOL, _pf)
                except Exception:
                    pass
                for _k in ("diag", "compress", "quant"):
                    _log[f"approx_applied/{_k}"] = _pf.get(f"applied_{_k}", 0)
                    _log[f"approx_skipped/{_k}"] = _pf.get(f"skipped_{_k}", 0)
                _log["approx_applied/total"] = _pf.get("applied", 0)
                _log["approx_skipped/total"] = (
                    _pf.get("skipped", 0) + _pf.get("skipped_raised", 0))
                _log["approx_applied/fraction"] = _pf.get("applied_fraction", 0.0)
                _MICRO_CHOICES.clear()
                _t_prev = _now
                wb.log(_log)
            except Exception:
                pass

    # --- 6. logging/dump ---
    json.dump({"best": best, "n_measured": n_meas, "config": vars(args),
               "micro": GAZ_MICRO}, open(os.path.join(args.out, "gaz_result.json"), "w"),
              indent=2, default=float)
    try:
        _pex = {"episode": ep, "n_measured": n_meas, "task": args.task, "seed": args.seed}
        _pareto.dump_front(os.path.join(args.out, "gaz_pareto_front.json"), extra=_pex)
        _pareto.dump_all_candidates(
            os.path.join(args.out, "gaz_all_front_candidates.json"), extra=_pex)
        print(f"[gaz] pareto: {len(_pareto.pts)} front pts / "
              f"{len(_pareto.all_candidates)} candidates, HV={_pareto.hypervolume():.4g}",
              flush=True)
    except Exception as _pexc:
        print(f"[gaz] final pareto dump failed: {_pexc}", flush=True)
    print(f"[gaz] DONE best={best['raw']} at n={best['at']}", flush=True)
    if wb is not None:
        try: wb.finish()
        except Exception: pass
    return 0


def main() -> int:
    """Entry point, mirroring `ppo_ray.main`.

    NOTE the asymmetry vs ppo_ray: az_gumbel ALSO parses at import time (the
    module-level `A`), because the module-level setup above needs the config
    before any search function is defined. This parse re-reads the same
    `sys.argv`, so `args` is value-identical to `A`; it exists so that `main`
    reads like `ppo_ray.main` and `_run` takes its config as an argument.
    """
    args = make_argparser().parse_args()
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
