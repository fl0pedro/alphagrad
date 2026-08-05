"""Sampled + Gumbel AlphaZero over vertex elimination, built from the PPO components.

CLEAN implementation (ignores autoscheduler_loop's surrogate-Gumbel search and mu0):
  * KNOWN dynamics: symbolic graph elimination — no learned model, no mctx
    (elimination is not jittable). Python MCTS. There is exactly ONE graph
    model, the one inside graphax's IncrementalPathTokenizer (`tk.ij.graph`),
    so the graph the search plans on IS the graph the tokens describe.
  * OBSERVATION = PPO's, not a parallel one: the append-only path tokenizer
    (base stream once, one token DELTA per decision) consumed by the shared
    `common/carry_stream.py` into the palimpsa carry + per-vertex memory.
    Speculative expansions branch with `live_faces._Snapshot`.
  * Value + prior = the mainline PPO Agent (palimpsa encoder + Set-pointer
    vertex head + per-channel value heads), read through
    `heads_from_memory` off that carry — the same call PPO's rollout makes.
    Leaf evaluation = value net; simulations NEVER measure.
  * GUMBEL (Danihelka 2022): root Gumbel-top-m without replacement over the prior
    logits, candidate-set halving with PROGRESSIVE DEEPENING (survivors get a
    2x deeper rollout each phase; total work ~= n_candidates x
    rollout_depth x ceil(log2 m) — there is no separate simulation budget
    knob), action chosen by
    argmax(g + logits + sigma(q)), policy trained by CE to the COMPLETED-Q improved
    target softmax(logits + sigma(completed_q)) over the legal set.
  * SAMPLED (Hubert 2021, pragmatic): with ALPHAGRAD_GAZ_MICRO=1 each root candidate
    is (vertex, micro-action) with the micro drawn from an explicit-range proposal
    (quant/diag/compress/skip, sparse); the search Q decides which survive. The
    learned heads stay vertex-level in v1 (micro-head learning = follow-up).
    SKIP = drop the contraction of every face of that vertex (graphax.SKIP_FACE);
    it is a SEARCH variant only -- see the ("s",) block below for why it is not
    a MicroActionPolicy output.
  * Real measurements ONLY at episode terminals (budget = --total-measurements).
  * Objective identical to the E2 campaign: equal-weight z-scored
    {cosine_sim, latency_ns, peak_memory}; flops unrewarded.

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
    consume_per_face_stats, MAX_DELTA_TOKENS, _record_delta_truncation)

# Full 8-channel reward of the most recent measurement (PPO REWARD_NAMES
# layout) — logged as ``mean_<name>`` for wandb parity with the PPO arms.
LAST_FULL_REWARD = np.zeros(len(REWARD_NAMES), dtype=np.float64)
from alphagrad.approx.common.examples import (
    get_fn, get_args, data_gen, infer_argnums, scalar_loss_fn)
from alphagrad.approx.common.eval_samples import generate_eval_samples
# The application hook the MEASUREMENT uses, needed here so the golden-stream
# check drives the env's tokenizer with exactly the hooks the measurement
# builds (a rule that does not fit one face is skipped for that face, not
# raised for the whole vertex).
from alphagrad.approx.common.masks import make_live_masked_hook
from alphagrad.approx.common.order_specs import build_order_specs
from alphagrad.approx.common.popart import PopArtStats
from alphagrad.approx.common.pareto_archive import ParetoArchive
# THE tokenizer: graphax's IncrementalPathTokenizer, driven exactly as PPO's
# env drives it (base once, one delta per decision) through the shared
# PlanTokenizer. VEJaxpr — graphax's own docstring calls it LEGACY and says
# not to build on it — is gone, and with it the whole-state TOKCAP re-encode
# and its ALL-ZERO eqn array (the relational gate was inert on AZ and live on
# PPO for the entire comparison; `tk.last_eqn_ids()` supplies real ids here).
from alphagrad.approx.common.plan_tokens import PlanTokenizer
from alphagrad.approx.common import carry_stream as _cs
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
TOTAL_V = len(jaxpr.eqns)
# ONE graph model. `PlanTokenizer.legal` reads the TOKENIZER's own
# `tk.ij.graph`, which is built by the same `_build_graph`/`_prune_graph` and
# advanced by the same `_eliminate_vertex` a separate search model would run.
# The parallel model (`_build_graph`/`copy_g`/`legal_set`/`step_state`'s own
# `_eliminate_vertex`) is DELETED: keeping two is the two-MDPs hazard — the
# search plans on one graph while the tokens describe another.
PT = PlanTokenizer(jaxpr, ARGN, list(closed.literals), list(xs),
                   max_faces=int(ENV_MAX_FACES))
_BASE_TOKS, _BASE_IDS = PT.base()
BASE_N = len(_BASE_TOKS)
BASE_W = max(BASE_N, 1)
BASE_TOK = jnp.asarray(np.asarray(_BASE_TOKS[:BASE_W], dtype=np.int32))
BASE_EQN = jnp.asarray(np.asarray(_BASE_IDS[:BASE_W], dtype=np.int32))
# Cross-check against the env's own producer: the two must agree bitwise or
# the search reads a different base than the measurement tokenizes.
_ebt, _ebe, _ebn = env.base_observation()
assert int(_ebn) == BASE_N, (
    f"base stream length disagrees: PlanTokenizer {BASE_N} vs "
    f"env.base_observation() {int(_ebn)}")
assert np.array_equal(np.asarray(_ebt)[:BASE_N], np.asarray(_BASE_TOKS)), (
    "base token stream disagrees between PlanTokenizer and "
    "env.base_observation()")
assert np.array_equal(np.asarray(_ebe)[:BASE_N], np.asarray(_BASE_IDS)), (
    "base eqn ids disagree between PlanTokenizer and env.base_observation()")
del _ebt, _ebe, _ebn
print(f"[gaz] path tokenizer: base={BASE_N} tokens "
      f"(per-step delta budget {MAX_DELTA_TOKENS}, total_v={TOTAL_V})",
      flush=True)

# --------------------------- 4. normalisation (PopArt) + Pareto archive / objective (campaign)
# Objective channels. There is ONE memory channel, ``peak_memory``: the
# deterministic memory_analysis() estimate is substituted INTO it in place
# where the runtime high-water mark is unavailable (see
# env._note_static_peak_fallback), so the 10-channel-era name
# ``xla_peak_memory`` names nothing this env emits. The alias table survives
# for the remaining legacy names; resolve by name and fail loudly (listing what
# IS available) rather than dying on a bare KeyError deep in module import.
_CH_ALIASES = {"xla_peak_memory": "peak_memory", "bkstep_acc": "cosine_sim"}
CH = ["latency_ns", "peak_memory", "flops", "cosine_sim"]
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
_pareto = ParetoArchive(["latency_ns", "peak_memory", "cosine_sim"], [0, 1, 3])

# ---------------------------------------------------------------- state <-> tokens
QD = [d for d in os.environ.get(
    "ALPHAGRAD_QUANT_ALLOWED", "int8,int16,bfloat16,float16").split(",") if d]
GAZ_MICRO = os.environ.get("ALPHAGRAD_GAZ_MICRO", "0") == "1"
MICRO_P = float(os.environ.get("ALPHAGRAD_GAZ_MICRO_P", "0.25"))
MAXAX = int(os.environ.get("ALPHAGRAD_GAZ_MAX_AX", "2"))
FACS = [int(x) for x in os.environ.get("ALPHAGRAD_GAZ_FACTORS", "2,3,4").split(",")]
# ---------------------------------------------------------------- SKIP ("s",)
# The fourth micro op: DROP the contraction of EVERY face of this vertex
# (graphax.SKIP_FACE per face, carried on the face_skips wire -- see measure).
#
# ASYMMETRY, deliberate and documented: PPO's skip is per FACE
# (UnifiedFaceHead.sample draws one Bernoulli per face); az decides per VERTEX,
# so "skip all of this vertex's faces" is the closest faithful variant az's
# action space can express. Exact parity is impossible until the per-face
# action space lands on az (W5).
#
# SKIP IS A SEARCH VARIANT, NOT A POLICY OUTPUT. It is deliberately NOT an
# output of MicroActionPolicy: that head is SHARED with PPO and a fourth op
# there would perturb PPO's heads. gumbel_search proposes it, the value net
# scores it, and the completed-Q improved-policy target teaches the VERTEX
# policy where skipping pays.
GAZ_SKIP = os.environ.get("ALPHAGRAD_GAZ_SKIP", "1") == "1"

def rand_micro(rng, force=False):
    """SAMPLED proposal for a candidate's micro-action (explicit ranges).
    force=True always returns a non-None micro (root candidate variants)."""
    if not GAZ_MICRO or (not force and rng.random() >= MICRO_P):
        return None
    op = int(rng.integers(4 if GAZ_SKIP else 3))
    if op == 0:
        return ("q", QD[int(rng.integers(len(QD)))])
    if op == 1:
        i = int(rng.integers(MAXAX)); j = (i + 1) % max(MAXAX, 2)
        return ("d", i, j, FACS[int(rng.integers(len(FACS)))])
    if op == 2:
        return ("c", int(rng.integers(MAXAX)),
                int(rng.integers(len(COMPRESS_KINDS))))
    return ("s",)

def micro_str(m):
    if m is None: return []
    if m[0] == "q": return ["quant('%s')" % m[1]]
    if m[0] == "d": return ["diag(%d,%d,%d)" % (m[1], m[2], m[3])]
    # WIRE FORMAT: build_order_specs/parse_calls recognise "skip()" and route
    # it to the face_skips array (it decodes to no rule spec), so seq_of ->
    # build_order_specs round-trips a skip without a format hack.
    if m[0] == "s": return ["skip()"]
    return ["compress('%s',%d)" % (COMPRESS_KINDS[m[2]], m[1])]

def _rules_of(m):
    """micro tuple -> graphax rule objects (for the append-only micro tokens).

    A SKIP is NOT a rule object -- it is ``graphax.SKIP_FACE`` applied per
    face -- so this returns () for ("s",), exactly like None. The skip travels
    on the face_skips array instead (see ``measure``). Where the STATE has to
    stay distinguishable, use ``_tok_rules_of``."""
    from graphax.sparse.micro_actions import Diag as _D, Compress as _C, Quant as _Q
    if m is None: return ()
    if m[0] == "q": return (_Q(dtype=m[1]),)
    if m[0] == "d": return (_D(i=int(m[1]), j=int(m[2]), factor=int(m[3])),)
    if m[0] == "s": return ()
    return (_C(axes=(int(m[1]),), kind=COMPRESS_KINDS[m[2]]),)

def _tok_rules_of(m):
    """TOKENIZER-side transforms for one micro: ``_rules_of`` except a SKIP
    emits ``graphax.SKIP_FACE``, which VEJaxpr encodes through its opaque
    -transform tag (``<crc32(repr(t))>``; repr is the stable string
    "graphax.SKIP_FACE", so trainer and measure actors agree).

    LOAD-BEARING for the search: with ``_rules_of`` here the skip variant of a
    vertex would tokenize IDENTICALLY to the plain variant, so the value net
    would score the two the same, and sequential halving (stable argsort over
    tied g+logit+sigma(Q)) would always keep the plain one -- the action would
    exist and never be chosen."""
    if m is not None and m[0] == "s":
        from graphax import SKIP_FACE as _SK
        return (_SK,)
    return _rules_of(m)

def seq_of(state):
    """state = list of (action_idx, micro-or-None) -> build_order_specs seq."""
    return [(int(a), micro_str(m)) for a, m in state]


# ---------------------------------------------------------------- the WIRE
# ONE builder for the arrays that go to BOTH the tokenizer and the
# measurement. Previously the tokenizer got graphax rule OBJECTS (VEJaxpr
# transforms) while `measure` built spec ROWS from `build_order_specs`; two
# encoders of the same decision is how they drift.
#
# `build_order_specs` is per-step independent -- `specs[k]` depends only on
# that step's calls and its resolved vertex id -- so one step's row can be
# built on its own and appended, which is what an incremental search needs.
def _step_wire(action_idx, micro):
    """One decision -> ``(spec_row (MAX_RULES,3), face_rows, face_skips)``."""
    _o, _s, _n, _sk = build_order_specs(
        [(int(action_idx), micro_str(micro))], env, return_skips=True)
    rows = np.full((int(ENV_MAX_FACES), FACE_SLOTS, 3), -1, dtype=np.int32)
    skips = np.zeros((int(ENV_MAX_FACES),), dtype=np.int32)
    if bool(_sk[0]):
        # "skip every face of this vertex" -- env._face_dict_for_vertex only
        # reads the first len(faces_of(v)) entries, so the padding is inert.
        skips[:] = 1
    return np.asarray(_s[0], dtype=np.int32), rows, skips


def plan_wires(state):
    """A whole plan -> ``(order, specs, face_rows, face_skips)``.

    THE wire. The tokenizer, the measurement and the dumps all read these
    same four arrays, so a decision cannot be expressed one way to the
    observation and another way to the measurement.
    """
    order, specs, _n, skips = build_order_specs(
        seq_of(state), env, return_skips=True)
    n = len(order)
    face_rows = np.full((n, int(ENV_MAX_FACES), FACE_SLOTS, 3), -1,
                        dtype=np.int32)
    face_skips = np.zeros((n, int(ENV_MAX_FACES)), dtype=np.int32)
    if n:
        face_skips[np.asarray(skips, dtype=bool)] = 1
    return (np.asarray(order, dtype=np.int32),
            np.asarray(specs, dtype=np.int32), face_rows, face_skips)


def _wire_delta(toks, ids):
    """A tokenizer block -> the ``(MAX_DELTA_TOKENS,)`` device buffers.

    Same clip-don't-raise policy as ``env._delta_observation`` (and the same
    truncation counter), minus its in-band header slot: AZ hands the count
    directly to ``encode_extend`` instead of shipping it through a callback.
    """
    n_raw = len(toks)
    _record_delta_truncation(n_raw)
    n = min(n_raw, MAX_DELTA_TOKENS)
    t = np.zeros((MAX_DELTA_TOKENS,), dtype=np.int32)
    e = np.full((MAX_DELTA_TOKENS,), -1, dtype=np.int32)
    if n:
        t[:n] = np.asarray(toks[:n], dtype=np.int32)
        e[:n] = np.asarray(ids[:n], dtype=np.int32)
    return t, e, n

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
        # THE SAME four wire arrays the tokenizer read (`plan_wires`). Slot 0
        # is the per-face rule rows, slot 1 the face_skips -- the latter is
        # where a ("s",) micro becomes real: env._face_dict_for_vertex turns a
        # 1 into graphax.SKIP_FACE, and it only reads the first
        # len(faces_of(v)) entries, so the padding is inert. Without them the
        # skip is silently dropped and the measurement is byte-identical to
        # the exact plan.
        #
        # Current 8-channel _callback: no raw_sink (that was the 9-channel-era
        # API — passing it raised TypeError, the blanket except returned None,
        # and every "measurement" silently failed). The reward VECTOR carries
        # the winsorized aggregates; costs are stored NEGATED (higher=better).
        order, specs, _face_rows, _skip_rows = plan_wires(state)
        n = len(order)
        _zface = (
            jnp.asarray(_face_rows, dtype=jnp.int32),
            jnp.asarray(_skip_rows, dtype=jnp.int32),
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


# ------------------------------------------------- the OBSERVATION CARRY
# PPO's two-move protocol, verbatim (common/carry_stream.py): consume the
# base stream ONCE into the palimpsa carry + per-vertex memory, then extend
# by one step's DELTA per decision. Both moves are pure jnp, so AZ calls
# them eagerly on its single env while PPO scans them.
#
# PER-VERTEX FEATURES ARE LOAD-BEARING, not decoration. The pointer is a
# SetPointerVertexPolicy (agent_factory: set_pointer=True) and it scores
# CONTENT: with `vertex_features=None` its slots for the un-eliminated
# vertices are all the same empty row, so every candidate gets an identical
# score up to its own query embedding -- the v31 uniform-pick failure. AZ ran
# that way (agent.encode(tok, eqn_ids=eqn) with no features) for the whole
# comparison. They are computed from the same calibration samples PPO uses.
from alphagrad.approx.ppo import _episode_vertex_features as _ep_vfeat  # noqa: E402

VFEAT = _ep_vfeat(_ns, jaxpr, tuple(closed.literals), tuple(xs),
                  eval_samples=ev, argnums=tuple(ARGN))


@eqx.filter_jit
def _carry_init(agent):
    return _cs.init_carry(agent, BASE_TOK, BASE_EQN, BASE_N,
                          window=BASE_W, total_v=TOTAL_V, embd_dim=EMBD)


@eqx.filter_jit
def _carry_advance(agent, enc, vs, vc, dtok, deqn, dcount, owner):
    return _cs.advance(agent, enc, vs, vc, dtok, deqn, dcount, owner,
                       window=MAX_DELTA_TOKENS)


@eqx.filter_jit
def _carry_heads(agent, vs, vc, residual):
    """(vertex_logits (total_v,), vertex_contexts (total_v, E), value (3,))."""
    return _cs.heads(agent, vs, vc, vertex_features=VFEAT,
                     residual_state=residual, preference=None)


@eqx.filter_jit
def _residual_update(agent, residual, v0, ctx_row):
    return agent.update_residual(residual, v0, ctx_row)


def _zero_residual():
    return jnp.zeros((TOTAL_V, EMBD), dtype=jnp.float32)


class Carry:
    """The DEVICE half of a search node (~35 KB on nn256).

    EncCarry(M, I, cumhist, nvalid, pos) + the per-vertex memory
    (sums/counts) + the elimination residual. Params-DEPENDENT, unlike the
    old `_tok_cache`: a carry cached across a `train_step` is silently STALE,
    not wrong-shaped, so carries live for exactly one `gumbel_search` call
    and every decision rebuilds from the committed root carry.
    """

    __slots__ = ("enc", "vs", "vc", "residual")

    def __init__(self, enc, vs, vc, residual):
        self.enc, self.vs, self.vc, self.residual = enc, vs, vc, residual


def _q_of(value3):
    """The 3 value heads [lat, mem, cos] scalarized with the CH weights."""
    v = np.asarray(value3, dtype=np.float64).reshape(-1)
    return float(W4[0] * v[0] + W4[1] * v[1] + W4[3] * v[2])


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
    attention_entropy_diagnostic as _attn_ent_diag,
    _ATTN_ENTROPY_ON,
)
from graphax.sparse.micro_actions import Compress as _GxC, Diag as _GxD, Quant as _GxQ

_LM_TABLES = None
_lm_oracle_cache: dict = {}


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


def learned_micro(ctxs, state, vertex, key):
    """One masked draw from the mainline micro policy for ``vertex`` at the
    graph produced by ``state``; None on END or any mask/replay failure (the
    candidate then enters plain, exactly like rand_micro's None).

    ``ctxs`` are the per-vertex contexts of the CURRENT node, already
    materialised by the carry heads -- there is no second encode any more.
    """
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
    v_idx = int(vertex) - 1
    feats = _axis_feats(env.axis_state_static[v_idx], env.axis_valid_static[v_idx])
    acts, *_r = agent.micro_action_policy.sample(
        ctxs[v_idx], feats, _LM_TABLES, key,
        pair_valid=jnp.asarray(pair, jnp.float32),
        compress_valid=jnp.asarray(comp, jnp.float32),
    )
    # APPROXIMATION-HEAD ENTROPY. MicroActionPolicy.sample returns
    # (actions, sum logp, sum entropy, sum arity, *dists), so _r[1] is the
    # summed sub-episode entropy and _r[2] the summed arity. Normalise by the
    # head's OWN arity, exactly as ppo's evaluate_action_dynamic does, so a
    # longer sub-episode does not inflate the number. Recorded here (before the
    # decode, which can still fail) because the entropy of the draw is real
    # regardless of whether the resulting rows decode to a usable rule.
    try:
        _AP_ENT.append(float(_r[1]) / max(float(_r[2]), 1.0))
    except Exception:
        pass
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

def _skey(state):
    return tuple((int(a), tuple(m) if m else None) for a, m in state)


# ------------------------------------------------------- known dynamics
# ONE elimination per expansion, on the tokenizer's own graph, producing that
# expansion's token DELTA and extending the carry by it. There is no
# `_eliminate_vertex` here and no `copy_g`: `PT.eliminate` advances the
# tokenizer's `ij.graph` through exactly that call, and `PT.branch()` undoes
# the whole speculative chain in one restore.
#
# NO CROSS-DECISION CACHE. `_tok_cache`/`_eval_cache` were params-INdependent
# and never cleared; a carry is params-DEPENDENT, and a carry cached across a
# `train_step` is silently STALE (right shape, wrong weights). Every decision
# rebuilds from the committed root carry, and nothing survives a search.
def _step(node, vertex, micro):
    """Eliminate ``vertex`` (with ``micro``) on PT and return the child node.

    ``node`` is ``(state, carry, ctxs)``; the returned child carries its own
    ``(state, carry)``. The caller decides whether the elimination is
    speculative (inside ``PT.branch()``) or committed.
    """
    state, carry, ctxs = node
    a = int(vertex) - 1                     # VALID is contiguous 1..NV
    spec_row, face_rows, face_skips = _step_wire(a, micro)
    # is_last mirrors the MEASUREMENT's decode exactly (`_callback` uses
    # is_last=(v_idx == last_v_idx) over the FULL order, and
    # `_face_dict_for_vertex`/`_face_transforms_for_order` the same): COMPRESS
    # is honored only on the genuinely TERMINAL vertex.
    #
    # Honoring it mid-plan is what makes the append-only stream
    # non-prefix-stable -- env's documented COMPRESS prefix-property
    # violation, which PPO mitigates with ALPHAGRAD_TOKENS_MID_COMPRESS=0.
    # Here a block is emitted ONCE and never rewritten, so it has to carry
    # the terminal decode from the start or the stream and the measurement
    # describe different graphs.
    #
    # "Terminal" is DETECTED, not counted: eliminating a vertex can make
    # OTHER vertices non-eliminable (nn256: NV=27 valid vertices but 24
    # decisions), so `len(state) == NV - 1` is simply wrong. The last
    # decision is the one taken when exactly one vertex is still legal; the
    # caller asserts the prediction against the graph afterwards, and only a
    # COMPRESS row makes a misprediction observable at all.
    is_last = (len(PT.legal(VALID)) == 1)
    toks, ids = PT.eliminate(vertex, spec_row, face_rows, face_skips,
                             is_last=is_last)
    dt, de, dc = _wire_delta(toks, ids)
    enc2, vs2, vc2 = _carry_advance(
        agent, carry.enc, carry.vs, carry.vc,
        jnp.asarray(dt), jnp.asarray(de), jnp.asarray(dc, jnp.int32),
        jnp.asarray(a, jnp.int32))
    resid2 = _residual_update(
        agent, carry.residual, jnp.asarray(a, jnp.int32), ctxs[a])
    st2 = list(state) + [(a, micro)]
    return (st2, Carry(enc2, vs2, vc2, resid2),
            {"tokens": toks, "eqn_ids": ids, "delta": (dt, de, dc),
             "owner": a, "spec_row": spec_row, "face_rows": face_rows,
             "face_skips": face_skips, "is_last": is_last})


def _assert_terminal_prediction(d):
    """The `is_last` prediction of the step just COMMITTED, against the graph.

    A misprediction only changes bytes when the step carried a COMPRESS row
    (that is the sole `is_last` sensitivity of `decode_vertex_rule_specs`), so
    the guard fires exactly when it matters instead of on every long tail.
    """
    from alphagrad.approx.env import COMPRESS_SENTINEL
    if d["is_last"] or PT.legal(VALID):
        return
    has_compress = (
        bool(np.any(np.asarray(d["spec_row"])[..., 0] == COMPRESS_SENTINEL))
        or bool(np.any(np.asarray(d["face_rows"])[..., 0] == COMPRESS_SENTINEL)))
    if has_compress:
        raise AssertionError(
            "the terminal decision was not predicted as terminal AND carried "
            "a COMPRESS row: its block was tokenized with is_last=False while "
            "the measurement will decode it with is_last=True, so the "
            "observation and the measured graph diverge. (Elimination made "
            "the remaining legal vertices vanish at the same step.)")


def _eval_node(state, carry):
    """``(vertex_logits (total_v,), contexts, scalar value in z-space)``."""
    vlog, ctxs, val = _carry_heads(agent, carry.vs, carry.vc, carry.residual)
    return np.asarray(vlog, dtype=np.float64), ctxs, _q_of(val)


def rollout_value(state, carry, depth):
    """Greedy descent of ``depth`` steps from an already-expanded child.

    DEPTH-FIRST, not lockstep. The old lockstep loop batched one device
    dispatch per level, but interleaving chains is exactly what a single
    stateful tokenizer cannot do: branching would force a cold prefix replay
    per chain per level (O(prefix) eliminations each). One `PT.branch()` per
    chain costs 0.02 ms to enter and 0.17 ms to leave at depth 23 against a
    5-17 ms elimination, so the sequential dispatches are the cheaper trade.

    C1 (kept): EVERY rollout is valued, terminal or not. Filtering terminals
    out and substituting the root value made every candidate's Q the same
    constant at the deciding halvings.
    """
    for _ in range(depth):
        legal = PT.legal(VALID)
        if not legal:
            break
        vlog, ctxs, _v = _eval_node(state, carry)
        la = [int(v) - 1 for v in legal]
        v = legal[int(np.argmax(vlog[la]))]
        state, carry, _d = _step((state, carry, ctxs), v, None)
    _vl, _cx, vz = _eval_node(state, carry)
    return vz

# ---------------------------------------------------------------- golden check
# ACCEPTANCE (a): for an episode, `base ++ concat(per-decision deltas)` must
# equal `env._incremental_stream_tokens` on the committed plan, BITWISE. The
# two producers are the same graphax tokenizer driven by the same decoded
# hooks, so a mismatch means a decode divergence (is_last, face wires) or a
# cache-state dependence -- both silent otherwise. On by default for the first
# ALPHAGRAD_GAZ_GOLD_EPISODES episodes; 0 disables.
_GOLD_EPISODES = int(os.environ.get("ALPHAGRAD_GAZ_GOLD_EPISODES", "1"))
_GOLD_CHECK = _GOLD_EPISODES > 0
_GOLD_FAILS = [0]


def _golden_equivalence(state, stream, seg_ids, ep):
    """Compare the streamed episode against the env's own full-stream builder."""
    global _GOLD_CHECK
    if ep > _GOLD_EPISODES:
        _GOLD_CHECK = False
        return
    from alphagrad.approx.env import (
        _incremental_stream_tokens, decode_vertex_rule_specs)
    order, specs, face_rows, face_skips = plan_wires(state)
    n = len(order)
    tok_rules_by_v = {}
    for k in range(n):
        rules = decode_vertex_rule_specs(
            jaxpr, int(order[k]), specs[k].tolist(), is_last=(k == n - 1))
        if rules:
            tok_rules_by_v[int(order[k])] = (
                make_live_masked_hook(tuple(rules)),)
    ref, ref_ids, _ft, _ls = _incremental_stream_tokens(
        env.config, env.consts, env.args, [int(v) for v in order],
        specs.tolist(), tok_rules_by_v,
        face_rows_list=face_rows.tolist(),
        face_skips_list=face_skips.tolist(),
        honor_last_compress=True,
        face_key=tuple(
            (tuple(int(x) for x in face_rows[k].reshape(-1)),
             tuple(int(x) for x in face_skips[k].reshape(-1)))
            for k in range(n)),
    )
    ok_t = list(ref) == list(stream)
    ok_i = list(ref_ids) == list(seg_ids)
    if ok_t and ok_i:
        print(f"[gaz][gold] ep={ep} BITWISE OK: {len(stream)} tokens, "
              f"{n} decisions (base {BASE_N} + deltas)", flush=True)
        return
    _GOLD_FAILS[0] += 1
    _bad = next((i for i in range(min(len(ref), len(stream)))
                 if ref[i] != stream[i]), min(len(ref), len(stream)))
    raise AssertionError(
        f"[gaz][gold] ep={ep} STREAM MISMATCH: streamed {len(stream)} tokens "
        f"vs env {len(ref)}; first differing index {_bad}; "
        f"eqn_ids match={ok_i}. The search reads a different graph than the "
        f"measurement tokenizes.")


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

# Per-episode POLICY-ENTROPY accumulators, one entry per DECISION; drained to
# their episode means in the wandb block and cleared alongside _MICRO_CHOICES.
#   _VE_ENT : entropy (nats) of the vertex-elimination prior the search acts
#             under -- ppo logs the same quantity as entropy/ve_head.
#   _AP_ENT : arity-normalised entropy of the approximation head.
# CAVEAT: az's approximation head is the PER-VERTEX MicroActionPolicy, while
# ppo's entropy/approx_head is the PER-FACE UnifiedFacePolicy. Both are divided
# by their own action arity, so the curves are comparable in SHAPE but NOT in
# absolute scale -- different action spaces, different alphabet sizes.
_VE_ENT: list = []
_AP_ENT: list = []


# SILENT-FAILURE ASSERT 1 (learned the hard way): an action that does not
# change the OBSERVATION is invisible to search. A per-vertex skip once
# tokenized IDENTICALLY to no-skip, so the value net scored the two the same
# and stable-argsort ties always kept the plain variant -- the action existed
# and was never chosen. Every halving phase now checks that candidates with
# distinct WIRE bytes have distinct TOKEN bytes; a non-zero steady state on
# `search/tied_candidates` is a bug, not noise.
_TIED_CANDIDATES = [0]
_TIED_TOTAL = [0]


def _check_distinct_observations(cands):
    by_tokens: dict = {}
    tied = 0
    for c in cands:
        wire = (int(c["v"]), c["_wire"])
        tok = c["_tokbytes"]
        prev = by_tokens.get(tok)
        if prev is None:
            by_tokens[tok] = wire
        elif prev != wire:
            tied += 1
    _TIED_CANDIDATES[0] += tied
    _TIED_TOTAL[0] += len(cands)
    return tied


# ---------------------------------------------------------------- Gumbel root search
def gumbel_search(state, carry, rng):
    """One decision. ``PT`` is positioned at the COMMITTED prefix on entry and
    is left there on exit -- every expansion runs inside ``PT.branch()``."""
    legal = PT.legal(VALID)
    vlog, ctxs, v_root = _eval_node(state, carry)
    la = np.array([int(v) - 1 for v in legal], dtype=np.int32)
    logits = vlog[la] - vlog[la].max()
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
                lm = learned_micro(ctxs, state, legal[int(ci)],
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
    # deeper rollout — the budget buys more accurate Q, not duplicates).
    surv = list(cands)
    depth = A.rollout_depth
    _phase = 0
    while True:
        for c in surv:
            # ONE branch per candidate covers its expansion AND its whole
            # rollout chain: _Snapshot.__exit__ truncates the append-only
            # lists back to their entry length, so any number of
            # eliminations inside are undone together.
            with PT.branch():
                st2, cy2, d = _step((state, carry, ctxs), c["v"], c["micro"])
                if _phase == 0:
                    c["_wire"] = (d["spec_row"].tobytes(),
                                  d["face_rows"].tobytes(),
                                  d["face_skips"].tobytes())
                    c["_tokbytes"] = (
                        np.asarray(d["tokens"], np.int32).tobytes(),
                        np.asarray(d["eqn_ids"], np.int32).tobytes())
                c["q"].append(rollout_value(st2, cy2, depth))
        if _phase == 0:
            _check_distinct_observations(cands)
        _phase += 1
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
    # VE-HEAD ENTROPY (nats) of the prior this decision actually acted under.
    # Free: the softmaxed vertex prior is already materialised for the
    # completed-Q target. One sample per decision; the episode mean is logged.
    _VE_ENT.append(float(-np.sum(_prior * np.log(_prior + 1e-12))))
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
    # ``ctxs`` rides out so the caller can commit the chosen action without a
    # second heads pass (the residual update needs this node's context row).
    return chosen, pi, la, legal, ctxs

# ---------------------------------------------------------------- training
def loss_fn(agent, enc_M, enc_I, enc_ch, enc_nv, enc_pos, vmem_s, vmem_c,
            resid, dtok, deqn, dcnt, owner, la_pad, la_mask, pi_pad,
            vtgt, vmask):
    """Vertex CE + value MSE, re-derived from the STORED PRE-step carry.

    §7b of the PPO design, applied here: the replay stores the carry synced to
    the PREVIOUS step's delta plus this step's delta, and the loss reproduces
    the step's encoding by the SAME extension the rollout ran. Gradient
    reaches palimpsa through that ``encode_extend``; it is truncated at the
    stored carry, exactly as in PPO.
    """
    from alphagrad.approx.ppo import EncCarry

    def per(M, I, ch, nv, pos, vs, vc, rs, dt, de, dc, ow,
            la, lam, pi, vt, vm):
        carry = EncCarry(M=M, I=I, cumhist=ch, nvalid=nv, pos=pos)
        _c2, vs2, vc2 = _cs.advance(agent, carry, vs, vc, dt, de, dc, ow,
                                    window=MAX_DELTA_TOKENS)
        vlog, _ctx, v3 = _cs.heads(agent, vs2, vc2, vertex_features=VFEAT,
                                   residual_state=rs, preference=None)
        lg = vlog[la]
        lg = jnp.where(lam > 0.5, lg, -jnp.inf)   # -1e9 collided with a sentinel
        logp = jax.nn.log_softmax(lg)
        ce = -jnp.sum(jnp.where(lam > 0.5, pi * logp, 0.0))
        # Predicted CH vector from the 3 heads: [lat, mem, 0 (flops: no
        # head, _CH_ACTIVE masks it), cos]; targets vt are PopArt-normalised.
        pred = jnp.stack([v3[0], v3[1], jnp.zeros_like(v3[0]), v3[2]])
        vl = jnp.sum(vm * _CH_ACTIVE * (pred - vt) ** 2)
        return ce + 0.5 * vl

    return jnp.mean(jax.vmap(per)(
        enc_M, enc_I, enc_ch, enc_nv, enc_pos, vmem_s, vmem_c, resid,
        dtok, deqn, dcnt, owner, la_pad, la_mask, pi_pad, vtgt, vmask))

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
    at MODULE level, because the search functions (`gumbel_search`, `_step`,
    `_eval_node`, `measure`, `scalarize`) read it through module globals —
    moving it into a function would silently turn those reads stale. For the
    same reason `gumbel_search` still reads `A.n_candidates` / `A.rollout_depth`
    off the module-level `A`, not off `args` (same values; see `main`).

    `global agent, opt_state` is LOAD-BEARING, not decoration: this loop REBINDS
    `agent` (PopArt head rescale, then `train_step`) and the jitted carry
    helpers must see the rebound agent. Without it both names would become
    `_run` locals and the net would either go stale in the search or raise
    UnboundLocalError.
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
    # Logging must never kill training, but it must never be SILENT either: a
    # swallowed exception here used to cost the episode's row, poison
    # approx_prob/* (uncleared counters) and skew time/sec_per_episode (stale
    # _t_prev) with nothing in the log to say so. Rate-limited so a systematic
    # failure cannot flood stdout.
    _log_fails = [0]

    def _log_failure(where, exc):
        _log_fails[0] += 1
        n = _log_fails[0]
        if n <= 3 or n % 50 == 0:
            print(f"[gaz] wandb logging FAILED in {where} "
                  f"(occurrence {n}): {type(exc).__name__}: {exc}",
                  flush=True)
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

    # --- 4b. PopArt WARM-START ------------------------------------------
    # Uniform on the standardised (ppo) form. Without it the normaliser starts
    # at (0, 1) and the first EMA step is fed M == 1 sample, whose variance is
    # exactly 0 -- sigma clips to sigma_min (0.1, or 0.2 on the quality
    # channels) while the raw memory channel is ~1e9, so the critic's targets
    # are ~1e10 for the first episodes.
    #
    # WHAT IS SEEDED: the RAW terminal 4-vector. Unlike ppo (which regresses a
    # DISCOUNTED RETURN and therefore has to build Monte-Carlo returns for its
    # seed) az's value target IS the terminal raw vector -- see ``vt`` below,
    # ``(raw4 - mu) / sigma``. Seeding on anything else would be seeding in a
    # different space from the one that is updated.
    #
    # Plans are drawn UNIFORMLY AT RANDOM over the legal set (gumbel_search is
    # bypassed): the point is the measurement SCALE, and an untrained search is
    # both slower and no more representative. Each episode IS logged to wandb
    # (minus a "loss" key -- no train step ran), uniform with ppo's warm-up,
    # but they do not advance ``n_meas``: the warm start is not spent from the
    # --total-measurements budget, exactly as ppo's warm-up rollouts do not
    # advance its episode counter.
    _pie = int(getattr(args, "popart_init_episodes", 0) or 0)
    if _pie > 0:
        _seed_rows = []
        for _wi in range(_pie):
            _wst = []
            # The warm start needs a PLAN, not an observation -- no carry, no
            # network. It still eliminates on the tokenizer's graph (the only
            # graph model there is) inside a branch, so the committed
            # tokenizer is untouched.
            with PT.branch():
                while True:
                    _wlegal = PT.legal(VALID)
                    if not _wlegal:
                        break
                    _wv = _wlegal[int(rng.integers(len(_wlegal)))]
                    _wm = rand_micro(rng)
                    _wa = int(_wv) - 1
                    _wrow, _wfr, _wfs = _step_wire(_wa, _wm)
                    PT.eliminate(_wv, _wrow, _wfr, _wfs,
                                 is_last=(len(_wlegal) == 1))
                    _wst.append((_wa, _wm))
            _wraw = measure(_wst)
            if _wraw is None:
                print(f"[popart-init] episode {_wi + 1}/{_pie} measure FAILED",
                      flush=True)
                continue
            _wraw = np.asarray(_wraw, dtype=np.float64)
            if not np.isfinite(_wraw).all():
                print(f"[popart-init] episode {_wi + 1}/{_pie} non-finite",
                      flush=True)
                continue
            _seed_rows.append(_wraw)
            # Log the warm-up episode. Uniform with ppo, which now also logs
            # its warm-start rollouts: the measurement IS real and the wandb
            # step counter must advance so these episodes are not a silent gap.
            # No "loss" key -- no train step ran, and a NaN loss would wreck
            # the panel's y-range for the whole run.
            if wb is not None:
                try:
                    _wnow = time.time()
                    _wlog = {
                        "popart_init/warmup_episode": 1,
                        "this_lat_us": float(_wraw[0]) / 1e3,
                        "measure/peak_memory_mb": float(_wraw[1]) / 1e6,
                        "time/sec_per_episode": _wnow - _t_prev,
                        "time/wall_seconds": _wnow - _t_start,
                        "time/wall_minutes": (_wnow - _t_start) / 60.0,
                    }
                    for _wj, _wnm in enumerate(REWARD_NAMES):
                        _wlog[f"mean_{_wnm}"] = float(LAST_FULL_REWARD[_wj])
                    wb.log(_wlog)
                except Exception as _lexc:
                    _log_failure("popart-init warm-up", _lexc)
                else:
                    # only once the row is actually on the wire
                    _t_prev = _wnow
        print(f"[popart-init] {_pie} random-plan episodes -> "
              f"{len(_seed_rows)}/{_pie} usable terminal measurements",
              flush=True)
        # >= 2 rows or keep the zero init: a single row has variance 0 on every
        # channel, so it would seed nothing but a mean and leave every sigma on
        # the floor -- the exact failure this block exists to remove.
        if len(_seed_rows) >= 2:
            _R = np.stack(_seed_rows)
            popart.seed(_R)
            _mu0 = _R.mean(axis=0)
            _sd0 = _R.std(axis=0)
            _sd_eff = np.maximum(_sd0, np.asarray(popart.sigma_min, np.float64))
            _zvar = (((_R - _mu0) / _sd_eff) ** 2).mean(axis=0)
            _cfg = {}
            for _k, _nm in enumerate(CH):
                if _sd0[_k] <= 1e-12:
                    _fl = "  <-- CONSTANT in the sample, left COLD"
                elif _sd0[_k] < _sd_eff[_k]:
                    _fl = "  <-- sigma FLOORED by popart sigma_min"
                else:
                    _fl = ""
                print(f"[popart-init]   {_nm}: mu={_mu0[_k]:.6g} "
                      f"sigma={_sd0[_k]:.6g} norm_var={_zvar[_k]:.4f}{_fl}",
                      flush=True)
                _cfg[f"popart_init_mu_{_nm}"] = float(_mu0[_k])
                _cfg[f"popart_init_sigma_{_nm}"] = float(_sd0[_k])
                _cfg[f"popart_init_normvar_{_nm}"] = float(_zvar[_k])
            _cfg["popart_init_samples"] = int(_R.shape[0])
            if wb is not None:
                try:
                    wb.config.update(_cfg, allow_val_change=True)
                except Exception:
                    pass
        else:
            print("[popart-init] too few usable measurements; keeping zero "
                  "init", flush=True)

    # --- 5. loop: act/search -> measure -> popart -> pareto -> train ---
    while n_meas < args.total_measurements:
        ep += 1
        # A fresh episode = a fresh tokenizer at the base, and a fresh carry
        # built from the base stream under the CURRENT weights. The carry is
        # params-dependent, so it can never outlive a train_step.
        PT.reset()
        state = []
        carry = Carry(*_carry_init(agent), _zero_residual())
        steps = []
        _memlog = os.environ.get("ALPHAGRAD_GAZ_MEMLOG", "0") == "1"
        _dstep = 0
        # Golden-equivalence accumulator: base ++ concat(per-decision deltas).
        _gold_stream = list(_BASE_TOKS)
        _gold_ids = list(_BASE_IDS)
        while True:
            legal = PT.legal(VALID)
            if not legal:
                break
            chosen, pi, la, legal, ctxs = gumbel_search(state, carry, rng)
            _mk = chosen.get("micro")
            _MICRO_CHOICES["none" if _mk is None else
                           {"q": "quant", "d": "diag", "c": "compress",
                            "s": "skip"}.get(_mk[0], "other")] += 1
            # PRE-step snapshot (§7b): the carry synced to the PREVIOUS step's
            # delta, plus THIS step's delta. The loss re-derives this step's
            # encoding by the same extension.
            _pre = carry
            state, carry, d = _step((state, carry, ctxs),
                                    chosen["v"], chosen["micro"])
            _assert_terminal_prediction(d)
            _gold_stream += list(d["tokens"])
            _gold_ids += list(d["eqn_ids"])
            steps.append({
                "enc_M": np.asarray(_pre.enc.M), "enc_I": np.asarray(_pre.enc.I),
                "enc_ch": np.asarray(_pre.enc.cumhist),
                "enc_nv": np.asarray(_pre.enc.nvalid),
                "enc_pos": np.asarray(_pre.enc.pos),
                "vmem_s": np.asarray(_pre.vs), "vmem_c": np.asarray(_pre.vc),
                "resid": np.asarray(_pre.residual),
                "dtok": d["delta"][0], "deqn": d["delta"][1],
                "dcnt": np.int32(d["delta"][2]), "owner": np.int32(d["owner"]),
                "la": la.copy(), "pi": pi.copy()})
            _dstep += 1
            if _memlog and _dstep % 5 == 0:
                try:
                    ms = jax.devices()[0].memory_stats()
                    print(f"[gaz][mem] ep={ep} decision={_dstep}/{len(legal)+_dstep} "
                          f"peak={ms.get('peak_bytes_in_use',0)/1e9:.2f}GB "
                          f"curr={ms.get('bytes_in_use',0)/1e9:.2f}GB "
                          f"delta={int(d['delta'][2])}", flush=True)
                except Exception:
                    pass
        if _GOLD_CHECK:
            _golden_equivalence(state, _gold_stream, _gold_ids, ep)
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
        # No cache to invalidate: carries are params-dependent and are rebuilt
        # from the base at every episode, so a rescaled value head cannot be
        # read through a stale one.
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
            # fixed shapes (delta buffers are MAX_DELTA_TOKENS-wide; legal set
            # <= NV) -> train_step compiles ONCE instead of re-jitting as
            # episode lengths vary
            MAXLA = NV
            def pad(x, n, v=0):
                return np.pad(x, (0, n - len(x)), constant_values=v)
            def stk(k):
                return jnp.asarray(np.stack([s[k] for s in flat]))
            enc_M, enc_I = stk("enc_M"), stk("enc_I")
            enc_ch, enc_nv, enc_pos = stk("enc_ch"), stk("enc_nv"), stk("enc_pos")
            vmem_s, vmem_c, resid = stk("vmem_s"), stk("vmem_c"), stk("resid")
            dtok, deqn = stk("dtok"), stk("deqn")
            dcnt, owner = stk("dcnt"), stk("owner")
            la_p = jnp.asarray([pad(s["la"], MAXLA) for s in flat])
            la_m = jnp.asarray([pad(np.ones(len(s["la"])), MAXLA) for s in flat])
            pi_p = jnp.asarray([pad(s["pi"], MAXLA) for s in flat])
            vt = jnp.asarray([(s["raw4"] - popart.mu) / popart.sigma for s in flat])  # PopArt-normalised
            vm = jnp.asarray(np.broadcast_to(np.abs(W4) > 0, (len(flat), 4)).astype(np.float32))
            _cols = (enc_M, enc_I, enc_ch, enc_nv, enc_pos, vmem_s, vmem_c,
                     resid, dtok, deqn, dcnt, owner, la_p, la_m, pi_p, vt, vm)
            # M5: resample the minibatch EVERY epoch. Taking one fixed 64-sample
            # draw and hitting it ``train_epochs`` times overfits that draw and
            # discards the rest of the replay for this update.
            _bs = int(os.environ.get("ALPHAGRAD_GAZ_BATCH", "64"))
            for _ in range(args.train_epochs):
                idx = jnp.asarray(rng.permutation(len(flat))[:_bs])
                batch = tuple(x[idx] for x in _cols)
                agent, opt_state, L = train_step(agent, opt_state, batch)
            L = float(L)
        else:
            L = float("nan")
        b = best["raw"]
        try:
            _scal = float(scalarize(raw))
        except Exception:
            _scal = float("nan")
        # PPO-COMPARABLE weighted score. PPO reports a convex combination of
        # normal CDFs Phi(z) in [0,1] (the running-distribution percentile of
        # this episode's channel value); AZ used to report the raw z-SUM under
        # the same key, which is unbounded and not the same quantity. Compute
        # PPO's form here so one panel means one thing.
        # sign(W4) orients each channel so higher = better: W4 is
        # [-w_cmp, -w_mem, 0, +w_acc], and raw holds POSITIVE cost magnitudes.
        try:
            from math import erf as _erf
            _mu_v = np.asarray(popart.mu, np.float64).reshape(-1)
            _sig_v = np.asarray(popart.sigma, np.float64).reshape(-1)
            _w4 = np.asarray(W4, np.float64).reshape(-1)
            _raw_v = np.asarray(raw, np.float64).reshape(-1)
            _z = (_raw_v - _mu_v) / np.maximum(_sig_v, 1e-8)
            _z = _z * np.sign(_w4)          # higher = better on every channel
            _phi_v = np.array(
                [0.5 * (1.0 + _erf(float(v) / np.sqrt(2.0))) for v in _z],
                dtype=np.float64)
            _aw = np.abs(_w4)
            _awsum = float(np.sum(_aw))
            _wn_v = (_aw / _awsum) if _awsum > 0 else np.zeros_like(_aw)
            _wmr = float(np.sum(_phi_v * _wn_v))
        except Exception:
            _phi_v, _wmr = None, float("nan")
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
            _now = time.time()
            _log = None
            try:
                _log = {
                    # --- AZ-native ---
                    "ep": ep, "n_meas": n_meas,
                    "best_scalar": best["scalar"],
                    "best_lat_us": b[0] / 1e3, "best_cos": b[3],
                    "this_lat_us": raw[0] / 1e3,
                    # --- PPO-parity keys (same names => same panels) ---
                    "pareto/hypervolume": _pareto.hypervolume(),
                    "pareto/archive_size": len(_pareto.pts),
                    "pareto/size": len(_pareto.pts),   # legacy AZ key
                    "weighted_mean_return": _wmr,
                    # the raw z-sum kept under its own honest name
                    "scalarized_return": float(_scal),
                    # MEASURED peak, in MB. Previously logged under PPO's
                    # "measure/xla_peak_memory" key, which on that arm was a
                    # STATIC estimate in BYTES -- two different quantities in
                    # two different units sharing one panel.
                    "measure/peak_memory_mb": raw[1] / 1e6,
                    "time/episode": ep,
                    "time/sec_per_episode": _now - _t_prev,
                    "time/wall_seconds": _now - _t_start,
                    "time/wall_minutes": (_now - _t_start) / 60.0,
                }
                # B3: while the PopArt gate (popart.n_updates < 8) still
                # holds no train step runs and L is NaN. wandb records NaN as a
                # DATA POINT, which wrecks the loss panel's y-range for the
                # whole run; omitting the key leaves a clean gap instead.
                if math.isfinite(L):
                    _log["loss"] = float(L)
                for _j, _nm in enumerate(REWARD_NAMES):
                    _log[f"mean_{_nm}"] = float(LAST_FULL_REWARD[_j])
                # --- approximation telemetry (identical keys to the PPO runs)
                _tot = sum(_MICRO_CHOICES.values()) or 1
                # ``skip`` is the REALIZED fraction of vertex decisions
                # that chose ("s",) -- every face of that vertex measured with
                # graphax.SKIP_FACE. Same key as PPO's, NOT the same
                # denominator: PPO reports the per-FACE skip probability of the
                # UnifiedFaceHead, az the per-VERTEX choice frequency. The two
                # curves share a panel and are comparable in trend only.
                for _nm in ("none", "diag", "compress", "quant", "skip"):
                    _log[f"approx_prob/{_nm}"] = _MICRO_CHOICES[_nm] / _tot
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
                # Mirror to stdout (same switch ppo.py uses), so a
                # --wandb disabled probe or a dead run's log still shows
                # what the policy chose. NOTE the quantities differ from
                # PPO's despite the identical keys: PPO counts realized
                # usage per FACE, AZ counts the chosen class per VERTEX.
                if os.environ.get("ALPHAGRAD_DEBUG_APPROX_PROB", "0") == "1":
                    _ap = {_k2: _v2 for _k2, _v2 in _log.items()
                           if _k2.startswith(("approx_prob/",
                                              "approx_applied/",
                                              "approx_skipped/"))}
                    if _ap:
                        print("[approx per-vertex] " + " ".join(
                            f"{_k2.split('/')[-1]}={float(_v2):.4g}"
                            for _k2, _v2 in sorted(_ap.items())), flush=True)
                # per-head percentile companions + the PopArt state that
                # produced them. AZ channel order is [lat, peak, flops, cos];
                # flops has W4 == 0 and no head, so it is skipped.
                try:
                    for _hi, _hn in ((0, "latency"), (1, "mem"), (3, "cos")):
                        if _phi_v is not None and _hi < _phi_v.shape[0]:
                            _log[f"weighted_mean_{_hn}"] = float(_phi_v[_hi])
                        _log[f"popart/mu_{_hn}"] = float(
                            np.asarray(popart.mu).reshape(-1)[_hi])
                        _log[f"popart/sigma_{_hn}"] = float(
                            np.asarray(popart.sigma).reshape(-1)[_hi])
                except Exception:
                    pass
                # Episode-mean policy entropies. Keys match ppo.py's so the
                # two arms share a panel -- see the module-level CAVEAT: the
                # VE head is the same distribution on both, the approximation
                # head is per-VERTEX here and per-FACE on ppo. Omitted (not
                # logged as 0) when the head is not in play, so an absent panel
                # means "not applicable" rather than "collapsed".
                if _VE_ENT:
                    _log["entropy/ve_head"] = float(np.mean(_VE_ENT))
                if _AP_ENT:
                    _log["entropy/approx_head"] = float(np.mean(_AP_ENT))
                # entropy/palimpsa: the ENCODER's mean attention-row entropy on
                # the ROOT state -- now the BASE stream itself (the root state
                # IS the base under the append-only tokenizer), same input
                # every episode, so the curve isolates the encoder's drift.
                # Representation-collapse diagnostic -- NOT a policy entropy,
                # NOT comparable in units to the two keys above. Same helper
                # ppo uses, so the two arms measure the identical quantity.
                if _ATTN_ENTROPY_ON:
                    try:
                        _pe = float(_attn_ent_diag(
                            agent, BASE_TOK, BASE_EQN,
                            env.axis_state_static, env.axis_valid_static))
                        if _pe == _pe:      # not NaN
                            _log["entropy/palimpsa"] = _pe
                    except Exception:
                        pass
                # SILENT-FAILURE 1 telemetry: candidates whose WIRE bytes
                # differ but whose TOKEN bytes do not. Non-zero steady state
                # means an action the search cannot see.
                _log["search/tied_candidates"] = int(_TIED_CANDIDATES[0])
                _log["search/tied_fraction"] = (
                    _TIED_CANDIDATES[0] / max(_TIED_TOTAL[0], 1))
                # Mirror the three head entropies to stdout under the same
                # switch the approximation telemetry uses, so a --wandb
                # disabled probe (or a dead run's log) still shows them.
                if os.environ.get("ALPHAGRAD_DEBUG_APPROX_PROB", "0") == "1":
                    _ek = {_k3: _v3 for _k3, _v3 in _log.items()
                           if _k3.startswith("entropy/")}
                    if _ek:
                        print("[entropy] " + " ".join(
                            f"{_k3.split('/')[-1]}={float(_v3):.4g}"
                            for _k3, _v3 in sorted(_ek.items())), flush=True)
            except Exception as _lexc:
                # NARROW. A failure while BUILDING the payload loses this
                # episode's row and nothing else -- it no longer takes the
                # state resets with it, and it is no longer invisible.
                _log_failure("episode payload build", _lexc)
                _log = None
            if _log is not None:
                try:
                    wb.log(_log)
                except Exception as _lexc:
                    _log_failure("wb.log", _lexc)
                else:
                    # ONLY after the row landed. If it did not, the counters
                    # keep accumulating and _t_prev keeps its old mark, so the
                    # NEXT successful row covers both episodes honestly instead
                    # of reporting a two-episode average as one episode.
                    _MICRO_CHOICES.clear()
                    _TIED_CANDIDATES[0] = 0
                    _TIED_TOTAL[0] = 0
                    _VE_ENT.clear()
                    _AP_ENT.clear()
                    _t_prev = _now

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
