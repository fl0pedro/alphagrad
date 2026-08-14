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
  * GUMBEL (Danihelka 2022) x SAMPLED (Hubert 2021): root Gumbel-top-m
    without replacement over the prior logits, sequential halving with
    WIDENING (#93: every evaluation is a depth-0 value bootstrap; round p
    gives each survivor 2**p new face-sequence draws from beta, so the
    budget buys lower-variance weighted Q, never depth-mixed Q -- legacy
    progressive deepening survives behind ALPHAGRAD_GAZ_DEEPEN=1 with
    same-depth targets), action chosen by argmax(g + logits + sigma(q)),
    policy trained by CE to the COMPLETED-Q improved target
    softmax(logits + sigma(completed_q)) over the legal set.
  * APPROXIMATION = PER FACE, from PPO's UnifiedFacePolicy, reading the SAME
    live per-face token chunks (`common/face_driver.py` -> `live_faces.py`)
    PPO's head reads. It is drawn ONCE per COMMITTED decision (~NV x
    mean_faces chunks per episode); the SEARCH proposes and scores at VERTEX
    granularity, because running the face loop at every sequential-halving
    expansion would be ~12x PPO's chunk count. Face variants inside the
    search are a later stage. The per-VERTEX micro-action space is gone.
  * The approximation head is TRAINED as the beta factor of a SAMPLED
    AlphaZero (Hubert et al. 2021) composite action a = (v, F): the search
    draws K i.i.d. face sequences F_k ~ beta per surviving vertex, weighs
    them w_k = rho_k * exp(sigma(q_k)), and the loss is the face
    cross-entropy - lambda_f sum_v pi'_ve(v) sum_k w_hat_{v,k} log beta(F_k)
    (common/sampled_az.py), with log beta recomputed through
    `Agent._face_replay` ("gradient reaches palimpsa through this scan").
    The old unclipped off-policy REINFORCE face term is GONE (#95), and so
    is its flat-coefficient arity problem (#76).
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
    consume_per_face_stats, MAX_DELTA_TOKENS, _record_delta_truncation,
    MAX_RULES_PER_VERTEX, MAX_AXES_PER_VERTEX)

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

# ------------------------------------------------------- 1. config/env (measure_worker pattern)
TASK = A.task; DSET = A.dataset
# ALPHAGRAD_GAZ_MEASURE_GRAD=0: measure the RAW JACOBIAN (parity with the
# v45 PPO arms, which run without --measure-grad); default 1 keeps the
# legacy gradient-pipeline target for existing scripts.
_GAZ_MGRAD = os.environ.get("ALPHAGRAD_GAZ_MEASURE_GRAD", "1") == "1"
LOSS = scalar_loss_fn(get_fn(TASK)) if _GAZ_MGRAD else get_fn(TASK)
# ORDER-ONLY / EXACT ARM (--no-approx-head): no plan can approximate anything,
# so every plan returns the EXACT gradient and the quality channel is a
# CONSTANT (measured on TLM order-only: cos=+0.885 on every episode). It
# therefore contributes zero gradient while the 200-step loss-drop walk that
# produces it costs 200 executions of the plan -- twice the entire latency
# budget of 100. Resolve the env default to "none" and say so; an explicit
# ALPHAGRAD_QUALITY_METRIC is always honoured.
if (os.environ.get("ALPHAGRAD_QUALITY_METRIC", "auto").strip().lower()
        in ("", "auto") and bool(A.no_approx_head)):
    os.environ["ALPHAGRAD_QUALITY_METRIC"] = "none"
    print("[gaz] ORDER-ONLY arm (--no-approx-head): the quality channel is "
          "constant by construction, so it is NOT computed "
          "(ALPHAGRAD_QUALITY_METRIC=none). Set it explicitly to override.",
          flush=True)
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

# FACE WIDTH. `from ... import MAX_FACES as ENV_MAX_FACES` binds a VALUE at
# import time, so the module default would stick even after the per-graph
# bound is configured -- every face wire, the head's max_faces and the replay
# zero-fills all read this. Derive and rebind BEFORE anything builds a shape
# from it, exactly as ppo.main() does under --face-actions.
from alphagrad.approx import env as _env_mod              # noqa: E402
_FACE_BOUND = _env_mod.derived_max_faces(
    closed.jaxpr, ARGN, closed.literals, xs)
_env_mod.configure_max_faces(_FACE_BOUND)
ENV_MAX_FACES = _env_mod.MAX_FACES
print(f"[gaz] face width: derived bound {_FACE_BOUND} "
      f"(max_v |anc|x|desc|; in force: {ENV_MAX_FACES})", flush=True)

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
# ``VALID.index(v)`` while every head indexes by ``v - 1`` (vertex logits,
# vertex contexts, vmem slots, axis_state rows). They agree only while VALID
# is contiguous 1..NV. Fail loudly rather than silently scoring the wrong
# vertex on a task whose jaxpr has an early output eqn.
assert VALID == list(range(1, NV + 1)), (
    f"VALID must be contiguous 1..{NV} for the vertex-index conventions to "
    f"agree; got {VALID[:8]}...")
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
# Per-token owning VERTEX, from the SAME env method PPO uses -- keying the
# per-vertex memory differently on the two arms is exactly the class of
# divergence this file exists to avoid.
try:
    BASE_OWN = env.base_owners()
except Exception:
    BASE_OWN = None
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
# #86: this defaulted to 0.2 while PPO passes its scalar popart_sigma_min
# (0.1) for EVERY channel, so AZ damped the quality advantage ~2x relative
# to PPO -- and did it exactly when quality converges (var -> 0), which is
# when that channel carries the reward decision. Defaulting to the base
# floor makes the arms agree; the knob stays for a deliberate re-raise.
_sig_min_qual = float(os.environ.get(
    "ALPHAGRAD_POPART_SIGMA_MIN_QUALITY", str(_sig_min_base)))
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


# --reward-mode mult parity (owner 2026-08-09): under
# ALPHAGRAD_REWARD_MODE=mult the ONE scalar everything consumes --
# search leaf values, completed-Q, CE draw weights, replay and the
# `scalarized_return` panel -- is the cosine-gated cheapness, so AZ
# optimizes exactly what PPO's mult arm optimizes. PopArt keeps
# normalizing the VALUE net targets downstream; the gate replaces the
# additive W4 z-sum, not the normalizer.
if os.environ.get("ALPHAGRAD_REWARD_MODE", "additive") == "mult":
    from alphagrad.approx.common.sampled_az import mult_gate_scalar
    _GATE_TAU = float(os.environ.get("ALPHAGRAD_GAZ_GATE_TAU", "0.5"))
    _GATE_W = float(os.environ.get("ALPHAGRAD_GAZ_GATE_W", "40.0"))
    _ADEG_P = float(os.environ.get(
        "ALPHAGRAD_GAZ_ANTI_DEGEN_P", "2.0"))
    _ADEG_TAU = float(os.environ.get(
        "ALPHAGRAD_GAZ_ANTI_DEGEN_TAU", "0.05"))

    def scalarize(raw4):  # noqa: F811 -- deliberate mult-mode override
        return mult_gate_scalar(raw4, _GATE_TAU, _GATE_W, _ADEG_P,
                                _ADEG_TAU)

    print(f"[gaz] REWARD MODE mult: g(cos>tau={_GATE_TAU}) x "
          f"max(0, {_GATE_W} - symlog costs); anti-degen P={_ADEG_P} "
          f"below cos={_ADEG_TAU} (PPO _apply_mult_gate parity)",
          flush=True)

# Pareto archive over the 3 focus OBJECTIVES {latency, xla_peak, cosine} in RAW
# native units (kept SEPARATE from PopArt -- never the normalized values). The
# archive MAXIMISES, so feed a sign-oriented raw vector (negate the cost
# channels, keep cosine) and select the lat/peak/cos slots via obj_idx (flops
# excluded). Exactly 3 objectives => hypervolume() uses the exact 2/3-D sweep,
# not the >=4-D normalized Monte-Carlo estimate.
_PSGN = np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float64)   # over [lat,peak,flops,cos]
_pareto = ParetoArchive(["latency_ns", "peak_memory", "cosine_sim"], [0, 1, 3])

# ------------------------------------------------- the APPROXIMATION space
# PER-VERTEX APPROXIMATIONS ARE GONE (owner: "we won't be keeping the vertex
# based approximations"). Deleted with them: rand_micro, micro_str, _rules_of,
# _tok_rules_of, seq_of, QD, GAZ_MICRO, MICRO_P, MAXAX, FACS, GAZ_SKIP,
# K_MICRO, learned_micro, _ctx_fwd, _LM_TABLES, GAZ_MICRO_LEARNED,
# ALPHAGRAD_GAZ_SEARCH_HOOKS and its M6 hook branch -- and the crc32 SKIP_FACE
# tag hack, which existed only because VEJaxpr had no way to render a skip;
# the path tokenizer emits a real ``approx SKIP`` header, so the opaque tag
# was dead weight AND the reason the search could not see a skip at all.
#
# AZ eliminates at the VERTEX level and approximates at the FACE level,
# through the SAME UnifiedFacePolicy PPO trains. The per-vertex rule rows are
# therefore ALWAYS the exact END rows, exactly as ppo.py sets them under
# --live-faces ("No per-vertex head: the vertex rules are ALWAYS the exact END
# rows -- approximation is purely per-face").
EXACT_SPEC_ROW = np.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
EXACT_SPEC_ROW[:, 2] = 0
EXACT_FACE_ROWS = np.full((int(ENV_MAX_FACES), FACE_SLOTS, 3), -1, dtype=np.int32)
EXACT_FACE_SKIPS = np.zeros((int(ENV_MAX_FACES),), dtype=np.int32)


# #79: the EXACT arm switch, resolved once from --no-approx-head. When set,
# the approximation head is not built, the rollout skips _face_plan, and the
# loss drops its face term -- so AZ searches the elimination ORDER only, the
# counterpart of PPO fq_v47e.
_EXACT_ARM = bool(A.no_approx_head)


# ---------------------------------------------------------------- the WIRE
# ONE builder for the arrays that go to the tokenizer, the measurement AND the
# dumps. Previously the tokenizer got graphax rule OBJECTS (VEJaxpr
# transforms) while `measure` built spec ROWS from `build_order_specs`; two
# encoders of the same decision is how they drift.
#
# A plan step is ``(action_idx, face_rows, face_skips)``. There is no
# per-vertex rule any more, so `build_order_specs` (whose entire job was
# parsing micro CALL STRINGS) leaves this path: the order is the action
# indices + 1 and the specs are the constant exact rows. The face arrays come
# straight off `to_env_action_dynamic`'s translator, so AZ and PPO write
# IDENTICAL wire bytes for the same FaceAction.
def plan_wires(state):
    """A whole plan -> ``(order, specs, face_rows, face_skips)``."""
    n = len(state)
    order = np.array([int(a) + 1 for a, _fr, _fs in state], dtype=np.int32)
    specs = np.broadcast_to(
        EXACT_SPEC_ROW, (n,) + EXACT_SPEC_ROW.shape).copy()
    if n:
        face_rows = np.stack([np.asarray(fr, np.int32) for _a, fr, _fs in state])
        face_skips = np.stack([np.asarray(fs, np.int32) for _a, _fr, fs in state])
    else:
        face_rows = np.zeros((0,) + EXACT_FACE_ROWS.shape, dtype=np.int32)
        face_skips = np.zeros((0,) + EXACT_FACE_SKIPS.shape, dtype=np.int32)
    return order, specs, face_rows, face_skips


def state_to_json(state):
    """Serialisable plan. The FACE WIRES ride along -- a dump that stored only
    the order (or only per-vertex micros, as it used to) is NOT replayable
    under per-face approximation: re-measuring it would build the EXACT plan
    and silently report a different cosine."""
    return [{"vertex": int(a) + 1,
             "face_rows": np.asarray(fr, np.int32).tolist(),
             "face_skips": np.asarray(fs, np.int32).tolist()}
            for a, fr, fs in state]


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
    """Terminal REAL measurement -> [lat, xla_peak, flops, cos] or None."""
    global _ms_client
    if _MS:
        # ALPHAGRAD_MEASURE_SERVER=1 used to measure in an isolated SUBPROCESS
        # over a CALL-STRING sequence (`measure_seq`). That wire has no
        # per-face representation at all, so under per-face approximation it
        # would silently measure the EXACT plan and report its cosine as the
        # approximated plan's. RAISE instead of measuring the wrong graph.
        raise NotImplementedError(
            "ALPHAGRAD_MEASURE_SERVER=1 cannot carry the per-face wires "
            "(face_rows / face_skips): its measure_seq protocol is a list of "
            "per-vertex CALL STRINGS. It would measure the exact plan and "
            "report it as the approximated one. Unset it, or extend the "
            "measure server's protocol first.")
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
    # W5: THE SAME approximation-head surface PPO builds. There is no longer a
    # legitimate divergence here -- `micro_action_policy` is None on both arms
    # and the approximation head is the per-FACE UnifiedFacePolicy on both.
    # AZ eliminates per VERTEX and approximates per FACE, exactly like PPO.
    dynamic_substeps=True,
    unified_head=False,
    # #79: these four were LITERAL CONSTANTS, so AZ had no exact arm and
    # ALPHAGRAD_GAZ_MICRO gated nothing. They now come from args_az, with
    # defaults reproducing the previous behaviour byte-for-byte.
    no_approx_head=bool(A.no_approx_head),
    face_actions=bool(A.face_actions) and not bool(A.no_approx_head),
    unified_face_head=bool(A.face_actions) and not bool(A.no_approx_head),
    live_faces=bool(A.live_faces) and not bool(A.no_approx_head),
    max_substeps=1,
    axis_group_embedding=False,
)
if A.no_approx_head:
    print("[gaz] EXACT arm: no approximation head; searching the elimination "
          "ORDER only (face actions and live faces forced off).", flush=True)
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
# The per-vertex IDENTITY replaced the hand-written feature matrix (see
# ppo.VertexIdentityPool): one encoder pass over the BASE stream, kept as
# rows so the pool runs -- and trains -- inside the heads. Params-dependent,
# so it is rebuilt whenever the weights move (per episode, below), exactly
# as the carry is.


@eqx.filter_jit
def _identity_stream(agent):
    return _cs.base_identity_stream(agent, BASE_TOK, BASE_EQN, BASE_N,
                                    window=BASE_W, total_v=TOTAL_V,
                                    base_owners=BASE_OWN)


VFEAT = None    # rebound to the identity stream once `agent` exists


@eqx.filter_jit
def _carry_init(agent):
    return _cs.init_carry(agent, BASE_TOK, BASE_EQN, BASE_N,
                          window=BASE_W, total_v=TOTAL_V, embd_dim=EMBD,
                          base_owners=BASE_OWN)


@eqx.filter_jit
def _carry_advance(agent, enc, vs, vc, dtok, deqn, dcount, owner):
    return _cs.advance(agent, enc, vs, vc, dtok, deqn, dcount, owner,
                       window=MAX_DELTA_TOKENS)


@eqx.filter_jit
def _carry_heads(agent, vs, vc):
    """(vertex_logits (total_v,), vertex_contexts (total_v, E), value (3,))."""
    return _cs.heads(agent, vs, vc, identity_stream=VFEAT,
                     preference=None)


class Carry:
    """The DEVICE half of a search node (~35 KB on nn256).

    EncCarry(M, I, cumhist, nvalid, pos) + the per-vertex memory
    (sums/counts). Params-DEPENDENT, unlike the
    old `_tok_cache`: a carry cached across a `train_step` is silently STALE,
    not wrong-shaped, so carries live for exactly one `gumbel_search` call
    and every decision rebuilds from the committed root carry.
    """

    __slots__ = ("enc", "vs", "vc")

    def __init__(self, enc, vs, vc):
        self.enc, self.vs, self.vc = enc, vs, vc


def _q_of(value3):
    """The 3 value heads [lat, mem, cos] scalarized with the CH weights."""
    v = np.asarray(value3, dtype=np.float64).reshape(-1)
    return float(W4[0] * v[0] + W4[1] * v[1] + W4[3] * v[2])


# ------------------------------------- the PER-FACE head (PPO's, verbatim)
# `agent.micro_action_policy` is None and `agent.face_path_policy` is the
# UnifiedFacePolicy -- the SAME two facts hold on PPO under --live-faces. The
# stream the head reads, the host callbacks that feed it and the per-step
# prefix binding all come from common/face_driver.py: the same objects
# ppo.main() drives, not a second copy. A second copy is exactly how the
# face-index shift survived unnoticed for months.
from alphagrad.approx.common.face_driver import (      # noqa: E402
    bind_step_callbacks, build_live_face_stream, make_face_callbacks)
from alphagrad.approx.heads import (                   # noqa: E402
    NUM_OPS as _NUM_OPS, OP_COMPRESS as _OP_COMPRESS, OP_DIAG as _OP_DIAG,
    OP_END as _OP_END, OP_QUANT as _OP_QUANT,
    precompute_factor_tables as _pft)
from alphagrad.approx.ppo import (                     # noqa: E402
    _axis_features_from_state as _axis_feats,
    attention_entropy_diagnostic as _attn_ent_diag,
    _ATTN_ENTROPY_ON,
)

assert agent.micro_action_policy is None, (
    "the per-VERTEX approximation head is still built -- AZ approximates per "
    "FACE now, and a live micro head would be a second, untrained action "
    "space PPO does not have")
# #79: the EXACT arm legitimately has no face head. On the APPROX arm its
# absence is still a hard error -- that is the W5 guarantee (AZ approximates
# per FACE, like PPO), and losing it silently is how the head went untrained
# before.
if _EXACT_ARM:
    assert agent.face_path_policy is None, (
        "--no-approx-head was passed but a per-face head was still built")
else:
    assert agent.face_path_policy is not None, (
        "no per-face head was built: check face_actions/unified_face_head/"
        "live_faces in the agent factory call above")

LIVE_FACES = build_live_face_stream(
    jaxpr, ARGN, list(closed.literals), list(xs),
    max_faces=int(ENV_MAX_FACES), max_axes=MAX_AXES_PER_VERTEX,
    # A chunk is a slice of the step delta, so the delta cap is the one
    # honest window -- identical to PPO's choice.
    window=MAX_DELTA_TOKENS,
    # AZ has no envs, so PPO's max(64, 4 * num_envs) is a formula in a number
    # that does not exist here. The live working set is ONE prefix (all faces
    # of the current vertex share it), so any capacity >= 1 is warm; 64 keeps
    # the previous decisions' prefixes around for the loss-free re-reads.
    cache=int(os.environ.get("ALPHAGRAD_GAZ_FACE_PREFIX_CACHE", "64")),
)
_live_face, _live_face_count = make_face_callbacks(
    LIVE_FACES, window=MAX_DELTA_TOKENS, prof_sink=None)

# ---------------- #109: bucket-compiled face width (search hot path) -------
# `_face_plan` used to be jitted at the CONFIGURED face bound (2538 on the
# TLM targets) although a vertex has ~1-8 live faces: the while_loop already
# runs the ACTUAL count, but every draw still uploaded the (prefix,
# MAX_FACES, SLOTS, 3) history wires, carried MAX_FACES-wide loop state
# through every pure_callback, and shipped MAX_FACES-wide outputs back to
# the host -- >99.9% padding. Each draw is now padded to the smallest bucket
# in {2, 4, 8, 32} (fallback: ENV_MAX_FACES) covering BOTH the drawn
# vertex's live count and the widest face slot the committed prefix uses
# (history shares the face axis), and jitted once per bucket via the face
# head's STATIC max_faces. Outputs are re-padded to ENV_MAX_FACES on the
# host with the exact bytes the unbucketed call produces for never-visited
# faces (see common/face_buckets.py), so replay storage, the loss and the
# PlanTokenizer wires are byte-identical either way.
# ALPHAGRAD_GAZ_FACE_BUCKETS=0 restores the single-shape path.
from alphagrad.approx.common.face_buckets import (      # noqa: E402
    bucket_width, hist_face_width, pad_face_outputs, with_face_width)

_FACE_BUCKETS_ON = (os.environ.get("ALPHAGRAD_GAZ_FACE_BUCKETS", "1") == "1"
                    and not _EXACT_ARM)
_BUCKET_AGENT_CACHE: dict = {}


def _agent_for_face_width(fb):
    """The CURRENT module-global agent with the face head's static width
    rebound to ``fb`` -- params shared, one `_face_plan` compile per width.
    Re-derived whenever `train_step`/PopArt rebinds `agent` (the cache entry
    keeps the source agent so staleness is an identity check, not a leak
    hazard: at most one superseded params set per bucket, replaced on the
    next draw)."""
    ent = _BUCKET_AGENT_CACHE.get(int(fb))
    if ent is not None and ent[0] is agent:
        return ent[1]
    ag2 = with_face_width(agent, int(fb))
    _BUCKET_AGENT_CACHE[int(fb)] = (agent, ag2)
    return ag2

# WARM THE QUANT HARDWARE SCAN EAGERLY, before anything is traced. Its own
# docstring demands it ("Warm this once at build (eagerly, before any jit) so
# the jnp.dot probes never run under trace") and PPO does it in main(). AZ has
# TWO separate jits over the face head -- `_face_plan` at rollout and
# `train_step` in the loss -- so a cache first filled under `_face_plan`'s
# while_body handed `train_step` a DEAD TRACER:
#   UnexpectedTracerError: float32[24] ... created at
#   micro_actions.py:299 (verify_hardware_compat), leaked from
#   Agent._face_loop._body traced for while_body.
# PPO never saw it because its rollout and its loss are traced inside the same
# jit. Warming here makes the tables concrete constants for both.
try:
    from graphax.sparse.micro_actions import report_hardware_scan as _hw_scan
    _hw_scan()
except Exception as _hwe:
    print(f"[quant-scan] unavailable: {_hwe!r}", flush=True)

FACT_TABLES = _pft(_ns.max_axis_size)
# (NUM_OPS,) all-ones: AZ runs no --variant curriculum, so nothing is masked
# out of the op alphabet. Sampling and the loss MUST pass the same array or
# the re-scored log-prob is not the behaviour policy's.
OP_OVERRIDE = jnp.ones((_NUM_OPS,), dtype=jnp.float32)
AXIS_STATE = env.axis_state_static
AXIS_VALID = env.axis_valid_static
_PREFIX_W = max(NV, 1)


def _prefix_arrays(state):
    """The committed prefix in the wire shapes `bind_step_callbacks` wants.

    `order` / `spec_hist` / `face_hist` / `skip_hist` are aligned 1:1 and
    sliced to `step_count` on the host side, exactly like the env state
    arrays PPO passes.
    """
    order = np.zeros((_PREFIX_W,), np.int32)
    spec_hist = np.broadcast_to(
        EXACT_SPEC_ROW, (_PREFIX_W,) + EXACT_SPEC_ROW.shape).copy()
    face_hist = np.broadcast_to(
        EXACT_FACE_ROWS, (_PREFIX_W,) + EXACT_FACE_ROWS.shape).copy()
    skip_hist = np.zeros((_PREFIX_W, int(ENV_MAX_FACES)), np.int32)
    for k, (a, fr, fs) in enumerate(state):
        order[k] = int(a) + 1
        face_hist[k] = fr
        skip_hist[k] = fs
    return order, spec_hist, face_hist, skip_hist


@eqx.filter_jit
def _face_plan(agent, precomputed, enc_carry, avail,
               order, spec_hist, step_count, face_hist, skip_hist, key):
    """Draw ONE face-sequence sample F ~ beta for the vertex the one-hot
    ``avail`` forces (Sampled AZ calls this per surviving candidate per
    widening round; see `_draw_face_sequence`).

    The vertex is forced by handing `sample_action_dynamic` a ONE-HOT
    availability mask: the categorical then has no choice, and every other
    gate in that function (masks, face loop, wire translation) runs exactly as
    it does for PPO. Reimplementing the face loop here instead would be a
    second copy of the pipeline this whole change exists to remove.
    """
    face_chunk_fn, face_count_fn = bind_step_callbacks(
        _live_face, _live_face_count, order, spec_hist, step_count,
        face_hist, skip_hist)
    (vertex_idx, actions, _vdist, _od, _id, _jd, _ed, _kd, _qlp,
     _vp, _vc, face_out, value, v_context) = agent.sample_action_dynamic(
        None, avail, AXIS_STATE, AXIS_VALID, FACT_TABLES, OP_OVERRIDE, key,
        identity_stream=VFEAT,
        precomputed=precomputed, enc_carry=enc_carry,
        face_chunk_fn=face_chunk_fn, face_count_fn=face_count_fn,
    )
    (fa, face_logp, face_ent, f_pair, f_comp, f_valid,
     f_cnt, f_dt, f_de, f_ends) = face_out
    # THE WIRE, from `to_env_action_dynamic`'s own translator -- so AZ and PPO
    # emit identical bytes for identical FaceActions. AZ's `measure()` already
    # passed correctly-shaped face arrays; they were filled with -1.
    env_action = agent.to_env_action_dynamic(
        vertex_idx, actions, AXIS_STATE, face_action=fa)
    features = _axis_feats(AXIS_STATE[vertex_idx], AXIS_VALID[vertex_idx])
    return (env_action.face_rows, env_action.face_skip, fa, f_pair, f_comp,
            f_valid, f_cnt, f_dt, f_de, f_ends, v_context, features, face_ent,
            vertex_idx)


# ------------------------------------------------------------------ 3. optimizer
# #95 training-loop hygiene: the SAME palimpsa gradient guard PPO runs --
# optax.clip_by_global_norm ahead of adam (ppo.py builds
# `optax.chain(clip_by_global_norm(args.max_grad_norm), adam(...))`, default
# --max-grad-norm 0.5). AZ trained on a bare adam, so one bad replayed batch
# could kick the shared palimpsa backbone arbitrarily far.
_MAX_GRAD_NORM = float(os.environ.get("ALPHAGRAD_GAZ_MAX_GRAD_NORM", "0.5"))
opt = optax.chain(optax.clip_by_global_norm(_MAX_GRAD_NORM),
                  optax.adam(A.lr))
opt_state = opt.init(eqx.filter(agent, eqx.is_array))

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
def _step(node, vertex, face_rows=None, face_skips=None, face_keys=None):
    """Eliminate ``vertex`` on PT and return the child node.

    ``node`` is ``(state, carry, ctxs)``; the returned child carries its own
    ``(state, carry)``. The caller decides whether the elimination is
    speculative (inside ``PT.branch()``) or committed. ``ctxs`` is INERT since
    the B.4 residual was deleted -- it fed the per-vertex residual update and
    nothing else -- but the node shape is the caller's, so it stays.

    ``face_rows``/``face_skips`` default to EXACT (the exact arm, the DEEPEN
    rollouts and the warm start). Under Sampled AZ the search's expansions
    pass each drawn face sequence's wires, so the search scores the composite
    (vertex, faces) action on the graph those wires actually build; the
    COMMITTED decision passes the executed draw's wires, so the committed
    carry is built from the same approximated delta the search evaluated.
    """
    state, carry, _ctxs = node
    a = int(vertex) - 1                     # VALID is contiguous 1..NV
    spec_row = EXACT_SPEC_ROW
    face_rows = EXACT_FACE_ROWS if face_rows is None else np.asarray(
        face_rows, np.int32)
    face_skips = EXACT_FACE_SKIPS if face_skips is None else np.asarray(
        face_skips, np.int32)
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
                             is_last=is_last, face_keys=face_keys)
    dt, de, dc = _wire_delta(toks, ids)
    enc2, vs2, vc2 = _carry_advance(
        agent, carry.enc, carry.vs, carry.vc,
        jnp.asarray(dt), jnp.asarray(de), jnp.asarray(dc, jnp.int32),
        jnp.asarray(a, jnp.int32))
    st2 = list(state) + [(a, face_rows, face_skips)]
    return (st2, Carry(enc2, vs2, vc2),
            {"tokens": toks, "eqn_ids": ids, "delta": (dt, de, dc),
             "owner": a, "spec_row": spec_row, "face_rows": face_rows,
             "face_skips": face_skips, "is_last": is_last})


_OP_NAME = {int(_OP_END): "none", int(_OP_DIAG): "diag",
            int(_OP_COMPRESS): "compress", int(_OP_QUANT): "quant"}


def _face_choice_counts(fa, f_valid):
    """Realized per-face-SLOT approximation classes -> approx_prob/* counts.

    THE UNIT IS ONE (face, slot) DECISION, over the faces that EXIST
    (``f_valid``) rather than the padded width -- padding faces never ran, so
    counting them would dilute every class toward "none". A SKIPPED face is
    dropped before any approximation can apply, so it contributes all of its
    slots to ``skip`` and nothing to the op classes. The five classes
    {skip, none, diag, compress, quant} therefore partition the episode's
    realized decisions and sum to 1.

    WANDB AUDIT (2026-08-07). This used to be a per-FACE, SET-DEDUPED count:
    a face with a diag in one of its 3 slots scored 1 for "diag" and 0 for
    "none", while ppo.py reported the same episode as diag = 1/3, none = 2/3
    (it counted per slot). Same key, same policy, two curves a factor of
    FACE_SLOTS apart. The per-slot form is kept as the shared definition
    because it counts what the head actually emits -- one op per slot -- and
    the set-dedupe could not distinguish one diag on a face from three.
    ppo.py's ``_face_op_freq`` now excludes skipped faces to match.
    """
    out = collections.Counter()
    ops = np.asarray(fa.op_type, np.int32)
    skips = np.asarray(fa.skip, np.int32)
    valid = np.asarray(f_valid) > 0.5
    n_slots = int(ops.shape[1]) if ops.ndim > 1 else 1
    for f in range(ops.shape[0]):
        if not valid[f]:
            continue
        if int(skips[f]) == 1:
            out["skip"] += n_slots
            continue
        for o in np.atleast_1d(ops[f]):
            out[_OP_NAME.get(int(o), "other")] += 1
    return out


_FACE_WINDOW_SATURATED = [0]
_FACE_STEPS = [0]
_LAST_PREFIX = {}


def _face_count_diag(vertex):
    """Recompute the LiveFaceStream count on the HOST, off the same prefix the
    jitted callback was handed, so a disagreement can be attributed to the
    tokenizers rather than to the callback plumbing."""
    try:
        pfx = _LAST_PREFIX
        direct = int(LIVE_FACES.n_faces(
            pfx["order"], pfx["specs"], pfx["n"], int(vertex),
            pfx["face_hist"], pfx["skip_hist"]))
        wires = [(int(pfx["order"][k]),
                  int(np.sum(pfx["skip_hist"][k])),
                  int(np.sum(pfx["face_hist"][k][..., 0] >= 0)))
                 for k in range(pfx["n"])]
        return (f"direct_live_n_faces={direct} n={pfx['n']} "
                f"prefix(v,skips,rulerows)={wires}")
    except Exception as _e:
        return f"diag failed: {type(_e).__name__}: {_e}"


def _assert_face_accounting(f_cnt, f_valid, d, n_faces, vertex, tk_faces):
    """SILENT-FAILURE 2: face/chunk accounting, with the one identity that
    actually holds.

    ``sum(face_counts) == step_delta_count`` does NOT hold, and cannot: face
    f's chunk is face f-1's approximation followed by face f's contraction
    emitted UNHOOKED (`live_faces.chunk`: "NO hook on face f ... a recording
    hook on an UNDECIDED face would make the head read a contraction the
    measurement will not build"), whereas the committed delta emits every
    face's contraction WITH its approximation. The chunks are a
    COUNTERFACTUAL of the delta, not a prefix of it, so the totals differ in
    both directions -- measured 1128 chunk tokens against an 874-token delta
    on nn256 vertex 25.

    What must hold, and is checked:

    (a) the concatenated emission window did not SATURATE. `_face_loop` clamps
        each chunk with ``ct_eff = min(ct_f, W - off)``, so once the window is
        full every later face reads an EMPTY chunk and decides blind -- a
        silent failure that reads as a healthy run. Counted per step and
        logged as ``faces/window_saturated``.
    (b) the number of decided faces equals the length of the key list the
        committed elimination was indexed by. Both come from the LiveFaceStream
        prefix tokenizer -- deliberately, see `PlanTokenizer.face_transforms`
        -- but by two different routes: `n_faces` through the jitted
        `face_count_fn` callback inside `sample_action_dynamic`, `tk_faces`
        through the host-side enumeration. A shift between them fed the head
        another face's contraction for months.
    """
    total = int(np.sum(np.asarray(f_cnt, np.int64)))
    _FACE_STEPS[0] += 1
    if total >= MAX_DELTA_TOKENS:
        _FACE_WINDOW_SATURATED[0] += 1
    assert total <= MAX_DELTA_TOKENS, (
        f"vertex {vertex}: face chunks total {total} > the emission window "
        f"{MAX_DELTA_TOKENS} -- the per-face clamp did not hold")
    if n_faces != tk_faces:
        _dbg = _face_count_diag(vertex)
        raise AssertionError(
            f"vertex {vertex}: the face loop decided {n_faces} faces but the "
            f"plan tokenizer enumerates {tk_faces} -- a face-index shift feeds "
            f"the head another face's contraction. "
            f"[diag] step={_FACE_STEPS[0] - 1} "
            f"pt_legal={len(PT.legal(VALID))} "
            f"chunk_counts={np.asarray(f_cnt)[:8].tolist()} "
            f"valid_prefix={np.asarray(f_valid)[:8].tolist()} "
            f"{_dbg}")


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
    """``(vertex_logits (host), the raw head triple, scalar value in z-space)``.

    The raw triple is `sample_action_dynamic`'s ``precomputed`` argument --
    handing it back means the committed decision's face draw conditions on the
    encoding the search actually acted under, with no second encode.
    """
    out = _carry_heads(agent, carry.vs, carry.vc)
    return np.asarray(out[0], dtype=np.float64), out, _q_of(out[2])


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
        if os.environ.get("ALPHAGRAD_FORCE_REV_ORDER", "0") == "1" and legal:
            legal = [max(legal)]   # rev: highest first
        if not legal:
            break
        vlog, out, _v = _eval_node(state, carry)
        la = [int(v) - 1 for v in legal]
        v = legal[int(np.argmax(vlog[la]))]
        state, carry, _d = _step((state, carry, out[1]), v)
    _vl, _out, vz = _eval_node(state, carry)
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
        _face_wire_keys, _incremental_stream_tokens, decode_vertex_rule_specs)
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
        # MUST be the env's own builder, not a hand-rolled tuple: the
        # ancestor-cut branch in _incremental_stream_tokens does
        # np.frombuffer(face_key[-1][0]) and needs the int32-BYTES form
        # _face_wire_keys emits. The dense-tuple form only survived because
        # that branch is skipped whenever the last vertex carries a COMPRESS
        # row -- which is every approximation episode and NO exact one, so
        # --no-approx-head crashed here on its first golden check.
        face_key=_face_wire_keys(face_rows, face_skips, n),
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
CSCALE = float(os.environ.get("ALPHAGRAD_GAZ_CSCALE", "0.1"))

# ------------------------------------------ Sampled AlphaZero (Hubert 2021)
# The shared, env-free arithmetic lives in common/sampled_az.py so the unit
# tests exercise the exact functions this trainer runs (this module cannot be
# imported by tests -- it builds its env at import time).
from alphagrad.approx.common.sampled_az import (      # noqa: E402
    assert_same_depth, draw_weights, face_ce_term, improved_policy,
    phase_plan, qscale_sigma, rho_from_beta_temp, weighted_q)

# #93: rollout depth defaults to 0 -- pure value bootstrap, textbook
# Gumbel-AZ. The halving budget WIDENS (2**p new face-sequence draws per
# survivor in round p) instead of deepening; depth-deepening survives only
# behind ALPHAGRAD_GAZ_DEEPEN=1, and even then every Q estimate entering one
# target comes from a single common depth (round 0) -- the 200:1 spread in
# the completed-Q target was attributable to depth alone.
_GAZ_DEEPEN = os.environ.get("ALPHAGRAD_GAZ_DEEPEN", "0") == "1"
# --- ORDER-SEARCH DIAGNOSTICS ------------------------------------------
# `_SEARCH_DIAG` is refreshed by gumbel_search on every root decision and
# read by the JSONL sink at the end of the episode, so the record carries the
# LAST root's completed-Q spread. At depth 0 the completed Q IS the value
# net's own output (there is no tree to back up), so a flat comp_q spread and
# a flat improved policy are the same statement: the bandit has no signal.
_SEARCH_DIAG = {}
# ACCUMULATOR over the episode's ROOTS. Roots with < 2 legal actions are
# excluded entirely -- they have no decision to spread over, and the last root
# of every episode is exactly that case.
_SEARCH_ACC: dict = {}


def _search_acc(k, v):
    _SEARCH_ACC.setdefault(k, []).append(float(v))
_GAZ_JSONL = os.environ.get("ALPHAGRAD_GAZ_JSONL", "")
if not _GAZ_DEEPEN and int(A.rollout_depth) != 0:
    print(f"[gaz] #93: --rollout-depth {A.rollout_depth} IGNORED (depth 0, "
          "pure value bootstrap); set ALPHAGRAD_GAZ_DEEPEN=1 to deepen",
          flush=True)
# rho = pi/beta == 1 while the draws come from the head we train. The knob
# exists so a future proposal temperature CANNOT silently bias the target:
# any value != 1.0 hard-fails here until the importance ratio is implemented.
_GAZ_BETA_TEMP = float(os.environ.get("ALPHAGRAD_GAZ_BETA_TEMP", "1.0"))
_RHO = rho_from_beta_temp(_GAZ_BETA_TEMP)
# DEEPEN mode has no widening rounds, so the face CE draws its K sequences
# for the CHOSEN vertex after the search (K >= 2 or the weighted CE carries
# no ranking signal: a single draw's w_hat is 1 and E[grad(-log beta)] = 0).
_DEEPEN_FACE_K = max(2, int(os.environ.get("ALPHAGRAD_GAZ_DEEPEN_FACE_K", "2")))
# Fixed storage pads for the per-decision search draws, from the FULL-width
# schedule (late-episode decisions have fewer legal vertices => fewer draws).
_PLAN_FULL = phase_plan(A.n_candidates, deepen=_GAZ_DEEPEN,
                        rollout_depth=A.rollout_depth)
if _GAZ_DEEPEN:
    D_MAX_DRAWS = _DEEPEN_FACE_K
else:
    D_MAX_DRAWS = sum(n * d for n, d, _dep in _PLAN_FULL)


def sigma(q, max_n=1, cs=CSCALE):
    """Danihelka et al. 2022 monotone Q-transform (mctx qtransform form):
    min-max-normalize Q over the candidate set, then scale by
    (c_visit + max_N) * c_scale, so evaluated Q outweighs the prior+Gumbel
    and increasingly so as the search invests more evaluations. max_N =
    evaluation rounds of the most-evaluated candidate. Delegates to the ONE
    shared transform in common/sampled_az.py (also used for the draw
    weights), replacing a per-set z-score that capped every Q-gap at ~1
    sigma and erased magnitudes (2026-07-16 review, Fix 1)."""
    return qscale_sigma(q, max_n=max_n, cvisit=CVISIT, cscale=cs)

# Per-episode POLICY-ENTROPY accumulators, drained to their episode means in
# the wandb block.
#   _VE_ENT : entropy (nats) of the vertex-elimination prior the search acts
#             under -- ppo logs the same quantity as entropy/ve_head.
#   _AP_ENT : arity-normalised entropy of the approximation head, taken from
#             `_face_replay` in the train step.
# The CAVEAT that used to sit here ("az's approximation head is the PER-VERTEX
# MicroActionPolicy while ppo's is the PER-FACE UnifiedFacePolicy ...
# comparable in SHAPE but NOT in absolute scale") is GONE with the head it
# described: both arms now report the same head's arity-normalised entropy
# over the same per-face action space, so the two curves are directly
# comparable.
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
def _draw_face_sequence(vertex, head_out, carry, prefix_arrays, rng,
                        n_live=None, hist_w=0):
    """ONE i.i.d. face-sequence draw F ~ beta for ``vertex`` at the committed
    prefix -- the SAME `_face_plan` call the old commit path made, so the
    chunks the head reads and the wires it emits are byte-identical to PPO's.
    Returns a host dict carrying the wires plus everything `_face_replay`
    needs to re-score the draw in the loss.

    ``n_live`` is the caller's authoritative face count for ``vertex`` (from
    the per-decision `face_keys_of` enumeration) and ``hist_w`` the widest
    face slot the committed prefix uses: together they pick the #109 compile
    bucket. ``n_live=None`` (or ALPHAGRAD_GAZ_FACE_BUCKETS=0) keeps the old
    single-shape ENV_MAX_FACES path."""
    _o_arr, _sp_h, _f_h, _s_h, _n = prefix_arrays
    # FRESH PREFIX TOKENIZER PER DRAW. `chunk()`'s speculative eliminations
    # force LazyEdge memos IN PLACE on the CACHED prefix tokenizer (the
    # `_Snapshot` cannot restore a forced memo), which was harmless while
    # exactly ONE face loop ran per prefix -- but Sampled AZ runs K of them,
    # and the second one already reads a SHRUNKEN face enumeration (observed
    # on the first smoke: a vertex whose draw decided 2 faces enumerated 0
    # at commit). Evicting the prefix cache makes every face loop start from
    # the same pristine enumeration the measurement's fresh replay will use;
    # the caller's loop-top `_lf_tk` reference drops out of the cache and is
    # never speculated on, so the commit keys stay authoritative. The chunk
    # cache stays: its keys include (prefix, vertex, decided rows, f), and a
    # fresh-start recompute of the same key is byte-deterministic.
    LIVE_FACES._prefix.clear()
    _avail = np.zeros((TOTAL_V,), np.float32)
    _avail[int(vertex) - 1] = 1.0
    _key = jax.random.PRNGKey(int(rng.integers(2 ** 31)))
    # #109: pick the compile bucket. The width must cover the CURRENT
    # vertex's enumeration (or the while_loop would silently truncate) AND
    # the widest face slot any committed decision wrote (the history wires
    # share the face axis and ride into the same jit).
    _fb = int(ENV_MAX_FACES)
    _ag = agent
    _f_use, _s_use = _f_h, _s_h
    if _FACE_BUCKETS_ON and n_live is not None:
        _fb = bucket_width(max(int(n_live), int(hist_w)), int(ENV_MAX_FACES))
        if _fb < int(ENV_MAX_FACES):
            _ag = _agent_for_face_width(_fb)
            _f_use = np.ascontiguousarray(_f_h[:, :_fb])
            _s_use = np.ascontiguousarray(_s_h[:, :_fb])
    (fr, fs, fa, f_pair, f_comp, f_valid, f_cnt, f_dt, f_de, f_ends,
     _vctx, _feat, face_ent, _vi) = _face_plan(
        _ag, head_out, carry.enc, jnp.asarray(_avail),
        jnp.asarray(_o_arr), jnp.asarray(_sp_h),
        jnp.asarray(_n, jnp.int32), jnp.asarray(_f_use), jnp.asarray(_s_use),
        _key)
    assert int(_vi) == int(vertex) - 1, (
        f"the one-hot availability mask did not force the searched vertex: "
        f"head picked {int(_vi) + 1}, search wanted {int(vertex)}")
    fr = np.asarray(fr, np.int32)
    fs = np.asarray(fs, np.int32)
    fa = jax.tree_util.tree_map(np.asarray, fa)
    f_pair = np.asarray(f_pair)
    f_comp = np.asarray(f_comp)
    f_valid = np.asarray(f_valid)
    f_cnt = np.asarray(f_cnt, np.int32)
    if _fb < int(ENV_MAX_FACES):
        # The callback count and the caller's key list must agree, or the
        # bucket could be too narrow and the loop would truncate SILENTLY --
        # the same divergence `_assert_face_accounting` catches at commit,
        # surfaced here for every draw.
        _ndec = int(np.sum(f_valid > 0.5))
        assert _ndec == int(n_live), (
            f"vertex {vertex}: bucketed draw decided {_ndec} faces at width "
            f"{_fb} but the authoritative enumeration has {n_live} -- "
            f"face_count_fn and face_keys_of disagree")
        (fr, fs, fa, f_pair, f_comp, f_valid, f_cnt,
         f_ends) = pad_face_outputs(
            int(ENV_MAX_FACES), fr, fs, fa, f_pair, f_comp, f_valid, f_cnt,
            f_ends)
    return {"fr": fr, "fs": fs,
            "fa": fa,
            "f_pair": f_pair, "f_comp": f_comp,
            "f_valid": f_valid,
            "f_cnt": f_cnt,
            "f_dt": np.asarray(f_dt, np.int32),
            "f_de": np.asarray(f_de, np.int32),
            "f_ends": np.asarray(f_ends, np.int32),
            "face_ent": float(face_ent)}


def _pack_search_draws(cands):
    """The decision's face-sequence draws, flattened to the FIXED
    ``D_MAX_DRAWS`` slots the loss vmaps over (padding: li == -1, w == 0,
    all-zero wires -- `face_ce_term` gates it to exactly 0). Each slot
    carries its own masks/window so the loss needs no per-vertex indirection.
    """
    D = int(D_MAX_DRAWS)
    li = np.full((D,), -1, np.int32)
    vidx = np.zeros((D,), np.int32)
    w = np.zeros((D,), np.float32)
    rows = [dd for c in cands for dd in c.get("draws", ())]
    assert rows, "no face draws to pack on the approx arm"
    assert len(rows) <= D, (
        f"{len(rows)} search draws exceed the D_MAX_DRAWS={D} storage pad")
    _z = lambda a: np.zeros((D,) + a.shape, a.dtype)
    fpair, fcomp = _z(rows[0]["f_pair"]), _z(rows[0]["f_comp"])
    fvalid, fcnt = _z(rows[0]["f_valid"]), _z(rows[0]["f_cnt"])
    fdt, fde = _z(rows[0]["f_dt"]), _z(rows[0]["f_de"])
    fends = _z(rows[0]["f_ends"])
    fa_list = []
    i = 0
    for c in cands:
        for dd in c.get("draws", ()):
            li[i] = int(c["li"])
            vidx[i] = int(c["v"]) - 1
            w[i] = float(dd["w_hat"])
            fpair[i] = dd["f_pair"]; fcomp[i] = dd["f_comp"]
            fvalid[i] = dd["f_valid"]; fcnt[i] = dd["f_cnt"]
            fdt[i] = dd["f_dt"]; fde[i] = dd["f_de"]
            fends[i] = dd["f_ends"]
            fa_list.append(dd["fa"])
            i += 1
    _fa0 = jax.tree_util.tree_map(np.zeros_like, fa_list[0])
    fa_list += [_fa0] * (D - len(fa_list))
    fa = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *fa_list)
    return {"sd_li": li, "sd_vidx": vidx, "sd_w": w, "sd_fpair": fpair,
            "sd_fcomp": fcomp, "sd_fvalid": fvalid, "sd_cnt": fcnt,
            "sd_dt": fdt, "sd_de": fde, "sd_fa": fa,
            "sd_fends": fends}


def gumbel_search(state, carry, rng, prefix_arrays, face_keys_of):
    """One decision, Sampled AlphaZero over the composite action a = (v, F)
    with Gumbel-AZ at the vertex level. ``PT`` is positioned at the COMMITTED
    prefix on entry and is left there on exit -- every expansion runs inside
    ``PT.branch()``.

    ``prefix_arrays`` / ``face_keys_of`` come from the caller's per-decision
    LiveFaceStream prefix tokenizer (the authoritative face enumeration; see
    the commit-path note on `_Snapshot` and forced LazyEdges).

    Returns ``(chosen, pi, la, legal, head_out, cands, exec_draw)``:
    ``cands`` carries every candidate's weighted draws for the face CE, and
    ``exec_draw`` is the face sequence to EXECUTE for the chosen vertex,
    sampled ~ w_hat (the improved beta at the root; None on the exact arm).
    """
    legal = PT.legal(VALID)
    if os.environ.get("ALPHAGRAD_FORCE_REV_ORDER", "0") == "1" and legal:
        legal = [max(legal)]   # rev: highest first
    vlog, head_out, v_root = _eval_node(state, carry)
    ctxs = head_out[1]
    la = np.array([int(v) - 1 for v in legal], dtype=np.int32)
    logits = vlog[la] - vlog[la].max()
    m = min(A.n_candidates, len(legal))
    g = rng.gumbel(size=len(legal))
    order_idx = np.argsort(-(logits + g))[:m]
    # ONE candidate per Gumbel-selected VERTEX; its Q now aggregates over its
    # own face-sequence draws, so the search scores the COMPOSITE action.
    cands = []
    for ci in order_idx:
        cands.append({"li": int(ci), "v": legal[int(ci)],
                      "q": [], "q_depths": [], "draws": [],
                      "g": float(g[int(ci)]),
                      "logit": float(logits[int(ci)])})
    # SEQUENTIAL HALVING. Default (#93): every evaluation is a DEPTH-0 value
    # bootstrap and each round WIDENS -- survivors draw 2**p new face
    # sequences from beta, so the budget buys lower-variance weighted Q over
    # the composite action, never depth-heterogeneous Q. ALPHAGRAD_GAZ_DEEPEN=1
    # restores the legacy progressive deepening (exact-wire rollouts, one
    # evaluation per round, 2x depth for survivors).
    plan = phase_plan(m, deepen=_GAZ_DEEPEN,
                      rollout_depth=(A.rollout_depth if _GAZ_DEEPEN else 0))
    # #109: the committed prefix's used face width, ONCE per decision -- with
    # the per-vertex live counts (via `face_keys_of`) it selects each draw's
    # compile bucket.
    _hist_wd = 0
    if _FACE_BUCKETS_ON and not _EXACT_ARM:
        _hist_wd = hist_face_width(prefix_arrays[2], prefix_arrays[3],
                                   int(prefix_arrays[4]))
    surv = list(cands)
    for _phase, (_n_expect, _n_draws, _depth) in enumerate(plan):
        assert len(surv) == _n_expect, (
            f"halving drifted from phase_plan: {len(surv)} survivors, "
            f"plan says {_n_expect}")
        depth = min(int(_depth), NV)
        for c in surv:
            if _GAZ_DEEPEN or _EXACT_ARM:
                # Legacy/exact: ONE exact-wire evaluation per round
                # (deterministic dynamics + deterministic value net => a
                # repeat at the same depth would be an identical duplicate;
                # on the exact arm later rounds add nothing at depth 0).
                reps = 1 if (_GAZ_DEEPEN or _phase == 0) else 0
            else:
                reps = int(_n_draws)
            for _r in range(reps):
                dr = None
                fr = fs = None
                if not (_GAZ_DEEPEN or _EXACT_ARM):
                    dr = _draw_face_sequence(
                        c["v"], head_out, carry, prefix_arrays, rng,
                        n_live=len(face_keys_of(c["v"])), hist_w=_hist_wd)
                    fr, fs = dr["fr"], dr["fs"]
                # ONE branch per evaluation covers the expansion AND its
                # whole rollout chain: _Snapshot.__exit__ truncates the
                # append-only lists back to their entry length.
                with PT.branch():
                    st2, cy2, d = _step((state, carry, ctxs), c["v"], fr, fs,
                                        face_keys=face_keys_of(c["v"]))
                    if _phase == 0 and _r == 0:
                        c["_wire"] = (d["spec_row"].tobytes(),
                                      d["face_rows"].tobytes(),
                                      d["face_skips"].tobytes())
                        c["_tokbytes"] = (
                            np.asarray(d["tokens"], np.int32).tobytes(),
                            np.asarray(d["eqn_ids"], np.int32).tobytes())
                    qk = rollout_value(st2, cy2, depth)
                if dr is not None:
                    dr["q"] = float(qk)
                    dr["depth"] = depth
                    # #95: the tokens this draw's elimination produced, kept
                    # so the caller can ASSERT the committed step re-produces
                    # them bitwise (search dynamics == measured graph).
                    dr["d_tokens"] = np.asarray(d["tokens"], np.int32)
                    dr["d_eqns"] = np.asarray(d["eqn_ids"], np.int32)
                    c["draws"].append(dr)
                c["q"].append(float(qk))
                c["q_depths"].append(depth)
        if _phase == 0:
            _check_distinct_observations(cands)
        if len(surv) <= 1:
            break
        if _GAZ_DEEPEN or _EXACT_ARM:
            # M2 (deepen): entries of ``q`` estimate different-depth
            # quantities; the halving comparison uses the latest (all
            # survivors in a round share one depth, so it is homogeneous).
            qbar = np.array([c["q"][-1] for c in surv])
        else:
            # Sampled-AZ weighted Q over each survivor's draws: q(v) =
            # sum_k w_k q_k / sum_k w_k, w_k = rho * exp(sigma(q_k)). All
            # draws are depth-0, so the comparison is depth-homogeneous.
            qbar = np.array([
                weighted_q([dd["q"] for dd in c["draws"]], rho=_RHO,
                           cvisit=CVISIT, cscale=CSCALE) for c in surv])
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
    chosen = surv[0]
    # completed-Q improved policy target over the FULL legal set; unvisited
    # vertices complete with v_mix (Danihelka completed-Q).
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
    # #93: one target, ONE depth. Default mode every estimate is depth-0; in
    # DEEPEN mode the target takes each candidate's ROUND-0 estimate (the one
    # depth every visited candidate has) -- never the survivors' deeper ones,
    # which would mix depths across candidates inside one softmax.
    _qv, _nv = {}, {}
    _target_depths = []
    for c in cands:
        if not c["q"]:
            continue
        if _GAZ_DEEPEN or _EXACT_ARM:
            qv = float(c["q"][0])
            _target_depths.append(c["q_depths"][0])
        else:
            qv = weighted_q([dd["q"] for dd in c["draws"]], rho=_RHO,
                            cvisit=CVISIT, cscale=CSCALE)
            _target_depths.extend(c["q_depths"])
        _qv[c["li"]] = qv
        _nv[c["li"]] = len(c["q"])
    assert_same_depth(_target_depths, context="completed-Q target")
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
    # Completed-Q spread at this root. `comp_q` is v_mix everywhere the
    # candidate was NOT visited, so a spread of ~0 means the search saw no
    # difference between the actions it sampled -- which at depth 0, where the
    # completed Q IS the value net's own output, is a statement about the
    # value net alone and not about the tree.
    try:
        _cq = np.asarray(comp_q, dtype=np.float64).reshape(-1)
        _lg = np.asarray(logits, dtype=np.float64).reshape(-1)
        _SEARCH_DIAG.clear()
        _SEARCH_DIAG.update({
            "search/comp_q_mean": float(_cq.mean()),
            "search/comp_q_std": float(_cq.std()),
            "search/comp_q_min": float(_cq.min()),
            "search/comp_q_max": float(_cq.max()),
            "search/comp_q_ptp": float(_cq.max() - _cq.min()),
            "search/comp_q_n": int(_cq.size),
            "search/n_visited": int(len(_qv)),
            "search/v_root": float(v_root),
            "search/v_mix": float(v_mix),
            "search/logit_std": float(_lg.std()),
        })
        if _cq.size >= 2:
            _search_acc("comp_q_std", _cq.std())
            _search_acc("comp_q_ptp", _cq.max() - _cq.min())
            _search_acc("n_legal", _cq.size)
            _search_acc("n_visited", len(_qv))
            _search_acc("v_root", v_root)
            _search_acc("logit_std", _lg.std())
            if len(_qv) > 1:
                _vq = np.asarray(list(_qv.values()), np.float64)
                _search_acc("visited_q_ptp", _vq.max() - _vq.min())
    except Exception:
        pass
    pi = improved_policy(
        logits, comp_q, max_n=max([len(c["q"]) for c in cands] + [1]),
        cvisit=CVISIT, cscale=CSCALE)
    try:
        _p = np.asarray(pi, dtype=np.float64).reshape(-1)
        _SEARCH_DIAG["search/pi_max"] = float(_p.max())
        _p = _p[_p > 0]
        _SEARCH_DIAG["search/pi_entropy"] = float(-(_p * np.log(_p)).sum())
        if _p.size >= 2:
            _search_acc("pi_entropy", -(_p * np.log(_p)).sum())
            _search_acc("pi_max", _p.max())
    except Exception:
        pass
    # ---- face-CE draws + the EXECUTED face sequence -----------------------
    exec_draw = None
    if not _EXACT_ARM:
        if _GAZ_DEEPEN:
            # No widening rounds ran, so the CE's K draws are taken here for
            # the CHOSEN vertex, each valued by the SAME depth-0 bootstrap
            # (one depth per target -- the w_hat ranking must not inherit the
            # depth artifact either).
            for _k in range(_DEEPEN_FACE_K):
                dr = _draw_face_sequence(
                    chosen["v"], head_out, carry, prefix_arrays, rng,
                    n_live=len(face_keys_of(chosen["v"])), hist_w=_hist_wd)
                with PT.branch():
                    st2, cy2, d = _step((state, carry, ctxs), chosen["v"],
                                        dr["fr"], dr["fs"],
                                        face_keys=face_keys_of(chosen["v"]))
                    dr["q"] = float(rollout_value(st2, cy2, 0))
                dr["depth"] = 0
                dr["d_tokens"] = np.asarray(d["tokens"], np.int32)
                dr["d_eqns"] = np.asarray(d["eqn_ids"], np.int32)
                chosen["draws"].append(dr)
        # Normalized draw weights w_hat per candidate (the CE's targets).
        for c in cands:
            if not c["draws"]:
                continue
            assert_same_depth([dd["depth"] for dd in c["draws"]],
                              context=f"draw weights v={c['v']}")
            _w, _w_hat = draw_weights([dd["q"] for dd in c["draws"]],
                                      rho=_RHO, cvisit=CVISIT, cscale=CSCALE)
            for dd, wh in zip(c["draws"], _w_hat):
                dd["w_hat"] = float(wh)
        _cd = chosen["draws"]
        assert _cd, "chosen vertex has no face draws on the approx arm"
        _wh = np.array([dd["w_hat"] for dd in _cd], dtype=np.float64)
        _wh = _wh / _wh.sum()
        exec_draw = _cd[int(rng.choice(len(_cd), p=_wh))]
    # ``head_out`` rides out so the caller can commit the chosen action
    # without a second heads pass.
    return chosen, pi, la, legal, head_out, cands, exec_draw

# ---------------------------------------------------------------- training
# THE APPROXIMATION TERM: Sampled AlphaZero (Hubert et al. 2021) over the
# composite action a = (v, F). The search drew K i.i.d. face sequences per
# surviving vertex and weighed them w_k = rho_k * exp(sigma(q_k)); the loss
# is the face cross-entropy toward those weighted draws,
#     - lambda_f sum_v pi'_ve(v) sum_k w_hat_{v,k} log beta_theta(F_{v,k}),
# with log beta recomputed through `Agent._face_replay` ("Gradient reaches
# palimpsa through this scan"). This REPLACES the old unclipped off-policy
# REINFORCE term over replayed plans (#95) and with it the flat-coefficient
# f_logp arity problem (#76): the CE targets are the CURRENT search's stored
# improved weights, not a raw return times a whole-plan log-prob.
_FACE_COEF = float(os.environ.get("ALPHAGRAD_GAZ_FACE_COEF", "1.0"))
_FACE_ENT_COEF = float(os.environ.get("ALPHAGRAD_GAZ_FACE_ENT_COEF", "0.01"))
_W4J = jnp.asarray(np.asarray(W4, dtype=np.float32))


def loss_fn(agent, enc_M, enc_I, enc_ch, enc_nv, enc_pos, vmem_s, vmem_c,
            dtok, deqn, dcnt, owner, vsel, la_pad, la_mask, pi_pad,
            vtgt, vmask, sd_li, sd_vidx, sd_w, sd_fpair, sd_fcomp, sd_fvalid,
            sd_cnt, sd_dt, sd_de, sd_fa, sd_fends):
    """Vertex CE + value MSE + the Sampled-AZ face CE, all re-derived from
    the STORED PRE-step carry.

    §7b of the PPO design, applied here: the replay stores the carry synced to
    the PREVIOUS step's delta plus that delta, and the loss reproduces the
    step's encoding by the SAME extension the rollout ran. Gradient reaches
    palimpsa through that ``encode_extend`` and, for the face head, through
    ``_face_replay``'s scan of each stored emission window; it is truncated at
    the stored carry, exactly as in PPO. On the exact arm the ``sd_*`` slots
    are None (leafless pytrees -- vmap passes them through untouched).
    """
    from alphagrad.approx.ppo import EncCarry

    def per(M, I, ch, nv, pos, vs, vc, dt, de, dc, ow, vsl,
            la, lam, pi, vt, vm, s_li, s_vidx, s_w, s_fp, s_fc, s_fv,
            s_cnt, s_dt, s_de, s_fa, s_fend):
        carry = EncCarry(M=M, I=I, cumhist=ch, nvalid=nv, pos=pos)
        # chunk=0: AZ's loss is reverse-differentiated through this extend
        # too, and the dynamic trip count is a lax.while_loop.
        c2, vs2, vc2 = _cs.advance(agent, carry, vs, vc, dt, de, dc, ow,
                                   window=MAX_DELTA_TOKENS, chunk=0)
        vlog, ctx, v3 = _cs.heads(agent, vs2, vc2, identity_stream=VFEAT,
                                  preference=None)
        lg = vlog[la]
        lg = jnp.where(lam > 0.5, lg, -jnp.inf)   # -1e9 collided with a sentinel
        logp = jax.nn.log_softmax(lg)
        ce = -jnp.sum(jnp.where(lam > 0.5, pi * logp, 0.0))
        # Predicted CH vector from the 3 heads: [lat, mem, 0 (flops: no
        # head, _CH_ACTIVE masks it), cos]; targets vt are PopArt-normalised.
        pred = jnp.stack([v3[0], v3[1], jnp.zeros_like(v3[0]), v3[2]])
        vl = jnp.sum(vm * _CH_ACTIVE * (pred - vt) ** 2)

        # --- the per-face head: Sampled-AZ cross-entropy --------------
        # #79 EXACT arm: the approximation head is not built, so there is
        # nothing to replay and no face term. This is a PYTHON-level branch,
        # taken before `_face_replay` is traced -- a jnp.where would still
        # trace a head that does not exist.
        if _EXACT_ARM:
            return ce + 0.5 * vl, (jnp.zeros((), jnp.float32), vl, ce)
        # Each stored draw carries its own vertex index (the search's top-m,
        # NOT only `vsl`), masks and emission window; ``pi`` supplies the
        # pi'_ve(v) factor via the draw's legal-set index. `face_ce_term`
        # gates the padding slots (li == -1 / w == 0) to exactly 0.
        face_ce, face_ent_norm = face_ce_term(
            agent._face_replay, ctx, c2, AXIS_STATE, AXIS_VALID, FACT_TABLES,
            OP_OVERRIDE, _axis_feats, pi,
            s_li, s_vidx, s_w, s_fp, s_fc, s_fv, s_cnt, s_dt, s_de, s_fa,
            s_fend)
        face_loss = _FACE_COEF * face_ce - _FACE_ENT_COEF * face_ent_norm
        return ce + 0.5 * vl + face_loss, (face_ent_norm, vl, ce)

    losses, (ents, vls, ces) = jax.vmap(per)(
        enc_M, enc_I, enc_ch, enc_nv, enc_pos, vmem_s, vmem_c,
        dtok, deqn, dcnt, owner, vsel, la_pad, la_mask, pi_pad, vtgt, vmask,
        sd_li, sd_vidx, sd_w, sd_fpair, sd_fcomp, sd_fvalid,
        sd_cnt, sd_dt, sd_de, sd_fa, sd_fends)
    return jnp.mean(losses), (jnp.mean(ents), jnp.mean(vls),
                             jnp.mean(ces))


@eqx.filter_jit
def train_step(agent, opt_state, batch):
    (l, aux), gr = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        agent, *batch)
    up, opt_state = opt.update(gr, opt_state, eqx.filter(agent, eqx.is_array))
    # aux = (approx-head entropy, value MSE, vertex cross-entropy). The two
    # loss halves are reported separately: `l` alone cannot say whether the
    # value net or the policy head is the one that stopped learning.
    return eqx.apply_updates(agent, up), opt_state, l, aux

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
    # #85: `env`, `ev` and `VFEAT` join the rebound globals. AZ used to
    # draw its eval samples ONCE at import and keep them for the whole
    # run, so its quality channel could overfit a fixed reference set
    # while PPO redraws every episode. Same load-bearing `global`
    # mechanism as `agent`/`opt_state`: the search functions read these
    # through module globals, so they must be REBOUND, never shadowed.
    global agent, opt_state, env, ev, VFEAT
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
          f"mode={'deepen(d0=%d)' % args.rollout_depth if _GAZ_DEEPEN else 'widen(depth=0)'} "
          f"draws/decision<={D_MAX_DRAWS} per-face-head={not _EXACT_ARM} "
          f"max_faces={ENV_MAX_FACES}", flush=True)

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
                    if os.environ.get("ALPHAGRAD_FORCE_REV_ORDER", "0") == "1" and _wlegal:
                        _wlegal = [max(_wlegal)]   # rev: highest first
                    if not _wlegal:
                        break
                    _wv = _wlegal[int(rng.integers(len(_wlegal)))]
                    _wa = int(_wv) - 1
                    # EXACT plans. The warm start exists to fix the measurement
                    # SCALE; drawing an untrained face plan would only add
                    # variance to the (mu, sigma) it seeds.
                    PT.eliminate(_wv, EXACT_SPEC_ROW, EXACT_FACE_ROWS,
                                 EXACT_FACE_SKIPS,
                                 is_last=(len(_wlegal) == 1))
                    _wst.append((_wa, EXACT_FACE_ROWS, EXACT_FACE_SKIPS))
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
        # TRAINER-SIDE cache clear (ALPHAGRAD_GAZ_TRAINER_CLEAR_EVERY,
        # default 0=off). Under GAZ_RAY_MEASURE=1 the in-process
        # measure-cache cadence never runs, and the TRAINER process
        # accumulated executables until jit_train_step's ~18.5GiB buffer
        # stopped fitting a 96GB card (~78GB resident by ep15; v54 jobs
        # 59332/59345 died this way at batch 4 AND 2 -- it is creep, not
        # batch size). Costs one recompile of the touched jits per clear.
        # Per-episode LIVE-BUFFER census (GAZ_MEMLOG=1): clear_caches()
        # did NOT stop the creep (59352 died at ~ep17 right after the
        # ep16 clear), so the ~78GB is LIVE ARRAYS someone retains, not
        # executables. The census names them by shape.
        if os.environ.get("ALPHAGRAD_GAZ_MEMLOG", "0") == "1":
            try:
                _la = jax.live_arrays()
                _tot = sum(int(a.nbytes) for a in _la)
                from collections import Counter as _C
                _by = _C()
                for _a in _la:
                    _by[(tuple(_a.shape), str(_a.dtype))] += int(_a.nbytes)
                _top = ", ".join(
                    f"{_sh}x{_dt}={_b/2**20:.0f}MB"
                    for (_sh, _dt), _b in _by.most_common(5))
                print(f"[gaz-mem] ep={ep} live={len(_la)} "
                      f"{_tot/2**30:.2f}GiB top: {_top}", flush=True)
            except Exception as _me:
                print(f"[gaz-mem] census failed: {_me}", flush=True)
        _tce = int(os.environ.get(
            "ALPHAGRAD_GAZ_TRAINER_CLEAR_EVERY", "0") or 0)
        if _tce > 0 and ep % _tce == 0:
            import gc as _gc
            jax.clear_caches()
            _gc.collect()
            print(f"[gaz] ep={ep} trainer jax.clear_caches() "
                  f"(every {_tce})", flush=True)
        # #85: fresh calibration samples per episode (the measurement's
        # data), and the per-vertex IDENTITY re-pooled under this episode's
        # weights. Mirrors ppo.main's per-episode `generate_eval_samples` +
        # `base_identity_stream`.
        _ev_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), ep)
        ev = generate_eval_samples(env, _ev_key, A.ndata)
        env = eqx.tree_at(lambda e: e.eval_args_samples, env, ev)
        # The identity is params-dependent, not sample-dependent: it is the
        # pool's reading of this graph's base tokens under the CURRENT
        # weights, so it is rebuilt here for the same reason the carry is.
        VFEAT = _identity_stream(agent)
        # A fresh episode = a fresh tokenizer at the base, and a fresh carry
        # built from the base stream under the CURRENT weights. The carry is
        # params-dependent, so it can never outlive a train_step.
        PT.reset()
        state = []
        carry = Carry(*_carry_init(agent))
        steps = []
        _memlog = os.environ.get("ALPHAGRAD_GAZ_MEMLOG", "0") == "1"
        _dstep = 0
        # Golden-equivalence accumulator: base ++ concat(per-decision deltas).
        _gold_stream = list(_BASE_TOKS)
        _gold_ids = list(_BASE_IDS)
        # The (carry, delta) pair the LOSS re-runs. PPO's step_fn scores step t
        # off the carry BEFORE step t-1's delta plus that delta; storing the
        # post-advance carry with THIS step's delta instead is an off-by-one
        # that scores step t's target against step t+1's encoding.
        _prev_pre = carry
        _prev_delta = (np.zeros((MAX_DELTA_TOKENS,), np.int32),
                       np.full((MAX_DELTA_TOKENS,), -1, np.int32), 0)
        _prev_owner = -1
        _ep_faces = 0
        _ep_chunks = 0
        # #109 smoke hook: per-decision wall time (search + commit), default
        # off -- the buckets-on/off A/B reads these lines.
        _dtl = os.environ.get("ALPHAGRAD_GAZ_DECISION_TIMELOG", "0") == "1"
        while True:
            legal = PT.legal(VALID)
            if os.environ.get("ALPHAGRAD_FORCE_REV_ORDER", "0") == "1" and legal:
                legal = [max(legal)]   # rev: highest first
            if not legal:
                break
            _t_dec = time.perf_counter() if _dtl else 0.0
            # The committed prefix + its LiveFaceStream tokenizer, BEFORE the
            # search: under Sampled AZ the SEARCH draws face sequences (K per
            # surviving vertex, the widened halving budget), so it needs the
            # same prefix binding and face enumeration the committed
            # execution will use.
            _o_arr, _sp_h, _f_h, _s_h = _prefix_arrays(state)
            _LAST_PREFIX.update(order=_o_arr, specs=_sp_h, n=len(state),
                                face_hist=_f_h, skip_hist=_s_h)
            # THE AUTHORITATIVE FACE KEYS, from the tokenizer the head is
            # about to read its chunks from. NOT from PT: `faces_of` filters
            # LazyEdges whose thunk has already run, speculation forces them
            # IN PLACE, and `_Snapshot` cannot restore that memo -- so PT's
            # enumeration SHRINKS across branches that are otherwise perfect
            # no-ops (measured on nn256: vertices 13/16/17 lost 1/2/2 faces
            # after 8 speculative chains, with a bit-identical legal set).
            # Indexing the per-face plan by a shrunk key list drops the head's
            # decisions or applies face_rows[f] to a different face. This
            # tokenizer is rebuilt per PREFIX and is queried before any
            # `chunk()` has speculated on it, so it carries the same
            # enumeration the measurement's fresh replay will.
            _lf_tk = LIVE_FACES._tokenizer_at(_o_arr, _sp_h, len(state),
                                              _f_h, _s_h)
            _fkeys_cache = {}

            def _face_keys_of(_vv, _tk=_lf_tk, _c=_fkeys_cache):
                if _vv not in _c:
                    _c[_vv] = list(_tk.ij.faces(int(_vv)))
                return _c[_vv]

            (chosen, pi, la, legal, head_out, cands,
             exec_draw) = gumbel_search(
                state, carry, rng,
                (_o_arr, _sp_h, _f_h, _s_h, len(state)), _face_keys_of)
            v = int(chosen["v"])
            _face_keys = _face_keys_of(v)
            _nf_pre = len(_face_keys)
            if _EXACT_ARM:
                # #79 EXACT arm: no approximation head exists, so there is
                # nothing to plan. Reuse the SAME no-approximation constants
                # the speculative path uses, so the committed step is the
                # graph the search reasoned about and the measurement runs a
                # fully exact plan. Every face-derived quantity below is the
                # empty/zero form.
                fr, fs = EXACT_FACE_ROWS, EXACT_FACE_SKIPS
                fa = None
                f_valid = np.zeros((int(ENV_MAX_FACES),), np.float32)
                f_cnt = np.zeros((int(ENV_MAX_FACES),), np.int32)
            else:
                # Sampled AZ EXECUTES one of the search's own draws for the
                # chosen vertex (sampled ~ w_hat in gumbel_search) -- the
                # committed graph is one the search actually evaluated, and
                # the commit-time extra face loop is gone with it.
                fr, fs = exec_draw["fr"], exec_draw["fs"]
                fa = exec_draw["fa"]
                f_valid = exec_draw["f_valid"]
                f_cnt = exec_draw["f_cnt"]
            fr = np.asarray(fr, np.int32)
            fs = np.asarray(fs, np.int32)
            if fa is not None:
                _MICRO_CHOICES.update(_face_choice_counts(fa, f_valid))
            _nf = int(np.sum(np.asarray(f_valid) > 0.5))
            _ep_faces += _nf
            _ep_chunks += int(np.sum(np.asarray(f_cnt) > 0))
            # ---- commit: re-tokenize the vertex WITH its face wires ----
            _pre_for_loss = (_prev_pre, _prev_delta, _prev_owner)
            _carry_before = carry
            state, carry, d = _step((state, carry, head_out[1]), v, fr, fs,
                                    face_keys=_face_keys)
            _assert_terminal_prediction(d)
            # #79 EXACT arm: there is no face loop, so `f_valid`/`f_cnt` are
            # the ZERO form by construction (see the branch above) and `_nf`
            # is 0 while the tokenizer still enumerates the vertex's real
            # faces. Check (b) -- "the face loop decided as many faces as the
            # plan tokenizer enumerates" -- is a statement about the
            # APPROXIMATION head's face INDEXING; with no head there is
            # nothing indexed and the identity is vacuously false on the first
            # vertex with >= 1 face. That is why --no-approx-head has never
            # completed a single episode. Pass the enumerated count so check
            # (a) (emission-window saturation) still runs and (b) is a no-op.
            _assert_face_accounting(
                f_cnt, f_valid, d, (_nf_pre if _EXACT_ARM else _nf), v,
                _nf_pre)
            # #95: search dynamics == measured graph, ASSERTED not assumed.
            # The committed elimination must re-produce BITWISE the tokens the
            # search's branch produced for the executed draw -- same prefix,
            # same wires, same face keys; a mismatch means the search scored
            # a different graph than the measurement will tokenize.
            if exec_draw is not None:
                if not (np.array_equal(np.asarray(d["tokens"], np.int32),
                                       exec_draw["d_tokens"])
                        and np.array_equal(np.asarray(d["eqn_ids"], np.int32),
                                           exec_draw["d_eqns"])):
                    raise AssertionError(
                        f"vertex {v}: committed step tokens diverge from the "
                        f"search branch that evaluated the executed face "
                        f"draw ({len(d['tokens'])} vs "
                        f"{len(exec_draw['d_tokens'])} tokens) -- the search "
                        f"dynamics and the measured graph disagree")
            _gold_stream += list(d["tokens"])
            _gold_ids += list(d["eqn_ids"])
            if _dtl:
                print(f"[gaz][dtime] ep={ep} d={_dstep} v={v} nf={_nf} "
                      f"{time.perf_counter() - _t_dec:.3f}s", flush=True)
            steps.append({
                "enc_M": np.asarray(_pre_for_loss[0].enc.M),
                "enc_I": np.asarray(_pre_for_loss[0].enc.I),
                "enc_ch": np.asarray(_pre_for_loss[0].enc.cumhist),
                "enc_nv": np.asarray(_pre_for_loss[0].enc.nvalid),
                "enc_pos": np.asarray(_pre_for_loss[0].enc.pos),
                "vmem_s": np.asarray(_pre_for_loss[0].vs),
                "vmem_c": np.asarray(_pre_for_loss[0].vc),
                "dtok": _pre_for_loss[1][0], "deqn": _pre_for_loss[1][1],
                "dcnt": np.int32(_pre_for_loss[1][2]),
                "owner": np.int32(_pre_for_loss[2]),
                "vsel": np.int32(v - 1),
                "la": la.copy(), "pi": pi.copy(),
                # the SEARCH's face-sequence draws, flattened to the fixed
                # D_MAX_DRAWS slots: per draw its legal-set index, vertex,
                # normalized weight w_hat, the SAMPLING masks, the emission
                # window the head read and the FaceAction itself -- so
                # `_face_replay` re-scores the same decisions against the
                # same gates and the CE reweights them by pi'_ve * w_hat.
                **({} if _EXACT_ARM else _pack_search_draws(cands))})
            # slide the (carry, delta) window forward by one decision
            _prev_pre = _carry_before
            _prev_delta = d["delta"]
            _prev_owner = d["owner"]
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
        print(f"[gaz] ep={ep} face pipeline: {_dstep} decisions, "
              f"{_ep_faces} faces, {_ep_chunks} non-empty chunks "
              f"(NV x mean_faces = {_ep_faces})", flush=True)
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
        _pareto.add(_PSGN * raw, state_to_json(state), ep)
        # best-tracker: scores from different normalizer epochs are NOT comparable
        # (pre-warmup raw-scale ~-1e6 vs z-scored O(1) let a worse order overwrite a
        # better one at n=8). Keep every (raw, state) and re-argmax under the CURRENT
        # normalizer each episode.
        _solutions.append((raw.copy(), state_to_json(state), n_meas))
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
            # HOST-SIDE replay columns (root cause of the ep15-17 OOM
            # saga, census-proven: uploading the FULL flattened replay --
            # sd_fpair alone is (rows, D, 2538, 8, 8) ~ 0.5GB/episode --
            # to the GPU every episode retained a generation stack of
            # 78GiB by ep15 on 96GB. Columns stay numpy; only the sampled
            # minibatch rows are uploaded per epoch (~5MB/row).
            def stk(k):
                return np.stack([s[k] for s in flat])
            enc_M, enc_I = stk("enc_M"), stk("enc_I")
            enc_ch, enc_nv, enc_pos = stk("enc_ch"), stk("enc_nv"), stk("enc_pos")
            vmem_s, vmem_c = stk("vmem_s"), stk("vmem_c")
            dtok, deqn = stk("dtok"), stk("deqn")
            dcnt, owner, vsel = stk("dcnt"), stk("owner"), stk("vsel")
            # The search-draw columns for the Sampled-AZ face CE. On the
            # exact arm there are no draws: None is a leafless pytree, so
            # vmap and the minibatch tree_map pass it through untouched and
            # the loss's _EXACT_ARM branch never reads it.
            if _EXACT_ARM:
                sd_li = sd_vidx = sd_w = sd_fpair = sd_fcomp = None
                sd_fvalid = sd_cnt = sd_dt = sd_de = sd_fa = None
                sd_fends = None
            else:
                sd_li, sd_vidx, sd_w = stk("sd_li"), stk("sd_vidx"), stk("sd_w")
                sd_fpair, sd_fcomp = stk("sd_fpair"), stk("sd_fcomp")
                sd_fvalid, sd_cnt = stk("sd_fvalid"), stk("sd_cnt")
                sd_dt, sd_de = stk("sd_dt"), stk("sd_de")
                sd_fends = stk("sd_fends")
                sd_fa = jax.tree_util.tree_map(
                    lambda *xs: np.stack(xs),
                    *[s["sd_fa"] for s in flat])
            la_p = np.stack([pad(s["la"], MAXLA) for s in flat])
            la_m = np.stack([pad(np.ones(len(s["la"])), MAXLA) for s in flat])
            pi_p = np.stack([pad(s["pi"], MAXLA) for s in flat])
            vt = np.stack([(s["raw4"] - popart.mu) / popart.sigma for s in flat])  # PopArt-normalised
            vm = np.broadcast_to(np.abs(W4) > 0, (len(flat), 4)).astype(np.float32)
            _cols = (enc_M, enc_I, enc_ch, enc_nv, enc_pos, vmem_s, vmem_c,
                     dtok, deqn, dcnt, owner, vsel, la_p, la_m, pi_p,
                     vt, vm, sd_li, sd_vidx, sd_w, sd_fpair, sd_fcomp,
                     sd_fvalid, sd_cnt, sd_dt, sd_de, sd_fa, sd_fends)
            # M5: resample the minibatch EVERY epoch. Taking one fixed 64-sample
            # draw and hitting it ``train_epochs`` times overfits that draw and
            # discards the rest of the replay for this update.
            _bs = int(os.environ.get("ALPHAGRAD_GAZ_BATCH", "64"))
            for _ in range(args.train_epochs):
                idx = rng.permutation(len(flat))[:_bs]     # numpy sample
                batch = tuple(
                    jax.tree_util.tree_map(
                        lambda a: jnp.asarray(a[idx]), x)
                    for x in _cols)
                agent, opt_state, L, _AUX = train_step(
                    agent, opt_state, batch)
            _AH, _VL, _CE = _AUX
            L = float(L)
            _VLOSS, _CELOSS = float(_VL), float(_CE)
            # ACCEPTANCE (b): the approximation head is no longer frozen. This
            # is the arity-normalised per-FACE entropy from `_face_replay`,
            # the same quantity PPO logs under this key.
            _AP_ENT.append(float(_AH))
        else:
            L = float("nan")
            _VLOSS, _CELOSS = float("nan"), float("nan")
        # --- approximation telemetry ---------------------------------
        # Built OUTSIDE the wandb block. It used to live inside it, so a
        # --wandb-off probe could not see `approx_applied/*` at all -- and
        # that key is precisely the one that proves the face wires reached
        # the measurement (a flat 0 means they never did).
        _approx_telemetry = {}
        _tot = sum(_MICRO_CHOICES.values()) or 1
        # REALIZED per-FACE class fractions, over the faces that exist. Same
        # head, same denominator as PPO's -- the per-VERTEX-vs-per-FACE
        # caveat that used to qualify this panel is gone with the per-vertex
        # action space.
        for _nm in ("none", "diag", "compress", "quant", "skip"):
            _approx_telemetry[f"approx_prob/{_nm}"] = _MICRO_CHOICES[_nm] / _tot
        _pf = consume_per_face_stats()
        # In-process measurement fills these directly; with a Ray measure pool
        # the hooks run in the actors, so merge their counters or every
        # approx_applied/* key logs as a flat 0.
        try:
            from alphagrad.approx.common.measure_pool import (
                merge_pool_face_stats as _merge_pf)
            _pf = _merge_pf(_MEASURE_POOL, _pf)
        except Exception:
            pass
        for _k in ("diag", "compress", "quant"):
            _approx_telemetry[f"approx_applied/{_k}"] = _pf.get(f"applied_{_k}", 0)
            _approx_telemetry[f"approx_skipped/{_k}"] = _pf.get(f"skipped_{_k}", 0)
        _approx_telemetry["approx_applied/total"] = _pf.get("applied", 0)
        _approx_telemetry["approx_skipped/total"] = (
            _pf.get("skipped", 0) + _pf.get("skipped_raised", 0))
        _approx_telemetry["approx_applied/fraction"] = _pf.get(
            "applied_fraction", 0.0)
        if os.environ.get("ALPHAGRAD_DEBUG_APPROX_PROB", "0") == "1":
            print("[approx per-face] " + " ".join(
                f"{_k2.split('/')[-1]}={float(_v2):.4g}"
                for _k2, _v2 in sorted(_approx_telemetry.items())), flush=True)
            _edbg = {}
            if _VE_ENT:
                _edbg["ve_head"] = float(np.mean(_VE_ENT))
            if _AP_ENT:
                _edbg["approx_head"] = float(np.mean(_AP_ENT))
            _edbg["tied_candidates"] = float(_TIED_CANDIDATES[0])
            _edbg["window_saturated"] = (
                _FACE_WINDOW_SATURATED[0] / max(_FACE_STEPS[0], 1))
            print("[entropy] " + " ".join(
                f"{_k3}={_v3:.4g}" for _k3, _v3 in sorted(_edbg.items())),
                flush=True)
        if wb is None:
            # No wandb row to gate on, so drain the per-episode counters here
            # or every mean silently becomes a run-to-date mean.
            _MICRO_CHOICES.clear(); _VE_ENT.clear(); _AP_ENT.clear()
            _TIED_CANDIDATES[0] = 0; _TIED_TOTAL[0] = 0
            _FACE_WINDOW_SATURATED[0] = 0; _FACE_STEPS[0] = 0
        # PER-EPISODE JSONL SINK (ALPHAGRAD_GAZ_JSONL, default off). Written
        # append-only and BEFORE any dump, so a killed run keeps every
        # measured episode -- the final gaz_result.json drops `solutions`.
        if _GAZ_JSONL:
            try:
                _o = np.asarray(plan_wires(state)[0]).reshape(-1).tolist()
                _rw = np.asarray(raw, dtype=np.float64).reshape(-1).tolist()
                _bb = best.get("raw")
                _rec = {
                    "ep": int(ep),
                    "n_meas": int(n_meas),
                    "order": [int(v) for v in _o],
                    "raw": _rw,
                    "raw_names": ["latency_ns", "xla_peak_bytes",
                                  "flops", "cos"],
                    "loss": float(L),
                    "value_loss": float(_VLOSS),
                    "vertex_ce": float(_CELOSS),
                    "best_scalar": float(best.get("scalar", float("nan"))),
                    "best_raw": (list(_bb) if _bb is not None else None),
                    "best_at": int(best.get("at", 0)),
                    # PRIMARY READOUT. The analytic entropy of an exactly
                    # UNIFORM policy over the shrinking legal set across a
                    # 95-step elimination is mean_{k=1..95} ln k = 3.58749
                    # nats; a run that ends there never left the random
                    # policy, whatever its best latency says.
                    "entropy/ve_head": (float(np.mean(_VE_ENT))
                                       if _VE_ENT else float("nan")),
                    "entropy/ve_head_std": (float(np.std(_VE_ENT))
                                           if _VE_ENT else float("nan")),
                    "entropy/ve_head_n": int(len(_VE_ENT)),
                    "popart_mu": np.asarray(
                        popart.mu, np.float64).reshape(-1).tolist(),
                    "popart_sigma": np.asarray(
                        popart.sigma, np.float64).reshape(-1).tolist(),
                }
                _rec.update(_SEARCH_DIAG)
                # EPISODE-level search telemetry: the completed-Q spread the
                # depth-0 bandit actually trained on, over every root that had
                # a real choice.
                for _k, _vs in _SEARCH_ACC.items():
                    if not _vs:
                        continue
                    _a = np.asarray(_vs, np.float64)
                    _rec["ep_search/" + _k + "_mean"] = float(_a.mean())
                    _rec["ep_search/" + _k + "_median"] = float(np.median(_a))
                    _rec["ep_search/" + _k + "_max"] = float(_a.max())
                _rec["ep_search/n_roots"] = int(
                    len(_SEARCH_ACC.get("comp_q_std", [])))
                _SEARCH_ACC.clear()
                with open(_GAZ_JSONL, "a") as _fh:
                    _fh.write(json.dumps(_rec, default=float) + "\n")
            except Exception as _je:
                print(f"[gaz-jsonl] write failed: {_je}", flush=True)
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
                   # PER-FACE plans. Each `state` entry carries {vertex,
                   # face_rows, face_skips} -- the full wire, so a dump
                   # re-measures to the cosine/latency the run reported. The
                   # old per-vertex-micro dumps are not replayable under
                   # per-face approximation.
                   "wire": "per_face_v1",
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
                    # 0-BASED, to match ppo. `ep` is incremented at the TOP of
                    # the loop here, so az's FIRST episode is ep == 1 while
                    # ppo's is ep == 0: plotting the shared `time/episode`
                    # axis put az one step to the right of ppo for the whole
                    # run. The az-native `ep` key above keeps the raw 1-based
                    # counter that the stdout lines and the pareto stamps use.
                    "time/episode": ep - 1,
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
                _log.update(_approx_telemetry)
                # per-head percentile companions + the PopArt state that
                # produced them. AZ channel order is [lat, peak, flops, cos];
                # flops has W4 == 0 and no head, so it is skipped.
                try:
                    for _hi, _hn in ((0, "latency"), (1, "mem"), (3, "cos")):
                        if _phi_v is not None and _hi < _phi_v.shape[0]:
                            _log[f"weighted_mean_{_hn}"] = float(_phi_v[_hi])
                        # SPACE CAVEAT (audited 2026-08-07): az's PopArt
                        # tracks the RAW TERMINAL measurement vector; ppo's
                        # tracks `estim_returns`, the GAE-bootstrapped
                        # DISCOUNTED return over `_symlog_rewards(reward)`.
                        # Under the campaign config
                        # (--terminal-rewards-only --no-symlog) the two agree
                        # up to the bootstrap and the discount; under any
                        # other ppo config they do not. The key names the same
                        # ROLE on both arms (the normaliser the value head is
                        # rescaled by), which is why it keeps one name.
                        _log[f"popart/mu_{_hn}"] = float(
                            np.asarray(popart.mu).reshape(-1)[_hi])
                        _log[f"popart/sigma_{_hn}"] = float(
                            np.asarray(popart.sigma).reshape(-1)[_hi])
                except Exception:
                    pass
                # Episode-mean policy entropies. Keys match ppo.py's and so
                # do the quantities: the VE head is the same distribution on
                # both arms, and the approximation head is the same per-FACE
                # UnifiedFacePolicy. Omitted (not logged as 0) when the head
                # has not been scored yet (before the first train step), so an
                # absent panel means "not applicable" rather than "collapsed".
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
                _log["faces/window_saturated"] = (
                    _FACE_WINDOW_SATURATED[0] / max(_FACE_STEPS[0], 1))
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
                    _FACE_WINDOW_SATURATED[0] = 0
                    _FACE_STEPS[0] = 0
                    _VE_ENT.clear()
                    _AP_ENT.clear()
                    _t_prev = _now

    # --- 6. logging/dump ---
    json.dump({"best": best, "n_measured": n_meas, "config": vars(args),
               "wire": "per_face_v1"},
              open(os.path.join(args.out, "gaz_result.json"), "w"),
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
