"""DEPRECATED — DOES NOT RUN against this tree.

Verified by two independent audits (2026-07-26): the Ray line dies before the
first update with at least five independent failures — the 8-vs-10 reward
channel skew (PPORayWorker.__init__ indexes channel 9 of an 8-vector), the
unported FactoredQuantHead (`quant_dtype_head.proj` no longer exists; MicroAction
needs `quant_scale_sign`), and 100% sentinel measurements (`_callback` does not
accept the `point_idx=` this line always passes, so every measurement is
swallowed into -1e10). The advertised async pipeline cannot start at all
(no `--p3o` flag; its learner `train_on_trajs` was removed but is still called).

KEPT AS A DESIGN REFERENCE ONLY. The parts worth reading are catalogued in
alphagrad/COMPONENTS.md; the ones worth having have been ported to ppo.py
(PopArt, Pareto + hypervolume, the multiplicative cosine gate, per-component KL).
Use `ppo.py`.

----------------------------------------------------------------------------
(The banner above was inserted as a SECOND string literal, which made the
`from __future__` import below a SyntaxError — the whole module, and through
`policy.py` every GAZ entry point, failed to import. Merged into one
docstring.)

Ray-actor host of PPO training using the external-tokenizer path.

First cut of the Phase-3+5 PPO Ray version from the migration plan:

* JAX-side rollout uses `env.step_external_jax_part` (no `io_callback`).
  Per timestep the worker JIT-runs vmap(act + step_external) over the
  env batch, ships the resulting `(order, specs)` triples to a pool of
  `CpuApproximationActor` Ray actors for the graphax tokenize / jacve
  compile / cost-analysis work, then stitches the tokenizer output back
  in via `env.assemble_step_result`. The result is functionally
  identical to `env.step()` but moves the host-side Python pass off the
  trainer thread.

* PPO update is a deliberately simple clip-loss + value-MSE + entropy.
  We do NOT replicate every feature of the single-process `ppo.py`
  trainer here (no dynamic-substeps /
  preference Dirichlet / replay buffer — those will land as follow-ups
  if the basic path proves out). The trainer scalarises the env's
  8-dim reward vector to a single scalar via a fixed weight vector
  derived from `--rewards / --cmp-type / --mem-type / --lambda-*` —
  same shape as `ppo._build_reward_weights`.

* Single-GPU for now. The mu0_ray_worker SPMD shard (Mesh + NamedSharding)
  is the natural follow-up but kept out of the first cut to limit
  surface area.

Interface (matches `mu0_ray_worker.SPMDServerWorker` shape so the driver
in `ppo_ray.py` can drop in identically):

    worker = PPORayWorker(args_dict, seed, cpu_workers=cpu_workers)
    stats = worker.run_rollout_and_train(rng_seed)
"""

from __future__ import annotations

import os
import time as _time
from functools import partial
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from alphagrad.approx.common import (
    build_pair_valid_mask,
    build_vertex_valid_static,
    data_gen,
    generate_eval_samples,
    get_args,
    get_fn,
    grad_target_setup,
    infer_argnums,
    init_linear_weights,
    vertex_avail_at_step,
)
from alphagrad.approx.common.gae import reward_normalization_fn
from alphagrad.utils import symlog
from alphagrad.approx.env import (
    AXIS_FEATURE_DIM,
    MAX_AXES_PER_VERTEX,
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    NUM_REWARDS,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
    micro_actions_to_rule_specs_jax,
    _AXIS_FEAT_SIZE,
)
from alphagrad.approx.heads import (
    MAX_PRIMES,
    MicroAction,
    MicroActionPolicy,
    OP_COMPRESS,
    OP_DIAG,
    OP_END,
    OP_QUANT,
    precompute_factor_tables,
)
from graphax.sparse.micro_actions import NUM_QUANT_DTYPES


# Which reward channels skip symlog when compared against a threshold
# (the --reward-condition gates). Currently just cosine_sim — it's
# already bounded to [0, 1], so symlog'ing it would only complicate the
# threshold semantics. Mirrored from ppo._NO_SYMLOG_REWARD_INDICES.
_NO_SYMLOG_REWARD_INDICES: tuple[int, ...] = (REWARD_INDEX["cosine_sim"],)

# Channel-index aliases used by the multiplicative cosine-gate reward.
COSINE_SIM_IDX: int = REWARD_INDEX["cosine_sim"]
FROB_RESIDUAL_IDX: int = REWARD_INDEX["frob_residual"]

# ADDITIVE_SYMLOG_COST: when ALPHAGRAD_ADDITIVE_SYMLOG_COST=1 (additive mode),
# symlog-compress the COST channels in the reward buffer before the scalar
# weighted sum + GAE. Without this the raw cost magnitudes (peak_memory ~1.7e6,
# latency ~2.4e4) dominate the scalar reward by 4-6 orders of magnitude over
# the [0,1] quality channels (cosine_sim / bkstep_acc), so bkstep(fracred) and
# the capped-cossim guide can never influence the advantage direction — the
# "flat-zero-basin" is really a scale-domination basin. Symlog brings
# latency->~10, peak_memory->~14 so lambda_cmp/lambda_mem (~0.005) put them on
# ~0.05-0.07 footing, commensurate with lambda_acc*bkstep (~0.5) and the guide.
# Mirrors ppo._symlog_rewards. cosine_sim + bkstep_acc are [0,1] quality
# channels and stay RAW (excluded from the mask).
_COST_SYMLOG_INDICES: tuple[int, ...] = tuple(
    i for i in range(NUM_REWARDS)
    if REWARD_NAMES[i] not in ("cosine_sim", "bkstep_acc")
)
_COST_SYMLOG_MASK_NP: np.ndarray = np.zeros((NUM_REWARDS,), dtype=bool)
_COST_SYMLOG_MASK_NP[list(_COST_SYMLOG_INDICES)] = True

# INNER-LAMBDA (per cost channel). The additive cost term was
# ``lambda_outer * symlog(raw_cost)``: because raw costs are HUGE (latency
# ~1.1e5 ns, peak_memory ~1.3e8 B) they saturate symlog's log regime, so a
# 2x-cheaper order moves the term by only ~symlog(2x)-symlog(1x) ~ log(2)/raw
# -> a few-e-4 delta once multiplied by lambda_outer. The cost channel was
# effectively CONSTANT (rank-blind). Fix: move the lambda INSIDE the symlog and
# make it a PER-CHANNEL RESCALE ~ 1/typical_raw_cost, so a typical cost lands in
# symlog's LINEAR regime (|lambda_inner*cost| ~ 1); order-of-magnitude cost
# differences are then PRESERVED (sensitive/rankable) while 10-100x outliers
# still get log-bounded. The scalar cost contribution becomes
#     w_outer * symlog(lambda_inner_c * raw_cost)
# with a SMALL w_outer so the whole cost term stays a minor nudge (~0.05-0.1)
# vs the bkstep quality term (~0.5). Defaults sized from the ep-20 best.json of
# job 51325 (raw latency ~1.13e5 ns, raw peak_memory ~1.30e8 B). Per-channel
# override via ALPHAGRAD_INNER_LAMBDA_<CHANNELNAME> (e.g.
# ALPHAGRAD_INNER_LAMBDA_LATENCY_NS). Any cost channel without an override
# defaults to 1.0 (i.e. legacy plain symlog(raw)); the two REWARDED cost
# channels (latency_ns, peak_memory) get calibrated defaults below.
_INNER_LAMBDA_DEFAULTS: dict[str, float] = {
    "latency_ns": 9e-6,     # 1/1.13e5  -> typical latency -> symlog arg ~1.0
    "peak_memory": 7.7e-9,  # 1/1.30e8  -> typical peak_mem -> symlog arg ~1.0
}



def _traced_inlined(target_fn, xs):
    """``jax.make_jaxpr(target_fn)(*xs)``, numbered on the form that is actually
    eliminated -- see the twin in ``cpu_approx_worker`` / ``ppo``.

    jacve and the AOJ splice jit/pjit bodies in before eliminating, which ADDS
    equations; numbering vertices from the raw trace addresses a different graph
    and leaves the spliced-in vertices un-eliminated.
    """
    import jax
    from graphax import inline_call_primitives

    cj = jax.make_jaxpr(target_fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    try:                                   # jax >= 0.4.31
        from jax.extend.core import ClosedJaxpr
    except ImportError:                    # older / internal layout
        from jax._src.core import ClosedJaxpr
    return ClosedJaxpr(jx, consts)


def _build_inner_lambda_vec() -> np.ndarray:
    """Per-channel INNER lambda applied INSIDE symlog for cost channels.

    ``symlog(lambda_inner_c * raw_cost)`` instead of ``symlog(raw_cost)``.
    Defaults from :data:`_INNER_LAMBDA_DEFAULTS` (calibrated for the rewarded
    cost channels), 1.0 elsewhere, overridable per-channel via
    ``ALPHAGRAD_INNER_LAMBDA_<UPPER_CHANNEL_NAME>``. Quality channels
    (cosine_sim, bkstep_acc — masked out of the symlog pass) always keep 1.0.
    """
    vec = np.ones((NUM_REWARDS,), dtype=np.float64)
    for i in range(NUM_REWARDS):
        name = REWARD_NAMES[i]
        default = _INNER_LAMBDA_DEFAULTS.get(name, 1.0)
        env_key = "ALPHAGRAD_INNER_LAMBDA_" + name.upper()
        val = os.environ.get(env_key)
        if val is not None and val.strip() != "":
            try:
                default = float(val)
            except ValueError:
                pass
        vec[i] = default
    return vec.astype(np.float32)

# Fix 1 (v2): NON-symlog quality channels (cosine_sim, bkstep_acc). The
# additive failed-penalty is stamped on one of these so no symlog pass
# reshapes it (the second additive-symlog pass symlog's bkstep_acc, so we
# must stamp AFTER it -- see run_rollout_and_train).
from alphagrad.approx.common.reward_scaling import (
    NO_SYMLOG_MASK_NP as _NO_SYMLOG_MASK_FP,
)



def _args_from_dict(args_dict: dict) -> SimpleNamespace:
    """Back-compat re-export of the canonical helper."""
    from alphagrad.approx.common.ray_runtime import _args_from_dict as _impl
    return _impl(args_dict)


def _setup_jax_compile_cache() -> None:
    """Back-compat wrapper around
    :func:`alphagrad.approx.common.compile_cache.setup_jax_compile_cache`."""
    from alphagrad.approx.common.compile_cache import setup_jax_compile_cache
    setup_jax_compile_cache()


def _build_reward_weights(args) -> np.ndarray:
    """Back-compat wrapper around
    :func:`alphagrad.approx.common.reward_scaling.build_reward_weights`."""
    from alphagrad.approx.common.reward_scaling import build_reward_weights
    return build_reward_weights(args)


def _quality_is_rewarded(args) -> bool:
    """True iff cosine_sim OR frob_residual carries a non-zero reward weight.

    Derived from the SAME ``build_reward_weights`` the reward uses, so the
    env's exact-Jacobian skip (config.quality_rewarded) can never diverge from
    the actual reward. When False the env skips the exact reference Jacobian
    (a full jacrev per terminal step). Any failure -> True (safe: compute it)."""
    try:
        from alphagrad.approx.common.reward_scaling import build_reward_weights
        from alphagrad.approx.env import REWARD_INDEX
        w = build_reward_weights(args)
        return bool(
            w[REWARD_INDEX["cosine_sim"]] != 0.0
            or w[REWARD_INDEX["frob_residual"]] != 0.0
        )
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Policy-V2 agent — the ONLY policy on the Ray PPO path: the rich head stack
# from heads.py / ppo.py.
#
#   * Vertex selection: `ppo.PointerVertexPolicy` — per-vertex learned queries
#     cross-attend over the PER-TOKEN encoder embeddings (not a mean-pool MLP);
#     the cross-attended per-vertex representation is the context that
#     conditions the micro-action sub-episode, exactly as in ppo.py::Agent.
#   * Rule head: `heads.MicroActionPolicy` — the autoregressive typed
#     sub-episode (op-type -> axis-i (op-conditional mask) -> axis-j | i ->
#     prime-exponent factor -> compress-kind | i -> quant-dtype), scanned to
#     `--max-substeps` with sticky END + the 2x-num-axes hard cap. Joint
#     log-prob / entropy are GATED per component (END contributes only lp_op),
#     killing the off-policy-heads issue the removed flat-head agent had.
#   * Value: 10-channel (NUM_REWARDS) head fed from its OWN learned
#     attention pool over the per-token embeddings — policy and value no
#     longer share one 128-d mean-pool bottleneck.
#
# The encoder honours the ALPHAGRAD_POLICY backbone, with eqn_ids + pad-mask
# threading for the palimpsa recurrence.
# ---------------------------------------------------------------------------
class MicroPPOAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: Any
    encoder: Any
    final_norm: eqx.nn.LayerNorm
    vertex_policy: Any        # ppo.PointerVertexPolicy
    value_pool_query: jax.Array
    value_head: Any           # MLP value-pooled -> (NUM_REWARDS,)
    cost_pool_query: jax.Array  # cost-head aux: own attention-pool query
    cost_head: Any            # MLP cost-pooled -> (NUM_REWARDS,) measured 4-tuple
    micro_action_policy: Any  # heads.MicroActionPolicy

    embd_dim: int = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)
    max_substeps: int = eqx.field(static=True)
    policy: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_size: int,
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_vertices: int,
        value_dims: tuple[int, ...],
        key,
        max_substeps: int = 16,
        policy: str = "transformer",
    ):
        from alphagrad.transformer import MLP, Encoder, PositionalEncoder, make_encoder
        # Lazy: only the micro-policy path pays for the 6k-line ppo module.
        from alphagrad.approx.ppo import PointerVertexPolicy

        keys = jrand.split(key, 8)
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=keys[0])
        self.pos_enc = PositionalEncoder(embd_dim, MAX_TOKENS)
        self.encoder = make_encoder(
            policy, num_layers, num_heads, embd_dim, hidden_dim,
            key=keys[1],
        )
        # Final LayerNorm over the residual stream. The encoder blocks are
        # pre-LN (norm INSIDE each block, residual outside), so the stream's
        # scale grows unbounded across layers — under the palimpsa mixers
        # (whose recurrence output isn't softmax-bounded) the per-token
        # embeddings reach ~1e6 at init, saturating every downstream head
        # (deterministic op/vertex sampling, entropy ~= 0, log(1e-8)
        # log-probs). Standard pre-LN closing norm; O(1) inputs for the
        # pointer / value-pool / micro heads.
        self.final_norm = eqx.nn.LayerNorm(embd_dim)
        self.vertex_policy = PointerVertexPolicy(
            num_vertices=num_vertices, embd_dim=embd_dim,
            num_heads=num_heads, key=keys[2],
        )
        self.value_pool_query = jrand.normal(keys[3], (embd_dim,)) * 0.02
        self.value_head = MLP(embd_dim, NUM_REWARDS, value_dims, key=keys[4])
        # Cost-head aux (ALPHAGRAD_COST_HEAD_AUX): own attention pool + MLP
        # predicting the TERMINAL MEASURED 4-tuple in symlog space. Same shape
        # as the Phase-1 probe head. Always constructed (cheap); only trained
        # when the flag is on (weight 0 otherwise => value/policy unchanged).
        self.cost_pool_query = jrand.normal(keys[6], (embd_dim,)) * 0.02
        self.cost_head = MLP(embd_dim, NUM_REWARDS, value_dims, key=keys[7])
        self.micro_action_policy = MicroActionPolicy(
            embd_dim=embd_dim, num_heads=num_heads,
            max_substeps=max_substeps, key=keys[5],
        )
        self.embd_dim = embd_dim
        self.num_vertices = num_vertices
        self.max_substeps = max_substeps
        self.policy = policy

    def encode_tokens(self, tokens, key, eqn_ids=None):
        """Per-token encoder pass. Returns ``(enc_x (S, E), token_mask (S,))``.

        Same eqn_ids + pad-mask threading as the encoder pass (the
        palimpsa recurrence must not accumulate the ~16k pad positions),
        but WITHOUT the mean-pool — the pointer / value heads consume the
        per-token embeddings directly.
        """
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        pad_tok = tokens > 0
        enc_mask = pad_tok if self.policy in ("palimpsa", "palimpsa_bi") else None
        enc_x = self.encoder(x, eqn_ids=eqn_ids, mask=enc_mask, key=key)
        enc_x = jax.vmap(self.final_norm)(enc_x)
        return enc_x, pad_tok

    def value_from_encoding(self, enc_x, token_mask):
        """10-channel value from the value head's OWN attention pool."""
        scores = (enc_x @ self.value_pool_query) / jnp.sqrt(
            jnp.float32(self.embd_dim)
        )
        scores = jnp.where(token_mask, scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        pooled = jnp.sum(attn[:, None] * enc_x, axis=0)
        return self.value_head(pooled)  # (NUM_REWARDS,)

    def cost_from_encoding(self, enc_x, token_mask):
        """Cost-head aux prediction: raw MEASURED 4-tuple (symlog space) from
        the cost head's OWN attention pool over the shared encoder output."""
        scores = (enc_x @ self.cost_pool_query) / jnp.sqrt(
            jnp.float32(self.embd_dim)
        )
        scores = jnp.where(token_mask, scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        pooled = jnp.sum(attn[:, None] * enc_x, axis=0)
        return self.cost_head(pooled)  # (NUM_REWARDS,) symlog measured cost

    def policy_value_from_encoding(self, enc_x, token_mask):
        """``(vertex_logits (V,), vertex_contexts (V, E), value (K,))``."""
        vertex_logits, vertex_contexts = self.vertex_policy(enc_x, token_mask)
        value = self.value_from_encoding(enc_x, token_mask)
        return vertex_logits, vertex_contexts, value

    def value(self, tokens, key, eqn_ids=None):
        """Bootstrap-path value from raw tokens (encode + value pool)."""
        enc_x, token_mask = self.encode_tokens(tokens, key=key, eqn_ids=eqn_ids)
        return self.value_from_encoding(enc_x, token_mask)


def _scale_micro_policy_heads(agent, scale: float):
    """Scale the V2 policy-head output weights for a near-uniform initial
    policy (ppo.py's ``--head-init-scale`` pattern, default 0.1).

    Without this the freshly orthogonal-initialised categorical projections
    ride on the un-normalised AxisSetEncoder summary and saturate — at init
    P(END) ~ 1e-8, so every sub-episode runs to the hard cap and the op head
    sees near-zero gradients (softmax saturated). Scaled heads start
    near-uniform, so END / DIAG / COMPRESS / QUANT all stay explorable.
    Value heads are left untouched (mirrors ppo._scale_output_heads).
    """
    from alphagrad.approx.common.init import scale_module_weight
    getters = [
        lambda a: a.vertex_policy.pointer_proj.weight,
        lambda a: a.micro_action_policy.head.op_head.proj.weight,
        lambda a: a.micro_action_policy.head.axis_i_head.query_proj.weight,
        lambda a: a.micro_action_policy.head.axis_j_head.query_proj.weight,
        lambda a: a.micro_action_policy.head.factor_head.head_proj.weight,
        lambda a: a.micro_action_policy.head.compress_kind_head.proj.weight,
        lambda a: a.micro_action_policy.head.quant_dtype_head.proj.weight,
    ]
    for g in getters:
        agent = scale_module_weight(agent, g, scale)
    return agent


def _rezero_encoder_rel_gates(agent):
    """Re-zero the encoder mixers' ``rel_gate`` after init_linear_weights.

    ``init_linear_weights`` orthogonally re-initialises EVERY ``eqx.nn.Linear``
    in the agent — including the Palimpsa mixers' relational DAG-degree gate
    ``rel_gate``, whose ZERO init is load-bearing (the mixer must start as
    pure paper-Palimpsa and only deviate once the structural modulation
    learns to). Restore the zeros here.
    """
    for li in range(len(agent.encoder.layers)):
        mixer = agent.encoder.layers[li].attn_layer
        rg = getattr(mixer, "rel_gate", None)
        if rg is None:
            continue
        zeroed = eqx.tree_at(lambda l: l.weight, rg, jnp.zeros_like(rg.weight))
        zeroed = eqx.tree_at(lambda l: l.bias, zeroed, jnp.zeros_like(rg.bias))
        agent = eqx.tree_at(
            lambda a, _li=li: a.encoder.layers[_li].attn_layer.rel_gate,
            agent,
            zeroed,
        )
    return agent


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class PPORayWorker:
    """Owns env + agent + opt_state for one Ray trainer."""

    def __init__(self, args_dict: dict, seed: int = 0, cpu_workers: list | None = None):
        self.args = _args_from_dict(args_dict)
        self.seed = int(seed)
        self.cpu_workers = list(cpu_workers) if cpu_workers else []
        # Phase 3: optional CpuApproxPool for timeout-bounded tokenisation.
        # Created lazily by `init_server` (mirrors mu0_ray_actors.SPMDActor's
        # API). When None, `_fan_out_tokenize` falls back to the naive
        # per-actor `ray.get` path (no timeout — original behaviour).
        self._cpu_pool = None
        # ALPHAGRAD_LOCAL_TOKENIZE=1: compute non-terminal state-tokens
        # in-process on the trainer (skip the per-step Ray round-trip).
        # Default OFF -> byte-identical to the all-Ray path. Only valid
        # under ``terminal_rewards_only`` (non-terminal rewards zeroed,
        # tokens a pure deterministic fn of (order, specs, step)).
        self._local_tokenize_enabled = (
            os.environ.get("ALPHAGRAD_LOCAL_TOKENIZE", "0") == "1"
        )
        # Process-local in-proc server for the local-tokenize path.
        # Built once in ``init_server`` (and lazily on first use as a
        # fallback). Reuses the same ``env._callback`` the CPU pool runs.
        self._in_proc_server = None
        # ``--terminal-rewards-only`` was renamed to
        # ``--intermediate-rewards`` (BooleanOptionalAction, default
        # False). Derive terminal_rewards_only the same way the CPU
        # actor does (cpu_approx_worker._build_env_from_args), so the
        # trainer's own env (and the in-proc server built from it) agree
        # with the pool on whether non-terminal steps skip the heavy
        # jacve compile/exec. Fall back to the legacy key.
        if hasattr(self.args, "intermediate_rewards"):
            self._terminal_rewards_only = not bool(
                self.args.intermediate_rewards
            )
        else:
            self._terminal_rewards_only = bool(
                getattr(self.args, "terminal_rewards_only", False)
            )
        # Checkpoint path is captured here; loading happens after the
        # agent + opt_state templates are constructed (so eqx has
        # something to fill the leaves into).
        self._checkpoint_path = getattr(self.args, "checkpoint_path", "") or ""
        self._checkpoint_every = int(getattr(self.args, "checkpoint_every", 0))
        # Env-var aliases (bridge-cse): ALPHAGRAD_CHECKPOINT_PATH /
        # ALPHAGRAD_CHECKPOINT_EVERY let a launcher enable periodic agent.eqx
        # saves without the CLI flags. Only OVERRIDE when the CLI left them
        # unset (path empty), so an explicit --checkpoint-path always wins and
        # the unset/no-env case stays byte-identical (checkpointing OFF).
        import os as _os_ck
        if not self._checkpoint_path:
            _env_path = str(_os_ck.environ.get("ALPHAGRAD_CHECKPOINT_PATH", "") or "").strip()
            if _env_path:
                self._checkpoint_path = _env_path
        _env_every = str(_os_ck.environ.get("ALPHAGRAD_CHECKPOINT_EVERY", "") or "").strip()
        if _env_every:
            self._checkpoint_every = int(_env_every)
        # Cost-head auxiliary task (bridge-cse). OFF => weight 0 => the aux
        # loss term vanishes and the update is byte-identical to baseline.
        self._cost_head_aux = (
            _os_ck.environ.get("ALPHAGRAD_COST_HEAD_AUX", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self._cost_head_aux_weight = float(
            _os_ck.environ.get("ALPHAGRAD_COST_HEAD_AUX_WEIGHT", "0.3") or 0.3
        ) if self._cost_head_aux else 0.0
        # Encoder-grad flow toggle (default ON = the representation-learning
        # benefit). OFF => stop_gradient on the encoder ctx for the aux loss.
        self._cost_head_aux_encoder_grad = (
            _os_ck.environ.get("ALPHAGRAD_COST_HEAD_AUX_ENCODER_GRAD", "1").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if self.args.no_jit:
            jax.config.update("jax_disable_jit", True)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        _setup_jax_compile_cache()

        key = jrand.PRNGKey(self.seed)
        key, args_key, eval_key, agent_key, init_key = jrand.split(key, 5)
        self._rng_key = key

        # Build env exactly like cpu_approx_worker but keep the JAX-side
        # references locally — we need both `args` (for tokenizer/eval)
        # and `valid_vertices` for the action mask.
        dataset_arg = None if self.args.dataset == "none" else self.args.dataset
        use_dataset = (
            dataset_arg is not None
            and self.args.example.endswith("NeuralNetwork")
        )
        dataset_for_call = dataset_arg if use_dataset else None
        target_fn = get_fn(self.args.example)
        xs = get_args(self.args.example, args_key, dataset=dataset_for_call)
        gen = data_gen(
            self.args.example,
            dataset=dataset_for_call,
            dataset_size=self.args.dataset_size,
        )
        # GRAD MODE (--measure-grad): wrap into the scalar-loss (+ optional
        # seed-vertex) graph and shift argnums, IDENTICALLY to the measure
        # actor (cpu_approx_worker._build_env_from_args). This makes the POLICY
        # env's jaxpr / valid_vertices / argnums match the graph the actor
        # actually differentiates, so the policy emits a COMPLETE elimination
        # order over the grad graph (incl. the output / loss-reduction
        # vertices). A Jacobian-graph order is incomplete for the grad graph
        # and graphax value_and_grad then returns a structurally-zero gradient.
        target_fn, xs, argnums = grad_target_setup(
            self.args, target_fn, xs, self.args.example
        )
        closed_jaxpr = _traced_inlined(target_fn, xs)
        env_target_fun = target_fn if "acc" in self.args.rewards else None
        measure_latency = bool(
            getattr(self.args, "measure_latency", False)
            or self.args.cmp_type == "latency"
        )
        env = VertexEliminationEnv.from_jaxpr(
            closed_jaxpr,
            args=xs,
            argnums=argnums,
            num_envs=0,
            data_gen=gen,
            target_fun=env_target_fun,
            cmp_type=self.args.cmp_type,
            mem_type=self.args.mem_type,
            exec_on_gpu=getattr(self.args, "exec_on_gpu", False),
            measure_latency=measure_latency,
            terminal_rewards_only=self._terminal_rewards_only,
            measure_grad=bool(getattr(self.args, "measure_grad", False)),
            # PERF (bridge-cse): skip the exact reference Jacobian when neither
            # cosine_sim nor frob_residual is rewarded — derived from the SAME
            # weight vector the reward uses, so it can never diverge.
            quality_rewarded=_quality_is_rewarded(self.args),
        )
        eval_samples = generate_eval_samples(
            env, eval_key, int(self.args.num_eval_samples),
        )
        self.env = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)

        self.total_v = len(closed_jaxpr.jaxpr.eqns)
        self.num_valid = len(env.valid_vertices)
        self.vertex_valid_static = build_vertex_valid_static(
            env.valid_vertices, self.total_v,
        )
        # EXACT per-vertex micro-action legality (one oracle per rollout env).
        # The tag-bit reconstruction inside heads.py models the vertex's
        # nominal dense Jacobian; the tensor a per-vertex Diag/Compress
        # actually lands on is the per-face CONTRACTION, whose rank / sizes /
        # diagonal pairings differ. Without these masks the policy proposes
        # actions graphax rejects ("TRANSFORM DID NOT FIT"), every one of which
        # sentinels the whole measurement. ALPHAGRAD_LIVE_MASKS=0 disables
        # them (restores the pre-fix behaviour for A/B measurement).
        self._live_masks_enabled = (
            os.environ.get("ALPHAGRAD_LIVE_MASKS", "1") != "0"
        )
        self._mask_jaxpr = closed_jaxpr.jaxpr
        self._mask_consts = list(closed_jaxpr.literals)
        self._mask_args = list(xs)
        self._mask_argnums = tuple(int(a) for a in argnums)
        self._mask_oracles = None
        self._mask_probe_seconds = 0.0
        self.rollout_length = self.num_valid
        # SPMD mesh — replicate agent / opt_state across all visible
        # devices, shard env_states along the env axis. When there's only
        # one device, the sharding annotations are no-ops; the same code
        # path runs on a laptop CPU and a 4-GPU node.
        devices = jax.devices()
        self.num_devices = len(devices)
        self.mesh = Mesh(np.asarray(devices), axis_names=("dev",))
        self.replicated_sharding = NamedSharding(self.mesh, PartitionSpec())
        self.data_sharding = NamedSharding(self.mesh, PartitionSpec("dev"))
        # Update_step batches go through a 2-D layout
        # ``(num_minibatches, mb_size, ...)`` — the data-parallel axis is
        # the mb_size dimension (axis 1) since the minibatch axis is the
        # Python-side loop variable. Mirrors `scan_data_sharding` in
        # mu0_ray_worker. Keeps the per-device working set bounded by
        # mb_size / num_devices instead of by mb_size.
        self.scan_data_sharding = NamedSharding(
            self.mesh, PartitionSpec(None, "dev"),
        )

        # Auto-tune num_envs to a multiple of num_devices so the shard
        # split is even. Mirrors mu0_ray_worker's tuning — we silently
        # round up rather than reject the config so the user can pass
        # round numbers without having to remember the device count.
        requested_envs = int(getattr(self.args, "num_envs", 4))
        if self.num_devices > 1 and requested_envs % self.num_devices != 0:
            self.num_envs = (
                max(self.num_devices, round(requested_envs / self.num_devices))
                * self.num_devices
            )
            print(
                f"[ppo_ray] num_envs tuned: {requested_envs} -> "
                f"{self.num_envs} (multiple of {self.num_devices} devices)"
            )
        else:
            self.num_envs = max(requested_envs, self.num_devices)
        self.minibatches = max(int(getattr(self.args, "minibatches", 1)), 1)
        self.ppo_eps = float(getattr(self.args, "ppo_eps", 0.2))
        self.value_coef = float(getattr(self.args, "value_coef", 0.5))
        # Entropy-coef annealing (Fix 3). The coefficient is decayed
        # linearly from ``entropy_coef`` (init) to ``entropy_coef_final``
        # (floor) over the run's episodes so the policy explores early
        # then COMMITS to the best rule. ``self.entropy_coef`` is the LIVE
        # (annealed) value, recomputed every episode in
        # ``run_rollout_and_train`` and PASSED AS A RUNTIME ARG into the
        # jit'd update step — NOT captured in the loss closure (the old
        # capture-once path is why annealing was dead code and entropy
        # stayed pinned ~1.99).
        self.entropy_coef_init = float(getattr(self.args, "entropy_coef", 0.01))
        self.entropy_coef_final = float(
            getattr(self.args, "entropy_coef_final", 0.001)
        )
        self.entropy_coef = self.entropy_coef_init
        # MONTE-CARLO CREDIT (KEEP AS-IS, bridge-cse). The shipped launcher
        # runs --discount 1.0 --gae-lambda 1.0, i.e. a pure Monte-Carlo return
        # with terminal-only reward (the measured {latency, peak_mem, quality}
        # vector lands on the last step; intermediate steps carry 0). A single
        # episode can be LONG — up to ~|V| x max_substeps steps (|V| vertex
        # eliminations, each an autoregressive micro sub-episode of up to
        # max_substeps=16 typed actions) — so the value head must learn a
        # LONG-HORIZON return: with lambda=discount=1.0 the terminal reward
        # flows back UNDISCOUNTED to every earlier step, and V must predict it
        # from states many steps before the reward is revealed.
        self.gae_lambda = float(getattr(self.args, "gae_lambda", 0.95))
        self.discount = float(getattr(self.args, "discount", 0.99))
        self.reward_weights_np = _build_reward_weights(self.args)

        # --------------------------------------------------------------
        # Multiplicative cosine-gate reward (ALPHAGRAD_REWARD_MODE=mult).
        # --------------------------------------------------------------
        # The default ("additive") path is byte-identical to the legacy
        # behaviour: the scalar reward is sum(reward_vec * weights), with
        # the cost channels stored negated (r = -cost, so cost->0 => r->0
        # = its max) and cosine_sim in [0,1] * lambda_acc. That additive
        # form is reward-hackable: a degenerate order can drive cost->0 and
        # cosine->0 and still beat a faithful order, because the cost term
        # alone reaches its maximum (0) while a faithful order pays cost.
        #
        # "mult" mode replaces the additive scalar with a FIDELITY-GATED
        # CHEAPNESS product computed in ``_apply_mult_reward_gate``:
        #     reward = g(cosine) * max(cheapness, 0)
        # where g(cosine) in [0,1] -> 0 as cosine -> 0, and ``cheapness``
        # is POSITIVE-oriented (larger = cheaper). cosine -> 0 => g -> 0 =>
        # reward -> ~0 regardless of how cheap the order is, so the
        # degenerate order can no longer be the reward-max.
        #
        # Implementation: the gated scalar is written into the cosine_sim
        # channel of the reward buffer and the scalarising weights are
        # collapsed to a one-hot on cosine_sim (weight 1.0). Every
        # downstream consumer that scalarises via ``reward_weights``
        # (per-channel GAE -> dot product, the scalar value-loss
        # ``priority_weights`` dot product, and the milestone best/mean
        # return) then picks out exactly the gated scalar. The RAW
        # per-channel buffer is preserved untouched for per-channel
        # logging (cosine_sim/cost means still report measured values).
        self.reward_mode = str(
            os.environ.get("ALPHAGRAD_REWARD_MODE", "additive")
        ).strip().lower()
        if self.reward_mode not in ("additive", "mult"):
            raise ValueError(
                f"ALPHAGRAD_REWARD_MODE must be 'additive' or 'mult', got "
                f"{self.reward_mode!r}",
            )
        # ADDITIVE COST-SYMLOG (opt-in, default OFF = plain weighted sum).
        # The additive scalar reward is ``sum(reward_vec * weights)`` over the
        # RAW per-channel buffer: cost channels are stored as ``-cost`` in their
        # native units (latency_ns ~1e6, peak_memory ~1e8), so a plain weighted
        # sum is dominated by the largest raw cost and is reward-hackable (a
        # degenerate cost->0/cossim->0 rule scores 0, beating a faithful rule's
        # huge negative cost). ``ALPHAGRAD_ADDITIVE_SYMLOG_COST=1`` symlog's the
        # COST channels (NOT cosine_sim / frob_residual) BEFORE the weighted sum
        # so the scalar becomes
        #     lam_cmp*(-symlog(latency)) + lam_mem*(-symlog(peak_mem))
        #         + lam_acc*cosine_sim
        # i.e. costs land in the ~14-21 symlog band, comparable to an O(10-30)
        # lam_acc*cosine term, so O(1)-O(30) lambdas balance the objective and a
        # large-enough lam_acc makes the cossim term strictly dominate the cost
        # gain (degenerate can never be the argmax). buf_reward_vec_raw stays RAW
        # for per-channel telemetry. No effect in mult mode (the
        # gate already symlog's cost in ``cheapness``).
        self.additive_symlog_cost = (
            os.environ.get("ALPHAGRAD_ADDITIVE_SYMLOG_COST", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        # Gate hyperparameters (only consulted in mult mode).
        #   tau: fidelity floor; cosine <= tau => g = 0.
        #   W:   cheapness offset; cheapness = W - sum_c |w_c|*symlog(cost_c).
        #        Pick W >= typical max weighted symlog(cost) so faithful
        #        orders keep cheapness > 0 (cheaper => larger). Clamped to
        #        >= 0 so an unusually expensive order floors at 0 rather
        #        than flipping sign (which would invert the incentive).
        self.reward_gate_tau = float(
            os.environ.get("ALPHAGRAD_REWARD_GATE_TAU", "0.5")
        )
        self.reward_gate_w = float(
            os.environ.get("ALPHAGRAD_REWARD_GATE_W", "5.0")
        )
        # ------------------------------------------------------------------
        # ANTI-DEGENERACY PENALTY (ALPHAGRAD_ANTI_DEGEN, default ON).
        # ------------------------------------------------------------------
        # The bare mult-gate makes the degenerate basin FLAT-ZERO: a
        # terminal rule with cosine_sim < tau drives g(cos)->0, so
        # gated = g*cheapness -> 0 REGARDLESS of how cheap it is. That
        # stops the policy from *being rewarded* for cossim->0 (good, no
        # hacking) but it does NOT *punish* it — the whole cossim<tau
        # region is a 0-gradient plateau the policy drifts into and can
        # never climb out of (every direction reads 0). Compounded: a
        # skipped (mem-gate) / sentineled (shape-storm) / exceptioned
        # measurement returns ZEROED cost channels which ``cheapness``
        # reads as "free perfect" (cheapness = W - symlog(0) = W), and a
        # cossim=0 zeroed row would also collapse to gated=0 — so the
        # degenerate and the failed-measure cases share the same flat-0
        # plateau and can corrupt best_overall as "free" rules.
        #
        # FIX: any TERMINAL transition that is degenerate (cosine_sim <
        # tau_deg, which also captures ||approx||~=0 since the env emits
        # cosine_sim=0 for a collapsed/failed Jacobian) OR a
        # failed/skipped measure gets a strictly NEGATIVE gated reward
        # -P, making the basin a penalised region with a gradient OUT
        # toward higher cosine. P is set strictly worse than the worst
        # VALID rule (cheapness floors at 0 -> worst valid gated == 0),
        # so -P < 0 <= any valid gated, i.e. degenerate/failed rules are
        # always the argmin and never the argmax / never best_overall.
        self.anti_degen = (
            os.environ.get("ALPHAGRAD_ANTI_DEGEN", "1").strip().lower()
            in ("1", "true", "yes", "on")
        )
        # Degeneracy fidelity floor: terminal cosine_sim below this is
        # treated as degenerate. Default 0.1 (well below the gate tau so
        # genuinely low-but-real fidelity rules still earn the small
        # positive g*cheapness rather than the penalty).
        self.anti_degen_tau = float(
            os.environ.get("ALPHAGRAD_ANTI_DEGEN_TAU", "0.1")
        )
        # Penalty magnitude. Default 2*W so a degenerate rule (gated=-P)
        # is strictly worse than the cheapest possible VALID rule
        # (gated -> g*W <= W). Tunable; clamped to be > 0.
        self.anti_degen_penalty = float(
            os.environ.get(
                "ALPHAGRAD_ANTI_DEGEN_PENALTY",
                str(2.0 * self.reward_gate_w),
            )
        )
        if self.anti_degen_penalty <= 0.0:
            self.anti_degen_penalty = 2.0 * self.reward_gate_w
        # In mult mode the gated scalar lives entirely on the cosine_sim
        # channel; collapse the scalarising weights to a one-hot so the
        # GAE / value / milestone dot products recover it exactly. The
        # gate already folds in the cost lambdas via ``cheapness``, so the
        # raw cost weights would double-count if left in the dot product.
        if self.reward_mode == "mult":
            # Preserve the user's full weight vector for per-channel
            # telemetry (the canonical weights collapse to one-hot below,
            # which would otherwise zero the cost columns in the
            # per-channel logging).
            self._logging_weights_np = self.reward_weights_np.astype(np.float32).copy()
            # Remember the user lambdas (|weight| per active cost channel)
            # BEFORE collapsing — the gate uses them to weight per-channel
            # symlog(cost) in ``cheapness``.
            self._mult_cost_weights_np = np.abs(
                self.reward_weights_np.astype(np.float32)
            )
            self._mult_cost_weights_np[COSINE_SIM_IDX] = 0.0
            self._mult_cost_weights_np[FROB_RESIDUAL_IDX] = 0.0
            collapsed = np.zeros_like(self.reward_weights_np)
            collapsed[COSINE_SIM_IDX] = 1.0
            self.reward_weights_np = collapsed.astype(np.float32)
            print(
                f"[ppo_ray] REWARD_MODE=mult: cosine-gate reward active "
                f"(tau={self.reward_gate_tau}, W={self.reward_gate_w}); "
                f"cost weights {self._mult_cost_weights_np.tolist()} folded "
                f"into cheapness; scalarising weight = one-hot[cosine_sim]."
            )
            if self.anti_degen:
                print(
                    f"[ppo_ray] ANTI_DEGEN active (SHAPED): terminal "
                    f"cosine_sim < {self.anti_degen_tau} -> gated reward ramps "
                    f"linearly from -{self.anti_degen_penalty} (cos=0) up to "
                    f"-{self.anti_degen_penalty * (1.0 - self.anti_degen_tau):.4g} "
                    f"(cos=tau); failed/sentinel measure -> "
                    f"-{self.anti_degen_penalty}. Positive d/dcos gives a "
                    f"gradient OUT of the cossim-0 basin; still strictly worse "
                    f"than the cheapest valid rule (excluded from best_overall)."
                )
        else:
            self._mult_cost_weights_np = np.zeros_like(self.reward_weights_np)
            self._logging_weights_np = self.reward_weights_np.astype(np.float32).copy()

        self.reward_weights = jnp.asarray(
            self.reward_weights_np, dtype=jnp.float32,
        )
        # ADDITIVE_SYMLOG_COST: symlog-compress cost channels in the reward
        # buffer before the scalar sum (see _COST_SYMLOG_MASK_NP). additive
        # mode only — mult mode does its own cost handling. Default off.
        self._additive_symlog_cost = (
            self.reward_mode == "additive"
            and os.environ.get("ALPHAGRAD_ADDITIVE_SYMLOG_COST", "0") == "1"
        )
        # Per-channel INNER lambda: cost channels are transformed as
        # ``symlog(lambda_inner_c * raw)`` (lambda MOVED INSIDE the symlog) so
        # the cost signal is sensitive/rankable instead of over-compressed.
        # See _build_inner_lambda_vec / _INNER_LAMBDA_DEFAULTS. The scalarising
        # OUTER weight (lambda_cmp/lambda_mem, set small ~0.06 in the launcher)
        # then keeps the term a minor nudge:  w_outer * symlog(lam_inner*raw).
        self._inner_lambda_np = _build_inner_lambda_vec()
        self._inner_lambda_j = jnp.asarray(self._inner_lambda_np, dtype=jnp.float32)
        if self._additive_symlog_cost:
            _il = {
                REWARD_NAMES[i]: float(self._inner_lambda_np[i])
                for i in range(NUM_REWARDS)
                if _COST_SYMLOG_MASK_NP[i] and self.reward_weights_np[i] != 0.0
            }
            print(
                f"[ppo_ray] ADDITIVE_SYMLOG_COST: cost term = "
                f"w_outer * symlog(lambda_inner * raw) (lambda INSIDE symlog). "
                f"Rewarded cost channels' lambda_inner = {_il}; outer weights "
                f"(lambda_cmp/lambda_mem) = "
                f"{[float(self.reward_weights_np[i]) for i in range(NUM_REWARDS) if _COST_SYMLOG_MASK_NP[i] and self.reward_weights_np[i] != 0.0]}."
            )

        # Fix 1 (v2 -- REPLACES the failed-row MASK): bounded-NEGATIVE
        # penalty for failed/sentinel transitions in additive mode. The
        # old mask zeroed failed rows out of the loss, which makes an
        # ALL-failed batch a zero-gradient NO-OP -- so once the policy
        # drifts into the all-failing region (COMPRESS/QUANT measure
        # failures) it is STRANDED (return exactly 0, flat). Instead we
        # stamp the failed row's SCALAR reward with a bounded negative
        # value and INCLUDE it in the advantage z-score + loss, so a
        # failed row is a negative advantage the policy is pushed AWAY
        # from (there is always a gradient out). The value must be
        # STRICTLY WORSE than the worst VALID rule (valid rewards run
        # ~[-1.5 (neg-cossim via the 1.5*min(cos,C) guide) .. +0.65]),
        # so failing is always the least-preferred outcome; NOT the
        # -1e10 sentinel (that dominates), NOT 0 (the trap). Default
        # -2.0 (< the worst valid ~-1.5).
        #
        # RETIRED (bridge-cse, USER-DIRECTED): ALPHAGRAD_FAILED_PENALTY is
        # INERT under the shipped neutral-mu sentinel (SENTINEL_K=0 +
        # SENTINEL_NEUTRAL_MU=1). A failed row's per-channel return is
        # overwritten with popart.mu (truly neutral), so ``_dyn_scalar`` is
        # always finite and the ``not _dyn_applied`` guard below skips this
        # penalty stamp entirely — a failed row is neither rewarded nor
        # punished (unknown != bad). The read/var is kept only so the LEGACY
        # non-PopArt / dynamic-sentinel-off path still functions; the launcher
        # no longer exports it.
        self.failed_penalty = float(
            os.environ.get("ALPHAGRAD_FAILED_PENALTY", "-2.0")
        )
        if self.failed_penalty >= 0.0:
            self.failed_penalty = -2.0
        if self.reward_mode == 'additive':
            print(
                f"[ppo_ray] ADDITIVE failed/sentinel penalty = "
                f"{self.failed_penalty} (bounded-negative, INCLUDED in "
                f"advantage+loss -- replaces the failed-row mask; failed "
                f"rows are repulsive, never a zero-gradient no-op)."
            )
        # V2 anti-hack fix 1 (ALL-FAIL BASIN IS ABSORBING): the -2.0
        # penalty above is z-scored together with everything else, so a
        # rollout where EVERY transition failed normalises a CONSTANT
        # reward to advantage ~0 -- the "gradient out" vanishes exactly
        # when it is needed most and the basin absorbs the run (observed
        # pf7ityh6: 4/60 -> 32/60 sentinels, mean_return pinned -2/-16
        # for 80+ eps, zero recovery). Fix: AFTER the scalar-mode
        # z-score, OVERWRITE failed rows' advantage with -|stamp| so a
        # failed action always carries an on-scale (~1 sigma) repulsive
        # gradient, all-fail batches included. 0 disables.
        #
        # RETIRED (bridge-cse, USER-DIRECTED): ALPHAGRAD_FAILED_ADV_STAMP is
        # INERT under the shipped stack. It only ever fired in (a) the
        # non-PopArt scalar z-score path, and (b) the PopArt scale/clip block
        # (constant -1.0) — BOTH now guarded off: (a) is skipped by the
        # neutral-mu ``not _dyn_applied`` guard, and (b) lives inside the
        # ALPHAGRAD_POPART_PURE_ADV legacy branch which the default (=1) does
        # not take. Under neutral-mu a failed row is already A==0 (truly
        # neutral). Kept as a var for the legacy revert path; not exported by
        # the launcher.
        self.failed_adv_stamp = float(
            os.environ.get("ALPHAGRAD_FAILED_ADV_STAMP", "1.0")
        )
        # V2 anti-hack fix 2 (COST-CHANNEL LEVERAGE): lambda_inner was
        # calibrated so TYPICAL costs land at symlog ~0.7 (term ~0.06*0.7
        # *2ch ~ 0.09, the designed ~8x-below-quality nudge). But the
        # untrained micro policy's graphs start ~20x typical (latency
        # symlog ~3.1), handing the cost term ~0.23 of scalar leverage --
        # cost-cutting via COMPRESS/QUANT spam then rivals/beats the
        # quality term and the policy hacks cost while bkstep decays
        # (pf7ityh6 ep0->36: mean_return 0.068->0.19 with bkstep
        # 0.22->0.16). Fix: clip the symlog'd cost channels to +/-CAP so
        # beyond ~2x typical the cost term SATURATES (zero gradient for
        # making the graph cheaper by making it worse) and quality keeps
        # the designed dominance at every operating point. 0 disables.
        self._cost_symlog_cap = float(
            os.environ.get("ALPHAGRAD_COST_SYMLOG_CAP", "0.0")
        )
        if self.reward_mode == "additive" and (
            self.failed_adv_stamp > 0 or self._cost_symlog_cap > 0
        ):
            print(
                f"[ppo_ray] V2 anti-hack: failed_adv_stamp="
                f"{self.failed_adv_stamp} (post-z-score advantage overwrite "
                f"on failed rows), cost_symlog_cap={self._cost_symlog_cap} "
                f"(symlog'd cost channels clipped to +/-cap)."
            )

        # Advantage-normalisation strategy. ``scalar`` is the only mode
        # this trainer supports — PopArt (below, unconditional) carries the
        # per-channel normalisation and rejects ``gdpo``, which would
        # double-normalise with its own per-minibatch z-score.
        self.advantage_norm = str(getattr(self.args, "advantage_norm", "scalar"))
        if self.advantage_norm not in ("gdpo", "scalar"):
            raise ValueError(
                f"--advantage-norm must be 'gdpo' or 'scalar', got "
                f"{self.advantage_norm!r}",
            )
        # Channel masks used by ``gdpo_normalise_advantages``. The
        # channel mask is 1 only for reward channels with non-zero
        # weight (user opted in via --rewards / --lambda-*). The sparse
        # mask flags channels (cosine_sim, frob_residual) whose value
        # is only meaningful at the terminal elimination step — z-score
        # uses rescale-without-recentre for them so the all-zero
        # intermediate steps don't artificially boost the std.
        from alphagrad.approx.common.reward_scaling import (
            SPARSE_TERMINAL_MASK_NP as _SPARSE_TERMINAL_MASK_NP,
        )
        self._channel_mask_j = jnp.asarray(
            (self.reward_weights_np != 0.0).astype(np.float32)
        )
        self._sparse_mask_j = jnp.asarray(
            _SPARSE_TERMINAL_MASK_NP.astype(np.float32)
        )
        # The no-symlog GAE variant — the only one used now that the
        # critic learns PopArt-normalised values. Built up-front so the
        # rollout loop is free of import / construction cost per call.
        from alphagrad.approx.common.gae import (
            make_get_advantages as _make_get_advantages,
        )
        self._gae_no_symlog = _make_get_advantages(use_symlog=False)

        # PopArt value-target normalisation (van Hasselt 2016,
        # multi-channel as in IMPALA). ALWAYS ON — it REPLACES the former
        # rollout-wide advantage z-score, the suspected collapse driver:
        # batch z-scoring AMPLIFIES noise as returns homogenise at
        # convergence (divide by a shrinking batch std), washes an
        # all-fail batch to ~0, and can sign-flip good samples around the
        # batch mean. The critic learns per-channel NORMALISED values
        # v_hat = (v - mu_k)/sigma_k against quasi-static EMA stats of the
        # GAE returns (beta ~1e-2/update, sigma FLOORED so it can never
        # amplify); the value head's final linear layer is rescaled
        # output-preservingly on every stats step (the "Art"); advantages
        # are formed per-channel in normalised space
        # (A_k/sigma_k == (G_k - mu_k)/sigma_k - v_hat_k) and collapsed
        # via reward_weights — O(1) WITHOUT batch coupling.
        if self.advantage_norm != "scalar":
            raise ValueError(
                "PopArt value normalisation (always on) requires "
                "--advantage-norm scalar (the gdpo path carries its own "
                "per-minibatch normalisation).",
            )
        # The critic predicts PopArt-NORMALISED values, so the legacy
        # symlog encoding inside GAE never applies; raw values are
        # recovered affinely (v = sigma*v_hat + mu) before the GAE deltas.
        from alphagrad.approx.common.popart import PopArtStats
        # Fix 4: per-channel sigma floor. The global default 0.1
        # over-amplified the near-homogeneous bkstep channel (idx 9,
        # returns ~0.64 everywhere at convergence -> var->0 -> floored
        # sigma -> A/sigma inflated). Floor bkstep(9) + cosine(6)
        # higher (ALPHAGRAD_POPART_SIGMA_MIN_QUALITY, default 0.2);
        # everything else keeps the base floor
        # (ALPHAGRAD_POPART_SIGMA_MIN, default 0.1). Both env-tunable
        # so the change is fully revertible (set the quality floor
        # equal to the base to restore the old scalar behaviour).
        _sig_min_base = float(
            os.environ.get("ALPHAGRAD_POPART_SIGMA_MIN", "0.1")
        )
        _sig_min_qual = float(
            os.environ.get("ALPHAGRAD_POPART_SIGMA_MIN_QUALITY", "0.2")
        )
        _sig_min_vec = np.full(NUM_REWARDS, _sig_min_base, dtype=np.float64)
        for _qi in (6, 9):  # cosine_sim, bkstep_acc
            _sig_min_vec[_qi] = max(_sig_min_base, _sig_min_qual)
        # sigma_max must admit the RAW cost scale (bridge-cse): now that
        # cost channels enter PopArt raw (no symlog), peak_memory returns
        # span ~1e9 with a true std >> the legacy 1e6 cap. A too-small
        # sigma_max CLAMPS sigma below the true scale and UNDER-normalises
        # the cost channel -> O(100) advantage (observed ep1-4). Raise the
        # cap to 1e12 so raw-cost sigma can reach its true magnitude.
        _sig_max = float(os.environ.get("ALPHAGRAD_POPART_SIGMA_MAX", "1e12"))
        # ROBUST STD (bridge-cse, USER-DIRECTED). Raw cost -> a single 100x
        # outlier would spike the per-channel sigma EMA and crush the other
        # channels' relative signal. WINSORIZE each channel's value targets
        # to mu +/- k*sigma (current running stats) BEFORE the EMA update so
        # an outlier cannot inflate sigma; output-preserving (only the
        # stats-tracking sees the winsorized batch, the returned old/new
        # mu,sigma still drive the head rescale). Flag ALPHAGRAD_POPART_
        # ROBUST_STD (default 1); k via ALPHAGRAD_POPART_WINSOR_K (default
        # 5.0). See PopArtStats.update.
        _robust = os.environ.get("ALPHAGRAD_POPART_ROBUST_STD", "1") == "1"
        _winsor_k = float(
            os.environ.get("ALPHAGRAD_POPART_WINSOR_K", "5.0")
        )
        self.popart = PopArtStats(
            NUM_REWARDS,
            beta=float(os.environ.get("ALPHAGRAD_POPART_BETA", "0.01")),
            sigma_min=_sig_min_vec,
            sigma_max=_sig_max,
            robust_std=_robust,
            winsor_k=_winsor_k,
        )
        print(
            f"[ppo_ray] PopArt value norm ON: beta={self.popart.beta} "
            f"sigma_min(base={_sig_min_base}, quality[6,9]="
            f"{_sig_min_qual}) sigma_max={_sig_max:g} "
            f"robust_std={_robust} (winsor k={_winsor_k}) — replaces the "
            f"rollout advantage z-score; under pure-PopArt advantage a "
            f"failed row is truly-neutral (A==0), no constant stamp.",
            flush=True,
        )

        # ------------------------------------------------------------------
        # DYNAMIC SENTINEL (bridge-cse). Replace the catastrophic -1e10
        # sentinel reward with a bounded, distribution-aware substitute:
        # per-channel ``mu_c - K*sigma_c`` over the running distribution of
        # REAL (non-failed) terminal reward vectors. Keeps failed rows a
        # *repulsive* (mildly-negative) signal without the -1e10 blow-up that
        # distorts every downstream mean / z-score / log. The boolean
        # ``buf_failed`` mask (value-independent) still drives every
        # failed_transitions count and PopArt exclusion — we change the
        # REWARD, never the event bookkeeping.
        #
        # Stats source: a DEDICATED per-channel EMA (mu, sigma) of the RAW
        # terminal reward vectors — NOT PopArt (which normalises RETURNS, not
        # per-step rewards, and whose mu/sigma live in a different, discounted
        # space). EMA of x and x^2 -> mu = E[x], sigma = sqrt(max(E[x^2]-mu^2,
        # eps)). beta is the per-rollout mixing rate.
        self.dynamic_sentinel = (
            os.environ.get("ALPHAGRAD_DYNAMIC_SENTINEL", "1") == "1"
        )
        self.sentinel_k = float(
            os.environ.get("ALPHAGRAD_SENTINEL_K", "2.0")
        )
        self._sent_ema_beta = float(
            os.environ.get("ALPHAGRAD_SENTINEL_EMA_BETA", "0.01")
        )
        # EMA state (per-channel). ``_sent_ema_n`` counts how many rollouts
        # have fed the EMA — until the first real update we have no
        # distribution, so the sentinel value falls back to the legacy
        # bounded penalty (failed_penalty) rather than a meaningless 0-K*0=0.
        self._sent_ema_mean = np.zeros(NUM_REWARDS, dtype=np.float64)
        self._sent_ema_sq = np.zeros(NUM_REWARDS, dtype=np.float64)
        self._sent_ema_n = 0
        # ------------------------------------------------------------------
        # SENTINEL DESIGN — CURRENT CHOICE + TWO FUTURE ALTERNATIVES
        # ------------------------------------------------------------------
        # A "failed/sentinel" transition is one whose measurement never
        # produced a usable value (CPU-pool timeout, mem-gate skip, shape
        # storm, exec exception). We must substitute SOMETHING for its
        # per-channel return before GAE/PopArt see it. THE CURRENT CHOICE is
        # NEUTRAL-MU (below): overwrite the failed row's per-channel return
        # with ``popart.mu`` so its PopArt advantage (G_k - mu_k)/sigma_k == 0
        # on every channel and its value target == 0 — the row is TRULY
        # NEUTRAL (unknown != bad; it neither rewards nor punishes, and
        # mean_return is unmoved). K=0 (no mu - K*sigma penalty).
        #
        # TWO ALTERNATIVES worth trying if neutral-mu proves insufficient
        # (documented here for future reference — NOT implemented):
        #   (a) MASKING — drop failed transitions ENTIRELY from the update
        #       (exclude them from the policy loss / value loss / advantage
        #       batch; NO substitute value at all), so a failed row contributes
        #       zero gradient rather than a neutral one. Risk: an ALL-failed
        #       rollout becomes a no-op (no gradient to escape the failing
        #       region) — the historical trap this neutral-mu path replaced.
        #   (b) SURROGATE MEASUREMENT HEAD + EMA — train a cheap regression
        #       head to PREDICT the expensive channels {latency_ns, peak_memory}
        #       from the cheap-to-compute {flops, bytes, xla_peak_memory}
        #       features, and use an EMA of {cosine_sim / bkstep_acc} for the
        #       quality channels on a sentinel. A failed row then gets a
        #       *predicted* (not neutral, not measured) return so the policy
        #       still receives a plausible learning signal without paying the
        #       measurement cost.
        # ------------------------------------------------------------------
        # TRULY-NEUTRAL SENTINEL (bridge-cse). EMA of the REAL per-env
        # post-transform WEIGHTED scalar return (the exact quantity GAE
        # scalarises + mean_return averages). With SENTINEL_K=0 the raw
        # per-channel substitute ``_dyn_vec=mu`` reads as average-quality
        # (quality channels are non-symlog, keeping mu~0.65-0.7) + near-free
        # cost (cost channels symlog-compressed to ~0) => a net-POSITIVE
        # ~+0.71 scalar that SPIKES mean_return and injects a spurious
        # POSITIVE GAE advantage on failed rows. When enabled, a failed
        # row's scalarised return is forced to THIS running mean so the row
        # is truly neutral: it neither rewards nor punishes (mean_return is
        # unmoved; GAE advantage ~= return - value ~= 0 since the critic
        # tracks the same mean). Flag-guarded (default on), revertible via
        # ALPHAGRAD_SENTINEL_NEUTRAL_MU=0 (falls back to raw mu-K*sigma).
        self._sentinel_neutral_mu = (
            os.environ.get("ALPHAGRAD_SENTINEL_NEUTRAL_MU", "1") == "1"
        )
        # KILL THE -2 FLOOR (bridge-cse). Extend the truly-neutral-mu handling
        # to the FIRST / un-warmed rollout (``_sent_ema_n == 0``): use
        # ``popart.mu`` (neutral) instead of falling through to the additive
        # -2.0 ``failed_penalty`` stamp that floored mean_return/best_return at
        # -2 before the EMA warmed. PopArt-only (mu is the neutral frame). Set 0
        # to revert to the legacy -2-on-un-warmed behaviour.
        self._neutral_unwarmed = (
            os.environ.get("ALPHAGRAD_NEUTRAL_UNWARMED", "1") == "1"
        )
        self._sent_ema_scalar = 0.0
        self._sent_ema_scalar_n = 0
        if self.dynamic_sentinel:
            print(
                f"[ppo_ray] DYNAMIC SENTINEL ON: failed/sentinel terminals "
                f"get per-channel mu_c - {self.sentinel_k}*sigma_c "
                f"(EMA beta={self._sent_ema_beta}) instead of -1e10. "
                f"The additive failed_penalty stamp AND the post-z-score "
                f"failed_adv_stamp are DISABLED (dynamic value supersedes "
                f"them); failed_transitions logging + PopArt exclusion "
                f"unchanged.",
                flush=True,
            )

        # RAW full-range cosine_sim threading (bridge-cse). The env used to
        # apply the ALPHAGRAD_COSSIM_GUIDE_CAP ``min(cossim, C)`` guide cap in
        # env._callback, so idx6 arrived already clamped to [.., C] and the
        # full-range cossim never reached the worker for honest logging /
        # EMA. When ON, the env emits RAW cossim in idx6 and the guide cap is
        # applied HERE — only into the reward/GAE buffer, leaving
        # buf_reward_vec_raw carrying the full-range value for
        # ``reward/cosine_sim_raw`` + the dynamic-sentinel EMA.
        self.raw_cossim_thread = (
            os.environ.get("ALPHAGRAD_RAW_COSSIM_THREAD", "1") == "1"
        )
        # UN-SCALE COSSIM (bridge-cse). ``ALPHAGRAD_COSSIM_GUIDE_CAP=0`` (or any
        # value <= 0, or empty/unset) DISABLES the ``min(cossim, C)`` guide cap
        # entirely so the cosine_sim channel enters the reward at its FULL RAW
        # range (PopArt then normalises it) instead of being clamped into
        # [.., 0.1]. Only a strictly-positive C installs a cap. Previously "0"
        # parsed to a literal ``min(cossim, 0.0)`` clamp — the un-scale is a
        # <=0 sentinel for "no cap".
        _gc = os.environ.get("ALPHAGRAD_COSSIM_GUIDE_CAP", "").strip()
        self._cossim_guide_cap = None
        if _gc:
            try:
                _gc_val = float(_gc)
                self._cossim_guide_cap = _gc_val if _gc_val > 0.0 else None
            except ValueError:
                self._cossim_guide_cap = None
        if self.raw_cossim_thread:
            print(
                f"[ppo_ray] RAW COSSIM THREAD ON: env emits full-range "
                f"cosine_sim in idx6; worker applies guide cap="
                f"{self._cossim_guide_cap} into the GAE buffer only, logs "
                f"reward/cosine_sim_raw from the uncapped value.",
                flush=True,
            )

        # Phase D — conditioned-reward gates. The user-supplied
        # ``--reward-condition`` specs are parsed once at init; the
        # rollout loop walks the list and zeroes the easier reward at
        # any transition where the gate fails. Sparse-terminal harder
        # channels (cosine_sim, frob_residual) only carry a signal at
        # the terminal step, so the gate is forced True on
        # intermediate steps to avoid zeroing the easier channel
        # everywhere.
        from alphagrad.approx.common.reward_scaling import (
            parse_reward_conditions as _parse_reward_conditions,
            SPARSE_TERMINAL_INDICES as _SPARSE_TI,
        )
        self._reward_conditions: list[tuple[int, int, str, float]] = (
            _parse_reward_conditions(
                getattr(self.args, "reward_condition", []) or []
            )
        )
        self._sparse_terminal_idx_set = set(_SPARSE_TI)


        # Agent + optimizer.
        policy_dims = self._parse_int_list(self.args.policy_dims)
        value_dims = self._parse_int_list(self.args.value_dims)
        self.dynamic_substeps = bool(
            getattr(self.args, "dynamic_substeps", False)
        )
        # MicroPPOAgent (pointer vertex head + autoregressive
        # MicroActionPolicy sub-episodes + separate value pool) is the ONLY
        # policy path; it always implies --dynamic-substeps.
        if not self.dynamic_substeps:
            print(
                "[ppo_ray_worker] MicroPPOAgent implies --dynamic-substeps; "
                "forcing it on."
            )
            self.dynamic_substeps = True
        # Variant support. When ``--variant`` is anything other than
        # ``custom``, we build the agent with the UNION factor table
        # and restrict the action space via the mask flags below.
        # ``custom`` falls back to the legacy ``--factors`` value.
        from alphagrad.approx.variants import (
            compute_ppo_variant_masks,
            ppo_full_factor_table,
        )
        self.current_variant = str(getattr(self.args, "variant", "custom"))
        if self.current_variant == "custom":
            self.factor_table = self._parse_int_list(
                getattr(self.args, "factors", "-1,2,3,4")
            )
            if not self.factor_table:
                self.factor_table = (-1, 2, 3, 4)
        else:
            self.factor_table = ppo_full_factor_table()
        self.factor_table_j = jnp.asarray(self.factor_table, dtype=jnp.int32)
        # Initial action masks from the ``--variant`` setting. For custom
        # there's no restriction (all-True); other variants restrict to
        # the variant's allowed set. ``compute_union_variant_masks``
        # accepts both single variants and compound "A+B" / "all_simple"
        # strings.
        from alphagrad.approx.variants import (
            compute_union_variant_masks as _compute_union_variant_masks,
        )
        from graphax.sparse.micro_actions import (
            NUM_QUANT_DTYPES as _NUM_QUANT_DTYPES_VAR,
        )
        self._num_quant_dtypes_for_mask = int(_NUM_QUANT_DTYPES_VAR)
        masks = _compute_union_variant_masks(
            self.current_variant,
            tuple(self.factor_table),
            self._num_quant_dtypes_for_mask,
        )
        self._current_op_mask_j = jnp.asarray(
            masks["op_type_mask"].astype(np.float32),
        )
        self._current_factor_mask_j = jnp.asarray(
            masks["factor_mask"].astype(np.float32),
        )
        self._current_quant_mask_j = jnp.asarray(
            masks["quant_dtype_mask"].astype(np.float32),
        )
        self.num_factors = len(self.factor_table)
        # Resolve the token-mixer policy exactly as ppo.py::make_agent does
        # (ppo.py:2984): env override first, then ALPHAGRAD_POLICY, default
        # "transformer". Prior to this the worker built a plain transformer
        # regardless of ALPHAGRAD_POLICY — palimpsa_bi was inert on the
        # ACTUAL Ray rollout path (only ppo.py::Agent honoured it).
        # Palimpsa, not the quadratic transformer. policy.py has documented
        # palimpsa as the default all along, but this path defaulted to
        # "transformer" -- and that is not a small difference at these sizes:
        # full attention materialises an (heads, T, T) score matrix, which for
        # the tokenized jaxpr came out as f32[16, 16384, 16384] = 16 GiB and
        # OOMd the GPU during autotuning. Palimpsas gated linear attention is
        # O(T) and never forms that matrix.
        _policy = getattr(self.args, "policy", None) or os.environ.get(
            "ALPHAGRAD_POLICY", "palimpsa"
        )
        _policy = str(_policy).strip().lower()
        if _policy not in ("transformer", "palimpsa", "palimpsa_bi"):
            raise ValueError(
                "ALPHAGRAD_POLICY must be 'transformer', 'palimpsa' or "
                f"'palimpsa_bi', got {_policy!r}"
            )
        self._policy = _policy
        # Policy V2 static inputs: substep budget + prime/gcd factor
        # tables sized from the env's static axis structure.
        self.max_substeps = int(getattr(self.args, "max_substeps", 16))
        # Per-vertex RUNTIME cap on micro-action slots. Traced jnp scalar
        # so it never recompiles act_step. Fixed at MAX_RULES_PER_VERTEX
        # (no cap).
        self._substep_budget_j = jnp.int32(MAX_RULES_PER_VERTEX)
        if self.max_substeps > MAX_RULES_PER_VERTEX:
            print(
                f"[ppo_ray_worker] --max-substeps {self.max_substeps} > "
                f"MAX_RULES_PER_VERTEX {MAX_RULES_PER_VERTEX}; clamping "
                f"(the env rule_specs row capacity would silently drop "
                f"the excess)."
            )
            self.max_substeps = int(MAX_RULES_PER_VERTEX)
        _axis_sizes_np = np.asarray(self.env.axis_state_static)[
            ..., _AXIS_FEAT_SIZE
        ]
        _observed_max_axis = int(_axis_sizes_np.max()) if _axis_sizes_np.size else 1
        _table_size = max(
            64,
            int(getattr(self.args, "max_axis_size", 0) or 0),
            _observed_max_axis,
        )
        if _table_size > 4096:
            print(
                f"[ppo_ray_worker] max axis size {_table_size} exceeds the "
                f"factor-table cap 4096; clamping (gcd gathers clamp to "
                f"the table edge for larger axes)."
            )
            _table_size = 4096
        self.factor_tables = precompute_factor_tables(_table_size)
        print(
            f"[ppo_ray_worker] POLICY V2 ACTIVE: "
            f"MicroPPOAgent = {_policy.upper()} backbone + PointerVertexPolicy "
            f"+ MicroActionPolicy(max_substeps={self.max_substeps}) + separate "
            f"value pool; factor tables up to axis size {_table_size} "
            f"(observed max {_observed_max_axis}).",
            flush=True,
        )
        self.agent = MicroPPOAgent(
            vocab_size=int(self.args.vocab_size),
            embd_dim=int(self.args.embd_dim),
            num_layers=int(self.args.num_layers),
            num_heads=int(self.args.num_heads),
            hidden_dim=int(self.args.hidden_dim),
            num_vertices=self.total_v,
            value_dims=value_dims,
            key=agent_key,
            max_substeps=self.max_substeps,
            policy=_policy,
        )
        # One-time assertion that the palimpsa_bi backbone was actually
        # instantiated on the worker path (guards against silent
        # transformer fallback regressions). Cheap: inspects a static field.
        if _policy == "palimpsa_bi":
            from alphagrad.transformer.encoder import BiPalimpsaMixer
            _layer0 = self.agent.encoder.layers[0]
            _mixer = getattr(_layer0, "attn_layer", None)
            assert isinstance(_mixer, BiPalimpsaMixer), (
                "ALPHAGRAD_POLICY=palimpsa_bi but MicroPPOAgent encoder "
                f"layer 0 mixer is {type(_mixer).__name__}, not "
                "BiPalimpsaMixer — policy did not reach the Ray rollout path."
            )
            print(
                "[ppo_ray_worker] VERIFIED BiPalimpsaMixer active on "
                f"encoder ({self.agent.encoder.num_layers} layers)",
                flush=True,
            )
        self.agent = init_linear_weights(self.agent, init_key)
        # init_linear_weights orthogonally re-inits every Linear,
        # including the palimpsa mixers' zero-init relational gate —
        # restore the zeros so the eqn_ids gate starts as a no-op.
        self.agent = _rezero_encoder_rel_gates(self.agent)
        # Near-uniform initial policy over all V2 heads (ppo.py's
        # head-init-scale pattern; keeps END/DIAG/COMPRESS/QUANT and the
        # vertex pointer explorable instead of softmax-saturated).
        _his = float(getattr(self.args, "head_init_scale", 0.1))
        self.agent = _scale_micro_policy_heads(self.agent, _his)
        print(f"[ppo_ray_worker] V2 policy heads scaled by {_his} "
              f"(near-uniform initial policy).")
        # Replicate the agent across all devices — under SPMD this is
        # cheap (params are < 100 MB) and lets `act_step` / loss path
        # run sharded without the trainer having to think about it.
        self.agent = self._replicate(self.agent)

        _lr = float(self.args.lr)
        _decay = int(self.args.episodes) * self.minibatches
        _min_mult = float(getattr(self.args, "lr_decay_min_mult", 0.1))
        _warmup = int(os.environ.get("ALPHAGRAD_LR_WARMUP_STEPS", "0") or "0")
        if _warmup > 0:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=_lr, warmup_steps=_warmup,
                decay_steps=max(_decay, _warmup + 1), end_value=_lr * _min_mult,
            )
        else:
            schedule = optax.cosine_decay_schedule(_lr, _decay, _min_mult)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(float(self.args.max_grad_norm)),
            optax.adamw(schedule, eps=float(self.args.adam_eps)),
        )
        self.opt_state = self.optimizer.init(
            eqx.filter(self.agent, eqx.is_inexact_array),
        )
        self.opt_state = self._replicate(self.opt_state)

        # Pre-reset batched env state. The env axis (axis 0) is the
        # data-parallel dimension; shard it across devices so each
        # GPU owns ``num_envs // num_devices`` envs.
        env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
        )
        self.env_states = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding)
            if eqx.is_array(x) else x,
            env_states,
        )

        # Cache the per-env vertex-valid mask. It's static (depends only on
        # the env's `valid_vertices`), so we vectorise once.
        self._vertex_valid_static_j = jnp.asarray(
            self.vertex_valid_static, dtype=jnp.float32,
        )

        self._episode_counter = 0

        # Trainer-side leak profile (mirror of the per-actor profiler in
        # cpu_approx_worker). Lets the user diff trainer-process RSS
        # growth against the worker-process growth in
        # ``slurm/logs/${JOB_ID}/leak-trainer-${PID}.log`` vs.
        # ``leak-actor-${PID}.log`` — confirms (or refutes) that the
        # JIT compile leak lives in the worker, not the trainer.
        from alphagrad.approx.common.leak_profile import maybe_install
        self._leak_profile = maybe_install("trainer")

        # Optional resume from a previously-written checkpoint. Must
        # happen AFTER the agent + opt_state are constructed (we need
        # template trees for eqx.tree_deserialise_leaves).
        if self._checkpoint_path:
            try:
                from alphagrad.approx.common.checkpoint import (
                    install_sigterm_handler, load_state,
                )
                restored = load_state(
                    self._checkpoint_path,
                    template_agent=self.agent,
                    template_opt_state=self.opt_state,
                )
                if restored is not None:
                    if restored["agent"] is not None:
                        self.agent = self._replicate(restored["agent"])
                    if restored["opt_state"] is not None:
                        self.opt_state = self._replicate(restored["opt_state"])
                    self._episode_counter = int(restored["episode_counter"])
                    if restored["reward_weights"] is not None:
                        self.reward_weights_np = restored["reward_weights"].astype(np.float32)
                        self.reward_weights = jnp.asarray(self.reward_weights_np, dtype=jnp.float32)
                    print(
                        f"[ppo_ray_worker] resumed from {self._checkpoint_path} "
                        f"at episode {self._episode_counter}"
                    )
                # Install the SIGTERM handler regardless of whether a
                # checkpoint already existed — we want the next SLURM
                # timeout to drop a checkpoint even if we started cold.
                install_sigterm_handler(self._save_checkpoint_safe)
            except Exception as exc:
                print(f"[ppo_ray_worker] checkpoint resume failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_int_list(raw: str) -> tuple[int, ...]:
        if not raw:
            return ()
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())

    def _replicate(self, tree):
        """Place every array leaf onto every device (replicated sharding).

        Static (non-array) leaves pass through unchanged. Matches the
        `shard_leaf` pattern in `mu0_ray_worker._build_actor_state`.
        """
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.replicated_sharding)
            if eqx.is_array(x) else x,
            tree,
        )

    def _vertex_avail(self, state):
        """`(num_envs, num_vertices)` 0/1 mask of vertices still in the
        active suffix of `state.order`. Matches the action-mask convention
        the policy uses to invalidate finished vertices."""
        return jax.vmap(
            lambda s: vertex_avail_at_step(
                s,
                self._vertex_valid_static_j,
                self.total_v,
                self.num_valid,
            ),
        )(state)

    def _reset_mask_oracles(self):
        """One :class:`LiveVertexMaskOracle` per env, rewound to step 0.

        Each env walks its own elimination order, so each needs its own live
        graph. Built lazily (and rebuilt every episode) because the oracle
        carries a jax trace whose equation list would otherwise grow across
        episodes.
        """
        if not self._live_masks_enabled:
            self._mask_oracles = None
            return
        from alphagrad.approx.common.masks import LiveVertexMaskOracle

        self._mask_oracles = [
            LiveVertexMaskOracle(
                self._mask_jaxpr, self._mask_consts, self._mask_args,
                self._mask_argnums, max_axes=MAX_AXES_PER_VERTEX,
            )
            for _ in range(self.num_envs)
        ]

    def _live_masks(self, avail_np):
        """``(pair_valid, compress_valid)`` for every env x vertex, this step.

        Shapes ``(num_envs, total_v + 1, N, N)`` / ``(num_envs, total_v + 1,
        N)``; row ``v`` is the mask for the 1-indexed vertex id, so the policy
        can gather with ``vertex_action + 1`` inside the jit. All-ones when the
        oracle is disabled, which reproduces the unmasked behaviour exactly.
        """
        N = MAX_AXES_PER_VERTEX
        shape_p = (self.num_envs, self.total_v + 1, N, N)
        shape_c = (self.num_envs, self.total_v + 1, N)
        if not self._live_masks_enabled:
            return np.ones(shape_p, np.float32), np.ones(shape_c, np.float32)
        if self._mask_oracles is None:
            self._reset_mask_oracles()
        t0 = _time.time()
        pair = np.zeros(shape_p, np.float32)
        comp = np.zeros(shape_c, np.float32)
        for e, oracle in enumerate(self._mask_oracles):
            cands = [v for v in range(1, self.total_v + 1)
                     if avail_np[e, v - 1] > 0.5]
            pair[e], comp[e] = oracle.masks(cands)
        self._mask_probe_seconds += _time.time() - t0
        return pair, comp

    def _advance_mask_oracles(self, order_np, step_np, specs_np):
        """Commit the vertex each env just eliminated, with ITS rules.

        The transforms applied at a vertex change the structure of every
        downstream edge, so the oracle has to replay the same approximation the
        measurement will, not a plain exact elimination — hence the shared
        ``rule_specs_to_transforms`` decode.
        """
        if not self._live_masks_enabled or self._mask_oracles is None:
            return
        from alphagrad.approx.env import rule_specs_to_transforms

        t0 = _time.time()
        for e, oracle in enumerate(self._mask_oracles):
            stop = int(step_np[e])
            if stop <= 0:
                continue
            v = int(order_np[e][stop - 1])
            o_list = [int(x) for x in order_np[e][:stop]]
            specs_list = np.asarray(specs_np[e][:stop]).tolist()
            tmap = dict(
                rule_specs_to_transforms(
                    self._mask_jaxpr, o_list, specs_list,
                    quant_once=bool(getattr(self.args, "quant_once", False)),
                )
            )
            oracle.advance(v, tmap.get(v, ()))
        self._mask_probe_seconds += _time.time() - t0

    def _scalar_reward(self, reward_vec):
        """Reduce the env's 8-component reward vector to a scalar via
        the per-component weight vector. Symlog is applied downstream
        by `get_advantages` (per the gae.py contract)."""
        return jnp.sum(reward_vec * self.reward_weights, axis=-1)

    def _apply_mult_reward_gate(
        self,
        buf_reward_vec: np.ndarray,
        terminal_mask: np.ndarray | None = None,
        failed_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transform the raw per-channel reward buffer into the gated
        representation used by the multiplicative cosine-gate reward.

        ``buf_reward_vec`` is ``(T, N, NUM_REWARDS)`` of RAW per-channel
        rewards: cost channels stored NEGATED (r = -cost, more negative =
        worse), cosine_sim in [0,1]. Returns a NEW buffer (the caller
        keeps the original for per-channel logging).

        Gate::

            g(cos)    = clip((cos - tau) / (1 - tau), 0, 1)      # fidelity
            cheapness = max(0, W - sum_c |w_c| * symlog(cost_c))  # >0 = cheap
            reward    = g(cos) * cheapness

        where ``cost_c = -r_c`` (cost is the negated stored reward) and
        symlog(cost_c) = symlog(-r_c) grows with cost, so a CHEAP order
        (small cost) yields a LARGE positive cheapness. The gated scalar
        is written into the cosine_sim channel and every other channel is
        zeroed; the scalarising weights (one-hot[cosine_sim], set in
        ``__init__``) then recover it exactly downstream.

        SIGN: ``cheapness`` is built from the POSITIVE cost magnitude
        (``-r_c``), NOT the negative stored reward, so it is never
        multiplied by g while carrying a negative sign — a low-cosine
        order (g -> 0) yields reward -> 0, never a spuriously-large
        less-negative value. cosine -> 0 => reward -> ~0 regardless of
        cheapness, which kills the cost->0/cosine->0 hack.

        ANTI-DEGENERACY (``self.anti_degen``): the bare gate above makes
        the cossim<tau region a flat-0 plateau (no *punishment* for
        cossim->0, just no reward) — the policy can drift in and never
        climb out. When enabled, any TERMINAL transition that is
        degenerate (cosine_sim < ``anti_degen_tau``, capturing
        ||approx||~=0 since the env emits cosine_sim=0 for a
        collapsed/failed Jacobian) OR a failed/skipped/sentineled
        measurement (``failed_mask``) gets a strictly NEGATIVE gated
        reward ``-anti_degen_penalty``. With cheapness floored at 0 the
        worst VALID gated reward is 0, so the penalty makes degenerate /
        failed rules strictly the argmin (never best_overall) and turns
        the flat plateau into a penalised slope with a gradient toward
        higher cosine. ``terminal_mask`` / ``failed_mask`` are ``(T, N)``
        bool; when both are None the penalty is skipped (back-compat).
        """
        out = np.array(buf_reward_vec, dtype=np.float32, copy=True)
        cos = out[..., COSINE_SIM_IDX]                       # (T, N), in [0,1]
        tau = np.float32(self.reward_gate_tau)
        denom = np.maximum(np.float32(1.0) - tau, np.float32(1e-6))
        g = np.clip((cos - tau) / denom, 0.0, 1.0)           # (T, N)

        # Per-channel positive cost magnitude = -stored_reward; symlog and
        # weight by the user's |lambda| for each active cost channel. Only
        # cost channels carry nonzero weight here (cosine/frob zeroed in
        # __init__), so this never picks up the quality channels.
        cost_mag = -out                                      # (T, N, R)
        cost_sl = np.sign(cost_mag) * np.log1p(np.abs(cost_mag))
        w = self._mult_cost_weights_np.astype(np.float32)    # (R,)
        weighted_cost = (cost_sl * w[None, None, :]).sum(axis=-1)  # (T, N)
        cheapness = np.maximum(
            0.0, np.float32(self.reward_gate_w) - weighted_cost,
        )                                                    # (T, N)

        gated = (g * cheapness).astype(np.float32)           # (T, N)

        # Anti-degeneracy penalty: replace the flat-0 plateau with a
        # strictly-negative reward for degenerate / failed TERMINAL
        # transitions so the policy gets a gradient out of the basin.
        if self.anti_degen:
            P = np.float32(self.anti_degen_penalty)
            tau = np.float32(self.anti_degen_tau)
            degen = cos < tau                                  # (T, N)
            fail = None
            if failed_mask is not None:
                fail = np.asarray(failed_mask, dtype=bool)
                degen = degen | fail
            # Only penalise terminal transitions — intermediate steps
            # legitimately carry cosine_sim=0 (sparse-terminal channel)
            # and must stay at the gated value (0) so GAE isn't poisoned
            # with -P on every non-terminal step.
            if terminal_mask is not None:
                degen = degen & np.asarray(terminal_mask, dtype=bool)
            # SHAPED penalty: instead of a FLAT -P (zero advantage inside the
            # whole basin -> value collapse, the cossim-0 absorbing attractor),
            # give the penalty a positive slope in cos so the policy gets a
            # gradient pointing OUT of the basin. Linear ramp on cos in [0, tau]:
            #   cos=0   -> -P
            #   cos=tau -> -P*(1-tau)   (still strictly < 0 for tau in (0,1))
            # which stays <= 0 <= the worst VALID gated reward (cheapness>=0 =>
            # gated>=0 for a valid rule), so best_overall ordering and the
            # anti-hack guarantee are preserved. Failed/sentinel measures have
            # no meaningful cos -> pinned to the floor -P.
            cos_basin = np.clip(cos, 0.0, tau).astype(np.float32)
            shaped = (-(P - cos_basin * P)).astype(np.float32)   # (T, N)
            if fail is not None:
                shaped = np.where(fail, -P, shaped).astype(np.float32)
            gated = np.where(degen, shaped, gated).astype(np.float32)
            # Record for telemetry / caller (best_overall exclusion).
            self._last_degen_mask = degen
        else:
            self._last_degen_mask = np.zeros_like(gated, dtype=bool)

        out[...] = 0.0
        out[..., COSINE_SIM_IDX] = gated
        return out

    # ------------------------------------------------------------------
    # JIT'd per-step act + JIT-side env update
    # ------------------------------------------------------------------
    def _make_act_step_fn_micro(self):
        """Policy-V2 act step: pointer vertex head + MicroActionPolicy
        sub-episode over the chosen vertex's LIVE per-axis features.

        Mirrors cmorl_ray_worker's micro act path, plus: eqn_ids threading
        into the encoder, the pointer per-vertex context feeding the
        sub-episode, and the LIVE (EnvState) axis features (returned to the
        rollout loop so the loss re-evaluates against the exact features the
        sample saw).
        """
        env = self.env
        factor_tables = self.factor_tables
        from alphagrad.approx.ppo import _axis_features_from_state

        @eqx.filter_jit
        def act_step(agent, state_batch, vert_avail_batch,
                     op_mask, factor_mask, quant_mask, key,
                     substep_budget=jnp.int32(MAX_RULES_PER_VERTEX),
                     pair_valid_batch=None, compress_valid_batch=None):
            """``op_mask`` (4,) gates op-types per variant
            (order DIAG/COMPRESS/QUANT/END): any disallowed sampled op_type is
            rewritten to END (ppo-style op_legality_override) so restricted
            variants restrict the action space. ``factor_mask`` /
            ``quant_mask`` are accepted for signature stability but unused —
            the factor / dtype choice sets are governed by the heads (QUANT
            additionally by ALPHAGRAD_QUANT_ALLOWED inside heads.py)."""
            keys = jrand.split(key, self.num_envs)

            def per_env(state_i, avail_i, pv_i, cv_i, k_i):
                k_enc, k_v, k_micro = jrand.split(k_i, 3)
                enc_x, token_mask = agent.encode_tokens(
                    state_i.tokens, key=k_enc, eqn_ids=state_i.eqn_ids,
                )
                vertex_logits, vertex_contexts, value = (
                    agent.policy_value_from_encoding(enc_x, token_mask)
                )
                masked = jnp.where(avail_i > 0.5, vertex_logits, -1e9)
                log_probs_v = jax.nn.log_softmax(masked)
                vertex_action = jrand.categorical(k_v, masked)
                log_prob_v = log_probs_v[vertex_action]
                vertex_id = vertex_action + 1  # env vertex IDs are 1-indexed

                # LIVE per-axis features of the chosen vertex (EnvState is
                # updated as rules land, unlike the cmorl static tables).
                axis_state_v = state_i.axis_state[vertex_action]
                axis_valid_v = state_i.axis_valid_mask[vertex_action]
                v_context = vertex_contexts[vertex_action]
                features = _axis_features_from_state(axis_state_v, axis_valid_v)
                # EXACT live-edge legality for the vertex just sampled. Rows
                # are 1-indexed by vertex id (row 0 is padding), matching
                # ``LiveVertexMaskOracle.masks``.
                pv_v = pv_i[vertex_id]
                cv_v = cv_i[vertex_id]

                (
                    actions, micro_lp, micro_ent, micro_arity,
                    *_dists,
                ) = agent.micro_action_policy.sample(
                    v_context, features, factor_tables, k_micro,
                    pair_valid=pv_v, compress_valid=cv_v,
                )
                op_seq = actions.op_type.astype(jnp.int32)
                i_seq = actions.i.astype(jnp.int32)
                j_seq = actions.j.astype(jnp.int32)
                exp_seq = actions.exponents.astype(jnp.int32)
                f_seq = actions.factor.astype(jnp.int32)
                kind_seq = actions.compress_kind.astype(jnp.int32)
                q_seq = actions.quant_dtype.astype(jnp.int32)
                # Variant gating (mirrors ppo/cmorl's
                # op_legality_override): rewrite disallowed op_types to END
                # and zero their stale args. NO-OP when op_mask is all-ones
                # (variant=full). The stored (rewritten) action is what the
                # loss re-evaluates, keeping rollout/loss consistent.
                _disallowed = (
                    ((op_seq == OP_DIAG) & (op_mask[OP_DIAG] <= 0.5))
                    | ((op_seq == OP_COMPRESS) & (op_mask[OP_COMPRESS] <= 0.5))
                    | ((op_seq == OP_QUANT) & (op_mask[OP_QUANT] <= 0.5))
                )
                op_seq = jnp.where(_disallowed, jnp.int32(OP_END), op_seq)
                i_seq = jnp.where(_disallowed, 0, i_seq)
                j_seq = jnp.where(_disallowed, 0, j_seq)
                exp_seq = jnp.where(_disallowed[:, None], 0, exp_seq)
                f_seq = jnp.where(_disallowed, 0, f_seq)
                kind_seq = jnp.where(_disallowed, 0, kind_seq)
                q_seq = jnp.where(_disallowed, 0, q_seq)
                # SUBSTEP BUDGET: cap the sub-episode at
                # ``substep_budget`` slots; positions >= budget -> OP_END
                # (no-op) + args zeroed (mirrors the _disallowed override,
                # so the stored rewritten action == what the loss
                # re-evaluates). budget=0 => every slot END => pure exact
                # elimination (order only).
                _pos = jnp.arange(op_seq.shape[0], dtype=jnp.int32)
                _over = _pos >= substep_budget
                op_seq = jnp.where(_over, jnp.int32(OP_END), op_seq)
                i_seq = jnp.where(_over, 0, i_seq)
                j_seq = jnp.where(_over, 0, j_seq)
                exp_seq = jnp.where(_over[:, None], 0, exp_seq)
                f_seq = jnp.where(_over, 0, f_seq)
                kind_seq = jnp.where(_over, 0, kind_seq)
                q_seq = jnp.where(_over, 0, q_seq)
                rule_specs = micro_actions_to_rule_specs_jax(
                    op_seq, i_seq, j_seq, f_seq,
                    axis_state_v,
                    compress_kinds=kind_seq,
                    quant_dtypes=q_seq,
                )
                # Joint log-prob = vertex + the sub-episode's GATED
                # autoregressive log-prob (already summed over the real
                # prefix; per-component contributions past END / for
                # non-participating heads are zero inside heads.py).
                log_prob_total = log_prob_v + micro_lp

                env_action = StepAction(
                    target_vertex=jnp.asarray(vertex_id, dtype=jnp.int32),
                    rule_specs=rule_specs,
                )
                partial, order, specs, step = env.step_external_jax_part(
                    state_i, env_action,
                )
                return (
                    vertex_action,
                    op_seq, i_seq, j_seq, exp_seq, f_seq, kind_seq, q_seq,
                    axis_state_v, axis_valid_v, pv_v, cv_v,
                    log_prob_total,
                    value,
                    partial, order, specs, step,
                )

            return jax.vmap(per_env)(
                state_batch, vert_avail_batch,
                pair_valid_batch, compress_valid_batch, keys,
            )

        return act_step

    def _make_assemble_fn(self):
        env = self.env

        @eqx.filter_jit
        def assemble(state_before, partial, tokens, eqn_ids, reward):
            def per_env(s_b, p, t, e, r):
                return env.assemble_step_result(s_b, p, t, e, r).state
            return jax.vmap(per_env)(
                state_before, partial, tokens, eqn_ids, reward,
            )

        return assemble

    # ------------------------------------------------------------------
    # PPO loss + minibatch update
    # ------------------------------------------------------------------
    def _make_update_step_micro(self):
        """Policy-V2 loss: pointer vertex head + MicroActionPolicy.evaluate
        over the stored typed sub-episode.

        Differences vs the legacy loss:
          * the batch carries per-substep action SEQUENCES (S = max_substeps
            per transition) plus the axis features + vertex-avail mask the
            rollout actually used, so the PPO ratio is exactly 1 at epoch 0;
          * the joint log-prob / entropy come from heads.py's GATED
            per-component sums (op=END => no i/j/factor/kind/quant
            contribution) instead of the always-on flat-head sums;
          * the sub-episode entropy is arity-normalised (ppo.py pattern) so a
            16-substep vertex doesn't get a ~16x larger entropy bonus.
        Value loss / gdpo-vs-scalar branches / nan-skip guard are identical
        to the legacy update step.
        """
        clip_eps = self.ppo_eps
        value_coef = self.value_coef
        is_gdpo = self.advantage_norm == "gdpo"
        channel_mask_j = self._channel_mask_j        # (NUM_REWARDS,)
        sparse_mask_j = self._sparse_mask_j          # (NUM_REWARDS,)
        priority_weights_j = self.reward_weights     # (NUM_REWARDS,)
        active_count = jnp.maximum(jnp.sum(channel_mask_j), jnp.float32(1.0))
        factor_tables = self.factor_tables
        # Cost-head aux constants (bridge-cse). Weight 0 when the flag is OFF
        # -> the aux term drops out -> update byte-identical.
        _cost_aux_weight_j = jnp.float32(self._cost_head_aux_weight)
        _cost_aux_enc_grad = bool(self._cost_head_aux_encoder_grad)
        # supervise ONLY the 4 measured target channels (latency_ns,
        # xla_peak_memory, flops, cosine_sim); mask over NUM_REWARDS.
        from alphagrad.approx.env import REWARD_INDEX as _RI
        _cost_tgt_names = ["latency_ns", "xla_peak_memory", "flops", "cosine_sim"]
        _cost_mask_np = np.zeros((NUM_REWARDS,), dtype=np.float32)
        for _cn in _cost_tgt_names:
            _cost_mask_np[_RI[_cn]] = 1.0
        _cost_target_mask_j = jnp.asarray(_cost_mask_np)
        _cost_active_k = jnp.float32(len(_cost_tgt_names))
        from alphagrad.approx.ppo import _axis_features_from_state
        from alphagrad.approx.common.gae import gdpo_normalise_advantages

        def loss_fn(agent, batch, op_mask, factor_mask, quant_mask, key,
                    entropy_coef):
            """``op_mask``/``factor_mask``/``quant_mask`` are threaded for
            signature parity with the legacy loss; the micro path applies
            variant gating at SAMPLE time (op rewrite in act_step), so the
            stored actions are already the gated ones."""
            (
                tokens, eqn_ids, avail,
                actions, op_a, i_a, j_a, exp_a, f_a, kind_a, q_a,
                axis_state_a, axis_valid_a, pair_valid_a, compress_valid_a,
                old_log_probs, returns, advantages, valid,
                cost_target, cost_valid,
            ) = batch
            if is_gdpo:
                adv_scalar = gdpo_normalise_advantages(
                    advantages, channel_mask_j, sparse_mask_j,
                    priority_weights_j,
                )  # (B,)
            else:
                adv_scalar = advantages
            keys = jrand.split(key, tokens.shape[0])

            def per_sample(tok, eqn, av, v_act, op, i_s, j_s, exp_s, f_s,
                           kind_s, q_s, ax_st, ax_va, pv_s, cv_s,
                           olp, ret, adv, k,
                           cost_target, cost_valid):
                enc_x, token_mask = agent.encode_tokens(
                    tok, key=k, eqn_ids=eqn,
                )
                vertex_logits, vertex_contexts, value = (
                    agent.policy_value_from_encoding(enc_x, token_mask)
                )
                # Cost-head aux prediction (symlog measured 4-tuple). Encoder
                # grad flow toggle: stop_gradient on enc_x when disabled so the
                # aux loss trains only the cost head (default: flows into the
                # shared encoder = the representation-learning benefit).
                _enc_for_cost = enc_x if _cost_aux_enc_grad else jax.lax.stop_gradient(enc_x)
                cost_pred = agent.cost_from_encoding(_enc_for_cost, token_mask)
                # Same avail mask the rollout sampled under, so the vertex
                # log-prob ratio is exact (legacy loss skipped this).
                masked_v = jnp.where(av > 0.5, vertex_logits, -1e9)
                log_probs = jax.nn.log_softmax(masked_v)
                lp_v = log_probs[v_act]
                p = jax.nn.softmax(masked_v)
                ent_v = -jnp.sum(p * log_probs)

                v_context = vertex_contexts[v_act]
                features = _axis_features_from_state(ax_st, ax_va)
                action = MicroAction(
                    op_type=op, i=i_s, j=j_s, exponents=exp_s,
                    factor=f_s, compress_kind=kind_s, quant_dtype=q_s,
                )
                (
                    micro_lp, micro_ent, micro_arity, *_dists,
                ) = agent.micro_action_policy.evaluate(
                    v_context, features, factor_tables, action,
                    pair_valid=pv_s, compress_valid=cv_s,
                )
                new_log_prob = lp_v + micro_lp
                # Arity-normalised sub-episode entropy (ppo.py / cmorl
                # pattern) so entropy_coef stays on the single-head scale.
                micro_ent_norm = micro_ent / jnp.maximum(micro_arity, 1.0)
                entropy = ent_v + micro_ent_norm
                z = jnp.float32(0.0)
                per_head = jnp.stack([ent_v, micro_ent_norm, z, z, z, z])

                logratio = new_log_prob - olp
                ratio = jnp.exp(logratio)
                surr1 = ratio * adv
                surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_loss = -jnp.minimum(surr1, surr2)
                # PER-COMPONENT KL (bridge-cse). The joint action log-prob is a
                # SUM over its active components: 1 vertex-selection head +
                # ``micro_arity`` active micro-action components (the same
                # active mask the gated ``micro_lp`` sums over). So the joint
                # logratio is a SUM of per-component logratios, and the joint
                # approx_kl is ~ (n_components)x a single-action KL — which is
                # why target_kl=0.15 trips after ~1 minibatch. Expose the
                # per-sample active-component count so the early-stop can use
                # the MEAN-per-component KL (standard PPO threshold semantics).
                n_components = jnp.float32(1.0) + jnp.maximum(
                    micro_arity, jnp.float32(0.0)
                )
                if is_gdpo:
                    per_channel_se = (value - ret) ** 2           # (K,)
                    normaliser = jnp.maximum(jnp.abs(ret), 1.0)   # (K,)
                    value_loss = jnp.sum(
                        per_channel_se / (normaliser ** 2) * channel_mask_j
                    ) / active_count
                else:
                    # PopArt (always on): ``ret`` is the (K,) PER-CHANNEL
                    # NORMALISED target (G_k - mu_k)/sigma_k staged upstream;
                    # ``value`` is the critic's normalised prediction v_hat.
                    # Plain MSE over the active channels.
                    per_channel_se = (value - ret) ** 2           # (K,)
                    value_loss = jnp.sum(
                        per_channel_se * channel_mask_j
                    ) / active_count
                # AUX cost loss for THIS sample: Huber(cost_pred, symlog(target))
                # over the 4 target channels, gated by cost_valid (terminal &
                # non-failed). symlog matches the Phase-1 probe target space.
                _cost_tgt_sl = jnp.sign(cost_target) * jnp.log1p(jnp.abs(cost_target))
                _cost_err = optax.huber_loss(cost_pred, _cost_tgt_sl, delta=1.0)  # (K,)
                _cost_err = jnp.sum(_cost_err * _cost_target_mask_j) / _cost_active_k
                cost_aux_l = _cost_err * cost_valid  # 0 on non-terminal/failed
                return (policy_loss, value_loss, entropy, per_head, logratio,
                        n_components, cost_aux_l, cost_valid)

            (p_l, v_l, ent, per_head_ent, logratio, n_comp,
             cost_aux_l, cost_valid_v) = jax.vmap(
                per_sample
            )(
                tokens, eqn_ids, avail,
                actions, op_a, i_a, j_a, exp_a, f_a, kind_a, q_a,
                axis_state_a, axis_valid_a, pair_valid_a, compress_valid_a,
                old_log_probs, returns, adv_scalar, keys,
                cost_target, cost_valid,
            )
            # Failed rows stay INCLUDED (bounded-negative penalty upstream
            # gives them a repulsive advantage) — same as the legacy loss.
            del valid
            ppo_loss = jnp.mean(p_l)
            value_loss = jnp.mean(v_l)
            entropy_loss = -jnp.mean(ent)
            # AUX cost-head loss: mean over the TERMINAL (cost_valid=1) samples
            # only, weighted by ALPHAGRAD_COST_HEAD_AUX_WEIGHT (0 when OFF ->
            # this term vanishes -> total is byte-identical to baseline).
            _cost_n = jnp.maximum(jnp.sum(cost_valid_v), 1.0)
            cost_aux_loss = jnp.sum(cost_aux_l) / _cost_n
            total = (ppo_loss + value_coef * value_loss
                     + entropy_coef * entropy_loss
                     + _cost_aux_weight_j * cost_aux_loss)
            head_means = jnp.mean(per_head_ent, axis=0)  # (6,)
            # Fix 3: approx_kl for logging + Fix 2 KL early-stop. Schulman
            # low-variance non-negative estimator mean((r-1) - logratio).
            #   * approx_kl        = JOINT KL (SUM over action components; the
            #                        legacy quantity — ~n_components x a single-
            #                        action KL).
            #   * approx_kl_percomp = PER-ACTIVE-COMPONENT KL: the joint
            #                        Schulman estimator DIVIDED by the number of
            #                        active components (1 vertex + micro_arity),
            #                        per sample, then averaged. This is a proper
            #                        single-action-scale KL so target_kl=0.15 is
            #                        a standard-PPO per-component threshold
            #                        rather than a joint-sum one (which tripped
            #                        after ~1 minibatch). The early-stop uses
            #                        THIS when ALPHAGRAD_KL_PER_COMPONENT=1
            #                        (default); both are logged.
            _ratio_kl = jnp.exp(logratio)
            _kl_per_sample = (_ratio_kl - 1.0) - logratio        # (B,)
            approx_kl = jnp.mean(_kl_per_sample)                  # joint (sum)
            approx_kl_percomp = jnp.mean(
                _kl_per_sample / jnp.maximum(n_comp, jnp.float32(1.0))
            )
            aux = {
                "ppo_loss": ppo_loss,
                "value_loss": value_loss,
                "approx_kl": approx_kl,
                "approx_kl_percomp": approx_kl_percomp,
                "entropy": jnp.mean(ent),
                "entropy_coef": entropy_coef,
                "entropy/vertex": head_means[0],
                "entropy/micro": head_means[1],
                "total_loss": total,
                "cost_head/aux_loss": cost_aux_loss,
                "cost_head/n_terminal": _cost_n,
            }
            return total, aux

        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        @eqx.filter_jit
        def update_step(agent, opt_state, batch,
                        op_mask, factor_mask, quant_mask, key,
                        entropy_coef):
            (loss, aux), grads = grad_fn(
                agent, batch, op_mask, factor_mask, quant_mask, key,
                entropy_coef,
            )
            # Fix 3: global grad norm BEFORE the optax clip/update — the
            # raw signal a catastrophic update is landing (previously
            # invisible in wandb).
            _gsq = jax.tree_util.tree_reduce(
                lambda acc, g: acc + jnp.sum(jnp.square(g)),
                eqx.filter(grads, eqx.is_inexact_array),
                jnp.float32(0.0),
            )
            aux = dict(aux)
            aux["grad_norm"] = jnp.sqrt(_gsq)
            # NaN-skip guard — identical to the legacy update step.
            loss_finite = jnp.isfinite(loss)
            zero_grads = jax.tree.map(jnp.zeros_like, grads)
            safe_grads = jax.tree.map(
                lambda g, z: jnp.where(loss_finite, g, z), grads, zero_grads,
            )
            updates, new_opt_state = self.optimizer.update(
                safe_grads, opt_state, agent,
            )
            new_agent = eqx.apply_updates(agent, updates)
            new_opt_state = jax.tree.map(
                lambda new, old: jnp.where(loss_finite, new, old),
                new_opt_state, opt_state,
            )
            aux = dict(aux)
            aux["nan_skip"] = jnp.where(loss_finite, 0, 1).astype(jnp.int32)
            return new_agent, new_opt_state, aux

        return update_step

    # ------------------------------------------------------------------
    # Driver entry — one episode (rollout + train)
    # ------------------------------------------------------------------
    def run_rollout_and_train(self, rng_seed: int) -> dict:
        """One on-policy rollout + ppo-epochs * minibatches gradient steps.

        Returns a metrics dict ready to log to wandb.
        """
        if not hasattr(self, "_act_step"):
            self._act_step = self._make_act_step_fn_micro()
            self._assemble = self._make_assemble_fn()
            self._update_step = self._make_update_step_micro()

        key = jrand.PRNGKey(int(rng_seed))
        key, reset_key = jrand.split(key)

        # Fresh on-policy episode: reset state for every env. Preserve
        # the sharded layout (data-parallel along the env axis).
        env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
        )
        self.env_states = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding)
            if eqx.is_array(x) else x,
            env_states,
        )
        self._reset_mask_oracles()
        self._mask_probe_seconds = 0.0

        # Rollout buffers — numpy is fine; we re-stage to JAX once for
        # the update step.
        T = int(self.rollout_length)
        N = int(self.num_envs)
        buf_tokens = np.zeros((T, N, MAX_TOKENS), dtype=np.int32)
        # Per-token equation ids for the relational gate (palimpsa_bi) /
        # relational bias (transformer). Recorded alongside tokens so the
        # loss path re-encodes with the SAME relational structure the
        # rollout policy used (mirrors ppo.py which carries state.eqn_ids
        # into the loss batch). Pad/non-eqn tokens are -1.
        buf_eqn_ids = np.zeros((T, N, MAX_TOKENS), dtype=np.int32)
        buf_actions = np.zeros((T, N), dtype=np.int32)
        # Policy V2: full typed sub-episode buffers — one sequence of up
        # to ``S = max_substeps`` DIAG/COMPRESS/QUANT micro-actions per
        # (step, env), plus the LIVE axis features + vertex-avail mask
        # the sample was drawn under (the loss re-evaluates against
        # exactly these, so the PPO ratio is 1 at epoch 0).
        S = int(self.max_substeps)
        P = int(MAX_PRIMES)
        buf_op = np.zeros((T, N, S), dtype=np.int32)
        buf_i = np.zeros((T, N, S), dtype=np.int32)
        buf_j = np.zeros((T, N, S), dtype=np.int32)
        buf_exp = np.zeros((T, N, S, P), dtype=np.int32)
        buf_f = np.zeros((T, N, S), dtype=np.int32)
        buf_kind = np.zeros((T, N, S), dtype=np.int32)
        buf_q = np.zeros((T, N, S), dtype=np.int32)
        buf_axis_state = np.zeros(
            (T, N, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), dtype=np.int32,
        )
        buf_axis_valid = np.zeros(
            (T, N, MAX_AXES_PER_VERTEX), dtype=np.float32,
        )
        # The EXACT live-edge masks the sample was drawn under. Buffered rather
        # than recomputed in the loss: the loss re-scores a stored action
        # against a graph state that no longer exists, and the PPO ratio is
        # only 1 at epoch 0 if both sides mask identically.
        buf_pair_valid = np.zeros(
            (T, N, MAX_AXES_PER_VERTEX, MAX_AXES_PER_VERTEX), dtype=np.float32,
        )
        buf_compress_valid = np.zeros(
            (T, N, MAX_AXES_PER_VERTEX), dtype=np.float32,
        )
        buf_avail = np.zeros((T, N, int(self.total_v)), dtype=np.float32)
        buf_log_probs = np.zeros((T, N), dtype=np.float32)
        # Per-channel value buffer (post-refactor). The value head is
        # always K=NUM_REWARDS wide; the scalar-mode path collapses
        # via dot product with ``reward_weights`` after rollout
        # collection, the gdpo-mode path keeps the K-vector through
        # GAE and the advantage stack.
        buf_values = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        # Raw 8-vec reward per step — the per-channel diagnostics read from
        # this buffer. The scalar reward is derived from this buffer +
        # ``reward_weights`` after the rollout loop terminates.
        buf_reward_vec = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)
        # Phase 3 (b): per-step sentinel mask. True when the CPU pool
        # timed out / errored on that env's tokenisation. Before GAE we
        # force `dones[t]=1` and `rewards[t]=0` for sentinel slots so
        # on-policy PPO doesn't bootstrap value over a -1e10 reward
        # (which would otherwise survive symlog'ing in GAE).
        buf_sentinel = np.zeros((T, N), dtype=bool)

        state = self.env_states
        for t in range(T):
            key, sub = jrand.split(key)
            avail = self._vertex_avail(state)
            # EXACT per-vertex micro-action legality, probed off the LIVE edges
            # of every still-available vertex (see LiveVertexMaskOracle). This
            # is what stops the policy proposing a Diag/Compress graphax will
            # reject -- each rejection sentinels the whole measurement.
            pv_np, cv_np = self._live_masks(np.asarray(avail))
            pv_j = jax.device_put(jnp.asarray(pv_np), self.data_sharding)
            cv_j = jax.device_put(jnp.asarray(cv_np), self.data_sharding)
            # Observation the policy ACTS ON this step (pre-elimination). BOTH
            # loss paths re-evaluate log-probs against these tokens: storing the
            # post-assemble tokens pairs a_t with s_{t+1}, which makes the PPO
            # ratio != 1 even at epoch 0 and regresses the critic onto G_t at
            # the wrong state.
            pre_tokens = np.asarray(state.tokens)
            pre_eqn_ids = np.asarray(state.eqn_ids)
            (
                actions, op_a, i_a, j_a, exp_a, f_a, kind_a, q_a,
                ax_st_a, ax_va_a, pv_a, cv_a,
                log_probs, values,
                partial, order, specs, step,
            ) = self._act_step(
                self.agent, state, avail,
                self._current_op_mask_j, self._current_factor_mask_j,
                self._current_quant_mask_j, sub,
                self._substep_budget_j, pv_j, cv_j,
            )

            # Convert to numpy for Ray fan-out.
            order_np = np.asarray(order)
            specs_np = np.asarray(specs)
            step_np = np.asarray(step)

            # Terminal step + --measure-queue: use the global per-point
            # measurement queue (full core utilisation). Non-terminal
            # steps stay on the cheap tokenize-only batch path. Grafted
            # from the approx branch — additive, gated entirely on the
            # ``--measure-queue`` flag (default off keeps theirs' path).
            _is_terminal_step = t == T - 1
            _local_tokenize = (
                not _is_terminal_step
                and self._local_tokenize_enabled
                and self._terminal_rewards_only
            )
            if (
                _is_terminal_step
                and getattr(self.args, "measure_queue", False)
                and self._terminal_rewards_only
            ):
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_terminal_queue(order_np, specs_np, step_np)
                )
            elif _local_tokenize:
                # Non-terminal step under ``terminal_rewards_only`` with
                # ALPHAGRAD_LOCAL_TOKENIZE=1: the reward is zeroed
                # (env.py early-return) and the tokens are a pure
                # deterministic fn of (order, specs, step), so compute
                # them in-process and skip the ~2 s/step Ray round-trip.
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._tokenize_local(order_np, specs_np, step_np)
                )
            else:
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_tokenize(order_np, specs_np, step_np)
                )
            buf_sentinel[t] = sentinel_mask

            # Shard the Ray-returned numpy back to the env axis so the
            # assemble JIT runs SPMD (otherwise JAX would gather the
            # already-sharded `state` / `partial` onto one device).
            tokens_j = jax.device_put(
                jnp.asarray(tokens_np, dtype=jnp.int32), self.data_sharding,
            )
            eqn_ids_j = jax.device_put(
                jnp.asarray(eqn_ids_np, dtype=jnp.int32), self.data_sharding,
            )
            reward_j = jax.device_put(
                jnp.asarray(reward_np, dtype=jnp.float32), self.data_sharding,
            )
            state = self._assemble(state, partial, tokens_j, eqn_ids_j, reward_j)
            # Keep the mask oracles in lockstep with the env: commit the vertex
            # each env just eliminated, WITH the rules that landed on it (an
            # approximation changes every downstream edge's structure).
            self._advance_mask_oracles(order_np, step_np, specs_np)

            # Record the PRE-step obs (the one a_t was sampled from) plus
            # the sub-episode sequences + the axis features / avail mask
            # the sample used.
            buf_tokens[t] = pre_tokens
            buf_eqn_ids[t] = pre_eqn_ids
            buf_avail[t] = np.asarray(avail)
            buf_exp[t] = np.asarray(exp_a)
            buf_kind[t] = np.asarray(kind_a)
            buf_axis_state[t] = np.asarray(ax_st_a)
            buf_axis_valid[t] = np.asarray(ax_va_a)
            buf_pair_valid[t] = np.asarray(pv_a)
            buf_compress_valid[t] = np.asarray(cv_a)
            buf_actions[t] = np.asarray(actions)
            buf_op[t] = np.asarray(op_a)
            buf_i[t] = np.asarray(i_a)
            buf_j[t] = np.asarray(j_a)
            buf_f[t] = np.asarray(f_a)
            buf_q[t] = np.asarray(q_a)
            buf_log_probs[t] = np.asarray(log_probs)
            buf_values[t] = np.asarray(values)  # (N, NUM_REWARDS)
            buf_reward_vec[t] = reward_np
            buf_dones[t] = np.asarray(state.terminated).astype(np.float32)

        # Bootstrap value at the final state (for the GAE next_value
        # term on the last timestep). The value head emits K=NUM_REWARDS
        # heads; the bootstrap is shape ``(N, NUM_REWARDS)``.
        key, boot_key = jrand.split(key)
        boot_keys = jax.device_put(
            jrand.split(boot_key, N), self.data_sharding,
        )
        bootstrap = np.asarray(
            jax.vmap(lambda s, k: self.agent.value(s.tokens, key=k, eqn_ids=s.eqn_ids))(state, boot_keys),
        )  # (N, NUM_REWARDS)

        # ANTI-DEGEN / failed-measure mask. A measurement is "failed"
        # when EITHER the CPU pool timed out / errored (``buf_sentinel``)
        # OR the env returned a sentinel reward vector (mem-gate skip /
        # shape-storm / exception path: cost channels == -1e10). Both
        # currently get their reward channels ZEROED below for GAE
        # stability — but a zeroed cost row reads as "free perfect"
        # (cheapness = W - symlog(0) = W) in the mult-gate and would
        # corrupt best_overall, while a zeroed cossim=0 row collapses to
        # gated=0 (the flat plateau). We capture the mask HERE, before
        # zeroing, so the mult-gate can stamp these with -P (anti-degen
        # penalty) and best_overall can exclude them — never zero.
        from alphagrad.approx.common.compile_cache import (
            SENTINEL_REWARD_VALUE as _SENTINEL_RV,
        )
        # NON-FINITE REWARD SCRUB. A single nan/inf ANYWHERE in the (T, N, K)
        # reward buffer poisons the ENTIRE episode's update: scalar-mode GAE
        # runs per-channel then dot-products with the weights (0 * nan = nan,
        # so even a ZERO-weight diagnostics channel leaks through), and the
        # rollout-wide advantage z-score then smears that one nan over every
        # transition -> loss non-finite on all minibatches -> nan_skip ==
        # mb_count and the episode trains NOTHING (observed as the permanent
        # ``loss(p/v):nan nan_skip:60`` signature).
        #   * non-finite in a ZERO-weight channel  -> zero the entry (it only
        #     feeds telemetry; the row stays a valid training sample);
        #   * non-finite in a REWARDED channel     -> the measurement is
        #     garbage: mark the transition failed so the existing sentinel
        #     machinery (zero + bounded penalty + best-exclusion) handles it.
        _nonfinite_rew = ~np.isfinite(buf_reward_vec)          # (T, N, K)
        if _nonfinite_rew.any():
            _w_nz = self.reward_weights_np != 0.0              # (K,)
            _bad_rewarded = (
                _nonfinite_rew & _w_nz[None, None, :]
            ).any(axis=-1)                                     # (T, N)
            print(
                f"[ppo_ray] non-finite reward scrub: "
                f"{int(_nonfinite_rew.sum())} entries "
                f"({int((_nonfinite_rew & ~_w_nz[None, None, :]).sum())} in "
                f"zero-weight channels zeroed; "
                f"{int(_bad_rewarded.sum())} transitions with non-finite "
                f"REWARDED channels -> failed)."
            )
            buf_reward_vec = np.where(
                _nonfinite_rew, 0.0, buf_reward_vec,
            ).astype(np.float32)
        else:
            _bad_rewarded = np.zeros((T, N), dtype=bool)
        # Same guard for the rollout value estimates + bootstrap (a nan value
        # poisons GAE identically).
        _nf_val = int((~np.isfinite(buf_values)).sum())
        _nf_boot = int((~np.isfinite(bootstrap)).sum())
        if _nf_val or _nf_boot:
            print(
                f"[ppo_ray] non-finite VALUE scrub: {_nf_val} rollout value "
                f"entries, {_nf_boot} bootstrap entries -> zeroed."
            )
            buf_values = np.nan_to_num(
                buf_values, nan=0.0, posinf=0.0, neginf=0.0,
            )
            bootstrap = np.nan_to_num(
                bootstrap, nan=0.0, posinf=0.0, neginf=0.0,
            )
        _nf_lp = int((~np.isfinite(buf_log_probs)).sum())
        if _nf_lp:
            print(
                f"[ppo_ray] non-finite rollout LOG-PROB scrub: {_nf_lp} "
                f"entries -> zeroed (ratio falls back to exp(new_lp))."
            )
            buf_log_probs = np.nan_to_num(
                buf_log_probs, nan=0.0, posinf=0.0, neginf=0.0,
            )
        # Env-side sentinel: any COST channel exactly == SENTINEL value.
        _cost_idx = [
            i for i in range(buf_reward_vec.shape[-1])
            if i not in (COSINE_SIM_IDX, FROB_RESIDUAL_IDX)
        ]
        _env_sentinel = np.any(
            buf_reward_vec[..., _cost_idx] == np.float32(_SENTINEL_RV),
            axis=-1,
        )  # (T, N)
        buf_failed = buf_sentinel | _env_sentinel | _bad_rewarded  # (T, N)

        # LOG ALL MEASUREMENT CHANNELS (bridge-cse). Snapshot the RAW measured
        # per-channel reward vector HERE — before any sentinel substitution,
        # symlog compression, guide cap or gate — so every one of the 10
        # REWARD_NAMES (incl. the ZERO-WEIGHT measured-but-not-rewarded channels
        # flops / max_io_sum / bytes_accessed / peak_memory / frob_residual /
        # xla_peak_memory) can be logged in its NATURAL raw units at
        # ``measure/<name>`` below. Cost channels are stored negated (env
        # convention); logged as-measured.
        buf_reward_vec_measured_raw = np.array(
            buf_reward_vec, dtype=np.float32, copy=True
        )

        # Phase 3 (b): mask sentinel transitions. Zero the per-channel
        # reward vector AND force dones=1 so GAE treats sentinel
        # timesteps as terminal and the value head bootstraps cleanly
        # at the next state. Per-channel zero (vs the legacy scalar
        # zero-out) means the gdpo path also sees zero reward on
        # those steps. NB: in mult+anti_degen mode the gate stamps the
        # cosine channel of these rows with -P AFTER this zeroing, so the
        # net effect is a strictly-negative gated reward (a gradient out),
        # NOT the misleading "free" zero — the zeroing here only clears the
        # raw -1e10 cost channels that would otherwise blow up symlog/GAE.
        # DYNAMIC SENTINEL (bridge-cse). Update the per-channel EMA from the
        # RAW, non-failed TERMINAL reward vectors FIRST (buf_reward_vec is
        # still the raw measured buffer here — cost channels raw-negative,
        # cosine_sim full-range now that the env defers the guide cap), so
        # substituted sentinel values NEVER feed back into the stats. Then
        # replace each failed row's whole reward vector with the bounded
        # ``mu_c - K*sigma_c`` (finite, != any value filter_sentinel_mask
        # keys on) instead of the legacy per-channel ZERO. Falls back to the
        # legacy zero until the EMA has seen at least one real rollout.
        _dyn_scalar = float("nan")
        # TRULY-NEUTRAL SENTINEL (bridge-cse). Post-transform per-channel
        # override applied to failed rows AFTER the cost-symlog pass; set in
        # the sentinel block below when PopArt + neutral-mu are on. None = no
        # override (leave the raw-mu substitute in place).
        _neutral_post_vec = None
        if self.dynamic_sentinel:
            _term_ok = buf_dones.astype(bool) & ~buf_failed        # (T, N)
            if _term_ok.any():
                _real = buf_reward_vec[_term_ok].astype(np.float64)  # (M, K)
                _b = self._sent_ema_beta
                _bm = _real.mean(axis=0)
                _bs = (_real * _real).mean(axis=0)
                if self._sent_ema_n == 0:
                    self._sent_ema_mean = _bm
                    self._sent_ema_sq = _bs
                else:
                    self._sent_ema_mean = (
                        (1.0 - _b) * self._sent_ema_mean + _b * _bm
                    )
                    self._sent_ema_sq = (
                        (1.0 - _b) * self._sent_ema_sq + _b * _bs
                    )
                self._sent_ema_n += 1
        if buf_failed.any():
            _use_dyn = self.dynamic_sentinel and self._sent_ema_n > 0
            if _use_dyn:
                _mu = self._sent_ema_mean
                _sig = np.sqrt(
                    np.maximum(self._sent_ema_sq - _mu * _mu, 1e-12)
                )
                _dyn_vec = (_mu - self.sentinel_k * _sig).astype(np.float32)  # (K,)

                # Helper: scalarise a RAW per-channel reward vector through the
                # EXACT transform the GAE reward consumes (cost symlog+cap,
                # cossim guide cap), then weighted-dot. Reused for the neutral
                # shift below AND the logged scalar.
                def _scalarise_dyn(_vec):
                    _s = np.asarray(_vec, dtype=np.float64).copy()
                    if self._additive_symlog_cost:
                        _mm = _COST_SYMLOG_MASK_NP
                        _sc = _s * self._inner_lambda_np
                        _slv = np.sign(_sc) * np.log1p(np.abs(_sc))
                        if self._cost_symlog_cap > 0.0:
                            _slv = np.clip(
                                _slv,
                                -self._cost_symlog_cap,
                                self._cost_symlog_cap,
                            )
                        _s = np.where(_mm, _slv, _s)
                    if (
                        self.raw_cossim_thread
                        and self._cossim_guide_cap is not None
                    ):
                        _s[COSINE_SIM_IDX] = min(
                            _s[COSINE_SIM_IDX],
                            float(self._cossim_guide_cap),
                        )
                    return _s

                buf_reward_vec = np.where(
                    buf_failed[..., None],
                    _dyn_vec[None, None, :],
                    buf_reward_vec,
                ).astype(np.float32)
                # ``_dyn_vec`` is in RAW reward space (cost channels ~ -1e6).
                # The reward the GAE eventually consumes runs the SAME
                # symlog(lambda_inner*cost) + cap pass over these rows as
                # over real measurements (below), so report the logged scalar
                # in that post-transform space — otherwise the raw cost EMA
                # dominates and the metric reads ~-1e5 instead of the ~mildly
                # -ve scalar the policy actually sees.
                _dv_scalarised = _scalarise_dyn(_dyn_vec)
                _dyn_scalar = float(
                    np.dot(_dv_scalarised, self.reward_weights_np)
                )
                # TRULY-NEUTRAL SENTINEL (bridge-cse). With SENTINEL_K=0 the
                # substitute above is the raw EMA mu: quality channels (non-
                # symlog) keep mu~0.65-0.7 while cost channels symlog-compress
                # to ~0, so the failed row's post-transform return reads as an
                # average-quality, near-free-cost row => a net-POSITIVE ~+0.71
                # scalar that SPIKES mean_return AND injects a spurious POSITIVE
                # GAE advantage. USER CHOICE "truly-neutral mu": a failed row
                # must not steer the policy either way. When enabled we OVERRIDE
                # each failed row's POST-TRANSFORM per-channel return to the
                # value the critic already predicts for an average state, so the
                # PER-CHANNEL PopArt advantage ``(G_k - mu_k)/sigma_k`` is
                # EXACTLY 0 on every channel (not merely scalar-matched, which a
                # per-channel-normalised advantage does NOT zero) and the value
                # target ``(G_k - mu_k)/sigma_k`` is 0 too. The override lands
                # AFTER the cost-symlog pass (below) since ``popart.mu`` lives
                # in POST-transform return space; we STASH the target here and
                # apply it there. Flag ALPHAGRAD_SENTINEL_NEUTRAL_MU (default 1).
                # The target is ``popart.mu`` — the per-channel return mean
                # the critic is normalised against (pre-update = the rollout
                # frame), so a failed row's per-channel advantage is exactly 0.
                if self._sentinel_neutral_mu:
                    _neutral_post_vec = self.popart.mu.astype(np.float32).copy()
                    _dyn_scalar = float(
                        np.dot(
                            _neutral_post_vec.astype(np.float64),
                            self.reward_weights_np.astype(np.float64),
                        )
                    )
            elif (
                self._sentinel_neutral_mu
                and self._neutral_unwarmed
            ):
                # KILL THE -2 FLOOR ON THE FIRST / UN-WARMED ROLLOUT (bridge-cse).
                # Before the sentinel EMA has warmed (``_sent_ema_n == 0``, i.e.
                # the very first rollout, or any rollout whose every env failed
                # before a real terminal ever landed) the legacy path zeroed the
                # failed rows and the additive ``failed_penalty=-2.0`` stamp
                # below then floored ``mean_return`` / ``best_return`` at -2.
                # That is exactly the -2 the user still saw. Under PopArt +
                # neutral-mu the truly-neutral substitute (``popart.mu``, which
                # is well-defined — zeros — from rollout 0) makes the failed
                # row's per-channel PopArt advantage ``(G_k - mu_k)/sigma_k``
                # EXACTLY 0 and its weighted return ``dot(mu, w)`` (0 at rollout
                # 0), so a failed / un-warmed row is neither rewarded nor
                # punished on the FIRST rollout too — no -2 floor, uniform
                # neutral handling. Set ``_dyn_scalar`` finite so the
                # ``failed_penalty`` stamp below is skipped, zero the raw buffer
                # (the post-transform override at the ``_neutral_post_vec`` block
                # then writes ``popart.mu``). Flag ALPHAGRAD_NEUTRAL_UNWARMED
                # (default 1) reverts to the -2 legacy path when 0.
                buf_reward_vec = np.where(
                    buf_failed[..., None], 0.0, buf_reward_vec,
                )
                _neutral_post_vec = self.popart.mu.astype(np.float32).copy()
                _dyn_scalar = float(
                    np.dot(
                        _neutral_post_vec.astype(np.float64),
                        self.reward_weights_np.astype(np.float64),
                    )
                )
            else:
                # Legacy path (dynamic off, or EMA not warmed yet): zero the
                # per-channel reward vector (the additive failed_penalty /
                # failed_adv_stamp below then carry the repulsive signal).
                buf_reward_vec = np.where(
                    buf_failed[..., None], 0.0, buf_reward_vec,
                )
            buf_dones = np.where(buf_failed, 1.0, buf_dones)
            n_sentinels = int(buf_failed.sum())
            print(
                f"[ppo_ray] {n_sentinels}/{T*N} failed/sentinel transitions "
                f"this episode (pool timeout / mem-gate / shape-storm); "
                + (
                    f"dynamic sentinel mu-{self.sentinel_k}*sigma "
                    f"(scalar={_dyn_scalar:+.4g}), forcing dones"
                    if _use_dyn
                    else "zeroing raw rewards, forcing dones"
                )
                + (
                    f", anti-degen penalty -{self.anti_degen_penalty} applied"
                    if (self.anti_degen and self.reward_mode == 'mult')
                    else ""
                )
                + "."
            )
        # Stash for wandb (surfaced in last_aux below). NaN when no sentinel
        # fired this rollout OR dynamic sentinel is off/unwarmed.
        self._last_dyn_sentinel_scalar = _dyn_scalar

        # ADDITIVE_SYMLOG_COST: symlog-compress the cost channels so the raw
        # ~1e6 cost magnitudes don't dwarf the [0,1] quality channels in the
        # scalar weighted sum (see _COST_SYMLOG_MASK_NP). Applied AFTER sentinel
        # zeroing (symlog(0)==0, so zeroed rows stay 0) and BEFORE reward
        # conditions / GAE. additive mode only.
        if self._additive_symlog_cost:
            _m = _COST_SYMLOG_MASK_NP[None, None, :]  # (1,1,NUM_REWARDS)
            # lambda INSIDE the symlog: symlog(lambda_inner_c * raw_cost).
            # lambda_inner ~ 1/typical_raw_cost puts a typical cost near the
            # LINEAR regime so order-of-magnitude differences are preserved
            # (sensitive), while outliers still log-bound. Sign-preserving
            # (stored cost is -raw). Quality channels (cosine_sim, bkstep_acc)
            # are masked out (lambda_inner=1 for them anyway).
            _scaled = buf_reward_vec * self._inner_lambda_np[None, None, :]
            _sl = np.sign(_scaled) * np.log1p(np.abs(_scaled))
            # V2 anti-hack fix 2: saturate the symlog'd cost channels at
            # +/-cap -- beyond ~2x-typical cost the term is CONSTANT, so
            # there is no reward for making an already-expensive graph
            # cheaper by making it WORSE (see __init__ for the pf7ityh6
            # post-mortem). Quality channels are outside ``_m`` and
            # untouched.
            if self._cost_symlog_cap > 0.0:
                _sl = np.clip(
                    _sl, -self._cost_symlog_cap, self._cost_symlog_cap,
                )
            buf_reward_vec = np.where(
                _m,
                _sl,
                buf_reward_vec,
            ).astype(np.float32)

        # TRULY-NEUTRAL SENTINEL (bridge-cse) — POST-TRANSFORM override. Now
        # that the cost channels are in symlog space, write each failed row's
        # per-channel return to ``popart.mu`` (the per-channel mean the critic
        # is normalised against, captured PRE this rollout's PopArt update =
        # the frame the rollout head was trained in). Under PopArt the
        # collapsed advantage is ``sum_k (G_k - mu_k)/sigma_k * w_k`` and the
        # value target is ``(G_k - mu_k)/sigma_k``; placing G_k == mu_k makes
        # BOTH EXACTLY 0 on every channel — a failed row is truly neutral in
        # advantage AND in mean_return (it reads as an average episode), never
        # the spurious +0.71 the raw-mu-with-K=0 substitute produced. Skipped
        # when the override is disabled or PopArt is off (``_neutral_post_vec``
        # is None), leaving the raw-mu substitute + non-PopArt shift in place.
        if _neutral_post_vec is not None and buf_failed.any():
            buf_reward_vec = np.where(
                buf_failed[..., None],
                _neutral_post_vec[None, None, :],
                buf_reward_vec,
            ).astype(np.float32)

        # Phase D: apply conditioned-reward gates. For each spec
        # ``(easier, harder, op, threshold)``, zero the easier channel
        # at any (t, n) where the harder channel doesn't satisfy
        # ``op(harder, threshold)``. Threshold comparison happens in
        # symlog space for cost channels (so the user can write
        # ``flops>=1e9`` without thinking about the squashing) and in
        # raw space for bounded channels (cosine_sim).
        reward_condition_stats: dict = {}
        if self._reward_conditions:
            for easier_idx, harder_idx, op, thresh in self._reward_conditions:
                harder_val = buf_reward_vec[..., harder_idx]  # (T, N)
                if harder_idx in _NO_SYMLOG_REWARD_INDICES:
                    h_compare = harder_val
                    t_compare = np.float32(thresh)
                else:
                    h_compare = np.sign(harder_val) * np.log1p(np.abs(harder_val))
                    t_compare = (
                        np.sign(np.float32(thresh))
                        * np.log1p(np.abs(np.float32(thresh)))
                    )
                if op == ">=":
                    gate = (h_compare >= t_compare).astype(np.float32)
                else:
                    gate = (h_compare <= t_compare).astype(np.float32)
                # Sparse-terminal harder channels carry signal only at
                # the terminal step. Force the gate True on
                # intermediate steps to avoid zeroing the easier
                # channel everywhere when the policy hasn't reached
                # the terminal step yet.
                if harder_idx in self._sparse_terminal_idx_set:
                    gate[:-1, :] = 1.0
                # ``buf_reward_vec`` is the canonical reward buffer; we
                # rebuild a writable copy because the sentinel path
                # above may have produced a view.
                buf_reward_vec = np.array(buf_reward_vec, copy=True)
                buf_reward_vec[..., easier_idx] = (
                    buf_reward_vec[..., easier_idx] * gate
                )
                gated_fraction = float(1.0 - gate.mean())
                easier_name = REWARD_NAMES[easier_idx]
                harder_name = REWARD_NAMES[harder_idx]
                reward_condition_stats[
                    f"reward_condition/{easier_name}_gated_fraction"
                ] = gated_fraction
                reward_condition_stats[
                    f"reward_condition/{harder_name}_threshold_{op}"
                ] = float(thresh)

        # Keep the RAW per-channel buffer for per-channel logging
        # (cosine_sim / cost means) — the mult-gate below overwrites
        # the canonical buffer with the gated scalar, which must not
        # corrupt the measured-channel telemetry.
        buf_reward_vec_raw = np.array(buf_reward_vec, dtype=np.float32, copy=True)

        # RAW COSSIM THREAD (bridge-cse): apply the cosine_sim GUIDE CAP HERE
        # — into the GAE/reward buffer ONLY — so ``buf_reward_vec_raw`` above
        # retains the full-range cosine_sim for honest ``reward/cosine_sim_raw``
        # logging + the dynamic-sentinel EMA, while the reward the policy is
        # optimised against still sees the bounded ``min(cossim, C)`` climb
        # guide. (Previously the cap was baked in env._callback so the full
        # range never reached the worker.) Failed rows already carry the
        # dynamic-sentinel value in idx6, which is < C, so min() leaves them
        # unchanged. Cosine_sim is a non-symlog quality channel, untouched by
        # the cost transforms above.
        if self.raw_cossim_thread and self._cossim_guide_cap is not None:
            buf_reward_vec = np.array(buf_reward_vec, copy=True)
            buf_reward_vec[..., COSINE_SIM_IDX] = np.minimum(
                buf_reward_vec[..., COSINE_SIM_IDX],
                np.float32(self._cossim_guide_cap),
            )

        # (Removed) The second additive cost-symlog pass used to live here. It
        # double-symlog'd the canonical buffer's cost channels (the pass above
        # already symlog'd them into buf_reward_vec, captured into
        # buf_reward_vec_raw), crushing e.g. symlog(latency)~11.6 -> ~2.5 and
        # rendering the GAE/return cost term rank-blind; it also excluded only
        # cosine_sim/frob_residual, so it wrongly symlog'd the [0,1] bkstep_acc
        # quality channel. The single lambda-inside-symlog pass above
        # (symlog(lambda_inner*raw), masked to true cost channels) is now the
        # ONLY cost transform. ``self.additive_symlog_cost`` retained for
        # compatibility but no longer drives a transform here.

        # Fix 1 (v2): ADDITIVE bounded-negative FAILED penalty. Stamped
        # HERE -- AFTER both symlog passes and the reward-condition gates,
        # BEFORE GAE -- so the value is exact (no later transform reshapes
        # it). Failed rows had every channel zeroed in the sentinel block
        # above; we now write the penalty onto the highest-|weight|
        # NON-symlog channel (bkstep_acc / cosine_sim) so the GAE
        # weighted-sum scalarisation recovers EXACTLY self.failed_penalty
        # (all other channels of a failed row are 0). With
        # discount=gae-lambda=1.0 (Monte-Carlo) a terminal-step penalty
        # flows back unchanged as the episode return. Included normally in
        # the advantage z-score + loss => failed rows get a negative
        # advantage (repulsive gradient), never a zero-gradient no-op.
        #
        # DYNAMIC SENTINEL supersede (bridge-cse): when the dynamic sentinel
        # actually stamped failed rows this rollout (``_dyn_scalar`` finite),
        # those rows already carry the bounded ``mu-K*sigma`` reward vector —
        # overwriting idx with the flat failed_penalty would DISCARD the
        # distribution-aware signal. Skip the stamp then. (It still runs on
        # the first, un-warmed rollout where failed rows were zeroed.)
        _dyn_applied = self.dynamic_sentinel and np.isfinite(_dyn_scalar)
        if (
            self.reward_mode == 'additive'
            and buf_failed.any()
            and not _dyn_applied
        ):
            _w = self.reward_weights_np.astype(np.float32)
            _cand = _w.copy()
            _cand[~_NO_SYMLOG_MASK_FP] = 0.0  # keep only non-symlog cols
            if np.any(_cand != 0.0):
                _pidx = int(np.argmax(np.abs(_cand)))
            else:
                _pidx = int(np.argmax(np.abs(_w)))
            _pw = float(_w[_pidx])
            if _pw == 0.0:
                _pw = 1.0
            buf_reward_vec = np.array(buf_reward_vec, copy=True)
            _fill = np.float32(self.failed_penalty / _pw)
            buf_reward_vec[..., _pidx] = np.where(
                buf_failed, _fill, buf_reward_vec[..., _pidx],
            )

        # Multiplicative cosine-gate reward (ALPHAGRAD_REWARD_MODE=mult).
        # Collapse the per-channel buffer into the fidelity-gated cheapness
        # scalar on the cosine_sim channel BEFORE GAE/scalarisation. The
        # one-hot[cosine_sim] weights set in __init__ then make every
        # downstream dot product (GAE scalarise, scalar value loss, the
        # milestone best/mean return) pick out exactly the gated scalar.
        # ``additive`` mode leaves the buffer untouched (byte-identical to
        # the legacy path).
        if self.reward_mode == "mult":
            _term_mask = buf_dones.astype(bool)              # (T, N)
            buf_reward_vec = self._apply_mult_reward_gate(
                buf_reward_vec,
                terminal_mask=_term_mask,
                failed_mask=buf_failed,
            )

        # GAE over the rollout. The critic learns PopArt-normalised
        # values, so symlog squashing inside GAE never applies — PopArt's
        # per-channel (mu, sigma) handles magnitude.
        _popart_log: dict = {}
        _gae = self._gae_no_symlog
        # rewards_b shape: (N, T, K). dones / discounts broadcast over K.
        rewards_b = jnp.asarray(np.transpose(buf_reward_vec, (1, 0, 2)))
        dones_b = jnp.asarray(buf_dones.T)             # (N, T)
        values_b = jnp.asarray(np.transpose(buf_values, (1, 0, 2)))  # (N, T, K)
        # The critic emits PopArt-NORMALISED values v_hat; GAE needs raw
        # values. De-normalise affinely with the CURRENT (pre-update)
        # stats — the frame the rollout head was trained in.
        _pa_mu = jnp.asarray(self.popart.mu, dtype=jnp.float32)
        _pa_sig = jnp.asarray(self.popart.sigma, dtype=jnp.float32)
        values_b = _pa_mu + _pa_sig * values_b
        bootstrap = (
            self.popart.mu + self.popart.sigma * np.asarray(bootstrap)
        ).astype(np.float32)
        next_values_b = jnp.concatenate(
            [values_b[:, 1:, :], jnp.asarray(bootstrap)[:, None, :]], axis=1,
        )  # (N, T, K)
        discounts_b = jnp.full_like(dones_b, self.discount)  # (N, T)
        _episodic_return, returns_b, advantages_b = _gae(
            rewards_b, dones_b, values_b, next_values_b, discounts_b,
            self.gae_lambda,
        )
        # returns_b / advantages_b shape: (N, T, K) in gdpo mode.
        # In scalar mode we still ran per-channel GAE above; collapse
        # to scalar via the priority-weighted dot product so the
        # downstream legacy path (global z-score, scalar value loss)
        # receives (N, T) tensors.
        if self.advantage_norm == "scalar":
            # ---- PopArt (van Hasselt 2016; multi-channel as in IMPALA).
            # DESIGN CHOICE: per-channel PopArt on the K return channels,
            # collapsed to the scalar advantage via reward_weights — NOT
            # scalar-space PopArt on the weighted return. The existing
            # scalar collapse is per-channel GAE followed by a weighted
            # dot product and the value head is already K-wide, so
            # per-channel stats slot into that contract without touching
            # the head shape, and the lambda weights keep expressing pure
            # PRIORITY over O(1)-normalised channels.
            weights_j = self.reward_weights
            returns_raw_b = returns_b                      # (N, T, K) raw
            adv_raw_np = np.asarray(advantages_b, dtype=np.float64)
            ret_raw_np = np.asarray(returns_raw_b, dtype=np.float64)
            # (1) EMA stats over the VALUE TARGETS (per-channel returns).
            # Envs with ANY failed/sentinel transition are excluded so the
            # failed-row machinery never interacts with the normaliser
            # (their returns carry the stamped penalty, not a measured
            # value). All-fail rollout => stats simply hold (quasi-static
            # anyway), advantages get the constant stamp below.
            _env_ok = ~buf_failed.any(axis=0)              # (N,)
            _upd_rows = ret_raw_np[_env_ok].reshape(-1, NUM_REWARDS)
            if _upd_rows.shape[0] > 0 and np.isfinite(_upd_rows).all():
                _o_mu, _o_sig, _n_mu, _n_sig = self.popart.update(_upd_rows)
                # (2) The "Art": output-preserving rescale of the value
                # head's final linear layer under mu->mu', sigma->sigma'
                # — the raw predictions are numerically unchanged, so the
                # stats step is invisible to the critic's gradients. (The
                # Adam moments of that layer keep their old scale — the
                # standard PopArt simplification; beta is quasi-static so
                # per-step rescales are ~1.)
                from alphagrad.approx.common.popart import (
                    popart_rescale_mlp_head,
                )
                self.agent = eqx.tree_at(
                    lambda a: a.value_head,
                    self.agent,
                    popart_rescale_mlp_head(
                        self.agent.value_head, _o_mu, _o_sig, _n_mu, _n_sig,
                    ),
                )
            _pa_mu = jnp.asarray(self.popart.mu, dtype=jnp.float32)
            _pa_sig = jnp.asarray(self.popart.sigma, dtype=jnp.float32)
            # (3) NORMALISED value targets for the critic loss (per
            # channel, staged as (B, K) below)…
            returns_b = (returns_raw_b - _pa_mu) / _pa_sig  # (N, T, K)
            # …and NORMALISED per-channel advantages. GAE constructs
            # estim_return = advantage + value_raw identically, so
            # A_k/sigma_k == (G_k - mu_k)/sigma_k - v_hat_k exactly.
            # sigma >= sigma_min => this can NEVER amplify noise.
            advantages_b = jnp.sum(
                (advantages_b / _pa_sig) * weights_j, axis=-1,
            )  # (N, T) scalar advantage
            # Telemetry: per-channel mu/sigma + explained variance over
            # the non-failed rows. GAE constructs estim_return =
            # advantage + value_raw identically, so A_k/sigma_k is the
            # normalised residual. The HEADLINE metric is the POOLED EV
            # (all active channels concatenated in normalised space,
            # standard multi-task form): a channel whose per-episode
            # returns went homogeneous (Var(G)~0 — exactly the
            # convergence regime this fix targets) contributes ~nothing
            # to either variance instead of blowing the mean up with a
            # -1e2 spike, as the per-channel 1 - Var(A_k)/Var(G_k)
            # otherwise does. Per-channel EVs stay logged as auxiliaries
            # (variance-floored in normalised space).
            _row_ok = ~buf_failed.T                        # (N, T)
            _act_idx = np.where(self.reward_weights_np != 0.0)[0]
            _mu64 = self.popart.mu.astype(np.float64)
            _sig64 = self.popart.sigma.astype(np.float64)
            _t_pool, _r_pool = [], []
            for _k in range(NUM_REWARDS):
                _name = REWARD_NAMES[_k]
                _popart_log[f"popart/mu_{_name}"] = float(self.popart.mu[_k])
                _popart_log[f"popart/sigma_{_name}"] = float(
                    self.popart.sigma[_k]
                )
                if _k in _act_idx and _row_ok.any():
                    _tn = (
                        ret_raw_np[_row_ok][:, _k] - _mu64[_k]
                    ) / _sig64[_k]
                    _rn = adv_raw_np[_row_ok][:, _k] / _sig64[_k]
                    _t_pool.append(_tn)
                    _r_pool.append(_rn)
                    _popart_log[f"popart/ev_{_name}"] = float(
                        1.0 - np.var(_rn) / max(np.var(_tn), 1e-6)
                    )
            if _t_pool:
                _tp = np.concatenate(_t_pool)
                _rp = np.concatenate(_r_pool)
                _popart_log["explained_variance"] = float(
                    1.0 - np.var(_rp) / max(np.var(_tp), 1e-8)
                )
        elif self.advantage_norm == "scalar":
            weights_j = self.reward_weights
            returns_b = jnp.sum(returns_b * weights_j, axis=-1)      # (N, T)
            advantages_b = jnp.sum(advantages_b * weights_j, axis=-1)  # (N, T)

        # V2 telemetry: any non-finite count here means the scrubs above
        # missed a contamination path — the update would nan-skip.
        _nf_ret = int((~np.isfinite(np.asarray(returns_b))).sum())
        _nf_adv = int((~np.isfinite(np.asarray(advantages_b))).sum())
        if _nf_ret or _nf_adv:
            print(
                f"[ppo_ray][v2-diag] POST-GAE non-finite: "
                f"returns={_nf_ret} advantages={_nf_adv} "
                f"(update will nan-skip these minibatches)."
            )

        # Advantage normalisation. Under PopArt the per-channel
        # (G_k - mu_k)/sigma_k IS the normalisation, so no rollout-wide
        # z-score is layered on top (see below).
        if self.advantage_norm == "scalar":
            # PURE-POPART ADVANTAGE (bridge-cse, USER-DIRECTED). The advantage
            # entering the update is EXACTLY the per-channel PopArt-normalised
            # residual collapsed by the priority weights,
            #     A = sum_k (A_k / sigma_k) * w_k
            # constructed above (line ~3234). PopArt's slow-EMA per-channel
            # (G_k - mu_k)/sigma_k IS the normalisation — it is already O(1)
            # and finite (sigma has a hard floor), so we do NOT apply any
            # additional rollout-wide z-score, scale-only divisor, std-floor,
            # hard clip, OR failed-row advantage stamp on top. Layering a
            # second (rollout-batch) normaliser over PopArt is DOUBLE
            # NORMALISATION: it re-couples the advantage to per-batch
            # statistics (the exact all-fail-zeroing / homogeneous-inflation
            # artifact PopArt was chosen to avoid) and throws away PopArt's
            # cross-rollout scale. KL early-stop (target_kl, below) remains the
            # blow-up guard instead of the clip.
            #
            # RETIRED here (were the double-normalisation): the SCALE-ONLY
            # divide-by-max(std, ADV_STD_FLOOR), the ADV_CLIP hard clip, and the
            # constant -1.0 failed-row advantage overwrite. All three are
            # skipped when ALPHAGRAD_POPART_PURE_ADV=1 (default). Failed rows
            # are already truly-neutral in advantage under the neutral-mu
            # sentinel (their per-channel return == popart.mu => A_k == 0 on
            # every channel), so no stamp is needed. Set
            # ALPHAGRAD_POPART_PURE_ADV=0 to restore the legacy
            # scale/clip/stamp path (kept below for revert / A-B).
            _pure_adv = (
                os.environ.get("ALPHAGRAD_POPART_PURE_ADV", "1") == "1"
            )
            if not _pure_adv:
                # LEGACY (retired) double-normalisation path. Env knobs:
                #   ALPHAGRAD_ADV_STD_FLOOR (default 0.5), ALPHAGRAD_ADV_CLIP
                #   (default 8.0). Set ADV_CLIP=0 to disable the bound.
                _adv_std_floor = float(
                    os.environ.get("ALPHAGRAD_ADV_STD_FLOOR", "0.5")
                )
                _adv_clip = float(os.environ.get("ALPHAGRAD_ADV_CLIP", "8.0"))
                _adv_np_pre = np.asarray(advantages_b, dtype=np.float64)
                _adv_std_pre = float(_adv_np_pre.std())
                _adv_absmax_pre = float(np.abs(_adv_np_pre).max())
                if _adv_clip > 0.0:
                    _adv_scale = jnp.maximum(
                        jnp.std(advantages_b), jnp.float32(_adv_std_floor)
                    )
                    advantages_b = advantages_b / _adv_scale     # scale-only
                    advantages_b = jnp.clip(
                        advantages_b, -_adv_clip, _adv_clip
                    )
                    _popart_log["popart/adv_std_pre_bound"] = _adv_std_pre
                    _popart_log["popart/adv_absmax_pre_bound"] = _adv_absmax_pre
                    _popart_log["popart/adv_scale"] = float(_adv_scale)
                # RETIRED failed-row advantage overwrite (constant -1.0),
                # guarded by the dynamic sentinel. Inert under neutral-mu
                # (failed rows already A==0). Only runs in the legacy path.
                if (
                    self.reward_mode == "additive"
                    and buf_failed.any()
                    and not (self.dynamic_sentinel and np.isfinite(_dyn_scalar))
                ):
                    advantages_b = jnp.where(
                        jnp.asarray(buf_failed.T),            # (N, T)
                        jnp.float32(-1.0),
                        advantages_b,
                    )
            _af = np.asarray(advantages_b, dtype=np.float64).reshape(-1)
            _popart_log["popart/adv_mean"] = float(_af.mean())
            _popart_log["popart/adv_std"] = float(_af.std())
            _popart_log["popart/adv_absmax"] = float(np.abs(_af).max())
            print(
                f"[ppo_ray][popart] ep={self._episode_counter} "
                f"ev={_popart_log.get('explained_variance', float('nan')):+.3f} "
                f"adv(mean/std/absmax)="
                f"{_af.mean():+.3f}/{_af.std():.3f}/{np.abs(_af).max():.3f} "
                + " ".join(
                    f"{REWARD_NAMES[k]}: mu={self.popart.mu[k]:+.3g} "
                    f"sig={self.popart.sigma[k]:.3g}"
                    for k in np.where(self.reward_weights_np != 0.0)[0]
                ),
                flush=True,
            )
        elif self.advantage_norm == "scalar":
            # Fix 1 (v2): z-score over ALL transitions (failed rows
            # INCLUDED). Failed rows now carry the bounded-negative
            # penalty (not a spurious 0), so including them in the
            # (mean, std) makes their post-norm advantage strongly
            # negative -> the policy is driven away from the failing
            # region. Valid rows keep their RELATIVE signal (advantage
            # is relative), so there is no dilution.
            adv_flat = advantages_b.reshape(-1)
            adv_mean = jnp.mean(adv_flat)
            adv_std = jnp.std(adv_flat) + 1e-8
            advantages_b = (advantages_b - adv_mean) / adv_std
            # V2 anti-hack fix 1: the z-score above maps a CONSTANT batch
            # (every row failed -> uniform -2 penalty) to advantage ~0,
            # deleting the repulsive gradient exactly in the all-fail
            # basin. Overwrite failed rows' advantage POST-normalisation
            # with a fixed on-scale negative so failed actions are always
            # pushed down -- including when the whole rollout failed.
            # DYNAMIC SENTINEL supersede (bridge-cse): skip the flat
            # post-z-score failed-row advantage override when the dynamic
            # sentinel carried the signal at the REWARD level this rollout —
            # the mu-K*sigma reward already produces a proportionate negative
            # advantage through GAE, and stamping a constant here would
            # flatten that distribution-aware gradient. Falls through on the
            # first, un-warmed rollout (failed rows zeroed) so the all-fail
            # basin is still handled.
            if (
                self.reward_mode == "additive"
                and self.failed_adv_stamp > 0.0
                and buf_failed.any()
                and not (self.dynamic_sentinel and np.isfinite(_dyn_scalar))
            ):
                advantages_b = jnp.where(
                    jnp.asarray(buf_failed.T),            # (N, T)
                    jnp.float32(-abs(self.failed_adv_stamp)),
                    advantages_b,
                )

        # Stage for the update. Flatten (T, N, ...) -> (N*T, ...) along the
        # env axis (each transition is independent for PPO). Returns and
        # advantages carry a trailing K axis in gdpo mode (per-channel
        # values flow into the per-mb gdpo normalisation inside the
        # loss); scalar mode collapsed them above and they're already
        # 1D here. Dict-driven so the micro path (per-substep sequences +
        # axis features + avail mask) and the legacy path share one
        # perm/truncate/reshape/shard pipeline.
        def _flat(buf, extra=()):
            arr = np.asarray(buf)
            axes = (1, 0) + tuple(range(2, arr.ndim))
            return jnp.asarray(
                arr.transpose(axes).reshape((N * T,) + tuple(extra))
            )

        parts: dict[str, tuple] = {}
        parts["tokens"] = (_flat(buf_tokens, (MAX_TOKENS,)), (MAX_TOKENS,))
        parts["eqn_ids"] = (_flat(buf_eqn_ids, (MAX_TOKENS,)), (MAX_TOKENS,))
        parts["actions"] = (_flat(buf_actions), ())
        _V = int(self.total_v)
        parts["avail"] = (_flat(buf_avail, (_V,)), (_V,))
        parts["op"] = (_flat(buf_op, (S,)), (S,))
        parts["i"] = (_flat(buf_i, (S,)), (S,))
        parts["j"] = (_flat(buf_j, (S,)), (S,))
        parts["exp"] = (_flat(buf_exp, (S, P)), (S, P))
        parts["f"] = (_flat(buf_f, (S,)), (S,))
        parts["kind"] = (_flat(buf_kind, (S,)), (S,))
        parts["q"] = (_flat(buf_q, (S,)), (S,))
        parts["axis_state"] = (
            _flat(buf_axis_state, (MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)),
            (MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM),
        )
        parts["axis_valid"] = (
            _flat(buf_axis_valid, (MAX_AXES_PER_VERTEX,)),
            (MAX_AXES_PER_VERTEX,),
        )
        parts["pair_valid"] = (
            _flat(buf_pair_valid,
                  (MAX_AXES_PER_VERTEX, MAX_AXES_PER_VERTEX)),
            (MAX_AXES_PER_VERTEX, MAX_AXES_PER_VERTEX),
        )
        parts["compress_valid"] = (
            _flat(buf_compress_valid, (MAX_AXES_PER_VERTEX,)),
            (MAX_AXES_PER_VERTEX,),
        )
        parts["log_probs"] = (_flat(buf_log_probs), ())
        # Fix 1: per-transition valid mask (1.0 = good measure, 0.0 =
        # failed/sentinel). Flows into every minibatch so the loss can drop
        # failed rows (see loss_fn's ``valid`` arg).
        parts["valid"] = (_flat((~buf_failed).astype(np.float32)), ())
        # Cost-head aux targets: the RAW MEASURED per-channel 4-tuple
        # (buf_reward_vec_measured_raw, (T,N,K)) + a terminal-and-non-failed
        # mask so the aux Huber loss only supervises real terminal
        # measurements. Flattened to (N*T, K) / (N*T,) like the other parts.
        parts["cost_target"] = (
            _flat(buf_reward_vec_measured_raw, (NUM_REWARDS,)), (NUM_REWARDS,),
        )
        parts["cost_valid"] = (
            _flat((buf_dones.astype(bool) & ~buf_failed).astype(np.float32)), (),
        )
        if self.advantage_norm == "gdpo":
            parts["returns"] = (
                returns_b.reshape(N * T, NUM_REWARDS), (NUM_REWARDS,),
            )
            parts["advantages"] = (
                advantages_b.reshape(N * T, NUM_REWARDS), (NUM_REWARDS,),
            )
        else:
            # PopArt (always on): PER-CHANNEL normalised value targets (B, K)
            # for the per-channel critic MSE; the advantage is already the
            # scalar weighted sum of normalised channels.
            parts["returns"] = (
                returns_b.reshape(N * T, NUM_REWARDS), (NUM_REWARDS,),
            )
            parts["advantages"] = (advantages_b.reshape(N * T), ())

        # Single epoch × `minibatches` mini-batches. We rotate the batch
        # split by a deterministic permutation so each episode's first
        # minibatch isn't always envs 0..k-1.
        perm_key = jrand.fold_in(key, int(self._episode_counter))
        perm = jrand.permutation(perm_key, N * T)
        parts = {k: (v[perm], ex) for k, (v, ex) in parts.items()}

        total = N * T
        mb_size = total // self.minibatches
        if mb_size == 0:
            mb_size = total

        # Guarantee that mb_size is a multiple of num_devices so we can always shard
        if mb_size % self.num_devices != 0:
            mb_size = max(self.num_devices, (mb_size // self.num_devices) * self.num_devices)

        mb_count = total // mb_size
        valid_total = mb_count * mb_size

        if valid_total < total:
            print(f"[ppo_ray] truncating batch from {total} to {valid_total} to ensure mb_size={mb_size} is a multiple of {self.num_devices}")
            parts = {k: (v[:valid_total], ex) for k, (v, ex) in parts.items()}

        # Reshape to ``(mb_count, mb_size, ...)`` so we can shard the
        # data axis (axis 1) across devices. Each minibatch index `i`
        # then yields a per-mb tensor that already lives in the
        # data-parallel layout — the loss vmap's leading axis is the
        # sharded one, and XLA distributes the work without a gather.
        mb = {}
        for _name, (_v, _ex) in parts.items():
            arr = _v.reshape((mb_count, mb_size, *_ex))
            mb[_name] = jax.device_put(arr, self.scan_data_sharding)

        agent = self.agent
        opt_state = self.opt_state

        last_aux = {}
        nan_skip_total = 0
        # Fix 3: per-episode approx_kl + grad_norm telemetry (previously
        # this collapse class was invisible in wandb). Accumulate over the
        # minibatches actually run.
        _kl_running = []          # JOINT KL (legacy sum-over-components)
        _kl_pc_running = []       # PER-COMPONENT KL (mean-over-components)
        _gradnorm_running = []
        _mb_done = 0
        # Fix 2: PPO KL early-stop. The update loop is a single pass over
        # ``mb_count`` minibatches; treat it as the ppo epoch and stop
        # taking further steps once the running-mean approx_kl crosses the
        # target, so a catastrophic ep49-type update can't land in full.
        # ALPHAGRAD_PPO_TARGET_KL=0 disables (revertible no-op).
        _target_kl = float(os.environ.get("ALPHAGRAD_PPO_TARGET_KL", "0.15"))
        # PER-COMPONENT KL EARLY-STOP (bridge-cse, USER-DIRECTED). The joint
        # approx_kl is a SUM over ~n_components action components, so it is
        # ~10-50x a single-action KL and trips target_kl=0.15 after ~1
        # minibatch (over-throttling). ALPHAGRAD_KL_PER_COMPONENT=1 (default)
        # switches the early-stop CHECK to the MEAN-per-active-component KL
        # (approx_kl_percomp) so target_kl is a proper standard-PPO
        # per-component threshold and more minibatches run. Both KLs are always
        # logged. Set 0 to revert to the legacy joint-sum check.
        _kl_per_component = (
            os.environ.get("ALPHAGRAD_KL_PER_COMPONENT", "1") == "1"
        )
        _kl_stopped = False

        # Fix 3: linear entropy-coef anneal from init → final over the run.
        # Recomputed EVERY episode and passed as a runtime arg into the jit'd
        # update step (NOT captured in the loss closure), so the decay
        # actually reaches the loss — the policy explores early then commits.
        _total_eps = max(int(getattr(self.args, "episodes", 1000)), 1)
        _progress = min(self._episode_counter / _total_eps, 1.0)
        self.entropy_coef = (
            self.entropy_coef_init
            + (self.entropy_coef_final - self.entropy_coef_init) * _progress
        )
        _ent_coef_j = jnp.asarray(self.entropy_coef, dtype=jnp.float32)
        for i in range(mb_count):
            batch = (
                mb["tokens"][i], mb["eqn_ids"][i], mb["avail"][i],
                mb["actions"][i],
                mb["op"][i], mb["i"][i], mb["j"][i], mb["exp"][i],
                mb["f"][i], mb["kind"][i], mb["q"][i],
                mb["axis_state"][i], mb["axis_valid"][i],
                mb["pair_valid"][i], mb["compress_valid"][i],
                mb["log_probs"][i], mb["returns"][i], mb["advantages"][i],
                mb["valid"][i],
                mb["cost_target"][i], mb["cost_valid"][i],
            )
            key, mb_key = jrand.split(key)
            agent, opt_state, aux = self._update_step(
                agent, opt_state, batch,
                self._current_op_mask_j, self._current_factor_mask_j,
                self._current_quant_mask_j, mb_key,
                _ent_coef_j,
            )
            nan_skip_total += int(aux.pop("nan_skip", 0))
            _gn = aux.pop("grad_norm", None)
            _kl = aux.pop("approx_kl", None)
            _kl_pc = aux.pop("approx_kl_percomp", None)
            if _gn is not None and np.isfinite(float(_gn)):
                _gradnorm_running.append(float(_gn))
            if _kl is not None and np.isfinite(float(_kl)):
                _kl_running.append(float(_kl))
            if _kl_pc is not None and np.isfinite(float(_kl_pc)):
                _kl_pc_running.append(float(_kl_pc))
            last_aux = {k: float(v) for k, v in aux.items()}
            _mb_done += 1
            # Fix 2: early-stop the remaining minibatches once the running
            # mean approx_kl exceeds the target. The current minibatch's
            # step has already been applied (standard PPO checks AFTER the
            # step); we simply take no further steps this episode. The CHECK
            # KL is the per-component mean (default) or the legacy joint sum
            # (ALPHAGRAD_KL_PER_COMPONENT=0).
            if _kl_per_component and _kl_pc_running:
                _kl_check = _kl_pc
                _kl_check_running = _kl_pc_running
                _kl_label = "approx_kl_percomp"
            else:
                _kl_check = _kl
                _kl_check_running = _kl_running
                _kl_label = "approx_kl(joint)"
            if (
                _target_kl > 0.0
                and _kl_check is not None
                and np.isfinite(float(_kl_check))
                and float(np.mean(_kl_check_running)) > _target_kl
            ):
                _kl_stopped = True
                print(
                    f"[ppo_ray][kl-stop] ep={self._episode_counter} "
                    f"{_kl_label}(mean)={float(np.mean(_kl_check_running)):.4f} "
                    f"> target {_target_kl} after mb {_mb_done}/{mb_count} "
                    f"(joint_kl mean="
                    f"{float(np.mean(_kl_running)) if _kl_running else float('nan'):.3f}) "
                    f"-- stopping remaining updates.",
                    flush=True,
                )
                break

        self.agent = agent
        self.opt_state = opt_state
        self._episode_counter += 1

        # Best / mean return computed from the RAW per-channel buffer +
        # the (calibration-aware) reward weights — NOT from the
        # scalarised `rewards_b` which has been symlog'd inside GAE
        # (and which masks magnitude ordering for large rewards). The
        # pre-refactor calc at this site is why PPO's best stayed at
        # ep 0 in the wandb log run-vsh0bv2c — `jnp.sum(rewards_b,
        # axis=1).max()` was comparing symlog'd sums.
        from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE
        from alphagrad.approx.common.reward_scaling import (
            aggregate_per_channel_stats,
            REWARD_NAMES as _RS_REWARD_NAMES,
        )

        per_env_weighted_sum = (
            buf_reward_vec * self.reward_weights_np
        ).sum(axis=(0, 2))  # (N,) — raw weighted per-env return
        # Fix 1 (v2): mean_return now INCLUDES all envs. Failed envs carry
        # the bounded-negative penalty as their return (not a spurious 0),
        # so averaging them in is the honest metric -- a mean that DROPS
        # when the policy samples failing rules is exactly the signal we
        # want to watch (does it climb back out?). ``best_return`` is the
        # max over all envs (the penalty can only under-count it).
        _env_valid = ~buf_failed[-1, :]  # (N,) -- terminal step ok
        _n_valid_env = int(_env_valid.sum())
        episode_return = float(per_env_weighted_sum.mean())
        best_return = float(per_env_weighted_sum.max())
        last_aux_env_valid_frac = float(_n_valid_env) / max(int(N), 1)

        # Per-env action sequence — what landed in the env at each
        # step. For the simple policy each entry is just the vertex
        # index; for ``dynamic_substeps`` we record the full 6-tuple
        # ``(vertex, op, i, j, factor, quant)`` so the JSON dump can
        # reproduce the exact micro-action sequence that won each
        # channel. ``buf_*`` are shape ``(T, N)``; transpose so the
        # outer dimension is per-env (N) and the inner is per-step (T).
        per_env_actions: list[list]
        # Decode each env's full typed sub-episode sequence into
        # copy-pastable ``(vertex, [diag(...)/compress(...)/quant(...)])``
        # rows — same human-readable form cmorl/mogfn dump; the
        # best-sequences JSON writer's ``to_typed_records`` accepts this
        # 2-tuple string shape natively. ``buf_*`` are ``(T, N, S)``;
        # slice per env (column n) → ``(T, S)``.
        from alphagrad.approx.ppo import _action_to_pylist_dynamic
        per_env_actions = [
            _action_to_pylist_dynamic(
                buf_actions[:, n],
                buf_op[:, n],
                buf_i[:, n],
                buf_j[:, n],
                buf_f[:, n],
                buf_kind[:, n],
                buf_q[:, n],
                int(self.max_substeps),
            )
            for n in range(N)
        ]

        # Per-channel best / mean / overall — same shape as MuZero's
        # `mu0_ray_worker` returns so the driver can use the shared
        # logging helpers.
        # Per-channel telemetry uses the RAW measured buffer + the user's
        # full weight vector. In mult mode the canonical ``buf_reward_vec``
        # has been collapsed to the gated scalar on the cosine channel and
        # ``reward_weights_np`` is one-hot, so feeding those here would
        # zero out cost-channel reporting and overstate cosine. ``_raw``
        # and ``_logging_weights_np`` are byte-identical to the canonical
        # ones in additive mode.
        # ANTI-DEGEN best_overall exclusion. The failed-measure zeroing
        # above turns a sentinel/mem-gate/shape-storm terminal into an
        # ALL-ZERO raw row, which ``aggregate_per_channel_stats`` would
        # read as "free perfect" (cost=0) and could crown as best_overall
        # — corrupting the milestone best. Likewise a genuinely degenerate
        # terminal (cosine_sim < anti_degen_tau) must never be best. Stamp
        # the cost channels of these envs' TERMINAL rows in the
        # aggregation copy with the SENTINEL value so the built-in
        # ``filter_sentinel_mask`` drops them from every per-channel mean,
        # the per-env sum, and the best_overall argmax. (The canonical
        # ``buf_reward_vec`` already carries -P on these via the gate, so
        # ``best_return`` from the gated buffer is correct independently.)
        buf_raw_for_agg = buf_reward_vec_raw.astype(np.float32)
        # SENTINEL AGG EXCLUSION (bridge-cse). A failed/sentinel row must be
        # EXCLUDED from every terminal aggregate regardless of reward_mode.
        # Previously this exclusion was gated ``anti_degen and mult``, so in
        # additive mode the dynamic-sentinel substitute (raw quality mu ~0.65
        # on the non-symlog channels + symlog-compressed ~0 cost) survived
        # ``filter_sentinel_mask`` (mu != -1e10) and was wrongly RETAINED in
        # ``terminal_means`` / ``per_reward_means`` / ``per_env_tot_for_best``
        # / ``best_overall`` — reading as an average-quality, near-free-cost
        # row that both spiked the means and could crown best_overall. Stamp
        # the cost channels of EVERY failed terminal row with SENTINEL so
        # filter_sentinel_mask drops them. Flag-guarded (default on),
        # revertible via ALPHAGRAD_EXCLUDE_FAILED_AGG=0.
        _term = buf_dones.astype(bool)                        # (T, N)
        _exclude_failed_agg = (
            os.environ.get("ALPHAGRAD_EXCLUDE_FAILED_AGG", "1") == "1"
        )
        _failed_terminal = buf_failed & _term if _exclude_failed_agg else (
            np.zeros_like(_term)
        )
        if self.anti_degen and self.reward_mode == "mult":
            _degen_terminal = (
                (buf_reward_vec_raw[..., COSINE_SIM_IDX]
                 < np.float32(self.anti_degen_tau))
                | buf_failed
            ) & _term                                        # (T, N)
            _degen_terminal = _degen_terminal | _failed_terminal
        else:
            _degen_terminal = _failed_terminal
        if _degen_terminal.any():
            buf_raw_for_agg = buf_raw_for_agg.copy()
            _cost_idx_arr = np.array(_cost_idx, dtype=np.int64)
            # Mark every cost channel of the degenerate terminal rows
            # as SENTINEL so filter_sentinel_mask excludes them.
            for _ci in _cost_idx_arr:
                buf_raw_for_agg[..., int(_ci)] = np.where(
                    _degen_terminal,
                    np.float32(SENTINEL_REWARD_VALUE),
                    buf_raw_for_agg[..., int(_ci)],
                )
            print(
                f"[ppo_ray] anti-degen: excluded "
                f"{int(_degen_terminal.sum())} degenerate/failed terminal "
                f"env(s) from best_overall."
            )

        ch_stats = aggregate_per_channel_stats(
            buf_raw_for_agg,
            self._logging_weights_np.astype(np.float32),
            sentinel=SENTINEL_REWARD_VALUE,
            action_seq=per_env_actions,
            dones_mask=buf_dones.astype(bool),
        )

        # V2 anti-hack telemetry: per-channel decomposition of the scalar
        # the GAE actually consumes (weight x transformed reward) at VALID
        # terminal rows -- quality (cosine guide + bkstep) vs cost
        # (everything else). This is the direct "is quality dominating the
        # scalar?" probe; quality_term should sit ~8x cost_term at a good
        # operating point (bkstep ~0.65 + guide ~0.15 vs cost ~-0.09).
        _bkstep_idx = REWARD_INDEX["bkstep_acc"]
        _valid_term = buf_dones.astype(bool) & ~buf_failed  # (T, N)
        if _valid_term.any():
            _tvec = buf_reward_vec_raw[_valid_term]        # (M, K)
            _contrib = (_tvec * self.reward_weights_np).mean(axis=0)  # (K,)
            _decomp_quality = float(
                _contrib[COSINE_SIM_IDX] + _contrib[_bkstep_idx]
            )
            _decomp_cost = float(_contrib.sum()) - _decomp_quality
        else:
            _decomp_quality = 0.0
            _decomp_cost = 0.0
        last_aux["decomp/quality_term"] = _decomp_quality
        last_aux["decomp/cost_term"] = _decomp_cost

        # TRULY-NEUTRAL SENTINEL (bridge-cse). Update the EMA of the REAL
        # per-env post-transform WEIGHTED scalar return from this rollout's
        # VALID terminal rows. ``buf_reward_vec_raw`` already carries the
        # cost symlog+cap transform; apply the cossim guide cap here so the
        # tracked scalar matches EXACTLY what the GAE reward consumes for a
        # real row. NEXT rollout's failed rows are shifted to this mean so
        # they land on the running-average return => neutral advantage.
        # (One-rollout lag by construction, same as the per-channel EMA.)
        if self._sentinel_neutral_mu and _valid_term.any():
            _rvec = buf_reward_vec_raw[_valid_term].astype(np.float64).copy()
            if self.raw_cossim_thread and self._cossim_guide_cap is not None:
                _rvec[:, COSINE_SIM_IDX] = np.minimum(
                    _rvec[:, COSINE_SIM_IDX], float(self._cossim_guide_cap)
                )
            _real_scalars = _rvec @ self.reward_weights_np.astype(np.float64)
            _bm_s = float(_real_scalars.mean())
            _b_s = self._sent_ema_beta
            if self._sent_ema_scalar_n == 0:
                self._sent_ema_scalar = _bm_s
            else:
                self._sent_ema_scalar = (
                    (1.0 - _b_s) * self._sent_ema_scalar + _b_s * _bm_s
                )
            self._sent_ema_scalar_n += 1
        last_aux["sentinel/neutral_mu_scalar"] = float(self._sent_ema_scalar)

        # RAW full-range cosine_sim + bkstep logging (bridge-cse). Over VALID
        # terminal rows ONLY (failed/sentinel rows carry the mu-K*sigma
        # substitute or a zero, neither of which is a measured cossim). Since
        # the env now emits the UNCAPPED cossim in idx6 and the guide cap is
        # deferred into the GAE buffer, ``buf_reward_vec_raw[..., 6]`` is the
        # honest full-range [-1, 1] value (NOT clamped to [.., C]). bkstep_acc
        # (idx9) is already raw accuracy in [0, 1] — logged from the same raw
        # buffer so it is the real measured value, never the sentinel 0.
        if _valid_term.any():
            _rawcos = buf_reward_vec_raw[_valid_term][:, COSINE_SIM_IDX]
            _rawbk = buf_reward_vec_raw[_valid_term][:, _bkstep_idx]
            last_aux["reward/cosine_sim_raw"] = float(np.mean(_rawcos))
            last_aux["reward/cosine_sim_raw_max"] = float(np.max(_rawcos))
            last_aux["reward/cosine_sim_raw_min"] = float(np.min(_rawcos))
            last_aux["reward/bkstep_acc_raw"] = float(np.mean(_rawbk))
        else:
            last_aux["reward/cosine_sim_raw"] = float("nan")
            last_aux["reward/cosine_sim_raw_max"] = float("nan")
            last_aux["reward/cosine_sim_raw_min"] = float("nan")
            last_aux["reward/bkstep_acc_raw"] = float("nan")
        # Dynamic-sentinel diagnostics: the collapsed mu-K*sigma scalar a
        # sentinel row now yields this rollout (NaN if none fired / dynamic
        # off / EMA un-warmed), plus the per-channel EMA mu the substitution
        # is built from.
        last_aux["sentinel/dynamic_value_scalar"] = float(
            getattr(self, "_last_dyn_sentinel_scalar", float("nan"))
        )
        if self.dynamic_sentinel and self._sent_ema_n > 0:
            last_aux["sentinel/ema_mu_cosine_sim"] = float(
                self._sent_ema_mean[COSINE_SIM_IDX]
            )
            last_aux["sentinel/ema_mu_bkstep_acc"] = float(
                self._sent_ema_mean[_bkstep_idx]
            )
        # Fix 3: surface approx_kl + grad_norm each episode (mean + max over
        # the minibatches run).
        last_aux["ppo/approx_kl"] = (
            float(np.mean(_kl_running)) if _kl_running else float("nan")
        )
        last_aux["ppo/approx_kl_max"] = (
            float(np.max(_kl_running)) if _kl_running else float("nan")
        )
        # PER-COMPONENT KL telemetry (bridge-cse) — the quantity the early-stop
        # checks by default; compare against ppo/approx_kl (joint sum) to see
        # the ~n_components scaling and the extra minibatches it unlocks.
        last_aux["ppo/approx_kl_percomp"] = (
            float(np.mean(_kl_pc_running)) if _kl_pc_running else float("nan")
        )
        last_aux["ppo/approx_kl_percomp_max"] = (
            float(np.max(_kl_pc_running)) if _kl_pc_running else float("nan")
        )
        last_aux["ppo/grad_norm"] = (
            float(np.mean(_gradnorm_running))
            if _gradnorm_running else float("nan")
        )
        last_aux["ppo/grad_norm_max"] = (
            float(np.max(_gradnorm_running))
            if _gradnorm_running else float("nan")
        )
        last_aux["ppo/minibatches_run"] = int(_mb_done)
        last_aux["ppo/kl_early_stopped"] = int(_kl_stopped)

        last_aux.update({
            "episode_return_mean": episode_return,
            "episode_return_max": best_return,
            "mean_return": episode_return,
            "best_return": best_return,
            "rollout_length": T,
            "num_envs": N,
            "nan_skip_count": int(nan_skip_total),
            "train_step": int(self._episode_counter),
            # Fix 1 telemetry: how much of the batch survived measurement.
            "sentinel/failed_transitions": int(buf_failed.sum()),
            "sentinel/failed_fraction": float(
                buf_failed.sum() / max(int(T * N), 1)
            ),
            "sentinel/valid_env_fraction": last_aux_env_valid_frac,
            # Fix 3 telemetry: the LIVE annealed entropy coefficient.
            "entropy_coef": float(self.entropy_coef),
            # Live-edge mask oracle: seconds spent probing this episode, and
            # how much of the micro-action space survived the exact masks.
            "mask/probe_seconds": float(self._mask_probe_seconds),
            "mask/compress_axes_mean": float(np.mean(buf_compress_valid)),
            "mask/diag_pairs_mean": float(np.mean(buf_pair_valid)),
        })
        # PopArt telemetry (per-channel mu/sigma, explained_variance —
        # the critic-health metric — and the advantage-scale probes).
        last_aux.update(_popart_log)
        # Per-channel raw-reward means — same key namespace MuZero
        # uses. Already part of `per_reward_means` in `ch_stats` but
        # re-emit as flat `reward_mean/...` keys for the driver's
        # back-compat wandb log dict.
        for name, val in ch_stats["per_reward_means"].items():
            last_aux[f"reward_mean/{name}"] = float(val)
        # Per-channel terminal-step means — the honest per-episode signal
        # for the sparse-terminal quality channels (cossim / frob), which
        # the per-step `reward_mean/<name>` dilutes with intermediate
        # zero-reward steps. Restored to match the pre-merge PPO behaviour
        # (and the C-MORL `reward_mean/<name>_terminal` definition):
        # mean over the dones-masked terminal transitions.
        for name, val in ch_stats.get("terminal_means", {}).items():
            last_aux[f"reward_mean/{name}_terminal"] = float(val)
        # LOG ALL MEASUREMENT CHANNELS (bridge-cse). Emit the terminal-step
        # mean (over VALID = non-failed terminal rows) of the RAW measured
        # value for EVERY one of the 10 REWARD_NAMES — including the channels
        # with ZERO reward weight (flops, max_io_sum, bytes_accessed,
        # peak_memory, frob_residual, xla_peak_memory) that never enter the
        # reward and so were invisible in wandb. Namespace ``measure/<name>``
        # (raw units, cost channels negated per env convention). Distinct from
        # the reward-space ``reward_mean/<name>`` keys — no double-count.
        _valid_term_meas = buf_dones.astype(bool) & ~buf_failed   # (T, N)
        if _valid_term_meas.any():
            _meas = buf_reward_vec_measured_raw[_valid_term_meas]  # (M, K)
            _meas_mean = _meas.mean(axis=0)
            for _mi, _mname in enumerate(REWARD_NAMES):
                last_aux[f"measure/{_mname}"] = float(_meas_mean[_mi])
        last_aux["per_reward_means"] = ch_stats["per_reward_means"]
        last_aux["per_reward_means_terminal"] = ch_stats.get("terminal_means", {})
        last_aux["best_per_reward"] = ch_stats["best_per_reward"]
        last_aux["best_overall_rewards"] = ch_stats["best_overall_rewards"]
        last_aux["best_overall_weighted"] = ch_stats["best_overall_weighted"]
        # ``best_seq`` is the canonical key the driver's
        # ``update_running_bests`` reads to populate
        # ``state["best_global_seq"]``. Pass through the overall-best
        # env's action sequence so the JSON dump can reconstruct
        # ``(a_i, b_i, c_i, r_i)`` AND the sequence that produced it.
        last_aux["best_seq"] = ch_stats.get("best_overall_seq", [])
        last_aux["best_overall_env"] = ch_stats.get("best_overall_env", -1)
        last_aux.update(reward_condition_stats)

        # Phase 3 follow-up: periodic CPU-pool recycle (matches the
        # MuZero side at mu0_ray_worker.py:1041-1050). Without this,
        # each ``CpuApproximationActor`` accumulates the ~9 MB-per-call
        # ``cost_analysis()`` C++ residual; on the 188 GB pgi15 nodes
        # this OOM-killed the PPO run at episode ~605 with 16 workers
        # holding 10 GB each. The recycle is a kill + respawn, takes a
        # few seconds, and resets each actor's RSS back to baseline.
        if self._cpu_pool is not None:
            last_aux.update(
                {f"pool/{k}": v for k, v in self._cpu_pool.stats().items()}
            )
            # Aggregate the per-actor tokenization-truncation telemetry.
            # All values are PER-EPISODE (the pool's
            # ``consume_*`` reset is invoked once per rollout). Mirrors
            # the ``nan_skip_count`` pattern. Best-effort — a Ray hiccup
            # here returns zeros rather than crashing the training step.
            #   * ``truncated_count``       — how often the clip bit this episode
            #   * ``overflow_sum_this_ep``  — total tokens of information thrown
            #                                 away this episode (sum of
            #                                 ``raw_len - MAX_TOKENS``)
            #   * ``mean_overflow_per_trunc`` — overflow_sum / count, or 0 if no
            #                                 truncations
            #   * ``max_observed_len``      — largest raw jaxpr length this episode;
            #                                 ``MAX_TOKENS`` is the live clip
            try:
                trunc = self._cpu_pool.fetch_tokenization_truncation_stats()
                count = int(trunc.get("count", 0))
                overflow_sum = int(trunc.get("overflow_sum", 0))
                max_len = int(trunc.get("max_observed_len", 0))
            except Exception:
                count = 0
                overflow_sum = 0
                max_len = 0
            last_aux["tokenization/truncated_count"] = count
            last_aux["tokenization/overflow_sum_this_ep"] = overflow_sum
            last_aux["tokenization/mean_overflow_per_trunc"] = (
                float(overflow_sum / count) if count > 0 else 0.0
            )
            last_aux["tokenization/max_observed_len"] = max_len
            recycle_every = getattr(self, "_cpu_pool_recycle_every", 0)
            if recycle_every > 0:
                # Cascading recycle: kill ONE actor every
                # ``recycle_every / N`` episodes so the full pool
                # rotates over ``recycle_every`` total episodes — same
                # effective per-actor lifetime as the old all-at-once
                # ``recycle()``, but the memory spike is smeared.
                n_pool = max(self._cpu_pool.size(), 1)
                interval = max(1, recycle_every // n_pool)
                if (
                    self._episode_counter > 0
                    and self._episode_counter % interval == 0
                ):
                    new_size = self._cpu_pool.recycle_one()
                    last_aux["pool/recycled_at_ep"] = self._episode_counter
                    last_aux["pool/size_after_recycle"] = new_size

        # Periodic checkpoint. The SIGTERM handler also fires a save
        # on slurm timeout, but periodic writes bound the worst-case
        # progress lost on a non-clean crash (OOM, segfault, network).
        if (
            self._checkpoint_path
            and self._checkpoint_every > 0
            and self._episode_counter > 0
            and self._episode_counter % self._checkpoint_every == 0
        ):
            self._save_checkpoint_safe()
            last_aux["checkpoint/saved_at_ep"] = self._episode_counter

        # Trainer-side leak sample. The actor-side profiler counts
        # `evaluate` calls; here we count episodes (each ep == one
        # full rollout + minibatch update). If RSS growth is local to
        # the trainer process, this log will show it; if the leak is
        # in the worker actors (where the per-call jit().compile()
        # lives), trainer RSS should stay roughly flat after JIT-warm.
        if self._leak_profile is not None:
            # Force-record on every episode (not gated by the every-N
            # counter) by sampling at a counter the profiler always
            # accepts — use the episode counter * `every` so the
            # ``n_call % every == 0`` gate always fires.
            self._leak_profile.record_call(
                self._episode_counter * self._leak_profile._every
            )
            last_aux["leak/trainer_rss_mb"] = float(
                self._leak_profile.rss_bytes() / 1024 / 1024
            )

        return last_aux

    # ==================================================================
    # SAMPLER HELPERS — rollout+MEASUREMENT (samplers, GPU1-3 + the measure
    # pool) decoupled from the learner. The synchronous
    # run_rollout_and_train path above is the only training entry point.
    #
    #   get_weights / set_weights : broadcast the learner's policy to the
    #       samplers (eqx array-leaf pytree; ray.put'd by the driver).
    #   collect_traj              : ONE rollout + measurement, returns the
    #       trajectory (numpy, Ray-serialisable) + telemetry. No
    #       gradient update.
    #
    # NOTE: the learner half of the old async pipeline (``train_on_trajs``)
    # was P3O-only and has been removed along with the P3O path. These
    # helpers have no in-tree training consumer left.
    # ==================================================================
    def get_weights(self):
        """Return the policy's array-leaf pytree (host numpy) for broadcast.

        Only the inexact-array leaves of ``self.agent`` (the eqx.Module) —
        the static structure is reconstructed via ``set_weights`` against the
        actor's own template. Numpy so it round-trips the Ray object store
        without device placement surprises."""
        _arrays = eqx.filter(self.agent, eqx.is_inexact_array)
        return jax.tree_util.tree_map(lambda x: np.asarray(x), _arrays)

    def set_compile_actor(self, compile_actor) -> bool:
        """STAGE-2: register the dedicated compile-actor handle. When set, the
        sampler PRE-compiles its terminal orders on this actor (into the shared
        cluster cache) before the measure fan-out — so measures exec-only."""
        self._compile_actor = compile_actor
        return True

    def set_weights(self, weights_np, learner_step: int = 0) -> int:
        """Overwrite ``self.agent``'s array leaves from a host-numpy pytree
        (mirrors ``get_weights``). Re-replicates to the actor's sharding so
        the JIT'd act-step sees device arrays. ``learner_step`` is the LEARNER's
        update count at broadcast time — stored so collect_traj can stamp it and
        the learner can compute a MEANINGFUL actor↔learner staleness (learner
        updates elapsed since this policy was synced). Returns the local version."""
        _w = jax.tree_util.tree_map(lambda x: jnp.asarray(x), weights_np)
        self.agent = eqx.combine(_w, self.agent)
        self.agent = self._replicate(self.agent)
        self._policy_version = int(getattr(self, "_policy_version", 0)) + 1
        self._synced_learner_step = int(learner_step)
        return self._policy_version

    def _collect_rollout_buffers(self, key):
        """Run ONE rollout+measurement pass with the CURRENT ``self.agent``
        and return the raw ``buf_*`` numpy arrays. This is the rollout half of
        ``run_rollout_and_train`` (act-step loop + measure fan-out), factored
        out for the async samplers. The synchronous path keeps its own inline
        copy (untouched)."""
        env_states = jax.vmap(lambda _: self.env.reset())(
            jnp.arange(self.num_envs),
        )
        self.env_states = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, self.data_sharding)
            if eqx.is_array(x) else x,
            env_states,
        )
        self._reset_mask_oracles()
        T = int(self.rollout_length)
        N = int(self.num_envs)
        S = int(self.max_substeps)
        P = int(MAX_PRIMES)
        buf_tokens = np.zeros((T, N, MAX_TOKENS), dtype=np.int32)
        buf_eqn_ids = np.zeros((T, N, MAX_TOKENS), dtype=np.int32)
        buf_actions = np.zeros((T, N), dtype=np.int32)
        buf_op = np.zeros((T, N, S), dtype=np.int32)
        buf_i = np.zeros((T, N, S), dtype=np.int32)
        buf_j = np.zeros((T, N, S), dtype=np.int32)
        buf_exp = np.zeros((T, N, S, P), dtype=np.int32)
        buf_f = np.zeros((T, N, S), dtype=np.int32)
        buf_kind = np.zeros((T, N, S), dtype=np.int32)
        buf_q = np.zeros((T, N, S), dtype=np.int32)
        buf_axis_state = np.zeros(
            (T, N, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), dtype=np.int32)
        buf_axis_valid = np.zeros((T, N, MAX_AXES_PER_VERTEX), dtype=np.float32)
        buf_pair_valid = np.zeros(
            (T, N, MAX_AXES_PER_VERTEX, MAX_AXES_PER_VERTEX), dtype=np.float32)
        buf_compress_valid = np.zeros(
            (T, N, MAX_AXES_PER_VERTEX), dtype=np.float32)
        buf_avail = np.zeros((T, N, int(self.total_v)), dtype=np.float32)
        buf_log_probs = np.zeros((T, N), dtype=np.float32)
        buf_values = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        buf_reward_vec = np.zeros((T, N, NUM_REWARDS), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)
        buf_sentinel = np.zeros((T, N), dtype=bool)
        state = self.env_states
        for t in range(T):
            key, sub = jrand.split(key)
            avail = self._vertex_avail(state)
            pv_np, cv_np = self._live_masks(np.asarray(avail))
            pv_j = jnp.asarray(pv_np)
            cv_j = jnp.asarray(cv_np)
            (
                actions, op_a, i_a, j_a, exp_a, f_a, kind_a, q_a,
                ax_st_a, ax_va_a, pv_a, cv_a, log_probs, values,
                partial, order, specs, step,
            ) = self._act_step(
                self.agent, state, avail,
                self._current_op_mask_j, self._current_factor_mask_j,
                self._current_quant_mask_j, sub,
                self._substep_budget_j, pv_j, cv_j,
            )
            pre_tokens = np.asarray(state.tokens)
            pre_eqn_ids = np.asarray(state.eqn_ids)
            order_np = np.asarray(order)
            specs_np = np.asarray(specs)
            step_np = np.asarray(step)
            _is_terminal_step = t == T - 1
            _local_tokenize = (
                not _is_terminal_step and self._local_tokenize_enabled
                and self._terminal_rewards_only
            )
            if (_is_terminal_step
                    and getattr(self.args, "measure_queue", False)
                    and self._terminal_rewards_only):
                # STAGE-2: PRE-COMPILE the terminal orders on the dedicated
                # compile-actor BEFORE the measure fan-out, so the measure
                # actors get a coordinator HIT (~10ms load) instead of the
                # ~1.9s inline jacve compile. Only the DISTINCT orders (one per
                # env) need compiling — n_points measure the SAME order per env.
                # We WAIT for the precompiles so the fan-out is a pure exec.
                _ca = getattr(self, "_compile_actor", None)
                if _ca is not None:
                    import ray as _ray
                    _futs = [
                        _ca.precompile.remote(
                            order_np[_e], specs_np[_e], int(step_np[_e]))
                        for _e in range(order_np.shape[0])
                    ]
                    try:
                        _ok = _ray.get(_futs)
                        self._precompile_ok = int(sum(bool(x) for x in _ok))
                        self._precompile_tot = int(len(_ok))
                    except Exception:
                        self._precompile_ok = 0
                        self._precompile_tot = int(order_np.shape[0])
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_terminal_queue(order_np, specs_np, step_np))
            elif _local_tokenize:
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._tokenize_local(order_np, specs_np, step_np))
            else:
                tokens_np, eqn_ids_np, reward_np, sentinel_mask = (
                    self._fan_out_tokenize(order_np, specs_np, step_np))
            buf_sentinel[t] = sentinel_mask
            tokens_j = jax.device_put(
                jnp.asarray(tokens_np, dtype=jnp.int32), self.data_sharding)
            eqn_ids_j = jax.device_put(
                jnp.asarray(eqn_ids_np, dtype=jnp.int32), self.data_sharding)
            reward_j = jax.device_put(
                jnp.asarray(reward_np, dtype=jnp.float32), self.data_sharding)
            state = self._assemble(state, partial, tokens_j, eqn_ids_j, reward_j)
            buf_tokens[t] = pre_tokens
            buf_eqn_ids[t] = pre_eqn_ids
            buf_avail[t] = np.asarray(avail)
            buf_exp[t] = np.asarray(exp_a)
            buf_kind[t] = np.asarray(kind_a)
            buf_axis_state[t] = np.asarray(ax_st_a)
            buf_axis_valid[t] = np.asarray(ax_va_a)
            buf_pair_valid[t] = np.asarray(pv_a)
            buf_compress_valid[t] = np.asarray(cv_a)
            buf_actions[t] = np.asarray(actions)
            buf_op[t] = np.asarray(op_a)
            buf_i[t] = np.asarray(i_a)
            buf_j[t] = np.asarray(j_a)
            buf_f[t] = np.asarray(f_a)
            buf_q[t] = np.asarray(q_a)
            buf_log_probs[t] = np.asarray(log_probs)
            buf_values[t] = np.asarray(values)
            buf_reward_vec[t] = reward_np
            buf_dones[t] = np.asarray(state.terminated).astype(np.float32)
        # Non-finite reward scrub (mirror the sync path's guard).
        buf_reward_vec = np.nan_to_num(
            buf_reward_vec, nan=0.0, posinf=0.0, neginf=0.0)
        buf_log_probs = np.nan_to_num(
            buf_log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        return dict(
            tokens=buf_tokens, eqn_ids=buf_eqn_ids, avail=buf_avail,
            actions=buf_actions, op=buf_op, i=buf_i, j=buf_j, exp=buf_exp,
            f=buf_f, kind=buf_kind, q=buf_q, axis_state=buf_axis_state,
            axis_valid=buf_axis_valid, pair_valid=buf_pair_valid,
            compress_valid=buf_compress_valid, log_probs=buf_log_probs,
            reward_vec=buf_reward_vec, dones=buf_dones, sentinel=buf_sentinel,
        )

    def collect_traj(self, rng_seed: int):
        """SAMPLER entry: one rollout+measurement with the last-synced policy.
        Returns a Ray-serialisable ``(traj_np, telemetry)`` where ``traj_np``
        is the (N,T,...) replay trajectory (numpy) ready for the buffer, and
        telemetry carries the policy version + measured mean return so the
        driver can watch staleness + progress. No gradient step here."""
        if not hasattr(self, "_act_step"):
            self._act_step = self._make_act_step_fn_micro()
            self._assemble = self._make_assemble_fn()
        key = jrand.PRNGKey(int(rng_seed))
        bufs = self._collect_rollout_buffers(key)
        _tp3 = lambda a: np.transpose(a, (1, 0) + tuple(range(2, a.ndim)))
        traj_np = {
            "tokens": _tp3(bufs["tokens"]), "eqn_ids": _tp3(bufs["eqn_ids"]),
            "avail": _tp3(bufs["avail"]), "v_act": bufs["actions"].T,
            "op": _tp3(bufs["op"]), "i": _tp3(bufs["i"]), "j": _tp3(bufs["j"]),
            "exp": _tp3(bufs["exp"]), "f": _tp3(bufs["f"]),
            "kind": _tp3(bufs["kind"]), "q": _tp3(bufs["q"]),
            "axis_state": _tp3(bufs["axis_state"]),
            "axis_valid": _tp3(bufs["axis_valid"]),
            "pair_valid": _tp3(bufs["pair_valid"]),
            "compress_valid": _tp3(bufs["compress_valid"]),
            "old_lp": bufs["log_probs"].T,
            "reward_vec": _tp3(bufs["reward_vec"]),
            "dones": bufs["dones"].T,
        }
        _wsum = (bufs["reward_vec"] * self.reward_weights_np).sum(axis=(0, 2))
        telemetry = {
            "policy_version": int(getattr(self, "_policy_version", 0)),
            # The learner update-count this policy was last synced to — the
            # learner computes staleness = (its current update count) - this.
            "synced_learner_step": int(getattr(self, "_synced_learner_step", 0)),
            "mean_return": float(np.mean(_wsum)),
            "sentinel_frac": float(np.mean(bufs["sentinel"])),
            "precompile_ok": int(getattr(self, "_precompile_ok", 0)),
            "precompile_tot": int(getattr(self, "_precompile_tot", 0)),
        }
        # STAGE-2 compile-cache HIT/MISS rate from the CLUSTER coordinator (the
        # global executable cache all measure actors consult): the measure
        # actors' inline compiles are MISSES; the compile-actor's precompiles
        # turn subsequent measures into HITS. Reported so we can see the
        # cluster hit-rate rise under --async-compile-split.
        try:
            import ray as _ray
            _coord = _ray.get_actor("alphagrad_compile_cache_coordinator")
            _cs = _ray.get(_coord.stats.remote())
            telemetry["cache_hits"] = int(_cs.get("hits", 0))
            telemetry["cache_misses"] = int(_cs.get("misses", 0))
            telemetry["cache_hit_rate"] = float(_cs.get("hit_rate", 0.0))
            telemetry["cache_size"] = int(_cs.get("size", 0))
        except Exception:
            pass
        return traj_np, telemetry

    # ------------------------------------------------------------------
    # Reward-vector statistics + weight-setting helpers (driver-callable).
    # ------------------------------------------------------------------
    def reward_vec_means(self, rng_seed: int, num_rollouts: int) -> dict:
        """Run `num_rollouts` zero-pref rollouts of the un-trained
        agent and return per-channel calibration statistics.

        Returns a dict (shape ``(NUM_REWARDS,)`` per entry):
          * ``mean``      — per-channel mean of valid transitions
          * ``median``    — per-channel median (raw)
          * ``q25``,``q75`` — quartiles (raw); ``iqr = q75 - q25``
          * ``median_symlog``, ``q25_symlog``, ``q75_symlog`` — same in
            symlog space (so IQR-on-symlog can be computed driver-side)
          * ``count``     — number of valid samples per channel
            (scalar — same across channels because the mask is row-wise)

        Sentinel transitions (any cost channel == SENTINEL_REWARD_VALUE)
        are filtered out — see Phase 4b note in the docstring of the
        MuZero sibling for why naive averaging breaks under sentinel
        contamination.

        Reward-vector statistics helper — a driver-side caller can pick
        a statistic (mean_abs / iqr / std) from this dict to derive a
        per-channel weight rescale.
        """
        from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE
        from alphagrad.approx.common.reward_scaling import (
            NUM_REWARDS as _NUM_REWARDS_RS,
            filter_sentinel_mask,
            symlog_np,
        )
        if not hasattr(self, "_act_step"):
            self._act_step = self._make_act_step_fn_micro()
            self._assemble = self._make_assemble_fn()
            self._update_step = self._make_update_step_micro()

        T = int(self.num_valid)
        N = int(self.num_envs)
        # Accumulate the FLAT valid samples per channel; small (≈ T·N
        # ·num_rollouts × K × 4 bytes ≈ 250 KB at defaults) so a single
        # contiguous buffer keeps the median/quartile computation simple.
        all_rows: list[np.ndarray] = []
        for r in range(int(num_rollouts)):
            key = jrand.PRNGKey(int(rng_seed) + r)
            key, reset_key = jrand.split(key)
            env_states = jax.vmap(lambda _: self.env.reset())(jnp.arange(N))
            state = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding)
                if eqx.is_array(x) else x,
                env_states,
            )
            self._reset_mask_oracles()
            buf_reward_vec = np.zeros((T, N, _NUM_REWARDS_RS), dtype=np.float32)
            for t in range(T):
                key, sub = jrand.split(key)
                avail = self._vertex_avail(state)
                pv_np, cv_np = self._live_masks(np.asarray(avail))
                act_out = self._act_step(
                    self.agent, state, avail,
                    self._current_op_mask_j, self._current_factor_mask_j,
                    self._current_quant_mask_j, sub,
                    self._substep_budget_j,
                    jnp.asarray(pv_np), jnp.asarray(cv_np),
                )
                partial = act_out[-4]
                order = act_out[-3]
                specs = act_out[-2]
                step = act_out[-1]
                tokens_np, eqn_ids_np, reward_np, _sentinel_mask = (
                    self._fan_out_tokenize(
                        np.asarray(order),
                        np.asarray(specs),
                        np.asarray(step),
                    )
                )
                tokens_j = jax.device_put(
                    jnp.asarray(tokens_np, dtype=jnp.int32), self.data_sharding,
                )
                eqn_ids_j = jax.device_put(
                    jnp.asarray(eqn_ids_np, dtype=jnp.int32), self.data_sharding,
                )
                reward_j = jax.device_put(
                    jnp.asarray(reward_np, dtype=jnp.float32), self.data_sharding,
                )
                state = self._assemble(state, partial, tokens_j, eqn_ids_j, reward_j)
                buf_reward_vec[t] = reward_np
            valid = filter_sentinel_mask(buf_reward_vec, SENTINEL_REWARD_VALUE)
            if valid.any():
                all_rows.append(buf_reward_vec[valid])  # (n_valid, K)
        if not all_rows:
            zeros = np.zeros((_NUM_REWARDS_RS,), dtype=np.float32)
            return {
                "mean": zeros, "median": zeros, "q25": zeros, "q75": zeros,
                "median_symlog": zeros, "q25_symlog": zeros, "q75_symlog": zeros,
                "count": 0,
            }
        samples = np.concatenate(all_rows, axis=0).astype(np.float32)  # (M, K)
        samples_sl = symlog_np(samples).astype(np.float32)
        return {
            "mean": samples.mean(axis=0).astype(np.float32),
            "median": np.median(samples, axis=0).astype(np.float32),
            "q25": np.quantile(samples, 0.25, axis=0).astype(np.float32),
            "q75": np.quantile(samples, 0.75, axis=0).astype(np.float32),
            "median_symlog": np.median(samples_sl, axis=0).astype(np.float32),
            "q25_symlog": np.quantile(samples_sl, 0.25, axis=0).astype(np.float32),
            "q75_symlog": np.quantile(samples_sl, 0.75, axis=0).astype(np.float32),
            "count": int(samples.shape[0]),
        }

    def set_reward_weights(self, weights_np) -> None:
        """Weight-setting helper — replace the scalarising weight
        vector in-place from a driver-side caller.
        """
        weights_np = np.asarray(weights_np, dtype=np.float32)
        assert weights_np.shape == self.reward_weights_np.shape, (
            f"weights shape mismatch: got {weights_np.shape}, "
            f"expected {self.reward_weights_np.shape}"
        )
        self.reward_weights_np = weights_np
        self.reward_weights = jnp.asarray(weights_np, dtype=jnp.float32)
        # Keep the GDPO channel mask in sync — calibration can change
        # which channels carry non-zero weight if a previously-disabled
        # channel gets a non-trivial scaling factor.
        self._channel_mask_j = jnp.asarray(
            (self.reward_weights_np != 0.0).astype(np.float32)
        )

    def _ensure_in_proc_server(self):
        """Lazily build the process-local ``CpuApproximationServer``.

        Reuses the already-constructed env so the in-proc path runs the
        exact same ``env._callback`` the CPU pool's actors run — the
        tokenizer output is therefore byte-identical to the pool path
        for the (deterministic) non-terminal steps.
        """
        if self._in_proc_server is None:
            from alphagrad.approx.cpu_approx_worker import (
                CpuApproximationServer,
            )
            self._in_proc_server = CpuApproximationServer.from_env(self.env)
            # Pin the non-terminal tokenize to a CPU device. The
            # tokenizer is pure symbolic/graph work (graphax
            # _build_graph) — it does NOT need the GPU, and running it
            # on the PPOActor's GPU (device 0) steals memory from
            # ``jit_update_step`` and OOMs the PPO weight update. The CPU
            # pool actors run on CPU devices too, so this also keeps the
            # local path byte-identical to the pool path.
            try:
                self._tokenize_device = jax.devices("cpu")[0]
            except Exception:
                self._tokenize_device = None
        return self._in_proc_server

    def _tokenize_local(self, order_np, specs_np, step_np):
        """Compute non-terminal state-tokens IN-PROCESS (no Ray).

        For the NON-TERMINAL steps under ``terminal_rewards_only`` the
        reward is zeroed (``env._callback`` early-returns before any
        jacve compile / cost_analysis) and the tokens are a pure
        deterministic fn of ``(order, specs, step)``. So we run the same
        ``env._callback`` locally via the in-proc server, skipping the
        ~2 s/step Ray round-trip (~99% of which is Ray overhead).

        Returns the same ``(tokens, eqn_ids, rewards, sentinel_mask)``
        contract as ``_fan_out_tokenize`` — ``rewards`` is all-zero and
        ``sentinel_mask`` is all-False (the local path never times out;
        a callback exception still raises, surfacing the bug rather than
        masking it on the cheap deterministic path).
        """
        server = self._ensure_in_proc_server()
        N = order_np.shape[0]
        dev = getattr(self, "_tokenize_device", None)
        if dev is not None:
            with jax.default_device(dev):
                out = [
                    server.evaluate(order_np[i], specs_np[i], int(step_np[i]))
                    for i in range(N)
                ]
        else:
            out = [
                server.evaluate(order_np[i], specs_np[i], int(step_np[i]))
                for i in range(N)
            ]
        tokens = np.stack([r[0] for r in out])
        eqn_ids = np.stack([r[1] for r in out])
        rewards = np.zeros((N, NUM_REWARDS), dtype=np.float32)
        sentinel_mask = np.zeros((N,), dtype=bool)
        return tokens, eqn_ids, rewards, sentinel_mask

    def _fan_out_tokenize(self, order_np, specs_np, step_np):
        """Ship the (order, specs, step) triple for each env to a CPU
        actor and collect (tokens, eqn_ids, reward) back.

        Returns ``(tokens, eqn_ids, rewards, sentinel_mask)`` where
        ``sentinel_mask`` is ``(N,) bool`` — True for env slots whose
        callback timed out / errored. The naive in-process and
        unpooled-ray paths never produce sentinels (they raise or
        block instead). Only the pooled path can mark a slot as
        sentinel.
        """
        N = order_np.shape[0]
        if self._cpu_pool is not None:
            # Phase 3 path: timeout-bounded fan-out via the shared
            # pool. Sentinels for timed-out slots flow back to the
            # rollout loop so it can mask the transition before GAE.
            tokens, eqn_ids, rewards, sentinel_mask = (
                self._cpu_pool.evaluate_batch(
                    order_np, specs_np, step_np, eval_samples=None, init=False,
                )
            )
            return tokens, eqn_ids, rewards, sentinel_mask

        if not self.cpu_workers:
            # In-process fallback — useful for local smoke tests where
            # we don't want to spin up a Ray cluster.
            server = self._ensure_in_proc_server()
            out = [
                server.evaluate(
                    order_np[i], specs_np[i], int(step_np[i]),
                )
                for i in range(N)
            ]
        else:
            import ray
            W = len(self.cpu_workers)
            futures = [
                self.cpu_workers[i % W].evaluate.remote(
                    order_np[i], specs_np[i], int(step_np[i]),
                )
                for i in range(N)
            ]
            out = ray.get(futures)
        tokens = np.stack([r[0] for r in out])
        eqn_ids = np.stack([r[1] for r in out])
        rewards = np.stack([r[2] for r in out])
        sentinel_mask = np.zeros((N,), dtype=bool)
        return tokens, eqn_ids, rewards, sentinel_mask

    def _fan_out_terminal_queue(self, order_np, specs_np, step_np):
        """Global measurement queue for the TERMINAL step.

        Instead of N env-tasks each measuring n_points data points
        SEQUENTIALLY on one actor's fixed core slice (which leaves
        gated/cheap envs' cores idle while a straggler grinds), decompose
        into ``N × n_points`` per-(env, data-point) tasks and dispatch
        them through ``ray.util.ActorPool``. Free actors pull the next
        task, so an env's points run in parallel and gated tasks
        immediately free workers for stragglers — full core utilisation.
        Each task measures ONE data point (R reps) on a fixed core slice,
        so the per-reading latency stays consistent. The driver then
        P60-aggregates each env's n_points reward vectors per channel.

        Returns the same ``(tokens, eqn_ids, rewards, sentinel_mask)``
        contract as ``_fan_out_tokenize``.
        """
        import ray
        from ray.util import ActorPool

        N = order_np.shape[0]
        n_points = max(int(getattr(self.args, "num_data_points", 5)), 1)
        pk = float(getattr(self.args, "percentile_keep", 0.60))

        actors = None
        if self._cpu_pool is not None:
            actors = self._cpu_pool.live_actors()
        elif self.cpu_workers:
            actors = list(self.cpu_workers)
        if not actors:
            # No Ray pool (in-proc tests) — fall back to the per-env path.
            return self._fan_out_tokenize(order_np, specs_np, step_np)

        # Build (env, point) task list, env-major so results[env*n_points+p].
        tasks = [
            (int(e), int(p), order_np[e], specs_np[e], int(step_np[e]))
            for e in range(N)
            for p in range(n_points)
        ]
        pool = ActorPool(actors)
        results = list(
            pool.map(
                lambda a, t: a.evaluate.remote(
                    t[2], t[3], t[4], point_idx=t[1],
                ),
                tasks,
            )
        )  # preserves task order

        tokens_out = np.zeros((N, MAX_TOKENS), dtype=np.int32)
        eqn_ids_out = np.zeros((N, MAX_TOKENS), dtype=np.int32)
        rewards_out = np.zeros((N, NUM_REWARDS), dtype=np.float32)
        for e in range(N):
            base = e * n_points
            # Per-point reward vectors for this env (each already P60'd
            # over its R reps inside _callback).
            pr = np.stack([results[base + p][2] for p in range(n_points)])  # (P, K)
            # Aggregate across points with the same "keep worst pk" rule
            # the single-call path uses. Cost channels are stored negated
            # (reward = -cost), so the worst-pk cost = -percentile(-reward).
            # cosine_sim (index COSINE) is "higher better" + weight 0 — use
            # the mean; everything else is a cost channel.
            agg = np.empty((NUM_REWARDS,), dtype=np.float32)
            _cos_idx = REWARD_INDEX["cosine_sim"]
            for k in range(NUM_REWARDS):
                if k == _cos_idx:
                    agg[k] = float(np.mean(pr[:, k]))
                else:
                    agg[k] = -float(np.percentile(-pr[:, k], pk * 100.0))
            rewards_out[e] = agg
            tokens_out[e] = results[base][0]
            eqn_ids_out[e] = results[base][1]
        sentinel_mask = np.zeros((N,), dtype=bool)
        return tokens_out, eqn_ids_out, rewards_out, sentinel_mask

    # ------------------------------------------------------------------
    # Lifecycle helpers used by the driver
    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return True

    def _save_checkpoint_safe(self) -> None:
        """Wrapped `save_state` that never propagates exceptions — used
        by both the periodic save in `run_rollout_and_train` and the
        SIGTERM handler. Logs to stdout instead of raising so a buggy
        serialisation doesn't take training down.
        """
        if not self._checkpoint_path:
            return
        try:
            from alphagrad.approx.common.checkpoint import save_state
            save_state(
                self._checkpoint_path,
                agent=self.agent,
                opt_state=self.opt_state,
                episode_counter=self._episode_counter,
                reward_weights=self.reward_weights_np,
            )
        except Exception as exc:
            print(f"[ppo_ray_worker] checkpoint save failed: {exc}")

    def init_server(
        self,
        cpu_workers: list,
        *,
        callback_timeout_s: float = 120.0,
        initial_timeout_s: float = 600.0,
        warm_after: int = 3,
        recycle_every: int = 50,
        cpu_actor_options: dict | None = None,
        starting_actor_id: int = 0,
    ) -> bool:
        """Construct the timeout-bounded `CpuApproxPool` from a list of
        Ray actor handles. Mirrors `mu0_ray_actors.SPMDActor.init_server`.

        After this call, `_fan_out_tokenize` dispatches through the
        pool — slow / hung actors are killed + respawned per the cold
        (`initial_timeout_s`) / warm (`callback_timeout_s`) budget,
        and timed-out env slots get sentinel rewards rather than
        blocking the rollout.
        """
        from alphagrad.approx.cpu_approx_pool import CpuApproxPool
        from alphagrad.approx.env import (
            MAX_TOKENS as _MAX_TOKENS,
            NUM_REWARDS as _NUM_REWARDS,
            REWARD_INDEX as _REWARD_INDEX,
        )

        self.cpu_workers = list(cpu_workers) if cpu_workers else []

        if cpu_actor_options is not None:
            from alphagrad.approx.cpu_approx_actors import CpuApproximationActor
            args_dict = vars(self.args)
            next_id = [int(starting_actor_id)]

            def _factory():
                next_id[0] += 1
                return CpuApproximationActor.options(**cpu_actor_options).remote(
                    args_dict, variant=None, actor_id=next_id[0],
                )
        else:
            _factory = None  # pool runs without respawn

        self._cpu_pool = CpuApproxPool(
            self.cpu_workers,
            timeout_s=float(callback_timeout_s),
            initial_timeout_s=float(initial_timeout_s),
            warm_after=int(warm_after),
            respawn_factory=_factory,
            max_tokens=int(_MAX_TOKENS),
            num_rewards=int(_NUM_REWARDS),
            cosine_sim_idx=int(_REWARD_INDEX["cosine_sim"]),
            frob_residual_idx=int(_REWARD_INDEX["frob_residual"]),
        )
        self._cpu_pool_recycle_every = int(recycle_every)
        # Build the process-local in-proc server up front when the
        # local-tokenize path is enabled, so the first non-terminal step
        # doesn't pay the env-reuse setup cost mid-rollout.
        if self._local_tokenize_enabled:
            self._ensure_in_proc_server()
        return True
