from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from itertools import zip_longest
from typing import Callable, Literal, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import jax.random as jrand
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

import numpy as np

from alphagrad.approx.common.relations import compute_eqn_ids_from_tokens
from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from graphax.jaxpr import get_vocab as _graphax_get_vocab
from jax_memory_monitor import ResourceMonitor

# Cache the graphax vocabulary used by `compute_eqn_ids_from_tokens` — the
# tokenizer always uses the same digit_base, so the vocab is constant and
# rebuilding it on every callback is pure overhead.
_TOKEN_VOCAB, _, _ = _graphax_get_vocab()

MAX_TOKENS = 4096
MAX_RULES_PER_VERTEX = 4
NUM_AXIS_PAIRS = 4

# Canonical 8-component reward vector layout. The env reports raw reward values
# in the convention "higher is better": every cost component is stored *negated*
# (so r = -cost), `cosine_sim` is in [0, 1] (1 = identical Jacobian), and
# `frob_residual` is stored as `-||J_e - J_a||_F / ||J_e||_F` so larger residuals
# correspond to lower reward. Downstream code can therefore treat all 8 entries
# uniformly as "reward to maximize".
#
# Compute family (indices 0..5):
#   0 muls_adds_fmas   — graphax `adds + muls + fmas` op count from VE.
#   1 flops            — XLA cost-analysis FLOPs of the compiled approx fn.
#   2 latency_ns       — wall-clock latency in ns (only populated when
#                        `EnvConfig.measure_latency` is True; else 0).
#   3 max_io_sum       — graphax `mem` accumulator = sum over Jacobian
#                        accumulations of `max(in_size, out_size, edge_out_size)
#                        * itemsize`. (This is the "sum of max-input/max-output
#                        sizes per Jacobian accumulation" metric in the spec.)
#   4 bytes_accessed   — XLA cost-analysis bytes-accessed of the approx fn.
#   5 peak_memory      — peak HBM bytes during a single execution of the approx
#                        fn, captured via `ResourceMonitor`.
# Quality family (indices 6..7):
#   6 cosine_sim       — cosine similarity between flattened approximated and
#                        exact Jacobians, averaged over the calibration samples.
#   7 frob_residual    — relative Frobenius residual ||J_e - J_a||_F / ||J_e||_F.
NUM_REWARDS = 8
REWARD_NAMES: tuple[str, ...] = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "cosine_sim",
    "frob_residual",
)
REWARD_INDEX = {name: i for i, name in enumerate(REWARD_NAMES)}
COMPUTE_REWARD_INDICES = tuple(range(0, 6))  # cost components
QUALITY_REWARD_INDICES = (6, 7)             # cosine, frobenius

# Sentinel reward returned when a sparsity_map matches an entry in the in-file
# blacklist (used during exploration to penalise pathological configurations).
_SENTINEL_BAD_REWARD = jnp.array(
    [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1.0, -1e10],
    dtype=jnp.float32,
)

# Axis pair index -> (base_idx1, base_idx2). base_idx1 picks output axis 0/1; base_idx2 picks input axis 0/1.
axis_pair_idx_to_base = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


class EnvState(NamedTuple):
    order: Array
    sparsity_specs: Array  # (N, MAX_RULES_PER_VERTEX, 3) int32; row [base_idx1, base_idx2, factor]; base_idx1 < 0 means slot unused
    tokens: Array
    eqn_ids: Array  # (MAX_TOKENS,) int32; per-token equation ID, -1 for non-eqn tokens
    step_count: Array
    max_steps: int
    reward: Array  # (NUM_REWARDS,) float32; see REWARD_NAMES for layout
    terminated: bool


class EnvOut(NamedTuple):
    state: EnvState
    reward: Array
    terminated: bool


class StepAction(NamedTuple):
    target_vertex: Array  # scalar int32
    rule_specs: Array  # (MAX_RULES_PER_VERTEX, 3) int32; row [base_idx1, base_idx2, factor]; base_idx1 < 0 marks unused


class EnvConfig(NamedTuple):
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    has_aux: bool
    sparse: bool
    # cmp_type / mem_type used to gate which compute/memory metric was measured.
    # The env now reports the full 8-component reward vector every step, so they
    # are kept only as *primary-metric hints* for legacy CLI/host-side reporting.
    # New callers should ignore them and pick the desired component from the
    # reward vector explicitly via REWARD_INDEX.
    cmp_type: Literal["graphax", "flops", "latency"]
    mem_type: Literal["graphax", "bytes_accessed", "peak_memory"]
    target_fun: Callable | None = None
    data_gen: Callable | None = None
    exec_on_gpu: bool = False
    # Latency requires running the compiled fn 10x per step, which roughly 10xs
    # rollout-to-reward time. Off by default; flip on when the latency component
    # of the reward is actually being weighted.
    measure_latency: bool = False
    # Skip the expensive jacve-compile/exec branch on every step EXCEPT the
    # terminal one. Tokens/eqn_ids are still produced (the agent needs them as
    # the next observation), but the reward vector is zero on intermediate
    # steps and fully populated only when the order is complete. This is the
    # paper-native form for AlphaZero / GDPO / GFlowNet and works fine for PPO
    # / MuZero (just yields a sparse reward signal).
    terminal_rewards_only: bool = False


def _get_partials(order, sparsity_specs, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order
    partial_specs = (
        sparsity_specs[:v_stop] if v_stop < len(sparsity_specs) else sparsity_specs
    )
    return partial_order, partial_specs


# Lookup row used to convert a legacy scalar sp_type ∈ {0..4} into a single-rule (MAX_RULES, 3) spec.
_LEGACY_SP_TO_RULE_ROW = jnp.array(
    [
        [-1, -1, 0],   # sp 0: unused
        [0, 0, -1],    # sp 1 -> (0,0)
        [0, 1, -1],    # sp 2 -> (0,1)
        [1, 0, -1],    # sp 3 -> (1,0)
        [1, 1, -1],    # sp 4 -> (1,1)
    ],
    dtype=jnp.int32,
)


def _legacy_sp_to_specs(sp_type: Array) -> Array:
    """Convert a scalar legacy sp_type ∈ {0..4} into (MAX_RULES_PER_VERTEX, 3) rule specs."""
    first = _LEGACY_SP_TO_RULE_ROW[sp_type]  # (3,)
    pad = jnp.tile(jnp.array([-1, -1, 0], dtype=jnp.int32), (MAX_RULES_PER_VERTEX - 1, 1))
    return jnp.concatenate([first[None, :], pad], axis=0)


@jax.jit
def cossim(target, preds):
    target = target / jnp.maximum(
        jnp.linalg.norm(target, keepdims=True), jnp.sqrt(1e-7)
    )
    preds = preds / jnp.maximum(jnp.linalg.norm(preds, keepdims=True), jnp.sqrt(1e-7))
    return jnp.sum(target * preds)


sp_type_to_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}

# things to try:
# error = MSE, cossim, Frobenius Norm
# other = {log, no log} x {div, no div}


def _flatten_jacobians(jac):
    """Concatenate all leaves of a (possibly nested) jacobian pytree to a flat 1-d array."""
    leaves = jax.tree_util.tree_leaves(jac)
    if not leaves:
        return None
    flats = [jnp.ravel(l) for l in leaves]
    return jnp.concatenate(flats)


def _quality_metrics(jac_exact, jac_approx):
    """`(cosine_sim, relative_frobenius)` of `jac_approx` against `jac_exact`.

    Returns the trivial `(1.0, 0.0)` (perfect agreement) when either side has
    no leaves, mismatched shapes, or zero size, mirroring the original `error`
    fallback so a degenerate plan can't poison downstream normalisation.
    """
    flat_exact = _flatten_jacobians(jac_exact)
    flat_approx = _flatten_jacobians(jac_approx)
    if flat_exact is None or flat_approx is None:
        return jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)
    if flat_approx.shape != flat_exact.shape or flat_approx.size == 0:
        return jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)
    cos = cossim(flat_exact, flat_approx)
    exact_norm = jnp.linalg.norm(flat_exact)
    resid_norm = jnp.linalg.norm(flat_exact - flat_approx)
    rel_frob = resid_norm / jnp.maximum(exact_norm, jnp.sqrt(1e-7))
    return cos, rel_frob


def _aggregate_samples(values, want_top_quartile: bool):
    """Reduce a list of per-sample scalars to a single jnp scalar.

    With ≥8 samples and `want_top_quartile`, takes the top-quartile mean
    (matching legacy behaviour for latency); otherwise falls back to a plain
    mean. Handles the empty-list case by returning `0.0`.
    """
    if not values:
        return jnp.array(0.0, dtype=jnp.float32)
    stack = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in values])
    if want_top_quartile and stack.shape[0] >= 8:
        return stack.sort()[6:8].mean()
    return stack.mean()


def _callback(
    config: EnvConfig,
    args,
    consts,
    order,
    sparsity_specs,
    stop,
    *eval_samples,
    init: bool = False,
):
    """Stage A reward harness: returns `(tokens, rewards)` where `rewards` is
    the canonical `(NUM_REWARDS,)` float32 vector documented at the top of this
    file. Every component is computed every (non-init) call, except `latency`
    which is gated behind `config.measure_latency`. When
    `config.terminal_rewards_only` is on, intermediate steps return tokens
    only — every reward component is zeroed so the heavy jacve compile/exec
    is skipped entirely until the elimination order is complete.
    """
    partial_order, partial_specs = _get_partials(order, sparsity_specs, stop)
    is_terminal = int(stop) >= len(order)

    o_list = [int(x) for x in partial_order.tolist()]
    specs_list = partial_specs.tolist()  # list of MAX_RULES x 3 lists

    sparsity_map = []
    for v_idx, v in enumerate(o_list):
        eqn = config.jaxpr.eqns[v - 1]
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue

        out_len = len(eqn.outvars[0].aval.shape)
        rules: list[tuple[int, int, int]] = []
        used_axes: set[int] = set()
        for slot in range(MAX_RULES_PER_VERTEX):
            row = specs_list[v_idx][slot]
            bi1 = int(row[0])
            bi2 = int(row[1])
            factor = int(row[2])
            if bi1 < 0:
                break  # stop / unused slot terminates the sequence
            idx1 = bi1
            idx2 = out_len + bi2
            # Skip rules that would reuse an axis (graphax filters them anyway, drop here for clarity)
            if idx1 in used_axes or idx2 in used_axes or idx1 == idx2:
                continue
            used_axes.add(idx1)
            used_axes.add(idx2)
            rules.append((idx1, idx2, factor))
        if rules:
            sparsity_map.append((v, tuple(rules)))

    blacklist = [
        # [(1, ((1, 2, -1),)), (7, ((1, 3, -1),)), (4, ((1, 2, -1),)), (2, ((0, 3, -1),))],
        # ...
    ]

    if sparsity_map != [] and any(
        all(
            a is not None and b is not None and a == b
            for a, b in zip_longest(sparsity_map, x)
        )
        for x in blacklist
    ):
        print("skipping blacklisted")
        return (
            jnp.zeros(MAX_TOKENS, dtype=jnp.int32),
            jnp.full(MAX_TOKENS, -1, dtype=jnp.int32),
            _SENTINEL_BAD_REWARD,
        )

    ve = extract_jaxpr(
        config.jaxpr,
        config.argnums,
        o_list,
        config.sparse,
        args,
        consts,
        sparsity_map=sparsity_map,
    )
    tokens = ve.tokenized()[:MAX_TOKENS]
    tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

    # Compute per-token equation IDs once for the relational-bias encoder
    # (Stage B.1). Cheap (single Python scan over a length-≤4096 numpy array)
    # and adds (MAX_TOKENS,) int32 to EnvState.
    tokens_np = np.asarray(tokens)
    eqn_ids_np = compute_eqn_ids_from_tokens(tokens_np, _TOKEN_VOCAB)
    eqn_ids = jnp.asarray(eqn_ids_np, dtype=jnp.int32)

    if init:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    # Terminal-only fast path: every reward component is sparse — only the
    # final step (when the elimination order is complete) gets a non-zero
    # signal, so we skip the expensive jacve compile/exec on every prior
    # step. Cumsum-style returns (alpha0/mu0) and per-rollout aggregations
    # (gdpo) collapse to the terminal reward; gfn already reads only the
    # last step. PPO sees a sparse-reward MDP, which GAE handles natively.
    if config.terminal_rewards_only and not is_terminal:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Compute family — graphax counters (always) → muls_adds_fmas, max_io_sum.
    # ------------------------------------------------------------------
    _, aux = vertex_elimination_jaxpr(
        config.jaxpr,
        o_list,
        consts,
        *args,
        argnums=config.argnums,
        count_ops=True,
        sparse_representation=config.sparse,
        sparsity_map=sparsity_map,
    )
    muls_adds_fmas = float(aux["adds"] + aux["muls"] + aux["fmas"])
    max_io_sum = float(aux["mem"])

    # If no `target_fun` is supplied, we can't compile/execute. Skip every
    # execution-derived metric and return a partial reward vector.
    if config.target_fun is None:
        rewards = jnp.array(
            [
                -muls_adds_fmas, 0.0, 0.0, -max_io_sum,
                0.0, 0.0, 1.0, 0.0,
            ],
            dtype=jnp.float32,
        )
        return tokens, eqn_ids, rewards

    # ------------------------------------------------------------------
    # Compile both the approximated and exact jacobian functions once.
    # ------------------------------------------------------------------
    callback_device = None
    if config.exec_on_gpu:
        gpu_devices = jax.devices("gpu")
        if len(gpu_devices) >= 2:
            callback_device = gpu_devices[1]

    args_for_lower = (
        jax.device_put(args, callback_device)
        if (config.exec_on_gpu and callback_device is not None)
        else args
    )

    def compiled(sp_map=None):
        return (
            jax.jit(
                jacve(
                    config.target_fun,
                    o_list,
                    argnums=config.argnums,
                    has_aux=config.has_aux,
                    sparse_representation=config.sparse,
                    sparsity_map=sp_map,
                ),
                keep_unused=True,
            )
            .lower(*args_for_lower)
            .compile()
        )

    compiled_approx = compiled(sparsity_map)
    compiled_exact = compiled()

    # XLA cost analysis — flops + bytes accessed. Falls back to 0 when the
    # backend doesn't expose them (CPU sometimes returns an empty dict).
    cost_analysis = compiled_approx.cost_analysis() or {}
    flops = float(cost_analysis.get("flops", 0))
    bytes_accessed = float(cost_analysis.get("bytes accessed", 0))

    # ------------------------------------------------------------------
    # Execution loop — runs once for peak_memory + quality, or 10x when
    # `measure_latency` is on (the latency reading is noisy enough that the
    # top-quartile-mean smoothing from the original code is worth keeping).
    # ------------------------------------------------------------------
    n_samples = 10 if config.measure_latency else 1

    monitoring_devices: list = []
    for x in jax.tree_util.tree_leaves(args):
        if hasattr(x, "devices"):
            monitoring_devices.extend(list(x.devices()))
    unique_devices = list(set(monitoring_devices))
    if (
        config.exec_on_gpu
        and callback_device is not None
        and callback_device not in unique_devices
    ):
        unique_devices.append(callback_device)
    if not unique_devices:
        unique_devices = jax.local_devices()

    out_approxs: list = []
    out_exacts: list = []
    latency_samples: list[float] = []
    peak_mem_samples: list[float] = []

    for i in range(n_samples):
        if eval_samples:
            eval_args_i = [arg[i] for arg in eval_samples]
        else:
            eval_args_i = list(args)
        if config.exec_on_gpu and callback_device is not None:
            eval_args_i = [jax.device_put(d, callback_device) for d in eval_args_i]

        with ResourceMonitor(devices=unique_devices) as monitor:
            out_approx = compiled_approx(*eval_args_i)
        latency_s, peak_bytes = monitor.stats.values()
        latency_samples.append(float(latency_s) * 1e9)  # → ns
        peak_mem_samples.append(float(peak_bytes))

        out_exact = compiled_exact(*eval_args_i)
        out_approxs.append(out_approx)
        out_exacts.append(out_exact)

    latency_ns = (
        float(_aggregate_samples(latency_samples, want_top_quartile=True))
        if config.measure_latency
        else 0.0
    )
    peak_memory = float(max(peak_mem_samples)) if peak_mem_samples else 0.0

    # ------------------------------------------------------------------
    # Quality family — cosine similarity + relative Frobenius residual.
    # ------------------------------------------------------------------
    cosines: list = []
    frobs: list = []
    for out_approx, out_exact in zip(out_approxs, out_exacts):
        jac_approx = out_approx[1] if config.has_aux else out_approx
        jac_exact = out_exact[1] if config.has_aux else out_exact
        cos, rel_frob = _quality_metrics(jac_exact, jac_approx)
        cosines.append(cos)
        frobs.append(rel_frob)

    cosine_sim = float(_aggregate_samples(cosines, want_top_quartile=True))
    frob_residual = float(_aggregate_samples(frobs, want_top_quartile=True))

    rewards = jnp.array(
        [
            -muls_adds_fmas,
            -flops,
            -latency_ns,
            -max_io_sum,
            -bytes_accessed,
            -peak_memory,
            cosine_sim,
            -frob_residual,
        ],
        dtype=jnp.float32,
    )

    return tokens, eqn_ids, rewards


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class VertexEliminationEnv:
    config: EnvConfig
    args: tuple
    consts: tuple
    valid_vertices: tuple
    num_envs: int | None = None
    eval_args_samples: tuple | None = None

    def __init__(
        self,
        config: EnvConfig,
        args: Sequence,
        consts: Sequence,
        valid_vertices: tuple | None = None,
        num_envs: int | None = None,
        eval_args_samples: tuple | None = None,
    ):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "eval_args_samples", eval_args_samples)

        if num_envs is None:
            num_envs = jax.local_device_count()
        object.__setattr__(self, "num_envs", num_envs)

        if valid_vertices is None:
            _, _, _, vo_vertices = _build_graph(
                config.jaxpr, args, consts, config.argnums
            )
            valid = []
            for i, eqn in enumerate(config.jaxpr.eqns, 1):
                if eqn.outvars[0] not in config.jaxpr.outvars or i in vo_vertices:
                    valid.append(i)
            valid_vertices = tuple(valid)
        object.__setattr__(self, "valid_vertices", valid_vertices)

    @classmethod
    def from_jaxpr(
        cls,
        jaxpr: core.ClosedJaxpr,
        argnums=None,
        args=None,
        has_aux=False,
        sparse=False,
        num_envs=None,
        data_gen: Callable | None = None,
        target_fun: Callable | None = None,
        cmp_type: str = "flops",
        mem_type: str = "peak_memory",
        exec_on_gpu: bool = False,
        measure_latency: bool = False,
        terminal_rewards_only: bool = False,
    ):
        assert (argnums is None and args is None) or not (args is None or args is None)
        config = EnvConfig(
            jaxpr=jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.invars)))
            if argnums is None
            else tuple(argnums),
            has_aux=has_aux,
            sparse=sparse,
            cmp_type=cmp_type,
            mem_type=mem_type,
            target_fun=target_fun,
            data_gen=data_gen,
            exec_on_gpu=exec_on_gpu,
            measure_latency=measure_latency,
            terminal_rewards_only=terminal_rewards_only,
        )
        return cls(
            config,
            args=jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            num_envs=num_envs,
        )

    def tree_flatten(self):
        children = (self.args, self.consts, self.eval_args_samples)
        aux_data = (self.config, self.valid_vertices, self.num_envs)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts, eval_args_samples = children
        config, valid_vertices, num_envs = aux_data
        return cls(config, args, consts, valid_vertices, num_envs, eval_args_samples)

    def tokenize(self, init: bool = False):
        return partial(_callback, self.config, init=init)

    @property
    def _callback_shape(self):
        return (
            jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32),
            jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32),
            jax.ShapeDtypeStruct((NUM_REWARDS,), jnp.float32),
        )

    def reset(self, num_envs: int | None = None) -> EnvState:
        if num_envs is None:
            num_envs = getattr(self, "num_envs", None)

        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)
        initial_specs = jnp.full(
            (initial_order.shape[0], MAX_RULES_PER_VERTEX, 3),
            -1,
            dtype=jnp.int32,
        )
        initial_specs = initial_specs.at[..., 2].set(0)  # factor=0 default for unused rows

        tokens, eqn_ids, _ = io_callback(
            self.tokenize(init=True),
            self._callback_shape,
            self.args,
            self.consts,
            initial_order,
            initial_specs,
            0,
            *(self.eval_args_samples if self.eval_args_samples is not None else ()),
        )

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = jnp.zeros(NUM_REWARDS, dtype=jnp.float32)
        terminated = jnp.array(False, dtype=jnp.bool_)

        state = EnvState(
            order=initial_order,
            sparsity_specs=initial_specs,
            tokens=tokens,
            eqn_ids=eqn_ids,
            step_count=step_count,
            max_steps=max_steps,
            reward=reward,
            terminated=terminated,
        )

        if num_envs is not None and num_envs > 0:
            state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)), state
            )

        return state

    @jit
    def step(self, state: EnvState, action) -> EnvOut:
        # Action may be either a `StepAction` (multi-rule) or a legacy scalar int
        # encoded as `sp_type * MAX_TOKENS + target_vertex`.
        if isinstance(action, StepAction):
            target_vertex = jnp.asarray(action.target_vertex, dtype=jnp.int32)
            rule_specs = jnp.asarray(action.rule_specs, dtype=jnp.int32)
        else:
            action = jnp.asarray(action, dtype=jnp.int32)
            sp_type = action // MAX_TOKENS
            target_vertex = action % MAX_TOKENS
            rule_specs = _legacy_sp_to_specs(sp_type)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        curr_specs = state.sparsity_specs

        pos = jnp.argwhere(curr_order == target_vertex, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)

        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(target_vertex)
        new_specs = curr_specs[shifted.astype(jnp.int32)].at[idx].set(rule_specs)

        tokens, eqn_ids, reward = io_callback(
            self.tokenize(),
            self._callback_shape,
            self.args,
            self.consts,
            new_order,
            new_specs,
            new_step,
            *(self.eval_args_samples if self.eval_args_samples is not None else ()),
        )

        terminated = new_step >= state.max_steps

        new_state = EnvState(
            order=new_order,
            sparsity_specs=new_specs,
            tokens=tokens,
            eqn_ids=eqn_ids,
            step_count=new_step,
            max_steps=state.max_steps,
            reward=reward,
            terminated=terminated,
        )

        def _step_process(_):
            return EnvOut(new_state, reward, terminated)

        def _step_done(_):
            return EnvOut(
                state,
                jnp.zeros(NUM_REWARDS, jnp.float32),
                jnp.array(True, dtype=jnp.bool_),
            )

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)
