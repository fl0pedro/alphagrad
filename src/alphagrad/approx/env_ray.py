from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

from alphagrad.approx.common.relations import compute_eqn_ids_from_tokens
from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from graphax.jaxpr import get_vocab as _graphax_get_vocab
from graphax.sparse.micro_actions import (
    COMPRESS_KINDS, QUANT_DTYPES, Compress, Diag, Quant,
)
from jax_memory_monitor import ResourceMonitor as _RealResourceMonitor


class _NoopResourceMonitor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    @property
    def peak(self) -> int:
        return 0

    @property
    def duration(self) -> float:
        return 0.0

    @property
    def stats(self) -> dict:
        return {"time": 0.0, "memory": 0.0}


ResourceMonitor = (
    _NoopResourceMonitor
    if os.environ.get("ALPHAGRAD_DISABLE_RESOURCE_MONITOR", "0") == "1"
    else _RealResourceMonitor
)

import math as _math

_TOKEN_VOCAB, _, _ = _graphax_get_vocab()

MAX_TOKENS = 8192
MAX_RULES_PER_VERTEX = 16
NUM_AXIS_PAIRS = 4
MAX_AXES_PER_VERTEX = 8
AXIS_FEATURE_DIM = 4
_AXIS_FEAT_SIZE = 0
_AXIS_FEAT_IS_OUTPUT = 1
_AXIS_FEAT_IS_COMPRESSED = 2
_AXIS_FEAT_GROUP_ID = 3

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
COMPUTE_REWARD_INDICES = tuple(range(0, 6))
QUALITY_REWARD_INDICES = (6, 7)

_SENTINEL_BAD_REWARD = jnp.array(
    [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1.0, -1e10],
    dtype=jnp.float32,
)

axis_pair_idx_to_base = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
COMPRESS_SENTINEL = -2
QUANT_SENTINEL = -3


class EnvState(NamedTuple):
    order: Array
    sparsity_specs: Array
    tokens: Array
    eqn_ids: Array
    axis_state: Array
    axis_valid_mask: Array
    step_count: Array
    max_steps: int
    reward: Array
    terminated: bool


class EnvOut(NamedTuple):
    state: EnvState
    reward: Array
    terminated: bool


class StepAction(NamedTuple):
    target_vertex: Array
    rule_specs: Array


class EnvConfig(NamedTuple):
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    has_aux: bool
    sparse: bool
    cmp_type: Literal["graphax", "flops", "latency"]
    mem_type: Literal["graphax", "bytes_accessed", "peak_memory"]
    target_fun: Callable | None = None
    data_gen: Callable | None = None
    exec_on_gpu: bool = False
    measure_latency: bool = False
    terminal_rewards_only: bool = False
    remote_eval_fn: Callable | None = None


def _get_partials(order, sparsity_specs, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order
    partial_specs = (
        sparsity_specs[:v_stop] if v_stop < len(sparsity_specs) else sparsity_specs
    )
    return partial_order, partial_specs


def _apply_rules_to_axis_state(axis_state_v: Array, rule_specs: Array) -> Array:
    is_output = axis_state_v[:, _AXIS_FEAT_IS_OUTPUT]
    n_out = jnp.sum(is_output).astype(jnp.int32)
    init_gid = jnp.max(axis_state_v[:, _AXIS_FEAT_GROUP_ID]) + 1

    def _body(slot, carry):
        state, gid = carry
        row = rule_specs[slot]
        bi1 = row[0]
        bi2 = row[1]
        factor = row[2]

        is_diag = bi1 >= 0
        is_compress = bi1 == COMPRESS_SENTINEL

        diag_out_tok = jnp.clip(bi1, 0, axis_state_v.shape[0] - 1)
        diag_prim_tok = jnp.clip(n_out + bi2, 0, axis_state_v.shape[0] - 1)
        safe_factor = jnp.maximum(factor, 1)

        def _apply_diag(s):
            s = s.at[diag_out_tok, _AXIS_FEAT_GROUP_ID].set(gid)
            s = s.at[diag_prim_tok, _AXIS_FEAT_GROUP_ID].set(gid)
            s = s.at[diag_out_tok, _AXIS_FEAT_SIZE].set(
                jnp.maximum(s[diag_out_tok, _AXIS_FEAT_SIZE] // safe_factor, 1)
            )
            s = s.at[diag_prim_tok, _AXIS_FEAT_SIZE].set(
                jnp.maximum(s[diag_prim_tok, _AXIS_FEAT_SIZE] // safe_factor, 1)
            )
            return s

        comp_tok = jnp.clip(bi2, 0, axis_state_v.shape[0] - 1)

        def _apply_compress(s):
            s = s.at[comp_tok, _AXIS_FEAT_IS_COMPRESSED].set(1)
            s = s.at[comp_tok, _AXIS_FEAT_SIZE].set(1)
            return s

        state = jax.lax.cond(is_diag, _apply_diag, lambda s: s, state)
        state = jax.lax.cond(is_compress, _apply_compress, lambda s: s, state)
        new_gid = jnp.where(is_diag, gid + 1, gid)
        return state, new_gid

    final_state, _ = jax.lax.fori_loop(
        0, MAX_RULES_PER_VERTEX, _body, (axis_state_v, init_gid)
    )
    return final_state


def compute_static_axis_state(jaxpr, total_v: int) -> tuple[np.ndarray, np.ndarray]:
    axis_state = np.zeros(
        (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM),
        dtype=np.int32,
    )
    axis_state[..., _AXIS_FEAT_GROUP_ID] = -1
    axis_valid = np.zeros((total_v, MAX_AXES_PER_VERTEX), dtype=np.float32)

    for v_idx, eqn in enumerate(jaxpr.eqns):
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue
        out_shape = eqn.outvars[0].aval.shape
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        primal_shape = invars[0].aval.shape if invars else ()

        slot = 0
        for size in out_shape:
            if slot >= MAX_AXES_PER_VERTEX:
                break
            axis_state[v_idx, slot, _AXIS_FEAT_SIZE] = int(size)
            axis_state[v_idx, slot, _AXIS_FEAT_IS_OUTPUT] = 1
            axis_valid[v_idx, slot] = 1.0
            slot += 1
        for size in primal_shape:
            if slot >= MAX_AXES_PER_VERTEX:
                break
            axis_state[v_idx, slot, _AXIS_FEAT_SIZE] = int(size)
            axis_state[v_idx, slot, _AXIS_FEAT_IS_OUTPUT] = 0
            axis_valid[v_idx, slot] = 1.0
            slot += 1

    return axis_state, axis_valid


def micro_actions_to_rule_specs(
    op_types,
    i_indices,
    j_indices,
    factors,
    *,
    axis_state_for_vertex,
    compress_kinds=None,
    quant_dtypes=None,
):
    from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT

    op_types_arr = np.asarray(op_types)
    i_arr = np.asarray(i_indices)
    j_arr = np.asarray(j_indices)
    f_arr = np.asarray(factors)
    if compress_kinds is None:
        k_arr = np.zeros_like(op_types_arr)
    else:
        k_arr = np.asarray(compress_kinds)
    if quant_dtypes is None:
        q_arr = np.zeros_like(op_types_arr)
    else:
        q_arr = np.asarray(quant_dtypes)
    axis_state_np = np.asarray(axis_state_for_vertex)

    n_out = int(np.sum(axis_state_np[:, _AXIS_FEAT_IS_OUTPUT]))

    def _to_base(token_idx: int) -> tuple[int, int]:
        token_idx = int(token_idx)
        is_out = int(axis_state_np[token_idx, _AXIS_FEAT_IS_OUTPUT])
        rel = token_idx if is_out else token_idx - n_out
        return (rel, is_out)

    specs = np.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, 2] = 0

    slot = 0
    for s_idx, op in enumerate(op_types_arr.tolist()):
        if op == OP_END:
            break
        if op == OP_COMPRESS:
            if slot >= MAX_RULES_PER_VERTEX:
                break
            physical_axis = int(i_arr[s_idx])
            specs[slot, 0] = COMPRESS_SENTINEL
            specs[slot, 1] = physical_axis
            specs[slot, 2] = int(k_arr[s_idx])
            slot += 1
            continue
        if op == OP_QUANT:
            if slot >= MAX_RULES_PER_VERTEX:
                break
            specs[slot, 0] = QUANT_SENTINEL
            specs[slot, 1] = int(q_arr[s_idx])
            specs[slot, 2] = 0
            slot += 1
            continue
        if op != OP_DIAG:
            raise ValueError(f"Unknown op_type {op!r} at sub-step {s_idx}.")
        if slot >= MAX_RULES_PER_VERTEX:
            break
        rel_i, is_out_i = _to_base(i_arr[s_idx])
        rel_j, is_out_j = _to_base(j_arr[s_idx])
        if is_out_i == is_out_j:
            break
        if is_out_i:
            bi1, bi2 = rel_i, rel_j
        else:
            bi1, bi2 = rel_j, rel_i
        specs[slot, 0] = bi1
        specs[slot, 1] = bi2
        specs[slot, 2] = int(f_arr[s_idx])
        slot += 1

    return specs


def micro_actions_to_rule_specs_jax(
    op_types,
    i_indices,
    j_indices,
    factors,
    axis_state_for_vertex,
    compress_kinds=None,
    quant_dtypes=None,
):
    from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END, OP_QUANT

    is_output = axis_state_for_vertex[:, _AXIS_FEAT_IS_OUTPUT].astype(jnp.int32)
    n_out = jnp.sum(is_output)

    if compress_kinds is None:
        compress_kinds = jnp.zeros_like(op_types)
    if quant_dtypes is None:
        quant_dtypes = jnp.zeros_like(op_types)

    is_end_per = op_types == OP_END
    prior_ends = jnp.cumsum(is_end_per.astype(jnp.int32)) - is_end_per.astype(jnp.int32)
    active = prior_ends == 0
    is_diag = op_types == OP_DIAG
    is_compress = op_types == OP_COMPRESS
    is_quant = op_types == OP_QUANT

    def _row(s_idx):
        i = i_indices[s_idx]
        j = j_indices[s_idx]
        is_out_i = is_output[i] > 0
        is_out_j = is_output[j] > 0
        rel_i = jnp.where(is_out_i, i, i - n_out)
        rel_j = jnp.where(is_out_j, j, j - n_out)
        same_side = is_out_i == is_out_j
        diag_bi1 = jnp.where(is_out_i, rel_i, rel_j)
        diag_bi2 = jnp.where(is_out_i, rel_j, rel_i)
        diag_used = active[s_idx] & is_diag[s_idx] & (~same_side)

        compress_used = active[s_idx] & is_compress[s_idx]
        compress_bi1 = jnp.asarray(COMPRESS_SENTINEL, dtype=jnp.int32)
        compress_bi2 = i.astype(jnp.int32)

        quant_used = active[s_idx] & is_quant[s_idx]
        quant_bi1 = jnp.asarray(QUANT_SENTINEL, dtype=jnp.int32)
        quant_bi2 = quant_dtypes[s_idx].astype(jnp.int32)

        bi1 = jnp.where(
            quant_used, quant_bi1,
            jnp.where(
                compress_used, compress_bi1,
                jnp.where(diag_used, diag_bi1, -1),
            ),
        ).astype(jnp.int32)
        bi2 = jnp.where(
            quant_used, quant_bi2,
            jnp.where(
                compress_used, compress_bi2,
                jnp.where(diag_used, diag_bi2, -1),
            ),
        ).astype(jnp.int32)
        f = jnp.where(
            quant_used, jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                compress_used, compress_kinds[s_idx],
                jnp.where(diag_used, factors[s_idx], 0),
            ),
        ).astype(jnp.int32)
        return jnp.stack([bi1, bi2, f])

    rows = jax.vmap(_row)(jnp.arange(op_types.shape[0]))
    rows_truncated = rows[:MAX_RULES_PER_VERTEX]
    pad_needed = MAX_RULES_PER_VERTEX - rows_truncated.shape[0]
    if pad_needed > 0:
        pad = jnp.tile(
            jnp.array([-1, -1, 0], dtype=jnp.int32),
            (pad_needed, 1),
        )
        rows_truncated = jnp.concatenate([rows_truncated, pad], axis=0)
    return rows_truncated


_LEGACY_SP_TO_RULE_ROW = jnp.array(
    [
        [-1, -1, 0],
        [0, 0, -1],
        [0, 1, -1],
        [1, 0, -1],
        [1, 1, -1],
    ],
    dtype=jnp.int32,
)


def _legacy_sp_to_specs(sp_type: Array) -> Array:
    first = _LEGACY_SP_TO_RULE_ROW[sp_type]
    pad = jnp.tile(
        jnp.array([-1, -1, 0], dtype=jnp.int32), (MAX_RULES_PER_VERTEX - 1, 1)
    )
    return jnp.concatenate([first[None, :], pad], axis=0)


@jax.jit
def cossim(target, preds):
    target = target / jnp.maximum(
        jnp.linalg.norm(target, keepdims=True), jnp.sqrt(1e-7)
    )
    preds = preds / jnp.maximum(jnp.linalg.norm(preds, keepdims=True), jnp.sqrt(1e-7))
    return jnp.sum(target * preds)


sp_type_to_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}


def _flatten_jacobians(jac):
    leaves = jax.tree_util.tree_leaves(jac)
    if not leaves:
        return None
    flats = [jnp.ravel(l) for l in leaves]
    return jnp.concatenate(flats)


def _quality_metrics(jac_exact, jac_approx):
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
    partial_order, partial_specs = _get_partials(order, sparsity_specs, stop)
    is_terminal = int(stop) >= len(order)

    o_list = [int(x) for x in partial_order.tolist()]
    specs_list = partial_specs.tolist()

    transforms: list[tuple[int, tuple]] = []
    last_v_idx = len(o_list) - 1
    for v_idx, v in enumerate(o_list):
        eqn = config.jaxpr.eqns[v - 1]
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue

        out_shape = eqn.outvars[0].aval.shape
        out_len = len(out_shape)
        primal_shapes = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
        if not primal_shapes:
            continue

        rules: list = []  # mixed list[Diag | Compress | Quant]
        used_axes: set[int] = set()
        for slot in range(MAX_RULES_PER_VERTEX):
            row = specs_list[v_idx][slot]
            bi1 = int(row[0])
            bi2 = int(row[1])
            factor = int(row[2])
            if bi1 == -1:
                break
            if bi1 == QUANT_SENTINEL:
                dtype_idx = bi2
                if not (0 <= dtype_idx < len(QUANT_DTYPES)):
                    dtype_idx = 0
                rules.append(Quant(dtype=QUANT_DTYPES[dtype_idx]))
                continue
            if bi1 == COMPRESS_SENTINEL:
                if v_idx != last_v_idx:
                    continue
                axis_idx = bi2
                kind_idx = factor
                fits_all = True
                if axis_idx < 0:
                    fits_all = False
                elif axis_idx < out_len:
                    pass
                else:
                    primal_pos = axis_idx - out_len
                    fits_all = all(primal_pos < len(ps) for ps in primal_shapes)
                if not fits_all:
                    continue
                if axis_idx in used_axes:
                    continue
                if not (0 <= kind_idx < len(COMPRESS_KINDS)):
                    kind_idx = 0
                used_axes.add(axis_idx)
                rules.append(Compress(axes=(axis_idx,), kind=COMPRESS_KINDS[kind_idx]))
                continue
            if bi1 < 0:
                continue
            idx1 = bi1
            idx2 = out_len + bi2
            if idx1 in used_axes or idx2 in used_axes or idx1 == idx2:
                continue
            if not (0 <= bi1 < out_len):
                continue
            n1 = int(out_shape[bi1])
            n2_list: list[int] = []
            fits_all = True
            for ps in primal_shapes:
                if not (0 <= bi2 < len(ps)):
                    fits_all = False
                    break
                n2_list.append(int(ps[bi2]))
            if not fits_all:
                continue
            if factor == 0 or factor == 1:
                continue
            if factor == -1:
                from functools import reduce as _reduce

                factor = _reduce(_math.gcd, [n1] + n2_list)
            if (
                factor <= 0
                or n1 % factor != 0
                or any(n2 % factor != 0 for n2 in n2_list)
            ):
                continue
            used_axes.add(idx1)
            used_axes.add(idx2)
            rules.append(Diag(i=idx1, j=idx2, factor=factor))
        if rules:
            transforms.append((int(v), tuple(rules)))

    ve = extract_jaxpr(
        config.jaxpr,
        config.argnums,
        o_list,
        config.sparse,
        args,
        consts,
        transforms=transforms,
    )
    # Truncation diagnostic mirrors the canonical env.py path so legacy
    # callers of this Ray variant get the same warn-once + counter
    # behaviour without re-implementing it here.
    raw_tokens = ve.tokenized()
    from alphagrad.approx.env import _record_tokenization_truncation
    _record_tokenization_truncation(int(raw_tokens.shape[0]))
    tokens = raw_tokens[:MAX_TOKENS]
    tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

    tokens_np = np.asarray(tokens)
    eqn_ids_np = compute_eqn_ids_from_tokens(tokens_np, _TOKEN_VOCAB)
    eqn_ids = jnp.asarray(eqn_ids_np, dtype=jnp.int32)

    if init:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    if config.terminal_rewards_only and not is_terminal:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    _, aux = vertex_elimination_jaxpr(
        config.jaxpr,
        o_list,
        consts,
        *args,
        argnums=config.argnums,
        count_ops=True,
        sparse_representation=config.sparse,
        transforms=transforms,
    )
    muls_adds_fmas = float(aux["adds"] + aux["muls"] + aux["fmas"])
    max_io_sum = float(aux["mem"])

    if config.target_fun is None:
        rewards = jnp.array(
            [
                -muls_adds_fmas,
                0.0,
                0.0,
                -max_io_sum,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
            dtype=jnp.float32,
        )
        return tokens, eqn_ids, rewards

    if config.remote_eval_fn is not None:
        flops, bytes_accessed, peak_memory, latency_ns, cosine_sim, frob_residual = (
            config.remote_eval_fn(o_list, transforms, eval_samples)
        )
    else:
        callback_device = None
        if config.exec_on_gpu:
            gpu_devices = jax.devices("gpu")
            if len(gpu_devices) < 2:
                raise RuntimeError(
                    "--exec-on-gpu requires at least two GPUs (one for the "
                    f"trainer, one for the env callback); got {len(gpu_devices)}."
                )
            callback_device = gpu_devices[-1]

        args_for_lower = (
            jax.device_put(args, callback_device)
            if callback_device is not None
            else args
        )

        compiled_approx = (
            jax.jit(
                jacve(
                    config.target_fun,
                    list(o_list),
                    argnums=config.argnums,
                    has_aux=config.has_aux,
                    sparse_representation=config.sparse,
                    transforms=transforms,
                ),
                keep_unused=True,
            )
            .lower(*args_for_lower)
            .compile()
        )

        # Exact reference Jacobian is order-invariant → use jax.jacrev
        # (native, order-free, XLA-optimised) instead of jacve with the
        # policy's (often catastrophic) dense order. Identical value,
        # ~0s vs ~125s. See env.py for the full investigation. NOTE:
        # env_ray.py is currently unused; kept consistent with env.py.
        if config.has_aux:
            def _exact_with_aux(*a):
                jac, aux = jax.jacrev(
                    config.target_fun, argnums=config.argnums, has_aux=True,
                )(*a)
                return (aux, jac)
            _exact_fn = _exact_with_aux
        else:
            _exact_fn = jax.jacrev(config.target_fun, argnums=config.argnums)
        compiled_exact = (
            jax.jit(_exact_fn, keep_unused=True)
            .lower(*args_for_lower)
            .compile()
        )

        if os.environ.get("ALPHAGRAD_SKIP_COST_ANALYSIS", "0") == "1":
            flops = 0.0
            bytes_accessed = 0.0
        else:
            cost_analysis = compiled_approx.cost_analysis() or {}
            flops = float(cost_analysis.get("flops", 0))
            bytes_accessed = float(cost_analysis.get("bytes accessed", 0))

        n_samples = 10 if config.measure_latency else 1
        monitoring_devices: list = []
        for x in jax.tree_util.tree_leaves(args_for_lower):
            if hasattr(x, "devices"):
                monitoring_devices.extend(list(x.devices()))
        unique_devices = list({id(d): d for d in monitoring_devices}.values())
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
            if callback_device is not None:
                eval_args_i = [jax.device_put(d, callback_device) for d in eval_args_i]

            if os.environ.get("ALPHAGRAD_BYPASS_RESOURCE_MONITOR", "0") == "1":
                out_approx = compiled_approx(*eval_args_i)
                latency_samples.append(0.0)
                peak_mem_samples.append(0.0)
            else:
                with ResourceMonitor(devices=unique_devices) as monitor:
                    out_approx = compiled_approx(*eval_args_i)
                latency_s = float(monitor.stats.get("time", 0.0))
                peak_bytes = float(monitor.stats.get("memory", 0.0))
                latency_samples.append(latency_s * 1e9)
                peak_mem_samples.append(peak_bytes)

            out_exact = compiled_exact(*eval_args_i)
            out_approxs.append(out_approx)
            out_exacts.append(out_exact)

        latency_ns = (
            float(_aggregate_samples(latency_samples, want_top_quartile=True))
            if config.measure_latency
            else 0.0
        )
        peak_memory = float(max(peak_mem_samples)) if peak_mem_samples else 0.0

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
    axis_state_static: Array | None = None
    axis_valid_static: Array | None = None
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
        axis_state_static: Array | None = None,
        axis_valid_static: Array | None = None,
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

        if axis_state_static is None or axis_valid_static is None:
            total_v = len(config.jaxpr.eqns)
            axis_state_np, axis_valid_np = compute_static_axis_state(
                config.jaxpr,
                total_v,
            )
            axis_state_static = jnp.asarray(axis_state_np, dtype=jnp.int32)
            axis_valid_static = jnp.asarray(axis_valid_np, dtype=jnp.float32)
        object.__setattr__(self, "axis_state_static", axis_state_static)
        object.__setattr__(self, "axis_valid_static", axis_valid_static)

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
        remote_eval_fn: Callable | None = None,
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
            remote_eval_fn=remote_eval_fn,
        )
        return cls(
            config,
            args=jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            num_envs=num_envs,
        )

    def tree_flatten(self):
        children = (
            self.args,
            self.consts,
            self.eval_args_samples,
            self.axis_state_static,
            self.axis_valid_static,
        )
        aux_data = (self.config, self.valid_vertices, self.num_envs)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts, eval_args_samples, axis_state_static, axis_valid_static = children
        config, valid_vertices, num_envs = aux_data
        return cls(
            config,
            args,
            consts,
            valid_vertices,
            num_envs,
            eval_args_samples,
            axis_state_static=axis_state_static,
            axis_valid_static=axis_valid_static,
        )

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
        initial_specs = initial_specs.at[..., 2].set(0)

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
            axis_state=self.axis_state_static,
            axis_valid_mask=self.axis_valid_static,
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

        v_idx = target_vertex - jnp.int32(1)
        updated_axis_v = _apply_rules_to_axis_state(
            state.axis_state[v_idx],
            rule_specs,
        )
        new_axis_state = state.axis_state.at[v_idx].set(updated_axis_v)

        new_state = EnvState(
            order=new_order,
            sparsity_specs=new_specs,
            tokens=tokens,
            eqn_ids=eqn_ids,
            axis_state=new_axis_state,
            axis_valid_mask=state.axis_valid_mask,
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
