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

from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from jax_memory_monitor import ResourceMonitor

MAX_TOKENS = 4096
MAX_RULES_PER_VERTEX = 4
NUM_AXIS_PAIRS = 4

# Axis pair index -> (base_idx1, base_idx2). base_idx1 picks output axis 0/1; base_idx2 picks input axis 0/1.
axis_pair_idx_to_base = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


class EnvState(NamedTuple):
    order: Array
    sparsity_specs: Array  # (N, MAX_RULES_PER_VERTEX, 3) int32; row [base_idx1, base_idx2, factor]; base_idx1 < 0 means slot unused
    tokens: Array
    step_count: Array
    max_steps: int
    reward: Array
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
    cmp_type: Literal["graphax", "flops", "latency"]
    mem_type: Literal["graphax", "bytes_accessed", "peak_memory"]
    target_fun: Callable | None = None
    data_gen: Callable | None = None
    exec_on_gpu: bool = False


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


# cmp_type:
#   graphax  -> adds + muls + fmas from vertex_elimination_jaxpr
#   flops    -> cost_analysis()["flops"]
#   latency  -> ResourceMonitor.duration over 10 runs (top-quartile mean)
# mem_type:
#   graphax        -> "mem" from vertex_elimination_jaxpr
#   bytes_accessed -> cost_analysis()["bytes accessed"]
#   peak_memory    -> ResourceMonitor.peak (single run; or max over runs if cmp_type == latency)
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
        # [(1, ((1, 2, -1),)), (7, ((1, 3, -1),)), (4, ((1, 2, -1),)), (9, ((0, 2, -1),)), (2, ((0, 3, -1),))],
        # [(5, ((0, 3, -1),)), (4, ((1, 3, -1),)), (3, ((1, 2, -1),)), (1, ((0, 3, -1),))],
        # [(4, ((0, 2, -1),)), (10, ((1, 3, -1),)), (12, ((0, 3, -1),)), (2, ((0, 3, -1),))],
        # [(3, ((1, 2, -1),)), (1, ((0, 3, -1),))],
        # [(4, ((0, 3, -1),)), (8, ((0, 2, -1),)), (6, ((0, 3, -1),)), (7, ((1, 2, -1),))],
        # [(10, ((0, 3, -1),)), (5, ((1, 2, -1),)), (1, ((1, 3, -1),)), (12, ((1, 3, -1),)), (7, ((1, 2, -1),)), (6, ((1, 3, -1),))],
        # [(6, ((1, 3, -1),)), (4, ((0, 2, -1),)), (1, ((0, 2, -1),)), (7, ((1, 2, -1),))],
        # [(10, ((0, 3, -1),)), (7, ((0, 3, -1),)), (11, ((0, 3, -1),)), (6, ((0, 2, -1),))],
        # [(1, ((1, 2, -1),)), (7, ((1, 3, -1),)), (4, ((1, 2, -1),)), (9, ((0, 2, -1),)), (2, ((0, 3, -1),)), (10, ((0, 2, -1),))],
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
            jnp.array(jnp.iinfo(jnp.int32).min, dtype=jnp.int32),
            jnp.array(jnp.iinfo(jnp.int32).min, dtype=jnp.int32),
            jnp.array(-1.0, dtype=jnp.float32),
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

    cmp = jnp.array(0, dtype=jnp.int32)
    mem = jnp.array(0, dtype=jnp.int32)
    error = jnp.array(0.0, dtype=jnp.float32)

    if not init and (config.cmp_type == "graphax" or config.mem_type == "graphax"):
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
        if config.cmp_type == "graphax":
            cmp = jnp.array(aux["adds"] + aux["muls"] + aux["fmas"], dtype=jnp.int32)
        if config.mem_type == "graphax":
            mem = jnp.array(aux["mem"], dtype=jnp.int32)

    needs_cost_analysis = (
        config.cmp_type == "flops" or config.mem_type == "bytes_accessed"
    )
    needs_runtime = config.cmp_type == "latency" or config.mem_type == "peak_memory"
    should_compile = not init and (
        needs_cost_analysis or needs_runtime or config.target_fun is not None
    )

    if init or not should_compile:
        return tokens, cmp, mem, error

    assert config.target_fun is not None, (
        "Must provide a valid Callable for `target_fun` if compilation is required"
    )

    callback_device = None
    if config.exec_on_gpu:
        gpu_devices = jax.devices("gpu")
        callback_device = gpu_devices[1]

    # Ensure compilation target matches execution device
    args_for_lower = (
        jax.device_put(args, callback_device) if config.exec_on_gpu else args
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

    if needs_cost_analysis:
        cost_analysis = compiled_approx.cost_analysis()
        if config.cmp_type == "flops":
            cmp = jnp.array(cost_analysis.get("flops", 0), dtype=jnp.int32)
        if config.mem_type == "bytes_accessed":
            mem = jnp.array(cost_analysis.get("bytes accessed", 0), dtype=jnp.int32)

    # only run multiple times when measuring latency; otherwise a single sample
    # is enough for cossim error and (optionally) peak_memory.
    n_samples = 10 if config.cmp_type == "latency" else 1
    out_approxs = []
    out_exacts = []
    stats = jnp.zeros((2, n_samples), jnp.int32)

    unique_devices = None
    if needs_runtime:
        # identify devices for monitoring from representative args
        monitoring_devices = []
        for x in jax.tree_util.tree_leaves(args):
            if hasattr(x, "devices"):
                monitoring_devices.extend(list(x.devices()))
        unique_devices = list(set(monitoring_devices))
        if config.exec_on_gpu and callback_device not in unique_devices:
            unique_devices.append(callback_device)
        if not unique_devices:
            unique_devices = jax.local_devices()

    for i in range(n_samples):
        if eval_samples:
            eval_args_i = [arg[i] for arg in eval_samples]
        else:
            eval_args_i = list(args)

        if config.exec_on_gpu:
            eval_args_i = [jax.device_put(d, callback_device) for d in eval_args_i]

        if needs_runtime:
            with ResourceMonitor(devices=unique_devices) as monitor:
                out_approx = compiled_approx(*eval_args_i)
            cmp_val, mem_val = monitor.stats.values()
            if config.cmp_type == "latency":
                stats = stats.at[0, i].set(cmp_val * 10**9)
            if config.mem_type == "peak_memory":
                stats = stats.at[1, i].set(mem_val)
        else:
            out_approx = compiled_approx(*eval_args_i)

        out_exact = compiled_exact(*eval_args_i)
        out_approxs.append(out_approx)
        out_exacts.append(out_exact)

    if config.cmp_type == "latency":
        cmp = stats[0].sort()[6:8].mean().astype(jnp.int32)  # top quartile
    if config.mem_type == "peak_memory":
        mem = stats[1].max()  # single sample when n_samples == 1

    all_cossims = []
    for out_approx, out_exact in zip(out_approxs, out_exacts):
        jac_approx = out_approx[1] if config.has_aux else out_approx
        jac_exact = out_exact[1] if config.has_aux else out_exact

        leaves_approx = jax.tree_util.tree_leaves(jac_approx)
        leaves_exact = jax.tree_util.tree_leaves(jac_exact)

        # flat_mse = [
        #     jnp.mean((e - a) ** 2) for e, a in zip(leaves_exact, leaves_approx)
        # ]

        if leaves_approx and leaves_exact:
            flat_approx = jnp.concatenate([jnp.ravel(l) for l in leaves_approx])
            flat_exact = jnp.concatenate([jnp.ravel(l) for l in leaves_exact])
            if flat_approx.shape == flat_exact.shape and flat_approx.size > 0:
                all_cossims.append(cossim(flat_exact, flat_approx))
            else:
                all_cossims.append(jnp.array(1.0, dtype=jnp.float32))
        else:
            all_cossims.append(jnp.array(1.0, dtype=jnp.float32))

    if not all_cossims:
        error = jnp.array(1.0, dtype=jnp.float32)
    else:
        error = jnp.stack(all_cossims).sort()[6:8].mean()  # top quartile

    return tokens, cmp, mem, error


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
            _, _, _, _, vo_vertices = _build_graph(
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
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.float32),
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

        tokens, _, _, _ = io_callback(
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
        reward = jnp.zeros(3, dtype=jnp.float32)
        terminated = jnp.array(False, dtype=jnp.bool_)

        state = EnvState(
            order=initial_order,
            sparsity_specs=initial_specs,
            tokens=tokens,  # Directly use the unpacked token array
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

        tokens, cmp, mem, error = io_callback(
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
        reward = jnp.array(
            [-cmp, error, -mem],
        )

        new_state = EnvState(
            order=new_order,
            sparsity_specs=new_specs,
            tokens=tokens,
            step_count=new_step,
            max_steps=state.max_steps,
            reward=reward,
            terminated=terminated,
        )

        def _step_process(_):
            return EnvOut(new_state, reward, terminated)

        def _step_done(_):
            return EnvOut(
                state, jnp.zeros(3, jnp.float32), jnp.array(True, dtype=jnp.bool_)
            )

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)
