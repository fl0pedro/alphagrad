from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, NamedTuple, Sequence
from jax_memory_monitor import ResourceMonitor

import time
import jax
import jax._src.core as core
import jax.numpy as jnp
import jax.random as jrand
from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class
from itertools import zip_longest

MAX_TOKENS = 4096


class EnvState(NamedTuple):
    order: Array
    sparsity_types: Array
    tokens: Array
    step_count: Array
    max_steps: int
    reward: Array
    terminated: bool


class EnvOut(NamedTuple):
    state: EnvState
    reward: Array
    terminated: bool


class EnvConfig(NamedTuple):
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    has_aux: bool
    sparse: bool
    reward_type: Literal["analytical", "estimated", "empirical"]
    target_fun: Callable | None = None
    data_gen: Callable | None = None
    exec_on_gpu: bool = False


def _get_partials(order, sparsity_types, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order
    partial_sp = (
        sparsity_types[:v_stop] if v_stop < len(sparsity_types) else sparsity_types
    )
    return partial_order, partial_sp


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


# analytical = fmas, sum(max(lhs.val.size, rhs.val.size, out.val.size), ...)
# estimated = cost_analysis() -> flops, bytes accessed
# empirical = latency, bytes accessed (?)
def _callback(
    config: EnvConfig,
    args,
    consts,
    order,
    sparsity_types,
    stop,
    *,
    init: bool = False,
):
    partial_order, partial_sp = _get_partials(order, sparsity_types, stop)

    o_list = [int(x) for x in partial_order.tolist()]
    sp_list = [int(x) for x in partial_sp.tolist()]

    sparsity_map = []
    for v, sp in zip(o_list, sp_list):
        if sp > 0 and sp in sp_type_to_map:  # sp == 0 skips rule creation
            eqn = config.jaxpr.eqns[v - 1]
            if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
                continue

            out_len = len(eqn.outvars[0].aval.shape)
            base_idx1, base_idx2 = sp_type_to_map[sp]

            idx1 = base_idx1
            idx2 = out_len + base_idx2

            rule = ((idx1, idx2, -1),)
            sparsity_map.append((v, rule))

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
        sparsity_map=sparsity_map,  # can we add the padding somewhere internally?
    )
    tokens = ve.tokenized()[:MAX_TOKENS]
    tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

    cmp = jnp.array(0, dtype=jnp.int32)
    mem = jnp.array(0, dtype=jnp.int32)
    error = jnp.array(0.0, dtype=jnp.float32)

    if not init and config.reward_type == "analytical":
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
        cmp = jnp.array(aux["adds"] + aux["muls"] + aux["fmas"], dtype=jnp.int32)
        mem = jnp.array(aux["mem"], dtype=jnp.int32)

    should_compile = not init and (
        config.reward_type != "analytical" or config.target_fun is not None
    )

    if init or not should_compile:
        return tokens, cmp, mem, error

    assert config.target_fun is not None, (
        "Must provide a valid Callable for `target_fun` if compilation is required"
    )

    eval_args = list(args)

    # TODO data generator or dataset retriever, most efficient would be to get some mini batches
    if config.data_gen is not None:
        data = config.data_gen(
            jrand.split(jrand.PRNGKey(int(stop)), 5)
        )  # must return tuples, don't provide a split key, let internals split as it goes
        for i, d in enumerate(data):
            eval_args[i] = jnp.asarray(d)

    if config.exec_on_gpu:
        gpu_devices = jax.devices("gpu")
        callback_device = gpu_devices[1]
        eval_args = [jax.device_put(d, callback_device) for d in eval_args]

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
            .lower(*eval_args)
            .compile()
        )

    compiled_approx = compiled(sparsity_map)
    compiled_exact = compiled()

    if config.reward_type == "estimated":
        cost_analysis = compiled_approx.cost_analysis()
        cmp = jnp.array(cost_analysis.get("flops", 0), dtype=jnp.int32)
        mem = jnp.array(cost_analysis.get("bytes accessed", 0), dtype=jnp.int32)

    if config.reward_type == "empirical":
        # Identify devices for monitoring from eval_args
        monitoring_devices = []
        for x in jax.tree_util.tree_leaves(eval_args):
            if hasattr(x, "devices"):
                monitoring_devices.extend(list(x.devices()))
        unique_devices = list(set(monitoring_devices))
        if not unique_devices:
            unique_devices = jax.local_devices()

        with ResourceMonitor(devices=unique_devices) as monitor:
            out_approx = compiled_approx(*eval_args)
        cmp, mem = monitor.stats.values()
        cmp = jnp.array(cmp*10**9, jnp.int32)
        mem = jnp.array(mem, jnp.int32)
    else:
        out_approx = compiled_approx(*eval_args)

    # out_approx = compiled_approx(*eval_args)
    out_exact = compiled_exact(*eval_args)

    jac_approx = out_approx[1] if config.has_aux else out_approx
    jac_exact = out_exact[1] if config.has_aux else out_exact

    leaves_approx = jax.tree_util.tree_leaves(jac_approx)
    leaves_exact = jax.tree_util.tree_leaves(jac_exact)

    # flat_mse = [
    #     jnp.mean((e - a) ** 2) for e, a in zip(leaves_exact, leaves_approx)
    # ]

    flat_cossim = []
    for e, a in zip(leaves_exact, leaves_approx):
        if hasattr(e, "shape") and hasattr(a, "shape") and e.shape == a.shape:
            flat_cossim.append(cossim(e, a))
        else:
            flat_cossim.append(jnp.array(1.0, dtype=jnp.float32))

        # maybe we can make this a weighted average based on the magnitude measuring error (e.g. MSE)
        # TODO figure out if this makes sense mathematically.
        error = jnp.mean(jnp.stack(flat_cossim))

    return tokens, cmp, mem, error


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class VertexEliminationEnv:
    config: EnvConfig
    args: tuple
    consts: tuple
    valid_vertices: tuple
    num_envs: int | None = None

    def __init__(
        self,
        config: EnvConfig,
        args: Sequence,
        consts: Sequence,
        valid_vertices: tuple | None = None,
        num_envs: int | None = None,
    ):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))

        if num_envs is None:
            num_envs = jax.local_device_count()
        object.__setattr__(self, "num_envs", num_envs)

        if valid_vertices is None:
            _, _, _, _, vo_vertices = _build_graph(config.jaxpr, args, consts, config.argnums)
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
        reward_type: str = "analytical",
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
            reward_type=reward_type,
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
        children = (self.args, self.consts)
        aux_data = (self.config, self.valid_vertices, self.num_envs)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts = children
        config, valid_vertices, num_envs = aux_data
        return cls(config, args, consts, valid_vertices, num_envs)

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
        initial_sp_types = jnp.zeros_like(initial_order)

        tokens, _, _, _ = io_callback(
            self.tokenize(init=True),
            self._callback_shape,
            self.args,
            self.consts,
            initial_order,
            initial_sp_types,
            0,
        )

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = jnp.zeros(3, dtype=jnp.float32)
        terminated = jnp.array(False, dtype=jnp.bool_)

        state = EnvState(
            order=initial_order,
            sparsity_types=initial_sp_types,
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
    def step(self, state: EnvState, action: int | Array) -> EnvOut:
        action = jnp.asarray(action, dtype=jnp.int32)

        sp_type = action // MAX_TOKENS
        target_vertex = action % MAX_TOKENS

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        curr_sp = state.sparsity_types

        pos = jnp.argwhere(curr_order == target_vertex, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)

        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(target_vertex)
        new_sp = curr_sp[shifted.astype(jnp.int32)].at[idx].set(sp_type)

        tokens, cmp, mem, error = io_callback(
            self.tokenize(),
            self._callback_shape,
            self.args,
            self.consts,
            new_order,
            new_sp,
            new_step,
        )

        terminated = new_step >= state.max_steps
        reward = jnp.array(
            [-cmp, error, -mem],
        )

        new_state = EnvState(
            order=new_order,
            sparsity_types=new_sp,
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
