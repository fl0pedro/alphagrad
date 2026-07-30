from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import jax.random as jrand
from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

MAX_TOKENS = 1024


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
    return -jnp.sum(target * preds)


def _callback(
    jaxpr,
    argnums,
    has_aux,
    sparse,
    target_fun,
    data_gen,
    args,
    consts,
    order,
    sparsity_types,
    stop,
    *,
    stage,
):
    partial_order, partial_sp = _get_partials(order, sparsity_types, stop)

    ve = extract_jaxpr(
        jaxpr,
        argnums,
        partial_order.tolist(),
        sparse,
        args,
        consts,
        partial_sp.tolist(),
    )
    tokens = ve.tokenized()[:MAX_TOKENS]
    tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

    if stage >= 1:
        _, aux = vertex_elimination_jaxpr(
            jaxpr,
            partial_order.tolist(),
            consts,
            *args,
            argnums=argnums,
            count_ops=True,
            sparse_representation=sparse,
            sparsity_types=partial_sp.tolist(),
        )
        fmas = jnp.array(aux["fmas"], dtype=jnp.int32)
    else:
        fmas = jnp.array(0, dtype=jnp.int32)

    if stage >= 2 and target_fun is not None:
        eval_args = list(args)
        if data_gen is not None:
            data = data_gen(jrand.split(jrand.PRNGKey(int(stop)), 4))

            if isinstance(data, tuple):
                for i, d in enumerate(data):
                    eval_args[i] = jnp.asarray(d)
            else:
                eval_args[0] = jnp.asarray(data)

        o_list = [int(x) for x in partial_order.tolist()]
        sp_list = [int(x) for x in partial_sp.tolist()]

        # todo mini batch to loop for some number of tests
        # this jit should be cached so its fine to compile regardless

        out_exact = jax.jit(
            jacve(
                target_fun,
                o_list,
                argnums=argnums,
                has_aux=has_aux,
                sparse_representation=sparse,
            )
        )(*eval_args)

        # this fn always changes so it won't be cached and should only be compiled for mini batches
        out_approx = jacve(
            target_fun,
            o_list,
            argnums=argnums,
            has_aux=has_aux,
            sparse_representation=sparse,
            sparsity_types=sp_list,
        )(*eval_args)

        jac_exact = out_exact[1] if has_aux else out_exact
        jac_approx = out_approx[1] if has_aux else out_approx

        leaves_exact = jax.tree_util.tree_leaves(jac_exact)
        leaves_approx = jax.tree_util.tree_leaves(jac_approx)

        # flat_mse = [
        #     jnp.mean((e - a) ** 2) for e, a in zip(leaves_exact, leaves_approx)
        # ]

        flat_cossim = [cossim(e, a) for e, a in zip(leaves_exact, leaves_approx)]

        error = jnp.mean(jnp.array(flat_cossim))
    else:
        error = jnp.array(0, dtype=jnp.float32)

    return tokens, fmas, error


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class VertexEliminationEnv:
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    args: tuple
    consts: tuple
    valid_vertices: tuple
    num_envs: int | None = None
    data_gen: Callable | None = None
    target_fun: Callable | None = None

    def __init__(
        self,
        jaxpr: core.Jaxpr,
        argnums: Sequence[int],
        args: Sequence,
        consts: Sequence,
        valid_vertices: tuple | None = None,
        has_aux: bool = False,
        sparse: bool = False,
        num_envs: int | None = None,
        data_gen: Callable | None = None,
        target_fun: Callable | None = None,
    ):
        object.__setattr__(self, "jaxpr", jaxpr)
        object.__setattr__(self, "argnums", tuple(argnums))
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "has_aux", has_aux)
        object.__setattr__(self, "sparse", sparse)
        object.__setattr__(self, "data_gen", data_gen)
        object.__setattr__(self, "target_fun", target_fun)

        if num_envs is None:
            num_envs = jax.local_device_count()
        object.__setattr__(self, "num_envs", num_envs)

        if valid_vertices is None:
            _, _, _, _, vo_vertices = _build_graph(jaxpr, args, consts)
            valid = []
            for i, eqn in enumerate(jaxpr.eqns, 1):
                if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices:
                    valid.append(i)
            valid_vertices = tuple(valid)
        object.__setattr__(self, "valid_vertices", valid_vertices)

    @classmethod
    def from_jaxpr(
        cls,
        jaxpr: core.ClosedJaxpr,
        argnums=None,
        args=None,
        sparse=False,
        num_envs=None,
        data_gen: Callable | None = None,
        target_fun: Callable | None = None,
    ):
        assert (argnums is None and args is None) or not (args is None or args is None)
        return cls(
            jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.invars))),
            args=jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            sparse=sparse,
            num_envs=num_envs,
            data_gen=data_gen,
            target_fun=target_fun,
        )

    def tree_flatten(self):
        children = (self.args, self.consts)
        aux_data = (
            self.jaxpr,
            self.argnums,
            self.valid_vertices,
            self.has_aux,
            self.sparse,
            self.num_envs,
            self.data_gen,
            self.target_fun,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts = children
        (
            jaxpr,
            argnums,
            valid_vertices,
            has_aux,
            sparse,
            num_envs,
            data_gen,
            target_fun,
        ) = aux_data
        obj = cls(
            jaxpr,
            argnums,
            args,
            consts,
            valid_vertices,
            has_aux,
            sparse,
            num_envs,
            data_gen,
            target_fun,
        )
        return obj

    def tokenize(self, stage=0):
        return partial(
            _callback,
            self.jaxpr,
            self.argnums,
            self.has_aux,
            self.sparse,
            self.target_fun,
            self.data_gen,
            stage=stage,
        )

    @property
    def _token_shape(self):
        return (
            jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.float32),
        )

    def reset(self, num_envs: int | None = None) -> EnvState:
        if num_envs is None:
            num_envs = getattr(self, "num_envs", None)

        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)
        initial_sp_types = jnp.zeros_like(initial_order)

        tokens, _, _ = io_callback(
            self.tokenize(stage=0),
            self._token_shape,
            self.args,
            self.consts,
            initial_order,
            initial_sp_types,
            0,
        )

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        fmas_init = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = jnp.zeros(2, dtype=jnp.float32)
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

        tokens, fmas, error = io_callback(
            self.tokenize(stage=2),
            self._token_shape,
            self.args,
            self.consts,
            new_order,
            new_sp,
            new_step,
        )

        terminated = new_step >= state.max_steps
        reward = jnp.array(
            [-fmas, -error],
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
                state, jnp.zeros(2, jnp.float32), jnp.array(True, dtype=jnp.bool_)
            )

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)
