from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np
import os
from jax import Array, jit
from jax.tree_util import register_pytree_node_class

from graphax.core import _build_graph, extract_jaxpr, vertex_elimination_jaxpr

_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count()//2)

class Dispatcher:
    def __init__(self, fn_name, env):
        self.fn_name = fn_name
        self.env = env

    def __call__(self, *args):
        # We know jax.pure_callback passes: (order, stop, args_tuple, consts)
        order, stop, args_tuple, consts = args
        jaxpr = self.env.jaxpr
        argnums = self.env.argnums
        has_aux = self.env.has_aux
        sparse = self.env.sparse
        
        if self.fn_name == '_tokenize':
            future = _EXECUTOR.submit(
                _tokenize, jaxpr, argnums, has_aux, sparse, order, stop, args_tuple, consts
            )
        else:
            future = _EXECUTOR.submit(
                _get_counts, jaxpr, argnums, has_aux, sparse, order, stop, args_tuple, consts
            )
        return future.result()


MAX_TOKENS = 1024

class EnvState(NamedTuple):
    order: Array
    tokens: Array
    step_count: Array
    max_steps: int
    reward: float
    terminated: bool


class EnvOut(NamedTuple):
    state: EnvState
    reward: float
    terminated: bool


def _tokenize(jaxpr, argnums, has_aux, sparse, order, stop, args, consts):
    args_np = jax.tree_util.tree_map(np.asarray, args)
    consts_np = jax.tree_util.tree_map(np.asarray, consts)
    order_np = np.asarray(order)

    ve = extract_jaxpr(
        jaxpr, argnums, order_np.tolist(), sparse, args_np, consts_np
    )
    result = np.zeros(MAX_TOKENS, dtype=np.int32)
    tokens = ve.tokenized()
    n = min(len(tokens), MAX_TOKENS)
    result[:n] = tokens[:n]
    return result


def _get_counts(jaxpr, argnums, has_aux, sparse, order, stop, args, consts):
    args_np = jax.tree_util.tree_map(np.asarray, args)
    consts_np = jax.tree_util.tree_map(np.asarray, consts)
    order_np = np.asarray(order)
    v_stop = int(np.asarray(stop))

    if v_stop < len(order_np):
        partial_order = order_np[:v_stop]
    else:
        partial_order = order_np

    outs, aux = vertex_elimination_jaxpr(
        jaxpr,
        partial_order.tolist(),
        consts_np,
        *args_np,
        argnums=argnums,
        count_ops=True,
        sparse_representation=sparse,
    )
    return np.int32(aux["num_muls"]), np.int32(aux["num_adds"])


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class VertexEliminationEnv:
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    args: tuple
    consts: tuple
    valid_vertices: tuple
    has_aux: bool = False
    sparse: bool = False
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
        target_fun: Callable | None = None,
    ):
        object.__setattr__(self, "jaxpr", jaxpr)
        object.__setattr__(self, "argnums", tuple(argnums))
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "has_aux", has_aux)
        object.__setattr__(self, "sparse", sparse)
        object.__setattr__(self, "target_fun", target_fun)
        
        if valid_vertices is None:
            # We must pass concrete or dummy shape structs to _build_graph
            args_np = jax.tree_util.tree_map(np.asarray, args)
            consts_np = jax.tree_util.tree_map(np.asarray, consts)
            _, _, _, _, vo_vertices = _build_graph(jaxpr, args_np, consts_np)
            valid = []
            for i, eqn in enumerate(jaxpr.eqns, 1):
                if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices:
                    valid.append(i)
            valid_vertices = tuple(valid)
        object.__setattr__(self, "valid_vertices", valid_vertices)

    @classmethod
    def from_jaxpr(
        cls, jaxpr: core.ClosedJaxpr, args=None, sparse=False, target_fun=None
    ):
        return cls(
            jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.jaxpr.invars))),
            args=jaxpr.jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            sparse=sparse,
            target_fun=target_fun,
        )

    def tree_flatten(self):
        children = (self.args, self.consts)
        aux_data = (
            self.jaxpr,
            self.argnums,
            self.has_aux,
            self.sparse,
            self.target_fun,
            self.valid_vertices,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts = children
        jaxpr, argnums, has_aux, sparse, target_fun, valid_vertices = aux_data
        return cls(jaxpr, argnums, args, consts, valid_vertices, has_aux, sparse, target_fun)

    def _make_callback(self, fn_name):
        return Dispatcher(fn_name, self)

    @property
    def _token_shape(self):
        return jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32)

    @property
    def _counts_shape(self):
        s = jax.ShapeDtypeStruct((), jnp.int32)
        return s, s

    @jit
    def reset(self, initial_order: Array) -> EnvState:
        initial_order = jnp.asarray(initial_order, dtype=jnp.int32)
        
        tokens = jax.pure_callback(
            self._make_callback('_tokenize'),
            self._token_shape,
            initial_order,
            0,
            self.args,
            self.consts,
            vmap_method="sequential"
        )

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = 0.0
        terminated = False

        return EnvState(
            order=initial_order,
            tokens=tokens,
            step_count=step_count,
            max_steps=max_steps,
            reward=reward,
            terminated=terminated,
        )

    @jit
    def step(
        self, state: EnvState, action: int | Array, tokens_input: Array | None = None
    ) -> EnvOut:
        action = jnp.asarray(action, dtype=jnp.int32)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        pos = jnp.argwhere(curr_order == action, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where(
            (indices > idx) & (indices <= pos), indices - 1, indices
        )
        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(action)
        

        muls, adds = jax.pure_callback(
            self._make_callback('_get_counts'),
            self._counts_shape,
            new_order,
            new_step,
            self.args,
            self.consts,
            vmap_method="sequential"
        )
        num_ops = muls + adds

        if tokens_input is not None:
            tokens = tokens_input
        else:
            tokens = jax.pure_callback(
                self._make_callback('_tokenize'),
                self._token_shape,
                new_order,
                new_step,
                self.args,
                self.consts,
                vmap_method="sequential"
            )

        terminated = new_step >= state.max_steps
        reward = -num_ops.astype(jnp.float32)

        new_state = EnvState(
            order=new_order,
            tokens=tokens,
            step_count=new_step,
            max_steps=state.max_steps,
            reward=reward,
            terminated=terminated,
        )

        def _step_process(_):
            return EnvOut(new_state, reward, terminated)

        def _step_done(_):
            return EnvOut(state, 0.0, True)

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)
