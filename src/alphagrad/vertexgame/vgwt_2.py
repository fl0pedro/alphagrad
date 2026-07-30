from __future__ import annotations

import os

# from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial, wraps
from itertools import count
from threading import Lock
from typing import Callable, NamedTuple, Sequence

# def _init_cpu_thread():
#     import jax
#     cpus = jax.devices("cpu")
#     jax.config.update("jax_default_device", cpus[0])
#     jax.config.update("jax_disable_jit", True)
# THREADPOOL = ThreadPoolExecutor(
#     max_workers=os.cpu_count() or 8,
#     thread_name_prefix="vertex_game_",
#     initializer=_init_cpu_thread,
# )
import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np
from graphax.core import _build_graph, extract_jaxpr, vertex_elimination_jaxpr
from jax import Array, jit
from jax.tree_util import register_pytree_node_class

MAX_TOKENS = 1024


class EnvState(NamedTuple):
    order: Array
    tokens: Array
    step_count: Array
    max_steps: int
    fmas: Array
    reward: float
    terminated: bool


class EnvOut(NamedTuple):
    state: EnvState
    reward: float
    terminated: bool


def _env_logic_python(jaxpr, argnums, sparse, args, consts, order, stop):
    """Computes both tokens and FMAs in one pass to save cycles."""
    v_stop = int(stop)
    # Ensure order is a list for graphax
    order_list = order[:v_stop].tolist() if v_stop < len(order) else order.tolist()

    # vertex_elimination_jaxpr is the heavy lifter
    ve, aux = vertex_elimination_jaxpr(
        jaxpr,
        order_list,
        consts,
        *args,
        argnums=argnums,
        count_ops=True,
        sparse_representation=sparse,
    )

    # Extract tokens from the VE object
    tokens = ve.tokenized()[:MAX_TOKENS]
    padded_tokens = np.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

    # Return as a single concatenated array [tokens..., fma]
    return np.concatenate([padded_tokens, [aux["fmas"]]]).astype(np.int32)


def _tokenize(jaxpr, argnums, has_aux, sparse, args, consts, order, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order

    ve = extract_jaxpr(jaxpr, argnums, partial_order.tolist(), sparse, args, consts)
    tokens = ve.tokenized()[:MAX_TOKENS]
    return jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))


def _tokenize_and_count(jaxpr, argnums, has_aux, sparse, args, consts, order, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order

    _, aux = vertex_elimination_jaxpr(
        jaxpr,
        partial_order.tolist(),
        consts,
        *args,
        argnums=argnums,
        count_ops=True,
        sparse_representation=sparse,
    )
    tokens = _tokenize(jaxpr, argnums, has_aux, sparse, args, consts, order, stop)
    return jnp.concatenate([tokens, jnp.array(aux["fmas"])[None]])


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class VertexEliminationEnv:
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    args: tuple
    consts: tuple
    valid_vertices: tuple
    target_fun: Callable | None = None
    num_envs: int | None = None

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
        num_envs: int | None = None,
    ):
        print("yes")
        object.__setattr__(self, "jaxpr", jaxpr)
        object.__setattr__(self, "argnums", tuple(argnums))
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "has_aux", has_aux)
        object.__setattr__(self, "sparse", sparse)
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
        target_fun=None,
        num_envs=None,
    ):
        assert (argnums is None and args is None) or not (args is None or args is None)
        return cls(
            jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.invars))),
            args=jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            sparse=sparse,
            target_fun=target_fun,
            num_envs=num_envs,
        )

    def tree_flatten(self):
        children = (self.args, self.consts)
        aux_data = (
            self.jaxpr,
            self.argnums,
            self.valid_vertices,
            self.has_aux,
            self.sparse,
            self.target_fun,
            self.num_envs,
        )
        print("squash")
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
            target_fun,
            num_envs,
        ) = aux_data
        obj = cls(
            jaxpr,
            argnums,
            args,
            consts,
            valid_vertices,
            has_aux,
            sparse,
            target_fun,
            num_envs,
        )
        print("yoink")
        return obj

    def _token_shape(self, counting=False):
        return jax.ShapeDtypeStruct((MAX_TOKENS + int(counting),), jnp.int32)

    def _call_python_logic(self, order, step_count):
        # We wrap the python logic here.
        # Note: self.args and self.consts are already in the Pytree
        res = jax.experimental.io_callback(
            _env_logic_python,
            jax.ShapeDtypeStruct((MAX_TOKENS + 1,), jnp.int32),
            self.jaxpr,
            self.argnums,
            self.sparse,
            self.args,
            self.consts,
            order,
            step_count,
        )
        return res[..., :-1], res[..., -1]

    def reset(self) -> EnvState:
        # No internal vmap here if the trainer calls vmap(env.reset)
        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)

        # Initial step count is a scalar 0
        tokens, fmas = self._call_python_logic(initial_order, 0)

        return EnvState(
            order=initial_order,
            tokens=tokens,
            step_count=jnp.array(0, jnp.int32),
            max_steps=initial_order.shape[0],
            fmas=fmas,
            reward=jnp.array(0.0, jnp.float32),
            terminated=jnp.array(False, jnp.bool_),
        )

    @jit
    def step(
        self,
        state: EnvState,
        action: int | Array,
    ) -> EnvOut:
        action = jnp.asarray(action, dtype=jnp.int32)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        pos = jnp.argwhere(curr_order == action, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)
        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(action)

        tokens, fmas = self._call_python_logic(new_order, new_step)

        terminated = new_step >= state.max_steps
        step_cost = fmas - state.fmas
        reward = -step_cost.astype(jnp.float32)

        new_state = EnvState(
            order=new_order,
            tokens=tokens,
            step_count=new_step,
            max_steps=state.max_steps,
            fmas=fmas,
            reward=reward,
            terminated=terminated,
        )

        return EnvOut(new_state, reward, terminated)
