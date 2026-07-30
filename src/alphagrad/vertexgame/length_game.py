from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np
from graphax.core import _build_graph, extract_jaxpr, vertex_elimination_jaxpr
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

MAX_TOKENS = 1024 * 4


class EnvState(NamedTuple):
    order: Array
    tokens: Array
    step_count: Array
    max_steps: int
    length: Array
    reward: float
    terminated: bool


class EnvOut(NamedTuple):
    state: EnvState
    reward: float
    terminated: bool


def _tokenize(jaxpr, argnums, has_aux, sparse, args, consts, order, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order

    ve = extract_jaxpr(jaxpr, argnums, partial_order.tolist(), sparse, args, consts)
    tokens = ve.tokenized()
    n = len(tokens)
    if n > MAX_TOKENS:
        jax.debug.print(
            "Tokens excceed limit: {}/{} ({}%)",
            n,
            MAX_TOKENS,
            round((n * 100.0) / MAX_TOKENS, 2),
        )
        tokens = tokens[:MAX_TOKENS]

    tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))
    return jnp.concatenate([tokens, jnp.array(n)[None]], dtype=jnp.int32)


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

    def _token_shape(self):
        return jax.ShapeDtypeStruct((MAX_TOKENS + 1,), jnp.int32)

    def tokenize(self, args, consts, order, stop):
        return _tokenize(
            self.jaxpr,
            self.argnums,
            self.has_aux,
            self.sparse,
            args,
            consts,
            order,
            stop,
        )

    def tokenize_and_count(self, args, consts, order, stop):
        return _tokenize(
            self.jaxpr,
            self.argnums,
            self.has_aux,
            self.sparse,
            args,
            consts,
            order,
            stop,
        )

    def reset(self, num_envs: int | None = None) -> EnvState:
        if num_envs is None:
            num_envs = getattr(self, "num_envs", None)

        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)

        tokens_and_count = io_callback(
            self.tokenize,
            self._token_shape(),
            self.args,
            self.consts,
            initial_order,
            0,
        )
        tokens = tokens_and_count[..., :-1]
        length = tokens_and_count[..., -1]

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = jnp.array(0.0, dtype=jnp.float32)
        terminated = jnp.array(False, dtype=jnp.bool_)

        state = EnvState(
            order=initial_order,
            tokens=tokens,
            step_count=step_count,
            max_steps=max_steps,
            length=length,
            reward=reward,
            terminated=terminated,
        )

        if num_envs is not None and num_envs > 0:
            state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)), state
            )

        return state

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

        tokens_and_count = io_callback(
            self.tokenize_and_count,
            self._token_shape(),
            self.args,
            self.consts,
            new_order,
            new_step,
        )
        tokens = tokens_and_count[..., :-1]
        length = tokens_and_count[..., -1]

        terminated = new_step >= state.max_steps
        reward = length.astype(jnp.float32)

        new_state = EnvState(
            order=new_order,
            tokens=tokens,
            step_count=new_step,
            max_steps=state.max_steps,
            length=length,
            reward=reward,
            terminated=terminated,
        )

        def _step_process(_):
            return EnvOut(new_state, reward, terminated)

        def _step_done(_):
            return EnvOut(
                state, jnp.array(0.0, jnp.float32), jnp.array(True, dtype=jnp.bool_)
            )

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)
