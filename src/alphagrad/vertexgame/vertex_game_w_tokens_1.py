from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np
from graphax.core import _build_graph, extract_jaxpr, vertex_elimination_jaxpr
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

# persist the worker
_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count())

_THREAD_METRICS = {
    "total_tasks_run": 0,
    "unique_threads": set(),
}

def print_thread_metrics():
    print("\n--- Thread Pool Metrics ---")
    print(f"Total tasks run: {_THREAD_METRICS['total_tasks_run']}")
    print(f"Unique threads used: {len(_THREAD_METRICS['unique_threads'])}")
    print("---------------------------\n")

def _process_batched_callback(func, *args):
    order_arg = args[4]
    
    is_batched = isinstance(order_arg, (np.ndarray, jax.Array)) and getattr(order_arg, 'ndim', 0) > 1
    
    if is_batched:
        B = order_arg.shape[0]
        
        def task(i):
            _THREAD_METRICS["total_tasks_run"] += 1
            _THREAD_METRICS["unique_threads"].add(threading.get_ident())
            
            def slice_if_batched(x):
                try:
                    if isinstance(x, (np.ndarray, jax.Array)) and getattr(x, 'ndim', 0) > 0:
                        if getattr(x, 'shape', (0,))[0] == B:
                            return x[i]
                        elif getattr(x, 'shape', (0,))[0] == 1:
                            return x[0]
                except Exception:
                    pass
                return x
                
            single_args = [jax.tree_util.tree_map(slice_if_batched, arg) for arg in args]
            return func(*single_args)
            
        futures = [_EXECUTOR.submit(task, i) for i in range(B)]
        results = [f.result() for f in futures]
        if isinstance(results[0], tuple):
            return tuple(np.stack(r) for r in zip(*results))
        return np.stack(results)
    else:
        def task_single():
            _THREAD_METRICS["total_tasks_run"] += 1
            _THREAD_METRICS["unique_threads"].add(threading.get_ident())
            return func(*args)
        return task_single()

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


def _tokenize_single(jaxpr, argnums, has_aux, sparse, order, stop, args_np, consts_np):
    v_stop = int(stop)

    if v_stop < len(order):
        partial_order = order[:v_stop]
    else:
        partial_order = order

    ve = extract_jaxpr(
        jaxpr, argnums, partial_order.tolist(), sparse, args_np, consts_np
    )
    result = np.zeros(MAX_TOKENS, dtype=np.int32)
    tokens = ve.tokenized()
    n = min(len(tokens), MAX_TOKENS)
    result[:n] = tokens[:n]
    return result

def _tokenize(jaxpr, argnums, has_aux, sparse, order, stop, args, consts):
    args_np = jax.tree_util.tree_map(np.asarray, args)
    consts_np = jax.tree_util.tree_map(np.asarray, consts)
    order_np = np.asarray(order)
    stop_np = np.asarray(stop)
    
    return _process_batched_callback(
        _tokenize_single, jaxpr, argnums, has_aux, sparse, order_np, stop_np, args_np, consts_np
    )


# not correct?
def _get_counts_single(jaxpr, argnums, has_aux, sparse, order, stop, args_np, consts_np):
    v_stop = int(stop)

    if v_stop < len(order):
        partial_order = order[:v_stop]
    else:
        partial_order = order

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

def _get_counts(jaxpr, argnums, has_aux, sparse, order, stop, args, consts):
    args_np = jax.tree_util.tree_map(np.asarray, args)
    consts_np = jax.tree_util.tree_map(np.asarray, consts)
    order_np = np.asarray(order)
    stop_np = np.asarray(stop)

    return _process_batched_callback(
        _get_counts_single, jaxpr, argnums, has_aux, sparse, order_np, stop_np, args_np, consts_np
    )


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
        cls, jaxpr: core.ClosedJaxpr, args=None, sparse=False, target_fun=None, num_envs=None
    ):
        return cls(
            jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.jaxpr.invars))),
            args=jaxpr.jaxpr.invars if args is None else args,
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
            self.has_aux,
            self.sparse,
            self.target_fun,
            self.valid_vertices,
            self.num_envs,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts = children
        jaxpr, argnums, has_aux, sparse, target_fun, valid_vertices, num_envs = aux_data
        return cls(
            jaxpr, argnums, args, consts, valid_vertices, has_aux, sparse, target_fun, num_envs
        )

    @property
    def _token_shape(self):
        return jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32)

    @property
    def _counts_shape(self):
        s = jax.ShapeDtypeStruct((), jnp.int32)
        return s, s

    def reset(self) -> EnvState:
        initial_order = jnp.zeros(len(self.valid_vertices), dtype=jnp.int32)

        target = self.target_fun if self.target_fun is not None else self.jaxpr

        tokenize_fn = partial(
            _tokenize,
            target,
            self.argnums,
            self.has_aux,
            self.sparse,
        )

        tokens = jax.pure_callback(
            tokenize_fn,
            self._token_shape,
            initial_order,
            0,
            self.args,
            self.consts,
            vmap_method="expand_dims",
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
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)
        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(action)

        target = self.target_fun if self.target_fun is not None else self.jaxpr

        counts_fn = partial(
            _get_counts,
            target,
            self.argnums,
            self.has_aux,
            self.sparse,
        )

        muls, adds = jax.pure_callback(
            counts_fn,
            self._counts_shape,
            new_order,
            new_step,
            self.args,
            self.consts,
            vmap_method="expand_dims",
        )
        num_ops = muls + adds

        if tokens_input is not None:
            tokens = tokens_input
        else:
            tokenize_fn = partial(
                _tokenize,
                target,
                self.argnums,
                self.has_aux,
                self.sparse,
            )

            tokens = jax.pure_callback(
                tokenize_fn,
                self._token_shape,
                new_order,
                new_step,
                self.args,
                self.consts,
                vmap_method="expand_dims",
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
