from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

from graphax.core import extract_jaxpr, vertex_elimination_jaxpr, _build_graph

import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
import itertools
from more_itertools import powerset
import heapq
import time

class PrefetchStats:
    def __init__(self):
        self.hits = 0
        self.pending_hits = 0
        self.misses = 0
        self.direct_calls = 0
        self.direct_hits = 0
        self.direct_misses = 0
        self.direct_pendings = 0
        self.tasks_submitted = 0
        self.tasks_skipped_cached = 0
        self.searches_aborted = 0
        self.items_capped = 0
        self.cache_lock_wait_time = 0.0
        self.semaphore_wait_count = 0
        self.last_report_time = time.time()
        self.reported_steps = 0

    def report(self, force=False):
        now = time.time()
        if not force and now - self.last_report_time < 2:
            return
        timestamp = time.strftime("%H:%M:%S", time.localtime(now))
        print(f"\n[{timestamp} | Stats @ {self.reported_steps}] "
              f"Direct (Env): {self.direct_calls} calls ({self.direct_hits} hit, {self.direct_pendings} pend, {self.direct_misses} miss) | "
              f"Pref (Daemon): {self.hits} hits, {self.pending_hits} pend, {self.misses} miss | "
              f"Mngt: {self.tasks_submitted} sub, {self.tasks_skipped_cached} skip, {self.searches_aborted} abort, {self.items_capped} cap | "
              f"Locks: {self.cache_lock_wait_time:.4f}s wait, {self.semaphore_wait_count} sem_wait")
        self.last_report_time = now
        self.reported_steps += 1

_STATS = PrefetchStats()

_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=8)
_PREFETCH_QUEUE = queue.Queue()
_SEMAPHORE = threading.Semaphore(16)
MAX_PREFETCH_ITEMS = 512
_CACHE = {}
_CACHE_LOCK = threading.Lock()
MAX_CACHE_SIZE = 5000

def _sequence_generator(mode, seq, committed=(), pref_tail=()):
    def iddfs(seq):
        for d in range(1, len(seq) + 1):
            yield from dfs(seq, d)

    def dfs(seq, depth=None):
        n = len(seq)
        if n == 0:
            return
        stack = [((), 0)]
        
        while stack:
            path, current_idx = stack.pop()
            if path: yield path
            if depth is not None and len(path) >= depth: continue
            for i in range(n - 1, current_idx - 1, -1):
                stack.append((path + (seq[i],), i + 1))

    def diagonal_powerset(seq):
        n = len(seq)
        if n == 0:
            return
        pq = [(i, i, (seq[i],)) for i in range(n)]
        heapq.heapify(pq)

        while pq:
            idx_sum, max_idx, subset = heapq.heappop(pq)
            yield subset

            for j in range(max_idx + 1, n):
                new_sum = idx_sum + j
                new_subset = subset + (seq[j],)
                heapq.heappush(pq, (new_sum, j, new_subset))

    if mode == "greedy":
        curr = committed
        for p in pref_tail:
            curr = curr + (p,)
            yield curr
        yield from iddfs(seq)
    elif mode == "breadth":
        yield from powerset(seq)
    elif mode == "iterative depth":
        yield from iddfs(seq)
    elif mode == "depth":
        yield from dfs(seq)
    elif mode == "diagonal":
        yield from diagonal_powerset(seq)

def _prefetch_daemon_loop():
    while True:
        try:
            state = _PREFETCH_QUEUE.get()
            while not _PREFETCH_QUEUE.empty():
                try:
                    state = _PREFETCH_QUEUE.get_nowait()
                except queue.Empty:
                    break
            
            env, mode, full_pref, step, c_args, c_consts = state
            committed = full_pref[:step]
            pref_tail = full_pref[step:]
            
            _STATS.report(force=True)
            count = 0
            for task_seq in _sequence_generator(mode, pref_tail, committed=committed, pref_tail=pref_tail):
                if not _PREFETCH_QUEUE.empty():
                    _STATS.searches_aborted += 1
                    break
                if count >= MAX_PREFETCH_ITEMS:
                    _STATS.items_capped += 1
                    break 
                count += 1
                
                key_tok = (env.env_id, '_tokenize', committed + task_seq)
                key_cnt = (env.env_id, '_get_counts', committed + task_seq)
                
                should_abort = False
                for task_type, key in [('_tokenize', key_tok), ('_get_counts', key_cnt)]:
                    t0 = time.time()
                    with _CACHE_LOCK:
                        _STATS.cache_lock_wait_time += time.time() - t0
                        if key in _CACHE:
                            _STATS.tasks_skipped_cached += 1
                            continue
                        f = Future()
                        _CACHE[key] = f
                        
                    while not _SEMAPHORE.acquire(timeout=0.1):
                        _STATS.semaphore_wait_count += 1
                        if not _PREFETCH_QUEUE.empty():
                            should_abort = True
                            _STATS.searches_aborted += 1
                            break
                    
                    if should_abort:
                        break
                        
                    def make_task(t=task_type, k=key, s=committed + task_seq):
                        def task():
                            try:
                                with jax.default_device(jax.devices("cpu")[0]):
                                    full_order = np.array(s, dtype=np.int32)
                                    stop = len(s)
                                    if t == '_tokenize':
                                        res = _tokenize(env.jaxpr, env.argnums, env.has_aux, env.sparse, full_order, stop, c_args, c_consts)
                                    else:
                                        res = _get_counts(env.jaxpr, env.argnums, env.has_aux, env.sparse, full_order, stop, c_args, c_consts)
                                    _CACHE[k].set_result(res)
                            except Exception as e:
                                _CACHE[k].set_exception(e)
                            finally:
                                _SEMAPHORE.release()
                        return task
                    
                    _STATS.tasks_submitted += 1
                    _PREFETCH_EXECUTOR.submit(make_task())
                
                if should_abort:
                    break
                _STATS.report() # Report stats during search
            _STATS.report(force=True) # Final report after search completes or aborts
        except Exception:
            pass

_DAEMON_THREAD = threading.Thread(target=_prefetch_daemon_loop, daemon=True)
_DAEMON_THREAD.start()

class DaemonDispatcher:
    def __init__(self, fn_name, env):
        self.fn_name = fn_name
        self.env = env

    def __call__(self, *args):
        # We know io_callback passes: (order, stop, args_tuple, consts)
        order, stop, args_tuple, consts = args
        jaxpr = self.env.jaxpr
        argnums = self.env.argnums
        has_aux = self.env.has_aux
        sparse = self.env.sparse
        
        order_np = np.asarray(order)
        v_stop = int(np.asarray(stop))
        seq = tuple(int(x) for x in order_np[:v_stop])
        key = (self.env.env_id, self.fn_name, seq)
        
        _STATS.direct_calls += 1
        t0 = time.time()
        with _CACHE_LOCK:
            _STATS.cache_lock_wait_time += time.time() - t0
            if key in _CACHE:
                f = _CACHE[key]
                if f.done():
                    _STATS.direct_hits += 1
                    _STATS.hits += 1
                else:
                    _STATS.direct_pendings += 1
                    _STATS.pending_hits += 1
                do_compute = False
            else:
                _STATS.direct_misses += 1
                _STATS.misses += 1
                if len(_CACHE) >= MAX_CACHE_SIZE:
                    _CACHE.clear()
                f = Future()
                _CACHE[key] = f
                do_compute = True
                
        if do_compute:
            try:
                if self.fn_name == '_tokenize':
                    res = _tokenize(jaxpr, argnums, has_aux, sparse, order, stop, args_tuple, consts)
                else:
                    res = _get_counts(jaxpr, argnums, has_aux, sparse, order, stop, args_tuple, consts)
                f.set_result(res)
            except Exception as e:
                f.set_exception(e)
        
        _STATS.report()
        return f.result()


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
    v_stop = int(np.asarray(stop))

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
    has_aux: bool = False
    sparse: bool = False
    target_fun: Callable | None = None
    prefetch_mode: str = "off"
    env_id: str = ""
    valid_vertices: tuple = ()

    def __init__(
        self,
        jaxpr: core.Jaxpr,
        argnums: Sequence[int],
        args: Sequence,
        consts: Sequence,
        has_aux: bool = False,
        sparse: bool = False,
        target_fun: Callable | None = None,
        prefetch_mode: str = "off",
        env_id: str = None,
        valid_vertices: tuple | None = None,
    ):
        object.__setattr__(self, "jaxpr", jaxpr)
        object.__setattr__(self, "argnums", tuple(argnums))
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "has_aux", has_aux)
        object.__setattr__(self, "sparse", sparse)
        object.__setattr__(self, "target_fun", target_fun)
        object.__setattr__(self, "prefetch_mode", prefetch_mode)
        
        if env_id is None:
            env_id = uuid.uuid4().hex
        object.__setattr__(self, "env_id", env_id)
        
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
        cls, jaxpr: core.ClosedJaxpr, args=None, sparse=False, target_fun=None, prefetch_mode="off"
    ):
        return cls(
            jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.jaxpr.invars))),
            args=jaxpr.jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            sparse=sparse,
            target_fun=target_fun,
            prefetch_mode=prefetch_mode,
        )

    def tree_flatten(self):
        children = (self.args, self.consts)
        aux_data = (
            self.jaxpr,
            self.argnums,
            self.has_aux,
            self.sparse,
            self.target_fun,
            self.prefetch_mode,
            self.env_id,
            self.valid_vertices,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts = children
        jaxpr, argnums, has_aux, sparse, target_fun, prefetch_mode, env_id, valid_vertices = aux_data
        return cls(jaxpr, argnums, args, consts, has_aux, sparse, target_fun, prefetch_mode, env_id, valid_vertices)

    def _make_callback(self, fn_name):
        return DaemonDispatcher(fn_name, self)

    def _trigger_prefetch(self, order, step_count, preferred_order, args_tuple, consts):
        if self.prefetch_mode == "off":
            return
        
        order_np = np.asarray(order)
        step = int(np.asarray(step_count))
        pref_np = np.asarray(preferred_order)
        
        if len(pref_np) > 0:
            committed = order_np[:step]
            committed_set = set(committed)
            
            tail = [int(x) for x in pref_np if int(x) in self.valid_vertices and int(x) not in committed_set]
            
            tail_set = set(tail)
            for v in self.valid_vertices:
                if v not in committed_set and v not in tail_set:
                    tail.append(v)
            
            full_pref_seq = tuple(committed) + tuple(tail)
        else:
            full_pref_seq = tuple(int(x) for x in order_np)

        with _PREFETCH_QUEUE.mutex:  
            _PREFETCH_QUEUE.queue.clear()
            _PREFETCH_QUEUE.all_tasks_done.notify_all()
            _PREFETCH_QUEUE.unfinished_tasks = 0
        _PREFETCH_QUEUE.put((self, self.prefetch_mode, full_pref_seq, step, args_tuple, consts))

    @property
    def _token_shape(self):
        return jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32)

    @property
    def _counts_shape(self):
        s = jax.ShapeDtypeStruct((), jnp.int32)
        return s, s

    @jit
    def reset(self, initial_order: Array, preferred_order: Array | None = None) -> EnvState:
        initial_order = jnp.asarray(initial_order, dtype=jnp.int32)
        if preferred_order is None:
            preferred_order = jnp.zeros(0, dtype=jnp.int32)
        else:
            preferred_order = jnp.asarray(preferred_order, dtype=jnp.int32)
        
        io_callback(self._trigger_prefetch, None, initial_order, 0, preferred_order, self.args, self.consts, ordered=False)

        tokens = io_callback(
            self._make_callback('_tokenize'),
            self._token_shape,
            initial_order,
            0,
            self.args,
            self.consts,
            ordered=False,
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
        self, state: EnvState, action: int | Array, tokens_input: Array | None = None, preferred_order: Array | None = None
    ) -> EnvOut:
        action = jnp.asarray(action, dtype=jnp.int32)
        if preferred_order is None:
            preferred_order_val = jnp.zeros(0, dtype=jnp.int32)
        else:
            preferred_order_val = jnp.asarray(preferred_order, dtype=jnp.int32)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        pos = jnp.argwhere(curr_order == action, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where(
            (indices > idx) & (indices <= pos), indices - 1, indices
        )
        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(action)
        
        io_callback(self._trigger_prefetch, None, new_order, new_step, preferred_order_val, self.args, self.consts, ordered=False)

        muls, adds = io_callback(
            self._make_callback('_get_counts'),
            self._counts_shape,
            new_order,
            new_step,
            self.args,
            self.consts,
            ordered=False,
        )
        num_ops = muls + adds

        if tokens_input is not None:
            tokens = tokens_input
        else:
            tokens = io_callback(
                self._make_callback('_tokenize'),
                self._token_shape,
                new_order,
                new_step,
                self.args,
                self.consts,
                ordered=False,
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

