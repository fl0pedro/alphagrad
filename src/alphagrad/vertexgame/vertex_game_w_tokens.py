from __future__ import annotations

import heapq
import os
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np
from graphax.core import _build_graph, extract_jaxpr, vertex_elimination_jaxpr
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

from alphagrad.utils.profiler import track_activity, register_thread, update_thread_registry, get_dashboard_summary

ENABLE_PREFETCH = True
MAX_TOKENS = 1024

# --- Prefetching Iterators ---

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
        if path:
            yield path
        if depth is not None and len(path) >= depth:
            continue
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

# --- Cache and Worker ---

class TwoTierCache:
    """Read-only global cache synced per batch, with NOGIL-safe sharded writes."""
    def __init__(self, num_shards: int = 64):
        self.num_shards = num_shards
        
        # Lock-free global read caches (safe in 3.13t if only updated synchronously)
        self.read_tokens = {}
        self.read_counts = {}
        
        # Sharded write caches
        self.write_tokens = [{} for _ in range(num_shards)]
        self.write_counts = [{} for _ in range(num_shards)]
        self.locks = [threading.Lock() for _ in range(num_shards)]
        self.sync_lock = threading.Lock()

    def get_tokens(self, key):
        val = self.read_tokens.get(key)
        if val is not None:
            return val
        idx = hash(key) % self.num_shards
        with self.locks[idx]:
            return self.write_tokens[idx].get(key)

    def set_tokens(self, key, value):
        idx = hash(key) % self.num_shards
        with self.locks[idx]:
            self.write_tokens[idx][key] = value

    def get_counts(self, key):
        val = self.read_counts.get(key)
        if val is not None:
            return val
        idx = hash(key) % self.num_shards
        with self.locks[idx]:
            return self.write_counts[idx].get(key)

    def set_counts(self, key, value):
        idx = hash(key) % self.num_shards
        with self.locks[idx]:
            self.write_counts[idx][key] = value

    def sync(self):
        """Merge sharded writes into the lock-free read cache."""
        with self.sync_lock:
            for i in range(self.num_shards):
                with self.locks[i]:
                    self.read_tokens.update(self.write_tokens[i])
                    self.read_counts.update(self.write_counts[i])
                    self.write_tokens[i].clear()
                    self.write_counts[i].clear()

# --- Optimization Globals ---

active_direct_calls = 0
direct_calls_cond = threading.Condition()

# --- Metrics ---

# _THREAD_METRICS: dict[str, int | set[int]] = {
#     "total_tasks_run": 0,
#     "unique_threads": set(),
#     "prefetch_tasks_generated": 0,
#     "prefetch_cache_skips": 0,
#     "cache_hits": 0,
#     "cache_misses": 0,
# }
# _METRICS_LOCK = threading.Lock()
# _LOCAL_METRICS = threading.local()

_SCHEDULED_TOKENS = set()
_SCHEDULED_COUNTS = set()
_SCHEDULED_LOCK = threading.Lock()

# def _get_local_metrics():
#     if not hasattr(_LOCAL_METRICS, "counts"):
#         _LOCAL_METRICS.counts = {}
#     return _LOCAL_METRICS.counts

# def _update_metrics(metric_name: str = "total_tasks_run", count: int = 1):
#     local_counts = _get_local_metrics()
#     local_counts[metric_name] = local_counts.get(metric_name, 0) + count
    
#     if metric_name == "unique_threads" or local_counts[metric_name] >= 10:
#         _sync_metrics_to_global()
    
#     if len(_SCHEDULED_TOKENS) > 10000:
#         _SCHEDULED_TOKENS.clear()
#     if len(_SCHEDULED_COUNTS) > 10000:
#         _SCHEDULED_COUNTS.clear()

# def _sync_metrics_to_global():
#     local_counts = _get_local_metrics()
#     if not local_counts:
#         return
        
#     with _METRICS_LOCK:
#         for name, val in local_counts.items():
#             if name == "unique_threads":
#                 _THREAD_METRICS["unique_threads"].add(threading.get_ident())
#             else:
#                 current_global = _THREAD_METRICS.get(name, 0)
#                 if isinstance(current_global, int):
#                     _THREAD_METRICS[name] = current_global + val
    
#     local_counts.clear()

# def log(msg):
#     import time
#     t = time.strftime("%H:%M:%S", time.localtime())
#     print(f"[{t}] [Thread {threading.get_ident()}] {msg}", flush=True)

# --- Thread Pool ---

class PriorityThreadPool:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or (os.cpu_count() or 4)
        self.task_queue = queue.PriorityQueue()
        self.workers = []
        self.task_counter = 0
        self.epoch_id = 0
        self._lock = threading.Lock()
        
        for i in range(self.max_workers):
            t = threading.Thread(
                target=self._worker_loop, 
                args=(i,), 
                daemon=True, 
                name=f"UnifiedWorker-{i}"
            )
            t.start()
            self.workers.append(t)

    def _worker_loop(self, worker_id: int):
        # register_thread()
        # _update_metrics("unique_threads")
        while True:
            with track_activity("worker_idle"):
                priority, _, task_epoch, fn, args, kwargs, future = self.task_queue.get()
            
            # Epoch invalidation for prefetch tasks
            if priority > 0 and task_epoch < self.epoch_id:
                self.task_queue.task_done()
                continue

            if not future.cancelled():
                activity_name = f"task_{fn.__name__}"
                try:
                    with track_activity(activity_name):
                        result = fn(*args, **kwargs)
                    future.set_result(result)
                    # _update_metrics("total_tasks_run")
                    # update_thread_registry()
                except Exception as e:
                    future.set_exception(e)
                    log(f"Error in unified worker {worker_id}: {e}")
            self.task_queue.task_done()
            # _sync_metrics_to_global()

    def submit(self, priority: int, fn: Callable, *args, **kwargs) -> Future:
        future = Future()
        with self._lock:
            count = self.task_counter
            self.task_counter += 1
            current_epoch = self.epoch_id
        self.task_queue.put((priority, count, current_epoch, fn, args, kwargs, future))
        return future
        
    def advance_epoch(self):
        with self._lock:
            self.epoch_id += 1

_UNIFIED_POOL = PriorityThreadPool()
_CACHE = TwoTierCache()

# --- Prefetcher ---

class GlobalPrefetcher:
    producer_thread: threading.Thread

    def __init__(self, num_workers: int | None = None):
        self.current_generator = None
        self.gen_lock = threading.Lock()
        self.condition = threading.Condition(self.gen_lock)
        
        # self.reporter_thread = threading.Thread(
        #     target=self._reporter_loop, daemon=True, name="PrefetchReporter"
        # )
        # self.reporter_thread.start()
        
        self.producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True, name="PrefetchProducer"
        )
        self.producer_thread.start()

    def _producer_loop(self):
        register_thread()
        while True:
            with track_activity("producer_idle"):
                gen = None
                with self.gen_lock:
                    while self.current_generator is None:
                        self.condition.wait()
                    gen = self.current_generator
            
            if gen and ENABLE_PREFETCH:
                with track_activity("producing_tasks"):
                    try:
                        for task in gen:
                            # Wait if direct callbacks are active to give them full priority
                            with direct_calls_cond:
                                while active_direct_calls > 0:
                                    direct_calls_cond.wait(timeout=0.1)
                                    # Re-check if generator changed during wait
                                    with self.gen_lock:
                                        if self.current_generator is not gen:
                                            break
                            
                            with self.gen_lock:
                                if self.current_generator is not gen:
                                    break
                            func, args = task
                            _UNIFIED_POOL.submit(1, func, *args)
                    except Exception as e:
                        log(f"Error in prefetch producer: {e}")
                    finally:
                        with self.gen_lock:
                            if self.current_generator is gen:
                                self.current_generator = None
            else:
                with self.gen_lock:
                    self.current_generator = None

    # def _reporter_loop(self):
    #     import time
    #     while True:
    #         time.sleep(5)
    #         with _METRICS_LOCK:
    #             metrics = _THREAD_METRICS.copy()
    #             unique_threads = len(metrics.get("unique_threads", set()))
    #             total_tasks = metrics.get("total_tasks_run", 0)
    #             gen_tasks = metrics.get("prefetch_tasks_generated", 0)
    #             skips = metrics.get("prefetch_cache_skips", 0)
            
    #         q_depth = _UNIFIED_POOL.task_queue.qsize()
    #         if gen_tasks > 0 or total_tasks > 0:
    #             hits = metrics.get("cache_hits", 0)
    #             misses = metrics.get("cache_misses", 0)
                
    #             if isinstance(hits, int) and isinstance(misses, int):
    #                 h, m = hits, misses
    #                 hit_rate = (h / (h + m)) * 100 if (h + m) > 0 else 0.0
    #             else:
    #                 h, m, hit_rate = 0, 0, 0.0

    #             log(f"Prefetch Status: Tasks Run={total_tasks}, Generated={gen_tasks}, Skips={skips}, Threads={unique_threads}, Queue Depth={q_depth}")
    #             log(f"Cache Stats: Hits={h}, Misses={m}, Hit Rate={hit_rate:.1f}%")
    #             log(get_dashboard_summary())

    def set_generator(self, gen: Iterator[tuple[Callable, tuple]]):
        with self.gen_lock:
            self.current_generator = gen
            self.condition.notify_all()

_PREFETCHER = GlobalPrefetcher()

# --- Core Logic ---

def extract_consensus_orders(batch_prefetch_orders: np.ndarray, top_k: int = 3) -> np.ndarray:
    if getattr(batch_prefetch_orders, 'ndim', 0) <= 1:
        return np.expand_dims(np.asarray(batch_prefetch_orders), 0)
    
    unique_orders, counts = np.unique(batch_prefetch_orders, axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    return unique_orders[sorted_indices[:top_k]]

def prefetch_task_generator(
    target: core.Jaxpr,
    argnums: tuple[int, ...],
    has_aux: bool,
    sparse: bool,
    args_ndims: Sequence[int],
    consts_ndims: Sequence[int],
    base_order: np.ndarray,
    step_idx: int,
    consensus_orders: np.ndarray,
    prefetch_type: str,
    args: Sequence,
    consts: Sequence,
) -> Iterator[tuple[Callable, tuple]]:
    log(f"DEBUG: prefetch_task_generator called with type={prefetch_type}, consensus_orders.shape={consensus_orders.shape}")
    try:
        target_id = id(target)
        # Cast to numpy and extract standard ints for hashing
        base_order_np = np.asarray(base_order)
        base_prefix = tuple(int(x) for x in base_order_np[:step_idx])
        tasks_yielded = 0
        MAX_TASKS_PER_STEP = 1000

        for p_order in consensus_orders:
            p_order_list = p_order.tolist()[:10]
            
            if prefetch_type == 'iddfs':
                iterator = iddfs(p_order_list)
            elif prefetch_type == 'dfs':
                iterator = dfs(p_order_list)
            elif prefetch_type == 'diagonal_powerset':
                iterator = diagonal_powerset(p_order_list)
            else:
                return

            for future_vertices in iterator:
                if tasks_yielded >= MAX_TASKS_PER_STEP:
                    break
                    
                future_vertices_ints = tuple(int(v) for v in future_vertices)
                future_step = step_idx + len(future_vertices_ints)
                path_key = base_prefix + future_vertices_ints
                
                token_key = (target_id, frozenset(path_key))
                count_key = (target_id, path_key)

                with _SCHEDULED_LOCK:
                    token_needs_work = token_key not in _SCHEDULED_TOKENS and _CACHE.get_tokens(token_key) is None
                    count_needs_work = count_key not in _SCHEDULED_COUNTS and _CACHE.get_counts(count_key) is None

                    if not token_needs_work and not count_needs_work:
                        # _update_metrics("prefetch_cache_skips")
                        continue

                    if token_needs_work:
                        _SCHEDULED_TOKENS.add(token_key)
                    if count_needs_work:
                        _SCHEDULED_COUNTS.add(count_key)

                if token_needs_work:
                    # _update_metrics("prefetch_tasks_generated")
                    yield _tokenize, (target, argnums, has_aux, sparse, args_ndims, consts_ndims, base_order_np, step_idx, future_vertices_ints, args, consts)
                    tasks_yielded += 1

                if count_needs_work:
                    # _update_metrics("prefetch_tasks_generated")
                    yield _get_counts, (target, argnums, has_aux, sparse, args_ndims, consts_ndims, base_order_np, step_idx, future_vertices_ints, args, consts)
                    tasks_yielded += 1

    except Exception as e:
        log(f"ERROR in prefetch_task_generator: {e}")

def _process_batched_callback(func, args_ndims, consts_ndims, *args):
    # *args indices:
    # 0: jaxpr, 1: argnums, 2: has_aux, 3: sparse, 
    # 4: base_order, 5: step_idx, 6: future_vertices_ints, 7: args_np, 8: consts_np
    
    idx_base_order = 4
    idx_step = 5
    idx_future = 6
    idx_args = 7
    idx_consts = 8
    
    order_arg = args[idx_base_order]
    
    is_batched = isinstance(order_arg, (np.ndarray, jax.Array)) and getattr(order_arg, 'ndim', 0) > 1
    B = order_arg.shape[0] if is_batched else 1

    def prepare_args(i=None):
        single_args = list(args)

        if is_batched and i is not None:
            if isinstance(single_args[idx_base_order], (np.ndarray, jax.Array)):
                single_args[idx_base_order] = single_args[idx_base_order][i]
            if isinstance(single_args[idx_step], (np.ndarray, jax.Array)):
                single_args[idx_step] = single_args[idx_step][i]
        
        def _slice(x, expected_ndim):
            if not isinstance(x, (np.ndarray, jax.Array)):
                return x
            curr_x = x
            while getattr(curr_x, 'ndim', 0) > expected_ndim:
                if i is not None and curr_x.shape[0] == B:
                    curr_x = curr_x[i]
                elif curr_x.shape[0] == 1:
                    curr_x = curr_x[0]
                elif i is None:
                    curr_x = curr_x[0]
                else:
                    break
            return curr_x

        single_args[idx_args] = jax.tree_util.tree_map(_slice, single_args[idx_args], args_ndims)
        single_args[idx_consts] = jax.tree_util.tree_map(_slice, single_args[idx_consts], consts_ndims)
        
        # Late Array Instantiation for Zero-Copy Generator
        base_order = single_args[idx_base_order]
        step = int(single_args[idx_step])
        future_vertices = single_args[idx_future]
        
        if future_vertices is not None:
            # Construct the target order lazily in the worker
            target_order = base_order.copy()
            for j, v in enumerate(future_vertices):
                if step + j < target_order.shape[0]:
                    target_order[step + j] = v
            single_args[idx_base_order] = target_order
            single_args[idx_step] = step + len(future_vertices)
            
        return single_args

    if is_batched:
        # Register active direct calls for prefetcher throttle
        with direct_calls_cond:
            global active_direct_calls
            active_direct_calls += 1

        try:
            def task(i):
                s_args = prepare_args(i)
                s_args.pop(idx_future) 
                return func(*s_args)
                
            futures = [_UNIFIED_POOL.submit(0, task, i) for i in range(B)]
            results = [f.result() for f in futures]
            _CACHE.sync()
            if isinstance(results[0], tuple):
                return tuple(np.stack(r) for r in zip(*results))
            return np.stack(results)
        finally:
            with direct_calls_cond:
                active_direct_calls -= 1
                direct_calls_cond.notify_all()
    else:
        with direct_calls_cond:
            active_direct_calls += 1
        
        try:
            def task_single():
                s_args = prepare_args(None)
                s_args.pop(idx_future)
                return func(*s_args)
                
            future = _UNIFIED_POOL.submit(0, task_single)
            res = future.result()
            _CACHE.sync()
            return res
        finally:
            with direct_calls_cond:
                active_direct_calls -= 1
                direct_calls_cond.notify_all()
    

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


def _tokenize_single(jaxpr, argnums, has_aux, sparse, order, stop, args_np, consts_np):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order

    ve = extract_jaxpr(
        jaxpr, argnums, partial_order.tolist(), sparse, args_np, consts_np
    )
    result = np.zeros(MAX_TOKENS, dtype=np.int32)
    tokens = ve.tokenized()
    n = min(len(tokens), MAX_TOKENS)
    result[:n] = tokens[:n]
    return result

def _tokenize(jaxpr, argnums, has_aux, sparse, args_ndims, consts_ndims, base_order, step_idx, future_vertices_ints, args, consts):
    args_np = jax.tree_util.tree_map(np.asarray, args)
    consts_np = jax.tree_util.tree_map(np.asarray, consts)
    
    def wrapped_tokenize_single(*args_for_single):
        _order = args_for_single[4]
        _stop = int(args_for_single[5])
        _order_slice = _order[:_stop]
        
        _partial_set = frozenset(int(x) if hasattr(_order_slice, "tolist") else _order_slice for x in _order_slice.tolist())
        cache_key = (id(args_for_single[0]), _partial_set)

        cached = _CACHE.get_tokens(cache_key)
        if cached is not None:
            # _update_metrics("cache_hits")
            return cached
        
        # _update_metrics("cache_misses")
        try:
            res = _tokenize_single(*args_for_single)
            _CACHE.set_tokens(cache_key, res)
            return res
        finally:
            with _SCHEDULED_LOCK:
                _SCHEDULED_TOKENS.discard(cache_key)

    return _process_batched_callback(
        wrapped_tokenize_single, args_ndims, consts_ndims, jaxpr, argnums, has_aux, sparse, base_order, step_idx, future_vertices_ints, args_np, consts_np
    )

def _get_counts_single(jaxpr, argnums, has_aux, sparse, order, stop, args_np, consts_np):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order

    outs, aux = vertex_elimination_jaxpr(
        jaxpr,
        partial_order.tolist(),
        consts_np,
        *args_np,
        argnums=argnums,
        count_ops=True,
        sparse_representation=sparse,
    )
    return np.int32(aux["fmas"])

def _get_counts(jaxpr, argnums, has_aux, sparse, args_ndims, consts_ndims, base_order, step_idx, future_vertices_ints, args, consts):
    args_np = jax.tree_util.tree_map(np.asarray, args)
    consts_np = jax.tree_util.tree_map(np.asarray, consts)

    def wrapped_counts_single(*args_for_single):
        _order = args_for_single[4]
        _stop = int(args_for_single[5])
        _order_slice = _order[:_stop]
        _partial_tuple = tuple(int(x) for x in _order_slice.tolist()) if hasattr(_order_slice, "tolist") else tuple(int(x) for x in _order_slice)
        cache_key = (id(args_for_single[0]), _partial_tuple)

        cached = _CACHE.get_counts(cache_key)
        if cached is not None:
            # _update_metrics("cache_hits")
            return cached
        
        # _update_metrics("cache_misses")
        try:
            res = _get_counts_single(*args_for_single)
            _CACHE.set_counts(cache_key, res)
            return res
        finally:
            with _SCHEDULED_LOCK:
                _SCHEDULED_COUNTS.discard(cache_key)

    return _process_batched_callback(
        wrapped_counts_single, args_ndims, consts_ndims, jaxpr, argnums, has_aux, sparse, base_order, step_idx, future_vertices_ints, args_np, consts_np
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

        args_ndims = jax.tree_util.tree_map(lambda x: getattr(x, "ndim", 0), self.args)
        consts_ndims = jax.tree_util.tree_map(lambda x: getattr(x, "ndim", 0), self.consts)
        object.__setattr__(self, "args_ndims", args_ndims)
        object.__setattr__(self, "consts_ndims", consts_ndims)

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
        cls,
        jaxpr: core.ClosedJaxpr,
        args=None,
        sparse=False,
        target_fun=None,
        num_envs=None,
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
            self.args_ndims,
            self.consts_ndims,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts = children
        (
            jaxpr,
            argnums,
            has_aux,
            sparse,
            target_fun,
            valid_vertices,
            num_envs,
            args_ndims,
            consts_ndims,
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
        object.__setattr__(obj, "args_ndims", args_ndims)
        object.__setattr__(obj, "consts_ndims", consts_ndims)
        return obj

    @property
    def _token_shape(self):
        return jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32)

    @property
    def _counts_shape(self):
        s = jax.ShapeDtypeStruct((), jnp.int32)
        return s

    def reset(self, num_envs: int | None = None) -> EnvState:
        if num_envs is None:
            num_envs = getattr(self, "num_envs", None)

        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)
        target = self.jaxpr

        tokenize_fn = partial(
            _tokenize,
            target,
            self.argnums,
            self.has_aux,
            self.sparse,
            self.args_ndims,
            self.consts_ndims,
        )

        tokens = jax.pure_callback(
            tokenize_fn,
            self._token_shape,
            initial_order,
            0,
            None,
            self.args,
            self.consts,
            vmap_method="expand_dims",
        )

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        fmas_init = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = jnp.array(0.0, dtype=jnp.float32)
        terminated = jnp.array(False, dtype=jnp.bool_)

        state = EnvState(
            order=initial_order,
            tokens=tokens,
            step_count=step_count,
            max_steps=max_steps,
            fmas=fmas_init,
            reward=reward,
            terminated=terminated,
        )

        if num_envs is not None and num_envs > 0:
            state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)),
                state
            )

        return state

    @jit
    def step(
        self, state: EnvState, action: int | Array, tokens_input: Array | None = None,
        prefetch_order: Array | None = None, prefetch_type: str = "diagonal_powerset"
    ) -> EnvOut:
        target = self.jaxpr
        if getattr(state.order, 'ndim', 0) > 1:
            out = jax.vmap(self.step, in_axes=(0, 0, 0 if tokens_input is not None else None, 0 if prefetch_order is not None else None, None))(state, action, tokens_input, prefetch_order, prefetch_type)
            
            # Trigger prefetch ONCE per batch if enabled
            if prefetch_order is not None:
                def trigger_prefetch_callback(o, s, po, a, c):
                    log("DEBUG: trigger_prefetch_callback triggered")
                    _UNIFIED_POOL.advance_epoch()
                    consensus_orders = extract_consensus_orders(po)
                    gen = prefetch_task_generator(
                        self.jaxpr, self.argnums, self.has_aux, self.sparse, 
                        self.args_ndims, self.consts_ndims, o[0], int(s[0]), 
                        consensus_orders, prefetch_type, a, c
                    )
                    _PREFETCHER.set_generator(gen)
                    return 0.0

                _unused = jax.pure_callback(
                    trigger_prefetch_callback,
                    jax.ShapeDtypeStruct((), jnp.float32),
                    out.state.order,
                    out.state.step_count,
                    prefetch_order,
                    self.args,
                    self.consts
                )
                # Ensure the callback is part of the graph by adding its (zero) result
                out = out._replace(reward = out.reward + _unused)
            return out

        action = jnp.asarray(action, dtype=jnp.int32)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        pos = jnp.argwhere(curr_order == action, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)
        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(action)

        counts_fn = partial(
            _get_counts,
            target,
            self.argnums,
            self.has_aux,
            self.sparse,
            self.args_ndims,
            self.consts_ndims,
        )
        fmas = jax.pure_callback(
            counts_fn,
            self._counts_shape,
            new_order,
            new_step,
            None,
            self.args,
            self.consts,
            vmap_method="expand_dims",
        )

        if tokens_input is not None:
            tokens = tokens_input
        else:
            tokenize_fn = partial(
                _tokenize,
                target,
                self.argnums,
                self.has_aux,
                self.sparse,
                self.args_ndims,
                self.consts_ndims,
            )

            tokens = jax.pure_callback(
                tokenize_fn,
                self._token_shape,
                new_order,
                new_step,
                None,
                self.args,
                self.consts,
                vmap_method="expand_dims",
            )

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

        def _step_process(_):
            return EnvOut(new_state, reward, terminated)

        def _step_done(_):
            return EnvOut(state, jnp.array(0.0, jnp.float32), jnp.array(True, dtype=jnp.bool_))

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)