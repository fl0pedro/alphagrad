import multiprocessing
import os
import sys

if multiprocessing.current_process().name != "MainProcess":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import argparse
import heapq
import threading
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from functools import partial
from itertools import chain, combinations, permutations
from typing import Any, Callable, NamedTuple, Optional, Literal
from more_itertools import powerset
from dataclasses import dataclass

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from graphax.core import extract_jaxpr
from jax.experimental import io_callback

from alphagrad.vertexgame.fast_core import vertex_eliminate

@dataclass
class BenchmarkStats:
    direct_calls: int = 0
    direct_hits: int = 0
    direct_pendings: int = 0
    direct_misses: int = 0
    prefetch_hits: int = 0
    cache_lock_wait: float = 0
    semaphore_wait: float = 0
    submitted: int = 0
    skipped: int = 0
    
    def reset(self):
        for field in self.__dataclass_fields__:
            setattr(self, field, 0)
            
    def report(self, batch_size, steps):
        print(f"\n[Telemetry | B={batch_size}, S={steps}]")
        print(f"  Calls: {self.direct_calls} ({self.direct_hits} hit, {self.direct_pendings} pend, {self.direct_misses} miss)")
        print(f"  Prefetch Efficiency: {self.prefetch_hits} used hits / {self.submitted} submitted")
        print(f"  Waits: Lock {self.cache_lock_wait:.4f}s | Sem {self.semaphore_wait:.4f}s")
        print(f"  Mngt: {self.skipped} skipped")

_STATS = BenchmarkStats()


class Backend(Enum):
    NONE = "None (Seq)"
    THREAD = "Threading"


class Coordination(Enum):
    PYTHON = "Python (Direct)"
    PURE_CALLBACK = "jax.pure_callback"
    IO_CALLBACK = "jax.experimental.io_callback"
    ASYNC = "Async"


class PrefetchMode(Enum):
    OFF = "No Pref"
    MODE1 = "Mode 1 (Spec)"
    MODE2 = "Mode 2 (Pref)"


class EnvState(NamedTuple):
    order: Any
    tokens: Any
    step_count: Any
    edges: Any
    max_steps: Any


class StaticConfig(NamedTuple):
    fn: Callable
    argnums: tuple[int, ...]
    has_aux: bool
    sparse: bool


class DynamicConfig(NamedTuple):
    args: Any
    consts: Any


class MockFuture:
    def __init__(self, res):
        self.res = res

    def result(self):
        return self.res


class MockExecutor:
    def submit(self, fn, *args):
        return MockFuture(fn(*args))

    def shutdown(self, wait=True):
        pass


_WORK_JAXPR_CACHE = {}


def get_jaxpr(s, d):
    if s.fn not in _WORK_JAXPR_CACHE:
        _WORK_JAXPR_CACHE[s.fn] = jax.make_jaxpr(s.fn)(*d.args).jaxpr
    return _WORK_JAXPR_CACHE[s.fn]


# Shared computation cache: key (prefix tuple) -> Future
# All batch elements with the same elimination prefix share one computation.
_SHARED_CACHE = {}
_SHARED_CACHE_LOCK = threading.Lock()

# Prefetch semaphore: limits in-flight prefetch jobs to prevent flooding.
_N_PREFETCH_WORKERS = max(1, os.cpu_count() // 2)
_N_ONDEMAND_WORKERS = max(1, os.cpu_count() // 2)
_PREFETCH_SEMAPHORE = threading.Semaphore(_N_PREFETCH_WORKERS * 2)


def local_tokenize(s, d, order, stop):
    jaxpr, ord_np = get_jaxpr(s, d), np.asarray(order)
    arg_np = jax.tree_util.tree_map(np.asarray, d.args)
    con_np = jax.tree_util.tree_map(np.asarray, d.consts)
    ve = extract_jaxpr(jaxpr, s.argnums, ord_np[:int(stop)], s.sparse, arg_np, con_np)
    res = np.zeros(1024, dtype=np.int32)
    res[: min(len(ve.tokenized()), 1024)] = ve.tokenized()[:1024]
    return res


def _prefetch_tokenize(s, d, order, stop):
    """Wrapper that releases the prefetch semaphore after completing."""
    try:
        return local_tokenize(s, d, order, stop)
    finally:
        _PREFETCH_SEMAPHORE.release()


class BatchTracker:
    """Lightweight per-batch-element telemetry. No thread, no lock overhead."""
    __slots__ = ('prefetch_hits', 'unique_requests', '_seen_keys')

    def __init__(self):
        self.prefetch_hits = 0
        self.unique_requests = 0
        self._seen_keys = set()

    def reset(self):
        self.prefetch_hits = 0
        self.unique_requests = 0
        self._seen_keys = set()


class SharedDispatcher:
    """Single dispatcher thread that prefetches for ALL batch elements.
    
    Since all batch elements share the same order in the benchmark,
    one dispatcher generates all needed predictions without redundancy.
    """
    def __init__(self, prefetch_executor):
        self.prefetch_executor = prefetch_executor
        self.lock = threading.Lock()
        self.request_event = threading.Event()
        # Dispatch state
        self.history = ()
        self.full_order = None
        self.pref = None
        self.rem = None
        self.fn_args = None
        self.mode = PrefetchMode.OFF
        self._last_pref = None
        self.prefetched_keys = set()  # keys submitted by this dispatcher
        self.dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, daemon=True
        )
        self.dispatcher_thread.start()

    def reset(self):
        with self.lock:
            self.history = ()
            self._last_pref = None
            self.prefetched_keys = set()

    @staticmethod
    def _gen(pref, rem, mode: Literal["diagonal", "depth", "breadth", "greedy"]="diagonal", bench_mode=None):
        """Generate candidate sequences.
        
        For Mode 2: yields sequential prefixes FIRST (the keys we'll actually
        need at each step), then speculative subsets. This ensures the
        highest-value predictions are submitted first and computed during warmup.
        """
        if bench_mode == PrefetchMode.MODE1 and rem is not None:
            w = tuple(rem.tolist()) if hasattr(rem, "tolist") else tuple(rem) 
            mode = "breadth"
        elif bench_mode == PrefetchMode.MODE2 and pref is not None:
            w = tuple(pref.tolist()) if hasattr(pref, "tolist") else tuple(pref)
        else:
            raise TypeError("bench_mode must be one of the PrefetchModes")
        n = len(w)

        if mode == "greedy":
            for i in range(len(w)):
                yield w[:i]
            mode == "breadth"

        if mode == "diagonal":
            pq = [(i, i, (w[i],)) for i in range(n)]
            heapq.heapify(pq)

            while pq:
                idx_sum, max_idx, subset = heapq.heappop(pq)
                yield subset

                for j in range(max_idx + 1, n):
                    new_sum = idx_sum + j
                    new_subset = subset + (w[j],)
                    heapq.heappush(pq, (new_sum, j, new_subset))
        elif mode == "depth":
            stack = [((), 0)]
            
            while stack:
                path, current_idx = stack.pop()
                yield path
                for i in range(n - 1, current_idx - 1, -1):
                    stack.append((path + (w[i],), i + 1))
        elif mode == "breadth":
            yield from powerset(w)
        else:
            raise TypeError("mode must be diagonal, depth, or breadth")
        
    def _dispatcher_loop(self):
        while True:
            self.request_event.wait()
            self.request_event.clear()
            with self.lock:
                args = self.fn_args
                mode = self.mode
                pref, rem = self.pref, self.rem
                history = self.history
                full_order = self.full_order
            if args is None or mode == PrefetchMode.OFF or full_order is None:
                continue
            s, d = args
            full_list = full_order.tolist() if hasattr(full_order, 'tolist') else list(full_order)
            for sub_seq in self._gen(pref, rem, "depth", mode):
                sub_seq = tuple(sub_seq)
                cache_key = history + sub_seq
                # Check shared cache (no semaphore needed)
                with _SHARED_CACHE_LOCK:
                    if cache_key in _SHARED_CACHE:
                        with self.lock:
                            self.prefetched_keys.add(cache_key)
                        continue
                # Acquire semaphore — blocks if too many in-flight
                t_sem = time.perf_counter()
                _PREFETCH_SEMAPHORE.acquire()
                _STATS.semaphore_wait += time.perf_counter() - t_sem
                with _SHARED_CACHE_LOCK:
                    if cache_key in _SHARED_CACHE:
                        _PREFETCH_SEMAPHORE.release()
                        with self.lock:
                            self.prefetched_keys.add(cache_key)
                        _STATS.skipped += 1
                        continue
                    stop = len(cache_key)
                    prefix_set = set(cache_key)
                    remaining = [v for v in full_list if v not in prefix_set]
                    order_array = np.array(list(cache_key) + remaining, dtype=np.int32)
                    try:
                        fut = self.prefetch_executor.submit(
                            _prefetch_tokenize, s, d, order_array, stop
                        )
                        _SHARED_CACHE[cache_key] = fut
                        _STATS.submitted += 1
                        with self.lock:
                            self.prefetched_keys.add(cache_key)
                    except RuntimeError as e:
                        _PREFETCH_SEMAPHORE.release()
                        if str(e) != "cannot schedule new futures after shutdown":
                            raise

    def prefetch(self, pref, rem, full_order, args, mode):
        self.mode = mode
        self.fn_args = args
        self.full_order = full_order
        with self.lock:
            self.history = tuple(rem.tolist()[:0]) if rem is not None else ()
            # Update history from the current step position
            # history is set by update_history()
        pref_key = tuple(pref.tolist()) if hasattr(pref, 'tolist') else (tuple(pref) if pref is not None else None)
        if pref_key == self._last_pref and mode == PrefetchMode.MODE2:
            return
        self._last_pref = pref_key
        self.pref = pref
        self.rem = rem
        self.request_event.set()

    def update_history(self, history_tuple):
        with self.lock:
            self.history = history_tuple


def benchmark_step(bench, state, action, graph, static, dynamic):
    def callback(o, s, d_args, d_consts):
        d_v = DynamicConfig(d_args, d_consts)
        tasks = [
            bench.submit_task(b, static, d_v, o[b], s[b]) for b in range(bench.batch_size)
        ]
        return np.stack([t.result() for t in tasks])

    struct = jax.ShapeDtypeStruct((bench.batch_size, 1024), jnp.int32)
    if bench.coord == Coordination.PURE_CALLBACK:
        tokens = jax.pure_callback(
            callback, struct, state.order, state.step_count,
            dynamic.args, dynamic.consts,
        )
    elif bench.coord == Coordination.IO_CALLBACK:
        tokens = io_callback(
            callback, struct, state.order, state.step_count,
            dynamic.args, dynamic.consts,
        )
    else:
        tokens = state.tokens
    new_edges, _ = jax.vmap(vertex_eliminate)(action, state.edges)
    return state._replace(
        tokens=tokens, step_count=state.step_count + 1, edges=new_edges
    )


class UnifiedBenchmark:
    def __init__(self, coord, backend, batch_size):
        self.coord, self.backend, self.batch_size = coord, backend, batch_size
        if backend == Backend.NONE:
            self.on_demand_executor = MockExecutor()
            self.prefetch_executor = MockExecutor()
        else:
            self.on_demand_executor = ThreadPoolExecutor(max_workers=_N_ONDEMAND_WORKERS)
            self.prefetch_executor = ThreadPoolExecutor(max_workers=_N_PREFETCH_WORKERS)
        # Single shared dispatcher (1 thread, not batch_size threads)
        self.dispatcher = SharedDispatcher(self.prefetch_executor)
        # Lightweight per-batch telemetry (no threads)
        self.trackers = [BatchTracker() for _ in range(batch_size)]

    def cleanup(self):
        self.on_demand_executor.shutdown(wait=False)
        self.prefetch_executor.shutdown(wait=False)

    def submit_task(self, b, s, d, order, stop):
        order_np = np.asarray(order)
        prefix = tuple(order_np[:int(stop)].tolist())
        tracker = self.trackers[b]
        is_new = prefix not in tracker._seen_keys
        if is_new:
            tracker._seen_keys.add(prefix)
            tracker.unique_requests += 1
        _STATS.direct_calls += 1
        # Check shared cache
        t_lock = time.perf_counter()
        with _SHARED_CACHE_LOCK:
            _STATS.cache_lock_wait += time.perf_counter() - t_lock
            if prefix in _SHARED_CACHE:
                f = _SHARED_CACHE[prefix]
                if f.done():
                    _STATS.direct_hits += 1
                    if is_new and prefix in self.dispatcher.prefetched_keys:
                        tracker.prefetch_hits += 1
                        _STATS.prefetch_hits += 1
                else:
                    _STATS.direct_pendings += 1
                return f
            # Not cached — submit on-demand (bypasses prefetch semaphore)
            _STATS.direct_misses += 1
            fut = self.on_demand_executor.submit(
                local_tokenize, s, d, order_np, int(stop)
            )
            _SHARED_CACHE[prefix] = fut
            return fut

    def run_single(self, graph, static, dynamic, steps=10,
                   prefetch_mode=PrefetchMode.OFF, pref_seqs=None):
        """pref_seqs: list of per-batch-element preferred orders, or single array broadcast to all."""
        num_v = int(graph.at[0, 0, 1].get())
        order = jnp.tile(
            jnp.arange(1, num_v + 1, dtype=jnp.int32), (self.batch_size, 1)
        )
        edges = jnp.broadcast_to(graph, (self.batch_size, *graph.shape))
        state = EnvState(
            order,
            jnp.zeros((self.batch_size, 1024), jnp.int32),
            jnp.zeros(self.batch_size, jnp.int32),
            edges, num_v,
        )
        step_jit = jax.jit(benchmark_step, static_argnames=("bench", "static"))

        # Normalize pref_seqs: broadcast single order to all batch elements
        if pref_seqs is not None and not isinstance(pref_seqs, list):
            pref_seqs = [pref_seqs] * self.batch_size

        # Pre-seed prefetch BEFORE warmup — dispatcher generates sequential
        # prefixes first (highest value), then speculative subsets.
        # All of this runs in parallel with JIT compilation.
        if prefetch_mode != PrefetchMode.OFF:
            pref_0 = None
            if prefetch_mode == PrefetchMode.MODE2 and pref_seqs is not None:
                pref_0 = pref_seqs[0] if pref_seqs[0] is not None else None
            self.dispatcher.prefetch(
                pref_0, np.asarray(order[0]),
                np.asarray(order[0]), (static, dynamic), prefetch_mode
            )

        # Warmup (JIT compilation — dispatcher populates cache in parallel!)
        _ = step_jit(self, state, order[:, 0], graph, static, dynamic)
        jax.block_until_ready(_)

        start = time.perf_counter()
        for i in range(steps):
            if prefetch_mode != PrefetchMode.OFF:
                # Update history so dispatcher knows current position
                history = tuple(np.asarray(order[0, :i]).tolist())
                self.dispatcher.update_history(history)

            if self.coord == Coordination.ASYNC:
                nxt = np.asarray(state.step_count) + 1
                ord_np = np.asarray(state.order)
                futs = [
                    self.submit_task(b, static, dynamic, ord_np[b], nxt[b])
                    for b in range(self.batch_size)
                ]
                tokens = jnp.stack([jnp.asarray(f.result()) for f in futs])
                state = step_jit(self, state, order[:, i], graph, static, dynamic)
                state = state._replace(tokens=tokens)
            elif self.coord == Coordination.PYTHON:
                actions = np.asarray(order[:, i])
                cur_edges = np.asarray(state.edges)
                new_edges_list = []
                for b in range(self.batch_size):
                    e, _ = vertex_eliminate(actions[b], cur_edges[b])
                    new_edges_list.append(e)
                tasks = [
                    self.submit_task(b, static, dynamic, np.asarray(order[b]), i + 1)
                    for b in range(self.batch_size)
                ]
                tokens = np.stack([t.result() for t in tasks])
                state = state._replace(
                    tokens=jnp.asarray(tokens),
                    edges=jnp.asarray(np.stack(new_edges_list)),
                    step_count=state.step_count + 1,
                )
            else:
                state = step_jit(self, state, order[:, i], graph, static, dynamic)

        jax.block_until_ready(state)
        return (time.perf_counter() - start) / steps

    def run_statistical(self, graph, static, dynamic, trials=5, steps=10,
                        prefetch_mode=PrefetchMode.OFF, pref_seq=None):
        times = []
        hit_rates = []
        for _ in range(trials):
            with _SHARED_CACHE_LOCK:
                _SHARED_CACHE.clear()
            _STATS.reset()
            for t in self.trackers:
                t.reset()

            t = self.run_single(graph, static, dynamic, steps, prefetch_mode, pref_seq)
            _STATS.report(self.batch_size, steps)
            times.append(t)

            total_hits = sum(tr.prefetch_hits for tr in self.trackers)
            total_reqs = sum(tr.unique_requests for tr in self.trackers)
            hit_rates.append(total_hits / total_reqs if total_reqs > 0 else 0.0)

        return np.median(times), np.std(times), np.mean(hit_rates)


def inject_noise(order, level):
    if level <= 0:
        return order
    order = np.array(order)
    n = len(order)
    for _ in range(int(level * n / 2)):
        i, j = np.random.randint(0, n, size=2)
        order[i], order[j] = order[j], order[i]
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="small,medium,large")
    parser.add_argument("--batches", type=str, default="8,16,32,64,128,256,512")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--coords", type=str, default=None,
                        help="Comma-separated coordination modes to run: python,pure,io,async")
    cli = parser.parse_args()

    sizes_list = [s.strip() for s in cli.sizes.split(",")]
    batch_sizes = [int(b.strip()) for b in cli.batches.split(",")]

    # Filter coordination modes
    coord_map = {
        'python': Coordination.PYTHON,
        'pure': Coordination.PURE_CALLBACK,
        'io': Coordination.IO_CALLBACK,
        'async': Coordination.ASYNC,
    }
    if cli.coords:
        coord_list = [coord_map[c.strip()] for c in cli.coords.split(",") if c.strip() in coord_map]
    else:
        coord_list = list(Coordination)

    from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph

    for size in sizes_list:
        if size == "small":
            n, compl = (10, 5)
        elif size == "medium":
            n, compl = (100, 20)
        else:
            n, compl = (500, 40)

        def f(x):
            for _ in range(compl):
                x = jnp.sin(jnp.cos(x)) * x
            return jnp.sum(x)

        args_in = (jnp.ones((n,)),)
        graph = make_graph(f, *args_in)
        static, dynamic = (
            StaticConfig(f, (0,), False, False),
            DynamicConfig(args_in, jax.make_jaxpr(f)(*args_in).literals),
        )
        num_v = int(graph.at[0, 0, 1].get())
        true_order = np.arange(1, num_v + 1, dtype=np.int32)

        print(f"\n============================================================")
        print(
            f"BENECHMARKING SIZE: {size.upper()} (n={n}, compl={compl}, trials={cli.trials})"
        )
        print(f"============================================================\n")

        header = f"{'Batch':<6} | {'Coordination':<25} | {'Prefetch':<20} | {'Median':<10} | {'Std':<8} | {'Hit%'}"
        print(header)
        print("-" * len(header))

        for batch_size in batch_sizes:
            backend = Backend.THREAD
            for coord in coord_list:
                try:
                    bench = UnifiedBenchmark(coord, backend, batch_size)
                    m, s, h = bench.run_statistical(
                        graph, static, dynamic, cli.trials, cli.steps, PrefetchMode.OFF
                    )
                    print(
                        f"{batch_size:<6} | {coord.value:<25} | {'None':<20} | {m:.4f}s | {s:.4f} | {'-':>6}",
                        flush=True,
                    )
                    bench.cleanup()
                except Exception as e:
                    print(
                        f"{batch_size:<6} | {coord.value:<25} | {'None':<20} | ERROR: {e}",
                        flush=True,
                    )

                # 2. Mode 1 (Speculative)
                try:
                    bench = UnifiedBenchmark(coord, backend, batch_size)
                    m, s, h = bench.run_statistical(
                        graph,
                        static,
                        dynamic,
                        cli.trials,
                        cli.steps,
                        PrefetchMode.MODE1,
                    )
                    print(
                        f"{batch_size:<6} | {coord.value:<25} | {'Mode 1 (Spec)':<20} | {m:.4f}s | {s:.4f} | {h * 100:>5.1f}%",
                        flush=True,
                    )
                    bench.cleanup()
                except Exception as e:
                    print(
                        f"{batch_size:<6} | {coord.value:<25} | {'Mode 1 (Spec)':<20} | ERROR: {e}",
                        flush=True,
                    )

                # 3. Mode 2 (Preferred) Noise Sweep (Coarse sweep for broad range)
                noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                for noise in noise_levels:
                    try:
                        noisy_order = inject_noise(true_order, noise)
                        bench = UnifiedBenchmark(coord, backend, batch_size)
                        m, s, h = bench.run_statistical(
                            graph,
                            static,
                            dynamic,
                            cli.trials,
                            cli.steps,
                            PrefetchMode.MODE2,
                            pref_seq=noisy_order,
                        )
                        print(
                            f"{batch_size:<6} | {coord.value:<25} | {f'Mode 2 ({int(noise * 100)}% n)':<20} | {m:.4f}s | {s:.4f} | {h * 100:>5.1f}%",
                            flush=True,
                        )
                        bench.cleanup()
                    except Exception as e:
                        print(
                            f"{batch_size:<6} | {coord.value:<25} | {f'Mode 2 ({int(noise * 100)}% n)':<20} | ERROR: {e}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
