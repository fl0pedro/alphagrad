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
from typing import Any, Callable, NamedTuple, Optional

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
from graphax.core import extract_jaxpr
from jax.experimental import io_callback

from alphagrad.vertexgame.fast_core import vertex_eliminate


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


_THREAD_SEMAPHORE = threading.Semaphore(os.cpu_count() // 2)


def local_tokenize(s, d, order, stop):
    jaxpr, ord_np = get_jaxpr(s, d), np.asarray(order)
    arg_np = jax.tree_util.tree_map(np.asarray, d.args)
    con_np = jax.tree_util.tree_map(np.asarray, d.consts)
    ve = extract_jaxpr(
        jaxpr, s.argnums, s.has_aux, ord_np, int(stop), s.sparse, arg_np, con_np
    )
    res = np.zeros(1024, dtype=np.int32)
    res[: min(len(ve.tokenized), 1024)] = ve.tokenized[:1024]
    _THREAD_SEMAPHORE.release()
    return res


class PrefetchManager:
    """Batched prefetch manager — per-batch-element caches and generators.
    
    Each batch element independently tracks its own elimination history,
    cache, and preferred order. The dispatcher generates predictions for
    ALL batch elements in parallel.
    """
    def __init__(self, on_demand_executor, prefetch_executor, batch_size):
        self.on_demand_executor = on_demand_executor
        self.prefetch_executor = prefetch_executor
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.request_event = threading.Event()
        # Per-batch-element state
        self.caches = [{} for _ in range(batch_size)]
        self.histories = [[] for _ in range(batch_size)]
        self.background_keys = [set() for _ in range(batch_size)]
        self._seen_keys = [set() for _ in range(batch_size)]
        # Shared state for dispatcher
        self.pref_seqs = [None] * batch_size  # per-element preferred orders
        self._last_prefs = [None] * batch_size
        self.full_orders = [None] * batch_size  # full N-element order per element
        self.remaining = [None] * batch_size
        self.fn_args = None
        self.mode = PrefetchMode.OFF
        self._dirty_batch = set()  # which batch elements need new dispatch
        # Telemetry
        self.prefetch_hits = 0
        self.unique_requests = 0
        self.pruned_count = 0
        self.dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, daemon=True
        )
        self.dispatcher_thread.start()

    def reset_telemetry(self):
        with self.lock:
            self.prefetch_hits = 0
            self.unique_requests = 0
            self.pruned_count = 0
            for b in range(self.batch_size):
                self.caches[b] = {}
                self.histories[b] = []
                self.background_keys[b] = set()
                self._seen_keys[b] = set()
                self._last_prefs[b] = None
            self._dirty_batch = set()

    def _gen(self, pref, rem):
        if self.mode == PrefetchMode.MODE2 and pref is not None:
            w = tuple(pref.tolist()) if hasattr(pref, "tolist") else tuple(pref)
            n = len(w)
            pq = [(i, i, (w[i],)) for i in range(n)]
            heapq.heapify(pq)
            while pq:
                idx_sum, max_idx, subset = heapq.heappop(pq)
                yield subset
                for j in range(max_idx + 1, n):
                    heapq.heappush(pq, (idx_sum + j, j, subset + (w[j],)))
        elif self.mode == PrefetchMode.MODE1 and rem is not None:
            w = tuple(rem.tolist()) if hasattr(rem, "tolist") else tuple(rem)
            for depth in range(1, len(w) + 1):
                for p in permutations(w, depth):
                    yield p

    def _dispatcher_loop(self):
        while True:
            self.request_event.wait()
            self.request_event.clear()
            with self.lock:
                args = self.fn_args
                if args is None or self.mode == PrefetchMode.OFF:
                    continue
                s, d = args
                dirty = set(self._dirty_batch)
                self._dirty_batch.clear()
                snapshots = []
                for b in dirty:
                    snapshots.append((
                        b,
                        tuple(self.histories[b]),
                        self.pref_seqs[b],
                        self.remaining[b],
                        self.full_orders[b],
                    ))

            for b, history, pref, rem, full_order in snapshots:
                if full_order is None:
                    continue
                full_list = full_order.tolist() if hasattr(full_order, 'tolist') else list(full_order)
                for sub_seq in self._gen(pref, rem):
                    sub_seq = tuple(sub_seq)
                    cache_key = history + sub_seq
                    stop = len(cache_key)
                    prefix_set = set(cache_key)
                    remaining = [v for v in full_list if v not in prefix_set]
                    order_array = list(cache_key) + remaining
                    _THREAD_SEMAPHORE.acquire()
                    with self.lock:
                        if cache_key not in self.caches[b]:
                            try:
                                self.caches[b][cache_key] = self.prefetch_executor.submit(
                                    local_tokenize, s, d,
                                    np.array(order_array, dtype=np.int32), stop,
                                )
                                self.background_keys[b].add(cache_key)
                            except RuntimeError as e:
                                _THREAD_SEMAPHORE.release()
                                if str(e) != "cannot schedule new futures after shutdown":
                                    raise

    def _prune_batch(self, b):
        """Prune cache for batch element b. Must be called with lock held."""
        to_remove = []
        h = tuple(self.histories[b])
        h_len = len(h)
        for key in list(self.caches[b].keys()):
            check_len = min(len(key), h_len)
            if check_len > 0 and key[:check_len] != h[:check_len]:
                to_remove.append(key)
        for k in to_remove:
            self.caches[b].pop(k, None)
            self.background_keys[b].discard(k)
            self.pruned_count += 1

    def get_or_submit(self, b, key, fn, args):
        """Look up or compute for batch element b."""
        with self.lock:
            # Update history for this batch element
            if len(key) > len(self.histories[b]):
                self.histories[b] = list(key)

            self._prune_batch(b)

            is_new = key not in self._seen_keys[b]
            if is_new:
                self._seen_keys[b].add(key)
                self.unique_requests += 1

            if key in self.caches[b]:
                if is_new and key in self.background_keys[b]:
                    self.prefetch_hits += 1
                return self.caches[b][key]
            else:
                fut = self.on_demand_executor.submit(fn, *args)
                self.caches[b][key] = fut
                return fut

    def prefetch(self, b, pref, rem, full_order, args, mode):
        """Trigger prefetch for a single batch element."""
        self.mode = mode
        self.fn_args = args
        self.full_orders[b] = full_order
        self.remaining[b] = rem
        pref_key = tuple(pref.tolist()) if hasattr(pref, 'tolist') else (tuple(pref) if pref is not None else None)
        if pref_key == self._last_prefs[b] and mode == PrefetchMode.MODE2:
            return
        self._last_prefs[b] = pref_key
        self.pref_seqs[b] = pref
        self._dirty_batch.add(b)
        self.request_event.set()

    def prefetch_all(self, prefs, rems, full_orders, args, mode):
        """Trigger prefetch for ALL batch elements at once."""
        self.mode = mode
        self.fn_args = args
        for b in range(self.batch_size):
            self.full_orders[b] = full_orders[b] if full_orders is not None else None
            self.remaining[b] = rems[b] if rems is not None else None
            pref = prefs[b] if prefs is not None else None
            pref_key = tuple(pref.tolist()) if hasattr(pref, 'tolist') else (tuple(pref) if pref is not None else None)
            if pref_key == self._last_prefs[b] and mode == PrefetchMode.MODE2:
                continue
            self._last_prefs[b] = pref_key
            self.pref_seqs[b] = pref
            self._dirty_batch.add(b)
        if self._dirty_batch:
            self.request_event.set()


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
            self.on_demand_executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)
            self.prefetch_executor = ThreadPoolExecutor(max_workers=os.cpu_count() // 2)
        self.manager = PrefetchManager(
            self.on_demand_executor, self.prefetch_executor, batch_size
        )

    def cleanup(self):
        self.on_demand_executor.shutdown(wait=False)
        self.prefetch_executor.shutdown(wait=False)

    def submit_task(self, b, s, d, order, stop):
        order_np = np.asarray(order)
        prefix = tuple(order_np[:int(stop)].tolist())
        return self.manager.get_or_submit(
            b, prefix, local_tokenize, (s, d, order_np, int(stop))
        )

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

        # Normalize pref_seqs to per-element list
        if pref_seqs is not None and not isinstance(pref_seqs, list):
            pref_seqs = [pref_seqs] * self.batch_size

        # Pre-seed prefetch BEFORE warmup — dispatcher works during JIT compilation
        if prefetch_mode != PrefetchMode.OFF and pref_seqs is not None:
            prefs = [ps[:10] if ps is not None else None for ps in pref_seqs]
            rems = [np.asarray(order[b]) for b in range(self.batch_size)]
            fulls = [np.asarray(order[b]) for b in range(self.batch_size)]
            self.manager.prefetch_all(prefs, rems, fulls, (static, dynamic), prefetch_mode)

        # Warmup (JIT compilation — dispatcher populates cache!)
        _ = step_jit(self, state, order[:, 0], graph, static, dynamic)
        jax.block_until_ready(_)

        start = time.perf_counter()
        for i in range(steps):
            if prefetch_mode != PrefetchMode.OFF:
                prefs = None
                if prefetch_mode == PrefetchMode.MODE2 and pref_seqs is not None:
                    prefs = [ps[i:i+10] if ps is not None else None for ps in pref_seqs]
                rems = [np.asarray(order[b, i:]) for b in range(self.batch_size)]
                fulls = [np.asarray(order[b]) for b in range(self.batch_size)]
                self.manager.prefetch_all(prefs, rems, fulls, (static, dynamic), prefetch_mode)

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
            self.manager.reset_telemetry()

            t = self.run_single(graph, static, dynamic, steps, prefetch_mode, pref_seq)
            times.append(t)

            total = self.manager.unique_requests
            hit_rates.append(self.manager.prefetch_hits / total if total > 0 else 0.0)

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
    cli = parser.parse_args()

    sizes_list = [s.strip() for s in cli.sizes.split(",")]
    batch_sizes = [int(b.strip()) for b in cli.batches.split(",")]

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
            for coord in Coordination:
                # 1. No Prefetch
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
