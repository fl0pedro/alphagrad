import os, time, uuid, threading, argparse, cloudpickle, numpy as np, jax, jax.numpy as jnp
from enum import Enum
from typing import NamedTuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import shared_memory
from jax.experimental import io_callback
import multiprocessing
from functools import partial
from graphax.core import extract_jaxpr
from alphagrad.vertexgame.fast_core import vertex_eliminate

class Backend(Enum):
    THREAD, PICKLE, SHM = "Threading", "Pooling (Pickle)", "Pooling (SHM)"
class Coordination(Enum):
    PURE_CALLBACK, IO_CALLBACK, ASYNC = "jax.pure_callback", "jax.experimental.io_callback", "Async Dispatcher"
class Prefetch(Enum):
    OFF, ON = "No Prefetch", "Prefetch"

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

from _cpu_workers import pickle_worker, shm_worker

_WORK_JAXPR_CACHE = {}
def get_jaxpr(s, d):
    if s.fn not in _WORK_JAXPR_CACHE: _WORK_JAXPR_CACHE[s.fn] = jax.make_jaxpr(s.fn)(*d.args).jaxpr
    return _WORK_JAXPR_CACHE[s.fn]

def local_tokenize(s, d, order, stop):
    jaxpr, ord_np = get_jaxpr(s, d), np.asarray(order)
    arg_np = jax.tree_util.tree_map(np.asarray, d.args)
    con_np = jax.tree_util.tree_map(np.asarray, d.consts)
    ve = extract_jaxpr(jaxpr, s.argnums, s.has_aux, ord_np, int(stop), s.sparse, arg_np, con_np)
    res = np.zeros(1024, dtype=np.int32)
    res[:min(len(ve.tokenized), 1024)] = ve.tokenized[:1024]
    return res

class PrefetchManager:
    def __init__(self, executor): self.executor, self.cache, self.lock = executor, {}, threading.Lock()
    def get_or_submit(self, k, fn, args):
        with self.lock:
            if k not in self.cache: self.cache[k] = self.executor.submit(fn, *args)
            return self.cache[k]

def benchmark_step(bench, state, action, graph, static, dynamic):
    def callback(o, s, d_args, d_consts):
        d_v = DynamicConfig(d_args, d_consts)
        tasks = [bench.submit_task(static, d_v, o[b], s[b]) for b in range(bench.batch_size)]
        return np.stack([t.result() for t in tasks])
    struct = jax.ShapeDtypeStruct((bench.batch_size, 1024), jnp.int32)
    cb_fn = jax.pure_callback if bench.coord == Coordination.PURE_CALLBACK else io_callback
    tokens = cb_fn(callback, struct, state.order, state.step_count, dynamic.args, dynamic.consts)
    new_edges, _ = jax.vmap(vertex_eliminate)(action, state.edges)
    return state._replace(tokens=tokens, step_count=state.step_count + 1, edges=new_edges)

class UnifiedBenchmark:
    def __init__(self, backend, coord, prefetch, batch_size):
        self.backend, self.coord, self.pref, self.batch_size = backend, coord, prefetch, batch_size
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count()*3//4) if backend == Backend.THREAD else ProcessPoolExecutor(max_workers=os.cpu_count()*3//4, mp_context=multiprocessing.get_context('spawn'))
        self.manager, self.shm_ctx = PrefetchManager(self.executor), None
    def setup_shm(self, s, d):
        data = cloudpickle.dumps((s, d)); self.shm_ctx = shared_memory.SharedMemory(create=True, size=len(data), name=f"/c_{uuid.uuid4().hex[:8]}")
        self.shm_ctx.buf.cast('B')[:len(data)] = data
    def cleanup(self):
        self.executor.shutdown(wait=False)
        if self.shm_ctx: self.shm_ctx.close(); self.shm_ctx.unlink()
    def submit_task(self, s, d, order, stop):
        key, ord_shm = tuple(order[int(stop):].tolist()), None
        if self.backend == Backend.THREAD: fn, args = local_tokenize, (s, d, order, stop)
        elif self.backend == Backend.PICKLE: fn, args = pickle_worker, (cloudpickle.dumps((s, d)), order, stop)
        else:
            buf = order.tobytes(); ord_shm = shared_memory.SharedMemory(create=True, size=len(buf), name=f"/o_{uuid.uuid4().hex[:8]}")
            ord_shm.buf.cast('B')[:len(buf)] = buf; fn, args = shm_worker, (self.shm_ctx.name, ord_shm.name, order.shape, stop)
        fut = self.manager.get_or_submit(key, fn, args) if self.pref == Prefetch.ON else self.executor.submit(fn, *args)
        if ord_shm:
            def clean(_):
                try: ord_shm.close(); ord_shm.unlink()
                except: pass
            fut.add_done_callback(clean)
        return fut
    def run(self, graph, static, dynamic, steps=5):
        num_v = int(graph.at[0, 0, 1].get())
        order, edges = jnp.tile(jnp.arange(1, num_v + 1, dtype=jnp.int32), (self.batch_size, 1)), jnp.broadcast_to(graph, (self.batch_size, *graph.shape))
        state, step_jit = EnvState(order, jnp.zeros((self.batch_size, 1024), jnp.int32), jnp.zeros(self.batch_size, jnp.int32), edges, num_v), jax.jit(benchmark_step, static_argnames=("bench", "static"))
        start = time.perf_counter()
        if self.coord == Coordination.ASYNC:
            for i in range(steps):
                actions = order[:, i]; nxt, ord_np = np.asarray(state.step_count) + 1, np.asarray(state.order)
                futs = [self.submit_task(static, dynamic, ord_np[b], nxt[b]) for b in range(self.batch_size)]
                tokens = jnp.stack([jnp.asarray(f.result()) for f in futs])
                state = step_jit(self, state, actions, graph, static, dynamic)
                state = state._replace(tokens=tokens)
        else:
            for i in range(steps): state = step_jit(self, state, order[:, i], graph, static, dynamic)
        state.tokens.block_until_ready()
        return (time.perf_counter() - start) / steps

def main():
    parser = argparse.ArgumentParser(); 
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--batches", type=str, default="8,16,32,64,128,256")
    cli = parser.parse_args(); 
    if cli.size == "small": n, compl, steps = (10, 5, 5)
    elif cli.size == "medium": n, compl, steps = (100, 20, 10)
    else: n, compl, steps = (500, 40, 10)
    
    batch_sizes = [int(b.strip()) for b in cli.batches.split(",")]
    from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph
    def f(x):
        for _ in range(compl): x = jnp.sin(jnp.cos(x)) * x
        return jnp.sum(x)
    args_in = (jnp.ones((n,)),); graph = make_graph(f, *args_in)
    static, dynamic = StaticConfig(f, (0,), False, False), DynamicConfig(args_in, jax.make_jaxpr(f)(*args_in).literals)
    results = {}
    for coord in Coordination:
        for pref in Prefetch:
            for back in Backend:
                name = f"{coord.value} | {pref.value} | {back.value}"; print(f"Benchmarking: {name}"); row = []
            for b in batch_sizes:
                bench = UnifiedBenchmark(back, coord, pref, b)
                try: row.append(bench.run(graph, static, dynamic, steps))
                except Exception as e: print(f"  Error {b}: {e}"); row.append(float('nan'))
                finally: bench.cleanup()
            results[name] = row
    header = f"{'Configuration':<60} | " + " | ".join([f"B={b:<7}" for b in batch_sizes])
    print("\n" + header + "\n" + "-" * len(header))
    for name, times in results.items(): print(f"{name:<60} | " + " | ".join([f"{t:.4f}s" for t in times]))

if __name__ == "__main__": main()
