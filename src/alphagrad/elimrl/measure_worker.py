"""alphagrad.elimrl.measure_worker -- M0: isolated, restartable real-measurement
worker.

The parent-side :class:`MeasureClient` spawns ``python -m
alphagrad.elimrl.measure_worker`` as a plain SUBPROCESS (no Ray) and speaks
one-JSON-object-per-line over the worker's stdin/stdout. Right after startup
the worker re-points fd 1 at stderr and keeps a private dup for the protocol,
so stray prints from jax or target builders can never corrupt the stream.
This module itself imports ONLY the stdlib in the parent; jax/graphax/
alphagrad are imported inside the worker process.

Measurement protocol (patterns follow alphagrad/src/alphagrad/approx/env.py --
used as a REFERENCE, deliberately not imported):

  latency    per rep: ``inner`` (default 5) dispatches, ONE block_until_ready,
             divide by inner. Primary statistic = MEDIAN over reps
             (``latency_ns``); a 20% winsorised mean is returned alongside
             (``latency_ns_winsor``) plus the raw samples.
  memory     objective = ``compiled.memory_analysis()`` temp+output bytes
             (deterministic, CV 0 by construction). The runtime
             peak_bytes_in_use delta (barrier -> clear_memory_stats ->
             barrier -> base -> run -> peak - base) is returned as
             ``peak_delta_bytes`` cross-check where the backend exposes
             allocator stats (CPU raises UNIMPLEMENTED -> None).
  compile    per-executable compiler options on GPU:
             {"xla_gpu_autotune_level": 0, "xla_gpu_enable_triton_gemm": False}.
  OOM class  r"allocate ([0-9.]+)\\s*([KMGT])iB" parsed from exception text;
             allocation failures score INFEASIBLE, not error.

Guards (all demonstrated by tests/elimrl/worker_test.py):
  * predicted-memory pre-check: if memory_analysis temp+output exceeds
    ``budget_bytes``, the plan is scored infeasible WITHOUT executing
    (``executed`` stays False).
  * hard timeout (parent side): kill + respawn + status "infeasible",
    reason "timeout".
  * worker crash (CUDA_ERROR_ILLEGAL_ADDRESS class): parent sees EOF,
    respawns, scores "infeasible"/"worker_died"; the NEXT measurement runs on
    the fresh worker.

Request (cmd="measure"):
    {"cmd": "measure",
     "target": {"builder": "module:attr", "kwargs": {...}},   # -> (fn, args, argnums)
     "method": "jacve" | "jacfe" | "elim_plan" | "jax.grad" | "jax.jacrev",
     "order":  "rev" | "fwd" | [ints],          # jacve; "rev" policy for jacfe
     "plan":   [["F", u, w], ["V", j], ...],    # elim_plan (env.history)
     "vertex_only": false,                      # elim_plan replay flag
     "budget_bytes": 4e10 | null,
     "reps": 10, "inner": 5,
     "check_against": {"method": ..., "order": ...} | null}

Response: {"status": "ok"|"infeasible"|"error", "executed": bool,
           "latency_ns", "latency_ns_winsor", "latency_samples_ns",
           "mem_temp_bytes", "mem_output_bytes", "mem_argument_bytes",
           "mem_total_bytes", "peak_delta_bytes", "compile_s", "backend",
           "reason", "oom_bytes", "check_maxdiff", "check_cos", ...}
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

_OOM_RE = re.compile(r"allocate ([0-9.]+)\s*([KMGT])iB")
_UNITS = {"K": 2 ** 10, "M": 2 ** 20, "G": 2 ** 30, "T": 2 ** 40}


def _oom_bytes(msg: str):
    m = _OOM_RE.search(msg)
    return float(m.group(1)) * _UNITS[m.group(2)] if m else None


# ===========================================================================
# worker side (runs in the subprocess; jax imported here only)
# ===========================================================================
def _resolve_target(spec):
    """spec = {"builder": "module:attr", "kwargs": {...}} -> (fn, args, argnums)."""
    import importlib
    mod, _, attr = spec["builder"].partition(":")
    builder = getattr(importlib.import_module(mod), attr)
    return builder(**spec.get("kwargs", {}))


def _build_jac_callable(req, fn, argnums):
    import jax
    method = req["method"]
    if method == "jax.grad":
        return jax.grad(fn, argnums=argnums)
    if method == "jax.jacrev":
        return jax.jacrev(fn, argnums=argnums)
    if method == "jacve":
        from graphax import jacve
        order = req.get("order", "rev")
        order = order if isinstance(order, str) else [int(o) for o in order]
        return jacve(fn, order, argnums=argnums)
    if method == "jacfe":
        # face engine driven by the built-in reverse policy; the per-step
        # choice is structural (topo rank + graph shape), so it is stable
        # under tracing and across processes.
        from alphagrad.elimrl.env import ElimEnv, rev_policy
        if req.get("order", "rev") != "rev":
            raise ValueError("jacfe: only the 'rev' policy is wired in M0")

        def jacf(*xs):
            env = ElimEnv(fn, xs, argnums)
            while not env.done:
                env.apply(rev_policy(env))
            return tuple(j for j in env.jacobian() if j is not None)

        return jacf
    if method == "elim_plan":
        from alphagrad.elimrl.env import ElimEnv
        plan = [tuple(a) for a in req["plan"]]
        vertex_only = bool(req.get("vertex_only", False))

        def jacp(*xs):
            env = ElimEnv(fn, xs, argnums, vertex_only=vertex_only)
            for a in plan:
                env.apply(a)
            if not env.done:
                raise ValueError(
                    "elim_plan did not terminate the elimination "
                    f"({len(env.faces())} faces left)")
            return tuple(j for j in env.jacobian() if j is not None)

        return jacp
    raise ValueError(f"unknown method {method!r}")


def _compile(jitted, args):
    import jax
    lowered = jitted.lower(*args)
    if jax.default_backend() == "gpu":
        try:
            return lowered.compile(compiler_options={
                "xla_gpu_autotune_level": 0,
                "xla_gpu_enable_triton_gemm": False,
            })
        except Exception as e:  # pragma: no cover - build-specific
            print(f"[worker] compiler_options rejected ({type(e).__name__}: {e});"
                  " falling back to a plain compile", file=sys.stderr, flush=True)
    return lowered.compile()


def _mem_fields(compiled) -> dict:
    out = {}
    try:
        ma = compiled.memory_analysis()
    except Exception:
        ma = None
    for key, attr in (("mem_temp_bytes", "temp_size_in_bytes"),
                      ("mem_output_bytes", "output_size_in_bytes"),
                      ("mem_argument_bytes", "argument_size_in_bytes"),
                      ("mem_alias_bytes", "alias_size_in_bytes")):
        out[key] = (float(getattr(ma, attr, 0) or 0.0)
                    if ma is not None else None)
    out["mem_total_bytes"] = (
        (out["mem_temp_bytes"] or 0.0) + (out["mem_output_bytes"] or 0.0)
        if ma is not None else None)
    return out


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _winsorized_mean(xs, frac=0.2):
    """Mean with the lowest/highest ``frac`` clamped to the boundary values
    (the reference stack's 20% winsorised aggregation)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    k = int(n * frac)
    lo, hi = s[k], s[n - 1 - k]
    return sum(min(max(x, lo), hi) for x in s) / n


def _measure(req) -> dict:
    import jax

    t0 = time.perf_counter()
    fn, args, argnums = _resolve_target(req["target"])
    f = _build_jac_callable(req, fn, tuple(argnums))
    compiled = _compile(jax.jit(f), args)
    compile_s = time.perf_counter() - t0

    res = {"status": "ok", "executed": False,
           "backend": jax.default_backend(), "compile_s": compile_s}
    res.update(_mem_fields(compiled))

    # -- predicted-memory pre-check: score infeasible WITHOUT executing ------
    budget = req.get("budget_bytes")
    if (budget is not None and res["mem_total_bytes"] is not None
            and res["mem_total_bytes"] > float(budget)):
        res.update(status="infeasible", reason="predicted_memory",
                   budget_bytes=float(budget))
        return res

    reps = int(req.get("reps", 10))
    inner = int(req.get("inner", 5))
    devices = list(jax.local_devices())

    out = compiled(*args)                      # warmup / first-touch
    jax.block_until_ready(out)

    # runtime-peak cross-check window (allocator stats; CPU backends expose
    # clear_memory_stats but raise UNIMPLEMENTED -> peak_delta_bytes = None).
    have_stats = True
    base = 0.0
    try:
        jax.effects_barrier()
        for d in devices:
            d.clear_memory_stats()
        jax.effects_barrier()
        base = sum(float((d.memory_stats() or {}).get("bytes_in_use", 0.0))
                   for d in devices)
    except Exception:
        have_stats = False

    samples = []
    for _ in range(reps):
        t1 = time.perf_counter()
        for _k in range(inner):
            out = compiled(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t1) / inner * 1e9)   # -> ns

    peak_delta = None
    if have_stats:
        try:
            peak = sum(float((d.memory_stats() or {}).get(
                "peak_bytes_in_use", 0.0)) for d in devices)
            peak_delta = max(0.0, peak - base)
        except Exception:
            peak_delta = None

    res.update(executed=True,
               latency_ns=_median(samples),
               latency_ns_winsor=_winsorized_mean(samples, 0.2),
               latency_samples_ns=samples,
               peak_delta_bytes=peak_delta,
               reps=reps, inner=inner)

    # -- optional numeric cross-check against a reference method -------------
    chk = req.get("check_against")
    if chk:
        import numpy as np
        import jax.tree_util as jtu
        ref_req = dict(req)
        ref_req.update(chk)
        ref_req.pop("check_against", None)
        fr = _build_jac_callable(ref_req, fn, tuple(argnums))
        ref_out = jax.jit(fr)(*args)
        a = np.concatenate([np.asarray(x, dtype=np.float64).ravel()
                            for x in jtu.tree_leaves(out)] or [np.zeros(0)])
        b = np.concatenate([np.asarray(x, dtype=np.float64).ravel()
                            for x in jtu.tree_leaves(ref_out)] or [np.zeros(0)])
        if a.size == b.size:
            res["check_maxdiff"] = float(np.max(np.abs(a - b))) if a.size else 0.0
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            res["check_cos"] = float(a @ b / den) if den > 0 else 0.0
        else:
            res["check_error"] = f"leaf-size mismatch {a.size} vs {b.size}"
    return res


def _worker_main():  # pragma: no cover - exercised via subprocess in tests
    # Protocol stream hygiene: keep a private dup of fd 1 for JSON responses
    # and re-point fd 1 at stderr so ANY stray print lands off-protocol.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    import jax  # heavy import inside the client's startup timeout window
    proto.write(json.dumps({"status": "ready",
                            "backend": jax.default_backend(),
                            "pid": os.getpid()}) + "\n")
    proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            proto.write(json.dumps(
                {"status": "error", "reason": f"bad request: {e}"}) + "\n")
            proto.flush()
            continue
        cmd = req.get("cmd", "measure")
        if cmd == "ping":
            resp = {"status": "ok", "pong": True}
        elif cmd == "shutdown":
            proto.write(json.dumps({"status": "ok", "bye": True}) + "\n")
            proto.flush()
            return
        elif cmd == "crash":
            # test/demo hook: simulate the CUDA_ERROR_ILLEGAL_ADDRESS class of
            # hard worker death (no response, process gone).
            proto.flush()
            os._exit(int(req.get("code", 139)))
        elif cmd == "sleep":
            # test/demo hook for the parent-side hard timeout.
            time.sleep(float(req.get("seconds", 1.0)))
            resp = {"status": "ok", "slept": float(req.get("seconds", 1.0))}
        elif cmd == "measure":
            try:
                resp = _measure(req)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                ob = _oom_bytes(str(e))
                if ob is not None or "RESOURCE_EXHAUSTED" in str(e):
                    resp = {"status": "infeasible", "reason": "oom",
                            "executed": False, "oom_bytes": ob,
                            "detail": msg[:500]}
                else:
                    resp = {"status": "error", "reason": msg[:1000]}
        else:
            resp = {"status": "error", "reason": f"unknown cmd {cmd!r}"}
        proto.write(json.dumps(resp) + "\n")
        proto.flush()


# ===========================================================================
# parent side (stdlib only -- never imports jax)
# ===========================================================================
class MeasureClient:
    """Owns one measurement-worker subprocess; kills + respawns it on timeout
    or crash and scores the offending plan infeasible. Safe to keep using
    after any failure -- the next request runs on a fresh worker."""

    def __init__(self, env: dict | None = None, python: str | None = None,
                 startup_timeout: float = 900.0,
                 default_timeout: float = 1800.0):
        self._env_overrides = dict(env or {})
        self._python = python or sys.executable
        self._startup_timeout = float(startup_timeout)
        self._default_timeout = float(default_timeout)
        self.respawns = -1          # first _spawn brings it to 0
        self.backend = None
        self._proc = None
        self._q = None
        self._spawn()

    # -- lifecycle -----------------------------------------------------------
    def _spawn(self):
        env = dict(os.environ)
        env.update(self._env_overrides)
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            [self._python, "-m", "alphagrad.elimrl.measure_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,            # NEVER devnull stderr -- keep diagnostics
            env=env, text=True, bufsize=1)
        self._q = queue.Queue()
        threading.Thread(target=self._reader,
                         args=(self._proc.stdout, self._q),
                         daemon=True).start()
        self.respawns += 1
        try:
            ready = self._q.get(timeout=self._startup_timeout)
        except queue.Empty:
            self._kill()
            raise RuntimeError("measure worker did not become ready within "
                               f"{self._startup_timeout}s")
        if ready is None:
            rc = self._proc.poll()
            raise RuntimeError(f"measure worker died during startup (rc={rc})")
        ready = json.loads(ready)
        if ready.get("status") != "ready":
            raise RuntimeError(f"unexpected worker handshake: {ready!r}")
        self.backend = ready.get("backend")

    @staticmethod
    def _reader(stream, q):
        try:
            for line in stream:
                q.put(line)
        except Exception:
            pass
        q.put(None)                 # EOF sentinel

    def _kill(self):
        try:
            self._proc.kill()
            self._proc.wait(timeout=30)
        except Exception:
            pass

    def restart(self):
        self._kill()
        self._spawn()

    def close(self):
        try:
            self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=10)
        except Exception:
            self._kill()

    # -- request/response ------------------------------------------------------
    def request(self, req: dict, timeout: float | None = None) -> dict:
        """Send one request; enforce the hard timeout; survive crashes.
        Timeout and worker death are scored status="infeasible" (reasons
        "timeout" / "worker_died") and the worker is respawned either way."""
        timeout = self._default_timeout if timeout is None else float(timeout)
        if self._proc.poll() is not None:
            self._spawn()
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            rc = self._proc.poll()
            self._spawn()
            return {"status": "infeasible", "reason": "worker_died",
                    "executed": False, "returncode": rc}
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            self._spawn()
            return {"status": "infeasible", "reason": "timeout",
                    "executed": False, "timeout_s": timeout}
        if line is None:
            rc = self._proc.poll()
            self._spawn()
            return {"status": "infeasible", "reason": "worker_died",
                    "executed": False, "returncode": rc}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"status": "error",
                    "reason": f"undecodable worker line: {line[:200]!r}"}

    def measure(self, target: dict, method: str,
                timeout: float | None = None, **kw) -> dict:
        req = {"cmd": "measure", "target": target, "method": method}
        req.update(kw)
        return self.request(req, timeout=timeout)

    def ping(self, timeout: float = 60.0) -> bool:
        return self.request({"cmd": "ping"}, timeout=timeout).get("pong", False)


if __name__ == "__main__":
    _worker_main()
