#!/usr/bin/env python3
"""Run the measure toolchain gate on THIS node and say what it found.

The verification tool for finding 03 / ticket dsnn-3qm.21, and the thing to
run on a node before trusting its measurements. It does exactly what the
first measure compile of every measuring process does -- ``env.
_measure_toolchain_check`` -- plus one CONTROL: a fixed (cacheable) module
compiled the same way (no compiler options, owner ruling 2026-09-26, Q5 b).
On a node whose link toolchain is broken, a warm persistent cache written on
a clean node makes the control pass (finding 03 sec 5a) while the gate still
refuses: that is the cache-immunity proof.

    python tools/toolchain_gate_probe.py [abort|warn|off]

Exit 0 = gate passed. 3 = gate aborted (the message names the node, the
ptxas / nvlink versions and the fault). Uses JAX_COMPILATION_CACHE_DIR as
set in the environment, so the caller decides whether the control is warm.
"""
from __future__ import annotations

import os
import socket
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "abort"
os.environ["ALPHAGRAD_MEASURE_TOOLCHAIN_GATE"] = mode
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

from alphagrad.approx import env                                # noqa: E402

print(f"[probe] host={socket.gethostname()} mode={mode} "
      f"backend={jax.default_backend()} "
      f"cache_dir={os.environ.get('JAX_COMPILATION_CACHE_DIR')} "
      f"PATH[0:2]={os.environ.get('PATH', '').split(':')[:2]}", flush=True)
print(f"[probe] ptxas (PATH): {env._tool_version('ptxas')}", flush=True)
print(f"[probe] ptxas (venv): {env._venv_ptxas_version()}", flush=True)
print(f"[probe] nvlink (PATH): {env._tool_version('nvlink')}", flush=True)
print(f"[probe] /usr/local/cuda -> {os.path.realpath('/usr/local/cuda')}",
      flush=True)


def _control(x):
    y = jnp.sin(x * 1.5)
    y = jnp.tanh(y) + 0.5
    y = y @ jnp.eye(y.shape[-1], dtype=y.dtype)
    return jnp.log1p(jnp.abs(y))


t0 = time.perf_counter()
try:
    jax.jit(_control).lower(jax.ShapeDtypeStruct((16, 16), jnp.float32)) \
        .compile()
    print(f"[probe] CONTROL (fixed, cacheable module) compiled OK in "
          f"{time.perf_counter() - t0:.2f}s", flush=True)
except Exception as exc:
    print(f"[probe] CONTROL FAILED in {time.perf_counter() - t0:.2f}s: "
          f"{' '.join(str(exc).split())[:300]}", flush=True)

t0 = time.perf_counter()
try:
    env._measure_toolchain_check()
except env.MeasureToolchainFault as exc:
    print(f"[probe] GATE ABORT in {time.perf_counter() - t0:.2f}s",
          flush=True)
    print(exc, flush=True)
    sys.exit(3)
print(f"[probe] GATE {'OK' if env._MEASURE_TOOLCHAIN['ok'] else 'WARNED'} "
      f"in {time.perf_counter() - t0:.2f}s: {env._MEASURE_TOOLCHAIN}",
      flush=True)
