"""Per-node persistent JAX compilation cache.

Single source of truth — `ppo.py`, `mu0.py`, `ppo_ray_worker.py`,
`cpu_approx_worker.py`, and any future trainer that touches JAX should
call `setup_jax_compile_cache()` from here. Three prior copies existed
with subtly different paths (some scoped by hostname, some not), which
let an NFS-homedir-shared cache trip the `cpu_aot_loader.cc:195`
warning when XLA loaded an AOT artefact compiled for a different CPU
generation.

Default cache dir is per-SLURM-job and per-node to match
`/tmp/dsnn-venv-${JOB_TAG}` from the sbatch scripts:

    /tmp/dsnn-jax-cache-${SLURM_JOB_ID}-${hostname}/

Outside SLURM (local dev), `SLURM_JOB_ID` is "local". Set
`DSNN_JAX_CACHE_REUSE=1` to opt into a cross-job, intra-node shared
cache at `${TMPDIR:-/tmp}/dsnn-jax-cache-shared-${hostname}/` — trades
correctness on heterogeneous job sequencing for a warmer cache.
"""

from __future__ import annotations

import os
import socket


SENTINEL_REWARD_VALUE: float = -1e10
"""Reward value the env / CPU pool emits when an `io_callback` or
`ray.get` hits the timeout path. Mirrored in `cpu_approx_pool._sentinel_callback_output`
and `env._callback`'s error branches. Reward channels that are
sentinels must be filtered out of calibration averages and replay
priorities — see `reward_scaling.filter_sentinel_mask`."""


def setup_jax_compile_cache() -> str:
    """Wire JAX's persistent disk cache to a node-local, job-scoped dir.

    Sets *both* `JAX_COMPILATION_CACHE_DIR` and
    `jax.config.update("jax_compilation_cache_dir", ...)` — the former
    is what XLA's `cpu_aot_loader.cc` checks for AOT-result lookup, the
    latter is what JAX's lowering pipeline checks before emitting a new
    artefact. Setting only one trips silent cache misses.

    Idempotent — re-calling in the same process is a no-op (the env-var
    is set via `setdefault`).
    """
    slurm_job = os.environ.get("SLURM_JOB_ID", "local")
    host = socket.gethostname().split(".", 1)[0]
    if os.environ.get("DSNN_JAX_CACHE_REUSE", "0") == "1":
        base = os.environ.get("TMPDIR", "/tmp")
        default = os.path.join(base, f"dsnn-jax-cache-shared-{host}")
    else:
        default = os.path.join("/tmp", f"dsnn-jax-cache-{slurm_job}-{host}")
    cache_dir = os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", default)
    os.makedirs(cache_dir, exist_ok=True)

    import jax
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    return cache_dir
