"""Unit tests for ``alphagrad.approx.common.compile_cache.setup_jax_compile_cache``.

The helper is the single source of truth for the JAX persistent disk
cache across `ppo.py`, `mu0.py`, the Ray PPO worker, and the CPU
approx worker. It must:

* Default to a per-SLURM-job, per-node ``/tmp/dsnn-jax-cache-...`` path
  so cross-node CPU-feature mismatches don't trip the
  ``cpu_aot_loader.cc:195`` warning.
* Honour ``JAX_COMPILATION_CACHE_DIR`` if the caller has set it
  explicitly (env var takes precedence).
* Switch to the cross-job, intra-node opt-in path when
  ``DSNN_JAX_CACHE_REUSE=1``.
* Be idempotent — re-calling in the same process is a no-op (env-var
  uses ``setdefault``).
* Set BOTH the env var (XLA's ``cpu_aot_loader.cc:195`` only checks
  this) AND ``jax.config.update`` (JAX's lowering pipeline checks this).
"""

from __future__ import annotations

import os
import socket

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Each test sees a clean env so we don't leak state across tests."""
    for k in ("JAX_COMPILATION_CACHE_DIR", "DSNN_JAX_CACHE_REUSE",
              "SLURM_JOB_ID", "TMPDIR"):
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_path_uses_slurm_job_id(monkeypatch):
    """Without env-var override, the helper builds
    ``/tmp/dsnn-jax-cache-${SLURM_JOB_ID}-${hostname}/``."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    from alphagrad.approx.common.compile_cache import setup_jax_compile_cache

    host = socket.gethostname().split(".", 1)[0]
    expected = f"/tmp/dsnn-jax-cache-12345-{host}"

    got = setup_jax_compile_cache()
    try:
        assert got == expected
        assert os.environ["JAX_COMPILATION_CACHE_DIR"] == expected
    finally:
        # Don't leave the test's choice of cache dir wired into JAX
        # for subsequent tests in the same process.
        if os.path.isdir(expected):
            try:
                os.rmdir(expected)
            except OSError:
                pass


def test_env_var_takes_precedence(monkeypatch, tmp_path):
    """If the caller pre-sets JAX_COMPILATION_CACHE_DIR, the helper
    respects it (uses ``setdefault``)."""
    custom = str(tmp_path / "custom-cache")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", custom)
    monkeypatch.setenv("SLURM_JOB_ID", "67890")

    from alphagrad.approx.common.compile_cache import setup_jax_compile_cache

    got = setup_jax_compile_cache()
    assert got == custom
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == custom


def test_reuse_flag_uses_shared_path(monkeypatch, tmp_path):
    """``DSNN_JAX_CACHE_REUSE=1`` switches to the cross-job intra-node
    path."""
    monkeypatch.setenv("DSNN_JAX_CACHE_REUSE", "1")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "99999")

    from alphagrad.approx.common.compile_cache import setup_jax_compile_cache

    got = setup_jax_compile_cache()
    host = socket.gethostname().split(".", 1)[0]
    assert got == str(tmp_path / f"dsnn-jax-cache-shared-{host}")
    # SLURM_JOB_ID is NOT in the path when REUSE=1.
    assert "99999" not in got


def test_idempotent(monkeypatch, tmp_path):
    """Re-calling in the same process should be a no-op (env-var sticks)."""
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "fixed"))

    from alphagrad.approx.common.compile_cache import setup_jax_compile_cache

    a = setup_jax_compile_cache()
    b = setup_jax_compile_cache()
    assert a == b == str(tmp_path / "fixed")


def test_sentinel_value_constant():
    """The sentinel value is exposed for downstream filtering and must
    match the ``cpu_approx_pool.py`` magic."""
    from alphagrad.approx.common.compile_cache import SENTINEL_REWARD_VALUE
    assert SENTINEL_REWARD_VALUE == -1e10
