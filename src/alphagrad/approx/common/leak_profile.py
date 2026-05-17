"""Per-process RSS / tracemalloc / jax.live_arrays sampler.

Single source of truth for the leak instrumentation that lives in
both the CPU approximation workers and the GPU trainer. Gated by
``ALPHAGRAD_LEAK_PROFILE=1`` so production runs aren't taxed.

Each process emits ``slurm/logs/${SLURM_JOB_ID}/leak-<role>-<pid>.log``
(or ``/tmp/leak-<role>-<pid>.log`` if slurm dir creation fails). The
log header carries the PID, cmdline, and role so worker vs. trainer
samples are unambiguous when cross-referenced — the user's confusion
about *where* the JIT compile leak lives was the trigger for splitting
the role out.

The expensive samples (tracemalloc snapshot diff, jax.live_arrays
sweep) only fire every ``ALPHAGRAD_LEAK_PROFILE_EVERY`` calls (default
100); cheap RSS reads are unconditional.
"""

from __future__ import annotations

import os
import time
from typing import Any


def _read_cmdline() -> str:
    """Return ``/proc/self/cmdline`` as a single space-separated string.

    Used in the log header so a downstream reader can distinguish a
    ``CpuApproximationActor`` process from the PPO/SPMD trainer
    process from a shell wrapper. Falls back to ``sys.argv[0]``.
    """
    try:
        with open("/proc/self/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        return raw.strip() or "<no-cmdline>"
    except OSError:
        import sys
        return sys.argv[0] if sys.argv else "<no-argv>"


def _rss_bytes() -> int:
    """Process RSS in bytes via ``/proc/self/status``. 0 on non-Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class LeakProfile:
    """Per-process leak sampler.

    Parameters
    ----------
    role
        Short tag (``"actor"``, ``"trainer"``, …) baked into the log
        filename so cross-process diff is easy. The corresponding
        log path is ``slurm/logs/${SLURM_JOB_ID}/leak-${role}-${pid}.log``.
    every
        Sampling interval (in ``record_call`` invocations). Default
        100 — cheap enough to leave on for an entire run.
    """

    def __init__(self, role: str, every: int | None = None):
        import tracemalloc

        self._tm = tracemalloc
        self._tm.start(25)
        if every is None:
            every = int(os.environ.get("ALPHAGRAD_LEAK_PROFILE_EVERY", "100"))
        self._every = max(1, int(every))

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        log_dir = os.path.join("slurm", "logs", str(job_id))
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(
                log_dir, f"leak-{role}-{os.getpid()}.log",
            )
        except OSError:
            log_path = f"/tmp/leak-{role}-{os.getpid()}.log"
        self._log_path = log_path

        self._snapshot_prev = None
        self._t0 = time.monotonic()
        self._rss0 = _rss_bytes()
        self._role = role
        self._n_records = 0

        with open(log_path, "w") as f:
            f.write(
                f"# leak-profile role={role} pid={os.getpid()} "
                f"slurm_job={job_id} t0={self._t0:.0f} rss0={self._rss0} "
                f"every={self._every}\n"
                f"# cmd: {_read_cmdline()}\n"
                f"# columns: n_call wall_s rss_mb d_rss_mb "
                f"jax_arrays jax_mb top_allocator\n"
            )

    @staticmethod
    def rss_bytes() -> int:
        """Public accessor — call sites occasionally want RSS without
        an associated call counter (e.g. an episode-boundary trainer
        sample)."""
        return _rss_bytes()

    def record_call(self, n_call: int) -> None:
        """Sample at call ``n_call``. No-op except every ``self._every``
        calls so the hot path stays cheap."""
        if n_call % self._every != 0:
            return
        rss_now = _rss_bytes()
        wall = time.monotonic() - self._t0

        # JAX live arrays — cheap (list of weakrefs, iterated once).
        try:
            import jax
            live = jax.live_arrays()
            n_live = len(live)
            mb_live = sum(int(a.nbytes) for a in live) / (1024 * 1024)
        except Exception:
            n_live, mb_live = -1, -1.0

        # Per-process compile-cache counters from the shared cache
        # client (only meaningful in actor processes; trainer never
        # calls ``cached_compile``). The coordinator's cluster-wide
        # stats are a separate ``coord.stats.remote()`` call — too
        # heavy for the hot path, so the driver polls those instead.
        cache_stats_str = ""
        try:
            from alphagrad.approx.common.compile_cache import local_stats
            ls = local_stats()
            total = ls["local_hits"] + ls["local_misses"]
            rate = (ls["local_hits"] / total) if total else 0.0
            cache_stats_str = (
                f" cache=hits/{ls['local_hits']} miss/{ls['local_misses']} "
                f"err/{ls['local_errors']} rate/{rate:.2f}"
            )
        except Exception:
            pass

        # Tracemalloc top diff vs prior snapshot.
        snap = self._tm.take_snapshot()
        top_str = "-"
        if self._snapshot_prev is not None:
            stats = snap.compare_to(self._snapshot_prev, "lineno")
            if stats:
                top = stats[0]
                top_str = (
                    f"{top.traceback.format()[-1].strip()[:80]} "
                    f"d_size={top.size_diff / 1024:+.0f}KB"
                )
        self._snapshot_prev = snap

        line = (
            f"{n_call} {wall:.1f} "
            f"{rss_now / 1024 / 1024:.0f} "
            f"{(rss_now - self._rss0) / 1024 / 1024:+.0f} "
            f"{n_live} {mb_live:.0f} "
            f"\"{top_str}\"{cache_stats_str}\n"
        )
        try:
            with open(self._log_path, "a") as f:
                f.write(line)
        except OSError:
            pass
        self._n_records += 1


def maybe_install(role: str) -> LeakProfile | None:
    """Return a fresh :class:`LeakProfile` if ``ALPHAGRAD_LEAK_PROFILE=1``,
    else ``None``. Catches construction failures so a buggy profiler
    can't take training down."""
    if os.environ.get("ALPHAGRAD_LEAK_PROFILE", "0") != "1":
        return None
    try:
        return LeakProfile(role)
    except Exception as exc:
        print(f"[leak_profile] init failed (role={role}): {exc}")
        return None
