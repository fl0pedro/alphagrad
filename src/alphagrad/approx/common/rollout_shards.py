"""Data parallelism over environments: one rollout shard per GPU.

WHY THIS MODULE EXISTS, and what the runtime actually does.

The trainer rolls out `E` environments on ONE device while the other GPUs of
the node serve a few seconds of measurement per episode. The obvious fix is
to give every GPU its own shard of environments. The obstacle is not the
arithmetic -- the shards are independent -- it is the HOST. Every decision of
every environment calls back into Python: the tokenizer, the face enumerator,
the legality oracle. Three facts about that, all MEASURED on the campaign node
(jobs 65801, 65805, 65806 on pgi15-gpu19, eight Blackwells):

1. A `jax.pure_callback` runs on the thread that DISPATCHED the program, not
   on a runtime thread of its own. Dispatching eight programs from one Python
   thread therefore executes their callbacks one after another, and the eight
   rollouts do not overlap at all: eight one-second callbacks took 8.01 s.
2. Dispatching each program from its OWN Python thread makes the callbacks
   concurrent: the same eight took 1.01 s on eight distinct threads. So the
   shards have to be dispatched from a thread each. That is what
   :func:`dispatch` does.
3. Concurrency is therefore REAL, and the host state the callbacks share --
   the face enumerator's prefix cache, the oracle memo, the plan log, the
   per-episode counters -- is not protected by anything. Two of those are
   caches whose corruption produces WRONG TOKENS rather than a wrong number.

So this module provides the two things that make shard concurrency safe:

* :data:`HOST_LOCK` and :func:`host_serial`, which serialise the host
  callbacks that own mutable caches. The GIL already serialises pure Python,
  so the lock costs throughput only where a callback releases it, and the
  callbacks it wraps do not.
* :class:`ShardGather`, a rendezvous that turns the N shards' per-step
  measurement callbacks into ONE batched call over all `N*E` environment rows,
  in global environment order, and hands each shard back its own slice.

THE GATHER IS NOT AN OPTIMISATION, IT IS REQUIRED. `CpuApproxPool._pick()`
takes actors off a free list and `_evaluate_batch_impl` sentinels every slot
it could not get an actor for ("pool-drained", `cpu_approx_pool.py`). Eight
concurrent `evaluate_batch` calls against seven actors would leave seven of
the eight with no actor at all and sentinel their rows -- a reward of -1e10
for environments that were measured perfectly well. One gathered call keeps
the pool's wave scheduler the only thing deciding how the rows are placed,
which is what it was written to be, and keeps the number of pool round trips
per step at ONE, exactly as it is without shards.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

__all__ = [
    "HOST_LOCK",
    "host_serial",
    "ShardGather",
    "ShardGatherTimeout",
    "resolve_devices",
    "parse_shard_devices",
    "shard_env_range",
    "dispatch",
    "concat_shards",
]


# ---------------------------------------------------------------- the lock

#: Serialises host callbacks that own mutable state. Reentrant, because a
#: callback may call another one (the face driver's edge table calls into the
#: same enumerator the chunk callback used).
HOST_LOCK = threading.RLock()


def host_serial(fn):
    """Run `fn` under :data:`HOST_LOCK`.

    Applied UNCONDITIONALLY, not only when shards are on. An uncontended
    reentrant lock costs about a hundred nanoseconds against callbacks that
    cost milliseconds, and a gate would be one more thing that can be wired
    wrong. It changes no number: it changes only who may run at once.
    """
    def _wrapped(*a, **k):
        with HOST_LOCK:
            return fn(*a, **k)
    _wrapped.__name__ = getattr(fn, "__name__", "host_serial")
    _wrapped.__doc__ = getattr(fn, "__doc__", None)
    _wrapped.__wrapped__ = fn
    return _wrapped


# ------------------------------------------------------------- the devices

def parse_shard_devices(spec, n_shards):
    """`"0,0,1"` -> `[0, 0, 1]`, one device INDEX per shard.

    Empty or None means the default, one shard per device in order. A list
    that names one device twice puts two shards ON THAT DEVICE, which is the
    owner's item 5 of 2026-09-15: two half-batch shards interleaved on one
    GPU, so that the device work of one half runs while the other half is on
    the host. The shards are separate threads and separate traces, so nothing
    else about them changes.
    """
    if spec is None or not str(spec).strip():
        return None
    out = []
    for piece in str(spec).replace(" ", "").split(","):
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError:
            raise ValueError(
                f"--rollout-shard-devices takes a comma separated list of "
                f"device indices, one per shard; {piece!r} is not an "
                f"integer.") from None
    if len(out) != int(n_shards):
        raise ValueError(
            f"--rollout-shard-devices names {len(out)} devices and "
            f"--rollout-shards asks for {int(n_shards)} shards. There has to "
            f"be exactly one device per shard.")
    return out


def resolve_devices(n_shards, devices=None, mapping=None):
    """The device each rollout shard runs on.

    Without `mapping` that is `devices[0..n-1]`, one shard per device in
    order. Shard 0 keeps device 0, which is the trainer's own device, so a run
    with one shard places exactly what it placed before.

    With `mapping` it is `[devices[i] for i in mapping]`, which may name one
    device more than once. See :func:`parse_shard_devices`.
    """
    n = int(n_shards)
    if n < 1:
        raise ValueError(
            f"--rollout-shards must be at least 1, got {n_shards}.")
    if devices is None:
        import jax
        devices = jax.local_devices()
    devices = list(devices)
    if mapping is not None:
        if len(mapping) != n:
            raise ValueError(
                f"--rollout-shard-devices names {len(mapping)} devices for "
                f"{n} shards.")
        bad = [i for i in mapping if not (0 <= int(i) < len(devices))]
        if bad:
            raise ValueError(
                f"--rollout-shard-devices names device index or indices "
                f"{bad}, and this process sees {len(devices)} devices: "
                f"{devices}.")
        return [devices[int(i)] for i in mapping]
    if n > len(devices):
        raise ValueError(
            f"--rollout-shards {n} needs {n} devices and this process sees "
            f"{len(devices)}: {devices}. A shard per GPU is the point; ask "
            f"for no more shards than the job holds, or name the placement "
            f"with --rollout-shard-devices.")
    return devices[:n]


def shard_env_range(shard, n_shards, envs_per_shard):
    """`(lo, hi)`: the GLOBAL environment indices shard `shard` rolls out.

    The block is contiguous and the blocks tile `0..n*E-1` in shard order, so
    a global environment index names one shard and one row inside it, and the
    concatenation of the shards' outputs is in global environment order
    without any permutation. `env_index` is that global index, which is what
    keeps it unique across shards and what makes a stream row's index its own
    environment's.
    """
    s, n, e = int(shard), int(n_shards), int(envs_per_shard)
    if not (0 <= s < n):
        raise ValueError(f"shard {s} is not in 0..{n - 1}")
    if e < 1:
        raise ValueError(f"envs per shard must be positive, got {e}")
    return s * e, (s + 1) * e


# ------------------------------------------------------------- the gather

class ShardGatherTimeout(RuntimeError):
    """A shard did not reach the rendezvous in time."""


class ShardGather:
    """Turn N per-shard host callbacks into ONE call over all the rows.

    Every shard's wrapped callback deposits its own `E` environment rows and
    waits. The LAST shard to arrive concatenates the rows in shard order --
    which is global environment order, see :func:`shard_env_range` -- calls
    the real host function once, and wakes the others; each then takes its own
    `E` rows out of the result.

    WHAT IT KEEPS UNCHANGED. The host function it calls is the same batched
    callback a run without shards calls, with `N*E` rows instead of `E`. The
    per-environment tokenization inside it is untouched, the rows are in the
    same order, the measurement pool sees ONE `evaluate_batch` per step as it
    always did, and the terminal step makes ONE submission under ONE ticket.

    WHAT IT ASSUMES, and checks. Every shard runs the same program for the
    same number of steps, so every shard reaches every round. A shard that
    dies before its rendezvous would leave the others waiting, so the wait has
    a timeout and raises rather than hanging for ever; the first episode's
    wait legitimately covers the other shards' COMPILATION, which is why the
    timeout is minutes rather than seconds and why a long wait logs a line
    instead of failing.
    """

    #: Each shard's per-round wait, seconds. The first round of the first
    #: episode waits out the other shards' tracing and compilation, which is
    #: minutes at the campaign's width.
    DEFAULT_TIMEOUT = 3600.0
    #: Log a line (once per round) when a shard has waited this long.
    WARN_AFTER = 120.0

    def __init__(self, n_shards, envs_per_shard, timeout_s=None,
                 log=print):
        n = int(n_shards)
        e = int(envs_per_shard)
        if n < 2:
            raise ValueError(
                f"ShardGather is for two or more shards; got {n}. With one "
                f"shard the callback is called directly and nothing is "
                f"wrapped.")
        if e < 2:
            # `_cb_slot` in env.py tells a per-environment operand from a
            # closed-over constant by its leading dimension: E for the first,
            # 1 for the second. At E == 1 those are the same number and the
            # merge below cannot tell a constant from a row, so it would
            # concatenate the constants too.
            raise ValueError(
                f"--rollout-shards {n} needs at least two environments per "
                f"shard; got {e}. At one environment per shard a callback "
                f"operand's leading dimension no longer says whether it is a "
                f"per-environment row or a broadcast constant.")
        self.n = n
        self.envs_per_shard = e
        self.timeout_s = float(
            os.environ.get("ALPHAGRAD_SHARD_GATHER_TIMEOUT",
                           timeout_s if timeout_s is not None
                           else self.DEFAULT_TIMEOUT))
        self._log = log
        self._cv = threading.Condition()
        self._round = 0
        self._arrivals: dict = {}
        self._results: dict = {}
        #: Rounds completed, for the driver's telemetry.
        self.rounds = 0

    # -- the public surface ------------------------------------------------

    def wrap(self, shard, fn):
        """`fn` as shard `shard` sees it: gather, call once, take my slice."""
        s = int(shard)
        if not (0 <= s < self.n):
            raise ValueError(f"shard {s} is not in 0..{self.n - 1}")

        def _gathered(*args):
            return self.call(s, fn, args)
        _gathered.__name__ = "shard_gathered_" + getattr(
            fn, "__name__", "callback")
        _gathered.__wrapped__ = fn
        return _gathered

    def call(self, shard, fn, args):
        import jax
        with self._cv:
            if shard in self._arrivals:
                raise RuntimeError(
                    f"shard {shard} reached the measurement rendezvous twice "
                    f"in round {self._round} without the round closing. Every "
                    f"shard runs the same program, so each makes exactly one "
                    f"call per step.")
            rnd = self._round
            self._arrivals[shard] = args
            if len(self._arrivals) == self.n:
                out = err = None
                # THE MERGE IS INSIDE THE TRY. It can fail -- two shards that
                # disagree on a broadcast operand's shape fail it by design --
                # and a failure that escaped here would leave the round open
                # and every other shard waiting on it until the timeout. The
                # rule is that this block always publishes a result, whether
                # that result is an answer or the exception that replaced it.
                try:
                    ordered = [self._arrivals[i] for i in range(self.n)]
                    merged = jax.tree_util.tree_map(self._merge, *ordered)
                    with HOST_LOCK:
                        out = fn(*merged)
                except BaseException as exc:      # noqa: BLE001
                    err = exc
                self._results[rnd] = [out, err, 0]
                self._arrivals = {}
                self._round = rnd + 1
                self.rounds += 1
                self._cv.notify_all()
            else:
                self._wait(rnd, shard)
            slot = self._results[rnd]
            slot[2] += 1
            if slot[2] == self.n:
                del self._results[rnd]
            out, err = slot[0], slot[1]
        if err is not None:
            raise err
        lo = shard * self.envs_per_shard
        hi = lo + self.envs_per_shard
        return jax.tree_util.tree_map(
            lambda x: np.asarray(x)[lo:hi], out)

    # -- internals ---------------------------------------------------------

    def _wait(self, rnd, shard):
        t0 = time.monotonic()
        warned = False
        while rnd not in self._results:
            self._cv.wait(timeout=5.0)
            if rnd in self._results:
                break
            waited = time.monotonic() - t0
            if waited > self.timeout_s:
                raise ShardGatherTimeout(
                    f"rollout shard {shard} waited {waited:.0f} s at the "
                    f"measurement rendezvous of round {rnd} and "
                    f"{self.n - len(self._arrivals)} of {self.n} shards never "
                    f"arrived. A shard that died before its callback leaves "
                    f"the others here; look for its traceback above.")
            if waited > self.WARN_AFTER and not warned:
                warned = True
                self._log(
                    f"[rollout-shards] shard {shard} has waited "
                    f"{waited:.0f} s at rendezvous round {rnd}; "
                    f"{len(self._arrivals)} of {self.n} shards have arrived "
                    f"(the first round of a bin waits out the other shards' "
                    f"compilation)", flush=True)

    def _merge(self, *leaves):
        """One callback operand, from N shards to one batch of N*E rows.

        `env._cb_slot`'s convention: under `vmap_method="expand_dims"` a
        per-environment operand has leading dimension E and a closed-over
        constant has leading dimension 1. Rows concatenate; a constant is
        taken from shard 0, and the shards are checked to agree on it, because
        a constant that differed between shards would mean the shards were not
        running the same episode.
        """
        first = leaves[0]
        a0 = np.asarray(first)
        if a0.ndim >= 1 and a0.shape[0] == self.envs_per_shard:
            return np.concatenate([np.asarray(x) for x in leaves], axis=0)
        for k, x in enumerate(leaves[1:], start=1):
            xk = np.asarray(x)
            if xk.shape != a0.shape:
                raise ValueError(
                    f"rollout shards disagree on the shape of a broadcast "
                    f"callback operand: shard 0 has {a0.shape}, shard {k} has "
                    f"{xk.shape}.")
        return a0


# ------------------------------------------------------------ the dispatch

def dispatch(fns, log=print):
    """Run `fns[i]()` in its OWN Python thread and return the results in order.

    THE THREAD IS THE POINT, not a convenience. A host callback runs on the
    thread that dispatched the program (probe job 65806), so dispatching the
    shards from one thread runs their callbacks one after another and the
    rollouts do not overlap. One thread per shard is what makes them concurrent.

    The first exception raised by any shard is re-raised here, after every
    thread has finished, so a failed shard cannot leave a live thread behind.
    """
    n = len(fns)
    if n == 1:
        return [fns[0]()]
    out = [None] * n
    err: list = [None] * n
    threads = []

    def _run(i):
        try:
            out[i] = fns[i]()
        except BaseException as exc:              # noqa: BLE001
            err[i] = exc

    for i in range(n):
        t = threading.Thread(target=_run, args=(i,),
                             name=f"rollout-shard-{i}", daemon=False)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for i in range(n):
        if err[i] is not None:
            raise err[i]
    return out


def concat_shards(outs, device=None):
    """Concatenate N shards' per-environment outputs along the env axis.

    Every leaf a sharded rollout returns is per environment -- the rollout is
    `jax.vmap`ped over environments, so its outputs all carry the environment
    as axis 0. The concatenation is therefore the whole join, it is in global
    environment order by :func:`shard_env_range`, and the result is exactly
    what one device would have produced for `N*E` environments.

    `device` is where the update runs; the leaves are moved there first so the
    concatenation itself does not have to pick.
    """
    import jax
    import jax.numpy as jnp
    if len(outs) == 1:
        return outs[0]

    def _join(*xs):
        if device is not None:
            xs = [jax.device_put(x, device) for x in xs]
        return jnp.concatenate([jnp.asarray(x) for x in xs], axis=0)

    return jax.tree_util.tree_map(_join, *outs)
