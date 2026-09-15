# -*- coding: utf-8 -*-
"""DATA PARALLELISM OVER ENVIRONMENTS: one rollout shard per GPU.

Owner ruling 2026-09-15, "use the idle GPUs for the rollout".

Four things are pinned here.

1. THE DIVISION. A shard's global environment indices tile `0..N*E-1` in
   shard order, so `env_index` is unique across shards and a concatenation of
   the shards' outputs is in environment order without a permutation.
2. THE RENDEZVOUS. The N shards' per-step measurement callbacks become ONE
   batched call over all the rows, in global environment order, with the
   per-environment operands concatenated and the broadcast constants taken
   once; each shard gets its own rows back. An exception inside the call
   reaches every shard, and a shard that never arrives raises instead of
   hanging for ever.
3. THE SERIALISATION. `host_serial` really excludes concurrent entry, which
   is what protects the face enumerator's prefix cache and the oracle memo
   once the shards run on threads of their own.
4. THE EXACTNESS THE RULING ASKS FOR BY NAME. A given environment's
   trajectory does not depend on which device rolled it out. That one runs in
   a SUBPROCESS, on a four-device CPU (`--xla_force_host_platform_device_count
   =4`), because the flag has to be set before the backend comes up and no
   other test in this suite may see it.

The whole apparatus under test is `common/rollout_shards.py` plus the two
hooks on the env (`with_rollout_shard`, `set_shard_gather`). `ppo.rollout_fn`
is a closure inside `ppo.main` and cannot be called from a test at all, which
`tests/one_stream_claim_test.py` documents at length; case 4 therefore drives
the same key rule, the same index rule, the same rendezvous and the same
concatenation through a rollout of its own.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402


def _rs():
    from alphagrad.approx.common import rollout_shards
    return rollout_shards


# --------------------------------------------------------------- the division

def test_the_shards_environment_blocks_tile_every_environment_once():
    rs = _rs()
    for n, e in ((1, 16), (4, 16), (8, 16), (7, 4), (2, 3)):
        seen = []
        for s in range(n):
            lo, hi = rs.shard_env_range(s, n, e)
            assert hi - lo == e
            seen.extend(range(lo, hi))
        assert seen == list(range(n * e))


def test_a_shard_outside_the_count_is_refused():
    rs = _rs()
    with pytest.raises(ValueError):
        rs.shard_env_range(4, 4, 16)
    with pytest.raises(ValueError):
        rs.shard_env_range(-1, 4, 16)
    with pytest.raises(ValueError):
        rs.shard_env_range(0, 1, 0)


def test_more_shards_than_devices_is_refused_by_name():
    rs = _rs()
    devs = ["d0", "d1", "d2"]
    assert rs.resolve_devices(3, devs) == devs
    assert rs.resolve_devices(1, devs) == ["d0"]
    with pytest.raises(ValueError) as exc:
        rs.resolve_devices(4, devs)
    assert "4" in str(exc.value) and "3" in str(exc.value)
    with pytest.raises(ValueError):
        rs.resolve_devices(0, devs)


# ------------------------------------------------------------- the rendezvous

def _drive(gather, fn, per_shard_args):
    """Call `gather` from one thread per shard; return the shards' results."""
    n = len(per_shard_args)
    out = [None] * n
    err = [None] * n

    def _go(i):
        try:
            out[i] = gather.call(i, fn, per_shard_args[i])
        except BaseException as exc:                    # noqa: BLE001
            err[i] = exc

    ts = [threading.Thread(target=_go, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    for t in ts:
        assert not t.is_alive(), "a shard thread did not finish"
    return out, err


def test_the_rendezvous_makes_one_call_over_every_row_in_environment_order():
    rs = _rs()
    g = rs.ShardGather(4, 2, timeout_s=30.0)
    calls = []

    def fn(rows, const):
        calls.append((np.asarray(rows).copy(), np.asarray(const).copy()))
        # One output row per input row, so the slicing has something to cut.
        return (np.asarray(rows) * 10,)

    args = [(np.array([2 * s, 2 * s + 1]), np.array([7])) for s in range(4)]
    out, err = _drive(g, fn, args)
    assert err == [None] * 4
    # ONE call, over all eight rows, in global environment order.
    assert len(calls) == 1
    assert calls[0][0].tolist() == list(range(8))
    # The broadcast constant is taken once, not concatenated.
    assert calls[0][1].tolist() == [7]
    # And every shard got its OWN two rows back.
    for s in range(4):
        assert out[s][0].tolist() == [10 * 2 * s, 10 * (2 * s + 1)]
    assert g.rounds == 1


def test_the_rendezvous_runs_round_after_round():
    rs = _rs()
    g = rs.ShardGather(3, 2, timeout_s=30.0)
    seen = []

    def fn(rows):
        seen.append(np.asarray(rows).tolist())
        return (np.asarray(rows) + 1,)

    for step in range(4):
        args = [(np.array([2 * s + 100 * step, 2 * s + 1 + 100 * step]),)
                for s in range(3)]
        out, err = _drive(g, fn, args)
        assert err == [None] * 3
        for s in range(3):
            assert out[s][0].tolist() == [2 * s + 100 * step + 1,
                                          2 * s + 1 + 100 * step + 1]
    assert len(seen) == 4
    assert g.rounds == 4


def test_a_failure_inside_the_gathered_call_reaches_every_shard():
    rs = _rs()
    g = rs.ShardGather(3, 2, timeout_s=30.0)

    class Boom(RuntimeError):
        pass

    def fn(rows):
        raise Boom("the pool refused")

    args = [(np.array([2 * s, 2 * s + 1]),) for s in range(3)]
    out, err = _drive(g, fn, args)
    assert out == [None] * 3
    assert all(isinstance(e, Boom) for e in err), err


def test_a_shard_that_never_arrives_raises_instead_of_hanging():
    rs = _rs()
    g = rs.ShardGather(3, 2, timeout_s=0.1)
    g.WARN_AFTER = 1e9

    def fn(rows):
        return (np.asarray(rows),)

    # Two of three shards arrive.
    out = [None, None]
    err = [None, None]

    def _go(i):
        try:
            out[i] = g.call(i, fn, (np.array([2 * i, 2 * i + 1]),))
        except BaseException as exc:                    # noqa: BLE001
            err[i] = exc

    ts = [threading.Thread(target=_go, args=(i,)) for i in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    for t in ts:
        assert not t.is_alive()
    assert all(isinstance(e, rs.ShardGatherTimeout) for e in err), err


def test_a_shard_cannot_enter_one_round_twice():
    rs = _rs()
    g = rs.ShardGather(2, 2, timeout_s=0.5)
    g.WARN_AFTER = 1e9
    err = []

    def _go():
        try:
            g.call(0, lambda r: (r,), (np.array([0, 1]),))
        except BaseException as exc:                    # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_go)
    t.start()
    # Wait until shard 0 is parked, then bring it back a second time.
    for _ in range(200):
        with g._cv:
            if 0 in g._arrivals:
                break
        time.sleep(0.01)
    with pytest.raises(RuntimeError):
        g.call(0, lambda r: (r,), (np.array([0, 1]),))
    t.join(timeout=30)
    assert not t.is_alive()


def test_the_rendezvous_refuses_a_shape_the_shards_disagree_on():
    rs = _rs()
    g = rs.ShardGather(2, 2, timeout_s=30.0)
    args = [(np.array([0, 1]), np.zeros((1, 3))),
            (np.array([2, 3]), np.zeros((1, 4)))]
    out, err = _drive(g, lambda r, c: (r,), args)
    assert all(isinstance(e, ValueError) for e in err), err


def test_the_rendezvous_refuses_one_shard_and_one_environment_per_shard():
    rs = _rs()
    with pytest.raises(ValueError):
        rs.ShardGather(1, 16)
    with pytest.raises(ValueError) as exc:
        rs.ShardGather(4, 1)
    assert "leading dimension" in str(exc.value)


# ------------------------------------------------------------ the dispatch

def test_dispatch_runs_every_shard_on_a_thread_of_its_own():
    rs = _rs()
    bar = threading.Barrier(4, timeout=30)
    tids = [None] * 4

    def _mk(i):
        def _f():
            tids[i] = threading.get_ident()
            bar.wait()          # only passes if the four really are concurrent
            return i * 3
        return _f

    got = rs.dispatch([_mk(i) for i in range(4)])
    assert got == [0, 3, 6, 9]
    assert len(set(tids)) == 4


def test_dispatch_reraises_a_shards_failure_after_joining_the_others():
    rs = _rs()
    done = []

    def _ok():
        time.sleep(0.05)
        done.append(1)
        return 1

    def _bad():
        raise ValueError("shard 1 died")

    with pytest.raises(ValueError):
        rs.dispatch([_ok, _bad, _ok])
    # The surviving shards ran to the end before the raise reached the caller.
    assert len(done) == 2


def test_concat_shards_joins_on_the_environment_axis_in_shard_order():
    import jax.numpy as jnp
    rs = _rs()
    outs = [(jnp.arange(2) + 2 * s, {"r": jnp.full((2, 3), float(s))})
            for s in range(3)]
    joined = rs.concat_shards(outs)
    assert np.asarray(joined[0]).tolist() == list(range(6))
    assert np.asarray(joined[1]["r"]).shape == (6, 3)
    assert np.asarray(joined[1]["r"])[:, 0].tolist() == [0, 0, 1, 1, 2, 2]
    # One shard is the identity, leaves and all.
    one = [(jnp.arange(2), None)]
    assert rs.concat_shards(one) is one[0]


# ------------------------------------------------------- the serialisation

def test_host_serial_excludes_concurrent_entry():
    rs = _rs()
    inside = [0]
    worst = [0]

    @rs.host_serial
    def _touch():
        inside[0] += 1
        worst[0] = max(worst[0], inside[0])
        time.sleep(0.005)
        inside[0] -= 1

    ts = [threading.Thread(target=_touch) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert worst[0] == 1


def test_host_serial_is_reentrant_so_a_callback_may_call_another():
    rs = _rs()

    @rs.host_serial
    def _inner():
        return 7

    @rs.host_serial
    def _outer():
        return _inner() + 1

    assert _outer() == 8


# --------------------------------------------------------------- the env

def test_the_env_carries_its_shard_through_the_pytree():
    import jax
    from alphagrad.approx import env as envmod
    e = _tiny_env()
    assert (e.rollout_shard, e.rollout_shards) == (0, 1)
    s2 = e.with_rollout_shard(2, 4)
    assert (s2.rollout_shard, s2.rollout_shards) == (2, 4)
    # The pair is AUX data, so it survives a flatten/unflatten round trip and
    # two shards are two different static arguments to one jit.
    leaves, treedef = jax.tree_util.tree_flatten(s2)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert (back.rollout_shard, back.rollout_shards) == (2, 4)
    assert jax.tree_util.tree_structure(e) != jax.tree_util.tree_structure(s2)
    # And the window bin carries it too, so `with_delta_window` cannot drop it.
    assert s2.with_delta_window(1024).rollout_shard == 2
    assert s2.with_delta_window(1024).rollout_shards == 4
    with pytest.raises(ValueError):
        e.with_rollout_shard(4, 4)
    del envmod


def test_a_sharded_env_without_a_rendezvous_refuses_to_trace():
    from alphagrad.approx import env as envmod
    e = _tiny_env().with_rollout_shard(1, 4)
    envmod.set_shard_gather(None)
    with pytest.raises(RuntimeError) as exc:
        e._shard_wrap(lambda *a: a)
    assert "rendezvous" in str(exc.value)


def test_a_sharded_env_refuses_the_per_environment_callback():
    """Without the BATCHED callback there is nothing to gather, and every
    shard would make its own pool call -- which sentinels the rows it cannot
    place an actor for. `tokenize` refuses rather than returning it."""
    from alphagrad.approx import env as envmod
    if envmod._BATCHED_CALLBACK:
        pytest.skip("ALPHAGRAD_BATCHED_CALLBACK is on in this process")
    e = _tiny_env().with_rollout_shard(1, 4)
    with pytest.raises(RuntimeError) as exc:
        e.tokenize(batched=True)
    assert "BATCHED_CALLBACK" in str(exc.value)
    # And the unbatched reset callback is untouched: it makes no measurement.
    e.tokenize(init=True)


def test_an_unsharded_env_wraps_nothing_at_all():
    from alphagrad.approx import env as envmod
    envmod.set_shard_gather(None)
    e = _tiny_env()
    fn = e.tokenize(batched=True)
    assert not hasattr(fn, "__wrapped__") or "shard" not in fn.__name__


def test_a_sharded_envs_callback_is_the_rendezvous_and_gathers_the_rows():
    """The env's OWN wrapper, driven by hand, is one call over every row."""
    from alphagrad.approx import env as envmod
    rs = _rs()
    g = rs.ShardGather(2, 2, timeout_s=30.0)
    envmod.set_shard_gather(g)
    try:
        envs = [_tiny_env().with_rollout_shard(s, 2) for s in range(2)]
        seen = []

        def _raw(rows):
            seen.append(np.asarray(rows).tolist())
            return (np.asarray(rows) * 2,)

        wrapped = [e._shard_wrap(_raw) for e in envs]
        args = [(np.array([2 * s, 2 * s + 1]),) for s in range(2)]
        out = [None, None]

        def _go(i):
            out[i] = wrapped[i](*args[i])

        ts = [threading.Thread(target=_go, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)
        assert seen == [[0, 1, 2, 3]]
        assert out[0][0].tolist() == [0, 2]
        assert out[1][0].tolist() == [4, 6]
    finally:
        envmod.set_shard_gather(None)


def test_a_rendezvous_for_the_wrong_number_of_shards_is_refused():
    from alphagrad.approx import env as envmod
    rs = _rs()
    envmod.set_shard_gather(rs.ShardGather(2, 2, timeout_s=1.0))
    try:
        e = _tiny_env().with_rollout_shard(1, 4)
        with pytest.raises(RuntimeError):
            e._shard_wrap(lambda *a: a)
    finally:
        envmod.set_shard_gather(None)


def _tiny_env():
    """The smallest real :class:`VertexEliminationEnv` this suite can build."""
    import jax
    import jax.numpy as jnp
    from alphagrad.approx.env import VertexEliminationEnv

    def f(x, y):
        return jnp.sin(x) * y + jnp.cos(y)

    jaxpr = jax.make_jaxpr(f)(jnp.ones((2,)), jnp.ones((2,)))
    return VertexEliminationEnv.from_jaxpr(
        jaxpr, args=(jnp.ones((2,)), jnp.ones((2,))), num_envs=2)


# ---------------------------------------- the exactness the ruling asks for

_SUBPROCESS = textwrap.dedent(
    """
    # Does a trajectory depend on which device rolled its environment out?
    #
    # Four CPU devices, the SAME key rule, index rule, rendezvous and
    # concatenation ppo.py uses, and a rollout whose every output depends on
    # the environment's own key, its global index and the host callback's
    # answer over the whole gathered batch.
    import os
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_force_host_platform_device_count=4").strip()
    os.environ["JAX_PLATFORMS"] = "cpu"

    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jrand
    from alphagrad.approx.common import rollout_shards as rs

    TOTAL = 8
    STEPS = 5
    devs = jax.local_devices()
    assert len(devs) >= 4, devs


    def host(rows, const):
        # The measurement callback's stand-in: per-row work, one batch. The
        # answer for a row depends on the row, so a mis-ordered gather or a
        # mis-cut slice shows up as a changed trajectory.
        r = np.asarray(rows)
        c = np.asarray(const)
        return (np.asarray(r * 3 + c[0], np.int32),)


    def make_rollout(hostfn):
        # ONE JIT PER SHARD. The host function is read at TRACE time, exactly
        # as `env.tokenize` reads it, so two shards must not share a trace --
        # in ppo.py the shard index rides the env's aux data and does that;
        # here a jit of its own does.
        @jax.jit
        def rollout(env_keys, idx):
            def step(carry, t):
                noise = jax.vmap(
                    lambda kk: jrand.randint(jrand.fold_in(kk, t), (), 0, 5)
                )(env_keys).astype(jnp.int32)
                got = jax.pure_callback(
                    hostfn,
                    (jax.ShapeDtypeStruct(idx.shape, jnp.int32),),
                    carry + noise + idx,
                    jnp.asarray([1], jnp.int32),
                    vmap_method="expand_dims")[0]
                return carry + got, got
            last, out = jax.lax.scan(
                step, jnp.zeros_like(idx), jnp.arange(STEPS, dtype=jnp.int32))
            return last, out.T          # (E,) and (E, STEPS)
        return rollout


    def run(n_shards):
        e = TOTAL // n_shards
        # THE KEY RULE: one key per environment of the WHOLE episode, and the
        # shard takes its own block. This is ppo._episode_rollout's rule.
        all_keys = jrand.split(jrand.PRNGKey(7), TOTAL)
        g = rs.ShardGather(n_shards, e, timeout_s=300.0) if n_shards > 1 \
            else None

        def one(s):
            lo, hi = rs.shard_env_range(s, n_shards, e)
            dev = devs[s]
            fn = make_rollout(host if g is None else g.wrap(s, host))
            return fn(jax.device_put(all_keys[lo:hi], dev),
                      jax.device_put(jnp.arange(lo, hi, dtype=jnp.int32), dev))

        if n_shards == 1:
            outs = [one(0)]
        else:
            outs = rs.dispatch([(lambda _s=_s: one(_s))
                                for _s in range(n_shards)])
        return rs.concat_shards(outs, devs[0])


    ref = run(1)
    for n in (2, 4):
        got = run(n)
        for a, b in zip(ref, got):
            a, b = np.asarray(a), np.asarray(b)
            assert a.shape == b.shape, (n, a.shape, b.shape)
            if not np.array_equal(a, b):
                raise SystemExit(
                    "shards=%d changed a trajectory: %r vs %r" % (n, a, b))
    print("SHARD EXACTNESS OK", flush=True)
    """
)


def test_a_trajectory_does_not_depend_on_which_device_rolled_it_out(tmp_path):
    """THE RULING'S OWN CASE, on four CPU devices, in its own process.

    `--xla_force_host_platform_device_count=4` has to be set before the
    backend comes up and must not leak into the rest of the suite, so this
    runs as a subprocess and asserts on what it printed.
    """
    script = tmp_path / "shard_exactness.py"
    script.write_text(_SUBPROCESS)
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run([sys.executable, str(script)], env=env,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SHARD EXACTNESS OK" in proc.stdout, proc.stdout + proc.stderr
