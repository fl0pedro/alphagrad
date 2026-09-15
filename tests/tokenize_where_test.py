"""WHERE THE PER-STEP TOKENIZATION RUNS (owner ruling 2026-09-15).

A non-terminal callback row measures nothing under terminal rewards: it
tokenizes the elimination prefix, decides face legality and returns the delta
observation. `--tokenize-where` says which process does that work -- the
measure actors (`pool`), the trainer itself (`local`), or a second pool of
CPU-only actors (`cpu-actors`).

What these tests pin:

* THE BYTES DO NOT MOVE. The same step, tokenized through each of the three
  routes with every incremental cache cleared in between, returns the same
  tokens, the same equation ids and the same reward rows -- and so does a
  whole episode's worth of steps, one after the other, which is the case the
  caches actually run in.
* THE ROUTING. Under `local` no non-terminal row reaches the measure pool;
  under `cpu-actors` every one of them reaches the tokenize pool and none the
  measure pool; under `pool` nothing changes. The terminal row goes where it
  always went in all three.
* THE PLAN LOG. A terminal plan recorded after a `local` prefix is the same
  record as one recorded after a `pool` prefix, field for field.
* THE DEFERRED SUBMISSION, which is what keeps the deep pipeline to one batch
  in the actors: the terminal step packages, the driver starts, and a collect
  before the start is an error rather than a hang.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------- helpers

class _DoneFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value

    def done(self):
        return True


def _small_env(delta_window=0):
    """A four-equation env on the delta-observation path."""
    import jax
    import jax.numpy as jnp
    from alphagrad.approx.env import EnvConfig, VertexEliminationEnv

    def fn(x, y):
        return jnp.tanh(jnp.sin(x) @ y) + jnp.exp(jnp.sin(x) @ y)

    args = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)
    cj = jax.make_jaxpr(fn)(*args)
    cfg = EnvConfig(
        jaxpr=cj.jaxpr, argnums=(0, 1), has_aux=False, sparse=False,
        cmp_type="flops", mem_type="peak_memory",
        terminal_rewards_only=True, delta_obs=True,
        delta_window=int(delta_window))
    return VertexEliminationEnv(cfg, args, list(cj.literals))


class _RealPool:
    """A pool that TOKENIZES FOR REAL, in this process.

    It stands in for a measure actor, which is the same `env._callback` in
    another process. Standing in for it here is the point: the claim under
    test is that the bytes do not depend on which process ran the call, and a
    fake that returned synthetic wires could not test that at all.
    """

    def __init__(self, env, name):
        self._env = env
        self.name = name
        self.rows = []              # [(step, ...)] per evaluate_batch call
        self.submitted = []
        self._eval_samples_ref = None

    def _serve(self, order_batch, specs_batch, step_batch, *, eval_samples,
               init, face_specs_batch, face_skips_batch, episode=None):
        from alphagrad.approx.env import _callback
        toks, rews = [], []
        for k in range(len(order_batch)):
            out = _callback(
                self._env.config, self._env.args, self._env.consts,
                order_batch[k], specs_batch[k],
                (np.zeros((int(step_batch[k]), 2, 1, 3), np.int32) - 1
                 if face_specs_batch is None else face_specs_batch[k]),
                (np.zeros((int(step_batch[k]), 2), np.int32)
                 if face_skips_batch is None else face_skips_batch[k]),
                int(step_batch[k]),
                *(eval_samples or ()),
                init=bool(init), face_joins=None)
            toks.append(np.asarray(out[0]))
            rews.append(np.asarray(out[-1]))
        n = len(order_batch)
        return (np.stack(toks).astype(self._env.wire_token_dtype),
                np.stack(rews).astype(np.float32),
                np.zeros((n,), bool))

    def evaluate_batch(self, order_batch, specs_batch, step_batch, **kw):
        self.rows.append([int(s) for s in step_batch])
        return self._serve(order_batch, specs_batch, step_batch,
                           eval_samples=kw.get("eval_samples"),
                           init=kw.get("init", False),
                           face_specs_batch=kw.get("face_specs_batch"),
                           face_skips_batch=kw.get("face_skips_batch"))

    def submit_batch(self, *args, **kw):
        self.submitted.append([int(s) for s in args[2]])
        return _DoneFuture(self.evaluate_batch(*args, **kw))


def _pooled_env(env, pool):
    return type(env)(
        env.config, env.args, env.consts, env.valid_vertices, env.num_envs,
        env.eval_args_samples,
        axis_state_static=env.axis_state_static,
        axis_valid_static=env.axis_valid_static,
        remote_pool=pool, remote_timeout_s=60.0)


def _clear_caches():
    """Force every route to replay the prefix COLD.

    Without this the second route would read the first route's incremental
    stream cache and the comparison would test nothing. A measure actor is a
    fresh process with an empty cache, so cold is also the honest model of it.
    """
    from alphagrad.approx import env as E
    E._INCR_STREAM_CACHE.clear()
    E._INCR_TOK_CACHE.clear()
    E._FACE_ENUM_CACHE.clear()


def _order_and_specs(n_envs, n_vertices, order):
    o = np.zeros((n_envs, n_vertices), np.int32)
    for i in range(n_envs):
        o[i] = np.asarray(order, np.int32)
    s = np.zeros((n_envs, n_vertices, 4), np.int32)
    return o, s


def _call(env, steps, order, n_vertices=4):
    """One batched host callback for `len(steps)` environments.

    THE ENV'S OWN ARGS AND CONSTS GO IN, not placeholders. The trainer-local
    route tokenizes from them (a measure actor tokenizes from its own copy of
    the same env), so a harness that handed in dummies would compare one
    route's real tokens against another route's tokens of a different graph.
    `_cb_slot` leaves them alone as long as their leading dimension is neither
    `len(steps)` nor 1, which is why every caller here uses two or three
    environments against a four-by-four graph.
    """
    n = len(steps)
    o, s = _order_and_specs(n, n_vertices, order)
    face_specs = -np.ones((n, n_vertices, 2, 1, 3), np.int32)
    face_skips = np.zeros((n, n_vertices, 2), np.int32)
    fn = env.tokenize(batched=True)
    return fn(env.args, env.consts, o, s, face_specs, face_skips, None,
              np.asarray(steps, np.int32))


@pytest.fixture(autouse=True)
def _clean_route():
    """No test may leave a route, a pool or a ticket behind."""
    from alphagrad.approx import env as E
    E.set_tokenize_where("pool")
    E.set_measure_defer(False)
    E._MEASURE_TICKETS.clear()
    E._MEASURE_TICKET_OPEN[0] = None
    yield
    E.set_tokenize_where("pool")
    E.set_measure_defer(False)
    E._MEASURE_TICKETS.clear()
    E._MEASURE_TICKET_OPEN[0] = None


# ------------------------------------------------------- 1. the routing

def test_local_keeps_every_non_terminal_row_out_of_the_measure_pool():
    from alphagrad.approx import env as E

    env0 = _small_env()
    pool = _RealPool(env0, "measure")
    env = _pooled_env(env0, pool)
    E.set_tokenize_where("local")
    _clear_caches()
    _call(env, [2, 2, 2], [0, 1, 2, 3])
    assert pool.rows == [] and pool.submitted == []


def test_pool_is_still_the_route_it_always_was():
    from alphagrad.approx import env as E

    env0 = _small_env()
    pool = _RealPool(env0, "measure")
    env = _pooled_env(env0, pool)
    E.set_tokenize_where("pool")
    _clear_caches()
    _call(env, [2, 2, 2], [0, 1, 2, 3])
    assert pool.rows == [[2, 2, 2]]


def test_cpu_actors_sends_the_step_to_the_tokenize_pool_and_not_the_other():
    from alphagrad.approx import env as E

    env0 = _small_env()
    measure = _RealPool(env0, "measure")
    tokenize = _RealPool(env0, "tokenize")
    env = _pooled_env(env0, measure)
    E.set_tokenize_where("cpu-actors", pool=tokenize)
    _clear_caches()
    _call(env, [2, 2, 2], [0, 1, 2, 3])
    assert tokenize.rows == [[2, 2, 2]]
    assert measure.rows == [] and measure.submitted == []


def test_cpu_actors_without_a_pool_is_refused_at_the_setter():
    from alphagrad.approx import env as E

    with pytest.raises(ValueError):
        E.set_tokenize_where("cpu-actors")
    with pytest.raises(ValueError):
        E.set_tokenize_where("somewhere-else")


def test_the_terminal_row_goes_where_it_always_went_under_every_route():
    """`--tokenize-where` moves the NON-TERMINAL work and nothing else."""
    from alphagrad.approx import env as E

    for where in ("local", "cpu-actors"):
        env0 = _small_env()
        measure = _RealPool(env0, "measure")
        tokenize = _RealPool(env0, "tokenize")
        env = _pooled_env(env0, measure)
        E.set_tokenize_where(
            where, pool=(tokenize if where == "cpu-actors" else None))
        _clear_caches()
        t = E.open_measure_ticket()
        _call(env, [4, 4], [0, 1, 2, 3])
        E.close_measure_ticket()
        # The terminal rows were SUBMITTED to the MEASURE pool, as under
        # `pool`; the tokenize pool never sees a terminal.
        assert measure.submitted == [[4, 4]], where
        assert tokenize.rows == [], where
        E.collect_measurement(t)


def test_a_mixed_step_splits_terminal_from_non_terminal_under_local():
    from alphagrad.approx import env as E

    env0 = _small_env()
    measure = _RealPool(env0, "measure")
    env = _pooled_env(env0, measure)
    E.set_tokenize_where("local")
    _clear_caches()
    t = E.open_measure_ticket()
    _call(env, [4, 2, 4], [0, 1, 2, 3])
    E.close_measure_ticket()
    assert measure.submitted == [[4, 4]]
    assert measure.rows == []          # row 1 was tokenized in the trainer
    assert E.measure_ticket_rows(t) == [0, 2]
    E.collect_measurement(t)


# ------------------------------------------------ 2. the bytes do not move

_PREFIX = [0, 1, 2, 3]


def _sweep(where, steps, n_envs=3):
    """A whole episode's worth of steps through one route. Returns the wires.

    The caches are cleared ONCE at the start and then left alone, which is
    how a real rollout runs: step k+1 extends step k's tokenizer.
    """
    from alphagrad.approx import env as E

    env0 = _small_env()
    measure = _RealPool(env0, "measure")
    tokenize = _RealPool(env0, "tokenize")
    env = _pooled_env(env0, measure)
    E.set_tokenize_where(
        where, pool=(tokenize if where == "cpu-actors" else None))
    _clear_caches()
    out = []
    for s in steps:
        tk, rw = _call(env, [s] * n_envs, _PREFIX)
        out.append((np.asarray(tk).copy(), np.asarray(rw).copy()))
    E.set_tokenize_where("pool")
    return out


def test_one_step_is_byte_identical_through_all_three_routes():
    ref = _sweep("pool", [2])
    for where in ("local", "cpu-actors"):
        got = _sweep(where, [2])
        assert np.array_equal(got[0][0], ref[0][0]), where
        assert np.array_equal(got[0][1], ref[0][1]), where


def test_a_whole_episodes_steps_are_byte_identical_through_all_three_routes():
    """THE CASE THE CACHES RUN IN. Step k+1 extends step k's tokenizer, so a
    route that got the incremental extension wrong would diverge here and not
    on a single cold step."""
    steps = [1, 2, 3]
    ref = _sweep("pool", steps)
    for where in ("local", "cpu-actors"):
        got = _sweep(where, steps)
        for k, s in enumerate(steps):
            assert np.array_equal(got[k][0], ref[k][0]), (where, s)
            assert np.array_equal(got[k][1], ref[k][1]), (where, s)


def test_every_environment_row_gets_its_own_tokens_under_local():
    """Two environments at DIFFERENT steps must not swap rows when the
    classification splits them."""
    from alphagrad.approx import env as E

    env0 = _small_env()
    measure = _RealPool(env0, "measure")
    env = _pooled_env(env0, measure)
    E.set_tokenize_where("pool")
    _clear_caches()
    ref_tk, _ = _call(env, [1, 2, 3], _PREFIX)
    E.set_tokenize_where("local")
    _clear_caches()
    got_tk, _ = _call(env, [1, 2, 3], _PREFIX)
    assert np.array_equal(np.asarray(got_tk), np.asarray(ref_tk))
    # And the three rows really are different from one another, so the
    # equality above is not the equality of three identical zero rows.
    assert not np.array_equal(np.asarray(ref_tk)[0], np.asarray(ref_tk)[1])


# ----------------------------------------------------------- 3. the plan log

def test_a_terminal_plan_is_the_same_record_after_a_local_prefix(monkeypatch):
    """The plan log is what the campaign reads a run back from, so the route
    the PREFIX took must not show up in it."""
    from alphagrad.approx import env as E

    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    recs = {}
    for where in ("pool", "local"):
        env0 = _small_env()
        measure = _RealPool(env0, "measure")
        env = _pooled_env(env0, measure)
        E.set_tokenize_where(where)
        _clear_caches()
        E.consume_plan_records()
        for s in (1, 2, 3):
            _call(env, [s], _PREFIX)
        # THE TERMINAL STEP, measured in this process either way: the route
        # under test is the prefix's, not the terminal's.
        E.set_tokenize_where("local")
        _call(env, [4], _PREFIX)
        drained = E.consume_plan_records()
        recs[where] = drained["records"]
        E.set_tokenize_where("pool")
    assert len(recs["pool"]) == 1 and len(recs["local"]) == 1
    a, b = recs["pool"][0], recs["local"][0]
    assert a["order"] == b["order"]
    from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX
    lat = REWARD_INDEX["latency_ns"]
    keep = [i for i in range(NUM_REWARDS) if i != lat]
    ra = np.asarray(a["rewards"], np.float64)[keep]
    rb = np.asarray(b["rewards"], np.float64)[keep]
    assert np.array_equal(ra, rb), (ra, rb)
    # The replayable wire is the field the campaign reads a plan back from.
    for key in ("wire", "requested", "counts", "schema"):
        if key in a or key in b:
            assert a.get(key) == b.get(key), key


# ------------------------------------------- 4. the deferred submission

def test_a_deferred_submission_is_not_started_by_the_callback():
    """THE DEEP PIPELINE'S WHOLE GUARANTEE. Episode e+1's batch must not be
    in the actors while the driver is draining episode e's records out of
    them, so the terminal step packages and the DRIVER starts."""
    from alphagrad.approx import env as E

    env0 = _small_env()
    measure = _RealPool(env0, "measure")
    env = _pooled_env(env0, measure)
    E.set_tokenize_where("local")
    E.set_measure_defer(True)
    _clear_caches()
    t = E.open_measure_ticket()
    tk, rw = _call(env, [4, 4], _PREFIX)
    E.close_measure_ticket()
    assert measure.submitted == []          # packaged, not started
    assert np.all(np.asarray(tk) == 0) and np.all(np.asarray(rw) == 0.0)
    # A collect before the start is an error, not a hang.
    with pytest.raises(E.MeasureTicketError):
        E.collect_measurement(t)
    assert E.start_measurement(t) is True
    assert measure.submitted == [[4, 4]]
    out = E.collect_measurement(t)
    assert out["rewards"].shape[0] == 2
    # And starting it twice is not a second submission.
    with pytest.raises(E.MeasureTicketError):
        E.start_measurement(t)


def test_a_discarded_deferred_attempt_needs_no_drain():
    """It never reached an actor, so there is nothing there to drain and
    nothing there to drop."""
    from alphagrad.approx import env as E

    env0 = _small_env()
    measure = _RealPool(env0, "measure")
    env = _pooled_env(env0, measure)
    E.set_tokenize_where("local")
    E.set_measure_defer(True)
    _clear_caches()
    t = E.open_measure_ticket()
    _call(env, [4], _PREFIX)
    E.close_measure_ticket()
    out = E.drop_measurement(t)
    assert out["drained"] is False and out.get("deferred") is True
    assert measure.submitted == []
    assert E.pending_measure_tickets() == []


def test_without_the_defer_armed_the_callback_starts_the_submission():
    """`--measure-pipeline 1 --tokenize-where pool` keeps the old behaviour."""
    from alphagrad.approx import env as E

    env0 = _small_env()
    measure = _RealPool(env0, "measure")
    env = _pooled_env(env0, measure)
    E.set_measure_defer(False)
    _clear_caches()
    t = E.open_measure_ticket()
    _call(env, [4], _PREFIX)
    E.close_measure_ticket()
    assert measure.submitted == [[4]]
    assert E.start_measurement(t) is False
    E.collect_measurement(t)


def test_the_pool_still_refuses_a_second_batch_in_flight():
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    class _Slow:
        pass

    pool = CpuApproxPool.__new__(CpuApproxPool)
    import threading
    pool._lock = threading.RLock()
    pool._closed = False
    pool._submit_exec = None
    pool._submit_inflight = []
    gate = threading.Event()

    def _block():
        gate.wait(5.0)
        return "done"

    pool.evaluate_batch = lambda *a, **k: _block()
    f1 = pool.submit_batch()
    with pytest.raises(RuntimeError):
        pool.submit_batch()
    gate.set()
    assert f1.result(5.0) == "done"
    # Once it is done the slot frees and the next submit is accepted.
    f2 = pool.submit_batch()
    assert f2.result(5.0) == "done"
