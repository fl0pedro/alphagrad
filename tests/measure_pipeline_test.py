"""THE PIPELINED TERMINAL MEASUREMENT (owner ruling 2026-09-14).

The terminal rewards of episode e are read by exactly one thing, e's PPO
update. So the terminal step SUBMITS its plans to the measure actors and
returns a placeholder, and the driver waits for them with the previous
episode's update already running on the trainer's GPU.

What these tests pin:

* the ticket protocol -- one open ticket, one submission per ticket, and a
  collect that puts every environment's reward vector on its own row;
* the routing -- with a ticket open the TERMINAL rows go to `submit_batch`
  and the non-terminal ones still go to `evaluate_batch`, because those
  carry the tokenization the next step's encoder reads;
* the discard -- an attempt that overflows its bin drains its submission and
  throws the measure actors' records away, which is the half of the rollback
  the trainer's own snapshot cannot reach;
* the pool -- one batch in flight at a time, because the same actors serve
  the next rollout's per-step tokenization.
"""

from __future__ import annotations

import concurrent.futures as _cf
import threading

import numpy as np
import pytest


# --------------------------------------------------------------- helpers

class _DoneFuture:
    """A future whose result is already there. Stands in for the pool's."""

    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value

    def done(self):
        return True


def _wire(n_rows, width, n_rewards, base=0.0):
    """One `evaluate_batch` return, `(tokens, rewards, sentinel)`."""
    tokens = np.zeros((n_rows, width), np.uint8)
    rewards = np.arange(n_rows * n_rewards, dtype=np.float32).reshape(
        n_rows, n_rewards) + base
    sentinel = np.zeros((n_rows,), bool)
    return tokens, rewards, sentinel


@pytest.fixture(autouse=True)
def _clean_tickets():
    """No test may leave a ticket open for the next one."""
    from alphagrad.approx import env as E
    E._MEASURE_TICKETS.clear()
    E._MEASURE_TICKET_OPEN[0] = None
    yield
    E._MEASURE_TICKETS.clear()
    E._MEASURE_TICKET_OPEN[0] = None


# ------------------------------------------------------- 1. the protocol

def test_only_one_ticket_can_be_open_at_a_time():
    """Two open tickets would mean two rollout attempts in flight, and the
    pool serves one batch at a time."""
    from alphagrad.approx import env as E

    t = E.open_measure_ticket()
    assert E.current_measure_ticket() == t
    with pytest.raises(E.MeasureTicketError):
        E.open_measure_ticket()
    assert E.close_measure_ticket() == t
    assert E.current_measure_ticket() is None
    # And after closing, a new one opens and gets a DIFFERENT number.
    assert E.open_measure_ticket() != t


def test_a_rollout_submits_its_terminal_plans_exactly_once():
    from alphagrad.approx import env as E

    t = E.open_measure_ticket()
    E._record_measure_submission(t, _DoneFuture(None), [0, 1], 2, 4)
    with pytest.raises(E.MeasureTicketError):
        E._record_measure_submission(t, _DoneFuture(None), [0, 1], 2, 4)


def test_the_collected_rewards_land_on_their_own_environment_rows():
    """The submission order is the row order the pool was handed, and the
    trajectory's order is the environment index. They are joined here."""
    from alphagrad.approx.env import NUM_REWARDS
    from alphagrad.approx import env as E

    t = E.open_measure_ticket()
    # Rows submitted OUT OF ORDER on purpose: row 2 first, then 0, then 1.
    rows = [2, 0, 1]
    _, rewards, sentinel = _wire(3, 8, NUM_REWARDS)
    sentinel[0] = True                       # the row-2 plan sentinelled
    E._record_measure_submission(
        t, _DoneFuture((np.zeros((3, 8), np.uint8), rewards, sentinel)),
        rows, 3, 7)
    E.close_measure_ticket()
    out = E.collect_measurement(t)
    assert out["step"] == 7 and out["ticket"] == t
    # Submission slot k carried environment rows[k].
    for k, i in enumerate(rows):
        assert np.array_equal(out["rewards"][i], rewards[k])
        assert bool(out["sentinel"][i]) == bool(sentinel[k])
    # And the ticket is gone: a second collect is an error, not a repeat.
    with pytest.raises(E.MeasureTicketError):
        E.collect_measurement(t)


def test_a_missing_environment_raises_instead_of_leaving_a_zero_row():
    """A gap here would train one environment on another's measurement."""
    from alphagrad.approx.env import NUM_REWARDS
    from alphagrad.approx import env as E

    t = E.open_measure_ticket()
    _, rewards, sentinel = _wire(2, 8, NUM_REWARDS)
    E._record_measure_submission(
        t, _DoneFuture((np.zeros((2, 8), np.uint8), rewards, sentinel)),
        [0, 2], 3, 4)                        # environment 1 never submitted
    E.close_measure_ticket()
    with pytest.raises(E.MeasureTicketError):
        E.collect_measurement(t)


def test_a_ticket_that_never_reached_a_terminal_step_cannot_be_collected():
    from alphagrad.approx import env as E

    t = E.open_measure_ticket()
    E.close_measure_ticket()
    with pytest.raises(E.MeasureTicketError):
        E.collect_measurement(t)
    # ... but it CAN be dropped, which is what a discarded attempt does.
    assert E.drop_measurement(t) == {"rows": [], "drained": False}
    assert E.pending_measure_tickets() == []


# ---------------------------------------------------------- 2. the routing

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


class _FakePool:
    """Records which rows went where. Nothing here touches Ray."""

    def __init__(self, width, n_rewards):
        self.width = width
        self.n_rewards = n_rewards
        self.evaluated = []
        self.submitted = []
        self._eval_samples_ref = None

    def evaluate_batch(self, order_batch, specs_batch, step_batch, **kw):
        self.evaluated.append(list(int(s) for s in step_batch))
        return _wire(len(order_batch), self.width, self.n_rewards, base=100.0)

    def submit_batch(self, order_batch, specs_batch, step_batch, **kw):
        self.submitted.append(list(int(s) for s in step_batch))
        return _DoneFuture(
            _wire(len(order_batch), self.width, self.n_rewards, base=200.0))


def _pooled_env(pool):
    env = _small_env()
    return type(env)(
        env.config, env.args, env.consts, env.valid_vertices, env.num_envs,
        env.eval_args_samples,
        axis_state_static=env.axis_state_static,
        axis_valid_static=env.axis_valid_static,
        remote_pool=pool, remote_timeout_s=60.0)


def _call(env, steps, n_vertices=4):
    """Drive the batched remote callback for one step of `len(steps)` envs."""
    E = len(steps)
    z = np.zeros((1,), np.int32)
    order = np.zeros((E, n_vertices), np.int32)
    specs = np.zeros((E, n_vertices, 4), np.int32)
    face_specs = -np.ones((E, n_vertices, 2, 1, 3), np.int32)
    face_skips = np.zeros((E, n_vertices, 2), np.int32)
    step = np.asarray(steps, np.int32)
    fn = env.tokenize(batched=True)
    return fn(z, z, order, specs, face_specs, face_skips, None, step)


def test_without_a_ticket_every_row_is_measured_in_the_callback():
    """The synchronous path is unchanged: no ticket, no submission."""
    from alphagrad.approx.env import NUM_REWARDS

    pool = _FakePool(_small_env().obs_width, NUM_REWARDS)
    env = _pooled_env(pool)
    tk, rw = _call(env, [4, 4])
    assert pool.submitted == []
    assert pool.evaluated == [[4, 4]]
    assert rw.shape == (2, NUM_REWARDS)
    assert not np.all(rw == 0.0)


def test_with_a_ticket_open_the_terminal_rows_are_submitted_and_return_zeros():
    from alphagrad.approx.env import NUM_REWARDS
    from alphagrad.approx import env as E

    pool = _FakePool(_small_env().obs_width, NUM_REWARDS)
    env = _pooled_env(pool)
    t = E.open_measure_ticket()
    tk, rw = _call(env, [4, 4, 4])
    E.close_measure_ticket()
    assert pool.evaluated == []              # nothing measured in-line
    assert pool.submitted == [[4, 4, 4]]
    # THE PLACEHOLDER. Zero tokens decode to a zero delta count, and the zero
    # reward vector is what the driver overwrites at collect time.
    assert np.all(tk == 0) and np.all(rw == 0.0)
    assert E.measure_ticket_rows(t) == [0, 1, 2]
    out = E.collect_measurement(t)
    assert out["rewards"].shape == (3, NUM_REWARDS)
    assert not np.all(out["rewards"] == 0.0)


def test_a_non_terminal_step_is_still_measured_in_line_with_a_ticket_open():
    """Every step's TOKENS are read by the next step's encoder, so only the
    terminal row can be left in flight."""
    from alphagrad.approx.env import NUM_REWARDS
    from alphagrad.approx import env as E

    pool = _FakePool(_small_env().obs_width, NUM_REWARDS)
    env = _pooled_env(pool)
    E.open_measure_ticket()
    _call(env, [2, 2])
    assert pool.submitted == [] and pool.evaluated == [[2, 2]]


def test_a_mixed_step_splits_the_terminal_rows_off_and_keeps_the_rest_inline():
    from alphagrad.approx.env import NUM_REWARDS
    from alphagrad.approx import env as E

    pool = _FakePool(_small_env().obs_width, NUM_REWARDS)
    env = _pooled_env(pool)
    t = E.open_measure_ticket()
    tk, rw = _call(env, [4, 2, 4])
    E.close_measure_ticket()
    assert pool.evaluated == [[2]]
    assert pool.submitted == [[4, 4]]
    assert E.measure_ticket_rows(t) == [0, 2]
    # Row 1 was measured in-line and carries a real reward; the two terminal
    # rows are placeholders until the collect.
    assert not np.all(rw[1] == 0.0)
    assert np.all(rw[0] == 0.0) and np.all(rw[2] == 0.0)


# ---------------------------------------------------------- 3. the discard

def test_a_discarded_attempt_drains_its_submission_and_drops_the_result():
    """The attempt is gone, so its rewards must reach no trajectory. The
    future is still DRAINED: the actors are measuring it, and the repeat's
    own tokenization queues behind whatever is still running."""
    from alphagrad.approx.env import NUM_REWARDS
    from alphagrad.approx import env as E

    drained = {"n": 0}

    class _CountingFuture(_DoneFuture):
        def result(self, timeout=None):
            drained["n"] += 1
            return super().result(timeout)

    t = E.open_measure_ticket()
    E._record_measure_submission(
        t, _CountingFuture(_wire(2, 8, NUM_REWARDS)), [0, 1], 2, 4)
    E.close_measure_ticket()
    out = E.drop_measurement(t)
    assert out == {"rows": [0, 1], "drained": True}
    assert drained["n"] == 1
    # Nothing is left behind: the repeat opens a clean ticket.
    assert E.pending_measure_tickets() == []
    assert E.current_measure_ticket() is None
    t2 = E.open_measure_ticket()
    assert t2 != t


def test_a_discarded_attempt_whose_measurement_failed_does_not_raise():
    """Its result is thrown away either way; a raise here would kill a run
    over an episode nobody is going to use."""
    from alphagrad.approx import env as E

    class _Boom(_DoneFuture):
        def result(self, timeout=None):
            raise RuntimeError("actor died")

    t = E.open_measure_ticket()
    E._record_measure_submission(t, _Boom(None), [0], 1, 4)
    E.close_measure_ticket()
    assert E.drop_measurement(t)["drained"] is True


def test_the_discarded_attempts_actor_records_are_dropped_not_logged():
    """`episode_telemetry_restore` rolls back the TRAINER's containers; the
    measure actors are separate processes, so the driver drains them and
    throws the drain away. This pins the drain, which is what makes the
    dropping possible."""
    from alphagrad.approx.common.measure_pool import merge_pool_plan_records

    class _Actor:
        def __init__(self, recs):
            self._recs = recs

        class _M:
            def __init__(self, v):
                self._v = v

            def remote(self):
                return self._v

        def __getattr__(self, name):
            if name == "consume_plan_records":
                out = {"records": self._recs, "dropped": 0, "terminals":
                       len(self._recs), "actor_id": 1, "enabled": True}
                self._recs = []               # a drain EMPTIES the actor
                return _Actor._M(out)
            raise AttributeError(name)

    class _Pool:
        def __init__(self, actors):
            self._actors = actors

        def live_actors(self):
            return list(self._actors)

    import sys
    import types
    fake_ray = types.ModuleType("ray")
    fake_ray.get = lambda x, timeout=None: x
    saved = sys.modules.get("ray")
    sys.modules["ray"] = fake_ray
    try:
        pool = _Pool([_Actor([{"plan_index": 0}, {"plan_index": 1}])])
        first = merge_pool_plan_records(pool)
        assert len(first["records"]) == 2 and first["actors_polled"] == 1
        # The discarded attempt's records were taken out of the actor here;
        # the repeat's drain sees only the repeat's.
        second = merge_pool_plan_records(pool)
        assert second["records"] == [] and second["terminals"] == 0
    finally:
        if saved is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = saved


# ------------------------------------------------------------- 4. the pool

class _StubPool:
    """`CpuApproxPool.submit_batch` on a stub, so the threading is the only
    thing under test."""

    from alphagrad.approx.cpu_approx_pool import CpuApproxPool as _C

    submit_batch = _C.submit_batch
    has_batch_in_flight = _C.has_batch_in_flight

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._closed = False
        self._submit_exec = None
        self._submit_inflight = None
        self._gate = threading.Event()
        self.calls = []

    def evaluate_batch(self, *a, **kw):
        self.calls.append(a)
        self._gate.wait(5.0)
        return "measured"


def test_the_pool_takes_one_batch_at_a_time():
    """The same actors serve the next rollout's per-step tokenization, so a
    second batch in flight would starve it. That is an error, not a queue."""
    p = _StubPool()
    fut = p.submit_batch([1], [2], [3])
    assert p.has_batch_in_flight()
    with pytest.raises(RuntimeError):
        p.submit_batch([1], [2], [3])
    p._gate.set()
    assert fut.result(timeout=10) == "measured"
    assert not p.has_batch_in_flight()
    # Once it is back, the next batch is accepted.
    p._gate.clear()
    p._gate.set()
    assert p.submit_batch([4], [5], [6]).result(timeout=10) == "measured"
    p._submit_exec.shutdown(wait=True)


def test_a_submitted_batch_runs_the_same_evaluate_batch_the_caller_would():
    """The measurement is not moved, only the wait."""
    p = _StubPool()
    p._gate.set()
    fut = p.submit_batch(["order"], ["specs"], [4], init=False)
    assert fut.result(timeout=10) == "measured"
    assert p.calls == [(["order"], ["specs"], [4])]
    p._submit_exec.shutdown(wait=True)


def test_a_closed_pool_refuses_a_submission():
    p = _StubPool()
    p._closed = True
    with pytest.raises(RuntimeError):
        p.submit_batch([1], [2], [3])


def test_fold_face_stats_is_the_one_summation_rule():
    """The pipelined driver drains the actors at COLLECT time and folds the
    result in later; it must fold them the way the synchronous merge does."""
    from alphagrad.approx.common.measure_pool import fold_face_stats

    out = fold_face_stats({"applied": 2, "skipped": 1},
                          {"applied": 3, "skipped": 4,
                           "applied_fraction": 0.99, "name": "ignored"})
    assert out["applied"] == 5 and out["skipped"] == 5
    assert "name" not in out
    assert out["applied_fraction"] == pytest.approx(5 / 10)


def test_an_unused_pipeline_leaves_the_executor_unbuilt():
    """`--measure-pipeline 0` must not start a thread."""
    p = _StubPool()
    assert p._submit_exec is None
    assert not p.has_batch_in_flight()


def test_concurrent_futures_is_what_the_pool_hands_back():
    p = _StubPool()
    p._gate.set()
    fut = p.submit_batch([1], [2], [3])
    assert isinstance(fut, _cf.Future)
    fut.result(timeout=10)
    p._submit_exec.shutdown(wait=True)
