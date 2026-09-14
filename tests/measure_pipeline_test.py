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


# ---------------------------- 5. the SYNCHRONOUS discard (job 65684, A5)

def test_a_sync_repeat_leaves_one_set_of_records_stamped_with_the_repeat():
    """THE DEFECT JOB 65684 FOUND, MODELLED.

    All three arms of that job hit one window-bin overflow at episode 0 and
    repeated it. The two pipelined arms kept 16 plan records for the episode.
    The SYNCHRONOUS arm kept 32: the discarded attempt's sixteen beside the
    repeat's sixteen, every one stamped `attempt: 0`.

    Two things were missing on that path and both are modelled here. The
    trainer's snapshot restore cannot reach a MEASURE ACTOR, which is a
    separate process, so the actors' records survived the discard. And
    `_record_plan` stamps the WRITING process's attempt counter, which in an
    actor is always 0, so the repeat's records did not say so either.

    The driver now drains the actors on every discard and drops the drain,
    and stamps the surviving pooled records from its own counter. This is
    that sequence, against the real `episode_stream.run_episode`, the real
    `env` containers and the real `merge_pool_plan_records`.
    """
    import sys
    import types

    from alphagrad.approx import env as ENV
    from alphagrad.approx.common import episode_stream as ES

    class _Actor:
        """One measure actor. A drain EMPTIES it, as the real one does."""

        def __init__(self):
            self.records = []

        class _M:
            def __init__(self, v):
                self._v = v

            def remote(self):
                return self._v

        def __getattr__(self, name):
            if name == "consume_plan_records":
                out = {"records": self.records, "dropped": 0,
                       "terminals": len(self.records), "actor_id": 7,
                       "enabled": True}
                self.records = []
                return _Actor._M(out)
            raise AttributeError(name)

    class _Pool:
        def __init__(self, actor):
            self._actor = actor

        def live_actors(self):
            return [self._actor]

    fake_ray = types.ModuleType("ray")
    fake_ray.get = lambda x, timeout=None: x
    saved_ray = sys.modules.get("ray")
    sys.modules["ray"] = fake_ray
    outer = ENV.episode_telemetry_snapshot()
    try:
        from alphagrad.approx.common.measure_pool import (
            merge_pool_plan_records)

        ENV._PLAN_RECORDS.clear()
        ENV._PLAN_LOG_TERMINALS[0] = 0
        actor = _Actor()
        pool = _Pool(actor)
        policy = ES.BinPolicy(8, history=4, margin=2.0)
        held = {"attempt": 0}

        def attempt(n):
            # What the driver does at the top of every attempt.
            held["snapshot"] = ENV.episode_telemetry_snapshot()
            ENV.set_plan_log_attempt(held["attempt"])
            # THE MEASURE ACTORS write this episode's terminal plans. The
            # actor's own stamp is 0 whatever the driver's counter says,
            # because nobody advances a counter in that process.
            for e in range(3):
                actor.records.append({"env_index": -1, "bin": n,
                                      "attempt": 0})
            return "attempt-%d" % n, (
                ES.StreamOverflow(env_index=1, step=4, length=300, log2=n)
                if n == 8 else None)

        def discard(_result):
            # WHAT `_ep_discard` NOW DOES, in order: drain the actors and
            # throw the drain away, then put this process's containers back.
            merge_pool_plan_records(pool)
            ENV.episode_telemetry_restore(held["snapshot"])
            held["attempt"] += 1

        assert ES.run_episode(policy, "episode 0", attempt,
                              log=lambda _l: None,
                              on_discard=discard) == "attempt-9"

        # THE DRAIN THE EPISODE'S LOGGING TAKES. One set, not two.
        drained = merge_pool_plan_records(pool)
        assert len(drained["records"]) == 3, drained["records"]
        assert [r["bin"] for r in drained["records"]] == [9, 9, 9]
        # And the driver stamps its own counter over the actor's, which is
        # the only place the repeat is known.
        for _r in drained["records"]:
            _r["attempt"] = held["attempt"]
        assert [r["attempt"] for r in drained["records"]] == [1, 1, 1]
    finally:
        ENV.episode_telemetry_restore(outer)
        ENV.set_plan_log_attempt(0)
        if saved_ray is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = saved_ray


def test_without_the_actor_drain_a_sync_repeat_keeps_both_attempts():
    """The defect itself, so the test above is known to be testing something.

    Same sequence with the drain removed from the discard: the actors keep
    both attempts and the log would hold six records where three were
    measured twice. That is job 65684's `(0, 0): 32`.
    """
    import sys
    import types

    from alphagrad.approx.common import episode_stream as ES

    class _Actor:
        def __init__(self):
            self.records = []

        class _M:
            def __init__(self, v):
                self._v = v

            def remote(self):
                return self._v

        def __getattr__(self, name):
            if name == "consume_plan_records":
                out = {"records": self.records, "dropped": 0,
                       "terminals": len(self.records), "actor_id": 7,
                       "enabled": True}
                self.records = []
                return _Actor._M(out)
            raise AttributeError(name)

    class _Pool:
        def __init__(self, actor):
            self._actor = actor

        def live_actors(self):
            return [self._actor]

    fake_ray = types.ModuleType("ray")
    fake_ray.get = lambda x, timeout=None: x
    saved_ray = sys.modules.get("ray")
    sys.modules["ray"] = fake_ray
    try:
        from alphagrad.approx.common.measure_pool import (
            merge_pool_plan_records)

        actor = _Actor()
        pool = _Pool(actor)
        policy = ES.BinPolicy(8, history=4, margin=2.0)

        def attempt(n):
            for _e in range(3):
                actor.records.append({"env_index": -1, "bin": n,
                                      "attempt": 0})
            return "attempt-%d" % n, (
                ES.StreamOverflow(env_index=1, step=4, length=300, log2=n)
                if n == 8 else None)

        ES.run_episode(policy, "episode 0", attempt, log=lambda _l: None,
                       on_discard=lambda _r: None)
        drained = merge_pool_plan_records(pool)
        assert len(drained["records"]) == 6
        assert [r["bin"] for r in drained["records"]] == [8, 8, 8, 9, 9, 9]
        assert {r["attempt"] for r in drained["records"]} == {0}
    finally:
        if saved_ray is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = saved_ray


# ------------------------------- 6. every counter host_log drains is parked

# THE PARTITION. `host_log` empties about a dozen host counters with calls
# that POP. Pipelined, `host_log` for episode e runs after episode e+1's
# rollout has already added to them, so an unparked one reports two episodes
# under a single number and then nothing under the next (job 65684:
# `prof/env_cb_host n=190` at ep0, a blank `live-faces` line at ep2).
#
# There are exactly two ways a counter can be safe, and every call in
# `host_log` has to be one of them.
#
# A. Its container is named in `env._EPISODE_TELEMETRY_NAMES`. The driver
#    parks that whole set with `episode_telemetry_snapshot` at collect time
#    and restores it around the logging, so the call sees this episode's.
_PARKED_BY_THE_SNAPSHOT = {
    "consume_degenerate_plan_count",
    "consume_fidelity_stats",
    "consume_memory_compression_stats",
    "consume_sparsity_stats",
    "consume_token_length_stats",
    "consume_tokenization_truncation_stats",
    "consume_truncated_plan_count",
    "consume_untraceable_plan_count",
    "consume_zero_work_plan_count",
}
# B. Its call site reads `_POOL_DRAIN` first, because the snapshot has no
#    name for it: it lives on an object or in another module's globals.
_PARKED_BY_THE_CALL_SITE = {
    "consume_live_chain_stats",
    "consume_per_face_stats",
    "consume_probe_failure_stats",
    "consume_static_peak_fallbacks",
    "consume_stats",          # _LIVE_FACES and _EDGE_TABLE
    "_merge_cs", "_merge_pf", # the measure actors own drains
    "_plog_consume", "_plog_merge",
}


def _host_log_ast():
    import ast
    import inspect

    from alphagrad.approx import ppo

    tree = ast.parse(inspect.getsource(ppo))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "host_log":
            return node
    raise AssertionError("host_log not found in ppo.py")


def _draining_calls(fn_ast):
    """Every call in `fn_ast` that EMPTIES a per-episode counter."""
    import ast

    out = {}
    for n in ast.walk(fn_ast):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else None)
        if name is None:
            continue
        if (name.startswith("consume_") or name.startswith("_merge_")
                or name in ("_plog_consume", "_plog_merge")):
            out.setdefault(name, []).append(n.lineno)
    return out


def test_every_counter_host_log_drains_is_parked_one_way_or_the_other():
    """A new `consume_*` in `host_log` fails here until it is parked.

    Silence is what makes this class of defect expensive: an unparked
    counter reports plausible numbers under the wrong episode, and nothing
    says so. So the rule is stated as a partition and checked as one.
    """
    known = _PARKED_BY_THE_SNAPSHOT | _PARKED_BY_THE_CALL_SITE
    found = _draining_calls(_host_log_ast())
    unknown = {k: v for k, v in found.items() if k not in known}
    assert not unknown, (
        f"{sorted(unknown)} empties a per-episode counter inside host_log and "
        f"is in neither parking list. Either its container is named in "
        f"env._EPISODE_TELEMETRY_NAMES (add it to _PARKED_BY_THE_SNAPSHOT) or "
        f"its call site has to read _POOL_DRAIN first (add it to "
        f"_PARKED_BY_THE_CALL_SITE and to _drain_measure_telemetry).")
    # And the lists are not allowed to rot: every name in them is still called.
    stale = known - set(found)
    assert not stale, (
        f"{sorted(stale)} is listed as parked but host_log no longer calls it")


def test_the_call_site_parked_counters_read_the_park_before_they_drain():
    """Group B's half of the partition, as source.

    Each of these has to appear inside an expression that mentions
    `_POOL_DRAIN`, because that is the whole mechanism: with a park in hand
    the call does not run at all.
    """
    import ast
    import inspect

    from alphagrad.approx import ppo

    src = inspect.getsource(ppo)
    fn = _host_log_ast()
    # Every statement of host_log that mentions _POOL_DRAIN, by line span.
    guarded_lines = set()
    for n in ast.walk(fn):
        if not isinstance(n, (ast.If, ast.IfExp, ast.Assign, ast.Expr)):
            continue
        seg = ast.get_source_segment(src, n) or ""
        if "_POOL_DRAIN" in seg:
            guarded_lines.update(
                range(n.lineno, (getattr(n, "end_lineno", n.lineno) or
                                 n.lineno) + 1))
    found = _draining_calls(fn)
    for name in sorted(_PARKED_BY_THE_CALL_SITE):
        if name not in found:
            continue
        for line in found[name]:
            assert line in guarded_lines, (
                f"{name} at line {line} of ppo.py drains a counter without "
                f"reading _POOL_DRAIN first; pipelined, it would report the "
                f"wrong episode's number")


# ----------------------- 7. the zero of a one-element counter (job 65715)

def test_the_fresh_state_keeps_every_one_element_counter_one_element_long():
    """CANARY JOB 65715. Nine of the per-episode accumulators are `[0]`, a
    number every reader indexes. The driver used to reset them by emptying
    every list, which is not zero but a MISSING ELEMENT, and the next
    `consume_plan_records` raised IndexError.
    """
    from alphagrad.approx import env as ENV

    fixed = ENV.episode_telemetry_fixed_counters()
    # The ones the traceback named have to be in there.
    for name in ("_PLAN_LOG_TERMINALS", "_PLAN_LOG_DROPPED",
                 "_TOKLEN_SUM", "_DEGENERATE_PLANS"):
        assert name in fixed, (name, fixed)
    outer = ENV.episode_telemetry_snapshot()
    try:
        # Fill them, then reset, and check the SHAPE survived.
        ENV._PLAN_LOG_TERMINALS[0] = 7
        ENV._PLAN_LOG_DROPPED[0] = 3
        ENV._TOKLEN_SUM[0] = 99
        ENV._PLAN_RECORDS.append({"what": "a record"})
        ENV.episode_telemetry_reset()
        for name in fixed:
            assert len(getattr(ENV, name)) == 1, name
            assert getattr(ENV, name)[0] == 0, name
        # And the collections really are empty.
        assert ENV._PLAN_RECORDS == []
    finally:
        ENV.episode_telemetry_restore(outer)


def test_a_discard_drain_and_restore_leaves_consume_plan_records_well_formed():
    """(a) of the follow-up. The exact call that raised, after the exact
    sequence a discarded attempt puts the containers through."""
    from alphagrad.approx import env as ENV

    outer = ENV.episode_telemetry_snapshot()
    try:
        ENV._PLAN_RECORDS.clear()
        ENV._PLAN_LOG_TERMINALS[0] = 0
        ENV._PLAN_LOG_DROPPED[0] = 0
        # What the driver does at the top of an attempt.
        snap = ENV.episode_telemetry_snapshot()
        # What one attempt's callbacks do.
        ENV._record_plan({"env_index": 0})
        ENV._PLAN_LOG_TERMINALS[0] += 1
        # What the discard does: drain (and drop), then put the containers
        # back to what the attempt found.
        dropped = ENV.consume_plan_records()
        assert dropped["terminals"] == 1
        ENV.episode_telemetry_restore(snap)
        # And now the call that raised in job 65715.
        out = ENV.consume_plan_records()
        assert isinstance(out, dict)
        assert out["records"] == [] and out["terminals"] == 0
        assert out["dropped"] == 0
        for key in ("compile_fallbacks", "compile_fallbacks_total",
                    "toolchain_ok", "mem_parity", "paired_ref", "enabled",
                    "pid"):
            assert key in out, key
    finally:
        ENV.episode_telemetry_restore(outer)


def test_two_overflow_repeats_then_a_collect_leaves_the_counters_readable():
    """(b) of the follow-up: the sequence of job 65715 at unit level.

    Pipeline armed, a pool present, TWO window-bin repeats of episode 0, then
    the successful attempt's collect, then the next episode's first read. The
    collect is where the reset happens, and the read after it is what used to
    raise.
    """
    import sys
    import types

    from alphagrad.approx import env as ENV
    from alphagrad.approx.common import episode_stream as ES

    class _Actor:
        def __init__(self):
            self.records = []

        class _M:
            def __init__(self, v):
                self._v = v

            def remote(self):
                return self._v

        def __getattr__(self, name):
            if name == "consume_plan_records":
                out = {"records": self.records, "dropped": 0,
                       "terminals": len(self.records), "actor_id": 1,
                       "enabled": True}
                self.records = []
                return _Actor._M(out)
            raise AttributeError(name)

    class _Pool:
        def __init__(self, actor):
            self._actor = actor

        def live_actors(self):
            return [self._actor]

    fake_ray = types.ModuleType("ray")
    fake_ray.get = lambda x, timeout=None: x
    saved_ray = sys.modules.get("ray")
    sys.modules["ray"] = fake_ray
    outer = ENV.episode_telemetry_snapshot()
    try:
        from alphagrad.approx.common.measure_pool import (
            merge_pool_plan_records)

        ENV.episode_telemetry_reset()
        actor = _Actor()
        pool = _Pool(actor)
        policy = ES.BinPolicy(8, history=4, margin=2.0)
        held = {"attempt": 0}

        def attempt(n):
            held["snapshot"] = ENV.episode_telemetry_snapshot()
            ENV.set_plan_log_attempt(held["attempt"])
            # One episode's worth of trainer-side counting, including the
            # one-element counters the crash was about.
            for e in range(2):
                ENV._record_plan({"env_index": e, "bin": n})
                actor.records.append({"env_index": -1, "bin": n})
            ENV._PLAN_LOG_TERMINALS[0] += 2
            ENV._record_token_length(1234)
            # TWO repeats: the bin has to climb 8 -> 9 -> 10.
            return "attempt-%d" % n, (
                ES.StreamOverflow(env_index=1, step=4, length=1 << (n + 1),
                                  log2=n)
                if n < 10 else None)

        def discard(_result):
            merge_pool_plan_records(pool)
            ENV.consume_plan_records()
            ENV.episode_telemetry_restore(held["snapshot"])
            held["attempt"] += 1

        assert ES.run_episode(policy, "episode 0", attempt,
                              log=lambda _l: None,
                              on_discard=discard) == "attempt-10"
        assert held["attempt"] == 2, "the sequence needs TWO repeats"

        # THE COLLECT. Park this episode's counters and reset the live ones.
        parked = ENV.episode_telemetry_snapshot()
        ENV.episode_telemetry_reset()

        # THE NEXT EPISODE'S FIRST READS, which is where job 65715 raised.
        ENV._record_token_length(99)          # indexes _TOKLEN_SUM[0]
        ENV._record_plan({"env_index": 0, "bin": "next"})
        ENV._PLAN_LOG_TERMINALS[0] += 1
        live = ENV.consume_plan_records()
        assert live["terminals"] == 1
        assert [r["bin"] for r in live["records"]] == ["next"]

        # And the parked episode still reads back as its own.
        ENV.episode_telemetry_restore(parked)
        kept = ENV.consume_plan_records()
        assert kept["terminals"] == 2
        assert [r["bin"] for r in kept["records"]] == [10, 10]
        assert [r["attempt"] for r in kept["records"]] == [2, 2]
    finally:
        ENV.episode_telemetry_restore(outer)
        ENV.set_plan_log_attempt(0)
        if saved_ray is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = saved_ray


def test_no_driver_fabricates_a_zero_by_emptying_every_list():
    """The shape of the bug, pinned as source.

    `episode_telemetry_reset` exists so nobody has to know which of these
    containers are counters and which are collections. A driver that builds
    its own zero is the defect of job 65715 coming back.
    """
    import inspect

    from alphagrad.approx import ppo

    src = inspect.getsource(ppo)
    assert "episode_telemetry_reset()" in src, (
        "ppo.py no longer resets the per-episode accumulators through env.py")
    assert "_telemetry_zero" not in src, (
        "ppo.py fabricates its own telemetry zero again; env.py owns what "
        "zero means for its containers (canary job 65715)")
