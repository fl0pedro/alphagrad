# dsnn-dfw.201 (owner ruling 2026-09-26, QD a): the pool takes each terminal call's plan records
# and refusal counts from its actor right after the call. An actor that the OOM path recycles or
# the deadline kills then takes none of them with it. In the NN256 pilot the recycle took 483
# records and all 151 run-time OOM refusals (jobs 68277, 68278, 68316).
from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

NUM_REWARDS = 12
COS = 6
FROB = 7
TOK = 8
CLEAN, OOM, STUCK = 0, 1, 2
SCORED = np.array([-0.2, -0.1, -13.9, -0.3, -0.4, -3.2, 0.0, 0.0, -1.0, 0.0, -1.0, -2.0],
                  np.float32)
MEASURED = np.array([-0.2, -0.1, -0.5, -0.3, -0.4, -0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.3],
                    np.float32)


class _Future:
    def __init__(self, fn, *a, **k):
        self._fn, self._a, self._k = fn, a, k

    def result(self):
        return self._fn(*self._a, **self._k)


class _Stuck:
    pass


@pytest.fixture
def fake_ray(monkeypatch):
    class GetTimeoutError(Exception):
        pass

    class RayActorError(Exception):
        pass

    def _get(fut, timeout=None):
        if isinstance(fut, _Stuck):
            raise GetTimeoutError("the call is still running at its deadline")
        return fut.result() if isinstance(fut, _Future) else fut

    def _wait(futures, num_returns=1, timeout=None):
        ready = [f for f in futures if not isinstance(f, _Stuck)][:num_returns]
        return ready, [f for f in futures if f not in ready]

    def _kill(actor, no_restart=False):
        actor.killed = True

    exc = types.SimpleNamespace(GetTimeoutError=GetTimeoutError,
                                RayActorError=RayActorError)
    ray = types.SimpleNamespace(get=_get, wait=_wait, kill=_kill,
                                cancel=lambda *a, **k: None, exceptions=exc)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.exceptions", exc)
    return ray


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return _Future(self._fn, *a, **k)


class _Evaluate:
    def __init__(self, actor):
        self._actor = actor

    def remote(self, order, specs, step, **kw):
        if int(np.asarray(order).reshape(-1)[1]) - 10 == STUCK:
            return _Stuck()
        return _Future(self._actor._evaluate, order, specs, step, **kw)


class _Actor:
    # order[0] is the plan's id and order[1] its kind plus 10. As in the real actor, a call leaves its
    # record and its refusal count in the actor until a drain takes them.
    def __init__(self, name):
        self.name = name
        self.killed = False
        self.records: list = []
        self.refused: dict = {}
        self.terminals = 0
        self._flag = False
        self.evaluate = _Evaluate(self)
        self.pop_oom_flag = _Remote(self._pop)
        self.ready = _Remote(lambda: True)
        self.consume_call_telemetry = _Remote(self._take)
        self.consume_plan_records = _Remote(self._plan)
        self.consume_collapse_stats = _Remote(self._collapse)
        self.consume_face_stats = _Remote(lambda: {})

    def _evaluate(self, order, specs, step, **kw):
        plan, kind = (int(x) for x in np.asarray(order).reshape(-1)[:2])
        kind -= 10
        self.terminals += 1
        rec = {"plan": plan, "pid": self.name}
        if kind == OOM:
            self._flag = True
            rec["refused"] = "oom:measurement"
            for k in ("oom", "total", "scored"):
                self.refused[k] = self.refused.get(k, 0) + 1
        self.records.append(rec)
        row = SCORED if kind == OOM else MEASURED
        return (np.full((TOK,), 7, np.int32), np.full((TOK,), 3, np.int32), row.copy())

    def _pop(self):
        was, self._flag = self._flag, False
        return was

    def _plan(self):
        out = {"records": self.records, "dropped": 0, "terminals": self.terminals,
               "actor_id": self.name, "enabled": True,
               "mem_parity": {"records": [], "measured": 0, "dropped": 0},
               "paired_ref": {"records": [], "dropped": 0}}
        self.records, self.terminals = [], 0
        return out

    def _collapse(self):
        out = {f"refused_{k}": v for k, v in self.refused.items()}
        self.refused = {}
        return out

    def _take(self):
        return {"plan": self._plan(), "collapse": self._collapse(), "face": {}}


def _pool(n_actors, timeout_s=300.0, respawn=True):
    from alphagrad.approx.cpu_approx_pool import CpuApproxPool

    first = [_Actor(f"a{i}") for i in range(n_actors)]
    fresh: list = []

    def factory():
        fresh.append(_Actor(f"fresh{len(fresh)}"))
        return fresh[-1]

    pool = CpuApproxPool(
        first, timeout_s=timeout_s, respawn_factory=factory if respawn else None,
        max_tokens=TOK, num_rewards=NUM_REWARDS, cosine_sim_idx=COS,
        frob_residual_idx=FROB)
    return pool, first, fresh


def _order(plan, kind):
    return np.array([plan, 10 + kind, 20], np.int32)


def _batch(pool, kinds):
    n = len(kinds)
    return pool.evaluate_batch([_order(i, k) for i, k in enumerate(kinds)],
                               [np.full((3, 1, 3), -1, np.int32)] * n, [3] * n)


@pytest.fixture
def trainer_counts(monkeypatch):
    from alphagrad.approx import env as env_mod
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    yield env_mod
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def test_an_actor_the_oom_path_recycles_leaves_its_records_and_its_refusal(fake_ray):
    from alphagrad.approx.common.measure_pool import (
        merge_pool_collapse_stats, merge_pool_plan_records)

    pool, (actor,), fresh = _pool(1)
    _tok, _eqn, rewards, mask = _batch(pool, [CLEAN, OOM, CLEAN])
    assert actor.killed and len(fresh) == 1 and pool.stats()["oom_recycles"] == 1
    assert mask.tolist() == [False, False, False]
    drain = merge_pool_plan_records(pool)
    assert sorted(r["plan"] for r in drain["records"]) == [0, 1, 2]
    assert drain["terminals"] == 3 and drain["dropped"] == 0
    refused = [r for r in drain["records"] if r.get("refused")]
    assert [(r["plan"], r["refused"], r["actor"]) for r in refused] == [
        (1, "oom:measurement", "a0")]
    counts = merge_pool_collapse_stats(pool, {})
    assert (counts["refused_oom"], counts["refused_total"], counts["refused_scored"]) == (1, 1, 1)
    again = merge_pool_plan_records(pool)
    assert again["records"] == [] and again["terminals"] == 0, "a drain takes each record once"
    assert merge_pool_collapse_stats(pool, {}) == {}


def test_an_actor_the_deadline_kills_leaves_the_records_of_its_earlier_plans(
        fake_ray, trainer_counts):
    from alphagrad.approx.common.measure_pool import merge_pool_plan_records

    pool, actors, _fresh = _pool(2, timeout_s=0.01, respawn=False)
    _tok, _eqn, rewards, mask = _batch(pool, [CLEAN, CLEAN, STUCK])
    killed = [a for a in actors if a.killed]
    assert len(killed) == 1 and pool.stats()["timeouts"] == 1
    assert mask.tolist() == [False, False, True]
    drain = merge_pool_plan_records(pool)
    assert sorted(r["plan"] for r in drain["records"]) == [0, 1]
    assert {r["pid"] for r in drain["records"]} == {"a0", "a1"}
    assert killed[0].name in {r["actor"] for r in drain["records"]}
    local = trainer_counts.consume_plan_records()["records"]
    assert [r["refused"] for r in local] == ["timeout"], "the killed call keeps its own record"


def test_the_single_dispatch_takes_the_records_before_the_recycle(fake_ray):
    from alphagrad.approx.common.measure_pool import (
        merge_pool_collapse_stats, merge_pool_plan_records)

    pool, (actor,), fresh = _pool(1)
    pool.evaluate(_order(7, OOM), np.full((3, 1, 3), -1, np.int32), 3, None)
    assert actor.killed and pool.live_actors() == [fresh[0]]
    assert [r["plan"] for r in merge_pool_plan_records(pool)["records"]] == [7]
    assert merge_pool_collapse_stats(pool, {})["refused_oom"] == 1


def test_a_non_terminal_call_is_not_taken(fake_ray):
    pool, (actor,), _fresh = _pool(1)
    pool.evaluate_batch([_order(0, CLEAN)],
                        [np.full((3, 1, 3), -1, np.int32)], [1])
    assert pool.take_kept("plan")["calls"] == 0
    assert [r["plan"] for r in actor.records] == [0]


def test_a_take_that_fails_is_counted_and_named(fake_ray, capsys):
    pool, (actor,), _fresh = _pool(1)

    def _broken():
        raise RuntimeError("the actor lost its records")

    actor.consume_call_telemetry = _Remote(_broken)
    _batch(pool, [CLEAN])
    assert pool.stats()["takes_failed"] == 1
    assert "could not be taken" in capsys.readouterr().out


def _unwrapped(cls):
    for attr in ("__ray_actor_class__", "__ray_metadata__"):
        inner = getattr(cls, attr, None)
        if inner is None:
            continue
        if attr == "__ray_metadata__":
            inner = getattr(inner, "modified_class", None) or getattr(inner, "cls", None)
        if inner is not None:
            return inner
    return cls


def test_the_measure_actor_hands_over_one_calls_records_and_keeps_the_episode_open(
        trainer_counts):
    env_mod = trainer_counts
    import alphagrad.approx.cpu_approx_actors as actors_mod

    cls = _unwrapped(actors_mod.CpuApproximationActor)
    fake = types.SimpleNamespace(_actor_id=5, _slot=1, _gpu_uuid=None, _device="2", _pid=4242)
    for name in ("consume_plan_records", "consume_collapse_stats", "consume_face_stats"):
        setattr(fake, name, types.MethodType(cls.__dict__.get(name) or getattr(cls, name), fake))
    episode = env_mod._MEASURE_EPISODE
    saved = {k: (list(v) if isinstance(v, list) else v) for k, v in episode.items()}
    try:
        episode.update(key=b"ep", label="3", n_plans=2, n_measured=2, secs=[1.0, 1.0],
                       cand_secs=[0.5, 0.5], ref_secs=[0.5, 0.5])
        env_mod._PLAN_RECORDS.append({"plan_hash": "aa", "refused": "oom:measurement"})
        env_mod._PLAN_LOG_TERMINALS[0] += 1
        env_mod._record_refusal("oom:measurement", scored=True)
        take = cls.__dict__.get("consume_call_telemetry") or getattr(cls, "consume_call_telemetry")
        out = take(fake)
        assert [r["plan_hash"] for r in out["plan"]["records"]] == ["aa"]
        assert out["plan"]["records"][0]["actor_id"] == {"pid": 4242, "slot": 1, "actor": 5}
        assert out["plan"]["terminals"] == 1 and out["plan"]["actor_id"] == 5
        assert out["collapse"]["refused_oom"] == 1 and out["collapse"]["refused_scored"] == 1
        assert "applied_fraction" in out["face"]
        assert episode["n_plans"] == 2, "the dedupe's plan index survives a take"
        env_mod.consume_plan_records()
        assert episode["n_plans"] == 0, "the episode drain still ends the episode"
    finally:
        episode.clear()
        episode.update(saved)
