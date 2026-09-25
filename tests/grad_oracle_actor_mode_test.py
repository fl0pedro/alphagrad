import inspect
import os
import sys
import threading
import time
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest                                                   # noqa: E402

from alphagrad.approx.common import grad_oracle_async as G      # noqa: E402


class _Ref:
    def __init__(self, value, ready):
        self.value = value
        self.ready = ready


class _Stored:
    def __init__(self, value):
        self.value = value


class _Actor:
    def __init__(self, box, hang):
        self.hang = hang
        self.box = box
        self.check = types.SimpleNamespace(remote=self._remote)

    def _remote(self, args_np, order, probe_seed, rule=None):
        self.box["sent"].append(args_np)
        # Ray gives the method the value of a top-level object ref.
        if isinstance(args_np, _Stored):
            args_np = args_np.value
        self.box["calls"].append((args_np, tuple(order), int(probe_seed), rule))
        return _Ref(("pass", 1e-15), ready=not self.hang)


@pytest.fixture
def fake_ray(monkeypatch):
    box = {"calls": [], "killed": [], "actors": [], "puts": [], "sent": []}

    def wait(refs, num_returns=1, timeout=None):
        return ([r for r in refs if r.ready], [r for r in refs if not r.ready])

    def kill(actor, no_restart=False):
        box["killed"].append((actor, bool(no_restart)))

    def put(value):
        box["puts"].append(_Stored(value))
        return box["puts"][-1]

    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(
        wait=wait, get=lambda ref: ref.value, kill=kill, put=put))
    return box


def _factory(box, hang_first=False):
    def make():
        box["actors"].append(
            _Actor(box, hang=hang_first and not box["actors"]))
        return box["actors"][-1]
    return make


def _in_process_check(*_a):
    raise AssertionError("the check ran in the trainer process")


# dsnn-dfw.22: with a measurement pool the trainer builds the oracle as a Ray actor, not a thread.
def test_a_run_with_a_measurement_pool_builds_the_oracle_as_an_actor():
    import alphagrad.approx.ppo as ppo
    src = inspect.getsource(ppo.main)
    assert "if _have_measure_pool:" in src, (
        "the trainer has no branch that builds the oracle on the pool")
    pool = src[src.index("if _have_measure_pool:"):]
    pool = pool[:pool.index("\n        el")]
    assert "_AsyncGradOracle(" in pool
    assert "actor_factory=_make_oracle_actor_factory(" in pool


def _fake_ray_remote(seen, options_seen):
    def remote(**kw):
        seen.update(kw)

        def wrap(cls):
            def options(**okw):
                options_seen.update(okw)
                return types.SimpleNamespace(
                    remote=lambda *a: ("handle", cls.__name__, a))
            return types.SimpleNamespace(options=options)
        return wrap
    return remote


def test_the_oracle_actor_is_one_process_with_no_gpu(monkeypatch):
    seen, options_seen = {}, {}
    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(
        remote=_fake_ray_remote(seen, options_seen)))
    handle = G.make_ray_oracle_actor_factory({"example": "Helmholtz"})()
    assert seen == {"num_cpus": G.ORACLE_ACTOR_NUM_CPUS, "num_gpus": 0}
    assert handle[0] == "handle" and handle[2] == ({"example": "Helmholtz"},)
    # dsnn-dfw.206 / dsnn-dfw.229: the actor sees no GPU and runs JAX on the
    # CPU, so it holds no context on a measure device.
    env_vars = options_seen["runtime_env"]["env_vars"]
    assert env_vars["CUDA_VISIBLE_DEVICES"] == ""
    assert env_vars["JAX_PLATFORMS"] == "cpu"
    assert env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"


def test_the_check_runs_in_the_actor_and_no_trainer_thread_starts(fake_ray):
    before = set(threading.enumerate())
    oracle = G.AsyncGradOracle(
        _in_process_check, timeout_s=60.0, actor_factory=_factory(fake_ray),
        arg_resolver=lambda ep: {"frozen_for_episode": ep})
    assert oracle.submit(3, 7, [{"order": [5, 4, 3], "plan_hashes": ["h"]}]) == 1
    res = oracle.take_results()
    oracle.close()
    assert [r["status"] for r in res] == ["pass"], res
    assert fake_ray["calls"] == [({"frozen_for_episode": 3}, (5, 4, 3), 7, None)]
    started = [t.name for t in set(threading.enumerate()) - before]
    assert "grad-oracle" not in started, started


# dsnn-dfw.202: every order sent its own copy of the episode's arguments, 14.4 GB each at B=64.
def test_the_orders_of_one_episode_share_one_copy_of_its_arguments(fake_ray):
    oracle = G.AsyncGradOracle(
        _in_process_check, timeout_s=60.0, actor_factory=_factory(fake_ray),
        arg_resolver=lambda ep: {"frozen_for_episode": ep})
    jobs = [{"order": [i, 9, 8], "plan_hashes": [f"h{i}"]} for i in range(3)]
    assert oracle.submit(4, 7, jobs) == 3
    assert len(fake_ray["puts"]) == 1, fake_ray["puts"]
    assert all(s is fake_ray["puts"][0] for s in fake_ray["sent"]), fake_ray["sent"]
    assert [c[0] for c in fake_ray["calls"]] == [{"frozen_for_episode": 4}] * 3
    assert oracle.submit(5, 7, [{"order": [1, 2], "plan_hashes": ["h5"]}]) == 1
    assert [p.value for p in fake_ray["puts"]] == [
        {"frozen_for_episode": 4}, {"frozen_for_episode": 5}]
    assert [r["status"] for r in oracle.take_results()] == ["pass"] * 4
    oracle.close()


# The pre-fix thread could only disown a hung check; a process can be killed and replaced.
def test_a_check_over_the_timeout_kills_the_actor_and_starts_a_fresh_one(fake_ray):
    oracle = G.AsyncGradOracle(
        _in_process_check, timeout_s=5.0,
        actor_factory=_factory(fake_ray, hang_first=True),
        arg_resolver=lambda ep: ep)
    oracle.submit(1, 0, [{"order": [2, 1], "plan_hashes": ["a"]}])
    res = oracle.take_results(now=time.monotonic() + 10.0)
    assert [r["status"] for r in res] == ["timeout"], res
    assert fake_ray["killed"] == [(fake_ray["actors"][0], True)]
    assert len(fake_ray["actors"]) == 2
    assert oracle.counts()["actor_kills"] == 1
    oracle.submit(2, 0, [{"order": [1, 2], "plan_hashes": ["b"]}])
    assert [r["status"] for r in oracle.take_results()] == ["pass"]
    oracle.close()


class _PastTheOracleCheck(Exception):
    pass


def _main_until_after_the_check(monkeypatch, *flags):
    import alphagrad.approx.ppo as ppo

    def stop(*_a, **_k):
        raise _PastTheOracleCheck

    monkeypatch.setattr(ppo, "_apply_variant_preset", stop)
    monkeypatch.setattr(sys, "argv", [
        "ppo.py", "--example", "Helmholtz", "--dataset", "none",
        "--wandb", "disabled", "--grad-oracle", "reference", *flags])
    keep = dict(os.environ)
    try:
        ppo.main()
    finally:
        os.environ.clear()
        os.environ.update(keep)


# dsnn-dfw.22: the oracle reads its orders off the plan log, so without the log it checked nothing.
def test_the_oracle_without_the_plan_log_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="needs the plan log"):
        _main_until_after_the_check(monkeypatch)


def test_the_oracle_with_the_plan_log_passes_the_refusal(monkeypatch, tmp_path):
    with pytest.raises(_PastTheOracleCheck):
        _main_until_after_the_check(
            monkeypatch, "--plan-log", str(tmp_path / "plans.jsonl"))


# dsnn-dfw.227: job 68195 counted 15 checks that died with the OOM-killed worker as wrong
# gradients and raised "THE EXACT GRADIENT IS WRONG" at max rel_l2 1.2e-14.
class _DeadRef:
    ready = True


class _DyingActor:
    # Answers the first check, then dies: every later ref raises RayActorError.
    def __init__(self, box, answers_before_death=1):
        self.box = box
        self.left = answers_before_death
        self.check = types.SimpleNamespace(remote=self._remote)

    def _remote(self, args_np, order, probe_seed, rule=None):
        self.box["calls"].append(tuple(order))
        if self.left > 0:
            self.left -= 1
            return _Ref(("pass", 1.2e-14), ready=True)
        return _DeadRef()


@pytest.fixture
def dying_ray(monkeypatch):
    box = {"calls": [], "killed": [], "actors": [], "puts": []}

    class RayActorError(Exception):
        pass

    class ActorDiedError(RayActorError):
        pass

    def get(ref):
        if isinstance(ref, _DeadRef):
            raise ActorDiedError("The actor died unexpectedly before finishing this task")
        return ref.value

    def wait(refs, num_returns=1, timeout=None):
        return ([r for r in refs if r.ready], [r for r in refs if not r.ready])

    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(
        wait=wait, get=get, put=lambda v: v,
        kill=lambda actor, no_restart=False: box["killed"].append(actor),
        exceptions=types.SimpleNamespace(RayActorError=RayActorError,
                                         ActorDiedError=ActorDiedError)))
    return box


def _dying_factory(box, answers_before_death=1):
    def make():
        box["actors"].append(_DyingActor(box, answers_before_death
                                         if not box["actors"] else 10 ** 6))
        return box["actors"][-1]
    return make


def test_a_check_that_dies_with_its_actor_drains_as_dead_and_raises_nothing(dying_ray, tmp_path):
    import json
    from alphagrad.approx.ppo import (
        _grad_oracle_boundary, _grad_oracle_exit_summary)

    oracle = G.AsyncGradOracle(
        _in_process_check, timeout_s=60.0, actor_factory=_dying_factory(dying_ray),
        arg_resolver=lambda ep: ep)
    jobs = [{"order": [i, 9, 8], "plan_hashes": [f"h{i}"]} for i in range(4)]
    assert oracle.submit(0, 7, jobs, batch=8) == 4
    path = str(tmp_path / "plan_log.jsonl")
    lines = []
    out = _grad_oracle_boundary(oracle, path, 1, 1e-3, log=lines.append)
    assert [r["status"] for r in out] == ["pass", "dead", "dead", "dead"], out
    assert all("ActorDiedError" in r["error"] for r in out[1:])
    c = oracle.counts()
    assert (c["pass"], c["fail"], c["dead"], c["pending"]) == (1, 0, 3, 0), c
    assert c["actor_deaths"] == 1 and len(dying_ray["actors"]) == 2, (
        "the dead actor is replaced by a fresh one")
    assert any("3 check(s) dead" in l and "missing" in l for l in lines), lines
    rows = [json.loads(l) for l in open(path) if l.strip()]
    assert [r["oracle"]["status"] for r in rows] == ["pass", "dead", "dead", "dead"]
    assert all(r["oracle"]["batch"] == 8 for r in rows)
    assert "1 pass, 0 fail, 0 timeout, 3 dead, 0 error" in _grad_oracle_exit_summary(oracle, 0.0)
    # NOT MEMOIZED: the dead orders run again, on the fresh actor.
    assert oracle.submit(50, 7, jobs[1:], batch=8) == 3
    out = _grad_oracle_boundary(oracle, path, 51, 1e-3, log=lambda _l: None)
    assert [r["status"] for r in out] == ["pass"] * 3
    assert [r["from_memo"] for r in out] == [False] * 3
    assert dying_ray["calls"][-3:] == [(1, 9, 8), (2, 9, 8), (3, 9, 8)]
    oracle.close()


def test_a_measured_rel_l2_above_the_bar_still_raises_in_actor_mode(dying_ray, tmp_path):
    from alphagrad.approx.env import GradientOracleFailure
    from alphagrad.approx.ppo import _grad_oracle_boundary

    class _Wrong:
        def __init__(self):
            self.check = types.SimpleNamespace(
                remote=lambda a, order, seed, rule=None: _Ref(("fail", 4.2e-2), ready=True))

    oracle = G.AsyncGradOracle(_in_process_check, timeout_s=60.0,
                               actor_factory=_Wrong, arg_resolver=lambda ep: ep)
    oracle.submit(0, 7, [{"order": [3, 2, 1], "plan_hashes": ["h"]}])
    with pytest.raises(GradientOracleFailure, match="THE EXACT GRADIENT IS WRONG"):
        _grad_oracle_boundary(oracle, str(tmp_path / "p.jsonl"), 1, 1e-3,
                              log=lambda _l: None)
    assert oracle.counts()["fail"] == 1
    oracle.close()


def test_a_check_that_raises_inside_the_actor_is_an_error_not_a_stop(dying_ray, tmp_path):
    from alphagrad.approx.ppo import _grad_oracle_boundary

    class _Raising:
        def __init__(self):
            self.check = types.SimpleNamespace(remote=lambda *a, **k: _ErrRef())

    class _ErrRef:
        ready = True
        value = None

    real_get = sys.modules["ray"].get

    def get(ref):
        if isinstance(ref, _ErrRef):
            raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating 4182023340032 bytes")
        return real_get(ref)
    sys.modules["ray"].get = get

    oracle = G.AsyncGradOracle(_in_process_check, timeout_s=60.0,
                               actor_factory=_Raising, arg_resolver=lambda ep: ep)
    oracle.submit(0, 7, [{"order": [3, 2, 1], "plan_hashes": ["h"]}])
    out = _grad_oracle_boundary(oracle, str(tmp_path / "p.jsonl"), 1, 1e-3,
                                log=lambda _l: None)
    assert [r["status"] for r in out] == ["error"]
    assert "4182023340032" in out[0]["error"]
    c = oracle.counts()
    assert (c["error"], c["fail"], c["actor_deaths"]) == (1, 0, 0)
    oracle.close()


def test_the_factory_hint_and_the_pin_follow_the_cores_given(monkeypatch):
    seen, options_seen = {}, {}
    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(
        remote=_fake_ray_remote(seen, options_seen)))
    G.make_ray_oracle_actor_factory({"example": "Helmholtz"}, num_cpus=38,
                                    core_ids=tuple(range(22, 60)))()
    assert seen == {"num_cpus": 38, "num_gpus": 0}
