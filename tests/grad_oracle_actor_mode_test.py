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


def test_the_oracle_actor_is_one_process_with_no_gpu(monkeypatch):
    seen = {}

    def remote(**kw):
        seen.update(kw)
        return lambda cls: types.SimpleNamespace(
            remote=lambda *a: ("handle", cls.__name__, a))

    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(remote=remote))
    handle = G.make_ray_oracle_actor_factory({"example": "Helmholtz"})()
    assert seen == {"num_cpus": G.ORACLE_ACTOR_NUM_CPUS, "num_gpus": 0}
    assert handle[0] == "handle" and handle[2] == ({"example": "Helmholtz"},)


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
