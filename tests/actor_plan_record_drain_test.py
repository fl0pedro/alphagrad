"""The measure actor's plan-record drain must run (finding 58 defect).

Ticket .49 added a memory-parity check to ``CpuApproximationActor.
consume_plan_records`` that formats ``os.getpid()`` without importing
``os``. Every ``--ray-measure`` run then polled no records, wrote an empty
plan log and exited 0. This test calls the drain in-process, on the unwrapped
actor class, so the failure is a red test and not a silent empty log.
"""
from __future__ import annotations

import types

import alphagrad.approx.cpu_approx_actors as actors_mod


def _unwrapped(cls):
    # ray.remote wraps the class; the original is kept on the wrapper.
    for attr in ("__ray_actor_class__", "__ray_metadata__"):
        inner = getattr(cls, attr, None)
        if inner is None:
            continue
        if attr == "__ray_metadata__":
            inner = getattr(inner, "modified_class", None) or getattr(inner, "cls", None)
        if inner is not None:
            return inner
    return cls


def test_module_imports_os():
    assert hasattr(actors_mod, "os"), "cpu_approx_actors must import os (finding 58)"


def test_consume_plan_records_runs_in_process(monkeypatch):
    import alphagrad.approx.env as env

    monkeypatch.setattr(env, "consume_plan_records",
                        lambda: {"records": [], "dropped": 0, "pid": 1,
                                 "mem_parity": {"records": [], "measured": 0, "dropped": 0}})
    monkeypatch.setattr(env, "check_mem_parity_complete", lambda parity, who: None)
    cls = _unwrapped(actors_mod.CpuApproximationActor)
    fn = cls.__dict__.get("consume_plan_records") or getattr(cls, "consume_plan_records")
    fake_self = types.SimpleNamespace(_actor_id="test-actor")
    out = fn(fake_self)
    assert out["actor_id"] == "test-actor"
    assert out["records"] == []
