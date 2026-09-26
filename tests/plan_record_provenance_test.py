# dsnn-dfw.229 (owner ruling 2026-09-25): every plan record carries the device the measuring
# process saw (CUDA_VISIBLE_DEVICES plus the GPU uuid when known) and the actor that timed it
# (pid and slot), so a plan measured next to a co-resident process can be told apart.
from __future__ import annotations

import inspect
import types

import alphagrad.approx.cpu_approx_actors as actors_mod
from alphagrad.approx.common.plan_log import stamp_provenance


def test_stamp_provenance_puts_both_fields_on_every_dict_record():
    recs = [{"plan_hash": "aa"}, "not a record", {"plan_hash": "bb", "device": {"x": 1}}]
    n = stamp_provenance(recs, device={"cuda_visible_devices": "1", "gpu_uuid": "GPU-x"},
                         actor_id={"pid": 4242, "slot": 0, "actor": 3})
    assert n == 2
    assert recs[0]["device"] == {"cuda_visible_devices": "1", "gpu_uuid": "GPU-x"}
    assert recs[0]["actor_id"] == {"pid": 4242, "slot": 0, "actor": 3}
    assert recs[2]["device"] == {"cuda_visible_devices": "1", "gpu_uuid": "GPU-x"}
    kept = [{"plan_hash": "cc", "device": {"cuda_visible_devices": "3", "gpu_uuid": None},
             "actor_id": {"pid": 1, "slot": 2, "actor": 9}}]
    assert stamp_provenance(kept, device={"cuda_visible_devices": "0", "gpu_uuid": None},
                            actor_id={"pid": 7, "slot": None, "actor": "trainer"},
                            overwrite=False) == 0
    assert kept[0]["actor_id"]["pid"] == 1, "a stamped record keeps its actor"


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


def test_the_measure_actors_drain_stamps_its_records(monkeypatch):
    import alphagrad.approx.env as env

    monkeypatch.setattr(env, "consume_plan_records", lambda flush_episode=True: {
        "records": [{"plan_hash": "aa", "order": [3, 2, 1]}], "dropped": 0, "pid": 1,
        "mem_parity": {"records": [], "measured": 0, "dropped": 0}})
    monkeypatch.setattr(env, "check_mem_parity_complete", lambda parity, who: None)
    cls = _unwrapped(actors_mod.CpuApproximationActor)
    fn = cls.__dict__.get("consume_plan_records") or getattr(cls, "consume_plan_records")
    fake_self = types.SimpleNamespace(_actor_id=3, _slot=2, _gpu_uuid="GPU-abc",
                                      _device="3", _pid=4242)
    out = fn(fake_self)
    rec = out["records"][0]
    assert rec["device"] == {"cuda_visible_devices": "3", "gpu_uuid": "GPU-abc"}
    assert rec["actor_id"] == {"pid": 4242, "slot": 2, "actor": 3}


def test_the_actor_constructor_takes_its_slot_and_uuid():
    cls = _unwrapped(actors_mod.CpuApproximationActor)
    params = inspect.signature(cls.__init__).parameters
    assert "slot" in params and "gpu_uuid" in params


def test_the_trainer_stamps_its_own_records_without_overwriting_the_actors():
    import alphagrad.approx.ppo as ppo

    src = inspect.getsource(ppo.main)
    block = src[src.index("_plog_stamp("):]
    block = block[:block.index("for _plog_j, _plog_r in enumerate(_plog_recs):")]
    assert '"actor": "trainer"' in block and "overwrite=False" in block
