# dsnn-dfw.229: the factory makes sure the device of a slot has no process of ours before an
# actor starts on it. If one is there after the grace period, it raises with the device, the
# pid and the slot, and starts nothing.
from __future__ import annotations

import types

import pytest

from alphagrad.approx.common import device_guard as DG


def _runner(apps_by_call, uuids="0, GPU-aaa\n1, GPU-bbb\n"):
    calls = []

    def run(cmd):
        calls.append(cmd)
        if "--query-gpu=index,uuid" in cmd:
            return types.SimpleNamespace(returncode=0, stdout=uuids, stderr="")
        k = min(len(calls) - 1, len(apps_by_call) - 1)
        return types.SimpleNamespace(returncode=0, stdout=apps_by_call[k], stderr="")
    return run, calls


def test_a_process_of_ours_on_the_device_refuses_the_start_with_device_pid_and_slot():
    run, calls = _runner(["4242, 1024 MiB\n"])
    t = [0.0]
    with pytest.raises(DG.DeviceInUse) as exc:
        DG.wait_device_free(1, slot=0, uuid="GPU-bbb", timeout_s=2.0, poll_s=0.5, run=run,
                            owner=lambda pid: 1000, uid=1000, sleep=lambda s: t.__setitem__(0, t[0] + s),
                            clock=lambda: t[0])
    msg = str(exc.value)
    assert "device 1" in msg and "GPU-bbb" in msg and "pid 4242" in msg and "slot 0" in msg
    assert calls and calls[0][:2] == ["nvidia-smi", "--query-compute-apps=pid,used_memory"]
    assert calls[0][-2:] == ["-i", "1"]


def test_a_dying_predecessor_that_leaves_within_the_grace_period_does_not_refuse():
    run, calls = _runner(["4242, 1024 MiB\n", "4242, 1024 MiB\n", ""])
    t = [0.0]
    DG.wait_device_free(1, slot=0, timeout_s=30.0, poll_s=0.5, run=run,
                        owner=lambda pid: 1000, uid=1000,
                        sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert len(calls) == 3


def test_another_users_process_does_not_refuse():
    run, _ = _runner(["7, 10 MiB\n"])
    DG.wait_device_free(0, slot=0, timeout_s=0.0, run=run,
                        owner=lambda pid: 2000 if pid == 7 else 1000, uid=1000)


def test_a_pid_whose_owner_is_unknown_counts_as_ours():
    run, _ = _runner(["4242, 1024 MiB\n"])
    with pytest.raises(DG.DeviceInUse, match="pid 4242"):
        DG.wait_device_free(0, slot=1, timeout_s=0.0, run=run, owner=lambda pid: None, uid=1000)


def test_a_failing_nvidia_smi_raises_instead_of_reporting_a_free_device():
    def run(cmd):
        return types.SimpleNamespace(returncode=9, stdout="", stderr="no devices were found")
    with pytest.raises(RuntimeError, match="nvidia-smi failed"):
        DG.wait_device_free(0, slot=0, timeout_s=0.0, run=run, uid=1000)


def test_gpu_uuids_map_the_index_to_the_uuid():
    run, _ = _runner([""])
    assert DG.gpu_uuids(run=run) == {0: "GPU-aaa", 1: "GPU-bbb"}


def test_a_refusal_is_an_actor_start_refusal():
    assert issubclass(DG.DeviceInUse, DG.ActorStartRefused)


# dsnn-dfw.245: each measure slot takes the device the row names, and half B stays off half A's GPUs.

def test_each_half_of_a_pair_maps_its_slots_to_its_own_devices():
    assert DG.measure_devices(3, "1,2,3", trainer_gpus="0", visible="0,1,2,3") == [1, 2, 3]
    assert DG.measure_devices(3, "5,6,7", trainer_gpus="4", visible="4,5,6,7") == [5, 6, 7]
    assert DG.measure_devices(3, "1,2,3", trainer_gpus="0") == [1, 2, 3]
    assert DG.measure_devices(7, "1,2,3,4,5,6,7", trainer_gpus="0",
                              visible="0,1,2,3,4,5,6,7") == [1, 2, 3, 4, 5, 6, 7]


def test_a_device_count_that_is_not_the_actor_count_raises():
    with pytest.raises(ValueError, match="names 2 devices for 3 measure actors"):
        DG.measure_devices(3, "5,6", trainer_gpus="4", visible="4,5,6,7")
    with pytest.raises(ValueError, match="names 3 devices for 7 measure actors"):
        DG.measure_devices(7, "1,2,3", trainer_gpus="0")


def test_a_measure_device_that_is_the_trainer_s_raises():
    with pytest.raises(ValueError, match=r"the trainer's device \[4\]"):
        DG.measure_devices(3, "4,5,6", trainer_gpus="4", visible="4,5,6,7")
    with pytest.raises(ValueError, match=r"the trainer's device \[0\]"):
        DG.measure_devices(3, "0,1,2", trainer_gpus="0", visible="0,1,2,3")


def test_half_b_on_half_a_s_gpus_raises():
    with pytest.raises(ValueError, match="does not hold"):
        DG.measure_devices(3, "1,2,3", trainer_gpus="4", visible="4,5,6,7")
    with pytest.raises(ValueError, match="is not the trainer's device"):
        DG.measure_devices(3, "5,6,7", trainer_gpus="0", visible="4,5,6,7")
    with pytest.raises(ValueError, match="is not the trainer's device"):
        DG.measure_devices(3, "5,6,7", trainer_gpus="4")


def test_a_repeated_or_malformed_device_raises():
    with pytest.raises(ValueError, match="twice"):
        DG.measure_devices(3, "1,1,2", trainer_gpus="0")
    with pytest.raises(ValueError, match="not a comma-separated list"):
        DG.measure_devices(3, "1,2,x", trainer_gpus="0")
    with pytest.raises(ValueError, match="not a comma-separated list"):
        DG.measure_devices(3, "1,2,3", trainer_gpus="0", visible="GPU-aaa")


def test_without_the_flag_the_slots_keep_the_first_gpu_rule():
    assert DG.measure_devices(3, None, trainer_gpus="0") == [1, 2, 3]
    assert DG.measure_devices(3, None, trainer_gpus="0", first_gpu="0") == [0, 1, 2]
    with pytest.raises(ValueError, match="both name the measure devices"):
        DG.measure_devices(3, "1,2,3", trainer_gpus="0", first_gpu="1")
