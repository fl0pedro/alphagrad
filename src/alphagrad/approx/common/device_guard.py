from __future__ import annotations

import os
import subprocess
import time


class ActorStartRefused(RuntimeError):
    # A measure actor is not started: its slot is held by a live actor, or its
    # device has a process of ours (owner ruling 2026-09-25, dsnn-dfw.229).
    pass


class DeviceInUse(ActorStartRefused):
    pass


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def _rows(result, what):
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed ({what}): rc {result.returncode}: "
            f"{(result.stderr or '').strip()[:300]}")
    out = []
    for line in (result.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        out.append(parts)
    return out


def compute_apps(index: int, run=_run) -> list[tuple[int, str]]:
    rows = _rows(run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader", "-i", str(int(index))]),
                 f"compute apps of device {index}")
    return [(int(r[0]), r[1]) for r in rows]


def gpu_uuids(run=_run) -> dict[int, str]:
    rows = _rows(run(["nvidia-smi", "--query-gpu=index,uuid",
                      "--format=csv,noheader"]), "gpu uuids")
    return {int(r[0]): r[1] for r in rows}


def _indices(text, what: str) -> list[int]:
    parts = [p.strip() for p in str(text).split(",")]
    if not all(p.isdigit() for p in parts):
        raise ValueError(
            f"{what} {text!r} is not a comma-separated list of GPU indices")
    return [int(p) for p in parts]


def measure_devices(n_actors: int, measure_gpus: str | None, *,
                    trainer_gpus: str, visible: str | None = None,
                    first_gpu: str | None = None) -> list[int]:
    # Slot k of the measure pool runs on the k-th device (dsnn-dfw.245).
    n = int(n_actors)
    if measure_gpus is None:
        first = 1 if first_gpu is None else int(first_gpu)
        return [first + k for k in range(n)]
    if first_gpu is not None:
        raise ValueError(
            f"--measure-gpus {measure_gpus} and ALPHAGRAD_MEASURE_FIRST_GPU="
            f"{first_gpu} both name the measure devices; set one of them")
    devs = _indices(measure_gpus, "--measure-gpus")
    if len(devs) != n:
        raise ValueError(
            f"--measure-gpus {measure_gpus} names {len(devs)} devices for "
            f"{n} measure actors")
    if len(set(devs)) != len(devs):
        raise ValueError(f"--measure-gpus {measure_gpus} names a device twice")
    trainer = _indices(trainer_gpus, "--gpus")
    shared = sorted(set(devs) & set(trainer))
    if shared:
        raise ValueError(
            f"--measure-gpus {measure_gpus} names the trainer's device "
            f"{shared} (--gpus {trainer_gpus})")
    own = None if visible is None else _indices(visible, "CUDA_VISIBLE_DEVICES")
    if trainer[0] != (0 if own is None else own[0]):
        raise ValueError(
            f"--gpus {trainer_gpus} is not the trainer's device: "
            f"CUDA_VISIBLE_DEVICES={visible!r} puts the trainer on GPU "
            f"{0 if own is None else own[0]}")
    outside = [d for d in devs if own is not None and d not in own]
    if outside:
        raise ValueError(
            f"--measure-gpus {measure_gpus} names {outside}, which the "
            f"trainer's CUDA_VISIBLE_DEVICES={visible} does not hold")
    return devs


def owner_uid(pid: int):
    try:
        return os.stat(f"/proc/{int(pid)}").st_uid
    except OSError:
        return None


def our_processes(index: int, *, uid=None, exclude_pids=(), run=_run,
                  owner=owner_uid) -> list[tuple[int, str]]:
    # A pid whose owner cannot be read counts as ours: a guard that cannot
    # tell refuses rather than starting a timed process next to it.
    uid = os.getuid() if uid is None else int(uid)
    skip = {int(p) for p in exclude_pids}
    out = []
    for pid, mem in compute_apps(index, run):
        if pid in skip:
            continue
        who = owner(pid)
        if who is None or int(who) == uid:
            out.append((pid, mem))
    return out


def wait_device_free(index: int, *, slot, uuid=None, exclude_pids=(),
                     timeout_s: float = 30.0, poll_s: float = 0.5, run=_run,
                     owner=owner_uid, uid=None, sleep=time.sleep,
                     clock=time.monotonic) -> None:
    # A killed predecessor releases its context a moment after its death, so
    # the device is polled up to timeout_s before a process of ours on it
    # refuses the start.
    deadline = clock() + float(timeout_s)
    while True:
        busy = our_processes(index, uid=uid, exclude_pids=exclude_pids,
                             run=run, owner=owner)
        if not busy:
            return
        if clock() >= deadline:
            break
        sleep(poll_s)
    pids = ", ".join(f"pid {p} ({m})" for p, m in busy)
    raise DeviceInUse(
        f"device {int(index)}" + (f" ({uuid})" if uuid else "")
        + f" for slot {slot} is in use by {pids} after {float(timeout_s):g}s; "
        f"no actor is started on it (dsnn-dfw.229)")
