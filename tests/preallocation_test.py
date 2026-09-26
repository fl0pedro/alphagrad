# dsnn-dfw.288: no process sets XLA_PYTHON_CLIENT_PREALLOCATE, so each one preallocates with JAX's
# default. The trainer sees only --gpus before its JAX backend starts, a measure actor sees only its
# own GPU, and the device guard refuses every process of ours on a measure device.
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
import types

import pytest

import importlib.util

from alphagrad.approx.common import device_guard as DG

SITES = ("alphagrad.approx.ppo", "alphagrad.approx.cpu_approx_worker",
         "alphagrad.approx.ppo_ray_worker")


@pytest.fixture
def fresh_job(monkeypatch):
    monkeypatch.setattr(DG, "_JOB_VISIBLE", [])


def test_no_preallocation_switch_at_the_three_sites():
    setter = re.compile(r"XLA_PYTHON_CLIENT_PREALLOCATE[\"']\s*(\]\s*=|,|:)")
    for name in SITES:
        with open(importlib.util.find_spec(name).origin) as fh:
            src = fh.read()
        assert not setter.search(src), f"{name} sets XLA_PYTHON_CLIENT_PREALLOCATE"


def test_the_trainer_is_narrowed_to_its_own_gpus_and_the_job_s_are_kept(fresh_job):
    env = {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    assert DG.own_gpus_only("4", environ=env, started=lambda: False) == "4"
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert DG.job_visible_devices() == "4,5,6,7"
    assert DG.own_gpus_only("4", environ=env, started=lambda: True) == "4"
    assert DG.job_visible_devices() == "4,5,6,7"


def test_a_backend_that_started_before_the_narrowing_raises(fresh_job):
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    with pytest.raises(RuntimeError, match="before it was narrowed"):
        DG.own_gpus_only("0", environ=env, started=lambda: True)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_a_measure_actor_sees_its_own_gpu_or_none():
    gpu = DG.measure_actor_env(3)
    assert gpu["CUDA_VISIBLE_DEVICES"] == "3"
    assert gpu["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu["XLA_PYTHON_CLIENT_ALLOCATOR"] == "default"
    cpu = DG.measure_actor_env(None)
    assert cpu["CUDA_VISIBLE_DEVICES"] == "" and cpu["JAX_PLATFORMS"] == "cpu"
    for env in (gpu, cpu):
        assert not any(k.startswith("XLA_PYTHON_CLIENT_PREALLOCATE") for k in env)


def _apps(rows):
    def run(cmd):
        return types.SimpleNamespace(returncode=0, stdout=rows, stderr="")
    return run


def test_the_guard_exempts_no_process_of_ours():
    run = _apps(f"{os.getpid()}, 73554 MiB\n")
    with pytest.raises(DG.DeviceInUse, match=f"pid {os.getpid()}"):
        DG.wait_device_free(1, slot=0, timeout_s=0.0, run=run, owner=lambda pid: 1000, uid=1000)
    with pytest.raises(TypeError):
        DG.wait_device_free(1, slot=0, timeout_s=0.0, run=run, exclude_pids=(os.getpid(),),
                            owner=lambda pid: 1000, uid=1000)


def _import_ppo(argv, visible):
    code = textwrap.dedent(f"""
        import os, sys
        sys.argv = {["ppo.py", *argv]!r}
        import jax._src.xla_bridge as xb
        seen = []
        orig = xb.backends
        def hooked(*a, **k):
            if not seen:
                seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
            return orig(*a, **k)
        xb.backends = hooked
        import alphagrad.approx.ppo
        from alphagrad.approx.common import device_guard
        job = getattr(device_guard, "job_visible_devices", lambda: None)()
        print("RESULT", repr((seen[0] if seen else None, os.environ.get("CUDA_VISIBLE_DEVICES"),
                              job)))
    """)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
    env.pop("ALPHAGRAD_POOL_TERMINAL_LOCAL", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [s for s in out.stdout.splitlines() if s.startswith("RESULT ")][-1]
    return ast.literal_eval(line[len("RESULT "):])


def test_the_trainer_s_jax_starts_on_its_own_gpus_when_actors_measure():
    first, after, job = _import_ppo(["--exec-on-gpu", "--ray-measure", "3", "--gpus", "0"],
                                    "0,1,2,3")
    assert (first, after, job) == ("0", "0", "0,1,2,3")


def test_the_trainer_keeps_the_job_s_gpus_when_it_measures_itself():
    first, after, job = _import_ppo(["--exec-on-gpu", "--gpus", "0"], "0,1,2,3")
    assert (first, after, job) == ("0,1,2,3", "0,1,2,3", "0,1,2,3")


def test_the_trainer_s_jax_must_see_exactly_its_gpus_when_actors_measure(monkeypatch):
    import jax
    from alphagrad.approx import ppo
    devs = [types.SimpleNamespace(platform="gpu", id=i) for i in range(4)]
    seen = {"n": 1}
    monkeypatch.setattr(jax, "devices", lambda *a, **k: devs[:seen["n"]])
    args = types.SimpleNamespace(exec_on_gpu=True, gpus="0")
    assert ppo._resolve_main_device(args, own_only=True) is devs[0]
    with pytest.raises(RuntimeError, match="only 1 GPU"):
        ppo._resolve_main_device(args)
    seen["n"] = 4
    with pytest.raises(RuntimeError, match="sees 4 GPUs"):
        ppo._resolve_main_device(args, own_only=True)
    assert ppo._resolve_main_device(args) is devs[0]
