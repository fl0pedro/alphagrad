"""Pin `_compile_measure`: INTERNAL GPU-compiler failures (ptxas exit-139,
Triton fusion) retry once with the degraded-fusion option set; anything else
(OOM, INVALID_ARGUMENT) re-raises untouched so the existing trunc/sentinel
machinery keeps handling it. No GPU needed -- the lowered object is stubbed."""
import pytest

from alphagrad.approx import env as env_mod


class _Lowered:
    def __init__(self, msg):
        self.calls = []
        self.msg = msg

    def compile(self, compiler_options=None):
        self.calls.append(compiler_options)
        if len(self.calls) == 1:
            raise RuntimeError(self.msg)
        return "EXE"


def test_fallback_on_ptxas_segfault():
    lo = _Lowered("INTERNAL: ptxas exited with non-zero error code 139, output:")
    before = env_mod._MEASURE_COMPILE_FALLBACKS["n"]
    assert env_mod._compile_measure(lo) == "EXE"
    assert len(lo.calls) == 2
    # the retry really uses the degraded set, not the standard options
    assert lo.calls[1]["xla_gpu_enable_dynamic_slice_fusion"] is False
    assert lo.calls[1]["xla_gpu_use_runtime_fusion"] is False
    assert env_mod._MEASURE_COMPILE_FALLBACKS["n"] == before + 1


def test_fallback_on_triton_fusion_failure():
    lo = _Lowered("INTERNAL: Failed to compile Triton kernel. Context: [Fusion: f32[32,128,32]]")
    assert env_mod._compile_measure(lo) == "EXE"
    assert len(lo.calls) == 2


def test_oom_reraises_for_trunc_machinery():
    lo = _Lowered("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.00GiB.")
    with pytest.raises(RuntimeError):
        env_mod._compile_measure(lo)
    assert len(lo.calls) == 1          # no second attempt


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_MEASURE_COMPILE_FALLBACK", "0")
    lo = _Lowered("INTERNAL: ptxas exited with non-zero error code 139, output:")
    with pytest.raises(RuntimeError):
        env_mod._compile_measure(lo)
    assert len(lo.calls) == 1


def test_fallback_on_shared_memory_kernel_config():
    """RESOURCE_EXHAUSTED by label, compiler kernel-config by nature."""
    lo = _Lowered("RESOURCE_EXHAUSTED: Shared memory size limit exceeded: "
                  "requested 131072, available: 101376, context: [Fusion")
    assert env_mod._compile_measure(lo) == "EXE"
    assert len(lo.calls) == 2


def test_fallback_on_fusion_cycle():
    lo = _Lowered("FAILED_PRECONDITION: A cycle is detected while visiting "
                  "instruction %fusion.113")
    assert env_mod._compile_measure(lo) == "EXE"
    assert len(lo.calls) == 2
