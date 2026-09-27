"""Pin `_compile_measure`: INTERNAL GPU-compiler failures (ptxas exit-139,
Triton fusion) go on to the allowed 0/1 compile tuples in their fixed order
(owner ruling 2026-09-25); anything else (OOM, INVALID_ARGUMENT) re-raises
untouched so the existing trunc/sentinel machinery keeps handling it. No GPU
needed -- the lowered object is stubbed."""
import pytest

from alphagrad.approx import env as env_mod

_TRY = env_mod.MEASURE_COMPILE_TRY_ORDER
_LIVE = env_mod.measure_compile_live_tuple()


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
    # the retry is the first allowed tuple on top of the live options
    assert lo.calls[1] == env_mod.measure_compile_options(_TRY[0])
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


# The error text of job 67947: VmappedTransformerLM at B=64 under the Markowitz order.
_SOFTMAX_TRITON = (
    "INTERNAL: Failed to compile Triton kernel. Context: [Fusion: fusion.260 = "
    "f32[64,32,128]{2,1,0} fusion(a_7_.1, constant_285_0, get-tuple-element.2.0, "
    "constant_471_0, fusion.499, input_reduce_fusion.39, fusion.517), kind=kCustom, "
    "calls=fused_computation.238, backend_config={\"operation_queue_id\":\"0\","
    "\"fusion_backend_config\":{\"kind\":\"__triton\",\"block_level_fusion_config\":"
    "{\"num_warps\":\"8\"}}}]")


class _SoftmaxLowered:
    def __init__(self):
        self.calls = []

    def compile(self, compiler_options=None):
        self.calls.append(compiler_options)
        off = str((compiler_options or {}).get("xla_disable_hlo_passes", ""))
        if "triton-softmax-rewriter" not in off.split(","):
            raise RuntimeError(_SOFTMAX_TRITON)
        return "EXE"


def test_fallback_turns_off_the_triton_softmax_rewriter():
    lo = _SoftmaxLowered()
    assert env_mod._compile_measure(lo) == "EXE"
    assert len(lo.calls) == 2
    assert "xla_disable_hlo_passes" not in (lo.calls[0] or {})
    assert lo.calls[1]["xla_disable_hlo_passes"] == "triton-softmax-rewriter"
    assert "xla_gpu_disable_gpuasm_optimizations" not in lo.calls[1]
    assert env_mod._LAST_COMPILE_NOTE[0] == {
        "used": list(_TRY[0]), "tried": [list(_LIVE), list(_TRY[0])]}


# ---------------------------------------------------------------------------
# THE ALLOWED LIST (owner ruling 2026-09-25): one constant, one fixed try
# order, and each tuple's options on their own -- no live options merged in
# (owner ruling 2026-09-26, Q5 b).
# ---------------------------------------------------------------------------
def test_the_allowed_list_and_its_try_order():
    assert env_mod.measure_compile_layout() == [
        "xla_disable_hlo_passes=triton-softmax-rewriter",
        "xla_gpu_disable_gpuasm_optimizations=True"]
    assert _LIVE == (0, 0)
    assert _TRY == ((1, 0), (1, 1))
    assert env_mod.measure_compile_options(_LIVE) is None
    for bad in ((1,), (1, 0, 0), (2, 0)):
        with pytest.raises(ValueError, match="allowed list"):
            env_mod.measure_compile_options(bad)


def test_a_tuple_sets_only_its_own_entries_no_live_options():
    """Owner ruling 2026-09-26, Q5 b: no live options, so an allowed tuple's
    entries are the whole compiler_options dict, not additions on top of a
    base one."""
    assert env_mod.measure_compile_options((0, 0)) is None
    assert env_mod.measure_compile_options((1, 0)) == {
        "xla_disable_hlo_passes": "triton-softmax-rewriter"}
    assert env_mod.measure_compile_options((1, 1)) == {
        "xla_disable_hlo_passes": "triton-softmax-rewriter",
        "xla_gpu_disable_gpuasm_optimizations": True}


class _OnlyWith:
    # Compiles only with these exact options; every other compile fails as ptxas did.
    def __init__(self, ok):
        self.ok = ok
        self.calls = []

    def compile(self, compiler_options=None):
        self.calls.append(compiler_options)
        if compiler_options != self.ok:
            raise RuntimeError(
                "INTERNAL: ptxas exited with non-zero error code 139, output:")
        return "EXE"


def test_the_live_compile_is_unchanged_when_it_compiles():
    lo = _OkLowered()
    assert env_mod._compile_measure(lo) == "EXE"
    assert lo.calls == [None]                 # no live options
    assert env_mod._LAST_COMPILE_NOTE[0] == {
        "used": list(_LIVE), "tried": [list(_LIVE)]}


def test_the_try_order_is_walked_until_a_tuple_compiles():
    lo = _OnlyWith(env_mod.measure_compile_options(_TRY[-1]))
    assert env_mod._compile_measure(lo) == "EXE"
    assert lo.calls == [None] + [
        env_mod.measure_compile_options(t) for t in _TRY]
    assert env_mod._LAST_COMPILE_NOTE[0] == {
        "used": list(_TRY[-1]),
        "tried": [list(_LIVE)] + [list(t) for t in _TRY]}


def test_a_plan_no_tuple_compiles_names_every_tuple_it_tried():
    before = env_mod._MEASURE_COMPILE_FALLBACKS["n"]
    lo = _OnlyWith({"never": True})
    with pytest.raises(env_mod.MeasureCompileFailure) as ei:
        env_mod._compile_measure(lo)
    assert len(lo.calls) == 1 + len(_TRY)
    assert ei.value.tried == [list(_LIVE)] + [list(t) for t in _TRY]
    assert env_mod._LAST_COMPILE_NOTE[0]["used"] is None
    assert env_mod._MEASURE_COMPILE_FALLBACKS["n"] == before + 1


def test_a_replay_compiles_exactly_its_tuple_and_nothing_else():
    want = _TRY[-1]
    lo = _OnlyWith(env_mod.measure_compile_options(want))
    assert env_mod._compile_measure(lo, compile_tuple=want) == "EXE"
    assert lo.calls == [env_mod.measure_compile_options(want)]
    assert env_mod._LAST_COMPILE_NOTE[0] == {
        "used": list(want), "tried": [list(want)]}
    lo = _OnlyWith({"never": True})
    with pytest.raises(env_mod.MeasureCompileFailure) as ei:
        env_mod._compile_measure(lo, compile_tuple=want)
    assert len(lo.calls) == 1
    assert ei.value.tried == [list(want)]


# ---------------------------------------------------------------------------
# MEASURE TOOLCHAIN GATE (finding 03, ticket dsnn-3qm.21).
#
# A link-toolchain fault (nvlink refusing a newer ptxas cubin) is an
# ENVIRONMENT fault that recurs on every plan; it must never be absorbed into
# the per-plan degraded-fusion retry, and the gate that probes for it must
# (a) stop the run under the default, (b) only warn-and-tag under warn,
# (c) be cache-proof by construction.
# ---------------------------------------------------------------------------
import json
import os
import socket
import subprocess
import sys

_NVLINK = ("INTERNAL: nvlink exited with non-zero error code 256, output: "
           "nvlink fatal   : Input file '/tmp/tempfile-x.cubin' newer than "
           "toolkit (129 vs 128)")


class _OkLowered:
    def __init__(self):
        self.calls = []

    def compile(self, compiler_options=None):
        self.calls.append(compiler_options)
        return "EXE"


@pytest.fixture
def fresh_gate(monkeypatch):
    """Un-latch the per-process gate, default mode, probe passes."""
    monkeypatch.delenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", raising=False)
    monkeypatch.setattr(env_mod, "_MEASURE_TOOLCHAIN", dict(
        env_mod._MEASURE_TOOLCHAIN, checked=False, ok=True, detail="",
        mode="", link_faults=0))
    monkeypatch.setattr(env_mod, "_toolchain_probe_compile", lambda: None)
    return env_mod._MEASURE_TOOLCHAIN


def test_nvlink_is_not_fallbackable_under_abort(fresh_gate):
    lo = _Lowered(_NVLINK)
    before = env_mod._MEASURE_COMPILE_FALLBACKS["n"]
    with pytest.raises(env_mod.MeasureToolchainFault) as ei:
        env_mod._compile_measure(lo)
    assert len(lo.calls) == 1                       # no degraded retry
    assert env_mod._MEASURE_COMPILE_FALLBACKS["n"] == before
    msg = str(ei.value)
    assert socket.gethostname() in msg              # names the node
    assert "ptxas:" in msg and "nvlink:" in msg     # names the versions
    assert "129 vs 128" in msg                      # names the fault
    assert fresh_gate["ok"] is False
    assert fresh_gate["link_faults"] == 1


def test_nvlink_goes_on_to_the_allowed_tuples_and_tags_under_warn(
        fresh_gate, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", "warn")
    lo = _Lowered(_NVLINK)
    before = env_mod._MEASURE_COMPILE_FALLBACKS["n"]
    assert env_mod._compile_measure(lo) == "EXE"
    assert len(lo.calls) == 2
    assert env_mod._MEASURE_COMPILE_FALLBACKS["n"] == before + 1
    assert fresh_gate["ok"] is False                # the plan gets tagged
    # ...and the drain carries the tag in THIS process.
    out = env_mod.consume_plan_records()
    assert out["toolchain_ok"] is False
    assert out["compile_fallbacks"] >= 1


def test_nvlink_with_fallback_kill_switch_still_names_the_fault(fresh_gate,
                                                                 monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", "warn")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_COMPILE_FALLBACK", "0")
    lo = _Lowered(_NVLINK)
    with pytest.raises(env_mod.MeasureToolchainFault):
        env_mod._compile_measure(lo)
    assert len(lo.calls) == 1


def test_gate_aborts_on_a_probe_failure(fresh_gate, monkeypatch):
    def _boom():
        raise RuntimeError(_NVLINK)
    monkeypatch.setattr(env_mod, "_toolchain_probe_compile", _boom)
    lo = _OkLowered()
    with pytest.raises(env_mod.MeasureToolchainFault) as ei:
        env_mod._compile_measure(lo)
    assert lo.calls == []                           # nothing was measured
    msg = str(ei.value)
    assert socket.gethostname() in msg
    assert "ptxas:" in msg
    assert "129 vs 128" in msg
    assert "gate probe" in msg
    assert fresh_gate["ok"] is False


def test_gate_warns_and_runs_on_a_probe_failure(fresh_gate, monkeypatch,
                                                capsys):
    monkeypatch.setenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", "warn")

    def _boom():
        raise RuntimeError(_NVLINK)
    monkeypatch.setattr(env_mod, "_toolchain_probe_compile", _boom)
    lo = _OkLowered()
    assert env_mod._compile_measure(lo) == "EXE"
    assert "TOOLCHAIN FAULT" in capsys.readouterr().err
    assert fresh_gate["ok"] is False


def test_gate_off_skips_the_probe_only(fresh_gate, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", "off")
    n = [0]

    def _count():
        n[0] += 1
    monkeypatch.setattr(env_mod, "_toolchain_probe_compile", _count)
    assert env_mod._compile_measure(_OkLowered()) == "EXE"
    assert n[0] == 0
    assert fresh_gate["ok"] is True
    # off skips the PROBE; a link fault in a real measure compile is still
    # not fallbackable.
    with pytest.raises(env_mod.MeasureToolchainFault):
        env_mod._compile_measure(_Lowered(_NVLINK))


def test_gate_runs_once_per_process(fresh_gate, monkeypatch):
    n = [0]

    def _count():
        n[0] += 1
    monkeypatch.setattr(env_mod, "_toolchain_probe_compile", _count)
    env_mod._compile_measure(_OkLowered())
    env_mod._compile_measure(_OkLowered())
    assert n[0] == 1
    assert fresh_gate["checked"] is True


def test_unknown_gate_mode_is_refused(fresh_gate, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_MEASURE_TOOLCHAIN_GATE", "maybe")
    with pytest.raises(ValueError, match="measure-toolchain-gate"):
        env_mod._compile_measure(_OkLowered())


_CACHE_SCRIPT = r"""
import json, os, sys, tempfile
d = tempfile.mkdtemp(prefix="t21_cache_")
os.environ.update(JAX_PLATFORMS="cpu", JAX_COMPILATION_CACHE_DIR=d,
                  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
                  JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
                  ALPHAGRAD_SKIP_COST_ANALYSIS="1",
                  ALPHAGRAD_SKIP_COUNT_OPS="1")
from alphagrad.approx import env
# Importing env runs a few eager module-level jnp ops, and those ARE cached
# (convert_element_type entries). Snapshot after the import: the claim is
# that the PROBE adds nothing.
before = set(os.listdir(d))
a = env._toolchain_probe_compile().as_text()
b = env._toolchain_probe_compile().as_text()
added_by_probe = sorted(set(os.listdir(d)) - before)
import jax, jax.numpy as jnp
jax.jit(lambda x: jnp.cos(x) * 2).lower(
    jax.ShapeDtypeStruct((8, 8), jnp.float32)).compile()
added_by_control = sorted(set(os.listdir(d)) - before)
print("RESULT " + json.dumps({"differ": a != b,
                               "added_by_probe": added_by_probe,
                               "added_by_control": added_by_control}))
"""


def test_probe_bypasses_the_persistent_compile_cache():
    """Finding 03 sec 5a: a cache entry written on a clean node is reused on
    a broken one and the fault never fires, so the probe must be unable to
    hit the cache and must not populate it. A fresh process, because the
    cache is initialised once per process from the environment."""
    r = subprocess.run([sys.executable, "-c", _CACHE_SCRIPT],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-2000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout[-2000:]
    res = json.loads(line[-1][len("RESULT "):])
    # unique key each call: the two modules are not even the same module
    assert res["differ"] is True
    # no write: two probes added NOTHING to the persistent cache...
    assert res["added_by_probe"] == [], res["added_by_probe"]
    # ...while an ordinary compile in the same process DID write, so the
    # cache was live and the silence above is the probe's doing.
    assert len(res["added_by_control"]) == 1, res["added_by_control"]
