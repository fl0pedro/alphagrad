# One reproduction per allowed compile option (owner ruling 2026-09-25, dsnn-dfw.212): a recorded
# TLM plan that the live options cannot compile on Blackwell, and the 0/1 tuple that compiles it.
# The plans are rebuilt with landscape_map.rebuild_record, the repro-bundle rebuild. Skips on the CPU.
import json
import os
import subprocess
import sys

import pytest

_GOLDEN = os.path.join(os.path.dirname(__file__), "golden")

# class: (record, example, ALPHAGRAD_NN_BATCH, the tuple that compiles it, the live options' error)
_CASES = {
    "verify": ("compile_fallback_tlm_b1_verify_plan.json", "TransformerLM",
               "-", (1, 0, 0, 0), "Failed to verify Triton module"),
    "shmem": ("compile_fallback_tlm_b1_shmem_plan.json", "TransformerLM",
              "-", (1, 0, 0, 0), "Shared memory size limit exceeded"),
    "tkernel": ("compile_fallback_tlm_b1_tkernel_plan.json", "TransformerLM",
                "-", (1, 0, 0, 0), "Failed to compile Triton kernel"),
    "ptxas": ("compile_fallback_tlm_b4_ptxas_plan.json",
              "VmappedTransformerLM", "4", (1, 1, 0, 0),
              "ptxas exited with non-zero error code 139"),
}

_SCRIPT = r"""
import json, os, sys
os.environ.update(ALPHAGRAD_TLM_SEQ="32", ALPHAGRAD_TLM_DMODEL="128",
                  ALPHAGRAD_TLM_VOCAB="1024", ALPHAGRAD_MAX_FACES="128",
                  ALPHAGRAD_APPROX_ADD="lossless",
                  ALPHAGRAD_REDUCE_AXIS_SPACE="physical",
                  ALPHAGRAD_PER_FACE_MASKS="1", ALPHAGRAD_PER_FACE_REPAIR_AXIS="1",
                  ALPHAGRAD_SKIP_COST_ANALYSIS="1", ALPHAGRAD_SKIP_COUNT_OPS="1")
example, batch, cases = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
if batch != "-":
    os.environ["ALPHAGRAD_NN_BATCH"] = batch
import jax
if jax.default_backend() != "gpu":
    print("RESULT " + json.dumps({"backend": jax.default_backend()}))
    sys.exit(0)
import alphagrad.approx.env as envmod
import alphagrad.approx.tools.landscape_map as lm
a = lm.make_argparser().parse_args(
    ["--example", example, "--dataset", "wikitext2", "--seed", "250197",
     "--exec-on-gpu", "--num-eval-samples", "1"])
env, _e, _c = lm.build_env(a)
live = envmod.measure_compile_live_tuple()
out = {"backend": "gpu", "x_shape": list(env.args[0].shape),
       "live": list(live), "try": [list(t) for t in envmod.MEASURE_COMPILE_TRY_ORDER],
       "cases": {}}
for name, path, want in cases:
    rec = json.load(open(path))
    lowered = lm.rebuild_record(env, rec)
    res = {}
    try:
        lm.compile_record(env, rec, compile_tuple=live)
        res["live"] = "compiled"
    except envmod.MeasureCompileFailure as exc:
        res["live"] = " ".join(str(exc).split())[:300]
    try:
        exe = envmod._compile_measure(lowered, compile_tuple=tuple(want))
        res["temp"] = int(exe.memory_analysis().temp_size_in_bytes)
    except envmod.MeasureCompileFailure as exc:
        res["tuple_err"] = " ".join(str(exc).split())[:300]
    n0 = envmod._MEASURE_COMPILE_FALLBACKS["n"]
    try:
        envmod._compile_measure(lowered)
    except envmod.MeasureCompileFailure as exc:
        res["search_err"] = " ".join(str(exc).split())[:300]
    res["note"] = envmod._LAST_COMPILE_NOTE[0]
    res["fallbacks"] = envmod._MEASURE_COMPILE_FALLBACKS["n"] - n0
    out["cases"][name] = res
print("RESULT " + json.dumps(out))
"""


def _run(names):
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        pytest.skip("the failures are in the GPU compiler; run this module "
                    "on a Blackwell node without JAX_PLATFORMS=cpu")
    example, batch = _CASES[names[0]][1], _CASES[names[0]][2]
    cases = [(n, os.path.join(_GOLDEN, _CASES[n][0]), list(_CASES[n][3]))
             for n in names]
    r = subprocess.run(
        [sys.executable, "-c", _SCRIPT, example, batch, json.dumps(cases)],
        capture_output=True, text=True, timeout=1800, env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout[-2000:] + r.stderr[-2000:]
    out = json.loads(line[-1][len("RESULT "):])
    if out["backend"] != "gpu":
        pytest.skip(f"no GPU backend in the subprocess: {out['backend']}")
    return out


def _check(out, name):
    want, signature = list(_CASES[name][3]), _CASES[name][4]
    res = out["cases"][name]
    assert signature in res["live"], res
    assert "temp" in res, res
    k = out["try"].index(want)
    assert res["note"] == {"used": want,
                           "tried": [out["live"]] + out["try"][:k + 1]}, res
    assert res["fallbacks"] == 1, res


def test_the_softmax_entry_compiles_the_recorded_tlm_b1_plans():
    out = _run(["verify", "shmem", "tkernel"])
    assert out["x_shape"] == [32, 128]
    for name in ("verify", "shmem", "tkernel"):
        _check(out, name)


def test_the_ptxas_entry_compiles_the_recorded_tlm_b4_plan():
    out = _run(["ptxas"])
    assert out["x_shape"] == [4, 32, 128]
    _check(out, "ptxas")
