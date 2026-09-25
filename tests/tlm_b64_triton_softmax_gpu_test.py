import json
import os
import subprocess
import sys

import pytest

_PLAN = os.path.join(os.path.dirname(__file__), "golden",
                     "tlm_b64_triton_softmax_plan.json")

_SCRIPT = r"""
import json, os, sys
os.environ.update(ALPHAGRAD_NN_BATCH="64", ALPHAGRAD_TLM_SEQ="32",
                  ALPHAGRAD_TLM_DMODEL="128", ALPHAGRAD_TLM_VOCAB="1024",
                  ALPHAGRAD_MAX_FACES="128", ALPHAGRAD_APPROX_ADD="lossless",
                  ALPHAGRAD_REDUCE_AXIS_SPACE="physical",
                  ALPHAGRAD_PER_FACE_MASKS="1", ALPHAGRAD_PER_FACE_REPAIR_AXIS="1",
                  ALPHAGRAD_SKIP_COST_ANALYSIS="1", ALPHAGRAD_SKIP_COUNT_OPS="1")
import jax
if jax.default_backend() != "gpu":
    print("RESULT " + json.dumps({"backend": jax.default_backend()}))
    sys.exit(0)
import alphagrad.approx.env as envmod
import alphagrad.approx.tools.landscape_map as lm
from alphagrad.approx.common.plan_log import decode_wires
rec = json.load(open(sys.argv[1]))
a = lm.make_argparser().parse_args(
    ["--example", "VmappedTransformerLM", "--dataset", "wikitext2",
     "--seed", "250197", "--exec-on-gpu"])
env, _e, _c = lm.build_env(a)
cfg = env.config
order, specs, fspecs, fskips = decode_wires(rec)
o = [int(v) for v in order]
ft = envmod._face_transforms_for_order(
    cfg, env.consts, env.args, o, specs.tolist(), fspecs, fskips)
tr, _ = envmod._decode_vertex_transforms(cfg, o, specs.tolist())
fn = envmod.measured_program(cfg, o, env.consts, transforms=tr,
                             face_transforms=ft)
args = jax.device_put(tuple(env.args), jax.devices()[0])
lowered = jax.jit(fn, keep_unused=True).lower(*args)
n0 = envmod._MEASURE_COMPILE_FALLBACKS["n"]
res = {"backend": "gpu", "x_shape": list(env.args[0].shape)}
try:
    ex = envmod._compile_measure(lowered)
    res.update(ok=True, temp=int(ex.memory_analysis().temp_size_in_bytes))
except envmod.MeasureCompileFailure as exc:
    res.update(ok=False, err=str(exc)[:400])
res["fallbacks"] = envmod._MEASURE_COMPILE_FALLBACKS["n"] - n0
print("RESULT " + json.dumps(res))
"""


def test_tlm_b64_plan_with_a_triton_softmax_fusion_compiles():
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        pytest.skip("the defect is in the GPU compiler (job 67947); "
                    "run this module on a Blackwell node without JAX_PLATFORMS=cpu")
    r = subprocess.run([sys.executable, "-c", _SCRIPT, _PLAN],
                       capture_output=True, text=True, timeout=1200,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-3000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(line[-1][len("RESULT "):])
    if res["backend"] != "gpu":
        pytest.skip(f"no GPU backend in the subprocess: {res['backend']}")
    assert res["x_shape"] == [64, 32, 128]
    assert res["ok"], res.get("err")
    assert res["fallbacks"] == 1
