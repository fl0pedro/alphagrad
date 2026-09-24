# dsnn-ct2: a raise under face actions prints the plan's face wires in the [SENTINEL-VERBOSE] block.
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace                               # noqa: E402

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402
from alphagrad.approx.common import plan_log                    # noqa: E402
from alphagrad.approx.cpu_approx_worker import (                # noqa: E402
    CpuApproximationServer,
)

_ORDER = np.asarray([2, 3, 1], np.int32)


@pytest.fixture
def actor(monkeypatch):
    def _refuse(*args, **kwargs):
        raise ValueError("synthetic refusal of the measured plan")

    monkeypatch.setattr(env_mod, "_callback", _refuse)
    a = object.__new__(CpuApproximationServer)
    a._config = SimpleNamespace(delta_obs=True)
    a._args = ()
    a._consts = ()
    a._eval_samples = ()
    a._leak_profile = None
    a._n_calls = 0
    a._n_oom = 0
    a._last_was_oom = False
    a._cache_clear_every = 0
    return a


def _end_rows():
    specs = np.full((len(_ORDER), env_mod.MAX_RULES_PER_VERTEX, 3), -1,
                    np.int32)
    specs[..., 2] = 0
    return specs


def _block(actor, capsys, specs, **wires):
    out = actor.evaluate(_ORDER, specs, len(_ORDER), **wires)
    assert float(np.asarray(out[-1]).min()) <= -1e9
    text = capsys.readouterr().out
    assert "[SENTINEL-VERBOSE] ORDER (3): [2, 3, 1]" in text
    assert "diagnostics dump failed" not in text
    return text


def _row(r):
    return json.dumps(r, separators=(",", ":"))


def test_a_raise_under_face_actions_prints_the_face_wires(actor, capsys):
    bf16 = plan_log.quant_dtype_id("bfloat16")
    faces = np.full((len(_ORDER), 4, env_mod.wire_slots(), 3), -1, np.int32)
    faces[..., 2] = 0
    skips = np.zeros((len(_ORDER), 4), np.int32)
    faces[0, 2, 0] = (0, 1, 2)
    faces[2, 1, 0] = (env_mod.QUANT_SENTINEL, bf16, 0)
    faces[2, 1, 1] = (env_mod.QUANT_SENTINEL, bf16, 0)
    skips[1, 3] = 1
    specs = _end_rows()
    text = _block(actor, capsys, specs, face_specs=faces, face_skips=skips)
    rows = plan_log.encode_wires(
        _ORDER, specs, faces, skips,
        compress_sentinel=env_mod.COMPRESS_SENTINEL,
        quant_sentinel=env_mod.QUANT_SENTINEL)["faces"]
    assert [r[:3] for r in rows] == [[0, 2, 0], [1, 3, 1], [2, 1, 0]]
    assert rows[2][3:9] == [env_mod.QUANT_SENTINEL, "bfloat16", 0] * 2
    assert "    (no active micro-action rules)" in text
    assert "[SENTINEL-VERBOSE] FACE WIRES" in text
    assert f"    v2(vertex_id=1) face 1: {_row(rows[2])}" in text
    for r in rows:
        assert _row(r) in text


def test_a_raise_without_face_wires_says_so(actor, capsys):
    specs = _end_rows()
    specs[1, 0] = (env_mod.QUANT_SENTINEL,
                   plan_log.quant_dtype_id("bfloat16"), 0)
    text = _block(actor, capsys, specs)
    assert "    v1(vertex_id=3): QUANT(dtype=bfloat16)" in text
    assert "[SENTINEL-VERBOSE] FACE WIRES" in text
    assert "    (no live face wires)" in text
