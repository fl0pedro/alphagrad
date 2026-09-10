"""D7 (ticket dsnn-3qm.19, finding 56): the plan log stores the QUANT dtype
NAME, not the runtime index of ``QUANT_DTYPES``.

``QUANT_DTYPES`` is enumerated from the runtime (graphax
``micro_actions._get_quant_dtypes``): ``jax_enable_x64`` prepends float64 /
int64 / uint64, so the same name has index 0 (float32) and 2 (bfloat16) with
x64 off and 3 and 5 with x64 on (job 63579). A record that carried the index
decoded to Quant(float64) / Quant(uint64) under the other setting.

What is pinned here:

1. A QUANT row's dtype column is the NAME in the record, for the vertex wire
   and the face wire alike; every other column is the integer it was.
2. The round trip through real JSON is still bit-exact on the int32 wire.
3. A schema-1 record (bare integer) decodes through the x64-OFF catalog --
   the catalog every archived plan log was written under -- so an archived
   index 0 stays float32 whatever the reader's x64 setting.
4. A name this runtime cannot resolve RAISES. Falling back to index 0 would
   measure a different plan and report it as this one (907c231).
5. ACROSS RUNTIMES (subprocesses with JAX_ENABLE_X64=0 and =1): the names in
   the record are identical, each runtime decodes them to ITS OWN index, and
   a legacy integer record decodes to the same names in both.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx.common import plan_log as plog            # noqa: E402
from graphax.sparse.micro_actions import QUANT_DTYPES           # noqa: E402

QS, CS = -3, -2
_I_F32 = QUANT_DTYPES.index("float32")
_I_BF16 = QUANT_DTYPES.index("bfloat16")


def _wires():
    order = np.arange(1, 4, dtype=np.int32)
    rules = np.full((3, 2, 3), -1, np.int32)
    rules[:, :, 2] = 0
    rules[0, 0] = (QS, _I_F32, 0)
    rules[1, 0] = (QS, _I_BF16, 7)         # a scale encoding rides along
    rules[1, 1] = (CS, 1, 2)
    rules[2, 0] = (2, 1, 3)
    faces = np.full((3, 2, 2, 3), -1, np.int32)
    skips = np.zeros((3, 2), np.int32)
    faces[0, 1, 0] = (QS, _I_BF16, 0)
    faces[2, 0, 1] = (QS, _I_F32, -4)
    faces[2, 1, 0] = (1, 0, 2)
    skips[1, 0] = 1
    return order, rules, faces, skips


def _encode(order, rules, faces, skips):
    rec = plog.encode_wires(order, rules, faces, skips,
                            compress_sentinel=CS, quant_sentinel=QS)
    return json.loads(json.dumps(plog.jsonable(rec), allow_nan=False))


# --------------------------------------------------------------------------
# 1/2. the record carries names; the wire round trip stays exact
# --------------------------------------------------------------------------

def test_quant_rows_carry_the_dtype_name():
    rec = _encode(*_wires())
    by_pos = {(r[0], r[1]): r for r in rec["rules"]}
    assert by_pos[(0, 0)] == [0, 0, QS, "float32", 0]
    assert by_pos[(1, 0)] == [1, 0, QS, "bfloat16", 7]
    assert by_pos[(1, 1)] == [1, 1, CS, 1, 2]           # not a QUANT: ints
    assert by_pos[(2, 0)] == [2, 0, 2, 1, 3]
    faces = {(r[0], r[1]): r for r in rec["faces"]}
    assert faces[(0, 1)][3:6] == [QS, "bfloat16", 0]
    assert faces[(2, 0)][6:9] == [QS, "float32", -4]
    assert faces[(2, 1)][3:6] == [1, 0, 2]
    # The dtype column changed representation, which is a MAJOR bump.
    assert plog.SCHEMA == "alphagrad.plan_log/2"


def test_roundtrip_through_json_is_bit_exact():
    order, rules, faces, skips = _wires()
    o2, r2, f2, k2 = plog.decode_wires(_encode(order, rules, faces, skips))
    assert np.array_equal(o2, order)
    assert np.array_equal(r2, rules)
    assert np.array_equal(f2, faces)
    assert np.array_equal(k2, skips)
    assert r2.dtype == np.int32 and f2.dtype == np.int32


# --------------------------------------------------------------------------
# 3/4. legacy integers and unknown names
# --------------------------------------------------------------------------

def test_legacy_integer_index_decodes_through_the_x64_off_catalog():
    order, rules, faces, skips = _wires()
    rec = _encode(order, rules, faces, skips)
    # Rewrite the record the way schema 1 wrote it: bare x64-OFF indices.
    off = plog.quant_dtype_catalog(x64=False)
    for r in rec["rules"]:
        if r[2] == QS:
            r[3] = off.index(r[3])
    for row in rec["faces"]:
        for s in range(rec["shape"]["face_slots"]):
            if row[3 + 3 * s] == QS:
                row[4 + 3 * s] = off.index(row[4 + 3 * s])
    rec["schema"] = "alphagrad.plan_log/1"
    o2, r2, f2, k2 = plog.decode_wires(rec)
    assert np.array_equal(r2, rules)
    assert np.array_equal(f2, faces)


def test_the_one_conversion_function_goes_both_ways():
    assert plog.quant_dtype_id(_I_F32) == "float32"
    assert plog.quant_dtype_id("float32") == _I_F32
    assert plog.quant_dtype_id(_I_BF16) == "bfloat16"
    assert plog.quant_dtype_id("bfloat16") == _I_BF16
    for i, name in enumerate(QUANT_DTYPES):
        assert plog.quant_dtype_id(plog.quant_dtype_id(i)) == i
        assert plog.quant_dtype_id(plog.quant_dtype_id(name)) == name
    # The x64-off catalog never carries the 64-bit entries, whatever the
    # reader's own setting.
    off = plog.quant_dtype_catalog(x64=False)
    assert not set(off) & {"float64", "int64", "uint64"}
    assert off.index("float32") == 0 and off.index("bfloat16") == 2


def test_an_unknown_dtype_name_is_refused_not_mapped_to_zero():
    with pytest.raises(ValueError, match="not-a-dtype"):
        plog.quant_dtype_id("not-a-dtype")
    with pytest.raises(ValueError):
        plog.quant_dtype_id(len(QUANT_DTYPES))
    order, rules, faces, skips = _wires()
    rec = _encode(order, rules, faces, skips)
    rec["rules"][0][3] = "not-a-dtype"
    with pytest.raises(ValueError, match="not-a-dtype"):
        plog.decode_wires(rec)


# --------------------------------------------------------------------------
# 5. across runtimes: JAX_ENABLE_X64 = 0 and 1 in subprocesses
# --------------------------------------------------------------------------

_CHILD = r"""
import json, sys
import numpy as np
import jax
from graphax.sparse.micro_actions import QUANT_DTYPES
from alphagrad.approx.common import plan_log as plog
QS = -3
given = json.loads(sys.argv[1])
legacy = json.loads(sys.argv[2])
i32, ibf = QUANT_DTYPES.index("float32"), QUANT_DTYPES.index("bfloat16")
order = np.arange(1, 3, dtype=np.int32)
rules = np.full((2, 2, 3), -1, np.int32); rules[:, :, 2] = 0
rules[0, 0] = (QS, i32, 0); rules[1, 0] = (QS, ibf, 5)
faces = np.full((2, 2, 2, 3), -1, np.int32); skips = np.zeros((2, 2), np.int32)
faces[1, 0, 1] = (QS, ibf, 0)
rec = plog.encode_wires(order, rules, faces, skips, quant_sentinel=QS)
rec = json.loads(json.dumps(plog.jsonable(rec), allow_nan=False))
_, rg, fg, _ = plog.decode_wires(given)
_, rl, fl, _ = plog.decode_wires(legacy)
print("RESULT " + json.dumps({
    "x64": bool(jax.config.jax_enable_x64),
    "n": len(QUANT_DTYPES), "i32": i32, "ibf": ibf,
    "rules": rec["rules"], "faces": rec["faces"],
    "given_idx": [int(rg[0, 0, 1]), int(rg[1, 0, 1]), int(fg[1, 0, 1, 1])],
    "given_names": [QUANT_DTYPES[int(rg[0, 0, 1])], QUANT_DTYPES[int(rg[1, 0, 1])],
                    QUANT_DTYPES[int(fg[1, 0, 1, 1])]],
    "legacy_names": [QUANT_DTYPES[int(rl[0, 0, 1])], QUANT_DTYPES[int(rl[1, 0, 1])],
                     QUANT_DTYPES[int(fl[1, 0, 1, 1])]],
}))
"""


def _run_child(x64: int, given: dict, legacy: dict) -> dict:
    import alphagrad
    import graphax
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = str(x64)
    env["JAX_PLATFORMS"] = "cpu"
    roots = [os.path.dirname(os.path.dirname(alphagrad.__file__)),
             os.path.dirname(os.path.dirname(graphax.__file__))]
    env["PYTHONPATH"] = os.pathsep.join(
        roots + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(given), json.dumps(legacy)],
        env=env, capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr[-4000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    return json.loads(line[-1][len("RESULT "):])


def test_names_survive_a_change_of_jax_enable_x64():
    # The record every child decodes: NAMES, as this process writes them.
    order = np.arange(1, 3, dtype=np.int32)
    rules = np.full((2, 2, 3), -1, np.int32)
    rules[:, :, 2] = 0
    rules[0, 0] = (QS, _I_F32, 0)
    rules[1, 0] = (QS, _I_BF16, 5)
    faces = np.full((2, 2, 2, 3), -1, np.int32)
    skips = np.zeros((2, 2), np.int32)
    faces[1, 0, 1] = (QS, _I_BF16, 0)
    given = _encode(order, rules, faces, skips)
    # A schema-1 record of the same plan: x64-OFF indices 0 and 2.
    legacy = json.loads(json.dumps(given))
    legacy["schema"] = "alphagrad.plan_log/1"
    legacy["rules"][0][3] = 0
    legacy["rules"][1][3] = 2
    legacy["faces"][0][7] = 2

    c0 = _run_child(0, given, legacy)
    c1 = _run_child(1, given, legacy)
    assert c0["x64"] is False and c1["x64"] is True
    # The runtime index of the same name really does move...
    assert c1["n"] == c0["n"] + 3
    assert (c1["i32"], c1["ibf"]) == (c0["i32"] + 3, c0["ibf"] + 3)
    # ...and the record does not: both children write the same names.
    assert c0["rules"] == c1["rules"] == given["rules"]
    assert c0["faces"] == c1["faces"] == given["faces"]
    # Each child resolves the names to ITS OWN index.
    assert c0["given_idx"] == [c0["i32"], c0["ibf"], c0["ibf"]]
    assert c1["given_idx"] == [c1["i32"], c1["ibf"], c1["ibf"]]
    assert c0["given_names"] == c1["given_names"] == [
        "float32", "bfloat16", "bfloat16"]
    # A legacy integer record means the same plan under both settings.
    assert c0["legacy_names"] == c1["legacy_names"] == [
        "float32", "bfloat16", "bfloat16"]
