import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                             # noqa: E402
import pytest                                                  # noqa: E402

from alphagrad.approx.common import carry_plan as CP           # noqa: E402
from alphagrad.approx.env import (COMPRESS_SENTINEL,           # noqa: E402
                                  MAX_RULES_PER_VERTEX, QUANT_SENTINEL)
from alphagrad.approx.unified_face_head import QUANT_SLOTS     # noqa: E402

# Vertex 2 of the policy's graph maps to vertex 2, which the measured graph does not eliminate.
VARIANT = {"vertex_map": {1: 1, 2: 2}, "valid": {1}, "alt_carry": ()}


def _exact_wires():
    rules = np.full((2, MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    rules[:, :, 2] = 0
    faces = np.full((2, 2, 3, 3), -1, dtype=np.int32)
    skips = np.zeros((2, 2), dtype=np.int32)
    return rules, faces, skips


def _wires_with(kind):
    rules, faces, skips = _exact_wires()
    if kind == "diag":
        faces[1, 0, 0] = (0, 0, -1)
    elif kind == "skip":
        skips[1, 0] = 1
    elif kind == "quant":
        for s in QUANT_SLOTS:
            faces[1, 0, s] = (QUANT_SENTINEL, 0, 0)
    elif kind == "reduce":
        faces[1, 0, 0] = (COMPRESS_SENTINEL, 0, 0)
    elif kind == "vertex_quant":
        rules[1, 0] = (QUANT_SENTINEL, 0, 0)
    else:
        raise ValueError(kind)
    return rules, faces, skips


def test_an_exact_row_on_a_vertex_the_measured_graph_keeps_is_dropped():
    rules, faces, skips = _exact_wires()
    order, _rs, _fs, _sk, _jn = CP.transport_wires(
        [1, 2], VARIANT, rules, faces, skips)
    assert order == [1]


@pytest.mark.parametrize("kind",
                         ["diag", "skip", "quant", "reduce", "vertex_quant"])
def test_a_decision_on_a_vertex_the_measured_graph_keeps_raises(kind):
    rules, faces, skips = _wires_with(kind)
    with pytest.raises(ValueError,
                       match="is not eliminable on the measured graph"):
        CP.transport_wires([1, 2], VARIANT, rules, faces, skips)
