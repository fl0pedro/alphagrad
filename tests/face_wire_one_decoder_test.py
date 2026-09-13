"""ONE face wire, ONE decoder (the frame the measurement applies in).

Ticket .17 (D1) made the four face-wire decoders -- the measurement
(``env._face_dict_for_vertex``), the live-face stream's prefix replay
(``live_faces.LiveFaceStream._decided``), the AZ plan tokenizer
(``plan_tokens.PlanTokenizer.face_transforms``) and the mask oracle's wire
replay (``masks.LiveVertexMaskOracle._face_ft``) -- emit the same ENTRY FORM
(``face_wire_one_transform_test``). They still decoded the rows differently:
the measurement in each SLOT TENSOR's frame at apply time
(``env.make_slot_frame_hook``), the three head-side decoders in the vertex's
nominal frame (``env.decode_vertex_rule_specs``).

Probe 65266/65270 on the dsnn-3qm.59 stage-2 smoke (TLM, 20 plans): the
stream's prefix carried different COMPRESS axes / DIAG pairs than the measured
graph, its operands drifted, and 7 of 839 rows the decide-time mask cleared
were idempotent no-ops on the real operand. With the stream built by the
measurement's own builder the mask refused all 7, on an operand identical to
the apply-time one.

Pinned here, on the ticket-.56 toy graph: every decoder's slot hook is the
slot-frame hook, and on the live slot tensors the measurement actually hands
the hooks, all four decode a wire row to the SAME rules -- for a QUANT, a
COMPRESS and a DIAG row.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.pop("ALPHAGRAD_PLAN_LOG", None)

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from graphax import SKIP_FACE, IncrementalJaxpr                 # noqa: E402
from graphax.core import _stable_var_index, _unpack_face_slots  # noqa: E402
from graphax.sparse.micro_actions import QUANT_DTYPES           # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.common.masks import LiveVertexMaskOracle  # noqa: E402
from alphagrad.approx.common.plan_tokens import PlanTokenizer   # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_FACES, QUANT_SENTINEL)
from alphagrad.approx.live_faces import LiveFaceStream          # noqa: E402

_B = jnp.asarray(
    (np.arange(16, dtype=np.float64).reshape(4, 4) * 0.1234567 + 0.7654321)
    .astype(np.float32))
_W = jnp.asarray(np.arange(20, dtype=np.float32).reshape(5, 4) / 19.0)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _join_mat(x):
    u = jnp.sin(x)                   # vertex 1
    v = _B @ x                       # vertex 2
    w = u + v                        # vertex 3
    return jnp.sum(_W @ w)           # vertices 4, 5


def _bf16_index() -> int:
    for i, d in enumerate(QUANT_DTYPES):
        if jnp.dtype(d) == jnp.dtype(jnp.bfloat16):
            return i
    raise AssertionError(f"no bfloat16 in QUANT_DTYPES: {QUANT_DTYPES}")


def _toy():
    closed = jax.make_jaxpr(_join_mat)(_X4)
    _stable_var_index(closed.jaxpr)
    return SimpleNamespace(jaxpr=closed.jaxpr, argnums=(0,),
                           consts=list(closed.literals), xs=[_X4])


def _rows(slot: int, row) -> tuple[np.ndarray, np.ndarray]:
    rows = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    rows[..., 2] = 0
    rows[0, slot] = row
    return rows, np.zeros((MAX_FACES,), np.int32)


def _four_dicts(T, v, prefix, rows, skips):
    def fresh():
        ij = IncrementalJaxpr(T.jaxpr, T.argnums, list(T.consts), list(T.xs))
        for p in prefix:
            ij.eliminate(p, ())
        return ij
    ij = fresh()
    cfg = SimpleNamespace(jaxpr=T.jaxpr)
    A = envmod._face_dict_for_vertex(cfg, ij, v, rows, skips)
    stream = LiveFaceStream(T.jaxpr, T.argnums, T.consts, T.xs, vocab=512)
    _keys, B = LiveFaceStream._decided(stream, SimpleNamespace(ij=ij), v,
                                       rows, skips, MAX_FACES)
    C = PlanTokenizer.face_transforms(
        SimpleNamespace(tk=SimpleNamespace(ij=ij), jaxpr=T.jaxpr),
        v, rows, skips) or {}
    D = LiveVertexMaskOracle._face_ft(
        SimpleNamespace(jaxpr=T.jaxpr), ij, v, (rows, skips)) or {}
    return ij, {"env": A, "live_faces": B, "plan_tokens": C, "masks": D}


def _slot_hooks(entry, v):
    lhs, rhs, jres, new, _join = _unpack_face_slots(entry, v)
    return {"lhs": lhs, "rhs": rhs, "new": new}


def _live_slot_tensors(T, v, prefix, entry_from_env):
    """The SparseTensors the measurement hands each slot hook of face 0."""
    seen: dict[str, list] = {"lhs": [], "rhs": [], "new": []}
    key = next(k for k, e in entry_from_env.items() if e is not SKIP_FACE)
    hooks = _slot_hooks(entry_from_env[key], v)
    wrapped = {}
    for name, h in hooks.items():
        if h is None:
            wrapped[name] = None
            continue

        def _rec(st, _h=h, _n=name):
            seen[_n].append(st)
            return _h(st)
        wrapped[name] = _rec
    from alphagrad.approx.env import face_entry_from_slots
    ij = IncrementalJaxpr(T.jaxpr, T.argnums, list(T.consts), list(T.xs))
    for p in prefix:
        ij.eliminate(p, ())
    ij.eliminate(v, (), {key: face_entry_from_slots(
        [wrapped["lhs"], wrapped["rhs"], wrapped["new"]])})
    return seen


_CASES = {
    "quant-new": (2, (QUANT_SENTINEL, None, 1)),          # dtype filled below
    "compress-lhs": (0, (COMPRESS_SENTINEL, 0, 0)),
    "diag-lhs": (0, (0, 0, 4)),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_four_decoders_decode_one_row_to_one_rule_set_on_the_live_slot(case):
    T = _toy()
    slot, row = _CASES[case]
    if row[1] is None:
        row = (row[0], _bf16_index(), row[2])
    rows, skips = _rows(slot, row)
    v, prefix = 2, ()                  # vertex 2's face x -> v -> w is merge-free
    _ij, dicts = _four_dicts(T, v, prefix, rows, skips)
    assert dicts["env"], "the measurement built no entry for the row"
    for name, d in dicts.items():
        assert set(d) == set(dicts["env"]), (name, set(d), set(dicts["env"]))

    # Every decoder's hook is the SLOT-FRAME hook: it decodes at apply time in
    # the frame of the tensor it is handed (``rules_for`` is that decode).
    slot_name = ("lhs", "rhs", "new")[slot]
    for name, d in dicts.items():
        for key, entry in d.items():
            if entry is SKIP_FACE:
                continue
            h = _slot_hooks(entry, v)[slot_name]
            assert h is not None, (name, key, slot_name)
            assert hasattr(h, "rules_for"), (
                f"{name}: the {slot_name} hook is not the slot-frame hook "
                f"(env.make_slot_frame_hook) -- a second decoder is back")

    # On the live slot tensor the measurement actually hands the hook, all four
    # decode the row to the same, non-empty rules.
    seen = _live_slot_tensors(T, v, prefix, dicts["env"])
    assert seen[slot_name], f"the {slot_name} hook was never handed a tensor"
    st = seen[slot_name][0]
    decoded = {}
    for name, d in dicts.items():
        key = next(k for k, e in d.items() if e is not SKIP_FACE)
        h = _slot_hooks(d[key], v)[slot_name]
        decoded[name] = tuple(repr(r) for r in h.rules_for(st))
    assert decoded["env"], f"the measurement decoded the row to nothing on {st}"
    for name, rules in decoded.items():
        assert rules == decoded["env"], (name, rules, decoded["env"])
