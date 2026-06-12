"""Human-readable rendering of recorded best-sequence action rows.

Each row in ``best_sequences.json`` is a 6-tuple
``[vertex, op_type, i, j, factor, quant_or_kind]`` produced by
:mod:`alphagrad.approx.ppo_ray_worker` (and equivalents) when
``--dynamic-substeps`` is on. The legacy single-int format
``[vertex]`` is also supported (each row stands alone, no approx).

The renderer groups consecutive rows by ``vertex`` and emits a list of
``(vertex_eliminated, [f_1, f_2, ..., f_n])`` tuples, where the
``f_i`` are stringified ``Diag(...)`` / ``Compress(...)`` / ``Quant(...)``
calls applied at that vertex elimination step in agent-temporal order.
``OP_END`` rows commit the vertex with no extra op and contribute the
empty list when no prior ops were queued for that vertex.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from graphax.sparse.micro_actions import COMPRESS_KINDS, QUANT_DTYPES

OP_DIAG = 0
OP_COMPRESS = 1
OP_QUANT = 2
OP_END = 3


def _safe_lookup(table: Sequence[str], idx: int) -> str:
    if 0 <= idx < len(table):
        return table[idx]
    return f"<idx={idx}>"


def render_action_row(row: Sequence[int]) -> tuple[int, str | None]:
    """Decode one 6-tuple ``[vertex, op_type, i, j, factor, q]`` into
    ``(vertex, op_repr_or_None)``. ``None`` is returned for ``OP_END`` rows
    (pure vertex elimination with no approximation at this sub-step).

    Legacy single-int rows ``[vertex]`` (non-dynamic-substeps mode) return
    ``(vertex, None)`` — equivalent to an OP_END.
    """
    if isinstance(row, int):
        return int(row), None
    if len(row) == 0:
        return 0, None
    if len(row) < 6:
        return int(row[0]), None
    vertex, op, i, j, factor, q = (int(x) for x in row[:6])
    if op == OP_END:
        return vertex, None
    if op == OP_DIAG:
        return vertex, f"diag(i={i}, j={j}, factor={factor})"
    if op == OP_COMPRESS:
        kind = _safe_lookup(COMPRESS_KINDS, q)
        return vertex, f'compress(axis={i}, kind="{kind}")'
    if op == OP_QUANT:
        dtype = _safe_lookup(QUANT_DTYPES, q)
        return vertex, f'quant("{dtype}")'
    return vertex, f"<unknown op={op}>"


def render_sequence(seq: Iterable[Sequence[int]]) -> list[tuple[int, list[str]]]:
    """Group decoded rows by elimination event.

    Consecutive rows targeting the same vertex are merged: their approximation
    ops are accumulated in agent-temporal order; the trailing ``OP_END`` (or
    end-of-list) commits the vertex. Returned tuples follow the user-specified
    format ``(vertex, [f_1, ..., f_n])``.
    """
    out: list[tuple[int, list[str]]] = []
    pending_vertex: int | None = None
    pending_ops: list[str] = []
    for row in seq:
        vertex, op_repr = render_action_row(row)
        if pending_vertex is None:
            pending_vertex = vertex
        elif vertex != pending_vertex:
            out.append((pending_vertex, pending_ops))
            pending_vertex, pending_ops = vertex, []
        if op_repr is None:
            out.append((pending_vertex, pending_ops))
            pending_vertex, pending_ops = None, []
        else:
            pending_ops.append(op_repr)
    if pending_vertex is not None:
        out.append((pending_vertex, pending_ops))
    return out


def render_sequence_str(seq: Iterable[Sequence[int]]) -> str:
    """Compact single-line repr of the rendered sequence."""
    return repr(render_sequence(seq))


def render_best_sequences_json(path: str | Path) -> dict[str, str]:
    """Read a ``best_sequences.json`` produced by the trainers and return a
    dict mapping each channel (``best_overall``, ``best_per_channel/<name>``)
    to its rendered string.

    Missing or malformed entries are silently skipped — best_sequences.json
    can be partial mid-run.
    """
    data = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    overall = data.get("best_overall")
    if isinstance(overall, dict) and isinstance(overall.get("seq"), list):
        out["best_overall"] = render_sequence_str(overall["seq"])
    per_channel = data.get("best_per_channel")
    if isinstance(per_channel, dict):
        for name, payload in per_channel.items():
            if isinstance(payload, dict) and isinstance(payload.get("seq"), list):
                out[f"best_per_channel/{name}"] = render_sequence_str(payload["seq"])
    return out
