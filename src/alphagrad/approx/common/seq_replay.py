"""Replay a recorded best-sequence as ``(order, transforms)`` for jacve.

The trainers (ppo / mu0 / gfn) write ``best_sequences.json`` with one
entry per channel; each entry has a ``seq`` field of 6-tuples
``[vertex, op_type, i, j, factor, quant_or_kind_idx]``. This module
converts that wire format into the ``(order, transforms)`` arguments
:func:`graphax.jacve` consumes, so a downstream training loop can build
the gradient function matching the policy that produced the recording.

Distinct from :mod:`alphagrad.approx.common.replay`, which is the
trajectory replay buffer for the RL trainers — that one stores
pytrees of past rollouts for sample-efficient learning; this one
re-builds a single recorded micro-action sequence as a usable gradient
callable.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence, Union

from graphax.sparse.micro_actions import (
    COMPRESS_KINDS,
    QUANT_DTYPES,
    Compress,
    Diag,
    Quant,
)

OP_DIAG = 0
OP_COMPRESS = 1
OP_QUANT = 2
OP_END = 3

MicroAction = Union[Diag, Compress, Quant]


_CALL_DIAG = re.compile(r"diag\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+)\s*\)")
_CALL_COMPRESS = re.compile(r"compress\(\s*'([^']+)'\s*,\s*(\d+)\s*\)")
_CALL_QUANT = re.compile(r"quant\(\s*'([^']+)'\s*\)")


def _parse_call_string(s: str) -> dict[str, Any] | None:
    """Parse one dynamic-substeps call string into a typed op dict.

    Grammar (must stay in sync with
    :func:`alphagrad.approx.ppo._action_to_pylist_dynamic`):
    ``diag(i, j, factor)`` / ``compress('kind', axis)`` / ``quant('dtype')``.
    Returns ``None`` for an unrecognised string (e.g. a bare ``op7(...)``
    fallback) so it's skipped rather than mis-decoded.
    """
    m = _CALL_DIAG.fullmatch(s.strip())
    if m:
        return {"op": "Diag", "i": int(m[1]), "j": int(m[2]), "factor": int(m[3])}
    m = _CALL_COMPRESS.fullmatch(s.strip())
    if m:
        return {"op": "Compress", "axes": [int(m[2])], "kind": m[1]}
    m = _CALL_QUANT.fullmatch(s.strip())
    if m:
        return {"op": "Quant", "dtype": m[1]}
    return None


def _row_to_typed_record(row: Any) -> dict[str, Any]:
    """Convert one raw action row (any of the legacy wire shapes) into a
    typed dict ``{"vertex": v, "ops": [{"op": "Diag", "i": ..., ...}, ...]}``.

    Wire shapes accepted (matches :func:`parse_recorded_seq` for decode
    parity, but emits dicts instead of named tuples so the result is
    JSON-native and self-describing — no need for the reader to know
    the ``COMPRESS_SENTINEL = -2`` etc. encoding):

    * **dict already** — typed format; pass through after normalising
      vertex / ops keys. Idempotent.
    * **PPO 6-tuple** ``[v, op_type, i, j, factor, q]`` — the canonical
      action recording from ``--dynamic-substeps`` PPO. ``op_type`` ∈
      ``{0=DIAG, 1=COMPRESS, 2=QUANT, 3=END}``. END rows produce an
      empty ``ops`` list (the vertex still appears so the rendered
      order is preserved).
    * **MuZero / GFN 2-tuple** ``(vertex, [(bi1, bi2, factor), ...])``
      — pre-resolved DIAG-only triples. Vertex is 1-indexed in this
      shape; we record 1-indexed in the output for consistency with
      jacve.
    * **bare int** — legacy non-dynamic-substeps recording (PPO
      ``--variant ve_only``); treated as pure VE at that vertex.
    * **str** — GFN's stringified op like ``"diag(0,1,2)"``; passed
      through as a single ``{"op": "raw", "repr": str}`` entry rather
      than parsed (rarely used; parse path is in :mod:`nn_mnist`).
    """
    if isinstance(row, dict):
        v = int(row.get("vertex", 0))
        ops_in = row.get("ops", []) or []
        ops_out: list[dict[str, Any]] = []
        for op in ops_in:
            if isinstance(op, dict):
                ops_out.append({k: v for k, v in op.items()})
        return {"vertex": v, "ops": ops_out}

    if isinstance(row, int):
        return {"vertex": int(row), "ops": []}

    if isinstance(row, str):
        return {"vertex": 0, "ops": [{"op": "raw", "repr": row}]}

    # Sequence types from here on.
    if len(row) >= 6:
        # PPO 6-tuple [vertex, op_type, i, j, factor, q].
        v, op, i, j, factor, q = (int(x) for x in row[:6])
        if op == OP_END:
            return {"vertex": v, "ops": []}
        if op == OP_DIAG:
            return {"vertex": v, "ops": [{
                "op": "Diag", "i": i, "j": j, "factor": factor,
            }]}
        if op == OP_COMPRESS:
            kind = (
                COMPRESS_KINDS[q] if 0 <= q < len(COMPRESS_KINDS)
                else COMPRESS_KINDS[0]
            )
            return {"vertex": v, "ops": [{
                "op": "Compress", "axes": [i], "kind": kind,
            }]}
        if op == OP_QUANT:
            dtype = (
                QUANT_DTYPES[q] if 0 <= q < len(QUANT_DTYPES)
                else QUANT_DTYPES[0]
            )
            return {"vertex": v, "ops": [{"op": "Quant", "dtype": dtype}]}
        return {"vertex": v, "ops": []}  # unknown op_type

    if len(row) == 2:
        # Two distinct 2-tuple shapes share this branch:
        #   * MuZero / GFN legacy: (vertex, [(bi1, bi2, factor), ...]) —
        #     numeric DIAG triples.
        #   * Dynamic-substeps: (vertex, ["diag(i, j, factor)",
        #     "compress('kind', axis)", "quant('dtype')", ...]) — the
        #     human-readable call strings emitted by
        #     ``ppo._action_to_pylist_dynamic`` (cmorl / gfn / ppo with
        #     --dynamic-substeps). These carry COMPRESS/QUANT, not just DIAG.
        v = int(row[0])
        ops_in = row[1] or []
        ops_out: list[dict[str, Any]] = []
        for op in ops_in:
            if isinstance(op, str):
                parsed = _parse_call_string(op)
                if parsed is not None:
                    ops_out.append(parsed)
            elif len(op) >= 3:
                bi1, bi2, factor = (int(x) for x in op[:3])
                ops_out.append({
                    "op": "Diag", "i": bi1, "j": bi2, "factor": factor,
                })
        return {"vertex": v, "ops": ops_out}

    if len(row) == 1:
        return {"vertex": int(row[0]), "ops": []}

    # Unknown shape: return a vertex=0 record with the raw row preserved
    # for debugging, instead of crashing the snapshot writer.
    return {"vertex": 0, "ops": [{"op": "raw", "repr": str(row)}]}


def to_typed_records(seq: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert any legacy / native action sequence to the typed format.

    Idempotent: if ``seq`` is already a list of typed dicts it round-trips
    unchanged. Use at the json.dump boundary so the on-disk
    ``best_sequences.json`` carries human-readable typed records, never
    raw int rows with sentinel encodings.

    Consecutive rows for the same vertex are coalesced — each PPO 6-tuple
    emits at most one op, but a vertex with multiple op rows (e.g. Diag
    then Compress) is produced as one dict with both in ``ops``.
    """
    if not seq:
        return []
    merged: list[dict[str, Any]] = []
    last_vertex: int | None = None
    for row in seq:
        rec = _row_to_typed_record(row)
        if merged and rec["vertex"] == last_vertex:
            merged[-1]["ops"].extend(rec["ops"])
        else:
            merged.append({"vertex": rec["vertex"], "ops": list(rec["ops"])})
            last_vertex = rec["vertex"]
    return merged


def _resolve_diag_factor(
    factor: int, i: int, j: int, axis_sizes: Sequence[int]
) -> int | None:
    """Resolve the legacy ``factor=-1`` sentinel ("gcd-auto") to an explicit
    integer using ``axis_sizes``. Returns ``None`` when the resolved
    factor is degenerate (0 or 1) or fails the divisibility checks
    ``apply_diag`` enforces.
    """
    if factor in (0, 1):
        return None
    if not (0 <= i < len(axis_sizes)) or not (0 <= j < len(axis_sizes)):
        return None
    n_i, n_j = int(axis_sizes[i]), int(axis_sizes[j])
    if factor == -1:
        factor = math.gcd(n_i, n_j)
    if factor <= 1:
        return None
    if n_i % factor != 0 or n_j % factor != 0:
        return None
    return factor


_LOW_PRECISION_QUANT_DTYPES = frozenset(
    d for d in QUANT_DTYPES if d.startswith(("float4", "float8", "int2", "int4", "uint2", "uint4"))
)


def _vertex_edge_shapes(jaxpr, vid_1indexed: int):
    """Return ``(out_len, primal_shapes)`` for the eqn at 1-indexed vertex
    ``vid_1indexed`` — the SAME shape metadata the env's
    :func:`alphagrad.approx.env._callback` reads to validate a row.

    ``out_len`` is the rank of the eqn's primary output; ``primal_shapes``
    is the list of input shapes (non-literal invars only). Returns
    ``(None, None)`` when the vertex is out of range or carries no usable
    output/input aval (the env skips such vertices for transforms).
    """
    eqns = jaxpr.eqns
    idx = vid_1indexed - 1
    if not (0 <= idx < len(eqns)):
        return None, None
    eqn = eqns[idx]
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        return None, None
    out_shape = eqn.outvars[0].aval.shape
    primal_shapes = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
    if not primal_shapes:
        return None, None
    return len(out_shape), [out_shape, *primal_shapes]


def _parse_typed_records(
    seq: Sequence[dict[str, Any]],
    *,
    axis_sizes: Sequence[int] | None,
    skip_low_precision_quant: bool,
    one_indexed: bool = True,
    jaxpr: Any | None = None,
) -> tuple[list[int], list[tuple[int, tuple[MicroAction, ...]]]]:
    """Decode the typed-dict format into ``(order, transforms)``.

    Records are ``{"vertex": v, "ops": [{"op": "Diag", ...}, ...]}`` whose
    ``vertex`` field is the 0-indexed agent action the trainers write to
    ``best_sequences.json`` (``vertex_action = jrand.categorical(...)`` in
    ``ppo_ray_worker.py``; the env applies ``+1`` only when it feeds the
    action into jacve). ``one_indexed=True`` (default) replays the same
    ``+1`` so the returned ``order`` / vertex keys match the 1-indexed
    convention ``graphax.jacve`` consumes — WITHOUT it the whole order is
    shifted by one, the real last vertex is dropped, and the replayed
    Jacobian is garbage (‖grad‖→0, cosine→nan); see the historical bug in
    the module docstring.

    When ``jaxpr`` is provided, the SAME per-vertex filtering the env's
    ``_callback`` applies is mirrored so the replayed rule == the rule the
    env actually executed:

    * **COMPRESS only on the terminal vertex** of ``order`` (the env
      restricts ``Compress`` to the last vertex because a reduced
      ``val.ndim`` trips graphax's shape-preservation assertion in any
      downstream elimination).
    * **Full-reduction COMPRESS cap** — never drop the last remaining
      physical axis of an edge (would canonicalize it to ``val=None``).
    * **Axis-fit / divisibility** — drop Diag/Compress rows whose axis
      indices don't fit every invar edge, and Diag factors that don't
      divide the relevant axis sizes, exactly as ``apply_diag`` requires.
    * **used_axes dedup** — at most one transform per logical axis.

    Vertices appear in ``order`` in the listed sequence; pure-VE vertices
    (and vertices whose every op was filtered) have an empty ``ops`` list
    and contribute to ``order`` only.
    """
    offset = 1 if one_indexed else 0
    order: list[int] = []
    transforms_by_vertex: dict[int, list[MicroAction]] = {}
    seen: set[int] = set()
    # Per-record vertex ids (already offset) so we can identify the terminal
    # vertex for the env's COMPRESS-on-last-vertex restriction.
    rec_vertices = [int(rec.get("vertex", 0)) + offset for rec in seq]
    # The env keys COMPRESS to the LAST vertex of the (deduplicated) order;
    # mirror that — the terminal vertex is the last distinct one.
    terminal_vertex = None
    for v in rec_vertices:
        terminal_vertex = v  # last wins; distinctness handled by `order` below

    for rec, v in zip(seq, rec_vertices):
        if v not in seen:
            order.append(v)
            seen.add(v)
        out_len = primal_shapes = None
        if jaxpr is not None:
            out_len, primal_shapes = _vertex_edge_shapes(jaxpr, v)
        used_axes: set[int] = set()
        n_compressed = 0
        edge_phys_axes = (
            max((len(ps) for ps in primal_shapes), default=0)
            if primal_shapes is not None else 0
        )
        for op in rec.get("ops", []) or []:
            kind = op.get("op")
            if kind == "Diag":
                i = int(op.get("i", 0))
                j = int(op.get("j", 0))
                if i == j:
                    continue
                factor = int(op.get("factor", 0))
                if jaxpr is not None:
                    # Mirror _callback: i/j are logical edge axes; both must
                    # fit every invar edge, and factor must divide each axis.
                    if i in used_axes or j in used_axes:
                        continue
                    if primal_shapes is None:
                        continue
                    n_i = _logical_axis_size(i, out_len, primal_shapes)
                    n_j = _logical_axis_size(j, out_len, primal_shapes)
                    if n_i is None or n_j is None:
                        continue
                    if factor == -1:
                        factor = math.gcd(n_i, n_j)
                    if factor <= 1 or n_i % factor != 0 or n_j % factor != 0:
                        continue
                    used_axes.add(i)
                    used_axes.add(j)
                    f = factor
                elif axis_sizes is not None:
                    f = _resolve_diag_factor(factor, i, j, axis_sizes)
                    if f is None:
                        continue
                else:
                    if factor in (-1, 0, 1):
                        continue
                    f = factor
                transforms_by_vertex.setdefault(v, []).append(
                    Diag(i=i, j=j, factor=f)
                )
            elif kind == "Compress":
                axes_in = op.get("axes", [])
                if not axes_in:
                    continue
                kind_name = str(op.get("kind", COMPRESS_KINDS[0]))
                if kind_name not in COMPRESS_KINDS:
                    kind_name = COMPRESS_KINDS[0]
                if jaxpr is not None:
                    # Env restriction: COMPRESS only on the terminal vertex.
                    if v != terminal_vertex:
                        continue
                    axis_idx = int(axes_in[0])
                    if axis_idx in used_axes:
                        continue
                    if primal_shapes is None:
                        continue
                    if _logical_axis_size(axis_idx, out_len, primal_shapes) is None:
                        continue
                    # Full-reduction cap: keep >=1 physical axis.
                    if n_compressed + 1 >= edge_phys_axes:
                        continue
                    used_axes.add(axis_idx)
                    n_compressed += 1
                transforms_by_vertex.setdefault(v, []).append(
                    Compress(axes=tuple(int(a) for a in axes_in), kind=kind_name)
                )
            elif kind == "Quant":
                dtype = str(op.get("dtype", QUANT_DTYPES[0]))
                if dtype not in QUANT_DTYPES:
                    dtype = QUANT_DTYPES[0]
                if skip_low_precision_quant and dtype in _LOW_PRECISION_QUANT_DTYPES:
                    continue
                transforms_by_vertex.setdefault(v, []).append(
                    Quant(dtype=dtype)
                )
            # Unknown op kinds are silently dropped — the vertex still
            # appears in ``order`` via the outer ``seen`` bookkeeping.

    transforms = [
        (v, tuple(transforms_by_vertex[v]))
        for v in order
        if v in transforms_by_vertex
    ]
    return order, transforms


def _logical_axis_size(
    axis_idx: int, out_len: int | None, primal_shapes: Sequence[Sequence[int]]
) -> int | None:
    """Resolve a logical edge-axis index to its size, requiring it to fit
    EVERY invar edge (mirrors the env's ``fits_all`` check).

    ``primal_shapes[0]`` is the output shape; ``primal_shapes[1:]`` are the
    input shapes. A logical axis ``< out_len`` is an output-side axis (always
    present); ``>= out_len`` indexes ``axis - out_len`` into each primal.
    Returns the (consistent) size, or ``None`` when it doesn't fit.
    """
    if axis_idx < 0 or out_len is None:
        return None
    out_shape = primal_shapes[0]
    inputs = primal_shapes[1:]
    if axis_idx < out_len:
        if axis_idx >= len(out_shape):
            return None
        return int(out_shape[axis_idx])
    primal_pos = axis_idx - out_len
    sizes = set()
    for ps in inputs:
        if primal_pos >= len(ps):
            return None
        sizes.add(int(ps[primal_pos]))
    if len(sizes) != 1:
        return None
    return sizes.pop()


def parse_recorded_seq(
    seq: Sequence[Any],
    *,
    axis_sizes: Sequence[int] | None = None,
    one_indexed: bool = True,
    skip_low_precision_quant: bool = False,
    jaxpr: Any | None = None,
) -> tuple[list[int], list[tuple[int, tuple[MicroAction, ...]]]]:
    """Convert a recorded best-sequence into ``(order, transforms)``.

    Args:
        seq: list of 6-tuples ``[vertex, op_type, i, j, factor, q]`` as
            written by the trainers. Legacy single-int rows ``[vertex]``
            (non-dynamic-substeps recording) are accepted and treated
            as ``OP_END`` at that vertex (pure VE).
        jaxpr: optional ``jax.core.Jaxpr`` (``make_jaxpr(target_fn).jaxpr``)
            for the model whose gradient is being replayed. When provided,
            the typed-record path mirrors the env's ``_callback`` per-vertex
            filtering exactly (COMPRESS restricted to the terminal vertex,
            axis-fit / factor-divisibility checks, full-reduction cap, axis
            dedup) so the replayed rule == the rule the env executed. When
            ``None``, only the looser ``axis_sizes`` checks apply (legacy).
        axis_sizes: optional flat list of axis sizes used to resolve
            ``factor=-1`` (gcd-auto). When ``None``, gcd-auto Diag rows
            are dropped (the elimination still happens at that vertex,
            but without block-diagonalisation).
        one_indexed: when True (default), the returned ``order`` and the
            vertex keys in ``transforms`` are 1-indexed — the convention
            ``graphax.jacve`` expects. The trainers store 0-indexed
            vertex actions in ``best_sequences.json``
            (``vertex_action = jrand.categorical(...)`` in
            ``ppo_ray_worker.py:800``, which is then ``+1``'d only when
            fed to the env at line 802). Set to False to keep the raw
            0-indexed values for debugging.
        skip_low_precision_quant: when True, ``Quant`` ops targeting
            dtypes below ``float16`` (float4_*, float8_*, int2/4) are
            silently dropped from the transforms list. The vertex still
            appears in ``order`` — only the op is omitted. Workaround
            for JAX's ``TypePromotionError`` on float4 multiplied with
            float32; until the jacve pipeline is dtype-aware,
            low-precision Quants can't replay end-to-end. **Caller-
            visible side effect**: the returned ``transforms`` list
            does NOT reflect the recorded sequence's full op set — log
            the drop count if analysis depends on it.

    Returns:
        ``(order, transforms)`` where ``order`` is the agent-temporal
        sequence of vertex ids (deduplicated, first-occurrence wins)
        and ``transforms`` is the list of ``(vertex_id, tuple_of_ops)``
        entries. Vertices whose only action was ``OP_END`` appear in
        ``order`` but not in ``transforms`` (pure VE, no approximation).

    Failure modes (silent drops the caller may want to log):

    * ``OP_DIAG`` with ``factor in (0, 1)`` or ``factor=-1`` with coprime
      ``axis_sizes`` → dropped (degenerate block size).
    * ``OP_DIAG`` with a factor that doesn't divide both axis sizes →
      dropped (``graphax.apply_diag`` would raise).
    * ``OP_QUANT`` with out-of-range dtype index → silently falls back
      to ``QUANT_DTYPES[0]`` rather than dropping the op.
    * ``OP_COMPRESS`` with out-of-range kind index → falls back to
      ``COMPRESS_KINDS[0]`` ("mean").
    * Truncated / empty rows → treated as ``OP_END`` at vertex 0.
    """
    offset = 1 if one_indexed else 0
    order: list[int] = []
    transforms_by_vertex: dict[int, list[MicroAction]] = {}
    seen: set[int] = set()

    # Detect typed-record format (post-2026-05 writer; see :func:`to_typed_records`).
    # When the first row is a dict, decode via the typed branch. The dict
    # ``vertex`` is the trainers' 0-indexed agent action (NOT pre-+1'd); the
    # typed decoder applies the same ``one_indexed`` offset as the legacy path
    # below, so both wire shapes 1-index identically for jacve.
    if seq and isinstance(seq[0], dict):
        return _parse_typed_records(
            seq, axis_sizes=axis_sizes,
            skip_low_precision_quant=skip_low_precision_quant,
            one_indexed=one_indexed, jaxpr=jaxpr,
        )

    # Dynamic-substeps raw format: rows are (vertex, ["diag(...)",
    # "compress('k', a)", "quant('d')", ...]) with 1-indexed vertices
    # (ppo._action_to_pylist_dynamic). Normalise via to_typed_records — which
    # understands the call-string grammar — then decode. Without this the loop
    # below would hit ``len(row) < 6`` and treat every such row as OP_END,
    # silently stripping ALL approximations. (Production writes these through
    # to_typed_records already, so best_sequences.json arrives as dicts above;
    # this covers a raw seq handed straight to the reader.)
    if seq and any(
        isinstance(r, (list, tuple)) and len(r) == 2
        and isinstance(r[1], (list, tuple))
        and any(isinstance(x, str) for x in r[1])
        for r in seq
    ):
        # NOTE: these call-string rows already carry 1-indexed vertices
        # (``ppo._action_to_pylist_dynamic``), so do NOT re-apply the +1
        # offset here — pass one_indexed=False to keep them as-is. jaxpr
        # filtering is still threaded through for env parity.
        return _parse_typed_records(
            to_typed_records(seq), axis_sizes=axis_sizes,
            skip_low_precision_quant=skip_low_precision_quant,
            one_indexed=False, jaxpr=jaxpr,
        )

    for row in seq:
        # Legacy wire formats accepted:
        # * bare int           — legacy non-dynamic-substeps row, treat as OP_END
        # * len(row) < 6       — truncated / legacy row, treat as OP_END
        # * len(row) >= 6      — full 6-tuple [vertex, op, i, j, factor, q]
        if isinstance(row, int):
            vertex, op, i, j, factor, q = int(row), OP_END, 0, 0, 0, 0
        elif len(row) < 6:
            vertex, op, i, j, factor, q = (
                int(row[0]) if len(row) else 0, OP_END, 0, 0, 0, 0,
            )
        else:
            vertex, op, i, j, factor, q = (int(x) for x in row[:6])

        vertex_out = vertex + offset
        if vertex_out not in seen:
            order.append(vertex_out)
            seen.add(vertex_out)

        if op == OP_END:
            continue
        if op == OP_DIAG:
            if i == j:
                continue
            if axis_sizes is not None:
                f = _resolve_diag_factor(factor, i, j, axis_sizes)
                if f is None:
                    continue
            else:
                if factor in (-1, 0, 1):
                    continue
                f = factor
            transforms_by_vertex.setdefault(vertex_out, []).append(
                Diag(i=i, j=j, factor=f)
            )
        elif op == OP_COMPRESS:
            if not (0 <= q < len(COMPRESS_KINDS)):
                q = 0
            transforms_by_vertex.setdefault(vertex_out, []).append(
                Compress(axes=(i,), kind=COMPRESS_KINDS[q])
            )
        elif op == OP_QUANT:
            if not (0 <= q < len(QUANT_DTYPES)):
                q = 0
            dtype = QUANT_DTYPES[q]
            # JAX's sub-float16 dtypes (float4_e*, float8_e*) and
            # sub-int8 dtypes (int2/4, uint2/4) do not support implicit
            # promotion against float32; an elimination step that
            # downcasts to float4 and then matmuls against a float32
            # input raises TypePromotionError. Callers that wire the
            # downstream gradient through a float32 optimizer should
            # opt into ``skip_low_precision_quant`` to silently drop
            # these rules from the replay (the rest of the sequence —
            # Diag / Compress / higher-precision Quants — still apply).
            if skip_low_precision_quant and dtype in _LOW_PRECISION_QUANT_DTYPES:
                continue
            transforms_by_vertex.setdefault(vertex_out, []).append(
                Quant(dtype=dtype)
            )

    transforms = [
        (v, tuple(transforms_by_vertex[v]))
        for v in order
        if v in transforms_by_vertex
    ]
    return order, transforms


def load_best_sequence(
    run_dir: str | Path, channel: str
) -> Sequence[Any]:
    """Read the recorded 6-tuple sequence for ``channel`` from a wandb
    run directory.

    Args:
        run_dir: path to a wandb ``run-XXX`` dir, its ``files`` subdir,
            or directly to a ``best_sequences.json``.
        channel: one of ``"best_overall"`` or
            ``"best_per_channel/<name>"`` (``<name>`` ∈ ``flops``,
            ``peak_memory``, ``cosine_sim``, ``frob_residual``, …).

    Returns:
        The list of 6-tuples ready for :func:`parse_recorded_seq`.
    """
    p = Path(run_dir)
    if p.is_dir() and (p / "files" / "best_sequences.json").exists():
        p = p / "files" / "best_sequences.json"
    elif p.is_dir() and (p / "best_sequences.json").exists():
        p = p / "best_sequences.json"
    elif not p.is_file():
        raise FileNotFoundError(
            f"best_sequences.json not found at {p}"
        )
    data = json.loads(p.read_text())
    if channel == "best_overall":
        payload = data.get("best_overall")
    elif channel.startswith("best_per_channel/"):
        name = channel.split("/", 1)[1]
        payload = data.get("best_per_channel", {}).get(name)
    else:
        raise ValueError(
            f"channel must be 'best_overall' or 'best_per_channel/<name>', "
            f"got {channel!r}"
        )
    if payload is None:
        raise ValueError(
            f"no entry for channel {channel!r} in {p}"
        )
    seq = payload.get("seq")
    if seq is None:
        raise ValueError(f"entry {channel!r} in {p} has no 'seq' field")
    return seq
