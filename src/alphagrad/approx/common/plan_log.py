"""A6 -- THE PLAN LOG: the schema for *every* terminal plan, losers included.

WHY THIS EXISTS. ``ppo._dump_pareto`` persists the FRONT. Nothing persisted
the plans that lost, so X3 ("diff the lowered graphs of the recorded losers
against identity and attribute the regression") had no input at all: the only
archived plans were the ones that won, which is the one sample from which no
regression can be attributed. This module defines the record and the two
halves of the round trip -- ``encode_wires`` (called from the measurement
callback, host-side, on wires that are already materialised) and
``decode_wires`` (called by any replay tool).

THE REPLAYABILITY RULE, AND THE BUG IT IS WRITTEN AGAINST.
``907c231`` fixed 264 archived Pareto points that could not be replayed at
all: the archive recorded a ``seq`` whose vertex column was the 1-BASED JAXPR
VERTEX ID while ``build_order_specs`` documented it as the agent's 0-BASED
ACTION INDEX and resolved it through ``env.valid_vertices``, so every replay
either walked off the end of that array (``IndexError: index 95 is out of
bounds for axis 0 with size 95``) or -- worse -- shifted the whole order by
one and measured a garbage Jacobian while reporting a healthy number.

This record does not repeat that, and not by being more careful about the
convention: it removes the reconstruction step entirely. What is written is
the ``(order, rule_specs, face_specs, face_skips)`` INTEGER WIRE that the
measurement callback handed to ``jacve`` -- verbatim, sparsely encoded -- so a
replay is an array rebuild and not an interpretation. There is no seq to
parse, no ``valid_vertices`` lookup, no convention to detect. The convention
is nevertheless NAMED in the record (``vertex_convention``), because a field
whose meaning lives only in a docstring is how the last one went wrong.

The shape metadata travels with the record too (``shape``: n / max_rules /
max_faces / face_slots), so ``decode_wires`` never reads the live process's
``env.MAX_FACES`` -- which is frozen at env.py's first import and is exactly
the kind of ambient constant that makes an archive un-replayable six months
later.

THE QUANT DTYPE COLUMN IS A NAME (ticket dsnn-3qm.19 D7, finding 56). On
the int32 wire a QUANT row is ``[QUANT_SENTINEL, dtype_idx, scale]`` with
``dtype_idx`` the runtime index into ``graphax.sparse.micro_actions
.QUANT_DTYPES`` -- a catalog enumerated from the runtime, which
``jax_enable_x64`` prepends float64 / int64 / uint64 to, so the same dtype is
index 0 (float32) and 2 (bfloat16) with x64 off and 3 and 5 with it on. A
record that carried the index decoded to Quant(float64) / Quant(uint64)
under the other setting. The record therefore carries the NAME, and
:func:`quant_dtype_id` is the one place the two are converted: on the way
in (``encode_wires``) the index becomes the name, on the way out
(``decode_wires``) the name becomes THIS runtime's index. A schema-1 record
carried a bare integer; ``decode_wires`` reads it as an index into the
x64-OFF catalog, which is the catalog every archived log was written under
(no launcher or campaign ever set jax_enable_x64; job 63579 confirms
x64=False on the cluster).

THE PALIMPSA OPERATOR PAIR TRAVELS WITH THE RECORD (owner ruling
2026-09-15). The trainer stamps ``palimpsa_read_rollout`` and
``palimpsa_read_loss`` on every record it writes, beside ``episode`` and
``plan_index``. They are ``"exact"`` or ``"fast"``. When the two differ, the
rollout sampled under one operator and the loss scored under another, so the
PPO ratio at epoch 0 was not 1 and the policy that produced this plan was
trained with a systematic off-policy bias. A plan log outlives the run it came
from and is read on its own, so nothing else in the file would say so. They
are written by the TRAINER, not by :func:`encode_wires`, which is why they do
not move ``SCHEMA`` -- readers already tolerate added fields.

NON-FINITE FLOATS. The coverage census legitimately contains ``nan`` (an
uncounted leaf) and can contain ``inf``. ``json.dumps`` would emit bare
``NaN``/``Infinity`` literals, which are not JSON and which several readers
refuse. They are written as the STRINGS ``"nan"`` / ``"inf"`` / ``"-inf"``
instead -- distinguishable from a missing value, and round-tripped by
:func:`unjson_float`.
"""

from __future__ import annotations

import json
import math

import numpy as np

# Bump the minor when a field is ADDED (readers must tolerate that), the
# major when one changes meaning or disappears.
#   /1  a QUANT row's dtype column was the runtime INDEX into QUANT_DTYPES.
#   /2  it is the dtype NAME. decode_wires reads both: a bare integer is a
#       /1 index and resolves through the x64-OFF catalog.
SCHEMA = "alphagrad.plan_log/2"

# The vertex column of ``order`` is the 1-based jaxpr vertex id -- the same
# integers ``env.valid_vertices`` holds and the same ones ``_callback`` hands
# to ``jacve``. NOT the agent's 0-based action index (see the module
# docstring / commit 907c231).
VERTEX_CONVENTION = "jaxpr-vertex-id-1-based"

_KIND_OF_SENTINEL = {-2: "compress", -3: "quant"}


def kind_of_slot(b0: int, compress_sentinel: int = -2,
                 quant_sentinel: int = -3) -> str | None:
    """Which approximation a wire row asks for, or None when the row is dead.

    The wire layout is the one env.py documents: column 0 >= 0 is a DIAG
    base index, ``COMPRESS_SENTINEL`` (-2) is a COMPRESS, ``QUANT_SENTINEL``
    (-3) is a QUANT and -1 is an unused row.
    """
    b0 = int(b0)
    if b0 == -1:
        return None
    if b0 >= 0:
        return "diag"
    if b0 == int(compress_sentinel):
        return "compress"
    if b0 == int(quant_sentinel):
        return "quant"
    return "other"


_X64_ONLY_DTYPES = ("float64", "int64", "uint64")


def quant_dtype_catalog(x64: bool | None = None) -> tuple[str, ...]:
    """``QUANT_DTYPES`` as this runtime enumerates it (``x64=None``), or as a
    runtime with the given ``jax_enable_x64`` setting would.

    The flag is the only thing that moves the catalog between runtimes of the
    same install (graphax ``_get_quant_dtypes``): on, it PREPENDS the three
    64-bit dtypes. So the x64-OFF catalog is the live one without them. The
    x64-ON catalog cannot be built from an x64-OFF process, which has no
    64-bit dtypes to name; nothing needs it, since names are what is stored.
    """
    import jax
    from graphax.sparse.micro_actions import QUANT_DTYPES
    live = tuple(str(d) for d in QUANT_DTYPES)
    if x64 is None or bool(x64) == bool(jax.config.jax_enable_x64):
        return live
    if x64:
        raise ValueError(
            "the x64-ON QUANT_DTYPES catalog cannot be built in an x64-OFF "
            "process")
    return tuple(n for n in live if n not in _X64_ONLY_DTYPES)


def quant_dtype_id(value, catalog=None):
    """THE conversion of a QUANT row's dtype column between the wire (the
    runtime index into ``QUANT_DTYPES``) and the record (the dtype name).

    An ``int`` is an index and comes back as the name; a ``str`` is a name
    and comes back as the index. ``catalog`` defaults to this runtime's.
    Anything unresolvable RAISES: index 0 is float32 with x64 off and
    float64 with it on, so a silent fallback to 0 (what env.py's decoder
    does with an out-of-range index) would measure a different plan and
    report it as this one.
    """
    cat = quant_dtype_catalog() if catalog is None else tuple(catalog)
    if isinstance(value, str):
        if value not in cat:
            raise ValueError(
                f"QUANT dtype {value!r} is not in this runtime's "
                f"QUANT_DTYPES {cat}")
        return cat.index(value)
    idx = int(value)
    if not 0 <= idx < len(cat):
        raise ValueError(
            f"QUANT dtype index {idx} is outside QUANT_DTYPES (len {len(cat)})")
    return cat[idx]


def jsonable(x):
    """Non-finite floats as unambiguous strings; everything else untouched."""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, (np.floating,)):
        return jsonable(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def unjson_float(x) -> float:
    """Inverse of :func:`jsonable` for one float slot."""
    if isinstance(x, str):
        return {"nan": float("nan"), "inf": float("inf"),
                "-inf": float("-inf")}[x]
    return float(x)


def encode_wires(order, rule_specs, face_specs, face_skips, *,
                 max_faces_recorded: int = 0,
                 compress_sentinel: int = -2,
                 quant_sentinel: int = -3) -> dict:
    """The replayable half of the record: the wires, sparsely.

    ``order`` is the ``(n,)`` elimination order actually eliminated (the
    TERMINAL prefix, i.e. the whole thing), ``rule_specs`` the
    ``(n, MAX_RULES_PER_VERTEX, 3)`` per-vertex micro-rule rows,
    ``face_specs`` the ``(n, MAX_FACES, FACE_SLOTS, 3)`` per-face rows and
    ``face_skips`` the ``(n, MAX_FACES)`` skip wire.

    Only LIVE entries are written -- a row is live when column 0 != -1, a
    face is live when any of its slots is live or its skip bit is set --
    because the dense buffers are 8.7 MB of mostly -1 at the flagship's
    n=95 / MAX_FACES=2538 and writing them per plan would cost more than
    the whole run's telemetry. The dense buffers are rebuilt exactly by
    :func:`decode_wires` from ``shape``.

    ``max_faces_recorded > 0`` caps how many live faces are written. It is
    OFF by default because a truncated face list is NOT REPLAYABLE, and a
    record that has been truncated says so: ``replayable`` goes False and
    ``faces_truncated`` counts what was dropped. Nothing is ever silently
    lost.
    """
    order = np.asarray(order, dtype=np.int64)
    rs = np.asarray(rule_specs, dtype=np.int64)
    fs = np.asarray(face_specs, dtype=np.int64)
    fk = np.asarray(face_skips, dtype=np.int64)
    n = int(order.shape[0])
    if rs.ndim == 2:                       # (n, 3) never happens, but be safe
        rs = rs.reshape(n, -1, 3)
    max_rules = int(rs.shape[1]) if rs.size else 0
    max_faces = int(fs.shape[1]) if fs.ndim == 4 else 0
    face_slots = int(fs.shape[2]) if fs.ndim == 4 else 0

    req = {"diag": 0, "compress": 0, "quant": 0, "other": 0}
    req_v = {"diag": 0, "compress": 0, "quant": 0, "other": 0}
    req_f = {"diag": 0, "compress": 0, "quant": 0, "other": 0}

    rules: list[list[int | str]] = []
    for k in range(min(n, int(rs.shape[0]) if rs.size else 0)):
        for r in range(max_rules):
            b0 = int(rs[k, r, 0])
            kind = kind_of_slot(b0, compress_sentinel, quant_sentinel)
            if kind is None:
                continue
            req_v[kind] += 1
            req[kind] += 1
            b1 = int(rs[k, r, 1])
            if kind == "quant":
                b1 = quant_dtype_id(b1)
            rules.append([k, r, b0, b1, int(rs[k, r, 2])])

    faces: list[list[int | str]] = []
    n_live = 0
    truncated = 0
    n_skip = 0
    if max_faces:
        for k in range(min(n, int(fs.shape[0]))):
            for f in range(max_faces):
                skip = int(fk[k, f]) if fk.ndim == 2 else 0
                slots = fs[k, f]
                live_slots = [s for s in range(face_slots)
                              if int(slots[s, 0]) != -1]
                if not live_slots and skip != 1:
                    continue
                n_live += 1
                if skip == 1:
                    n_skip += 1
                for s in live_slots:
                    kind = kind_of_slot(int(slots[s, 0]), compress_sentinel,
                                        quant_sentinel)
                    if kind is not None:
                        req_f[kind] += 1
                        req[kind] += 1
                if max_faces_recorded and len(faces) >= int(max_faces_recorded):
                    truncated += 1
                    continue
                row = [k, f, skip]
                for s in range(face_slots):
                    b0, b1, b2 = (int(v) for v in slots[s])
                    if b0 == int(quant_sentinel):
                        b1 = quant_dtype_id(b1)
                    row.extend((b0, b1, b2))
                faces.append(row)

    req["total"] = sum(req[k] for k in ("diag", "compress", "quant", "other"))
    req["skip"] = n_skip
    req_v["total"] = sum(req_v[k]
                         for k in ("diag", "compress", "quant", "other"))
    req_f["total"] = sum(req_f[k]
                         for k in ("diag", "compress", "quant", "other"))
    req_f["skip"] = n_skip

    return {
        "vertex_convention": VERTEX_CONVENTION,
        "order": [int(v) for v in order.tolist()],
        "shape": {"n": n, "max_rules": max_rules, "max_faces": max_faces,
                  "face_slots": face_slots},
        # [k, rule_row, b0, b1, b2]; b1 is the dtype NAME on a QUANT row
        "rules": rules,
        # [k, face, skip, s0b0, s0b1, s0b2, s1b0, ...] -- face_slots triples,
        # again with the dtype name in a QUANT slot's middle column
        "faces": faces,
        "n_live_faces": n_live,
        "faces_truncated": truncated,
        "replayable": truncated == 0,
        "requested": req,
        "requested_vertex": req_v,
        "requested_face": req_f,
    }


def decode_wires(rec: dict, *, quant_sentinel: int = -3):
    """``(order, rule_specs, face_specs, face_skips)`` -- the dense int32
    buffers ``env._callback`` consumes, rebuilt EXACTLY.

    Feed them straight back with ``stop=len(order)``. Raises on a record
    that was truncated rather than handing back a plan that would measure
    as something else -- the failure mode 907c231 is about.

    A QUANT row's dtype column comes back as THIS runtime's index for the
    recorded name. A bare integer there is a schema-1 record: it is read as
    an index into the x64-OFF catalog (see the module docstring), then
    resolved by name like any other. A name this runtime cannot resolve
    raises.
    """
    legacy_catalog: list = []

    def _dtype_col(b0, b1):
        if int(b0) != int(quant_sentinel):
            return int(b1)
        if not isinstance(b1, str):
            if not legacy_catalog:
                legacy_catalog.append(quant_dtype_catalog(x64=False))
            b1 = quant_dtype_id(int(b1), legacy_catalog[0])
        return quant_dtype_id(b1)

    # Only truncation refuses: old refused records say replayable False.
    if int(rec.get("faces_truncated") or 0) > 0:
        raise ValueError(
            f"plan-log record is TRUNCATED and NOT replayable: "
            f"{rec['faces_truncated']} live faces were dropped by "
            f"ALPHAGRAD_PLAN_LOG_MAX_FACES. "
            f"Re-run with the cap off (0) -- replaying the truncated wire "
            f"would measure a DIFFERENT plan and report it as this one.")
    sh = rec["shape"]
    n = int(sh["n"])
    order = np.asarray(rec["order"], dtype=np.int32)
    if order.shape[0] != n:
        raise ValueError(f"order length {order.shape[0]} != shape.n {n}")
    rule_specs = np.full((n, int(sh["max_rules"]), 3), -1, dtype=np.int32)
    rule_specs[:, :, 2] = 0
    for row in rec.get("rules", ()):
        k, r, b0, b1, b2 = row
        rule_specs[int(k), int(r)] = (int(b0), _dtype_col(b0, b1), int(b2))
    mf, fslots = int(sh["max_faces"]), int(sh["face_slots"])
    face_specs = np.full((n, mf, fslots, 3), -1, dtype=np.int32)
    face_skips = np.zeros((n, mf), dtype=np.int32)
    for row in rec.get("faces", ()):
        k, f, skip = int(row[0]), int(row[1]), int(row[2])
        face_skips[k, f] = skip
        for s in range(fslots):
            b0 = row[3 + 3 * s]
            face_specs[k, f, s] = (int(b0), _dtype_col(b0, row[4 + 3 * s]),
                                   int(row[5 + 3 * s]))
    return order, rule_specs, face_specs, face_skips


def stamp_provenance(records, *, device: dict, actor_id: dict,
                     overwrite: bool = True) -> int:
    # The process and the device that timed a plan (owner ruling 2026-09-25,
    # dsnn-dfw.229): device is the CUDA_VISIBLE_DEVICES it saw plus the GPU
    # uuid when known, actor_id its pid and slot.
    n = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if not overwrite and ("device" in rec or "actor_id" in rec):
            continue
        rec["device"] = dict(device)
        rec["actor_id"] = dict(actor_id)
        n += 1
    return n


def append_records(path: str, records) -> int:
    """Append records to the run's JSONL. Returns how many were written.

    One line per record, ``json.dumps`` with ``allow_nan=False`` so a
    non-finite that escaped :func:`jsonable` raises here rather than
    producing a file no strict JSON reader will load.
    """
    n = 0
    with open(path, "a") as fh:
        for rec in records:
            fh.write(json.dumps(jsonable(rec), allow_nan=False,
                                separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n


def read_records(path: str) -> list[dict]:
    """Load a plan log. Blank lines tolerated; a bad line raises with its
    number, because a silently skipped record is a silently missing plan."""
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{i}: {exc}") from exc
    return out
