"""AUTOREGRESSIVE per-face loop: emit -> re-encode -> decide -> emit -> ...

The trainer's default path calls ``Agent.encode()`` ONCE per env step and then
runs the whole micro sub-episode as a ``lax.scan`` over cached
``vertex_contexts``. So every approximation decision for a vertex is conditioned
on an encoding made BEFORE that vertex was chosen — the structure is
hierarchical but not autoregressive, the ``approx:`` tokens are write-only, and
palimpsa (a linear-attention RECURRENCE, picked precisely so it can absorb
deltas) is used exactly once and then asked to re-read a whole buffer.

This module closes that loop. Per face, in order:

    1. emit the local path / face accumulation      (REAL tokenizer vocabulary)
    2. extend the palimpsa carry with ONLY those tokens
    3. ask the policy: skip this path, or approximate it?
    4. emit  ``approx <SKIP|DIAG|COMPRESS|QUANT> <args>``
       - not skip -> followed by the NEW post-approx accumulation
       - skip     -> no accumulation; that edge is deleted
    5. extend the carry again, so the NEXT decision sees step 4

repeating (3)-(5) up to ``max_substeps``, then moving to the next face.

Vertex elimination itself needs no special token: the path header already names
the eliminated vertex (``path <central>, <pred>, <succ>``).

WHY IT HANGS OFF ``face_transforms``: ``IncrementalJaxpr`` exposes no per-face
eliminate — only ``eliminate(vertex, rules, face_transforms)`` — so rules would
otherwise have to be committed before ANY token exists. A callable in a
``face_transforms`` slot is the one hook graphax invokes mid-elimination, per
face, holding that face's live operand. That callable is therefore the decision
point, and this driver is that callable.
"""

from __future__ import annotations

from graphax.jaxpr import int_to_base
from graphax.sparse.micro_actions import Compress, Diag, Quant

from alphagrad.approx.common.masks import (
    couple_quant_rules,
    legal_compress_actions,
    legal_diag_actions,
    rule_is_legal,
)


def _shape_of(st) -> tuple:
    """Logical extents of a live operand: out dims then primal dims."""
    out = []
    for dims in (getattr(st, "out_dims", ()) or (),
                 getattr(st, "primal_dims", ()) or ()):
        for d in dims:
            out.append(int(getattr(d, "logical_size", None)
                          or getattr(d, "size", 0)))
    return tuple(out)


class AutoregressiveFaceDriver:
    """Per-face emit → re-encode → decide → emit loop.

    ``policy(ctx) -> action | None`` is consulted once per approximation slot.
    ``ctx`` carries the live ``operand``, the ``legal`` actions on it RIGHT NOW,
    the running ``tokens``, the palimpsa ``enc_state`` (already advanced past
    everything emitted so far), plus ``face_index`` / ``substep``. Returning
    ``None`` is the per-path SKIP.

    ``tokenizer`` must be a ``graphax.IncrementalPathTokenizer`` — its vocabulary
    and emitters are used directly, so the stream the policy reads is the SAME
    stream the tokenizer would have produced, not a parallel synthetic one.
    ``encoder`` is the ``alphagrad.approx.incremental_encoder`` module (or any
    object exposing ``init_state`` / ``extend``); pass ``None`` to run as a pure
    token/decision recorder.
    """

    def __init__(self, policy, tokenizer, *, encoder=None, agent=None,
                 max_substeps: int = 1, max_dims: int = 8, max_axes: int = 8,
                 compress_kinds: tuple = ("mean",)):
        self.policy = policy
        self.tk = tokenizer
        self.encoder = encoder
        self.agent = agent
        self.max_substeps = int(max_substeps)
        self.max_dims = int(max_dims)
        self.max_axes = int(max_axes)
        self.compress_kinds = tuple(compress_kinds)

        self.tokens: list[int] = []
        self.decisions: list = []
        self.stats: dict = {"faces": 0, "applied": 0, "skipped": 0, "encodes": 0}
        self.enc_state = (
            encoder.init_state(agent)
            if (encoder is not None and agent is not None) else None
        )

    # ---- emission through the REAL tokenizer vocabulary -------------------
    def _word(self, w: str, out: list) -> None:
        self.tk._emit_word(w, out)

    def _accumulation_tokens(self, st) -> list[int]:
        """``path < d0 * d1 * ... >`` — the face accumulation's live shape.

        Uses the tokenizer's own ``path`` word and ``<``/``*``/``>`` shape
        delimiters plus its digit scheme, so these ids are drawn from the same
        vocabulary as everything else in the stream.
        """
        out: list[int] = []
        self._word("path", out)
        self.tk._emit_atoms(self.tk._format_shape(_shape_of(st)), out)
        return out

    def _approx_tokens(self, action) -> list[int]:
        """``approx <TYPE> <args>`` via the tokenizer's own approx emitter."""
        out: list[int] = []
        if action is None:
            self._word("approx", out)
            self._word("SKIP", out)
            return out
        if isinstance(action, Diag):
            atype, params = "DIAG", {"i": action.i, "j": action.j,
                                     "factor": action.factor}
        elif isinstance(action, Compress):
            atype, params = "COMPRESS", {"kind": action.kind,
                                         "axes": tuple(action.axes)}
        elif isinstance(action, Quant):
            atype, params = "QUANT", {"dtype": str(action.dtype)}
        else:
            self._word("approx", out)
            self._word("SKIP", out)
            return out
        self.tk._emit_approx_head(atype, params, out)
        return out

    def _emit(self, new_tokens: list[int]) -> None:
        """Append, then advance the palimpsa carry by ONLY these tokens."""
        if not new_tokens:
            return
        self.tokens.extend(new_tokens)
        if self.enc_state is not None:
            self.enc_state = self.encoder.extend(
                self.agent, self.enc_state, new_tokens)
            self.stats["encodes"] += 1

    def _legal(self, st) -> list:
        return (legal_diag_actions(st, self.max_dims)
                + legal_compress_actions(st, self.max_axes, self.compress_kinds))

    # ---- the graphax face hook -------------------------------------------
    def slot(self):
        def _hook(st):
            face_index = self.stats["faces"]
            self.stats["faces"] += 1

            # 1+2. the path/accumulation, then RE-ENCODE before deciding
            self._emit(self._accumulation_tokens(st))

            cur = st
            dtype_in_force = None
            for substep in range(self.max_substeps):
                legal = self._legal(cur)
                ctx = {
                    "operand": cur, "legal": legal, "tokens": self.tokens,
                    "enc_state": self.enc_state, "face_index": face_index,
                    "substep": substep,
                }
                # 3. decide, conditioned on everything emitted so far
                action = self.policy(ctx) if legal else None

                if action is not None:
                    (action,), dtype_in_force = couple_quant_rules(
                        (action,), applied_dtype=dtype_in_force)
                    if not rule_is_legal(cur, action, max_dims=self.max_dims,
                                         max_axes=self.max_axes):
                        action = None

                if action is None:
                    # 4. SKIP: emit the decision, NO new accumulation — the
                    #    edge is deleted, so there is nothing to describe.
                    self._emit(self._approx_tokens(None))
                    self.stats["skipped"] += 1
                    break

                cur = _apply(cur, action)
                self.decisions.append((face_index, substep, action))
                self.stats["applied"] += 1
                # 4+5. the action AND the resulting accumulation, then
                #      re-encode so the next substep sees both.
                self._emit(self._approx_tokens(action)
                           + self._accumulation_tokens(cur))
            return cur

        return _hook


def _apply(st, action):
    from graphax.sparse.micro_actions import (
        apply_compress, apply_diag, apply_quant)
    if isinstance(action, Diag):
        return apply_diag(st, action)
    if isinstance(action, Compress):
        return apply_compress(st, action)
    if isinstance(action, Quant):
        return apply_quant(st, action)
    return st


def face_transforms_for(incr, vertex: int, driver, slots=("lhs", "rhs", "res")):
    """``{face_key: (lhs, rhs, res)}`` installing ``driver`` in the named slots.

    ``slots`` selects which of the pre / post / new operands the policy may
    approximate; the others are left exact.
    """
    hook = driver.slot()
    sel = tuple(hook if name in slots else None
                for name in ("lhs", "rhs", "res"))
    return {key: sel for key in incr.faces(int(vertex))}


# Backwards-compatible alias: the previous name emitted synthetic shape tokens
# and never re-encoded. Anything importing it now gets the real loop.
InterleavedFaceDriver = AutoregressiveFaceDriver
