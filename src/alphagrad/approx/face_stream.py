"""INTERLEAVED per-face token stream — emit, decide, emit.

The batch path tokenizes the whole jaxpr up front and the policy commits a rule
list per vertex BEFORE anything is emitted, so the observation is
``[base] -> [everything vertex 1 did] -> [everything vertex 2 did] -> ...``.
The spec asks for the finer interleaving::

    [base]
      -> [path 1 tokens]      -> (skip this path?)
      -> [approx head tokens] -> (another approximation?)
      -> [path 2 tokens]      -> ...

i.e. every decision is conditioned on tokens emitted for the thing being
decided. That is only expressible if emission happens DURING elimination, and
graphax gives exactly one such hook: a callable in a ``face_transforms`` slot is
invoked once per face, mid-elimination, holding that face's live operand.

:class:`InterleavedFaceDriver` is that hook. Per face it

  1. emits a compact token block describing the live operand (the "path"),
  2. extends the palimpsa carry with ONLY those new tokens
     (:mod:`~alphagrad.approx.incremental_encoder` — a recurrence, so this is
     O(new tokens), not O(stream)),
  3. asks the policy to skip or pick an approximation,
  4. emits the ``approx <TYPE> <args>`` head for what was picked and extends
     again, so the next decision sees it,

repeating (3)-(4) up to ``max_substeps``. Legality comes from the live operand
(:func:`~alphagrad.approx.common.masks.legal_diag_actions` etc.), so an illegal
action is unrepresentable rather than merely unlikely, and a face where nothing
is legal — or where the policy declines — is left exact: the per-path SKIP.

The driver is deliberately encoder-agnostic: pass ``encoder=None`` to use it as
a pure token/decision recorder (what the tests do), or an object exposing
``init_state`` / ``extend`` to carry a real palimpsa state.
"""

from __future__ import annotations

from graphax.sparse.micro_actions import Compress, Diag, Quant

from alphagrad.approx.common.masks import (
    couple_quant_rules,
    legal_compress_actions,
    legal_diag_actions,
    rule_is_legal,
)

# Token vocabulary for the interleaved stream. These are SYMBOLIC ids local to
# this driver: they name the decision points ("a path opened", "the policy
# skipped", "a DIAG was applied") rather than re-describing the jaxpr, which
# the base tokens already did. Keeping them tiny and fixed means the policy's
# embedding table doesn't have to grow with the graph.
TOK_PATH = 1        # a face/path opened; followed by its shape descriptors
TOK_SKIP = 2        # the policy declined to approximate this operand
TOK_DIAG = 3
TOK_COMPRESS = 4
TOK_QUANT = 5
TOK_END = 6         # no more approximations on this path
TOK_SHAPE = 7       # prefix for a dimension size
NUM_STREAM_TOKENS = 8


def _shape_tokens(st) -> list[int]:
    """Compact descriptor of the live operand: its out/primal extents."""
    out = [TOK_PATH]
    for dims in (getattr(st, "out_dims", ()), getattr(st, "primal_dims", ())):
        for d in dims:
            size = getattr(d, "logical_size", None) or getattr(d, "size", 0)
            out += [TOK_SHAPE, int(size)]
    return out


def _action_tokens(action) -> list[int]:
    """The ``approx <TYPE> <args>`` head for a chosen action."""
    if isinstance(action, Diag):
        return [TOK_DIAG, int(action.i), int(action.j), int(action.factor)]
    if isinstance(action, Compress):
        return [TOK_COMPRESS] + [int(a) for a in action.axes]
    if isinstance(action, Quant):
        return [TOK_QUANT]
    return [TOK_SKIP]


class InterleavedFaceDriver:
    """Per-face emit -> decide -> emit driver.

    ``policy(context) -> action | None`` is called once per approximation slot.
    ``context`` carries ``operand`` (the live SparseTensor), ``legal`` (actions
    legal on it RIGHT NOW), ``tokens`` (everything emitted so far this episode),
    ``enc_state`` (the carried encoder state, or None), ``face_index`` and
    ``substep``. Returning ``None`` skips — which is how the per-path skip is
    expressed.
    """

    def __init__(self, policy, *, encoder=None, agent=None, max_substeps: int = 1,
                 max_dims: int = 8, max_axes: int = 8,
                 compress_kinds: tuple = ("mean",)):
        self.policy = policy
        self.encoder = encoder
        self.agent = agent
        self.max_substeps = int(max_substeps)
        self.max_dims = int(max_dims)
        self.max_axes = int(max_axes)
        self.compress_kinds = tuple(compress_kinds)
        self.tokens: list[int] = []
        self.decisions: list = []
        self.stats: dict = {"faces": 0, "applied": 0, "skipped": 0}
        self.enc_state = (
            encoder.init_state(agent) if (encoder is not None and agent is not None)
            else None
        )

    # -- token plumbing ----------------------------------------------------
    def _emit(self, new_tokens: list[int]) -> None:
        """Append ``new_tokens`` and advance the encoder by ONLY those tokens."""
        if not new_tokens:
            return
        self.tokens.extend(new_tokens)
        if self.enc_state is not None:
            self.enc_state = self.encoder.extend(
                self.agent, self.enc_state, new_tokens)

    def _legal(self, st) -> list:
        return (legal_diag_actions(st, self.max_dims)
                + legal_compress_actions(st, self.max_axes, self.compress_kinds))

    # -- the graphax hook --------------------------------------------------
    def slot(self):
        """Return the callable to install in a ``face_transforms`` slot.

        graphax invokes it once per face with that face's live operand; the
        return value is the transformed operand (unchanged == exact).
        """
        def _hook(st):
            face_index = self.stats["faces"]
            self.stats["faces"] += 1
            # 1. emit the path, so the decision below is conditioned on it
            self._emit(_shape_tokens(st))

            cur = st
            dtype_in_force = None
            for substep in range(self.max_substeps):
                legal = self._legal(cur)
                ctx = {
                    "operand": cur, "legal": legal, "tokens": self.tokens,
                    "enc_state": self.enc_state, "face_index": face_index,
                    "substep": substep,
                }
                action = self.policy(ctx) if legal else None
                if action is None:
                    self._emit([TOK_SKIP if substep == 0 else TOK_END])
                    self.stats["skipped"] += 1
                    break
                # one quantization decision per turn: a later Quant inherits
                (action,), dtype_in_force = couple_quant_rules(
                    (action,), applied_dtype=dtype_in_force)
                if not rule_is_legal(cur, action, max_dims=self.max_dims,
                                     max_axes=self.max_axes):
                    self._emit([TOK_SKIP])
                    self.stats["skipped"] += 1
                    break
                cur = _apply(cur, action)
                self.decisions.append((face_index, substep, action))
                self.stats["applied"] += 1
                # 2. emit what was applied, so the NEXT decision sees it
                self._emit(_action_tokens(action))
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


def face_transforms_for(incr, vertex: int, driver: InterleavedFaceDriver,
                        slots=("lhs", "rhs", "res")):
    """``{face_key: (lhs, rhs, res)}`` installing ``driver`` in the named slots.

    ``slots`` selects which of the pre / post / new operands the policy gets to
    approximate; the others are left exact.
    """
    hook = driver.slot()
    sel = tuple(hook if name in slots else None
                for name in ("lhs", "rhs", "res"))
    return {key: sel for key in incr.faces(int(vertex))}
