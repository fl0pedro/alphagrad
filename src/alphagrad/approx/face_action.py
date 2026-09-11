"""THE face action record. ONE declaration; every use DERIVED from it.

WHY THIS MODULE EXISTS
----------------------
One face decision used to be written out by hand in SEVEN places that had to
agree or PPO broke silently. The four the 2026-09-11 brief named:

1. ``heads.FaceAction`` -- the fields ``UnifiedFacePolicy.sample`` emits;
2. ``ppo.Trajectory`` -- ten ``face_*`` leaves the rollout stored;
3. ``ppo.TrainBatch`` + ``Agent.evaluate``'s ``FaceAction(...)`` rebuild -- the
   ten leaves the loss re-assembled before re-scoring them;
4. ``Agent.to_env_action_dynamic`` -- where eight of them become the env's
   ``(F, S, 3)`` wire rows.

and three nobody had counted, each restating the names AND the canonical
inactive values a second and third time:

5. ``ppo._zero_face_action``;
6. ``common.face_buckets.pad_face_outputs``;
7. ``Agent._face_loop``'s ``wire0`` carry init.

(``Agent._face_row_specs`` restated use 4's eight argument names a second time
as well.)

FORGETTING NUMBER 3 IS SILENT: ``evaluate`` then scores a variable ``sample``
never drew, no error is raised, no shape disagrees, and the PPO ratio simply
drifts off 1 while every number the run reports stays plausible. That failure
mode is what this module deletes -- and it deletes it for uses 2 and 3
STRUCTURALLY: the carriers hold the record as ONE leaf, so there is no list of
ten names left to keep in step.

WHAT IS DECLARED AND WHAT IS DERIVED
------------------------------------
:data:`FACE_ACTION_FIELDS` is the declaration: per field, the name, the dtype,
whether it is PER SLOT or PER FACE, its trailing dims, its canonical inactive
value, whether the head DRAWS it or the wire encoder DERIVES it, whether it
feeds ``env.micro_actions_to_rule_specs_jax``, and which ``--approx-add``
widths carry it at all.

Everything else is computed from that list:

* :data:`FaceAction` -- the NamedTuple itself, built by :func:`_namedtuple`;
* :func:`zeros` -- the canonical inactive record (use 1's padding, and
  ``pad_face_outputs``' fill values);
* :func:`shapes` -- every field's shape at one width, so a MIS-SLOTTED row
  raises instead of broadcasting;
* :func:`check` -- the validator the sampling and wire boundaries call;
* :func:`translator_names` -- the ordered argument list of
  ``micro_actions_to_rule_specs_jax``, i.e. use 4;
* :func:`check_carrier` -- uses 2 and 3 no longer restate anything at all
  (``Trajectory``/``TrainBatch`` carry the record as ONE leaf), and this
  function is what pins that.

A NEW FIELD IS ADDED IN EXACTLY ONE PLACE: a line in
:data:`FACE_ACTION_FIELDS`.

THE SLOT AXIS FOLLOWS THE HEAD'S WIDTH
--------------------------------------
Per-slot fields are ``(MAX_FACES, n_slots)`` where ``n_slots`` is
``unified_face_head.head_layout(mode).n_slots`` -- 3 under ``lossy`` /
``lossless`` / ``choose``, 4 under ``learned1``, 5 under ``learned2``. That is
the SAME table the head's logit width and ``env.wire_slots`` come from, so a
record, a head and a wire row cannot be built at three different widths.

``FACE_SLOTS`` stays 3 and keeps meaning "the CONTRACTION slots" (lhs, rhs,
new), so every existing ``range(FACE_SLOTS)`` loop stays correct. The SHAPE is
the widened quantity, and it is spelled :func:`n_slots` / ``env.wire_slots()``
-- never ``FACE_SLOTS``.

IMPORT WEIGHT
-------------
This module imports ``heads`` (for ``MAX_PRIMES`` / ``OP_END``) and
``unified_face_head`` (for the layout table). It does NOT import ``env`` or
``ppo`` at module scope: ``ppo`` is 13k lines and ``env`` pulls graphax, and a
declaration every one of them has to agree with must be importable by all of
them. ``MAX_FACES`` is read from ``env`` LAZILY, only when a caller omits it.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np

from alphagrad.approx.heads import MAX_PRIMES, OP_END
from alphagrad.approx.unified_face_head import (
    FACE_SLOTS, JOIN_LOSSY, head_layout,
)

#: A field the head DRAWS: ``sample`` picks it from a logit and ``score`` must
#: read it back off the stored record, or the two score different variables.
SCORED = "scored"
#: A field the wire encoder DERIVES from the scored ones plus the face's live
#: sizes -- ``factor`` / ``exponents`` (the gcd factorisation) and the two
#: quant-scale fields (constants). ``score`` does NOT read these, and must not:
#: they carry no decision. What the agreement test requires of them instead is
#: that the REPLAY re-derives them bit-for-bit (round trip in
#: ``tests/face_action_record_test.py``).
DERIVED = "derived"

#: Every ``--approx-add`` value, as ``unified_face_head._LAYOUT_SPEC`` knows
#: them. Named here so :data:`FACE_ACTION_FIELDS`' ``modes`` entries can be
#: checked against the layout table at import rather than being free text.
from alphagrad.approx.unified_face_head import _LAYOUT_SPEC as _MODES  # noqa: E402

ALL_MODES: tuple[str, ...] = tuple(_MODES)


@dataclass(frozen=True)
class FaceField:
    """ONE field of the face action record.

    ``per_slot``   -- leading shape ``(MAX_FACES, n_slots)`` rather than
                      ``(MAX_FACES,)``. The slot axis follows the HEAD's width
                      (module docstring), never ``FACE_SLOTS``.
    ``tail``       -- dims after the face/slot axes, e.g. ``(MAX_PRIMES,)``.
    ``fill``       -- the canonical INACTIVE value. One number, used by the
                      padding record, by the bucketed re-pad and by the
                      dead-face forcing, so those three cannot disagree about
                      what "no approximation" looks like.
    ``role``       -- :data:`SCORED` or :data:`DERIVED`.
    ``translator`` -- does it feed ``env.micro_actions_to_rule_specs_jax``?
                      ``exponents`` does not: it is the factorisation of
                      ``factor``, carried for the plan log, and the translator
                      takes the integer.
    ``modes``      -- ``None`` = present at every width. A tuple restricts the
                      field to those ``--approx-add`` values, and at any other
                      width the record's entry is ``None`` -- "this decision
                      has no field", exactly as ``FaceFields.join`` is ``None``
                      at a width with no logit for it.
    """

    name: str
    dtype: str
    per_slot: bool
    fill: Any
    role: str
    translator: bool
    tail: tuple[int, ...] = ()
    modes: tuple[str, ...] | None = None
    doc: str = ""

    def __post_init__(self):
        if self.role not in (SCORED, DERIVED):
            raise ValueError(f"{self.name}: role must be scored/derived")
        if self.modes is not None:
            bad = set(self.modes) - set(ALL_MODES)
            if bad:
                raise ValueError(
                    f"{self.name}: modes {sorted(bad)} are not --approx-add "
                    f"values ({ALL_MODES}); the field list and the head's "
                    f"layout table would describe different widths.")
        if self.translator and not self.per_slot:
            raise ValueError(
                f"{self.name}: the rule-spec translator is per (face, slot); "
                f"a per-FACE field cannot be one of its arguments.")

    @property
    def np_dtype(self):
        return np.dtype(self.dtype)

    @property
    def carrier(self) -> str:
        """The leaf name a flat carrier would give it: ``face_<name>``.

        Kept because the plan log and the telemetry still speak that dialect;
        ``Trajectory`` / ``TrainBatch`` carry the record whole (see
        :func:`check_carrier`).
        """
        return f"face_{self.name}"

    def present(self, mode: str) -> bool:
        return self.modes is None or mode in self.modes

    def shape(self, mode: str, max_faces: int) -> tuple[int, ...]:
        lead = ((int(max_faces), n_slots(mode)) if self.per_slot
                else (int(max_faces),))
        return lead + tuple(self.tail)


#: THE DECLARATION. Order is the NamedTuple's field order, so ``skip`` stays
#: first and new fields go LAST -- positional construction anywhere keeps
#: working, and ``join`` (2026-09-11) is appended rather than inserted.
FACE_ACTION_FIELDS: tuple[FaceField, ...] = (
    FaceField(
        "skip", "int32", per_slot=False, fill=0, role=SCORED,
        translator=False,
        doc="1 => this face's contraction is dropped (graphax.SKIP_FACE) and "
            "every slot is forced to its canonical inactive value."),
    FaceField(
        "op_type", "int32", per_slot=True, fill=int(OP_END), role=SCORED,
        translator=True,
        doc="the slot's approximation op; OP_END/OP_NONE = none."),
    FaceField(
        "i", "int32", per_slot=True, fill=0, role=SCORED, translator=True,
        doc="DIAG's first axis-pair index, or COMPRESS's axis (the wire "
            "parks the reduce axis in `i`)."),
    FaceField(
        "j", "int32", per_slot=True, fill=0, role=SCORED, translator=True,
        doc="DIAG's second axis-pair index; 0 for every other op."),
    FaceField(
        "exponents", "int32", per_slot=True, fill=0, role=DERIVED,
        translator=False, tail=(MAX_PRIMES,),
        doc="the prime factorisation of `factor`. DERIVED from the face's "
            "live gcd, not drawn, and NOT a translator argument -- the "
            "translator takes the integer `factor`."),
    FaceField(
        "factor", "int32", per_slot=True, fill=0, role=DERIVED,
        translator=True,
        doc="gcd(N_i, N_j) on the slot's own live sizes. DERIVED: sampling it "
            "was removed because the env clamped it to a divisor of this gcd "
            "anyway."),
    FaceField(
        "compress_kind", "int32", per_slot=True, fill=0, role=SCORED,
        translator=True,
        doc="COMPRESS's reduction fn index."),
    FaceField(
        "quant_dtype", "int32", per_slot=True, fill=0, role=SCORED,
        translator=True,
        doc="QUANT's target dtype slot."),
    FaceField(
        "quant_scale_sign", "int32", per_slot=True, fill=1, role=DERIVED,
        translator=True,
        doc="CONSTANT +1. The face head draws no quant scale; the field exists "
            "because the shared translator takes one."),
    FaceField(
        "quant_scale_frac", "float32", per_slot=True, fill=0.0, role=DERIVED,
        translator=True,
        doc="CONSTANT 0.0, for the same reason as quant_scale_sign."),
    FaceField(
        "join", "int32", per_slot=False, fill=int(JOIN_LOSSY), role=SCORED,
        translator=False, modes=("choose",),
        doc="--approx-add choose: ONE Bernoulli per face, 0 = lossy, "
            "1 = lossless, drawn at logit `1 + 31*n_slots`. PRESENT IFF the "
            "width has the bit; `None` otherwise, because a width without the "
            "bit has no logit to score it against and substituting JOIN_LOSSY "
            "would measure the plan under a join the policy did not pick."),
)

_BY_NAME = {f.name: f for f in FACE_ACTION_FIELDS}
if len(_BY_NAME) != len(FACE_ACTION_FIELDS):
    raise RuntimeError("duplicate field name in FACE_ACTION_FIELDS")


def field(name: str) -> FaceField:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a face action field. The record is declared in "
            f"alphagrad.approx.face_action.FACE_ACTION_FIELDS and has "
            f"{[f.name for f in FACE_ACTION_FIELDS]}.") from None


# ---------------------------------------------------------------- the record
def _namedtuple():
    """Build :data:`FaceAction` from the declaration.

    A namedtuple rather than a dataclass because the rest of the trainer
    already treats the action as a jax pytree and stacks it with ``lax.scan``
    (jax registers every namedtuple subclass automatically).

    Width-restricted fields default to ``None`` and must therefore be the
    SUFFIX of the declaration -- asserted here rather than assumed, because
    ``namedtuple``'s ``defaults`` are right-aligned and a field inserted in the
    middle would silently give its default to its neighbour.
    """
    optional = [i for i, f in enumerate(FACE_ACTION_FIELDS)
                if f.modes is not None]
    if optional and optional != list(
            range(len(FACE_ACTION_FIELDS) - len(optional),
                  len(FACE_ACTION_FIELDS))):
        raise RuntimeError(
            "width-restricted face action fields "
            f"{[FACE_ACTION_FIELDS[i].name for i in optional]} must be the "
            "LAST entries of FACE_ACTION_FIELDS: namedtuple defaults are "
            "right-aligned, so an optional field in the middle would hand its "
            "default to the field after it.")
    cls = namedtuple("FaceAction", [f.name for f in FACE_ACTION_FIELDS],
                     defaults=[None] * len(optional))
    cls.__doc__ = (
        "Per-face decisions for ONE vertex elimination, padded to MAX_FACES.\n"
        "\nDERIVED from face_action.FACE_ACTION_FIELDS -- do not add a field "
        "here, add it there.\n\n"
        + "\n".join(
            f"  {f.name}: "
            f"{'(F, S)' if f.per_slot else '(F,)'}"
            f"{''.join('+(%d,)' % d for d in f.tail)} {f.dtype}"
            f"  [{f.role}]"
            f"{'' if f.modes is None else '  [%s only]' % ','.join(f.modes)}"
            f"\n      {f.doc}"
            for f in FACE_ACTION_FIELDS))
    return cls


FaceAction = _namedtuple()

#: The wire keys ``Agent._face_loop`` carries per face, in declaration order:
#: every field except the per-FACE ones, which travel beside the rows.
WIRE_KEYS: tuple[str, ...] = tuple(
    f.name for f in FACE_ACTION_FIELDS if f.per_slot)


# ------------------------------------------------------------------ queries
def n_slots(mode: str) -> int:
    """The record's SLOT-axis width at one ``--approx-add`` value.

    ``head_layout(mode).n_slots`` and nothing else -- the same number the head
    sizes its logits from and ``env.wire_slots()`` returns.
    """
    return head_layout(mode).n_slots


def fields(mode: str | None = None) -> tuple[FaceField, ...]:
    """The fields present at ``mode``; every field when ``mode`` is None."""
    if mode is None:
        return FACE_ACTION_FIELDS
    head_layout(mode)  # raises on an unknown value, with the layout's message
    return tuple(f for f in FACE_ACTION_FIELDS if f.present(mode))


def names(mode: str | None = None) -> tuple[str, ...]:
    return tuple(f.name for f in fields(mode))


def scored_names(mode: str | None = None) -> tuple[str, ...]:
    return tuple(f.name for f in fields(mode) if f.role == SCORED)


def derived_names(mode: str | None = None) -> tuple[str, ...]:
    return tuple(f.name for f in fields(mode) if f.role == DERIVED)


def per_slot_names(mode: str | None = None) -> tuple[str, ...]:
    return tuple(f.name for f in fields(mode) if f.per_slot)


def per_face_names(mode: str | None = None) -> tuple[str, ...]:
    return tuple(f.name for f in fields(mode) if not f.per_slot)


def translator_names() -> tuple[str, ...]:
    """USE 4, declared: the per-slot fields ``env.micro_actions_to_rule_specs_jax``
    consumes, in the order it takes them.

    ``_face_row_specs`` and ``to_env_action_dynamic`` both build their vmapped
    call from this tuple, so neither can pass a field the other does not.
    """
    return tuple(f.name for f in FACE_ACTION_FIELDS if f.translator)


def shapes(mode: str, max_faces: int) -> dict[str, tuple[int, ...]]:
    """Every present field's exact shape at one width."""
    return {f.name: f.shape(mode, max_faces) for f in fields(mode)}


# -------------------------------------------------------------- constructors
def _max_faces(max_faces: int | None) -> int:
    if max_faces is not None:
        return int(max_faces)
    from alphagrad.approx.env import MAX_FACES
    return int(MAX_FACES)


def _mode(mode: str | None) -> str:
    if mode is not None:
        return mode
    from alphagrad.approx.env import approx_add
    return approx_add()


def zeros(mode: str | None = None, max_faces: int | None = None, *,
          backend="jax"):
    """The canonical INACTIVE record: no skip, END slots, fills as declared.

    ONE definition of "no approximation", so ``ppo._zero_face_action``, the
    bucketed re-pad's fill values and the dead-face forcing cannot disagree.
    ``backend="numpy"`` for the host-side callers (``face_buckets``).
    """
    mode, F = _mode(mode), _max_faces(max_faces)
    if backend == "numpy":
        def _full(shape, fill, dt):
            return np.full(shape, fill, dtype=dt)
    else:
        import jax.numpy as jnp

        def _full(shape, fill, dt):
            return jnp.full(shape, fill, dtype=dt)
    return FaceAction(**{
        f.name: _full(f.shape(mode, F), f.fill, f.np_dtype)
        for f in fields(mode)})


def fill_like(fa, max_faces: int, mode: str | None = None):
    """``fa`` re-padded up to ``max_faces`` faces with the declared fills.

    The host-side half of a bucketed draw (``face_buckets.pad_face_outputs``):
    faces the device while_loop never visited must carry EXACTLY the bytes the
    unbucketed call would have produced, and those bytes are one column of the
    declaration.
    """
    mode = _mode(mode)
    F = int(max_faces)
    out = {}
    for f in fields(mode):
        x = getattr(fa, f.name)
        if x is None:
            out[f.name] = None
            continue
        x = np.asarray(x)
        o = np.full((F,) + x.shape[1:], f.fill, dtype=x.dtype)
        o[:x.shape[0]] = x
        out[f.name] = o
    return FaceAction(**out)


# --------------------------------------------------------------- validation
def check(fa, mode: str | None = None, max_faces: int | None = None, *,
          where: str = "") -> None:
    """Raise unless ``fa`` is EXACTLY the record this width declares.

    Checks presence, absence, shape and dtype, field by field, and says which
    field and which width in the message. A MIS-SLOTTED row therefore raises
    here rather than broadcasting into a neighbouring slot's decision --
    which is the whole point of widening the slot axis from the layout table
    instead of from ``FACE_SLOTS``.
    """
    mode = _mode(mode)
    tag = f" ({where})" if where else ""
    if max_faces is None:
        # `skip` is the one field present at every width and always `(F,)`, so
        # it is the only honest place to infer the face count from.
        shape = np.shape(getattr(fa, "skip", None))
        if len(shape) != 1:
            raise ValueError(
                f"face action record{tag}: `skip` has shape {shape}, expected "
                f"(MAX_FACES,), so the face count cannot be inferred. Pass "
                f"max_faces explicitly.")
        max_faces = int(shape[0])
    F, S = int(max_faces), n_slots(mode)
    for f in FACE_ACTION_FIELDS:
        x = getattr(fa, f.name, None)
        if not f.present(mode):
            if x is not None:
                raise ValueError(
                    f"face action record{tag} carries {f.name!r} but "
                    f"--approx-add {mode!r} HAS NO SUCH FIELD (it is declared "
                    f"for {f.modes}). A field this width does not contain has "
                    f"no logit, so `score` cannot read it back and `sample` "
                    f"never drew it.")
            continue
        if x is None:
            raise ValueError(
                f"face action record{tag} is missing {f.name!r}, which "
                f"--approx-add {mode!r} DOES contain. Leaving it out would let "
                f"`evaluate` score a default the head never drew -- the PPO "
                f"ratio drifts off 1 with no error anywhere.")
        want = f.shape(mode, F)
        got = tuple(int(d) for d in np.shape(x))
        if got != want:
            raise ValueError(
                f"face action record{tag} field {f.name!r} has shape {got}, "
                f"expected {want} at --approx-add {mode!r} "
                f"(MAX_FACES={F}, slots={S}: {FACE_SLOTS} contraction + "
                f"{S - FACE_SLOTS} learned join). The slot axis follows the "
                f"HEAD's width (unified_face_head._LAYOUT_SPEC); a row at the "
                f"wrong width would be applied to a tensor nothing was "
                f"decided for.")
        dt = np.dtype(getattr(x, "dtype", f.np_dtype))
        if dt != f.np_dtype:
            raise ValueError(
                f"face action record{tag} field {f.name!r} has dtype {dt}, "
                f"declared {f.np_dtype} in face_action.FACE_ACTION_FIELDS.")


def check_carrier(cls, *, where: str = "") -> None:
    """USES 2 AND 3: the carrier must hold the record WHOLE, not field by field.

    ``Trajectory`` and ``TrainBatch`` carry ONE ``face_action`` leaf. That is
    the structural answer to the drift this module exists for: there is no list
    of ten names to keep in step, so a field added to the declaration reaches
    the rollout and the loss replay with no edit at all.

    This function refuses the old shape explicitly -- a carrier that grew a
    ``face_<name>`` leaf back has re-created the second copy, and the next
    field added to the declaration would be silently absent from it.
    """
    fs = set(getattr(cls, "_fields", ()))
    if "face_action" not in fs:
        raise TypeError(
            f"{cls.__name__}{' (' + where + ')' if where else ''} must carry "
            f"the face action record as ONE leaf named `face_action` (type "
            f"face_action.FaceAction). It is the declaration's only carrier: "
            f"ten separate `face_*` leaves is the four-places restatement this "
            f"module removed.")
    flat = sorted(f.carrier for f in FACE_ACTION_FIELDS if f.carrier in fs)
    if flat:
        raise TypeError(
            f"{cls.__name__} carries BOTH `face_action` and the flat leaves "
            f"{flat}. Two copies of one decision is exactly the drift this "
            f"module removed: drop the flat leaves and read "
            f"`{cls.__name__.lower()}.face_action.<field>`.")


def check_wire_row_keys(row, mode: str | None = None, *,
                        where: str = "") -> None:
    """The per-slot wire dict ``UnifiedFacePolicy._rows`` emits must have
    EXACTLY the declared per-slot field names.

    This is the seam between the head's ``FaceFields`` and the action record.
    A key the declaration does not know is a field nothing stores; a declared
    key the encoder does not write is a field ``sample`` leaves at a default
    and ``evaluate`` then scores.
    """
    mode = _mode(mode)
    want = set(per_slot_names(mode))
    got = set(row)
    if got != want:
        raise ValueError(
            f"the face wire row{' (' + where + ')' if where else ''} has keys "
            f"{sorted(got)}, the declaration says {sorted(want)} at "
            f"--approx-add {mode!r} (missing {sorted(want - got)}, unexpected "
            f"{sorted(got - want)}). Add the field in "
            f"face_action.FACE_ACTION_FIELDS, not here.")
