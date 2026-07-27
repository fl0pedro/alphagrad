"""JAX-free variant presets and per-variant action masks.

Lives outside ``alphagrad.approx.common`` so JAX-free driver entry points can
import variant metadata without pulling JAX into the driver process.

``MAX_RULES_PER_VERTEX`` is duplicated from ``alphagrad.approx.env`` (which
imports JAX). The constant is part of the action-space contract — bumping it
requires changing it in both places.
"""

from __future__ import annotations


MAX_RULES_PER_VERTEX = 16


VARIANT_PRESETS: dict[str, dict] = {
    "custom": {},
    "ve_only": {"pin_rules_to_exact": True},
    # SIMPLE family — one fixed operator with NO inner choice (the
    # policy only picks WHICH vertex to act on; the operator's payload
    # is pinned to its simplest option). Pairs with the "DIFFICULT"
    # family below where the same operator gets its full inner choice
    # set.
    #
    # diag_gcd:   DIAG with factor = -1 (auto gcd per edge).
    # compress_scalar: COMPRESS with kind = "mean" (kind index 0). The
    #                  agent's compress_kind head isn't sampled today
    #                  (env defaults to mean) so this is equivalent to
    #                  ``compress`` for now.
    # quant_smallest_float: QUANT with dtype = smallest float
    #                  available (``float4_e2m1fn``, index 13 in
    #                  ``graphax.sparse.micro_actions.QUANT_DTYPES``).
    #                  The agent's quant_dtype head IS sampled today,
    #                  so this variant restricts the head's logits to
    #                  just that single dtype.
    "diag_gcd": {"factors": "-1", "max_rules": 1, "pin_rules_to_exact": False},
    "compress_scalar": {
        "factors": "-1",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "quant_smallest_float": {
        "factors": "-1",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    # DIFFICULT family — same operator, full inner-choice palette.
    "diag_factor": {
        "factors": "2,3,4,8,16",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "compress": {
        "factors": "-1",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    # quantize: 1 substep per vertex emitting a QUANT spec (op_type=2 in
    # heads.OP_QUANT). The factor table is irrelevant for QUANT — the
    # micro action only carries a quant_dtype index — but we keep the
    # placeholder "-1" so the agent's factor head still exists (and is
    # masked out via the op-type gating in act_step). The PPO-ray
    # worker is the only trainer that currently exercises QUANT.
    "quantize": {
        "factors": "-1",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    # PAIR family — two DIFFICULT operators legal simultaneously. The
    # factor table is the union of the constituent variants (DIAG needs
    # its full factor table; COMPRESS / QUANT only need -1). max_rules
    # stays at 1 because the action space is "pick one op per vertex"
    # — the pair just widens the op_type choice from {one op, END} to
    # {two ops, END}.
    "diag_compress": {
        "factors": "-1,2,3,4,8,16",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "diag_quant": {
        "factors": "-1,2,3,4,8,16",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "compress_quant": {
        "factors": "-1",
        "max_rules": 1,
        "pin_rules_to_exact": False,
    },
    "full": {
        "factors": "-1,2,3,4,8,16",
        "max_rules": MAX_RULES_PER_VERTEX,
        "pin_rules_to_exact": False,
    },
}


# Operator family used by :func:`compute_union_variant_masks` when unioning
# the three SIMPLE operators into a single ``all_simple`` mask.
SIMPLE_OPERATORS: tuple[str, ...] = (
    "diag_gcd", "compress_scalar", "quant_smallest_float",
)


def _apply_variant_preset(args, variant: str | None = None) -> None:
    """In-place apply a ``--variant`` preset to ``args``.

    ``variant`` overrides ``args.variant`` if given.
    """
    name = variant if variant is not None else getattr(args, "variant", "custom")
    if name not in VARIANT_PRESETS:
        raise ValueError(
            f"Unknown --variant '{name}'. Valid: {list(VARIANT_PRESETS)}."
        )
    for k, v in VARIANT_PRESETS[name].items():
        setattr(args, k, v)


# Per-variant op-legality / pin-rules helper. ``ve_only`` forces every
# pair-slot to STOP, which is exactly the "no rule" output the pinned
# legacy path produces.
def _pin_rules_for_variant(variant: str) -> bool:
    return variant == "ve_only"


# ---------------------------------------------------------------------------
# PPO-side per-variant action masks
# ---------------------------------------------------------------------------
# The PPO worker builds the agent with the FULL action footprint
# (op_type ∈ {DIAG, COMPRESS, QUANT, END}, factor_table = the union of
# every variant's legal factors). The chosen variant then restricts the
# effective action space by masking the policy logits before sampling, so
# the agent weights survive across variants.
#
# ``compute_ppo_variant_masks(variant, full_factor_table) → dict``
# returns ``{op_type_mask: (4,) bool, factor_mask: (F,) bool}`` where:
#   * ``op_type_mask[k] = True`` means op_type sample ``k`` is allowed.
#   * ``factor_mask[j] = True`` means ``full_factor_table[j]`` is
#     allowed. ``factor_mask`` defaults to all-True for variants that
#     don't restrict the factor table.
#
# Convention: indices match the OP_* constants in alphagrad.approx.heads:
#   0 = OP_DIAG, 1 = OP_COMPRESS, 2 = OP_QUANT, 3 = OP_END

_OP_DIAG = 0
_OP_COMPRESS = 1
_OP_QUANT = 2
_OP_END = 3


def _quant_allowed_mask(num_quant_dtypes: int):
    """(num_quant_dtypes,) bool mask of dtypes permitted by
    ``ALPHAGRAD_QUANT_ALLOWED`` (comma-list of QUANT_DTYPES names).

    Mirrors ``heads._quant_dtype_mask`` so the variant action-mask
    honours the same env-var restriction as the policy head. Unset env
    var → all-True (no restriction). Names not in QUANT_DTYPES are
    ignored. Returns None if QUANT_DTYPES cannot be imported (JAX-free
    fallback: caller leaves the mask unchanged).
    """
    import os
    import numpy as _np

    _env = os.environ.get("ALPHAGRAD_QUANT_ALLOWED", "").strip()
    if not _env:
        return None
    try:
        from graphax.sparse.micro_actions import QUANT_DTYPES as _QUANT_DTYPES
    except ImportError:
        return None
    _allowed = {s.strip() for s in _env.split(",") if s.strip()}
    mask = _np.array(
        [d in _allowed for d in _QUANT_DTYPES], dtype=_np.bool_
    )
    # Guard against a head/dtype-list size mismatch: pad/truncate to the
    # mask length so the intersection is always well-formed.
    if mask.shape[0] != num_quant_dtypes:
        out = _np.zeros((num_quant_dtypes,), dtype=_np.bool_)
        n = min(mask.shape[0], num_quant_dtypes)
        out[:n] = mask[:n]
        mask = out
    return mask


def compute_ppo_variant_masks(
    variant: str,
    full_factor_table: tuple[int, ...] | list[int],
    num_quant_dtypes: int | None = None,
) -> dict:
    """Per-variant action masks for the PPO agent.

    Args:
        variant: variant name (one of :data:`VARIANT_PRESETS`).
        full_factor_table: the agent's full factor table at init.
            Variants with a restricted factor set produce a mask True
            only at indices whose value is in the variant's factor list.
        num_quant_dtypes: size of the agent's quant_dtype head. When
            None, defaults to the import of
            ``graphax.sparse.micro_actions.NUM_QUANT_DTYPES``. Variants
            with a restricted quant set produce a mask True only at the
            allowed indices.

    Returns ``{"op_type_mask": (4,) bool, "factor_mask": (F,) bool,
    "quant_dtype_mask": (NUM_QUANT_DTYPES,) bool}``.

    Semantics:
      * ``ve_only`` — op_type forced to END; the agent emits no rules,
        the env does pure vertex elimination. Other masks are all True
        (irrelevant since op_type=END).
      * ``diag_gcd`` — op_type ∈ {DIAG, END}; factor restricted to ``-1``.
      * ``compress_scalar`` — op_type ∈ {COMPRESS, END}; the agent's
        compress_kind head defaults to ``mean`` (kind 0); factor and
        quant masks are unrestricted (irrelevant under COMPRESS).
      * ``quant_smallest_float`` — op_type ∈ {QUANT, END}; quant_dtype
        restricted to the smallest float (``float4_e2m1fn``, index 13
        in QUANT_DTYPES); factor mask unrestricted (irrelevant).
      * ``diag_factor`` — op_type ∈ {DIAG, END}; factor table
        ``{2,3,4,8,16} ∩ full_factor_table`` (no -1).
      * ``compress`` — op_type ∈ {COMPRESS, END}; factor=-1 only
        (compress_kind head not yet sampled; reserved for future
        expansion); quant mask unrestricted.
      * ``quantize`` — op_type ∈ {QUANT, END}; full quant_dtype set;
        factor=-1 only.
      * ``full`` / ``custom`` — no restriction.
    """
    import numpy as _np

    if num_quant_dtypes is None:
        try:
            from graphax.sparse.micro_actions import (
                NUM_QUANT_DTYPES as _NUM_QUANT_DTYPES,
            )
            num_quant_dtypes = int(_NUM_QUANT_DTYPES)
        except ImportError:
            num_quant_dtypes = 27  # known value at time of writing

    op_mask = _np.ones((4,), dtype=_np.bool_)
    factor_mask = _np.ones((len(full_factor_table),), dtype=_np.bool_)
    quant_dtype_mask = _np.ones((num_quant_dtypes,), dtype=_np.bool_)

    if variant == "ve_only":
        op_mask[:] = False
        op_mask[_OP_END] = True
    elif variant == "diag_gcd":
        op_mask[_OP_COMPRESS] = False
        op_mask[_OP_QUANT] = False
        factor_mask[:] = False
        for j, f in enumerate(full_factor_table):
            if int(f) == -1:
                factor_mask[j] = True
    elif variant == "compress_scalar":
        # COMPRESS-only with the simplest reduction (kind=mean is the
        # env default when compress_kinds is not supplied). The "simple"
        # vs "compress" distinction is reserved for when a
        # compress_kind head lands; today both reduce-to-mean.
        op_mask[_OP_DIAG] = False
        op_mask[_OP_QUANT] = False
    elif variant == "quant_smallest_float":
        # QUANT-only with smallest-float dtype.
        # graphax.sparse.micro_actions.QUANT_DTYPES has
        # ``float4_e2m1fn`` at index 13 (counted from the canonical
        # tuple defined there). If the head size is smaller than 14
        # (downstream changed the dtype list), fall back to the
        # highest available index.
        op_mask[_OP_DIAG] = False
        op_mask[_OP_COMPRESS] = False
        quant_dtype_mask[:] = False
        smallest_float_idx = min(13, num_quant_dtypes - 1)
        quant_dtype_mask[smallest_float_idx] = True
    elif variant == "diag_factor":
        op_mask[_OP_COMPRESS] = False
        op_mask[_OP_QUANT] = False
        factor_mask[:] = False
        allowed = {2, 3, 4, 8, 16}
        for j, f in enumerate(full_factor_table):
            if int(f) in allowed:
                factor_mask[j] = True
    elif variant == "compress":
        op_mask[_OP_DIAG] = False
        op_mask[_OP_QUANT] = False
    elif variant == "quantize":
        op_mask[_OP_DIAG] = False
        op_mask[_OP_COMPRESS] = False
    elif variant == "diag_compress":
        # DIAG ∪ COMPRESS legal — only disable QUANT. Factor + quant
        # masks stay all-True (DIAG uses factors; COMPRESS doesn't —
        # the op-type gate in the worker handles that).
        op_mask[_OP_QUANT] = False
    elif variant == "diag_quant":
        # DIAG ∪ QUANT legal — only disable COMPRESS.
        op_mask[_OP_COMPRESS] = False
    elif variant == "compress_quant":
        # COMPRESS ∪ QUANT legal — only disable DIAG.
        op_mask[_OP_DIAG] = False
    elif variant in ("full", "custom"):
        pass  # no restriction
    else:
        raise ValueError(
            f"Unknown PPO variant '{variant}'. Valid: "
            f"{list(VARIANT_PRESETS)}"
        )

    # Intersect the variant quant mask with ALPHAGRAD_QUANT_ALLOWED so
    # int/uint/bool/complex/exotic-float dtypes (which zero the gradient ->
    # cossim 0) are unsampleable in BOTH the policy head (heads._quant_dtype_mask)
    # AND the variant action mask. Closes the `full`-variant bypass where
    # the quant mask was all-True regardless of the env var.
    _allowed = _quant_allowed_mask(quant_dtype_mask.shape[0])
    if _allowed is not None:
        quant_dtype_mask &= _allowed

    # Defensive: must have at least one legal op_type / factor / quant
    # entry, otherwise the policy can't sample any valid action.
    if not op_mask.any():
        raise ValueError(
            f"Variant '{variant}' produces empty op_type mask — bug?"
        )
    if not factor_mask.any():
        raise ValueError(
            f"Variant '{variant}' produces empty factor mask — likely "
            f"the full_factor_table {tuple(full_factor_table)} does not "
            f"include any of the variant's allowed factors."
        )
    if not quant_dtype_mask.any():
        raise ValueError(
            f"Variant '{variant}' produces empty quant_dtype mask — "
            f"num_quant_dtypes={num_quant_dtypes} too small for the "
            f"variant's required dtype index."
        )
    return {
        "op_type_mask": op_mask,
        "factor_mask": factor_mask,
        "quant_dtype_mask": quant_dtype_mask,
    }


def ppo_full_factor_table() -> tuple[int, ...]:
    """The factor table the PPO agent should be built with so that
    every variant's allowed factors are addressable. Union of every
    variant's factor list."""
    return (-1, 2, 3, 4, 8, 16)


def compute_union_variant_masks(
    variant_or_union: str,
    full_factor_table: tuple[int, ...] | list[int],
    num_quant_dtypes: int | None = None,
) -> dict:
    """Like :func:`compute_ppo_variant_masks` but accepts compound
    variant strings of the form ``"A+B"`` or ``"all_simple"`` and
    returns the LOGICAL UNION of the constituent masks.

    ``compute_ppo_variant_masks`` stays the single-variant primitive.
    """
    import numpy as _np

    if variant_or_union == "all_simple":
        members = list(SIMPLE_OPERATORS)
    elif "+" in variant_or_union:
        members = variant_or_union.split("+")
    else:
        return compute_ppo_variant_masks(
            variant_or_union, full_factor_table, num_quant_dtypes,
        )

    op_mask = None
    factor_mask = None
    quant_mask = None
    for v in members:
        m = compute_ppo_variant_masks(
            v, full_factor_table, num_quant_dtypes,
        )
        if op_mask is None:
            op_mask = m["op_type_mask"].copy()
            factor_mask = m["factor_mask"].copy()
            quant_mask = m["quant_dtype_mask"].copy()
        else:
            op_mask |= m["op_type_mask"]
            factor_mask |= m["factor_mask"]
            quant_mask |= m["quant_dtype_mask"]
    # Members already intersected with ALPHAGRAD_QUANT_ALLOWED in
    # compute_ppo_variant_masks; re-intersect defensively (idempotent) so the
    # union can never re-admit a disallowed int/exotic dtype.
    _allowed = _quant_allowed_mask(quant_mask.shape[0])
    if _allowed is not None:
        quant_mask &= _allowed
    return {
        "op_type_mask": op_mask,
        "factor_mask": factor_mask,
        "quant_dtype_mask": quant_mask,
    }


# ---------------------------------------------------------------------------
# Curriculum helpers — restored for mu0.py (dropped from the ppo line in the
# 9cdd258 trim). Identical to the pre-trim implementations except the deleted
# seven-stage rotation names are no longer valid stage names: stages must be
# VARIANT_PRESETS keys.
# ---------------------------------------------------------------------------


def _parse_curriculum(spec: str) -> list[tuple[str, int]]:
    """Parse ``stage1:N1,stage2:N2,...`` into ``[(stage_name, episodes), ...]``."""
    if not spec.strip():
        return []
    stages: list[tuple[str, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Curriculum stage '{chunk}' missing ':'. Expected "
                "`stage:episode_count`."
            )
        name, n_str = chunk.split(":", 1)
        name = name.strip()
        if name not in VARIANT_PRESETS:
            raise ValueError(
                f"Curriculum stage '{name}' unknown. Valid stage names: "
                f"{sorted(VARIANT_PRESETS)}."
            )
        try:
            n_episodes = int(n_str.strip())
        except ValueError as e:
            raise ValueError(
                f"Curriculum stage '{chunk}': episode count '{n_str}' is "
                "not an integer."
            ) from e
        if n_episodes <= 0:
            raise ValueError(
                f"Curriculum stage '{chunk}': episode count must be > 0."
            )
        stages.append((name, n_episodes))
    return stages


def _default_full_curriculum(total_episodes: int) -> list[tuple[str, int]]:
    """Split ``total_episodes`` across diag_gcd → diag_factor → full."""
    base = max(total_episodes // 3, 1)
    return [
        ("diag_gcd", base),
        ("diag_factor", base),
        ("full", max(total_episodes - 2 * base, 1)),
    ]


def _current_stage_at(stages: list[tuple[str, int]], ep: int) -> str:
    cumulative = 0
    for name, n in stages:
        if ep < cumulative + n:
            return name
        cumulative += n
    return stages[-1][0] if stages else ""


def _current_stage_index(stages: list[tuple[str, int]], ep: int) -> int:
    cumulative = 0
    for idx, (_, n) in enumerate(stages):
        if ep < cumulative + n:
            return idx
        cumulative += n
    return len(stages) - 1
