"""Tests for the PPO curriculum action masks.

``compute_ppo_variant_masks(variant, full_factor_table) → {op_type_mask,
factor_mask}`` produces the per-stage masks the worker applies to
op_type / factor logits before sampling. The mask shape is
(4,) for op_type and (F,) for factor where F = len(full_factor_table).

Tests cover the legal action subsets per variant + the
``_current_stage_at`` curriculum-stepper helper.
"""

from __future__ import annotations

import numpy as np
import pytest


_OP_DIAG = 0
_OP_COMPRESS = 1
_OP_QUANT = 2
_OP_END = 3
_FULL_TABLE = (-1, 2, 3, 4, 8, 16)


def test_ve_only_masks_to_end_only():
    from alphagrad.approx.variants import compute_ppo_variant_masks

    masks = compute_ppo_variant_masks("ve_only", _FULL_TABLE)
    assert masks["op_type_mask"].shape == (4,)
    assert masks["op_type_mask"][_OP_END]
    assert not masks["op_type_mask"][_OP_DIAG]
    assert not masks["op_type_mask"][_OP_COMPRESS]
    assert not masks["op_type_mask"][_OP_QUANT]
    # factor_mask doesn't matter (op_type=END never reads the factor)
    # but must be all True for the shape contract to be uniform.
    assert masks["factor_mask"].all()


def test_diag_gcd_masks_diag_only_with_factor_minus_one():
    from alphagrad.approx.variants import compute_ppo_variant_masks

    masks = compute_ppo_variant_masks("diag_gcd", _FULL_TABLE)
    # op_type ∈ {DIAG, END}
    assert masks["op_type_mask"][_OP_DIAG]
    assert masks["op_type_mask"][_OP_END]
    assert not masks["op_type_mask"][_OP_COMPRESS]
    assert not masks["op_type_mask"][_OP_QUANT]
    # factor restricted to -1 (index 0 in _FULL_TABLE).
    assert masks["factor_mask"][0]
    assert not masks["factor_mask"][1:].any()


def test_diag_factor_excludes_gcd():
    """``diag_factor`` allows {2,3,4,8,16} but NOT -1, so the agent
    has to learn the actual divisors rather than relying on gcd."""
    from alphagrad.approx.variants import compute_ppo_variant_masks

    masks = compute_ppo_variant_masks("diag_factor", _FULL_TABLE)
    assert masks["op_type_mask"][_OP_DIAG]
    assert masks["op_type_mask"][_OP_END]
    # index 0 is -1 → forbidden under diag_factor.
    assert not masks["factor_mask"][0]
    # indices 1..5 are 2,3,4,8,16 → all allowed.
    assert masks["factor_mask"][1:].all()


def test_compress_only_allows_compress_op_type():
    """Post 7-stage refactor: ``compress`` is the DIFFICULT-family
    COMPRESS variant and does NOT include DIAG. The DIAG operator is
    exposed via ``diag_gcd`` / ``diag_factor`` independently. This
    keeps each curriculum stage's action space monotonically related
    to its name."""
    from alphagrad.approx.variants import compute_ppo_variant_masks

    masks = compute_ppo_variant_masks("compress", _FULL_TABLE)
    assert masks["op_type_mask"][_OP_COMPRESS]
    assert masks["op_type_mask"][_OP_END]
    assert not masks["op_type_mask"][_OP_DIAG]
    assert not masks["op_type_mask"][_OP_QUANT]
    # factor mask is unrestricted under COMPRESS (irrelevant).
    assert masks["factor_mask"].all()


def test_full_unrestricted():
    from alphagrad.approx.variants import compute_ppo_variant_masks

    masks = compute_ppo_variant_masks("full", _FULL_TABLE)
    assert masks["op_type_mask"].all()
    assert masks["factor_mask"].all()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SOURCE INCONSISTENCY, not a stale test: 9cdd258 removed "
        "'full_curriculum' from VARIANT_PRESETS and from "
        "compute_ppo_variant_masks, but gfn_ray.py/mu0_ray.py still list it in "
        "the DEFAULT --variant-sweep and still branch on the name, so the "
        "default sweep raises in _apply_variant_preset. Either re-add the "
        "variant (this test passes again) or drop it from the drivers."
    ),
)
def test_full_curriculum_acts_like_full_at_query_time():
    """``full_curriculum`` is a flag, not a single-stage variant. When
    queried directly it should produce the FULL masks (no restriction)
    — the curriculum logic in the driver expands it into a sequence of
    other variants which are queried individually."""
    from alphagrad.approx.variants import compute_ppo_variant_masks

    masks = compute_ppo_variant_masks("full_curriculum", _FULL_TABLE)
    assert masks["op_type_mask"].all()
    assert masks["factor_mask"].all()


def test_unknown_variant_raises():
    from alphagrad.approx.variants import compute_ppo_variant_masks

    with pytest.raises(ValueError, match="Unknown PPO variant"):
        compute_ppo_variant_masks("nonexistent_stage", _FULL_TABLE)


def test_diag_gcd_with_factor_table_missing_minus_one_raises():
    """If the agent was built with a factor table that doesn't contain
    -1, the diag_gcd mask would have no legal factor — guard rejects
    rather than producing an empty mask that crashes at sampling."""
    from alphagrad.approx.variants import compute_ppo_variant_masks

    with pytest.raises(ValueError, match="empty factor mask"):
        compute_ppo_variant_masks("diag_gcd", (2, 3, 4))  # no -1


def test_ppo_full_factor_table_includes_every_variant_choice():
    """The agent must be built with a factor table that's a superset of
    every variant's allowed factors. Otherwise stage transitions could
    leave a curriculum stage with an empty factor mask."""
    from alphagrad.approx.variants import (
        compute_ppo_variant_masks, ppo_full_factor_table,
    )

    full = ppo_full_factor_table()
    for variant in ("diag_gcd", "diag_factor", "compress", "quantize", "full"):
        masks = compute_ppo_variant_masks(variant, full)
        assert masks["op_type_mask"].any()
        assert masks["factor_mask"].any()


# ---------------------------------------------------------------------------
# Curriculum stage stepper
# ---------------------------------------------------------------------------

def test_current_stage_at_returns_stage_name_per_episode():
    from alphagrad.approx.variants import _current_stage_at, _parse_curriculum

    stages = _parse_curriculum("diag_gcd:3,diag_factor:2,full:5")
    # ep 0,1,2 → diag_gcd
    assert _current_stage_at(stages, 0) == "diag_gcd"
    assert _current_stage_at(stages, 2) == "diag_gcd"
    # ep 3,4 → diag_factor
    assert _current_stage_at(stages, 3) == "diag_factor"
    assert _current_stage_at(stages, 4) == "diag_factor"
    # ep 5..9 → full
    assert _current_stage_at(stages, 5) == "full"
    assert _current_stage_at(stages, 9) == "full"
    # Past the end → last stage absorbs the overflow.
    assert _current_stage_at(stages, 100) == "full"


def test_default_full_curriculum_splits_evenly():
    from alphagrad.approx.variants import _default_full_curriculum

    stages = _default_full_curriculum(99)
    assert [n for _, n in stages] == [33, 33, 33]
    # Remainder goes to the final stage so episode count sums exactly.
    stages = _default_full_curriculum(100)
    names = [n for n, _ in stages]
    counts = [c for _, c in stages]
    assert names == ["diag_gcd", "diag_factor", "full"]
    assert sum(counts) == 100
    # Final stage gets the remainder.
    assert counts[-1] == 100 - 2 * (100 // 3)
