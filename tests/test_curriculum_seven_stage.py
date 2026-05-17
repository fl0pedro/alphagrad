"""Tests for the 7-stage curriculum, per-episode round-robin rotation,
and the new ``compress_scalar`` + ``quant_smallest_float`` variants.

See ``alphagrad/src/alphagrad/approx/CURRICULUM.md`` for the design.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# VARIANT_PRESETS sanity
# ---------------------------------------------------------------------------

def test_new_variants_registered():
    from alphagrad.approx.variants import (
        VARIANT_PRESETS, SIMPLE_OPERATORS, DIFFICULT_OPERATORS,
    )

    assert "compress_scalar" in VARIANT_PRESETS
    assert "quant_smallest_float" in VARIANT_PRESETS
    # SIMPLE_OPERATORS exactly matches the simple-family triple.
    assert SIMPLE_OPERATORS == (
        "diag_gcd", "compress_scalar", "quant_smallest_float",
    )
    # DIFFICULT_OPERATORS exactly matches the difficult-family triple.
    assert DIFFICULT_OPERATORS == ("diag_factor", "compress", "quantize")


# ---------------------------------------------------------------------------
# compute_ppo_variant_masks — new variants
# ---------------------------------------------------------------------------

def test_compress_scalar_mask():
    from alphagrad.approx.variants import (
        compute_ppo_variant_masks, ppo_full_factor_table,
    )

    m = compute_ppo_variant_masks("compress_scalar", ppo_full_factor_table())
    op = m["op_type_mask"]
    # COMPRESS and END are legal; DIAG and QUANT are blocked.
    assert op.tolist() == [False, True, False, True]
    # factor_mask is unrestricted under COMPRESS.
    assert m["factor_mask"].all()
    # quant_dtype_mask is unrestricted (irrelevant under COMPRESS).
    assert m["quant_dtype_mask"].all()


def test_quant_smallest_float_mask():
    from alphagrad.approx.variants import (
        compute_ppo_variant_masks, ppo_full_factor_table,
    )

    m = compute_ppo_variant_masks(
        "quant_smallest_float", ppo_full_factor_table(),
    )
    op = m["op_type_mask"]
    # QUANT and END are legal; DIAG and COMPRESS blocked.
    assert op.tolist() == [False, False, True, True]
    # quant_dtype_mask has exactly one True entry at index 13 (float4_e2m1fn).
    assert m["quant_dtype_mask"].sum() == 1
    assert m["quant_dtype_mask"][13]


def test_diag_factor_excludes_minus_one():
    """Regression: ``diag_factor`` must NOT include the -1 (gcd auto)
    factor — that's the diag_gcd variant's territory."""
    from alphagrad.approx.variants import (
        compute_ppo_variant_masks, ppo_full_factor_table,
    )

    table = ppo_full_factor_table()
    m = compute_ppo_variant_masks("diag_factor", table)
    minus_one_idx = table.index(-1)
    assert not m["factor_mask"][minus_one_idx]


# ---------------------------------------------------------------------------
# 7-stage curriculum schedule
# ---------------------------------------------------------------------------

def test_seven_stage_sums_to_total_episodes():
    """The seven-stage schedule must always sum exactly to the input
    episode budget (the final stage absorbs any rounding remainder)."""
    from alphagrad.approx.variants import compute_seven_stage_curriculum

    for total in [100, 500, 1000, 1500, 2000, 5000, 10000]:
        sched = compute_seven_stage_curriculum(total, "ppo")
        assert sum(n for _, n in sched) == total


def test_seven_stage_names_and_order():
    from alphagrad.approx.variants import compute_seven_stage_curriculum

    sched = compute_seven_stage_curriculum(1000, "ppo")
    names = [name for name, _ in sched]
    assert names == [
        "ve_only",
        "rot1_simple",
        "rot2_simple",
        "all_simple",
        "rot1_difficult",
        "rot2_difficult",
        "full",
    ]


def test_seven_stage_floor_respected():
    """Per-trainer floors keep early stages from being too tiny."""
    from alphagrad.approx.variants import (
        compute_seven_stage_curriculum, TRAINER_STAGE_FLOOR,
    )

    sched = compute_seven_stage_curriculum(1000, "ppo")
    floor = TRAINER_STAGE_FLOOR["ppo"]
    for name, n in sched[:-1]:  # final stage absorbs remainder
        assert n >= floor, f"stage {name} got {n} < floor {floor}"


def test_seven_stage_mu0_floor_larger_than_ppo():
    """MuZero floor is bigger than PPO's, so early stages get more
    episodes for the same budget."""
    from alphagrad.approx.variants import compute_seven_stage_curriculum

    ppo_sched = compute_seven_stage_curriculum(300, "ppo")
    mu0_sched = compute_seven_stage_curriculum(300, "mu0")
    # ve_only stage: PPO at floor 10, MuZero at floor 20.
    assert ppo_sched[0][1] == 10
    assert mu0_sched[0][1] == 20


def test_seven_stage_too_small_raises():
    """Below the per-trainer minimum, the schedule should reject the
    budget rather than silently emit zero-length stages."""
    from alphagrad.approx.variants import compute_seven_stage_curriculum

    with pytest.raises(ValueError, match="below the minimum"):
        # 7 stages × floor 20 = 140 minimum for MuZero.
        compute_seven_stage_curriculum(100, "mu0")


def test_seven_stage_full_stage_dominates():
    """The 'full' stage should be the largest by a wide margin
    (geometric pacing puts 256/319 = ~80% there)."""
    from alphagrad.approx.variants import compute_seven_stage_curriculum

    sched = compute_seven_stage_curriculum(2000, "ppo")
    full_idx = [i for i, (n, _) in enumerate(sched) if n == "full"][0]
    full_n = sched[full_idx][1]
    others_max = max(n for name, n in sched if name != "full")
    assert full_n > 4 * others_max


# ---------------------------------------------------------------------------
# Per-episode rotation
# ---------------------------------------------------------------------------

def test_rotation_round_robin_simple():
    """rot1_simple cycles through the three simple operators every 3 eps."""
    from alphagrad.approx.variants import rotation_variant_at_episode

    expected = ["diag_gcd", "compress_scalar", "quant_smallest_float"]
    for ep in range(9):
        variant = rotation_variant_at_episode("rot1_simple", ep)
        assert variant == expected[ep % 3]


def test_rotation_round_robin_difficult():
    """rot1_difficult cycles through the three difficult operators."""
    from alphagrad.approx.variants import rotation_variant_at_episode

    expected = ["diag_factor", "compress", "quantize"]
    for ep in range(6):
        variant = rotation_variant_at_episode("rot1_difficult", ep)
        assert variant == expected[ep % 3]


def test_rotation_pairs_simple():
    """rot2_simple cycles through the 3 pairs of simple operators."""
    from alphagrad.approx.variants import rotation_variant_at_episode

    expected = [
        "diag_gcd+compress_scalar",
        "diag_gcd+quant_smallest_float",
        "compress_scalar+quant_smallest_float",
    ]
    for ep in range(9):
        variant = rotation_variant_at_episode("rot2_simple", ep)
        assert variant == expected[ep % 3]


def test_rotation_pairs_difficult():
    from alphagrad.approx.variants import rotation_variant_at_episode

    expected = [
        "diag_factor+compress",
        "diag_factor+quantize",
        "compress+quantize",
    ]
    for ep in range(6):
        variant = rotation_variant_at_episode("rot2_difficult", ep)
        assert variant == expected[ep % 3]


def test_rotation_monolithic_stages_unchanged():
    """Non-rotation stages emit their canonical name regardless of ep."""
    from alphagrad.approx.variants import rotation_variant_at_episode

    for stage in ("ve_only", "all_simple", "full"):
        for ep in range(5):
            assert rotation_variant_at_episode(stage, ep) == stage


# ---------------------------------------------------------------------------
# compute_variant_at_episode — end-to-end episode → variant mapping
# ---------------------------------------------------------------------------

def test_episode_to_variant_within_first_stage():
    from alphagrad.approx.variants import (
        compute_seven_stage_curriculum, compute_variant_at_episode,
    )

    sched = compute_seven_stage_curriculum(1000, "ppo")
    stage, variant, within = compute_variant_at_episode(0, sched)
    assert stage == "ve_only"
    assert variant == "ve_only"
    assert within == 0


def test_episode_to_variant_at_stage_boundary():
    """Episode = sum of first N stage sizes should be the start of stage N+1."""
    from alphagrad.approx.variants import (
        compute_seven_stage_curriculum, compute_variant_at_episode,
    )

    sched = compute_seven_stage_curriculum(1000, "ppo")
    boundary = sched[0][1]  # exit ve_only, enter rot1_simple
    stage, variant, within = compute_variant_at_episode(boundary, sched)
    assert stage == "rot1_simple"
    assert variant == "diag_gcd"
    assert within == 0


def test_episode_to_variant_overflow_clamps_to_last_stage():
    """An ep > total_episodes returns the last stage's last episode."""
    from alphagrad.approx.variants import (
        compute_seven_stage_curriculum, compute_variant_at_episode,
    )

    sched = compute_seven_stage_curriculum(1000, "ppo")
    total = sum(n for _, n in sched)
    stage, variant, _ = compute_variant_at_episode(total + 50, sched)
    assert stage == "full"
    assert variant == "full"


# ---------------------------------------------------------------------------
# compute_union_variant_masks — compound strings
# ---------------------------------------------------------------------------

def test_union_mask_two_simple_operators():
    """A+B union should be the OR of A's and B's masks."""
    from alphagrad.approx.variants import (
        compute_ppo_variant_masks, compute_union_variant_masks,
        ppo_full_factor_table,
    )

    table = ppo_full_factor_table()
    union = compute_union_variant_masks(
        "diag_gcd+compress_scalar", table,
    )
    a = compute_ppo_variant_masks("diag_gcd", table)
    b = compute_ppo_variant_masks("compress_scalar", table)
    np.testing.assert_array_equal(
        union["op_type_mask"], a["op_type_mask"] | b["op_type_mask"],
    )
    np.testing.assert_array_equal(
        union["factor_mask"], a["factor_mask"] | b["factor_mask"],
    )
    np.testing.assert_array_equal(
        union["quant_dtype_mask"], a["quant_dtype_mask"] | b["quant_dtype_mask"],
    )


def test_union_mask_all_simple():
    """``all_simple`` is the union of all three simple operators."""
    from alphagrad.approx.variants import (
        compute_ppo_variant_masks, compute_union_variant_masks,
        ppo_full_factor_table, SIMPLE_OPERATORS,
    )

    table = ppo_full_factor_table()
    union = compute_union_variant_masks("all_simple", table)
    expected_op = compute_ppo_variant_masks(
        SIMPLE_OPERATORS[0], table,
    )["op_type_mask"]
    for v in SIMPLE_OPERATORS[1:]:
        expected_op |= compute_ppo_variant_masks(v, table)["op_type_mask"]
    np.testing.assert_array_equal(union["op_type_mask"], expected_op)
    # Should at minimum include DIAG (from diag_gcd), COMPRESS (from
    # compress_scalar), QUANT (from quant_smallest_float), and END.
    assert union["op_type_mask"].all()


def test_parse_curriculum_accepts_seven_stage_names():
    """Regression: ``_parse_curriculum`` must accept the rotation-stage
    names (``rot1_simple``, ``rot2_difficult``, ``all_simple``, etc.)
    even though they're NOT in ``VARIANT_PRESETS``. Job 45309 crashed
    at startup because the parser rejected the auto-generated 7-stage
    spec."""
    from alphagrad.approx.variants import _parse_curriculum

    spec = (
        "ve_only:31,rot1_simple:63,rot2_simple:125,all_simple:251,"
        "rot1_difficult:502,rot2_difficult:1003,full:8025"
    )
    parsed = _parse_curriculum(spec)
    assert len(parsed) == 7
    names = [n for n, _ in parsed]
    assert names == [
        "ve_only", "rot1_simple", "rot2_simple", "all_simple",
        "rot1_difficult", "rot2_difficult", "full",
    ]
    assert sum(n for _, n in parsed) == 10000


def test_parse_curriculum_rejects_truly_unknown_name():
    """Sanity: completely nonsense names still raise."""
    from alphagrad.approx.variants import _parse_curriculum

    with pytest.raises(ValueError, match="unknown"):
        _parse_curriculum("totally_made_up:50,full:100")


def test_union_mask_single_variant_passthrough():
    """A non-compound name should match compute_ppo_variant_masks exactly."""
    from alphagrad.approx.variants import (
        compute_ppo_variant_masks, compute_union_variant_masks,
        ppo_full_factor_table,
    )

    table = ppo_full_factor_table()
    a = compute_ppo_variant_masks("diag_gcd", table)
    b = compute_union_variant_masks("diag_gcd", table)
    np.testing.assert_array_equal(a["op_type_mask"], b["op_type_mask"])
    np.testing.assert_array_equal(a["factor_mask"], b["factor_mask"])
    np.testing.assert_array_equal(
        a["quant_dtype_mask"], b["quant_dtype_mask"],
    )
