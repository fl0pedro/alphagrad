"""Unit tests for :mod:`alphagrad.approx.common.anti_degeneracy`.

Also serves as a regression for the dual-constraint parsing bug fixed
in Infra 6 — pre-fix, repeated ``--lagrangian-constraint`` flags
overwrote (``nargs="*"``) instead of appending. The PPO run
``j2yl2wn7`` was therefore launched with both ``cosine_sim>=0.8`` and
``cosine_sim<=0.9`` on the CLI but only ``cosine_sim<=0.9`` reached the
trainer.
"""

import argparse

from alphagrad.approx.common.anti_degeneracy import desugar_anti_degeneracy


def test_none_passes_user_constraints_through():
    out, lo, hi = desugar_anti_degeneracy(
        ["cosine_sim>=0.5"],
        anti_degeneracy="none",
        delta=0.01,
        cosine_lower_bound=0.8,
        cosine_upper_bound=0.9,
    )
    assert out == ["cosine_sim>=0.5"]
    assert lo is None and hi is None


def test_delta_ceiling_adds_single_constraint():
    out, lo, hi = desugar_anti_degeneracy(
        [],
        anti_degeneracy="delta_ceiling",
        delta=0.01,
        cosine_lower_bound=0.8,
        cosine_upper_bound=0.9,
    )
    assert out == ["cosine_sim<=0.99"]
    assert lo is None
    assert hi == 0.99


def test_delta_ceiling_sweep_values():
    for delta, expected_high in [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]:
        out, lo, hi = desugar_anti_degeneracy(
            [],
            anti_degeneracy="delta_ceiling",
            delta=delta,
            cosine_lower_bound=0.0,
            cosine_upper_bound=1.0,
        )
        assert out == [f"cosine_sim<={expected_high:g}"]
        assert hi == expected_high


def test_corridor_adds_both_bounds():
    out, lo, hi = desugar_anti_degeneracy(
        [],
        anti_degeneracy="corridor",
        delta=0.01,
        cosine_lower_bound=0.8,
        cosine_upper_bound=0.9,
    )
    assert out == ["cosine_sim>=0.8", "cosine_sim<=0.9"]
    assert lo == 0.8 and hi == 0.9


def test_corridor_with_disabled_floor():
    out, lo, hi = desugar_anti_degeneracy(
        [],
        anti_degeneracy="corridor",
        delta=0.01,
        cosine_lower_bound=0.0,
        cosine_upper_bound=0.9,
    )
    assert out == ["cosine_sim<=0.9"]
    assert lo is None and hi == 0.9


def test_corridor_with_disabled_ceiling():
    out, lo, hi = desugar_anti_degeneracy(
        [],
        anti_degeneracy="corridor",
        delta=0.01,
        cosine_lower_bound=0.8,
        cosine_upper_bound=1.0,
    )
    assert out == ["cosine_sim>=0.8"]
    assert lo == 0.8 and hi is None


def test_ppo_args_lagrangian_constraint_appends_correctly():
    """Regression: ``ppo_args.py`` used to declare ``--lagrangian-constraint``
    with ``nargs="*"`` which caused repeated occurrences to OVERWRITE
    rather than append. Confirm the fixed declaration appends."""
    from alphagrad.approx.ppo_args import make_argparser

    parser = make_argparser()
    args = parser.parse_args(
        [
            "--example", "VmappedNeuralNetwork",
            "--lagrangian-constraint", "cosine_sim>=0.8",
            "--lagrangian-constraint", "cosine_sim<=0.9",
        ]
    )
    assert args.lagrangian_constraint == [
        "cosine_sim>=0.8",
        "cosine_sim<=0.9",
    ]


def test_anti_degeneracy_flag_choices():
    from alphagrad.approx.ppo_args import make_argparser

    parser = make_argparser()
    for choice in ("none", "delta_ceiling", "corridor"):
        args = parser.parse_args(
            ["--example", "VmappedNeuralNetwork", "--anti-degeneracy", choice]
        )
        assert args.anti_degeneracy == choice


def test_anti_degeneracy_flag_default_is_none():
    from alphagrad.approx.ppo_args import make_argparser

    parser = make_argparser()
    args = parser.parse_args(["--example", "VmappedNeuralNetwork"])
    assert args.anti_degeneracy == "none"
    assert args.anti_degeneracy_delta == 0.01


def test_anti_degeneracy_invalid_choice_rejected():
    from alphagrad.approx.ppo_args import make_argparser

    parser = make_argparser()
    try:
        parser.parse_args(
            ["--example", "VmappedNeuralNetwork", "--anti-degeneracy", "garbage"]
        )
    except SystemExit:
        return
    raise AssertionError("argparse should have rejected --anti-degeneracy garbage")


def test_threshold_format_small_delta_no_scientific_notation():
    """δ=1e-4 → 0.9999 should render as a fixed-point string, not 9.999e-01."""
    from alphagrad.approx.common.anti_degeneracy import _format_threshold

    assert _format_threshold(0.9999) == "0.9999"
    assert _format_threshold(0.99) == "0.99"
    assert _format_threshold(0.9) == "0.9"
    assert _format_threshold(0.999999) == "0.999999"
    # Edge: thresholds we'd never use but should still parse.
    assert _format_threshold(0.0) == "0"


def test_threshold_format_round_trips_through_parse():
    """Formatted threshold must parse back to the same float via the
    ``parse_lagrangian_constraints`` path."""
    from alphagrad.approx.common.anti_degeneracy import _format_threshold

    for delta in (0.01, 0.05, 0.1, 0.0001, 0.5):
        s = _format_threshold(1.0 - delta)
        parsed = float(s)
        assert abs(parsed - (1.0 - delta)) < 1e-9, (s, parsed, 1.0 - delta)
