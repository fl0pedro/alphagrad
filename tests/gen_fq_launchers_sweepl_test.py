"""dsnn-dfw epic, 2026-09-20 sweep ruling: the Lagrangian dual sweep on
NN256, arm C, around the rung-1 center.

The stated grid (lag-eta 3 values, lag-max 2, lag-init 3, quality-floor 3,
one factor at a time around a shared center) admits 8 distinct
configurations -- the center plus 2+1+2+2 single-factor perturbations -- not
9: lag-max names only one non-center value.  8 configurations x 3 seeds is
24 rows, not 27.  This file pins the 24 rows the stated value sets actually
admit and does not invent a 9th configuration to reach 27.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")
#: The stack every launcher here is rendered for: a generation-time input
#: with no default (owner ruling 2026-09-27).
STACK = "/Scratch/assmuth/mrg/test-stack"

SEEDS = ("250197", "250198", "250199")
CENTER = {"--lag-eta": "2.0", "--lag-max": "64", "--lag-init": "16",
          "--quality-floor": "0.90"}
#: tag -> the cli overrides that differ from CENTER.
NON_CENTER = {
    "eta_0.5": {"--lag-eta": "0.5"},
    "eta_8.0": {"--lag-eta": "8.0"},
    "lagmax_256": {"--lag-max": "256"},
    "laginit_8": {"--lag-init": "8", "--lag-min": "8"},
    "laginit_40": {"--lag-init": "40"},
    "qfloor_0.85": {"--quality-floor": "0.85"},
    "qfloor_0.95": {"--quality-floor": "0.95"},
}
TAGS = ("center",) + tuple(NON_CENTER)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def test_the_grid_is_eight_configurations_not_nine(gen):
    configs = gen.sweepl_configs()
    assert len(configs) == 8, len(configs)
    tags = [t for t, _ in configs]
    assert len(tags) == len(set(tags)), "the center rendered more than once"
    assert set(tags) == set(TAGS)


def test_the_sweep_is_24_rows_three_seeds_of_eight_configurations(gen):
    rows = gen.sweepl_single_arms()
    assert len(rows) == 24, len(rows)
    got = {(a["sweepl_tag"], a["thesis_seed"]) for a in rows}
    want = {(t, s) for t in TAGS for s in SEEDS}
    assert got == want


def test_every_row_is_arm_c_on_nn256_at_the_rung1_init(gen):
    for a in gen.sweepl_single_arms():
        assert a["thesis_arm"] == "C", a["name"]
        assert a["thesis_target"] == "nn256", a["name"]
        cli = _cli(gen, a)
        assert cli["--face-init-approx-per-plan"] == \
            gen.RUNG1_APPROX_PER_PLAN == "3", a["name"]
        assert cli["--face-init-skips-per-plan"] == \
            gen.RUNG1_SKIPS_PER_PLAN == "0.3", a["name"]
        assert "--face-none-bias" not in cli, a["name"]
        assert cli["--face-entropy-floor"] == \
            gen.THESIS_FACE_ENTROPY_FLOOR == "0.05", a["name"]
        assert cli["--episodes"] == "100", a["name"]
        assert "--auto-stop" not in cli, a["name"]
        assert cli["--checkpoint-every"] == gen.THESIS_CHECKPOINT_EVERY \
            == "50", a["name"]
        assert cli["--plan-log"] == "auto", a["name"]
        assert cli["--reward-mode"] == "lagrangian", a["name"]
        assert a["gpus"] == 4, a["name"]


def test_the_center_row_matches_arm_cs_own_dual_defaults(gen):
    for a in gen.sweepl_single_arms():
        if a["sweepl_tag"] != "center":
            continue
        cli = _cli(gen, a)
        for flag, value in CENTER.items():
            assert cli[flag] == value, (a["name"], flag)
        assert cli["--lag-min"] == gen.DUAL_LAMBDA_MIN == "12", a["name"]


def test_every_non_center_row_moves_exactly_one_factor(gen):
    for a in gen.sweepl_single_arms():
        tag = a["sweepl_tag"]
        if tag == "center":
            continue
        cli = _cli(gen, a)
        want = dict(CENTER)
        want.update(NON_CENTER[tag])
        for flag, value in want.items():
            assert cli[flag] == value, (a["name"], flag)
        if "--lag-min" not in NON_CENTER[tag]:
            assert cli["--lag-min"] == gen.DUAL_LAMBDA_MIN == "12", a["name"]


def test_laginit_8_rows_lower_lag_min_and_say_so(gen):
    for a in gen.sweepl_single_arms():
        if a["sweepl_tag"] != "laginit_8":
            continue
        cli = _cli(gen, a)
        assert cli["--lag-min"] == "8", a["name"]
        assert int(cli["--lag-min"]) <= int(cli["--lag-init"]) \
            <= int(cli["--lag-max"]), a["name"]
        assert "--lag-min 8" in a["purpose"], a["name"]


def test_names_follow_the_ruled_pattern(gen):
    for a in gen.sweepl_single_arms():
        tag, seed = a["sweepl_tag"], a["thesis_seed"]
        want = (f"sweepL_nn256_center_s{seed}" if tag == "center"
                else f"sweepL_nn256_{tag}_s{seed}")
        assert a["name"] == want, a["name"]


def test_the_sweep_is_not_a_matrix_coordinate(gen):
    core = {a["name"] for a in gen.thesis_core_arms()}
    block1 = {a["name"] for a in gen.thesis_block1_arms()}
    for a in gen.sweepl_arms():
        assert a["name"] not in core, a["name"]
        assert a["name"] not in block1, a["name"]


def test_sweep_pairs_are_excluded_from_the_matrixs_own_pair_list(gen):
    matrix_pairs = gen.thesis_pair_arms()
    sweep_pairs = [a for a in gen.sweepl_arms() if a.get("paired")]
    assert sweep_pairs, "no sweep configuration paired on an 8-GPU node"
    matrix_names = {p["name"] for p in matrix_pairs}
    for p in sweep_pairs:
        assert p["name"] not in matrix_names, p["name"]
        assert p["node"] in ("pgi15-gpu19", "pgi15-gpu20"), p["name"]
        assert p["gpus"] == 8, p["name"]
        assert len(p["halves"]) == 2, p["name"]
        seeds = [h["seed"] for h in p["halves"]]
        assert len(set(seeds)) == 2, p["name"]


def test_every_sweep_half_points_at_its_own_pair_field(gen):
    """Never the matrix's `paired_into` -- the two pairing records must not
    cross."""
    for a in gen.sweepl_single_arms():
        if a.get("half") is not None and a.get("sweepl_paired_into"):
            assert "paired_into" not in a, a["name"]


def test_every_row_lands_on_a_permitted_thesis_node(gen):
    for a in gen.sweepl_arms():
        assert a["node"] in gen.THESIS_NODES, a["name"]
        assert a["node"] != "pgi15-gpu17", a["name"]


def test_every_launcher_bash_n_checks(gen):
    for a in gen.sweepl_arms():
        text = gen.render(a, STACK)
        assert gen._bash_n(text) is None, a["name"]
