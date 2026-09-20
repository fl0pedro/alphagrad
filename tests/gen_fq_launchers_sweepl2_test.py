"""dsnn-dfw epic, 2026-09-20 sweep round 2 ruling: after round 1's read-out
(no configuration on NN256 arm C left the center seeds' own band within 100
episodes), round 2 asks whether the actor responds to the dual given more
time and a bigger PPO update, not whether the dual's constants matter.

Two finalists (center, cap256 = --lag-max 256 else center) x two update
budgets (b1 = today's --ppo-epochs 1 --minibatches 4, b4 = --ppo-epochs 2
--minibatches 8) x three seeds = 12 rows, 300 episodes, no --auto-stop.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

SEEDS = ("250197", "250198", "250199")
CENTER = {"--lag-eta": "2.0", "--lag-max": "64", "--lag-init": "16",
          "--quality-floor": "0.90"}
DUALS = {"center": {}, "cap256": {"--lag-max": "256"}}
BUDGETS = {"b1": {}, "b4": {"--ppo-epochs": "2", "--minibatches": "8"}}
TAGS = tuple(f"{d}_{b}" for d in DUALS for b in BUDGETS)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def test_the_grid_is_four_configurations(gen):
    configs = gen.sweepl2_configs()
    assert len(configs) == 4, len(configs)
    tags = [t for t, _ in configs]
    assert len(tags) == len(set(tags))
    assert set(tags) == set(TAGS)


def test_the_sweep_is_12_rows_three_seeds_of_four_configurations(gen):
    rows = gen.sweepl2_single_arms()
    assert len(rows) == 12, len(rows)
    got = {(a["sweepl2_tag"], a["thesis_seed"]) for a in rows}
    want = {(t, s) for t in TAGS for s in SEEDS}
    assert got == want


def test_every_row_is_arm_c_on_nn256_at_the_rung1_init(gen):
    for a in gen.sweepl2_single_arms():
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
        assert cli["--episodes"] == "300", a["name"]
        assert "--auto-stop" not in cli, a["name"]
        assert cli["--checkpoint-every"] == gen.THESIS_CHECKPOINT_EVERY \
            == "50", a["name"]
        assert cli["--plan-log"] == "auto", a["name"]
        assert cli["--reward-mode"] == "lagrangian", a["name"]
        assert a["gpus"] == 4, a["name"]


def test_dual_and_budget_flags_match_the_tag(gen):
    for a in gen.sweepl2_single_arms():
        dual, budget = a["sweepl2_tag"].rsplit("_", 1)
        cli = _cli(gen, a)
        want = dict(CENTER)
        want.update(DUALS[dual])
        want.update(BUDGETS[budget])
        for flag, value in want.items():
            assert cli[flag] == value, (a["name"], flag)
        if budget == "b1":
            assert cli["--ppo-epochs"] == "1", a["name"]
            assert cli["--minibatches"] == "4", a["name"]
        assert cli["--lag-min"] == gen.DUAL_LAMBDA_MIN == "12", a["name"]


def test_names_follow_the_ruled_pattern(gen):
    for a in gen.sweepl2_single_arms():
        tag, seed = a["sweepl2_tag"], a["thesis_seed"]
        assert a["name"] == f"sweepL2_nn256_{tag}_s{seed}", a["name"]


def test_the_sweep_is_not_a_matrix_coordinate(gen):
    core = {a["name"] for a in gen.thesis_core_arms()}
    block1 = {a["name"] for a in gen.thesis_block1_arms()}
    for a in gen.sweepl2_arms():
        assert a["name"] not in core, a["name"]
        assert a["name"] not in block1, a["name"]


def test_sweep_pairs_are_excluded_from_the_matrixs_own_pair_list(gen):
    matrix_pairs = gen.thesis_pair_arms()
    sweep_pairs = [a for a in gen.sweepl2_arms() if a.get("paired")]
    assert sweep_pairs, "no sweep round 2 configuration paired on an " \
        "8-GPU node"
    matrix_names = {p["name"] for p in matrix_pairs}
    round1_names = {p["name"] for p in gen.sweepl_arms() if p.get("paired")}
    for p in sweep_pairs:
        assert p["name"] not in matrix_names, p["name"]
        assert p["name"] not in round1_names, p["name"]
        assert p["node"] in ("pgi15-gpu19", "pgi15-gpu20"), p["name"]
        assert p["gpus"] == 8, p["name"]
        assert len(p["halves"]) == 2, p["name"]
        seeds = [h["seed"] for h in p["halves"]]
        assert len(set(seeds)) == 2, p["name"]


def test_every_sweep_half_points_at_its_own_pair_field(gen):
    """Never round 1's `sweepl_paired_into` or the matrix's `paired_into`
    -- the three pairing records must not cross."""
    for a in gen.sweepl2_single_arms():
        if a.get("half") is not None and a.get("sweepl2_paired_into"):
            assert "paired_into" not in a, a["name"]
            assert "sweepl_paired_into" not in a, a["name"]


def test_every_row_lands_on_a_permitted_thesis_node(gen):
    for a in gen.sweepl2_arms():
        assert a["node"] in gen.THESIS_NODES, a["name"]
        assert a["node"] != "pgi15-gpu17", a["name"]


def test_no_other_row_changed(gen):
    """The sweep round 2 marker excludes these 12 rows (plus their pairs)
    from every matrix count the way round 1's marker does; the matrix and
    round 1's own rows are untouched by this file's grid."""
    assert len(gen.thesis_core_arms()) == 50, len(gen.thesis_core_arms())
    assert len(gen.sweepl_single_arms()) == 24, len(gen.sweepl_single_arms())


def test_every_launcher_bash_n_checks(gen):
    for a in gen.sweepl2_arms():
        text = gen.render(a)
        assert gen._bash_n(text) is None, a["name"]
