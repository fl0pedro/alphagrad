"""The update-overlap test (dsnn-dfw.190; owner rulings 2026-09-25 and
2026-09-27), pinned on `tools/gen_fq_launchers.py`:

  1. FOUR ROWS: arm C_popart on TLM, seeds 250197 and 250198, 100 episodes,
     each seed at --update-overlap 1 and at --update-overlap 0, named
     `ovl<value>_C_popart_tlm_s<seed>`, released, on pgi15-gpu20 and holding
     the whole node the way a TLM thesis row does (8 GPUs, --ray-measure 7).
  2. THE TWO ARMS OF A SEED DIFFER IN --update-overlap ALONE: the command
     lines, the environment and the rendered launchers are the same but for
     that value and the row's own --name.
  3. EVERY OTHER ARGUMENT IS THE MATRIX'S: the rows are the TLM C_popart
     matrix row of their seed with --episodes 100, their own --name, the
     flag, and the node's own size.
  4. ppo.py's preconditions for --update-overlap 1 hold on every row: the deep
     pipeline (--measure-pipeline 1 with --tokenize-where local or
     cpu-actors), one rollout shard, one temporal rule, no preference sweep,
     the jitted update.
  5. ppo.py's own argparse accepts every row (the thesis test's parse check).
  6. NOT A MATRIX COORDINATE: block 1, the core rows and the matrix counts
     do not move.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import tempfile

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")
_PPO = os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "ppo.py")
#: The stack every launcher here is rendered for: a generation-time input
#: with no default (owner ruling 2026-09-27).
STACK = "/Scratch/assmuth/mrg/test-stack"

# THE OWNER'S NUMBERS, TYPED HERE ON PURPOSE.
ARM = "C_popart"
TARGET = "tlm"
SEEDS = ("250197", "250198")
EPISODES = "100"
NODE = "pgi15-gpu20"
GPUS = 8
ACTORS = "7"
VALUES = ("1", "0")
NAMES = ["ovl1_C_popart_tlm_s250197", "ovl0_C_popart_tlm_s250197",
         "ovl1_C_popart_tlm_s250198", "ovl0_C_popart_tlm_s250198"]
#: What follows the node's size rather than the arm (the thesis test's set).
NODE_DERIVED = {"--ray-measure", "--cpu-cores-per-actor",
                "--reserved-driver-cores", "--grad-oracle-host-budget-gb",
                "--measure-gpus"}
_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]*\}")


class _Missing:
    def __repr__(self) -> str:
        return "<absent>"


_MISSING = _Missing()


@pytest.fixture(scope="module")
def gen():
    # The default node assignment: the matrix rows this module compares
    # against are where `thesis_submission_order` puts them.
    saved = {k: os.environ.pop(k, None)
             for k in ("THESIS_PAIRS", "THESIS_TARGET_NODES")}
    try:
        spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    return mod


@pytest.fixture(scope="module")
def rows(gen):
    return {a["name"]: a for a in gen.overlap_arms()}


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def _parse(gen, a):
    from alphagrad.approx.ppo import make_argparser
    toks = [_PLACEHOLDER.sub("/tmp/ckpt", t) for t in gen.cli_tokens(a)]
    try:
        return make_argparser().parse_args(toks)
    except SystemExit as exc:      # argparse exits; say WHICH row
        raise AssertionError(f"{a['name']}: argparse rejected {toks}") from exc


def _bash_n(text: str) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fh:
        fh.write(text)
    try:
        chk = subprocess.run(["bash", "-n", fh.name],
                             capture_output=True, text=True)
        return None if chk.returncode == 0 else chk.stderr
    finally:
        os.unlink(fh.name)


# ------------------------------------------------------------ 1. four rows

def test_the_four_rows_the_owner_named(gen, rows):
    assert [a["name"] for a in gen.overlap_arms()] == NAMES
    assert gen.OVERLAP_SEEDS == SEEDS and gen.OVERLAP_VALUES == VALUES
    for s in SEEDS:
        for v in VALUES:
            a = rows[gen.overlap_run_name(v, s)]
            assert a["name"] == f"ovl{v}_{ARM}_{TARGET}_s{s}"
            assert (a["thesis_arm"], a["thesis_target"], a["thesis_seed"]) \
                == (ARM, TARGET, s)
            cli = _cli(gen, a)
            assert cli["--update-overlap"] == v, a["name"]
            assert cli["--episodes"] == EPISODES, a["name"]
            assert cli["--seed"] == s, a["name"]
            assert cli["--name"] == a["name"], a["name"]
            assert "--update-overlap" in a["required_flags"], a["name"]
    with pytest.raises(gen.CampaignRowError):
        gen.overlap_run_name("2", SEEDS[0])
    with pytest.raises(gen.CampaignRowError):
        gen.overlap_run_name("1", "250199")


def test_the_rows_are_released_and_hold_the_whole_node(gen, rows):
    b = gen.THESIS_CORE_BUDGET[GPUS]
    for name, a in rows.items():
        assert not a.get("held") and not a.get("paired_into"), name
        assert a["node"] == NODE and a["gpus"] == GPUS, name
        assert a["job"] == f"node-{NODE}" and a.get("singleton"), name
        cli = _cli(gen, a)
        assert cli["--ray-measure"] == ACTORS, name
        assert cli["--gpus"] == "0", name
        assert cli["--measure-gpus"] == "1,2,3,4,5,6,7", name
        assert cli["--reserved-driver-cores"] == str(b["trainer"]), name
        assert cli["--cpu-cores-per-actor"] == str(b["per_actor"]), name
        assert cli["--grad-oracle-cores"] == "0", name
        assert cli["--grad-oracle-host-budget-gb"] \
            == gen.THESIS_GRAD_ORACLE_HOST_BUDGET_GB[GPUS], name
        text = gen.render(a, STACK)
        assert _bash_n(text) is None, name
        assert "ABORT(73)" not in text and "ABORT(74)" not in text, name
        assert f"#SBATCH -w {NODE}\n" in text, name
        assert ("#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_"
                f"workstation_edition:{GPUS}\n") in text, name
        assert f"#SBATCH -c {gen.BLACKWELL_CPUS[GPUS]}\n" in text, name
        assert f"#SBATCH --mem={gen.THESIS_ROW_MEM[GPUS]}\n" in text, name
        assert f"#SBATCH -J node-{NODE}\n" in text, name
        assert "#SBATCH --dependency=singleton\n" in text, name
        assert f"#SBATCH -o {gen.CAMPAIGN_RUNS}/{name}_%j.log\n" in text, name
        assert "THE UPDATE-OVERLAP TEST (dsnn-dfw.190)" in text, name
        assert "# REGISTERED PREDICTION" in text, name
        assert "# FALSIFICATION CRITERION:" in text, name


# ------------------------------ 2. the two arms of a seed: the flag alone

def test_the_two_arms_of_a_seed_differ_in_update_overlap_alone(gen, rows):
    for s in SEEDS:
        on = rows[gen.overlap_run_name("1", s)]
        off = rows[gen.overlap_run_name("0", s)]
        c1, c0 = _cli(gen, on), _cli(gen, off)
        diff = {k for k in set(c1) | set(c0)
                if c1.get(k, _MISSING) != c0.get(k, _MISSING)}
        assert diff == {"--update-overlap", "--name"}, (s, sorted(diff))
        assert (c1["--update-overlap"], c0["--update-overlap"]) == ("1", "0")
        # the same tokens in the same order, the value and the name aside
        t1, t0 = gen.cli_tokens(on), gen.cli_tokens(off)
        assert len(t1) == len(t0)
        assert [i for i, (x, y) in enumerate(zip(t1, t0)) if x != y] == [
            t1.index("--name") + 1, t1.index("--update-overlap") + 1]
        # the same environment, the same pre-flight, the same hardware
        assert on["env"] == off["env"], s
        assert on["required_flags"] == off["required_flags"], s
        for k in ("node", "gpus", "mem", "time", "job", "jax_cache",
                  "shd_dir", "purpose", "prediction", "falsifier"):
            assert on.get(k) == off.get(k), (s, k)
        # and the two launchers: one line apart once each row's own name
        # is taken out
        l1 = gen.render(on, STACK).replace(on["name"], "NAME").splitlines()
        l0 = gen.render(off, STACK).replace(off["name"], "NAME").splitlines()
        assert len(l1) == len(l0), s
        assert [(x, y) for x, y in zip(l1, l0) if x != y] == [
            ("  --update-overlap 1", "  --update-overlap 0")], s


# ------------------------------------------ 3. the matrix's TLM C_popart

def test_every_other_argument_is_the_matrix_row_of_the_seed(gen, rows):
    matrix = {a["name"]: a for a in gen.thesis_core_arms()}
    for s in SEEDS:
        m = matrix[gen.thesis_run_name(ARM, TARGET, s)]
        cm = _cli(gen, m)
        for v in VALUES:
            a = rows[gen.overlap_run_name(v, s)]
            ca = _cli(gen, a)
            diff = {k for k in set(ca) | set(cm)
                    if ca.get(k, _MISSING) != cm.get(k, _MISSING)}
            assert diff - NODE_DERIVED == {"--name", "--episodes",
                                           "--update-overlap"}, \
                (a["name"], sorted(diff))
            assert "--update-overlap" not in cm
            assert a["env"] == m["env"], a["name"]
            assert a["required_flags"] == (m["required_flags"]
                                           + ["--update-overlap"]), a["name"]
            assert a["time"] == m["time"] and a["jax_cache"] == m["jax_cache"]
    assert gen.thesis_episodes(TARGET) != EPISODES


# ------------------------------------------- 4. ppo.py's preconditions

def test_every_row_meets_ppos_preconditions_for_the_overlap(gen, rows):
    """ppo.py refuses --update-overlap 1 unless all five hold (the refusal
    list beside `_OVERLAP` in ppo.py).  Checked on the parsed namespace the
    way ppo.py derives each one, on both arms, so the two arms run the same
    pipeline."""
    from alphagrad.approx.common.rsnn_shd import (resolve_temporal_rules,
                                                  temporal_rule_list)
    src = open(_PPO).read()
    for why in ("it needs the deep pipeline: --measure-pipeline ",
                "it needs --rollout-shards 1", "it needs one temporal rule",
                "a preference sweep runs no update",
                "it needs the jitted update"):
        assert why in src, why
    for name, a in rows.items():
        ns = _parse(gen, a)
        # the deep pipeline
        assert ns.measure_pipeline == 1, name
        assert ns.tokenize_where in ("local", "cpu-actors"), name
        # one rollout shard
        assert ns.rollout_shards == 1, name
        # one temporal rule: the graphs the run holds, as ppo.py counts them
        rules = temporal_rule_list(
            resolve_temporal_rules(ns.example, ns.temporal_rule))
        assert len(list(rules) or [None]) == 1, name
        # no preference sweep, the jitted update
        assert not ns.preference_sweep_checkpoint, name
        assert not ns.no_jit, name


# ------------------------------------------------- 5. ppo.py's argparse

def test_ppo_argparse_accepts_every_overlap_row(gen, rows):
    for name, a in rows.items():
        ns = _parse(gen, a)
        cli = _cli(gen, a)
        assert ns.name == name
        assert ns.update_overlap == int(cli["--update-overlap"]), name
        assert ns.episodes == int(EPISODES), name
        assert ns.seed == int(a["thesis_seed"]), name
        assert ns.ray_measure == int(ACTORS), name
        assert ns.example == "VmappedTransformerLM", name
        assert ns.temporal_rule is None, name
        assert ns.fixed_order == gen.THESIS_ORDER, name
        assert ns.checkpoint_every == int(cli["--checkpoint-every"]), name
        assert ns.auto_stop is False, name


# -------------------------------------------- 6. not a matrix coordinate

def test_the_overlap_rows_are_not_matrix_coordinates(gen, rows):
    names = set(rows)
    assert len(names) == 4
    assert names <= {a["name"] for a in gen.thesis_arms()}
    assert not names & {a["name"] for a in gen.thesis_core_arms()}
    assert not names & {a["name"] for a in gen.thesis_block1_arms()}
    assert not names & {a["name"] for a in gen.thesis_smoke_arms()}
    assert len(gen.thesis_core_arms()) == 40
    assert len(gen.thesis_block1_arms()) == 24
