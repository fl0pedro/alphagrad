"""The NN256 pace probe (the orchestrator, 2026-09-27), pinned on
`tools/gen_fq_launchers.py`.  A launcher bakes its arguments in and takes no
episode override, so the pace of the NN256 thesis row is measured by a row of
its own:

  1. ONE ROW, probe3_C_popart_nn256_s250197: released, on pgi15-gpu15 as a
     whole 4-GPU row like the NN256 rows.
  2. IT DIFFERS FROM ITS THESIS ROW, C_popart_nn256_s250197, IN --episodes 3
     AND --wandb offline ALONE (AGENTS.md: probes run offline): every other
     argument, in order, and every exported variable are the thesis row's,
     --name included.
  3. An offline row carries no online round trip (pre-flight layer 3), and
     no other row is offline.
  4. ppo.py's own argparse accepts it.
  5. NOT A MATRIX COORDINATE: block 1, the core rows and the matrix counts do
     not move.
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
_COPY = os.path.join(_ALPHAGRAD, "tools", "thesis_nightly_copy.sh")
#: The stack every launcher here is rendered for: a generation-time input
#: with no default (owner ruling 2026-09-27).
STACK = "/Scratch/assmuth/mrg/test-stack"

# THE ORCHESTRATOR'S NUMBERS, TYPED HERE ON PURPOSE.
NAME = "probe3_C_popart_nn256_s250197"
ROW = "C_popart_nn256_s250197"
ARM, TARGET, SEED = "C_popart", "nn256", "250197"
EPISODES = "3"
NODE = "pgi15-gpu15"
GPUS = 4
_EXPORT = re.compile(r"^\s*export\s.*$", re.M)


@pytest.fixture(scope="module")
def gen():
    # The default node assignment, so the thesis row is where the matrix
    # puts it.
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
def probe(gen):
    (a,) = gen.pace_probe_arms()
    return a


@pytest.fixture(scope="module")
def row(gen):
    (a,) = [a for a in gen.ARMS if a["name"] == ROW]
    return a


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def _bash_n(text: str) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fh:
        fh.write(text)
    try:
        chk = subprocess.run(["bash", "-n", fh.name],
                             capture_output=True, text=True)
        return None if chk.returncode == 0 else chk.stderr
    finally:
        os.unlink(fh.name)


# ------------------------------------------------------------- 1. one row

def test_the_probe_is_one_released_whole_four_gpu_row(gen, probe):
    assert [a["name"] for a in gen.pace_probe_arms()] == [NAME]
    assert gen.PACE_PROBE_NAME == NAME
    assert (probe["thesis_arm"], probe["thesis_target"],
            probe["thesis_seed"]) == (ARM, TARGET, SEED)
    assert probe["node"] == NODE and probe["gpus"] == GPUS
    assert gen.node_gpu_count(NODE) == GPUS       # the whole node
    assert not probe.get("held") and not probe.get("paired_into")
    assert "half" not in probe
    cli = _cli(gen, probe)
    assert cli["--ray-measure"] == str(GPUS - 1)
    assert cli["--gpus"] == "0" and cli["--measure-gpus"] == "1,2,3"
    text = gen.render(probe, STACK)
    assert _bash_n(text) is None
    assert "ABORT(73)" not in text and "ABORT(74)" not in text
    assert f"#SBATCH -w {NODE}\n" in text
    assert ("#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_"
            f"workstation_edition:{GPUS}\n") in text
    assert f"#SBATCH -c {gen.BLACKWELL_CPUS[GPUS]}\n" in text
    assert f"#SBATCH --mem={gen.THESIS_ROW_MEM[GPUS]}\n" in text
    assert f"#SBATCH -J node-{NODE}\n" in text
    assert "#SBATCH --dependency=singleton\n" in text
    assert f"#SBATCH -o {gen.CAMPAIGN_RUNS}/{NAME}_%j.log\n" in text
    assert "THE NN256 PACE PROBE" in text


# ----------------------------------- 2. the thesis row, but for two things

def test_the_probe_differs_from_its_thesis_row_in_episodes_and_wandb_alone(
        gen, probe, row):
    tp, tr = gen.cli_tokens(probe), gen.cli_tokens(row)
    assert len(tp) == len(tr)
    diff = [(i, tr[i], tp[i]) for i in range(len(tp)) if tp[i] != tr[i]]
    assert diff == [
        (tr.index("--episodes") + 1, _cli(gen, row)["--episodes"], EPISODES),
        (tr.index("--wandb") + 1, "online", "offline")], diff
    assert _cli(gen, row)["--episodes"] != EPISODES
    # --name is an argument like any other: the thesis row's
    assert _cli(gen, probe)["--name"] == ROW == _cli(gen, row)["--name"]
    # every exported variable, in order, and what decides them
    assert probe["env"] == row["env"]
    assert _EXPORT.findall(gen.render(probe, STACK)) \
        == _EXPORT.findall(gen.render(row, STACK))
    for k in ("required_flags", "required_flags_file", "jax_cache", "shd_dir",
              "gpus", "mem", "time", "kind", "runtime"):
        assert probe.get(k) == row.get(k), k


# ------------------------------------------------ 3. offline, and alone

def test_an_offline_row_has_no_online_round_trip(gen, probe, row):
    text, ref = gen.render(probe, STACK), gen.render(row, STACK)
    assert (f"  --wandb offline --wandb-entity {gen.WANDB_ENTITY}"
            f" --wandb-project {gen.WANDB_PROJECT}\n") in text
    assert "ABORT(71)" not in text and "FQ_SKIP_WANDB_CHECK" not in text
    assert "# THREE layers, each failing LOUDLY" in text
    assert "ABORT(71)" in ref and "# FOUR layers, each failing LOUDLY" in ref
    assert gen.WANDB == ("--wandb online --wandb-entity dll-streetview"
                         " --wandb-project dsnn-vertex")
    assert [a["name"] for a in gen.ARMS if gen.wandb_cli(a) != gen.WANDB] \
        == [NAME]


def test_the_nightly_copy_does_not_read_an_offline_run():
    """The probe's run directory is offline-run-*; the copy reads run-*."""
    src = open(_COPY).read()
    assert 'for RD in "$SRC_WANDB"/run-*; do' in src


# ------------------------------------------------- 4. ppo.py's argparse

def test_ppo_argparse_accepts_the_probe(gen, probe):
    from alphagrad.approx.ppo import make_argparser
    ns = make_argparser().parse_args(gen.cli_tokens(probe))
    assert ns.episodes == int(EPISODES)
    assert ns.wandb == "offline"
    assert ns.name == ROW and ns.seed == int(SEED)
    assert ns.example == "VmappedNeuralNetwork"
    assert ns.ray_measure == GPUS - 1


# -------------------------------------------- 5. not a matrix coordinate

def test_the_probe_is_not_a_matrix_coordinate(gen, probe):
    assert NAME in {a["name"] for a in gen.thesis_arms()}
    for rows in (gen.thesis_core_arms(), gen.thesis_block1_arms(),
                 gen.thesis_smoke_arms(), gen.overlap_arms()):
        assert NAME not in {a["name"] for a in rows}
    assert ROW in {a["name"] for a in gen.thesis_core_arms()}
    assert len(gen.thesis_core_arms()) == 40
    assert len(gen.thesis_block1_arms()) == 24
