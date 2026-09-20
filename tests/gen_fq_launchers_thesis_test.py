"""Ticket `dsnn-dfw.4` -- the THESIS MATRIX on the launcher generator.

Launchers are generated (`tools/gen_fq_launchers.py`), never hand-edited, so
the owner's rulings of 2026-09-15 and 2026-09-16 are pinned on the generator
rather than on 150 files:

  1. THE CORE MATRIX: five arms (A, B, C, C_popart, condC) x two targets
     (nn256, tlm) x five seeds (250197..250201) = 50 runs, named
     `<arm>_<target>_s<seed>`.
  2. THE FLAGS, per arm: the face-head init bias, the reward form, the
     advantage normalisation and the preference conditioning are the ONLY
     things that differ between arms; everything else is byte-equal.
  3. NO XLA env, and the one per-arm export a thesis launcher may carry is
     the NeuralNetwork target shape, which has no flag.
  4. THE SCHEDULING: one job per run, a node constraint, and
     `--dependency=singleton` on a job name that IS the node.
  5. THE NODES and THE ACTORS: six Blackwell nodes in the matrix, the four
     released to this agent in the generated set; --ray-measure 7 on the
     8-GPU nodes and 3 on the 4-GPU nodes.
  6. THE SMOKE: 20 episodes of arm C on TLM with --checkpoint-every 10 and
     NO auto-stop, its resume twin, and a condC run on NN256.
  7. `thesis_arm` RAISES on a row outside the rulings.
  8. ppo.py's own argparse accepts every thesis command line.
  9. THE RECURRENT BLOCK: --example RSNN_SHD --dataset shd crossed with four
     temporal rules (tbptt, bptt, rtrl, window2), the same five arms and the
     same five seeds = 100 further runs, named `<arm>_rsnn_<rule>_s<seed>`,
     carrying the NN256/TLM flags unchanged and GENERATED BUT HELD.  The
     rule is part of the target key, so a recurrent run is one
     (arm, target, seed) triple like every other row.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

# THE OWNER'S NUMBERS, TYPED HERE ON PURPOSE.  A change to any of them must be
# a deliberate edit in this file as well as in the generator; that is the
# whole point of a pin.
SEEDS = ("250197", "250198", "250199", "250200", "250201")
ARMS = ("A", "B", "C", "C_popart", "condC")
#: The two targets of the core matrix.  The recurrent target's four rules are
#: four FURTHER targets; RSNN_TARGETS below spells them.
TARGETS = ("nn256", "tlm")
#: THE FOUR TEMPORAL RULES of --example RSNN_SHD (owner ruling 2026-09-16).
#: They are ppo.py's `--temporal-rule` choices, which it builds from
#: common/rsnn_shd.TEMPORAL_RULES; section 9 pins that they are the same four.
TEMPORAL_RULES = ("tbptt", "bptt", "rtrl", "window2")
RSNN_TARGETS = tuple(f"rsnn_{r}" for r in TEMPORAL_RULES)
ALL_TARGETS = TARGETS + RSNN_TARGETS
RSNN_EXAMPLE = "RSNN_SHD"
RSNN_DATASET = "shd"
EPISODES = "1000"
CHECKPOINT_EVERY = "50"
PARETO_DUMP_EVERY = "10"
TAU = "0.90"
LAMBDA_Q = "16"
DUAL_ETA = "2.0"
DUAL_MIN = "12"
#: THE BLOCK SETTINGS (owner rulings 2026-09-19): the cap is 64, the floor is
#: byte, the memory channel is watermark, and a FINAL row carries no
#: --auto-stop.  The tuning rows keep --auto-stop; their own test modules pin
#: that.
DUAL_MAX = "64"
PAIRED_COST_FLOOR = "byte"
MEM_CHANNEL = "watermark"
ORDER = "free"
GRAD_ORACLE_CADENCE = "50"

_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]*\}")
_EXPORT = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", re.M)


class _Missing:
    """Sentinel for "this flag is absent", so that an absent flag and a flag
    whose value happens to be None are never confused in a diff."""

    def __repr__(self) -> str:
        return "<absent>"


_MISSING = _Missing()


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def matrix(gen):
    """Every run of the matrix -- the 50 core rows AND the 100 recurrent
    rows -- without the three smoke arms.  What is shared is asserted over
    this whole set, so a recurrent row cannot quietly drift away from the
    NN256 and TLM rows.

    The order-only tuning rows of tickets dsnn-dfw.29 and dsnn-dfw.45 are
    thesis arms too -- they carry the per-node singleton and the target
    shape, so the tests below that sweep EVERY thesis arm must keep reaching
    them -- but they are not matrix coordinates.  They are out of this
    fixture and pinned by tests/gen_fq_launchers_orderonly_test.py and
    tests/gen_fq_launchers_snnsweep_test.py instead.
    """
    arms = [a for a in gen.thesis_arms()
            if not a.get("smoke") and not a.get("orderonly")
            and not a.get("orderonly_rsnn")
            and not a.get("orderonly_final")]
    assert arms, "the generator emits no thesis arm"
    return arms


@pytest.fixture(scope="module")
def core(gen):
    """The 50 NN256/TLM rows."""
    arms = gen.thesis_core_arms()
    assert arms, "the generator emits no core thesis arm"
    return arms


@pytest.fixture(scope="module")
def snn(gen):
    """The 100 recurrent rows (--example RSNN_SHD)."""
    arms = gen.thesis_snn_arms()
    assert arms, "the generator emits no recurrent thesis arm"
    return arms


@pytest.fixture(scope="module")
def smoke(gen):
    return gen.thesis_smoke_arms()


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def _by_name(arms, name):
    return next(a for a in arms if a["name"] == name)


def _bash_n(text: str) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fh:
        fh.write(text)
    try:
        chk = subprocess.run(["bash", "-n", fh.name],
                             capture_output=True, text=True)
        return None if chk.returncode == 0 else chk.stderr
    finally:
        os.unlink(fh.name)


# ------------------------------------------------------------- 1. the matrix

def test_the_core_matrix_is_five_arms_two_targets_five_seeds(gen, core):
    assert gen.THESIS_SEEDS == SEEDS
    assert gen.THESIS_ARMS == ARMS
    assert len(core) == len(ARMS) * len(TARGETS) * len(SEEDS) == 50
    got = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"])
           for a in core}
    want = {(arm, t, s) for arm in ARMS for t in TARGETS for s in SEEDS}
    assert got == want
    # and every one of them is a distinct run name of the ruled shape
    names = [a["name"] for a in core]
    assert len(set(names)) == len(names)
    for a in core:
        assert a["name"] == (f"{a['thesis_arm']}_{a['thesis_target']}"
                             f"_s{a['thesis_seed']}")
        assert _cli(gen, a)["--name"] == a["name"]
    assert "C_popart_tlm_s250197" in names      # the owner's own example


def test_the_whole_matrix_is_the_core_fifty_and_the_recurrent_hundred(
        gen, matrix, core, snn):
    """One naming rule, one (arm, target, seed) triple per row, six targets.

    The recurrent rules are TARGETS and not a fourth coordinate, so the whole
    matrix is still `THESIS_ARMS x THESIS_TARGETS x THESIS_SEEDS` and
    `thesis_run_name` spells every row of it.
    """
    assert tuple(sorted(gen.THESIS_TARGETS)) == tuple(sorted(ALL_TARGETS))
    assert len(gen.THESIS_TARGETS) == 6
    assert len(core) == 50 and len(snn) == 100
    assert len(matrix) == len(core) + len(snn) == 150
    assert len(matrix) == len(ARMS) * len(ALL_TARGETS) * len(SEEDS)
    got = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"])
           for a in matrix}
    want = {(arm, t, s) for arm in ARMS for t in ALL_TARGETS for s in SEEDS}
    assert got == want
    names = [a["name"] for a in matrix]
    assert len(set(names)) == len(names) == 150
    for a in matrix:
        assert a["name"] == gen.thesis_run_name(
            a["thesis_arm"], a["thesis_target"], a["thesis_seed"])
        assert _cli(gen, a)["--name"] == a["name"]


def test_the_priority_order_is_the_owners(gen):
    order = gen.thesis_submission_order()
    assert len(order) == 50 and len(set(order)) == 50
    # the owner's priority list is the CORE matrix; no recurrent row is
    # released to be submitted, so none of them appears here
    assert {o[1] for o in order} == set(TARGETS)
    # 1. C and C_popart, both targets, five seeds
    assert [o[0] for o in order[:20]] == ["C"] * 10 + ["C_popart"] * 10
    assert {o[2] for o in order[:20]} == set(SEEDS)
    # 2. condC, both targets, five seeds
    assert [o[0] for o in order[20:30]] == ["condC"] * 10
    assert {o[1] for o in order[20:30]} == set(TARGETS)
    # 3. A and B at seed 250197 only
    assert sorted(order[30:34]) == sorted(
        [(arm, t, SEEDS[0]) for arm in ("A", "B") for t in TARGETS])
    # 4. the rest is A and B at the other four seeds, and nothing else
    assert {o[0] for o in order[34:]} == {"A", "B"}
    assert {o[2] for o in order[34:]} == set(SEEDS[1:])
    assert gen.THESIS_BLOCK1 == 34


def test_only_the_first_block_is_submittable_the_rest_is_held(gen, matrix):
    block = gen.thesis_block1_arms()
    assert len(block) == gen.THESIS_BLOCK1 == 34
    assert not any(a.get("held") for a in block)
    held = [a for a in matrix if a.get("held")]
    # 16 core rows (A and B at the four later seeds) and every recurrent row
    assert len(held) == 16 + 100 == 116
    for a in held:
        text = gen.render(a)
        assert "*** HELD" in text and "ABORT(73)" in text, a["name"]
    # every held CORE row is an A or B row at a seed other than the first
    core_held = [a for a in held if not a.get("thesis_rule")]
    assert len(core_held) == 16
    for a in core_held:
        assert a["thesis_arm"] in ("A", "B"), a["name"]
        assert a["thesis_seed"] != SEEDS[0], a["name"]
    # the released block is the CORE block and nothing else: no recurrent row
    # is submittable, whatever its arm or seed
    assert not any(a.get("thesis_rule") for a in block)
    # and the first block holds exactly the runs the owner authorised
    got = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"])
           for a in block}
    want = {(arm, t, s) for arm in ("C", "C_popart", "condC")
            for t in TARGETS for s in SEEDS}
    want |= {(arm, t, SEEDS[0]) for arm in ("A", "B") for t in TARGETS}
    assert got == want


def test_every_thesis_arm_renders_to_valid_bash(gen, matrix, smoke):
    for a in matrix + smoke:
        err = _bash_n(gen.render(a))
        assert err is None, (a["name"], err)


# -------------------------------------------------------------- 2. the flags

def test_every_arm_carries_the_shared_thesis_flags(gen, matrix):
    for a in matrix:
        cli = _cli(gen, a)
        # the run
        assert cli["--episodes"] == EPISODES, a["name"]
        assert cli["--checkpoint-every"] == CHECKPOINT_EVERY, a["name"]
        assert cli["--grad-oracle-cadence"] == GRAD_ORACLE_CADENCE, a["name"]
        assert cli["--pareto-dump-every"] == PARETO_DUMP_EVERY, a["name"]
        assert cli["--plan-log"] == "auto", a["name"]
        # A FINAL ROW RUNS ITS FULL THOUSAND EPISODES (owner 2026-09-19):
        # --auto-stop is absent, and the absence is asserted rather than
        # trusted.
        assert "--auto-stop" not in cli, a["name"]
        assert gen.THESIS_FINAL_AUTO_STOP is False
        assert cli["--seed"] == a["thesis_seed"], a["name"]
        # the search space: the order is FREE in every arm
        assert cli["--fixed-order"] == ORDER, a["name"]
        assert cli["--approx-profile"] == "all", a["name"]
        assert cli["--approx-add"] == gen.APPROX_ADD == "lossless", a["name"]
        # the reward stack the campaign settled
        assert cli["--cost-form"] == "paired-log", a["name"]
        assert cli["--mem-channel"] == MEM_CHANNEL, a["name"]
        assert gen.THESIS_MEM_CHANNEL == MEM_CHANNEL
        assert cli["--quality-metric"] == "grad_cosine", a["name"]
        assert cli["--paired-cost-floor"] == PAIRED_COST_FLOOR, a["name"]
        assert gen.THESIS_PAIRED_COST_FLOOR == PAIRED_COST_FLOOR
        assert cli["--rewards"] == "cmp mem acc", a["name"]
        assert cli["--lambda-cmp"] == "1" and cli["--lambda-mem"] == "1"
        assert cli["--discount"] == "1.0" and cli["--gae-lambda"] == "1.0"
        assert "--terminal-rewards-only" in cli, a["name"]
        # the face head at init
        assert cli["--scale-face-head"] == "0.1", a["name"]
        assert cli["--face-logit-clamp"] == "15", a["name"]
        assert cli["--face-entropy-weight"] == "0.05", a["name"]
        assert cli["--face-entropy-floor"] == "0.3", a["name"]
        assert cli["--face-entropy-floor-weight"] == "10.0", a["name"]
        # the measurement protocol
        assert cli["--measure-pipeline"] == "1", a["name"]
        assert cli["--tokenize-where"] == "local", a["name"]
        assert cli["--face-wire-faces"] == "64", a["name"]
        assert cli["--ray-measure-timeout"] == "600", a["name"]
        assert cli["--rollout-shards"] == "1", a["name"]
        # the gate inputs, resolved from THIS arm's order
        assert (cli["--gate-winners-table"]
                == gen.CAMPAIGN_GATE_WINNERS_TABLES[ORDER]), a["name"]
        assert (cli["--gate-offline-contrast"]
                == gen.GATE_OFFLINE_CONTRAST[ORDER]), a["name"]


def test_the_targets_are_the_ones_the_owner_named(gen, matrix):
    for a in matrix:
        cli = _cli(gen, a)
        if a["thesis_target"] == "nn256":
            assert cli["--example"] == "NeuralNetwork", a["name"]
            assert cli["--dataset"] == "mnist", a["name"]
            assert a["env"] == {"ALPHAGRAD_NN_HIDDEN": "256"}, a["name"]
            text = gen.render(a)
            assert "export ALPHAGRAD_NN_HIDDEN=256\n" in text, a["name"]
        elif a["thesis_target"] == "tlm":
            assert cli["--example"] == "TransformerLM", a["name"]
            assert cli["--dataset"] == "wikitext2", a["name"]
            assert a["env"] == {}, a["name"]
            text = gen.render(a)
            assert "ALPHAGRAD_NN_HIDDEN" not in text, a["name"]
        else:
            assert a["thesis_target"] in RSNN_TARGETS, a["name"]
            assert cli["--example"] == RSNN_EXAMPLE, a["name"]
            assert cli["--dataset"] == RSNN_DATASET, a["name"]
            # the recurrent target's shape is module constants of
            # common/rsnn_shd.py, not an environment variable
            assert a["env"] == {}, a["name"]
            text = gen.render(a)
            assert "ALPHAGRAD_NN_HIDDEN" not in text, a["name"]
    # the hidden width really is read from that variable and has no flag
    ex = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "common",
                           "examples.py")).read()
    assert 'environ.get("ALPHAGRAD_NN_HIDDEN"' in ex
    ppo = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx",
                            "ppo.py")).read()
    assert '"--nn-hidden"' not in ppo


def test_arms_a_and_b_are_the_fixed_form_with_no_quality_floor(gen, matrix):
    """Owner ruling 2026-09-16 (night): A and B run the fixed additive form
    at lambda_q 16 with RAW quality and NO floor.  A starts at bias 0 and is
    predicted to collapse; B starts at the identity (bias 4) and is predicted
    to stay there.  The prediction is the point: the floor is NOT added to
    make them work."""
    for a in matrix:
        if a["thesis_arm"] not in ("A", "B"):
            continue
        cli = _cli(gen, a)
        assert cli["--reward-mode"] == "additive", a["name"]
        assert "--quality-floor" not in cli, a["name"]
        assert cli["--lambda-acc"] == LAMBDA_Q, a["name"]
        assert cli["--advantage-norm"] == "none", a["name"]
        assert "--preference-conditioned" not in cli, a["name"]
        assert "--no-symlog" not in cli, a["name"]
        assert cli["--symlog-channels"] == "cost", a["name"]
        for flag in ("--lag-eta", "--lag-init", "--lag-min", "--lag-max"):
            assert flag not in cli, (a["name"], flag)
        assert cli["--face-none-bias"] == ("0" if a["thesis_arm"] == "A"
                                           else "4"), a["name"]
        # and the launcher says so, in the rendered command line
        text = gen.render(a)
        assert "\n  --quality-floor" not in text, a["name"]


def test_the_three_c_arms_are_the_lagrangian_dual(gen, matrix):
    for a in matrix:
        if a["thesis_arm"] not in ("C", "C_popart", "condC"):
            continue
        cli = _cli(gen, a)
        assert cli["--reward-mode"] == "lagrangian", a["name"]
        assert cli["--quality-floor"] == TAU, a["name"]
        assert cli["--lag-eta"] == DUAL_ETA, a["name"]
        assert cli["--lag-min"] == DUAL_MIN, a["name"]
        # THE CAP IS 64 (owner 2026-09-19) so the multiplier can dominate the
        # skip-all plan; --lag-init and --lag-min do not move with it.
        assert cli["--lag-max"] == DUAL_MAX, a["name"]
        assert gen.THESIS_DUAL_LAMBDA_MAX == DUAL_MAX
        assert gen.DUAL_LAMBDA_MIN == DUAL_MIN
        assert cli["--lag-init"] == LAMBDA_Q, a["name"]
        assert cli["--face-none-bias"] == "2", a["name"]


def test_c_is_not_conditioned_and_condc_is(gen, matrix):
    """The one composition nothing has run: ppo.py ties nothing here, but
    `campaign_arm` does (its L rows are all conditioned), so C without the
    conditioning and condC with it is the difference this matrix needs."""
    for a in matrix:
        cli = _cli(gen, a)
        conditioned = "--preference-conditioned" in cli
        assert conditioned == (a["thesis_arm"] == "condC"), a["name"]
    # and ppo.py really does compose them rather than refuse the pair
    ppo = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx",
                            "ppo.py")).read()
    assert "lagrangian + preference-conditioned" in ppo


def test_popart_sets_no_symlog_at_every_site_and_only_on_c_popart(gen, matrix):
    """The recorded trap of ticket .53: PopArt needs --no-symlog, and ppo.py
    refuses a command line whose two symlog sites disagree."""
    for a in matrix:
        cli = _cli(gen, a)
        if a["thesis_arm"] == "C_popart":
            assert cli["--advantage-norm"] == "popart", a["name"]
            assert "--no-symlog" in cli, a["name"]
            assert cli["--symlog-channels"] == "none", a["name"]
        else:
            assert cli["--advantage-norm"] == "none", a["name"]
            assert "--no-symlog" not in cli, a["name"]
            assert cli["--symlog-channels"] == "cost", a["name"]


def test_the_arms_differ_only_where_the_matrix_says_they_do(gen, matrix):
    """Every pair of runs with the same arm and the same target differs ONLY
    in --seed and --name; every pair with the same target and seed differs
    only in the keys the matrix defines.

    --ray-measure is excluded from both comparisons because it is NOT a
    matrix coordinate: it is the node's GPU count minus the trainer's one
    GPU, and the runs are spread round-robin over nodes of two sizes.
    `test_the_nodes_and_the_actors_per_node_size` is what pins it, per arm,
    against that node's own size.
    """
    node_derived = {"--ray-measure", "--cpu-cores-per-actor",
                    "--reserved-driver-cores"}
    allowed_between_arms = {
        "--name", "--face-none-bias", "--reward-mode", "--quality-floor",
        "--advantage-norm", "--no-symlog", "--symlog-channels",
        "--preference-conditioned", "--lag-eta", "--lag-init", "--lag-min",
        "--lag-max",
    }
    by_key = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"]): a
              for a in matrix}
    for arm in ARMS:
        for t in ALL_TARGETS:
            ref = _cli(gen, by_key[(arm, t, SEEDS[0])])
            for s in SEEDS[1:]:
                cli = _cli(gen, by_key[(arm, t, s)])
                diff = {k for k in set(ref) | set(cli)
                        if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
                diff -= node_derived
                assert diff == {"--seed", "--name"}, (arm, t, s, sorted(diff))
    for t in ALL_TARGETS:
        ref = _cli(gen, by_key[("C", t, SEEDS[0])])
        for arm in ARMS:
            cli = _cli(gen, by_key[(arm, t, SEEDS[0])])
            diff = {k for k in set(ref) | set(cli)
                    if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
            diff -= node_derived
            assert diff <= allowed_between_arms, (arm, t, sorted(diff))


# ---------------------------------------------------- 3. the environment

def test_no_thesis_launcher_exports_an_xla_flag_or_a_promoted_var(gen,
                                                                  matrix,
                                                                  smoke):
    jax_cache_exports = {f"export {k}={v}" for k, v in gen.JAX_CACHE_ENV}
    jax_cache_mkdir = f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}"
    for a in matrix + smoke:
        text = gen.render(a)
        for var in gen.PROMOTED_ENV_VARS:
            assert f"export {var}=" not in text, (a["name"], var)
        assert "ALPHAGRAD_FORCE_REV_ORDER" not in text, a["name"]
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            if stripped in jax_cache_exports or stripped == jax_cache_mkdir:
                continue
            assert "XLA_" not in line, (a["name"], line)
            assert "export JAX_" not in line, (a["name"], line)


def test_every_export_in_a_thesis_launcher_is_allowed(gen, matrix, smoke):
    allowed = set(gen.THESIS_ENV_ALLOWED)
    assert allowed == set(gen.CAMPAIGN_ENV_ALLOWED) | {"ALPHAGRAD_NN_HIDDEN"}
    for a in matrix + smoke:
        exported = set(_EXPORT.findall(gen.render(a)))
        assert exported <= allowed, (a["name"], sorted(exported - allowed))
        # the campaign's whole set is always there; the target shape only on
        # the NeuralNetwork arms
        assert set(gen.CAMPAIGN_ENV_ALLOWED) <= exported, a["name"]
        if a.get("thesis_target") == "nn256":
            assert "ALPHAGRAD_NN_HIDDEN" in exported, a["name"]
        else:
            assert "ALPHAGRAD_NN_HIDDEN" not in exported, a["name"]


def test_a_thesis_arm_with_an_unlisted_env_key_is_refused_at_render(gen):
    a = dict(gen.thesis_arms()[0])
    a["env"] = {"ALPHAGRAD_NN_HIDDEN": "256", "XLA_FLAGS": "--nope"}
    with pytest.raises(gen.CampaignRowError) as e:
        gen.render(a)
    assert "XLA_FLAGS" in str(e.value)


# ----------------------------------------------- 4. the singleton scheduling

def test_every_thesis_job_is_a_per_node_singleton(gen, matrix, smoke):
    """THE NAME IS `node-<node>` (ticket dsnn-dfw.65).  Singleton serializes
    only jobs that SHARE a name and several agents submit to these nodes, so
    the matrix's old `thesis-<node>` serialized the matrix against itself and
    let an order-only row hold the same node -- the epilog then kills both."""
    for a in matrix + smoke:
        text = gen.render(a)
        assert a["job"] == f"node-{a['node']}", a["name"]
        assert f"#SBATCH -J node-{a['node']}\n" in text, a["name"]
        assert f"#SBATCH -J thesis-{a['node']}\n" not in text, a["name"]
        assert gen.thesis_job_name(a["node"]) \
            == gen.orderonly_job_name(a["node"]), a["name"]
        assert "#SBATCH --dependency=singleton\n" in text, a["name"]
        assert f"#SBATCH -w {a['node']}\n" in text, a["name"]
        # the RUN's identity is --name and the log file, never -J
        assert f"#SBATCH -o {gen.CAMPAIGN_RUNS}/{a['name']}_%j.log\n" in text
    # one name per node, shared by every job pinned there
    by_node = {}
    for a in matrix + smoke:
        by_node.setdefault(a["node"], set()).add(a["job"])
    for node, jobs in by_node.items():
        assert jobs == {f"node-{node}"}, (node, jobs)
    # and no campaign or wave arm gained a singleton dependency
    for a in gen.ARMS:
        if a.get("thesis"):
            continue
        assert "--dependency=singleton" not in gen.render(a), a["name"]


def test_the_nodes_and_the_actors_per_node_size(gen, matrix, smoke):
    assert gen.THESIS_NODES_ALL == (
        "pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18", "pgi15-gpu20")
    # pgi15-gpu17 has no matched CUDA 12.9 ptxas (dsnn-dfw.69); pgi15-gpu19
    # belongs to another group.  Neither is ever a node source.
    assert "pgi15-gpu17" not in gen.THESIS_NODES_ALL
    assert "pgi15-gpu19" not in gen.THESIS_NODES_ALL
    assert gen.THESIS_NODES == gen.THESIS_NODES_ALL
    assert set(gen.THESIS_NODES) <= set(gen.THESIS_NODES_ALL)
    assert gen.THESIS_RAY_MEASURE == {4: "3", 8: "7"}
    used = {a["node"] for a in matrix}
    assert used == set(gen.THESIS_NODES), sorted(used)
    for a in matrix + smoke:
        gpus = gen.THESIS_NODE_GPUS[a["node"]]
        assert a["gpus"] == gpus, a["name"]
        cli = _cli(gen, a)
        # ONE PPO GPU, every other GPU a measure actor
        assert cli["--ray-measure"] == str(gpus - 1), a["name"]
        assert cli["--ray-measure"] == gen.THESIS_RAY_MEASURE[gpus], a["name"]
        text = gen.render(a)
        assert (f"#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_"
                f"workstation_edition:{gpus}\n") in text, a["name"]
        assert f"#SBATCH -c {gen.BLACKWELL_CPUS[gpus]}\n" in text, a["name"]
        assert f"#SBATCH --mem={gen.BLACKWELL_MEM[gpus]}\n" in text, a["name"]
        assert "#SBATCH -p pgi15\n" in text, a["name"]


def test_the_core_budget_is_the_ruling_and_is_disjoint(gen, matrix, smoke):
    """The node's 64 logical CPUs, per node type (owner ruling Q3,
    2026-09-18): the trainer, the timing actors and the gradient oracle hold
    DISJOINT slices, and no arm carries a budget the node cannot hold."""
    from alphagrad.approx.common.core_budget import check_disjoint
    assert gen.THESIS_CORE_BUDGET_CPUS == 64
    assert gen.THESIS_CORE_BUDGET == {
        8: {"trainer": 8, "per_actor": 2, "oracle": 4},
        4: {"trainer": 8, "per_actor": 2, "oracle": 4},
    }
    for gpus in (4, 8):
        lay = gen.thesis_core_layout(gpus)
        check_disjoint(lay)
        assert len(lay.timing_actors) == int(gen.THESIS_RAY_MEASURE[gpus])
    for a in matrix + smoke:
        gpus = gen.THESIS_NODE_GPUS[a["node"]]
        b = gen.THESIS_CORE_BUDGET[gpus]
        cli = _cli(gen, a)
        assert cli["--reserved-driver-cores"] == str(b["trainer"]), a["name"]
        assert cli["--cpu-cores-per-actor"] == str(b["per_actor"]), a["name"]


def test_the_first_block_spreads_over_every_released_node(gen):
    """The node is assigned round-robin over THESIS_NODES IN SUBMISSION
    ORDER, so the authorised block occupies all five queues at once instead
    of stacking behind one of them."""
    block = gen.thesis_block1_arms()
    order = {a["name"]: i for i, a in enumerate(block)}
    assert len(order) == 34
    counts = {}
    for a in block:
        counts[a["node"]] = counts.get(a["node"], 0) + 1
    assert set(counts) == set(gen.THESIS_NODES)
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    # the first submissions go to a different node each
    n = len(gen.THESIS_NODES)
    assert len({a["node"] for a in block[:n]}) == n


def test_the_campaign_hardware_did_not_move(gen):
    """The thesis section made the gres/cpu/mem lines a function of the arm's
    GPU count.  The campaign arms are 8-GPU rows and must render exactly the
    three lines they always did."""
    assert gen.blackwell_gres(gen.CAMPAIGN_GPUS) == gen.CAMPAIGN_GRES
    assert gen.BLACKWELL_CPUS[8] == gen.CAMPAIGN_CPUS == 128
    assert gen.BLACKWELL_MEM[8] == gen.CAMPAIGN_MEM == "800G"
    for a in gen.campaign_arms():
        text = gen.render(a)
        assert ("#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_"
                "workstation_edition:8\n") in text, a["name"]
        assert "#SBATCH -c 128\n" in text and "#SBATCH --mem=800G\n" in text


# --------------------------------------------------------------- 5. the smoke

def test_the_smoke_is_the_three_runs_the_owner_asked_for(gen, smoke):
    assert [a["name"] for a in smoke] == [
        "smoke_C_tlm", "smoke_C_tlm_resume", "smoke_condC_nn256"]
    for a in smoke:
        assert a["node"] == gen.THESIS_SMOKE_NODE == "pgi15-gpu16", a["name"]
        cli = _cli(gen, a)
        # auto-stop is OFF on every smoke run
        assert "--auto-stop" not in cli, a["name"]
        assert cli["--checkpoint-every"] == "10", a["name"]
        assert cli["--pareto-dump-every"] == PARETO_DUMP_EVERY, a["name"]
        assert cli["--plan-log"] == "auto", a["name"]

    first, resume, cond = smoke
    assert _cli(gen, first)["--episodes"] == "20"
    assert _cli(gen, first)["--example"] == "TransformerLM"
    assert _cli(gen, first)["--reward-mode"] == "lagrangian"
    assert _cli(gen, first)["--grad-oracle-cadence"] == "10"
    assert "--preference-conditioned" not in _cli(gen, first)
    assert _cli(gen, resume)["--grad-oracle-cadence"] == "10"

    assert _cli(gen, cond)["--episodes"] == "5"
    assert _cli(gen, cond)["--example"] == "NeuralNetwork"
    assert _cli(gen, cond)["--dataset"] == "mnist"
    assert "--preference-conditioned" in _cli(gen, cond)
    assert _cli(gen, cond)["--reward-mode"] == "lagrangian"
    assert cond["env"] == {"ALPHAGRAD_NN_HIDDEN": "256"}


def test_the_resume_leg_differs_in_resume_alone(gen, smoke):
    """A resume refuses any command line that differs from the checkpoint's
    in anything but --episodes and --resume (common/checkpoint.py), and
    --name is one of the fields it compares.  So the two legs are generated
    from one call and this test is the proof that they did not drift."""
    first, resume, _cond = smoke
    a, b = _cli(gen, first), _cli(gen, resume)
    diff = {k for k in set(a) | set(b) if a.get(k, _MISSING) != b.get(k, _MISSING)}
    assert diff == {"--resume"}, sorted(diff)
    assert a["--name"] == b["--name"] == "smoke_C_tlm"
    assert resume["name"] == "smoke_C_tlm_resume"     # the FILE differs
    assert _PLACEHOLDER.match(b["--resume"]), b["--resume"]
    text = gen.render(resume)
    assert "THESIS_RESUME" in text
    ck = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "common",
                           "checkpoint.py")).read()
    assert "Only --episodes and --resume may differ." in ck


# ------------------------------------------------------- 6. the pre-flight

def test_the_preflight_greps_the_files_that_define_the_new_flags(gen, matrix,
                                                                 smoke):
    """--checkpoint-every / --resume live in common/checkpoint.py and
    --auto-stop in common/auto_stop.py, not in ppo.py.  A launcher whose
    layer-1 grep read ppo.py alone would ABORT(64) naming a flag that is
    defined."""
    assert "src/alphagrad/approx/common/checkpoint.py" in gen.THESIS_FLAGS_FILES
    assert "src/alphagrad/approx/common/auto_stop.py" in gen.THESIS_FLAGS_FILES
    texts = []
    for rel in gen.THESIS_FLAGS_FILES:
        with open(os.path.join(_ALPHAGRAD, rel)) as fh:
            texts.append(fh.read())
    src = "\n".join(texts)
    for flag in gen.THESIS_REQUIRED_FLAGS:
        assert f'"{flag}"' in src, flag
    for a in matrix + smoke:
        text = gen.render(a)
        assert " ".join(gen.THESIS_FLAGS_FILES) in text, a["name"]
        for flag in ("--auto-stop", "--checkpoint-every", "--resume"):
            assert flag in text, (a["name"], flag)
    # every flag a thesis arm actually passes is defined somewhere it greps
    for a in matrix + smoke:
        for flag in _cli(gen, a):
            assert f'"{flag}"' in src, (a["name"], flag)


# ----------------------------------------------------------- 7. the refusals

def test_thesis_arm_raises_on_a_row_outside_the_rulings(gen):
    ok = dict(arm="C", target="tlm", seed=SEEDS[0], node=gen.THESIS_NODES[0])
    gen.thesis_arm(**{**ok, "name": "throwaway_ok"})
    gen.ARMS.pop()
    for bad, frag in (
        ({"arm": "D"}, "is not one of"),
        ({"target": "snn"}, "is not one of"),
        # the recurrent targets are the four RULED rules and nothing else: a
        # plausible-looking fifth is refused like any other unknown target
        ({"target": "rsnn"}, "is not one of"),
        ({"target": "rsnn_window3"}, "is not one of"),
        ({"seed": "42"}, "is not one of"),
        ({"node": "pgi15-gpu14"}, "permitted"),
        ({"node": "pgi15-cpu1"}, "permitted"),
    ):
        with pytest.raises(gen.CampaignRowError) as e:
            gen.thesis_arm(**{**ok, **bad, "name": "throwaway_bad"})
        assert frag in str(e.value), (bad, str(e.value))
    with pytest.raises(gen.CampaignRowError):
        gen.blackwell_gres(6)


def test_the_run_name_helper_refuses_an_unknown_coordinate(gen):
    assert gen.thesis_run_name("C_popart", "tlm", "250197") == \
        "C_popart_tlm_s250197"
    # THE RECURRENT SPELLING.  The rule is part of the target key, so the one
    # naming helper produces `<arm>_rsnn_<rule>_s<seed>` with no second rule.
    assert gen.thesis_run_name("C_popart", "rsnn_bptt", "250199") == \
        "C_popart_rsnn_bptt_s250199"
    assert gen.thesis_run_name("A", "rsnn_window2", "250201") == \
        "A_rsnn_window2_s250201"
    for bad in (("X", "tlm", "250197"), ("C", "snn", "250197"),
                ("C", "rsnn", "250197"), ("C", "rsnn_window3", "250197"),
                ("C", "tlm", "1")):
        with pytest.raises(gen.CampaignRowError):
            gen.thesis_run_name(*bad)


def test_the_temporal_rule_helper_reads_the_rule_off_the_target(gen):
    assert gen.thesis_temporal_rule("nn256") is None
    assert gen.thesis_temporal_rule("tlm") is None
    for rule in TEMPORAL_RULES:
        assert gen.thesis_temporal_rule(f"rsnn_{rule}") == rule
    with pytest.raises(gen.CampaignRowError):
        gen.thesis_temporal_rule("rsnn_window3")


# --------------------------------------------------------- 8. ppo's argparse

def test_ppo_argparse_accepts_every_thesis_command_line(gen, matrix, smoke):
    from alphagrad.approx.ppo import make_argparser
    for a in matrix + smoke:
        toks = [_PLACEHOLDER.sub("/tmp/ckpt", t) for t in gen.cli_tokens(a)]
        try:
            ns = make_argparser().parse_args(toks)
        except SystemExit as exc:      # argparse exits; say WHICH arm
            raise AssertionError(f"{a['name']}: argparse rejected "
                                 f"{toks}") from exc
        assert ns.name == _cli(gen, a)["--name"]
        assert ns.fixed_order == ORDER
        assert ns.episodes == int(_cli(gen, a)["--episodes"])
        assert ns.checkpoint_every == int(_cli(gen, a)["--checkpoint-every"])
        assert ns.auto_stop == ("--auto-stop" in _cli(gen, a))
        assert ns.temporal_rule == a.get("thesis_rule"), a["name"]


def test_auto_stop_needs_the_checkpoint_the_matrix_gives_it(gen, matrix):
    """common/auto_stop.check_auto_stop_args refuses --auto-stop with
    --checkpoint-every 0, because an auto-stopped run can only be continued
    from a checkpoint.  Every matrix row passes 50, so the refusal never
    fires -- and this is the test that keeps it that way."""
    from alphagrad.approx.common import auto_stop as _auto
    from alphagrad.approx.ppo import make_argparser
    checked = 0
    for a in matrix:
        ns = make_argparser().parse_args(gen.cli_tokens(a))
        _auto.check_auto_stop_args(ns)          # raises on a bad pair
        assert _auto.check_points(ns) == (250, 500), a["name"]
        checked += 1
    assert checked == 150


def test_target_nodes_routing(monkeypatch):
    monkeypatch.setenv("THESIS_TARGET_NODES", "1")
    spec = importlib.util.spec_from_file_location("gen_fq_launchers_tgt", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The order-only tuning rows and the order-only BASELINE (the five
    # Blackwell rows of 2026-09-19) pin their own node by seed; the target
    # switch pins the MATRIX and must not reach either of them.
    matrix = [a for a in mod.thesis_arms()
              if not a.get("smoke") and not a.get("orderonly")
              and not a.get("orderonly_final")]
    for a in matrix:
        if a["thesis_target"] == "tlm":
            assert a["node"] in ("pgi15-gpu20", "pgi15-gpu16"), a["name"]
            expected_gpus = 8 if a["node"] == "pgi15-gpu20" else 4
            expected_actors = "7" if a["node"] == "pgi15-gpu20" else "3"
            assert a["gpus"] == expected_gpus, a["name"]
            assert _cli(mod, a)["--ray-measure"] == expected_actors, a["name"]
        elif a["thesis_target"] == "nn256":
            assert a["node"] in ("pgi15-gpu18", "pgi15-gpu15"), a["name"]
            assert a["gpus"] == 4, a["name"]
            assert _cli(mod, a)["--ray-measure"] == "3", a["name"]


# ------------------------------------------------- 9. the recurrent block

def test_the_recurrent_block_is_four_rules_five_arms_five_seeds(gen, snn):
    """4 x 5 x 5 = 100 rows (owner ruling 2026-09-16)."""
    assert gen.THESIS_TEMPORAL_RULES == TEMPORAL_RULES
    assert gen.THESIS_RSNN_TARGETS == RSNN_TARGETS
    assert len(snn) == len(TEMPORAL_RULES) * len(ARMS) * len(SEEDS) == 100
    got = {(a["thesis_rule"], a["thesis_arm"], a["thesis_seed"]) for a in snn}
    want = {(r, arm, s) for r in TEMPORAL_RULES for arm in ARMS
            for s in SEEDS}
    assert got == want
    # each rule carries the whole five-arm five-seed block
    for rule in TEMPORAL_RULES:
        rows = [a for a in snn if a["thesis_rule"] == rule]
        assert len(rows) == 25, rule
        assert {a["thesis_arm"] for a in rows} == set(ARMS), rule
        assert {a["thesis_seed"] for a in rows} == set(SEEDS), rule
        assert {a["thesis_target"] for a in rows} == {f"rsnn_{rule}"}, rule


def test_every_recurrent_row_is_the_rsnn_shd_target(gen, snn):
    for a in snn:
        cli = _cli(gen, a)
        assert cli["--example"] == RSNN_EXAMPLE == "RSNN_SHD", a["name"]
        assert cli["--dataset"] == RSNN_DATASET == "shd", a["name"]
        assert cli["--temporal-rule"] == a["thesis_rule"], a["name"]
        assert a["thesis_rule"] in TEMPORAL_RULES, a["name"]
        # and the flag really is spelled that way on the rendered command line
        text = gen.render(a)
        assert f"\n  --temporal-rule {a['thesis_rule']}\n" in text, a["name"]
        assert "\n  --example RSNN_SHD\n" in text, a["name"]
        assert "\n  --dataset shd\n" in text, a["name"]
    # all four rules are present, each on 25 rows
    assert {a["thesis_rule"] for a in snn} == set(TEMPORAL_RULES)
    # --dataset shd is a value ppo.py's own argparse offers
    ppo = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx",
                            "ppo.py")).read()
    assert '"--temporal-rule"' in ppo and '"shd"' in ppo


def test_the_recurrent_seeds_are_the_five_the_owner_named(gen, snn):
    assert {a["thesis_seed"] for a in snn} == set(SEEDS)
    for a in snn:
        assert a["thesis_seed"] in SEEDS, a["name"]
        assert _cli(gen, a)["--seed"] == a["thesis_seed"], a["name"]
    # every (rule, arm) pair runs all five, so no seed is short
    for rule in TEMPORAL_RULES:
        for arm in ARMS:
            rows = [a for a in snn
                    if a["thesis_rule"] == rule and a["thesis_arm"] == arm]
            assert {a["thesis_seed"] for a in rows} == set(SEEDS), (rule, arm)


def test_the_recurrent_run_names_are_unique_and_spelled_as_ruled(gen, snn,
                                                                 matrix):
    names = [a["name"] for a in snn]
    assert len(set(names)) == len(names) == 100
    for a in snn:
        assert a["name"] == (f"{a['thesis_arm']}_rsnn_{a['thesis_rule']}"
                             f"_s{a['thesis_seed']}"), a["name"]
        assert _cli(gen, a)["--name"] == a["name"], a["name"]
    assert "C_rsnn_bptt_s250199" in names
    assert "C_popart_rsnn_tbptt_s250197" in names
    assert "condC_rsnn_rtrl_s250201" in names
    # and no recurrent name collides with a core name
    assert len({a["name"] for a in matrix}) == 150


def test_every_recurrent_row_carries_the_nn256_and_tlm_flags_unchanged(
        gen, snn, core):
    """The strongest form of "everything else matches": diff each recurrent
    row against its own arm-and-seed twin on each core target.  The ONLY
    keys allowed to differ are the run name and the three that say which
    target this is.

    --ray-measure is excluded because it is the node's GPU count minus one
    and the rows are spread over nodes of two sizes;
    `test_the_nodes_and_the_actors_per_node_size` pins it per row.
    """
    node_derived = {"--ray-measure"}
    target_keys = {"--name", "--example", "--dataset", "--temporal-rule"}
    by_key = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"]): a
              for a in core}
    for a in snn:
        cli = _cli(gen, a)
        for t in TARGETS:
            ref = _cli(gen, by_key[(a["thesis_arm"], t, a["thesis_seed"])])
            diff = {k for k in set(ref) | set(cli)
                    if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
            diff -= node_derived
            assert diff == target_keys, (a["name"], t, sorted(diff))


def test_no_recurrent_row_carries_an_xla_flag(gen, snn):
    """The same rule the whole matrix runs under, asserted again on the
    recurrent rows on their own: no XLA_*, no JAX_* beyond the shared
    compilation cache, and no per-arm export at all."""
    jax_cache_exports = {f"export {k}={v}" for k, v in gen.JAX_CACHE_ENV}
    jax_cache_mkdir = f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}"
    for a in snn:
        assert a["env"] == {}, a["name"]
        text = gen.render(a)
        assert "XLA_FLAGS" not in text, a["name"]
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            if stripped in jax_cache_exports or stripped == jax_cache_mkdir:
                continue
            assert "XLA_" not in line, (a["name"], line)
            assert "export JAX_" not in line, (a["name"], line)
        exported = set(_EXPORT.findall(text))
        assert exported <= set(gen.THESIS_ENV_ALLOWED), a["name"]
        assert "ALPHAGRAD_NN_HIDDEN" not in exported, a["name"]


def test_the_recurrent_scheduling_matches_the_rest_of_the_matrix(gen, snn):
    """One job per run, one released node, the per-node singleton name, and
    the actor count the node's own size gives."""
    for a in snn:
        assert a["node"] in gen.THESIS_NODES, a["name"]
        gpus = gen.THESIS_NODE_GPUS[a["node"]]
        assert a["gpus"] == gpus, a["name"]
        assert a["job"] == f"node-{a['node']}", a["name"]
        cli = _cli(gen, a)
        assert cli["--ray-measure"] == gen.THESIS_RAY_MEASURE[gpus], a["name"]
        assert cli["--ray-measure"] == str(gpus - 1), a["name"]
        text = gen.render(a)
        assert "#SBATCH --dependency=singleton\n" in text, a["name"]
        assert f"#SBATCH -w {a['node']}\n" in text, a["name"]
    # 100 rows round-robin over the released nodes: every node carries its
    # share and no node carries two more than another. The node count is
    # THESIS_NODES', which grew from four to six when gpu17 and gpu19 were
    # released, so the share is derived and not typed.
    counts = {}
    for a in snn:
        counts[a["node"]] = counts.get(a["node"], 0) + 1
    assert set(counts) == set(gen.THESIS_NODES)
    _n = len(gen.THESIS_NODES)
    assert sum(counts.values()) == 100
    assert set(counts.values()) <= {100 // _n, -(-100 // _n)}, counts


def test_every_recurrent_row_is_generated_and_held_never_submitted(gen, snn):
    """The rows exist as files so the matrix is reviewable; not one of them
    can start.  `main` only ever WRITES and DIFFS launchers -- it has no
    sbatch path at all -- so "not submitted" is the HELD guard in the file."""
    for a in snn:
        assert a.get("held"), a["name"]
        text = gen.render(a)
        assert "*** HELD" in text, a["name"]
        assert f'ABORT(73): {a["name"]} is HELD' in text, a["name"]
        assert "exit 73" in text, a["name"]
    assert not any(a.get("thesis_rule") for a in gen.thesis_block1_arms())
    # THE GENERATOR HAS NO SUBMIT PATH.  It renders, syntax-checks, diffs and
    # writes; the only subprocess it ever starts is `bash -n`.  So "generated
    # but not submitted" is the HELD guard above, and nothing here needs a
    # second mechanism.
    gen_src = open(_GEN).read()
    assert gen_src.count("subprocess.run(") == 1
    assert 'subprocess.run(["bash", "-n"' in gen_src
    assert "os.system" not in gen_src


def test_the_four_rules_are_the_four_the_tree_defines(gen, snn):
    """The generator imports nothing from alphagrad, so the four rules are
    typed in two places.  This is where they are held together: ppo.py builds
    `--temporal-rule`'s choices from common/rsnn_shd.TEMPORAL_RULES, and a
    rule the matrix names that the tree does not define would fail every
    recurrent launcher at its layer-2 pre-flight, by value, after the
    name-only layer-1 grep said yes."""
    from alphagrad.approx.common import rsnn_shd
    from alphagrad.approx.ppo import make_argparser
    assert tuple(rsnn_shd.TEMPORAL_RULES) == TEMPORAL_RULES
    assert gen.THESIS_TEMPORAL_RULES == TEMPORAL_RULES
    for rule in TEMPORAL_RULES:
        ns = make_argparser().parse_args(
            ["--example", RSNN_EXAMPLE, "--temporal-rule", rule])
        assert ns.temporal_rule == rule
    # a rule nobody ruled is refused by the same choices list
    with pytest.raises(SystemExit):
        make_argparser().parse_args(
            ["--example", RSNN_EXAMPLE, "--temporal-rule", "window3"])
    # window2 is the TWO-COPY window and it is asked for by the RULE, not by
    # a second --example: the launcher names RSNN_SHD and rsnn_shd resolves
    # the target the rule builds.
    assert rsnn_shd.target_example(RSNN_EXAMPLE, "window2") == \
        rsnn_shd.RSNN_W2_TARGET
    assert rsnn_shd.target_example(RSNN_EXAMPLE, "tbptt") == RSNN_EXAMPLE
    for a in snn:
        assert _cli(gen, a)["--example"] == RSNN_EXAMPLE, a["name"]
        assert rsnn_shd.resolve_temporal_rule(
            RSNN_EXAMPLE, a["thesis_rule"]) == a["thesis_rule"], a["name"]
