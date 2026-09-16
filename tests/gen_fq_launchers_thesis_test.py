"""Ticket `dsnn-dfw.4` -- the THESIS MATRIX on the launcher generator.

Launchers are generated (`tools/gen_fq_launchers.py`), never hand-edited, so
the owner's rulings of 2026-09-15 and 2026-09-16 are pinned on the generator
rather than on 50 files:

  1. THE MATRIX: five arms (A, B, C, C_popart, condC) x two targets (nn256,
     tlm) x five seeds (250197..250201) = 50 runs, named
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
TARGETS = ("nn256", "tlm")
EPISODES = "1000"
CHECKPOINT_EVERY = "50"
PARETO_DUMP_EVERY = "10"
TAU = "0.90"
LAMBDA_Q = "16"
DUAL_ETA = "2.0"
DUAL_MIN = "12"
DUAL_MAX = "32"
ORDER = "free"

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
    """The 50 runs of the matrix, without the three smoke arms."""
    arms = [a for a in gen.thesis_arms() if not a.get("smoke")]
    assert arms, "the generator emits no thesis arm"
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

def test_the_matrix_is_five_arms_two_targets_five_seeds(gen, matrix):
    assert gen.THESIS_SEEDS == SEEDS
    assert gen.THESIS_ARMS == ARMS
    assert tuple(sorted(gen.THESIS_TARGETS)) == tuple(sorted(TARGETS))
    assert len(matrix) == len(ARMS) * len(TARGETS) * len(SEEDS) == 50
    got = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"])
           for a in matrix}
    want = {(arm, t, s) for arm in ARMS for t in TARGETS for s in SEEDS}
    assert got == want
    # and every one of them is a distinct run name of the ruled shape
    names = [a["name"] for a in matrix]
    assert len(set(names)) == len(names)
    for a in matrix:
        assert a["name"] == (f"{a['thesis_arm']}_{a['thesis_target']}"
                             f"_s{a['thesis_seed']}")
        assert _cli(gen, a)["--name"] == a["name"]
    assert "C_popart_tlm_s250197" in names      # the owner's own example


def test_the_priority_order_is_the_owners(gen):
    order = gen.thesis_submission_order()
    assert len(order) == 50 and len(set(order)) == 50
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
    assert len(held) == 16
    # every held row is an A or B row at a seed other than the first
    for a in held:
        assert a["thesis_arm"] in ("A", "B"), a["name"]
        assert a["thesis_seed"] != SEEDS[0], a["name"]
        text = gen.render(a)
        assert "*** HELD" in text and "ABORT(73)" in text, a["name"]
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
        assert cli["--pareto-dump-every"] == PARETO_DUMP_EVERY, a["name"]
        assert cli["--plan-log"] == "auto", a["name"]
        assert "--auto-stop" in cli and cli["--auto-stop"] is None, a["name"]
        assert cli["--seed"] == a["thesis_seed"], a["name"]
        # the search space: the order is FREE in every arm
        assert cli["--fixed-order"] == ORDER, a["name"]
        assert cli["--approx-profile"] == "all", a["name"]
        assert cli["--approx-add"] == gen.APPROX_ADD == "lossless", a["name"]
        # the reward stack the campaign settled
        assert cli["--cost-form"] == "paired-log", a["name"]
        assert cli["--mem-channel"] == "temp", a["name"]
        assert cli["--quality-metric"] == "grad_cosine", a["name"]
        assert cli["--paired-cost-floor"] == "reference", a["name"]
        assert cli["--rewards"] == "cmp mem acc", a["name"]
        assert cli["--lambda-cmp"] == "1" and cli["--lambda-mem"] == "1"
        assert cli["--discount"] == "1.0" and cli["--gae-lambda"] == "1.0"
        assert "--terminal-rewards-only" in cli, a["name"]
        # the face head at init
        assert cli["--scale-face-head"] == "0.1", a["name"]
        assert cli["--face-logit-clamp"] == "15", a["name"]
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


def test_the_targets_are_the_two_the_owner_named(gen, matrix):
    for a in matrix:
        cli = _cli(gen, a)
        if a["thesis_target"] == "nn256":
            assert cli["--example"] == "NeuralNetwork", a["name"]
            assert cli["--dataset"] == "mnist", a["name"]
            assert a["env"] == {"ALPHAGRAD_NN_HIDDEN": "256"}, a["name"]
            text = gen.render(a)
            assert "export ALPHAGRAD_NN_HIDDEN=256\n" in text, a["name"]
        else:
            assert cli["--example"] == "TransformerLM", a["name"]
            assert cli["--dataset"] == "wikitext2", a["name"]
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
        assert cli["--lag-max"] == DUAL_MAX, a["name"]
        assert cli["--lag-init"] == LAMBDA_Q, a["name"]
        assert cli["--face-none-bias"] == "0", a["name"]


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
    only in the five keys the matrix defines."""
    allowed_between_arms = {
        "--name", "--face-none-bias", "--reward-mode", "--quality-floor",
        "--advantage-norm", "--no-symlog", "--symlog-channels",
        "--preference-conditioned", "--lag-eta", "--lag-init", "--lag-min",
        "--lag-max",
    }
    by_key = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"]): a
              for a in matrix}
    for arm in ARMS:
        for t in TARGETS:
            ref = _cli(gen, by_key[(arm, t, SEEDS[0])])
            for s in SEEDS[1:]:
                cli = _cli(gen, by_key[(arm, t, s)])
                diff = {k for k in set(ref) | set(cli)
                        if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
                assert diff == {"--seed", "--name"}, (arm, t, s, sorted(diff))
    for t in TARGETS:
        ref = _cli(gen, by_key[("C", t, SEEDS[0])])
        for arm in ARMS:
            cli = _cli(gen, by_key[(arm, t, SEEDS[0])])
            diff = {k for k in set(ref) | set(cli)
                    if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
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
    for a in matrix + smoke:
        text = gen.render(a)
        assert a["job"] == f"thesis-{a['node']}", a["name"]
        assert f"#SBATCH -J thesis-{a['node']}\n" in text, a["name"]
        assert "#SBATCH --dependency=singleton\n" in text, a["name"]
        assert f"#SBATCH -w {a['node']}\n" in text, a["name"]
        # the RUN's identity is --name and the log file, never -J
        assert f"#SBATCH -o {gen.CAMPAIGN_RUNS}/{a['name']}_%j.log\n" in text
    # one name per node, shared by every job pinned there
    by_node = {}
    for a in matrix + smoke:
        by_node.setdefault(a["node"], set()).add(a["job"])
    for node, jobs in by_node.items():
        assert jobs == {f"thesis-{node}"}, (node, jobs)
    # and no campaign or wave arm gained a singleton dependency
    for a in gen.ARMS:
        if a.get("thesis"):
            continue
        assert "--dependency=singleton" not in gen.render(a), a["name"]


def test_the_nodes_and_the_actors_per_node_size(gen, matrix, smoke):
    assert gen.THESIS_NODES_ALL == (
        "pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu17", "pgi15-gpu18",
        "pgi15-gpu19", "pgi15-gpu20")
    # the released subset: the owner held gpu17 and gpu19 on 2026-09-16
    assert gen.THESIS_NODES == ("pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18",
                                "pgi15-gpu20")
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
        # a held node must not appear anywhere in a generated launcher
        for held_node in ("pgi15-gpu17", "pgi15-gpu19"):
            assert held_node not in text, (a["name"], held_node)


def test_the_first_block_spreads_over_every_released_node(gen):
    """The node is assigned round-robin over THESIS_NODES IN SUBMISSION
    ORDER, so the authorised block occupies all four queues at once instead
    of stacking behind one of them."""
    block = gen.thesis_block1_arms()
    order = {a["name"]: i for i, a in enumerate(block)}
    assert len(order) == 34
    counts = {}
    for a in block:
        counts[a["node"]] = counts.get(a["node"], 0) + 1
    assert set(counts) == set(gen.THESIS_NODES)
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    # the first four submissions go to four different nodes
    assert len({a["node"] for a in block[:4]}) == 4


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
    assert "--preference-conditioned" not in _cli(gen, first)

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
        ({"seed": "42"}, "is not one of"),
        ({"node": "pgi15-gpu19"}, "released"),
        ({"node": "pgi15-cpu1"}, "released"),
    ):
        with pytest.raises(gen.CampaignRowError) as e:
            gen.thesis_arm(**{**ok, **bad, "name": "throwaway_bad"})
        assert frag in str(e.value), (bad, str(e.value))
    with pytest.raises(gen.CampaignRowError):
        gen.blackwell_gres(6)


def test_the_run_name_helper_refuses_an_unknown_coordinate(gen):
    assert gen.thesis_run_name("C_popart", "tlm", "250197") == \
        "C_popart_tlm_s250197"
    for bad in (("X", "tlm", "250197"), ("C", "snn", "250197"),
                ("C", "tlm", "1")):
        with pytest.raises(gen.CampaignRowError):
            gen.thesis_run_name(*bad)


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


def test_auto_stop_needs_the_checkpoint_the_matrix_gives_it(gen, matrix):
    """common/auto_stop.check_auto_stop_args refuses --auto-stop with
    --checkpoint-every 0, because an auto-stopped run can only be continued
    from a checkpoint.  Every matrix row passes 50, so the refusal never
    fires -- and this is the test that keeps it that way."""
    from alphagrad.approx.common import auto_stop as _auto
    from alphagrad.approx.ppo import make_argparser
    for a in matrix:
        ns = make_argparser().parse_args(gen.cli_tokens(a))
        _auto.check_auto_stop_args(ns)          # raises on a bad pair
        assert _auto.check_points(ns) == (250, 500), a["name"]
