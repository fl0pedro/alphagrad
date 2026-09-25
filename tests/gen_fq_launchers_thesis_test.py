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
  8b. RUNG 1 OF THE LADDER (owner rulings 2026-09-20 and 2026-09-21): arms C
     and C_popart on NN256, five seeds each, state the face-head init as a
     PLAN -- `--face-init-approx-per-plan 3 --face-init-skips-per-plan 0.3`
     -- instead of `--face-none-bias 2`.  The recurrent target's four rules
     take the SAME rung, for C, C_popart AND condC (2026-09-21: the init
     probe reproduces on RSNN_SHD too), but its own approximation count
     dropped to 1 on a later ruling the same day (dsnn-dfw.84 and
     dsnn-dfw.78): `--face-init-approx-per-plan 1
     --face-init-skips-per-plan 0.3`.  No other arm, no other target and no
     other round (the order-only tuning rows are arm C on NN256 too) moves.
  8c. THE LONG CONDITIONED ROWS (owner ruling 2026-09-20): condC on NN256,
     five seeds, renders --episodes 2000.  Every other row keeps 1000.
  8d. CONDC RENDERS ON THE POPART FORM ON NN256, TLM, AND NOW THE FOUR
     RECURRENT TARGETS TOO (owner rulings 2026-09-21): the same three flags
     as arm C_popart -- --advantage-norm popart, --no-symlog,
     --symlog-channels none -- read from C_popart's own row on the SAME
     target, not retyped, in place of the symlog form THESIS_ARM_SPEC
     records for condC.  condC on TLM ALSO takes its own face-head init
     plan (TLM_INIT_APPROX_PER_PLAN / TLM_INIT_SKIPS_PER_PLAN, not rung
     1's).  Arm C itself never moved: it keeps the symlog form everywhere,
     recurrent targets included.
  9. THE RECURRENT BLOCK: --example RSNN_SHD --dataset shd crossed with four
     temporal rules (tbptt, bptt, rtrl, window2), the same five arms and the
     same five seeds = 100 further runs, named `<arm>_rsnn_<rule>_s<seed>`,
     carrying the NN256/TLM flags unchanged (apart from 8b/8d above) and
     GENERATED BUT HELD.  The rule is part of the target key, so a
     recurrent run is one (arm, target, seed) triple like every other row.
     rtrl renders on an 8-GPU node (round-robin by seed, gpu19 then gpu20,
     never paired) because its measurement pipeline stalls at 3 actors
     (owner ruling 2026-09-21); the other three rules keep the 4-GPU
     profile on gpu15/16/18.
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
#: THE LONG CONDITIONED ROWS (owner ruling 2026-09-20): condC on NN256, five
#: seeds, runs twice the episodes.  No other coordinate moves.
LONG_EPISODES = "2000"
LONG_EPISODES_ARMS = ("condC",)
LONG_EPISODES_TARGET = "nn256"
CHECKPOINT_EVERY = "50"


def _is_long_episodes(a) -> bool:
    return (a["thesis_target"] == LONG_EPISODES_TARGET
            and a["thesis_arm"] in LONG_EPISODES_ARMS)
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
#: THE ACTOR UPDATE BUDGET (owner ruling 2026-09-20): a matrix row runs at
#: --ppo-epochs 2 --minibatches 8, not today's --ppo-epochs 1 --minibatches 4
#: (the order-only tuning rows and the smoke keep the old pair; their own
#: test modules pin that).
PPO_EPOCHS = "2"
MINIBATCHES = "8"
#: DUAL-CLIP PPO (owner ruling 2026-09-21, dsnn-dfw.95): every row
#: `thesis_arm` emits caps the negative-advantage branch of the PPO
#: surrogate at c * A with c = 3.  The order-only tuning rows and the sweep
#: sections call `thesis_cli` directly and keep the flag off.
DUAL_CLIP = "3.0"
#: TARGET-KL TRUST REGION BOUND (dsnn-dfw.98): every row
#: `thesis_arm` emits bounds the PPO update at a KL of 0.1 to the
#: rollout policy.  The order-only tuning rows and the sweep
#: sections call `thesis_cli` directly and keep the flag off.
TARGET_KL = "0.1"
#: THE MEASURE ACTORS' EXECUTABLE RETENTION BOUND (dsnn-dfw.99): every
#: row `thesis_arm` emits exports the clear cadence, so the actor drops
#: its in-process JAX caches every 100 measurements instead of holding
#: one executable per plan until the measure GPU refuses.  The
#: order-only tuning rows and the three sweep rounds keep it off.
CACHE_CLEAR_EVERY = "100"
#: What every row `thesis_arm` emits now carries beside its target shape.
CLEAR_ENV = {"ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY": CACHE_CLEAR_EVERY}
#: THE ACTOR PROCESS RECYCLE (dsnn-dfw.99 follow-up, owner ruling
#: 2026-09-22, 08:58): the in-process clear above does not return the
#: executables' device memory to the pool, so every row `thesis_arm` emits
#: also exports the proactive recycle, on the SAME footprint as
#: CACHE_CLEAR_EVERY above.  The order-only tuning rows and the three sweep
#: rounds keep it off too.  The retry of an OOM'd plan is gone (dsnn-dfw.120).
PROACTIVE_RECYCLE_EVERY = "100"
RECYCLE_ENV = {
    "ALPHAGRAD_PROACTIVE_RECYCLE_EVERY": PROACTIVE_RECYCLE_EVERY,
}
#: What every row `thesis_arm` emits now carries beside its target shape:
#: the retention bound and the process recycle together.
MATRIX_ENV = {**CLEAR_ENV, **RECYCLE_ENV}
#: THE FACE-ENTROPY WEIGHT PER FAMILY (owner ruling 2026-09-21, dsnn-dfw.84
#: and dsnn-dfw.78): the recurrent target and TLM render the near-zero
#: bonus; NN256 keeps the campaign's 0.05.
ENTROPY_WEIGHT_LOW = "0.005"
ENTROPY_WEIGHT_NN256 = "0.05"
#: THE FACE-WIRE BUDGET (owner ruling 2026-09-22, dsnn-dfw.104): TLM alone
#: renders the raised budget; every other thesis target keeps the
#: campaign's 64.
FACE_WIRE_FACES = "64"
TLM_FACE_WIRE_FACES = "128"

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
            and not a.get("orderonly_final")
            and not a.get("orderonly_tlm_final")
            and not a.get("paired") and not a.get("sweepl")
              and not a.get("sweepl2") and not a.get("sweepl3")]
    assert arms, "the generator emits no thesis arm"
    return arms


@pytest.fixture(scope="module")
def pairs(gen):
    """The paired launchers (owner ruling 2026-09-20): two NN256 seeds of
    one arm inside ONE sbatch on an 8-GPU node.  Not matrix coordinates --
    their two halves are, and both halves are rows of `core`."""
    return gen.thesis_pair_arms()


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
        assert cli["--episodes"] == (LONG_EPISODES if _is_long_episodes(a)
                                     else EPISODES), a["name"]
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
        # the actor's update budget (owner ruling 2026-09-20, sweep rounds
        # 2-3): today's SHARED_CLI defaults (1, 4) do not meet the
        # Lagrangian constraint; 2, 8 does.
        assert cli["--ppo-epochs"] == PPO_EPOCHS, a["name"]
        assert cli["--minibatches"] == MINIBATCHES, a["name"]
        assert gen.THESIS_PPO_EPOCHS == PPO_EPOCHS
        assert gen.THESIS_MINIBATCHES == MINIBATCHES
        # dsnn-dfw.95 (owner ruling 2026-09-21): the face head's PPO ratio
        # reached 2e4 on the recurrent target and PPO's one-sided clip
        # bounds the ratio only for A > 0, so a violating plan pushed with
        # unbounded weight.  Every matrix row caps the negative branch.
        assert cli["--dual-clip"] == DUAL_CLIP, a["name"]
        assert gen.THESIS_DUAL_CLIP == DUAL_CLIP
        assert cli["--target-kl"] == TARGET_KL, a["name"]
        assert gen.THESIS_TARGET_KL == TARGET_KL
        # the face head at init
        assert cli["--scale-face-head"] == "0.1", a["name"]
        assert cli["--face-logit-clamp"] == "15", a["name"]
        # dsnn-dfw.84 and dsnn-dfw.78 (owner ruling 2026-09-21): the
        # recurrent target and TLM render the near-zero bonus; NN256 keeps
        # the campaign's 0.05.
        want_weight = (ENTROPY_WEIGHT_NN256 if a["thesis_target"] == "nn256"
                       else ENTROPY_WEIGHT_LOW)
        assert cli["--face-entropy-weight"] == want_weight, a["name"]
        assert gen.THESIS_FACE_ENTROPY_WEIGHT_NN256 == ENTROPY_WEIGHT_NN256
        assert gen.THESIS_FACE_ENTROPY_WEIGHT_LOW == ENTROPY_WEIGHT_LOW
        # dsnn-dfw.78: 0.3 is an always-on igniter at identity-like init;
        # matrix rows use 0.05 (SEC-12 finding 2026-08-25).
        assert cli["--face-entropy-floor"] == "0.05", a["name"]
        assert cli["--face-entropy-floor-weight"] == "10.0", a["name"]
        # the measurement protocol
        assert cli["--measure-pipeline"] == "1", a["name"]
        assert cli["--tokenize-where"] == "local", a["name"]
        # dsnn-dfw.104 (owner ruling 2026-09-22): TLM alone renders the
        # raised face-wire budget; every other target keeps the campaign's.
        want_faces = (TLM_FACE_WIRE_FACES if a["thesis_target"] == "tlm"
                      else FACE_WIRE_FACES)
        assert cli["--face-wire-faces"] == want_faces, a["name"]
        assert gen.CAMPAIGN_FACE_WIRE_FACES == FACE_WIRE_FACES
        assert gen.THESIS_TLM_FACE_WIRE_FACES == TLM_FACE_WIRE_FACES
        assert cli["--ray-measure-timeout"] == "300", a["name"]
        assert cli["--rollout-shards"] == "1", a["name"]
        # the gate inputs, resolved from THIS arm's order
        assert (cli["--gate-winners-table"]
                == gen.CAMPAIGN_GATE_WINNERS_TABLES[ORDER]), a["name"]
        assert (cli["--gate-offline-contrast"]
                == gen.GATE_OFFLINE_CONTRAST[ORDER]), a["name"]


def test_dual_clip_reaches_every_row_thesis_arm_emits_and_no_other(gen,
                                                                   matrix,
                                                                   smoke):
    """dsnn-dfw.95.  The cap is on the matrix -- A, B, C, C_popart and condC,
    on every target -- and on the smoke, which has to start the same command
    line the matrix runs.

    It is OFF on the order-only tuning rows and on the three Lagrangian
    sweep rounds, and the absence is asserted rather than trusted.  Those
    are frozen running comparisons: off is bit-identical to the loss before
    the flag existed, which is what keeps them comparable with what they
    already measured, and a launcher that moved under them would invalidate
    the round.
    """
    for a in matrix + smoke:
        cli = _cli(gen, a)
        assert cli["--dual-clip"] == DUAL_CLIP, a["name"]
        assert f"--dual-clip {DUAL_CLIP}" in gen.render(a), a["name"]
    off = (gen.orderonly_arms() + gen.orderonly_rsnn_arms()
           + gen.orderonly_final_arms() + gen.orderonly_tlm_final_arms()
           + gen.sweepl_arms() + gen.sweepl2_arms() + gen.sweepl3_arms())
    assert off
    for a in off:
        assert "--dual-clip" not in _cli(gen, a), a["name"]
        assert "--dual-clip" not in gen.render(a), a["name"]
    # Layer 1 of a launcher greps ppo.py for every flag its OWN command line
    # uses, so the flag is named by the rows that pass it and by no other --
    # a running comparison's launcher does not change for a guard its row
    # does not need (the same rule RUNG1_REQUIRED_FLAGS follows).
    assert gen.DUAL_CLIP_REQUIRED_FLAGS == ["--dual-clip"]
    assert "--dual-clip" not in gen.THESIS_REQUIRED_FLAGS
    for a in matrix + smoke:
        assert "--dual-clip" in a["required_flags"], a["name"]
    for a in off:
        assert "--dual-clip" not in a["required_flags"], a["name"]


def test_target_kl_reaches_every_row_thesis_arm_emits_and_no_other(gen,
                                                                   matrix,
                                                                   smoke):
    """dsnn-dfw.98.  The trust-region bound is on the matrix -- A, B, C,
    C_popart and condC, on every target -- and on the smoke, which has to
    start the same command line the matrix runs.

    It is OFF on the order-only tuning rows and on the three Lagrangian
    sweep rounds, and the absence is asserted rather than trusted, for
    the same reason dual-clip is off there: those are frozen running
    comparisons and a launcher that moved under them would invalidate
    the round.
    """
    for a in matrix + smoke:
        cli = _cli(gen, a)
        assert cli["--target-kl"] == TARGET_KL, a["name"]
        assert f"--target-kl {TARGET_KL}" in gen.render(a), a["name"]
    off = (gen.orderonly_arms() + gen.orderonly_rsnn_arms()
           + gen.orderonly_final_arms() + gen.orderonly_tlm_final_arms()
           + gen.sweepl_arms() + gen.sweepl2_arms() + gen.sweepl3_arms())
    assert off
    for a in off:
        assert "--target-kl" not in _cli(gen, a), a["name"]
        assert "--target-kl" not in gen.render(a), a["name"]
    assert gen.TARGET_KL_REQUIRED_FLAGS == ["--target-kl"]
    assert "--target-kl" not in gen.THESIS_REQUIRED_FLAGS
    for a in matrix + smoke:
        assert "--target-kl" in a["required_flags"], a["name"]
    for a in off:
        assert "--target-kl" not in a["required_flags"], a["name"]


def test_the_targets_are_the_ones_the_owner_named(gen, matrix):
    for a in matrix:
        cli = _cli(gen, a)
        if a["thesis_target"] == "nn256":
            assert cli["--example"] == "VmappedNeuralNetwork", a["name"]
            assert cli["--dataset"] == "mnist", a["name"]
            assert a["env"] == {"ALPHAGRAD_NN_HIDDEN": "256",
                                "ALPHAGRAD_NN_BATCH": "4096",
                                **MATRIX_ENV}, a["name"]
            assert cli["--face-wire-faces"] == FACE_WIRE_FACES, a["name"]
            text = gen.render(a)
            assert "export ALPHAGRAD_NN_HIDDEN=256\n" in text, a["name"]
            assert "export ALPHAGRAD_NN_BATCH=4096\n" in text, a["name"]
        elif a["thesis_target"] == "tlm":
            assert cli["--example"] == "VmappedTransformerLM", a["name"]
            assert cli["--dataset"] == "wikitext2", a["name"]
            assert a["env"] == {"ALPHAGRAD_NN_BATCH": "64",
                                **MATRIX_ENV}, a["name"]
            assert cli["--face-wire-faces"] == TLM_FACE_WIRE_FACES, a["name"]
            text = gen.render(a)
            assert "ALPHAGRAD_NN_HIDDEN" not in text, a["name"]
            assert "export ALPHAGRAD_NN_BATCH=64\n" in text, a["name"]
        else:
            assert a["thesis_target"] in RSNN_TARGETS, a["name"]
            assert cli["--example"] == RSNN_EXAMPLE, a["name"]
            assert cli["--dataset"] == RSNN_DATASET, a["name"]
            # the recurrent target's shape is module constants of
            # common/rsnn_shd.py, not an environment variable
            assert a["env"] == MATRIX_ENV, a["name"]
            assert cli["--face-wire-faces"] == FACE_WIRE_FACES, a["name"]
            text = gen.render(a)
            assert "ALPHAGRAD_NN_HIDDEN" not in text, a["name"]
    # the hidden width really is read from that variable and has no flag
    ex = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "common",
                           "examples.py")).read()
    assert 'environ.get("ALPHAGRAD_NN_HIDDEN"' in ex
    ppo = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx",
                            "ppo.py")).read()
    assert '"--nn-hidden"' not in ppo


def test_a_vmapped_row_carries_its_batch_and_no_frozen_round_moves(
        gen, matrix, smoke):
    var = gen.NN_BATCH_VAR
    assert var == "ALPHAGRAD_NN_BATCH"
    for a in matrix + smoke:
        vmapped = _cli(gen, a)["--example"].startswith("Vmapped")
        assert (var in a["env"]) == vmapped, a["name"]
    for a in _frozen_rounds(gen):
        assert not _cli(gen, a)["--example"].startswith("Vmapped"), a["name"]
        assert var not in (a.get("env") or {}), a["name"]
        assert var not in gen.render(a), a["name"]
    assert var in gen.THESIS_ENV_ALLOWED
    assert var not in gen.CAMPAIGN_ENV_ALLOWED
    ds = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "common",
                           "datasets.py")).read()
    assert 'environ.get("ALPHAGRAD_NN_BATCH"' in ds


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
        if _has_normalized_init(a):
            # rung 1 (or condC-on-TLM's own init) states the plan instead of
            # the bias; the two are mutually exclusive and ppo.py refuses
            # both at once
            assert "--face-none-bias" not in cli, a["name"]
        else:
            assert cli["--face-none-bias"] == "2", a["name"]


# ------------------------------------------------- 2b. rung 1: normalized init

#: The owner's rung-1 numbers, typed here on purpose (2026-09-20 / -21).
RUNG1_ARMS = ("C", "C_popart")
RUNG1_TARGET = "nn256"
RUNG1_A = "3"
RUNG1_KAPPA = "0.3"
#: The recurrent target's rung-1 arm set is WIDER than NN256's: condC moves
#: with C and C_popart there because it has no PopArt-form ruling of its own
#: on that target to hold it back (owner ruling 2026-09-21).
RUNG1_RSNN_ARMS = ("C", "C_popart", "condC")
#: The recurrent target drops to a=1 (owner ruling 2026-09-21, dsnn-dfw.84
#: and dsnn-dfw.78); NN256 keeps a=3 above.  kappa is unchanged for both.
RUNG1_RSNN_A = "1"
#: condC on TLM renders on PopArt (like NN256) but takes its OWN face-init
#: plan, not rung 1's (owner ruling 2026-09-21; the orchestrator sets these
#: from a running probe before the merge).
TLM_INIT_A = "1"
TLM_INIT_KAPPA = "0.1"


def _is_nn256_rung1(a) -> bool:
    """The ORIGINAL rung-1 rows (dsnn-dfw.74): NN256, C and C_popart only.
    These are the rows on the uniform four-GPU NN256 hardware profile."""
    return (a["thesis_target"] == RUNG1_TARGET
            and a["thesis_arm"] in RUNG1_ARMS)


def _is_rsnn_rung1(a) -> bool:
    return (a["thesis_target"] in RSNN_TARGETS
            and a["thesis_arm"] in RUNG1_RSNN_ARMS)


def _is_rung1(a) -> bool:
    """Mirrors `gen.rung1_row`: NN256's rung 1 OR the recurrent target's,
    which is a wider arm set."""
    return _is_nn256_rung1(a) or _is_rsnn_rung1(a)


def _is_condc_tlm_init(a) -> bool:
    return a["thesis_target"] == "tlm" and a["thesis_arm"] == "condC"


def _has_normalized_init(a) -> bool:
    """Any row whose face-head init is a PLAN rather than a bias."""
    return _is_rung1(a) or _is_condc_tlm_init(a)


def test_rung1_reaches_nn256_and_the_recurrent_target_and_nothing_else(
        gen, matrix):
    """THE ROWS.  dsnn-dfw.74 is an NN256 C/C_popart defect, so rung 1 was
    those ten rows -- five seeds each.  The recurrent target's init probe
    (agent rsnnpace, 2026-09-21) reproduced the same finding for its four
    rules, so rung 1 reaches it too, for C, C_popart AND condC -- a wider
    arm set, since the recurrent target has no PopArt ruling to hold condC
    back.  condC on TLM ALSO carries a face-init plan, but its own, not
    rung 1's, so it is counted separately.  No other arm and no other
    target moves, or the ladder's steps are not controlled ones."""
    seen = set()
    for a in matrix:
        cli = _cli(gen, a)
        normalized = "--face-init-approx-per-plan" in cli
        assert normalized == _has_normalized_init(a), a["name"]
        if normalized and not _is_condc_tlm_init(a):
            seen.add((a["thesis_arm"], a["thesis_target"], a["thesis_seed"]))
    want = {(arm, RUNG1_TARGET, s) for arm in RUNG1_ARMS for s in SEEDS}
    want |= {(arm, t, s) for arm in RUNG1_RSNN_ARMS for t in RSNN_TARGETS
             for s in SEEDS}
    assert seen == want


def test_rung1_asks_for_the_approximation_count_and_skip_fraction_per_family(
        gen, matrix):
    """NN256's rung 1 asks for RUNG1_A (3) approximations per plan; the
    recurrent target's own rung 1 dropped to RUNG1_RSNN_A (1) on owner
    ruling 2026-09-21 (dsnn-dfw.84 and dsnn-dfw.78).  kappa
    (RUNG1_KAPPA, 0.3) is unchanged and the same for both families."""
    for a in matrix:
        if not _is_rung1(a):
            continue
        cli = _cli(gen, a)
        want_a = RUNG1_RSNN_A if _is_rsnn_rung1(a) else RUNG1_A
        assert cli["--face-init-approx-per-plan"] == want_a, a["name"]
        assert cli["--face-init-skips-per-plan"] == RUNG1_KAPPA, a["name"]
        assert "--face-none-bias" not in cli, a["name"]
        assert "--face-skip-bias" not in cli, a["name"]
        # and the rendered command line carries both
        text = gen.render(a)
        assert f"--face-init-approx-per-plan {want_a}" in text, a["name"]
        assert f"--face-init-skips-per-plan {RUNG1_KAPPA}" in text, a["name"]
    assert gen.RUNG1_APPROX_PER_PLAN == RUNG1_A
    assert gen.RUNG1_RSNN_APPROX_PER_PLAN == RUNG1_RSNN_A
    assert gen.RUNG1_SKIPS_PER_PLAN == RUNG1_KAPPA


def test_the_conditioned_nn256_rows_run_two_thousand_episodes(gen, matrix):
    """Owner ruling 2026-09-20. The conditioning has to amortise over the
    whole weight span, so those five rows get twice the budget; every other
    coordinate, condC on TLM and on the four recurrent targets included,
    keeps the thousand."""
    long_rows = set()
    for a in matrix:
        cli = _cli(gen, a)
        is_long = cli["--episodes"] == LONG_EPISODES
        assert is_long == _is_long_episodes(a), (a["name"], cli["--episodes"])
        if not is_long:
            assert cli["--episodes"] == EPISODES, a["name"]
        else:
            long_rows.add((a["thesis_arm"], a["thesis_target"],
                           a["thesis_seed"]))
            # the rendered file says so, in the command line and in the header
            text = gen.render(a)
            assert f"--episodes {LONG_EPISODES}" in text, a["name"]
            assert f"THIS ROW RUNS {LONG_EPISODES} EPISODES" in text, a["name"]
    assert long_rows == {(arm, LONG_EPISODES_TARGET, s)
                         for arm in LONG_EPISODES_ARMS for s in SEEDS}
    assert gen.THESIS_EPISODES_LONG == LONG_EPISODES
    assert gen.THESIS_EPISODES == EPISODES


def test_the_long_budget_does_not_reach_a_row_that_names_its_own(gen, smoke):
    """The smoke rows pass an episode count of their own -- 20, its resume
    twin, and 5 for the condC NN256 smoke.  A default that overrode them
    would turn a 5-episode smoke into a 2000-episode run on a node."""
    for a in smoke:
        cli = _cli(gen, a)
        assert cli["--episodes"] != LONG_EPISODES, a["name"]
        assert f"THIS ROW RUNS {LONG_EPISODES} EPISODES" not in gen.render(a)


def test_only_a_normalized_init_launcher_greps_for_the_two_new_flags(gen,
                                                                     matrix):
    """Layer 1 of a launcher greps ppo.py for every flag its own command
    line uses, so a row whose init is a plan (rung 1, or condC-on-TLM's own
    plan) must name the two; and NO other row may, because that list is
    rendered into the file and a running comparison's launcher does not
    change for a guard its row does not need."""
    for a in matrix:
        text = gen.render(a)
        line = [ln for ln in text.splitlines()
                if ln.startswith("for F in --quality-metric")][0]
        greps = set(line[len("for F in "):].rstrip("; do").split())
        for flag in ("--face-init-approx-per-plan",
                     "--face-init-skips-per-plan"):
            assert (flag in greps) == _has_normalized_init(a), (a["name"], flag)


def test_only_a_matrix_coordinate_can_be_a_rung1_row(gen):
    """The order-only tuning rounds are arm C on NN256 too and they build
    their command line with the same `thesis_cli`.  They are a different
    round with its own record, so they must keep --face-none-bias 2; a
    launcher that moved under them would invalidate that comparison."""
    for arm in RUNG1_ARMS:
        assert gen.rung1_row(arm, RUNG1_TARGET, True)
        assert not gen.rung1_row(arm, RUNG1_TARGET, False)
    # and the recurrent target's wider arm set (owner ruling 2026-09-21)
    for rule in TEMPORAL_RULES:
        t = f"rsnn_{rule}"
        for arm in RUNG1_RSNN_ARMS:
            assert gen.rung1_row(arm, t, True)
            assert not gen.rung1_row(arm, t, False)
        assert not gen.rung1_row("A", t, True)
        assert not gen.rung1_row("B", t, True)
    # and the default of thesis_cli is the safe one
    cli = gen.thesis_cli(arm="C", target=RUNG1_TARGET, seed=SEEDS[0],
                         node=gen.THESIS_NODES[0], name="x",
                         episodes="1", checkpoint_every="1", auto_stop=False)
    assert "--face-none-bias" in cli
    assert "--face-init-approx-per-plan" not in cli


def test_rung1_keeps_the_four_gpu_profile_and_the_paired_slots(gen, matrix):
    """The init is the ONLY thing NN256's rung 1 changes.  The uniform NN256
    hardware profile (dsnn-dfw.69) and the paired 8-GPU slots are what make
    the five seeds one distribution; a row that moved off them would not be
    comparable with the seeds beside it.  (The recurrent target's rung-1
    rows are on a DIFFERENT profile, pinned by
    `test_the_recurrent_scheduling_matches_the_rest_of_the_matrix`.)"""
    rows = [a for a in matrix if _is_nn256_rung1(a)]
    assert len(rows) == len(RUNG1_ARMS) * len(SEEDS), len(rows)
    for a in rows:
        assert a["gpus"] == 4, a["name"]
        cli = _cli(gen, a)
        assert cli["--ray-measure"] == "3", a["name"]
    paired_halves = {h["name"] for p in gen.thesis_pair_arms()
                     for h in p["halves"]}
    assert paired_halves, "the paired slots are gone"


# ---------------------------- 2c. condC renders on popart on nn256 and tlm

#: The owner's condC-on-PopArt rulings, typed here on purpose (2026-09-21).
CONDC_POPART_TARGETS = ("nn256", "tlm") + RSNN_TARGETS


def _is_condc_popart(a) -> bool:
    return (a["thesis_target"] in CONDC_POPART_TARGETS
            and a["thesis_arm"] == "condC")


def test_condc_on_all_six_targets_render_on_the_popart_form(gen, matrix):
    """Owner rulings 2026-09-21: condC on NN256, then TLM, then the four
    recurrent targets, all take arm C_popart's own magnitude scaling --
    read from C_popart's own row on that SAME target, not retyped -- in
    place of the symlog form.  All five seeds each, on every one of the
    six targets, singles and pairs both building from this same `cli`."""
    for t in CONDC_POPART_TARGETS:
        c_popart_cli = next(
            _cli(gen, a) for a in matrix
            if a["thesis_arm"] == "C_popart" and a["thesis_target"] == t)
        seen = set()
        for a in matrix:
            if not (_is_condc_popart(a) and a["thesis_target"] == t):
                continue
            cli = _cli(gen, a)
            for flag in ("--advantage-norm", "--no-symlog",
                        "--symlog-channels"):
                assert cli.get(flag, _MISSING) == \
                    c_popart_cli.get(flag, _MISSING), (a["name"], flag)
            assert cli["--advantage-norm"] == "popart", a["name"]
            assert "--no-symlog" in cli, a["name"]
            assert cli["--symlog-channels"] == "none", a["name"]
            seen.add(a["thesis_seed"])
        assert seen == set(SEEDS), t


def test_condc_on_tlm_takes_its_own_face_init_plan(gen, matrix):
    """condC on TLM renders on PopArt like condC on NN256, but its
    face-head init is its OWN plan (TLM_INIT_APPROX_PER_PLAN /
    TLM_INIT_SKIPS_PER_PLAN), not rung 1's 3/0.3 -- rung 1 was probed on
    NN256 alone (owner ruling 2026-09-21, orchestrator sets the final
    numbers from a running probe before the merge)."""
    assert gen.TLM_INIT_APPROX_PER_PLAN == TLM_INIT_A
    assert gen.TLM_INIT_SKIPS_PER_PLAN == TLM_INIT_KAPPA
    seen = set()
    for a in matrix:
        if not (a["thesis_target"] == "tlm" and a["thesis_arm"] == "condC"):
            continue
        cli = _cli(gen, a)
        assert cli["--face-init-approx-per-plan"] == TLM_INIT_A, a["name"]
        assert cli["--face-init-skips-per-plan"] == TLM_INIT_KAPPA, a["name"]
        assert "--face-none-bias" not in cli, a["name"]
        text = gen.render(a)
        assert f"--face-init-approx-per-plan {TLM_INIT_A}" in text, a["name"]
        assert f"--face-init-skips-per-plan {TLM_INIT_KAPPA}" in text, \
            a["name"]
        seen.add(a["thesis_seed"])
    assert seen == set(SEEDS)


def test_c_on_the_recurrent_targets_keeps_the_symlog_form(gen, matrix):
    """condC now renders on PopArt everywhere (owner ruling 2026-09-21
    reaches the recurrent targets too), but arm C itself never moved: it
    keeps the symlog form on every target, recurrent included."""
    seen = set()
    for a in matrix:
        if a["thesis_arm"] != "C":
            continue
        cli = _cli(gen, a)
        assert cli["--advantage-norm"] == "none", a["name"]
        assert "--no-symlog" not in cli, a["name"]
        assert cli["--symlog-channels"] == "cost", a["name"]
        if a["thesis_target"] in RSNN_TARGETS:
            seen.add(a["thesis_target"])
    assert seen == set(RSNN_TARGETS)


def test_only_a_matrix_coordinate_can_be_a_condc_popart_row(gen):
    """Mirrors `test_only_a_matrix_coordinate_can_be_a_rung1_row`: the
    order-only preference row (dsnn-dfw.29/.45's `orderonly_arm`/
    `orderonly_rsnn_arm`, `pref=True`) is condC on NN256 too and builds its
    command line with the same `thesis_cli`, but it is a different round
    (quality inert, --approx-profile none) whose record reads "unchanged
    from the matrix"; `matrix_row=False`, thesis_cli's own default,
    protects it."""
    for t in CONDC_POPART_TARGETS:
        assert gen.condc_popart_row("condC", t, True)
        assert not gen.condc_popart_row("condC", t, False)
        assert not gen.condc_popart_row("C", t, True)
    # and the default of thesis_cli is the safe one
    cli = gen.thesis_cli(arm="condC", target="nn256",
                         seed=SEEDS[0], node=gen.THESIS_NODES[0], name="x",
                         episodes="1", checkpoint_every="1", auto_stop=False)
    assert cli["--advantage-norm"] == "none"
    assert "--no-symlog" not in cli


def test_only_a_matrix_coordinate_can_be_a_condc_tlm_init_row(gen):
    assert gen.condc_tlm_init_row("condC", "tlm", True)
    assert not gen.condc_tlm_init_row("condC", "tlm", False)
    assert not gen.condc_tlm_init_row("C", "tlm", True)
    assert not gen.condc_tlm_init_row("condC", "nn256", True)
    # and the default of thesis_cli is the safe one
    cli = gen.thesis_cli(arm="condC", target="tlm", seed=SEEDS[0],
                         node=gen.THESIS_NODES[0], name="x",
                         episodes="1", checkpoint_every="1", auto_stop=False)
    assert "--face-none-bias" in cli
    assert "--face-init-approx-per-plan" not in cli


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


def test_popart_sets_no_symlog_at_every_site_on_c_popart_and_condc_popart(
        gen, matrix):
    """The recorded trap of ticket .53: PopArt needs --no-symlog, and ppo.py
    refuses a command line whose two symlog sites disagree.  condC on NN256
    and on TLM render on the PopArt form too (owner rulings 2026-09-21,
    section 8d, `test_condc_on_nn256_and_tlm_render_on_the_popart_form`
    pins the detail); every other row keeps the symlog form."""
    for a in matrix:
        cli = _cli(gen, a)
        if a["thesis_arm"] == "C_popart" or _is_condc_popart(a):
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
        # rung 1 (owner 2026-09-20): the face-head init of the NN256 C rows
        # is stated as a plan, not as a bias.  It is the same coordinate as
        # --face-none-bias, expressed in the other of the two ways.
        "--face-init-approx-per-plan", "--face-init-skips-per-plan",
        # and the conditioned NN256 rows run twice the episodes (owner
        # 2026-09-20); `test_the_conditioned_nn256_rows_run_two_thousand_
        # episodes` is what pins which rows that is.
        "--episodes",
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
    assert allowed == (set(gen.CAMPAIGN_ENV_ALLOWED)
                       | {"ALPHAGRAD_NN_HIDDEN", "ALPHAGRAD_NN_BATCH"}
                       | set(MATRIX_ENV))
    for a in matrix + smoke:
        exported = set(_EXPORT.findall(gen.render(a)))
        assert exported <= allowed, (a["name"], sorted(exported - allowed))
        # the campaign's whole set is always there but the compile cache,
        # which a free-order row does not export (owner ruling 2026-09-23);
        # the target shape only on the NeuralNetwork arms
        assert (set(gen.CAMPAIGN_ENV_ALLOWED)
                - {k for k, _ in gen.JAX_CACHE_ENV}) <= exported, a["name"]
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
        "pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18", "pgi15-gpu19",
        "pgi15-gpu20")
    # pgi15-gpu17 has no matched CUDA 12.9 ptxas (dsnn-dfw.69) and is never a
    # node source.  pgi15-gpu19 came back to us on 2026-09-20.
    assert "pgi15-gpu17" not in gen.THESIS_NODES_ALL
    assert gen.THESIS_NODES == gen.THESIS_NODES_ALL
    assert set(gen.THESIS_NODES) <= set(gen.THESIS_NODES_ALL)
    assert gen.THESIS_RAY_MEASURE == {4: "3", 8: "7"}
    used = {a["node"] for a in matrix}
    assert used == set(gen.THESIS_NODES), sorted(used)
    for a in matrix + smoke:
        # THE ROW'S PROFILE, not the node's size: an NN256 row is four GPUs
        # on every Blackwell node (owner ruling 2026-09-20).
        gpus = gen.thesis_row_gpus(a["thesis_target"], a["node"])
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
        gpus = gen.thesis_row_gpus(a["thesis_target"], a["node"])
        b = gen.THESIS_CORE_BUDGET[gpus]
        cli = _cli(gen, a)
        assert cli["--reserved-driver-cores"] == str(b["trainer"]), a["name"]
        assert cli["--cpu-cores-per-actor"] == gen.THESIS_CORES_PER_ACTOR, \
            a["name"]


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
        # the smoke is outside the thesis matrix (owner ruling 2026-09-20):
        # it keeps today's update budget, not THESIS_PPO_EPOCHS/MINIBATCHES.
        assert cli["--ppo-epochs"] == "1", a["name"]
        assert cli["--minibatches"] == "4", a["name"]

    first, resume, cond = smoke
    assert _cli(gen, first)["--episodes"] == "20"
    assert _cli(gen, first)["--example"] == "VmappedTransformerLM"
    assert first["env"]["ALPHAGRAD_NN_BATCH"] == "64"
    assert _cli(gen, first)["--reward-mode"] == "lagrangian"
    assert _cli(gen, first)["--grad-oracle-cadence"] == "10"
    assert "--preference-conditioned" not in _cli(gen, first)
    assert _cli(gen, resume)["--grad-oracle-cadence"] == "10"
    # dsnn-dfw.104: the TLM smoke rows render the raised face-wire budget too.
    assert _cli(gen, first)["--face-wire-faces"] == TLM_FACE_WIRE_FACES
    assert _cli(gen, resume)["--face-wire-faces"] == TLM_FACE_WIRE_FACES

    assert _cli(gen, cond)["--episodes"] == "5"
    assert _cli(gen, cond)["--example"] == "VmappedNeuralNetwork"
    assert _cli(gen, cond)["--dataset"] == "mnist"
    assert "--preference-conditioned" in _cli(gen, cond)
    assert _cli(gen, cond)["--reward-mode"] == "lagrangian"
    assert _cli(gen, cond)["--face-wire-faces"] == FACE_WIRE_FACES
    assert cond["env"] == {"ALPHAGRAD_NN_HIDDEN": "256",
                           "ALPHAGRAD_NN_BATCH": "4096", **MATRIX_ENV}


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
              and not a.get("orderonly_final")
              and not a.get("orderonly_tlm_final") and not a.get("sweepl")
              and not a.get("sweepl2") and not a.get("sweepl3")]
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
    keys allowed to differ are the run name, the three that say which
    target this is, the face-head init, the face-entropy weight, the PopArt
    magnitude-scaling flags and (for condC on NN256) the episode count.

    --ray-measure is excluded because it is the node's GPU count minus one
    and the rows are spread over nodes of several sizes;
    `test_the_nodes_and_the_actors_per_node_size` and
    `test_the_recurrent_scheduling_matches_the_rest_of_the_matrix` pin it
    per row.

    The face-head init and the PopArt flags are excluded WHOLESALE rather
    than case-by-case: which rows carry the normalized init (rung 1, section
    8b) and which render on PopArt (section 8d) is pinned by their own
    dedicated tests, on both the NN256 and the TLM side (a recurrent C row
    now shares rung 1's numbers with its NN256 twin but not with its TLM
    one, and a recurrent condC row shares neither TLM condC's own init nor
    either twin's PopArt form) -- this test's job is that NOTHING ELSE
    differs.

    The face-entropy weight is excluded too (owner ruling 2026-09-21,
    dsnn-dfw.84 and dsnn-dfw.78), but NOT wholesale: the recurrent row
    renders 0.005 against its NN256 twin's 0.05 (always a diff) and against
    its TLM twin's OWN 0.005 (never a diff, since TLM is on the same number).
    Both directions are asserted below; the actual values are pinned by
    `test_every_arm_carries_the_shared_thesis_flags` above.

    The face-wire budget (dsnn-dfw.104, owner ruling 2026-09-22) is
    excluded the same way, in the OPPOSITE direction: the recurrent row
    stays on the campaign's 64, same as its NN256 twin (never a diff), but
    its TLM twin renders 128 (always a diff).
    """
    node_derived = {"--ray-measure"}
    face_init = {"--face-none-bias", "--face-init-approx-per-plan",
                 "--face-init-skips-per-plan"}
    popart_flags = {"--advantage-norm", "--no-symlog", "--symlog-channels"}
    entropy_weight = {"--face-entropy-weight"}
    target_keys = {"--name", "--example", "--dataset", "--temporal-rule"}
    by_key = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"]): a
              for a in core}
    for a in snn:
        cli = _cli(gen, a)
        for t in TARGETS:
            twin = by_key[(a["thesis_arm"], t, a["thesis_seed"])]
            ref = _cli(gen, twin)
            diff = {k for k in set(ref) | set(cli)
                    if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
            diff -= node_derived
            diff -= face_init
            diff -= popart_flags
            if t == "nn256":
                # the recurrent row renders 0.005 (dsnn-dfw.84 and
                # dsnn-dfw.78) against NN256's 0.05: always a real diff.
                assert "--face-entropy-weight" in diff, (a["name"], t)
            else:
                # TLM is on the SAME 0.005 as the recurrent target, so this
                # key does not even appear as a diff against the TLM twin.
                assert "--face-entropy-weight" not in diff, (a["name"], t)
            diff -= entropy_weight
            if t == "tlm":
                # dsnn-dfw.104: the TLM twin renders 128, the recurrent row
                # stays on the campaign's 64: always a real diff here.
                assert "--face-wire-faces" in diff, (a["name"], t)
            else:
                # NN256 is on the SAME 64 as the recurrent target.
                assert "--face-wire-faces" not in diff, (a["name"], t)
            diff -= {"--face-wire-faces"}
            if _is_long_episodes(twin):
                # the NN256 twin runs the long budget; the recurrent row is
                # not on that ruling
                assert "--episodes" in diff, (a["name"], t)
                diff -= {"--episodes"}
            assert diff == target_keys, (a["name"], t, sorted(diff))


def test_no_recurrent_row_carries_an_xla_flag(gen, snn):
    """The same rule the whole matrix runs under, asserted again on the
    recurrent rows on their own: no XLA_*, no JAX_* beyond the shared
    compilation cache, and no per-arm export but the measure actors'
    retention bound and process recycle (dsnn-dfw.99), which are
    ALPHAGRAD_ variables."""
    jax_cache_exports = {f"export {k}={v}" for k, v in gen.JAX_CACHE_ENV}
    jax_cache_mkdir = f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}"
    for a in snn:
        assert a["env"] == MATRIX_ENV, a["name"]
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
    # 100 rows, NOT one even round robin any more (owner ruling 2026-09-21):
    # the 75 non-rtrl rows split evenly over the three 4-GPU nodes, and the
    # 25 rtrl rows split over the two 8-GPU nodes BY SEED (uneven, since the
    # five seeds do not divide two nodes evenly).
    counts = {}
    for a in snn:
        counts[a["node"]] = counts.get(a["node"], 0) + 1
    assert counts == {"pgi15-gpu15": 25, "pgi15-gpu16": 25,
                      "pgi15-gpu18": 25, "pgi15-gpu19": 15,
                      "pgi15-gpu20": 10}, counts
    assert sum(counts.values()) == 100


def test_rtrl_renders_on_an_eight_gpu_node_round_robin_by_seed(gen, snn):
    """Owner ruling 2026-09-21 (agent rsnnpace's pace diagnosis): rtrl's
    measurement pipeline stalls at 3 actors once the plan carries real
    approximations, so every rtrl row -- every arm -- renders on an 8-GPU
    node with --ray-measure 7, round-robin BY SEED (gpu19 first) so a seed's
    node does not depend on which arm generated it, and never paired with a
    sibling on the same node.  bptt, tbptt and window2 keep the 4-GPU
    profile on the three other released nodes."""
    assert gen.THESIS_RSNN_RTRL_NODES == ("pgi15-gpu19", "pgi15-gpu20")
    assert gen.THESIS_RSNN_OTHER_NODES == (
        "pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18")
    want_rtrl_node = {
        s: gen.THESIS_RSNN_RTRL_NODES[i % 2] for i, s in enumerate(SEEDS)}
    for a in snn:
        cli = _cli(gen, a)
        if a["thesis_rule"] == "rtrl":
            assert a["node"] == want_rtrl_node[a["thesis_seed"]], a["name"]
            assert a["gpus"] == 8, a["name"]
            assert cli["--ray-measure"] == "7", a["name"]
        else:
            assert a["node"] in gen.THESIS_RSNN_OTHER_NODES, a["name"]
            assert a["gpus"] == 4, a["name"]
            assert cli["--ray-measure"] == "3", a["name"]
    # every rtrl seed's five arms land on the ONE node that seed maps to
    for s in SEEDS:
        nodes = {a["node"] for a in snn
                 if a["thesis_rule"] == "rtrl" and a["thesis_seed"] == s}
        assert nodes == {want_rtrl_node[s]}, s
    # no rtrl row is paired with a sibling on its node (the recurrent block
    # never calls `thesis_pair_arm`)
    paired_names = {h["name"] for p in gen.thesis_pair_arms()
                    for h in p["halves"]}
    assert not ({a["name"] for a in snn if a["thesis_rule"] == "rtrl"}
                & paired_names)


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


def test_the_retention_bound_reaches_every_row_thesis_arm_emits(
        gen, matrix, smoke):
    """dsnn-dfw.99.  The measure actors held one XLA executable per plan
    and nothing dropped them, so on the recurrent rows the actors' pool
    climbed to 72.4 GiB of a 96 GB card and refused 5-6 plans of 16 per
    episode (job 67410, from episode 120).  The bound is an EXPORT, not a
    flag: ppo.py has none, the actor reads the variable at construction.

    Its footprint is --dual-clip's and --target-kl's: the matrix on every
    target and the smoke.  It is OFF on the order-only tuning rows and on
    the three Lagrangian sweep rounds, and the absence is asserted rather
    than trusted -- they are frozen running comparisons and a launcher
    that moved under them would invalidate the round."""
    var = gen.MEASURE_CACHE_CLEAR_EVERY_VAR
    assert var == "ALPHAGRAD_MEASURE_CACHE_CLEAR_EVERY"
    assert gen.THESIS_MEASURE_CACHE_CLEAR_EVERY == CACHE_CLEAR_EVERY
    for a in matrix + smoke:
        assert a["env"][var] == CACHE_CLEAR_EVERY, a["name"]
        assert f"export {var}={CACHE_CLEAR_EVERY}" in gen.render(a), \
            a["name"]
    off = (gen.orderonly_arms() + gen.orderonly_rsnn_arms()
           + gen.orderonly_final_arms() + gen.orderonly_tlm_final_arms()
           + gen.sweepl_arms() + gen.sweepl2_arms() + gen.sweepl3_arms())
    assert off
    for a in off:
        assert var not in (a.get("env") or {}), a["name"]
        assert var not in gen.render(a), a["name"]
    # The export is admitted by the thesis allow-list only: a campaign arm
    # carries no per-arm environment at all.
    assert var in gen.THESIS_TARGET_ENV_ALLOWED
    assert var in gen.THESIS_ENV_ALLOWED
    assert var not in gen.CAMPAIGN_ENV_ALLOWED


def test_the_process_recycle_reaches_every_row_thesis_arm_emits(
        gen, matrix, smoke):
    """dsnn-dfw.99, follow-up ruling 2026-09-22 (08:58 cluster time), and
    dsnn-dfw.120 (owner ruling 2026-09-23).  The proactive recycle rides on
    the SAME footprint as the retention bound: the matrix on every target
    and the smoke, off on the order-only tuning rows and the three
    Lagrangian sweep rounds.  The retry of an OOM'd plan on a fresh actor
    is gone: no row exports its switch and the generator names it nowhere."""
    every_var = gen.PROACTIVE_RECYCLE_EVERY_VAR
    retry_var = "ALPHAGRAD_RECYCLE_RETRY_ON_OOM"
    assert every_var == "ALPHAGRAD_PROACTIVE_RECYCLE_EVERY"
    assert gen.THESIS_PROACTIVE_RECYCLE_EVERY == PROACTIVE_RECYCLE_EVERY
    assert not hasattr(gen, "RECYCLE_RETRY_ON_OOM_VAR")
    assert not hasattr(gen, "THESIS_RECYCLE_RETRY_ON_OOM")
    for a in gen.ARMS:
        assert retry_var not in (a.get("env") or {}), a["name"]
        assert retry_var not in gen.render(a), a["name"]
    for a in matrix + smoke:
        assert a["env"][every_var] == PROACTIVE_RECYCLE_EVERY, a["name"]
        assert (f"export {every_var}={PROACTIVE_RECYCLE_EVERY}"
               in gen.render(a)), a["name"]
    off = (gen.orderonly_arms() + gen.orderonly_rsnn_arms()
           + gen.orderonly_final_arms() + gen.orderonly_tlm_final_arms()
           + gen.sweepl_arms() + gen.sweepl2_arms() + gen.sweepl3_arms())
    assert off
    for a in off:
        assert every_var not in (a.get("env") or {}), a["name"]
        assert every_var not in gen.render(a), a["name"]
    assert every_var in gen.THESIS_TARGET_ENV_ALLOWED
    assert every_var in gen.THESIS_ENV_ALLOWED
    assert retry_var not in gen.THESIS_ENV_ALLOWED
    assert every_var not in set(gen.CAMPAIGN_ENV_ALLOWED)


def _frozen_rounds(gen):
    return (gen.orderonly_arms() + gen.orderonly_rsnn_arms()
            + gen.orderonly_final_arms() + gen.orderonly_tlm_final_arms()
            + gen.sweepl_arms() + gen.sweepl2_arms() + gen.sweepl3_arms())


def test_the_measure_timeout_and_the_actor_cores_of_a_thesis_row(
        gen, matrix, smoke, pairs):
    """Owner rulings 2026-09-23 and 2026-09-24 Q48: every row `thesis_arm`
    emits measures under --ray-measure-timeout 300, one deadline without a
    cold budget, and gives each timing actor 8 cores, on every node class.  The
    frozen rounds keep 600 and the budget's per-actor width."""
    assert gen.THESIS_RAY_MEASURE_TIMEOUT == "300"
    assert gen.CAMPAIGN_RAY_MEASURE_TIMEOUT == "600"
    assert gen.THESIS_CORES_PER_ACTOR == "8"
    for a in matrix + smoke + pairs:
        cli = _cli(gen, a)
        assert cli["--ray-measure-timeout"] == "300", a["name"]
        assert cli["--cpu-cores-per-actor"] == "8", a["name"]
        text = gen.render(a)
        assert "--ray-measure-timeout 300" in text, a["name"]
        assert "--cpu-cores-per-actor 8" in text, a["name"]
    for a in _frozen_rounds(gen):
        cli = _cli(gen, a)
        gpus = gen.thesis_row_gpus(a["thesis_target"], a["node"])
        assert cli["--ray-measure-timeout"] == \
            gen.CAMPAIGN_RAY_MEASURE_TIMEOUT, a["name"]
        assert cli["--cpu-cores-per-actor"] == \
            str(gen.THESIS_CORE_BUDGET[gpus]["per_actor"]), a["name"]
    from alphagrad.approx.common.core_budget import check_disjoint
    for gpus in (4, 8):
        lay = gen.thesis_row_core_layout(gpus)
        check_disjoint(lay)
        assert lay.n_logical == gen.BLACKWELL_CPUS[gpus]
        assert all(w == 8 for _, w in lay.timing_actors)
        assert len(lay.timing_actors) == int(gen.THESIS_RAY_MEASURE[gpus])


def _jax_cache_lines_in(gen, text):
    lines = [f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}"] + [
        f"export {k}={v}" for k, v in gen.JAX_CACHE_ENV]
    return [ln for ln in lines if ln + "\n" in text]


def test_the_compile_cache_is_exported_only_under_a_fixed_order(
        gen, matrix, smoke, pairs):
    """Owner ruling 2026-09-23: under a free order every plan is a new
    program and the per-node compile cache does not hit, so a thesis row
    with --fixed-order free exports none of JAX_COMPILATION_CACHE_DIR and
    its JAX_PERSISTENT_CACHE_* siblings; a fixed order exports all four.
    The frozen rounds keep what they ran with."""
    import copy
    for a in matrix + smoke + pairs:
        assert _cli(gen, a)["--fixed-order"] == "free", a["name"]
        text = gen.render(a)
        assert _jax_cache_lines_in(gen, text) == [], a["name"]
        assert "JAX_COMPILATION_CACHE_DIR" not in text, a["name"]
        assert "JAX_PERSISTENT_CACHE_" not in text, a["name"]
    for order in ("markowitz", "reverse"):
        for a in (matrix[0], smoke[0], pairs[0]):
            b = copy.deepcopy(a)
            b["cli"]["--fixed-order"] = order
            text = gen.render(b)
            assert len(_jax_cache_lines_in(gen, text)) == 5, (a["name"], order)
    for a in _frozen_rounds(gen):
        assert len(_jax_cache_lines_in(gen, gen.render(a))) == 5, a["name"]


def test_the_tlm_face_wire_budget_reaches_every_tlm_row_and_no_other(
        gen, matrix, smoke):
    """dsnn-dfw.104, owner ruling 2026-09-22.  Vertex 40 of a TLM plan
    carried 72 live faces at episode 55, past the campaign's 64, and the
    trainer raised by design.  Its footprint mirrors `rung1_row`,
    `condc_popart_row` and `condc_tlm_init_row`: ONLY A MATRIX COORDINATE
    (everything `thesis_arm` emits -- the matrix, the pair launchers and
    the smoke) renders it.  The order-only TLM final row is also target
    "tlm" and also goes through `thesis_cli`, but it is a DIFFERENT round
    with its own record, so it keeps 64 -- the same reason its rung-1 init
    and PopArt form do not move either."""
    assert gen.THESIS_TLM_FACE_WIRE_FACES == TLM_FACE_WIRE_FACES
    assert gen.CAMPAIGN_FACE_WIRE_FACES == FACE_WIRE_FACES
    for a in matrix + smoke:
        cli = _cli(gen, a)
        if a["thesis_target"] == "tlm":
            assert cli["--face-wire-faces"] == TLM_FACE_WIRE_FACES, a["name"]
            assert (f"--face-wire-faces {TLM_FACE_WIRE_FACES}"
                   in gen.render(a)), a["name"]
        else:
            assert cli["--face-wire-faces"] == FACE_WIRE_FACES, a["name"]
            assert (f"--face-wire-faces {FACE_WIRE_FACES}"
                   in gen.render(a)), a["name"]
    # the order-only row that also targets TLM is a DIFFERENT round (its
    # own record) and keeps the campaign's 64, exactly as its rung-1 init
    # and PopArt form stay off too.
    off = (gen.orderonly_arms() + gen.orderonly_rsnn_arms()
           + gen.orderonly_final_arms() + gen.orderonly_tlm_final_arms()
           + gen.sweepl_arms() + gen.sweepl2_arms() + gen.sweepl3_arms())
    assert off
    assert gen.orderonly_tlm_final_arms()
    for a in off:
        assert _cli(gen, a)["--face-wire-faces"] == FACE_WIRE_FACES, a["name"]
