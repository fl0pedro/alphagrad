"""Ticket `dsnn-dfw.45` -- ORDER-ONLY SCALARIZATION TUNING ON THE RECURRENT
TARGET, plus the no-temporal-edge reference.

Owner rulings 2026-09-19, pinned on `tools/gen_fq_launchers.py` rather than
on 39 files:

  1. THE ROWS: the NN256 order-only section (dsnn-dfw.29) on rsnn_bptt and
     rsnn_rtrl instead of nn256 -- five weight pairs x three seeds plus one
     preference-conditioned row per seed, per rule (18 x 2 = 36) -- named
     `orderonly_rsnn_<rule>_l<X>m<Y>_s<seed>` and
     `orderonly_rsnn_<rule>_pref_s<seed>`; plus `rsnn_tbptt` at (2, 0) for the
     three seeds, the no-temporal-edge reference, named
     `orderonly_rsnn_tbptt_l2m0_s<seed>` (3 rows, 39 total).
  2. THE ARM: --approx-profile none with --fixed-order free, the C form, the
     same weights, on RSNN_SHD with --temporal-rule set from the row's rule.
  3. THE NODE IS THE RULE (not the seed, unlike the NN256 round): bptt on
     pgi15-gpu14, rtrl on pgi15-gpu8, tbptt on pgi15-gpu8 after rtrl.
     pgi15-gpu8 is a node this generator did not know before this ticket:
     NODE_GRES_TYPE, NODE_GPUS, NODE_CPUS, NODE_MEM, NODE_CUDA_BIN and
     NODE_CUDA_WANT all gain a row (owner ruling 2026-09-19, evidence job
     66542: a matched 12.8 pair at /usr/local/cuda-12).
  4. NO XLA env; the one per-arm export is empty (the recurrent target's
     shape has no env var, same as the thesis matrix's recurrent block).
  5. The refusals.
  6. THE REGRESSION: the NN256 order-only round (dsnn-dfw.29) and the thesis
     matrix's own counts (50 core, 100 recurrent, 34 block 1) did not move.
"""
from __future__ import annotations

import importlib.util
import os
import re

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

# THE OWNER'S NUMBERS, TYPED HERE ON PURPOSE (bead dsnn-dfw.45).
RULES = ("bptt", "rtrl")
SEEDS = ("250197", "250198", "250199")
WEIGHTS = (("2", "0"), ("1.5", "0.5"), ("1", "1"), ("0.5", "1.5"), ("0", "2"))
NODES = {"bptt": "pgi15-gpu14", "rtrl": "pgi15-gpu8"}
TBPTT_NODE = "pgi15-gpu8"
TBPTT_WEIGHTS = ("2", "0")
PROFILE = "none"
ORDER = "free"
EPISODES = "1000"
CHECKPOINT_EVERY = "50"
PARETO_DUMP_EVERY = "10"
TAU = "0.90"
N_RUNS_PER_RULE = 18
N_RSNN_RUNS = N_RUNS_PER_RULE * len(RULES)          # 36
N_TBPTT_RUNS = len(SEEDS)                           # 3
N_TOTAL_RUNS = N_RSNN_RUNS + N_TBPTT_RUNS           # 39

_EXPORT = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", re.M)


class _Missing:
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
def rows(gen):
    arms = gen.orderonly_rsnn_arms()
    assert arms, "the generator emits no order-only recurrent arm"
    return arms


@pytest.fixture(scope="module")
def rsnn_rows(rows):
    """The 36 swept rows (bptt and rtrl), without the 3 tbptt reference
    rows -- most tests below are about the sweep, not the reference."""
    return [a for a in rows if a["thesis_target"] != "rsnn_tbptt"]


@pytest.fixture(scope="module")
def tbptt_rows(rows):
    return [a for a in rows if a["thesis_target"] == "rsnn_tbptt"]


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def _by_name(rows, name):
    return next(a for a in rows if a["name"] == name)


# --------------------------------------------------------------- 1. the rows

def test_the_rows_are_two_rules_of_eighteen_plus_three_tbptt_references(
        gen, rows, rsnn_rows, tbptt_rows):
    assert gen.ORDERONLY_RSNN_RULES == RULES
    assert gen.ORDERONLY_RSNN_RUNS_PER_RULE == N_RUNS_PER_RULE
    assert len(rsnn_rows) == N_RSNN_RUNS
    assert len(tbptt_rows) == N_TBPTT_RUNS
    assert len(rows) == N_TOTAL_RUNS
    names = [a["name"] for a in rows]
    assert len(set(names)) == len(names)
    want = {f"orderonly_rsnn_{r}_l{lc}m{lm}_s{s}"
            for r in RULES for lc, lm in WEIGHTS for s in SEEDS}
    want |= {f"orderonly_rsnn_{r}_pref_s{s}" for r in RULES for s in SEEDS}
    want |= {f"orderonly_rsnn_tbptt_l2m0_s{s}" for s in SEEDS}
    assert set(names) == want
    for a in rows:
        assert _cli(gen, a)["--name"] == a["name"]
    # the owner's own example spellings
    assert "orderonly_rsnn_bptt_l1.5m0.5_s250198" in names
    assert "orderonly_rsnn_rtrl_pref_s250197" in names
    assert "orderonly_rsnn_tbptt_l2m0_s250199" in names


def test_the_seeds_and_weights_are_the_nn256_rounds_own(gen):
    """One ruling, one set of numbers: dsnn-dfw.45 sweeps the identical five
    weights and three seeds dsnn-dfw.29 did, never a new one."""
    assert gen.ORDERONLY_SEEDS == SEEDS == gen.THESIS_SEEDS[:3]
    assert gen.ORDERONLY_WEIGHTS == WEIGHTS


def test_the_submission_order_is_seed_major_per_rule(gen):
    for rule in RULES:
        order = gen.orderonly_rsnn_submission_order(rule)
        assert len(order) == N_RUNS_PER_RULE == len(set(order))
        for i, s in enumerate(SEEDS):
            block = order[i * 6:(i + 1) * 6]
            assert {o[0] for o in block} == {s}
            assert [(o[1], o[2]) for o in block[:5]] == list(WEIGHTS)
            assert block[5][1] is None and block[5][2] is None


def test_every_order_only_recurrent_arm_renders_to_valid_bash(gen, rows):
    for a in rows:
        assert gen._bash_n(gen.render(a)) is None, a["name"]


# ---------------------------------------------------------------- 2. the arm

def test_the_arm_is_order_only_on_the_recurrent_target(gen, rsnn_rows):
    for a in rsnn_rows:
        cli = _cli(gen, a)
        assert cli["--approx-profile"] == PROFILE, a["name"]
        assert cli["--fixed-order"] == ORDER, a["name"]
        assert cli["--example"] == "RSNN_SHD", a["name"]
        assert cli["--dataset"] == "shd", a["name"]
        assert a["env"] == {}, a["name"]
        rule = a["thesis_target"].removeprefix("rsnn_")
        assert rule in RULES, a["name"]
        assert cli["--temporal-rule"] == rule, a["name"]


def test_the_tbptt_reference_carries_no_temporal_rule_edge(gen, tbptt_rows):
    for a in tbptt_rows:
        cli = _cli(gen, a)
        assert cli["--approx-profile"] == PROFILE, a["name"]
        assert cli["--fixed-order"] == ORDER, a["name"]
        assert cli["--example"] == "RSNN_SHD", a["name"]
        assert cli["--dataset"] == "shd", a["name"]
        assert cli["--temporal-rule"] == "tbptt", a["name"]
        assert a["thesis_target"] == "rsnn_tbptt", a["name"]
        assert cli["--lambda-cmp"] == "2" and cli["--lambda-mem"] == "0"
        assert "--preference-conditioned" not in cli, a["name"]


def test_every_row_is_the_c_form(gen, rows):
    for a in rows:
        cli = _cli(gen, a)
        assert cli["--reward-mode"] == "lagrangian", a["name"]
        assert cli["--quality-floor"] == TAU, a["name"]
        assert cli["--lag-eta"] == gen.DUAL_ETA, a["name"]
        assert cli["--lag-min"] == gen.DUAL_LAMBDA_MIN, a["name"]
        # THE CAP IS 64 ON A THESIS ROW (owner 2026-09-19), not the
        # campaign's 32; --lag-init and --lag-min do not move with it.
        assert cli["--lag-max"] == gen.THESIS_DUAL_LAMBDA_MAX == "64", a["name"]
        assert cli["--lag-init"] == gen.THESIS_LAMBDA_Q, a["name"]
        assert cli["--face-none-bias"] == "2", a["name"]
        assert cli["--advantage-norm"] == "none", a["name"]
        assert "--no-symlog" not in cli, a["name"]


def test_the_swept_weights_are_the_five_ruled_pairs_per_rule(gen, rsnn_rows):
    for rule in RULES:
        for lc, lm in WEIGHTS:
            for s in SEEDS:
                a = _by_name(rsnn_rows,
                            f"orderonly_rsnn_{rule}_l{lc}m{lm}_s{s}")
                cli = _cli(gen, a)
                assert cli["--lambda-cmp"] == lc, a["name"]
                assert cli["--lambda-mem"] == lm, a["name"]
                assert a["orderonly_weights"] == (lc, lm), a["name"]
                assert "--preference-conditioned" not in cli, a["name"]


def test_the_conditioned_row_sweeps_nothing_per_rule(gen, rsnn_rows):
    for rule in RULES:
        for s in SEEDS:
            a = _by_name(rsnn_rows, f"orderonly_rsnn_{rule}_pref_s{s}")
            cli = _cli(gen, a)
            assert "--preference-conditioned" in cli, a["name"]
            assert cli["--lambda-cmp"] == "1" and cli["--lambda-mem"] == "1"
            assert a["orderonly_weights"] is None, a["name"]
            assert a["thesis_arm"] == "condC", a["name"]


def test_the_run_shape_is_the_matrix_row(gen, rows):
    for a in rows:
        cli = _cli(gen, a)
        assert cli["--episodes"] == EPISODES, a["name"]
        assert cli["--checkpoint-every"] == CHECKPOINT_EVERY, a["name"]
        # A TUNING ROW KEEPS --auto-stop (owner 2026-09-19); only the FINAL
        # rows drop it.
        assert "--auto-stop" in cli and cli["--auto-stop"] is None, a["name"]
        assert cli["--pareto-dump-every"] == PARETO_DUMP_EVERY, a["name"]
        assert cli["--plan-log"] == "auto", a["name"]
        assert cli["--seed"] == a["thesis_seed"], a["name"]
        assert cli["--cost-form"] == "paired-log", a["name"]
        assert cli["--mem-channel"] == gen.THESIS_MEM_CHANNEL == "watermark", \
            a["name"]
        assert cli["--paired-cost-floor"] == gen.THESIS_PAIRED_COST_FLOOR \
            == "byte", a["name"]
        assert cli["--grad-oracle-cadence"] == "50", a["name"]
        text = gen.render(a)
        assert f"--wandb {gen.WANDB_MODE}" in text, a["name"]
        assert f"--wandb-project {gen.WANDB_PROJECT}" in text, a["name"]


# ------------------------------------------------------------- 3. the nodes

def test_the_node_is_the_rule(gen, rsnn_rows):
    assert gen.ORDERONLY_RSNN_NODES == NODES
    for a in rsnn_rows:
        rule = a["thesis_target"].removeprefix("rsnn_")
        assert a["node"] == NODES[rule], a["name"]
    # every rule's whole set (five weights x three seeds + three conditioned
    # rows = 18) is on the one node
    per_node = {}
    for a in rsnn_rows:
        per_node[a["node"]] = per_node.get(a["node"], 0) + 1
    assert per_node == {"pgi15-gpu14": 18, "pgi15-gpu8": 18}


def test_the_tbptt_reference_is_on_gpu8_after_rtrl(gen, tbptt_rows):
    """Owner ruling 2026-09-19: tbptt goes on pgi15-gpu8 after the rtrl rows
    (gpu13 is taken by the dsnn-dfw.44 run today), overriding the
    description's 'shorter queue; gpu13 if free'."""
    assert gen.ORDERONLY_RSNN_TBPTT_NODE == TBPTT_NODE == "pgi15-gpu8"
    for a in tbptt_rows:
        assert a["node"] == TBPTT_NODE, a["name"]
        assert a["job"] == gen.orderonly_job_name(TBPTT_NODE), a["name"]


def test_every_order_only_recurrent_job_is_a_cross_agent_per_node_singleton(
        gen, rows):
    for a in rows:
        text = gen.render(a)
        assert a["job"] == f"node-{a['node']}", a["name"]
        assert f"#SBATCH -J node-{a['node']}\n" in text, a["name"]
        assert "#SBATCH --dependency=singleton\n" in text, a["name"]
        assert f"#SBATCH -w {a['node']}\n" in text, a["name"]
        assert (f"#SBATCH -o {gen.CAMPAIGN_RUNS}/{a['name']}_%j.log\n"
                in text), a["name"]


def test_pgi15_gpu8_is_now_a_node_the_generator_knows(gen):
    """Ticket dsnn-dfw.45, owner ruling 2026-09-19 (evidence: job 66542,
    'toolchain gate OK on pgi15-gpu8: ptxas 12.8.93 ... nvlink 12.8.93').
    NODE_CUDA_BIN names it explicitly even though its matched pair is
    already inside /usr/local, because that is the one fact the job
    measured, and NODE_CUDA_WANT gives it its own release: the venv's own
    ptxas is 12.9, gpu8's matched pair is 12.8."""
    assert gen.NODE_GRES_TYPE["pgi15-gpu8"] == "nvidia_rtx_6000_ada_generation"
    assert gen.NODE_GPUS["pgi15-gpu8"] == 4
    assert gen.NODE_CPUS["pgi15-gpu8"] == 64
    assert gen.NODE_MEM["pgi15-gpu8"] == "340G"
    assert "pgi15-gpu8" not in gen.NODE_PARTITION
    assert gen.node_partition("pgi15-gpu8") == "pgi15"
    assert gen.NODE_CUDA_BIN["pgi15-gpu8"] == "/usr/local/cuda-12/bin"
    assert gen.NODE_CUDA_WANT["pgi15-gpu8"] == "12.8"
    assert gen.NODE_CUDA_WANT.get("pgi15-gpu14", gen.CUDA_WANT) == gen.CUDA_WANT
    assert gen.node_gres("pgi15-gpu8", 4) == "gpu:nvidia_rtx_6000_ada_generation:4"


def test_the_hardware_lines_follow_the_node(gen, rsnn_rows, tbptt_rows):
    for a in rsnn_rows + tbptt_rows:
        text = gen.render(a)
        node, gpus = a["node"], a["gpus"]
        assert gpus == gen.node_gpu_count(node), a["name"]
        assert f"#SBATCH -p {gen.node_partition(node)}\n" in text, a["name"]
        assert (f"#SBATCH --gres={gen.node_gres(node, gpus)}\n"
                in text), a["name"]
        assert f"#SBATCH -c {gen.node_cpus(node, gpus)}\n" in text, a["name"]
        assert f"#SBATCH --mem={gen.node_mem(node, gpus)}\n" in text, a["name"]


def test_the_gpu8_toolchain_wants_12_8_not_the_venvs_12_9(gen, rows):
    """The measure-toolchain block must read its wanted release PER NODE
    (NODE_CUDA_WANT), not always CUDA_WANT: gpu8's matched pair is 12.8, and
    a block that still asked for 12.9 there would abort 72 on a node the
    owner just cleared."""
    for a in rows:
        text = gen.render(a)
        if a["node"] == "pgi15-gpu8":
            assert "FQ_CUDA_WANT=12.8" in text, a["name"]
            assert "FQ_CUDA_WANT=12.9" not in text, a["name"]
            export_line = f'export PATH="{gen.NODE_CUDA_BIN["pgi15-gpu8"]}:$PATH"'
            assert export_line in text, a["name"]
        elif a["node"] == "pgi15-gpu14":
            assert "FQ_CUDA_WANT=12.9" in text, a["name"]
            assert "FQ_CUDA_WANT=12.8" not in text, a["name"]
        assert _cli(gen, a)["--measure-toolchain-gate"] == "abort", a["name"]
        assert "exit 72" in text, a["name"]


def test_the_toolchain_block_still_searches_usr_local_for_every_row(gen, rows):
    marker = re.compile(r"@[A-Z][A-Z0-9_]*@")
    for a in rows:
        text = gen.render(a)
        assert "\nfor d in /usr/local/cuda-*/bin; do\n" in text, a["name"]
        left = marker.findall(text)
        assert not left, (a["name"], sorted(set(left)))


# ------------------------------------------------------- 4. the environment

def test_no_order_only_recurrent_launcher_exports_an_xla_flag(gen, rows):
    jax_cache_exports = {f"export {k}={v}" for k, v in gen.JAX_CACHE_ENV}
    jax_cache_mkdir = f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}"
    for a in rows:
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


def test_every_export_in_an_order_only_recurrent_launcher_is_allowed(gen,
                                                                     rows):
    allowed = set(gen.THESIS_ENV_ALLOWED)
    for a in rows:
        exported = set(_EXPORT.findall(gen.render(a)))
        assert exported <= allowed, (a["name"], sorted(exported - allowed))
        assert set(gen.CAMPAIGN_ENV_ALLOWED) <= exported, a["name"]


def test_the_preflight_greps_the_swept_flags(gen, rows):
    assert "--lambda-cmp" in gen.ORDERONLY_REQUIRED_FLAGS
    assert "--lambda-mem" in gen.ORDERONLY_REQUIRED_FLAGS
    assert "--temporal-rule" in gen.ORDERONLY_REQUIRED_FLAGS
    for a in rows:
        text = gen.render(a)
        assert " ".join(gen.THESIS_FLAGS_FILES) in text, a["name"]


# ---------------------------------------------------------- 5. the refusals

def test_the_helpers_refuse_a_row_outside_the_rulings(gen):
    assert gen.orderonly_rsnn_run_name("bptt", "1.5", "0.5", "250198") == \
        "orderonly_rsnn_bptt_l1.5m0.5_s250198"
    assert gen.orderonly_rsnn_pref_run_name("rtrl", "250199") == \
        "orderonly_rsnn_rtrl_pref_s250199"
    assert gen.orderonly_rsnn_arm_name("250197") == \
        "orderonly_rsnn_tbptt_l2m0_s250197"
    for bad_rule in ("tbptt", "window2", "bogus"):
        with pytest.raises(gen.CampaignRowError):
            gen.orderonly_rsnn_run_name(bad_rule, "2", "0", "250197")
        with pytest.raises(gen.CampaignRowError):
            gen.orderonly_rsnn_pref_run_name(bad_rule, "250197")
        with pytest.raises(gen.CampaignRowError):
            gen.orderonly_rsnn_node(bad_rule)
    for bad in (("bptt", "3", "0", "250197"), ("rtrl", "1", "1", "250200")):
        with pytest.raises(gen.CampaignRowError):
            gen.orderonly_rsnn_run_name(*bad)
    with pytest.raises(gen.CampaignRowError):
        gen.orderonly_rsnn_arm_name("250200")
    # a conditioned row that also names a fixed weight says two things
    with pytest.raises(gen.CampaignRowError):
        gen.orderonly_rsnn_arm(rule="bptt", seed="250197", lam_cmp="1",
                               lam_mem="1", pref=True)
    # and a fixed-weight row that names neither
    with pytest.raises(gen.CampaignRowError):
        gen.orderonly_rsnn_arm(rule="rtrl", seed="250197", pref=False)


# ---------------------------------------------------------- 6. ppo's argparse

def test_ppo_argparse_accepts_every_order_only_recurrent_command_line(gen,
                                                                      rows):
    from alphagrad.approx.ppo import make_argparser
    for a in rows:
        toks = gen.cli_tokens(a)
        try:
            ns = make_argparser().parse_args(toks)
        except SystemExit as exc:
            raise AssertionError(f"{a['name']}: argparse rejected "
                                 f"{toks}") from exc
        assert ns.name == _cli(gen, a)["--name"]
        assert ns.fixed_order == ORDER
        assert ns.approx_profile == PROFILE
        assert ns.seed == int(a["thesis_seed"])
        assert ns.auto_stop is True
        assert ns.checkpoint_every == int(CHECKPOINT_EVERY)


# -------------------------------------------------------- 7. the regression

def test_the_nn256_order_only_round_did_not_move(gen):
    """dsnn-dfw.29's 18 rows are untouched by this ticket's new section."""
    nn256_rows = gen.orderonly_arms()
    assert len(nn256_rows) == 18
    assert all(a["thesis_target"] == "nn256" for a in nn256_rows)
    assert not any(a.get("orderonly_rsnn") for a in nn256_rows)


def test_the_matrix_and_the_campaign_still_have_their_own_counts(gen):
    # `orderonly_final`, the five-seed order-only baseline of 2026-09-19, is
    # excluded here exactly as the two tuning rounds are.
    matrix = [a for a in gen.thesis_arms()
              if not a.get("smoke") and not a.get("orderonly")
              and not a.get("orderonly_rsnn")
              and not a.get("orderonly_final")
              and not a.get("orderonly_tlm_final")
              and not a.get("paired") and not a.get("sweepl")]
    assert len(gen.thesis_core_arms()) == 50
    assert len(gen.thesis_snn_arms()) == 100
    assert len(matrix) == 150
    assert len(gen.thesis_smoke_arms()) == 3
    assert len(gen.orderonly_arms()) == 18
    assert len(gen.orderonly_rsnn_arms()) == N_TOTAL_RUNS
    block = gen.thesis_block1_arms()
    assert len(block) == gen.THESIS_BLOCK1 == 34
    assert not any(a.get("orderonly") for a in block)
    assert not any(a.get("orderonly_rsnn") for a in block)


def test_no_arm_outside_this_section_gained_the_orderonly_rsnn_flag(gen):
    for a in gen.ARMS:
        if a.get("orderonly_rsnn"):
            assert a["thesis_target"] in ("rsnn_bptt", "rsnn_rtrl",
                                          "rsnn_tbptt"), a["name"]
        else:
            assert a.get("thesis_target") not in ("rsnn_bptt", "rsnn_rtrl") \
                or not a.get("orderonly"), a["name"]
