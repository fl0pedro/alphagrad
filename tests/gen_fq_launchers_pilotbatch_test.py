"""The generator batch before the pilot (owner rulings 2026-09-25, epic
dsnn-dfw, grill rounds 1-3), pinned on `tools/gen_fq_launchers.py`:

  1. dsnn-dfw.169: ALPHAGRAD_DIRECT_MEASURE=1 and ALPHAGRAD_UNIFIED_FACE_ENUM=1.
  2. No rendered row sets GRAPHAX_KEEP_BLOCKDIAG=0.
  3. dsnn-mep: the static memory objective (slot 11) at weight 1; "mem"
     leaves --rewards and slot 5 keeps the watermark for the log.
  4. dsnn-dfw.230: no compile cache under any order.
  5. dsnn-dfw.208: the gradient oracle takes every core the trainer and the
     timing actors leave.
  6. report dsnn-dfw.237: the oracle's batch by target, its host budget and
     the row's memory by node size.
  7. condC left the matrix.
  8. dsnn-dfw.231: the defense arms A_popart at lambda_q 16, 4 and 64.

"Every row `thesis_arm` emits" is the matrix, the defense rows, the pair
launchers and the smoke.  The frozen rounds (the order-only rounds and the
three sweep rounds) keep what they ran with, and each test asserts that side.
"""
from __future__ import annotations

import importlib.util
import os
import re

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

# THE OWNER'S NUMBERS, TYPED HERE ON PURPOSE.
SEEDS = ("250197", "250198", "250199", "250200", "250201")
MEASURE_PATH_ENV = {"ALPHAGRAD_DIRECT_MEASURE": "1",
                    "ALPHAGRAD_UNIFIED_FACE_ENUM": "1"}
MEM_OBJECTIVE_WEIGHT = "1"
REWARDS = "cmp acc"
FROZEN_REWARDS = "cmp mem acc"
#: sinfo, 2026-09-25: CPUs and RealMemory (MB) of a 4-GPU Blackwell node
#: (pgi15-gpu15 to 18) and of an 8-GPU one (pgi15-gpu19, 20).
NODE_CPUS = {4: 64, 8: 128}
NODE_REAL_MEMORY_MB = {4: 770000, 8: 1540000}
ROW_MEM = {4: "740G", 8: "1480G"}
TRAINER_CORES = 8
CORES_PER_ACTOR = 8
#: every core the trainer and the timing actors leave: 64 - 8 - 3 x 8 and
#: 128 - 8 - 7 x 8
ORACLE_CORES = {4: 32, 8: 64}
HOST_BUDGET_GB = {4: "300", 8: "600"}
#: the peak RSS of a check over the size the budget bars (dsnn-dfw.236)
RSS_OVER_BUDGET = 1.34
ORACLE_BATCH = {"nn256": "0", "tlm": "2"}
RSNN_ORACLE_BATCH = "0"
ARMS = ("A", "B", "C", "C_popart")
DEFENSE_LAMBDA_Q = {"A_popart": "16", "A_popart_lq4": "4",
                    "A_popart_lq64": "64"}
DEFENSE_SEEDS = SEEDS[:3]
ORACLE_FLAGS = ("--grad-oracle-cores", "--grad-oracle-batch",
                "--grad-oracle-host-budget-gb")
_FROZEN_KEYS = ("orderonly", "orderonly_rsnn", "orderonly_final",
                "orderonly_tlm_final", "sweepl", "sweepl2", "sweepl3")
_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]*\}")


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
def emitted(gen):
    rows = [a for a in gen.thesis_arms()
            if not any(a.get(k) for k in _FROZEN_KEYS)]
    assert rows
    return rows


@pytest.fixture(scope="module")
def frozen(gen):
    rows = [a for a in gen.thesis_arms()
            if any(a.get(k) for k in _FROZEN_KEYS)]
    assert rows
    return rows


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def _parse(gen, a):
    from alphagrad.approx.ppo import make_argparser
    toks = [_PLACEHOLDER.sub("/tmp/ckpt", t) for t in gen.cli_tokens(a)]
    return make_argparser().parse_args(toks)


def _profile(gen, a) -> int:
    """The GPU profile of the row's own command line (a pair's halves are
    4-GPU rows on an 8-GPU node)."""
    return gen.thesis_row_gpus(a["thesis_target"], a["node"])


# ------------------------------------------------------ 1. the measure path

def test_every_row_thesis_arm_emits_exports_the_measure_path(gen, emitted,
                                                             frozen):
    assert gen.THESIS_MEASURE_PATH_ENV == MEASURE_PATH_ENV
    for a in emitted:
        text = gen.render(a)
        for k, v in MEASURE_PATH_ENV.items():
            assert a["env"][k] == v, (a["name"], k)
            assert f"\nexport {k}={v}\n" in text, (a["name"], k)
            assert text.count(f"export {k}=") == 1, (a["name"], k)
    for a in frozen:
        text = gen.render(a)
        for k in MEASURE_PATH_ENV:
            assert k not in (a.get("env") or {}), (a["name"], k)
            assert f"export {k}=" not in text, (a["name"], k)
    assert set(MEASURE_PATH_ENV) <= gen.THESIS_TARGET_ENV_ALLOWED
    assert not set(MEASURE_PATH_ENV) & set(gen.CAMPAIGN_ENV_ALLOWED)
    env_py = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx",
                               "env.py")).read()
    for k in MEASURE_PATH_ENV:
        assert re.search(r'environ\.get\(\s*"%s"' % k, env_py), k


# ------------------------------------------------- 2. the block diagonal

def test_no_rendered_row_turns_the_block_diagonal_off(gen):
    spec = importlib.util.find_spec("graphax")
    micro = os.path.join(spec.submodule_search_locations[0], "sparse",
                         "micro_actions.py")
    src = open(micro).read()
    assert re.search(r'_KEEP_BLOCKDIAG = _os\.environ\.get\('
                     r'"GRAPHAX_KEEP_BLOCKDIAG", "1"\) != "0"', src)
    off = re.compile(r"GRAPHAX_KEEP_BLOCKDIAG=[\"']?0\b")
    for a in gen.ARMS:
        assert str((a.get("env") or {}).get("GRAPHAX_KEEP_BLOCKDIAG")) != "0"
        for line in gen.render(a).splitlines():
            if line.lstrip().startswith("#"):
                continue
            assert not off.search(line), (a["name"], line)
    assert dict(gen.SHARED_ENV).get("GRAPHAX_KEEP_BLOCKDIAG") in (None, "1")


# --------------------------------------------- 3. the memory objective

def test_every_row_thesis_arm_emits_trains_the_static_memory_objective(
        gen, emitted, frozen):
    assert gen.THESIS_MEM_OBJECTIVE_WEIGHT == MEM_OBJECTIVE_WEIGHT
    # the wave arms inherit SHARED_CLI; its value does not move
    assert dict(gen.SHARED_CLI)["--rewards"] == FROZEN_REWARDS
    for a in emitted:
        cli = _cli(gen, a)
        assert cli["--rewards"] == REWARDS, a["name"]
        assert cli["--mem-objective-weight"] == MEM_OBJECTIVE_WEIGHT, a["name"]
        # slot 5 still holds the watermark and is logged, with weight 0
        assert cli["--mem-channel"] == "watermark", a["name"]
        assert cli["--mem-type"] == "peak_memory", a["name"]
        assert cli["--cost-form"] == "paired-log", a["name"]
        assert "--mem-objective-weight" in a["required_flags"], a["name"]
        text = gen.render(a)
        assert f"\n  --rewards {REWARDS}\n" in text, a["name"]
        assert (f"\n  --mem-objective-weight {MEM_OBJECTIVE_WEIGHT}\n"
                in text), a["name"]
        assert a["purpose"].startswith(gen.THESIS_MATRIX_HEAD), a["name"]
        ns = _parse(gen, a)
        assert ns.rewards == ["cmp", "acc"], a["name"]
        assert ns.mem_objective_weight == 1.0, a["name"]
        assert ns.mem_channel == "watermark", a["name"]
    for a in frozen:
        cli = _cli(gen, a)
        assert cli["--rewards"] == FROZEN_REWARDS, a["name"]
        assert "--mem-objective-weight" not in cli, a["name"]
        assert "--mem-objective-weight" not in a["required_flags"], a["name"]
    for a in gen.sweepl_arms() + gen.sweepl2_arms() + gen.sweepl3_arms():
        assert a["purpose"].startswith(gen.THESIS_HEAD), a["name"]
    assert "slot 11" in gen.THESIS_MATRIX_HEAD
    assert "runtime watermark memory" in gen.THESIS_HEAD


# ------------------------------------------------- 4. no compile cache

def test_no_row_thesis_arm_emits_sets_a_compile_cache_under_any_order(
        gen, emitted, frozen):
    for a in emitted:
        assert a["jax_cache"] is False, a["name"]
        for order in ("free", "markowitz", "reverse"):
            b = dict(a, cli=dict(a["cli"], **{"--fixed-order": order}))
            if a.get("halves"):
                b["halves"] = [dict(h, cli=dict(h["cli"],
                                                **{"--fixed-order": order}))
                               for h in a["halves"]]
            text = gen.render(b)
            assert "JAX_COMPILATION_CACHE_DIR" not in text, (a["name"], order)
            assert "JAX_PERSISTENT_CACHE_" not in text, (a["name"], order)
            assert f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}" not in text, \
                (a["name"], order)
    for a in frozen:
        text = gen.render(a)
        assert (f"export JAX_COMPILATION_CACHE_DIR={gen.JAX_CACHE_DIR_EXPR}\n"
                in text), a["name"]


# ---------------------------------------------- 5. every spare core

def test_the_oracle_takes_every_core_the_trainer_and_the_actors_leave(
        gen, emitted, frozen):
    from alphagrad.approx.common.core_budget import (check_disjoint,
                                                      node_core_layout)
    assert gen.BLACKWELL_CPUS == NODE_CPUS
    assert gen.THESIS_CORE_BUDGET == {
        g: {"trainer": TRAINER_CORES, "per_actor": CORES_PER_ACTOR,
            "oracle": ORACLE_CORES[g]} for g in (4, 8)}
    for g in (4, 8):
        lay = gen.thesis_core_layout(g)
        check_disjoint(lay)
        assert lay.n_logical == NODE_CPUS[g]
        assert lay.spare == ()
        assert lay.oracle == (NODE_CPUS[g] - ORACLE_CORES[g], ORACLE_CORES[g])
    for a in emitted:
        g = _profile(gen, a)
        cli = _cli(gen, a)
        assert cli["--grad-oracle-cores"] == "0", a["name"]
        for flag in ORACLE_FLAGS:
            assert flag in a["required_flags"], (a["name"], flag)
        ns = _parse(gen, a)
        assert ns.grad_oracle_cores == 0, a["name"]
        assert ns.reserved_driver_cores == TRAINER_CORES, a["name"]
        assert ns.cpu_cores_per_actor == CORES_PER_ACTOR, a["name"]
        assert ns.ray_measure == g - 1, a["name"]
        text = gen.render(a)
        if a.get("halves"):
            # a pair holds the node; each half is a 4-GPU row on its own half
            # of the cores, which is the affinity mask ppo.py reads
            assert f"#SBATCH -c {NODE_CPUS[8]}\n" in text, a["name"]
            widths = []
            for h in a["halves"]:
                lo, hi = (int(x) for x in h["cores"].split("-"))
                widths.append(hi - lo + 1)
            assert widths == [NODE_CPUS[4]] * 2, a["name"]
            cpus = NODE_CPUS[4]
        else:
            assert f"#SBATCH -c {NODE_CPUS[g]}\n" in text, a["name"]
            cpus = NODE_CPUS[g]
        # the oracle's width ppo.py computes from --grad-oracle-cores 0 is the
        # budget's, and the layout it builds is the one checked at import
        spare = (cpus - ns.reserved_driver_cores
                 - ns.ray_measure * ns.cpu_cores_per_actor)
        assert spare == ORACLE_CORES[g] == gen.THESIS_CORE_BUDGET[g]["oracle"]
        assert node_core_layout(
            cpus, ns.ray_measure, trainer_cores=ns.reserved_driver_cores,
            cores_per_actor=ns.cpu_cores_per_actor,
            oracle_cores=spare) == gen.thesis_core_layout(g), a["name"]
    for a in frozen:
        cli = _cli(gen, a)
        assert "--grad-oracle-cores" not in cli, a["name"]
        assert cli["--cpu-cores-per-actor"] == "2", a["name"]


# --------------------------------- 6. the oracle's batch, budget and memory

def test_the_oracle_batch_and_host_budget_follow_the_target_and_the_node(
        gen, emitted, frozen):
    assert gen.THESIS_GRAD_ORACLE_BATCH == {
        **ORACLE_BATCH, **{t: RSNN_ORACLE_BATCH
                           for t in gen.THESIS_RSNN_TARGETS}}
    assert gen.THESIS_GRAD_ORACLE_HOST_BUDGET_GB == HOST_BUDGET_GB
    assert gen.THESIS_ROW_MEM == ROW_MEM
    for g in (4, 8):
        mem_gib = int(ROW_MEM[g].rstrip("G"))
        # schedulable under sinfo's RealMemory, with a margin left to the OS
        assert mem_gib * 1024 < NODE_REAL_MEMORY_MB[g]
        # the largest check the budget admits fits inside the row's memory
        assert RSS_OVER_BUDGET * float(HOST_BUDGET_GB[g]) < mem_gib
    # a pair runs two 4-GPU rows and their two checks on one 8-GPU node
    assert (2 * RSS_OVER_BUDGET * float(HOST_BUDGET_GB[4])
            < int(ROW_MEM[8].rstrip("G")))
    for a in emitted:
        cli = _cli(gen, a)
        want = ORACLE_BATCH.get(a["thesis_target"], RSNN_ORACLE_BATCH)
        assert cli["--grad-oracle-batch"] == want, a["name"]
        assert cli["--grad-oracle-host-budget-gb"] == \
            HOST_BUDGET_GB[_profile(gen, a)], a["name"]
        ns = _parse(gen, a)
        assert ns.grad_oracle_batch == int(want), a["name"]
        assert ns.grad_oracle_host_budget_gb == \
            float(HOST_BUDGET_GB[_profile(gen, a)]), a["name"]
        text = gen.render(a)
        assert f"#SBATCH --mem={ROW_MEM[a['gpus']]}\n" in text, a["name"]
    for a in frozen:
        cli = _cli(gen, a)
        for flag in ORACLE_FLAGS:
            assert flag not in cli, (a["name"], flag)
        text = gen.render(a)
        assert (f"#SBATCH --mem={gen.node_mem(a['node'], a['gpus'])}\n"
                in text), a["name"]


# ------------------------------------------------------- 7. condC out

def test_condc_left_the_matrix_and_the_frozen_preference_rows_keep_it(
        gen, emitted):
    assert gen.THESIS_ARMS == ARMS
    assert "condC" not in gen.THESIS_ARM_SPEC
    for a in emitted:
        assert a["thesis_arm"] != "condC", a["name"]
        assert "--preference-conditioned" not in _cli(gen, a), a["name"]
    assert all(arm != "condC" for arm, _t, _s in
               gen.thesis_submission_order())
    got = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"])
           for a in gen.thesis_block1_arms()}
    want = {(arm, t, s) for arm in ("C", "C_popart")
            for t in ("nn256", "tlm") for s in SEEDS}
    want |= {(arm, t, SEEDS[0]) for arm in ("A", "B")
             for t in ("nn256", "tlm")}
    assert got == want
    assert gen.THESIS_BLOCK1 == len(want) == 24
    n0 = len(gen.ARMS)
    with pytest.raises(gen.CampaignRowError):
        gen.thesis_arm(arm="condC", target="nn256", seed=SEEDS[0],
                       node=gen.THESIS_NODES[0], name="throwaway_condc")
    assert len(gen.ARMS) == n0
    assert [a["name"] for a in gen.thesis_smoke_arms()] == [
        "smoke_C_tlm", "smoke_C_tlm_resume"]
    pref = [a for a in gen.orderonly_arms() + gen.orderonly_rsnn_arms()
            if a["thesis_arm"] == "condC"]
    assert len(pref) == 3 + 6
    for a in pref:
        cli = _cli(gen, a)
        assert "--preference-conditioned" in cli, a["name"]
        assert cli["--reward-mode"] == "lagrangian", a["name"]
        assert cli["--advantage-norm"] == "none", a["name"]
        assert cli["--face-none-bias"] == "2", a["name"]
        assert cli["--lag-init"] == "16", a["name"]


# ---------------------------------------------------- 8. the defense arms

def test_the_defense_arms_are_arm_a_on_popart_at_three_lambdas(gen):
    assert gen.THESIS_DEFENSE_ARMS == tuple(DEFENSE_LAMBDA_Q)
    rows = gen.thesis_defense_arms()
    got = {(a["thesis_arm"], a["thesis_target"], a["thesis_seed"])
           for a in rows}
    assert got == {(arm, "nn256", s) for arm in DEFENSE_LAMBDA_Q
                   for s in DEFENSE_SEEDS}
    assert len(rows) == 9
    block = {a["name"] for a in gen.thesis_block1_arms()}
    a_rows = {a["thesis_seed"]: a for a in gen.thesis_core_arms()
              if a["thesis_arm"] == "A" and a["thesis_target"] == "nn256"}
    for a in rows:
        lq = DEFENSE_LAMBDA_Q[a["thesis_arm"]]
        assert a["name"] == f"{a['thesis_arm']}_nn256_s{a['thesis_seed']}"
        # the full experiments only: held, not in block 1, not the pilot
        assert a.get("held"), a["name"]
        assert a["name"] not in block, a["name"]
        cli = _cli(gen, a)
        assert cli["--face-none-bias"] == "0", a["name"]
        assert cli["--reward-mode"] == "additive", a["name"]
        assert "--quality-floor" not in cli, a["name"]
        assert cli["--lambda-acc"] == lq, a["name"]
        assert cli["--advantage-norm"] == "popart", a["name"]
        assert "--no-symlog" in cli and cli["--no-symlog"] is None, a["name"]
        assert cli["--symlog-channels"] == "none", a["name"]
        assert "--preference-conditioned" not in cli, a["name"]
        for flag in ("--lag-eta", "--lag-init", "--lag-min", "--lag-max"):
            assert flag not in cli, (a["name"], flag)
        # arm A at the same seed differs in the PopArt form and lambda_q only
        ref = _cli(gen, a_rows[a["thesis_seed"]])
        diff = {k for k in set(ref) | set(cli)
                if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
        assert diff == ({"--name", "--advantage-norm", "--no-symlog",
                         "--symlog-channels"}
                        | ({"--lambda-acc"} if lq != "16" else set())), \
            (a["name"], sorted(diff))
        text = gen.render(a)
        assert "*** HELD" in text and f"ABORT(73): {a['name']} is HELD" in text
        for line in (f"--lambda-acc {lq}", "--advantage-norm popart",
                     "--no-symlog", "--symlog-channels none",
                     "--face-none-bias 0", "--reward-mode additive"):
            assert f"\n  {line}\n" in text, (a["name"], line)
        ns = _parse(gen, a)
        assert ns.advantage_norm == "popart" and ns.no_symlog, a["name"]
        assert ns.symlog_channels == "none", a["name"]
        assert ns.lambda_acc == float(lq), a["name"]
        assert ns.reward_mode == "additive", a["name"]
        assert ns.quality_floor is None, a["name"]
    with pytest.raises(gen.CampaignRowError):
        gen.thesis_arm(arm="A_popart", target="tlm", seed=SEEDS[0],
                       node=gen.THESIS_NODES[0], name="throwaway_defense")
