"""Ticket `dsnn-dfw.29` -- ORDER-ONLY SCALARIZATION TUNING, ROUND 1.

Launchers are generated, never hand-edited, so the owner's rulings of
2026-09-18 are pinned on `tools/gen_fq_launchers.py` rather than on 18 files:

  1. THE ROWS: five weight pairs x three seeds (250197..250199) plus one
     preference-conditioned run per seed = 18, named
     `orderonly_nn256_l<X>m<Y>_s<seed>` and `orderonly_nn256_pref_s<seed>`.
  2. THE ARM: --approx-profile none with --fixed-order free on NN256, the C
     form, and the two swept weights.  Everything else is the matrix row.
  3. THE NODE IS THE SEED: one node per seed, so a weight comparison never
     straddles two GPU models, with the per-node singleton and the node's own
     gres, partition, CPUs and memory.
  4. NO XLA env; the one per-arm export is the NeuralNetwork target shape.
  5. The refusals, and ppo.py's own argparse on every command line.
  6. THE REGRESSION: the campaign and thesis-matrix arms did not move.  The
     hardware lines became node-keyed for this round and every arm that
     existed before it must render exactly what it rendered before.
"""
from __future__ import annotations

import importlib.util
import os
import re

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

# THE OWNER'S NUMBERS, TYPED HERE ON PURPOSE (bead dsnn-dfw.29).
SEEDS = ("250197", "250198", "250199")
WEIGHTS = (("2", "0"), ("1.5", "0.5"), ("1", "1"), ("0.5", "1.5"), ("0", "2"))
NODES = ("pgi15-gpu14", "pgi15-gpu13", "pgi15-gpu14")
PROFILE = "none"
ORDER = "free"
EPISODES = "1000"
#: The NN256 baseline runs 2000 (owner, 2026-09-26). The finished tuning round and TLM keep EPISODES.
FINAL_EPISODES = "2000"
CHECKPOINT_EVERY = "50"
PARETO_DUMP_EVERY = "10"
TAU = "0.90"
N_RUNS = 18

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
    arms = gen.orderonly_arms()
    assert arms, "the generator emits no order-only arm"
    return arms


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def _by_name(rows, name):
    return next(a for a in rows if a["name"] == name)


# --------------------------------------------------------------- 1. the rows

def test_round_one_is_five_weights_three_seeds_plus_the_conditioned_row(gen,
                                                                       rows):
    assert gen.ORDERONLY_SEEDS == SEEDS
    assert gen.ORDERONLY_WEIGHTS == WEIGHTS
    assert gen.ORDERONLY_RUNS == N_RUNS == len(rows)
    names = [a["name"] for a in rows]
    assert len(set(names)) == len(names)
    want = {f"orderonly_nn256_l{lc}m{lm}_s{s}"
            for lc, lm in WEIGHTS for s in SEEDS}
    want |= {f"orderonly_nn256_pref_s{s}" for s in SEEDS}
    assert set(names) == want
    for a in rows:
        assert _cli(gen, a)["--name"] == a["name"]
    # the owner's own example spellings
    assert "orderonly_nn256_l1.5m0.5_s250198" in names
    assert "orderonly_nn256_pref_s250197" in names


def test_the_seeds_are_the_first_three_of_the_matrix_never_a_new_one(gen):
    """Three seeds for the tuning, five for the baseline that follows it.  A
    tuning seed the matrix does not know would make the baseline a different
    experiment."""
    assert gen.ORDERONLY_SEEDS == gen.THESIS_SEEDS[:3]


def test_the_submission_order_is_seed_major_with_the_conditioned_row_last(gen):
    order = gen.orderonly_submission_order()
    assert len(order) == N_RUNS and len(set(order)) == N_RUNS
    for i, s in enumerate(SEEDS):
        block = order[i * 6:(i + 1) * 6]
        assert {o[0] for o in block} == {s}
        assert [(o[1], o[2]) for o in block[:5]] == list(WEIGHTS)
        assert block[5][1] is None and block[5][2] is None


def test_every_order_only_arm_renders_to_valid_bash(gen, rows):
    for a in rows:
        assert gen._bash_n(gen.render(a)) is None, a["name"]


# ---------------------------------------------------------------- 2. the arm

def test_the_arm_is_order_only_on_nn256(gen, rows):
    """--approx-profile none IS the arm: the policy chooses the elimination
    order and nothing else, so the grad cosine is 1 on every plan."""
    for a in rows:
        cli = _cli(gen, a)
        assert cli["--approx-profile"] == PROFILE, a["name"]
        assert cli["--fixed-order"] == ORDER, a["name"]
        assert cli["--example"] == "NeuralNetwork", a["name"]
        assert cli["--dataset"] == "mnist", a["name"]
        assert a["env"] == {"ALPHAGRAD_NN_HIDDEN": "256"}, a["name"]
        assert a["thesis_target"] == "nn256", a["name"]


def test_every_row_is_the_c_form(gen, rows):
    """The C form, unchanged from the matrix.  It is INERT here -- quality is
    constant 1 so the constraint never binds -- and it is kept so that this
    round and the matrix's C rows differ in the swept flags alone."""
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


def test_the_swept_weights_are_the_five_ruled_pairs(gen, rows):
    for lc, lm in WEIGHTS:
        for s in SEEDS:
            a = _by_name(rows, f"orderonly_nn256_l{lc}m{lm}_s{s}")
            cli = _cli(gen, a)
            assert cli["--lambda-cmp"] == lc, a["name"]
            assert cli["--lambda-mem"] == lm, a["name"]
            assert a["orderonly_weights"] == (lc, lm), a["name"]
            assert "--preference-conditioned" not in cli, a["name"]
    # the weights sum to 2 on every row: the five rows differ in DIRECTION,
    # not in the scale of the advantage
    for lc, lm in WEIGHTS:
        assert float(lc) + float(lm) == 2.0, (lc, lm)


def test_the_conditioned_row_sweeps_nothing_and_carries_the_conditioning(gen,
                                                                        rows):
    """The Dirichlet preference over (latency, memory) IS the weight there, so
    the row keeps the matrix's own 1/1 and adds the flag."""
    for s in SEEDS:
        a = _by_name(rows, f"orderonly_nn256_pref_s{s}")
        cli = _cli(gen, a)
        assert "--preference-conditioned" in cli, a["name"]
        assert cli["--lambda-cmp"] == "1" and cli["--lambda-mem"] == "1"
        assert a["orderonly_weights"] is None, a["name"]
        assert a["thesis_arm"] == "condC", a["name"]


def test_the_run_shape_is_the_matrix_row(gen, rows):
    """Auto-stop, checkpoint 50, the dumps, the plan log -- unchanged."""
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
        assert cli["--rewards"] == "cmp mem acc", a["name"]
        assert cli["--discount"] == "1.0" and cli["--gae-lambda"] == "1.0"
        assert "--terminal-rewards-only" in cli, a["name"]
        assert cli["--grad-oracle-cadence"] == "50", a["name"]
        # wandb online, to the project the matrix already writes to
        text = gen.render(a)
        assert f"--wandb {gen.WANDB_MODE}" in text, a["name"]
        assert f"--wandb-project {gen.WANDB_PROJECT}" in text, a["name"]


def test_the_rows_differ_only_in_the_swept_flags(gen, rows):
    """Every pair of fixed-weight rows at one seed differs only in the two
    weights and the name; the same row at two seeds differs only in the seed,
    the name and what the NODE derives."""
    node_derived = {"--ray-measure"}
    by_key = {(a["thesis_seed"], a["orderonly_weights"]): a for a in rows}
    for s in SEEDS:
        ref = _cli(gen, by_key[(s, WEIGHTS[0])])
        for w in WEIGHTS[1:]:
            cli = _cli(gen, by_key[(s, w)])
            diff = {k for k in set(ref) | set(cli)
                    if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
            assert diff == {"--name", "--lambda-cmp", "--lambda-mem"}, (s, w)
    for w in WEIGHTS:
        ref = _cli(gen, by_key[(SEEDS[0], w)])
        for s in SEEDS[1:]:
            cli = _cli(gen, by_key[(s, w)])
            diff = {k for k in set(ref) | set(cli)
                    if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
            diff -= node_derived
            assert diff == {"--name", "--seed"}, (w, s, sorted(diff))
    # and the conditioned row differs from its seed's (1, 1) row in the
    # conditioning alone
    for s in SEEDS:
        ref = _cli(gen, by_key[(s, ("1", "1"))])
        cli = _cli(gen, _by_name(rows, f"orderonly_nn256_pref_s{s}"))
        diff = {k for k in set(ref) | set(cli)
                if ref.get(k, _MISSING) != cli.get(k, _MISSING)}
        assert diff == {"--name", "--preference-conditioned"}, (s, sorted(diff))


# ------------------------------------------------------------- 3. the nodes

def test_the_node_is_the_seed(gen, rows):
    """AGENTS.md: latency and memory are not comparable across GPU models.
    Three models carry this round, so all six configurations of one seed run
    on ONE node and the weight comparison never straddles two."""
    assert gen.ORDERONLY_NODES == NODES
    for a in rows:
        assert a["node"] == NODES[SEEDS.index(a["thesis_seed"])], a["name"]
    # THE INVARIANT: a seed is never split.  A node may carry more than one
    # seed (gpu14 does, since another group took every Blackwell node on
    # 2026-09-18), but no seed may span two nodes -- that would put a weight
    # comparison across two GPU models, which is the one thing this routing
    # exists to prevent.
    by_seed = {}
    for a in rows:
        by_seed.setdefault(a["thesis_seed"], set()).add(a["node"])
    assert set(by_seed) == set(SEEDS)
    for seed, nodes in by_seed.items():
        assert len(nodes) == 1, (seed, nodes)
    # and every seed carries its whole set: five weights and the conditioned
    # row, all six on that one node
    counts = {}
    for a in rows:
        counts[a["thesis_seed"]] = counts.get(a["thesis_seed"], 0) + 1
    assert set(counts.values()) == {len(WEIGHTS) + 1}
    # the node totals follow, and every node used is one of the cleared ones
    per_node = {}
    for a in rows:
        per_node[a["node"]] = per_node.get(a["node"], 0) + 1
    assert per_node == {"pgi15-gpu14": 12, "pgi15-gpu13": 6}


def test_every_order_only_job_is_a_cross_agent_per_node_singleton(gen, rows):
    """Orchestrator ruling 2026-09-18: `node-<node>`, not `thesis-<node>`.
    Singleton serializes jobs that share a NAME, three agents submit to
    pgi15-gpu14, and a name only one agent uses serializes nothing -- two
    jobs of this user on one node and the epilog kills both."""
    for a in rows:
        text = gen.render(a)
        assert a["job"] == f"node-{a['node']}", a["name"]
        assert f"#SBATCH -J node-{a['node']}\n" in text, a["name"]
        assert f"#SBATCH -J thesis-{a['node']}\n" not in text, a["name"]
        assert "#SBATCH --dependency=singleton\n" in text, a["name"]
        assert f"#SBATCH -w {a['node']}\n" in text, a["name"]
        assert (f"#SBATCH -o {gen.CAMPAIGN_RUNS}/{a['name']}_%j.log\n"
                in text), a["name"]
    # SINCE TICKET dsnn-dfw.65 the matrix carries the SAME name: a name only
    # the matrix used serialized the matrix against itself and let an
    # order-only row hold the same node.  Every thesis job of ours now shares
    # one name per node, which is what singleton needs to serialize them.
    for a in gen.thesis_arms():
        assert a["job"] == f"node-{a['node']}", a["name"]


def test_the_hardware_lines_follow_the_node(gen, rows):
    """gres, partition, CPUs and memory come from the NODE now, because this
    round spans three GPU models.  A --mem above the node's SLURM limit is
    never scheduled at all, which is why the memory is per node and not per
    GPU count."""
    assert gen.NODE_GRES_TYPE["pgi15-gpu13"] == "nvidia_geforce_rtx_4090"
    assert gen.NODE_GRES_TYPE["pgi15-gpu14"] == "nvidia_h100_80gb_hbm3"
    # pgi15-gpu8 joined 2026-09-19 (ticket dsnn-dfw.45); this round never
    # uses it, but the table it shares with gpu13/gpu14 now carries it too.
    assert gen.NODE_GPUS == {"pgi15-gpu8": 4, "pgi15-gpu13": 4,
                             "pgi15-gpu14": 8}
    assert gen.NODE_PARTITION == {"pgi15-gpu14": "pgi15-h100"}
    for a in rows:
        text = gen.render(a)
        node, gpus = a["node"], a["gpus"]
        assert gpus == gen.node_gpu_count(node), a["name"]
        assert f"#SBATCH -p {gen.node_partition(node)}\n" in text, a["name"]
        assert (f"#SBATCH --gres={gen.node_gres(node, gpus)}\n"
                in text), a["name"]
        assert f"#SBATCH -c {gen.node_cpus(node, gpus)}\n" in text, a["name"]
        assert f"#SBATCH --mem={gen.node_mem(node, gpus)}\n" in text, a["name"]
        # ONE trainer GPU, every other GPU a measure actor
        assert _cli(gen, a)["--ray-measure"] == str(gpus - 1), a["name"]
        assert _cli(gen, a)["--ray-measure"] == gen.THESIS_RAY_MEASURE[gpus]
    # the three models, spelled out
    assert gen.node_gres("pgi15-gpu16", 4).startswith(
        "gpu:nvidia_rtx_pro_6000_blackwell")
    assert gen.node_gres("pgi15-gpu13", 4) == "gpu:nvidia_geforce_rtx_4090:4"
    assert gen.node_gres("pgi15-gpu14", 8) == "gpu:nvidia_h100_80gb_hbm3:8"
    assert gen.node_partition("pgi15-gpu14") == "pgi15-h100"
    assert gen.node_partition("pgi15-gpu13") == "pgi15"


def test_the_node_whose_toolkit_is_outside_usr_local_puts_it_on_path(gen,
                                                                     rows):
    """Finding 03: a 12.8 nvlink refuses the venv's 12.9 cubins and every
    measurement degrades SILENTLY.  gpu14's matched pair is in the HPC SDK,
    which the measure-toolchain block does not search, so the arm puts it on
    PATH -- and the block still PROVES both versions, so a wrong path aborts
    72 instead of producing degraded numbers."""
    # pgi15-gpu8 joined 2026-09-19 (ticket dsnn-dfw.45); it carries the
    # recurrent order-only rows, tested in gen_fq_launchers_snnsweep_test.py.
    assert set(gen.NODE_CUDA_BIN) == {"pgi15-gpu14", "pgi15-gpu8"}
    bin_dir = gen.NODE_CUDA_BIN["pgi15-gpu14"]
    assert "12.9" in bin_dir and "13.2" not in bin_dir
    for a in rows:
        text = gen.render(a)
        export_line = f'export PATH="{bin_dir}:$PATH"'
        if a["node"] == "pgi15-gpu14":
            assert a["cuda_bin"] == bin_dir, a["name"]
            assert export_line in text, a["name"]
            # BEFORE the gate, or the gate reads the unpatched PATH
            assert (text.index(export_line)
                    < text.index("FQ_CUDA_WANT=")), a["name"]
        else:
            assert "cuda_bin" not in a, a["name"]
            assert bin_dir not in text, a["name"]
        # the gate itself is on, and abort is the mode, on every row
        assert _cli(gen, a)["--measure-toolchain-gate"] == "abort", a["name"]
        assert "exit 72" in text, a["name"]


def test_the_toolchain_block_still_searches_usr_local(gen):
    """A PLACEHOLDER THAT SURVIVES RENDERING IS A SILENT NODE-WIDE FAILURE.

    An earlier attempt at the gpu14 path put a @EXTRA_DIRS@ marker in the
    search glob and never substituted it.  Every launcher then globbed
    `@EXTRA_DIRS@/usr/local/cuda-*/bin`, which matches nothing, so the gate
    found no toolkit on ANY node and aborted 72 even where /usr/local carried
    the matched pair (jobs 66328 and 66329, pgi15-gpu16 and pgi15-gpu13).
    The glob is pinned literally, and no rendered launcher may carry an
    unsubstituted @NAME@ marker.
    """
    marker = re.compile(r"@[A-Z][A-Z0-9_]*@")
    for a in gen.ARMS:
        text = gen.render(a)
        if a["kind"] != "cpu" or a.get("needs_tool"):
            assert "\nfor d in /usr/local/cuda-*/bin; do\n" in text, a["name"]
        left = marker.findall(text)
        assert not left, (a["name"], sorted(set(left)))


# ------------------------------------------------------- 4. the environment

def test_no_order_only_launcher_exports_an_xla_flag_or_a_promoted_var(gen,
                                                                      rows):
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


def test_every_export_in_an_order_only_launcher_is_allowed(gen, rows):
    allowed = set(gen.THESIS_ENV_ALLOWED)
    for a in rows:
        exported = set(_EXPORT.findall(gen.render(a)))
        assert exported <= allowed, (a["name"], sorted(exported - allowed))
        # dsnn-dfw.264: DSNN_SHD_DIR is on the rows thesis_arm emits only;
        # a frozen round keeps the environment it ran with.
        assert "DSNN_SHD_DIR" not in exported, a["name"]
        assert set(gen.CAMPAIGN_ENV_ALLOWED) - {"DSNN_SHD_DIR"} <= exported, \
            a["name"]
        assert "ALPHAGRAD_NN_HIDDEN" in exported, a["name"]


def test_the_preflight_greps_the_swept_flags(gen, rows):
    """--lambda-cmp and --lambda-mem are what this round sweeps, so the
    launcher's layer-1 grep must name them rather than trust the default."""
    assert "--lambda-cmp" in gen.ORDERONLY_REQUIRED_FLAGS
    assert "--lambda-mem" in gen.ORDERONLY_REQUIRED_FLAGS
    texts = []
    for rel in gen.THESIS_FLAGS_FILES:
        with open(os.path.join(_ALPHAGRAD, rel)) as fh:
            texts.append(fh.read())
    src = "\n".join(texts)
    for flag in gen.ORDERONLY_REQUIRED_FLAGS:
        assert f'"{flag}"' in src, flag
    for a in rows:
        text = gen.render(a)
        assert " ".join(gen.THESIS_FLAGS_FILES) in text, a["name"]
        for flag in _cli(gen, a):
            assert f'"{flag}"' in src, (a["name"], flag)


# ---------------------------------------------------------- 5. the refusals

def test_the_helpers_refuse_a_row_outside_the_rulings(gen):
    assert gen.orderonly_run_name("1.5", "0.5", "250198") == \
        "orderonly_nn256_l1.5m0.5_s250198"
    assert gen.orderonly_pref_run_name("250199") == \
        "orderonly_nn256_pref_s250199"
    for bad in (("3", "0", "250197"), ("1", "1", "250200"),
                ("2", "2", "250197")):
        with pytest.raises(gen.CampaignRowError):
            gen.orderonly_run_name(*bad)
    for bad_seed in ("250200", "250201", "42"):
        with pytest.raises(gen.CampaignRowError):
            gen.orderonly_pref_run_name(bad_seed)
        with pytest.raises(gen.CampaignRowError):
            gen.orderonly_node(bad_seed)
    # a conditioned row that also names a fixed weight says two things
    with pytest.raises(gen.CampaignRowError):
        gen.orderonly_arm(seed="250197", lam_cmp="1", lam_mem="1", pref=True)
    # and a fixed-weight row that names neither
    with pytest.raises(gen.CampaignRowError):
        gen.orderonly_arm(seed="250197", pref=False)
    with pytest.raises(gen.CampaignRowError):
        gen.node_gpu_count("pgi15-gpu99")
    with pytest.raises(gen.CampaignRowError):
        gen.thesis_job_name("pgi15-gpu99")


# ---------------------------------------------------------- 6. ppo's argparse

def test_ppo_argparse_accepts_every_order_only_command_line(gen, rows):
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


def test_auto_stop_has_the_checkpoint_it_needs(gen, rows):
    from alphagrad.approx.common import auto_stop as _auto
    from alphagrad.approx.ppo import make_argparser
    for a in rows:
        ns = make_argparser().parse_args(gen.cli_tokens(a))
        _auto.check_auto_stop_args(ns)
        assert _auto.check_points(ns) == (250, 500), a["name"]


# -------------------------------------------------------- 7. the regression

def test_no_arm_that_existed_before_this_round_moved(gen):
    """The hardware lines became node-keyed.  Every arm that is NOT an
    order-only row renders the hardware it always did, so a live campaign is
    not invalidated by a tuning round added beside it."""
    for a in gen.ARMS:
        if a.get("orderonly") or a.get("orderonly_rsnn"):
            continue
        node, gpus = a["node"], a.get("gpus", 0)
        text = gen.render(a)
        # the fallbacks are the old expressions, exactly
        assert gen.node_partition(node) == (
            "pgi15-cpu" if node == "pgi15-cpu1" else "pgi15"), a["name"]
        if a.get("runtime", "home") == "scratch":
            assert gen.node_gres(node, gpus) == gen.blackwell_gres(gpus)
            assert gen.node_cpus(node, gpus) == gen.BLACKWELL_CPUS[gpus]
            assert gen.node_mem(node, gpus) == gen.BLACKWELL_MEM[gpus]
            assert (f"#SBATCH --gres={gen.blackwell_gres(gpus)}\n"
                    in text), a["name"]
            assert f"#SBATCH -c {gen.BLACKWELL_CPUS[gpus]}\n" in text
            # a row `thesis_arm` emits asks for the node's memory since the
            # owner rulings of 2026-09-25 (THESIS_ROW_MEM); every other row
            # keeps the old expression
            assert (f"#SBATCH --mem="
                    f"{a.get('mem') or gen.BLACKWELL_MEM[gpus]}\n") in text
            assert a.get("mem") in (None, gen.THESIS_ROW_MEM[gpus]), a["name"]
        # and nothing outside gpu14 gained a PATH prepend
        assert "hpc_sdk" not in text, a["name"]


def test_the_matrix_and_the_campaign_still_have_their_own_counts(gen):
    """The order-only rows are thesis arms -- they carry the singleton and the
    target shape -- but they are not matrix coordinates, and the matrix plus
    the two smoke runs must still be exactly what the rulings say: the 40
    core rows and the 40 recurrent rows once condC left (2026-09-25) and
    tbptt and window2 were deprecated (2026-09-26, dsnn-dfw.232), plus the 9
    defense rows of dsnn-dfw.231."""
    # dsnn-dfw.45 added a second order-only round (the recurrent target,
    # `orderonly_rsnn`), excluded here exactly as `orderonly` (NN256) is, the
    # 2026-09-19 ruling added the five-seed BASELINE (`orderonly_final`),
    # excluded the same way, and the same evening added the one-row TLM
    # order-only final (`orderonly_tlm_final`, owner: "run a TLM run for 1k
    # episodes") -- also a final row, not a matrix coordinate.
    matrix = [a for a in gen.thesis_arms()
              if not a.get("smoke") and not a.get("orderonly")
              and not a.get("orderonly_rsnn")
              and not a.get("orderonly_final")
              and not a.get("orderonly_tlm_final")
              and not a.get("paired") and not a.get("sweepl")
              and not a.get("sweepl2") and not a.get("sweepl3")]
    assert len(gen.thesis_core_arms()) == 40
    assert len(gen.thesis_snn_arms()) == 40
    assert len(gen.thesis_defense_arms()) == 9
    assert len(matrix) == 89
    assert len(gen.thesis_smoke_arms()) == 2
    assert len(gen.orderonly_arms()) == N_RUNS
    assert len(gen.orderonly_final_arms()) == FINAL_N_RUNS
    assert len(gen.orderonly_tlm_final_arms()) == 1
    assert all(a.get("thesis") for a in gen.orderonly_arms())
    # the matrix rows never carry --approx-profile none
    assert {_cli(gen, a)["--approx-profile"] for a in matrix} == {"all"}
    # block 1 is the 24 the owner authorised (2026-09-16, condC out since
    # 2026-09-25), and a tuning row is not one of them: a stray `sbatch` over
    # the block must not start one
    block = gen.thesis_block1_arms()
    assert len(block) == gen.THESIS_BLOCK1 == 24
    assert not any(a.get("orderonly") for a in block)
    assert not any(a.get("orderonly_rsnn") for a in block)


def test_the_target_node_switch_does_not_move_the_tuning_rows(gen):
    """THESIS_TARGET_NODES=1 pins the MATRIX to target-specific nodes.  The
    tuning round's node is its seed, which that switch must not touch --
    moving one seed's runs to another node would split a weight comparison
    across two GPU models."""
    import importlib.util as _ilu

    os.environ["THESIS_TARGET_NODES"] = "1"
    try:
        spec = _ilu.spec_from_file_location("gen_fq_launchers_tgt_oo", _GEN)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del os.environ["THESIS_TARGET_NODES"]
    for a in mod.orderonly_arms():
        assert a["node"] == NODES[SEEDS.index(a["thesis_seed"])], a["name"]


# =========  THE 5-SEED ORDER-ONLY BASELINE ON NN256 (owner 2026-09-19)  =====
# The FINAL row of this arm: the same order-only arm at the latency-only
# weight, five seeds, one Blackwell node each, NO --auto-stop, and the four
# block settings of 2026-09-19.  The owner's numbers are typed here as they
# are for the tuning round above.

FINAL_SEEDS = ("250197", "250198", "250199", "250200", "250201")
FINAL_WEIGHTS = ("2", "0")
#: pgi15-gpu17 excluded (dsnn-dfw.69: job 66740 aborted 72, no matched CUDA
#: 12.9 ptxas); pgi15-gpu19 released back to us 2026-09-20, so five seeds
#: now sit one per node over these five nodes.
FINAL_NODES = ("pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18", "pgi15-gpu19",
               "pgi15-gpu20")
FINAL_N_RUNS = 5


@pytest.fixture(scope="module")
def final_rows(gen):
    arms = gen.orderonly_final_arms()
    assert arms, "the generator emits no order-only baseline arm"
    return arms


def test_the_baseline_is_five_rows_named_for_the_weight_and_the_seed(
        gen, final_rows):
    assert len(final_rows) == FINAL_N_RUNS == gen.ORDERONLY_FINAL_RUNS
    assert gen.ORDERONLY_FINAL_SEEDS == FINAL_SEEDS
    assert gen.ORDERONLY_FINAL_WEIGHTS == FINAL_WEIGHTS
    lc, lm = FINAL_WEIGHTS
    want = {f"orderonly_nn256_final_l{lc}m{lm}_s{s}" for s in FINAL_SEEDS}
    assert {a["name"] for a in final_rows} == want
    for s in FINAL_SEEDS:
        assert gen.orderonly_final_run_name(s) in want
    with pytest.raises(gen.CampaignRowError):
        gen.orderonly_final_run_name("999999")


def test_the_baseline_is_the_order_only_arm_at_the_latency_weight(
        gen, final_rows):
    lc, lm = FINAL_WEIGHTS
    for a in final_rows:
        cli = _cli(gen, a)
        assert cli["--approx-profile"] == PROFILE, a["name"]
        assert cli["--fixed-order"] == ORDER, a["name"]
        assert cli["--lambda-cmp"] == lc and cli["--lambda-mem"] == lm
        assert cli["--reward-mode"] == "lagrangian", a["name"]
        assert cli["--quality-floor"] == TAU, a["name"]
        assert a["thesis_arm"] == gen.ORDERONLY_ARM == "C", a["name"]
        assert a["thesis_target"] == "nn256", a["name"]
        assert a["env"] == {"ALPHAGRAD_NN_HIDDEN": "256"}, a["name"]
        assert "--preference-conditioned" not in cli, a["name"]


def test_the_baseline_is_a_final_row_with_the_four_block_settings(
        gen, final_rows):
    """A FINAL row: all its episodes (2000 on NN256 since 2026-09-26), no
    --auto-stop, and the four settings of 2026-09-19 exactly as the A/B/C arms
    carry them."""
    for a in final_rows:
        cli = _cli(gen, a)
        assert cli["--episodes"] == FINAL_EPISODES, a["name"]
        assert cli["--checkpoint-every"] == CHECKPOINT_EVERY, a["name"]
        assert cli["--pareto-dump-every"] == PARETO_DUMP_EVERY, a["name"]
        assert cli["--plan-log"] == "auto", a["name"]
        assert "--auto-stop" not in cli, a["name"]
        assert cli["--paired-cost-floor"] == gen.THESIS_PAIRED_COST_FLOOR \
            == "byte", a["name"]
        assert cli["--mem-channel"] == gen.THESIS_MEM_CHANNEL == "watermark", \
            a["name"]
        assert cli["--lag-max"] == gen.THESIS_DUAL_LAMBDA_MAX == "64", a["name"]
        assert cli["--lag-min"] == gen.DUAL_LAMBDA_MIN == "12", a["name"]
        assert cli["--lag-init"] == gen.THESIS_LAMBDA_Q == "16", a["name"]
        text = gen.render(a)
        assert f"--wandb {gen.WANDB_MODE}" in text and gen.WANDB_MODE == "online"
        assert f"--wandb-project {gen.WANDB_PROJECT}" in text, a["name"]


def test_the_baseline_runs_one_seed_per_blackwell_node(gen, final_rows):
    """Latency and memory are not comparable across GPU models, so a seed is
    measured on ONE node; five seeds over the five cleared Blackwell nodes
    (pgi15-gpu17 excluded, dsnn-dfw.69), one seed each.

    EVERY seed renders the SAME profile -- 4 GPUs, --ray-measure 3 -- even on
    the two 8-GPU nodes (owner ruling 2026-09-20): a baseline whose fifth
    seed measured with seven actors while the others measured with three was
    not five samples of one distribution."""
    seen = {}
    for a in final_rows:
        node = a["node"]
        i = FINAL_SEEDS.index(a["thesis_seed"])
        assert node == FINAL_NODES[i % len(FINAL_NODES)]
        assert node in gen.THESIS_NODES_ALL, a["name"]
        assert node != "pgi15-gpu17", a["name"]
        seen.setdefault(node, []).append(a["name"])
        gpus = gen.thesis_row_gpus(a["thesis_target"], node)
        assert gpus == 4, a["name"]
        assert a["gpus"] == gpus, a["name"]
        cli = _cli(gen, a)
        assert cli["--ray-measure"] == gen.THESIS_RAY_MEASURE[gpus] \
            == str(gpus - 1), a["name"]
        text = gen.render(a)
        assert f"#SBATCH -w {node}\n" in text, a["name"]
        assert f"#SBATCH -J node-{node}\n" in text, a["name"]
        assert "#SBATCH --dependency=singleton\n" in text, a["name"]
        assert a["job"] == gen.orderonly_job_name(node) == f"node-{node}"
        assert f"#SBATCH --gres={gen.blackwell_gres(gpus)}\n" in text, a["name"]
    assert sorted(seen) == sorted(FINAL_NODES)
    counts = sorted(len(v) for v in seen.values())
    assert counts == [1, 1, 1, 1, 1], seen
    with pytest.raises(gen.CampaignRowError):
        gen.orderonly_final_node("999999")


def test_the_baseline_is_not_a_matrix_row_and_not_a_tuning_row(
        gen, final_rows, rows):
    """It is its own kind: out of the 40 core rows, out of block 1, and out
    of the three-seed tuning round it is the baseline for."""
    names = {a["name"] for a in final_rows}
    assert not (names & {a["name"] for a in rows})
    assert not any(a.get("orderonly") for a in final_rows)
    assert not any(a.get("orderonly_rsnn") for a in final_rows)
    assert not (names & {a["name"] for a in gen.thesis_core_arms()})
    assert len(gen.thesis_core_arms()) == 40
    assert not any(a.get("orderonly_final") for a in gen.thesis_block1_arms())
    assert len(gen.thesis_block1_arms()) == gen.THESIS_BLOCK1 == 24
    # it IS a thesis arm, so every sweep over thesis arms reaches it
    assert names <= {a["name"] for a in gen.thesis_arms()}


def test_the_target_node_switch_does_not_move_the_baseline(gen):
    """THESIS_TARGET_NODES=1 pins the MATRIX to target-specific nodes. The
    baseline's node is its seed, exactly as the tuning round's is, and that
    switch must not touch it: moving a seed would put two seeds on one GPU
    model and leave another model unmeasured."""
    import importlib.util as _ilu

    os.environ["THESIS_TARGET_NODES"] = "1"
    try:
        spec = _ilu.spec_from_file_location("gen_fq_launchers_tgt_fin", _GEN)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del os.environ["THESIS_TARGET_NODES"]
    for a in mod.orderonly_final_arms():
        i = FINAL_SEEDS.index(a["thesis_seed"])
        assert a["node"] == FINAL_NODES[i % len(FINAL_NODES)]


# ===============  ONE ORDER-ONLY FINAL ROW ON TLM (2026-09-19 evening)  ====
# owner: "run a TLM run for 1k episodes".  One named row, not a family: the
# order-only arm's shape on TransformerLM, at (--lambda-cmp 1, --lambda-mem
# 1), a final row (no --auto-stop) on pgi15-gpu15.

TLM_FINAL_NAME = "orderonly_tlm_final_l1m1_s250197"
TLM_FINAL_SEED = "250197"
TLM_FINAL_WEIGHTS = ("1", "1")
TLM_FINAL_NODE = "pgi15-gpu15"


@pytest.fixture(scope="module")
def tlm_final_row(gen):
    arms = gen.orderonly_tlm_final_arms()
    assert len(arms) == 1, "the generator emits more or fewer than one row"
    return arms[0]


def test_the_tlm_final_row_is_named_and_shaped_as_ruled(gen, tlm_final_row):
    a = tlm_final_row
    assert a["name"] == TLM_FINAL_NAME == gen.ORDERONLY_TLM_FINAL_NAME
    assert gen.ORDERONLY_TLM_FINAL_SEED == TLM_FINAL_SEED
    assert gen.ORDERONLY_TLM_FINAL_WEIGHTS == TLM_FINAL_WEIGHTS
    assert gen.ORDERONLY_TLM_FINAL_NODE == TLM_FINAL_NODE
    cli = _cli(gen, a)
    assert cli["--example"] == "TransformerLM", a["name"]
    assert cli["--dataset"] == "wikitext2", a["name"]
    assert cli["--approx-profile"] == PROFILE, a["name"]
    assert cli["--fixed-order"] == ORDER, a["name"]
    lc, lm = TLM_FINAL_WEIGHTS
    assert cli["--lambda-cmp"] == lc and cli["--lambda-mem"] == lm, a["name"]
    assert a["thesis_arm"] == gen.ORDERONLY_ARM == "C", a["name"]
    assert a["thesis_target"] == "tlm", a["name"]
    assert a["thesis_seed"] == TLM_FINAL_SEED, a["name"]
    assert a["orderonly_weights"] == (lc, lm), a["name"]
    assert "--preference-conditioned" not in cli, a["name"]
    assert cli["--reward-mode"] == "lagrangian", a["name"]
    assert cli["--quality-floor"] == TAU, a["name"]


def test_the_tlm_final_row_carries_the_tlm_env_and_the_block_settings(
        gen, tlm_final_row):
    a = tlm_final_row
    text = gen.render(a)
    assert "export ALPHAGRAD_TLM_SEQ=" in text, a["name"]
    assert "export ALPHAGRAD_TLM_DMODEL=" in text, a["name"]
    assert "export ALPHAGRAD_TLM_VOCAB=" in text, a["name"]
    cli = _cli(gen, a)
    assert cli["--episodes"] == EPISODES, a["name"]
    assert cli["--checkpoint-every"] == CHECKPOINT_EVERY, a["name"]
    assert cli["--pareto-dump-every"] == PARETO_DUMP_EVERY, a["name"]
    assert cli["--plan-log"] == "auto", a["name"]
    assert "--auto-stop" not in cli, a["name"]
    assert cli["--paired-cost-floor"] == gen.THESIS_PAIRED_COST_FLOOR \
        == "byte", a["name"]
    assert cli["--mem-channel"] == gen.THESIS_MEM_CHANNEL == "watermark", \
        a["name"]
    assert cli["--lag-max"] == gen.THESIS_DUAL_LAMBDA_MAX == "64", a["name"]
    assert f"--wandb {gen.WANDB_MODE}" in text and gen.WANDB_MODE == "online"
    assert f"--wandb-project {gen.WANDB_PROJECT}" in text, a["name"]


def test_the_tlm_final_row_is_on_gpu15_with_the_singleton(gen, tlm_final_row):
    a = tlm_final_row
    node = TLM_FINAL_NODE
    assert node in gen.THESIS_NODES_ALL, a["name"]
    assert a["node"] == node, a["name"]
    gpus = gen.THESIS_NODE_GPUS[node]
    assert a["gpus"] == gpus, a["name"]
    cli = _cli(gen, a)
    assert cli["--ray-measure"] == gen.THESIS_RAY_MEASURE[gpus] \
        == str(gpus - 1), a["name"]
    text = gen.render(a)
    assert f"#SBATCH -w {node}\n" in text, a["name"]
    assert f"#SBATCH -J node-{node}\n" in text, a["name"]
    assert "#SBATCH --dependency=singleton\n" in text, a["name"]
    assert a["job"] == gen.orderonly_job_name(node) == f"node-{node}"
    assert f"#SBATCH --gres={gen.blackwell_gres(gpus)}\n" in text, a["name"]
    assert gen._bash_n(text) is None, a["name"]


def test_the_tlm_final_row_is_not_a_matrix_row_and_not_a_tuning_row(
        gen, tlm_final_row, rows, final_rows):
    a = tlm_final_row
    assert a["name"] not in {r["name"] for r in rows}
    assert a["name"] not in {r["name"] for r in final_rows}
    assert not a.get("orderonly") and not a.get("orderonly_final") \
        and not a.get("orderonly_rsnn"), a["name"]
    assert a["name"] not in {r["name"] for r in gen.thesis_core_arms()}
    assert len(gen.thesis_core_arms()) == 40
    assert not any(a2.get("orderonly_tlm_final")
                  for a2 in gen.thesis_block1_arms())
    assert len(gen.thesis_block1_arms()) == gen.THESIS_BLOCK1 == 24
    assert a["name"] in {r["name"] for r in gen.thesis_arms()}
