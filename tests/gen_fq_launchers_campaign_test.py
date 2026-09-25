"""Ticket .43 -- the campaign arms (phases 1-5, tickets .50-.54) on the
launcher generator, regenerated 2026-09-13 under the owner's rulings.

Launchers are generated (tools/gen_fq_launchers.py), never hand-edited, so
the campaign contract is pinned on the generator:

  1. THE PHASE-1 TABLE (ticket .50): SKIP-only, Reduce-only, Quant-only,
     Diag-only, order-only (profile none, --fixed-order free), free (all,
     free); the all-rev x2 arms are gone; every arm comes from PHASE1_TABLE.
  2. THE WIDTH: the face head's logit count in every campaign header is
     head_layout(APPROX_ADD).width, and no rendered launcher says "94 logits".
  3. THE REQUIRED FLAGS on every campaign arm: --fixed-order markowitz
     (free on the order arms), --approx-add lossless and nothing else,
     --face-none-bias 4, --mem-channel temp, --quality-metric grad_cosine,
     --cost-form paired-log, --reward-mode additive, --terminal-rewards-only,
     --ray-measure 1 --ray-measure-timeout 600, --per-face-masks with
     --face-actions, --gate-winners-table.
  4. THE ENVIRONMENT: no promoted env var, no XLA_* flag, no
     ALPHAGRAD_FORCE_REV_ORDER; every `export` is in CAMPAIGN_ENV_ALLOWED.
     The one JAX_* exception (owner ruling 2026-09-14, small fixes #3) is
     the four-line per-node persistent compile cache in JAX_CACHE_ENV; no
     other JAX_* export is allowed. The no-flag knobs are exported AND
     named in the header's TODO block.
  5. THE HARDWARE: one 8-GPU Blackwell job per node on gpu19/gpu20, -c 128,
     the /Scratch stack of finding 57 (no ~/dsnn, no uv run).
  6. THE GATE TELEMETRY (ticket .45): ppo.py has no switch for it; the
     launcher passes G1's input.
  7. campaign_arm RAISES (CampaignRowError) on a row outside the rulings.
  8. ppo.py's own argparse accepts every campaign command line.
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
_PPO = os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "ppo.py")

# THE APPROVED REWARD, PINNED (owner ruling 2026-09-13, priced in finding
# 63).  These are the numbers the whole campaign is judged at; a change to
# either must be a deliberate edit HERE as well as in the generator.  The
# token is in every arm NAME because an arm called `lq5` that passes
# --lambda-acc 16 -- or one called lq16 with no quality floor -- is exactly
# the drift that made this morning's launchers wrong.
LAMBDA_Q = "16"
TAU = "0.90"
REWARD_TOKEN = "hinge_tau09_lq16"

PHASE1_NAMES = [
    f"p1a_skip_{REWARD_TOKEN}",
    f"p1b_reduce_{REWARD_TOKEN}",
    f"p1c_quant_{REWARD_TOKEN}",
    f"p1d_diag_{REWARD_TOKEN}",
    f"p1e_none_free_{REWARD_TOKEN}",
    f"p1f_all_free_{REWARD_TOKEN}",
]
PHASE1_PROFILES = ["skip", "reduce", "quant", "diag", "none", "all"]
_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]*\}")
_EXPORT = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", re.M)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def campaign(gen):
    arms = gen.campaign_arms()
    assert arms, "the generator emits no campaign arm"
    return arms


@pytest.fixture(scope="module")
def flag_sources(gen):
    texts = []
    for rel in gen.REQUIRED_FLAGS_FILES:
        with open(os.path.join(_ALPHAGRAD, rel)) as fh:
            texts.append(fh.read())
    return "\n".join(texts)


def _cli(gen, a) -> dict:
    return dict(gen._merge_cli(a.get("cli", {})))


def _by_name(campaign, name):
    return next(a for a in campaign if a["name"] == name)


def _tokens(gen, a, profile="skip") -> list[str]:
    """The arm's argv with shell placeholders substituted."""
    return [_PLACEHOLDER.sub(profile, t) for t in gen.cli_tokens(a)]


def _bash_n(text: str) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as fh:
        fh.write(text)
    try:
        chk = subprocess.run(["bash", "-n", fh.name],
                             capture_output=True, text=True)
        return None if chk.returncode == 0 else chk.stderr
    finally:
        os.unlink(fh.name)


# ------------------------------------------------------------ 1. the table

def test_the_phase1_table_is_the_one_ticket_50_asks_for(gen, campaign):
    p1 = [a for a in campaign if a["phase"] == 1]
    assert [a["name"] for a in p1] == PHASE1_NAMES
    assert [_cli(gen, a)["--approx-profile"] for a in p1] == PHASE1_PROFILES
    # one table: the rows ARE the arms, in order
    assert [r[0] for r in gen.PHASE1_TABLE] == list("abcdef")
    assert [r[1] for r in gen.PHASE1_TABLE] == PHASE1_PROFILES
    assert [r[2] for r in gen.PHASE1_TABLE] == ["markowitz"] * 4 + ["free"] * 2
    # the all-rev x2 arms (the .56 lossy/lossless pair) are gone
    assert not any("_all_lq" in a["name"] for a in p1)
    assert not any("addloss" in a["name"] for a in campaign)
    phases = sorted({a["phase"] for a in campaign})
    assert phases == [1, 2, 3, 4, 5]
    assert len([a for a in campaign if a["phase"] == 2]) == 2
    assert len([a for a in campaign if a["phase"] == 3]) == 3
    assert len([a for a in campaign if a["phase"] == 4]) == 1
    assert len([a for a in campaign if a["phase"] == 5]) == 5
    assert len(campaign) == 17
    assert len({a["name"] for a in campaign}) == len(campaign)


def test_no_campaign_arm_is_held(gen, campaign):
    # Diag-only is a LIVE arm under the Markowitz order (finding 59); the
    # hold on ticket .25 is named in its DEPENDS line, not enforced.
    for a in campaign:
        assert not a.get("held"), a["name"]
        assert "ABORT(73)" not in gen.render(a), a["name"]
    diag = _by_name(campaign, f"p1d_diag_{REWARD_TOKEN}")
    assert ".25" in diag["depends"]


def test_every_campaign_arm_renders_to_valid_bash(gen, campaign):
    for a in campaign:
        text = gen.render(a)
        assert text.startswith("#!/bin/bash\n"), a["name"]
        err = _bash_n(text)
        assert err is None, (a["name"], err)


# ------------------------------------------------------------ 2. the width

def test_the_face_head_width_is_derived_from_head_layout(gen, campaign):
    sys.path.insert(0, os.path.join(_ALPHAGRAD, "src"))
    from alphagrad.approx.unified_face_head import head_layout
    from alphagrad.approx.common.masks import FACE_QUANT_DTYPES
    assert gen.APPROX_ADD == "lossless"
    assert gen.FACE_HEAD_WIDTH == head_layout(gen.APPROX_ADD).width
    assert gen.FACE_QUANT_DTYPES == tuple(FACE_QUANT_DTYPES)
    assert len(gen.FACE_QUANT_DTYPES) == 2
    # the head is 89 wide since 2026-09-23 (skip, the quant bit, three
    # 29-wide slots); the number in the launchers is the library's, not the
    # generator's
    assert gen.FACE_HEAD_WIDTH == 2 + 3 * 29
    for a in campaign:
        text = gen.render(a)
        assert f"{gen.FACE_HEAD_WIDTH} logits" in text, a["name"]
        for dt in FACE_QUANT_DTYPES:
            assert dt in text, (a["name"], dt)
    for a in gen.ARMS:
        assert "94 logits" not in gen.render(a), a["name"]
    src = open(_GEN).read()
    assert "94 logits" not in src
    assert "103 logits" not in src


def test_the_width_derivation_raises_instead_of_falling_back(gen):
    with pytest.raises(ValueError):
        gen._face_head_geometry("no-such-approx-add")


# -------------------------------------------------- 3. the required flags

def test_required_flags_are_all_defined_where_the_preflight_greps(
        gen, flag_sources):
    # The pre-flight does `grep -qF -- "\"$F\"" $FLAGSRC`; this is that grep.
    missing = [f for f in gen.REQUIRED_FLAGS if f'"{f}"' not in flag_sources]
    assert not missing, missing
    for f in ("--approx-profile", "--cost-form", "--quality-floor",
              "--mem-channel", "--gate-winners-table", "--fixed-order",
              "--ray-measure", "--ray-measure-timeout", "--measure-pipeline",
              # DATA PARALLELISM OVER ENVIRONMENTS (owner ruling 2026-09-15):
              # an arm that inherited a default would roll out a different
              # number of environments than its --num-envs implies.
              "--rollout-shards",
              "--tokenize-where", "--face-wire-faces",
              "--per-face-masks",
              "--face-none-bias", "--approx-add", "--terminal-rewards-only"):
        assert f in gen.REQUIRED_FLAGS, f


def test_every_flag_a_campaign_arm_passes_is_defined(gen, campaign,
                                                     flag_sources):
    for a in campaign:
        flags = [t for t in _tokens(gen, a) if t.startswith("--")]
        undefined = [f for f in flags if f'"{f}"' not in flag_sources]
        assert not undefined, (a["name"], undefined)
        text = gen.render(a)
        # The launcher's own pre-flight greps the same two files.
        assert 'FLAGSRC="src/alphagrad/approx/ppo.py ' \
               'src/alphagrad/approx/common/gate_telemetry.py"' in text, a["name"]
        for f in gen.REQUIRED_FLAGS:
            assert f" {f} " in text or f" {f};" in text, (a["name"], f)


def test_every_campaign_arm_carries_the_required_flags(gen, campaign):
    for a in campaign:
        toks = _tokens(gen, a)
        cli = _cli(gen, a)
        joined = " " + " ".join(toks) + " "
        for frag in (" --example TransformerLM ", " --exec-on-gpu ",
                     " --measure-latency ", " --latency-inner-reps 50 ",
                     # THE PAIRED REFERENCE'S OWN BUDGET (owner ruling
                     # 2026-09-14), named explicitly on every training arm.
                     # The inner reps stay SHARED at 50, above.
                     " --ref-num-data-points 5 ",
                     " --ref-reps-per-point 32 ",
                     # THE PER-PLAN TIME BUDGET (owner ruling 2026-09-14).
                     # --num-data-points x --reps-per-point is the CAP on
                     # the candidate's timed windows now, and these two say
                     # how many of them a plan of a given cost earns.
                     " --measure-budget-secs 1.0 ",
                     " --measure-window-secs 0.05 ",
                     " --cmp-type latency ", " --mem-type peak_memory ",
                     " --cost-form paired-log ", " --mem-channel temp ",
                     " --quality-metric grad_cosine ",
                     " --discount 1.0 ", " --gae-lambda 1.0 ",
                     " --terminal-rewards-only ",
                     f" --approx-add {gen.APPROX_ADD} ",
                     f" --face-none-bias {gen.FACE_NONE_BIAS_MVP} ",
                     f" --scale-face-head {gen.SCALE_FACE_HEAD_MVP} ",
                     f" --face-logit-clamp {gen.FACE_LOGIT_CLAMP_MVP} ",
                     " --face-actions ", " --per-face-masks ",
                     " --unified-face-head ", " --live-faces ",
                     " --dynamic-substeps ", " --set-pointer ",
                     " --incremental-encode ",
                     f" --ray-measure {gen.CAMPAIGN_RAY_MEASURE} ",
                     f" --ray-measure-timeout {gen.CAMPAIGN_RAY_MEASURE_TIMEOUT} ",
                     # PIPELINE THE MEASUREMENT (owner ruling 2026-09-14).
                     f" --measure-pipeline {gen.CAMPAIGN_MEASURE_PIPELINE} ",
                     # HOW MANY DEVICES ROLL AN EPISODE OUT (owner ruling
                     # 2026-09-15), and --num-envs is PER SHARD from that
                     # ruling on.
                     f" --rollout-shards {gen.CAMPAIGN_ROLLOUT_SHARDS} ",
                     " --num-envs 16 ",
                     # THE PER-STEP TOKENIZATION IS OFF THE MEASURE ACTORS
                     # (owner ruling 2026-09-15), which is what lets the
                     # pipeline hide a measurement behind the NEXT rollout.
                     f" --tokenize-where {gen.CAMPAIGN_TOKENIZE_WHERE} ",
                     # AND THE FACE WIRE CARRIES THE COLUMNS THAT ARE USED.
                     f" --face-wire-faces {gen.CAMPAIGN_FACE_WIRE_FACES} ",
                     " --plan-log auto ", " --episodes 250 ",
                     " --measure-toolchain-gate abort ",
                     " --reduce-axis-space physical ",
                     f" --gate-winners-table "
                     f"{gen.CAMPAIGN_GATE_WINNERS_TABLES[cli['--fixed-order']]} ",
                     f" --gate-offline-contrast "
                     f"{gen.GATE_OFFLINE_CONTRAST[cli['--fixed-order']]} ",
                     " --lambda-cmp 1 ", " --lambda-mem 1 ",
                     " --wandb online "):
            assert frag in joined, (a["name"], frag)
        assert gen.FACE_NONE_BIAS_MVP == "4"
        assert cli["--approx-add"] == "lossless", a["name"]
        assert toks.count("--approx-add") == 1, a["name"]
        assert cli["--rewards"] in ("cmp mem acc", "cmp acc", "mem acc")
        if cli.get("--reward-mode") != "lagrangian":
            assert " --reward-mode additive " in joined, a["name"]
        if cli.get("--advantage-norm") != "popart":
            assert " --advantage-norm none " in joined, a["name"]
        # one seed per arm, one order per arm
        assert toks.count("--seed") == 1, a["name"]
        assert toks.count("--fixed-order") == 1, a["name"]
        # no quality gate anywhere in the campaign
        assert "QUALITY_GATE" not in gen.render(a), a["name"]
        # registered prediction and falsifier in the header
        text = gen.render(a)
        assert "# REGISTERED PREDICTION" in text, a["name"]
        assert "# FALSIFICATION CRITERION:" in text, a["name"]
        assert "REGISTERED BEFORE THE RUN" in text, a["name"]
        assert "ABORT(72)" in text, a["name"]   # the toolchain gate (.21)


def test_the_order_is_markowitz_except_on_the_two_order_arms(gen, campaign):
    for a in campaign:
        cli = _cli(gen, a)
        text = gen.render(a)
        assert "ALPHAGRAD_FORCE_REV_ORDER" not in text, a["name"]
        if a["name"] in (f"p1e_none_free_{REWARD_TOKEN}", f"p1f_all_free_{REWARD_TOKEN}"):
            assert cli["--fixed-order"] == "free", a["name"]
            assert a["time"] == "24:00:00", a["name"]
            assert "_free_" in a["name"]
        else:
            assert cli["--fixed-order"] == "markowitz", a["name"]
            assert "_free" not in a["name"] and "_reverse" not in a["name"]
    order_only = _cli(gen, _by_name(campaign, f"p1e_none_free_{REWARD_TOKEN}"))
    assert order_only["--approx-profile"] == "none"
    assert "--no-approx-head" not in order_only   # the profile IS the switch
    assert "--exact" not in order_only
    assert _cli(gen, _by_name(campaign, f"p1f_all_free_{REWARD_TOKEN}"))["--approx-profile"] == "all"


def test_the_approved_reward_is_on_every_arm(gen, campaign):
    """Owner ruling 2026-09-13, priced in finding 63: P1 hinge,
    --quality-floor 0.90, --lambda-acc 16, --lambda-cmp 1, --lambda-mem 1.

    The generator's own constants are pinned against this test's copies, so
    changing one without the other fails here rather than shipping 17
    launchers that run a reward nobody approved.
    """
    assert gen.LAMBDA_Q_MVP == LAMBDA_Q
    assert gen.QUALITY_FLOOR_TAU == TAU
    for a in campaign:
        cli = _cli(gen, a)
        assert cli["--lambda-cmp"] == "1", a["name"]
        assert cli["--lambda-mem"] == "1", a["name"]
        assert cli["--cost-form"] == "paired-log", a["name"]
        assert cli["--mem-channel"] == "temp", a["name"]
        if a["phase"] == 3 and a["name"].endswith(f"pref_lq{LAMBDA_Q}"):
            # P0 is the ONE arm that lifts the floor: that IS its question.
            assert "--quality-floor" not in cli, a["name"]
            continue
        assert cli["--quality-floor"] == TAU, a["name"]
        if cli.get("--reward-mode") == "lagrangian":
            # lambda is the dual variable here; --lambda-acc is ignored.
            assert cli["--lag-eta"] == gen.DUAL_ETA, a["name"]
            assert cli["--lag-min"] == gen.DUAL_LAMBDA_MIN, a["name"]
            assert cli["--lag-max"] == gen.DUAL_LAMBDA_MAX, a["name"]
        else:
            assert cli["--lambda-acc"] == LAMBDA_Q, a["name"]


def test_no_arm_carries_the_retired_lambda_five(gen, campaign):
    """The launchers this generator emitted on the morning of 2026-09-13
    passed --lambda-acc 5 with no quality floor.  Finding 63 prices that at
    contrast -20.6 on the Markowitz order: the skip-everything absorber
    outscores every honest plan.  Neither the flag nor the name may come
    back."""
    for a in campaign:
        cli = _cli(gen, a)
        assert cli.get("--lambda-acc") != "5", a["name"]
        assert "lq5" not in a["name"], a["name"]
        assert "lq5" not in gen.render(a), a["name"]


def test_gate_g1_points_at_the_sweep64_table_for_this_arms_order(gen, campaign):
    """The staged copy at campaign/sweep41/winners.csv is byte-for-byte the
    sweep64 MARKOWITZ table under a name that says sweep41.  A reverse-order
    arm pointed at it would report a recovery against the other order's
    winners."""
    assert "sweep41" not in str(gen.CAMPAIGN_GATE_WINNERS_TABLES)
    for a in campaign:
        cli = _cli(gen, a)
        want = gen.CAMPAIGN_GATE_WINNERS_TABLES[cli["--fixed-order"]]
        assert cli["--gate-winners-table"] == want, a["name"]
        assert "sweep64" in want
        assert "sweep41" not in gen.render(a), a["name"]


def test_gate_g6_carries_the_pre_run_contrast_of_this_arms_order(gen, campaign):
    """G6's number cannot be measured by the run it judges, so the launcher
    carries it (finding 63: +0.19 on Markowitz, 0.00 on reverse).  Without
    it gate/g6/offline_contrast reads NaN for 250 episodes."""
    for a in campaign:
        cli = _cli(gen, a)
        want = gen.GATE_OFFLINE_CONTRAST[cli["--fixed-order"]]
        assert cli["--gate-offline-contrast"] == want, a["name"]
        assert float(want) >= 0.0


def test_every_arm_passes_the_approved_cost_floor(gen, campaign):
    """--paired-cost-floor landed in ppo.make_argparser (alphagrad b2c89170)
    and finding 63's contrast assumes its `reference` setting. Every arm
    passes it EXPLICITLY -- a launcher must record the reward it ran under,
    not inherit a default that moved once already -- and every header says
    what the old floor would have priced."""
    assert gen.PAIRED_COST_FLOOR == "reference"
    for a in campaign:
        cli = _cli(gen, a)
        assert cli["--paired-cost-floor"] == "reference", a["name"]
        text = gen.render(a)
        assert "--paired-cost-floor reference" in text, a["name"]
        assert "-13.4" in text, (a["name"], "the header must price the old floor")


def test_the_cost_floor_flag_exists_in_the_trainer(gen):
    """The generator may not emit a flag the trainer would refuse."""
    import subprocess
    import sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import alphagrad.approx.ppo as P;"
         "a=P.make_argparser().parse_args(['--paired-cost-floor','reference']);"
         "print(a.paired_cost_floor)"],
        capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("reference")


# ------------------------------------------------- 4. the environment

def test_no_campaign_arm_exports_a_promoted_var_or_an_xla_flag(gen, campaign):
    for a in campaign:
        text = gen.render(a)
        assert a.get("env", {}) == {}, a["name"]
        for var in gen.PROMOTED_ENV_VARS:
            assert f"{var}=" not in text, (a["name"], var)
        for frag in ("QUALITY_GATE_MIN", "NEW_SLOT_JOIN", "GRAPHAX_ALLOW_PARTIAL_ORDER",
                     "ALPHAGRAD_FORCE_REV_ORDER", "GRAPHAX_PLANNER_EXACT",
                     "GRAPHAX_QUANT_PULLDOWN", "ALPHAGRAD_MAX_FACES"):
            assert frag not in text, (a["name"], frag)
        # no XLA memory flag, no XLA flag at all, no JAX cache/platform var --
        # EXCEPT the per-node persistent JAX compile cache (owner ruling
        # 2026-09-14, small fixes #3): exactly the mkdir and the four
        # JAX_CACHE_ENV exports are allowed, verbatim, nothing else.
        _jax_cache_exports = {f"export {k}={v}" for k, v in gen.JAX_CACHE_ENV}
        _jax_cache_mkdir = f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}"
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            if stripped in _jax_cache_exports or stripped == _jax_cache_mkdir:
                continue
            assert "XLA_" not in line, (a["name"], line)
            assert "export JAX_" not in line, (a["name"], line)
            assert "JAX_COMPILATION_CACHE_DIR" not in line, (a["name"], line)


def test_jax_cache_is_per_node_and_the_autotune_race_is_disabled(gen):
    """Owner ruling 2026-09-14 (small fixes #3): one persistent JAX compile
    cache dir per NODE (the nodes differ in GPUs/CPUs, and it must survive
    across job submissions -- a per-$SLURM_JOB_ID dir never warms).

    A live canary (job 65500 on gpu19, 7 measure actors sharing one node's
    cache dir) showed WHY a shared per-node dir is not free: one actor died
    in `evaluate` with a NOT_FOUND on
    `xla_gpu_per_fusion_autotune_cache_dir/tmp/tmp_per_fusion_cache__..._textproto`.
    This JAX build (0.10.2.dev0+selfbuilt,
    /Scratch/assmuth/t57/stack/venv) defaults
    `jax_persistent_cache_enable_xla_caches` to
    'xla_gpu_per_fusion_autotune_cache_dir' (jax/_src/config.py ~1427), and
    `jax/_src/compiler.py`'s `get_compile_options` (~262-283) arms that
    subcache UPDATE-mode for `distributed.global_state.process_id == 0` --
    which EVERY independent measure-actor process defaults to, since none
    of them joins one `jax.distributed` cluster. All N actors on a node
    then race UPDATE-mode writes to the same per-fusion cache files. The
    fix keeps ONE shared executable-cache directory per node (so it still
    warms across jobs, this ticket's actual ask) but disables the raced
    autotune subcache outright, rather than routing it per actor PID."""
    want = dict(gen.JAX_CACHE_ENV)
    # per NODE: keyed by hostname, never by the job id (would defeat
    # persistence across submissions and was never the race's cause anyway).
    assert "hostname" in gen.JAX_CACHE_DIR_EXPR
    assert "SLURM_JOB_ID" not in gen.JAX_CACHE_DIR_EXPR
    assert want["JAX_COMPILATION_CACHE_DIR"] == gen.JAX_CACHE_DIR_EXPR
    # "any compile time / any artifact size" -- both default otherwise.
    assert want["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
    assert want["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "0"
    # the autotune race: disabled outright (empty), not given a per-process
    # subdirectory -- a per-PID path would also defeat the per-node
    # executable-cache sharing this ticket asks for.
    assert want["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] == ""

    # Every arm the generator renders (wave, cpu, tool, probe, campaign --
    # not just the campaign fixture) runs python and must carry the SAME
    # cache dir, created once, before python starts.  The one exception
    # (owner ruling 2026-09-23): a thesis row under --fixed-order free
    # exports none of it, since every plan is a new program there.
    for a in gen.ARMS:
        text = gen.render(a)
        if (a.get("jax_cache_fixed_order_only") and dict(gen._merge_cli(
                a.get("cli", {}))).get("--fixed-order") == "free"):
            assert "export JAX_" not in text, a["name"]
            continue
        assert f"mkdir -p {gen.JAX_CACHE_DIR_EXPR}" in text, a["name"]
        for k, v in gen.JAX_CACHE_ENV:
            assert f"export {k}={v}\n" in text, (a["name"], k)
        # ONE cache directory for the whole node -- not one per actor PID --
        # so each export line appears exactly once per rendered launcher.
        for k, _ in gen.JAX_CACHE_ENV:
            assert text.count(f"export {k}=") == 1, (a["name"], k)


def test_every_export_in_a_campaign_launcher_is_allowed(gen, campaign):
    allowed = set(gen.CAMPAIGN_ENV_ALLOWED)
    for a in campaign:
        text = gen.render(a)
        exported = set(_EXPORT.findall(text))
        assert exported <= allowed, (a["name"], sorted(exported - allowed))
        # and the allowed set is exactly what is exported (nothing dormant)
        assert exported == allowed, (a["name"], sorted(allowed - exported))
        for k, v in gen.CAMPAIGN_ENV:
            assert f"export {k}={v}\n" in text, (a["name"], k)
    # the three measurement-plumbing vars and the TLM shape, nothing else of
    # the ALPHAGRAD_* / RAY_* kind
    campaign_env = {k for k, _ in gen.CAMPAIGN_ENV}
    assert campaign_env == {"ALPHAGRAD_TLM_SEQ", "ALPHAGRAD_TLM_DMODEL",
                            "ALPHAGRAD_TLM_VOCAB", "ALPHAGRAD_BATCHED_CALLBACK",
                            "RAY_TMPDIR",
                            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"}
    assert dict(gen.CAMPAIGN_ENV)["ALPHAGRAD_BATCHED_CALLBACK"] == "1"
    assert dict(gen.CAMPAIGN_ENV)["RAY_TMPDIR"] == "/tmp/ray_$SLURM_JOB_ID"


def test_no_flag_knobs_are_exported_and_named_in_the_todo_header(gen, campaign):
    """A knob without a flag is a departure from 'args only'; it is exported
    (the run is wrong or dead without it) AND declared in the header."""
    names = {k for k, _, _ in gen.NO_FLAG_ENV}
    assert "ALPHAGRAD_POLICY" in names          # ppo.py has no --policy flag
    for k, v, why in gen.NO_FLAG_ENV:
        assert why and len(why) > 40, k
    for a in campaign:
        text = gen.render(a)
        assert "*** TODO (ticket .43): ENV VARS WITHOUT A FLAG" in text, a["name"]
        for k, v, _why in gen.NO_FLAG_ENV:
            assert f"export {k}={v}\n" in text, (a["name"], k)
            assert f"#   {k}={v}\n" in text, (a["name"], k)
    # and the claim "no flag" is TRUE against ppo.py's argparser today
    src = open(_PPO).read()
    assert '"--policy"' not in src
    assert '"--skip-cost-analysis"' not in src


# ------------------------------------------------- 5. the hardware / stack

def test_one_eight_gpu_blackwell_job_per_node_on_gpu20(gen, campaign):
    # pgi15-gpu19 belongs to another group (dsnn-dfw.69): gpu20 is the only
    # 8-GPU node we may use, so every campaign row runs there now.
    assert gen.CAMPAIGN_NODES == ("pgi15-gpu20",)
    assert gen.CAMPAIGN_GPUS == 8 and gen.CAMPAIGN_CPUS == 128
    for a in campaign:
        text = gen.render(a)
        assert a["kind"] == "train" and a["gpus"] == 8, a["name"]
        assert gen.is_scratch(a), a["name"]
        assert a["node"] in gen.CAMPAIGN_NODES, (a["name"], a["node"])
        assert ("#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_"
                "workstation_edition:8\n") in text, a["name"]
        assert "#SBATCH -c 128\n" in text, a["name"]
        assert f"#SBATCH -w {a['node']}\n" in text
        assert f"#SBATCH -D {gen.CAMPAIGN_STACK}/alphagrad\n" in text
        assert f"#SBATCH -o {gen.CAMPAIGN_RUNS}/" in text
        for bad in ("pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu17", "pgi15-gpu18",
                    "pgi15-gpu19"):
            assert bad not in text, (a["name"], bad)
        # the /Scratch stack of finding 57: no home, no uv
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            assert "uv run" not in line, (a["name"], line)
            assert "~/dsnn" not in line and "$HOME/dsnn" not in line, (a["name"], line)
            assert "/Users/assmuth" not in line, (a["name"], line)
        assert f"PY={gen.CAMPAIGN_PY}\n" in text
        assert f"cd {gen.CAMPAIGN_STACK}/alphagrad\n" in text
        assert "ABORT(66)" in text
        assert 'src/alphagrad/approx/ppo.py "${ARGS[@]}"' in text
        assert "CUDA_VISIBLE_DEVICES=0,1,2,3" not in text, a["name"]
    # one node, so phase 1 runs every tag on it
    p1 = [a for a in campaign if a["phase"] == 1]
    assert [a["node"] for a in p1] == [gen.CAMPAIGN_NODES[0]] * 6
    assert {a["node"] for a in campaign} == set(gen.CAMPAIGN_NODES)


# ------------------------------------------------- 6. the gate telemetry

def test_gate_telemetry_has_no_switch_and_its_input_is_passed(gen, campaign):
    """Ticket .45: ppo.py emits gate/*, paired/*, measure/* every episode from
    host_log with no flag to turn it on or off; the launcher's part is G1's
    input.  Pinned on the source so a future switch cannot silently default
    to off."""
    from alphagrad.approx.ppo import make_argparser
    from alphagrad.approx.common import gate_telemetry
    ap = make_argparser()
    opts = {s for act in ap._actions for s in act.option_strings}
    assert "--gate-winners-table" in opts and "--gate-offline-contrast" in opts
    assert not [o for o in opts if "telemetry" in o], opts
    src = open(_PPO).read()
    assert "_gate_telemetry.episode_fields(" in src
    assert "log_dict.update(_gate_telemetry.episode_fields(" in src
    assert callable(gate_telemetry.episode_fields)
    assert callable(gate_telemetry.add_gate_args)
    assert os.path.exists(os.path.join(_ALPHAGRAD, "docs", "GATE_TELEMETRY.md"))
    for a in campaign:
        cli = _cli(gen, a)
        assert (cli["--gate-winners-table"]
                == gen.CAMPAIGN_GATE_WINNERS_TABLES[cli["--fixed-order"]])
        assert (cli["--gate-offline-contrast"]
                == gen.GATE_OFFLINE_CONTRAST[cli["--fixed-order"]])
        text = gen.render(a)
        assert "gate G1 winners table" in text, a["name"]
        assert "episode_fields" in text, a["name"]   # the header says where
        assert "measure/drain/" in text, a["name"]   # the .7 audit is named


def test_every_contract_field_the_arm_will_emit_is_documented(gen, campaign):
    """Ticket .45: the launcher carries the two inputs, and ppo.py emits the
    whole table unconditionally.  This pins that the table the arms will be
    read against is the one docs/GATE_TELEMETRY.md states, and that the nine
    quantities the ticket names all have a field."""
    from alphagrad.approx.common import gate_telemetry as gt
    heads = ("latency", "mem", "quality")
    names = gt.documented_fields(heads)
    for want in ("paired/lat_ratio_best",          # paired latency ratio
                 "paired/temp_ratio_best",         # paired memory ratio (temp)
                 "paired/watermark_ratio_best",    # the watermark beside it
                 "paired/grad_cosine_mean",        # grad-cosine
                 "gate/g4/q_zero_frac",            # the q = 0 fraction
                 "gate/g3/uniform_floor_nats",     # entropy against its floor
                 "gate/g2/ev_latency",             # EV per value head
                 "gate/g1/recovery",               # recovered sweep winners
                 "gate/g5/spread_lat",             # front spread at the corners
                 "gate/g5/drift_floor_lat",        # what the spread must beat
                 "measure/drain/ok"):              # ticket .7
        assert want in names, want
    # the floor is derived from the head the arms run, not from a literal
    geom = gt.face_head_geometry(gen.APPROX_ADD)
    assert geom["width"] == gen.FACE_HEAD_WIDTH
    assert geom["n_quant_dtypes"] == len(gen.FACE_QUANT_DTYPES) == 2


# ------------------------------------------------- 7. the builder raises

def test_campaign_arm_raises_on_a_row_outside_the_rulings(gen):
    n0 = len(gen.ARMS)
    ok = dict(phase=9, tag="z", profile="skip", node=gen.CAMPAIGN_NODES[0],
              what="x", prediction="x", falsifier="x")
    bad = [
        dict(ok, node="pgi15-gpu17"),
        dict(ok, node="pgi15-gpu19"),
        dict(ok, node="pgi15-gpu15"),
        dict(ok, approx_add="lossy"),
        dict(ok, approx_add="learned2"),
        dict(ok, profile="everything"),
        dict(ok, order="random"),
        dict(ok, form="P9"),
        dict(ok, rewards="cmp"),
        dict(ok, advantage_norm="zscore"),
        dict(ok, what=""),
    ]
    try:
        for kw in bad:
            with pytest.raises(gen.CampaignRowError):
                gen.campaign_arm(**kw)
        assert len(gen.ARMS) == n0
        a = gen.campaign_arm(**ok)
        assert a["name"] == f"p9z_skip_{REWARD_TOKEN}" and len(gen.ARMS) == n0 + 1
        assert a["cli"]["--approx-add"] == "lossless"
        assert a["cli"]["--fixed-order"] == "markowitz"
    finally:
        del gen.ARMS[n0:]
    assert len(gen.ARMS) == n0
    assert issubclass(gen.CampaignRowError, ValueError)


def test_a_campaign_arm_with_a_per_arm_env_is_refused_at_render(gen):
    a = dict(next(x for x in gen.ARMS if x.get("phase")))
    a["env"] = {"ALPHAGRAD_MAX_FACES": "2538"}
    with pytest.raises(gen.CampaignRowError):
        gen.render(a)


def test_setting_the_winner_constant_puts_the_profile_in_the_name(gen):
    n0 = len(gen.ARMS)
    saved = gen.P1_WINNER_PROFILE
    try:
        gen.P1_WINNER_PROFILE = "skip"
        a = gen.campaign_arm(phase=9, tag="z", profile="WINNER",
                             node=gen.CAMPAIGN_NODES[0], what="x",
                             prediction="x", falsifier="x")
        assert a["name"] == f"p9z_skip_{REWARD_TOKEN}"
        assert a["cli"]["--approx-profile"] == "skip"
        assert "P1_PROFILE" not in gen.render(a)
    finally:
        gen.P1_WINNER_PROFILE = saved
        del gen.ARMS[n0:]


# ------------------------------------------ 8. names and later phases

def test_each_name_encodes_profile_order_channels_and_price(gen, campaign):
    for a in campaign:
        cli = _cli(gen, a)
        name = a["name"]
        prof = cli["--approx-profile"]
        prof_tok = "winner" if _PLACEHOLDER.search(prof) else prof
        assert f"_{prof_tok}_" in name, (name, prof)
        if cli.get("--reward-mode") == "lagrangian":
            assert "_dual" in name, name
            assert f"eta{gen.DUAL_ETA.replace('.', '')}" in name, name
        else:
            # The price is in the name, and it is the price the flag carries.
            assert f"_lq{cli['--lambda-acc']}" in name, name
        if "--preference-conditioned" in cli:
            assert "_pref" in name or "_dual" in name, name
        else:
            assert "_hinge_" in name, name
        if "--quality-floor" in cli:
            # "0.90" -> "09": the name states tau without a trailing zero.
            tok = cli["--quality-floor"].replace(".", "").rstrip("0") or "0"
            assert f"tau{tok}" in name, name
        else:
            assert "tau" not in name, name
        if cli.get("--advantage-norm") == "popart":
            assert name.endswith("_popart"), name
        if cli["--seed"] != gen.CAMPAIGN_SEED:
            assert f"_s{cli['--seed']}" in name, name
        assert a["job"] == name.replace("_", "-")
        assert cli["--name"] == a["job"]


def test_phase2_channel_arms(gen, campaign):
    p2 = {a["name"]: _cli(gen, a) for a in campaign if a["phase"] == 2}
    assert set(p2) == {f"p2a_winner_latq_{REWARD_TOKEN}", f"p2b_winner_memq_{REWARD_TOKEN}"}
    assert p2[f"p2a_winner_latq_{REWARD_TOKEN}"]["--rewards"] == "cmp acc"
    assert p2[f"p2b_winner_memq_{REWARD_TOKEN}"]["--rewards"] == "mem acc"
    for cli in p2.values():
        assert _PLACEHOLDER.search(cli["--approx-profile"]), cli["--approx-profile"]
        assert "P1_PROFILE" in cli["--approx-profile"]


def test_phase3_ladder_p0_p1_l(gen, campaign):
    p3 = {a["name"]: _cli(gen, a) for a in campaign if a["phase"] == 3}
    n0, n1, nl = (f"p3a_winner_pref_lq{LAMBDA_Q}",
                  f"p3b_winner_pref_tau09_lq{LAMBDA_Q}",
                  f"p3c_winner_dual_tau09_eta20")
    assert set(p3) == {n0, n1, nl}
    p0, p1, lag = p3[n0], p3[n1], p3[nl]
    for cli in (p0, p1, lag):
        assert "--preference-conditioned" in cli
    assert "--quality-floor" not in p0 and p0["--reward-mode"] == "additive"
    assert p1["--quality-floor"] == gen.QUALITY_FLOOR_TAU
    assert p1["--reward-mode"] == "additive"
    assert lag["--quality-floor"] == gen.QUALITY_FLOOR_TAU
    assert lag["--reward-mode"] == "lagrangian"
    # eta 2.0, lambda in [12, 32] (owner ruling 2026-09-13): one episode of
    # full violation (q = 0 at tau = 0.90) moves lambda by 1.8, and 12 is the
    # smallest weight that puts the Markowitz absorber below the baseline.
    assert lag["--lag-eta"] == "2.0"
    assert lag["--lag-min"] == "12" and lag["--lag-max"] == "32"
    assert lag["--lag-init"] == LAMBDA_Q


def test_phase4_popart_sets_no_symlog_at_every_site(gen, campaign):
    (p4,) = [a for a in campaign if a["phase"] == 4]
    cli = _cli(gen, p4)
    assert cli["--advantage-norm"] == "popart"
    assert "--no-symlog" in cli
    assert cli["--symlog-channels"] == "none"   # ppo.py: the two must agree
    assert p4["name"].endswith("_popart")


def test_phase5_is_five_seeds_of_one_configuration(gen, campaign):
    p5 = [a for a in campaign if a["phase"] == 5]
    seeds = [_cli(gen, a)["--seed"] for a in p5]
    assert len(seeds) == 5 and len(set(seeds)) == 5
    assert seeds == list(gen.FIVE_SEEDS)
    ref = _cli(gen, p5[0])
    for a in p5[1:]:
        cli = _cli(gen, a)
        for k in set(ref) | set(cli):
            if k in ("--seed", "--name"):
                continue
            assert ref.get(k) == cli.get(k), (a["name"], k)
    assert {a["node"] for a in p5} == set(gen.CAMPAIGN_NODES)


# ------------------------------------ 9. the wave arms keep their runtime

def test_every_arm_including_the_wave_arms_gets_a_real_winners_table(gen):
    """One mechanism: SHARED_CLI carries the _ByOrder sentinel and _merge_cli
    resolves it from the arm's own --fixed-order.  The wave arms previously
    carried ~/dsnn/run_analysis/sweep41/winners.csv, a path nothing has ever
    written, so gate/g1/present read 0 in every one of them."""
    for a in gen.ARMS:
        if a.get("kind") != "train":
            continue
        cli = _cli(gen, a)
        if cli["--example"].startswith("Vmapped"):
            # dsnn-qaht: a batched thesis row passes no table.
            assert "--gate-winners-table" not in cli, a["name"]
            continue
        tbl = cli["--gate-winners-table"]
        assert isinstance(tbl, str), (a["name"], tbl)
        assert tbl == gen.CAMPAIGN_GATE_WINNERS_TABLES[cli["--fixed-order"]]
        assert "run_analysis" not in tbl and "sweep41" not in tbl
        assert tbl.startswith("/Scratch/")


def test_an_order_without_a_sweep_raises_instead_of_borrowing_one(gen):
    saved = dict(gen.CAMPAIGN_GATE_WINNERS_TABLES)
    try:
        del gen.CAMPAIGN_GATE_WINNERS_TABLES["reverse"]
        with pytest.raises(ValueError) as e:
            gen._merge_cli({"--fixed-order": "reverse"})
        assert "winners table" in str(e.value)
    finally:
        gen.CAMPAIGN_GATE_WINNERS_TABLES.clear()
        gen.CAMPAIGN_GATE_WINNERS_TABLES.update(saved)


def test_the_wave_arms_keep_their_own_env_but_run_the_campaign_stack(gen):
    """Owner ruling 2026-09-14: the wave arms and fq_face_attrib no longer
    `cd ~/dsnn/alphagrad` or `uv run` -- $HOME/dsnn is 281 commits stale and
    the export it lives on is read-only, so every arm now stages against
    CAMPAIGN_STACK like the campaign arms.  They keep their OWN per-arm
    SHARED_ENV (not args-only: `is_scratch` -- and the full-node Blackwell
    hardware and env-purity check that comes with it -- stays False), and
    their own 4-GPU hardware request."""
    # The THESIS arms (ticket dsnn-dfw.4) are a third family: they run the
    # /Scratch stack like a campaign arm (`is_scratch` True) and they carry no
    # phase, so they are excluded by name here and pinned in
    # tests/gen_fq_launchers_thesis_test.py instead.
    waves = [a for a in gen.ARMS
             if a["kind"] == "train" and not a.get("phase")
             and not a.get("thesis")]
    assert waves
    for a in waves:
        assert not gen.is_scratch(a), a["name"]
        text = gen.render(a)
        assert "uv run" not in text, a["name"]
        assert "$HOME/dsnn" not in text and "~/dsnn" not in text, a["name"]
        assert f"PY={gen.CAMPAIGN_PY}\n" in text, a["name"]
        assert f"cd {gen.CAMPAIGN_STACK}/alphagrad\n" in text, a["name"]
        assert f"#SBATCH -D {gen.CAMPAIGN_STACK}/alphagrad\n" in text, a["name"]
        assert f"#SBATCH -o {gen.CAMPAIGN_RUNS}/" in text, a["name"]
        assert "ABORT(66)" in text, a["name"]
        assert "#SBATCH --gres=gpu:4\n" in text, a["name"]
        assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in text, a["name"]
        assert f"  --approx-add {gen.APPROX_ADD}\n" in text, a["name"]
        assert "  --fixed-order " in text, a["name"]
        assert "ALPHAGRAD_FORCE_REV_ORDER" not in text, a["name"]


def test_no_rendered_launcher_of_any_kind_references_home_dsnn(gen):
    """The 2026-09-14 audit found 19 wave/cpu/tool launchers still pointed at
    $HOME/dsnn (281 commits stale, and the export it lives on is read-only)
    while the 17 campaign arms already ran the /Scratch stack of finding 57.
    Every arm runs that stack now; this is the whole-fleet guard the
    per-family tests above do not give.

    The guard is about what a launcher RUNS, so it is applied to the
    executable lines, exactly as the `uv run` check below it always was.  A
    COMMENT may name a home path: the thesis arms' header says where the
    nightly copy job puts the run data (/Users/assmuth/thesis-runs), which is
    a fact the reader of the launcher needs and is not a thing this job does.
    The stale checkout itself stays forbidden everywhere, comments included,
    because naming it is how a launcher comes back to it."""
    for a in gen.ARMS:
        text = gen.render(a)
        assert "$HOME/dsnn" not in text, a["name"]
        assert "~/dsnn" not in text, a["name"]
        assert "/Users/assmuth/dsnn" not in text, a["name"]
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            assert "uv run" not in line, (a["name"], line)
            assert "/Users/assmuth" not in line, (a["name"], line)


# ------------------------------------------- 10. --dry-run writes outside

def test_dry_run_writes_outside_the_tree_and_diffs_against_it(gen, tmp_path):
    tree = tmp_path / "tree"
    out = tmp_path / "out"
    tree.mkdir()
    # one launcher "in the tree" that drifted, one identical, the rest missing
    arms = gen.ARMS
    (tree / f"fq_{arms[0]['name']}.sbatch").write_text("#!/bin/bash\n# stale\n")
    (tree / f"fq_{arms[1]['name']}.sbatch").write_text(gen.render(arms[1]))
    rc = gen.main(["--dry-run", "--out", str(out), "--against", str(tree)])
    assert rc == 0
    written = sorted(p.name for p in out.glob("fq_*.sbatch"))
    assert written == sorted(f"fq_{a['name']}.sbatch" for a in arms)
    assert (out / "DRIFT.diff").exists() and (out / "SUMMARY.txt").exists()
    summary = (out / "SUMMARY.txt").read_text()
    assert f"{len(arms)} launchers rendered" in summary
    assert "1 ok, 1 DRIFT" in summary and f"{len(arms) - 2} MISSING" in summary
    assert f"DRIFT    {tree / ('fq_' + arms[0]['name'] + '.sbatch')}" in summary
    assert "# stale" in (out / "DRIFT.diff").read_text()
    # the tree was not touched
    assert sorted(p.name for p in tree.glob("*")) == sorted(
        [f"fq_{arms[0]['name']}.sbatch", f"fq_{arms[1]['name']}.sbatch"])
    assert (tree / f"fq_{arms[0]['name']}.sbatch").read_text() == "#!/bin/bash\n# stale\n"
    # refuses to write INTO the tree
    with pytest.raises(SystemExit):
        gen.main(["--dry-run", "--out", str(tree), "--against", str(tree)])
    with pytest.raises(SystemExit):
        gen.main(["--dry-run", "--out", str(tree / "sub"), "--against", str(tree)])
    with pytest.raises(SystemExit):
        gen.main(["--dry-run", "--check", "--out", str(out), "--against", str(tree)])


# --------------------------------- 11. ppo.py's own argparse accepts them

def test_ppo_argparse_accepts_every_campaign_command_line(gen, campaign):
    # Layer 2 of the launcher's pre-flight, run here without a node.
    from alphagrad.approx.ppo import make_argparser
    ap = make_argparser()
    prof_action = next(act for act in ap._actions
                       if "--approx-profile" in act.option_strings)
    assert tuple(prof_action.choices) == gen.PROFILES
    order_action = next(act for act in ap._actions
                        if "--fixed-order" in act.option_strings)
    assert set(order_action.choices) == set(gen.FIXED_ORDERS)
    for a in campaign:
        toks = _tokens(gen, a, profile="skip")
        ns = ap.parse_args(toks)
        cli = _cli(gen, a)
        assert ns.discount == 1.0 and ns.gae_lambda == 1.0, a["name"]
        assert ns.terminal_rewards_only, a["name"]
        assert ns.cost_form == "paired-log" and ns.mem_channel == "temp", a["name"]
        assert ns.quality_metric == "grad_cosine", a["name"]
        assert ns.approx_add == "lossless", a["name"]
        assert ns.approx_old is None, a["name"]
        assert ns.fixed_order == cli["--fixed-order"], a["name"]
        assert ns.approx_profile == ("skip" if "P1_PROFILE" in cli["--approx-profile"]
                                     else cli["--approx-profile"]), a["name"]
        assert ns.face_actions and ns.per_face_masks and ns.unified_face_head
        assert ns.live_faces and ns.dynamic_substeps and ns.incremental_encode
        # One PPO GPU, every other GPU a measure actor (owner ruling
        # 2026-09-14, replacing the one-actor ruling of 2026-09-13).
        assert ns.ray_measure == gen.CAMPAIGN_GPUS - 1, a["name"]
        assert ns.ray_measure_timeout == 600.0, a["name"]
        # THE MEASUREMENT IS PIPELINED on every campaign arm (owner ruling
        # 2026-09-14): the terminal step submits and the driver waits for the
        # rewards with the previous episode's update already on the GPU.
        assert ns.measure_pipeline == 1, a["name"]
        # HOW MANY DEVICES ROLL AN EPISODE OUT (owner ruling 2026-09-15).
        # --num-envs is PER SHARD, so the episode holds the product, and the
        # arm has to state both.
        assert ns.rollout_shards == int(gen.CAMPAIGN_ROLLOUT_SHARDS), a["name"]
        assert ns.rollout_shards >= 1, a["name"]
        assert ns.num_envs == 16, a["name"]
        # AND THE TOKENIZATION IS OFF THE ACTORS (owner ruling 2026-09-15).
        # Without this the pipeline can only hide a measurement behind the
        # previous update, because the actors serve every rollout step.
        assert ns.tokenize_where == gen.CAMPAIGN_TOKENIZE_WHERE, a["name"]
        assert ns.tokenize_where != "pool", a["name"]
        # THE FACE WIRE IS NARROWED and it is a positive width: 0 would ship
        # the whole 1920-column prefix history on every callback.
        assert ns.face_wire_faces == int(gen.CAMPAIGN_FACE_WIRE_FACES), a["name"]
        assert ns.face_wire_faces > 0, a["name"]
        assert ns.face_none_bias == float(gen.FACE_NONE_BIAS_MVP), a["name"]
        assert ns.scale_face_head == float(gen.SCALE_FACE_HEAD_MVP), a["name"]
        assert ns.face_logit_clamp == float(gen.FACE_LOGIT_CLAMP_MVP), a["name"]
        assert ns.plan_log == "auto" and ns.episodes == 250, a["name"]
        assert (ns.gate_winners_table
                == gen.CAMPAIGN_GATE_WINNERS_TABLES[ns.fixed_order]), a["name"]
        assert ns.gate_offline_contrast == float(
            gen.GATE_OFFLINE_CONTRAST[ns.fixed_order]), a["name"]
        assert ns.advantage_norm == cli.get("--advantage-norm", "none"), a["name"]
        assert ns.reward_mode == cli.get("--reward-mode", "additive"), a["name"]
        assert ns.no_approx_head is False, a["name"]
        if "--quality-floor" in cli:
            assert ns.quality_floor == float(cli["--quality-floor"]), a["name"]
        else:
            assert ns.quality_floor is None, a["name"]
