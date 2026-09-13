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
  4. THE ENVIRONMENT: no promoted env var, no XLA_* / JAX_* flag, no
     ALPHAGRAD_FORCE_REV_ORDER; every `export` is in CAMPAIGN_ENV_ALLOWED;
     the no-flag knobs are exported AND named in the header's TODO block.
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

PHASE1_NAMES = [
    "p1a_skip_lq5",
    "p1b_reduce_lq5",
    "p1c_quant_lq5",
    "p1d_diag_lq5",
    "p1e_none_free_lq5",
    "p1f_all_free_lq5",
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
    diag = _by_name(campaign, "p1d_diag_lq5")
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
    assert len(gen.FACE_QUANT_DTYPES) == 4
    # the head grew from 94 to 103 with the four-dtype set; the number in
    # the launchers is the library's, not the generator's
    assert gen.FACE_HEAD_WIDTH == 1 + 3 * (30 + len(FACE_QUANT_DTYPES))
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
              "--ray-measure", "--ray-measure-timeout", "--per-face-masks",
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
                     " --plan-log auto ", " --episodes 250 ",
                     " --measure-toolchain-gate abort ",
                     " --reduce-axis-space physical ",
                     f" --gate-winners-table {gen.CAMPAIGN_GATE_WINNERS_TABLE} ",
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
        if a["name"] in ("p1e_none_free_lq5", "p1f_all_free_lq5"):
            assert cli["--fixed-order"] == "free", a["name"]
            assert a["time"] == "24:00:00", a["name"]
            assert "_free_" in a["name"]
        else:
            assert cli["--fixed-order"] == "markowitz", a["name"]
            assert "_free" not in a["name"] and "_reverse" not in a["name"]
    order_only = _cli(gen, _by_name(campaign, "p1e_none_free_lq5"))
    assert order_only["--approx-profile"] == "none"
    assert "--no-approx-head" not in order_only   # the profile IS the switch
    assert "--exact" not in order_only
    assert _cli(gen, _by_name(campaign, "p1f_all_free_lq5"))["--approx-profile"] == "all"


def test_phase1_and_phase2_run_raw_quality_no_floor(gen, campaign):
    for a in campaign:
        cli = _cli(gen, a)
        if a["phase"] in (1, 2):
            assert "--quality-floor" not in cli, a["name"]
            assert "--preference-conditioned" not in cli, a["name"]
            assert cli["--lambda-acc"] == gen.LAMBDA_Q_MVP, a["name"]


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
        # no XLA memory flag, no XLA flag at all, no JAX cache/platform var
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            assert "XLA_" not in line, (a["name"], line)
            assert "export JAX_" not in line, (a["name"], line)
            assert "JAX_COMPILATION_CACHE_DIR" not in line, (a["name"], line)


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

def test_one_eight_gpu_blackwell_job_per_node_on_gpu19_gpu20(gen, campaign):
    assert gen.CAMPAIGN_NODES == ("pgi15-gpu19", "pgi15-gpu20")
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
        for bad in ("pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu17", "pgi15-gpu18"):
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
    # both nodes are used, and phase 1 alternates them
    p1 = [a for a in campaign if a["phase"] == 1]
    assert [a["node"] for a in p1] == [gen.CAMPAIGN_NODES[i % 2] for i in range(6)]
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
        assert cli["--gate-winners-table"] == gen.CAMPAIGN_GATE_WINNERS_TABLE
        text = gen.render(a)
        assert "gate G1 winners table" in text, a["name"]
        assert "episode_fields" in text, a["name"]   # the header says where


# ------------------------------------------------- 7. the builder raises

def test_campaign_arm_raises_on_a_row_outside_the_rulings(gen):
    n0 = len(gen.ARMS)
    ok = dict(phase=9, tag="z", profile="skip", node=gen.CAMPAIGN_NODES[0],
              what="x", prediction="x", falsifier="x")
    bad = [
        dict(ok, node="pgi15-gpu17"),
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
        assert a["name"] == "p9z_skip_lq5" and len(gen.ARMS) == n0 + 1
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
        assert a["name"] == "p9z_skip_lq5"
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
        if "--preference-conditioned" not in cli:
            assert f"_lq{cli['--lambda-acc']}" in name, name
        elif cli.get("--reward-mode") == "lagrangian":
            assert "_dual" in name, name
        else:
            assert "_pref" in name, name
        if "--quality-floor" in cli:
            assert f"tau{cli['--quality-floor'].replace('.', '')}" in name, name
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
    assert set(p2) == {"p2a_winner_latq_lq5", "p2b_winner_memq_lq5"}
    assert p2["p2a_winner_latq_lq5"]["--rewards"] == "cmp acc"
    assert p2["p2b_winner_memq_lq5"]["--rewards"] == "mem acc"
    for cli in p2.values():
        assert _PLACEHOLDER.search(cli["--approx-profile"]), cli["--approx-profile"]
        assert "P1_PROFILE" in cli["--approx-profile"]


def test_phase3_ladder_p0_p1_l(gen, campaign):
    p3 = {a["name"]: _cli(gen, a) for a in campaign if a["phase"] == 3}
    assert set(p3) == {"p3a_winner_pref", "p3b_winner_pref_tau09",
                       "p3c_winner_dual_tau09"}
    p0, p1, lag = (p3["p3a_winner_pref"], p3["p3b_winner_pref_tau09"],
                   p3["p3c_winner_dual_tau09"])
    for cli in (p0, p1, lag):
        assert "--preference-conditioned" in cli
    assert "--quality-floor" not in p0 and p0["--reward-mode"] == "additive"
    assert p1["--quality-floor"] == gen.QUALITY_FLOOR_TAU
    assert p1["--reward-mode"] == "additive"
    assert lag["--quality-floor"] == gen.QUALITY_FLOOR_TAU
    assert lag["--reward-mode"] == "lagrangian"


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

def test_the_wave_arms_keep_the_home_runtime_and_the_shared_env(gen):
    waves = [a for a in gen.ARMS if a["kind"] == "train" and not a.get("phase")]
    assert waves
    for a in waves:
        assert not gen.is_scratch(a), a["name"]
        text = gen.render(a)
        assert "uv run --no-sync python" in text, a["name"]
        assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in text, a["name"]
        assert f"  --approx-add {gen.APPROX_ADD}\n" in text, a["name"]
        assert "  --fixed-order " in text, a["name"]
        assert "ALPHAGRAD_FORCE_REV_ORDER" not in text, a["name"]


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
        assert ns.ray_measure == 1 and ns.ray_measure_timeout == 600.0, a["name"]
        assert ns.face_none_bias == float(gen.FACE_NONE_BIAS_MVP), a["name"]
        assert ns.scale_face_head == float(gen.SCALE_FACE_HEAD_MVP), a["name"]
        assert ns.face_logit_clamp == float(gen.FACE_LOGIT_CLAMP_MVP), a["name"]
        assert ns.plan_log == "auto" and ns.episodes == 250, a["name"]
        assert ns.gate_winners_table == gen.CAMPAIGN_GATE_WINNERS_TABLE, a["name"]
        assert ns.advantage_norm == cli.get("--advantage-norm", "none"), a["name"]
        assert ns.reward_mode == cli.get("--reward-mode", "additive"), a["name"]
        assert ns.no_approx_head is False, a["name"]
        if "--quality-floor" in cli:
            assert ns.quality_floor == float(cli["--quality-floor"]), a["name"]
        else:
            assert ns.quality_floor is None, a["name"]
