"""Ticket .43 -- the campaign arms (phases 1-5, tickets .50-.54) on the
launcher generator.

Launchers are generated (tools/gen_fq_launchers.py), never hand-edited, so
the campaign contract is pinned on the generator: every campaign arm renders
to valid bash; every flag it passes is defined in the files the pre-flight
greps (ppo.py, gate_telemetry.py); no arm carries an env-var knob that has a
flag (owner ruling 2026-09-03: args only); every name encodes its profile,
its old-edge value and its price; the shared contract (TLM, three channels,
paired-log cost form, static temp memory, grad-cosine, gamma = GAE lambda =
1, terminal rewards, additive, raw advantages, no quality gate, one seed,
250 episodes, plan log on, gate telemetry inputs, gpu15/16/18 only) holds
on every arm; and the per-phase shapes (the .56 paired pair, the held Diag
arm, the two order arms, P0 / P1 / L, PopArt, five seeds) are what the
tickets ask for.
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

CAMPAIGN_NODES_ALLOWED = {"pgi15-gpu15", "pgi15-gpu16", "pgi15-gpu18"}
PHASE1_NAMES = [
    "p1a_skip_oldsame_lq5",
    "p1b_all_oldsame_lq5",
    "p1c_all_oldexact_lq5",
    "p1d_reduce_oldsame_lq5",
    "p1e_quant_oldsame_lq5",
    "p1f_diag_oldsame_lq5",
    "p1g_none_free_oldsame_lq5",
    "p1h_all_free_oldsame_lq5",
]
_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]*\}")


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


# ------------------------------------------------------------ 1. the shape

def test_the_phase1_table_is_the_one_ticket_50_asks_for(campaign):
    p1 = [a["name"] for a in campaign if a["phase"] == 1]
    assert p1 == PHASE1_NAMES
    phases = sorted({a["phase"] for a in campaign})
    assert phases == [1, 2, 3, 4, 5]
    assert len([a for a in campaign if a["phase"] == 2]) == 2
    assert len([a for a in campaign if a["phase"] == 3]) == 3
    assert len([a for a in campaign if a["phase"] == 4]) == 1
    assert len([a for a in campaign if a["phase"] == 5]) == 5


def test_every_campaign_arm_renders_to_valid_bash(gen, campaign):
    for a in campaign:
        text = gen.render(a)
        assert text.startswith("#!/bin/bash\n"), a["name"]
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch",
                                         delete=False) as fh:
            fh.write(text)
        try:
            chk = subprocess.run(["bash", "-n", fh.name],
                                 capture_output=True, text=True)
            assert chk.returncode == 0, (a["name"], chk.stderr)
        finally:
            os.unlink(fh.name)


# -------------------------------------------------- 2. REQUIRED_FLAGS check

def test_required_flags_are_all_defined_where_the_preflight_greps(
        gen, flag_sources):
    # The pre-flight does `grep -qF -- "\"$F\"" $FLAGSRC`; this is that grep.
    missing = [f for f in gen.REQUIRED_FLAGS if f'"{f}"' not in flag_sources]
    assert not missing, missing
    assert "--approx-profile" in gen.REQUIRED_FLAGS
    assert "--cost-form" in gen.REQUIRED_FLAGS
    assert "--quality-floor" in gen.REQUIRED_FLAGS
    assert "--mem-channel" in gen.REQUIRED_FLAGS
    assert "--gate-winners-table" in gen.REQUIRED_FLAGS


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


# ------------------------------------------------------- 3. no env-var knob

def test_no_campaign_arm_carries_an_env_var_knob(gen, campaign):
    for a in campaign:
        text = gen.render(a)
        for var in gen.PROMOTED_ENV_VARS:
            assert var not in a.get("env", {}), (a["name"], var)
            assert f"export {var}=" not in text, (a["name"], var)
            assert f"{var}=" not in text, (a["name"], var)
        # the two deleted knobs of ticket .9 / .56 by name as well
        assert "QUALITY_GATE_MIN" not in text, a["name"]
        assert "NEW_SLOT_JOIN" not in text, a["name"]
        assert "GRAPHAX_ALLOW_PARTIAL_ORDER" not in text, a["name"]


def test_the_measurement_environment_is_still_exported(gen, campaign):
    # What the owner allows to stay an env var: the measurement environment
    # and the import-time settings that have no flag.
    for a in campaign:
        text = gen.render(a)
        for line in ("export ALPHAGRAD_SKIP_COUNT_OPS=1",
                     "export ALPHAGRAD_TLM_SEQ=32",
                     "export ALPHAGRAD_TLM_DMODEL=128",
                     "export ALPHAGRAD_TLM_VOCAB=1024",
                     "export ALPHAGRAD_MAX_FACES=2538",
                     "export JAX_COMPILATION_CACHE_DIR=$HOME/.jaxcache_$(hostname -s)"):
            assert line in text, (a["name"], line)


# ------------------------------------------- 4. names encode the knobs

def test_each_name_encodes_profile_old_edge_and_price(gen, campaign):
    for a in campaign:
        cli = _cli(gen, a)
        name = a["name"]
        prof = cli["--approx-profile"]
        prof_tok = "winner" if _PLACEHOLDER.search(prof) else prof
        assert f"_{prof_tok}_" in name, (name, prof)
        assert f"_old{cli['--approx-old']}" in name, name
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
    assert len({a["name"] for a in campaign}) == len(campaign)


# ------------------------------------------------ 5. the shared contract

def test_every_campaign_arm_carries_the_shared_contract(gen, campaign):
    for a in campaign:
        toks = _tokens(gen, a)
        text = gen.render(a)
        cli = _cli(gen, a)
        joined = " " + " ".join(toks) + " "
        for frag in (" --example TransformerLM ", " --exec-on-gpu ",
                     " --measure-latency ", " --latency-inner-reps 50 ",
                     " --cmp-type latency ", " --mem-type peak_memory ",
                     " --cost-form paired-log ", " --mem-channel temp ",
                     " --quality-metric grad_cosine ",
                     " --discount 1.0 ", " --gae-lambda 1.0 ",
                     " --terminal-rewards-only ",
                     " --plan-log auto ", " --episodes 250 ",
                     " --measure-toolchain-gate abort ",
                     " --face-slot-frames slot ",
                     " --reduce-axis-space physical ",
                     f" --gate-winners-table {gen.GATE_WINNERS_TABLE} ",
                     f" --face-none-bias {gen.FACE_NONE_BIAS_MVP} ",
                     f" --scale-face-head {gen.SCALE_FACE_HEAD_MVP} ",
                     f" --face-logit-clamp {gen.FACE_LOGIT_CLAMP_MVP} ",
                     " --lambda-cmp 1 ", " --lambda-mem 1 ",
                     " --wandb online "):
            assert frag in joined, (a["name"], frag)
        assert cli["--rewards"] in ("cmp mem acc", "cmp acc", "mem acc")
        if cli.get("--reward-mode") != "lagrangian":
            assert " --reward-mode additive " in joined, a["name"]
        if cli.get("--advantage-norm") != "popart":
            assert " --advantage-norm none " in joined, a["name"]
        # one seed per arm
        assert toks.count("--seed") == 1, a["name"]
        # every arm is a 4-GPU training job holding its node
        assert a["kind"] == "train" and a.get("gpus") == 4, a["name"]
        assert a["node"] in CAMPAIGN_NODES_ALLOWED, (a["name"], a["node"])
        assert "pgi15-gpu17" not in text, a["name"]
        # the toolchain gate (ticket .21) is in every launcher
        assert "ABORT(72)" in text, a["name"]
        # registered prediction and falsifier in the header
        assert "# REGISTERED PREDICTION" in text, a["name"]
        assert "# FALSIFICATION CRITERION:" in text, a["name"]
        assert "REGISTERED BEFORE THE RUN" in text, a["name"]


def test_phase1_and_phase2_run_raw_quality_no_floor(gen, campaign):
    for a in campaign:
        cli = _cli(gen, a)
        if a["phase"] in (1, 2):
            assert "--quality-floor" not in cli, a["name"]
            assert "--preference-conditioned" not in cli, a["name"]
            assert cli["--lambda-acc"] == gen.LAMBDA_Q_MVP, a["name"]


# ------------------------------------------------- 6. per-phase shapes

def test_the_all_rev_pair_differs_only_in_the_old_edge(gen, campaign):
    same = _by_name(campaign, "p1b_all_oldsame_lq5")
    exact = _by_name(campaign, "p1c_all_oldexact_lq5")
    cs, ce = _cli(gen, same), _cli(gen, exact)
    assert cs["--approx-old"] == "same" and ce["--approx-old"] == "exact"
    assert cs["--approx-profile"] == ce["--approx-profile"] == "all"
    for k in set(cs) | set(ce):
        if k in ("--approx-old", "--name"):
            continue
        assert cs.get(k) == ce.get(k), k
    for k in ("kind", "node", "time", "gpus", "env", "phase"):
        if k == "node":
            continue  # one ppo job per node: the pair sits on two nodes
        assert same[k] == exact[k], k
    ts, te = gen.render(same), gen.render(exact)
    assert "test_face_two_op_form.py" in ts     # the two-op pre-flight (.56)
    assert "test_face_two_op_form.py" not in te
    assert "  --approx-old exact\n" in te


def test_the_diag_arm_is_emitted_and_held(gen, campaign):
    diag = _by_name(campaign, "p1f_diag_oldsame_lq5")
    assert diag.get("held") and "dsnn-3qm.25" in diag["held"]
    text = gen.render(diag)
    assert "*** HELD" in text
    assert 'if [ "${FQ_RELEASE_HELD:-0}" != "1" ]; then' in text
    assert "ABORT(73)" in text and "  exit 73\n" in text
    # the guard sits before anything runs
    assert text.index("exit 73") < text.index("export RAY_TMPDIR")
    for a in campaign:
        if a is diag:
            continue
        assert not a.get("held"), a["name"]
        assert "ABORT(73)" not in gen.render(a), a["name"]


def test_the_order_arms_lift_the_pin_and_the_others_keep_it(gen, campaign):
    for a in campaign:
        text = gen.render(a)
        cli = _cli(gen, a)
        if a["name"] in ("p1g_none_free_oldsame_lq5",
                         "p1h_all_free_oldsame_lq5"):
            # not set (masks.py:149 reads == "1", default off), never "0"
            assert "export ALPHAGRAD_FORCE_REV_ORDER" not in text, a["name"]
            assert a["env"]["ALPHAGRAD_FORCE_REV_ORDER"] is gen._DELETE, a["name"]
            assert a["time"] == "24:00:00", a["name"]
        else:
            assert "export ALPHAGRAD_FORCE_REV_ORDER=1\n" in text, a["name"]
    order_only = _cli(gen, _by_name(campaign, "p1g_none_free_oldsame_lq5"))
    assert order_only["--approx-profile"] == "none"
    assert "--no-approx-head" not in order_only   # the profile IS the switch
    assert "--exact" not in order_only
    free = _cli(gen, _by_name(campaign, "p1h_all_free_oldsame_lq5"))
    assert free["--approx-profile"] == "all"
    assert _cli(gen, _by_name(campaign, "p1a_skip_oldsame_lq5"))["--approx-profile"] == "skip"
    assert _cli(gen, _by_name(campaign, "p1d_reduce_oldsame_lq5"))["--approx-profile"] == "reduce"
    assert _cli(gen, _by_name(campaign, "p1e_quant_oldsame_lq5"))["--approx-profile"] == "quant"
    assert _cli(gen, _by_name(campaign, "p1f_diag_oldsame_lq5"))["--approx-profile"] == "diag"


def test_phase2_channel_arms(gen, campaign):
    p2 = {a["name"]: _cli(gen, a) for a in campaign if a["phase"] == 2}
    assert set(p2) == {"p2a_winner_oldsame_latq_lq5", "p2b_winner_oldsame_memq_lq5"}
    assert p2["p2a_winner_oldsame_latq_lq5"]["--rewards"] == "cmp acc"
    assert p2["p2b_winner_oldsame_memq_lq5"]["--rewards"] == "mem acc"
    for cli in p2.values():
        assert _PLACEHOLDER.search(cli["--approx-profile"]), cli["--approx-profile"]
        assert "P1_PROFILE" in cli["--approx-profile"]


def test_phase3_ladder_p0_p1_l(gen, campaign):
    p3 = {a["name"]: _cli(gen, a) for a in campaign if a["phase"] == 3}
    assert set(p3) == {"p3a_winner_oldsame_pref",
                       "p3b_winner_oldsame_pref_tau09",
                       "p3c_winner_oldsame_dual_tau09"}
    p0, p1, lag = (p3["p3a_winner_oldsame_pref"],
                   p3["p3b_winner_oldsame_pref_tau09"],
                   p3["p3c_winner_oldsame_dual_tau09"])
    for cli in (p0, p1, lag):
        assert "--preference-conditioned" in cli
    assert "--quality-floor" not in p0 and p0["--reward-mode"] == "additive"
    assert p1["--quality-floor"] == gen.QUALITY_FLOOR_TAU
    assert p1["--reward-mode"] == "additive"
    assert lag["--quality-floor"] == gen.QUALITY_FLOOR_TAU
    assert lag["--reward-mode"] == "lagrangian"
    assert lag.get("--loss-mode", "multi_head") == "multi_head"


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
    assert len({a["node"] for a in p5}) == 3   # spread over the three nodes


# ------------------------------------------ 7. the campaign_arm builder

def test_campaign_arm_refuses_a_node_outside_the_allowed_set(gen):
    n0 = len(gen.ARMS)
    try:
        with pytest.raises(AssertionError):
            gen.campaign_arm(phase=9, tag="z", profile="skip",
                             node="pgi15-gpu17", what="x",
                             prediction="x", falsifier="x")
        with pytest.raises(AssertionError):
            gen.campaign_arm(phase=9, tag="z", profile="skip",
                             node="pgi15-gpu15", form="P9", what="x",
                             prediction="x", falsifier="x")
    finally:
        del gen.ARMS[n0:]
    assert len(gen.ARMS) == n0


def test_setting_the_winner_constant_puts_the_profile_in_the_name(gen):
    n0 = len(gen.ARMS)
    saved = gen.P1_WINNER_PROFILE
    try:
        gen.P1_WINNER_PROFILE = "skip"
        a = gen.campaign_arm(phase=9, tag="z", profile="WINNER",
                             node="pgi15-gpu15", what="x",
                             prediction="x", falsifier="x")
        assert a["name"] == "p9z_skip_oldsame_lq5"
        assert a["cli"]["--approx-profile"] == "skip"
        assert "P1_PROFILE" not in gen.render(a)
    finally:
        gen.P1_WINNER_PROFILE = saved
        del gen.ARMS[n0:]


# --------------------------------- 8. ppo.py's own argparse accepts them

def test_ppo_argparse_accepts_every_campaign_command_line(gen, campaign):
    # Layer 2 of the launcher's pre-flight, run here without a node.
    from alphagrad.approx.ppo import make_argparser
    ap = make_argparser()
    for a in campaign:
        toks = _tokens(gen, a, profile="skip")
        ns = ap.parse_args(toks)
        cli = _cli(gen, a)
        assert ns.discount == 1.0 and ns.gae_lambda == 1.0, a["name"]
        assert ns.terminal_rewards_only, a["name"]
        assert ns.cost_form == "paired-log" and ns.mem_channel == "temp", a["name"]
        assert ns.quality_metric == "grad_cosine", a["name"]
        assert ns.approx_old == cli["--approx-old"], a["name"]
        assert ns.face_none_bias == float(gen.FACE_NONE_BIAS_MVP), a["name"]
        assert ns.scale_face_head == float(gen.SCALE_FACE_HEAD_MVP), a["name"]
        assert ns.face_logit_clamp == float(gen.FACE_LOGIT_CLAMP_MVP), a["name"]
        assert ns.plan_log == "auto" and ns.episodes == 250, a["name"]
        assert ns.gate_winners_table == gen.GATE_WINNERS_TABLE, a["name"]
        assert ns.advantage_norm == cli.get("--advantage-norm", "none"), a["name"]
        assert ns.reward_mode == cli.get("--reward-mode", "additive"), a["name"]
        if "--quality-floor" in cli:
            assert ns.quality_floor == float(cli["--quality-floor"]), a["name"]
        else:
            assert ns.quality_floor is None, a["name"]
