# dsnn-dfw.232: no row thesis_arm emits runs the tbptt rule.
from __future__ import annotations

import importlib.util
import os
import re

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

SEEDS = ("250197", "250198", "250199", "250200", "250201")
ARMS = ("A", "B", "C", "C_popart")
MATRIX_RULES = ("bptt", "rtrl", "window2")
_FROZEN_KEYS = ("orderonly", "orderonly_rsnn", "orderonly_final",
                "orderonly_tlm_final", "sweepl", "sweepl2", "sweepl3")
_TEMPORAL_RULE = re.compile(r"^\s+--temporal-rule (\S+)$", re.M)


@pytest.fixture(scope="module")
def gen():
    old = os.environ.pop("THESIS_PAIRS", None)
    os.environ["THESIS_PAIRS"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.pop("THESIS_PAIRS", None)
        if old is not None:
            os.environ["THESIS_PAIRS"] = old
    return mod


@pytest.fixture(scope="module")
def emitted(gen):
    rows = [a for a in gen.thesis_arms()
            if not any(a.get(k) for k in _FROZEN_KEYS)]
    assert any(a.get("paired") for a in rows)
    assert any(a.get("smoke") for a in rows)
    return rows


@pytest.fixture(scope="module")
def frozen(gen):
    rows = [a for a in gen.thesis_arms()
            if any(a.get(k) for k in _FROZEN_KEYS)]
    assert rows
    return rows


def test_no_row_thesis_arm_emits_runs_the_tbptt_rule(gen, emitted, frozen):
    for a in emitted:
        assert a.get("thesis_rule") != "tbptt", a["name"]
        assert a.get("thesis_target") != "rsnn_tbptt", a["name"]
        assert "tbptt" not in _TEMPORAL_RULE.findall(gen.render(a)), a["name"]
    snn = gen.thesis_snn_arms()
    got = {(a["thesis_rule"], a["thesis_arm"], a["thesis_seed"]) for a in snn}
    assert got == {(r, arm, s) for r in MATRIX_RULES for arm in ARMS
                   for s in SEEDS}
    assert len(snn) == len(got) == 60
    assert gen.THESIS_MATRIX_RULES == MATRIX_RULES
    reference = [a for a in frozen if a.get("thesis_rule") == "tbptt"]
    assert sorted(a["name"] for a in reference) == [
        f"orderonly_rsnn_tbptt_l2m0_s{s}" for s in SEEDS[:3]]
    for a in reference:
        assert _TEMPORAL_RULE.findall(gen.render(a)) == ["tbptt"], a["name"]


def test_thesis_arm_refuses_the_tbptt_rule(gen):
    n0 = len(gen.ARMS)
    try:
        for arm in ARMS:
            with pytest.raises(gen.CampaignRowError) as e:
                gen.thesis_arm(arm=arm, target="rsnn_tbptt", seed=SEEDS[0],
                               node="pgi15-gpu15",
                               name=f"throwaway_{arm}_rsnn_tbptt")
            assert "dsnn-dfw.232" in str(e.value), str(e.value)
    finally:
        del gen.ARMS[n0:]
    assert gen.thesis_temporal_rule("rsnn_tbptt") == "tbptt"
    cli = gen.thesis_cli(arm="C", target="rsnn_tbptt", seed=SEEDS[0],
                         node="pgi15-gpu8", name="throwaway", episodes="1",
                         checkpoint_every="1", auto_stop=True)
    assert cli["--temporal-rule"] == "tbptt"
