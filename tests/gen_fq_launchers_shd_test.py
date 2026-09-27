# dsnn-dfw.232, .264: no row runs tbptt or window2, thesis rows get DSNN_SHD_DIR.
from __future__ import annotations

import importlib.util
import os
import re

import pytest

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")
#: The stack every launcher here is rendered for: a generation-time input
#: with no default (owner ruling 2026-09-27).
STACK = "/Scratch/assmuth/mrg/test-stack"

SEEDS = ("250197", "250198", "250199", "250200", "250201")
ARMS = ("A", "B", "C", "C_popart")
MATRIX_RULES = ("bptt", "rtrl")
DEPRECATED_RULES = ("tbptt", "window2")
DEPRECATED_TARGETS = ("rsnn_tbptt", "rsnn_window2")
SHD_DIR = "/Scratch/assmuth/mrg/cache/dsnn_shd"
MNIST_LINE = "export DSNN_MNIST_DIR=/Scratch/assmuth/mrg/cache/dsnn_mnist\n"
SHD_LINE = f"export DSNN_SHD_DIR={SHD_DIR}\n"
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


def test_no_row_runs_a_deprecated_rule(gen):
    for a in gen.ARMS:
        assert a.get("thesis_rule") not in DEPRECATED_RULES, a["name"]
        assert a.get("thesis_target") not in DEPRECATED_TARGETS, a["name"]
        text = gen.render(a, STACK)
        assert not set(_TEMPORAL_RULE.findall(text)) & set(DEPRECATED_RULES), \
            a["name"]
        assert "RSNN_SHD_W2" not in text, a["name"]
    snn = gen.thesis_snn_arms()
    got = {(a["thesis_rule"], a["thesis_arm"], a["thesis_seed"]) for a in snn}
    assert got == {(r, arm, s) for r in MATRIX_RULES for arm in ARMS
                   for s in SEEDS}
    assert len(snn) == len(got) == 40
    assert gen.THESIS_MATRIX_RULES == MATRIX_RULES
    assert {a["thesis_rule"] for a in gen.orderonly_rsnn_arms()} == \
        set(MATRIX_RULES)


def test_thesis_arm_refuses_the_deprecated_rules(gen):
    n0 = len(gen.ARMS)
    try:
        for target in DEPRECATED_TARGETS:
            for arm in ARMS:
                with pytest.raises(gen.CampaignRowError) as e:
                    gen.thesis_arm(arm=arm, target=target, seed=SEEDS[0],
                                   node="pgi15-gpu15",
                                   name=f"throwaway_{arm}_{target}")
                assert "deprecated" in str(e.value), str(e.value)
                assert "dsnn-dfw.232" in str(e.value), str(e.value)
    finally:
        del gen.ARMS[n0:]


def test_every_row_thesis_arm_emits_exports_the_shd_directory(gen, emitted,
                                                              frozen):
    for a in emitted:
        text = gen.render(a, STACK)
        assert MNIST_LINE + SHD_LINE in text, a["name"]
        assert text.count("export DSNN_SHD_DIR=") == 1, a["name"]
        assert a.get("shd_dir") is True, a["name"]
        assert "DSNN_SHD_DIR" not in (a.get("env") or {}), a["name"]
    ids = {id(a) for a in emitted}
    others = [a for a in gen.ARMS if id(a) not in ids]
    assert len(others) > len(frozen)
    for a in others:
        assert "DSNN_SHD_DIR" not in gen.render(a, STACK), a["name"]
    names = {a["name"] for a in emitted}
    assert {"C_popart_rsnn_rtrl_s250197", "C_popart_nn256_s250197",
            "smoke_C_tlm"} <= names


def test_the_shd_directory_goes_through_the_stack_allow_lists(gen):
    names = gen.STACK_ENV_NAMES
    assert "DSNN_SHD_DIR" in names
    assert names.index("DSNN_SHD_DIR") == names.index("DSNN_MNIST_DIR") + 1
    assert "DSNN_SHD_DIR" in gen.CAMPAIGN_ENV_ALLOWED
    assert "DSNN_SHD_DIR" in gen.THESIS_ENV_ALLOWED
    assert "DSNN_SHD_DIR" not in gen.THESIS_TARGET_ENV_ALLOWED
    assert f"{gen.CAMPAIGN_CACHE}/dsnn_shd" == SHD_DIR
    on = "\n".join(gen._scratch_stack_block(STACK, {}, jax_cache=False,
                                            shd_dir=True)) + "\n"
    off = "\n".join(gen._scratch_stack_block(STACK, {}, jax_cache=False)) + "\n"
    assert MNIST_LINE + SHD_LINE in on
    assert "DSNN_SHD_DIR" not in off
    ds = open(os.path.join(_ALPHAGRAD, "src", "alphagrad", "approx", "common",
                           "datasets.py")).read()
    assert 'os.environ.get("DSNN_SHD_DIR")' in ds
