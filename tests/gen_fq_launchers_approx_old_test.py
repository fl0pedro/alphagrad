"""Ticket .56 -- the old-edge knob on the launcher generator.

Launchers are generated (tools/gen_fq_launchers.py), never hand-edited, so the
contract is pinned on the generator: every training arm names ``--approx-old``
on its command line (the declared default ``same``), no launcher mentions the
deleted ``ALPHAGRAD_NEW_SLOT_JOIN`` variable, the two-op pre-flight runs
exactly when an arm runs ``same``, and ``arm_per_approx_old`` emits one arm per
configuration so ticket .43 can render the paired all-rev pair of .50.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile

import pytest

_GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "gen_fq_launchers.py")


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shared_cli_declares_the_default_and_the_preflight_greps_for_it(gen):
    assert dict(gen.SHARED_CLI)["--approx-old"] == "same"
    assert "--approx-old" in gen.REQUIRED_FLAGS
    assert "--approx-old" in gen.LANDSCAPE_FLAGS
    assert "ALPHAGRAD_NEW_SLOT_JOIN" not in dict(gen.SHARED_ENV)


def test_no_rendered_launcher_mentions_the_deleted_env_var(gen):
    for a in gen.ARMS:
        text = gen.render(a)
        assert "NEW_SLOT_JOIN" not in text, a["name"]
        if a["kind"] == "train":
            assert "  --approx-old same\n" in text, a["name"]


def _train_arm(gen, **cli):
    base = next(a for a in gen.ARMS if a["kind"] == "train")
    a = dict(base)
    a["cli"] = dict(base.get("cli", {}), **cli)
    return a


def test_the_two_op_preflight_runs_under_same_and_not_under_exact(gen):
    same = gen.render(_train_arm(gen))
    exact = gen.render(_train_arm(gen, **{"--approx-old": "exact"}))
    assert "test_face_two_op_form.py" in same
    assert "ABORT(70)" in same
    assert "test_face_two_op_form.py" not in exact
    assert "  --approx-old exact\n" in exact
    assert "  --approx-old same\n" not in exact
    for text in (same, exact):
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch",
                                         delete=False) as fh:
            fh.write(text)
        try:
            chk = subprocess.run(["bash", "-n", fh.name],
                                 capture_output=True, text=True)
            assert chk.returncode == 0, chk.stderr
        finally:
            os.unlink(fh.name)


def test_arm_per_approx_old_emits_one_arm_per_configuration(gen):
    base = next(a for a in gen.ARMS if a["kind"] == "train")
    n0 = len(gen.ARMS)
    kw = dict(base)
    kw["name"] = "t56_allrev"
    kw["job"] = "t56-allrev"
    try:
        gen.arm_per_approx_old(**kw)
        new = gen.ARMS[n0:]
        assert [a["name"] for a in new] == ["t56_allrev_oldsame",
                                            "t56_allrev_oldexact"]
        assert [a["job"] for a in new] == ["t56-allrev-oldsame",
                                           "t56-allrev-oldexact"]
        assert [a["cli"]["--approx-old"] for a in new] == ["same", "exact"]
        # everything else byte-identical between the pair
        for k in kw:
            if k in ("name", "job", "cli"):
                continue
            assert new[0][k] == new[1][k], k
        rendered = [gen.render(a) for a in new]
        assert "  --approx-old same\n" in rendered[0]
        assert "  --approx-old exact\n" in rendered[1]
    finally:
        del gen.ARMS[n0:]
