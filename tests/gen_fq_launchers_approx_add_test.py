"""Ticket .56 / finding 73 -- the face-ADD knob on the launcher generator.

Launchers are generated (tools/gen_fq_launchers.py), never hand-edited, so the
contract is pinned on the generator: every training arm names ``--approx-add``
on its command line (the declared default ``lossy``), no launcher mentions the
deleted ``ALPHAGRAD_NEW_SLOT_JOIN`` variable or the RETIRED ``--approx-old``,
the two-op pre-flight runs under BOTH values (both emit graphax's two-op face
form now), and ``arm_per_approx_add`` emits one arm per configuration so
ticket .43 can render the paired all-rev pair of .50.
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
    assert dict(gen.SHARED_CLI)["--approx-add"] == "lossy"
    assert "--approx-add" in gen.REQUIRED_FLAGS
    assert "--approx-add" in gen.LANDSCAPE_FLAGS
    assert "ALPHAGRAD_NEW_SLOT_JOIN" not in dict(gen.SHARED_ENV)
    # the RETIRED flag must not be declared anywhere the generator emits
    assert "--approx-old" not in dict(gen.SHARED_CLI)
    assert "--approx-old" not in gen.REQUIRED_FLAGS


def test_no_rendered_launcher_mentions_the_deleted_env_var(gen):
    for a in gen.ARMS:
        text = gen.render(a)
        assert "NEW_SLOT_JOIN" not in text, a["name"]
        if a["kind"] == "train":
            # Every training arm NAMES the face ADD; the campaign's paired
            # pair (ticket .43) carries lossless on one of the two.
            val = dict(gen._merge_cli(a.get("cli", {})))["--approx-add"]
            assert val in gen.APPROX_ADD_CONFIGS, (a["name"], val)
            assert f"  --approx-add {val}\n" in text, a["name"]
        assert "--approx-old" not in text, a["name"]


def _train_arm(gen, **cli):
    base = next(a for a in gen.ARMS if a["kind"] == "train")
    a = dict(base)
    a["cli"] = dict(base.get("cli", {}), **cli)
    return a


def test_the_two_op_preflight_runs_under_BOTH_values(gen):
    """#73: both values emit the two-op face form, so both need the
    pre-flight. Under the retired names only ``same`` did, because ``exact``
    emitted the bare triple -- that is gone: ``lossless`` is the two-op form
    with an all-None join triple, and ``lossy`` adds a join policy."""
    lossy = gen.render(_train_arm(gen))
    lossless = gen.render(_train_arm(gen, **{"--approx-add": "lossless"}))
    for text in (lossy, lossless):
        assert "test_face_two_op_form.py" in text
        assert "ABORT(70)" in text
    assert "  --approx-add lossless\n" in lossless
    assert "  --approx-add lossy\n" not in lossless
    assert "  --approx-add lossy\n" in lossy
    for text in (lossy, lossless):
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch",
                                         delete=False) as fh:
            fh.write(text)
        try:
            chk = subprocess.run(["bash", "-n", fh.name],
                                 capture_output=True, text=True)
            assert chk.returncode == 0, chk.stderr
        finally:
            os.unlink(fh.name)


def test_arm_per_approx_add_emits_one_arm_per_configuration(gen):
    base = next(a for a in gen.ARMS if a["kind"] == "train")
    n0 = len(gen.ARMS)
    kw = dict(base)
    kw["name"] = "t56_allrev"
    kw["job"] = "t56-allrev"
    try:
        gen.arm_per_approx_add(**kw)
        new = gen.ARMS[n0:]
        assert [a["name"] for a in new] == ["t56_allrev_addlossy",
                                            "t56_allrev_addlossless"]
        assert [a["job"] for a in new] == ["t56-allrev-addlossy",
                                           "t56-allrev-addlossless"]
        assert [a["cli"]["--approx-add"] for a in new] == ["lossy", "lossless"]
        # everything else byte-identical between the pair
        for k in kw:
            if k in ("name", "job", "cli"):
                continue
            assert new[0][k] == new[1][k], k
        rendered = [gen.render(a) for a in new]
        assert "  --approx-add lossy\n" in rendered[0]
        assert "  --approx-add lossless\n" in rendered[1]
    finally:
        del gen.ARMS[n0:]
