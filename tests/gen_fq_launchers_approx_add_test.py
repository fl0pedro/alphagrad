"""Ticket .56 / finding 73, re-ruled 2026-09-13 -- the face-ADD knob on the
launcher generator has ONE value.

Launchers are generated (tools/gen_fq_launchers.py), never hand-edited, so the
contract is pinned on the generator: ``APPROX_ADD`` is ``lossless``; every
training arm names ``--approx-add lossless`` on its command line exactly once;
no launcher mentions the deleted ``ALPHAGRAD_NEW_SLOT_JOIN`` / ``ALPHAGRAD_
APPROX_ADD`` variables or the RETIRED ``--approx-old``; the two-op pre-flight
runs on every GPU arm; the learned join values (choose, learned1, learned2)
and the dropped ``lossy`` appear on no command line; ``arm_per_approx_add``
(the .56 paired pair) is gone and ``campaign_arm`` refuses any other value.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile

import pytest

_GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "gen_fq_launchers.py")
#: The stack every launcher here is rendered for: a generation-time input
#: with no default (owner ruling 2026-09-27).
STACK = "/Scratch/assmuth/mrg/test-stack"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv_lines(text: str) -> list[str]:
    """The non-comment lines of a rendered launcher."""
    return [l for l in text.splitlines() if not l.lstrip().startswith("#")]


def test_shared_cli_declares_the_one_value_and_the_preflight_greps_for_it(gen):
    assert gen.APPROX_ADD == "lossless"
    assert dict(gen.SHARED_CLI)["--approx-add"] == gen.APPROX_ADD
    assert "--approx-add" in gen.REQUIRED_FLAGS
    assert "--approx-add" in gen.LANDSCAPE_FLAGS
    assert "ALPHAGRAD_NEW_SLOT_JOIN" not in dict(gen.SHARED_ENV)
    assert "ALPHAGRAD_APPROX_ADD" not in dict(gen.SHARED_ENV)
    # the RETIRED flag must not be declared anywhere the generator emits
    assert "--approx-old" not in dict(gen.SHARED_CLI)
    assert "--approx-old" not in gen.REQUIRED_FLAGS
    # the paired pair is gone
    assert not hasattr(gen, "arm_per_approx_add")
    assert not hasattr(gen, "APPROX_ADD_CONFIGS")


def test_every_launcher_names_lossless_once_and_no_other_value(gen):
    train = 0
    for a in gen.ARMS:
        text = gen.render(a, STACK)
        assert "NEW_SLOT_JOIN" not in text, a["name"]
        assert "--approx-old" not in text, a["name"]
        body = "\n".join(_argv_lines(text))
        for other in ("lossy", "choose", "learned1", "learned2", "same", "exact"):
            assert f"--approx-add {other}" not in body, (a["name"], other)
        if a["kind"] == "train":
            train += 1
            val = dict(gen._merge_cli(a.get("cli", {})))["--approx-add"]
            assert val == gen.APPROX_ADD, (a["name"], val)
            # ONE occurrence per rendered ARGS array.  A paired launcher
            # (owner ruling 2026-09-20) carries one array per half.
            n_arrays = len(a.get("halves") or [None])
            assert text.count("  --approx-add lossless\n") == n_arrays, a["name"]
            assert gen.cli_tokens(a).count("--approx-add") == 1, a["name"]
        elif a.get("needs_tool"):
            # the landscape arm passes it to the tool it invokes
            assert "--approx-add lossless" in body, a["name"]
    assert train >= 17


def test_the_two_op_preflight_runs_on_every_gpu_arm(gen):
    """#73: lossless is the two-op face form with an all-None join triple, so
    graphax must accept the form before any plan is measured."""
    for a in gen.ARMS:
        text = gen.render(a, STACK)
        if a["kind"] == "cpu":
            assert "test_face_two_op_form.py" not in text, a["name"]
            continue
        assert "test_face_two_op_form.py" in text, a["name"]
        assert "ABORT(70)" in text, a["name"]
        assert "# --approx-add lossless emits the res-slot two-op face form." in text
        with tempfile.NamedTemporaryFile("w", suffix=".sbatch",
                                         delete=False) as fh:
            fh.write(text)
        try:
            chk = subprocess.run(["bash", "-n", fh.name],
                                 capture_output=True, text=True)
            assert chk.returncode == 0, (a["name"], chk.stderr)
        finally:
            os.unlink(fh.name)


def test_campaign_arm_refuses_every_other_face_add(gen):
    n0 = len(gen.ARMS)
    try:
        for other in ("lossy", "choose", "learned1", "learned2", "same", "exact", ""):
            with pytest.raises(gen.CampaignRowError):
                gen.campaign_arm(phase=9, tag="z", profile="skip",
                                 node=gen.CAMPAIGN_NODES[0], approx_add=other,
                                 what="x", prediction="x", falsifier="x")
    finally:
        del gen.ARMS[n0:]
    assert len(gen.ARMS) == n0
