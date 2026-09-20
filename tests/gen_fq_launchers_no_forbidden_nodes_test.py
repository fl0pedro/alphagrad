"""Ticket dsnn-dfw.69 follow-up -- no rendered launcher may target
pgi15-gpu17 (no matched CUDA 12.9 ptxas or nvlink, job 66740 aborted 72) or
pgi15-gpu19 (belongs to another group), anywhere in the generator.

The rule is for the whole generator, not one constant: THESIS_NODES_ALL,
CAMPAIGN_NODES, and the four wave tuning rows that used to pin
pgi15-gpu17 (w1c, w2c, w3c, w4c) are all node sources, and every one of
them must stay off both forbidden nodes for every arm the generator emits.
"""
from __future__ import annotations

import importlib.util
import os

_ALPHAGRAD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ALPHAGRAD, "tools", "gen_fq_launchers.py")

FORBIDDEN = ("pgi15-gpu17", "pgi15-gpu19")


def _gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_arm_records_a_forbidden_node():
    gen = _gen()
    assert gen.ARMS, "the generator emits no arm"
    offenders = [(a["name"], a.get("node")) for a in gen.ARMS
                 if a.get("node") in FORBIDDEN]
    assert not offenders, (
        f"{len(offenders)} row(s) target a forbidden node: "
        f"{offenders[:10]}{' ...' if len(offenders) > 10 else ''}")


def test_no_rendered_sbatch_line_names_a_forbidden_node():
    """Belt and braces on the actual rendered text, not just the node field
    the builder recorded, for every arm the generator knows about."""
    gen = _gen()
    checked = 0
    for a in gen.ARMS:
        text = gen.render(a)
        for node in FORBIDDEN:
            assert f"#SBATCH -w {node}\n" not in text, (a["name"], node)
        checked += 1
    assert checked == len(gen.ARMS)
