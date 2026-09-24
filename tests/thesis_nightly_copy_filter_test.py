"""The nightly copy saves every thesis run and nothing else (finding dsnn-6e7).

tools/thesis_nightly_copy.sh selects the wandb run directories it copies by
run name, against the NAME_RE default it declares. Every name
tools/gen_fq_launchers.py's thesis_run_name can produce must match it, the
recurrent targets included: before 2026-09-24 the filter named only nn256 and
tlm, so no RSNN run was ever copied off /Scratch.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEN = os.path.join(_ROOT, "tools", "gen_fq_launchers.py")
_COPY = os.path.join(_ROOT, "tools", "thesis_nightly_copy.sh")


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_fq_launchers", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _name_re():
    with open(_COPY) as fh:
        for line in fh:
            m = re.match(r"^NAME_RE=\$\{NAME_RE:-(.*)\}$", line.strip())
            if m:
                return re.compile(m.group(1))
    raise AssertionError(f"{_COPY} declares no NAME_RE default")


def test_every_thesis_run_name_is_copied(gen):
    pat = _name_re()
    names = [gen.thesis_run_name(a, t, s) for a, t, s in itertools.product(
        gen.THESIS_ARMS, gen.THESIS_TARGETS, gen.THESIS_SEEDS)]
    missed = [n for n in names if not pat.search(n)]
    assert not missed, missed


def test_other_experiments_in_the_same_tree_are_not_copied():
    pat = _name_re()
    for name in ("nn256diag_after", "tlmdiag_after",
                 "condC_rsnn_bptt_s250197_2k_h100", "condC_nn256_s250197_20k"):
        assert not pat.search(name), name
