"""pytest entry point for the bit-identical policy-path regression gate.

The gate itself, its rationale and its one-liner live in
``tests/policy_regression_gate.py``. This file exists only so the gate is
collected by a plain ``pytest tests/`` run alongside everything else.

Standalone (this is the form to use after every Phase 1-2 edit)::

    JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py
"""
import importlib.util
import sys
from pathlib import Path

# Loaded by PATH, not by name: `tests/` is not a package and the gate must
# also be runnable as a bare script, so neither form may depend on the other's
# sys.path.
_spec = importlib.util.spec_from_file_location(
    "policy_regression_gate",
    Path(__file__).with_name("policy_regression_gate.py"))
G = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = G
_spec.loader.exec_module(G)


def test_policy_path_is_bit_identical_to_golden():
    """Fails with the gate's own readable diff, not a bare assert."""
    G.check()


def test_the_golden_is_not_trivial():
    """A golden that pinned the fail-soft path would be green forever while
    the thing it protects was dead. Re-checked against the RECORDED golden,
    not only against the live trace, so a future re-record cannot quietly
    downgrade the fixture."""
    import json
    from pathlib import Path
    G.assert_nontrivial(json.loads(Path(G.GOLDEN).read_text()))
