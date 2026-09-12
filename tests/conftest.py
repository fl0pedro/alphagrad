"""THE COLLECTION-TIME CONFIGURATION GUARD, for the ``tests/`` root.

Kept deliberately beside the root ``conftest.py`` (owner ruling 2026-09-12:
"COPY, do not move. Divergence does not have to be strictly prohibited.") so
that a run scoped to this subtree still gets the guard. When both are active
the root one is primary and this one's hooks are no-ops.

The guard's logic lives in ``_pytest_config_guard.py`` at the REPOSITORY ROOT
and is shared by both conftests rather than duplicated; read that file for what
the guard does and why, for the ``EXHAUSTIVE=1`` noise-floor mode, and for the
measurement that says a root-level conftest changes nothing. This file is a
shim: it loads that module (by absolute path, so nothing depends on
``sys.path``) and forwards the hooks.

IDEMPOTENCE. ``pytest`` runs every conftest in the chain, so when the root and
``tests/`` conftests are both active the hooks would fire twice for a test under
``tests/``. ``register()`` hands primacy to the FIRST conftest loaded -- the
root one, when it is present -- and the other shim's hooks are no-ops, so
nothing is counted, printed or failed twice.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_GUARD_NAME = "_alphagrad_config_guard"
_GUARD_PATH = (pathlib.Path(__file__).resolve().parent.parent
               / "_pytest_config_guard.py")

if _GUARD_NAME in sys.modules:
    _G = sys.modules[_GUARD_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_GUARD_NAME, _GUARD_PATH)
    _G = importlib.util.module_from_spec(_spec)
    sys.modules[_GUARD_NAME] = _G
    _spec.loader.exec_module(_G)

#: ``True`` only for the first conftest to load the guard; see IDEMPOTENCE.
PRIMARY = _G.register(__file__)


def pytest_configure(config):
    if PRIMARY:
        _G.configure(config)


def pytest_collectstart(collector):
    if PRIMARY:
        _G.collectstart(collector)


def pytest_collectreport(report):
    if PRIMARY:
        _G.collectreport(report)


def pytest_generate_tests(metafunc):
    if PRIMARY:
        _G.generate_tests(metafunc)


def pytest_runtest_logreport(report):
    if PRIMARY:
        _G.runtest_logreport(report)


def pytest_collection_finish(session):
    if PRIMARY:
        _G.collection_finish(session)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if PRIMARY:
        _G.terminal_summary(terminalreporter, exitstatus, config)


def pytest_sessionfinish(session, exitstatus):
    if PRIMARY:
        _G.sessionfinish(session, exitstatus)


@pytest.fixture(autouse=True)
def _report_run_phase_configuration_leaks(request):
    """REPORT -- deliberately not repair -- any ALPHAGRAD_*/GRAPHAX_* variable a
    test changes and does not put back. See the guard module for why it reports
    rather than restores."""
    if not PRIMARY:
        yield
        return
    yield from _G.run_phase_leak(request)
