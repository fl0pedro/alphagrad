# -*- coding: utf-8 -*-
"""Make the suite order-independent with respect to the project's env knobs.

THE PROBLEM
-----------
195 module-scope ``os.environ`` writes live across the two test roots -- almost
all ``setdefault`` at import time, because most ``ALPHAGRAD_*`` knobs have to be
set BEFORE ``alphagrad.approx.env`` is imported to take effect. pytest imports
every collected module into ONE process, so each of those writes leaks into
every module imported after it, and a module's behaviour depends on who was
collected first.

That was not theoretical. On 2026-08-31, with both test roots collected for the
first time, ``src/alphagrad/approx/tests/test_env_callback.py`` produced two
DISJOINT results:

    run alone            -> test_env_step_roundtrip_on_helmholtz FAILS, 8 pass
    run in the suite     -> that one PASSES, and three test_callback_* FAIL

Bisecting the 21 variables that differed between the two environments found a
single sufficient cause: ``ALPHAGRAD_INCREMENTAL_TOKENS=1``, leaked from
whichever of ~12 other modules was imported first. ``env.py:5202`` reads it per
call and takes a second code path into the real ``_eliminate_vertex`` -- which
``test_env_callback``'s ``extract_jaxpr`` mock does not cover.

THE RULE
--------
Before each test, the environment is reset to:

    the environment pytest STARTED with,  plus  whatever THIS test's own module
    set while it was being imported.

So a module gets exactly the knobs it asked for and none that a neighbour asked
for, and the result of a test no longer depends on collection order.

SCOPE, AND WHAT THIS DELIBERATELY DOES NOT FIX
----------------------------------------------
Only ``ALPHAGRAD_*`` and ``GRAPHAX_*`` are managed. ``JAX_*`` and ``XLA_*`` are
read once when jax initialises its backend, so restoring them per test would be
theatre -- it would change the string without changing the behaviour.

For the same reason this does NOT fix the knobs that ``env.py`` freezes into
module constants at ITS first import (``MAX_DELTA_TOKENS``, ``MAX_FACES``,
``MAX_TOKENS``). Once frozen they are frozen for the process, and no amount of
environment restoration reaches them. ``tests/_scale_guard.request_scale``
already handles that class by comparing the request against what ``env.py``
actually froze and turning a mismatch into a NAMED skip; that mechanism is
unaffected by this file and still needed. Only process isolation
(``tools/ratio_gates.sh``) makes those modules run for real.

Set ``ALPHAGRAD_TEST_NO_ENV_ISOLATION=1`` to disable, e.g. to confirm that a
failure really is order contamination.
"""
from __future__ import annotations

import os

import pytest

#: Knob prefixes this file owns. See "SCOPE" above for why JAX_/XLA_ are not here.
_MANAGED = ("ALPHAGRAD_", "GRAPHAX_")

_DISABLED = os.environ.get("ALPHAGRAD_TEST_NO_ENV_ISOLATION") == "1"


def _snapshot() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith(_MANAGED)}


#: The environment pytest started with, captured at conftest import -- before
#: any test module has had a chance to run its module-scope setdefaults.
_PRISTINE: dict[str, str] = _snapshot()

#: module nodeid -> ({key: value it set}, {keys it deleted})
_OWNED: dict[str, tuple[dict[str, str], set[str]]] = {}


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """Record what a test module changes while it is being imported.

    ``collect()`` is what performs the import, so wrapping the report that
    surrounds it is the one place where "before import" and "after import" are
    both observable.
    """
    if not isinstance(collector, pytest.Module) or _DISABLED:
        return (yield)

    before = _snapshot()
    try:
        return (yield)
    finally:
        after = _snapshot()
        added = {k: v for k, v in after.items() if before.get(k) != v}
        removed = {k for k in before if k not in after}
        _OWNED[collector.nodeid] = (added, removed)


def _owning_module(item) -> str | None:
    node = item
    while node is not None:
        if isinstance(node, pytest.Module):
            return node.nodeid
        node = node.parent
    return None


def pytest_runtest_setup(item):
    if _DISABLED:
        return
    added, removed = _OWNED.get(_owning_module(item), ({}, set()))

    for key in [k for k in os.environ if k.startswith(_MANAGED)]:
        del os.environ[key]
    os.environ.update(_PRISTINE)
    os.environ.update(added)
    for key in removed:
        os.environ.pop(key, None)
