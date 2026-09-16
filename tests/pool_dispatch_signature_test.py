"""THE POOL'S DISPATCH KEYWORDS MUST BIND TO THE ACTOR AND TO THE SERVER.

Twice now a field was added to ``CpuApproxPool`` and forwarded to
``actor.evaluate.remote(...)`` while the Ray wrapper in
``cpu_approx_actors.CpuApproximationActor`` was left behind:

  * ``episode`` (A3's pooled walk rotation, commit 4c4d872), and
  * ``env_row`` (the per-environment step-position draw, 2026-09-16).

Both failures look the same from outside: every pooled dispatch dies with
``TypeError: got an unexpected keyword argument ...``, the pool sentinels the
row and kills the actor, the remaining slots come back "pool-drained", and
under ``--ray-measure`` NO plan is measured at all. The only visible symptom
is a ``[SENTINEL]`` line the trainer does not gate on, so a whole campaign can
run to the end having measured nothing.

These tests read the keywords the pool ACTUALLY passes out of its own source
and bind them against the two signatures downstream. There is no Ray, no
actor and no env here: it is a signature contract and it is checked as one.
"""

from __future__ import annotations

import inspect
import re

import pytest

from alphagrad.approx import cpu_approx_pool as pool_mod
from alphagrad.approx.cpu_approx_actors import CpuApproximationActor
from alphagrad.approx.cpu_approx_worker import CpuApproximationServer

#: The keywords the pool passes to ``actor.evaluate.remote``. Read off the
#: source rather than written out here, so a new one is picked up the day it
#: is added instead of the day a campaign measures nothing.
_DISPATCH = re.compile(
    r"\.evaluate\.remote\((?P<body>.*?)\n\s*\)", re.S)


def _dispatch_keywords() -> set:
    src = inspect.getsource(pool_mod)
    out: set = set()
    for m in _DISPATCH.finditer(src):
        body = m.group("body")
        # one level of nesting is enough for these call sites; a keyword is
        # ``name=`` at the start of a line or just after a comma.
        for kw in re.findall(r"(?:^|,)\s*([a-z_][a-z0-9_]*)\s*=", body):
            out.add(kw)
    return out


def test_the_pool_has_dispatch_sites_to_read():
    kws = _dispatch_keywords()
    assert kws, "no `.evaluate.remote(...)` call site found in the pool"
    # the two that were forgotten, and the ones that were not
    for name in ("eval_samples", "init", "episode", "env_row"):
        assert name in kws, (name, sorted(kws))


@pytest.mark.parametrize("fn,where", [
    (CpuApproximationActor.evaluate, "the Ray wrapper"),
    (CpuApproximationServer.evaluate, "the server"),
])
def test_every_dispatch_keyword_binds(fn, where):
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        pytest.skip(f"{where} takes **kwargs; nothing to pin")
    missing = sorted(k for k in _dispatch_keywords() if k not in params)
    assert not missing, (
        f"{where} does not accept {missing}, and the pool passes them on "
        f"every dispatch. Every pooled measurement would die with "
        f"TypeError and the trainer would sentinel every plan.")


def test_the_wrapper_forwards_what_it_accepts():
    """A parameter the wrapper takes and drops on the floor is the same
    defect with a quieter symptom: the request arrives without the field."""
    src = inspect.getsource(CpuApproximationActor.evaluate)
    body = src.split("return self._impl.evaluate", 1)
    assert len(body) == 2, "the wrapper no longer forwards to the server"
    forwarded = set(re.findall(r"([a-z_][a-z0-9_]*)\s*=", body[1]))
    taken = {n for n in inspect.signature(
        CpuApproximationActor.evaluate).parameters
        if n not in ("self", "order", "sparsity_specs", "step")}
    dropped = sorted(taken - forwarded)
    assert not dropped, (
        f"the Ray wrapper accepts {dropped} and does not forward them")
