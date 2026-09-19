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

READ FROM THE SOURCE, NOT FROM THE OBJECTS. ``CpuApproximationActor`` is
decorated with ``@ray.remote``, so the name in the module is a Ray
``ActorClass`` and not the class that was written; its ``evaluate`` is not a
plain function and ``inspect.signature`` does not describe the wire. Every
check here therefore parses the three modules with ``ast``. No Ray, no actor,
no env: it is a signature contract and it is checked as one.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from alphagrad.approx import cpu_approx_actors as actors_mod
from alphagrad.approx import cpu_approx_pool as pool_mod
from alphagrad.approx import cpu_approx_worker as worker_mod


def _tree(mod):
    return ast.parse(inspect.getsource(mod))


def _dispatch_keywords() -> set:
    """Every keyword any ``<actor>.evaluate.remote(...)`` call in the pool
    passes.

    PARSED, not matched. A regular expression over the source picks up the
    ``dtype=`` of a nested ``np.asarray`` call as well, and then the test
    fails for a reason that has nothing to do with the contract.
    """
    out: set = set()
    for node in ast.walk(_tree(pool_mod)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "remote"):
            continue
        inner = fn.value
        if not (isinstance(inner, ast.Attribute) and inner.attr == "evaluate"):
            continue
        for kw in node.keywords:
            if kw.arg:
                out.add(kw.arg)
    return out


def _method(mod, cls_name: str, fn_name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(mod)):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == fn_name:
                    return sub
    raise AssertionError(f"{cls_name}.{fn_name} not found in {mod.__name__}")


def _parameters(fn: ast.FunctionDef) -> set:
    a = fn.args
    names = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add("*")
    if a.kwarg:
        names.add("**")
    return names


_SITES = [
    (actors_mod, "CpuApproximationActor", "the Ray wrapper"),
    (worker_mod, "CpuApproximationServer", "the server"),
]


def test_the_pool_has_dispatch_sites_to_read():
    kws = _dispatch_keywords()
    assert kws, "no `.evaluate.remote(...)` call site found in the pool"
    # the two that were forgotten, and some of the ones that were not
    for name in ("eval_samples", "init", "episode", "env_row"):
        assert name in kws, (name, sorted(kws))


@pytest.mark.parametrize("mod,cls,where", _SITES)
def test_every_dispatch_keyword_binds(mod, cls, where):
    params = _parameters(_method(mod, cls, "evaluate"))
    if "**" in params:
        pytest.skip(f"{where} takes **kwargs; nothing to pin")
    missing = sorted(k for k in _dispatch_keywords() if k not in params)
    assert not missing, (
        f"{where} does not accept {missing}, and the pool passes them on "
        f"every dispatch. Every pooled measurement would die with a "
        f"TypeError and the trainer would sentinel every plan.")


def test_the_wrapper_forwards_what_it_accepts():
    """A parameter the wrapper takes and drops on the floor is the same
    defect with a quieter symptom: the request arrives without the field."""
    fn = _method(actors_mod, "CpuApproximationActor", "evaluate")
    forwarded: set = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "evaluate"):
            forwarded |= {kw.arg for kw in node.keywords if kw.arg}
    assert forwarded, "the wrapper no longer forwards to the server"
    taken = {n for n in _parameters(fn)
             if n not in ("self", "order", "sparsity_specs", "step")}
    dropped = sorted(taken - forwarded)
    assert not dropped, (
        f"the Ray wrapper accepts {dropped} and does not forward them")
