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

#: Dispatch keywords the WRAPPER consumes and does not forward.
#:
#: ``rule`` names the graph the plan was acted on (owner ruling 2026-09-22,
#: the alternating run). The wrapper holds one server per graph and the
#: keyword is how it picks between them, so it MUST stop there: a server is
#: one graph and has nothing to do with the question. Every other keyword is
#: part of the request and must reach the server.
_CONSUMED_BY_THE_WRAPPER = {"rule"}


def test_the_pool_has_dispatch_sites_to_read():
    kws = _dispatch_keywords()
    assert kws, "no `.evaluate.remote(...)` call site found in the pool"
    # the two that were forgotten, and some of the ones that were not
    for name in ("eval_samples", "init", "episode", "env_row", "rule"):
        assert name in kws, (name, sorted(kws))


@pytest.mark.parametrize("mod,cls,where", _SITES)
def test_every_dispatch_keyword_binds(mod, cls, where):
    params = _parameters(_method(mod, cls, "evaluate"))
    if "**" in params:
        pytest.skip(f"{where} takes **kwargs; nothing to pin")
    expected = _dispatch_keywords()
    if cls == "CpuApproximationServer":
        # The wrapper answers these itself; see _CONSUMED_BY_THE_WRAPPER.
        expected = expected - _CONSUMED_BY_THE_WRAPPER
    missing = sorted(k for k in expected if k not in params)
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
    dropped = sorted(taken - forwarded - _CONSUMED_BY_THE_WRAPPER)
    assert not dropped, (
        f"the Ray wrapper accepts {dropped} and does not forward them")


def test_every_env_side_dispatch_names_the_graph():
    """EVERY call into the pool says which graph the plan was acted on.

    This is the same defect class as ``episode`` and ``env_row``, one hop
    earlier: the pool and the wrapper both took ``rule`` and the PIPELINED
    submission in ``env.py`` (``pool.submit_batch``) did not pass it, so
    under ``--measure-pipeline 1`` every terminal batch reached an actor that
    holds two graphs with no graph named. Measured: job 67527 on gpu14 and
    probe 67528 on cpu1, every terminal plan sentinelled.

    Read from the source, like the rest of this module: these are call sites,
    not signatures, and only the source says what a call site passes.
    """
    from alphagrad.approx import env as env_mod

    tree = _tree(env_mod)
    # `name = dict(...)` keyword sets, so a site that passes its request as
    # `**_pipe_kw` is read through the dict it unpacks rather than counted as
    # passing nothing.
    dict_kw: dict = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "dict"):
            dict_kw.setdefault(node.targets[0].id, set()).update(
                kw.arg for kw in node.value.keywords if kw.arg)
    # A deferred submission carries its request into a lambda as a DEFAULT
    # argument (`lambda _k=_pipe_kw: pool.submit_batch(**_k)`), so the alias
    # is followed; otherwise the one site that packages a batch for the
    # pipeline reads as passing nothing, which is the site that was wrong.
    for _ in range(4):                       # aliases of aliases
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                for arg, default in zip(node.args.args[::-1],
                                        node.args.defaults[::-1]):
                    if (isinstance(default, ast.Name)
                            and default.id in dict_kw):
                        dict_kw.setdefault(arg.arg, set()).update(
                            dict_kw[default.id])
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)
                  and isinstance(node.value, ast.Name)
                  and node.value.id in dict_kw):
                dict_kw.setdefault(node.targets[0].id, set()).update(
                    dict_kw[node.value.id])
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            continue
        if fn.attr not in ("evaluate", "evaluate_batch", "submit_batch"):
            continue
        # the tokenize-only pool serves its own actors and has no graph to
        # name; it is reached through `_tokpool`, never through `pool`.
        base = fn.value
        if isinstance(base, ast.Name) and base.id.startswith("_tok"):
            continue
        kws = {kw.arg for kw in node.keywords if kw.arg}
        for kw in node.keywords:
            if kw.arg is None and isinstance(kw.value, ast.Name):
                kws |= dict_kw.get(kw.value.id, set())
        sites.append((fn.attr, node.lineno, kws))
    assert sites, "no pool dispatch site found in env.py"
    missing = [(n, ln) for n, ln, kws in sites if "rule" not in kws]
    assert not missing, (
        f"these env.py dispatch sites do not name the graph: {missing}. An "
        f"actor that holds two graphs raises on a dispatch that names none, "
        f"and every plan in that batch comes back sentinelled.")


def test_a_keyword_the_wrapper_consumes_is_actually_read():
    """A consumed keyword that nothing reads is the dropped-field defect with
    the exemption written in. ``rule`` decides WHICH graph's server measures
    the plan, so the wrapper's body has to mention it."""
    fn = _method(actors_mod, "CpuApproximationActor", "evaluate")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    for kw in _CONSUMED_BY_THE_WRAPPER:
        assert kw in _parameters(fn), (
            f"{kw} is listed as consumed by the wrapper and the wrapper does "
            f"not take it")
        assert kw in names, (
            f"the wrapper takes {kw} and its body never reads it, so the "
            f"dispatch that sends it changes nothing")
