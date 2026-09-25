from __future__ import annotations

import ast
import collections
import os

import pytest

_ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src", "alphagrad", "approx", "env.py")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "dsnn-dfw.56 is still true at f825993: env.py defines "
    "_warn_cosine_is_now_grad_cosine and _grad_cosine_k twice"))
def test_env_defines_each_top_level_name_once():
    with open(_ENV, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=_ENV)
    lines = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            lines[node.name].append(node.lineno)
    twice = {name: at for name, at in lines.items() if len(at) > 1}
    assert not twice, (
        f"env.py defines these top-level names more than once, and only the "
        f"last definition is live: {twice}")
