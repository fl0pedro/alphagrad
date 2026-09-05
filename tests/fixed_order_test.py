"""The fixed elimination order as an argument (ticket dsnn-3qm.64).

``vertex_avail_at_step(..., fixed_order=table)`` keeps exactly ONE legal
vertex per step (the table's entry), terminal all-zero stays all-zero, and
``fixed_order=None`` is the identity. The reverse table reproduces what the
deleted ``ALPHAGRAD_FORCE_REV_ORDER=1`` pin did (the highest remaining valid
vertex). A process that still exports that variable fails at import.
"""
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import jax.numpy as jnp

from alphagrad.approx.common import masks as M
from alphagrad.approx.common.order import (
    FIXED_ORDER_CHOICES, fixed_order_table, reverse_order)


def _avail(chosen, step, static, total_v, num_valid, table=None):
    st = NS(order=jnp.asarray(chosen, jnp.int32),
            step_count=jnp.asarray(step, jnp.int32))
    return np.asarray(M.vertex_avail_at_step(st, static, total_v, num_valid,
                                             fixed_order=table))


def test_reverse_table_selects_highest_remaining():
    static = M.build_vertex_valid_static([1, 2, 3, 4, 5], 6)
    table = reverse_order([1, 2, 3, 4, 5])
    assert table.tolist() == [5, 4, 3, 2, 1]
    a = _avail([5, 0, 0, 0, 0], 1, static, 6, 5, table)   # 5 eliminated
    assert a.tolist() == [0, 0, 0, 1, 0, 0]                # vertex 4 forced


def test_first_step_is_the_table_head():
    static = M.build_vertex_valid_static([1, 2, 3, 4, 5], 6)
    table = np.array([2, 5, 1, 4, 3], dtype=np.int32)
    a = _avail([0, 0, 0, 0, 0], 0, static, 6, 5, table)
    assert a.tolist() == [0, 1, 0, 0, 0, 0]                # vertex 2 first


def test_exactly_one_legal_vertex_at_every_step():
    valid = [1, 2, 3, 4, 5]
    static = M.build_vertex_valid_static(valid, 6)
    table = np.array([2, 5, 1, 4, 3], dtype=np.int32)
    chosen = [0] * 5
    seen = []
    for k in range(5):
        a = _avail(chosen, k, static, 6, 5, table)
        assert int(a.sum()) == 1, (k, a)
        v = int(np.argmax(a)) + 1
        seen.append(v)
        chosen[k] = v
    assert seen == table.tolist()


def test_terminal_all_zero():
    static = M.build_vertex_valid_static([1, 2], 3)
    table = np.array([2, 1], dtype=np.int32)
    a = _avail([2, 1], 2, static, 3, 2, table)
    assert a.tolist() == [0, 0, 0]


def test_none_is_identity():
    static = M.build_vertex_valid_static([1, 2, 3, 4, 5], 6)
    a = _avail([5, 0, 0, 0, 0], 1, static, 6, 5, None)
    assert a.tolist() == [1, 1, 1, 1, 0, 0]


def test_choices_and_free_table():
    assert FIXED_ORDER_CHOICES == ("free", "reverse", "markowitz")
    assert fixed_order_table("free", None, (), (), (), [1, 2]) is None
    assert fixed_order_table("reverse", None, (), (), (), [1, 2]).tolist() == [2, 1]


def test_the_env_var_fails_loudly_at_import():
    env = dict(os.environ, ALPHAGRAD_FORCE_REV_ORDER="1")
    r = subprocess.run(
        [sys.executable, "-c", "import alphagrad.approx.common.masks"],
        env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "ALPHAGRAD_FORCE_REV_ORDER is no longer read" in r.stderr
