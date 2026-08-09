"""Pin ALPHAGRAD_FORCE_REV_ORDER: availability reduces to the single highest
remaining vertex; terminal all-zero stays all-zero; flag off is identity."""
import numpy as np
import jax.numpy as jnp
import pytest
from types import SimpleNamespace as NS

from alphagrad.approx.common import masks as M


def _avail(chosen, step, static, total_v, num_valid):
    st = NS(order=jnp.asarray(chosen, jnp.int32),
            step_count=jnp.asarray(step, jnp.int32))
    return np.asarray(M.vertex_avail_at_step(st, static, total_v, num_valid))


def test_force_rev_selects_highest_remaining(monkeypatch):
    monkeypatch.setattr(M, "_FORCE_REV", True)
    static = M.build_vertex_valid_static([1, 2, 3, 4, 5], 6)
    a = _avail([5, 0, 0, 0, 0], 1, static, 6, 5)   # 5 eliminated
    assert a.tolist() == [0, 0, 0, 1, 0, 0]        # vertex 4 forced


def test_force_rev_first_step_is_top_vertex(monkeypatch):
    monkeypatch.setattr(M, "_FORCE_REV", True)
    static = M.build_vertex_valid_static([1, 2, 3, 4, 5], 6)
    a = _avail([0, 0, 0, 0, 0], 0, static, 6, 5)
    assert a.tolist() == [0, 0, 0, 0, 1, 0]        # vertex 5 first ('rev')


def test_force_rev_terminal_all_zero(monkeypatch):
    monkeypatch.setattr(M, "_FORCE_REV", True)
    static = M.build_vertex_valid_static([1, 2], 3)
    a = _avail([1, 2], 2, static, 3, 2)
    assert a.tolist() == [0, 0, 0]


def test_flag_off_identity(monkeypatch):
    monkeypatch.setattr(M, "_FORCE_REV", False)
    static = M.build_vertex_valid_static([1, 2, 3, 4, 5], 6)
    a = _avail([5, 0, 0, 0, 0], 1, static, 6, 5)
    assert a.tolist() == [1, 1, 1, 1, 0, 0]
