"""The host-side elimination-prefix face history, and the row wire on it
(owner ruling 2026-09-15, item 2).

TWO HALVES.

``env.face_prefix_step`` is the store: one row per environment per step in,
the whole history out, and a RAISE on any step-index disagreement. Those tests
run in this process on plain arrays -- no graph, no policy.

The exactness half runs ``tests/face_row_wire.py`` in two child interpreters,
one per setting of ``ALPHAGRAD_FACE_ROW_WIRE``, and compares the sha of
``policy_regression_gate.run_trace``'s whole semantic trace. In children
because the gate pins the interpreter at module scope and because
``face_driver`` resolves the wire once per process.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alphagrad.approx.env import (
    face_prefix_reset, face_prefix_stats, face_prefix_step)

_HARNESS = Path(__file__).with_name("face_row_wire.py")

W, S, T = 4, 3, 6          # face columns, wire slots, episode steps
B = 2                      # environments


def _row(v):
    return np.full((B, W, S, 3), v, np.int32)


def _skip(v):
    return np.full((B, W), v, np.int32)


@pytest.fixture(autouse=True)
def _fresh():
    face_prefix_reset()
    face_prefix_stats()
    yield
    face_prefix_reset()
    face_prefix_stats()


# ------------------------------------------------------------- the store
def test_the_prefix_grows_one_row_per_step_and_reads_back_whole():
    """Row n-1 in at prefix n; rows 0..n-1 out, in order."""
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    for n in range(1, T + 1):
        rows, skips, joins = face_prefix_step(
            np.full((B,), n), _row(n), _skip(n), T)
        assert joins is None
        for i in range(B):
            for k in range(n):
                np.testing.assert_array_equal(rows[i, k], np.full(
                    (W, S, 3), k + 1, np.int32))
                np.testing.assert_array_equal(skips[i, k], np.full(
                    (W,), k + 1, np.int32))
    st = face_prefix_stats()
    assert st["append"] == B * T, st


def test_a_sibling_call_of_the_same_step_verifies_instead_of_appending():
    """Four callbacks a step ask at the same prefix; only the first extends."""
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    face_prefix_step(np.ones((B,), np.int64), _row(7), _skip(7), T)
    for _ in range(3):
        face_prefix_step(np.ones((B,), np.int64), _row(7), _skip(7), T)
    st = face_prefix_stats()
    assert st["append"] == B, st
    assert st["verify"] == 3 * B, st


def test_a_gap_in_the_step_index_raises():
    """Never truncate, never resynchronise: the prefix cannot be rebuilt."""
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    face_prefix_step(np.ones((B,), np.int64), _row(1), _skip(1), T)
    with pytest.raises(RuntimeError, match=r"holds 1 rows and the device is "
                                           r"at step 3"):
        face_prefix_step(np.full((B,), 3), _row(3), _skip(3), T)


def test_a_row_that_changed_under_the_prefix_raises():
    """A repeat at the same prefix length with other decisions is a defect,
    not something to patch over: the tokenizer would replay a graph the plan
    was never measured on."""
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    face_prefix_step(np.ones((B,), np.int64), _row(1), _skip(1), T)
    with pytest.raises(RuntimeError, match="carries a face row the host "
                                           "prefix does not hold"):
        face_prefix_step(np.ones((B,), np.int64), _row(9), _skip(1), T)


def test_step_zero_empties_the_prefix_and_so_does_the_reset():
    """A repeat restarts at step 0, which is the first of the two guards; the
    driver calls `face_prefix_reset` at the top of every attempt, which is the
    second."""
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    for n in (1, 2, 3):
        face_prefix_step(np.full((B,), n), _row(n), _skip(n), T)
    face_prefix_stats()
    # step 0 again: the repeat's own first callback
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    rows, _sk, _j = face_prefix_step(np.ones((B,), np.int64),
                                     _row(50), _skip(50), T)
    np.testing.assert_array_equal(rows[0, 0], np.full((W, S, 3), 50, np.int32))
    assert face_prefix_stats()["append"] == B

    # and the explicit reset, without a step 0 in front of it
    for n in (2, 3):
        face_prefix_step(np.full((B,), n), _row(n), _skip(n), T)
    face_prefix_reset()
    face_prefix_stats()
    rows, _sk, _j = face_prefix_step(np.ones((B,), np.int64),
                                     _row(60), _skip(60), T)
    np.testing.assert_array_equal(rows[0, 0], np.full((W, S, 3), 60, np.int32))


def test_the_join_channel_rides_the_same_prefix():
    """--approx-add choose carries a per-face join bit; the env step callback
    is the only caller that has one, and it lands at the same index."""
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    for n in (1, 2):
        _r, _s, joins = face_prefix_step(
            np.full((B,), n), _row(n), _skip(n), T,
            joins=np.full((B, W), n * 10, np.int32))
        assert joins is not None
        np.testing.assert_array_equal(
            joins[0, n - 1], np.full((W,), n * 10, np.int32))


def test_a_prefix_past_the_episode_length_raises():
    face_prefix_step(np.zeros((B,), np.int64), _row(-1), _skip(0), T)
    with pytest.raises(RuntimeError, match="the history holds"):
        face_prefix_step(np.full((B,), T + 1), _row(1), _skip(1), T)


# ------------------------------------------------------------- exactness
def _run(row_wire: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
    env["ALPHAGRAD_FACE_ROW_WIRE"] = row_wire
    return subprocess.run([sys.executable, str(_HARNESS)],
                          env=env, capture_output=True, text=True,
                          timeout=3600)


def _field(out: str, name: str) -> str:
    m = re.search(rf"^\[row-wire\] {name}=(\S+)$", out, re.M)
    assert m is not None, f"no {name} in:\n{out}"
    return m.group(1)


def test_the_row_wire_leaves_the_whole_policy_trace_identical():
    """One row on the wire, or the whole history: the same rollout."""
    on = _run("1")
    off = _run("0")
    for name, r in (("row wire on", on), ("row wire off", off)):
        assert r.returncode == 0, (
            f"{name} (rc={r.returncode})\n--- stdout ---\n{r.stdout}\n"
            f"--- stderr ---\n{r.stderr}")
    assert "-> 1" in on.stdout, on.stdout
    assert "-> 0" in off.stdout, off.stdout
    assert _field(on.stdout, "sha") == _field(off.stdout, "sha"), (
        f"the row wire moved the policy trace\n--- on ---\n{on.stdout}\n"
        f"--- off ---\n{off.stdout}")
    assert int(_field(on.stdout, "steps")) >= 2, on.stdout
