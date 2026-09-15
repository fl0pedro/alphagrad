"""THE LIVE-FACE WIRE: a compact prefix key, and a narrower operand.

Two changes, one defect seen from two sides. The rollout profile of
2026-09-15 measured the face path at 246 milliseconds of host Python and 117
megabytes of device-to-host copy per callback per step, against a MEASURED
occupancy of one face per vertex, maximum thirteen, of a cap of 1920.

* `LiveFaceStream.hist_key` replaces the dense `frh[:n].tobytes()` prefix key
  -- 6.5 megabytes, hashed about five times per environment per step -- with
  the positions and values of the entries that are not padding, built once
  per step by extending the previous step's key.
* `--face-wire-faces N` hands the callbacks the first N face columns instead
  of all of them.

What these tests pin:

  THE KEY IS INJECTIVE and the chain equals the cold rebuild, over an
  episode, over a repeat of an episode, and when the last row changes at the
  same prefix length.

  THE NARROW WIRE IS LOUD. A vertex with more faces than the wire carries
  stops the run, at the face count and again at the prefix replay. The
  builder's own loop would have run its tail exact in silence, which is the
  fault class the face width exists for.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import numpy as np
import pytest

import alphagrad.approx.live_faces as LF


# --------------------------------------------------------------- helpers

MAXF, SLOTS, N = 8, 3, 6


def _stream():
    """A stream object with only what `hist_key` touches.

    `hist_key` reads `self._histkeys` and `self.stats` and nothing else, and
    building a real graph here would test graphax rather than the key.
    """
    s = LF.LiveFaceStream.__new__(LF.LiveFaceStream)
    s._histkeys = {}
    s.stats = {"hist_key_ext": 0, "hist_key_cold": 0}
    return s


def _blank(n=N):
    """An empty history: all faces padded, no skips."""
    return (-np.ones((n, MAXF, SLOTS, 3), np.int32),
            np.zeros((n, MAXF), np.int32))


def _decide(frh, fsh, step, face, slot=0, row=(1, 2, 3)):
    frh[step, face, slot] = np.asarray(row, np.int32)


def _cold(frh, fsh, n):
    """The key the chain is supposed to agree with."""
    return tuple(LF._face_row_key(frh[j], fsh[j]) for j in range(n))


# ------------------------------------------------------- 1. the row key

def test_a_row_key_separates_two_different_rows():
    frh, fsh = _blank()
    a = LF._face_row_key(frh[0], fsh[0])
    _decide(frh, fsh, 0, face=2)
    b = LF._face_row_key(frh[0], fsh[0])
    assert a != b
    # And it is a FUNCTION of the row: the same row twice is the same key.
    assert b == LF._face_row_key(frh[0], fsh[0])


def test_a_row_key_separates_a_skip_from_no_decision():
    frh, fsh = _blank()
    a = LF._face_row_key(frh[0], fsh[0])
    fsh[0, 3] = 1
    assert LF._face_row_key(frh[0], fsh[0]) != a


def test_a_row_key_separates_the_same_value_at_a_different_face():
    frh, fsh = _blank()
    _decide(frh, fsh, 0, face=1)
    a = LF._face_row_key(frh[0], fsh[0])
    frh[0] = -1
    _decide(frh, fsh, 0, face=5)
    assert LF._face_row_key(frh[0], fsh[0]) != a


def test_a_row_key_is_far_smaller_than_the_row():
    """The whole point, at the CAMPAIGN's width. 1920 face columns of
    padding with one decision in them become a few dozen bytes."""
    row = -np.ones((1920, SLOTS, 3), np.int32)
    skip = np.zeros((1920,), np.int32)
    row[7, 0] = np.asarray((1, 2, 3), np.int32)
    k = LF._face_row_key(row, skip)
    assert sum(len(x) for x in k) < row.nbytes // 1000


# --------------------------------------------------------- 2. the chain

def test_the_chain_equals_the_cold_rebuild_over_an_episode():
    """The rollout asks for prefixes 0, 1, 2, ... and writes row n-1 as it
    goes. The chain must give the same key a cold rebuild would."""
    s = _stream()
    frh, fsh = _blank()
    for n in range(N + 1):
        if n > 0:
            _decide(frh, fsh, n - 1, face=(n % 3))
        got = s.hist_key(0, frh, fsh, n)
        assert got == _cold(frh, fsh, n), n
    assert s.stats["hist_key_ext"] == N


def test_the_same_step_asked_twice_gives_one_key_and_no_rebuild():
    """Five callbacks per step ask for the same prefix; the key is built
    once."""
    s = _stream()
    frh, fsh = _blank()
    _decide(frh, fsh, 0, face=1)
    first = s.hist_key(0, frh, fsh, 1)
    n_ext = s.stats["hist_key_ext"]
    n_cold = s.stats["hist_key_cold"]
    for _ in range(4):
        assert s.hist_key(0, frh, fsh, 1) == first
    assert s.stats["hist_key_ext"] == n_ext
    assert s.stats["hist_key_cold"] == n_cold


def test_a_repeated_episode_does_not_reuse_the_discarded_attempts_chain():
    """A window-bin overflow repeats the episode from step_count 0 with
    DIFFERENT decisions. Serving the discarded attempt's key would serve its
    tokenizer, and the head would read another plan's tokens."""
    s = _stream()
    frh, fsh = _blank()
    for n in range(1, N + 1):
        _decide(frh, fsh, n - 1, face=0)
        s.hist_key(0, frh, fsh, n)
    keep = _cold(frh, fsh, N)
    # The repeat: same prefix lengths, other decisions.
    frh2, fsh2 = _blank()
    for n in range(N + 1):
        if n > 0:
            _decide(frh2, fsh2, n - 1, face=4, row=(7, 7, 7))
        got = s.hist_key(0, frh2, fsh2, n)
        assert got == _cold(frh2, fsh2, n), n
    assert s.hist_key(0, frh2, fsh2, N) != keep


def test_a_changed_last_row_at_the_same_prefix_length_changes_the_key():
    """The last row is the one this step decided, so it is never taken on
    trust."""
    s = _stream()
    frh, fsh = _blank()
    _decide(frh, fsh, 0, face=1)
    a = s.hist_key(0, frh, fsh, 1)
    frh[0] = -1
    _decide(frh, fsh, 0, face=2)
    b = s.hist_key(0, frh, fsh, 1)
    assert a != b and b == _cold(frh, fsh, 1)


def test_each_environment_keeps_its_own_chain():
    s = _stream()
    a_r, a_s = _blank()
    b_r, b_s = _blank()
    for n in range(1, N + 1):
        _decide(a_r, a_s, n - 1, face=0)
        _decide(b_r, b_s, n - 1, face=3, row=(9, 9, 9))
        ka = s.hist_key(0, a_r, a_s, n)
        kb = s.hist_key(1, b_r, b_s, n)
        assert ka == _cold(a_r, a_s, n)
        assert kb == _cold(b_r, b_s, n)
        assert ka != kb


def test_a_prefix_asked_out_of_order_is_rebuilt_cold():
    """Anything that is not "one more than last time" takes the honest
    rebuild rather than guessing."""
    s = _stream()
    frh, fsh = _blank()
    for j in range(N):
        _decide(frh, fsh, j, face=j % 4)
    s.hist_key(0, frh, fsh, 1)
    n_cold = s.stats["hist_key_cold"]
    assert s.hist_key(0, frh, fsh, 5) == _cold(frh, fsh, 5)
    assert s.stats["hist_key_cold"] == n_cold + 1


def test_the_switch_off_returns_no_key_so_the_caller_takes_the_dense_one():
    s = _stream()
    frh, fsh = _blank()
    _decide(frh, fsh, 0, face=1)
    old = LF._FACE_KEY_CHAIN
    try:
        LF._FACE_KEY_CHAIN = False
        assert s.hist_key(0, frh, fsh, 1) is None
    finally:
        LF._FACE_KEY_CHAIN = old


def test_verification_catches_a_chain_that_has_gone_stale():
    """ALPHAGRAD_FACE_KEY_VERIFY=1 rebuilds the chain every call. Corrupt a
    row BELOW the last one, which is the one case the chain takes on trust,
    and the check must fire."""
    s = _stream()
    frh, fsh = _blank()
    for n in range(1, 4):
        _decide(frh, fsh, n - 1, face=1)
        s.hist_key(0, frh, fsh, n)
    frh[0] = -1
    _decide(frh, fsh, 0, face=6)        # a row the chain will not re-read
    old = LF._FACE_KEY_VERIFY
    try:
        LF._FACE_KEY_VERIFY = True
        _decide(frh, fsh, 3, face=1)
        with pytest.raises(RuntimeError, match="desynchronised"):
            s.hist_key(0, frh, fsh, 4)
    finally:
        LF._FACE_KEY_VERIFY = old


# ------------------------------------------------- 3. the narrowed wire

def test_the_wire_width_is_the_full_cap_until_it_is_configured():
    from alphagrad.approx import env as E

    E.configure_face_wire_faces(0)
    try:
        assert E.face_wire_faces() == E.MAX_FACES
        E.configure_face_wire_faces(4)
        assert E.face_wire_faces() == min(4, E.MAX_FACES)
        # And it never widens past the provable bound.
        E.configure_face_wire_faces(E.MAX_FACES + 1000)
        assert E.face_wire_faces() == E.MAX_FACES
        with pytest.raises(ValueError):
            E.configure_face_wire_faces(-1)
    finally:
        E.configure_face_wire_faces(0)


def test_the_face_count_refuses_a_vertex_wider_than_the_wire():
    """The check that covers the WHOLE history: every vertex passes through
    the face count before its faces are decided."""
    s = LF.LiveFaceStream.__new__(LF.LiveFaceStream)
    s.max_faces = 64
    s.stats = {"failures": 0}
    s._tokenizer_at = lambda *a, **k: SimpleNamespace(
        ij=SimpleNamespace(faces=lambda v: [(i, i + 1) for i in range(5)]))
    frh, fsh = (-np.ones((3, 3, SLOTS, 3), np.int32),
                np.zeros((3, 3), np.int32))
    with pytest.raises(RuntimeError, match="face wire"):
        s.n_faces(np.zeros((3,), np.int32), np.zeros((3, 1, 3), np.int32),
                  2, 1, frh, fsh)
    # Wide enough: no refusal, and the count comes back.
    frh8, fsh8 = (-np.ones((3, 8, SLOTS, 3), np.int32),
                  np.zeros((3, 8), np.int32))
    assert s.n_faces(np.zeros((3,), np.int32),
                     np.zeros((3, 1, 3), np.int32), 2, 1, frh8, fsh8) == 5


def test_the_prefix_replay_refuses_a_row_narrower_than_its_face_list():
    """The backstop. `env._face_dict_for_vertex` BREAKS out of its loop when
    the row runs out, which would run the tail of a wide vertex exact while
    every counter reported a healthy run."""
    s = LF.LiveFaceStream.__new__(LF.LiveFaceStream)
    s.jaxpr = None
    s.max_faces = 64
    tk = SimpleNamespace(
        ij=SimpleNamespace(faces=lambda v: [(i, i + 1) for i in range(5)]))
    rows = -np.ones((3, SLOTS, 3), np.int32)
    skips = np.zeros((3,), np.int32)
    with pytest.raises(RuntimeError, match="face wire"):
        s._decided(tk, 1, rows, skips, 3)
