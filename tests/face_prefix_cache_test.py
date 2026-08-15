"""Prefix caches must be INVISIBLE: an extended replay must equal a cold one.

Covers the two O(T^2) -> O(T) host-overhead fixes:

  * `_face_transforms_for_order` pop-and-extend cache
    (ALPHAGRAD_FACE_ENUM_CACHE=1): the k-th step advances the cached
    IncrementalJaxpr by ONE elimination instead of rebuilding and replaying
    the whole prefix; and
  * `_incremental_stream_tokens` ancestor extension under FACE actions
    (per-step `face_key` signatures instead of one whole-prefix blob).

Soundness (asserted in stream_prefix_property_test.py): the decode has NO
position sensitivity, so the stream for a prefix is a byte-prefix of the
stream for any extension for every rule kind. Both caches therefore store
every state and extend from every parent — there is no COMPRESS carve-out any
more. There used to be one, because `decode_vertex_rule_specs` emitted
COMPRESS only when `is_last=True`; the tests below pin that COMPRESS is now
ordinary, in the per-vertex specs AND in the face rows (the v40 failure mode
it caused: ext=1/431).

Every test asserts the ENGAGEMENT COUNTERS too — an equality proof over a
fast path that never ran is vacuous.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alphagrad.approx.env as E
from alphagrad.approx.common.masks import make_live_masked_hook
from alphagrad.approx.env import (
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX,
    decode_vertex_rule_specs,
)
from graphax import SKIP_FACE


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)
ARGNUMS = (0, 1)


def _mk():
    cj = jax.make_jaxpr(_fn)(*ARGS)
    cfg = SimpleNamespace(jaxpr=cj.jaxpr, argnums=ARGNUMS, per_face=True)
    return cfg, list(cj.literals), list(ARGS)


def _noop_rows():
    return [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]


def _episode(compress_at=None, face_compress_at=None, seed=0):
    """A full random episode: order + per-vertex specs + per-face rows/skips.

    `compress_at` / `face_compress_at` are POSITIONS IN THE ORDER (0-based):
    the prefix of length > that position carries the COMPRESS row.
    """
    rng = np.random.default_rng(seed)
    cfg, consts, args = _mk()
    T = len(cfg.jaxpr.eqns)
    order = [int(v) for v in rng.permutation(np.arange(1, T + 1))]
    specs, faces, skips = [], [], []
    for i in range(T):
        rows = _noop_rows()
        if rng.random() < 0.4:
            rows[0] = [0, 0, 2]
        if compress_at is not None and i == compress_at:
            rows[0] = [COMPRESS_SENTINEL, 0, 0]
        specs.append(rows)
        fr = np.full((MAX_FACES, FACE_SLOTS, 3), -1, dtype=np.int64)
        fr[..., 2] = 0
        sk = np.zeros((MAX_FACES,), dtype=np.int64)
        for f in range(4):
            r = rng.random()
            if r < 0.3:
                fr[f, 0] = (0, 0, 2)
            elif r < 0.4:
                sk[f] = 1
        if face_compress_at is not None and i == face_compress_at:
            fr[0, 0] = (COMPRESS_SENTINEL, 0, 0)
        faces.append(fr)
        skips.append(sk)
    return cfg, consts, args, order, specs, faces, skips


def _ft_sig(ft):
    """Structural signature of a ft_by_vertex dict: face keys, SKIP placement,
    slot occupancy. (Rule CONTENT equality is proven end-to-end by the stream
    tests below — the hooks are consumed by the tokenizer's eliminations.)"""
    sig = {}
    for v, per_face in ft.items():
        entry = {}
        for key, val in per_face.items():
            entry[key] = "SKIP" if val is SKIP_FACE else tuple(
                s is not None for s in val)
        sig[int(v)] = entry
    return sig


def _tok_rules(cfg, o_list, specs_list):
    d = {}
    for i, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(cfg.jaxpr, int(v), specs_list[i])
        if rules:
            d[int(v)] = (make_live_masked_hook(tuple(rules)),)
    return d


def _fk(faces, skips, t):
    return tuple(
        (tuple(int(x) for x in np.asarray(faces[k]).reshape(-1)),
         tuple(int(x) for x in np.asarray(skips[k]).reshape(-1)))
        for k in range(t))


def _check_last_start(prev_out, stream, last_start):
    """`last_start` must index the first token of THIS elimination's block.

    Under EnvConfig.delta_obs that slice is the whole observation, so if it
    is off by anything the policy reads a different graph than the one the
    measurement builds -- and with face wires in play (which is what this
    module drives) the block is where every approximation echo lives.

    The prefix property holds unconditionally now, so this is asserted for
    every step (the guard below is kept because the very first prefix has no
    predecessor to compare against).
    """
    assert 0 <= last_start <= len(stream)
    if not prev_out:
        # Length-1 prefix: the block is the FIRST elimination, so it starts
        # after the base -- strictly inside the stream, never at 0.
        assert 0 < last_start < len(stream)
        return
    prev = prev_out[-1][0]
    if list(stream[:len(prev)]) == list(prev):
        assert last_start == len(prev), (
            f"delta block starts at {last_start}, previous prefix ended at "
            f"{len(prev)} -- the emitted delta would be the wrong tokens")


def _reset(d):
    d.update({k: 0 for k in d})


def _ft_pass(cfg, consts, args, order, specs, faces, skips, cached):
    os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "1" if cached else "0"
    # This file is about the BRANCHING-search prefix cache. The default path
    # is now the live elimination state (face_live_state_test.py), which would
    # otherwise serve every call here and leave the cache counters at 0.
    # getattr/delattr rather than a bare read: `_FACE_LIVE_STATE` is an
    # IN-FLIGHT symbol (the live-elimination-state rework) that env.py does not
    # define yet, and a bare read makes every test in this file error out on a
    # tree that does not have it. Once it lands this is exactly the same
    # save/restore.
    _live = getattr(E, "_FACE_LIVE_STATE", None)
    E._FACE_LIVE_STATE = False
    try:
        E._FACE_ENUM_CACHE.clear()
        _reset(E._FACE_ENUM_STATS)
        out = []
        for t in range(1, len(order) + 1):
            out.append(E._face_transforms_for_order(
                cfg, consts, args, order[:t], specs[:t],
                [np.asarray(f).tolist() for f in faces[:t]],
                [np.asarray(s).tolist() for s in skips[:t]]))
        return out, dict(E._FACE_ENUM_STATS)
    finally:
        if _live is None:
            delattr(E, "_FACE_LIVE_STATE")
        else:
            E._FACE_LIVE_STATE = _live
        os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "0"


def _stream_pass(cfg, consts, args, order, specs, faces, skips, fts, chained):
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    out = []
    for t in range(1, len(order) + 1):
        if not chained:
            E._INCR_STREAM_CACHE.clear()
        stream, seg, _ft, ls = E._incremental_stream_tokens(
            cfg, consts, args, order[:t], specs[:t],
            _tok_rules(cfg, order[:t], specs[:t]),
            ft_by_vertex=fts[t - 1], face_key=_fk(faces, skips, t))
        _check_last_start(out, stream, ls)
        out.append((list(stream), list(seg)))
    return out, dict(E._INCR_STREAM_STATS)


# --------------------------------------------------------------------------- #
def test_face_enum_cache_equals_cold_and_engages():
    ep = _episode()
    T = len(ep[3])
    on, stats_on = _ft_pass(*ep, cached=True)
    off, _ = _ft_pass(*ep, cached=False)
    for t, (a, b) in enumerate(zip(on, off), start=1):
        assert _ft_sig(a) == _ft_sig(b), f"face enum diverged at step {t}"
    # engagement: every step after the first must EXTEND, not rebuild
    assert {k: stats_on[k] for k in ("ext", "cold")} == {
        "ext": T - 1, "cold": 1}, stats_on


def test_face_enum_cache_compress_is_not_special():
    """A COMPRESS in the per-vertex specs costs the cache NOTHING.

    It used to cost one cold replay per COMPRESS decision (and, before the
    refined bound, the rest of the episode).
    """
    T = len(_episode()[3])
    c = T // 2
    ep = _episode(compress_at=c)
    on, stats_on = _ft_pass(*ep, cached=True)
    off, _ = _ft_pass(*ep, cached=False)
    for t, (a, b) in enumerate(zip(on, off), start=1):
        assert _ft_sig(a) == _ft_sig(b), f"face enum diverged at step {t}"
    assert {k: stats_on[k] for k in ("ext", "cold")} == {
        "ext": T - 1, "cold": 1}, stats_on


def test_stream_extension_under_faces_equals_cold_and_engages():
    ep = _episode()
    cfg, consts, args, order, specs, faces, skips = ep
    T = len(order)
    fts, _ = _ft_pass(*ep, cached=False)
    chained, stats_ch = _stream_pass(cfg, consts, args, order, specs, faces,
                                     skips, fts, chained=True)
    cold, _ = _stream_pass(cfg, consts, args, order, specs, faces, skips,
                           fts, chained=False)
    for t, (a, b) in enumerate(zip(chained, cold), start=1):
        assert a[0] == b[0], f"stream diverged at step {t}"
        assert a[1] == b[1], f"seg_ids diverged at step {t}"
    assert stats_ch == {"hit": 0, "ext": T - 1, "cold": 1, "nostore": 0}, (
        stats_ch)


def test_stream_compress_in_specs_extends_like_any_other_rule():
    """A COMPRESS in the per-vertex specs neither breaks the chain nor blocks
    storage: one cold replay for the whole episode, T-1 extensions, and the
    chained streams are byte-identical to the cold ones."""
    T = len(_episode()[3])
    c = T // 2
    ep = _episode(compress_at=c)
    cfg, consts, args, order, specs, faces, skips = ep
    fts, _ = _ft_pass(*ep, cached=False)
    chained, stats_ch = _stream_pass(cfg, consts, args, order, specs, faces,
                                     skips, fts, chained=True)
    cold, _ = _stream_pass(cfg, consts, args, order, specs, faces, skips,
                           fts, chained=False)
    for t, (a, b) in enumerate(zip(chained, cold), start=1):
        assert a[0] == b[0] and a[1] == b[1], f"diverged at step {t}"
    assert stats_ch == {"hit": 0, "ext": T - 1, "cold": 1, "nostore": 0}, (
        stats_ch)


def test_stream_face_row_compress_same_rule():
    ep = _episode(face_compress_at=0)
    cfg, consts, args, order, specs, faces, skips = ep
    T = len(order)
    fts, _ = _ft_pass(*ep, cached=False)
    chained, stats_ch = _stream_pass(cfg, consts, args, order, specs, faces,
                                     skips, fts, chained=True)
    cold, _ = _stream_pass(cfg, consts, args, order, specs, faces, skips,
                           fts, chained=False)
    for t, (a, b) in enumerate(zip(chained, cold), start=1):
        assert a[0] == b[0] and a[1] == b[1], f"diverged at step {t}"
    # a COMPRESS inside a FACE row is equally ordinary now.
    assert stats_ch == {"hit": 0, "ext": T - 1, "cold": 1, "nostore": 0}, (
        stats_ch)


def test_stream_empty_prefix_init_call_does_not_crash():
    """The env INIT call tokenizes a ZERO-length order (smoke B2 regression:
    steps[-1] IndexError). Must return the base stream and store cleanly."""
    cfg, consts, args = _mk()
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    s1, g1, _, ls1 = E._incremental_stream_tokens(
        cfg, consts, args, [], [], {}, ft_by_vertex=None, face_key=None)
    assert len(s1) > 0 and len(s1) == len(g1)
    # Empty prefix: the base IS the block, so the delta starts at 0.
    assert ls1 == 0
    assert E._INCR_STREAM_STATS["cold"] == 1


def test_stream_terminal_compress_extends_and_stores():
    """A COMPRESS on the FINAL vertex used to be the one is_last-sensitive
    state: served cold, never stored. It is ordinary now -- one cold replay
    at t=1 and an extension at every step including the terminal one."""
    T = len(_episode()[3])
    ep = _episode(compress_at=T - 1, seed=4)
    cfg, consts, args, order, specs, faces, skips = ep
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    for t in range(1, T + 1):
        E._incremental_stream_tokens(
            cfg, consts, args, order[:t], specs[:t],
            _tok_rules(cfg, order[:t], specs[:t]),
            ft_by_vertex=None, face_key=None)
    assert E._INCR_STREAM_STATS == {
        "hit": 0, "ext": T - 1, "cold": 1, "nostore": 0}, E._INCR_STREAM_STATS


def test_unified_face_enum_equals_standalone_and_skips_second_replay():
    """P2: face enumeration riding the tokenizer's own IncrementalJaxpr must
    produce the SAME streams and the SAME face structure as the standalone
    `_face_transforms_for_order` replay — with zero second-replay activity
    (the face-enum engagement counters stay at 0)."""
    ep = _episode(seed=5)
    cfg, consts, args, order, specs, faces, skips = ep
    T = len(order)
    fts, _ = _ft_pass(*ep, cached=False)  # standalone (legacy) ft dicts
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    _reset(E._FACE_ENUM_STATS)
    uni_streams, uni_fts = [], []
    for t in range(1, T + 1):
        stream, seg, ft, ls = E._incremental_stream_tokens(
            cfg, consts, args, order[:t], specs[:t],
            _tok_rules(cfg, order[:t], specs[:t]),
            ft_by_vertex=None, face_key=_fk(faces, skips, t),
            face_rows_list=[np.asarray(f).tolist() for f in faces[:t]],
            face_skips_list=[np.asarray(s).tolist() for s in skips[:t]])
        _check_last_start(uni_streams, stream, ls)
        uni_streams.append((list(stream), list(seg)))
        uni_fts.append(ft)
    stats_uni = dict(E._INCR_STREAM_STATS)
    legacy, _ = _stream_pass(cfg, consts, args, order, specs, faces, skips,
                             fts, chained=True)
    for t in range(T):
        assert uni_streams[t] == legacy[t], f"stream diverged at t={t + 1}"
        assert _ft_sig(uni_fts[t]) == _ft_sig(fts[t]), f"ft diverged t={t + 1}"
    assert stats_uni["ext"] == T - 1, stats_uni
    assert {k: E._FACE_ENUM_STATS[k] for k in ("ext", "cold")} == {
        "ext": 0, "cold": 0}, E._FACE_ENUM_STATS
