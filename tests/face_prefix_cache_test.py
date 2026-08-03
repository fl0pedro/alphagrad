"""Prefix caches must be INVISIBLE: an extended replay must equal a cold one.

Covers the two O(T^2) -> O(T) host-overhead fixes:

  * `_face_transforms_for_order` pop-and-extend cache
    (ALPHAGRAD_FACE_ENUM_CACHE=1): the k-th step advances the cached
    IncrementalJaxpr by ONE elimination instead of rebuilding and replaying
    the whole prefix; and
  * `_incremental_stream_tokens` ancestor extension under FACE actions
    (per-step `face_key` signatures instead of one whole-prefix blob).

Soundness bound (measured in stream_prefix_property_test.py): only the
CURRENT LAST vertex's decode is is_last-sensitive — COMPRESS is emitted only
when `is_last=True`. The rule both caches implement: a COMPRESS-last state is
SERVED (cold, without consuming its parent) but never STORED, so every stored
state is is_last-insensitive by induction and extension from any stored
parent is sound. One COMPRESS decision costs one cold replay, not a dead
cache for the rest of the episode (the v40 failure mode: ext=1/431).

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
    last = len(o_list) - 1
    for i, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(
            cfg.jaxpr, int(v), specs_list[i], is_last=(i == last))
        if rules:
            d[int(v)] = (make_live_masked_hook(tuple(rules)),)
    return d


def _fk(faces, skips, t):
    return tuple(
        (tuple(int(x) for x in np.asarray(faces[k]).reshape(-1)),
         tuple(int(x) for x in np.asarray(skips[k]).reshape(-1)))
        for k in range(t))


def _reset(d):
    d.update({k: 0 for k in d})


def _ft_pass(cfg, consts, args, order, specs, faces, skips, cached):
    os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "1" if cached else "0"
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
        os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "0"


def _stream_pass(cfg, consts, args, order, specs, faces, skips, fts, chained):
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    out = []
    for t in range(1, len(order) + 1):
        if not chained:
            E._INCR_STREAM_CACHE.clear()
        stream, seg = E._incremental_stream_tokens(
            cfg, consts, args, order[:t], specs[:t],
            _tok_rules(cfg, order[:t], specs[:t]),
            ft_by_vertex=fts[t - 1], face_key=_fk(faces, skips, t))
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
    assert stats_on == {"ext": T - 1, "cold": 1}, stats_on


def test_face_enum_cache_compress_last_served_cold_chain_survives():
    T = len(_episode()[3])
    c = T // 2
    ep = _episode(compress_at=c)
    on, stats_on = _ft_pass(*ep, cached=True)
    off, _ = _ft_pass(*ep, cached=False)
    for t, (a, b) in enumerate(zip(on, off), start=1):
        assert _ft_sig(a) == _ft_sig(b), f"face enum diverged at step {t}"
    # step c+1 (COMPRESS-last) is served cold with NO cache interaction; its
    # parent survives, so step c+2 extends from cut=c across two vertices —
    # the chain loses exactly one extension, not the rest of the episode.
    assert stats_on == {"ext": T - 2, "cold": 1}, stats_on


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


def test_stream_compress_last_served_cold_not_stored_chain_survives():
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
    # t=1 cold(store); t=c+1 COMPRESS-last: cold + nostore, parent kept;
    # t=c+2 extends from cut=c (re-eliminating vertex c with is_last=False,
    # which drops the COMPRESS — the exact divergence the rule guards).
    assert stats_ch == {"hit": 0, "ext": T - 2, "cold": 2, "nostore": 1}, (
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
    # a COMPRESS inside a FACE row has the same is_last sensitivity: t=1 is
    # cold+nostore, t=2 cold (nothing stored yet), t>=3 extend.
    assert stats_ch == {"hit": 0, "ext": T - 2, "cold": 2, "nostore": 1}, (
        stats_ch)


def test_stream_empty_prefix_init_call_does_not_crash():
    """The env INIT call tokenizes a ZERO-length order (smoke B2 regression:
    steps[-1] IndexError). Must return the base stream and store cleanly."""
    cfg, consts, args = _mk()
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    s1, g1 = E._incremental_stream_tokens(cfg, consts, args, [], [], {},
                                          ft_by_vertex=None, face_key=None)
    assert len(s1) > 0 and len(s1) == len(g1)
    assert E._INCR_STREAM_STATS["cold"] == 1


def _tok_rules_mid(cfg, o_list, specs_list, honor):
    """Tokenizer rules as the callback builds them under
    ALPHAGRAD_TOKENS_MID_COMPRESS=0: the intermediate last vertex decodes
    with is_last=False; `honor` is True only on the terminal step."""
    d = {}
    last = len(o_list) - 1
    for i, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(
            cfg.jaxpr, int(v), specs_list[i],
            is_last=(i == last and honor))
        if rules:
            d[int(v)] = (make_live_masked_hook(tuple(rules)),)
    return d


def test_stream_mid_compress_off_extends_through_compress_and_terminal():
    """Flag=0 world: every intermediate COMPRESS-last state is storable, the
    chain never dies, and the TERMINAL call (honor=True) soundly extends the
    honor=False chain - its final elimination is the only difference."""
    T = len(_episode()[3])
    ep = _episode(compress_at=T // 2, seed=3)
    cfg, consts, args, order, specs, faces, skips = ep
    fts = [
        E._face_transforms_for_order(
            cfg, consts, args, order[:t], specs[:t],
            [np.asarray(f).tolist() for f in faces[:t]],
            [np.asarray(s).tolist() for s in skips[:t]],
            honor_last_compress=(t == T))
        for t in range(1, T + 1)
    ]

    def _pass(chained):
        E._INCR_STREAM_CACHE.clear()
        _reset(E._INCR_STREAM_STATS)
        out = []
        for t in range(1, T + 1):
            if not chained:
                E._INCR_STREAM_CACHE.clear()
            honor = (t == T)
            stream, seg = E._incremental_stream_tokens(
                cfg, consts, args, order[:t], specs[:t],
                _tok_rules_mid(cfg, order[:t], specs[:t], honor),
                ft_by_vertex=fts[t - 1], face_key=_fk(faces, skips, t),
                honor_last_compress=honor)
            out.append((list(stream), list(seg)))
        return out, dict(E._INCR_STREAM_STATS)

    chained, stats_ch = _pass(chained=True)
    cold, _ = _pass(chained=False)
    for t, (a, b) in enumerate(zip(chained, cold), start=1):
        assert a[0] == b[0] and a[1] == b[1], f"diverged at step {t}"
    # every step after the first extends - the COMPRESS-at-mid step included
    assert stats_ch["ext"] == T - 1 and stats_ch["cold"] == 1, stats_ch


def test_stream_mid_compress_off_terminal_compress_not_stored():
    """Terminal step with a COMPRESS-carrying FINAL vertex: is_last-sensitive
    again (honor=True), so it takes the one honest cold replay of the
    episode, leaves its parent untouched, and is never stored."""
    T = len(_episode()[3])
    ep = _episode(compress_at=T - 1, seed=4)
    cfg, consts, args, order, specs, faces, skips = ep
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    for t in range(1, T + 1):
        honor = (t == T)
        E._incremental_stream_tokens(
            cfg, consts, args, order[:t], specs[:t],
            _tok_rules_mid(cfg, order[:t], specs[:t], honor),
            ft_by_vertex=None, face_key=None, honor_last_compress=honor)
    # t=1 cold, t=2..T-1 extend, t=T cold + nostore (parent preserved)
    assert E._INCR_STREAM_STATS == {
        "hit": 0, "ext": T - 2, "cold": 2, "nostore": 1}, E._INCR_STREAM_STATS
