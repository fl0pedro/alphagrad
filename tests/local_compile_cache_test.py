"""The process-local memo in `cached_compile` must (a) return the SAME
object on a repeated key without re-invoking compile_fn, (b) evict
LEAST-RECENTLY-USED at the cap -- a hit refreshes an entry's position
(owner ruling 2026-09-14, small fixes #2), so a plain sequential fill with
no interleaved hits still evicts oldest-first, but a key touched by hits
survives past keys it FIFO-outlived before, (c) stay disabled at cap 0 --
all WITHOUT a Ray coordinator (the ALPHAGRAD_DIRECT_MEASURE single-process
mode, where the function used to be a pass-through that recompiled every
repeated plan).

Before the ruling, a hit never moved its entry: the table was FIFO by
FIRST WRITE, so the rev-exact reference (looked up once per plan under key
``b"paired-ref:" + paired_ref_key``, env.py ~7614 -- hit far more often
than any single approx-plan key, but never refreshed) was evicted under
load and recompiled exactly like a key touched only once.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

import alphagrad.approx.common.compile_cache as cc


def _fresh(cap):
    cc._LOCAL_CACHE.clear()
    cc._LOCAL_CACHE_CAP = cap


def test_repeat_key_skips_compile_fn():
    _fresh(cap=8)
    calls = []

    def fn():
        calls.append(1)
        return object()

    a = cc.cached_compile(b"k1", fn)
    b = cc.cached_compile(b"k1", fn)
    assert a is b, "second call must serve the memoized executable"
    assert len(calls) == 1, "compile_fn must run exactly once"


def test_lru_eviction_at_cap_with_no_interleaved_hits():
    """With nothing but sequential first-writes (no hits in between), LRU
    and FIFO agree: the oldest entry is the least-recently-used one."""
    _fresh(cap=2)
    objs = {k: cc.cached_compile(k, lambda k=k: ("exe", k))
            for k in (b"a", b"b", b"c")}
    assert b"a" not in cc._LOCAL_CACHE, "oldest entry must be evicted"
    assert set(cc._LOCAL_CACHE) == {b"b", b"c"}
    # evicted key recompiles (fresh object), retained key serves the memo
    fresh = cc.cached_compile(b"a", lambda: ("exe2", b"a"))
    assert fresh == ("exe2", b"a")
    assert cc.cached_compile(b"c", lambda: ("never", 0)) is objs[b"c"]


def test_a_hit_refreshes_lru_order_so_the_touched_key_survives():
    """33 distinct compiles at the real cap (32), interleaved with a HIT on
    the first key after every later compile: the first key must stay
    resident throughout, because each hit refreshes its position -- under
    the old FIFO behaviour it would have been evicted by the 33rd distinct
    key regardless of how often it was touched."""
    _fresh(cap=32)
    keys = [f"k{i}".encode() for i in range(33)]
    first = cc.cached_compile(keys[0], lambda: ("exe", keys[0]))

    def _must_not_recompile():
        pytest.fail("k0 was evicted -- a hit must not re-invoke compile_fn")

    for k in keys[1:]:
        cc.cached_compile(k, lambda k=k: ("exe", k))
        # touch k0 -- a hit, not a recompile, and it must move k0 to the
        # most-recently-used end so the NEXT distinct compile evicts
        # someone else instead.
        touched = cc.cached_compile(keys[0], _must_not_recompile)
        assert touched is first, "a hit must return the memoized object"

    assert keys[0] in cc._LOCAL_CACHE, "the repeatedly-touched key must survive"
    assert len(cc._LOCAL_CACHE) == 32, "capacity must be unchanged"


def test_cap_zero_disables():
    _fresh(cap=0)
    calls = []

    def fn():
        calls.append(1)
        return object()

    a = cc.cached_compile(b"k", fn)
    b = cc.cached_compile(b"k", fn)
    assert a is not b and len(calls) == 2
    assert not cc._LOCAL_CACHE
