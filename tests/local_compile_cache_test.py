"""The process-local memo in `cached_compile` must (a) return the SAME
object on a repeated key without re-invoking compile_fn, (b) evict FIFO at
the cap, (c) stay disabled at cap 0 — all WITHOUT a Ray coordinator (the
ALPHAGRAD_DIRECT_MEASURE single-process mode, where the function used to be
a pass-through that recompiled every repeated plan)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

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


def test_fifo_eviction_at_cap():
    _fresh(cap=2)
    objs = {k: cc.cached_compile(k, lambda k=k: ("exe", k))
            for k in (b"a", b"b", b"c")}
    assert b"a" not in cc._LOCAL_CACHE, "oldest entry must be evicted"
    assert set(cc._LOCAL_CACHE) == {b"b", b"c"}
    # evicted key recompiles (fresh object), retained key serves the memo
    fresh = cc.cached_compile(b"a", lambda: ("exe2", b"a"))
    assert fresh == ("exe2", b"a")
    assert cc.cached_compile(b"c", lambda: ("never", 0)) is objs[b"c"]


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
