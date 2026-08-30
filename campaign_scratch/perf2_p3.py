import io
p = "tests/face_prefix_cache_test.py"
s = io.open(p).read()
old = '''def _ft_pass(cfg, consts, args, order, specs, faces, skips, cached):
    os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "1" if cached else "0"
    try:
        E._FACE_ENUM_CACHE.clear()'''
new = '''def _ft_pass(cfg, consts, args, order, specs, faces, skips, cached):
    os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "1" if cached else "0"
    # This file is about the BRANCHING-search prefix cache. The default path
    # is now the live elimination state (face_live_state_test.py), which would
    # otherwise serve every call here and leave the cache counters at 0.
    _live = E._FACE_LIVE_STATE
    E._FACE_LIVE_STATE = False
    try:
        E._FACE_ENUM_CACHE.clear()'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = '''        return out, dict(E._FACE_ENUM_STATS)
    finally:
        os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "0"'''
new2 = '''        return out, dict(E._FACE_ENUM_STATS)
    finally:
        E._FACE_LIVE_STATE = _live
        os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "0"'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
io.open(p, "w").write(s)
print("ok")
