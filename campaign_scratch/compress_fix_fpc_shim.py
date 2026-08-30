import sys
P = "/Users/assmuth/dsnn/alphagrad/tests/face_prefix_cache_test.py"
A_OLD = """    _live = E._FACE_LIVE_STATE
    E._FACE_LIVE_STATE = False"""
A_NEW = """    # getattr/delattr rather than a bare read: `_FACE_LIVE_STATE` is an
    # IN-FLIGHT symbol (the live-elimination-state rework) that env.py does not
    # define yet, and a bare read makes every test in this file error out on a
    # tree that does not have it. Once it lands this is exactly the same
    # save/restore.
    _live = getattr(E, "_FACE_LIVE_STATE", None)
    E._FACE_LIVE_STATE = False"""
B_OLD = """    finally:
        E._FACE_LIVE_STATE = _live
        os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "0\""""
B_NEW = """    finally:
        if _live is None:
            delattr(E, "_FACE_LIVE_STATE")
        else:
            E._FACE_LIVE_STATE = _live
        os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "0\""""
s = open(P).read()
if A_NEW in s:
    print("already"); sys.exit(0)
assert A_OLD in s and B_OLD in s
open(P, "w").write(s.replace(A_OLD, A_NEW, 1).replace(B_OLD, B_NEW, 1))
print("ok")
