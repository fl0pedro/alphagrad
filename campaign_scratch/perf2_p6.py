import io
p = "tests/face_live_state_test.py"
s = io.open(p).read()
reps = [
("""    T = 8
    eps = [_episode(seed=s, T=T) for s in range(n_envs)]
    live, stats = _pass(eps, live=True)""",
 """    eps = [_episode(seed=s, T=8) for s in range(n_envs)]
    T = len(eps[0][3])          # the toy graph has fewer than 8 vertices
    live, stats = _pass(eps, live=True)"""),
("""    T = 8
    plain = [_episode(seed=0, T=T)]
    comp = [_episode(seed=0, T=T, compress_at=T // 2)]""",
 """    plain = [_episode(seed=0, T=8)]
    T = len(plain[0][3])
    comp = [_episode(seed=0, T=8, compress_at=T // 2)]"""),
("""    T = 6
    eps = [_episode(seed=s, T=T) for s in range(4)]
    old = E._LIVE_CHAIN_CAP""",
 """    eps = [_episode(seed=s, T=6) for s in range(4)]
    old = E._LIVE_CHAIN_CAP"""),
("""    T = 6
    ep = _episode(seed=3, T=T)
    prev = E._FACE_LIVE_STATE""",
 """    ep = _episode(seed=3, T=6)
    T = len(ep[3])
    prev = E._FACE_LIVE_STATE"""),
]
for a, b in reps:
    assert s.count(a) == 1, a[:40]
    s = s.replace(a, b)
io.open(p, "w").write(s)
print("ok")
