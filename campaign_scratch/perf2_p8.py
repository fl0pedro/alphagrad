import io
p = "tests/face_live_state_test.py"
s = io.open(p).read()

old = """def _episode(seed, T, compress_at=None):
    \"\"\"One env's wire arrays. Faces are SKIP rows: `_face_dict_for_vertex`
    short-circuits a skipped face BEFORE `decode_vertex_rule_specs`, so a
    synthetic rule can never raise on an operand it does not fit, while the
    enumeration and the replay are exercised identically.\"\"\"
    cfg, consts, args = _mk()
    rng = np.random.default_rng(seed)
    nv = len(cfg.jaxpr.eqns)
    order = [int(x) for x in rng.permutation(np.arange(1, nv + 1))[:T]]"""
new = """def _episode(seed, T, compress_at=None, first=None):
    \"\"\"One env's wire arrays. Faces are SKIP rows: `_face_dict_for_vertex`
    short-circuits a skipped face BEFORE `decode_vertex_rule_specs`, so a
    synthetic rule can never raise on an operand it does not fit, while the
    enumeration and the replay are exercised identically.

    `first` pins the leading vertex. On a graph this small four random orders
    can share a 1-prefix, and two envs that share a prefix legitimately share
    ONE chain -- whichever advances first keeps it and the other rebuilds
    (correctly: the match is exact, so the shared state is the state both
    asked for). That is a real property, but it is not the property these
    tests are about, so the callers that assert `restart == 0` make the
    chains distinct from step 1.\"\"\"
    cfg, consts, args = _mk()
    rng = np.random.default_rng(seed)
    nv = len(cfg.jaxpr.eqns)
    order = [int(x) for x in rng.permutation(np.arange(1, nv + 1))[:T]]
    if first is not None:
        order.remove(int(first))
        order = [int(first)] + order[:T - 1]"""
assert s.count(old) == 1
s = s.replace(old, new)

old2 = """    eps = [_episode(seed=s, T=8) for s in range(n_envs)]
    T = len(eps[0][3])          # the toy graph has fewer than 8 vertices"""
new2 = """    eps = [_episode(seed=s, T=8, first=s + 1) for s in range(n_envs)]
    T = len(eps[0][3])          # the toy graph has fewer than 8 vertices"""
assert s.count(old2) == 1
s = s.replace(old2, new2)

old3 = """    eps = [_episode(seed=s, T=6) for s in range(4)]
    old = E._LIVE_CHAIN_CAP"""
new3 = """    eps = [_episode(seed=s, T=6, first=s + 1) for s in range(4)]
    old = E._LIVE_CHAIN_CAP"""
assert s.count(old3) == 1
s = s.replace(old3, new3)
io.open(p, "w").write(s)
print("ok")
