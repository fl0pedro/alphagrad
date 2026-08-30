import io

p = "src/alphagrad/approx/ppo.py"
s = io.open(p).read()
old = """    _ORACLE_ONE_VERTEX = os.environ.get(
        "ALPHAGRAD_ORACLE_ONE_VERTEX", "1") == "1\""""
new = """    # Live elimination chains: one per concurrent env, plus the previous
    # episode's, which the LRU only sheds once the new ones exist. Sized like
    # the face-prefix cache above and for the same reason -- a capacity below
    # the concurrent chain count turns every step into a cold O(T) rebuild,
    # i.e. straight back to O(T^2) per episode. That is not silent (the
    # `restart` counter reports it in the [prof] line), but it should not be
    # possible by default. ALPHAGRAD_FACE_LIVE_CHAINS still overrides.
    if not os.environ.get("ALPHAGRAD_FACE_LIVE_CHAINS"):
        from alphagrad.approx import env as _env_chain_mod
        _env_chain_mod._LIVE_CHAIN_CAP = max(
            8, 4 * _resolve_num_envs(args.num_envs, args.example))

    _ORACLE_ONE_VERTEX = os.environ.get(
        "ALPHAGRAD_ORACLE_ONE_VERTEX", "1") == "1\""""
assert s.count(old) == 1, s.count(old)
io.open(p, "w").write(s.replace(old, new))
print("ok")
