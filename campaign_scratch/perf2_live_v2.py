import io

ENV = "src/alphagrad/approx/env.py"
PPO = "src/alphagrad/approx/ppo.py"


def sub1(s, old, new, tag):
    assert s.count(old) == 1, f"{tag}: {s.count(old)} matches"
    return s.replace(old, new)


src = io.open(ENV).read()

NEW_STATE = '''# ---------------------------------------------------------------------------
# LIVE ELIMINATION STATE
#
# `_face_transforms_for_order` is a PURE FUNCTION of the whole plan prefix, so
# every env step rebuilt an `IncrementalJaxpr` and replayed all k eliminations
# again: O(k) per step, O(T^2) per episode. `_FACE_ENUM_CACHE` tried to buy
# that back with a dict, but in a LINEAR ROLLOUT no prefix is ever visited
# twice, so that dict could never hit AS A CACHE -- the only path that could
# fire was pop-and-extend, i.e. "keep the live object", written as a lookup
# with an LRU cap that can evict the one entry the next step needs. And the
# lookup rebuilt its key from the DENSE wires (MAX_FACES x FACE_SLOTS x 3 ints
# per vertex, for every vertex of the prefix, every step -- 45936 ints per
# vertex at the 3-block TLM's derived bound of 5104), so the O(T^2) term the
# cache existed to remove was still being paid, inside the cache.
#
# This OWNS the state instead. A small pool of live chains -- one per
# concurrent env, since `pure_callback(vmap_method="sequential")` walks the
# batch one host call at a time -- each holding its `IncrementalJaxpr`, the
# face-transform dicts it has produced, and the exact prefix it has consumed.
# A step whose prefix extends a chain's ADVANCES that chain by the one new
# elimination. Work per step is constant, so O(T) per episode is STRUCTURAL:
# a mid-episode rebuild is a counted anomaly (`restart`), never a silent
# return to O(T^2).
#
# This is only possible because a vertex's decode no longer depends on where
# the prefix ends (COMPRESS is honored at every position -- the `is_last` gate
# is gone). Under that gate the last vertex of a prefix was decoded
# differently from the same vertex one step later, so a live builder would
# have had to speculate and roll back; that is what the deleted COMPRESS
# carve-outs were, and why v40 measured the cache essentially dead (ext=1/431)
# whenever COMPRESS was in the action space. The elimination stream is now
# append-only, and so is this.
#
# Identity is EXACT, never hashed: a chain matches on byte-equality of the
# order and spec prefixes plus tuple-equality of `_face_wire_keys` -- the
# sparse injective wire encoding `_INCR_STREAM_CACHE` already keys on,
# computed ONCE per callback and shared with it, so the check is free rather
# than a second O(T x MAX_FACES) pass.
#
# ALPHAGRAD_FACE_LIVE_STATE=0 restores the stateless rebuild, and with it
# ALPHAGRAD_FACE_ENUM_CACHE -- which is kept for BRANCHING search (GAZ/MCTS
# really do revisit prefixes, and there a cache is a cache).
# ---------------------------------------------------------------------------
_FACE_LIVE_STATE = os.environ.get("ALPHAGRAD_FACE_LIVE_STATE", "1") != "0"
_LIVE_CHAIN_CAP = int(os.environ.get("ALPHAGRAD_FACE_LIVE_CHAINS", "8"))
_LIVE_CHAINS: list = []
# `elims` is the asymptotic quantity: T per env per episode is O(T),
# T(T+1)/2 per env is O(T^2). `restart` counts mid-episode cold rebuilds --
# the failure mode this design exists to make impossible, so a nonzero value
# is a bug report, not noise.
_LIVE_CHAIN_STATS = {"step": 0, "elims": 0, "cold": 0, "restart": 0,
                     "evict": 0, "poison": 0}


class _ElimChain:
    """One env's live elimination state: `ij` has consumed `n` vertices and
    `out` holds their `{face_key: slots|SKIP_FACE}` dicts."""

    __slots__ = ("base", "ij", "out", "n", "okey", "skey", "fsig")

    def __init__(self, base, ij):
        self.base = base
        self.ij = ij
        self.out: dict[int, dict] = {}
        self.n = 0
        self.okey = b""
        self.skey = b""
        self.fsig: tuple = ()


def consume_live_chain_stats() -> dict:
    out = dict(_LIVE_CHAIN_STATS)
    for k in _LIVE_CHAIN_STATS:
        _LIVE_CHAIN_STATS[k] = 0
    return out


def _live_face_transforms(config, consts, args, o_list, specs_list,
                          face_rows_list, face_skips_list, wire_sig):
    """`_face_transforms_for_order` served from live state. Same result."""
    from graphax.incremental import IncrementalJaxpr
    from alphagrad.approx.common.masks import make_live_masked_hook

    K = len(o_list)
    base = (id(config.jaxpr), tuple(config.argnums))
    if wire_sig is None:
        wire_sig = _face_wire_keys(np.asarray(face_rows_list),
                                   np.asarray(face_skips_list), K)
    _ord = np.ascontiguousarray(np.asarray(o_list, dtype=np.int64))
    _sp = np.ascontiguousarray(np.asarray(specs_list, dtype=np.int64))
    okey, skey = _ord.tobytes(), _sp.tobytes()
    _ob, _sb = _ord.itemsize, int(_sp[0].nbytes)

    # LONGEST chain whose consumed prefix is a prefix of this request. A fixed
    # per-step stride makes a BYTE prefix exactly an ARRAY prefix.
    ch = None
    for c in _LIVE_CHAINS:
        if (c.base == base and c.n <= K
                and okey.startswith(c.okey) and skey.startswith(c.skey)
                and c.fsig == wire_sig[:c.n]
                and (ch is None or c.n > ch.n)):
            ch = c
    _LIVE_CHAIN_STATS["step"] += 1
    if ch is None:
        _LIVE_CHAIN_STATS["cold"] += 1
        if K > 1:
            _LIVE_CHAIN_STATS["restart"] += 1
        ch = _ElimChain(base, IncrementalJaxpr(
            config.jaxpr, tuple(config.argnums), list(consts), list(args),
            track_faces=False))
        _LIVE_CHAINS.append(ch)
        while len(_LIVE_CHAINS) > _LIVE_CHAIN_CAP:
            _LIVE_CHAINS.pop(0)
            _LIVE_CHAIN_STATS["evict"] += 1
    else:
        _LIVE_CHAINS.remove(ch)          # LRU: most recently used last
        _LIVE_CHAINS.append(ch)

    try:
        while ch.n < K:
            k = ch.n
            v = int(o_list[k])
            per_face = _face_dict_for_vertex(
                config, ch.ij, v, face_rows_list[k], face_skips_list[k])
            if per_face:
                ch.out[v] = per_face
            vertex_rules = decode_vertex_rule_specs(
                config.jaxpr, v, specs_list[k])
            ch.ij.eliminate(
                v,
                (make_live_masked_hook(tuple(vertex_rules)),)
                if vertex_rules else (),
                ch.out.get(v))
            ch.n += 1
            ch.okey = okey[:ch.n * _ob]
            ch.skey = skey[:ch.n * _sb]
            ch.fsig = wire_sig[:ch.n]
            _LIVE_CHAIN_STATS["elims"] += 1
    except BaseException:
        # A half-applied elimination leaves the trace inconsistent; drop the
        # chain so the next step rebuilds instead of extending the damage.
        if ch in _LIVE_CHAINS:
            _LIVE_CHAINS.remove(ch)
        _LIVE_CHAIN_STATS["poison"] += 1
        raise
    # the chain keeps growing -- hand back a snapshot, like the cache does
    return dict(ch.out)


'''

src = sub1(src,
    "def _face_transforms_for_order(config, consts, args, o_list, specs_list,\n"
    "                               face_rows_list, face_skips_list):",
    NEW_STATE
    + "def _face_transforms_for_order(config, consts, args, o_list, specs_list,\n"
      "                               face_rows_list, face_skips_list,\n"
      "                               wire_sig=None):",
    "insert+signature")

src = sub1(src,
    "    from graphax import SKIP_FACE, faces_of\n"
    "    from graphax.incremental import IncrementalJaxpr\n"
    "    from alphagrad.approx.common.masks import make_live_masked_hook\n"
    "\n"
    "    ij = None\n",
    "    from graphax import SKIP_FACE, faces_of\n"
    "    from graphax.incremental import IncrementalJaxpr\n"
    "    from alphagrad.approx.common.masks import make_live_masked_hook\n"
    "\n"
    "    if _FACE_LIVE_STATE and len(o_list):\n"
    "        return _live_face_transforms(\n"
    "            config, consts, args, o_list, specs_list, face_rows_list,\n"
    "            face_skips_list, wire_sig)\n"
    "\n"
    "    ij = None\n",
    "dispatch")

src = sub1(src,
    "    if _have_face_actions and not _unified_fe:\n"
    "        _fr_list = _faces_np.tolist()\n"
    "        _fs_list = _skips_np.tolist()\n"
    "        _pf(\"cb.face_tolist\")\n"
    "        ft_by_vertex = _face_transforms_for_order(\n"
    "            config, consts, args, o_list, specs_list,\n"
    "            _fr_list, _fs_list,\n"
    "        )\n"
    "    _pf(\"cb.face_enum\")",
    "    # ONE sparse per-vertex encoding of the face wires per callback,\n"
    "    # shared by the live elimination state below and by the stream\n"
    "    # cache's `face_key` further down -- they used to build one each.\n"
    "    _wire_sig = None\n"
    "    if _have_face_actions:\n"
    "        _wire_sig = _face_wire_keys(_faces_np, _skips_np, len(o_list))\n"
    "    _pf(\"cb.face_wire_sig\")\n"
    "    if _have_face_actions and not _unified_fe:\n"
    "        # NUMPY, not `.tolist()`: `_face_dict_for_vertex` pulls the three\n"
    "        # wire ints out of `face_row[f][s]` explicitly and reads\n"
    "        # `face_skip[f]` scalar-wise, so it never needed Python lists --\n"
    "        # and materialising T x MAX_FACES x FACE_SLOTS x 3 Python ints\n"
    "        # per callback is O(T x MAX_FACES) per step and\n"
    "        # O(T^2 x MAX_FACES) per episode, >99% of it -1 padding (measured\n"
    "        # live occupancy is ~1.24 faces per vertex). Same argument, same\n"
    "        # fix as the `_fe_inline` wires below already use.\n"
    "        ft_by_vertex = _face_transforms_for_order(\n"
    "            config, consts, args, o_list, specs_list,\n"
    "            _faces_np, _skips_np, wire_sig=_wire_sig,\n"
    "        )\n"
    "    _pf(\"cb.face_enum\")",
    "caller")

src = sub1(src,
    "            face_key=_face_wire_keys(_faces_np, _skips_np, len(o_list))\n"
    "            if (ft_by_vertex is not None or _fe_inline) else None,",
    "            face_key=_wire_sig\n"
    "            if (ft_by_vertex is not None or _fe_inline) else None,",
    "reuse-sig")

io.open(ENV, "w").write(src)

p = io.open(PPO).read()
old = ("                            f\"  face_enum(calls/elims/build)=\"\n"
       "                            f\"{_fs['calls']}/{_fs['elims']}/\"\n"
       "                            f\"{_fs['build']}\")\n")
new = ("                            f\"  face_enum(calls/elims/build)=\"\n"
       "                            f\"{_fs['calls']}/{_fs['elims']}/\"\n"
       "                            f\"{_fs['build']}\"\n"
       "                            f\"  live_chain=\"\n"
       "                            f\"{_envmod.consume_live_chain_stats()}\")\n")
assert p.count(old) == 1, p.count(old)
io.open(PPO, "w").write(p.replace(old, new))
print("perf2_live_v2: OK")
