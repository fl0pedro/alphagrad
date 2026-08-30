import io

ENV = "src/alphagrad/approx/env.py"


def sub1(s, old, new, tag):
    assert s.count(old) == 1, f"{tag}: {s.count(old)} matches"
    return s.replace(old, new)


src = io.open(ENV).read()

# ------------------------------------------------------------------ state --
NEW_STATE = '''
# ---------------------------------------------------------------------------
# LIVE ELIMINATION STATE
#
# `_face_transforms_for_order` is a PURE FUNCTION of the whole plan prefix, so
# every env step rebuilt an `IncrementalJaxpr` and replayed all k eliminations
# again: O(k) per step, O(T^2) per episode. `_FACE_ENUM_CACHE` tried to buy
# that back with a dict, but in a LINEAR ROLLOUT no prefix is ever visited
# twice, so that dict could never hit AS A CACHE -- the only path that could
# fire was pop-and-extend, i.e. "keep the live object" written as a lookup
# with an LRU cap that could evict the one entry the next step needs. And its
# key was rebuilt from the DENSE wires (MAX_FACES x FACE_SLOTS x 3 ints per
# vertex, for every vertex of the prefix, every step), so the O(T^2) term it
# existed to remove was still being paid -- in the lookup.
#
# This OWNS the state instead. A small pool of live chains, one per concurrent
# env, each holding its `IncrementalJaxpr` plus the exact prefix it has
# consumed; a step whose prefix extends a chain's ADVANCES that chain by the
# one new elimination. Work per step is constant, so O(T) per episode is
# STRUCTURAL, not cached: a mid-episode rebuild is a counted anomaly, never a
# silent return to O(T^2).
#
# THE COMPRESS CARVE-OUT IS GONE, and this is the reason the design is shaped
# this way. `decode_vertex_rule_specs` emits COMPRESS only at `is_last=True`,
# so the CURRENT LAST vertex is the only one whose decode is is_last-sensitive
# -- which is why the old cache had to refuse to store a COMPRESS-last state
# (v40 measured ext=1/431: essentially dead whenever COMPRESS was live). A
# chain therefore never eliminates the last vertex at all: it ENUMERATES that
# vertex's faces on the live graph and stops. The vertex is committed one step
# later, when it is no longer last and its `is_last=False` decode is final.
# Every elimination a chain performs is permanent, so the state is append-only
# by construction and there is nothing to speculate or roll back. (The one
# place append-only still breaks is the last vertex ITSELF, which is exactly
# the non-monotonicity the separate COMPRESS workstream is looking at; nothing
# here depends on it being preserved.)
#
# Identity is EXACT, not hashed: a chain is matched by byte-equality of the
# order and spec prefixes plus tuple-equality of `_face_wire_keys` -- the same
# sparse injective wire encoding `_INCR_STREAM_CACHE` keys on, computed ONCE
# per callback and shared, so verification is free rather than a second
# O(T x MAX_FACES) pass.
#
# ALPHAGRAD_FACE_LIVE_STATE=0 restores the stateless rebuild (and with it
# ALPHAGRAD_FACE_ENUM_CACHE, which is kept for BRANCHING search -- GAZ/MCTS
# really do revisit prefixes, and there a cache is a cache).
# ---------------------------------------------------------------------------
_FACE_LIVE_STATE = os.environ.get("ALPHAGRAD_FACE_LIVE_STATE", "1") != "0"
_LIVE_CHAIN_CAP = int(os.environ.get("ALPHAGRAD_FACE_LIVE_CHAINS", "8"))
_LIVE_CHAINS: list = []
# `elims` is the asymptotic quantity: T + 1 per env per episode means O(T).
# `restart` counts mid-episode cold rebuilds -- the failure mode this design
# exists to make impossible, so a nonzero value is a bug report, not noise.
_LIVE_CHAIN_STATS = {"step": 0, "elims": 0, "cold": 0, "restart": 0,
                     "evict": 0, "poison": 0}


class _ElimChain:
    """One env's live elimination state.

    ``ij`` has consumed ``n`` vertices, each in its PERMANENT ``is_last=False``
    form, and ``out`` holds their face-transform dicts. The current last
    vertex is deliberately absent -- see the module comment above.
    """

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
                          face_rows_list, face_skips_list,
                          honor_last_compress, wire_sig):
    """`_face_transforms_for_order` served from live state. Same result."""
    from graphax.incremental import IncrementalJaxpr
    from alphagrad.approx.common.masks import make_live_masked_hook

    K = len(o_list)
    n = K - 1                     # vertices that must already be eliminated
    base = (id(config.jaxpr), tuple(config.argnums))
    if wire_sig is None:
        wire_sig = _face_wire_keys(np.asarray(face_rows_list),
                                   np.asarray(face_skips_list), K)
    _ord = np.ascontiguousarray(np.asarray(o_list, dtype=np.int64))
    _sp = np.ascontiguousarray(np.asarray(specs_list, dtype=np.int64))
    okey, skey = _ord.tobytes(), _sp.tobytes()
    _ob, _sb = _ord.itemsize, int(_sp[0].nbytes)

    # LONGEST chain whose consumed prefix is a prefix of this request. Fixed
    # per-step stride makes a BYTE prefix exactly an ARRAY prefix.
    ch = None
    for c in _LIVE_CHAINS:
        if (c.base == base and c.n <= n
                and okey.startswith(c.okey) and skey.startswith(c.skey)
                and c.fsig == wire_sig[:c.n]
                and (ch is None or c.n > ch.n)):
            ch = c
    _LIVE_CHAIN_STATS["step"] += 1
    if ch is None:
        _LIVE_CHAIN_STATS["cold"] += 1
        if n > 0:
            _LIVE_CHAIN_STATS["restart"] += 1
        ch = _ElimChain(base, IncrementalJaxpr(
            config.jaxpr, tuple(config.argnums), list(consts), list(args),
            track_faces=False))
        _LIVE_CHAINS.append(ch)
        while len(_LIVE_CHAINS) > _LIVE_CHAIN_CAP:
            _LIVE_CHAINS.pop(0)
            _LIVE_CHAIN_STATS["evict"] += 1
    else:
        _LIVE_CHAINS.remove(ch)          # LRU: most recent at the end
        _LIVE_CHAINS.append(ch)

    # ADVANCE. Each vertex committed here is no longer the last one, so its
    # decode is final and the elimination never has to be undone.
    try:
        while ch.n < n:
            k = ch.n
            v = int(o_list[k])
            per_face = _face_dict_for_vertex(
                config, ch.ij, v, face_rows_list[k], face_skips_list[k],
                is_last_honored=False)
            if per_face:
                ch.out[v] = per_face
            vrules = decode_vertex_rule_specs(
                config.jaxpr, v, specs_list[k], is_last=False)
            ch.ij.eliminate(
                v,
                (make_live_masked_hook(tuple(vrules)),) if vrules else (),
                ch.out.get(v))
            ch.n += 1
            ch.okey = okey[:ch.n * _ob]
            ch.skey = skey[:ch.n * _sb]
            ch.fsig = wire_sig[:ch.n]
            _LIVE_CHAIN_STATS["elims"] += 1
        # The LAST vertex is enumerated on the live graph and NOT eliminated.
        out = dict(ch.out)
        v = int(o_list[n])
        per_face = _face_dict_for_vertex(
            config, ch.ij, v, face_rows_list[n], face_skips_list[n],
            is_last_honored=honor_last_compress)
    except BaseException:
        # A half-applied elimination leaves the trace inconsistent; drop the
        # chain so the next step rebuilds instead of extending damage.
        if ch in _LIVE_CHAINS:
            _LIVE_CHAINS.remove(ch)
        _LIVE_CHAIN_STATS["poison"] += 1
        raise
    if per_face:
        out[v] = per_face
    return out


'''

src = sub1(src,
    "def _face_transforms_for_order(config, consts, args, o_list, specs_list,",
    NEW_STATE.lstrip("\n")
    + "def _face_transforms_for_order(config, consts, args, o_list, specs_list,",
    "insert-live-state")

# ---------------------------------------------------------- entry + dispatch
src = sub1(src,
    "                               face_rows_list, face_skips_list,\n"
    "                               honor_last_compress=True):",
    "                               face_rows_list, face_skips_list,\n"
    "                               honor_last_compress=True, wire_sig=None):",
    "signature")

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
    "            face_skips_list, honor_last_compress, wire_sig)\n"
    "\n"
    "    ij = None\n",
    "dispatch")

# ------------------------------------------------- caller: numpy + one sig --
src = sub1(src,
    "    if _have_face_actions and not _unified_fe:\n"
    "        _fr_list = _faces_np.tolist()\n"
    "        _fs_list = _skips_np.tolist()\n"
    "        _pf(\"cb.face_tolist\")\n"
    "        ft_by_vertex = _face_transforms_for_order(\n"
    "            config, consts, args, o_list, specs_list,\n"
    "            _fr_list, _fs_list,\n"
    "            honor_last_compress=_honor_mid_compress,\n"
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
    "        # per callback is O(T x MAX_FACES) per step and O(T^2 x MAX_FACES)\n"
    "        # per episode, >99% of it -1 padding (measured live occupancy is\n"
    "        # ~1.24 faces per vertex). Same argument, same fix as the\n"
    "        # `_fe_inline` wires below already use.\n"
    "        ft_by_vertex = _face_transforms_for_order(\n"
    "            config, consts, args, o_list, specs_list,\n"
    "            _faces_np, _skips_np,\n"
    "            honor_last_compress=_honor_mid_compress,\n"
    "            wire_sig=_wire_sig,\n"
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
print("perf2_live: OK")
