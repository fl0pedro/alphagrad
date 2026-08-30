"""Remove the COMPRESS `is_last` gate: COMPRESS becomes representable at EVERY
position of a plan, so the token stream is genuinely append-only and the
`is_last` / `honor_last_compress` / `is_last_honored` machinery and the two
prefix-cache COMPRESS carve-outs disappear with it.

Deterministic exact-string patcher (the tree is edited over ssh; sed/awk-level
precision on purpose). Idempotent: re-running is a no-op once applied.
"""
import sys

ROOT = "/Users/assmuth/dsnn/alphagrad/"

EDITS: list[tuple[str, str, str]] = []


def E(path, old, new):
    EDITS.append((path, old, new))


# ---------------------------------------------------------------- env.py ----
ENV = "src/alphagrad/approx/env.py"

E(ENV,
  "def decode_vertex_rule_specs(jaxpr, vertex, spec_rows, is_last: bool) -> tuple:",
  "def decode_vertex_rule_specs(jaxpr, vertex, spec_rows) -> tuple:")

E(ENV, """      * ``row[0] == COMPRESS_SENTINEL`` COMPRESS, physical axis ``row[1]``,
        kind ``row[2]`` — honored only when ``is_last`` (COMPRESS reduces
        ``val.ndim``, which trips graphax's shape-preservation assertion when
        the compressed edge feeds a later elimination).""",
  """      * ``row[0] == COMPRESS_SENTINEL`` COMPRESS, physical axis ``row[1]``,
        kind ``row[2]`` — honored at EVERY position of the plan.

        There used to be an ``is_last`` gate here that emitted COMPRESS only
        for the NEWEST vertex of the prefix, on the stated grounds that a
        ``val.ndim`` reduction upstream of a later elimination trips graphax's
        shape-preservation assertion. It is gone, because the premise is false
        and the cost was severe:

        * graphax's nominal-shape asserts are EXACT-AD only — core.py gates
          them on ``not _perpath and not _is_approx_cfg and not approx_active()``
          — and ``apply_compress`` drops the axis POINTER (``Index.axis =
          None``), not the logical size, so a compressed edge still contracts.
          The over-conservative structural guard the note referred to was
          removed from ``apply_compress`` on 2026-07-15.
        * MEASURED (2026-08-15, ``compress_probe``/``compress_probe2``, CPU):
          the same decoded COMPRESS applied at all 23 positions of the
          NeuralNetwork plan and at 11 sampled positions of the TransformerLM
          plan, RAW and ``make_live_masked_hook``-wrapped, raised NOTHING
          (0/46 exceptions) and genuinely changed the Jacobian (cos vs exact
          AD 0.55–0.99). So the gate was not preventing a failure, it was
          silently DISCARDING every COMPRESS the policy placed anywhere but
          the terminal vertex: the terminal measurement of a mid-plan COMPRESS
          came back bit-identical to exact AD (cos 1.000000) while the honest
          application scores cos 0.653 (NN) / 0.9957 (TLM).
        * It is also what made the append-only stream non-prefix-stable: the
          same vertex tokenized WITH its COMPRESS at prefix length k and
          WITHOUT it at k+1 (``tests/stream_prefix_property_test.py``), which
          forced the COMPRESS carve-outs in both prefix caches
          (v40: ext=1/431).""")

E(ENV, """        if bi1 == COMPRESS_SENTINEL:
            # COMPRESS: physical axis row[1], kind row[2]. Only honored on the
            # LAST vertex of the partial order (val.ndim reduction upstream of
            # a later elimination trips graphax's shape assertion).
            if not is_last:
                continue
            axis_idx = bi2""",
  """        if bi1 == COMPRESS_SENTINEL:
            # COMPRESS: physical axis row[1], kind row[2]. Honored wherever it
            # sits in the plan — see the docstring for why the last-vertex
            # gate is gone.
            axis_idx = bi2""")

E(ENV, """def _face_dict_for_vertex(config, ij, v, face_row, face_skip,
                          is_last_honored):""",
  """def _face_dict_for_vertex(config, ij, v, face_row, face_skip):""")

E(ENV, """            rules = decode_vertex_rule_specs(
                config.jaxpr, int(v), one_row, is_last=is_last_honored)""",
  """            rules = decode_vertex_rule_specs(config.jaxpr, int(v), one_row)""")

# --- _incremental_stream_tokens ---------------------------------------------
E(ENV, """                               face_key=None, honor_last_compress=True,
                               face_rows_list=None, face_skips_list=None):""",
  """                               face_key=None,
                               face_rows_list=None, face_skips_list=None):""")

E(ENV, '''    # ANCESTOR EXTENSION IS ONLY SOUND ACROSS is_last-INSENSITIVE STATES.
    #
    # Measured (tests/stream_prefix_property_test.py): the stream for a
    # length-k prefix is NOT a byte-prefix of the length-k+1 stream when the
    # prefix carries COMPRESS. `decode_vertex_rule_specs` emits Compress only
    # when `is_last=True`, so vertex k-1 is tokenized WITH its Compress at
    # length k and WITHOUT it at length k+1 — the streams diverge (token 681
    # of 931 on the test graph). Extending a cached parent would then hand the
    # policy a different observation than a cold replay, i.e. the observation
    # would depend on cache state.
    #
    # REFINED BOUND (v40 counters showed the prefix-wide scan left the cache
    # disengaged, ext=1/431: with allow_compress a random policy plants a
    # COMPRESS row within a step or two and every later step replayed cold):
    # the divergence lives ONLY at the vertex whose decode flips is_last
    # between lengths — the CURRENT last vertex. So (a) a state is STORED
    # only if its last vertex is COMPRESS-free (specs AND face rows — same
    # decoder, same sensitivity), making every stored state
    # is_last-insensitive by induction; (b) extension from any stored parent
    # is then always sound; (c) a COMPRESS-last call replays cold WITHOUT
    # consuming its parent (extending would mutate it) and is not stored —
    # one O(t) replay per COMPRESS decision instead of a dead cache.
    # `honor_last_compress=False` (ALPHAGRAD_TOKENS_MID_COMPRESS=0 on a
    # non-terminal step): the caller decoded the last vertex with
    # is_last=False, so its COMPRESS row was DROPPED and the state is not
    # sensitive — storable. The terminal call (honor=True) extends this
    # chain soundly: vertices 0..T-2 decode identically in both worlds.
    _last_has_compress = honor_last_compress and bool(steps) and any(
        int(r[0]) == COMPRESS_SENTINEL for r in steps[-1][1])
    if (honor_last_compress and not _last_has_compress and steps
            and isinstance(face_key, tuple) and face_key):
        # `_face_wire_keys` entries are (flat positions of the non -1 face
        # entries, their values, ...) as int32 bytes. The wire row is
        # ``[r0, r1, r2]``, so the r0 column is exactly the positions
        # divisible by 3 -- the same set the dense ``[0::3]`` slice picked.
        _c = np.frombuffer(face_key[-1][0], dtype=np.int32)
        _v = np.frombuffer(face_key[-1][1], dtype=np.int32)
        _last_has_compress = bool(
            np.any((_c % 3 == 0) & (_v == COMPRESS_SENTINEL)))
    if not _last_has_compress:
        for cut in range(len(steps) - 1, 0, -1):
            parent = _INCR_STREAM_CACHE.pop(
                base_key + (tuple(steps[:cut]),
                            face_key[:cut] if isinstance(face_key, tuple)
                            else None), None)
            if parent is not None:
                _INCR_STREAM_STATS["ext"] += 1
                tk, stream, seg_ids, done = (
                    parent[0], list(parent[1]), list(parent[2]), cut)
                last_start = parent[4]
                if len(parent) > 3 and parent[3]:
                    ft_out = dict(parent[3])
                break''',
  '''    # ANCESTOR EXTENSION IS SOUND FOR EVERY STATE.
    #
    # It used not to be. `decode_vertex_rule_specs` used to emit COMPRESS only
    # when `is_last=True`, so vertex k-1 was tokenized WITH its Compress at
    # prefix length k and WITHOUT it at k+1 — the streams diverged
    # (tests/stream_prefix_property_test.py measured token 681 of 931), and
    # extending a cached parent would have handed the policy a different
    # observation than a cold replay. Two carve-outs lived here: a prefix-wide
    # COMPRESS scan (which left the cache dead, ext=1/431 at v40 — a random
    # policy plants a COMPRESS within a step or two) and then a refined
    # "store only COMPRESS-free-last states" bound.
    #
    # The `is_last` gate is GONE (see decode_vertex_rule_specs), so the decode
    # of a vertex no longer depends on where the prefix ends: the stream for a
    # prefix is a byte-prefix of the stream for any extension for EVERY rule
    # kind, COMPRESS included, and the test asserts it. Nothing to carve out —
    # every state is storable and every parent is extendable.
    for cut in range(len(steps) - 1, 0, -1):
        parent = _INCR_STREAM_CACHE.pop(
            base_key + (tuple(steps[:cut]),
                        face_key[:cut] if isinstance(face_key, tuple)
                        else None), None)
        if parent is not None:
            _INCR_STREAM_STATS["ext"] += 1
            tk, stream, seg_ids, done = (
                parent[0], list(parent[1]), list(parent[2]), cut)
            last_start = parent[4]
            if len(parent) > 3 and parent[3]:
                ft_out = dict(parent[3])
            break''')

E(ENV, """            _pf_v = _face_dict_for_vertex(
                config, tk.ij, int(v), face_rows_list[_ki],
                face_skips_list[_ki],
                is_last_honored=(_ki == len(steps) - 1
                                 and honor_last_compress))""",
  """            _pf_v = _face_dict_for_vertex(
                config, tk.ij, int(v), face_rows_list[_ki],
                face_skips_list[_ki])""")

E(ENV, """    _ft_ret = ft_out if _unified else ft_by_vertex
    if _last_has_compress:
        # is_last-SENSITIVE state — serving it is fine, but it must never
        # become a parent (its last elimination differs from a longer cold
        # replay's view of the same vertex).
        _INCR_STREAM_STATS["nostore"] += 1
        return stream, seg_ids, _ft_ret, last_start
    if len(_INCR_STREAM_CACHE) > _INCR_STREAM_CACHE_CAP:""",
  """    _ft_ret = ft_out if _unified else ft_by_vertex
    if len(_INCR_STREAM_CACHE) > _INCR_STREAM_CACHE_CAP:""")

# --- _face_transforms_for_order ---------------------------------------------
E(ENV, """def _face_transforms_for_order(config, consts, args, o_list, specs_list,
                               face_rows_list, face_skips_list,
                               honor_last_compress=True):""",
  """def _face_transforms_for_order(config, consts, args, o_list, specs_list,
                               face_rows_list, face_skips_list):""")

E(ENV, """        # Pop-and-extend prefix cache: step k's replay is step k-1's replay
        # plus ONE elimination, so advance the cached builder instead of
        # rebuilding it from scratch every env step — O(T) eliminations per
        # episode instead of O(T^2). Only sound while decode is
        # is_last-insensitive, i.e. no COMPRESS anywhere in the prefix
        # (specs OR face rows) — same policy, same reason as
        # _INCR_STREAM_CACHE. Extension MUTATES the builder, so the parent
        # entry is POPPED; a sibling chain that misses takes the honest
        # cold replay.""",
  """        # Pop-and-extend prefix cache: step k's replay is step k-1's replay
        # plus ONE elimination, so advance the cached builder instead of
        # rebuilding it from scratch every env step — O(T) eliminations per
        # episode instead of O(T^2). Sound for EVERY prefix now that the
        # COMPRESS `is_last` gate is gone and a vertex's decode no longer
        # depends on where the prefix ends — same reason as
        # _INCR_STREAM_CACHE. Extension MUTATES the builder, so the parent
        # entry is POPPED; a sibling chain that misses takes the honest
        # cold replay.""")

E(ENV, """        _base = (id(config.jaxpr), tuple(config.argnums))
        # Same refined bound as _INCR_STREAM_CACHE: only the LAST vertex's
        # decode is is_last-sensitive. A COMPRESS-last call is served cold,
        # keeps its parent cached, and is not stored.
        _k_last = len(o_list) - 1
        _last_compress = honor_last_compress and bool(
            COMPRESS_SENTINEL in np.asarray(specs_list[_k_last])[..., 0]
            or COMPRESS_SENTINEL in np.asarray(face_rows_list[_k_last])[..., 0])
        if _last_compress:
            _FACE_ENUM_STATS["compress"] += 1
        if not _last_compress:
            _cache_key = _base + (_sigs,)
            for cut in range(len(_sigs) - 1, 0, -1):
                parent = _FACE_ENUM_CACHE.pop(_base + (_sigs[:cut],), None)
                if parent is not None:
                    ij, out, _start = parent[0], parent[1], cut
                    _FACE_ENUM_STATS["ext"] += 1
                    break
            if ij is None:
                _FACE_ENUM_STATS["cold"] += 1""",
  """        _base = (id(config.jaxpr), tuple(config.argnums))
        # No COMPRESS carve-out: the decode is position-independent, so every
        # prefix is a legal parent and every result is storable.
        _cache_key = _base + (_sigs,)
        for cut in range(len(_sigs) - 1, 0, -1):
            parent = _FACE_ENUM_CACHE.pop(_base + (_sigs[:cut],), None)
            if parent is not None:
                ij, out, _start = parent[0], parent[1], cut
                _FACE_ENUM_STATS["ext"] += 1
                break
        if ij is None:
            _FACE_ENUM_STATS["cold"] += 1""")

E(ENV, """        per_face = _face_dict_for_vertex(
            config, ij, v, face_rows_list[k], face_skips_list[k],
            is_last_honored=(k == last and honor_last_compress))
        if per_face:
            out[v] = per_face
        vertex_rules = decode_vertex_rule_specs(
            config.jaxpr, v, specs_list[k],
            is_last=(k == last and honor_last_compress),
        )""",
  """        per_face = _face_dict_for_vertex(
            config, ij, v, face_rows_list[k], face_skips_list[k])
        if per_face:
            out[v] = per_face
        vertex_rules = decode_vertex_rule_specs(
            config.jaxpr, v, specs_list[k])""")

E(ENV, """    _FACE_ENUM_STATS["elims"] += len(o_list) - _start
    last = len(o_list) - 1
    for k in range(_start, len(o_list)):""",
  """    _FACE_ENUM_STATS["elims"] += len(o_list) - _start
    for k in range(_start, len(o_list)):""")

# --- _callback ---------------------------------------------------------------
E(ENV, """    # ALPHAGRAD_TOKENS_MID_COMPRESS=0: intermediate steps tokenize their last
    # vertex with is_last=False (terminal steps always honor COMPRESS).
    _honor_mid_compress = is_terminal or os.environ.get(
        "ALPHAGRAD_TOKENS_MID_COMPRESS", "1") == "1"
    _faces_np""",
  """    _faces_np""")

E(ENV, """        ft_by_vertex = _face_transforms_for_order(
            config, consts, args, o_list, specs_list,
            _fr_list, _fs_list,
            honor_last_compress=_honor_mid_compress,
        )""",
  """        ft_by_vertex = _face_transforms_for_order(
            config, consts, args, o_list, specs_list,
            _fr_list, _fs_list,
        )""")

E(ENV, """    transforms: list[tuple[int, tuple]] = []
    tok_rules_by_v: dict[int, tuple] = {}
    last_v_idx = len(o_list) - 1
    for v_idx, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(
            config.jaxpr, int(v), specs_list[v_idx],
            is_last=(v_idx == last_v_idx),
        )
        # TOKENIZER-side rules: under ALPHAGRAD_TOKENS_MID_COMPRESS=0 an
        # INTERMEDIATE last vertex is tokenized WITHOUT its COMPRESS
        # (is_last=False decode). The measurement `transforms` above keeps
        # the legacy decode — this changes the observation only, and only
        # where the incremental encoder's carry was already being extended
        # across a rewritten history (the COMPRESS prefix-property
        # violation). The terminal step is unchanged.
        tok_rules = rules
        if v_idx == last_v_idx and not _honor_mid_compress:
            tok_rules = decode_vertex_rule_specs(
                config.jaxpr, int(v), specs_list[v_idx], is_last=False)
        if rules or tok_rules:""",
  """    transforms: list[tuple[int, tuple]] = []
    tok_rules_by_v: dict[int, tuple] = {}
    for v_idx, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(
            config.jaxpr, int(v), specs_list[v_idx])
        # TOKENIZER-side rules are the SAME rules: the decode no longer
        # depends on the vertex's position in the prefix, so the observation
        # and the measured graph cannot disagree about a COMPRESS.
        tok_rules = rules
        if rules or tok_rules:""")

E(ENV, """        stream, seg_ids, _ft_ret, _last_start = _incremental_stream_tokens(
            config, consts, args, o_list, specs_list, tok_rules_by_v,
            ft_by_vertex=ft_by_vertex,
            honor_last_compress=_honor_mid_compress,""",
  """        stream, seg_ids, _ft_ret, _last_start = _incremental_stream_tokens(
            config, consts, args, o_list, specs_list, tok_rules_by_v,
            ft_by_vertex=ft_by_vertex,""")

# The deleted knob must not be settable silently (house rule: a deleted knob is
# a HARD ERROR, cf. ALPHAGRAD_MAX_TOKENS in aa0774c).
E(ENV, """_INCR_STREAM_STATS = {"hit": 0, "ext": 0, "cold": 0, "nostore": 0}""",
  '''_INCR_STREAM_STATS = {"hit": 0, "ext": 0, "cold": 0, "nostore": 0}

if os.environ.get("ALPHAGRAD_TOKENS_MID_COMPRESS") is not None:
    raise RuntimeError(
        "ALPHAGRAD_TOKENS_MID_COMPRESS is GONE. It existed to mitigate the "
        "COMPRESS prefix-property violation by tokenizing an intermediate "
        "last vertex WITHOUT its COMPRESS -- which made the observation "
        "describe an exact contraction while the measurement applied a "
        "reduction. The `is_last` gate it worked around has been removed: "
        "COMPRESS is now honored at every position, so the stream is "
        "append-only and there is nothing to mitigate. Unset the variable.")''')


# --------------------------------------------------------- live_faces.py ----
LF = "src/alphagrad/approx/live_faces.py"

E(LF, """                rules = decode_vertex_rule_specs(
                    self.jaxpr, v, specs[k], is_last=False)""",
  """                rules = decode_vertex_rule_specs(self.jaxpr, v, specs[k])""")

E(LF, """            # (env._face_transforms_for_order -> ft_by_vertex) built an
            # approximated one: the observation and the measured object
            # diverged. ``is_last=False`` is what that builder uses for every
            # vertex but the last of the ORDER, and a prefix vertex here is
            # never that one.
            ft = None
            if frh is not None:
                try:
                    _keys, ft = self._decided(
                        tk, v, frh[k], fsh[k], int(frh.shape[1]),
                        is_last=False)
                except Exception:
                    ft = None""",
  """            # (env._face_transforms_for_order -> ft_by_vertex) built an
            # approximated one: the observation and the measured object
            # diverged. The decode is position-independent now, so replaying a
            # prefix vertex here and deciding it fresh give the same rules.
            ft = None
            if frh is not None:
                try:
                    _keys, ft = self._decided(
                        tk, v, frh[k], fsh[k], int(frh.shape[1]))
                except Exception:
                    ft = None""")

E(LF, """    def _decided(self, tk, vertex, face_rows, face_skips, upto,
                 is_last=True):
        \"\"\"``{face_key: slots|SKIP_FACE}`` for faces ``0..upto-1``.

        ``is_last`` mirrors ``env._face_dict_for_vertex``'s
        ``is_last_honored``: True for the vertex whose faces are being decided
        right now (the newest of the prefix), False when replaying an OLDER
        prefix vertex, which is exactly how ``_face_transforms_for_order``
        decodes it.
        \"\"\"""",
  """    def _decided(self, tk, vertex, face_rows, face_skips, upto):
        \"\"\"``{face_key: slots|SKIP_FACE}`` for faces ``0..upto-1``.

        Position-independent: the same wire rows decode to the same rules for
        the vertex being decided now and for an OLDER prefix vertex being
        replayed, which is what ``_face_transforms_for_order`` also does.
        \"\"\"""")

E(LF, """                try:
                    # is_last=True: the vertex the face loop is deciding is
                    # the NEWEST of the prefix, which is exactly when
                    # decode_vertex_rule_specs admits COMPRESS. Decoding it
                    # with is_last=False drops every COMPRESS row SILENTLY,
                    # so the chunk would describe an exact contraction while
                    # the env applied a reduction.
                    rules = decode_vertex_rule_specs(
                        self.jaxpr, int(vertex), row, is_last=bool(is_last))""",
  """                try:
                    # No position gate: COMPRESS is admitted for every vertex,
                    # so the chunk the head reads and the graph the env
                    # measures carry the same reduction.
                    rules = decode_vertex_rule_specs(
                        self.jaxpr, int(vertex), row)""")

E(LF, """            vrules = decode_vertex_rule_specs(
                self.jaxpr, vertex, vspecs.tolist(), is_last=True)""",
  """            vrules = decode_vertex_rule_specs(
                self.jaxpr, vertex, vspecs.tolist())""")


# -------------------------------------------------------- plan_tokens.py ----
PT = "src/alphagrad/approx/common/plan_tokens.py"

E(PT, """    def _hooks(self, vertex, vertex_specs, is_last):""",
  """    def _hooks(self, vertex, vertex_specs):""")

E(PT, """            rules = decode_vertex_rule_specs(
                self.jaxpr, int(vertex), rows, is_last=bool(is_last))""",
  """            rules = decode_vertex_rule_specs(self.jaxpr, int(vertex), rows)""")

E(PT, """    def face_transforms(self, vertex, face_rows, face_skips, *, is_last=True,
                        keys=None):""",
  """    def face_transforms(self, vertex, face_rows, face_skips, *, keys=None):""")

E(PT, """                    r = decode_vertex_rule_specs(
                        self.jaxpr, int(vertex), row, is_last=bool(is_last))""",
  """                    r = decode_vertex_rule_specs(
                        self.jaxpr, int(vertex), row)""")

E(PT, """    def eliminate(self, vertex, vertex_specs=None, face_rows=None,
                  face_skips=None, *, is_last=True, face_keys=None):""",
  """    def eliminate(self, vertex, vertex_specs=None, face_rows=None,
                  face_skips=None, *, face_keys=None):""")

E(PT, """        hooks = self._hooks(vertex, vertex_specs, is_last)
        ft = self.face_transforms(vertex, face_rows, face_skips,
                                  is_last=is_last, keys=face_keys)""",
  """        hooks = self._hooks(vertex, vertex_specs)
        ft = self.face_transforms(vertex, face_rows, face_skips,
                                  keys=face_keys)""")


# --------------------------------------------------------------- ppo.py ----
PPO = "src/alphagrad/approx/ppo.py"
E(PPO, """                    _oracle_jaxpr, v, specs[k], is_last=(k == n - 1)""",
  """                    _oracle_jaxpr, v, specs[k]""")  # applied 3x below


# ------------------------------------------------------- tools/ppo.py ------
# Legacy snapshot of the trainer, still tracked and still importing the env's
# decoder at runtime -- keep it callable.
E("tools/ppo.py", """                    _oracle_jaxpr, v, specs[k], is_last=(k == n - 1)""",
  """                    _oracle_jaxpr, v, specs[k]""")


# ---------------------------------------------------------- az_gumbel.py ----
AZ = "src/alphagrad/approx/az_gumbel.py"

E(AZ, """    # is_last mirrors the MEASUREMENT's decode exactly (`_callback` uses
    # is_last=(v_idx == last_v_idx) over the FULL order, and
    # `_face_dict_for_vertex`/`_face_transforms_for_order` the same): COMPRESS
    # is honored only on the genuinely TERMINAL vertex.
    #
    # Honoring it mid-plan is what makes the append-only stream
    # non-prefix-stable -- env's documented COMPRESS prefix-property
    # violation, which PPO mitigates with ALPHAGRAD_TOKENS_MID_COMPRESS=0.
    # Here a block is emitted ONCE and never rewritten, so it has to carry
    # the terminal decode from the start or the stream and the measurement
    # describe different graphs.
    #
    # "Terminal" is DETECTED, not counted: eliminating a vertex can make
    # OTHER vertices non-eliminable (nn256: NV=27 valid vertices but 24
    # decisions), so `len(state) == NV - 1` is simply wrong. The last
    # decision is the one taken when exactly one vertex is still legal; the
    # caller asserts the prediction against the graph afterwards, and only a
    # COMPRESS row makes a misprediction observable at all.
    is_last = (len(PT.legal(VALID)) == 1)
    toks, ids = PT.eliminate(vertex, spec_row, face_rows, face_skips,
                             is_last=is_last, face_keys=face_keys)""",
  """    # A block is emitted ONCE here and never rewritten, which is only
    # coherent because the decode is POSITION-INDEPENDENT: COMPRESS is honored
    # at every vertex, so the block a step emits is the block the terminal
    # measurement will decode. (This used to need an `is_last` prediction —
    # "terminal" DETECTED as `len(PT.legal(VALID)) == 1`, because eliminating
    # a vertex can make OTHER vertices non-eliminable — plus a guard asserting
    # the prediction against the graph afterwards. Both are gone with the
    # gate.)
    toks, ids = PT.eliminate(vertex, spec_row, face_rows, face_skips,
                             face_keys=face_keys)""")

E(AZ, """             "face_skips": face_skips, "is_last": is_last})""",
  """             "face_skips": face_skips})""")

E(AZ, '''def _assert_terminal_prediction(d):
    """The `is_last` prediction of the step just COMMITTED, against the graph.

    A misprediction only changes bytes when the step carried a COMPRESS row
    (that is the sole `is_last` sensitivity of `decode_vertex_rule_specs`), so
    the guard fires exactly when it matters instead of on every long tail.
    """
    from alphagrad.approx.env import COMPRESS_SENTINEL
    if d["is_last"] or PT.legal(VALID):
        return
    has_compress = (
        bool(np.any(np.asarray(d["spec_row"])[..., 0] == COMPRESS_SENTINEL))
        or bool(np.any(np.asarray(d["face_rows"])[..., 0] == COMPRESS_SENTINEL)))
    if has_compress:
        raise AssertionError(
            "the terminal decision was not predicted as terminal AND carried "
            "a COMPRESS row: its block was tokenized with is_last=False while "
            "the measurement will decode it with is_last=True, so the "
            "observation and the measured graph diverge. (Elimination made "
            "the remaining legal vertices vanish at the same step.)")''',
  '''def _assert_terminal_prediction(d):
    """No-op kept as a named seam.

    It used to check the `is_last` prediction of the step just COMMITTED
    against the graph, because mispredicting terminality changed the emitted
    bytes whenever the step carried a COMPRESS row. `decode_vertex_rule_specs`
    has no position sensitivity any more, so there is no prediction to be
    wrong about.
    """
    return''')

E(AZ, """        rules = decode_vertex_rule_specs(
            jaxpr, int(order[k]), specs[k].tolist(), is_last=(k == n - 1))""",
  """        rules = decode_vertex_rule_specs(
            jaxpr, int(order[k]), specs[k].tolist())""")

E(AZ, """        face_skips_list=face_skips.tolist(),
        honor_last_compress=True,
        # MUST be the env's own builder, not a hand-rolled tuple: the
        # ancestor-cut branch in _incremental_stream_tokens does
        # np.frombuffer(face_key[-1][0]) and needs the int32-BYTES form
        # _face_wire_keys emits. The dense-tuple form only survived because
        # that branch is skipped whenever the last vertex carries a COMPRESS
        # row -- which is every approximation episode and NO exact one, so
        # --no-approx-head crashed here on its first golden check.""",
  """        face_skips_list=face_skips.tolist(),
        # MUST be the env's own builder, not a hand-rolled tuple: the
        # ancestor-cut branch in _incremental_stream_tokens does
        # np.frombuffer(face_key[-1][0]) and needs the int32-BYTES form
        # _face_wire_keys emits. The dense-tuple form only survived because
        # that branch used to be skipped whenever the last vertex carried a
        # COMPRESS row -- which is every approximation episode and NO exact
        # one, so --no-approx-head crashed here on its first golden check.
        # That skip is gone, so the branch is now always live.""")

E(AZ, """                    PT.eliminate(_wv, EXACT_SPEC_ROW, EXACT_FACE_ROWS,
                                 EXACT_FACE_SKIPS,
                                 is_last=(len(_wlegal) == 1))""",
  """                    PT.eliminate(_wv, EXACT_SPEC_ROW, EXACT_FACE_ROWS,
                                 EXACT_FACE_SKIPS)""")


def main():
    dry = "--dry" in sys.argv
    n_ok = 0
    for path, old, new in EDITS:
        full = ROOT + path
        src = open(full).read()
        if old not in src:
            if new in src:
                print(f"SKIP (already applied) {path}: {old.splitlines()[0][:70]}")
                continue
            print(f"!! NOT FOUND in {path}: {old.splitlines()[0][:90]}")
            sys.exit(1)
        cnt = src.count(old)
        src = src.replace(old, new)
        if not dry:
            open(full, "w").write(src)
        n_ok += 1
        print(f"ok x{cnt} {path}: {old.splitlines()[0][:70]}")
    print(f"{n_ok}/{len(EDITS)} edits applied{' (DRY)' if dry else ''}")


if __name__ == "__main__":
    main()
