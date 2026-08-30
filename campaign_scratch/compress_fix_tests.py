"""Test-side half of the COMPRESS `is_last` removal."""
import sys

ROOT = "/Users/assmuth/dsnn/alphagrad/"
EDITS: list[tuple[str, str, str]] = []


def E(path, old, new):
    EDITS.append((path, old, new))


# ------------------------------------------------- stream_prefix_property ---
SPP = "tests/stream_prefix_property_test.py"
SPP_NEW = '''"""Does the append-only token stream ACTUALLY have the prefix property?

`_incremental_stream_tokens` caches a tokenizer per elimination prefix and
extends it, on the stated invariant: "The stream for a prefix is a byte-wise
prefix of the stream for any extension."

That invariant is what the ancestor-extension fast path, the face-enum prefix
cache and any prefix memo for the oracle rest on. It USED TO BE FALSE for
COMPRESS: `decode_vertex_rule_specs` took an `is_last` flag and emitted
Compress only when it was set, so the vertex at index k-1 was tokenized WITH
its Compress at prefix length k and WITHOUT it at length k+1 -- the streams
diverged (measured: token 681 of 931 on this graph), and BOTH prefix caches
carried a COMPRESS carve-out because of it (v40: ext=1/431).

The gate is GONE (2026-08-15). Measured before removing it: the same decoded
COMPRESS applied at all 23 positions of the NeuralNetwork plan and 11 sampled
positions of the TransformerLM plan, raw and hook-wrapped, raised NOTHING and
genuinely changed the Jacobian -- the gate was discarding real approximations,
not preventing a failure.

So the property now holds for EVERY rule kind and these tests ASSERT it. If a
future change re-introduces any position sensitivity in the decode, the
`compress` case here fails first.
"""
import inspect
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.micro_actions import Compress

from alphagrad.approx.env import (
    COMPRESS_SENTINEL, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs,
)


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)


def _stream_for(prefix, specs_by_v, vocab=248):
    """Cold replay: tokenize exactly `prefix`, decoding each vertex's rules
    the way `_callback` does for a partial order."""
    from graphax import IncrementalPathTokenizer

    cj = jax.make_jaxpr(_fn)(*ARGS)
    tk = IncrementalPathTokenizer(
        cj.jaxpr, (0, 1), list(cj.literals), list(ARGS), vocab_size=vocab)
    stream = [int(t) for t in tk.base_tokens()]
    for v in prefix:
        rules = decode_vertex_rule_specs(cj.jaxpr, int(v), specs_by_v[int(v)])
        stream += [int(t) for t in tk.eliminate(int(v), tuple(rules))]
    return stream


def _rows(kind):
    """MAX_RULES-shaped spec rows: 'none', 'diag', or 'compress'."""
    rows = [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]
    if kind == "diag":
        rows[0] = [0, 0, 2]
    elif kind == "compress":
        rows[0] = [COMPRESS_SENTINEL, 0, 0]
    return rows


@pytest.mark.parametrize("kind", ["none", "diag", "compress"])
def test_prefix_property(kind):
    """stream(prefix[:k]) must be a byte-prefix of stream(prefix[:k+1]),
    at EVERY k and for EVERY rule kind -- COMPRESS included."""
    prefix = [1, 2, 3]
    specs = {v: _rows(kind) for v in prefix}
    for k in range(1, len(prefix)):
        short = _stream_for(prefix[:k], specs)
        long = _stream_for(prefix[:k + 1], specs)
        if long[: len(short)] != short:
            i = next((i for i, (a, b) in enumerate(zip(short, long))
                      if a != b), len(short))
            pytest.fail(
                f"PREFIX PROPERTY VIOLATED for {kind} at k={k}: streams "
                f"diverge at token {i} of {len(short)} "
                f"(short={short[max(0, i-2):i+3]}, "
                f"long={long[max(0, i-2):i+3]}). Ancestor extension in "
                f"_incremental_stream_tokens and the _FACE_ENUM_CACHE "
                f"pop-extend are UNSOUND in this state.")


def test_decode_has_no_position_parameter():
    """The gate is gone at the level of the signature, not just its default.

    A `decode_vertex_rule_specs(..., is_last=...)` that silently defaulted to
    True would leave every caller free to re-introduce the divergence.
    """
    params = inspect.signature(decode_vertex_rule_specs).parameters
    assert "is_last" not in params, (
        f"decode_vertex_rule_specs regrew a position parameter: {list(params)}")


def test_compress_decodes_at_any_position():
    """The same COMPRESS row decodes to the same rule for every vertex."""
    cj = jax.make_jaxpr(_fn)(*ARGS)
    rows = _rows("compress")
    got = [decode_vertex_rule_specs(cj.jaxpr, v, rows)
           for v in (1, 2, 3)]
    assert all(len(g) == 1 and isinstance(g[0], Compress) for g in got), got
    assert got[0] == got[1] == got[2], got
'''
E(SPP, "@@WHOLE@@", SPP_NEW)


# -------------------------------------------------------- rule_replay_test --
RR = "tests/rule_replay_test.py"

E(RR, """pin (a) faithful decoding of each row type incl. the is_last COMPRESS gate and
the -1 (joint gcd) factor sentinel, and (b) that advancing the oracle WITH the""",
  """pin (a) faithful decoding of each row type -- including a COMPRESS that is
honored at EVERY position of the plan, no longer only on the terminal vertex --
and the -1 (joint gcd) factor sentinel, and (b) that advancing the oracle WITH the""")

E(RR, """    rules = decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, -1]), is_last=True)""",
  """    rules = decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, -1]))""")

E(RR, """    rules = decode_vertex_rule_specs(
        cj.jaxpr, 1, _rows([QUANT_SENTINEL, 0, 0]), is_last=False)""",
  """    rules = decode_vertex_rule_specs(
        cj.jaxpr, 1, _rows([QUANT_SENTINEL, 0, 0]))""")

E(RR, """    assert decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, 1]), is_last=True) == ()
    assert decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, 5]), is_last=True) == ()""",
  """    assert decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, 1])) == ()
    assert decode_vertex_rule_specs(cj.jaxpr, 1, _rows([0, 1, 5])) == ()""")

E(RR, '''def test_compress_honored_only_on_last_vertex():
    fn = lambda x, y: jnp.tanh(x @ y)
    cj = _mk(fn, (jnp.ones((4, 6)), jnp.ones((6, 4))))
    row = _rows([COMPRESS_SENTINEL, 0, 0])
    assert decode_vertex_rule_specs(cj.jaxpr, 1, row, is_last=False) == ()
    got = decode_vertex_rule_specs(cj.jaxpr, 1, row, is_last=True)
    assert len(got) == 1 and isinstance(got[0], Compress)''',
  '''def test_compress_honored_at_every_position():
    """COMPRESS decodes for any vertex, not only the terminal one.

    The removed `is_last` gate returned () for every non-final vertex, which
    silently discarded the action: the terminal measurement came back
    bit-identical to exact AD while the policy had been credited with an
    approximation.
    """
    fn = lambda x, y: jnp.tanh(x @ y)
    cj = _mk(fn, (jnp.ones((4, 6)), jnp.ones((6, 4))))
    row = _rows([COMPRESS_SENTINEL, 0, 0])
    got = decode_vertex_rule_specs(cj.jaxpr, 1, row)
    assert len(got) == 1 and isinstance(got[0], Compress)


def test_mid_plan_compress_reaches_the_measured_jacobian():
    """A COMPRESS on a NON-final vertex must survive into the measurement.

    Numeric, not structural: eliminate a COMPLETE order with one COMPRESS
    planted at each position in turn and compare against exact AD. Every
    position must (a) not raise -- graphax's nominal-shape asserts are
    exact-AD-only and apply_compress drops the axis POINTER, not the logical
    size -- and (b) at least one NON-FINAL position must actually move the
    Jacobian, which is precisely what the old gate suppressed.
    """
    from graphax import jacve
    from graphax.core import _build_graph
    from alphagrad.approx.common.masks import make_live_masked_hook

    fn = lambda x, y: jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)
    args = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)
    cj = jax.make_jaxpr(fn)(*args)
    _, _, _, vo = _build_graph(cj.jaxpr, args, list(cj.literals), (0, 1))
    valid = [i for i, eqn in enumerate(cj.jaxpr.eqns, 1)
             if eqn.outvars[0] not in cj.jaxpr.outvars or i in vo]
    order = list(reversed(valid))

    def _flat(t):
        return np.concatenate(
            [np.asarray(x, np.float64).ravel()
             for x in jax.tree_util.tree_leaves(t)])

    ref = _flat(jax.jit(jacve(fn, order, argnums=(0, 1)))(*args))
    moved = 0
    for idx, v in enumerate(order[:-1]):          # NON-final positions only
        rules = decode_vertex_rule_specs(
            cj.jaxpr, int(v), _rows([COMPRESS_SENTINEL, 0, 0]))
        if not rules:
            continue
        out = _flat(jax.jit(jacve(                # must not raise
            fn, order, argnums=(0, 1),
            transforms=[(int(v),
                         (make_live_masked_hook(tuple(rules)),))]))(*args))
        cos = float(out @ ref / (np.linalg.norm(out) * np.linalg.norm(ref)))
        if cos < 0.999:
            moved += 1
    assert moved > 0, (
        "no NON-FINAL COMPRESS changed the measured Jacobian -- the position "
        "gate is back, or the rules are being dropped somewhere downstream")''')

E(RR, """                        rules = decode_vertex_rule_specs(
                            cj.jaxpr, v, _rows([bi1, bi2, -1]), is_last=False)""",
  """                        rules = decode_vertex_rule_specs(
                            cj.jaxpr, v, _rows([bi1, bi2, -1]))""")


# --------------------------------------------------- delta_obs_emission ------
DOE = "tests/delta_obs_emission_test.py"

E(DOE, '''    """Reference replay: tokenize exactly `prefix` with the same rule decode
    `_callback` uses, so `is_last` lands on the same vertex."""''',
  '''    """Reference replay: tokenize exactly `prefix` with the same rule decode
    `_callback` uses (position-independent, so a prefix vertex and the newest
    vertex decode identically)."""''')

E(DOE, """    last = len(prefix) - 1
    for k, v in enumerate(prefix):
        rules = decode_vertex_rule_specs(
            _CJ.jaxpr, int(v), specs_np[int(v) - 1], is_last=(k == last))""",
  """    for v in prefix:
        rules = decode_vertex_rule_specs(
            _CJ.jaxpr, int(v), specs_np[int(v) - 1])""")

E(DOE, """    if os.environ.get("ALPHAGRAD_TOKENS_MID_COMPRESS", "1") == "0":
        pytest.skip("mid-compress decode changes is_last semantics")
    specs = _specs(diag)""",
  """    specs = _specs(diag)""")

E(DOE, """    # NON-TERMINAL steps only: the terminal step would run the measurement,
    # and its is_last decode is the one the prefix property does not cover.""",
  """    # NON-TERMINAL steps only: the terminal step would run the measurement.""")


# --------------------------------------------------- live_face_prefix --------
LFP = "tests/live_face_prefix_test.py"

E(LFP, '''    """Face 0 of ``v_cur`` on the prefix graph THE MEASUREMENT BUILDS.

    ``honor_last_compress=False`` puts every prefix vertex on ``is_last=False``
    -- the decode the terminal measurement gives a vertex that is not the last
    of the order, and the one ``_tokenizer_at`` replays with. The searched
    decisions below are DIAG, for which ``is_last`` is irrelevant, so nothing
    here depends on that choice.
    """''',
  '''    """Face 0 of ``v_cur`` on the prefix graph THE MEASUREMENT BUILDS.

    The decode is position-independent, so this replay, ``_tokenizer_at``'s
    and the terminal measurement's all give a vertex the same rules.
    """''')

E(LFP, """        [np.asarray(skips[k]).tolist() for k in range(n)],
        honor_last_compress=False,
    )""",
  """        [np.asarray(skips[k]).tolist() for k in range(n)],
    )""")


# --------------------------------------------------- az_plan_tokens ----------
AZT = "tests/az_plan_tokens_test.py"
E(AZT, """            t, i = pt.eliminate(v, is_last=(len(legal) == 1))""",
  """            t, i = pt.eliminate(v)""")
E(AZT, """        cfg, consts, xs, order, specs.tolist(), {}, honor_last_compress=True)""",
  """        cfg, consts, xs, order, specs.tolist(), {})""")


# --------------------------------------------------- face_prefix_cache -------
FPC = "tests/face_prefix_cache_test.py"

E(FPC, """Soundness bound (measured in stream_prefix_property_test.py): only the
CURRENT LAST vertex's decode is is_last-sensitive — COMPRESS is emitted only
when `is_last=True`. The rule both caches implement: a COMPRESS-last state is
SERVED (cold, without consuming its parent) but never STORED, so every stored
state is is_last-insensitive by induction and extension from any stored
parent is sound. One COMPRESS decision costs one cold replay, not a dead
cache for the rest of the episode (the v40 failure mode: ext=1/431).""",
  """Soundness (asserted in stream_prefix_property_test.py): the decode has NO
position sensitivity, so the stream for a prefix is a byte-prefix of the
stream for any extension for every rule kind. Both caches therefore store
every state and extend from every parent — there is no COMPRESS carve-out any
more. There used to be one, because `decode_vertex_rule_specs` emitted
COMPRESS only when `is_last=True`; the tests below pin that COMPRESS is now
ordinary, in the per-vertex specs AND in the face rows (the v40 failure mode
it caused: ext=1/431).""")

E(FPC, """        rules = decode_vertex_rule_specs(
            cfg.jaxpr, int(v), specs_list[i], is_last=(i == last))
        if rules:
            d[int(v)] = (make_live_masked_hook(tuple(rules)),)
    return d""",
  """        rules = decode_vertex_rule_specs(cfg.jaxpr, int(v), specs_list[i])
        if rules:
            d[int(v)] = (make_live_masked_hook(tuple(rules)),)
    return d""")

E(FPC, """def _tok_rules(cfg, o_list, specs_list):
    d = {}
    last = len(o_list) - 1
    for i, v in enumerate(o_list):""",
  """def _tok_rules(cfg, o_list, specs_list):
    d = {}
    for i, v in enumerate(o_list):""")

E(FPC, '''    Only asserted where the prefix property actually holds: a COMPRESS-last
    state is is_last-SENSITIVE and its predecessor is deliberately not a
    byte-prefix of it (see `_incremental_stream_tokens`).
    """''',
  '''    The prefix property holds unconditionally now, so this is asserted for
    every step (the guard below is kept because the very first prefix has no
    predecessor to compare against).
    """''')

E(FPC, '''def test_face_enum_cache_compress_last_served_cold_chain_survives():
    T = len(_episode()[3])
    c = T // 2
    ep = _episode(compress_at=c)
    on, stats_on = _ft_pass(*ep, cached=True)
    off, _ = _ft_pass(*ep, cached=False)
    for t, (a, b) in enumerate(zip(on, off), start=1):
        assert _ft_sig(a) == _ft_sig(b), f"face enum diverged at step {t}"
    # step c+1 (COMPRESS-last) is served cold with NO cache interaction; its
    # parent survives, so step c+2 extends from cut=c across two vertices —
    # the chain loses exactly one extension, not the rest of the episode.
    assert {k: stats_on[k] for k in ("ext", "cold")} == {
        "ext": T - 2, "cold": 1}, stats_on''',
  '''def test_face_enum_cache_compress_is_not_special():
    """A COMPRESS in the per-vertex specs costs the cache NOTHING.

    It used to cost one cold replay per COMPRESS decision (and, before the
    refined bound, the rest of the episode).
    """
    T = len(_episode()[3])
    c = T // 2
    ep = _episode(compress_at=c)
    on, stats_on = _ft_pass(*ep, cached=True)
    off, _ = _ft_pass(*ep, cached=False)
    for t, (a, b) in enumerate(zip(on, off), start=1):
        assert _ft_sig(a) == _ft_sig(b), f"face enum diverged at step {t}"
    assert {k: stats_on[k] for k in ("ext", "cold")} == {
        "ext": T - 1, "cold": 1}, stats_on''')

E(FPC, '''def test_stream_compress_last_served_cold_not_stored_chain_survives():
    T = len(_episode()[3])
    c = T // 2
    ep = _episode(compress_at=c)
    cfg, consts, args, order, specs, faces, skips = ep
    fts, _ = _ft_pass(*ep, cached=False)
    chained, stats_ch = _stream_pass(cfg, consts, args, order, specs, faces,
                                     skips, fts, chained=True)
    cold, _ = _stream_pass(cfg, consts, args, order, specs, faces, skips,
                           fts, chained=False)
    for t, (a, b) in enumerate(zip(chained, cold), start=1):
        assert a[0] == b[0] and a[1] == b[1], f"diverged at step {t}"
    # t=1 cold(store); t=c+1 COMPRESS-last: cold + nostore, parent kept;
    # t=c+2 extends from cut=c (re-eliminating vertex c with is_last=False,
    # which drops the COMPRESS — the exact divergence the rule guards).
    assert stats_ch == {"hit": 0, "ext": T - 2, "cold": 2, "nostore": 1}, (
        stats_ch)''',
  '''def test_stream_compress_in_specs_extends_like_any_other_rule():
    """A COMPRESS in the per-vertex specs neither breaks the chain nor blocks
    storage: one cold replay for the whole episode, T-1 extensions, and the
    chained streams are byte-identical to the cold ones."""
    T = len(_episode()[3])
    c = T // 2
    ep = _episode(compress_at=c)
    cfg, consts, args, order, specs, faces, skips = ep
    fts, _ = _ft_pass(*ep, cached=False)
    chained, stats_ch = _stream_pass(cfg, consts, args, order, specs, faces,
                                     skips, fts, chained=True)
    cold, _ = _stream_pass(cfg, consts, args, order, specs, faces, skips,
                           fts, chained=False)
    for t, (a, b) in enumerate(zip(chained, cold), start=1):
        assert a[0] == b[0] and a[1] == b[1], f"diverged at step {t}"
    assert stats_ch == {"hit": 0, "ext": T - 1, "cold": 1, "nostore": 0}, (
        stats_ch)''')

E(FPC, '''    # a COMPRESS inside a FACE row has the same is_last sensitivity: t=1 is
    # cold+nostore, t=2 cold (nothing stored yet), t>=3 extend.
    assert stats_ch == {"hit": 0, "ext": T - 2, "cold": 2, "nostore": 1}, (
        stats_ch)''',
  '''    # a COMPRESS inside a FACE row is equally ordinary now.
    assert stats_ch == {"hit": 0, "ext": T - 1, "cold": 1, "nostore": 0}, (
        stats_ch)''')

# the two ALPHAGRAD_TOKENS_MID_COMPRESS=0 tests describe a knob that no longer
# exists -- delete them together with their helper.
E(FPC, '''def _tok_rules_mid(cfg, o_list, specs_list, honor):
    """Tokenizer rules as the callback builds them under
    ALPHAGRAD_TOKENS_MID_COMPRESS=0: the intermediate last vertex decodes
    with is_last=False; `honor` is True only on the terminal step."""
    d = {}
    last = len(o_list) - 1
    for i, v in enumerate(o_list):
        rules = decode_vertex_rule_specs(
            cfg.jaxpr, int(v), specs_list[i],
            is_last=(i == last and honor))
        if rules:
            d[int(v)] = (make_live_masked_hook(tuple(rules)),)
    return d


def test_stream_mid_compress_off_extends_through_compress_and_terminal():
    """Flag=0 world: every intermediate COMPRESS-last state is storable, the
    chain never dies, and the TERMINAL call (honor=True) soundly extends the
    honor=False chain - its final elimination is the only difference."""
    T = len(_episode()[3])
    ep = _episode(compress_at=T // 2, seed=3)
    cfg, consts, args, order, specs, faces, skips = ep
    fts = [
        E._face_transforms_for_order(
            cfg, consts, args, order[:t], specs[:t],
            [np.asarray(f).tolist() for f in faces[:t]],
            [np.asarray(s).tolist() for s in skips[:t]],
            honor_last_compress=(t == T))
        for t in range(1, T + 1)
    ]

    def _pass(chained):
        E._INCR_STREAM_CACHE.clear()
        _reset(E._INCR_STREAM_STATS)
        out = []
        for t in range(1, T + 1):
            if not chained:
                E._INCR_STREAM_CACHE.clear()
            honor = (t == T)
            stream, seg, _ft, ls = E._incremental_stream_tokens(
                cfg, consts, args, order[:t], specs[:t],
                _tok_rules_mid(cfg, order[:t], specs[:t], honor),
                ft_by_vertex=fts[t - 1], face_key=_fk(faces, skips, t),
                honor_last_compress=honor)
            _check_last_start(out, stream, ls)
            out.append((list(stream), list(seg)))
        return out, dict(E._INCR_STREAM_STATS)

    chained, stats_ch = _pass(chained=True)
    cold, _ = _pass(chained=False)
    for t, (a, b) in enumerate(zip(chained, cold), start=1):
        assert a[0] == b[0] and a[1] == b[1], f"diverged at step {t}"
    # every step after the first extends - the COMPRESS-at-mid step included
    assert stats_ch["ext"] == T - 1 and stats_ch["cold"] == 1, stats_ch


def test_stream_mid_compress_off_terminal_compress_not_stored():
    """Terminal step with a COMPRESS-carrying FINAL vertex: is_last-sensitive
    again (honor=True), so it takes the one honest cold replay of the
    episode, leaves its parent untouched, and is never stored."""
    T = len(_episode()[3])
    ep = _episode(compress_at=T - 1, seed=4)
    cfg, consts, args, order, specs, faces, skips = ep
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    for t in range(1, T + 1):
        honor = (t == T)
        E._incremental_stream_tokens(
            cfg, consts, args, order[:t], specs[:t],
            _tok_rules_mid(cfg, order[:t], specs[:t], honor),
            ft_by_vertex=None, face_key=None, honor_last_compress=honor)
    # t=1 cold, t=2..T-1 extend, t=T cold + nostore (parent preserved)
    assert E._INCR_STREAM_STATS == {
        "hit": 0, "ext": T - 2, "cold": 2, "nostore": 1}, E._INCR_STREAM_STATS


''',
  '''def test_stream_terminal_compress_extends_and_stores():
    """A COMPRESS on the FINAL vertex used to be the one is_last-sensitive
    state: served cold, never stored. It is ordinary now -- one cold replay
    at t=1 and an extension at every step including the terminal one."""
    T = len(_episode()[3])
    ep = _episode(compress_at=T - 1, seed=4)
    cfg, consts, args, order, specs, faces, skips = ep
    E._INCR_STREAM_CACHE.clear()
    _reset(E._INCR_STREAM_STATS)
    for t in range(1, T + 1):
        E._incremental_stream_tokens(
            cfg, consts, args, order[:t], specs[:t],
            _tok_rules(cfg, order[:t], specs[:t]),
            ft_by_vertex=None, face_key=None)
    assert E._INCR_STREAM_STATS == {
        "hit": 0, "ext": T - 1, "cold": 1, "nostore": 0}, E._INCR_STREAM_STATS


''')


def main():
    dry = "--dry" in sys.argv
    for path, old, new in EDITS:
        full = ROOT + path
        if old == "@@WHOLE@@":
            if not dry:
                open(full, "w").write(new)
            print(f"ok REWRITE {path}")
            continue
        src = open(full).read()
        if old not in src:
            if new in src:
                print(f"SKIP (applied) {path}: {old.splitlines()[0][:70]}")
                continue
            print(f"!! NOT FOUND in {path}: {old.splitlines()[0][:90]}")
            sys.exit(1)
        src = src.replace(old, new)
        if not dry:
            open(full, "w").write(src)
        print(f"ok {path}: {old.splitlines()[0][:70]}")
    print(f"{len(EDITS)} test edits{' (DRY)' if dry else ''}")


if __name__ == "__main__":
    main()
