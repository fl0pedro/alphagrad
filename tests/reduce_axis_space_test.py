"""``--reduce-axis-space`` (ticket dsnn-3qm.20, defect D5).

A Reduce (Compress) axis lived in three coordinate spaces that nobody
converted between: the head and the wire named a LOGICAL dim of the slot's
tensor (``out_dims ++ primal_dims``), ``masks.compress_valid_mask`` masked a
PHYSICAL ``val`` axis (``k < val.ndim``), and graphax's ``apply_compress``
read the same integer as a CANONICAL slot (``canonical_axis_order``). Finding
56 (job 63579) measured that on TLM lhs edges the three never coincide
(0/114 records): a decoder axis was a no-op, a masked-but-real reduction of a
different dim, or a raise, depending on the space.

The owner's ruling: Diag on logical dims, Reduce on PHYSICAL val axes. Under
``--reduce-axis-space physical`` (the default) ``Compress.axes`` inside
alphagrad ARE physical val axes, and ``masks.reduce_axis_spaces`` is the one
conversion: logical -> physical at decode (``env.make_slot_frame_hook``,
``masks.slot_legality``), physical -> canonical at the graphax boundary
(``masks.make_live_masked_hook``). ``canonical`` is the pre-ticket path.

What is pinned:

  * the conversion table agrees with graphax's own slot list, dim by dim;
  * on the finding-56 shapes (the v81 dot_general lhs, the v92 two-pair lhs,
    a tensor whose canonical slot k is another dim than logical dim k, a
    block-diagonal pair) an emitted ``Compress.axes`` names exactly the
    physical axis graphax drops, and a masked axis is never applied;
  * the same, end to end, on all three slots (lhs, rhs, new) of a real
    elimination under ``--face-slot-frames slot``, with the per-slot mask
    equal to the hook's verdict, with and without ``--per-face-masks``;
  * ``canonical`` is the pre-ticket code path, byte for byte, in every
    predicate and in the hook;
  * a drift between the mirrored traversal and graphax raises.
"""
import contextlib
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from graphax import inline_call_primitives                      # noqa: E402
from graphax.incremental import IncrementalJaxpr                # noqa: E402
from graphax.sparse.indexes import DenseIndex, DiagonalIndex    # noqa: E402
from graphax.sparse.micro_actions import (                      # noqa: E402
    Compress, Diag, apply_compress, apply_diag, canonical_axis_order)
from graphax.sparse.tensor import SparseTensor                  # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.common import masks as M                  # noqa: E402
from alphagrad.approx.common.masks import (                     # noqa: E402
    arm_face_counts, compress_rules_to_physical, compress_to_graphax,
    disarm_face_counts, reduce_axes_physical, reduce_axis_mask,
    reduce_axis_space, reduce_axis_spaces, set_face_slot_frames,
    set_per_face_masks, set_reduce_axis_space, slot_legality)
from alphagrad.approx.env import (                              # noqa: E402
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX,
    _face_dict_for_vertex, make_slot_frame_hook)

try:
    from jax.extend.core import ClosedJaxpr
except ImportError:                                        # pragma: no cover
    from jax._src.core import ClosedJaxpr

N_AX = 8
SLOTS = ("lhs", "rhs", "new")


@pytest.fixture(autouse=True)
def _defaults():
    """Every test starts from the defaults (physical, slot frames, per-face
    masks off) and leaves them."""
    set_reduce_axis_space("physical")
    set_face_slot_frames(True)
    set_per_face_masks(False)
    try:
        yield
    finally:
        set_reduce_axis_space("physical")
        set_face_slot_frames(True)
        set_per_face_masks(False)
        disarm_face_counts()


# --------------------------------------------------------------------------
# Hand-built tensors on the finding-56 shapes (sizes shrunk; the layouts are
# the recorded ones).
# --------------------------------------------------------------------------
def _val(shape):
    return jnp.asarray(np.arange(1, int(np.prod(shape)) + 1,
                                 dtype=np.float32).reshape(shape) / 7.0)


def _st(out_dims, primal_dims, shape):
    return SparseTensor(tuple(out_dims), tuple(primal_dims), _val(shape))


def _dense():
    """No pair, no implicit dim: the one layout on which the three spaces
    coincide (the control)."""
    return _st([DenseIndex(0, 4, 0), DenseIndex(1, 8, 1)],
               [DenseIndex(2, 6, 2)], (4, 8, 6))


def _v81():
    """Finding 56 D5b, the v81 dot_general lhs: ``out=(64, 2048)
    primal=(128, 2048) val=(64, 128) canon=(0, None, 1)``. out1 and primal1
    are an IMPLICIT coupled pair, out0 and primal0 are dense."""
    return _st([DenseIndex(0, 4, 0), DiagonalIndex(1, 8, None, other_id=3)],
               [DenseIndex(2, 6, 1), DiagonalIndex(3, 8, None, other_id=1)],
               (4, 6))


def _two_pairs():
    """Finding 56 D5b, the v92 mul lhs: two coupled pairs (out0, primal0)
    and (out1, primal1), ``val=(64, 2048) canon=(0, 1)``. The decoder's
    tokens 2 and 3 (the primal dims) RAISED in graphax."""
    return _st([DiagonalIndex(0, 4, 0, other_id=2),
                DiagonalIndex(1, 8, 1, other_id=3)],
               [DiagonalIndex(2, 4, 0, other_id=0),
                DiagonalIndex(3, 8, 1, other_id=1)], (4, 8))


def _shifted():
    """An implicit pair FIRST, then dense dims: canonical slot 2 is primal1's
    axis while logical dim 2 is primal0, and ``val.ndim`` is 3 so the old
    physical mask admits k=2. The pre-ticket path applied the head's request
    for primal0 to primal1 (the D5c shape, ``canon=(None, 0, 1, 2)``)."""
    return _st([DiagonalIndex(0, 8, None, other_id=2), DenseIndex(1, 4, 0)],
               [DiagonalIndex(2, 8, None, other_id=0), DenseIndex(3, 6, 1),
                DenseIndex(4, 5, 2)], (4, 6, 5))


def _block_pair():
    """A block-diagonal pair from graphax's own Diag: meta axis plus a block
    axis per side, ``canon=(meta, block_of_first)``."""
    st = _st([DenseIndex(0, 8, 0)], [DenseIndex(1, 8, 1)], (8, 8))
    return apply_diag(st, Diag(i=0, j=1, factor=4))


_FIXTURES = {"dense": _dense, "v81": _v81, "two_pairs": _two_pairs,
             "shifted": _shifted, "block_pair": _block_pair}


def _dims(st):
    return (*st.out_dims, *st.primal_dims)


def _dim_by_id(st, did):
    return next(d for d in _dims(st) if d.id == did)


# ==========================================================================
# 0. THE DEFAULT
# ==========================================================================
def test_default_is_physical():
    assert reduce_axis_space() == "physical"
    assert reduce_axes_physical()


def test_the_setter_republishes_to_the_environment():
    set_reduce_axis_space("canonical")
    assert os.environ["ALPHAGRAD_REDUCE_AXIS_SPACE"] == "canonical"
    assert not reduce_axes_physical()
    set_reduce_axis_space("physical")
    assert os.environ["ALPHAGRAD_REDUCE_AXIS_SPACE"] == "physical"
    with pytest.raises(ValueError):
        set_reduce_axis_space("logical")


# ==========================================================================
# 1. THE TABLE agrees with graphax, dim by dim
# ==========================================================================
@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_the_conversion_table_agrees_with_graphax(name):
    st = _FIXTURES[name]()
    sp = reduce_axis_spaces(st)
    canon = tuple(canonical_axis_order(st))
    assert sp.phys_of_slot == canon
    assert sp.n_dims == len(_dims(st)) == len(sp.phys_of_dim)
    assert sp.n_phys == st.val.ndim == len(sp.slot_of_phys)
    seen_pairs = {}
    for a, d in enumerate(_dims(st)):
        k = sp.slot_of_dim[a]
        assert 0 <= k < len(canon)
        if d.other_id is None:
            assert canon[k] == d.axis
            assert sp.phys_of_dim[a] == d.axis
        else:
            key = frozenset((d.id, d.other_id))
            # both dims of a pair share ONE slot, the first-seen dim's
            assert seen_pairs.setdefault(key, k) == k
            assert sp.phys_of_dim[a] == canon[k]
    for p, k in enumerate(sp.slot_of_phys):
        if k is not None:
            assert canon[k] == p
    # every physical axis some slot points at is addressable
    for k, p in enumerate(canon):
        if p is not None:
            assert sp.slot_of_phys[p] is not None


def test_the_fixtures_span_the_finding_56_layouts():
    """The control: the shapes finding 56 reported, and that the three
    spaces DO differ on all but the dense one."""
    assert reduce_axis_spaces(_dense()).phys_of_dim == (0, 1, 2)
    assert canonical_axis_order(_v81()) == (0, None, 1)
    assert reduce_axis_spaces(_v81()).phys_of_dim == (0, None, 1, None)
    assert canonical_axis_order(_two_pairs()) == (0, 1)
    assert reduce_axis_spaces(_two_pairs()).phys_of_dim == (0, 1, 0, 1)
    assert canonical_axis_order(_shifted()) == (None, 0, 1, 2)
    assert reduce_axis_spaces(_shifted()).phys_of_dim == (None, 0, None, 1, 2)
    bp = _block_pair()
    sp = reduce_axis_spaces(bp)
    assert bp.val.ndim == 3 and len(canonical_axis_order(bp)) == 2
    assert sp.phys_of_dim == (0, 0)
    assert sp.slot_of_phys.count(None) == 1     # the partner's block axis
    for name in ("v81", "two_pairs", "shifted", "block_pair"):
        sp = reduce_axis_spaces(_FIXTURES[name]())
        assert not (sp.n_dims == len(sp.phys_of_slot) == sp.n_phys), name


def test_a_drift_from_graphax_raises(monkeypatch):
    import graphax.sparse.micro_actions as mod
    monkeypatch.setattr(mod, "canonical_axis_order", lambda st: (0,))
    with pytest.raises(RuntimeError):
        reduce_axis_spaces(_two_pairs())


# ==========================================================================
# 2. AN EMITTED AXIS NAMES THE AXIS GRAPHAX DROPS
# ==========================================================================
def _emit(st, a):
    """What the slot-frame decode emits for wire token ``a`` on ``st``."""
    return compress_rules_to_physical(st, (Compress(axes=(a,), kind="mean"),))[0]


def _apply_through_hook(st, rule):
    stats = {}
    out = M.make_live_masked_hook((rule,), stats=stats)(st)
    return out, stats


def _shape(st):
    """``val.shape``; a fully reduced tensor folds to ``val=None`` with the
    value in ``scalar_mult`` (graphax's uniform form), i.e. shape ``()``."""
    return () if st.val is None else tuple(st.val.shape)


def _dropped_axis(before, after):
    """The one physical axis ``after`` lost against ``before``, by the dim
    pointers: every dim that pointed at it now points at None."""
    lost = None
    for d0 in _dims(before):
        d1 = _dim_by_id(after, d0.id)
        for attr in ("axis", "block_axis"):
            p0, p1 = getattr(d0, attr), getattr(d1, attr)
            if p0 is not None and p1 is None:
                assert lost in (None, p0), (lost, p0)
                lost = p0
    return lost


@pytest.mark.parametrize("per_face", [False, True])
@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_emitted_axis_names_the_axis_graphax_drops(name, per_face):
    set_per_face_masks(per_face)
    st = _FIXTURES[name]()
    sp = reduce_axis_spaces(st)
    L = slot_legality(st, N_AX)
    checked = applied_n = 0
    for a in range(sp.n_dims):
        rule = _emit(st, a)
        p = sp.phys_of_dim[a]
        # the emitted Compress.axes IS the physical axis of logical dim a
        assert rule.axes == (() if p is None else (p,)), (name, a, rule)
        out, stats = _apply_through_hook(st, rule)
        applied = stats.get("applied_compress", 0) == 1
        assert applied == bool(L.comp[a]), (name, a, stats, L.comp)
        legal = p is not None and st.val.shape[p] > 1
        assert applied == legal, (name, a, p, stats)
        if applied:
            # graphax dropped exactly p: val lost that axis and only the
            # dims stored there lost their pointer
            want = tuple(s for i, s in enumerate(st.val.shape) if i != p)
            assert _shape(out) == want, (name, a, p, _shape(out))
            assert _dropped_axis(st, out) == p, (name, a)
            for b in range(sp.n_dims):
                d1 = _dim_by_id(out, _dims(st)[b].id)
                assert (d1.axis is None) == (sp.phys_of_dim[b] in (p, None)), (
                    name, a, b)
            # and it is what apply_compress does with the converted action
            direct = apply_compress(st, compress_to_graphax(st, rule))
            assert _shape(direct) == _shape(out)
            assert np.allclose(np.asarray(direct.dense()),
                               np.asarray(out.dense()))
            applied_n += 1
        else:
            assert out is st, (name, a)
            if p is None:
                assert stats.get("skipped_compress_noop", 0) == int(per_face)
        checked += 1
    assert checked == sp.n_dims
    assert applied_n == sum(1 for p in sp.phys_of_dim
                            if p is not None and st.val.shape[p] > 1)


def test_the_finding_56_cases_are_closed():
    """The three failure kinds of D5b / D5c, each now the head's own dim."""
    # v81: k=1 was a legal no-op, k=2 masked-but-real, k=3 a raise.
    st = _v81()
    assert reduce_axis_mask(st, N_AX).tolist() == [True, True] + [False] * 6
    assert slot_legality(st, N_AX).comp.tolist() == (
        [True, False, True, False] + [False] * 4)
    assert _emit(st, 1).axes == () and _emit(st, 3).axes == ()
    out, stats = _apply_through_hook(st, _emit(st, 2))
    assert stats == {"applied": 1, "applied_compress": 1}
    assert tuple(out.val.shape) == (4,) and _dropped_axis(st, out) == 1
    # v92: the primal dims of the two pairs raised; they are the pairs' own
    # meta axes.
    st = _two_pairs()
    assert slot_legality(st, N_AX).comp.tolist() == [True] * 4 + [False] * 4
    for a, p in ((2, 0), (3, 1)):
        with pytest.raises(ValueError):
            apply_compress(st, Compress(axes=(a,)))          # the old read
        out, _ = _apply_through_hook(st, _emit(st, a))
        assert _dropped_axis(st, out) == p
    # shifted: the old read of token 2 (primal0, implicit) reduced primal1.
    st = _shifted()
    old = apply_compress(st, Compress(axes=(2,)))
    assert _dropped_axis(st, old) == 1 and _dim_by_id(old, 3).axis is None
    assert _emit(st, 2).axes == ()
    out, stats = _apply_through_hook(st, _emit(st, 2))
    assert out is st and stats.get("applied", 0) == 0
    out, _ = _apply_through_hook(st, _emit(st, 3))          # primal1 itself
    assert _dropped_axis(st, out) == 1


def test_a_block_axis_is_a_physical_axis_but_not_a_token():
    """The pair's block slot is addressable in PHYSICAL space (a rule may
    name it, and graphax reduces it) but no logical dim maps to it, so the
    head's axis vocabulary cannot request it; the partner's block axis has
    no slot at all and is refused before graphax sees it."""
    st = _block_pair()
    sp = reduce_axis_spaces(st)
    mask = reduce_axis_mask(st, N_AX)
    meta, blk = canonical_axis_order(st)
    assert mask[meta] and mask[blk]
    other = next(p for p in range(sp.n_phys) if p not in (meta, blk))
    assert not mask[other] and sp.slot_of_phys[other] is None
    assert blk not in sp.phys_of_dim
    out, stats = _apply_through_hook(st, Compress(axes=(blk,)))
    assert stats.get("applied_compress", 0) == 1
    assert _dropped_axis(st, out) == blk
    with pytest.raises(ValueError):
        compress_to_graphax(st, Compress(axes=(other,)))
    out, stats = _apply_through_hook(st, Compress(axes=(other,)))
    assert out is st and stats.get("skipped_raised", 0) == 0


# ==========================================================================
# 3. ALL THREE SLOTS, end to end, under --face-slot-frames slot
# ==========================================================================
#     e = A @ x        A: (6, 4)  x: (4,)   e: (6,)
#     h = tanh(e)                                 (vertex 2: lhs is a PAIR)
#     y = B @ h        B: (3, 6)             y: (3,)
#     return sum(y)
_A = jnp.asarray(np.linspace(0.1, 0.9, 6 * 4, dtype=np.float32).reshape(6, 4))
_B = jnp.asarray(np.linspace(0.2, 0.8, 3 * 6, dtype=np.float32).reshape(3, 6))
_X = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _chain(x, A, B):
    return jnp.sum(B @ jnp.tanh(A @ x))


_ARGS = (_X, _A, _B)


def _closed(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    return cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)


def _fresh_ij(closed, xs):
    return IncrementalJaxpr(closed.jaxpr, tuple(range(len(xs))),
                            list(closed.literals), list(xs))


@contextlib.contextmanager
def _in_trace(rec):
    from jax._src import core as _jcore
    with _jcore.set_current_trace(rec.trace):
        yield


_REC = {}


def _walk_to(closed, xs, upto_vertex):
    """Eliminate reverse order exactly down to (not including)
    ``upto_vertex``; record that vertex's faces' slot tensors."""
    total_v = len(closed.jaxpr.eqns)
    ij = _fresh_ij(closed, xs)
    for v in range(total_v, upto_vertex, -1):
        ij.eliminate(v, ())
    keys = list(ij.faces(upto_vertex))
    store = {}
    rec = _fresh_ij(closed, xs)
    for v in range(total_v, upto_vertex, -1):
        rec.eliminate(v, ())
    ft = {}
    for key in keys:
        st = store.setdefault(key, {})

        def mk(site, st=st):
            def h(t):
                st[site] = t
                return t
            return h
        ft[key] = ((mk("lhs"), mk("rhs"), mk("new")), (None, None, None))
    rec.eliminate(upto_vertex, (), face_transforms=ft)
    _REC[id(store)] = rec
    return ij, keys, store


def _slot_sweep(keys, store):
    checked = applied_n = pairs = 0
    for key in keys:
        by_site = store.get(key)
        if not by_site or "lhs" not in by_site:
            continue
        for site in SLOTS:
            st = by_site[site]
            sp = reduce_axis_spaces(st)
            L = slot_legality(st, N_AX)
            pairs += int(any(d.other_id is not None for d in _dims(st)))
            for a in range(N_AX):
                hook = make_slot_frame_hook((COMPRESS_SENTINEL, a, 0))
                rules = hook.rules_for(st)
                if a >= sp.n_dims:
                    assert rules == (), (site, a, rules)
                    assert not L.comp[a]
                    continue
                p = sp.phys_of_dim[a]
                assert rules == (Compress(axes=() if p is None else (p,),
                                          kind="mean"),), (site, a, rules)
                stats = {}
                out = M.make_live_masked_hook(rules, stats=stats)(st)
                applied = stats.get("applied_compress", 0) == 1
                assert applied == bool(L.comp[a]), (site, a, stats, L.comp)
                if applied:
                    want = tuple(s for i, s in enumerate(st.val.shape)
                                 if i != p)
                    assert _shape(out) == want, (site, a, p, _shape(out))
                    assert _dropped_axis(st, out) == p, (site, a, p)
                    applied_n += 1
                else:
                    assert out is st
                checked += 1
    return checked, applied_n, pairs


@pytest.mark.parametrize("per_face", [False, True])
@pytest.mark.parametrize("vertex", [1, 2])
def test_all_three_slots_end_to_end(vertex, per_face):
    """Per slot and per token of every face of ``vertex``: the slot-frame
    hook emits the physical axis of the token's dim, the per-slot mask is
    the hook's verdict, and an applied Reduce drops exactly that axis.
    Vertex 2 (tanh) has a coupled-pair lhs, where the pre-ticket read of the
    primal token raised."""
    set_per_face_masks(per_face)
    closed = _closed(_chain, _ARGS)
    _ij, keys, store = _walk_to(closed, _ARGS, vertex)
    with _in_trace(_REC[id(store)]):
        checked, applied_n, pairs = _slot_sweep(keys, store)
    assert checked > 0 and applied_n > 0, (checked, applied_n)
    if vertex == 2:
        assert pairs > 0, "the tanh lhs is a coupled pair"


def test_the_pair_lhs_reduces_through_the_face_dict():
    """Through ``_face_dict_for_vertex`` and a real elimination at the tanh
    vertex: the primal token of its diagonal lhs is APPLIED under physical
    (the pair's shared meta axis), and was skipped -- the old read raised,
    or masked it -- under canonical."""
    closed = _closed(_chain, _ARGS)
    config = SimpleNamespace(jaxpr=closed.jaxpr)

    def run(space, row):
        set_reduce_axis_space(space)
        ij, keys, store = _walk_to(closed, _ARGS, 2)
        key = next(k for k in keys
                   if any(d.other_id is not None
                          for d in _dims(store[k]["lhs"])))
        f = keys.index(key)
        rows = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
        rows[:, :, 2] = 0
        rows[f, 0] = row
        skips = np.zeros((MAX_FACES,), np.int32)
        envmod._PER_FACE_STATS.clear()
        per_face = _face_dict_for_vertex(config, ij, 2, rows, skips)
        assert key in per_face
        arm_face_counts()
        try:
            ij.eliminate(2, (), per_face)
        finally:
            disarm_face_counts()
        out = dict(envmod._PER_FACE_STATS)
        envmod._PER_FACE_STATS.clear()
        return out

    primal_token = (COMPRESS_SENTINEL, 1, 0)
    st = run("physical", primal_token)
    assert st.get("applied_compress", 0) == 1, st
    st = run("canonical", primal_token)
    assert st.get("applied_compress", 0) == 0, st
    assert st.get("skipped_compress", 0) == 1, st
    # the out token names the same storage; both spaces apply it
    for space in ("physical", "canonical"):
        st = run(space, (COMPRESS_SENTINEL, 0, 0))
        assert st.get("applied_compress", 0) == 1, (space, st)


# ==========================================================================
# 4. FLAG OFF -- the pre-ticket path
# ==========================================================================
@pytest.mark.parametrize("per_face", [False, True])
@pytest.mark.parametrize("name", sorted(_FIXTURES))
def test_canonical_is_the_pre_ticket_path(name, per_face):
    """Every predicate reads its historical mask, and the hook hands the
    integer to graphax unconverted."""
    set_reduce_axis_space("canonical")
    set_per_face_masks(per_face)
    st = _FIXTURES[name]()
    valid = M.compress_valid_mask(st, N_AX)
    slot = M.compress_slot_mask(st, N_AX)
    for a in range(N_AX):
        rule = Compress(axes=(a,), kind="mean")
        assert M.rule_is_legal(st, rule, max_axes=N_AX) == bool(valid[a])
        assert M.face_rule_is_legal(st, rule, max_axes=N_AX) == bool(
            valid[a] and slot[a])
        assert M.compress_is_noop(st, rule, max_axes=N_AX) == (not slot[a])
        assert M.hook_rule_is_legal(st, rule, max_axes=N_AX) == bool(
            valid[a] and (slot[a] if per_face else True))
    assert [int(r.axes[0]) for r in M.legal_compress_actions(st, N_AX)] == [
        a for a in range(N_AX) if valid[a]]
    assert [int(r.axes[0]) for r in
            M.legal_compress_actions(st, N_AX, strict=True)] == [
        a for a in range(N_AX) if slot[a]]
    L = slot_legality(st, N_AX)
    n_dims = reduce_axis_spaces(st).n_dims
    for a in range(N_AX):
        assert bool(L.comp[a]) == (a < n_dims and M.hook_rule_is_legal(
            st, Compress(axes=(a,), kind="mean"), max_dims=N_AX,
            max_axes=N_AX))
    # the hook applies the integer as graphax reads it: a canonical slot
    hook = make_slot_frame_hook((COMPRESS_SENTINEL, 2, 0))
    if reduce_axis_spaces(st).n_dims > 2:
        assert hook.rules_for(st) == (Compress(axes=(2,), kind="mean"),)
    if name == "shifted":
        # the D5 wrong-dim application, reachable: the head's token 2 is
        # primal0 (implicit); the old hook reduces primal1's axis instead
        stats = {}
        out = M.make_live_masked_hook((Compress(axes=(2,)),),
                                      stats=stats)(st)
        assert stats.get("applied_compress", 0) == 1, stats
        assert _dropped_axis(st, out) == 1 and _dim_by_id(out, 3).axis is None


def test_canonical_never_touches_the_conversion(monkeypatch):
    set_reduce_axis_space("canonical")
    calls = []
    monkeypatch.setattr(M, "reduce_axis_spaces",
                        lambda st: calls.append(st) or (_ for _ in ()).throw(
                            AssertionError("converted under canonical")))
    st = _two_pairs()
    for per_face in (False, True):
        set_per_face_masks(per_face)
        slot_legality(st, N_AX)
        M.make_live_masked_hook((Compress(axes=(0,)),), stats={})(st)
        make_slot_frame_hook((COMPRESS_SENTINEL, 0, 0)).rules_for(st)
    assert calls == []
