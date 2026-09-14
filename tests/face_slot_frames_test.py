"""``--face-slot-frames`` (ticket dsnn-3qm.18, defects D2 and D3).

D2. ``env._face_dict_for_vertex`` decoded each of a face's three slot rows
with ``decode_vertex_rule_specs(jaxpr, v, row)``, whose frame
``(out_len, primal_shapes)`` is the eliminated vertex's OWN equation. That
frame describes the ``lhs`` tensor (d central / d in_edge) and nothing else:
``rhs`` (d out_edge / d central) and ``new`` (d out_edge / d in_edge) carry a
different out rank and different primal dims (finding 54: under a scalar loss
their out side is EMPTY on every face), so ``Diag.j = out_len + bi2`` and the
Reduce axis range were resolved in the wrong frame for two of three slots.
Now each slot's hook decodes its row in the frame of the tensor it is handed.

D3. ``UnifiedFacePolicy._face_masks`` broadcast one legality vector to all
three slots, and ``LiveFaceStream.face_dim_sizes`` probed only the result
tensor. Now the probe records lhs / rhs / new and the head masks each slot
with that slot's own sizes, Diag-pair, Reduce-axis and Quant legality.

What is pinned:

  * the decoded Diag.j and Reduce axis per slot are the slot tensor's own
    (out_dims ++ primal_dims) positions, on a graph whose three slots have
    three different shapes;
  * per-slot legality equals the apply-time hook's verdict, slot by slot
    (the seam tickets .59 / .40 build on);
  * masked == pruned: log-prob, entropy and the sampled distribution of the
    masked 94-logit head equal those of a head over the legal choices only,
    per slot, with a slot-dependent legal set;
  * sampling == replay under per-slot masks (the PPO ratio at epoch 0);
  * flag off (``vertex``) is the pre-ticket code path, byte for byte.
"""
import contextlib
import os
from collections import Counter
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from graphax import inline_call_primitives                      # noqa: E402
from graphax.incremental import IncrementalJaxpr                # noqa: E402
from graphax.sparse.micro_actions import (                      # noqa: E402
    Compress, Diag, QUANT_DTYPES)

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.common import masks as M                  # noqa: E402
from alphagrad.approx.common.masks import (                     # noqa: E402
    FACE_QUANT_DTYPES, arm_face_counts, disarm_face_counts,
    slot_legality)
from alphagrad.approx.env import (                              # noqa: E402
    COMPRESS_SENTINEL, FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX,
    _face_dict_for_vertex, decode_rule_specs_in_frame,
    decode_vertex_rule_specs, make_slot_frame_hook, slot_frame,
    vertex_frame)
from alphagrad.approx.live_faces import LiveFaceStream          # noqa: E402
from alphagrad.approx.common.masks import NUM_FACE_QUANT_DTYPES

try:
    from jax.extend.core import ClosedJaxpr
except ImportError:                                        # pragma: no cover
    from jax._src.core import ClosedJaxpr

N_AX = 8
MAX_F = 8
SLOTS = ("lhs", "rhs", "new")


# --------------------------------------------------------------------------
# The fixture. A scalar loss, so every rhs / new tensor has an EMPTY out side
# (finding 54), while lhs keeps the vertex's own out dims:
#
#     e = A @ x        A: (6, 4)  x: (4,)   e: (6,)
#     h = tanh(e)
#     y = B @ h        B: (3, 6)             y: (3,)
#     return sum(y)
#
# At vertex 1 (A @ x, eliminated LAST under reverse order) the x-face's slots
# are lhs = d e/d x: out (6,) primal (4,); rhs = d loss/d e: out () primal
# (6,); new = d loss/d x: out () primal (4,). Three slots, three shapes.
# --------------------------------------------------------------------------
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
    """The recorded tensors are tracers of ``rec``'s persistent trace; a rule
    applied to one must run inside it (as ``per_face_masks_test`` does)."""
    from jax._src import core as _jcore
    with _jcore.set_current_trace(rec.trace):
        yield


_REC = {}


def _walk_to(closed, xs, upto_vertex):
    """Eliminate reverse order EXACTLY down to (not including) ``upto_vertex``;
    return the ij and the recorded ``{face key: {site: tensor}}`` of that
    vertex's faces (recorded on a throwaway copy of the elimination, kept in
    ``_REC[id(store)]`` so :func:`_in_trace` can reopen its trace)."""
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


def _x_face(closed, xs, v=1):
    """The face of vertex ``v`` whose in-edge is the graph input ``x``."""
    ij, keys, store = _walk_to(closed, xs, v)
    for key in keys:
        st = store.get(key, {})
        if "lhs" in st and len(st["lhs"].primal_dims) == 1:
            return ij, keys, key, st
    raise AssertionError("no x-face recorded")            # pragma: no cover


def _blank_rows():
    rows = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    rows[:, :, 2] = 0
    return rows, np.zeros((MAX_FACES,), np.int32)


def _pad(row):
    return [list(row)] + [[-1, -1, 0]] * (MAX_RULES_PER_VERTEX - 1)


@pytest.fixture(autouse=True)
def _counters_disarmed():
    """Per-slot frames are the ONLY behaviour, so there is nothing to set."""
    try:
        yield
    finally:
        disarm_face_counts()


# ==========================================================================
# 0. THE RETIRED SWITCH
# ==========================================================================
def test_the_retired_env_var_is_refused_not_ignored():
    """A launcher that still sets ALPHAGRAD_FACE_SLOT_FRAMES believes it chose
    a decode frame. The variable is gone, so importing masks with it set must
    FAIL rather than silently run the per-slot path the launcher asked to
    leave. Checked in a subprocess because the guard runs at import."""
    import subprocess
    import sys
    env = dict(os.environ)
    env["ALPHAGRAD_FACE_SLOT_FRAMES"] = "0"
    r = subprocess.run(
        [sys.executable, "-c", "import alphagrad.approx.common.masks"],
        env=env, capture_output=True, text=True)
    assert r.returncode != 0, r.stdout
    assert "ALPHAGRAD_FACE_SLOT_FRAMES is retired" in r.stderr, r.stderr


# ==========================================================================
# 1. D2 -- the decode frame is the SLOT's own tensor
# ==========================================================================
def test_the_three_slots_have_three_shapes():
    """The fixture control: if the slots agreed, every assertion below
    would hold in the vertex frame too and prove nothing."""
    closed = _closed(_chain, _ARGS)
    _ij, _keys, _key, st = _x_face(closed, _ARGS)
    frames = {s: slot_frame(st[s]) for s in SLOTS}
    assert frames["lhs"] == ((6,), [(4,)]), frames
    assert frames["rhs"] == ((), [(6,)]), frames
    assert frames["new"] == ((), [(4,)]), frames
    vf = vertex_frame(closed.jaxpr, 1)
    assert vf[0] == (6,), vf
    assert frames["lhs"] != vf          # lhs primal = x only; vertex = A, x
    assert all(f != vf for f in (frames["rhs"], frames["new"]))


def test_diag_j_is_decoded_in_each_slots_frame():
    """``Diag.j = out_len + bi2`` with the SLOT's out_len, not the vertex's.

    On lhs (out (6,), primal (4,)) the wire pair (0, 0) is Diag(i=0, j=1)
    with the joint gcd 2. On rhs and new there is no out side at all, so the
    same row names no pair and must decode to NOTHING -- where the vertex
    frame decoded it to Diag(0, 1, 2) for all three slots and left it to the
    apply-time mask to refuse.
    """
    closed = _closed(_chain, _ARGS)
    _ij, _keys, _key, st = _x_face(closed, _ARGS)
    row = (0, 0, -1)
    hook = make_slot_frame_hook(row)
    assert hook.rules_for(st["lhs"]) == (Diag(i=0, j=1, factor=2),)
    assert hook.rules_for(st["rhs"]) == ()
    assert hook.rules_for(st["new"]) == ()
    # The vertex frame (the pre-ticket decode) admits it for every slot.
    old = decode_vertex_rule_specs(closed.jaxpr, 1, _pad(row))
    assert old == (Diag(i=0, j=1, factor=2),)
    # The head writes an EXPLICIT factor (gcd of the slot's own sizes); the
    # slot frame checks it against the slot's own dims.
    assert make_slot_frame_hook((0, 0, 2)).rules_for(st["lhs"]) == (
        Diag(i=0, j=1, factor=2),)
    assert make_slot_frame_hook((0, 0, 3)).rules_for(st["lhs"]) == ()


def test_reduce_axis_is_decoded_in_each_slots_frame():
    """A Reduce (Compress) axis is a logical position of THIS slot's tensor.

    Axis 1 exists on lhs (its primal dim, size 4) and on no other slot.
    Axis 0 exists on every slot but names a DIFFERENT dim on each: the out
    dim (6) on lhs, the central primal dim (6) on rhs, the in-edge primal
    dim (4) on new. The vertex frame admitted axis 1 for every slot.
    """
    closed = _closed(_chain, _ARGS)
    _ij, _keys, _key, st = _x_face(closed, _ARGS)
    ax1 = make_slot_frame_hook((COMPRESS_SENTINEL, 1, 0))
    assert ax1.rules_for(st["lhs"]) == (Compress(axes=(1,), kind="mean"),)
    assert ax1.rules_for(st["rhs"]) == ()
    assert ax1.rules_for(st["new"]) == ()
    old = decode_vertex_rule_specs(closed.jaxpr, 1,
                                   _pad((COMPRESS_SENTINEL, 1, 0)))
    assert old == (Compress(axes=(1,), kind="mean"),)

    ax0 = make_slot_frame_hook((COMPRESS_SENTINEL, 0, 0))
    for s in SLOTS:
        assert ax0.rules_for(st[s]) == (Compress(axes=(0,), kind="mean"),)
    sizes = {s: M.dim_logical_sizes(st[s], N_AX)[0] for s in SLOTS}
    assert (sizes["lhs"], sizes["rhs"], sizes["new"]) == (6, 6, 4)


def test_frame_decode_is_the_vertex_decode_when_the_frames_agree():
    """``decode_vertex_rule_specs`` is ``decode_rule_specs_in_frame`` on the
    vertex's own frame -- one decoder, two frames -- so the per-vertex path
    and the lhs slot (whose tensor IS the vertex frame) cannot drift."""
    closed = _closed(_chain, _ARGS)
    rows = [(0, 0, -1), (COMPRESS_SENTINEL, 1, 0), (COMPRESS_SENTINEL, 0, 2),
            (envmod.QUANT_SENTINEL, 1, 0), (0, 0, 2), (0, 0, 4)]
    for v in range(1, len(closed.jaxpr.eqns) + 1):
        eqn = closed.jaxpr.eqns[v - 1]
        out_shape = eqn.outvars[0].aval.shape
        prim = [iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")]
        for row in rows:
            assert (decode_vertex_rule_specs(closed.jaxpr, v, _pad(row))
                    == decode_rule_specs_in_frame(out_shape, prim,
                                                  _pad(row)))


def test_face_dict_applies_in_the_slot_frame_end_to_end():
    """Through ``_face_dict_for_vertex`` and a real elimination.

    One slot at a time: lhs = Reduce axis 1 (its primal dim, size 4), rhs =
    Reduce axis 0, new = Reduce axis 0. Each is legal in its OWN frame and
    applies. (Not all three in one run: ``new`` is the product of the two
    operands, and once lhs and rhs have each lost an axis the contraction
    result is a scalar-``val`` tensor on which a Reduce is a no-op -- the
    slots interact through the contraction, which is exactly why legality
    is decided on the live tensor at apply time.)

    The row that separates the two frames is the lhs axis-1 row on rhs: the
    vertex frame decodes a Compress the mask must then refuse (counted
    ``skipped_compress``), the slot frame decodes nothing (also counted
    ``skipped_compress`` -- a requested rule that names no dim of the slot is
    a miss, not silence).
    """
    closed = _closed(_chain, _ARGS)
    config = SimpleNamespace(jaxpr=closed.jaxpr)

    def run(rows_by_slot):
        ij, keys, key, _st = _x_face(closed, _ARGS)
        f = keys.index(key)
        rows, skips = _blank_rows()
        for s, r in rows_by_slot.items():
            rows[f, SLOTS.index(s)] = r
        envmod._PER_FACE_STATS.clear()
        per_face = _face_dict_for_vertex(config, ij, 1, rows, skips)
        assert key in per_face
        arm_face_counts()
        try:
            ij.eliminate(1, (), per_face)
        finally:
            disarm_face_counts()
        out = dict(envmod._PER_FACE_STATS)
        envmod._PER_FACE_STATS.clear()
        return out

    legal = {"lhs": (COMPRESS_SENTINEL, 1, 0), "rhs": (COMPRESS_SENTINEL, 0, 0),
             "new": (COMPRESS_SENTINEL, 0, 0)}
    for site, row in legal.items():
        st = run({site: row})
        assert st.get("applied_compress", 0) == 1, (site, st)
        assert st.get("skipped_compress", 0) == 0, (site, st)

    # Axis 1 exists on lhs only. Requested on rhs it names no dim of that
    # slot, so it is a COUNTED miss, not silence.
    st = run({"rhs": (COMPRESS_SENTINEL, 1, 0)})
    assert st.get("applied_compress", 0) == 0, st
    assert st.get("skipped_compress", 0) == 1, st


def test_two_op_form_keeps_the_new_slot_hook_OFF_the_old_edge():
    """CHANGED contract (ticket .56, finding 73).

    The entry is still graphax's TWO-OP form -- ``new`` hooks the fresh
    contraction, BEFORE any join -- but the ``jr`` position no longer holds the
    slot's hook. Under ``--approx-add lossy`` it holds a join POLICY, which is
    handed both addends and makes them share one container; under ``lossless``
    it holds nothing. Either way the slot's wire row is decoded in exactly ONE
    frame, which is the frame its legality mask was computed in.

    The retired ``--approx-old same`` put the SAME hook object in ``jr`` too,
    so one row decoded in two different frames and the mask answered for one of
    them (finding 72, fault 1).
    """
    from graphax.sparse.ops.join import FaceJoinPolicy
    closed = _closed(_chain, _ARGS)
    config = SimpleNamespace(jaxpr=closed.jaxpr)
    ij, keys, key, _st = _x_face(closed, _ARGS)
    rows, skips = _blank_rows()
    rows[keys.index(key), 2] = (COMPRESS_SENTINEL, 0, 0)
    per_face = _face_dict_for_vertex(config, ij, 1, rows, skips)
    entry = per_face[key]
    assert len(entry) == 2 and len(entry[0]) == 3 and len(entry[1]) == 3
    assert entry[0][2] is not None
    assert entry[0][2] is not entry[1][1], \
        "the old edge must not carry the new slot's hook any more"
    assert entry[1][1] is None or isinstance(entry[1][1], FaceJoinPolicy)
    assert entry[1][0] is None and entry[1][2] is None
    assert entry[0][0] is None and entry[0][1] is None
    # the mask's site list agrees: ONE site for the `new` slot
    import alphagrad.approx.env as _e
    assert _e.face_slot_sites()[2] == ("res:new",), _e.face_slot_sites()


# ==========================================================================
# 2. D3 -- per-slot legality FROM EACH SLOT'S LIVE TENSOR
# ==========================================================================
def _exact_prefix(total_v, k):
    order = np.zeros((total_v,), np.int32)
    rev = list(range(total_v, 0, -1))
    order[:k] = np.asarray(rev[:k], np.int32)
    specs = -np.ones((total_v, 3, 3), np.int32)
    return order, specs, k


def _stream(closed, xs):
    return LiveFaceStream(closed.jaxpr, tuple(range(len(xs))),
                          list(closed.literals), list(xs),
                          vocab=256, max_faces=MAX_F, max_axes=N_AX)


def test_slot_legality_is_per_slot_and_matches_the_recorded_tensors():
    """``face_slot_legality`` returns, per face and per slot, exactly what
    ``slot_legality`` computes on the tensor graphax hands that slot -- and
    the three slots DIFFER, where ``face_dim_sizes`` returned one vector."""
    closed = _closed(_chain, _ARGS)
    lf = _stream(closed, _ARGS)
    total_v = len(closed.jaxpr.eqns)
    order, specs, n = _exact_prefix(total_v, total_v - 1)   # everything but 1
    sizes, quant, pair, comp, nout, nf = lf.face_slot_legality(
        order, specs, n, 1)
    # ONE ROW PER SLOT THE ENTRY BUILDER HOOKS, which is as many as
    # --approx-add has: three under the default lossless, four under learned1
    # (+ the old edge), five under learned2 (+ the summed edge). The width comes
    # from `env.face_slot_sites()` so the mask cannot go stale behind the
    # topology, and the contraction slots are its PREFIX -- which is what lets
    # the trainer path narrow to them.
    from alphagrad.approx.env import face_slot_sites as _sites
    S_ALL = len(_sites())
    assert tuple(x[0] for x in _sites()[:FACE_SLOTS]) == (
        "lhs", "rhs", "res:new"), _sites()
    assert sizes.shape == (MAX_F, S_ALL, N_AX)
    assert quant.shape == (MAX_F, S_ALL, NUM_FACE_QUANT_DTYPES)
    assert pair.shape == (MAX_F, S_ALL, N_AX, N_AX)
    assert comp.shape == (MAX_F, S_ALL, N_AX)
    assert nout.shape == (MAX_F, S_ALL)
    assert int(nf) >= 1

    _ij, keys, store = _walk_to(closed, _ARGS, 1)
    old_sizes, old_quant, old_n = lf.face_dim_sizes(order, specs, n, 1)
    assert int(old_n) == int(nf)
    differing = 0
    for f, key in enumerate(keys[:int(nf)]):
        st = store[key]
        for s, site in enumerate(SLOTS):
            want = slot_legality(st[site], N_AX)
            assert np.array_equal(sizes[f, s], want.sizes), (f, site)
            assert int(nout[f, s]) == want.n_out, (f, site)
            assert np.array_equal(pair[f, s] > 0.5, want.pair), (f, site)
            assert np.array_equal(comp[f, s] > 0.5, want.comp), (f, site)
            assert np.array_equal(quant[f, s] > 0.5, want.quant), (f, site)
        if not (np.array_equal(sizes[f, 0], sizes[f, 1])
                and np.array_equal(sizes[f, 1], sizes[f, 2])):
            differing += 1
        # The pre-ticket vector is the res-site one, broadcast to all
        # three; on a merge-free face res == new.
        assert np.array_equal(old_sizes[f], sizes[f, 2]), f
    assert differing > 0, "every face's slots agree -- fixture cannot tell"


def test_slot_legality_equals_the_hooks_verdict():
    """THE SEAM for .59 / .40: an action the mask admits on a slot is
    applied by that slot's hook, and an action it refuses is not.

    Enumerated over every Diag pair, every Reduce axis and both Quant dtypes
    on every slot of every face of vertex 1, with the head's own factor rule
    (``factor = gcd(N_i, N_j)``) on the Diag wire.
    """
    closed = _closed(_chain, _ARGS)
    _ij, keys, store = _walk_to(closed, _ARGS, 1)
    checked = admitted = 0
    with contextlib.ExitStack() as stack:
        stack.enter_context(_in_trace(_REC[id(store)]))
        checked, admitted = _verdict_sweep(keys, store)
    assert checked > 0 and admitted > 0, (checked, admitted)


def _verdict_sweep(keys, store):
    checked = admitted = 0
    for key in keys:
        st_by_site = store.get(key)
        if not st_by_site or "lhs" not in st_by_site:
            continue
        for site in SLOTS:
            st = st_by_site[site]
            L = slot_legality(st, N_AX)
            n_out = L.n_out
            for i in range(N_AX):
                for j in range(N_AX):
                    if i == j:
                        continue
                    io, jp = (i, j) if i < n_out else (j, i)
                    g = int(np.gcd(max(int(L.sizes[i]), 1),
                                   max(int(L.sizes[j]), 1)))
                    hook = make_slot_frame_hook((io, jp - n_out, g),
                                                stats=None)
                    stats = {}
                    M_ = M.make_live_masked_hook(hook.rules_for(st),
                                                 stats=stats)
                    M_(st)
                    applied = stats.get("applied", 0) == 1
                    assert applied == bool(L.pair[i, j]), (
                        site, i, j, L.pair[i, j], stats)
                    checked += 1
                    admitted += int(applied)
            for a in range(N_AX):
                hook = make_slot_frame_hook((COMPRESS_SENTINEL, a, 0))
                stats = {}
                M.make_live_masked_hook(hook.rules_for(st), stats=stats)(st)
                applied = stats.get("applied", 0) == 1
                assert applied == bool(L.comp[a]), (site, a, stats)
                checked += 1
                admitted += int(applied)
            for d, name in enumerate(FACE_QUANT_DTYPES):
                hook = make_slot_frame_hook(
                    (envmod.QUANT_SENTINEL, list(QUANT_DTYPES).index(name), 0))
                stats = {}
                out = M.make_live_masked_hook(hook.rules_for(st),
                                              stats=stats)(st)
                real = (stats.get("applied", 0) == 1
                        and not M.quant_is_noop(st, name))
                assert real == bool(L.quant[d]), (site, name, stats)
                del out
                checked += 1
    return checked, admitted


def test_the_slot_probe_is_a_pure_read_and_memoized():
    closed = _closed(_chain, _ARGS)
    lf = _stream(closed, _ARGS)
    total_v = len(closed.jaxpr.eqns)
    order, specs, n = _exact_prefix(total_v, 0)
    rows = -np.ones((MAX_F, 3, 3), np.int32)
    skips = np.zeros((MAX_F,), np.int32)
    vspecs = -np.ones((3, 3), np.int32)
    before = lf.chunk(order, specs, n, total_v, vspecs, rows, skips, 0)
    lf._chunks.clear()
    lf.consume_stats()
    for _ in range(4):
        lf.face_slot_legality(order, specs, n, total_v)
    st = lf.consume_stats()
    # ONE probe: the single engine has one dispatch mode (dsnn-3qm.65).
    assert st["slot_probe"] == 1, st
    assert st["slot_hit"] == 3, st
    after = lf.chunk(order, specs, n, total_v, vspecs, rows, skips, 0)
    for a, b in zip(before, after):
        assert np.array_equal(np.asarray(a), np.asarray(b))


# ==========================================================================
# 3. MASKED == PRUNED, per slot, on the 94-logit head
# ==========================================================================
def _policy(E=32, F=4):
    from alphagrad.approx.heads import precompute_factor_tables
    from alphagrad.approx.unified_face_policy import UnifiedFacePolicy
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(E, num_heads=2, max_faces=F, key=jrand.PRNGKey(0))
    return pol, tables


def _features(n=N_AX):
    from alphagrad.approx.heads import AXIS_TAG_BITS, AxisTokenFeatures
    sz = jnp.asarray([6, 4, 6, 4, 2, 2, 1, 1][:n], jnp.int32)
    return AxisTokenFeatures(
        size=sz, log_size=jnp.log(sz.astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def _slot_inputs():
    """Per-slot inputs shaped like ``face_slot_legality`` returns for ONE
    face: lhs has a Diag pair and two Reduce axes, rhs one axis and no
    pair, new one axis, no pair and no legal Quant."""
    S, N = FACE_SLOTS, N_AX
    sizes = np.zeros((S, N), np.int32)
    sizes[0, :2] = (6, 4)
    sizes[1, :1] = (6,)
    sizes[2, :1] = (4,)
    pair = np.zeros((S, N, N), np.float32)
    pair[0, 0, 1] = pair[0, 1, 0] = 1.0
    comp = np.zeros((S, N), np.float32)
    comp[0, :2] = 1.0
    comp[1, 0] = 1.0
    comp[2, 0] = 1.0
    quant = np.zeros((S, NUM_FACE_QUANT_DTYPES), np.float32)
    quant[0, 1] = 1.0
    quant[1, 1] = 1.0
    return (jnp.asarray(sizes), jnp.asarray(quant), jnp.asarray(pair),
            jnp.asarray(comp))


def _pruned_cat(logits, legal, idx):
    """Reference: a categorical over the LEGAL entries only."""
    z = np.asarray(logits, np.float64)[legal]
    lp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
    p = np.exp(lp)
    pos = int(np.flatnonzero(legal).tolist().index(int(idx)))
    return float(lp[pos]), float(-(p * lp).sum()), p


def test_masked_head_equals_pruned_head_per_slot():
    """Log-prob and entropy of the masked head == those of a head built over
    the legal choices only, slot by slot, for a slot-dependent legal set.

    The reference re-derives every categorical from the raw logits ``z`` and
    the per-slot legal sets ``face_slot_legality`` would supply; the head's
    branch rule (only the fields the chosen op consumes count) is applied on
    both sides.
    """
    from alphagrad.approx.unified_face_head import (
        NUM_REDUCE_FNS, OP_BLOCKDIAG, OP_NONE, OP_QUANT, OP_REDUCE, S_AXIS,
        SLOT_WIDTH, S_DTYPE, S_I, S_J, S_OP, S_RFN, j_mask_given_i, slot_base)
    from alphagrad.approx.unified_micro import face_dtype_idx_of, _KIND_MAP
    kinds = np.asarray(_KIND_MAP)
    # The reference inverts fn -> compress_kind, as evaluate_face does.
    assert len(set(kinds.tolist())) == len(kinds), kinds

    pol, tables = _policy()
    feats = _features()
    sizes, quant, pair, comp = _slot_inputs()
    ctx = jnp.asarray(np.linspace(-1, 1, pol.embd_dim, dtype=np.float32))
    om, im, jm, am, pair_ok, dm = pol._face_masks(
        [pol._face_feats_1(feats, sizes[s]) for s in range(FACE_SLOTS)],
        pair, comp, quant, None, tables)
    # Slot-dependent legal sets: what D3 is about.
    assert not np.array_equal(np.asarray(om[0]), np.asarray(om[1]))
    assert float(om[1][OP_BLOCKDIAG]) == 0.0 and float(om[0][OP_BLOCKDIAG]) == 1.0
    assert float(om[2][OP_QUANT]) == 0.0 and float(om[1][OP_QUANT]) == 1.0
    for s in range(FACE_SLOTS):      # axis head is NUM_REDUCE_AXES (9) wide
        assert np.array_equal(np.asarray(am[s] > 0.5)[:N_AX],
                              np.asarray(comp[s] > 0.5)), s
        assert not bool(am[s][N_AX:].any())

    z = pol.head.logits(ctx)
    zn = np.asarray(z, np.float64)
    checked = 0
    for k in range(24):
        (skip, row, lp, ent, _ar, _sp, _od) = pol.sample_face(
            feats, tables, jrand.PRNGKey(100 + k), 0,
            pair, comp, jnp.asarray(1.0), face_context=ctx,
            face_sizes_f=sizes, face_quant_f=quant)
        lp2, ent2 = pol.evaluate_face(
            feats, tables, _as_face_action(row, skip), 0, pair, comp,
            jnp.asarray(1.0), face_context=ctx, face_sizes_f=sizes,
            face_quant_f=quant)[:2]
        assert abs(float(lp) - float(lp2)) < 1e-5
        assert abs(float(ent) - float(ent2)) < 1e-5
        # The reference.
        p_skip = 1.0 / (1.0 + np.exp(-zn[0]))
        ref_lp = np.log(p_skip if int(skip) else 1.0 - p_skip)
        ref_e = -(p_skip * np.log(p_skip) + (1 - p_skip) * np.log(1 - p_skip))
        if int(skip) == 0:
            for s in range(FACE_SLOTS):
                b = slot_base(s)
                op = int(row["op_type"][s])
                legal_op = np.asarray(om[s]) > 0.5
                assert legal_op[op], (s, op)
                l, e, _ = _pruned_cat(zn[b + S_OP:b + S_I], legal_op, op)
                ref_lp += l
                ref_e += e
                if op == OP_BLOCKDIAG:
                    i, j = int(row["i"][s]), int(row["j"][s])
                    legal_i = np.asarray(im[s]) > 0.5
                    assert legal_i[i]
                    l, e, _ = _pruned_cat(zn[b + S_I:b + S_J], legal_i, i)
                    ref_lp += l
                    ref_e += e
                    jmask = np.asarray(j_mask_given_i(i, jm[s], pair_ok[s]))
                    legal_j = jmask > 0.5
                    assert legal_j[j] and float(pair[s, i, j]) == 1.0
                    l, e, _ = _pruned_cat(zn[b + S_J:b + S_AXIS], legal_j, j)
                    ref_lp += l
                    ref_e += e
                elif op == OP_REDUCE:
                    a = int(row["i"][s])
                    legal_a = np.asarray(am[s]) > 0.5
                    assert legal_a[a] and float(comp[s, a]) == 1.0
                    l, e, _ = _pruned_cat(zn[b + S_AXIS:b + S_RFN], legal_a, a)
                    ref_lp += l
                    ref_e += e
                    fidx = int(np.flatnonzero(
                        kinds == int(row["compress_kind"][s]))[0])
                    l, e, _ = _pruned_cat(zn[b + S_RFN:b + S_DTYPE],
                                          np.ones(NUM_REDUCE_FNS, bool), fidx)
                    ref_lp += l
                    ref_e += e
                elif op == OP_QUANT:
                    legal_dt = np.asarray(dm[s]) > 0.5
                    assert legal_dt.any()
                    dti = int(face_dtype_idx_of(int(row["quant_dtype"][s])))
                    l, e, _ = _pruned_cat(zn[b + S_DTYPE:b + SLOT_WIDTH],
                                          legal_dt, dti)
                    ref_lp += l
                    ref_e += e
                else:
                    assert op == OP_NONE
        assert abs(float(lp) - ref_lp) < 2e-4, (k, float(lp), ref_lp)
        assert abs(float(ent) - ref_e) < 2e-4, (k, float(ent), ref_e)
        checked += 1
    assert checked == 24


def _as_face_action(row, skip, F=4):
    from alphagrad.approx.face_action import FaceAction

    def _pad(v):
        v = jnp.asarray(v)
        z = jnp.zeros((F,) + tuple(v.shape), v.dtype)
        return z.at[0].set(v)

    return FaceAction(
        skip=jnp.zeros((F,), jnp.int32).at[0].set(jnp.asarray(skip)),
        op_type=_pad(row["op_type"]), i=_pad(row["i"]), j=_pad(row["j"]),
        exponents=_pad(row["exponents"]), factor=_pad(row["factor"]),
        compress_kind=_pad(row["compress_kind"]),
        quant_dtype=_pad(row["quant_dtype"]),
        quant_scale_sign=_pad(row["quant_scale_sign"]),
        quant_scale_frac=_pad(row["quant_scale_frac"]))


def test_sampled_distribution_equals_the_pruned_distribution_per_slot():
    """Frequencies of the masked draw match the pruned softmax, per slot,
    and an illegal choice is never drawn.

    Uses the head's own ``_sample_cat`` on each slot's op / axis logits so
    the check is on the sampler, not on a re-implementation of it.
    """
    from alphagrad.approx.unified_face_head import (
        S_AXIS, S_I, S_OP, S_RFN, _sample_cat, slot_base)

    pol, tables = _policy()
    feats = _features()
    sizes, quant, pair, comp = _slot_inputs()
    ctx = jnp.asarray(np.linspace(-1, 1, pol.embd_dim, dtype=np.float32))
    om, _im, _jm, am, _pk, _dm = pol._face_masks(
        [pol._face_feats_1(feats, sizes[s]) for s in range(FACE_SLOTS)],
        pair, comp, quant, None, tables)
    z = pol.head.logits(ctx)
    zn = np.asarray(z, np.float64)
    K = 6000
    keys = jrand.split(jrand.PRNGKey(7), K)
    for s in range(FACE_SLOTS):
        b = slot_base(s)
        for lo, hi, mask in ((b + S_OP, b + S_I, om[s]),
                             (b + S_AXIS, b + S_RFN, am[s])):
            draws = jax.vmap(lambda k: _sample_cat(z[lo:hi], mask, k))(keys)
            legal = np.asarray(mask) > 0.5
            cnt = Counter(np.asarray(draws).tolist())
            assert all(legal[c] for c in cnt), (s, cnt, legal)
            _l, _e, p = _pruned_cat(zn[lo:hi], legal, int(next(iter(cnt))))
            freq = np.asarray([cnt.get(c, 0) / K
                               for c in np.flatnonzero(legal)])
            assert np.max(np.abs(freq - p)) < 0.03, (s, freq, p)


def test_rank_2_inputs_take_the_pre_ticket_broadcast_path():
    """Vertex-frame inputs (one sizes vector, one pair mask, one comp mask,
    one quant bit) never enter the per-slot path: ``_slot_inputs`` declines
    them, ``_face_masks`` gets a single ``AxisTokenFeatures`` (a tuple, not
    a list) and returns slot-identical masks -- the historical broadcast.
    The same-key equality with the per-slot call replicated three times holds
    for the op / i / axis masks; the j mask and the pair table DIFFER on
    purpose (the per-slot path reads the exact (i, j) table, the broadcast
    path keeps its historical row-0 read), which is why the flag-off identity
    is pinned trajectory-wide by ALPHAGRAD_EQ_DUMP and not here."""
    from alphagrad.approx.heads import quant_hardware_masks
    pol, tables = _policy()
    feats = _features()
    n = N_AX
    fpv = jnp.ones((n, n), jnp.float32) * (1.0 - jnp.eye(n))
    fcv = jnp.ones((n,), jnp.float32)
    sz = jnp.asarray([4, 12, 6, 3, 0, 0, 0, 0], jnp.int32)
    qhw = quant_hardware_masks()[0]
    assert pol._slot_inputs(feats, fpv, fcv, qhw, sz, jnp.asarray(1.0)) is None
    assert pol._slot_inputs(feats, fpv, fcv, qhw, None, None) is None
    single = pol._face_masks(pol._face_feats_1(feats, sz), fpv, fcv, qhw,
                             None, tables)
    for m in single:
        for s in range(1, FACE_SLOTS):
            assert bool(jnp.all(m[s] == m[0]))
    per = pol._slot_inputs(feats, jnp.broadcast_to(fpv, (FACE_SLOTS, n, n)),
                           jnp.broadcast_to(fcv, (FACE_SLOTS, n)), qhw,
                           jnp.broadcast_to(sz, (FACE_SLOTS, n)),
                           jnp.ones((FACE_SLOTS,), jnp.float32))
    assert per is not None and isinstance(per[0], list)
    rep = pol._face_masks(*per, None, tables)
    for k in (0, 1, 3):                       # op, i, axis masks agree
        assert bool(jnp.all(rep[k] == single[k])), k


# ==========================================================================
# 4. EVERY REQUESTED ROW GETS A SLOT-FRAME HOOK
# ==========================================================================
def test_every_requested_row_gets_a_slot_frame_hook():
    """The face dict carries a ``make_slot_frame_hook`` for every wire row
    that is not the end sentinel, whether or not the row names a dim of that
    slot. A row that names nothing is a counted miss at apply time (see the
    D2 test above), which is why the hook must exist to count it."""
    closed = _closed(_chain, _ARGS)
    config = SimpleNamespace(jaxpr=closed.jaxpr)
    ij, keys, key, store = _x_face(closed, _ARGS)
    rows, skips = _blank_rows()
    f = keys.index(key)
    rows[f, 0] = (COMPRESS_SENTINEL, 2, 0)      # axis 2: in no frame here
    rows[f, 1] = (COMPRESS_SENTINEL, 1, 0)      # axis 1: lhs frame only
    new = _face_dict_for_vertex(config, ij, 1, rows, skips)
    n_ = new[key] if not (len(new[key]) == 2 and len(new[key][0]) == 3) \
        else new[key][0]
    assert n_[0] is not None and n_[1] is not None and n_[2] is None
    assert hasattr(n_[0], "rules_for")
    assert hasattr(n_[1], "rules_for")
    # Row 0 names axis 2, which no slot of this face has. The hook exists (so
    # the miss is counted) and decodes to NOTHING on the lhs tensor.
    assert n_[0].rules_for(store["lhs"]) == ()
    # Row 1 names axis 1, which the lhs tensor does have.
    assert len(n_[1].rules_for(store["lhs"])) == 1


def test_make_live_masked_hook_records_noop_when_operand_unchanged():
    """When a transform returns the operand unchanged (cur is prev), the hook
    must record skipped and skipped_{kind}_noop, never applied (dsnn-3qm.73)."""
    from alphagrad.approx.common.masks import make_live_masked_hook
    from graphax.sparse.indexes import DiagonalIndex
    from graphax.sparse.tensor import SparseTensor
    import jax.numpy as jnp

    # A tensor where dimension 1 is implicit (partner of dim 0)
    st = SparseTensor(
        out_dims=(DiagonalIndex(id=0, size=64, axis=0, other_id=1),),
        primal_dims=(DiagonalIndex(id=1, size=64, axis=None, other_id=0),),
        val=jnp.ones((64,)),
    )
    stats = {}
    rule = Compress(axes=(), kind="mean")
    hook = make_live_masked_hook([rule], stats=stats)
    res = hook(st)
    assert res is st
    assert stats.get("applied", 0) == 0
    assert stats.get("applied_compress", 0) == 0
    assert stats.get("skipped", 0) == 1
    assert stats.get("skipped_compress_noop", 0) == 1

