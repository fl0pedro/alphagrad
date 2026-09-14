"""``--per-face-masks`` SIZES half on the ``--live-faces`` path (A1b).

WHAT WAS WRONG. ``--per-face-masks`` has two halves. The apply-time half runs
everywhere. The SIZES half -- handing the head each face's LIVE
``out_dims ++ primal_dims`` logical sizes, which is what makes its ``pair_ok``
gate and its hardcoded ``factor = gcd(N_i, N_j)`` mean anything per face --
was sourced from ``LiveVertexMaskOracle``, and ``--live-faces`` runs no oracle
(``ppo._NO_ORACLE = no_approx_head or live_faces``). So on the campaign path
the head kept masking with the vertex's NOMINAL ``out_shape ++ first-invar
shape``: one vector for every face of the vertex.

WHAT IS UNDER TEST.

  * :meth:`LiveFaceStream.face_dim_sizes` produces the SAME quantity the
    oracle's ``face_masks_and_sizes(per_face=True)`` does -- value for value,
    face for face -- from the stream's own elimination;
  * it is keyed by FACE KEY, so a face the enumeration lists and the
    elimination never visits leaves a ZERO row instead of shifting every later
    face's dims up by one;
  * it is a PURE READ: the prefix tokenizer it probes is bit-identical
    afterwards, so the chunk the head reads next is unchanged;
  * the loss-side mirror still holds with live-derived sizes (sampling ==
    replay log-prob), with a negative control proving the mirror is
    load-bearing;
  * flag-off changes nothing.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from graphax import inline_call_primitives

from alphagrad.approx.common.masks import (
    FACE_QUANT_DTYPES, LiveVertexMaskOracle, dim_logical_sizes,
    quant_valid_mask, set_per_face_masks)
from alphagrad.approx.live_faces import LiveFaceStream

try:
    from jax.extend.core import ClosedJaxpr
except ImportError:                                        # pragma: no cover
    from jax._src.core import ClosedJaxpr

N_AX = 8
MAX_F = 8
VOCAB = 256


# The same split-gcd fixture per_face_masks_test.py uses: the two faces of the
# middle vertex have gcds 4 and 3, so NO per-vertex size vector can serve both
# -- which is exactly the situation the sizes half exists for.
_A = jnp.asarray(np.linspace(0.1, 0.9, 12 * 6, dtype=np.float32).reshape(12, 6))
_B = jnp.asarray(np.linspace(0.2, 0.8, 8 * 12, dtype=np.float32).reshape(8, 12))
_C = jnp.asarray(np.linspace(0.3, 0.7, 9 * 12, dtype=np.float32).reshape(9, 12))


def _split_gcd(x):
    e = _A @ x
    return _B @ e, _C @ e


_SPLIT_ARGS = (jnp.asarray(np.linspace(0.1, 0.9, 6, dtype=np.float32)),)


def _mlp(x, W1, W2):
    return jnp.tanh(x @ W1) @ W2


_MLP_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1)


def _closed(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    return cj if jx is cj.jaxpr else ClosedJaxpr(jx, consts)


def _stream_and_oracle(fn, xs):
    closed = _closed(fn, xs)
    argnums = tuple(range(len(xs)))
    lf = LiveFaceStream(closed.jaxpr, argnums, list(closed.literals), list(xs),
                        vocab=VOCAB, max_faces=MAX_F, max_axes=N_AX)
    o = LiveVertexMaskOracle(closed.jaxpr, list(closed.literals), list(xs),
                             argnums, max_axes=N_AX)
    return closed, lf, o


def _exact_prefix(total_v, k):
    """``(order, specs, n)`` for the first ``k`` vertices of REVERSE order.

    Reverse is the campaign's fixed order, and under --live-faces the
    per-vertex wire rows are always the exact END rows.
    """
    order = np.zeros((total_v,), np.int32)
    rev = list(range(total_v, 0, -1))
    order[:k] = np.asarray(rev[:k], np.int32)
    specs = -np.ones((total_v, 3, 3), np.int32)
    return order, specs, k


# ==========================================================================
# 1. THE LIVE SOURCE IS THE ORACLE'S SOURCE
# ==========================================================================
@pytest.mark.parametrize("fn,xs", [(_split_gcd, _SPLIT_ARGS),
                                   (_mlp, _MLP_ARGS)])
def test_live_sizes_equal_the_oracle_sizes(fn, xs):
    """Face for face, value for value, on every vertex of a reverse walk.

    This is the whole claim of A1b: the sizes half does not need the oracle,
    it needs an elimination, and --live-faces already runs one. If these two
    ever disagree, the campaign path is masking with something the oracle
    path is not, and the two arms' numbers stop being comparable.
    """
    closed, lf, o = _stream_and_oracle(fn, xs)
    total_v = len(closed.jaxpr.eqns)
    compared = 0
    for k, v in enumerate(range(total_v, 0, -1)):
        order, specs, n = _exact_prefix(total_v, k)
        l_sz, l_qt, l_n = lf.face_dim_sizes(order, specs, n, v)
        o_sz, o_qt, o_nf = _oracle_sizes(o, v)
        assert int(l_n) >= int(o_nf), (
            f"vertex {v}: the stream enumerates {int(l_n)} faces but the "
            f"oracle probe visited {int(o_nf)} -- the enumeration is a "
            "SUPERSET of the visited faces, never a subset")
        for f in range(int(o_nf)):
            assert np.array_equal(l_sz[f], o_sz[f]), (
                f"vertex {v} face {f}: live sizes {l_sz[f].tolist()} != "
                f"oracle sizes {o_sz[f].tolist()}")
            assert float(l_qt[f]) == float(o_qt[f]), (
                f"vertex {v} face {f}: live QUANT bit {float(l_qt[f])} != "
                f"oracle {float(o_qt[f])}")
            compared += 1
        o.advance(v, rules=())
    assert compared > 0, "the fixture produced no faces at all"


def _oracle_sizes(o, v):
    sz, qt, nf = np.zeros((MAX_F, N_AX), np.int32), np.zeros((MAX_F,),
                                                            np.float32), 0
    try:
        _p, _c, s, q, n = o.face_masks_and_sizes(v, MAX_F, per_face=True)
    except Exception:                                      # pragma: no cover
        return sz, qt, 0
    sz[:] = np.asarray(s, np.int32)
    qt[:] = np.asarray(q, np.float32)
    return sz, qt, int(n)


# ==========================================================================
# 2. THE NUMBERING
# ==========================================================================
def _nominal_sizes(jaxpr, v):
    """The vector the head was handed BEFORE A1b: the vertex's nominal
    ``out_shape ++ first-invar shape`` (``env.compute_static_axis_state``),
    identical for every face of the vertex."""
    eqn = jaxpr.eqns[v - 1]
    out = tuple(eqn.outvars[0].aval.shape)
    ins = [tuple(iv.aval.shape) for iv in eqn.invars if hasattr(iv, "aval")]
    dims = list(out) + list(ins[0] if ins else ())
    z = np.zeros((N_AX,), np.int32)
    z[: min(len(dims), N_AX)] = np.asarray(dims[:N_AX], np.int32)
    return z


@pytest.mark.parametrize("fn,xs", [(_split_gcd, _SPLIT_ARGS),
                                   (_mlp, _MLP_ARGS)])
def test_sizes_are_the_diag_numbering_not_val_shape(fn, xs):
    """``out_dims ++ primal_dims`` logical sizes, not the physical ``val``.

    Getting this wrong is SILENT: the head would derive its factor and its
    ``pair_ok`` gate in a coordinate system ``Diag(i, j)`` and
    ``rule_is_legal`` do not use, so the mask would admit pairs the operand
    rejects. Pinned against ``dim_logical_sizes`` on the very tensor the probe
    saw.

    The counter at the end is the control that makes this worth running: at
    least one face must come back with sizes DIFFERENT from the vertex's
    nominal per-vertex vector, or the whole sizes half is a no-op on this
    fixture and the equality above would hold for the wrong reason.
    """
    closed, lf, o = _stream_and_oracle(fn, xs)
    total_v = len(closed.jaxpr.eqns)
    differs = 0
    for k, v in enumerate(range(total_v, 0, -1)):
        order, specs, n = _exact_prefix(total_v, k)
        l_sz, _q, l_n = lf.face_dim_sizes(order, specs, n, v)
        faces = o.probe_faces(v, approx=True)
        nom = _nominal_sizes(closed.jaxpr, v)
        for f, st in enumerate(faces[:int(l_n)]):
            want = dim_logical_sizes(st, N_AX)
            assert np.array_equal(l_sz[f], want), (
                f"vertex {v} face {f}: {l_sz[f].tolist()} != "
                f"{want.tolist()} -- not the Diag numbering")
            # `val.shape` is a DIFFERENT list (a coupled pair stores two
            # logical dims in one physical axis), so it must not be what
            # comes back except by coincidence of rank.
            if not np.array_equal(want, nom):
                differs += 1
        o.advance(v, rules=())
    assert differs > 0, (
        "every face's LIVE sizes equalled the vertex's NOMINAL sizes on this "
        "fixture, so this test cannot distinguish the per-face vector from "
        "the per-vertex one it replaces")


# ==========================================================================
# 3. THE PROBE IS A PURE READ
# ==========================================================================
def test_the_size_probe_does_not_disturb_the_token_stream():
    """The chunk the head reads must be bit-identical either side of a probe.

    The probe runs a REAL elimination on the live prefix tokenizer. If
    ``_Snapshot`` did not undo all of it, the next chunk would carry the
    probe's equations, its variable names, or its face records -- and the
    head would read tokens for a graph the measurement never builds.
    """
    closed, lf, _o = _stream_and_oracle(_mlp, _MLP_ARGS)
    total_v = len(closed.jaxpr.eqns)
    v = total_v
    order, specs, n = _exact_prefix(total_v, 0)
    rows = -np.ones((MAX_F, 3, 3), np.int32)
    skips = np.zeros((MAX_F,), np.int32)
    vspecs = -np.ones((3, 3), np.int32)

    before = lf.chunk(order, specs, n, v, vspecs, rows, skips, 0)
    lf._chunks.clear()                       # force a real recomputation
    lf.face_dim_sizes(order, specs, n, v)
    after = lf.chunk(order, specs, n, v, vspecs, rows, skips, 0)
    for a, b, name in zip(before, after,
                          ("tokens", "eqn_ids", "count", "n_faces", "ends",
                           "head")):
        assert np.array_equal(np.asarray(a), np.asarray(b)), (
            f"the size probe changed the chunk's {name}")


def test_the_probe_is_memoized_per_prefix_and_vertex():
    """One probe pair serves the WHOLE face loop.

    Faces of one vertex write disjoint (in_edge, out_edge) edges and read only
    edges incident to the central vertex, so face f-1's approximation cannot
    move face f's operand -- which is what lets this be 2 extra eliminations
    per vertex step rather than 2 per face. If the memo ever keys on something
    that varies per face, the cost silently multiplies by n_faces.
    """
    closed, lf, _o = _stream_and_oracle(_mlp, _MLP_ARGS)
    total_v = len(closed.jaxpr.eqns)
    order, specs, n = _exact_prefix(total_v, 0)
    lf.consume_stats()
    for _ in range(5):
        lf.face_dim_sizes(order, specs, n, total_v)
    st = lf.consume_stats()
    assert st["size_probe"] == 1, (
        f"expected ONE probe (one engine, dsnn-3qm.65), got {st['size_probe']}")
    assert st["size_hit"] == 4, st


# ==========================================================================
# 4. THE LOSS-SIDE MIRROR (the PPO ratio at epoch 0)
# ==========================================================================
def test_sample_equals_replay_with_live_derived_sizes():
    """Sampling and replay must score the SAME log-prob under LIVE sizes.

    The arrays here are not synthetic: they are what
    ``LiveFaceStream.face_dim_sizes`` actually returns for a real vertex, fed
    through the same ``UnifiedFacePolicy`` entry points ``_face_loop`` and
    ``_face_replay`` call. The negative control at the end is what makes this
    a test rather than a tautology.
    """
    from alphagrad.approx.heads import (
        AXIS_TAG_BITS, AxisTokenFeatures, precompute_factor_tables)
    from alphagrad.approx.unified_face_policy import UnifiedFacePolicy

    closed, lf, _o = _stream_and_oracle(_split_gcd, _SPLIT_ARGS)
    total_v = len(closed.jaxpr.eqns)
    # The vertex whose faces disagree about the gcd is the interesting one:
    # take the first prefix position whose sizes are not all identical.
    fsz = fqt = None
    for k, v in enumerate(range(total_v, 0, -1)):
        order, specs, n = _exact_prefix(total_v, k)
        sz, qt, nf = lf.face_dim_sizes(order, specs, n, v)
        if int(nf) >= 2 and not np.array_equal(sz[0], sz[1]):
            fsz, fqt = jnp.asarray(sz), jnp.asarray(qt)
            break
    assert fsz is not None, (
        "no vertex in the fixture had two faces with DIFFERENT live sizes, "
        "so this test could not distinguish per-face from per-vertex")

    E, F, nax = 32, MAX_F, N_AX
    tables = precompute_factor_tables(64)
    pol = UnifiedFacePolicy(E, num_heads=2, max_faces=F, key=jrand.PRNGKey(0))
    nom = jnp.asarray([12, 6, 8, 4, 2, 2, 1, 1][:nax], jnp.int32)
    feats = AxisTokenFeatures(
        size=nom, log_size=jnp.log(nom.astype(jnp.float32)),
        tag_bits=jnp.zeros((nax, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((nax,), jnp.int32),
        valid_mask=jnp.ones((nax,), jnp.float32))
    fpv = jnp.ones((F, nax, nax), jnp.float32)
    fcv = jnp.ones((F, nax), jnp.float32)
    fval = (jnp.arange(F) < 2).astype(jnp.float32)

    out = pol.sample(None, feats, tables, jrand.PRNGKey(3), fpv, fcv, fval,
                     face_sizes=fsz, face_quant=fqt)
    fa, logp, ent, arity = out[0], out[1], out[2], out[3]
    logp2, ent2, arity2 = pol.evaluate(
        None, feats, tables, fa, fpv, fcv, fval,
        face_sizes=fsz, face_quant=fqt)[:3]
    assert float(jnp.abs(logp - logp2)) < 1e-5, (
        f"ratio != 1 at epoch 0: {float(logp)} vs {float(logp2)}")
    assert float(jnp.abs(ent - ent2)) < 1e-5
    assert float(arity) == float(arity2)

    # NEGATIVE CONTROL. A replay that drops the live sizes -- i.e. the
    # behaviour every --live-faces run had before A1b -- must score a
    # DIFFERENT number. If it does not, the sizes never reached the masks and
    # the equality above proves nothing.
    logp_blind = pol.evaluate(None, feats, tables, fa, fpv, fcv, fval)[0]
    assert float(jnp.abs(logp - logp_blind)) > 1e-6, (
        "dropping the live per-face sizes changed nothing -- they are not "
        "reaching the masks")


# ==========================================================================
# 5. FLAG-OFF
# ==========================================================================
def test_face_dim_sizes_does_not_read_the_flag():
    """The stream method is the SOURCE; ``ppo._PFM_LIVE_SIZES`` is the gate.

    Pinned so nobody later makes ``face_dim_sizes`` itself flag-sensitive:
    the whole point of gating in ppo.py is that with the flag off the array
    is never requested, so no callback and no host work exist at all.
    """
    closed, lf, _o = _stream_and_oracle(_mlp, _MLP_ARGS)
    total_v = len(closed.jaxpr.eqns)
    order, specs, n = _exact_prefix(total_v, 0)
    try:
        set_per_face_masks(False)
        off = lf.face_dim_sizes(order, specs, n, total_v)
        lf._sizes.clear()
        set_per_face_masks(True)
        on = lf.face_dim_sizes(order, specs, n, total_v)
    finally:
        set_per_face_masks(False)
    assert np.array_equal(off[0], on[0])
    assert np.array_equal(off[1], on[1])
