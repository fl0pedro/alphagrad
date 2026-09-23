"""THE FACE QUANT BIT ON THE WIRE (owner ruling 2026-09-23).

A face Quant is a narrow CONTRACTION -- both operands bfloat16, float32 sums,
bfloat16 result -- and graphax refuses a Quant on one contraction slot only
(``FaceTransformIllegal``). On the alphagrad side that is ONE decision per
face, ``FaceAction.quant``, and every carrier of the dtype is derived from it.
Pinned here, end to end:

1. THE DECLARATION: ``quant`` is a per-face SCORED field, ``quant_dtype`` is
   DERIVED, and the translator's argument list did not change.
2. THE HEAD CANNOT WRITE A ONE-SIDED ROW: every sampled face carries the
   QUANT row on lhs AND rhs (one dtype, the narrow float's runtime index) or
   on neither, at every ``--approx-add`` width.
3. THE TRANSLATOR: ``Agent.to_env_action_dynamic`` turns the bit into two
   QUANT rows and leaves the ``new`` slot and the learned slots alone.
4. THE SEAM REFUSES THE ONE-SIDED FORM: ``env.check_face_quant_rows`` and
   ``env._face_dict_for_vertex`` raise on a hand-built one-sided pair and on
   two dtypes, before graphax sees a face.
5. THE TRANSPORT keeps the bit on the carry faces: a plan with a Quant on a
   carried face lands two-sided QUANT rows on every face of every carry
   vertex of the measured graph, while its Diag stays in the value.
6. THE PLAN RECORD's carry bytes count the given blocks, not the reference
   weights.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import equinox as eqx                                           # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import face_action as REC                 # noqa: E402
from alphagrad.approx.common import carry_plan as CP            # noqa: E402
from alphagrad.approx.common.masks import (                     # noqa: E402
    FACE_QUANT_DTYPES, FACE_QUANT_NARROW)
from alphagrad.approx.env import (                              # noqa: E402
    AXIS_FEATURE_DIM, MAX_AXES_PER_VERTEX, QUANT_SENTINEL, StepAction,
    _face_dict_for_vertex, check_face_quant_rows)
from alphagrad.approx.face_action import FaceAction             # noqa: E402
from alphagrad.approx.heads import (                            # noqa: E402
    AXIS_TAG_BITS, AxisTokenFeatures, MicroAction, OP_END, OP_QUANT,
    precompute_factor_tables)
from alphagrad.approx.unified_face_head import (                # noqa: E402
    FACE_SLOTS, O_QUANT, QUANT_SLOTS, head_layout)
from alphagrad.approx.unified_face_policy import (              # noqa: E402
    UnifiedFacePolicy, _NARROW_SLOT)

MODES = list(REC.ALL_MODES)
E, F = 32, 4
N_AX = 6
NARROW = int(_NARROW_SLOT)


def _feats(sizes=(8, 8, 4, 16, 6, 4)):
    sz = jnp.asarray(sizes, jnp.int32)
    n = len(sizes)
    return AxisTokenFeatures(
        size=sz, log_size=jnp.log(jnp.maximum(sz, 1).astype(jnp.float32)),
        tag_bits=jnp.zeros((n, AXIS_TAG_BITS), jnp.float32),
        group_id=-jnp.ones((n,), jnp.int32),
        valid_mask=jnp.ones((n,), jnp.float32))


def _policy(mode, seed=0, quant_up=0.0):
    pol = UnifiedFacePolicy(E, num_heads=2, max_faces=F,
                            key=jrand.PRNGKey(seed), approx_add=mode)
    if quant_up:
        pol = eqx.tree_at(
            lambda p: p.head.proj.layers[-1].bias, pol,
            pol.head.proj.layers[-1].bias.at[O_QUANT].add(quant_up))
    return pol


def _env(mode, quant_up=0.0):
    pol = _policy(mode, quant_up=quant_up)
    f = _feats()
    n = f.size.shape[0]
    return (pol, f, precompute_factor_tables(64),
            jnp.ones((F, n, n), jnp.float32), jnp.ones((F, n), jnp.float32),
            jnp.ones((F,), jnp.float32))


# ============================================================ 1. declaration
def test_the_declaration_carries_the_bit_and_derives_the_dtype():
    q = REC.field("quant")
    assert not q.per_slot and q.role == REC.SCORED and not q.translator
    assert q.fill == 0 and q.modes is None
    d = REC.field("quant_dtype")
    assert d.per_slot and d.role == REC.DERIVED and d.translator
    assert REC.translator_names() == (
        "op_type", "i", "j", "factor", "compress_kind", "quant_dtype",
        "quant_scale_sign", "quant_scale_frac")
    assert FACE_QUANT_DTYPES == ("float32", "bfloat16")
    assert FACE_QUANT_DTYPES[FACE_QUANT_NARROW] == "bfloat16"
    for mode in MODES:
        assert "quant" in REC.per_face_names(mode)
        assert REC.zeros(mode, F).quant.shape == (F,)


# ============================================== 2. the head writes two rows
@pytest.mark.parametrize("mode", MODES)
def test_the_head_writes_the_bit_on_both_operand_slots_or_neither(mode):
    pol, f, tables, pair, comp, valid = _env(mode, quant_up=1.0)
    n_on = n_off = 0
    for seed in range(40):
        fa, *_ = pol.sample(None, f, tables, jrand.PRNGKey(seed), pair, comp,
                            valid)
        REC.check(fa, mode, F)
        op = np.asarray(fa.op_type)
        dt = np.asarray(fa.quant_dtype)
        q = np.asarray(fa.quant)
        sk = np.asarray(fa.skip)
        for k in range(F):
            lhs, rhs = (op[k, s] for s in QUANT_SLOTS)
            if q[k]:
                assert sk[k] == 0
                assert lhs == OP_QUANT and rhs == OP_QUANT, (seed, k, op[k])
                assert all(dt[k, s] == NARROW for s in QUANT_SLOTS)
                n_on += 1
            else:
                assert OP_QUANT not in op[k].tolist(), (seed, k, op[k])
                assert not dt[k].any()
                n_off += 1
            # the new slot and the learned slots never carry a Quant
            assert OP_QUANT not in op[k, FACE_SLOTS - 1:].tolist()
    assert n_on > 0 and n_off > 0, (n_on, n_off)


# ================================================= 3. the translator, use 4
@pytest.mark.parametrize("mode", MODES)
def test_the_translator_writes_a_quant_row_on_both_slots(mode, monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", mode)
    from alphagrad.approx.ppo import Agent
    pol, f, tables, pair, comp, valid = _env(mode, quant_up=1.0)
    S = REC.n_slots(mode)
    # a face with the bit and one without, from the head's own encoder
    fa = None
    for seed in range(60):
        cand, *_ = pol.sample(None, f, tables, jrand.PRNGKey(seed), pair,
                              comp, valid)
        q = np.asarray(cand.quant)
        if q[0] == 1 and q[1] == 0 and int(cand.skip[0]) == 0:
            fa = cand
            break
    assert fa is not None, "no draw with face 0 quantized and face 1 not"
    zero = jnp.zeros((1,), jnp.int32)
    act = MicroAction(
        op_type=jnp.full((1,), OP_END, jnp.int32), i=zero, j=zero,
        exponents=jnp.zeros((1, 9), jnp.int32), factor=zero,
        compress_kind=zero, quant_dtype=zero,
        quant_scale_sign=jnp.ones((1,), jnp.int32),
        quant_scale_frac=jnp.zeros((1,), jnp.float32))
    ax = jnp.zeros((2, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), jnp.int32)
    sa = Agent.to_env_action_dynamic(None, 0, act, ax, face_action=fa)
    assert isinstance(sa, StepAction)
    rows = np.asarray(sa.face_rows)
    assert rows.shape == (F, S, 3)
    for s in QUANT_SLOTS:
        assert rows[0, s].tolist() == [QUANT_SENTINEL, NARROW, 0], rows[0]
    assert rows[0, FACE_SLOTS - 1, 0] != QUANT_SENTINEL
    assert not (rows[1, :, 0] == QUANT_SENTINEL).any(), rows[1]
    # the rows pass the seam's own check, face by face
    for k in range(F):
        check_face_quant_rows(rows[k], where=f"face {k}")


# ============================================ 4. the seam refuses one side
def _one_face_rows(lhs, rhs, new=(-1, -1, 0)):
    return np.asarray([lhs, rhs, new], np.int32)


def test_a_one_sided_row_is_refused_at_the_seam():
    q = (QUANT_SENTINEL, NARROW, 0)
    none = (-1, -1, 0)
    check_face_quant_rows(_one_face_rows(q, q))
    check_face_quant_rows(_one_face_rows(none, none))
    check_face_quant_rows(_one_face_rows(none, none, q))
    with pytest.raises(ValueError, match="lhs contraction slot only"):
        check_face_quant_rows(_one_face_rows(q, none), where="t")
    with pytest.raises(ValueError, match="rhs contraction slot only"):
        check_face_quant_rows(_one_face_rows(none, q), where="t")
    with pytest.raises(ValueError, match="two dtypes"):
        check_face_quant_rows(_one_face_rows(q, (QUANT_SENTINEL, NARROW + 1, 0)))
    # a structural row on the other operand is still one-sided
    with pytest.raises(ValueError, match="only"):
        check_face_quant_rows(_one_face_rows(q, (0, 0, 2)))


def _mlp(x, W1, W2, W3):
    h = jnp.tanh(x @ W1)
    a = jnp.tanh(h @ W2)
    b = jnp.tanh(h @ W3)
    return a * b


def test_the_face_dict_builder_refuses_a_one_sided_plan(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_APPROX_ADD", "lossless")
    monkeypatch.setenv("ALPHAGRAD_MAX_FACES", "16")
    from types import SimpleNamespace
    from graphax import faces_of
    from graphax.incremental import IncrementalJaxpr
    import alphagrad.approx.env as envmod
    args = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1,
            jnp.ones((32, 16)) * 0.1, jnp.ones((32, 16)) * 0.1)
    cj = jax.make_jaxpr(_mlp)(*args)
    argnums = (1, 2, 3)
    ij = IncrementalJaxpr(cj.jaxpr, argnums, list(cj.literals), list(args),
                          track_faces=False)
    cfg = SimpleNamespace(jaxpr=cj.jaxpr)
    v = next(v for v in range(1, len(cj.jaxpr.eqns) + 1)
             if faces_of(ij.graph, ij.tgraph, v, cj.jaxpr))
    Fm = envmod.MAX_FACES
    rows = np.full((Fm, FACE_SLOTS, 3), -1, np.int32)
    rows[..., 2] = 0
    skips = np.zeros((Fm,), np.int32)
    rows[0, 0] = (QUANT_SENTINEL, NARROW, 0)
    with pytest.raises(ValueError, match="lhs contraction slot only"):
        _face_dict_for_vertex(cfg, ij, v, rows, skips)
    rows[0, 1] = (QUANT_SENTINEL, NARROW, 0)
    per_face = _face_dict_for_vertex(cfg, ij, v, rows, skips)
    assert len(per_face) == 1
    # a skipped face is not a contraction and carries no rows to check
    rows[0, 1] = (-1, -1, 0)
    skips[0] = 1
    per_face = _face_dict_for_vertex(cfg, ij, v, rows, skips)
    assert len(per_face) == 1


# ================================================== 5. the carry transport
def _variant():
    return {"container": "quant",
            "vertex_map": {1: 1, 2: 2, 4: 4},
            "alt_carry": (3,),
            "valid": {1, 2, 3, 4}}


def _plan(quant_on_carry, diag_on_carry=True):
    T, Fm, S = 4, 2, 3
    o_list = [4, 3, 2, 1]                     # vertex 3 is the carry
    rs = np.full((T, 2, 3), -1, np.int32)
    rs[:, :, 2] = 0
    fs = np.full((T, Fm, S, 3), -1, np.int32)
    fs[..., 2] = 0
    sk = np.zeros((T, Fm), np.int32)
    fs[0, 0, 2] = (0, 0, 2)                   # a Diag on the body vertex 4
    if diag_on_carry:
        fs[1, 1, 2] = (0, 0, 2)               # a Diag on the carry vertex 3
    if quant_on_carry:
        for s in QUANT_SLOTS:
            fs[1, 0, s] = (QUANT_SENTINEL, NARROW, 7)
    return o_list, rs, fs, sk


@pytest.mark.parametrize("quant_on_carry", [True, False])
def test_the_transport_keeps_the_plans_quant_bit_on_the_carry_faces(
        quant_on_carry):
    o_list, rs, fs, sk = _plan(quant_on_carry)
    order, rs2, fs2, sk2, jn2 = CP.transport_wires(o_list, _variant(), rs, fs,
                                                   sk, None)
    assert sorted(order) == [1, 2, 3, 4] and jn2 is None
    pos = {int(v): k for k, v in enumerate(order)}
    # the body rows travel by position
    np.testing.assert_array_equal(fs2[pos[4]], fs[0])
    np.testing.assert_array_equal(fs2[pos[2]], fs[2])
    np.testing.assert_array_equal(fs2[pos[1]], fs[3])
    carry = fs2[pos[3]]
    # Diag on the carry face never travels: it is in the value
    assert carry[1, 2, 0] == -1
    assert not (carry[..., 0] >= 0).any()
    assert int(sk2[pos[3]].max()) == 0
    if quant_on_carry:
        # ... and the Quant bit lands two-sided on EVERY carry face, the
        # plan's own row copied dtype column and all
        for f in range(carry.shape[0]):
            for s in QUANT_SLOTS:
                assert carry[f, s].tolist() == [QUANT_SENTINEL, NARROW, 7], (
                    f, s, carry[f])
            assert carry[f, 2, 0] == -1
            check_face_quant_rows(carry[f], where=f"carry face {f}")
    else:
        assert int(carry[..., 0].max()) == -1


def test_a_skipped_carry_face_does_not_lend_its_quant_row():
    o_list, rs, fs, sk = _plan(True)
    sk[1, 0] = 1
    order, _rs2, fs2, _sk2, _ = CP.transport_wires(o_list, _variant(), rs, fs,
                                                   sk, None)
    pos = {int(v): k for k, v in enumerate(order)}
    assert int(fs2[pos[3]][..., 0].max()) == -1


# ============================================ 6. the container's bytes
def test_carry_at_rest_bytes_counts_the_given_blocks_only():
    from alphagrad.approx.common.rsnn_shd import RSNN_HEAD_SLOTS
    head = tuple(np.zeros((3,), np.float32) for _ in range(RSNN_HEAD_SLOTS))
    weights = tuple(np.ones((4, 4), np.float32) for _ in range(3))
    blocks = (np.ones((5, 7), np.float32), np.ones((2, 3), np.float32))
    var = {"args": head + weights + blocks}
    assert CP.carry_at_rest_bytes(var, "rtrl") == 4 * (35 + 6)
    narrow = {"args": head + weights + tuple(
        b.astype(jnp.bfloat16) for b in blocks)}
    assert CP.carry_at_rest_bytes(narrow, "rtrl") == 2 * (35 + 6)
    adj = {"args": head + blocks}
    assert CP.carry_at_rest_bytes(adj, "bptt") == 4 * (35 + 6)
    assert CP.carry_at_rest_bytes({"args": head}, "tbptt") == 0
