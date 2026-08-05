"""The per-vertex micro-action masks must be read off the LIVE edges.

The per-vertex ``transforms`` list is NOT applied to the vertex's own elemental
Jacobian. ``graphax.core._eliminate_vertex`` applies it to ``edge_outval`` --
the per-face CONTRACTION of the in-edge and out-edge Jacobians (the same site
as the per-face ``res`` slot) -- so its rank, sizes and diagonal pairings come
from a different pair of graph variables and depend on the whole elimination
prefix. Deciding legality from the vertex's nominal ``(out_shape ++
primal_shape)`` therefore proposes actions graphax rejects with "TRANSFORM DID
NOT FIT", and every rejection sentinels a whole measurement.

These tests pin the two halves of the fix:

* the ORACLE is faithful -- anything :class:`LiveVertexMaskOracle` admits, a
  real elimination accepts (this is the property that was violated in
  production: 13312 rejections in one PPO run);
* the oracle is NOT vacuous -- it still admits real actions;
* the historical failure ``Diag(i=1, j=3, factor=3)`` at nn256 vertex 4 is
  reproduced from the nominal shapes and rejected by the live mask.
"""
import os

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from graphax import inline_call_primitives
from graphax.sparse.micro_actions import Compress, Diag

from alphagrad.approx.common import get_args, get_fn
from alphagrad.approx.common.masks import (
    LiveVertexMaskOracle, compress_valid_mask, diag_pair_factor_space,
    diag_valid_mask,
)
from alphagrad.approx.env import diag_row_to_pair

ARGNUMS = (2, 3, 4, 5)


def _traced_inlined(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    if jx is cj.jaxpr:
        return cj
    from jax.extend.core import ClosedJaxpr

    return ClosedJaxpr(jx, consts)


@pytest.fixture(scope="module")
def nn256():
    """The graph the failing PPO run used: VmappedNeuralNetwork on mnist."""
    _, args_key, _ = jrand.split(jrand.PRNGKey(7), 3)
    fn = get_fn("VmappedNeuralNetwork")
    xs = get_args("VmappedNeuralNetwork", args_key, dataset="mnist")
    cj = _traced_inlined(fn, xs)
    return cj.jaxpr, list(cj.literals), list(xs)


def _oracle(nn256):
    jaxpr, consts, xs = nn256
    return LiveVertexMaskOracle(jaxpr, consts, xs, ARGNUMS)


def test_the_transform_site_is_a_join_intermediate(nn256):
    """The tensor a per-vertex Diag lands on is NOT the vertex's Jacobian.

    This is the premise the whole fix rests on: if the per-vertex site really
    saw the vertex's own elemental Jacobian, a mask built from the jaxpr
    equation would be exact and there would be no bug.
    """
    jaxpr, _, _ = nn256
    o = _oracle(nn256)
    for v in range(1, 5):
        o.advance(v, ())
    eqn = jaxpr.eqns[4]                       # vertex 5, tanh (16,63)->(16,63)
    nominal = (tuple(eqn.outvars[0].aval.shape),
               tuple(eqn.invars[0].aval.shape))
    assert nominal == ((16, 63), (16, 63))
    faces = o.probe_faces(5)
    assert faces, "vertex 5 must still have faces after eliminating 1..4"
    live = [(tuple(int(d.size) for d in st.out_dims),
             tuple(int(d.size) for d in st.primal_dims)) for st in faces]
    assert any(l != nominal for l in live), (
        f"expected the live edges to differ from the nominal signature, got {live}"
    )


def test_nn256_vertex4_diag_1_3_factor_3_is_masked(nn256):
    """The exact production failure, reproduced and then masked.

    ``Diag(i=1, j=3, factor=3)`` at vertex 4 passes every nominal-shape screen
    (3 divides both 63s) but the live edge is already a coupled diagonal with
    meta 63, so graphax only accepts a multiple of 63.
    """
    jaxpr, _, _ = nn256
    eqn = jaxpr.eqns[3]
    out_shape = tuple(eqn.outvars[0].aval.shape)
    primal_shape = tuple(eqn.invars[0].aval.shape)
    assert out_shape == (16, 63) and primal_shape == (16, 63)
    # the nominal screen rule_specs_to_transforms applies: 3 | 63 both sides
    assert out_shape[1] % 3 == 0 and primal_shape[1] % 3 == 0
    assert diag_row_to_pair(jaxpr, 4, 1, 1) == (1, 3)

    o = _oracle(nn256)
    faces = o.probe_faces(4)
    st = faces[0]
    # the live edge: fully diagonal, both pairs coupled, val = (16, 63)
    assert tuple(st.val.shape) == (16, 63)
    assert all(d.is_sparse for d in (*st.out_dims, *st.primal_dims))
    base, span = diag_pair_factor_space(st, 1, 3)
    assert (base, span) == (63, 1), "pair (1,3) is coupled with meta 63"
    assert not diag_valid_mask(st, 8)[1, 3]

    pair, _ = o.vertex_mask(4)
    assert not pair[1, 3], "the mask must not admit the production failure"


def test_nn256_reduce_max_compress_axis2_is_masked(nn256):
    """The other production failure: a physical axis past the live val.ndim.

    The vertex is addressed by POSITION in the jaxpr and the position moved
    (the graph now carries an extra ``max`` guard for the log-sum-exp), so the
    (16,10) join intermediate this test was written against is vertex 10, not
    11. The equation identity is asserted first so a future graph change fails
    with a clear message instead of silently probing an unrelated vertex.
    """
    jaxpr, _, _ = nn256
    V = 10
    assert jaxpr.eqns[V - 1].primitive.name == "reduce_max", (
        f"nn256 graph moved again: expected the (16,10)->(16,) reduce_max at "
        f"vertex {V}, found {jaxpr.eqns[V - 1].primitive.name}"
    )
    o = _oracle(nn256)
    st = o.probe_faces(V)[0]
    assert tuple(st.val.shape) == (16, 10)      # two logical pairs, ONE per axis
    assert not compress_valid_mask(st, 8)[2]
    _, comp = o.vertex_mask(V)
    assert not comp[2]
    assert comp[0] and comp[1], "axes 0/1 are real actions and must survive"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_admitted_action_is_accepted_by_graphax(nn256, seed):
    """SAFETY: mask says legal => a real elimination accepts it.

    Walks a random elimination order; at every prefix, every action the mask
    admits for every remaining vertex is applied on a fresh replay of that
    prefix. A single "TRANSFORM DID NOT FIT" here is the production bug.
    """
    jaxpr, consts, xs = nn256
    total_v = len(jaxpr.eqns)
    rng = np.random.RandomState(seed)
    order = [int(v) for v in rng.permutation(np.arange(1, total_v + 1))]

    live = LiveVertexMaskOracle(jaxpr, consts, xs, ARGNUMS)
    n_checked = 0
    for t, v in enumerate(order):
        pair, comp = live.vertex_mask(v)
        actions = [Compress(axes=(int(a),), kind="mean")
                   for a in range(comp.shape[0]) if comp[a]]
        for i in range(pair.shape[0]):
            for j in range(pair.shape[1]):
                if not pair[i, j]:
                    continue
                st = live.probe_faces(v)[0]
                base, span = diag_pair_factor_space(st, i, j)
                for d in range(2, span + 1):
                    if span % d == 0:
                        actions.append(Diag(i=i, j=j, factor=base * d))
        for act in actions:
            trial = LiveVertexMaskOracle(jaxpr, consts, xs, ARGNUMS)
            for u in order[:t]:
                trial.advance(u, ())
            trial.advance(v, (act,))       # raises on "TRANSFORM DID NOT FIT"
            n_checked += 1
        live.advance(v, ())
    assert n_checked > 0, "the mask admitted nothing at all — vacuous test"


def test_mask_is_not_vacuous(nn256):
    """The oracle must leave the policy a usable action space."""
    jaxpr, _, _ = nn256
    o = _oracle(nn256)
    n_comp = n_pair = 0
    for v in range(1, len(jaxpr.eqns) + 1):
        pair, comp = o.vertex_mask(v)
        n_comp += int(comp.sum())
        n_pair += int(pair.sum())
    assert n_comp >= len(jaxpr.eqns), (
        f"only {n_comp} legal COMPRESS axes across the whole graph")


# ---------------------------------------------------------------------------
# The policy side: masks must reach the head, and sample/evaluate must agree
# ---------------------------------------------------------------------------


def _policy_bits(nn256, max_substeps=1):
    from alphagrad.approx.env import compute_static_axis_state
    from alphagrad.approx.heads import (
        MicroActionPolicy, precompute_factor_tables)
    from alphagrad.approx.ppo import _axis_features_from_state

    jaxpr, _, _ = nn256
    total_v = len(jaxpr.eqns)
    ax_st, ax_va = compute_static_axis_state(jaxpr, total_v)
    max_size = max(
        int(s) for e in jaxpr.eqns
        for v in list(e.outvars) + list(e.invars)
        if hasattr(v, "aval") for s in (v.aval.shape or (1,))
    )
    policy = MicroActionPolicy(
        embd_dim=16, num_heads=2, max_substeps=max_substeps,
        num_encoder_layers=1, key=jrand.PRNGKey(3))
    tables = precompute_factor_tables(max_size)
    feats = _axis_features_from_state(jnp.asarray(ax_st[3]),
                                      jnp.asarray(ax_va[3]))
    return policy, tables, feats, jnp.zeros((16,), jnp.float32)


def test_masked_policy_never_samples_a_masked_pair(nn256):
    """The head must actually honour ``pair_valid`` / ``compress_valid``.

    Vertex 4 (the production failure) has NO legal Diag at all, so a masked
    policy must never emit OP_DIAG there however it is seeded.
    """
    from alphagrad.approx.heads import OP_DIAG

    policy, tables, feats, ctx = _policy_bits(nn256)
    pair, comp = _oracle(nn256).vertex_mask(4)
    assert not pair.any(), "vertex 4 should afford no legal Diag at all"
    pv = jnp.asarray(pair.astype(np.float32))
    cv = jnp.asarray(comp.astype(np.float32))
    for seed in range(40):
        acts, *_ = policy.sample(ctx, feats, tables, jrand.PRNGKey(seed),
                                 pair_valid=pv, compress_valid=cv)
        assert int(acts.op_type[0]) != OP_DIAG, (
            f"seed {seed}: masked policy still sampled DIAG on vertex 4")
        if int(acts.op_type[0]) == 1:            # OP_COMPRESS
            assert comp[int(acts.i[0])], "sampled a masked-out compress axis"


def test_sample_and_evaluate_agree_under_the_same_masks(nn256):
    """PPO needs ratio == 1 at epoch 0, so the loss must mask identically."""
    policy, tables, feats, ctx = _policy_bits(nn256)
    pair, comp = _oracle(nn256).vertex_mask(4)
    pv = jnp.asarray(pair.astype(np.float32))
    cv = jnp.asarray(comp.astype(np.float32))
    acts, lp, *_ = policy.sample(ctx, feats, tables, jrand.PRNGKey(11),
                                 pair_valid=pv, compress_valid=cv)
    lp2, *_ = policy.evaluate(ctx, feats, tables, acts,
                              pair_valid=pv, compress_valid=cv)
    assert float(lp) == pytest.approx(float(lp2), abs=1e-5)


def test_unmasked_call_is_unchanged(nn256):
    """Omitting the masks must reproduce the pre-fix behaviour bit-for-bit."""
    policy, tables, feats, ctx = _policy_bits(nn256)
    a1, lp1, *_ = policy.sample(ctx, feats, tables, jrand.PRNGKey(5))
    a2, lp2, *_ = policy.sample(ctx, feats, tables, jrand.PRNGKey(5),
                                pair_valid=None, compress_valid=None)
    assert float(lp1) == float(lp2)
    assert int(a1.op_type[0]) == int(a2.op_type[0])
