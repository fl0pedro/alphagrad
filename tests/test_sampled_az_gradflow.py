"""Sampled-AZ loss gradient flow -- the check that caught the frozen micro
head: AZ's approximation head once sat at init for a whole campaign because
its log-prob never entered a loss.

Builds the REAL agent (the same factory call az_gumbel makes), fabricates a
tiny batch (no env, no measurement), runs the Sampled-AZ loss pieces --
vertex CE to an improved-policy target + `face_ce_term` through
`Agent._face_replay` -- and asserts non-zero gradient reaches BOTH the
face head and the vertex head (and the shared palimpsa encoder, which the
emission-window replay must touch: "Gradient reaches palimpsa through this
scan").
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrand  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx import ppo as P  # noqa: E402
from alphagrad.approx.common import carry_stream as _cs  # noqa: E402
from alphagrad.approx.common.agent_factory import (  # noqa: E402
    apply_policy_arch, build_and_init_agent)
from alphagrad.approx.common.sampled_az import face_ce_term  # noqa: E402
from alphagrad.approx.env import (  # noqa: E402
    FACE_SLOTS, MAX_AXES_PER_VERTEX, MAX_DELTA_TOKENS)
from alphagrad.approx.heads import (  # noqa: E402
    NUM_OPS, precompute_factor_tables)

TOTAL_V = 12
SEED = 5
N = MAX_AXES_PER_VERTEX


def _ns():
    ns = P.make_argparser().parse_args([])
    # az_gumbel's exact head surface (agent_factory call, W1/W5 parity).
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=False,
        face_actions=True, unified_face_head=True, live_faces=True,
        max_substeps=1, axis_group_embedding=False)
    ns.seed = SEED
    return ns


@pytest.fixture(scope="module")
def setup():
    ns = _ns()
    tbl, tpy, nfac, mrules = P._build_factor_table(ns)
    agent = build_and_init_agent(ns, TOTAL_V, nfac, mrules, seed=SEED)
    assert agent.face_path_policy is not None
    tables = precompute_factor_tables(ns.max_axis_size)

    # Fabricated axis state, layout per _axis_features_from_state's
    # docstring: [size, is_output, is_compressed, group_id].
    axis_state = np.zeros((TOTAL_V, N, 4), np.int32)
    axis_state[:, :, 0] = 4                      # sizes
    axis_state[:, :, 3] = -1                     # no diag groups
    axis_valid = np.zeros((TOTAL_V, N), np.int32)
    axis_valid[:, :2] = 1                        # two live axes per vertex
    axis_state = jnp.asarray(axis_state)
    axis_valid = jnp.asarray(axis_valid)

    # A real carry off a tiny fabricated base stream.
    W0 = 16
    base_tok = jnp.asarray((np.arange(W0) % 7 + 1).astype(np.int32))
    base_eqn = jnp.zeros((W0,), jnp.int32)
    enc, vs, vc = _cs.init_carry(
        agent, base_tok, base_eqn, W0, window=W0, total_v=TOTAL_V,
        embd_dim=ns.embd_dim, base_owners=None)

    F = agent.face_path_policy.max_faces
    S = FACE_SLOTS
    pair = jnp.zeros((F, N, N)).at[:, 0, 1].set(1.0).at[:, 1, 0].set(1.0)
    comp = jnp.zeros((F, N)).at[:, 0].set(1.0)
    valid = jnp.zeros((F,)).at[0].set(1.0).at[1].set(1.0)

    # Two real draws sampled from the head itself (F ~ beta), one padding.
    feats0 = P._axis_features_from_state(axis_state[0], axis_valid[0])
    ctx0 = jnp.ones((ns.embd_dim,)) * 0.1
    fa0, *_ = agent.face_path_policy.sample(
        ctx0, feats0, tables, jrand.PRNGKey(0), pair, comp, valid)
    fa1, *_ = agent.face_path_policy.sample(
        ctx0, feats0, tables, jrand.PRNGKey(1), pair, comp, valid)
    fa_pad = jax.tree_util.tree_map(jnp.zeros_like, fa0)

    D, W = 3, MAX_DELTA_TOKENS
    sd_li = jnp.asarray([0, 1, -1], jnp.int32)
    sd_vidx = jnp.asarray([0, 1, 0], jnp.int32)
    sd_w = jnp.asarray([0.7, 1.0, 0.0], jnp.float32)
    sd_fpair = jnp.stack([pair, pair, jnp.zeros_like(pair)])
    sd_fcomp = jnp.stack([comp, comp, jnp.zeros_like(comp)])
    sd_fvalid = jnp.stack([valid, valid, jnp.zeros_like(valid)])
    cnt = np.zeros((D, F), np.int32)
    cnt[0, 0], cnt[0, 1] = 3, 2                  # draw 0 read 5 chunk tokens
    cnt[1, 0] = 4                                # draw 1 read 4
    sd_cnt = jnp.asarray(cnt)
    dt = np.zeros((D, W), np.int32)
    dt[0, :5] = [1, 2, 3, 4, 5]
    dt[1, :4] = [2, 3, 4, 5]
    de = np.full((D, W), -1, np.int32)
    de[0, :5] = 0
    de[1, :4] = 0
    sd_dt, sd_de = jnp.asarray(dt), jnp.asarray(de)
    sd_fa = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), fa0, fa1, fa_pad)

    n_legal = 6
    pi_np = np.zeros((TOTAL_V,), np.float32)
    pi_np[:n_legal] = 1.0 / n_legal
    return dict(
        ns=ns, agent=agent, tables=tables, axis_state=axis_state,
        axis_valid=axis_valid, enc=enc, vs=vs, vc=vc,
        la=jnp.arange(n_legal, dtype=jnp.int32),
        pi=jnp.asarray(pi_np[:n_legal]), pi_pad=jnp.asarray(pi_np),
        sd=(sd_li, sd_vidx, sd_w, sd_fpair, sd_fcomp, sd_fvalid,
            sd_cnt, sd_dt, sd_de, sd_fa,
            # endpoint ids per draw: face 0 -> (1, 2), rest "no vertex"
            jnp.asarray(np.pad(np.array([[[1, 2]]] * D, np.int32),
                               ((0, 0), (0, F - 1), (0, 0)))),),
        op_override=jnp.ones((NUM_OPS,), jnp.float32))


def _loss_pieces(agent, s):
    vlog, ctx, _v3 = _cs.heads(agent, s["vs"], s["vc"],
                               vertex_features=None)
    logp = jax.nn.log_softmax(vlog[s["la"]])
    vertex_ce = -jnp.sum(s["pi"] * logp)
    face_ce, ent = face_ce_term(
        agent._face_replay, ctx, s["enc"], s["axis_state"], s["axis_valid"],
        s["tables"], s["op_override"], P._axis_features_from_state,
        s["pi_pad"], *s["sd"])
    return vertex_ce, face_ce, ent


def _norm(tree):
    leaves = [x for x in jax.tree_util.tree_leaves(
        eqx.filter(tree, eqx.is_inexact_array)) if x is not None]
    return float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves)))


def test_gradient_reaches_face_head_vertex_head_and_encoder(setup):
    s = setup

    def loss(agent):
        vce, fce, _ent = _loss_pieces(agent, s)
        return vce + fce

    grads = eqx.filter_grad(loss)(s["agent"])
    g_face = _norm(grads.face_path_policy)
    g_vertex = _norm(grads.vertex_policy)
    g_enc = _norm(grads.encoder)
    assert g_face > 0.0, "face CE reaches no face-head parameter"
    assert g_vertex > 0.0, "vertex CE reaches no vertex-head parameter"
    assert g_enc > 0.0, ("no gradient reaches the palimpsa encoder -- the "
                         "emission-window replay must touch it")
    assert np.isfinite(g_face) and np.isfinite(g_vertex) and np.isfinite(g_enc)


def test_face_ce_alone_reaches_face_head(setup):
    """The frozen-head regression in its sharpest form: the CE term ALONE
    must move the face head."""
    s = setup

    def loss(agent):
        _vce, fce, _ent = _loss_pieces(agent, s)
        return fce

    grads = eqx.filter_grad(loss)(s["agent"])
    assert _norm(grads.face_path_policy) > 0.0


def test_padding_slots_contribute_exactly_zero(setup):
    """All-padding draws (li == -1, w == 0) must yield CE == 0 -- the fixed
    D_MAX_DRAWS pads may never leak into the loss."""
    s = setup
    (sd_li, sd_vidx, sd_w, *rest) = s["sd"]
    sd0 = (jnp.full_like(sd_li, -1), sd_vidx, jnp.zeros_like(sd_w), *rest)
    _vlog, ctx, _v3 = _cs.heads(s["agent"], s["vs"], s["vc"],
                                vertex_features=None)
    ce, _ent = face_ce_term(
        s["agent"]._face_replay, ctx, s["enc"], s["axis_state"],
        s["axis_valid"], s["tables"], s["op_override"],
        P._axis_features_from_state, s["pi_pad"], *sd0)
    assert float(ce) == 0.0
