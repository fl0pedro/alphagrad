# -*- coding: utf-8 -*-
"""--face-endpoint-read (docs/FACE_LATENT_INFO_LOSS.md section 4).

The face head's input becomes ``[chunk_mean || slot_i || slot_j]`` -- the
chunk-mean latent concatenated with the face's two ENDPOINT slot rows,
``read(base+dyn)[face_endpoints-1]`` with a zero row for endpoint 0. Pins:

1. ``test_flag_off_is_bitwise_blind_to_endpoint_rows`` -- with the flag OFF
   (the v62 configuration), handing the face path garbage endpoint rows
   changes NOTHING: draws, wires, log-probs, entropies bit-identical. This
   is the v62-reproducibility pin: the disabled path must not read the new
   input at all.
2. ``test_flag_on_widens_only_the_face_head`` -- the flag-on agent differs
   from the flag-off agent ONLY in the face head's first (widened) layer;
   every other parameter is bit-identical (the key stream is untouched, so
   seeded runs with the flag off reproduce v62 exactly).
3. ``test_flag_on_rollout_equals_replay`` -- the loss-side `_face_replay`
   handed the SAME endpoint rows + stored endpoints reproduces the sampled
   joint log-prob (the ratio-1-at-epoch-0 contract, extended to the
   concatenated input). A control run with DIFFERENT endpoint rows must
   change the logp -- proving the head actually reads the endpoint half.
4. ``test_flag_on_probe_receives_concat_width`` -- `_face_replay`'s 4th
   return (the var probe's face input) is the (F, 3E) concatenation, its
   endpoint halves gathered from the stored ids, and a
   ``VarProbes(face_in_dim=3E)`` decodes it to a finite loss.
5. ``test_flag_on_requires_rows`` -- an endpoint_read policy without rows
   fails loudly (the az_gumbel guard), never silently zero-fills.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
# Read by env.py at import, so set before any alphagrad import.
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "64")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "128")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrand  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import equinox as eqx  # noqa: E402

MAXF = 64
TOTAL_V = 6
EMBD = 32
W = int(os.environ["ALPHAGRAD_MAX_DELTA_TOKENS"])


# ------------------------------------------------------------------ fixtures
def _agent(endpoint_read):
    from alphagrad.approx.common.agent_factory import (
        apply_policy_arch, build_and_init_agent)
    from alphagrad.approx import ppo as P

    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=False,
        face_actions=True, unified_face_head=True, live_faces=True,
        max_substeps=1, axis_group_embedding=False)
    ns.vocab_size = 64
    ns.embd_dim = EMBD
    ns.num_heads = 2
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.face_endpoint_read = bool(endpoint_read)
    nfac = 4
    mrules = 4
    return build_and_init_agent(ns, TOTAL_V, nfac, mrules, seed=11)


def _warm_quant_scan():
    from graphax.sparse.micro_actions import report_hardware_scan
    report_hardware_scan()


@pytest.fixture(scope="module")
def agent_off():
    _warm_quant_scan()
    a = _agent(False)
    assert a.face_path_policy is not None
    assert not a.face_path_policy.endpoint_read
    return a


@pytest.fixture(scope="module")
def agent_on():
    _warm_quant_scan()
    a = _agent(True)
    assert a.face_path_policy.endpoint_read
    return a


def _decision_inputs(agent):
    from alphagrad.approx.common import carry_stream as _cs
    from alphagrad.approx.env import MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM
    from alphagrad.approx.heads import NUM_OPS, precompute_factor_tables

    toks = jrand.randint(jrand.PRNGKey(1), (48,), 1, 60)
    eqns = jnp.repeat(jnp.arange(8), 6)
    enc, vs, vc = _cs.init_carry(
        agent, toks, eqns, 48, window=48, total_v=TOTAL_V, embd_dim=EMBD)
    pre = agent.heads_from_memory(vs, vc)
    avail = jnp.zeros((TOTAL_V,), jnp.float32).at[2].set(1.0)  # force v=3
    ax_state = jnp.zeros(
        (TOTAL_V, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), jnp.int32)
    ax_state = ax_state.at[..., 0].set(8)
    ax_mask = jnp.zeros(
        (TOTAL_V, MAX_AXES_PER_VERTEX), jnp.float32).at[:, :3].set(1.0)
    ft = precompute_factor_tables(16)
    ovr = jnp.ones((NUM_OPS,), jnp.float32)
    return enc, vs, vc, pre, avail, ax_state, ax_mask, ft, ovr


def _chunk_fns(n):
    """Deterministic LiveFaceStream stand-ins (face_bucket_test's, verbatim:
    face f reads (f % 3) + 1 tokens, endpoints ((f%3)+1, (f%2)+1))."""

    def face_chunk_fn(f, vertex_idx, vertex_specs, rows, skips):
        ct = (f % 3) + 1
        ar = jnp.arange(W, dtype=jnp.int32)
        tok = jnp.where(ar < ct, (f + ar) % 50 + 1, 0).astype(jnp.int32)
        eqn = jnp.where(ar < ct, f.astype(jnp.int32), -1).astype(jnp.int32)
        ends = jnp.stack([(f % 3) + 1, (f % 2) + 1]).astype(jnp.int32)
        return tok, eqn, jnp.asarray(ct, jnp.int32), ends

    def face_count_fn(vertex_idx):
        return jnp.asarray(n, jnp.int32)

    return face_chunk_fn, face_count_fn


def _run_face_path(agent, n, key, endpoint_rows=None):
    enc, vs, vc, pre, avail, ax_state, ax_mask, ft, ovr = (
        _decision_inputs(agent))
    chunk_fn, count_fn = _chunk_fns(n)
    (vertex_idx, actions, _vd, _od, _id, _jd, _ed, _kd, _qlp, _vp, _vc,
     face_out, value, v_ctx) = agent.sample_action_dynamic(
        None, avail, ax_state, ax_mask, ft, ovr, key,
        precomputed=pre, enc_carry=enc,
        face_chunk_fn=chunk_fn, face_count_fn=count_fn,
        endpoint_rows=endpoint_rows)
    (fa, face_logp, face_ent, f_pair, f_comp, f_valid, f_cnt, f_dt, f_de,
     f_ends) = face_out
    return dict(
        v=int(vertex_idx), fa=fa, logp=np.asarray(face_logp),
        ent=np.asarray(face_ent), f_pair=f_pair, f_comp=f_comp,
        f_valid=f_valid, f_cnt=f_cnt, f_dt=f_dt, f_de=f_de, f_ends=f_ends,
        enc=enc, ax_state=ax_state, ax_mask=ax_mask, ft=ft, ovr=ovr)


def _rand_rows(seed, scale=1.0):
    return scale * jrand.normal(jrand.PRNGKey(seed), (TOTAL_V + 1, EMBD),
                                jnp.float32)


# ------------------------------------------------- 1. flag-off bit identity
def test_flag_off_is_bitwise_blind_to_endpoint_rows(agent_off):
    key = jrand.PRNGKey(7)
    a = _run_face_path(agent_off, 3, key, endpoint_rows=None)
    b = _run_face_path(agent_off, 3, key, endpoint_rows=_rand_rows(0, 100.0))
    assert a["v"] == b["v"]
    assert np.array_equal(a["logp"], b["logp"])
    assert np.array_equal(a["ent"], b["ent"])
    for name in a["fa"]._fields:
        assert np.array_equal(np.asarray(getattr(a["fa"], name)),
                              np.asarray(getattr(b["fa"], name))), name
    # ... and the loss-side replay is equally blind to garbage endpoint args.
    from alphagrad.approx.ppo import _axis_features_from_state
    feats = _axis_features_from_state(a["ax_state"][a["v"]],
                                      a["ax_mask"][a["v"]])
    lp0, e0, ar0, lat0 = agent_off._face_replay(
        feats, a["ft"], a["fa"], a["f_pair"], a["f_comp"], a["f_valid"],
        a["enc"], (a["f_cnt"], a["f_dt"], a["f_de"]), a["ovr"])
    lp1, e1, ar1, lat1 = agent_off._face_replay(
        feats, a["ft"], a["fa"], a["f_pair"], a["f_comp"], a["f_valid"],
        a["enc"], (a["f_cnt"], a["f_dt"], a["f_de"]), a["ovr"],
        endpoint_rows=_rand_rows(1, 100.0), face_ends=a["f_ends"])
    assert np.array_equal(np.asarray(lp0), np.asarray(lp1))
    assert np.array_equal(np.asarray(e0), np.asarray(e1))
    assert np.array_equal(np.asarray(lat0), np.asarray(lat1))
    assert lat0.shape == (MAXF, EMBD)


# --------------------------------------- 2. only the face head widens
def test_flag_on_widens_only_the_face_head(agent_off, agent_on):
    # head width: E -> 3E on the first (trunk) layer.
    def _in_width(agent):
        h = agent.face_path_policy.head
        leaves = [x for x in jax.tree_util.tree_leaves(eqx.filter(
            h, eqx.is_array)) if x.ndim == 2]
        return max(x.shape[1] for x in leaves)

    assert _in_width(agent_off) == EMBD
    assert _in_width(agent_on) == 3 * EMBD
    # every parameter OUTSIDE the face head is bit-identical: the key
    # stream did not move, so flag-off seeded runs reproduce v62.
    strip = lambda a: eqx.tree_at(  # noqa: E731
        lambda t: t.face_path_policy.head, a, None)
    la = jax.tree_util.tree_leaves(eqx.filter(strip(agent_off), eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(strip(agent_on), eqx.is_array))
    assert len(la) == len(lb) and len(la) > 0
    for x, y in zip(la, lb):
        assert x.shape == y.shape and np.array_equal(
            np.asarray(x), np.asarray(y))


# --------------------------------------------- 3. rollout == replay (flag on)
@pytest.mark.parametrize("n", [1, 3, 5])
def test_flag_on_rollout_equals_replay(agent_on, n):
    from alphagrad.approx.ppo import _axis_features_from_state
    rows = _rand_rows(42)
    key = jrand.PRNGKey(200 + n)
    r = _run_face_path(agent_on, n, key, endpoint_rows=rows)
    feats = _axis_features_from_state(r["ax_state"][r["v"]],
                                      r["ax_mask"][r["v"]])
    lp, ent, ar, lat = agent_on._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        endpoint_rows=rows, face_ends=r["f_ends"])
    np.testing.assert_allclose(np.asarray(lp), r["logp"], rtol=1e-5,
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(ent), r["ent"], rtol=1e-5,
                               atol=1e-6)
    # control: DIFFERENT endpoint rows must move the replayed logp -- the
    # head actually reads the endpoint half (all stand-in endpoints > 0).
    lp2, _, _, _ = agent_on._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        endpoint_rows=_rand_rows(43, 10.0), face_ends=r["f_ends"])
    assert not np.allclose(np.asarray(lp2), r["logp"], rtol=1e-5, atol=1e-6)


# ------------------------------------------------- 4. probe input width
def test_flag_on_probe_receives_concat_width(agent_on):
    from alphagrad.approx.ppo import _axis_features_from_state
    from alphagrad.approx.common import var_probe as VP
    rows = _rand_rows(5)
    r = _run_face_path(agent_on, 3, jrand.PRNGKey(9), endpoint_rows=rows)
    feats = _axis_features_from_state(r["ax_state"][r["v"]],
                                      r["ax_mask"][r["v"]])
    _, _, _, lat = agent_on._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        endpoint_rows=rows, face_ends=r["f_ends"])
    assert lat.shape == (MAXF, 3 * EMBD)
    # the endpoint halves ARE the gathered slot rows (1-based, 0 -> zero row)
    fe = np.asarray(r["f_ends"], np.int64)
    R = np.asarray(rows)
    lat_np = np.asarray(lat)
    for f in range(3):
        for s in range(2):
            want = R[fe[f, s] - 1] if fe[f, s] > 0 else np.zeros(EMBD)
            got = lat_np[f, EMBD * (1 + s):EMBD * (2 + s)]
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    # a 3E-wide VarProbes decodes this input to a finite loss.
    probes = VP.VarProbes(embd_dim=EMBD, face_in_dim=3 * EMBD,
                          key=jax.random.PRNGKey(0))
    m = 4
    tgt = np.zeros((m, VP.N_SLOTS, VP.TGT_COLS), np.float32)
    val = np.ones((m, VP.N_SLOTS), np.float32)
    for i in range(m):
        for s in range(VP.N_SLOTS):
            tgt[i, s] = VP.encode_var((16, 10), "float32")
    loss, _aux = VP.var_probe_loss(
        probes, jnp.asarray(lat_np[:m]), jnp.asarray(tgt), jnp.asarray(val),
        jnp.asarray(lat_np[:m, :EMBD]), jnp.asarray(tgt), jnp.asarray(val))
    assert np.isfinite(float(loss))


# ------------------------------------------------- 5. loud failure, no rows
def test_flag_on_requires_rows(agent_on):
    with pytest.raises(ValueError, match="endpoint_rows"):
        _run_face_path(agent_on, 2, jrand.PRNGKey(3), endpoint_rows=None)
