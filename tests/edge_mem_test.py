# -*- coding: utf-8 -*-
"""--face-edge-mem (docs/FACE_LATENT_INFO_LOSS.md section 8).

The face head's input gains the face's two EDGE-KEYED memory rows:
``[chunk || (vmem_i||vmem_j)? || (emem_lhs||emem_rhs)?]`` = E / 3E / 5E.
The edge memory is the SAME parameter-free scatter as the vertex memory,
keyed by the res-edge id each face creates; the host assigns slots on first
emission (EdgeSlotTable) and the write rides `carry_stream.advance`'s fold.
Pins (template: tests/endpoint_read_test.py):

1. ``test_flag_off_is_bitwise_blind_to_edge_rows`` -- with the flag OFF,
   garbage edge rows / slots change NOTHING (v63-reproducibility pin).
2. ``test_flag_on_widens_only_the_face_head`` -- 3E with edge-mem alone,
   5E with both flags; every parameter outside the face head bit-identical.
3. ``test_slot_table_determinism_and_eviction`` -- EdgeSlotTable: identical
   assignment across fresh replays, evict-oldest past capacity with the
   `evictions` telemetry counter, reset on step regression, lookup never
   assigns and counts `nonzero_reads`.
4. ``test_edge_write_ids_spans`` -- `_edge_write_ids` recovers the
   `last_face_segments` tiling (contraction_f + approx-echo_f) from the
   stored (counts, heads): the offline arm's exact attribution.
5. ``test_write_then_read_binding`` -- THE BINDING PROPERTY the feature
   exists for: a delta scattered by res-edge slot through
   `carry_stream.advance(edge_mem=...)`, then a later face whose lhs IS
   that res edge reads back a NONZERO row equal to the mean of exactly the
   creation event's rows (and a -1 slot reads a zero row -- the
   primitive-operand convention).
6. ``test_flag_on_rollout_equals_replay`` (edge alone and BOTH flags at
   5E=160-wide input) -- `_face_replay` handed the same edge rows + stored
   slots reproduces the sampled joint log-prob; different rows move it.
7. ``test_flag_on_probe_receives_concat_width`` -- the replay's 4th return
   is the (F, 3E) concatenation with the emem halves gathered from the
   stored slots, and a matching VarProbes decodes it to a finite loss.
8. ``test_flag_on_requires_rows`` -- edge_mem without rows fails loudly
   (the az_gumbel guard), never silently zero-fills.
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
KE = MAXF  # edge-memory capacity = the face bound


# ------------------------------------------------------------------ fixtures
def _agent(edge_mem, endpoint_read=False):
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
    ns.face_edge_mem = bool(edge_mem)
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
    assert not a.face_path_policy.edge_mem
    return a


@pytest.fixture(scope="module")
def agent_on():
    _warm_quant_scan()
    a = _agent(True)
    assert a.face_path_policy.edge_mem
    return a


@pytest.fixture(scope="module")
def agent_both():
    _warm_quant_scan()
    a = _agent(True, endpoint_read=True)
    assert a.face_path_policy.edge_mem
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


def _chunk_fns(n, with_einfo):
    """Deterministic LiveFaceStream stand-ins (endpoint_read_test's,
    plus the --face-edge-mem einfo wire: face f reads lhs slot f%3 (>= 0,
    so the read half is LIVE), rhs slot -1 for odd f (the zero-row
    convention), res slot f, approx-head length f%2)."""

    def face_chunk_fn(f, vertex_idx, vertex_specs, rows, skips):
        ct = (f % 3) + 1
        ar = jnp.arange(W, dtype=jnp.int32)
        tok = jnp.where(ar < ct, (f + ar) % 50 + 1, 0).astype(jnp.int32)
        eqn = jnp.where(ar < ct, f.astype(jnp.int32), -1).astype(jnp.int32)
        ends = jnp.stack([(f % 3) + 1, (f % 2) + 1]).astype(jnp.int32)
        if not with_einfo:
            return tok, eqn, jnp.asarray(ct, jnp.int32), ends
        ei = jnp.stack([
            f % 3,
            jnp.where(f % 2 == 1, -1, (f + 1) % 4),
            f,
            f % 2,
        ]).astype(jnp.int32)
        return tok, eqn, jnp.asarray(ct, jnp.int32), ends, ei

    def face_count_fn(vertex_idx):
        return jnp.asarray(n, jnp.int32)

    return face_chunk_fn, face_count_fn


def _run_face_path(agent, n, key, edge_rows=None, endpoint_rows=None,
                   with_einfo=None):
    if with_einfo is None:
        with_einfo = bool(getattr(agent.face_path_policy, "edge_mem", False))
    enc, vs, vc, pre, avail, ax_state, ax_mask, ft, ovr = (
        _decision_inputs(agent))
    chunk_fn, count_fn = _chunk_fns(n, with_einfo)
    (vertex_idx, actions, _vd, _od, _id, _jd, _ed, _kd, _qlp, _vp, _vc,
     face_out, value, v_ctx) = agent.sample_action_dynamic(
        None, avail, ax_state, ax_mask, ft, ovr, key,
        precomputed=pre, enc_carry=enc,
        face_chunk_fn=chunk_fn, face_count_fn=count_fn,
        endpoint_rows=endpoint_rows, edge_rows=edge_rows)
    (fa, face_logp, face_ent, f_pair, f_comp, f_valid, f_cnt, f_dt, f_de,
     f_ends) = face_out[:10]
    out = dict(
        v=int(vertex_idx), fa=fa, logp=np.asarray(face_logp),
        ent=np.asarray(face_ent), f_pair=f_pair, f_comp=f_comp,
        f_valid=f_valid, f_cnt=f_cnt, f_dt=f_dt, f_de=f_de, f_ends=f_ends,
        enc=enc, ax_state=ax_state, ax_mask=ax_mask, ft=ft, ovr=ovr)
    if len(face_out) > 10:
        out["f_eslots"], out["f_ewr"] = face_out[10], face_out[11]
    return out


def _edge_rows(seed, scale=1.0):
    return scale * jrand.normal(jrand.PRNGKey(seed), (KE, EMBD), jnp.float32)


def _ep_rows(seed, scale=1.0):
    return scale * jrand.normal(jrand.PRNGKey(seed), (TOTAL_V + 1, EMBD),
                                jnp.float32)


# ------------------------------------------------- 1. flag-off bit identity
def test_flag_off_is_bitwise_blind_to_edge_rows(agent_off):
    key = jrand.PRNGKey(7)
    a = _run_face_path(agent_off, 3, key, edge_rows=None, with_einfo=False)
    b = _run_face_path(agent_off, 3, key, edge_rows=_edge_rows(0, 100.0),
                       with_einfo=False)
    assert a["v"] == b["v"]
    assert np.array_equal(a["logp"], b["logp"])
    assert np.array_equal(a["ent"], b["ent"])
    for name in a["fa"]._fields:
        assert np.array_equal(np.asarray(getattr(a["fa"], name)),
                              np.asarray(getattr(b["fa"], name))), name
    # ... and the loss-side replay is equally blind to garbage edge args.
    from alphagrad.approx.ppo import _axis_features_from_state
    feats = _axis_features_from_state(a["ax_state"][a["v"]],
                                      a["ax_mask"][a["v"]])
    lp0, e0, ar0, lat0 = agent_off._face_replay(
        feats, a["ft"], a["fa"], a["f_pair"], a["f_comp"], a["f_valid"],
        a["enc"], (a["f_cnt"], a["f_dt"], a["f_de"]), a["ovr"])
    lp1, e1, ar1, lat1 = agent_off._face_replay(
        feats, a["ft"], a["fa"], a["f_pair"], a["f_comp"], a["f_valid"],
        a["enc"], (a["f_cnt"], a["f_dt"], a["f_de"]), a["ovr"],
        edge_rows=_edge_rows(1, 100.0),
        face_eslots=jnp.zeros((MAXF, 2), jnp.int32))
    assert np.array_equal(np.asarray(lp0), np.asarray(lp1))
    assert np.array_equal(np.asarray(e0), np.asarray(e1))
    assert np.array_equal(np.asarray(lat0), np.asarray(lat1))
    assert lat0.shape == (MAXF, EMBD)


# --------------------------------------- 2. only the face head widens
def test_flag_on_widens_only_the_face_head(agent_off, agent_on, agent_both):
    def _in_width(agent):
        h = agent.face_path_policy.head
        leaves = [x for x in jax.tree_util.tree_leaves(eqx.filter(
            h, eqx.is_array)) if x.ndim == 2]
        return max(x.shape[1] for x in leaves)

    assert _in_width(agent_off) == EMBD
    assert _in_width(agent_on) == 3 * EMBD
    # BOTH flags: [chunk || vmem_i || vmem_j || emem_lhs || emem_rhs] = 5E
    # (= 160 at the campaign's E=32).
    assert _in_width(agent_both) == 5 * EMBD
    # every parameter OUTSIDE the face head is bit-identical: the key
    # stream did not move, so flag-off seeded runs reproduce v63.
    strip = lambda a: eqx.tree_at(  # noqa: E731
        lambda t: t.face_path_policy.head, a, None)
    la = jax.tree_util.tree_leaves(eqx.filter(strip(agent_off), eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(strip(agent_on), eqx.is_array))
    assert len(la) == len(lb) and len(la) > 0
    for x, y in zip(la, lb):
        assert x.shape == y.shape and np.array_equal(
            np.asarray(x), np.asarray(y))


# ------------------------------- 3. slot table: determinism + eviction
def test_slot_table_determinism_and_eviction():
    from alphagrad.approx.common.face_driver import EdgeSlotTable

    def drive(t):
        out = []
        t.begin(0, 0)
        for k in [(1, 5), (2, 5), (1, 5), (3, 7), (4, 7), (5, 8)]:
            out.append(t.assign(0, k))
        return out

    t1, t2 = EdgeSlotTable(4), EdgeSlotTable(4)
    s1, s2 = drive(t1), drive(t2)
    # Deterministic across fresh replays.
    assert s1 == s2
    # First 4 distinct keys fill 0..3; the 5th distinct key evicts the
    # OLDEST ((1,5)) and reuses its slot.
    assert s1[:4] == [0, 1, 0, 2]
    assert s1[4] == 3
    assert s1[5] == 0
    assert t1.stats["evictions"] == 1
    # Lookup never assigns; hits count as nonzero_reads.
    assert t1.lookup(0, [(9, 9)]) == -1
    n0 = t1.stats["nonzero_reads"]
    assert t1.lookup(0, [(9, 9), (3, 7)]) == 2
    assert t1.stats["nonzero_reads"] == n0 + 1
    # Step regression = new episode = fresh table.
    t1.begin(0, 5)
    t1.begin(0, 0)
    assert t1.stats["resets"] == 1
    assert t1.lookup(0, [(3, 7)]) == -1
    # Per-env isolation.
    t1.begin(1, 0)
    assert t1.assign(1, (3, 7)) == 0


# ------------------------------------------------- 4. write-id spans
def test_edge_write_ids_spans():
    from alphagrad.approx.ppo import _edge_write_ids

    F = 8
    cnt = np.zeros((F,), np.int32)
    head = np.zeros((F,), np.int32)
    slot = -np.ones((F,), np.int32)
    # face 0: chunk = 3 tokens, no echo prefix; face 1: chunk = 2 tokens,
    # 1 of which is face 0's approx echo. Emission length 6, so face 0's
    # true span is [0, 3+1) = contraction_0 + echo_0, face 1's is [4, 6)
    # (its own echo is the tail no chunk contains).
    cnt[:2] = [3, 2]
    head[:2] = [0, 1]
    slot[:2] = [5, 9]
    ids = np.asarray(_edge_write_ids(
        jnp.asarray(cnt), jnp.asarray(head), jnp.asarray(slot),
        jnp.asarray(2), jnp.asarray(6), W))
    want = -np.ones((W,), np.int32)
    want[0:4] = 5
    want[4:6] = 9
    assert np.array_equal(ids, want)
    # No live faces -> nothing written.
    ids0 = np.asarray(_edge_write_ids(
        jnp.asarray(cnt), jnp.asarray(head), jnp.asarray(slot),
        jnp.asarray(0), jnp.asarray(6), W))
    assert np.all(ids0 == -1)
    # -1 res slot (a dropped face) writes nothing for its span.
    slot2 = slot.copy()
    slot2[0] = -1
    ids2 = np.asarray(_edge_write_ids(
        jnp.asarray(cnt), jnp.asarray(head), jnp.asarray(slot2),
        jnp.asarray(2), jnp.asarray(6), W))
    assert np.all(ids2[0:4] == -1) and np.all(ids2[4:6] == 9)


# --------------------------------------- 5. write -> read binding
def test_write_then_read_binding(agent_on):
    """A second face's lhs IS the first face's res edge: its emem_lhs row
    must be nonzero and equal the scattered rows of the creation event."""
    from alphagrad.approx.common import carry_stream as _cs
    from alphagrad.approx import vertex_memory as _vmem
    from alphagrad.approx.ppo import (_axis_features_from_state,
                                      _edge_write_ids)

    agent = agent_on
    enc, vs, vc, pre, avail, ax_state, ax_mask, ft, ovr = (
        _decision_inputs(agent))
    # One step's emission: 6 tokens, two faces with the spans of
    # test_edge_write_ids_spans (slots 5 and 9).
    nd = 6
    dt = jnp.zeros((W,), jnp.int32).at[:nd].set(
        jnp.asarray([7, 8, 9, 10, 11, 12]))
    de = (-jnp.ones((W,), jnp.int32)).at[:nd].set(0)
    cnt = jnp.zeros((MAXF,), jnp.int32).at[0].set(3).at[1].set(2)
    head = jnp.zeros((MAXF,), jnp.int32).at[1].set(1)
    # NOTE: parenthesised -- unary minus binds LOOSER than .at, so
    # ``-jnp.ones(...).at[0].set(5)`` would negate the 5 too.
    slot = (-jnp.ones((MAXF,), jnp.int32)).at[0].set(5).at[1].set(9)
    ids = _edge_write_ids(cnt, head, slot, jnp.asarray(2), jnp.asarray(nd), W)
    es0, ec0 = _cs.zero_edge_memory(KE, EMBD)
    part = jnp.zeros((TOTAL_V + 1,), jnp.float32).at[0].set(1.0)
    _c2, _s2, _n2, es1, ec1 = _cs.advance(
        agent, enc, vs, vc, dt, de, jnp.asarray(nd), jnp.asarray(0),
        window=W, participants=part, edge_mem=(es0, ec0), edge_ids=ids)
    # Reference: the SAME encode, rows pooled by hand over the true spans.
    _c2r, rows, valid, _e = agent.encode_extend(
        enc, dt, de, jnp.asarray(nd), window=W, start=0, chunk=0)
    rows = np.asarray(rows)
    want5 = rows[0:4].mean(0)
    want9 = rows[4:6].mean(0)
    emem = np.asarray(_vmem.read(es1, ec1))
    np.testing.assert_allclose(emem[5], want5, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(emem[9], want9, rtol=2e-5, atol=1e-6)
    assert float(np.abs(emem[5]).sum()) > 0.0
    # every unwritten slot reads exactly zero
    mask = np.ones((KE,), bool)
    mask[[5, 9]] = False
    assert np.all(emem[mask] == 0.0)

    # NOW the read: a later face whose lhs slot is 5 (= face 0's res edge)
    # and whose rhs slot is -1 (a primitive operand, never written).
    r = _run_face_path(agent, 2, jrand.PRNGKey(3),
                       edge_rows=jnp.asarray(emem))
    feats = _axis_features_from_state(r["ax_state"][r["v"]],
                                      r["ax_mask"][r["v"]])
    eslots = (-jnp.ones((MAXF, 2), jnp.int32)).at[0, 0].set(5)
    _lp, _e2, _ar, lat = agent._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        edge_rows=jnp.asarray(emem), face_eslots=eslots)
    lat = np.asarray(lat)
    assert lat.shape == (MAXF, 3 * EMBD)
    # emem_lhs half of face 0 == the creation event's pooled rows, nonzero.
    np.testing.assert_allclose(lat[0, EMBD:2 * EMBD], emem[5],
                               rtol=1e-6, atol=1e-7)
    assert float(np.abs(lat[0, EMBD:2 * EMBD]).sum()) > 0.0
    # rhs half (slot -1): the zero row.
    assert np.all(lat[0, 2 * EMBD:] == 0.0)


# --------------------------------------------- 6. rollout == replay
@pytest.mark.parametrize("n", [1, 3, 5])
def test_flag_on_rollout_equals_replay(agent_on, n):
    from alphagrad.approx.ppo import _axis_features_from_state
    rows = _edge_rows(42)
    key = jrand.PRNGKey(200 + n)
    r = _run_face_path(agent_on, n, key, edge_rows=rows)
    feats = _axis_features_from_state(r["ax_state"][r["v"]],
                                      r["ax_mask"][r["v"]])
    lp, ent, ar, lat = agent_on._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        edge_rows=rows, face_eslots=r["f_eslots"])
    np.testing.assert_allclose(np.asarray(lp), r["logp"], rtol=1e-5,
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(ent), r["ent"], rtol=1e-5,
                               atol=1e-6)
    # control: DIFFERENT edge rows must move the replayed logp -- the head
    # actually reads the edge half (every face's lhs slot is >= 0 in the
    # stand-ins).
    lp2, _, _, _ = agent_on._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        edge_rows=_edge_rows(43, 10.0), face_eslots=r["f_eslots"])
    assert not np.allclose(np.asarray(lp2), r["logp"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n", [2, 4])
def test_both_flags_rollout_equals_replay(agent_both, n):
    """The 160-wide (5E) composition: endpoint + edge reads together."""
    from alphagrad.approx.ppo import _axis_features_from_state
    erows = _edge_rows(11)
    prows = _ep_rows(12)
    key = jrand.PRNGKey(300 + n)
    r = _run_face_path(agent_both, n, key, edge_rows=erows,
                       endpoint_rows=prows)
    feats = _axis_features_from_state(r["ax_state"][r["v"]],
                                      r["ax_mask"][r["v"]])
    lp, ent, _ar, lat = agent_both._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        endpoint_rows=prows, face_ends=r["f_ends"],
        edge_rows=erows, face_eslots=r["f_eslots"])
    assert lat.shape == (MAXF, 5 * EMBD)
    np.testing.assert_allclose(np.asarray(lp), r["logp"], rtol=1e-5,
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(ent), r["ent"], rtol=1e-5,
                               atol=1e-6)
    # the emem halves sit AFTER the endpoint halves: [3E:4E] is the lhs row
    # of slot f%3 for face 0 (slot 0).
    lat = np.asarray(lat)
    np.testing.assert_allclose(lat[0, 3 * EMBD:4 * EMBD],
                               np.asarray(erows)[0], rtol=1e-6, atol=1e-7)


# ------------------------------------------------- 7. probe input width
def test_flag_on_probe_receives_concat_width(agent_on):
    from alphagrad.approx.ppo import _axis_features_from_state
    from alphagrad.approx.common import var_probe as VP
    rows = _edge_rows(5)
    r = _run_face_path(agent_on, 3, jrand.PRNGKey(9), edge_rows=rows)
    feats = _axis_features_from_state(r["ax_state"][r["v"]],
                                      r["ax_mask"][r["v"]])
    _, _, _, lat = agent_on._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"], r["f_de"]), r["ovr"],
        edge_rows=rows, face_eslots=r["f_eslots"])
    assert lat.shape == (MAXF, 3 * EMBD)
    # the edge halves ARE the gathered slot rows (-1 -> zero row).
    fs = np.asarray(r["f_eslots"], np.int64)
    R = np.asarray(rows)
    lat_np = np.asarray(lat)
    for f in range(3):
        for s in range(2):
            want = R[fs[f, s]] if fs[f, s] >= 0 else np.zeros(EMBD)
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


# ------------------------------------------------- 8. loud failure, no rows
def test_flag_on_requires_rows(agent_on):
    with pytest.raises(ValueError, match="edge_rows"):
        _run_face_path(agent_on, 2, jrand.PRNGKey(3), edge_rows=None)
