# -*- coding: utf-8 -*-
"""#109: bucket-compiled face candidate counts for the Gumbel-AZ search.

The az trainer's `_face_plan` used to run every face draw at the configured
MAX_FACES width (2538 on the TLM targets) although a vertex has ~1-8 live
faces. Bucketing pads each draw to the smallest width in {2, 4, 8, 32}
(fallback: the configured MAX_FACES) and jits once per bucket, via the face
head's STATIC ``max_faces``. This must be a pure padding/compile-granularity
change:

1. ``test_bucketed_draw_is_bitwise_identical`` -- the full face-decision path
   (`Agent.sample_action_dynamic` -> `_face_loop` -> `to_env_action_dynamic`)
   at full width vs at the bucket width, same key, for live counts
   {1, 2, 3, 5, 9}: identical draws, wires, log-probs, entropies, masks,
   chunk counts and emission windows (bucketed outputs re-padded with
   `pad_face_outputs`, which must reproduce the never-visited faces' bytes).
2. ``test_bucket_width_selection`` -- the bucket function itself.
3. ``test_compile_count_bounded`` -- after processing live counts
   {1, 2, 3, 5, 9, 40} (twice), at most 5 distinct shapes were compiled.

az_gumbel.py itself builds its env at import time and cannot be imported
here; the kill-switch (ALPHAGRAD_GAZ_FACE_BUCKETS=0) short-circuits before
any of the machinery tested here is touched.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
# The test's "configured MAX_FACES". Read by env.py at import, so it must be
# set before any alphagrad import.
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "64")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "128")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrand  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common.face_buckets import (  # noqa: E402
    FACE_BUCKETS, bucket_width, hist_face_width, pad_face_outputs,
    with_face_width)

MAXF = 64
TOTAL_V = 6
W = int(os.environ["ALPHAGRAD_MAX_DELTA_TOKENS"])


# ------------------------------------------------------------------ fixtures
def _agent():
    """The az head inventory (AZ_HEADS of test_ppo_az_parity), shrunk."""
    from alphagrad.approx.common.agent_factory import (
        apply_policy_arch, build_and_init_agent)
    from alphagrad.approx import ppo as P

    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=False,
        face_actions=True, unified_face_head=True, live_faces=True,
        max_substeps=1, axis_group_embedding=False)
    ns.vocab_size = 64
    ns.embd_dim = 32
    ns.num_heads = 2
    ns.num_layers = 2
    ns.hidden_dim = 32
    nfac = 4
    mrules = 4
    return build_and_init_agent(ns, TOTAL_V, nfac, mrules, seed=11)


def _warm_quant_scan():
    """az_gumbel warms the quant hardware scan EAGERLY before any trace (its
    tables are jnp probes; first-touch inside a while_loop body hands later
    traces a dead tracer -- the exact UnexpectedTracerError its comment
    documents). Same requirement here."""
    from graphax.sparse.micro_actions import report_hardware_scan
    report_hardware_scan()


@pytest.fixture(scope="module")
def agent():
    _warm_quant_scan()
    a = _agent()
    assert a.face_path_policy is not None
    assert int(a.face_path_policy.max_faces) == MAXF
    return a


def _decision_inputs(agent):
    """Everything one `_face_plan`-shaped call needs, deterministic."""
    from alphagrad.approx.common import carry_stream as _cs
    from alphagrad.approx.env import MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM
    from alphagrad.approx.heads import NUM_OPS, precompute_factor_tables

    EMBD = 32
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
    return enc, pre, avail, ax_state, ax_mask, ft, ovr


def _chunk_fns(n):
    """Deterministic stand-ins for the LiveFaceStream pure_callbacks: face f
    reads a (f-dependent) chunk of (f % 3) + 1 tokens; the count callback
    reports ``n`` live faces. Pure jnp, so the values cannot depend on the
    padded width."""

    def face_chunk_fn(f, vertex_idx, vertex_specs, rows, skips):
        ct = (f % 3) + 1
        ar = jnp.arange(W, dtype=jnp.int32)
        tok = jnp.where(ar < ct, (f + ar) % 50 + 1, 0).astype(jnp.int32)
        eqn = jnp.where(ar < ct, f.astype(jnp.int32), -1).astype(jnp.int32)
        # 4th output: the face's ENDPOINT vertices (1-based, 0 = none).
        # Deterministic stand-in, same shape the real callback returns.
        ends = jnp.stack([(f % 3) + 1, (f % 2) + 1]).astype(jnp.int32)
        return tok, eqn, jnp.asarray(ct, jnp.int32), ends

    def face_count_fn(vertex_idx):
        return jnp.asarray(n, jnp.int32)

    return face_chunk_fn, face_count_fn


def _run_face_path(agent, n, key):
    """One full face decision (eager -- op-for-op identical arithmetic, so
    equality below is exact, not tolerance-based)."""
    enc, pre, avail, ax_state, ax_mask, ft, ovr = _decision_inputs(agent)
    chunk_fn, count_fn = _chunk_fns(n)
    (vertex_idx, actions, _vd, _od, _id, _jd, _ed, _kd, _qlp, _vp, _vc,
     face_out, value, v_ctx) = agent.sample_action_dynamic(
        None, avail, ax_state, ax_mask, ft, ovr, key,
        precomputed=pre, enc_carry=enc,
        face_chunk_fn=chunk_fn, face_count_fn=count_fn)
    (fa, face_logp, face_ent, f_pair, f_comp, f_valid, f_cnt, f_dt, f_de,
     f_ends) = face_out
    ea = agent.to_env_action_dynamic(
        vertex_idx, actions, ax_state, face_action=fa)
    return {
        "v": int(vertex_idx),
        "fa": jax.tree_util.tree_map(np.asarray, fa),
        "logp": np.asarray(face_logp),
        "ent": np.asarray(face_ent),
        "f_pair": np.asarray(f_pair),
        "f_comp": np.asarray(f_comp),
        "f_valid": np.asarray(f_valid),
        "f_cnt": np.asarray(f_cnt, np.int32),
        "f_dt": np.asarray(f_dt, np.int32),
        "f_de": np.asarray(f_de, np.int32),
        "fr": np.asarray(ea.face_rows, np.int32),
        "fs": np.asarray(ea.face_skip, np.int32),
        "value": np.asarray(value),
    }


# ------------------------------------------------------- (a) bitwise identity
@pytest.mark.parametrize("n", [1, 2, 3, 5, 9])
def test_bucketed_draw_is_bitwise_identical(agent, n):
    key = jrand.PRNGKey(100 + n)
    full = _run_face_path(agent, n, key)

    fb = bucket_width(n, MAXF)
    assert fb >= n
    ag_b = with_face_width(agent, fb)
    assert int(ag_b.face_path_policy.max_faces) == fb
    bck = _run_face_path(ag_b, n, key)

    assert bck["v"] == full["v"]
    # re-pad the bucketed outputs exactly like az_gumbel._draw_face_sequence
    fr2, fs2, fa2, fp2, fc2, fv2, fn2 = pad_face_outputs(
        MAXF, bck["fr"], bck["fs"], bck["fa"], bck["f_pair"],
        bck["f_comp"], bck["f_valid"], bck["f_cnt"])

    assert np.array_equal(fr2, full["fr"]), "face_rows diverge"
    assert np.array_equal(fs2, full["fs"]), "face_skip diverges"
    for name in fa2._fields:
        assert np.array_equal(
            np.asarray(getattr(fa2, name)),
            np.asarray(getattr(full["fa"], name))), f"FaceAction.{name}"
    assert np.array_equal(fp2, full["f_pair"])
    assert np.array_equal(fc2, full["f_comp"])
    assert np.array_equal(fv2, full["f_valid"])
    assert np.array_equal(fn2, full["f_cnt"])
    # scalar joint log-prob / entropy: padding faces are gated to exactly 0
    # in the full-width loop (it never iterates them), so these are the sum
    # of the SAME per-face terms in the SAME order.
    assert np.array_equal(bck["logp"], full["logp"]), (
        bck["logp"], full["logp"])
    assert np.array_equal(bck["ent"], full["ent"])
    # the emission window the head read (loss replay input)
    assert np.array_equal(bck["f_dt"], full["f_dt"])
    assert np.array_equal(bck["f_de"], full["f_de"])
    assert np.array_equal(bck["value"], full["value"])
    # sanity: the real faces were actually decided
    assert int(np.sum(full["f_valid"] > 0.5)) == n


def test_with_face_width_shares_params_and_leaves_original(agent):
    ag_b = with_face_width(agent, 8)
    la = jax.tree_util.tree_leaves(eqx.filter(agent, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(ag_b, eqx.is_array))
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert x is y, "bucketing must SHARE params, never copy"
    assert int(agent.face_path_policy.max_faces) == MAXF  # original untouched
    # identity when the width already matches
    assert with_face_width(agent, MAXF) is agent


# ------------------------------------------------------- (b) bucket selection
def test_bucket_width_selection():
    assert FACE_BUCKETS == (2, 4, 8, 32)
    mf = 2538
    expect = {0: 2, 1: 2, 2: 2, 3: 4, 4: 4, 5: 8, 8: 8, 9: 32, 32: 32,
              33: 2538, 40: 2538, 2538: 2538}
    for n, b in expect.items():
        assert bucket_width(n, mf) == b, (n, b, bucket_width(n, mf))
    # never below the live count, never above the configured max
    for n in range(0, 70):
        b = bucket_width(n, mf)
        assert n <= b <= mf
    # a tiny configured max caps every bucket
    assert bucket_width(1, 3) == 2
    assert bucket_width(3, 3) == 3
    assert bucket_width(5, 3) == 3


def test_hist_face_width():
    F, S = 16, 3
    fh = np.full((4, F, S, 3), -1, np.int32)
    sh = np.zeros((4, F), np.int32)
    assert hist_face_width(fh, sh, 0) == 0
    assert hist_face_width(fh, sh, 3) == 0          # all padding
    fh[1, 2, 0] = [0, 1, 2]                          # decided wire, face 2
    assert hist_face_width(fh, sh, 3) == 3
    sh[2, 6] = 1                                     # skip at face 6
    assert hist_face_width(fh, sh, 3) == 7
    assert hist_face_width(fh, sh, 2) == 3           # row 2 outside prefix
    # END translator rows ([-1, -1, 0]) are padding, not decided wires
    fh[0, 9, :] = [-1, -1, 0]
    assert hist_face_width(fh, sh, 3) == 7


# ------------------------------------------------------ (c) compile count
def test_compile_count_bounded(agent):
    """Counts {1, 2, 3, 5, 9, 40} (each twice) -> at most 5 distinct
    compiled shapes: {2, 4, 8, 32, MAXF}. The counter appends at TRACE time,
    so a cache hit adds nothing."""
    traces = []

    @eqx.filter_jit
    def probe(ag, x):
        traces.append((int(ag.face_path_policy.max_faces), tuple(x.shape)))
        return x * 2.0

    for n in (1, 2, 3, 5, 9, 40, 1, 2, 3, 5, 9, 40):
        fb = bucket_width(n, MAXF)
        probe(with_face_width(agent, fb), jnp.zeros((fb,), jnp.float32))
    assert len(traces) <= 5, f"{len(traces)} compiles: {traces}"
    assert len(set(traces)) == len(traces)
    assert set(t[0] for t in traces) == {2, 4, 8, 32, MAXF}


if __name__ == "__main__":
    _warm_quant_scan()
    a = _agent()
    test_bucket_width_selection()
    test_hist_face_width()
    test_with_face_width_shares_params_and_leaves_original(a)
    for _n in (1, 2, 3, 5, 9):
        test_bucketed_draw_is_bitwise_identical(a, _n)
    test_compile_count_bounded(a)
    print("ALL PASS")
