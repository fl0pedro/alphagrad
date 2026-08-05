# -*- coding: utf-8 -*-
"""STAGE 1 EQUIVALENCE PROOF: per-step delta buffers == the growing stream.

The trajectory stores the WHOLE ``state.tokens`` per step -- (E, T, MAX_TOKENS)
= 16 x 27 x 32768 x 4 B ~ 57 MB per array on the flagship -- while nothing ever
re-reads an earlier token: the rollout consumes ``stream[pos : pos + delta]``
and the loss re-derives the SAME window from the stored per-step carry. The
growing array is an artefact of the cursor being ABSOLUTE.

``ALPHAGRAD_DELTA_TOKENS=1`` adds the alternative: a BASE buffer
(MAX_BASE_TOKENS) filled once at episode start, and a per-step DELTA buffer
(MAX_DELTA_TOKENS) with a RELATIVE cursor -- the shape the FACE stream already
has. This file freezes ONE golden episode (fixed seed, four envs, real
tokenizer streams) and asserts the two paths agree BITWISE, so it can be re-run
unchanged after stage 2 deletes the old one:

  1. the delta buffer IS ``stream[pos : pos + W]``, token for token AND
     eqn-id for eqn-id -- the absolute-cursor path never read anything the
     relative-cursor path does not carry;
  2. the encoder carry (M, I, cumhist, nvalid, pos), the emitted rows, the
     valid mask and the eqn window are bitwise identical after EVERY step of
     EVERY env, and so is the vertex-memory fold they feed;
  3. the base buffer reproduces the wide-window base encode bitwise;
  4. the per-face chunks and their counts are unchanged.

ONE QUANTITY IS NOT BITWISE-EQUAL, AND IT IS NOT A FLOAT REASSOCIATION: the
delta COUNT. ``_stream_len`` counts non-zero tokens, but id 0 is not a reserved
pad -- it is the literal '-' graphax emits for a negative value. This graph's
base stream is 300 tokens with one id 0 at index 207 (the '-' of ``max -inf``),
so ``_stream_len`` returns 299 and the old path's carry starts one token short;
over the whole episode it loses 2-5 tokens per env. Measured the same way, the
flagship nn256 base is 419 tokens with 1 interior zero and its full stream
34590 tokens with 9. ``test_stream_len_hazard_is_live`` pins that with numbers.
The equivalence tests therefore drive BOTH paths from ``_stream_end`` (one past
the last non-zero, which is correct in both worlds) -- comparing two paths
under a length that is provably wrong would prove nothing about the buffers.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import jax.random as jrand                                        # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402
from graphax import IncrementalPathTokenizer                      # noqa: E402

from alphagrad.approx.env import MAX_BASE_TOKENS                  # noqa: E402
from alphagrad.approx.ppo import (                                # noqa: E402
    EncCarry, _build_agent, _stream_end, _window_copy, make_argparser,
)
# The OLD length function. Imported softly because stage 2 deletes it along
# with the growing-stream path; everything else in this file survives that.
try:
    from alphagrad.approx.ppo import _stream_len                   # noqa: E402
except ImportError:                                                # pragma: no cover
    _stream_len = None
from alphagrad.approx import vertex_memory as _vmem               # noqa: E402

VOCAB_TOK = 248     # ALPHAGRAD_INCR_TOKEN_VOCAB, as the env uses
EMBD = 16
N_ENVS = 4
SEED = 20260805

# Window widths are LOCAL, not the module constants: the proof is about the
# buffer SHAPE, and it must hold whatever ALPHAGRAD_MAX_*_TOKENS the ambient
# process happens to carry (the suite shares one interpreter).
STREAM_W = 32768    # stands in for MAX_TOKENS -- the growing buffer
BASE_W = 8192       # stands in for MAX_BASE_TOKENS
LARGEST_MEASURED_BASE = 4219   # BlackScholes_Jacobian, 2026-08-05 sweep


# --------------------------------------------------------------------------- #
# the golden episode
# --------------------------------------------------------------------------- #
def _graph():
    """A real graph with real faces (the one tests/test_live_faces.py drives),
    so the streams are the tokenizer's, not a synthetic stand-in."""
    from graphax.examples import Perceptron

    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (2, 4))
    y = jrand.normal(jrand.fold_in(key, 5), (2, 3))
    W1 = jrand.normal(jrand.fold_in(key, 1), (4, 6))
    b1 = jrand.normal(jrand.fold_in(key, 2), (6,))
    W2 = jrand.normal(jrand.fold_in(key, 3), (6, 3))
    b2 = jrand.normal(jrand.fold_in(key, 4), (3,))
    gamma = jrand.normal(jrand.fold_in(key, 6), (6,))
    beta = jrand.normal(jrand.fold_in(key, 7), (6,))
    args = (x, y, W1, b1, W2, b2, gamma, beta)
    cj = jax.make_jaxpr(Perceptron)(*args)
    return cj.jaxpr, cj.literals, args, (2, 3, 4, 5)


def _observations(order):
    """The per-step (tokens, eqn_ids, true_len) buffers the env emits for
    ``order``, built exactly the way ``env._incremental_stream_tokens`` +
    ``env._callback`` build them: base tokens, then one block per elimination,
    pad-filled (0 for tokens, -1 for eqn ids)."""
    jaxpr, consts, args, argnums = _graph()
    tk = IncrementalPathTokenizer(jaxpr, argnums, list(consts), list(args),
                                  vocab_size=VOCAB_TOK)
    stream = [int(t) for t in tk.base_tokens()]
    seg = [int(g) for g in tk.last_eqn_ids()]
    snaps = [(list(stream), list(seg))]
    for v in order:
        stream += [int(t) for t in tk.eliminate(int(v))]
        seg += [int(g) for g in tk.last_eqn_ids()]
        snaps.append((list(stream), list(seg)))
    bufs = []
    for s, g in snaps:
        assert len(s) <= STREAM_W, f"golden stream {len(s)} > STREAM_W"
        t = np.zeros((STREAM_W,), np.int32)
        e = np.full((STREAM_W,), -1, np.int32)
        t[:len(s)] = np.asarray(s, np.int32)
        e[:len(g)] = np.asarray(g, np.int32)
        bufs.append((jnp.asarray(t), jnp.asarray(e), len(s)))
    return bufs


def _golden():
    jaxpr, _c, _a, _an = _graph()
    V = len(jaxpr.eqns)
    rng = np.random.default_rng(SEED)
    orders = [list(rng.permutation(np.arange(1, V + 1))) for _ in range(N_ENVS)]
    return V, orders, [_observations(o) for o in orders]


def _agent(total_v):
    args = make_argparser().parse_args([])
    args.dynamic_substeps = True
    args.vocab_size = 512
    args.embd_dim = EMBD
    args.num_heads = 2
    args.num_layers = 2
    args.hidden_dim = 32
    return _build_agent(args, total_v, num_factors=4, max_rules=4,
                        key=jrand.PRNGKey(7))


def _carry_equal(a, b, where):
    for name in EncCarry._fields:
        na, nb = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert np.array_equal(na, nb), (
            f"{where}: carry field {name} differs, max|diff|="
            f"{np.max(np.abs(na - nb)) if na.size else 0}")


def _raw_window(buf, pos, width, fill):
    """What the ABSOLUTE cursor reads out of the growing stream."""
    idx = jnp.asarray(pos, jnp.int32) + jnp.arange(width, dtype=jnp.int32)
    return jnp.take(buf, idx, mode="fill", fill_value=fill)


_V, _ORDERS, _OBS = _golden()
_AGENT = _agent(_V)


# --------------------------------------------------------------------------- #
# 0. sizing + the hazard, with numbers
# --------------------------------------------------------------------------- #
def test_base_buffer_is_sized_from_the_measured_distribution():
    """MAX_BASE_TOKENS must cover the largest base stream actually measured
    (len(base_tokens()) is order-independent). 2026-08-05 sweep: Helmholtz 60,
    nn256 419, ViT 1457, TransformerLM 1406, EncoderDecoder 1627,
    RoeFlux_3d 1777, BlackScholes_Jacobian 4219."""
    assert MAX_BASE_TOKENS >= LARGEST_MEASURED_BASE, (
        f"MAX_BASE_TOKENS={MAX_BASE_TOKENS} does not cover the largest "
        f"measured base ({LARGEST_MEASURED_BASE}); the base encode would be "
        f"clipped and the carry would desync for the whole episode")
    for e, obs in enumerate(_OBS):
        assert obs[0][2] <= BASE_W, f"env {e}: base {obs[0][2]} > BASE_W"


def test_stream_len_hazard_is_live():
    """THE id-0 COLLISION, on a real stream. Token id 0 is the vocabulary's
    '-', not a pad, so ``_stream_len`` (a non-zero COUNT) undercounts by one
    per negative literal -- here the '-' of ``max -inf``. ``_stream_end`` (one
    past the last non-zero) is the correct length in both worlds.

    This is exactly the quantity the delta-buffer design removes: the count is
    computed ONCE, on the rollout side, and STORED in the trajectory, so the
    loss never re-derives a window from a growing buffer at all."""
    if _stream_len is None:
        pytest.skip("_stream_len is gone -- the hazard went with it (stage 2)")
    seen_interior_zero = False
    losses = []
    for e, obs in enumerate(_OBS):
        for k, (tok, _eq, raw) in enumerate(obs):
            n_end = int(_stream_end(tok))
            n_count = int(_stream_len(tok))
            assert n_end == raw, (
                f"env {e} step {k}: _stream_end {n_end} != true length {raw}")
            if n_count != raw:
                seen_interior_zero = True
        losses.append(obs[-1][2] - int(_stream_len(obs[-1][0])))
    base_true = _OBS[0][0][2]
    base_counted = int(_stream_len(_OBS[0][0][0]))
    assert seen_interior_zero, (
        "this golden episode no longer exercises the id-0 collision -- the "
        "hazard assertion below is then vacuous, pick a graph with a negative "
        "literal (e.g. -inf from a max/softmax)")
    assert base_counted == base_true - 1, (
        f"expected the base stream to carry exactly one interior id 0 "
        f"(true={base_true}, _stream_len={base_counted})")
    assert all(l > 0 for l in losses), (
        f"per-env end-of-episode undercount by _stream_len: {losses}")


# --------------------------------------------------------------------------- #
# 1./2./3. the delta buffer IS the window, and the encode is bitwise identical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env_i", range(N_ENVS))
@pytest.mark.parametrize("delta_w", [4096, 1024])
def test_delta_buffer_equals_the_stream_window_bitwise(env_i, delta_w):
    """``delta_w=4096`` is wider than every delta in this episode (max 3056);
    ``delta_w=1024`` is NARROWER, so both paths clip -- they must clip
    identically too, which is the regime the campaign actually runs in
    (MAX_DELTA_TOKENS=2048 against nn256 deltas up to 5664)."""
    agent, obs = _AGENT, _OBS[env_i]
    total_v = _V

    def _fold(rows, eqns, valid):
        ids = jnp.where((eqns >= 0) & (eqns < total_v), eqns, -1)
        s = jnp.zeros((total_v + 1, EMBD), jnp.float32)
        c = jnp.zeros((total_v + 1,), jnp.float32)
        return _vmem.update_ids(s, c, rows, ids, valid)

    # --- base: wide window over the stream vs a base-sized buffer ---------
    tok0, eqn0, raw0 = obs[0]
    n0 = _stream_end(tok0)
    assert int(n0) == raw0

    old, r_old, v_old, e_old = agent.encode_extend(
        agent.carry_init(), tok0, eqn0, n0, window=STREAM_W)
    btok, beqn = _window_copy(tok0, eqn0, 0, n0, BASE_W)
    assert np.array_equal(np.asarray(btok)[:raw0], np.asarray(tok0)[:raw0])
    assert np.array_equal(np.asarray(beqn)[:raw0], np.asarray(eqn0)[:raw0])
    assert np.all(np.asarray(btok)[raw0:] == 0)
    assert np.all(np.asarray(beqn)[raw0:] == -1)
    new, r_new, v_new, e_new = agent.encode_extend(
        agent.carry_init(), btok, beqn, n0, window=BASE_W, start=0)
    _carry_equal(old, new, f"env {env_i} base")
    assert np.array_equal(np.asarray(r_old)[:raw0], np.asarray(r_new)[:raw0])
    assert np.all(np.asarray(r_old)[raw0:] == 0.0)
    assert np.all(np.asarray(r_new)[raw0:] == 0.0)
    assert np.array_equal(np.asarray(v_old)[:raw0], np.asarray(v_new)[:raw0])
    assert np.array_equal(np.asarray(e_old)[:raw0], np.asarray(e_new)[:raw0])
    s_old, c_old = _fold(r_old, e_old, v_old)
    s_new, c_new = _fold(r_new, e_new, v_new)
    assert np.array_equal(np.asarray(c_old), np.asarray(c_new))
    assert np.array_equal(np.asarray(s_old), np.asarray(s_new)), (
        "base vertex-memory fold differs, max|diff|="
        f"{np.max(np.abs(np.asarray(s_old) - np.asarray(s_new)))}")

    # --- one step per elimination ----------------------------------------
    for k in range(1, len(obs)):
        tok, eqn, raw = obs[k]
        # The two cursors must agree; they only equal the TRUE stream position
        # while no step has clipped (delta_w=1024 clips on this graph, exactly
        # as MAX_DELTA_TOKENS=2048 clips nn256's 5664-token deltas -- and both
        # paths must then clip identically, which is what this loop pins).
        assert int(old.pos) == int(new.pos), (
            f"env {env_i} step {k}: cursors diverged "
            f"{int(old.pos)} vs {int(new.pos)}")
        cnt = _stream_end(tok) - old.pos
        if int(old.pos) == obs[k - 1][2]:
            assert int(cnt) == raw - obs[k - 1][2]

        dtok, deqn = _window_copy(tok, eqn, new.pos, cnt, delta_w)
        # (1) TOKEN FOR TOKEN against what the absolute cursor reads.
        assert np.array_equal(
            np.asarray(dtok),
            np.asarray(_raw_window(tok, old.pos, delta_w, 0))), (
            f"env {env_i} step {k}: delta tokens != stream[pos:pos+W]")
        assert np.array_equal(
            np.asarray(deqn),
            np.asarray(_raw_window(eqn, old.pos, delta_w, -1))), (
            f"env {env_i} step {k}: delta eqn ids != stream[pos:pos+W]")

        old, ro, vo, eo = agent.encode_extend(
            old, tok, eqn, cnt, window=delta_w)
        new, rn, vn, en = agent.encode_extend(
            new, dtok, deqn, cnt, window=delta_w, start=0)
        # (2) bitwise-identical encoder state after EVERY step
        _carry_equal(old, new, f"env {env_i} step {k}")
        assert np.array_equal(np.asarray(ro), np.asarray(rn)), (
            f"env {env_i} step {k}: rows differ, max|diff|="
            f"{np.max(np.abs(np.asarray(ro) - np.asarray(rn)))}")
        assert np.array_equal(np.asarray(vo), np.asarray(vn))
        assert np.array_equal(np.asarray(eo), np.asarray(en))

        so, co = _fold(ro, eo, vo)
        sn, cn = _fold(rn, en, vn)
        assert np.array_equal(np.asarray(co), np.asarray(cn))
        assert np.array_equal(np.asarray(so), np.asarray(sn))


def test_loss_replay_needs_only_the_stored_buffer_and_count():
    """Ratio-1 without the stream: extending the stored PRE-step carry with
    the STORED (buffer, count) reproduces the rollout's post-step carry
    bitwise. This is what ``_carry_heads`` does under ALPHAGRAD_DELTA_TOKENS=1
    -- no ``_stream_len``, no ``batch.tokens``."""
    agent, obs = _AGENT, _OBS[0]
    W = 4096
    tok0, eqn0, raw0 = obs[0]
    btok, beqn = _window_copy(tok0, eqn0, 0, raw0, BASE_W)
    carry, _r, _v, _e = agent.encode_extend(
        agent.carry_init(), btok, beqn, raw0, window=BASE_W, start=0)
    for k in range(1, len(obs)):
        tok, eqn, _raw = obs[k]
        cnt = _stream_end(tok) - carry.pos
        dtok, deqn = _window_copy(tok, eqn, carry.pos, cnt, W)
        stored_carry, stored_cnt = carry, jnp.asarray(cnt, jnp.int32)
        carry, rows, _v, _e = agent.encode_extend(
            carry, dtok, deqn, cnt, window=W, start=0)
        replay, rows2, _v2, _e2 = agent.encode_extend(
            stored_carry, dtok, deqn, stored_cnt, window=W, start=0)
        _carry_equal(carry, replay, f"loss replay step {k}")
        assert np.array_equal(np.asarray(rows), np.asarray(rows2))


# --------------------------------------------------------------------------- #
# 4. the per-face chunks and their counts are unchanged
# --------------------------------------------------------------------------- #
def test_face_chunks_and_counts_unchanged():
    """Face chunks are ALREADY standalone buffers read with ``start=0``; their
    only coupling to the vertex stream is the branch-point carry. Same chunks,
    same counts, and -- fed the two paths' carries -- bitwise the same side
    carry and pooled context, so every face decision is the same one."""
    from alphagrad.approx.live_faces import LiveFaceStream

    jaxpr, consts, args, argnums = _graph()
    V = len(jaxpr.eqns)
    MR, F, S = 8, 8, 3
    lfs = LiveFaceStream(jaxpr, argnums, consts, args, vocab=VOCAB_TOK,
                         max_faces=F, max_axes=8, window=8192)
    order = np.zeros((V,), np.int32)
    specs = -np.ones((V, MR, 3), np.int32)
    vspecs = -np.ones((MR, 3), np.int32)
    rows = -np.ones((F, S, 3), np.int32)
    skips = np.zeros((F,), np.int32)

    vertex, n_faces = None, 0
    for cand in range(1, V + 1):
        _t, _i, _c, n = lfs.chunk(order, specs, 0, cand, vspecs, rows, skips, 0)
        if int(n) > n_faces:
            vertex, n_faces = cand, int(n)
        if n_faces >= 3:
            break
    if vertex is None or n_faces < 2:
        pytest.skip("no multi-face vertex on this graph")

    agent, obs = _AGENT, _OBS[0]
    tok0, eqn0, raw0 = obs[0]
    n0 = _stream_end(tok0)
    c_old, _r, _v, _e = agent.encode_extend(
        agent.carry_init(), tok0, eqn0, n0, window=STREAM_W)
    btok, beqn = _window_copy(tok0, eqn0, 0, n0, BASE_W)
    c_new, _r, _v, _e = agent.encode_extend(
        agent.carry_init(), btok, beqn, n0, window=BASE_W, start=0)
    _carry_equal(c_old, c_new, "face branch point")

    counts = []
    for f in range(n_faces):
        t, i, c, n = lfs.chunk(order, specs, 0, vertex, vspecs, rows, skips, f)
        t2, i2, c2, n2 = lfs.chunk(order, specs, 0, vertex, vspecs, rows,
                                   skips, f)
        # the chunk itself does not depend on the encoder at all
        assert np.array_equal(np.asarray(t), np.asarray(t2))
        assert np.array_equal(np.asarray(i), np.asarray(i2))
        assert int(c) == int(c2) and int(n) == int(n2)
        counts.append(int(c))
        c_old, s_old = agent._face_encode(c_old, jnp.asarray(t),
                                          jnp.asarray(i), jnp.asarray(c))
        c_new, s_new = agent._face_encode(c_new, jnp.asarray(t),
                                          jnp.asarray(i), jnp.asarray(c))
        _carry_equal(c_old, c_new, f"face {f} side carry")
        assert np.array_equal(np.asarray(s_old), np.asarray(s_new)), (
            f"face {f} pooled context differs")
    assert all(c > 0 for c in counts), f"empty face chunk: {counts}"


if __name__ == "__main__":
    test_base_buffer_is_sized_from_the_measured_distribution()
    test_stream_len_hazard_is_live()
    for _w in (4096, 1024):
        for _i in range(N_ENVS):
            test_delta_buffer_equals_the_stream_window_bitwise(_i, _w)
    test_loss_replay_needs_only_the_stored_buffer_and_count()
    test_face_chunks_and_counts_unchanged()
    print("ALL PASS")
