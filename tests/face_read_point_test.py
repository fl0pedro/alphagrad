# -*- coding: utf-8 -*-
"""--face-read {chunk-mean, own-span-mean, last-row}.

docs/FACE_READ_POINT_TRACE.md: face ``f``'s chunk is
``[approx-echo(f-1) || header+contraction(f)]`` and the head pools the WHOLE
of it, so the 94 logits come from a token-count-weighted blend of the
PREVIOUS face's approximation with THIS face's contraction, with no marker
separating them. The fix masks the pooling at the already-computed
approx-echo prefix; the stronger variant reads the recurrence's last row
instead of any mean.

Pins:

1. FLAG-OFF BIT-IDENTITY. ``chunk-mean`` is the historical path: same draws,
   same wires, same log-probs, same entropies, same face latents, and the
   per-face ``head`` wire does not exist at all (``_face_read_needs_head()``
   is False, so the callback arity -- the TRACE -- is unchanged too).
2. THE READ IS WHAT IT SAYS. ``own-span-mean``'s latent is the mean of rows
   ``[head, count)``; ``last-row``'s is row ``count-1``; both computed
   independently from ``encode_extend``'s own output rows.
3. THE CARRY IS UNCHANGED. The side carry after ``_face_encode`` is BITWISE
   identical under all three settings -- only the readout moves, which is
   what keeps the recurrence causal.
4. ROLLOUT == REPLAY for all three. `_face_encode` (sampling) and
   `_face_replay` (loss) must move together or the PPO ratio is silently
   != 1 at epoch 0 with no error anywhere.
5. EMPTY POOL -> ZERO VECTOR, the documented absent-latent convention.
6. A replay handed no ``face_heads`` under a non-default mode FAILS LOUDLY
   rather than silently pooling the wrong span.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_MAX_FACES", "64")
os.environ.setdefault("ALPHAGRAD_MAX_DELTA_TOKENS", "128")

import equinox as eqx                                           # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

# READ THE LIVE CONSTANTS, not the env vars: `env.py` freezes both at ITS
# import, and in a multi-file pytest session another test module may already
# have imported it under different settings. `_face_loop` builds its buffers
# from `env.MAX_DELTA_TOKENS`, so the stand-in chunks must match that.
from alphagrad.approx.env import (                               # noqa: E402
    DELTA_TOKEN_DTYPE as _TOKEN_DTYPE, MAX_DELTA_TOKENS as W,
    MAX_FACES as MAXF)

TOTAL_V = 6
EMBD = 32
MODES = ("chunk-mean", "own-span-mean", "last-row")


# ------------------------------------------------------------------ fixtures
def _agent():
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
    ns.face_endpoint_read = False
    return build_and_init_agent(ns, TOTAL_V, 4, 4, seed=11)


@pytest.fixture(scope="module")
def agent():
    from graphax.sparse.micro_actions import report_hardware_scan
    report_hardware_scan()
    a = _agent()
    assert a.face_path_policy is not None
    return a


@pytest.fixture(autouse=True)
def _restore_mode():
    from alphagrad.approx import ppo as P
    old = P._FACE_READ[0]
    yield
    P._FACE_READ[0] = old


def _decision_inputs(agent):
    from alphagrad.approx.common import carry_stream as _cs
    from alphagrad.approx.env import MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM
    from alphagrad.approx.heads import NUM_OPS, precompute_factor_tables

    toks = jrand.randint(jrand.PRNGKey(1), (48,), 1, 60)
    enc, vs, vc = _cs.init_carry(
        agent, toks, 48, window=48, total_v=TOTAL_V, embd_dim=EMBD)
    pre = agent.heads_from_memory(vs, vc)
    avail = jnp.zeros((TOTAL_V,), jnp.float32).at[2].set(1.0)   # force v=3
    ax_state = jnp.zeros(
        (TOTAL_V, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), jnp.int32)
    ax_state = ax_state.at[..., 0].set(8)
    ax_mask = jnp.zeros(
        (TOTAL_V, MAX_AXES_PER_VERTEX), jnp.float32).at[:, :3].set(1.0)
    ft = precompute_factor_tables(16)
    ovr = jnp.ones((NUM_OPS,), jnp.float32)
    return enc, vs, vc, pre, avail, ax_state, ax_mask, ft, ovr


def _ct_of(f):
    return (f % 3) + 2                      # 2, 3, 4


def _head_of(f, empty=False):
    return _ct_of(f) if empty else _ct_of(f) - 2      # 0, 1, 2


def _chunk_fns(n, emit_head, empty_pool=False):
    """Deterministic LiveFaceStream stand-in, with the approx-echo prefix."""

    def face_chunk_fn(f, vertex_idx, vertex_specs, rows, skips):
        ct = (f % 3) + 2
        ar = jnp.arange(W, dtype=jnp.int32)
        tok = jnp.where(ar < ct, (f * 7 + ar) % 50 + 1, 0).astype(
            jnp.dtype(_TOKEN_DTYPE))
        ends = jnp.stack([(f % 3) + 1, (f % 2) + 1]).astype(jnp.int32)
        # The stand-in mirrors the real callback: tokens, count, endpoints.
        # There is no equation-id buffer any more.
        out = (tok, jnp.asarray(ct, jnp.int32), ends)
        if emit_head:
            hd = ct if empty_pool else ct - 2
            out = out + (jnp.asarray(hd, jnp.int32),)
        return out

    def face_count_fn(vertex_idx):
        return jnp.asarray(n, jnp.int32)

    return face_chunk_fn, face_count_fn


def _run(agent, mode, n, key, empty_pool=False):
    from alphagrad.approx import ppo as P
    P.set_face_read(mode)
    rh = P._face_read_needs_head()
    enc, vs, vc, pre, avail, ax_state, ax_mask, ft, ovr = (
        _decision_inputs(agent))
    chunk_fn, count_fn = _chunk_fns(n, rh, empty_pool)
    (vertex_idx, actions, _vd, _od, _id, _jd, _ed, _kd, _qlp, _vp, _vc,
     face_out, value, v_ctx) = agent.sample_action_dynamic(
        None, avail, ax_state, ax_mask, ft, ovr, key,
        precomputed=pre, enc_carry=enc,
        face_chunk_fn=chunk_fn, face_count_fn=count_fn)
    (fa, face_logp, face_ent, f_pair, f_comp, f_valid, f_cnt, f_dt,
     f_ends) = face_out[:9]
    heads = np.asarray(face_out[-1]) if rh else None
    return dict(
        v=int(vertex_idx), fa=fa, logp=np.asarray(face_logp),
        ent=np.asarray(face_ent), f_pair=f_pair, f_comp=f_comp,
        f_valid=f_valid, f_cnt=f_cnt, f_dt=f_dt, f_ends=f_ends,
        heads=heads, enc=enc, ax_state=ax_state, ax_mask=ax_mask, ft=ft,
        ovr=ovr)


def _replay(agent, r, face_heads="stored"):
    from alphagrad.approx.ppo import _axis_features_from_state
    feats = _axis_features_from_state(r["ax_state"][r["v"]],
                                      r["ax_mask"][r["v"]])
    fh = r["heads"] if face_heads == "stored" else face_heads
    fh = None if fh is None else jnp.asarray(fh, jnp.int32)
    return agent._face_replay(
        feats, r["ft"], r["fa"], r["f_pair"], r["f_comp"], r["f_valid"],
        r["enc"], (r["f_cnt"], r["f_dt"]), r["ovr"],
        face_heads=fh)


# ------------------------------------------------ 1. flag-off bit identity
def test_chunk_mean_needs_no_head_wire(agent):
    from alphagrad.approx import ppo as P
    P.set_face_read("chunk-mean")
    assert P._face_read_needs_head() is False
    for m in ("own-span-mean", "last-row"):
        P.set_face_read(m)
        assert P._face_read_needs_head() is True
    with pytest.raises(ValueError):
        P.set_face_read("nonsense")


def test_chunk_mean_face_out_arity_is_historical(agent):
    """No extra element on the sampling tuple: the flag-off TRACE is the
    v63 one, not merely the flag-off numbers."""
    r = _run(agent, "chunk-mean", 3, jrand.PRNGKey(7))
    assert r["heads"] is None


def test_chunk_mean_matches_the_unmasked_scatter_mean(agent):
    """The default readout IS the whole-chunk mean, recomputed here from
    encode_extend's own rows -- i.e. nothing about the default moved."""
    from alphagrad.approx import ppo as P
    from alphagrad.approx import vertex_memory as _vmem
    P.set_face_read("chunk-mean")
    enc = _decision_inputs(agent)[0]
    ct = 4
    ar = jnp.arange(W, dtype=jnp.int32)
    tok = jnp.where(ar < ct, (ar * 3) % 50 + 1, 0).astype(jnp.int32)
    c2, summ = agent._face_encode(enc, tok, jnp.asarray(ct, jnp.int32))
    _c, rows, valid = agent.encode_extend(
        enc, tok, jnp.asarray(ct, jnp.int32), window=W, start=0)
    want = _vmem.scatter_mean(
        rows, jnp.zeros((rows.shape[0],), jnp.int32), valid, 1)[0]
    np.testing.assert_array_equal(np.asarray(summ), np.asarray(want))


# ------------------------------- 2/3. the readout moves, the carry does not
def test_readout_is_what_it_says_and_carry_is_invariant(agent):
    from alphagrad.approx import ppo as P
    enc = _decision_inputs(agent)[0]
    ct, hd = 5, 2
    ar = jnp.arange(W, dtype=jnp.int32)
    tok = jnp.where(ar < ct, (ar * 11) % 50 + 1, 0).astype(jnp.int32)
    ctj, hdj = jnp.asarray(ct, jnp.int32), jnp.asarray(hd, jnp.int32)

    # the ground truth rows, from the recurrence itself
    _c, rows, valid = agent.encode_extend(
        enc, tok, ctj, window=W, start=0)
    R = np.asarray(rows)

    carries, summs = {}, {}
    for m in MODES:
        P.set_face_read(m)
        c2, summ = agent._face_encode(enc, tok, ctj, pool_from=hdj)
        carries[m] = [np.asarray(x) for x in jax.tree_util.tree_leaves(c2)]
        summs[m] = np.asarray(summ)

    # 3. THE CARRY IS BITWISE IDENTICAL ACROSS ALL THREE.
    base = carries["chunk-mean"]
    for m in MODES[1:]:
        assert len(carries[m]) == len(base)
        for i, (x, y) in enumerate(zip(base, carries[m])):
            np.testing.assert_array_equal(x, y, err_msg=f"{m} carry leaf {i}")

    # 2. THE READOUT IS WHAT IT SAYS.
    np.testing.assert_allclose(summs["chunk-mean"], R[:ct].mean(axis=0),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(summs["own-span-mean"], R[hd:ct].mean(axis=0),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(summs["last-row"], R[ct - 1],
                               rtol=1e-6, atol=1e-6)
    # and the three are genuinely different vectors (hd > 0, ct > 1)
    assert not np.allclose(summs["chunk-mean"], summs["own-span-mean"])
    assert not np.allclose(summs["own-span-mean"], summs["last-row"])


# ------------------------------------------------------- 5. empty pool -> 0
def test_empty_pool_is_the_zero_vector(agent):
    from alphagrad.approx import ppo as P
    enc = _decision_inputs(agent)[0]
    ct = 4
    ar = jnp.arange(W, dtype=jnp.int32)
    tok = jnp.where(ar < ct, (ar * 5) % 50 + 1, 0).astype(jnp.int32)
    P.set_face_read("own-span-mean")
    _c, summ = agent._face_encode(enc, tok, jnp.asarray(ct, jnp.int32),
                                  pool_from=jnp.asarray(ct, jnp.int32))
    np.testing.assert_array_equal(np.asarray(summ), 0.0)


# ------------------------------------------------ 4. rollout == replay, x3
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", [1, 3, 5])
def test_rollout_equals_replay(agent, mode, n):
    r = _run(agent, mode, n, jrand.PRNGKey(300 + n))
    lp, ent, ar, lat = _replay(agent, r)
    np.testing.assert_allclose(np.asarray(lp), r["logp"],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ent), r["ent"],
                               rtol=1e-5, atol=1e-6)
    assert lat.shape == (MAXF, EMBD)


@pytest.mark.parametrize("mode", ["own-span-mean", "last-row"])
def test_replay_with_the_wrong_span_breaks_the_ratio(agent, mode):
    """Control: the mirror is load-bearing. Replaying the SAME actions with
    the chunk-mean span (i.e. the un-mirrored loss) must move the log-prob,
    which is exactly the silent ratio break the fix sketch warns about."""
    from alphagrad.approx import ppo as P
    r = _run(agent, mode, 4, jrand.PRNGKey(77))
    lp_ok, _, _, _ = _replay(agent, r)
    np.testing.assert_allclose(np.asarray(lp_ok), r["logp"],
                               rtol=1e-5, atol=1e-6)
    P.set_face_read("chunk-mean")
    lp_bad, _, _, _ = _replay(agent, r, face_heads=None)
    assert not np.allclose(np.asarray(lp_bad), r["logp"],
                           rtol=1e-5, atol=1e-6)


# --------------------------------------------------- 6. loud, never silent
@pytest.mark.parametrize("mode", ["own-span-mean", "last-row"])
def test_replay_without_heads_fails_loudly(agent, mode):
    r = _run(agent, mode, 3, jrand.PRNGKey(5))
    with pytest.raises(ValueError, match="face_heads"):
        _replay(agent, r, face_heads=None)


# --------------------------- the wire: live_faces.chunk + the host callback
class _StubStream:
    """A LiveFaceStream stand-in: `chunk` returns the 5-tuple including the
    approx-echo prefix length, exactly as the real one now does. FIVE, not
    six: the equation-id buffer was removed on 2026-09-13. The tokens are
    uint8 for the same reason the real ones are (``token_vocab``)."""

    def __init__(self, ct=5, head=2):
        self.ct, self.head = int(ct), int(head)

    def chunk(self, order, specs, n, vertex, vspecs, rows, skips, f,
              fh=None, kh=None):
        tok = np.zeros((W,), _TOKEN_DTYPE)
        tok[:self.ct] = np.arange(1, self.ct + 1, dtype=_TOKEN_DTYPE)
        return (tok, np.int32(self.ct), np.int32(3),
                np.asarray([1, 2], np.int32), np.int32(self.head))

    def n_faces(self, *a, **k):
        return 3


def _call_stub(emit_head):
    from alphagrad.approx.common.face_driver import make_face_callbacks
    cb, _ = make_face_callbacks(_StubStream(), window=W, emit_head=emit_head)
    zi = jnp.zeros((4,), jnp.int32)
    return cb(jnp.asarray(0, jnp.int32), zi, jnp.zeros((4, 4, 3), jnp.int32),
              jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
              jnp.zeros((4, 3), jnp.int32), jnp.zeros((3, 2, 3), jnp.int32),
              jnp.zeros((3,), jnp.int32), jnp.zeros((2, 3, 2, 3), jnp.int32),
              jnp.zeros((2, 3), jnp.int32))


def test_callback_head_wire_is_gated_and_correct():
    off = _call_stub(False)
    assert len(off) == 3      # (tokens, count, endpoints) -- was 4 with ids
    on = _call_stub(True)
    assert len(on) == 4
    assert int(on[-1]) == 2                   # the stub's approx-echo prefix
    for a, b in zip(off, on[:3]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_three_modes_give_three_different_policies(agent):
    """The read point actually changes what the head decides from."""
    lat = {}
    for m in MODES:
        r = _run(agent, m, 4, jrand.PRNGKey(31))
        _, _, _, l = _replay(agent, r)
        lat[m] = np.asarray(l)[:4]
    assert not np.allclose(lat["chunk-mean"], lat["own-span-mean"])
    assert not np.allclose(lat["own-span-mean"], lat["last-row"])
