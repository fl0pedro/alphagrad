# -*- coding: utf-8 -*-
"""The Fast Palimpsa Pallas kernel against its pure-JAX oracle.

WHAT THIS PINS
--------------
``alphagrad.transformer.fast_palimpsa_pallas`` ships two things that must
agree: ``fast_palimpsa_ref``, a pure-JAX transcription of upstream's
differentiable contract, and a pair of hand-written Pallas kernels wrapped in
a ``custom_vjp``. Nothing else in the repo compares them, so this file does,
in both directions (forward and every input gradient) and at the one place
the whole design rests on -- the chunk boundary, where the carried state is
claimed to be EXACT.

WHY THE TOLERANCES DIFFER BY AXIS
---------------------------------
On CPU the kernel runs under ``interpret=True``, so it executes the same JAX
ops the oracle does, just in a different order -- the gap is float
reassociation and lands near machine epsilon. On GPU the same code lowers to
Triton and the dots go through tensor cores; upstream quotes about 1e-3 to
5e-3 relative there. One tolerance covers both, sized for the GPU, except
where an EXACT claim is being made (the carry round trip, the padded tail),
which is pinned at zero.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import lax

from alphagrad.transformer.fast_palimpsa_pallas import (
    CHUNK_C, fast_palimpsa, fast_palimpsa_ref)

jax.config.update("jax_enable_x64", False)

#: Sized for the GPU tensor-core path; CPU interpret comes in far under it.
RTOL = 5e-3
ATOL = 5e-4


def _inputs(B=2, T=64, H=3, DK=16, DV=16, seed=0):
    """Inputs in the range the encoder actually produces.

    ``b >= 0`` and ``gt >= 0`` are not cosmetic: the recurrence is only
    well-posed for a non-negative precision increment (see ``palimpsa_beta``
    in ``palimpsa_encoder``), and a negative one would let ``I`` cross zero
    and make every relative comparison below meaningless.
    """
    ks = jax.random.split(jax.random.PRNGKey(seed), 7)
    q = jax.random.normal(ks[0], (B, T, H, DK), jnp.float32)
    k = jax.random.normal(ks[1], (B, T, H, DK), jnp.float32)
    v = jax.random.normal(ks[2], (B, T, H, DV), jnp.float32)
    b = jax.nn.sigmoid(jax.random.normal(ks[3], (B, T, H, DV), jnp.float32))
    gt = jax.nn.softplus(jax.random.normal(ks[4], (B, T, H), jnp.float32))
    g = jax.nn.softplus(jax.random.normal(ks[5], (H,), jnp.float32))
    Ip = jax.nn.softplus(jax.random.normal(ks[6], (H,), jnp.float32)) + 0.2
    return q, k, v, b, gt, g, Ip


def _state0(B, H, DV, DK, Ip):
    M0 = jnp.zeros((B, H, DV, DK), jnp.float32)
    I0 = jnp.broadcast_to(Ip.reshape(1, H, 1, 1), (B, H, DV, DK))
    return M0, jnp.asarray(I0, jnp.float32)


def _exact_state(q, k, v, b, gt, g, Ip, M0, I0):
    """The token-by-token (M, I) the EXACT recurrence ends at.

    Deliberately a separate scan rather than a call into
    ``palimpsa_pallas``: the claim under test is that Fast Palimpsa's chunk
    boundary reproduces exact Palimpsa's state, and an oracle that shared
    code with either side would not test it.
    """
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    gh = g.reshape(1, H, 1, 1)
    Iph = Ip.reshape(1, H, 1, 1)

    def step(carry, inp):
        M, I = carry
        k_t, v_t, b_t, gt_t = inp
        f = jnp.exp(-gt_t[:, :, None, None] * gh)
        M2 = v_t[:, :, :, None] * k_t[:, :, None, :] + f * M
        I2 = (b_t[:, :, :, None] * (k_t[:, :, None, :] ** 2)
              + (1.0 - f) * Iph + f * I)
        return (M2, I2), None

    xs = (jnp.transpose(k, (1, 0, 2, 3)), jnp.transpose(v, (1, 0, 2, 3)),
          jnp.transpose(b, (1, 0, 2, 3)), jnp.transpose(gt, (1, 0, 2)))
    (M, I), _ = lax.scan(step, (M0, I0), xs)
    return M, I


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------
@pytest.mark.parametrize("DK,DV", [(16, 16), (12, 20), (32, 32)])
def test_the_pallas_kernel_matches_the_pure_jax_oracle_in_the_forward(DK, DV):
    q, k, v, b, gt, g, Ip = _inputs(DK=DK, DV=DV)
    M0, I0 = _state0(q.shape[0], q.shape[2], DV, DK, Ip)
    got = fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                        initial_M=M0, initial_I=I0)
    want = fast_palimpsa_ref(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                             initial_M=M0, initial_I=I0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=RTOL, atol=ATOL)


def test_the_kernel_and_the_oracle_agree_with_no_carry_handed_in():
    """The default state (M = 0, I = Ip) has to be built the same on both
    sides. It is the state every fresh ``carry_init`` hands over."""
    q, k, v, b, gt, g, Ip = _inputs(seed=3)
    got = fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C)
    want = fast_palimpsa_ref(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=RTOL, atol=ATOL)


def test_the_returned_final_state_matches_the_oracles_final_state():
    q, k, v, b, gt, g, Ip = _inputs(seed=4)
    _o, Mf, If = fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                               output_final_state=True)
    _w, Mw, Iw = fast_palimpsa_ref(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                                   output_final_state=True)
    np.testing.assert_allclose(np.asarray(Mf), np.asarray(Mw),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.asarray(If), np.asarray(Iw),
                               rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------
# the exactness claim at the boundary
# --------------------------------------------------------------------------
def test_the_chunk_boundary_state_is_the_exact_recurrences_own_state():
    """THE claim Fast Palimpsa rests on. Only the within-chunk READ is
    approximated; the state handed to the next chunk is what exact Palimpsa's
    token-by-token recurrence would have produced at that position."""
    q, k, v, b, gt, g, Ip = _inputs(seed=5)
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    M0, I0 = _state0(B, H, DV, DK, Ip)
    _o, Mf, If = fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                               initial_M=M0, initial_I=I0,
                               output_final_state=True)
    Me, Ie = _exact_state(q, k, v, b, gt, g, Ip, M0, I0)
    assert _rel(Mf, Me) < RTOL, f"M drifted by {_rel(Mf, Me):.3e}"
    assert _rel(If, Ie) < RTOL, f"I drifted by {_rel(If, Ie):.3e}"


# --------------------------------------------------------------------------
# the carry round trip
# --------------------------------------------------------------------------
def test_two_calls_with_the_carry_equal_one_call_over_the_concatenation():
    """EXACT, tolerance zero. This is what lets the incremental encoder read
    a step's delta in pieces and the loss read it in one go, or the other way
    round, and still land on the same numbers -- which is the whole reason the
    PPO ratio is 1 at epoch 0."""
    q, k, v, b, gt, g, Ip = _inputs(T=4 * CHUNK_C, seed=6)
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    M0, I0 = _state0(B, H, DV, DK, Ip)
    whole, Mw, Iw = fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                                  initial_M=M0, initial_I=I0,
                                  output_final_state=True)
    half = 2 * CHUNK_C
    o1, M1, I1 = fast_palimpsa(
        q[:, :half], k[:, :half], v[:, :half], b[:, :half], gt[:, :half],
        g, Ip, chunk_size=CHUNK_C, initial_M=M0, initial_I=I0,
        output_final_state=True)
    o2, M2, I2 = fast_palimpsa(
        q[:, half:], k[:, half:], v[:, half:], b[:, half:], gt[:, half:],
        g, Ip, chunk_size=CHUNK_C, initial_M=M1, initial_I=I1,
        output_final_state=True)
    split = jnp.concatenate([o1, o2], axis=1)
    assert np.array_equal(np.asarray(whole), np.asarray(split))
    assert np.array_equal(np.asarray(Mw), np.asarray(M2))
    assert np.array_equal(np.asarray(Iw), np.asarray(I2))


def test_a_split_that_is_not_on_a_chunk_boundary_is_a_different_operator():
    """Stated so nobody wires a caller that splits mid-chunk by accident. The
    chunk grid is part of the operator: cutting at 48 tokens with C = 32 makes
    the second piece start a fresh chunk, so its first 16 tokens are read
    against a boundary the one-shot call never had."""
    q, k, v, b, gt, g, Ip = _inputs(T=4 * CHUNK_C, seed=7)
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    M0, I0 = _state0(B, H, DV, DK, Ip)
    whole = fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                          initial_M=M0, initial_I=I0)
    cut = CHUNK_C + CHUNK_C // 2
    o1, M1, I1 = fast_palimpsa(
        q[:, :cut], k[:, :cut], v[:, :cut], b[:, :cut], gt[:, :cut],
        g, Ip, chunk_size=CHUNK_C, initial_M=M0, initial_I=I0,
        output_final_state=True)
    o2 = fast_palimpsa(
        q[:, cut:], k[:, cut:], v[:, cut:], b[:, cut:], gt[:, cut:],
        g, Ip, chunk_size=CHUNK_C, initial_M=M1, initial_I=I1)
    split = jnp.concatenate([o1, o2], axis=1)
    assert not np.allclose(np.asarray(whole), np.asarray(split),
                           rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------
# the padded tail
# --------------------------------------------------------------------------
def test_a_tail_shorter_than_one_chunk_is_padded_without_touching_real_rows():
    """The padding rule, pinned. ``gt = 0`` gives ``f = 1`` and ``k = v = b =
    0`` contributes nothing, so appending pad tokens changes no real row and
    no state a real row reads."""
    q, k, v, b, gt, g, Ip = _inputs(T=3 * CHUNK_C, seed=8)
    T_short = 2 * CHUNK_C + 11
    short = fast_palimpsa(q[:, :T_short], k[:, :T_short], v[:, :T_short],
                          b[:, :T_short], gt[:, :T_short], g, Ip,
                          chunk_size=CHUNK_C)
    assert short.shape[1] == T_short

    def _zpad(x, n):
        z = [(0, 0)] * x.ndim
        z[1] = (0, n)
        return jnp.pad(x, z)

    npad = (-T_short) % CHUNK_C
    padded = fast_palimpsa(
        _zpad(q[:, :T_short], npad), _zpad(k[:, :T_short], npad),
        _zpad(v[:, :T_short], npad), _zpad(b[:, :T_short], npad),
        _zpad(gt[:, :T_short], npad), g, Ip, chunk_size=CHUNK_C)
    assert np.array_equal(np.asarray(short),
                          np.asarray(padded[:, :T_short]))


def test_the_oracle_refuses_a_length_that_is_not_a_whole_number_of_chunks():
    """The oracle takes whole chunks only, and says so instead of silently
    truncating or silently padding behind the caller's back."""
    q, k, v, b, gt, g, Ip = _inputs(T=CHUNK_C + 5, seed=9)
    with pytest.raises(ValueError, match="not a multiple of chunk_size"):
        fast_palimpsa_ref(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C)


def test_a_non_positive_chunk_size_is_rejected():
    q, k, v, b, gt, g, Ip = _inputs(T=CHUNK_C, seed=10)
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=0)


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------
def _loss_factory(fn, chunk_size):
    def loss(q, k, v, b, gt, g, Ip, M0, I0):
        out = fn(q, k, v, b, gt, g, Ip, chunk_size=chunk_size,
                 initial_M=M0, initial_I=I0)
        w = jnp.arange(1, out.size + 1, dtype=jnp.float32).reshape(out.shape)
        return jnp.sum(out * w) / out.size
    return loss


@pytest.mark.parametrize("arg,name", list(enumerate(
    ["q", "k", "v", "b", "gt", "g", "Ip", "initial_M", "initial_I"])))
def test_the_pallas_kernel_matches_the_oracle_on_every_input_gradient(arg, name):
    q, k, v, b, gt, g, Ip = _inputs(T=2 * CHUNK_C, seed=11)
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    M0 = jax.random.normal(jax.random.PRNGKey(21), (B, H, DV, DK), jnp.float32)
    I0 = (jnp.broadcast_to(Ip.reshape(1, H, 1, 1), (B, H, DV, DK))
          + jax.nn.softplus(jax.random.normal(
              jax.random.PRNGKey(22), (B, H, DV, DK), jnp.float32)))
    args = (q, k, v, b, gt, g, Ip, M0, jnp.asarray(I0, jnp.float32))
    got = jax.grad(_loss_factory(fast_palimpsa, CHUNK_C), argnums=arg)(*args)
    want = jax.grad(_loss_factory(fast_palimpsa_ref, CHUNK_C),
                    argnums=arg)(*args)
    r = _rel(got, want)
    assert r < RTOL, f"d{name} relative gap {r:.3e}"


def test_the_gradient_reaches_the_carry_that_the_next_chunk_receives():
    """A zero here would be silent: the loss differentiates a whole chain of
    per-step extends through the carry, so a kernel that dropped the
    state cotangent would train only the last step."""
    q, k, v, b, gt, g, Ip = _inputs(T=2 * CHUNK_C, seed=12)
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    M0, I0 = _state0(B, H, DV, DK, Ip)

    def loss(M0_, I0_):
        out = fast_palimpsa(q, k, v, b, gt, g, Ip, chunk_size=CHUNK_C,
                            initial_M=M0_, initial_I=I0_)
        return jnp.sum(out ** 2)

    dM, dI = jax.grad(loss, argnums=(0, 1))(M0, I0)
    assert float(jnp.max(jnp.abs(dM))) > 0.0
    assert float(jnp.max(jnp.abs(dI))) > 0.0


def test_the_final_state_cotangent_is_carried_back_through_every_chunk():
    """The carry interface is differentiated from BOTH ends: the loss reaches
    a step's output rows and, through the next step, its final state."""
    q, k, v, b, gt, g, Ip = _inputs(T=2 * CHUNK_C, seed=13)

    def loss_kernel(qq):
        _o, Mf, If = fast_palimpsa(qq, k, v, b, gt, g, Ip,
                                   chunk_size=CHUNK_C,
                                   output_final_state=True)
        return jnp.sum(Mf ** 2) + jnp.sum(If ** 2)

    def loss_ref(qq):
        _o, Mf, If = fast_palimpsa_ref(qq, k, v, b, gt, g, Ip,
                                       chunk_size=CHUNK_C,
                                       output_final_state=True)
        return jnp.sum(Mf ** 2) + jnp.sum(If ** 2)

    # M and I do not depend on q at all, so this one must be exactly zero,
    # and the v gradient must not be.
    assert float(jnp.max(jnp.abs(jax.grad(loss_kernel)(q)))) == 0.0

    def lk(vv):
        _o, Mf, If = fast_palimpsa(q, k, vv, b, gt, g, Ip,
                                   chunk_size=CHUNK_C,
                                   output_final_state=True)
        return jnp.sum(Mf ** 2) + jnp.sum(If ** 2)

    def lr(vv):
        _o, Mf, If = fast_palimpsa_ref(q, k, vv, b, gt, g, Ip,
                                       chunk_size=CHUNK_C,
                                       output_final_state=True)
        return jnp.sum(Mf ** 2) + jnp.sum(If ** 2)

    assert _rel(jax.grad(lk)(v), jax.grad(lr)(v)) < RTOL
    assert float(jnp.max(jnp.abs(jax.grad(loss_ref)(q)))) == 0.0


# --------------------------------------------------------------------------
# it survives the transforms the loss actually applies
# --------------------------------------------------------------------------
def test_the_kernel_survives_vmap_and_remat_the_way_the_loss_wraps_it():
    """The loss vmaps over minibatch samples and rematerialises the extend.
    A custom_vjp that only worked unbatched would fail there and nowhere
    else."""
    q, k, v, b, gt, g, Ip = _inputs(B=1, T=2 * CHUNK_C, seed=14)
    n = 3
    stack = lambda x: jnp.broadcast_to(x, (n,) + x.shape)

    @jax.checkpoint
    def one(qq, kk, vv, bb, gg):
        return fast_palimpsa(qq, kk, vv, bb, gg, g, Ip, chunk_size=CHUNK_C)

    def loss(qs, ks, vs, bs, gs):
        return jnp.sum(jax.vmap(one)(qs, ks, vs, bs, gs) ** 2)

    args = (stack(q), stack(k), stack(v), stack(b), stack(gt))
    val = loss(*args)
    dq = jax.grad(loss)(*args)
    assert np.isfinite(float(val))
    assert dq.shape == args[0].shape
    assert float(jnp.max(jnp.abs(dq))) > 0.0
