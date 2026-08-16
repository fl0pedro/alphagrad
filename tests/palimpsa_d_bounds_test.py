# -*- coding: utf-8 -*-
"""Palimpsa-D: the two bounds on the precision state I.

Paper A.1 puts two bounds on

    I_t = alpha*I_{t-1} + (1-alpha)*I_prior + beta (x) k^2

that this codebase was missing: Palimpsa-D L2-normalises q and k, and the
reference bounds beta as sigmoid(.)*softplus(scale) rather than an unbounded
softplus. With neither, nothing bounds ||k||^2 or beta, so one large
beta*||k||^2 drives I up, 1/I -> 0, and that head stops integrating -- the
catastrophic remembering the forgetting term exists to prevent. It presents as
silent quality loss, never as an error, which is why it is pinned here.

The measured gap on one extreme token: 449083.0 unbounded vs 0.818 bounded.
"""
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import pytest

from alphagrad.transformer.palimpsa_encoder import (
    PalimpsaMixer, palimpsa_qk_norm, palimpsa_beta,
    PALIMPSA_QK_NORM, PALIMPSA_BETA_BOUNDED)

H, D = 2, 16

on_only = pytest.mark.skipif(
    not (PALIMPSA_QK_NORM and PALIMPSA_BETA_BOUNDED),
    reason="ALPHAGRAD_PALIMPSA_{QK_NORM,BETA_BOUNDED} disabled")


def _x_with_zero_row(scale=3.0):
    """Random rows plus one all-zero row -- a padding token."""
    x = jrand.normal(jrand.PRNGKey(0), (7, H, D)) * scale
    return x.at[3].set(0.0)


@on_only
def test_qk_norm_gives_unit_rows():
    xn = palimpsa_qk_norm(_x_with_zero_row())
    n = jnp.sqrt(jnp.sum(xn * xn, axis=-1))
    live = jnp.concatenate([n[:3], n[4:]]).ravel()
    assert float(jnp.max(jnp.abs(live - 1.0))) < 1e-5


def test_qk_norm_gradient_is_finite_on_a_zero_row():
    """REGRESSION. The eps must be INSIDE the sqrt.

    Two spellings that look safe and are not:
      * ``where(n > 0, x / n, x)`` -- guards the forward, still differentiates
        the unsafe branch.
      * ``x / maximum(n, eps)`` -- guards the DIVISION and sends a zero
        cotangent back to n, but n = sqrt(sum(x*x)) has d sqrt/du -> inf at
        u = 0, so reverse mode evaluates 0 * inf = NaN. This was MEASURED as
        max|grad| = nan before the rsqrt form landed.
    """
    x = _x_with_zero_row()
    g = jax.grad(lambda z: jnp.sum(palimpsa_qk_norm(z) ** 2))(x)
    assert bool(jnp.all(jnp.isfinite(g))), "NaN/inf gradient on the zero row"


@on_only
def test_beta_is_bounded_above():
    mixer = PalimpsaMixer(num_heads=H, embd_dim=H * D, key=jrand.PRNGKey(0))
    raw = jrand.normal(jrand.PRNGKey(1), (200, H, D)) * 20.0   # extreme
    b = palimpsa_beta(raw, mixer.b_scale_raw)
    cap = float(jnp.max(jnn.softplus(mixer.b_scale_raw)))
    assert float(jnp.max(b)) <= cap + 1e-6
    assert bool(jnp.all(b > 0)), "beta must stay positive: I_t must not cross 0"


@on_only
def test_beta_init_matches_the_legacy_softplus():
    """The bounded form must START where the unbounded one did.

    Otherwise this is not a bound, it is a re-scaling, and every downstream
    comparison against earlier runs silently shifts.
    """
    mixer = PalimpsaMixer(num_heads=H, embd_dim=H * D, key=jrand.PRNGKey(0))
    b0 = float(jnp.mean(palimpsa_beta(jnp.zeros((1, H, D)), mixer.b_scale_raw)))
    assert abs(b0 - float(jnn.softplus(jnp.array(0.0)))) < 0.02


@on_only
def test_precision_increment_stays_bounded():
    """The property the whole change exists to buy."""
    mixer = PalimpsaMixer(num_heads=H, embd_dim=H * D, key=jrand.PRNGKey(0))
    raw = jrand.normal(jrand.PRNGKey(1), (200, H, D)) * 20.0
    b = palimpsa_beta(raw, mixer.b_scale_raw)
    k = palimpsa_qk_norm(raw)
    inc = b[:, :, :, None] * (k[:, :, None, :] ** 2)
    assert float(jnp.max(inc)) < 10.0


def test_b_scale_exists_regardless_of_the_flag():
    """The parameter tree must not depend on the env flag, or a checkpoint
    written with the bound on cannot be read with it off."""
    mixer = PalimpsaMixer(num_heads=H, embd_dim=H * D, key=jrand.PRNGKey(0))
    assert tuple(mixer.b_scale_raw.shape) == (H,)


def test_mixer_forward_is_finite():
    mixer = PalimpsaMixer(num_heads=H, embd_dim=H * D, key=jrand.PRNGKey(0))
    toks = jrand.normal(jrand.PRNGKey(2), (11, H * D))
    out = mixer(toks, toks, toks)
    assert out.shape == (11, H * D)
    assert bool(jnp.all(jnp.isfinite(out)))
