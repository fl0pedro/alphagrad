"""PopArt and symlog are two DIFFERENT value encodings; applying both overflows.

`get_advantages` is `make_get_advantages(use_symlog=True)`: its scan decodes the
value head with `symexp`. That is right when the head emits symlog. Under PopArt
the head emits a z-score which ppo.py has *already* decoded affinely
(`value * sigma + mu`) before calling GAE — so the symlog variant exponentiates
an already-decoded value.

The failure is delayed by exactly one update, which is what made it read as a
slow collapse instead of a type error: on the first update PopArt is cold
(sigma=1, mu=0), v_raw stays small, and symexp of it is harmless. On the second
PopArt is warm, v_raw reaches the tens, and symexp overflows float32.
"""
import jax.numpy as jnp
import numpy as np

from alphagrad.approx.common.gae import get_advantages, make_get_advantages

GAE_POPART = make_get_advantages(use_symlog=False)


def _traj(value_scale):
    """One env, 8 steps, symlog-scale rewards and a PopArt-decoded value."""
    T = 8
    rewards = jnp.full((1, T, 1), 17.4)          # symlog scale, as observed
    dones = jnp.zeros((1, T, 1))
    values = jnp.full((1, T, 1), value_scale)
    next_values = jnp.full((1, T, 1), value_scale)
    discounts = jnp.full((1, T, 1), 0.99)
    return rewards, dones, values, next_values, discounts


def test_symlog_gae_overflows_on_a_popart_decoded_value():
    """The bug, pinned. v_raw ~89 is what round 2 actually produced."""
    _, est, adv = get_advantages(*_traj(89.33), 0.95)
    assert not np.all(np.isfinite(np.asarray(adv))), (
        "expected symexp(89) to overflow float32 — if this now passes, the "
        "symlog GAE variant changed and the guard below needs rechecking"
    )


def test_popart_gae_stays_finite_on_the_same_input():
    """The fix: no second decode, so the same input is unremarkable."""
    _, est, adv = GAE_POPART(*_traj(89.33), 0.95)
    assert np.all(np.isfinite(np.asarray(adv))), "PopArt GAE must not overflow"
    assert np.all(np.isfinite(np.asarray(est)))
    # Bounded by the geometric sum of the TD residual, not by exp().
    assert np.max(np.abs(np.asarray(adv))) < 1e4


def test_the_two_variants_agree_when_the_value_is_near_zero():
    """symexp(x) ~ x for small x, so a cold-PopArt round is identical.

    This is the reason the bug survived a full first update undetected.
    """
    a = np.asarray(get_advantages(*_traj(1e-4), 0.95)[2])
    b = np.asarray(GAE_POPART(*_traj(1e-4), 0.95)[2])
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_popart_gae_scales_linearly_not_exponentially():
    """Doubling the decoded value must not square the advantage."""
    small = np.max(np.abs(np.asarray(GAE_POPART(*_traj(10.0), 0.95)[2])))
    large = np.max(np.abs(np.asarray(GAE_POPART(*_traj(20.0), 0.95)[2])))
    assert large < 4 * max(small, 1e-6), (
        f"advantage grew super-linearly ({small} -> {large}); the value is "
        "being decoded more than once again"
    )
