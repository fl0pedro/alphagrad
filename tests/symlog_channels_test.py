"""--symlog-channels: the RAW ADDITIVE reward with an asymmetric transform.

Radical-simplification design (2026-08-26), element 1. Under
``--reward-mode additive --symlog-channels cost`` the composition is

    R = lambda_cmp*symlog(lat) + lambda_mem*symlog(mem) + lambda_acc*q_raw

i.e. the two cost channels keep the magnitude compression that makes 1e5-ns
latency and O(1) quality commensurable, and the QUALITY channel is left raw
so ``--lambda-acc`` prices one unit of loss_drop directly.

What is pinned here:

1. FLAG-OFF BIT-IDENTITY. ``--symlog-channels all`` (the default) reproduces
   HEAD's dispatch exactly, including the ``--reward-mode lagrangian``
   carve-out that already exempted the violation slot, and ``--no-symlog``
   stays a strict alias for ``none``.
2. ``cost`` exempts EXACTLY the quality slot; latency and memory are still
   symlog'd, bitwise as under ``all``.
3. THE THREE-SITES TRAP (CLEAN_DESIGN_AUDIT sec (c), c7/e6). The symlog trio
   is (reward transform, value target, GAE symexp). The reward transform is
   PER CHANNEL; the other two are uniform across channels AND mutually
   inverse, so a partial exemption cannot desynchronise them: whatever space
   a channel's reward was left in, the critic encodes into symlog space and
   GAE decodes straight back out of it. The round trip is tested PER CHANNEL,
   under every mode, and end to end through ``get_advantages``: a critic
   sitting exactly at ``_value_target(G)`` produces advantage 0 on every
   channel, whether or not that channel was symlog'd.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.ppo as ppo                              # noqa: E402
from alphagrad.approx.ppo import (                              # noqa: E402
    HEAD_REWARD_INDICES,
    NUM_REWARDS,
    REWARD_INDEX,
    _GAE_POPART,
    _set_no_symlog_indices,
    _symlog_rewards,
    _value_decode,
    _value_target,
    configure_symlog,
    get_advantages,
    resolve_symlog_channels,
)

QSLOT = int(REWARD_INDEX["cosine_sim"])
LAT = int(REWARD_INDEX["latency_ns"])
MEM = int(REWARD_INDEX["peak_memory"])


def _reset():
    _set_no_symlog_indices(())
    ppo._NO_SYMLOG_ALL[0] = False


def _ns(**kw):
    base = dict(symlog_channels="all", no_symlog=False, reward_mode="additive",
                advantage_norm="none", lambda_cmp=1.0, lambda_mem=1.0,
                lambda_acc=16.0)
    base.update(kw)
    return argparse.Namespace(**base)


def _rewards(seed=0, E=3, T=4):
    rng = np.random.default_rng(seed)
    r = rng.uniform(1.0, 3.0, (E, T, NUM_REWARDS)).astype(np.float32)
    r[..., LAT] = -rng.uniform(1e4, 3e5, (E, T))       # negated cost
    r[..., MEM] = -rng.uniform(1e6, 9e7, (E, T))       # negated cost
    r[..., QSLOT] = rng.uniform(-1.0, 1.0, (E, T))     # raw loss_drop range
    return jnp.asarray(r)


# ---------------------------------------------------------------- resolution

def test_resolve_defaults_and_alias():
    assert resolve_symlog_channels(_ns()) == "all"
    assert resolve_symlog_channels(_ns(symlog_channels="cost")) == "cost"
    assert resolve_symlog_channels(_ns(symlog_channels="none")) == "none"
    # --no-symlog IS --symlog-channels none.
    assert resolve_symlog_channels(_ns(no_symlog=True)) == "none"
    assert resolve_symlog_channels(
        _ns(no_symlog=True, symlog_channels="none")) == "none"
    # a namespace that predates the flag entirely still resolves.
    assert resolve_symlog_channels(argparse.Namespace(no_symlog=False)) == "all"


def test_resolve_rejects_contradiction():
    with pytest.raises(ValueError):
        resolve_symlog_channels(_ns(no_symlog=True, symlog_channels="cost"))
    with pytest.raises(ValueError):
        resolve_symlog_channels(_ns(symlog_channels="bogus"))


# --------------------------------------------------------- flag-off identity

def test_default_is_bitwise_the_module_import_state():
    """--symlog-channels all + additive == HEAD: every channel symlog'd."""
    r = _rewards()
    try:
        _reset()
        want = np.asarray(_symlog_rewards(r))
        _set_no_symlog_indices((QSLOT,))          # perturb, then reconfigure
        assert configure_symlog(_ns()) == "all"
        np.testing.assert_array_equal(np.asarray(_symlog_rewards(r)), want)
        assert ppo._NO_SYMLOG_ALL[0] is False
        assert ppo._NO_SYMLOG_REWARD_INDICES == ()
    finally:
        _reset()


def test_all_keeps_the_lagrangian_carveout():
    """HEAD exempted the violation slot in lagrangian mode; `all` must too."""
    r = _rewards(seed=1)
    try:
        assert configure_symlog(_ns(reward_mode="lagrangian")) == "all"
        out = np.asarray(_symlog_rewards(r))
        np.testing.assert_array_equal(out[..., QSLOT],
                                      np.asarray(r)[..., QSLOT])
        _reset()
        full = np.asarray(_symlog_rewards(r))
        np.testing.assert_array_equal(out[..., LAT], full[..., LAT])
        np.testing.assert_array_equal(out[..., MEM], full[..., MEM])
    finally:
        _reset()


def test_none_is_full_identity():
    r = _rewards(seed=2)
    try:
        assert configure_symlog(_ns(symlog_channels="none")) == "none"
        np.testing.assert_array_equal(np.asarray(_symlog_rewards(r)),
                                      np.asarray(r))
        _reset()
        assert configure_symlog(_ns(no_symlog=True)) == "none"
        np.testing.assert_array_equal(np.asarray(_symlog_rewards(r)),
                                      np.asarray(r))
    finally:
        _reset()


# --------------------------------------------------------------- cost mode

def test_cost_exempts_exactly_the_quality_channel():
    r = _rewards(seed=3)
    try:
        _reset()
        full = np.asarray(_symlog_rewards(r))
        assert configure_symlog(_ns(symlog_channels="cost")) == "cost"
        out = np.asarray(_symlog_rewards(r))
        # quality: bitwise RAW, and materially different from symlog'd.
        np.testing.assert_array_equal(out[..., QSLOT],
                                      np.asarray(r)[..., QSLOT])
        assert np.all(out[..., QSLOT] != full[..., QSLOT])
        # every other channel, latency and memory in particular: bitwise
        # the all-symlog baseline.
        other = [j for j in range(NUM_REWARDS) if j != QSLOT]
        np.testing.assert_array_equal(out[..., other], full[..., other])
        assert np.abs(out[..., LAT]).max() < 30.0
        assert np.abs(out[..., MEM]).max() < 30.0
    finally:
        _reset()


def test_cost_composition_is_plain_additive():
    """R == l_cmp*symlog(lat) + l_mem*symlog(mem) + l_acc*q_raw."""
    r = _rewards(seed=4)
    l_cmp, l_mem, l_acc = 1.0, 1.0, 16.0
    try:
        assert configure_symlog(
            _ns(symlog_channels="cost", lambda_cmp=l_cmp, lambda_mem=l_mem,
                lambda_acc=l_acc)) == "cost"
        sl = np.asarray(_symlog_rewards(r))
        head = sl[..., np.asarray(HEAD_REWARD_INDICES)]     # (E,T,3)
        got = head @ np.asarray([l_cmp, l_mem, l_acc], np.float32)
        raw = np.asarray(r)
        sym = np.asarray(ppo.reward_normalization_fn(r))
        want = (l_cmp * sym[..., LAT] + l_mem * sym[..., MEM]
                + l_acc * raw[..., QSLOT])
        np.testing.assert_allclose(got, want, rtol=0, atol=0)
    finally:
        _reset()


# ------------------------------------------------- the three-sites round trip

@pytest.mark.parametrize("mode", ["all", "cost", "none"])
def test_value_encode_decode_round_trips_per_channel(mode):
    """Site 2 (``_value_target``) and site 3 (the GAE value decode) are
    mutually inverse on EVERY channel, so a per-channel exemption at site 1
    cannot desynchronise them."""
    rng = np.random.default_rng(7)
    try:
        configure_symlog(_ns(symlog_channels=mode))
        for j in range(NUM_REWARDS):
            scale = 1e5 if j in (LAT, MEM) else 1.0
            x = jnp.asarray(rng.normal(size=(64,)).astype(np.float32) * scale)
            back = _value_decode(_value_target(x))
            np.testing.assert_allclose(np.asarray(back), np.asarray(x),
                                       rtol=2e-4, atol=1e-4)
    finally:
        _reset()


def test_gae_variant_follows_the_value_encoding_not_popart():
    """REGRESSION (2026-08-26). The GAE variant is chosen by which decode
    inverts `_value_target`, i.e. `use_popart or _NO_SYMLOG_ALL[0]`. With
    `--no-symlog --advantage-norm none` the OLD predicate (`use_popart`
    alone) picked the symexp variant against a RAW critic target and
    exponentiated every baseline."""
    import inspect
    src = inspect.getsource(ppo)
    assert ("_gae = (_GAE_POPART if (use_popart or _NO_SYMLOG_ALL[0])\n"
            "                else get_advantages)") in src
    # and the two variants really do differ where it matters.
    E, T, H = 1, 3, 1
    r = jnp.zeros((E, T, H)).at[:, -1, :].set(4.0)
    done = jnp.zeros((E, T), jnp.float32).at[:, -1].set(1.0)
    disc = jnp.ones((E, T), jnp.float32)
    v = jnp.full((E, T, H), 2.0, jnp.float32)
    a_sym = np.asarray(get_advantages(r, done, v, v, disc, 1.0)[2])
    a_raw = np.asarray(_GAE_POPART(r, done, v, v, disc, 1.0)[2])
    assert not np.allclose(a_sym, a_raw)


@pytest.mark.parametrize("mode", ["all", "cost", "none"])
def test_perfect_critic_gives_zero_advantage_on_every_channel(mode):
    """End-to-end through GAE, in the terminal-only / gamma=lambda=1 regime
    the campaign runs: a critic parked at ``_value_target(G)`` must produce
    advantage 0 on EVERY channel -- including the one whose reward was left
    raw. This is the actual failure mode a mismatched trio would show."""
    E, T = 4, 6
    r = _rewards(seed=8, E=E, T=T)
    try:
        configure_symlog(_ns(symlog_channels=mode))
        sl = _symlog_rewards(r)
        head = sl[..., jnp.asarray(HEAD_REWARD_INDICES)]         # (E,T,H)
        # terminal-only: only the last step carries reward.
        head = jnp.zeros_like(head).at[:, -1, :].set(head[:, -1, :])
        done = jnp.zeros((E, T), jnp.float32).at[:, -1].set(1.0)
        disc = jnp.ones((E, T), jnp.float32)
        # G_t = the terminal reward for every t (gamma = 1, terminal only).
        G = jnp.broadcast_to(head[:, -1, :][:, None, :], head.shape)
        value = _value_target(G)                                 # site 2
        nvalue = jnp.concatenate([value[:, 1:, :], value[:, -1:, :]], axis=1)
        # site 3, selected exactly as train_episode selects it (no PopArt).
        _gae = _GAE_POPART if ppo._NO_SYMLOG_ALL[0] else get_advantages
        _, estim, adv = _gae(head, done, value, nvalue, disc, 1.0)
        np.testing.assert_allclose(np.asarray(adv), 0.0, atol=2e-2)
        np.testing.assert_allclose(np.asarray(estim), np.asarray(G),
                                   rtol=2e-3, atol=2e-2)
    finally:
        _reset()
