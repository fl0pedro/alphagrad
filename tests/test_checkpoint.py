"""Smoke test for ``alphagrad.approx.common.checkpoint``.

Verifies save → load roundtrip on a tiny equinox module + optax state
plus the meta fields the trainers care about (episode counter,
reward weights, multipliers).
"""

from __future__ import annotations

import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax


class _TinyModel(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, key):
        self.linear = eqx.nn.Linear(4, 2, key=key)

    def __call__(self, x):
        return self.linear(x)


def test_save_and_load_roundtrip(tmp_path):
    from alphagrad.approx.common.checkpoint import save_state, load_state

    key = jrand.PRNGKey(0)
    model = _TinyModel(key)
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    # Mutate the model so we have something distinct to detect.
    altered = eqx.tree_at(
        lambda m: m.linear.weight, model, jnp.ones_like(model.linear.weight),
    )

    rw = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
    mults = np.array([0.3, 0.7], dtype=np.float32)
    ckpt_dir = str(tmp_path / "ckpt")

    save_state(
        ckpt_dir,
        agent=altered,
        opt_state=opt_state,
        episode_counter=42,
        reward_weights=rw,
        multipliers=mults,
        best_state={"best_global_return": -1.23e6, "best_global_ep": 7},
        extras={"variant": "full"},
    )

    # Files written.
    for name in ("agent.eqx", "opt_state.pkl", "meta.json"):
        assert (tmp_path / "ckpt" / name).exists()

    # Load round-trips via template.
    template = _TinyModel(key)
    restored = load_state(
        ckpt_dir,
        template_agent=template,
        template_opt_state=opt_state,
    )
    assert restored is not None
    assert restored["episode_counter"] == 42
    np.testing.assert_allclose(restored["reward_weights"], rw)
    np.testing.assert_allclose(restored["multipliers"], mults)
    assert restored["best_state"]["best_global_ep"] == 7
    assert restored["extras"]["variant"] == "full"
    # Agent params actually match the SAVED (altered) state — i.e. the
    # roundtrip preserves the leaves, doesn't return the template.
    np.testing.assert_allclose(
        np.asarray(restored["agent"].linear.weight),
        np.asarray(altered.linear.weight),
    )


def test_load_returns_none_for_missing_dir(tmp_path):
    from alphagrad.approx.common.checkpoint import load_state
    template = _TinyModel(jrand.PRNGKey(0))
    assert load_state(str(tmp_path / "missing"), template_agent=template) is None
    assert load_state("", template_agent=template) is None


def test_partial_writes_are_recoverable(tmp_path):
    """meta.json missing → load returns None (rather than crashing)."""
    from alphagrad.approx.common.checkpoint import save_state, load_state

    key = jrand.PRNGKey(0)
    model = _TinyModel(key)
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    ckpt_dir = str(tmp_path / "ckpt")
    save_state(
        ckpt_dir, agent=model, opt_state=opt_state, episode_counter=0,
    )
    # Simulate a partial write — remove meta.json.
    os.remove(os.path.join(ckpt_dir, "meta.json"))
    restored = load_state(
        ckpt_dir, template_agent=model, template_opt_state=opt_state,
    )
    assert restored is None
