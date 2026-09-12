"""P1a: per-path actions flow env-side — StepAction.face_rows/face_skip →
EnvState histories → _callback → graphax face_transforms (SKIP_FACE / hooks).

Hand-built actions, no policy: a skipped face must move the measured quality
away from exact while the same episode without face actions stays cos == 1.
"""
import os


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.env import (
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    REWARD_INDEX,
    StepAction,
    VertexEliminationEnv,
)

_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)
_Q = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 17.0 + 0.3)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _square(x):
    e = _M @ x
    return _P @ e, _Q @ e


@pytest.fixture(autouse=True)
def _the_quality_channel_this_module_measures(monkeypatch):
    """THE DEFAULT QUALITY METRIC, declared rather than assumed.

    Both assertions below read the quality slot (``REWARD_INDEX["cosine_sim"]``)
    and one of them requires it to be 1.0 on the exact plan. With
    ALPHAGRAD_QUALITY_METRIC=none the channel is switched off and the slot reads
    0.0, so the module fails with no hint that a configuration -- not the face
    wire -- is what moved. That is what happened for the whole campaign:
    tests/mem_channel_test.py and tests/paired_log_reward_test.py set
    QUALITY_METRIC=none at MODULE SCOPE, pytest imports every module during
    collection, and this module is collected earlier (f < m < p), so by the time
    its tests ran the channel was off. Those two modules now declare that in a
    fixture of their own; this one declares the default it needs, so it no
    longer depends on nobody else having spoken.
    """
    monkeypatch.delenv("ALPHAGRAD_QUALITY_METRIC", raising=False)


def _make_env():
    closed = jax.make_jaxpr(_square)(_X4)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=[_X4], argnums=(0,), num_envs=0, target_fun=_square,
    )


def _run_episode(env, skip_face_of_vertex=None):
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
        face_skip = jnp.zeros((MAX_FACES,), jnp.int32)
        if skip_face_of_vertex is not None and v == skip_face_of_vertex:
            face_skip = face_skip.at[0].set(1)
        out = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        )
        state = out.state
    return np.asarray(state.reward)


def test_no_face_actions_is_exact():
    env = _make_env()
    r = _run_episode(env)
    assert r[REWARD_INDEX["cosine_sim"]] > 0.999999
    assert r[REWARD_INDEX["frob_residual"]] == 0.0  # stored as -residual


def test_skipping_one_face_moves_the_measured_quality():
    env = _make_env()
    r_exact = _run_episode(env)
    r_skip = _run_episode(env, skip_face_of_vertex=1)
    cos_i = REWARD_INDEX["cosine_sim"]
    assert r_skip[cos_i] < r_exact[cos_i] - 1e-6, (
        "SKIP_FACE must reach the measured Jacobian"
    )
    # A dropped contraction can only make the count pass cheaper (costs are
    # stored NEGATED: cheaper == greater-or-equal).
    fmas_i = REWARD_INDEX["muls_adds_fmas"]
    assert r_skip[fmas_i] >= r_exact[fmas_i]


def test_legacy_two_field_step_action_still_works():
    env = _make_env()
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    v = int(np.asarray(env.valid_vertices)[0])
    out = env.step(state, StepAction(jnp.asarray(v, jnp.int32), no_rules))
    assert bool(np.asarray(out.state.step_count) == 1)
