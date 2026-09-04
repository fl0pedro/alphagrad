"""GRAD-COSINE IS THE DEFAULT QUALITY METRIC (ticket dsnn-3qm.39).

Owner ruling, 2026-09-02: quality = the gradient cosine between the candidate
plan's gradient and the rev-exact gradient on the same probe batch, in every
run and every sweep. ``--quality-metric auto`` therefore resolves to
``grad_cosine`` on every scalar-loss target. ``loss_drop`` (the 200-step Adam
walk) stays selectable by name and is never the default: finding 51 shows a
plan scoring 0.885 on loss_drop while its gradient points elsewhere.

Pinned here, on the smallest scalar-loss target the suite already uses (the
16x6 -> 8 -> 3 tanh MLP of ``walk_heldout_test``; never TLM):

1. ``quality_metric``'s ``auto`` is ``grad_cosine`` for a scalar target and
   ``jac_cosine`` for an analytic (full-Jacobian) one.
2. ``from_jaxpr`` reads the scalar-target fact off the jaxpr, so an env built
   from a loss resolves to ``grad_cosine`` without any flag.
3. A full episode under ``auto`` puts the GRADIENT COSINE in reward slot 6
   (``REWARD_NAMES[6] == "quality"``): the number is bit-identical to the one
   an explicit ``--quality-metric grad_cosine`` produces, it is 1.0 for the
   exact plan (its gradient IS the reference), and it is not the loss-drop
   walk's number.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import jax.random as jrand                                      # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as envmod                           # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    FACE_SLOTS,
    MAX_FACES,
    MAX_RULES_PER_VERTEX,
    REWARD_INDEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
    quality_metric,
)

_QM_ENV = ("ALPHAGRAD_QUALITY_METRIC", "ALPHAGRAD_WALK_STEPS")


@pytest.fixture(autouse=True)
def _no_quality_metric_in_the_environment():
    """Every test starts from "nothing selected" -- the ``auto`` path -- and
    leaves the process environment as it found it."""
    saved = {k: os.environ.get(k) for k in _QM_ENV}
    for k in _QM_ENV:
        os.environ.pop(k, None)
    envmod._PROBE_BATCH.clear()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    envmod._PROBE_BATCH.clear()


# ------------------------------------------------------------- the target
# The MLP fixture of ``walk_heldout_test``: a fixed teacher so the labels are
# a learnable function of the inputs.
_B, _D, _H, _C = 16, 6, 8, 3
_TEACHER = jrand.normal(jrand.PRNGKey(11), (_D, _C))


def _labels(x):
    return jax.nn.one_hot(jnp.argmax(x @ _TEACHER, axis=-1), _C)


def _data_gen(keys):
    x = jrand.normal(keys[0], (_B, _D))
    return x, _labels(x)


def _loss(x, y, w1, w2):
    h = jnp.tanh(x @ w1)
    logits = h @ w2
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(jnp.sum(y * logp, axis=-1))


def _base_args(seed=0):
    k1, k3, k4 = jrand.split(jrand.PRNGKey(seed), 3)
    x = jrand.normal(k1, (_B, _D))
    w1 = 0.5 * jrand.normal(k3, (_D, _H))
    w2 = 0.5 * jrand.normal(k4, (_H, _C))
    return [x, _labels(x), w1, w2]


_GEN = jax.jit(_data_gen)


def _make_env():
    xs = _base_args()
    closed = jax.make_jaxpr(_loss)(*xs)
    return VertexEliminationEnv.from_jaxpr(
        closed, args=xs, argnums=(2, 3), num_envs=0,
        data_gen=_GEN, target_fun=_loss, terminal_rewards_only=True,
    )


def _run_episode(env) -> np.ndarray:
    """The EXACT plan: every vertex in jaxpr order, no rule, no skip."""
    state = env.reset()
    no_rules = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, jnp.int32)
    no_rules = no_rules.at[..., 2].set(0)
    face_rows = jnp.full((MAX_FACES, FACE_SLOTS, 3), -1, jnp.int32)
    face_skip = jnp.zeros((MAX_FACES,), jnp.int32)
    for v in [int(x) for x in np.asarray(env.valid_vertices)]:
        state = env.step(
            state,
            StepAction(jnp.asarray(v, jnp.int32), no_rules,
                       face_rows, face_skip),
        ).state
    return np.asarray(state.reward)


# ------------------------------------------------------------- 1. resolution

def test_auto_is_grad_cosine_for_a_scalar_loss():
    scalar = SimpleNamespace(scalar_target=True)
    analytic = SimpleNamespace(scalar_target=False)
    assert quality_metric(scalar) == "grad_cosine"
    assert quality_metric(analytic) == "jac_cosine"
    os.environ["ALPHAGRAD_QUALITY_METRIC"] = "auto"
    assert quality_metric(scalar) == "grad_cosine"
    assert quality_metric(analytic) == "jac_cosine"


def test_loss_drop_stays_selectable_by_name_and_is_never_the_default():
    scalar = SimpleNamespace(scalar_target=True)
    os.environ["ALPHAGRAD_QUALITY_METRIC"] = "loss_drop"
    assert quality_metric(scalar) == "loss_drop"
    os.environ.pop("ALPHAGRAD_QUALITY_METRIC")
    assert quality_metric(scalar) != "loss_drop"


def test_deprecated_cosine_alias_keeps_its_mapping():
    os.environ["ALPHAGRAD_QUALITY_METRIC"] = "cosine"
    assert quality_metric(SimpleNamespace(scalar_target=True)) == "grad_cosine"
    assert quality_metric(SimpleNamespace(scalar_target=False)) == "jac_cosine"


# ------------------------------------------------------------- 2. the env

def test_env_built_from_a_loss_resolves_to_grad_cosine_without_a_flag():
    env = _make_env()
    assert env.config.scalar_target is True
    assert quality_metric(env.config) == "grad_cosine"


# ------------------------------------------------------------- 3. slot 6

def test_scalar_loss_episode_reports_grad_cosine_in_reward_slot_6():
    q = REWARD_INDEX["quality"]
    assert q == 6 and REWARD_NAMES[6] == "quality"

    r_auto = _run_episode(_make_env())

    os.environ["ALPHAGRAD_QUALITY_METRIC"] = "grad_cosine"
    r_named = _run_episode(_make_env())

    # ``auto`` IS ``grad_cosine``: the same number, bit for bit.
    assert float(r_auto[q]) == float(r_named[q]), (r_auto[q], r_named[q])
    # The exact plan's gradient is the reference gradient, so its cosine is
    # 1.0 up to floating-point summation order.
    assert abs(float(r_auto[q]) - 1.0) < 1e-4, r_auto[q]

    # ...and it is NOT the loss-drop walk's number. (On this fixture the walk
    # declares itself UNDEFINED inside ``_callback`` and reads 0.0 -- which is
    # what the old ``auto`` silently put in slot 6 for this env.)
    os.environ["ALPHAGRAD_QUALITY_METRIC"] = "loss_drop"
    os.environ["ALPHAGRAD_WALK_STEPS"] = "5"
    r_walk = _run_episode(_make_env())
    assert float(r_walk[q]) != float(r_auto[q]), (r_walk[q], r_auto[q])
    assert abs(float(r_walk[q]) - 1.0) > 1e-2, r_walk[q]
