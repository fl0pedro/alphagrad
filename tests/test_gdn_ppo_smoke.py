"""
Smoke test for gdn_vertex_ppo components.
Verifies that env, agent, rollout, and one training step work end-to-end.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as jnn
import pytest

import graphax.examples as examples
from alphagrad.vertexgame.vertex_game_w_tokens import (
    VertexEliminationEnv,
    MAX_TOKENS,
)
from alphagrad.ppo.gdn_vertex_ppo import (
    TransformerPPOAgent,
    get_log_probs_and_value,
    get_advantages,
)


@pytest.fixture
def env_and_meta():
    """Create a VertexEliminationEnv from the Simple example."""
    target_fn = examples.Helmholtz
    xs = (jnp.array([0.05, 0.15, 0.15, 0.2]),)
    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(closed_jaxpr, args=xs)
    total_v = len(closed_jaxpr.jaxpr.eqns)
    num_valid = len(env.valid_vertices)
    return env, total_v, num_valid


@pytest.fixture
def agent(env_and_meta):
    _, total_v, _ = env_and_meta
    key = jrand.PRNGKey(42)
    return TransformerPPOAgent(
        vocab_size=256,
        embd_dim=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        num_actions=total_v,
        policy_dims=[64, 32],
        value_dims=[64, 32],
        key=key,
    )


def test_env_reset(env_and_meta):
    env, total_v, num_valid = env_and_meta
    state = env.reset()
    assert state.tokens.shape == (MAX_TOKENS,)
    assert state.order.shape == (num_valid,)
    assert int(state.step_count) == 0
    assert state.terminated == False


def test_env_step(env_and_meta):
    env, total_v, num_valid = env_and_meta
    state = env.reset()
    valid = env.valid_vertices
    action = valid[0]  # first valid vertex
    out = env.step(state, action)
    assert out.state.tokens.shape == (MAX_TOKENS,)
    assert int(out.state.step_count) == 1


def test_env_vmap_reset(env_and_meta):
    env, _, num_valid = env_and_meta
    batch_size = 4
    states = jax.vmap(lambda _: env.reset())(jnp.arange(batch_size))
    assert states.tokens.shape == (batch_size, MAX_TOKENS)
    assert states.order.shape == (batch_size, num_valid)


def test_agent_forward(agent, env_and_meta):
    env, total_v, _ = env_and_meta
    state = env.reset()
    key = jrand.PRNGKey(0)
    logits, value = agent(state.tokens, key=key)
    assert logits.shape == (total_v,)
    assert value.shape == ()


def test_one_training_step(agent, env_and_meta):
    """Run one rollout + one gradient step and check loss is finite."""
    import equinox as eqx
    import optax
    from functools import partial
    import jax.lax as lax
    import distrax

    env, total_v, num_valid = env_and_meta
    valid_vertices = jnp.array(env.valid_vertices, dtype=jnp.int32)
    OBS_SHAPE = MAX_TOKENS
    NUM_ACTIONS = total_v
    ROLLOUT_LENGTH = num_valid
    NUM_ENVS = 2

    key = jrand.PRNGKey(123)

    # Reset envs
    env_states = jax.vmap(lambda _: env.reset())(jnp.arange(NUM_ENVS))

    # Simple sequential rollout (no vmap, just loop)
    all_samples = []
    for env_idx in range(NUM_ENVS):
        state_i = jax.tree.map(lambda x: x[env_idx], env_states)
        samples_i = []
        for step_idx in range(ROLLOUT_LENGTH):
            key, net_key, act_key = jrand.split(key, 3)
            logits, value = agent(state_i.tokens, key=net_key)
            prob_dist = jnn.softmax(logits, axis=-1)

            distribution = distrax.Categorical(probs=prob_dist)
            action_idx = distribution.sample(seed=act_key)
            action = action_idx + 1

            env_out = env.step(state_i, action)
            next_state = env_out.state
            reward = env_out.reward
            done = float(env_out.terminated)

            key, nk = jrand.split(key)
            _, next_value = agent(next_state.tokens, key=nk)

            sample = jnp.concatenate((
                state_i.tokens.astype(jnp.float32),
                jnp.array([action], dtype=jnp.float32),
                jnp.array([reward]),
                jnp.array([done]),
                jnp.array([value]),
                jnp.array([next_value]),
                prob_dist,
                jnp.array([1.0]),
            ))
            samples_i.append(sample)
            state_i = next_state

        all_samples.append(jnp.stack(samples_i))

    trajectories = jnp.stack(all_samples)  # (NUM_ENVS, ROLLOUT_LENGTH, features)

    # Compute advantages
    adv_data = get_advantages(
        trajectories[:, :, OBS_SHAPE + 1],
        trajectories[:, :, OBS_SHAPE + 2],
        trajectories[:, :, OBS_SHAPE + 3],
        trajectories[:, :, OBS_SHAPE + 4],
        trajectories[:, :, OBS_SHAPE + 5 + NUM_ACTIONS],
        0.95,
    )
    trajectories = jnp.concatenate([trajectories, adv_data], axis=-1)

    # Flatten for training
    batch = trajectories.reshape(-1, trajectories.shape[-1])

    # Compute loss
    keys = jrand.split(key, batch.shape[0])
    EPS = 0.2
    VALUE_WEIGHT = 0.5
    ENTROPY_WEIGHT = 0.01

    from alphagrad.ppo.gdn_vertex_ppo import (
        reward_normalization_fn,
        inverse_reward_normalization_fn,
        get_num_clipping_triggers,
    )
    from alphagrad.utils import entropy as ent_fn, explained_variance

    tokens = batch[:, :OBS_SHAPE].astype(jnp.int32)
    actions = batch[:, OBS_SHAPE].astype(jnp.int32)

    log_probs, prob_dist, values, entropies = get_log_probs_and_value(
        agent, tokens, actions, keys
    )

    assert jnp.all(jnp.isfinite(log_probs)), "log_probs not finite"
    assert jnp.all(jnp.isfinite(values)), "values not finite"

    # Check gradient flows
    def simple_loss(agent, batch, keys):
        tokens = batch[:, :OBS_SHAPE].astype(jnp.int32)
        actions = batch[:, OBS_SHAPE].astype(jnp.int32)
        log_probs, _, values, entropies = get_log_probs_and_value(
            agent, tokens, actions, keys
        )
        return jnp.mean(-log_probs) + 0.5 * jnp.mean(values ** 2)

    grads = eqx.filter_grad(simple_loss)(agent, batch, keys)
    # Just check that grads exist and are finite
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert len(grad_leaves) > 0, "No gradients computed"
    for i, g in enumerate(grad_leaves):
        assert jnp.all(jnp.isfinite(g)), f"Gradient leaf {i} not finite"
