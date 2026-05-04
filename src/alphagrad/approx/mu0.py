import argparse
import os
from functools import partial

import distrax
import equinox as eqx
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import mctx
import numpy as np
import optax
import wandb
from tqdm import tqdm

from alphagrad.approx.common import (
    build_legacy_sp_valid_mask,
    data_gen,
    get_args,
    get_fn,
)
from alphagrad.approx.env import MAX_TOKENS, VertexEliminationEnv
from alphagrad.transformer import MLP, Encoder, PositionalEncoder


# ---------------------------------------------------------------------------
# MuZero Agent
# ---------------------------------------------------------------------------


class TransformerMuZeroAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder

    action_embedding: eqx.nn.Embedding
    dynamics_mlp: MLP
    reward_head: MLP

    policy_head: MLP
    value_head: MLP

    num_actions: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size,
        embd_dim,
        num_layers,
        num_heads,
        hidden_dim,
        latent_dim,
        num_actions,
        policy_dims,
        value_dims,
        seq_len,
        key,
    ):
        k1, k2, k3, k4, k5, k6, k7 = jrand.split(key, 7)
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k2)

        self.action_embedding = eqx.nn.Embedding(num_actions, latent_dim, key=k3)
        self.dynamics_mlp = MLP(latent_dim * 2, latent_dim, [hidden_dim], key=k4)
        self.reward_head = MLP(latent_dim, 1, value_dims, key=k5)

        self.policy_head = MLP(latent_dim, num_actions, policy_dims, key=k6)
        self.value_head = MLP(latent_dim, 1, value_dims, key=k7)

    def representation(self, tokens, key=None):
        x = jax.vmap(self.embedding)(tokens)
        x = self.pos_enc(x)
        x = self.encoder(x, key=key if key is not None else jrand.PRNGKey(0))
        # Project to latent_dim via simple linear or use hidden_dim == latent_dim
        return jnp.mean(x, axis=0)

    def dynamics(self, latent, action):
        a_emb = self.action_embedding(action)
        x = jnp.concatenate([latent, a_emb], axis=-1)
        next_latent = self.dynamics_mlp(x)
        reward = self.reward_head(next_latent)[0]
        return next_latent, reward

    def prediction(self, latent):
        logits = self.policy_head(latent)
        value = self.value_head(latent)[0]
        return logits, value


# ---------------------------------------------------------------------------
# MCTX Interfaces
# ---------------------------------------------------------------------------


@eqx.filter_vmap(in_axes=(None, 0))
def root_fn(agent, tokens):
    latent = agent.representation(tokens)
    logits, value = agent.prediction(latent)
    return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=latent)


@eqx.filter_vmap(in_axes=(None, None, 0, 0))
def recurrent_fn(agent, rng_key, action, embedding):
    next_latent, reward = agent.dynamics(embedding, action)
    logits, value = agent.prediction(next_latent)
    return mctx.RecurrentFnOutput(
        reward=reward, discount=jnp.ones_like(reward), prior_logits=logits, value=value
    ), next_latent


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch_sequences(trajectories, minibatches, key):
    num_envs, num_windows, unroll_steps, features = trajectories.shape

    size = num_envs * num_windows // minibatches
    valid_samples = size * minibatches

    trajectories = trajectories.reshape(-1, unroll_steps, features)

    trajectories = jrand.permutation(key, trajectories, axis=0)

    trajectories = trajectories[:valid_samples]

    return trajectories.reshape(minibatches, size, unroll_steps, features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="MuZero_Vertex")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=int, default=250197)
    parser.add_argument("--wandb", type=str, default="disabled")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--example", type=str, default="Helmholtz")
    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key)
    gen = data_gen(args.example)

    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr, args=xs, num_envs=0, data_gen=gen, target_fun=target_fn
    )

    total_v = len(closed_jaxpr.jaxpr.eqns)
    valid_vertices = jnp.array(env.valid_vertices, dtype=jnp.int32)
    num_valid = len(env.valid_vertices)

    sp_valid_mask = build_legacy_sp_valid_mask(
        closed_jaxpr.jaxpr,
        total_v,
        num_sp_types=3,
        use_min_in_ndim=False,
    )

    EPISODES = args.episodes
    NUM_ENVS = os.cpu_count() or 64
    LR = 1e-4
    MINIBATCHES = 32
    NUM_SIMULATIONS = 25
    UNROLL_STEPS = 2

    OBS_SHAPE = 1024
    NUM_ACTIONS = 3 * total_v
    ROLLOUT_LENGTH = num_valid

    agent_key, key = jrand.split(key)
    agent = TransformerMuZeroAgent(
        vocab_size=256,
        embd_dim=64,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        latent_dim=64,
        num_actions=NUM_ACTIONS,
        policy_dims=[64, 32],
        value_dims=[64, 32],
        seq_len=OBS_SHAPE,
        key=agent_key,
    )

    def reset_envs():
        def _single_reset(_):
            return env.reset()

        return jax.vmap(_single_reset)(jnp.arange(NUM_ENVS))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def rollout_fn(agent, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, key):
            search_key, act_key = jrand.split(key)

            vertex_valid = (
                jnp.zeros(total_v, dtype=jnp.float32).at[valid_vertices - 1].set(1.0)
            )
            arange_v = jnp.arange(num_valid)
            active_mask = (arange_v < jnp.expand_dims(state.step_count, -1)).astype(
                jnp.float32
            )
            already_chosen = (
                jnp.zeros(total_v, dtype=jnp.float32)
                .at[state.order - 1]
                .add(active_mask)
            )

            vertex_available = vertex_valid * (1.0 - jnp.clip(already_chosen, 0.0, 1.0))
            available_matrix = jnp.expand_dims(vertex_available, 0) * sp_valid_mask
            available_flat = available_matrix.reshape(-1)

            invalid_actions_mask = 1.0 - available_flat

            # Expand tokens for batch compatibility with root_fn vmap
            batched_tokens = jnp.expand_dims(state.tokens, 0)
            roots = root_fn(agent, batched_tokens)

            policy_output = mctx.muzero_policy(
                params=agent,
                rng_key=search_key,
                root=roots,
                recurrent_fn=recurrent_fn,
                num_simulations=NUM_SIMULATIONS,
                invalid_actions=jnp.expand_dims(invalid_actions_mask, 0),
                dirichlet_fraction=0.25,
                dirichlet_alpha=0.3,
                temperature=1.0,
            )

            mcts_policy = policy_output.action_weights[0]
            mcts_value = policy_output.search_tree.summary().value[0]

            distribution = distrax.Categorical(probs=mcts_policy)
            action_idx = distribution.sample(seed=act_key)

            sp_type = action_idx // total_v
            target_vertex = (action_idx % total_v) + 1
            env_action = sp_type * MAX_TOKENS + target_vertex

            env_out = env.step(state, env_action)

            # Pack current step data for unrolling
            new_sample = jnp.concatenate(
                (
                    state.tokens.astype(jnp.float32),
                    jnp.array([action_idx], dtype=jnp.float32),
                    jnp.atleast_1d(env_out.rewards),
                    jnp.array([mcts_value], dtype=jnp.float32),
                    mcts_policy,
                ),
                axis=-1,
            )

            return env_out.state, new_sample

        return lax.scan(step_fn, env_state, keys)

    schedule = optax.cosine_decay_schedule(LR, EPISODES, 0.0)
    optimizer = optax.chain(
        optax.adamw(schedule, eps=1e-7), optax.clip_by_global_norm(1.0)
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    BASE_LEN = OBS_SHAPE + 3 + NUM_ACTIONS

    def loss(agent, batch):
        def unroll_loss(sequence):
            # sequence shape: (UNROLL_STEPS + 1, BASE_LEN)
            tokens = sequence[0, :OBS_SHAPE].astype(jnp.int32)
            actions = sequence[:, OBS_SHAPE].astype(jnp.int32)
            rewards = sequence[:, OBS_SHAPE + 1]
            target_values = sequence[:, OBS_SHAPE + 2]
            target_policies = sequence[:, OBS_SHAPE + 3 : OBS_SHAPE + 3 + NUM_ACTIONS]

            latent = agent.representation(tokens)
            l_pi, l_v, l_r = 0.0, 0.0, 0.0

            for k in range(UNROLL_STEPS + 1):
                logits, value = agent.prediction(latent)

                l_pi += -jnp.sum(target_policies[k] * jnn.log_softmax(logits + 1e-7))
                l_v += 0.5 * jnp.square(value - target_values[k])

                if k < UNROLL_STEPS:
                    latent, pred_reward = agent.dynamics(latent, actions[k])
                    l_r += 0.5 * jnp.square(pred_reward - rewards[k])
                    # Half gradient trick
                    latent = eqx.tree_at(
                        lambda x: x,
                        latent,
                        replace_fn=lambda x: x * 0.5 + lax.stop_gradient(x) * 0.5,
                    )

            return l_pi + l_v + l_r, (l_pi, l_v, l_r)

        batch_loss, (p_loss, v_loss, r_loss) = jax.vmap(unroll_loss)(batch)
        return jnp.mean(batch_loss), (
            jnp.mean(p_loss),
            jnp.mean(v_loss),
            jnp.mean(r_loss),
        )

    @eqx.filter_jit
    def train_agent(agent, opt_state, batch):
        grads, metrics = eqx.filter_grad(loss, has_aux=True)(agent, batch)
        updates, opt_state = optimizer.update(grads, opt_state, agent)
        new_agent = eqx.apply_updates(agent, updates)
        return new_agent, opt_state, metrics

    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else "offline",
    )
    elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

    pbar = tqdm(range(EPISODES))
    best_global_return = -float("inf")
    best_global_act_seq = None

    for episode in pbar:
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, NUM_ENVS)

        env_states = reset_envs()
        env_states, trajectories = rollout_fn(
            agent, ROLLOUT_LENGTH, env_states, rollout_keys
        )

        # Replace MCTS root value with actual discounted MC return for stronger value targets
        rewards = trajectories[:, :, OBS_SHAPE + 1]
        returns = jnp.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
        trajectories = trajectories.at[:, :, OBS_SHAPE + 2].set(returns)

        # Create unrolled sequence windows of size (UNROLL_STEPS + 1)
        windows = []
        for i in range(ROLLOUT_LENGTH - UNROLL_STEPS):
            windows.append(trajectories[:, i : i + UNROLL_STEPS + 1, :])

        sequence_trajectories = jnp.stack(
            windows, axis=1
        )  # (NUM_ENVS, VALID_STEPS, UNROLL_STEPS+1, FEATURES)

        batches = shuffle_and_batch_sequences(
            sequence_trajectories, MINIBATCHES, subkey
        )

        for i in range(MINIBATCHES):
            agent, opt_state, metrics = train_agent(agent, opt_state, batches[i])

        p_loss, v_loss, r_loss = metrics

        total_rewards = jnp.sum(rewards, axis=1)
        max_idx = jnp.argmax(total_rewards)
        best_reward = total_rewards[max_idx]
        best_act_seq = trajectories[max_idx, :, OBS_SHAPE]

        if best_reward > best_global_return:
            best_global_return = best_reward
            best_global_act_seq = best_act_seq
            elim_order_table.add_data(
                episode, float(best_reward), np.array(best_act_seq)
            )

        wandb.log(
            {
                "best_return": float(best_reward),
                "mean_return": float(jnp.mean(total_rewards)),
                "policy loss": float(p_loss),
                "value loss": float(v_loss),
                "reward loss": float(r_loss),
            }
        )

        pbar.set_description(
            f"best: {best_reward:.1f}, mean: {jnp.mean(total_rewards):.1f}"
        )

    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
