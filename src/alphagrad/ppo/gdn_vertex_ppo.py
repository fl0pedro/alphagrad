"""
PPO with token-based VertexEliminationEnv.
Migrated from vertex_ppo.py to use vertex_game_w_tokens.py environment.
Uses a small transformer agent model on a small example function.
"""

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
import numpy as np
import optax
import wandb
from graphax import examples
from tqdm import tqdm

from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.utils import entropy, explained_variance, symexp, symlog
from alphagrad.vertexgame.vertex_game_w_tokens_and_just_better import (
    VertexEliminationEnv,
)


def get_args(fn_str, key):
    basic_args = {
        "Simple": (5.0, 7.0),
        "Lighthouse": (0.02,) * 4,
        "Helmholtz": (jnp.array([0.05, 0.15, 0.25, 0.35]),),
        "RobotArm_6DOF": [0.02] * 6,
        "RoFlux_1d": (0.01, 0.02, 0.02, 0.01, 0.03, 0.03),
        "RoeFlux_3d": (
            jnp.array([0.1]),
            jnp.array([0.1, 0.2, 0.3]),
            jnp.array([0.5]),
            jnp.array([0.2]),
            jnp.array([0.2, 0.2, 0.4]),
            jnp.array([0.6]),
        ),
        "BlackScholes_Jacobian": (1.0,) * 5,
    }
    shapes = []
    if fn_str.endswith("NeuralNetwork") or fn_str.endswith("Perceptron"):
        shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,)]
    elif fn_str.startswith("Encoder"):
        shapes = [(4, 4), (2, 4), (4, 4) * 6, (4, 4), (4,), (2, 4), (2, 1)]

    if fn_str.startswith("Vmapped"):
        shapes[0] = (16,) + shapes[0]
        shapes[1] = (16,) + shapes[1]
    elif fn_str.endswith("Decoder"):
        shapes = shapes[:8] + [(4, 4) * 3] + shapes[8:]

    args = []
    for shape in shapes:
        key, k = jrand.split(key)
        args.append(jrand.normal(k, shape))

    if args == []:
        args = basic_args[fn_str]

    return args


def get_fn(fn_str):
    if fn_str.endswith("NeuralNetwork"):

        def NeuralNetwork(x, y, W1, b1, W2, b2):
            y1 = W1 @ x
            z1 = y1 + b1
            a1 = jnp.tanh(z1)

            y2 = W2 @ a1
            z2 = y2 + b2
            return 0.5 * (jnp.tanh(z2) - y) ** 2

        fn = NeuralNetwork
    elif fn_str.endswith("Perceptron"):
        fn = examples.Perceptron
    else:
        fn = getattr(examples, fn_str)
        if fn is None:
            raise ValueError

    if fn_str.startswith("Vmapped"):
        fn = jax.vmap(fn, in_axes=(0, 0) + (None,) * 4)

    return fn


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TransformerPPOAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
    value_head: MLP
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size,
        embd_dim,
        num_layers,
        num_heads,
        hidden_dim,
        num_actions,
        policy_dims,
        value_dims,
        seq_len,
        key,
    ):
        print("Agent init started...")
        k1, k2, k3, k4 = jrand.split(key, 4)
        print("Key split done.")
        self.num_actions = num_actions
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        print("Embedding init done.")
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        print("PosEnc init done.")
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k2)
        print("Encoder init done.")
        self.policy_head = MLP(embd_dim, num_actions, policy_dims, key=k3)
        print("PolicyHead init done.")
        self.value_head = MLP(embd_dim, 1, value_dims, key=k4)
        print("ValueHead init done.")

    def __call__(self, tokens, key=None, inference=False):
        if tokens.ndim == 1:
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            enc_key = key if key is not None else jrand.PRNGKey(0)
            x = self.encoder(x, key=enc_key)
            summary = jnp.mean(x, axis=0)
            logits = self.policy_head(summary)
            value = self.value_head(summary)
            return logits, value[0]
        else:
            batched_call = jax.vmap(self, in_axes=(0, None, None))
            return batched_call(tokens, key, inference)


# ---------------------------------------------------------------------------
# Reward normalisation helpers
# ---------------------------------------------------------------------------


def reward_normalization_fn(reward):
    return symlog(reward)


def inverse_reward_normalization_fn(reward):
    return symexp(reward)


# ---------------------------------------------------------------------------
# RL helpers
# ---------------------------------------------------------------------------


def get_num_clipping_triggers(ratio, eps):
    _ratio = jnp.where(ratio <= 1.0 + eps, ratio, 0.0)
    _ratio = jnp.where(ratio >= 1.0 - eps, 1.0, 0.0)
    return jnp.sum(_ratio)


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def get_log_probs_and_value(agent, tokens, action, key):
    logits, value = agent(tokens, key=key)
    # actions are 1-indexed vertex ids → convert to 0-indexed for logits
    action_idx = action - 1
    prob_dist = jnn.softmax(logits, axis=-1)
    log_prob = jnp.log(prob_dist[action_idx] + 1e-7)
    return log_prob, prob_dist, value, entropy(prob_dist)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
def get_advantages(rewards, dones, values, next_values, discounts, gae_lambda):
    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward, done, value, next_value, discount = traj

        mask = 1.0 - done
        episodic_return = reward + discount * episodic_return * mask

        value_raw = inverse_reward_normalization_fn(value)
        next_value_raw = inverse_reward_normalization_fn(next_value)

        delta = reward + next_value_raw * discount * mask - value_raw
        advantage = delta + discount * gae_lambda * lastgaelam * mask

        estim_return = advantage + value_raw
        return (episodic_return, advantage), jnp.array(
            [episodic_return, estim_return, advantage]
        )

    inputs = jnp.stack([rewards, dones, values, next_values, discounts], axis=1)
    _, output = lax.scan(loop_fn, (0.0, 0.0), inputs[::-1])
    return output[::-1]


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch(trajectories, minibatches, key):
    num_envs, rollout_length, features = trajectories.shape
    size = num_envs * rollout_length // minibatches
    trajectories = trajectories.reshape(-1, features)
    trajectories = jrand.permutation(key, trajectories, axis=0)
    return trajectories.reshape(minibatches, size, features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="GDN_Simple_Test")
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

    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)
    env = VertexEliminationEnv.from_jaxpr(closed_jaxpr, args=xs, num_envs=0)

    total_v = len(closed_jaxpr.jaxpr.eqns)
    valid_vertices = jnp.array(env.valid_vertices, dtype=jnp.int32)
    num_valid = len(env.valid_vertices)

    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"Valid set: {env.valid_vertices}"
    )

    # ---- Hyperparameters (small) ----
    ENTROPY_WEIGHT = 0.01
    VALUE_WEIGHT = 0.5
    EPISODES = args.episodes
    NUM_ENVS = os.cpu_count() or 64
    LR = 3e-4

    GAE_LAMBDA = 0.95
    EPS = 0.2
    MINIBATCHES = 40

    OBS_SHAPE = 1024
    NUM_ACTIONS = total_v
    ROLLOUT_LENGTH = num_valid
    MINIBATCHSIZE = NUM_ENVS * ROLLOUT_LENGTH // MINIBATCHES

    print(
        f"NUM_ACTIONS={NUM_ACTIONS}, ROLLOUT_LENGTH={ROLLOUT_LENGTH}, "
        f"MINIBATCHSIZE={MINIBATCHSIZE}"
    )

    # ---- Small agent model ----
    agent_key, key = jrand.split(key)
    agent = TransformerPPOAgent(
        vocab_size=256,
        embd_dim=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        num_actions=NUM_ACTIONS,
        policy_dims=[64, 32],
        value_dims=[64, 32],
        seq_len=OBS_SHAPE,
        key=agent_key,
    )

    # ---- wandb ----
    run_config = {
        "seed": args.seed,
        "entropy_weight": ENTROPY_WEIGHT,
        "value_weight": VALUE_WEIGHT,
        "lr": LR,
        "episodes": EPISODES,
        "num_envs": NUM_ENVS,
        "gae_lambda": GAE_LAMBDA,
        "eps": EPS,
        "minibatches": MINIBATCHES,
        "minibatchsize": MINIBATCHSIZE,
        "num_actions": NUM_ACTIONS,
        "rollout_length": ROLLOUT_LENGTH,
    }
    print("Initializing wandb...")
    # wandb.init(
    #     project="AlphaGrad",
    #     group="Simple",
    #     mode=args.wandb,
    #     config=run_config,
    # )
    # wandb.run.name = "GDN_PPO_Simple_" + args.name
    print("wandb skipped.")

    # ------------------------------------------------------------------
    # Reset helper  –  env.reset() returns a single EnvState; to get a
    # batch we vmap over a dummy axis so that pure_callback is invoked
    # once per env (vmap_method="sequential").
    # ------------------------------------------------------------------
    def reset_envs():
        def _single_reset(_):
            return env.reset()

        return jax.vmap(_single_reset)(jnp.arange(NUM_ENVS))

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def rollout_fn(agent, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, key):
            net_key, next_net_key, act_key = jrand.split(key, 3)

            logits, value = agent(state.tokens, key=net_key, inference=True)

            # Build action mask: valid vertices that have NOT been placed
            # in the order yet (positions 0..step_count-1 are filled).
            # state.order has shape (num_valid,).  Already-chosen vertices
            # occupy indices < step_count.
            chosen = state.order  # (num_valid,)
            step_idx = state.step_count

            # Mark which of the total_v vertices are valid at all
            vertex_valid = jnp.zeros(total_v, dtype=jnp.float32)
            vertex_valid = vertex_valid.at[valid_vertices - 1].set(1.0)

            # (only indices < step_idx are meaningful)
            arange_v = jnp.arange(num_valid)
            # arange_v: (num_valid,) and step_idx: (...,)
            # Need to broadcast arange_v to (..., num_valid)
            active_mask = (arange_v < jnp.expand_dims(step_idx, -1)).astype(jnp.float32)
            # chosen vertices (1-indexed) → 0-indexed
            already_chosen = jnp.zeros(total_v, dtype=jnp.float32)
            # Scatter: for each position i, if active_mask[i]==1,
            # mark chosen[i]-1 as used.
            already_chosen = already_chosen.at[chosen - 1].add(active_mask)

            # available = valid AND not-yet-chosen
            available = vertex_valid * (1.0 - jnp.clip(already_chosen, 0.0, 1.0))

            masked_logits = jnp.where(available > 0.5, logits, -1e9)
            prob_dist = jnn.softmax(masked_logits, axis=-1)

            distribution = distrax.Categorical(probs=prob_dist)
            action_idx = distribution.sample(seed=act_key)
            action = action_idx + 1  # 1-indexed vertex id

            # Calculate prefetch_order: sort all 1-indexed vertices by logits descending
            # We filter for available vertices afterwards in the worker or just use the top N.
            # Here we just pass the sorted order of ALL vertices, or better, just the available ones.

            env_out = env.step(state, action)
            next_state = env_out.state
            reward = env_out.reward
            done = env_out.terminated.astype(jnp.float32)
            discount = 1.0

            _, next_value = agent(next_state.tokens, key=next_net_key, inference=True)

            # Trajectory sample layout:
            #   tokens          [0 : OBS_SHAPE]              1024
            #   action          [OBS_SHAPE]                  1
            #   reward          [OBS_SHAPE+1]                1
            #   done            [OBS_SHAPE+2]                1
            #   value           [OBS_SHAPE+3]                1
            #   next_value      [OBS_SHAPE+4]                1
            #   prob_dist       [OBS_SHAPE+5 : OBS_SHAPE+5+NUM_ACTIONS]
            #   discount        [OBS_SHAPE+5+NUM_ACTIONS]    1
            new_sample = jnp.concatenate(
                (
                    state.tokens.astype(jnp.float32),
                    jnp.expand_dims(action, -1).astype(jnp.float32),
                    jnp.expand_dims(reward, -1),
                    jnp.expand_dims(done, -1),
                    jnp.expand_dims(value, -1),
                    jnp.expand_dims(next_value, -1),
                    prob_dist,
                    jnp.broadcast_to(jnp.array([discount]), (state.tokens.shape[0], 1))
                    if state.tokens.ndim > 1
                    else jnp.array([discount]),
                ),
                axis=-1,
            )

            return next_state, new_sample

        return lax.scan(step_fn, env_state, keys)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    schedule = optax.cosine_decay_schedule(LR, EPISODES, 0.0)
    optimizer = optax.chain(
        optax.adam(schedule, b1=0.9, eps=1e-7),
        optax.clip_by_global_norm(0.5),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss(agent, trajectories, keys):
        tokens = trajectories[:, :OBS_SHAPE].astype(jnp.int32)
        actions = trajectories[:, OBS_SHAPE].astype(jnp.int32)
        rewards = trajectories[:, OBS_SHAPE + 1]

        old_prob_dist = trajectories[:, OBS_SHAPE + 5 : OBS_SHAPE + 5 + NUM_ACTIONS]
        discounts = trajectories[:, OBS_SHAPE + 5 + NUM_ACTIONS]

        episodic_returns = trajectories[:, -3]
        estim_returns = trajectories[:, -2]
        advantages = trajectories[:, -1]

        next_values = trajectories[:, OBS_SHAPE + 4]

        log_probs, prob_dist, values, entropies = get_log_probs_and_value(
            agent, tokens, actions, keys
        )
        norm_adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-7)

        old_log_probs = jnp.log(
            jax.vmap(lambda p, a: p[a - 1])(old_prob_dist, actions) + 1e-7
        )
        ratio = jnp.exp(log_probs - old_log_probs)

        num_triggers = get_num_clipping_triggers(ratio, EPS)
        trigger_ratio = num_triggers / len(ratio)

        clipping_objective = jnp.minimum(
            ratio * norm_adv,
            jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * norm_adv,
        )
        ppo_loss = jnp.mean(-clipping_objective)
        entropy_loss = jnp.mean(entropies)
        value_loss = jnp.mean((values - reward_normalization_fn(estim_returns)) ** 2)

        dV = (
            estim_returns
            - rewards
            - discounts * inverse_reward_normalization_fn(next_values)
        )
        fit_quality = jnp.mean(jnp.abs(dV))
        explained_var = explained_variance(advantages, estim_returns)
        kl_div = jnp.mean(optax.kl_divergence(jnp.log(prob_dist + 1e-7), old_prob_dist))

        total_loss = (
            ppo_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy_loss
        )

        return total_loss, (
            kl_div,
            entropy_loss,
            fit_quality,
            explained_var,
            ppo_loss,
            VALUE_WEIGHT * value_loss,
            ENTROPY_WEIGHT * entropy_loss,
            total_loss,
            trigger_ratio,
        )

    @eqx.filter_jit
    def train_agent(agent, opt_state, trajectories, key):
        keys = jrand.split(key, trajectories.shape[0])
        grads, metrics = eqx.filter_grad(loss, has_aux=True)(agent, trajectories, keys)
        updates, opt_state = optimizer.update(grads, opt_state, agent)
        new_agent = eqx.apply_updates(agent, updates)
        return new_agent, opt_state, metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else "offline",
    )
    elim_order_table = wandb.Table(columns=["episode", "return", "order"])

    pbar = tqdm(range(EPISODES))
    samplecounts = 0
    best_global_return = -float("inf")
    best_global_act_seq = None

    elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

    for episode in pbar:
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, NUM_ENVS)

        # Reset all envs each episode (episodic)
        print(f"Episode {episode}: Resetting envs...")
        env_states = reset_envs()
        print(f"Episode {episode}: Envs reset.")

        print(f"Episode {episode}: Starting rollout...")
        env_states, trajectories = rollout_fn(
            agent, ROLLOUT_LENGTH, env_states, rollout_keys
        )
        print(f"Episode {episode}: Rollout complete.")

        # Compute advantages
        adv_data = get_advantages(
            trajectories[:, :, OBS_SHAPE + 1],  # rewards
            trajectories[:, :, OBS_SHAPE + 2],  # dones
            trajectories[:, :, OBS_SHAPE + 3],  # values
            trajectories[:, :, OBS_SHAPE + 4],  # next_values
            trajectories[:, :, OBS_SHAPE + 5 + NUM_ACTIONS],  # discounts
            GAE_LAMBDA,
        )
        # adv_data: (num_envs, rollout_length, 3) → [episodic_return, estim_return, advantage]
        trajectories = jnp.concatenate([trajectories, adv_data], axis=-1)

        batches = shuffle_and_batch(trajectories, MINIBATCHES, subkey)

        print(f"Episode {episode}: Starting training...")
        for i in range(MINIBATCHES):
            train_key, key = jrand.split(key)
            agent, opt_state, metrics = train_agent(
                agent, opt_state, batches[i], train_key
            )
        print(f"Episode {episode}: Training complete.")
        samplecounts += NUM_ENVS * ROLLOUT_LENGTH

        (
            kl_div,
            policy_entropy,
            fit_quality,
            explained_var,
            ppo_loss,
            value_loss,
            entropy_loss,
            total_loss,
            clipping_trigger_ratio,
        ) = metrics

        actions = trajectories[:, :, OBS_SHAPE]
        rewards = trajectories[:, :, OBS_SHAPE + 1]

        total_rewards = jnp.sum(rewards, axis=1)
        max_idx = jnp.argmax(total_rewards)
        best_reward = total_rewards[max_idx]
        best_act_seq = actions[max_idx]

        if best_reward > best_global_return:
            best_global_return = best_reward
            best_global_act_seq = best_act_seq
            print(f"\nNew best return: {best_reward}")
            vertex_elimination_order = [int(i) for i in best_act_seq]
            print(f"New best action sequence: {vertex_elimination_order}")
            elim_order_table.add_data(
                episode, float(best_reward), np.array(best_act_seq)
            )

        wandb.log(
            {
                "best_return": float(best_reward),
                "mean_return": float(jnp.mean(total_rewards)),
                "KL divergence": float(kl_div),
                "entropy evolution": float(policy_entropy),
                "value function fit quality": float(fit_quality),
                "explained variance": float(explained_var),
                "sample count": samplecounts,
                "ppo loss": float(ppo_loss),
                "value loss": float(value_loss),
                "entropy loss": float(entropy_loss),
                "total loss": float(total_loss),
                "clipping trigger ratio": float(clipping_trigger_ratio),
            }
        )

        pbar.set_description(
            f"ent: {policy_entropy:.4f}, best: {best_reward:.1f}, "
            f"mean: {jnp.mean(total_rewards):.1f}, "
            f"fit: {fit_quality:.2f}, ev: {explained_var:.4f}, "
            f"kl: {kl_div:.4f}"
        )

    wandb.log({"Elimination order": elim_order_table})
    vertex_elimination_order = [int(i) for i in best_global_act_seq]
    print(
        f"\nBest vertex elimination sequence after {EPISODES} episodes: "
        f"{vertex_elimination_order} with {best_global_return} score."
    )


if __name__ == "__main__":
    main()
