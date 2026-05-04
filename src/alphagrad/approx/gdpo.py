import argparse
import os
from functools import partial
from typing import NamedTuple

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
from tqdm import tqdm

from alphagrad.approx.common import (
    build_legacy_sp_valid_mask,
    data_gen,
    get_advantages,
    get_args,
    get_fn,
    get_num_clipping_triggers,
    init_linear_weights,
    reward_normalization_fn,
    shuffle_and_batch,
)
from alphagrad.approx.env import MAX_TOKENS, VertexEliminationEnv
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.utils import entropy, explained_variance


class Trajectory(NamedTuple):
    tokens: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    next_value: jax.Array
    prob_dist: jax.Array
    discount: jax.Array


class TrainBatch(NamedTuple):
    tokens: jax.Array
    action: jax.Array
    old_prob_dist: jax.Array
    estim_returns: jax.Array
    norm_adv: jax.Array


class TransformerPPOAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
    value_head: MLP
    num_actions: int = eqx.field(static=True)
    num_rewards: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size,
        embd_dim,
        num_layers,
        num_heads,
        hidden_dim,
        num_actions,
        num_rewards,
        policy_dims,
        value_dims,
        seq_len,
        key,
    ):
        k1, k2, k3, k4 = jrand.split(key, 4)
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k2)
        self.policy_head = MLP(embd_dim, num_actions, policy_dims, key=k3)
        self.value_head = MLP(embd_dim, num_rewards, value_dims, key=k4)

    def __call__(self, tokens, key=None, inference=False):
        if tokens.ndim == 1:
            mask = (tokens != 0)[..., None]
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            enc_key = key if key is not None else jrand.PRNGKey(0)
            x = self.encoder(x, key=enc_key)
            summary = jnp.sum(x * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1e-9)
            logits = self.policy_head(summary)
            value = self.value_head(summary) 
            return logits, value
        else:
            batched_call = jax.vmap(self, in_axes=(0, None, None))
            return batched_call(tokens, key, inference)


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def get_log_probs_and_value(agent, tokens, action, key):
    logits, value = agent(tokens, key=key)
    action_idx = action - 1
    prob_dist = jnn.softmax(logits, axis=-1)
    log_prob = jnp.log(prob_dist[action_idx] + 1e-8)
    return log_prob, prob_dist, value, entropy(prob_dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="Decoupled_PPO_MultiReward")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=int, default=250197)
    parser.add_argument("--wandb", type=str, default="disabled")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--example", type=str, default="Helmholtz")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--disable-sparsification", action="store_true")
    parser.add_argument("--disable-eval", action="store_true")
    args = parser.parse_args()

    if args.no_jit:
        jax.config.update('jax_disable_jit', True)

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

    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"Valid set: {env.valid_vertices}"
    )

    sp_valid_mask = build_legacy_sp_valid_mask(
        closed_jaxpr.jaxpr,
        total_v,
        num_sp_types=3,
        use_min_in_ndim=False,
        disable_sparsification=args.disable_sparsification,
    )

    ENTROPY_WEIGHT = 0.01
    VALUE_WEIGHT = 0.5
    EPISODES = args.episodes
    NUM_ENVS = os.cpu_count() or 64
    LR = 3e-4
    GAE_LAMBDA = 0.95
    EPS = 0.2
    MINIBATCHES = 32
    PPO_EPOCHS = 4

    NUM_REWARDS = 1 if args.disable_eval else 2
    OBS_SHAPE = MAX_TOKENS
    NUM_ACTIONS = 3 * total_v
    ROLLOUT_LENGTH = num_valid

    print(
        f"NUM_ACTIONS={NUM_ACTIONS}, ROLLOUT_LENGTH={ROLLOUT_LENGTH}, "
        f"MINIBATCHES={MINIBATCHES}"
    )

    agent_key, init_key, key = jrand.split(key, 3)
    agent = TransformerPPOAgent(
        vocab_size=256,
        embd_dim=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        num_actions=NUM_ACTIONS,
        num_rewards=NUM_REWARDS,
        policy_dims=[64, 32],
        value_dims=[64, 32],
        seq_len=OBS_SHAPE,
        key=agent_key,
    )
    agent = init_linear_weights(agent, init_key)

    def reset_envs():
        def _single_reset(_):
            return env.reset()

        return jax.vmap(_single_reset)(jnp.arange(NUM_ENVS))

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def rollout_fn(agent, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, key):
            net_key, next_net_key, act_key = jrand.split(key, 3)

            logits, value = agent(state.tokens, key=net_key, inference=True)

            chosen = state.order
            step_idx = state.step_count

            vertex_valid = jnp.zeros(total_v, dtype=jnp.float32)
            vertex_valid = vertex_valid.at[valid_vertices - 1].set(1.0)

            arange_v = jnp.arange(num_valid)
            active_mask = (arange_v < jnp.expand_dims(step_idx, -1)).astype(jnp.float32)

            already_chosen = jnp.zeros(total_v, dtype=jnp.float32)
            already_chosen = already_chosen.at[chosen - 1].add(active_mask)

            vertex_available = vertex_valid * (1.0 - jnp.clip(already_chosen, 0.0, 1.0))
            available_matrix = jnp.expand_dims(vertex_available, 0) * sp_valid_mask
            available_flat = available_matrix.reshape(-1)

            masked_logits = jnp.where(available_flat > 0.5, logits, -1e9)
            prob_dist = jnn.softmax(masked_logits, axis=-1)

            distribution = distrax.Categorical(probs=prob_dist)
            action_idx = distribution.sample(seed=act_key)

            sp_type = action_idx // total_v
            target_vertex = (action_idx % total_v) + 1
            env_action = sp_type * MAX_TOKENS + target_vertex

            env_out = env.step(state, env_action)
            next_state = env_out.state
            rewards = env_out.reward[:NUM_REWARDS]
            done = env_out.terminated.astype(jnp.float32)

            _, next_value = agent(next_state.tokens, key=next_net_key, inference=True)

            transition = Trajectory(
                tokens=state.tokens.astype(jnp.int32),
                action=jnp.array(action_idx + 1, dtype=jnp.int32),
                reward=jnp.atleast_1d(rewards),
                done=jnp.array(done, dtype=jnp.float32),
                value=jnp.atleast_1d(value),
                next_value=jnp.atleast_1d(next_value),
                prob_dist=prob_dist,
                discount=jnp.array(0.99),
            )

            return next_state, transition

        return lax.scan(step_fn, env_state, keys)

    schedule = optax.cosine_decay_schedule(
        LR, EPISODES * PPO_EPOCHS * MINIBATCHES, 0.0
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(schedule, b1=0.9, eps=1e-7),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    def loss(agent, batch: TrainBatch, keys):
        log_probs, prob_dist, values, entropies = get_log_probs_and_value(
            agent, batch.tokens, batch.action, keys
        )

        old_log_probs = jnp.log(
            jax.vmap(lambda p, a: p[a - 1])(batch.old_prob_dist, batch.action) + 1e-8
        )
        ratio = jnp.exp(log_probs - old_log_probs)

        num_triggers = get_num_clipping_triggers(ratio, EPS)
        trigger_ratio = num_triggers / len(ratio)

        clipping_objective = jnp.minimum(
            ratio * batch.norm_adv,
            jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * batch.norm_adv,
        )
        ppo_loss = jnp.mean(-clipping_objective)
        entropy_loss = jnp.mean(entropies)

        value_loss = jnp.mean(
            jnp.sum((values - reward_normalization_fn(batch.estim_returns)) ** 2, axis=-1)
        )

        explained_var = explained_variance(
            batch.norm_adv, jnp.sum(batch.estim_returns, axis=-1)
        )
        kl_div = jnp.mean(optax.kl_divergence(jnp.log(prob_dist + 1e-7), batch.old_prob_dist))

        total_loss = (
            ppo_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy_loss
        )

        return total_loss, (
            kl_div,
            entropy_loss,
            0.0,
            explained_var,
            ppo_loss,
            VALUE_WEIGHT * value_loss,
            ENTROPY_WEIGHT * entropy_loss,
            total_loss,
            trigger_ratio,
        )

    def train_episode(agent, opt_state, env_states, key):
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, NUM_ENVS)

        env_states, traj = rollout_fn(agent, ROLLOUT_LENGTH, env_states, rollout_keys)

        _, estim_returns, advantages = get_advantages(
            traj.reward,
            traj.done,
            traj.value,
            traj.next_value,
            traj.discount,
            GAE_LAMBDA,
        )

        # Normalize each reward-head advantage globally, then sum for a single policy signal
        def normalize(x):
            return (x - jnp.mean(x)) / (jnp.std(x) + 1e-7)

        norm_adv = jnp.sum(
            jax.vmap(normalize, in_axes=-1, out_axes=-1)(advantages.reshape(-1, advantages.shape[-1])).reshape(advantages.shape),
            axis=-1,
        )

        full_batch = TrainBatch(
            tokens=traj.tokens,
            action=traj.action,
            old_prob_dist=traj.prob_dist,
            estim_returns=estim_returns,
            norm_adv=norm_adv,
        )

        dynamic_carry, static_carry = eqx.partition((agent, opt_state), eqx.is_array)

        def train_epoch(carry, epoch_key):
            batches = shuffle_and_batch(full_batch, MINIBATCHES, epoch_key)
            mb_keys = jrand.split(epoch_key, MINIBATCHES)

            def train_minibatch(c, batch_and_key):
                comb_agent, comb_opt_state = eqx.combine(c, static_carry)
                batch, t_key = batch_and_key
                keys = jrand.split(t_key, batch.tokens.shape[0])
                grads, metrics = eqx.filter_grad(loss, has_aux=True)(comb_agent, batch, keys)
                updates, new_opt_state = optimizer.update(grads, comb_opt_state, comb_agent)
                new_agent = eqx.apply_updates(comb_agent, updates)
                next_carry, _ = eqx.partition((new_agent, new_opt_state), eqx.is_array)
                return next_carry, metrics

            return lax.scan(train_minibatch, carry, (batches, mb_keys))

        epoch_keys = jrand.split(subkey, PPO_EPOCHS)
        dynamic_carry, metrics_seq = lax.scan(train_epoch, dynamic_carry, epoch_keys)

        agent, opt_state = eqx.combine(dynamic_carry, static_carry)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_seq)

        total_rewards = jnp.sum(traj.reward[..., 0], axis=-1)

        return agent, opt_state, env_states, metrics, total_rewards, traj.action

    if not args.no_jit:
        train_episode = eqx.filter_jit(train_episode)

    wandb.init(
        project="dsnn-vertex",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else "offline",
    )
    elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

    pbar = tqdm(total=EPISODES)
    
    host_state = {
        "samplecounts": 0,
        "best_global_return": -float("inf"),
        "best_global_act_seq": None,
    }

    def host_log(ep, b_ret, b_seq, mean_r, mets):
        ep = int(ep)
        b_ret = float(b_ret)
        mean_r = float(mean_r)
        
        host_state["samplecounts"] += NUM_ENVS * ROLLOUT_LENGTH
        (
            kl_div, policy_entropy, fit_quality, explained_var,
            ppo_loss, value_loss, entropy_loss, total_loss, clipping_trigger_ratio
        ) = [float(m) for m in mets]

        if b_ret > host_state["best_global_return"]:
            host_state["best_global_return"] = b_ret
            host_state["best_global_act_seq"] = b_seq

            action_pairs = [
                (int((i - 1) % total_v) + 1, int((i - 1) // total_v))
                for i in b_seq
            ]
            print(f"\nNew best return: {b_ret}")
            print(f"New best action sequence (vertex, sp_type): {action_pairs}")
            elim_order_table.add_data(ep, b_ret, np.array(b_seq))

        wandb.log({
            "best_return": host_state["best_global_return"],
            "mean_return": mean_r,
            "KL divergence": kl_div,
            "entropy evolution": policy_entropy,
            "explained variance": explained_var,
            "sample count": host_state["samplecounts"],
            "ppo loss": ppo_loss,
            "value loss": value_loss,
            "total loss": total_loss,
        })
        pbar.update(1)
        pbar.set_description(f"ent: {policy_entropy:.4f}, best: {b_ret:.1f}, mean: {mean_r:.1f}")

    for ep in range(EPISODES):
        ep_key, key = jrand.split(key)

        env_states = reset_envs()
        agent, opt_state, _, metrics, total_rewards, actions = train_episode(
            agent, opt_state, env_states, ep_key
        )
        
        max_idx = jnp.argmax(total_rewards)
        best_reward = total_rewards[max_idx]
        best_act_seq = actions[max_idx]
        
        host_log(ep, best_reward, best_act_seq, jnp.mean(total_rewards), metrics)

    pbar.close()
    wandb.log({"Elimination order": elim_order_table})
    if host_state["best_global_act_seq"] is not None:
        best_pairs = [
            (int((i - 1) % total_v) + 1, int((i - 1) // total_v))
            for i in host_state["best_global_act_seq"]
        ]
        print(f"\nBest vertex elimination sequence after {EPISODES} episodes: {best_pairs} with {host_state['best_global_return']} score.")


if __name__ == "__main__":
    main()