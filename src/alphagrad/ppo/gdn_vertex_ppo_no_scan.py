import argparse
from functools import partial

import distrax
import equinox as eqx
import graphax.examples as examples
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import optax
import wandb
from tqdm import tqdm

from alphagrad.transformer.fla_wrappers import GatedDeltaNet
from alphagrad.utils import entropy
from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph
from alphagrad.vertexgame.vertex_game_w_tokens import VertexEliminationEnv


class PPOAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    gdn: GatedDeltaNet
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    num_actions: int = eqx.field(static=True)

    def __init__(self, vocab_size, embd_dim, h, k, hidden_size, num_actions, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.num_actions = num_actions
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        self.gdn = GatedDeltaNet(h, k, hidden_size, n_layers=2)
        self.policy_head = eqx.nn.Linear(hidden_size, num_actions, key=k2)
        self.value_head = eqx.nn.Linear(hidden_size, 1, key=k3)

    def __call__(self, tokens, key=None, inference=False):
        x = jax.vmap(self.embedding)(tokens)
        x = x[None, ...]
        x = self.gdn(x, inference=inference)
        summary = jnp.mean(x[0], axis=0)

        logits = self.policy_head(summary)
        value = self.value_head(summary)
        return logits, value[0]


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def get_log_probs_and_value(agent, tokens, action, key):
    logits, value = agent(tokens, key=key)
    # action is 1-indexed vertex, so we convert to 0-indexed for logits
    action_idx = action - 1

    prob_dist = jnn.softmax(logits, axis=-1)
    log_prob = jnp.log(prob_dist[action_idx] + 1e-7)
    return log_prob, prob_dist, value, entropy(prob_dist)


# @jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None))
def get_advantages(
    rewards, dones, values, next_values, discounts, gae_lambda, discount_factor
):
    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward, done, value, next_value, discount = traj

        mask = 1.0 - done
        episodic_return = reward + discount_factor * episodic_return * mask
        delta = reward + next_value * discount_factor * mask - value
        advantage = delta + discount_factor * gae_lambda * lastgaelam * mask

        return (episodic_return, advantage), jnp.array(
            [episodic_return, advantage + value, advantage]
        )

    inputs = jnp.stack([rewards, dones, values, next_values, discounts], axis=1)
    _, output = lax.scan(loop_fn, (0.0, 0.0), inputs[::-1])
    return output[::-1]


def scan(f, init, xs, length: int | None = None):
    if xs is None and length is not None:
        xs = [None] * length
    else:
        assert xs is not None
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jax.tree_util.tree_map(lambda *x: jnp.stack(x), *ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Encoder")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--rollout_length", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--value_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", type=str, default="online")
    args = parser.parse_args()

    # Setup Encoder task
    key = jrand.PRNGKey(args.seed)

    def simple_fn(x, y):
        return jnp.sin(x * y) + jnp.cos(x)

    x_init = jnp.ones((4, 4))
    y_init = jnp.ones((4, 4))
    closed_jaxpr = jax.make_jaxpr(simple_fn)(x_init, y_init)

    env = VertexEliminationEnv.from_jaxpr(closed_jaxpr, args=(x_init, y_init))
    num_v = len(closed_jaxpr.jaxpr.eqns)
    initial_order = jnp.arange(1, num_v + 1)
    from alphagrad.vertexgame.interpreter.from_jaxpr import make_graph

    initial_edges = make_graph(simple_fn, x_init, y_init)

    key, agent_key = jrand.split(key)
    agent = PPOAgent(256, 128, 4, 32, 128, num_v, agent_key)

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_array))

    wandb.init(project="AlphaGrad-PPO", config=vars(args), mode=args.wandb)

    # @eqx.filter_jit
    def rollout(agent, env, key):
        def step_fn(carry, k):
            state, key = carry
            k1, k2 = jrand.split(k)
            logits, value = agent(state.tokens, key=k1, inference=True)

            available_indices = jnp.arange(state.step_count, num_v)
            available_vertices = state.order[available_indices]

            mask = jnp.zeros(num_v)
            mask = mask.at[available_vertices - 1].set(1.0)

            mask = mask.astype(bool)
            masked_logits = jnp.where(mask, logits, -jnp.inf)
            prob_dist = jnn.softmax(masked_logits)

            dist = distrax.Categorical(probs=prob_dist)
            action_idx = dist.sample(seed=k2)
            action = action_idx + 1

            env_out = env.step(state, action)
            new_state = env_out.state

            return (new_state, key), (
                state.tokens,
                action,
                env_out.reward,
                env_out.terminated,
                value,
                prob_dist,
            )

        keys = jrand.split(key, args.rollout_length)
        state = env.reset()
        _, rollout_data = scan(step_fn, (state, key), keys)  # lax.scan
        return rollout_data

    # @eqx.filter_jit
    def train_step(agent, opt_state, rollout_data, key):
        tokens, actions, rewards, dones, values, old_probs = rollout_data

        next_values = jnp.roll(values, -1).at[-1].set(0.0)
        discounts = jnp.ones_like(rewards)

        adv_data = get_advantages(
            rewards[None, :],
            dones[None, :],
            values[None, :],
            next_values[None, :],
            discounts[None, :],
            args.gae_lambda,
            1.0,
        )
        returns, advantages = adv_data[0, :, 1], adv_data[0, :, 2]

        def loss_fn(agent):
            log_probs, probs, v_preds, entropies = get_log_probs_and_value(
                agent, tokens, actions, None
            )

            old_log_probs = jnp.log(
                jax.vmap(lambda p, a: p[a - 1])(old_probs, actions) + 1e-7
            )
            ratio = jnp.exp(log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = (
                jnp.clip(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param)
                * advantages
            )
            ppo_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            value_loss = jnp.mean(jnp.square(v_preds - returns))
            entropy_loss = -jnp.mean(entropies)

            total_loss = (
                ppo_loss
                + args.value_weight * value_loss
                + args.entropy_weight * entropy_loss
            )
            return total_loss, (ppo_loss, value_loss, entropy_loss)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(agent)
        updates, opt_state = optimizer.update(grads, opt_state)
        agent = eqx.apply_updates(agent, updates)
        return agent, opt_state, loss, aux

    best_reward = -float("inf")
    best_actions = None

    pbar = tqdm(range(args.episodes))
    for i in pbar:
        key, rollout_key, train_key = jrand.split(key, 3)
        rollout_data = rollout(agent, env, rollout_key)

        # Track best rollout seen during training
        tokens, actions, rewards, dones, values, old_probs = rollout_data
        total_reward = float(jnp.sum(rewards))

        # In this env, reward is negative ops. Higher reward = Fewer ops.
        # "Best" means the sequence that resulted in the least total math operations.
        if total_reward > best_reward:
            best_reward = total_reward
            best_actions = actions

        agent, opt_state, loss, aux = train_step(
            agent, opt_state, rollout_data, train_key
        )

        ppo_l, val_l, ent_l = aux
        wandb.log(
            {
                "loss": loss,
                "ppo_loss": ppo_l,
                "value_loss": val_l,
                "entropy_loss": ent_l,
                "mean_reward": jnp.mean(rewards),
                "best_reward": best_reward,
            }
        )
        pbar.set_description(
            f"Loss: {loss:.4f} Rew: {jnp.mean(rewards):.2f} Best: {best_reward:.2f}"
        )

    # --- Final Solution Extraction ---
    # We run one last rollout using the greedy policy (no sampling)
    # to see what the agent actually "decided" on as the optimal path.

    def greedy_eval(agent, env, key):
        def step_fn(carry, _):
            state, k = carry
            logits, _ = agent(state.tokens, key=k, inference=True)

            available_indices = jnp.arange(state.step_count, num_v)
            available_vertices = state.order[available_indices]
            mask = jnp.zeros(num_v)
            mask = mask.at[available_vertices - 1].set(1.0)
            mask = mask.astype(bool)

            # Pick the single best action according to the policy
            masked_logits = jnp.where(mask, logits, -jnp.inf)
            action = jnp.argmax(masked_logits) + 1

            env_out = env.step(state, action)
            return (env_out.state, k), (action, env_out.reward)

        state = env.reset(, initial_edges)
        _, (eval_actions, eval_rewards) = lax.scan(
            step_fn, (state, key), None, length=num_v
        )
        return eval_actions, jnp.sum(eval_rewards)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Meaning of 'Best': Highest Reward = Least Mathematical Operations")
    print(f"Best Reward seen during training: {best_reward}")

    # Run greedy evaluation
    eval_order, eval_total_reward = greedy_eval(agent, env, key)
    print(f"Final Agent Greedy Order: {eval_order}")
    print(f"Final Agent Total Reward: {eval_total_reward}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
