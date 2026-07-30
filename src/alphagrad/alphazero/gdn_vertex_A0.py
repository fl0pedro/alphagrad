import equinox as eqx
import graphax.examples as examples
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import optax
import wandb
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

import alphagrad.alphazero.gdn_tree_search as ts
import alphagrad.utils as u
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.vertexgame.vertex_game_w_tokens import (
    MAX_TOKENS,
    VertexEliminationEnv,
    print_thread_metrics,
)


class AlphaZeroAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
    value_head: MLP
    num_actions: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int,
        embd_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        num_actions: int,
        policy_dims: list,
        value_dims: list,
        key: jrand.PRNGKey,
    ):
        self.num_actions = num_actions
        k_emb, k_pos, k_enc, k_pol, k_val = jrand.split(key, 5)

        self.embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size, embedding_size=embd_dim, key=k_emb
        )
        self.pos_enc = PositionalEncoder(embd_dim, MAX_TOKENS)
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            embd_dim=embd_dim,
            hidden_dim=hidden_dim,
            key=k_enc,
        )

        self.policy_head = MLP(
            in_size=embd_dim, out_size=num_actions, layers=policy_dims, key=k_pol
        )

        self.value_head = MLP(
            in_size=embd_dim, out_size=1, layers=value_dims, key=k_val
        )

    def __call__(self, tokens, key=None):
        if tokens.ndim == 1:
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            x = self.encoder(x, key=key)
            summary = jnp.mean(x, axis=0)
            logits = self.policy_head(summary)
            value = self.value_head(summary).squeeze(-1)
            # concatenate [value, logits] on axis=-1
            return jnp.concatenate([jnp.array([value]), logits], axis=-1)
        else:
            if key is not None and key.ndim > 1:
                batched_call = jax.vmap(self, in_axes=(0, 0))
            else:
                batched_call = jax.vmap(self, in_axes=(0, None))
            return batched_call(tokens, key)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Helmholtz")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_simulations", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", type=str, default="disabled")
    args = parser.parse_args()

    key = jrand.PRNGKey(args.seed)

    target_fn = examples.Helmholtz
    xs = (jnp.array([0.05, 0.15, 0.15, 0.2]),)
    # target_fn = examples.g
    # xs = [jrand.uniform(key, (1,)) for _ in range(15)]

    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)

    env = VertexEliminationEnv.from_jaxpr(closed_jaxpr, args=xs, num_envs=0)
    num_v = len(closed_jaxpr.jaxpr.eqns)
    valid_vertices = jnp.array(env.valid_vertices, dtype=jnp.int32)

    key, agent_key = jrand.split(key)
    agent = AlphaZeroAgent(
        vocab_size=256,
        embd_dim=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        num_actions=num_v,
        policy_dims=[64, 32],
        value_dims=[64, 32],
        key=agent_key,
    )

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_array))

    mesh = Mesh(jax.devices(), axis_names=("batch",))
    repl_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("batch"))

    # Replicate model and optimizer state across all GPUs
    agent = jtu.tree_map(
        lambda x: jax.device_put(x, repl_sharding) if eqx.is_array(x) else x, agent
    )
    opt_state = jtu.tree_map(
        lambda x: jax.device_put(x, repl_sharding) if eqx.is_array(x) else x, opt_state
    )

    value_transform, inverse_value_transform = u.get_value_tf("log")

    tree_search_fn = ts.make_tree_search(
        agent,
        env.step,
        num_v,
        num_v,
        valid_vertices,
        inverse_value_transform,
        num_simulations=args.num_simulations,
        num_considered_actions=min(5, num_v),
    )

    @eqx.filter_jit
    def train_step(agent, opt_state, data, key):
        def loss_fn(agent):
            Rollout, B, T = data["obs"].shape

            # Swap to (B, Rollout, ...) before flattening to preserve device sharding isolation
            obs_flat = jnp.swapaxes(data["obs"], 0, 1).reshape(-1, T)

            keys = jrand.split(key, obs_flat.shape[0])
            output = agent(obs_flat, keys)
            v_preds = output[:, 0]
            policy_logits = output[:, 1:]

            targets_flat = jnp.swapaxes(data["policy"], 0, 1).reshape(-1, num_v)
            policy_loss = jnp.mean(
                optax.softmax_cross_entropy(policy_logits, targets_flat)
            )

            value_targets_flat = jnp.swapaxes(data["value"], 0, 1).reshape(-1)
            value_loss = jnp.mean(
                jnp.square(v_preds - value_transform(value_targets_flat))
            )

            return policy_loss + value_loss, (policy_loss, value_loss)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(agent)
        updates, opt_state = optimizer.update(grads, opt_state)
        agent = eqx.apply_updates(agent, updates)
        return agent, opt_state, loss, aux

    wandb.init(project="AlphaGrad-A0", config=vars(args), mode=args.wandb)

    # Need dummy array to vmap reset
    dummy_axes = jax.device_put(jnp.arange(args.num_envs), data_sharding)
    batched_reset = jax.vmap(lambda _: env.reset())

    pbar = tqdm(range(args.episodes))

    best_return = -jnp.inf
    best_strategy = None

    for ep in pbar:
        key, ts_key, train_key = jrand.split(key, 3)

        states = batched_reset(dummy_axes)
        num_muls = jnp.zeros(args.num_envs)
        init_carry = (states, num_muls, ts_key)

        final_state, total_rewards, data = tree_search_fn(agent, init_carry)

        agent, opt_state, loss, aux = train_step(agent, opt_state, data, train_key)

        # In AlphaZero the reward is the negated num. multiplications / vertices evaluated
        # The smaller (closer to 0), the better.
        mean_reward = jnp.mean(total_rewards)

        # Track best performance
        max_batch_reward = jnp.max(total_rewards)
        best_idx = jnp.argmax(total_rewards)
        if max_batch_reward > best_return:
            best_return = max_batch_reward
            best_strategy = final_state.order[best_idx].tolist()
            print(f"\nNew best return: {best_return}")
            print(f"New best action sequence: {best_strategy}\n")

        p_loss, v_loss = aux
        wandb.log(
            {
                "loss": loss,
                "policy_loss": p_loss,
                "value_loss": v_loss,
                "mean_reward": mean_reward,
                "best_return": best_return,
            }
        )
        pbar.set_description(f"Loss: {loss:.4f} Rew: {mean_reward:.2f}")

    print(
        f"\nBest vertex elimination sequence after {args.episodes} episodes: {best_strategy} with {best_return} score."
    )


if __name__ == "__main__":
    main()

    print_thread_metrics()
