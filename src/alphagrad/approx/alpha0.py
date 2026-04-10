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
from graphax import examples, jacve
from tqdm import tqdm

from alphagrad.approx.env import VertexEliminationEnv
from alphagrad.transformer import MLP, Encoder, PositionalEncoder

MAX_TOKENS = 4096


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


def data_gen(fn_str):
    if fn_str == "NeuralNetwork":

        @jax.jit
        def get_kinematics_data(keys):
            r1 = jrand.uniform(keys[0])
            th1 = jrand.uniform(keys[1], minval=-jnp.pi, maxval=jnp.pi)
            r2 = jrand.uniform(keys[2])
            th2 = jrand.uniform(keys[3], minval=-jnp.pi, maxval=jnp.pi)
            x = jnp.stack([r1, th1 / jnp.pi, r2, th2 / jnp.pi], axis=-1)
            y = jnp.stack(
                [
                    r1 * jnp.cos(th1),
                    r1 * jnp.sin(th1),
                    r2 * jnp.cos(th2),
                    r2 * jnp.sin(th2),
                ],
                axis=-1,
            )
            return x, y

        return get_kinematics_data


# ---------------------------------------------------------------------------
# AlphaZero Agent
# ---------------------------------------------------------------------------


class TransformerAlphaZeroAgent(eqx.Module):
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
        k1, k2, k3, k4 = jrand.split(key, 4)
        self.num_actions = num_actions

        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k2)

        self.policy_head = MLP(embd_dim, num_actions, policy_dims, key=k3)
        self.value_head = MLP(embd_dim, 1, value_dims, key=k4)

    def __call__(self, tokens, key=None):
        if tokens.ndim == 1:
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            x = self.encoder(x, key=key if key is not None else jrand.PRNGKey(0))
            summary = jnp.mean(x, axis=0)
            logits = self.policy_head(summary)
            value = self.value_head(summary)[0]
            return logits, value
        else:
            return jax.vmap(self, in_axes=(0, None))(tokens, key)


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch(trajectories, minibatches, key):
    num_envs, rollout_length, features = trajectories.shape
    size = num_envs * rollout_length // minibatches
    valid_samples = size * minibatches

    trajectories = trajectories.reshape(-1, features)
    trajectories = jrand.permutation(key, trajectories, axis=0)
    trajectories = trajectories[:valid_samples]

    return trajectories.reshape(minibatches, size, features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="AlphaZero_Vertex")
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

    sp_valid_mask_np = np.zeros((3, total_v), dtype=np.float32)
    for i, eqn in enumerate(closed_jaxpr.jaxpr.eqns):
        sp_valid_mask_np[0, i] = 1.0
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue
        out_ndim = len(eqn.outvars[0].aval.shape)
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        if invars:
            max_in_ndim = max(len(v.aval.shape) for v in invars)
            if out_ndim >= 1 and max_in_ndim >= 1:
                sp_valid_mask_np[1, i] = 1.0
            if out_ndim >= 1 and max_in_ndim >= 2:
                sp_valid_mask_np[2, i] = 1.0

    sp_valid_mask = jnp.array(sp_valid_mask_np)

    EPISODES = args.episodes
    NUM_ENVS = os.cpu_count() or 64
    LR = 1e-4
    MINIBATCHES = 32
    NUM_SIMULATIONS = 50

    OBS_SHAPE = 1024
    NUM_ACTIONS = 3 * total_v
    ROLLOUT_LENGTH = num_valid

    agent_key, key = jrand.split(key)
    agent = TransformerAlphaZeroAgent(
        vocab_size=256,
        embd_dim=64,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
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

    def get_action_mask(state):
        vertex_valid = (
            jnp.zeros(total_v, dtype=jnp.float32).at[valid_vertices - 1].set(1.0)
        )
        arange_v = jnp.arange(num_valid)
        active_mask = (arange_v < jnp.expand_dims(state.step_count, -1)).astype(
            jnp.float32
        )
        already_chosen = (
            jnp.zeros(total_v, dtype=jnp.float32).at[state.order - 1].add(active_mask)
        )
        vertex_available = vertex_valid * (1.0 - jnp.clip(already_chosen, 0.0, 1.0))
        available_matrix = jnp.expand_dims(vertex_available, 0) * sp_valid_mask
        return available_matrix.reshape(-1)

    # ---------------------------------------------------------------------------
    # MCTX Interfaces for AlphaZero (using exact env dynamics)
    # ---------------------------------------------------------------------------

    @eqx.filter_vmap(in_axes=(None, 0))
    def root_fn(agent, state):
        logits, value = agent(state.tokens)
        mask = get_action_mask(state)
        logits = jnp.where(mask > 0.5, logits, -1e9)
        return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

    @eqx.filter_vmap(in_axes=(None, None, 0, 0))
    def recurrent_fn(agent, rng_key, action, state):
        sp_type = action // total_v
        target_vertex = (action % total_v) + 1
        env_action = sp_type * MAX_TOKENS + target_vertex

        env_out = env.step(state, env_action)
        next_state = env_out.state

        logits, value = agent(next_state.tokens)
        mask = get_action_mask(next_state)
        logits = jnp.where(mask > 0.5, logits, -1e9)

        return mctx.RecurrentFnOutput(
            reward=env_out.rewards,
            discount=jnp.ones_like(env_out.rewards),
            prior_logits=logits,
            value=value,
        ), next_state

    # ---------------------------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------------------------

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def rollout_fn(agent, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, key):
            search_key, act_key = jrand.split(key)

            invalid_actions_mask = 1.0 - get_action_mask(state)

            roots = root_fn(agent, jax.tree.map(lambda x: jnp.expand_dims(x, 0), state))

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

            distribution = distrax.Categorical(probs=mcts_policy)
            action_idx = distribution.sample(seed=act_key)

            sp_type = action_idx // total_v
            target_vertex = (action_idx % total_v) + 1
            env_action = sp_type * MAX_TOKENS + target_vertex

            env_out = env.step(state, env_action)

            new_sample = jnp.concatenate(
                (
                    state.tokens.astype(jnp.float32),
                    jnp.array([action_idx], dtype=jnp.float32),
                    jnp.atleast_1d(env_out.rewards),
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

    def loss(agent, batch):
        tokens = batch[:, :OBS_SHAPE].astype(jnp.int32)
        target_returns = batch[:, OBS_SHAPE + 1]
        target_policies = batch[:, OBS_SHAPE + 2 : OBS_SHAPE + 2 + NUM_ACTIONS]

        logits, values = agent(tokens)

        p_loss = jnp.mean(
            -jnp.sum(target_policies * jnn.log_softmax(logits + 1e-7), axis=-1)
        )
        v_loss = jnp.mean(jnp.square(values - target_returns))

        return p_loss + v_loss, (p_loss, v_loss)

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

        rewards = trajectories[:, :, OBS_SHAPE + 1]
        returns = jnp.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
        trajectories = trajectories.at[:, :, OBS_SHAPE + 1].set(returns)

        batches = shuffle_and_batch(trajectories, MINIBATCHES, subkey)

        for i in range(MINIBATCHES):
            agent, opt_state, metrics = train_agent(agent, opt_state, batches[i])

        p_loss, v_loss = metrics

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
            }
        )

        pbar.set_description(
            f"best: {best_reward:.1f}, mean: {jnp.mean(total_rewards):.1f}"
        )

    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
