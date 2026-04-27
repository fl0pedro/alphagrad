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

from alphagrad.approx.env import VertexEliminationEnv
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.utils import entropy

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
# Agent (GDPO - No Value Network)
# ---------------------------------------------------------------------------
class TransformerGDPOAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
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
        seq_len,
        key,
    ):
        k1, k2, k3 = jrand.split(key, 3)
        self.num_actions = num_actions
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k2)
        self.policy_head = MLP(embd_dim, num_actions, policy_dims, key=k3)

    def __call__(self, tokens, key=None, inference=False):
        if tokens.ndim == 1:
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            enc_key = key if key is not None else jrand.PRNGKey(0)
            x = self.encoder(x, key=enc_key)
            summary = jnp.mean(x, axis=0)
            logits = self.policy_head(summary)
            return logits
        else:
            batched_call = jax.vmap(self, in_axes=(0, None, None))
            return batched_call(tokens, key, inference)


# ---------------------------------------------------------------------------
# RL helpers
# ---------------------------------------------------------------------------
def get_num_clipping_triggers(ratio, eps):
    _ratio = jnp.where(ratio <= 1.0 + eps, ratio, 0.0)
    _ratio = jnp.where(ratio >= 1.0 - eps, 1.0, 0.0)
    return jnp.sum(_ratio)


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def get_log_probs(agent, tokens, action, key):
    logits = agent(tokens, key=key)
    action_idx = action - 1
    prob_dist = jnn.softmax(logits, axis=-1)
    log_prob = jnp.log(prob_dist[action_idx] + 1e-7)
    return log_prob, prob_dist, entropy(prob_dist)


@jax.jit
def compute_gdpo_advantages(rewards):
    """
    Computes GDPO decoupled advantages for multi-reward vectors[cite: 180].
    Expects rewards shape: (NUM_QUESTIONS, GROUP_SIZE, num_rewards)
    """
    # Group-wise normalization per reward [cite: 182]
    mu_g = jnp.mean(rewards, axis=1, keepdims=True)
    std_g = jnp.std(rewards, axis=1, keepdims=True) + 1e-7
    A_k = (rewards - mu_g) / std_g

    # Sum across reward objectives [cite: 227]
    A_sum = jnp.sum(A_k, axis=-1)

    # Batch-wise normalization for numerical stability
    mu_b = jnp.mean(A_sum)
    std_b = jnp.std(A_sum) + 1e-7
    A_hat = (A_sum - mu_b) / std_b

    return A_hat


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
    parser.add_argument("--name", type=str, default="GDPO_Vertex_MultiReward")
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

    # ---- Sparsity Constraints Masking ----
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

    # ---- GDPO Specific Config ----
    NUM_QUESTIONS = 4  # Number of parallel initial states
    GROUP_SIZE = 16  # G rollouts per state [cite: 96]
    NUM_ENVS = NUM_QUESTIONS * GROUP_SIZE
    NUM_REWARDS = 2  # [FMAS, Accuracy]

    ENTROPY_WEIGHT = 0.1
    EPISODES = args.episodes
    LR = 5e-5
    EPS = 0.2
    MINIBATCHES = 32

    OBS_SHAPE = 1024
    NUM_ACTIONS = 3 * total_v
    ROLLOUT_LENGTH = num_valid

    print(
        f"NUM_ACTIONS={NUM_ACTIONS}, ROLLOUT_LENGTH={ROLLOUT_LENGTH}, "
        f"MINIBATCHES={MINIBATCHES}, NUM_ENVS={NUM_ENVS}"
    )

    # ---- Agent Model ----
    agent_key, key = jrand.split(key)
    agent = TransformerGDPOAgent(
        vocab_size=256,
        embd_dim=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        num_actions=NUM_ACTIONS,
        policy_dims=[64, 32],
        seq_len=OBS_SHAPE,
        key=agent_key,
    )

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
            logits = agent(state.tokens, key=net_key, inference=True)

            chosen = state.order
            step_idx = state.step_count

            vertex_valid = (
                jnp.zeros(total_v, dtype=jnp.float32).at[valid_vertices - 1].set(1.0)
            )
            arange_v = jnp.arange(num_valid)
            active_mask = (arange_v < jnp.expand_dims(step_idx, -1)).astype(jnp.float32)
            already_chosen = (
                jnp.zeros(total_v, dtype=jnp.float32).at[chosen - 1].add(active_mask)
            )

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

            # Assumes the updated env_out returns array [fmas_reward, acc_reward]
            rewards = env_out.rewards

            new_sample = jnp.concatenate(
                (
                    state.tokens.astype(jnp.float32),
                    jnp.expand_dims(action_idx + 1, -1).astype(jnp.float32),
                    jnp.atleast_1d(
                        rewards
                    ),  # Now a multi-reward vector natively shape [2]
                    prob_dist,
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

        old_prob_dist = trajectories[
            :, OBS_SHAPE + 1 + NUM_REWARDS : OBS_SHAPE + 1 + NUM_REWARDS + NUM_ACTIONS
        ]
        advantages = trajectories[:, -1]  # Appended to features during training step

        log_probs, prob_dist, entropies = get_log_probs(agent, tokens, actions, keys)
        old_log_probs = jnp.log(
            jax.vmap(lambda p, a: p[a - 1])(old_prob_dist, actions) + 1e-2
        )
        ratio = jnp.exp(log_probs - old_log_probs)

        clipping_objective = jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * advantages,
        )
        ppo_loss = jnp.mean(-clipping_objective)
        entropy_loss = jnp.mean(entropies)
        total_loss = ppo_loss - ENTROPY_WEIGHT * entropy_loss

        return total_loss, (
            ppo_loss,
            entropy_loss,
            total_loss,
            get_num_clipping_triggers(ratio, EPS) / len(ratio),
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

        # Retrieve sequence-level episodic rewards across all objectives
        all_rewards = trajectories[:, :, OBS_SHAPE + 1 : OBS_SHAPE + 1 + NUM_REWARDS]
        seq_rewards = jnp.sum(all_rewards, axis=1)  # Shape: (NUM_ENVS, NUM_REWARDS)

        # Compute Group-Relative Decoupled Advantages
        seq_rewards_grouped = seq_rewards.reshape(
            NUM_QUESTIONS, GROUP_SIZE, NUM_REWARDS
        )
        A_hat = compute_gdpo_advantages(
            seq_rewards_grouped
        )  # Shape: (NUM_QUESTIONS, GROUP_SIZE)

        # Broadcast advantages to all tokens per environment rollout
        A_hat_flat = A_hat.reshape(NUM_ENVS)
        A_hat_tokens = jnp.repeat(
            A_hat_flat[:, None], ROLLOUT_LENGTH, axis=1
        )  # (NUM_ENVS, ROLLOUT_LENGTH)

        # Append advantages to trajectories to pass to the dataloader
        trajectories = jnp.concatenate(
            [trajectories, jnp.expand_dims(A_hat_tokens, -1)], axis=-1
        )
        batches = shuffle_and_batch(trajectories, MINIBATCHES, subkey)

        for i in range(MINIBATCHES):
            train_key, key = jrand.split(key)
            agent, opt_state, metrics = train_agent(
                agent, opt_state, batches[i], train_key
            )

        ppo_loss, entropy_loss, total_loss, trigger_ratio = metrics

        # Logging best logic relies on total compounded return, summing all multi-reward vectors arbitrarily. Adjust as needed.
        total_compound_returns = jnp.sum(seq_rewards, axis=-1)
        max_idx = jnp.argmax(total_compound_returns)
        best_reward = total_compound_returns[max_idx]
        best_act_seq = trajectories[max_idx, :, OBS_SHAPE]

        if best_reward > best_global_return:
            best_global_return = best_reward
            best_global_act_seq = best_act_seq
            action_pairs = [
                (int((i - 1) % total_v) + 1, int((i - 1) // total_v))
                for i in best_act_seq
            ]

            print(f"\nNew best return: {best_reward}")
            print(f"New best action sequence (vertex, sp_type): {action_pairs}")

            elim_order_table.add_data(
                episode, float(best_reward), np.array(best_act_seq)
            )

        wandb.log(
            {
                "best_return": float(best_reward),
                "ppo loss": float(ppo_loss),
                "entropy loss": float(entropy_loss),
                "total loss": float(total_loss),
                "clipping trigger ratio": float(trigger_ratio),
            }
        )

        pbar.set_description(
            f"best: {best_reward:.1f}, mean: {jnp.mean(total_compound_returns):.1f}"
        )

    wandb.log({"Elimination order": elim_order_table})
    best_pairs = [
        (int((i - 1) % total_v) + 1, int((i - 1) // total_v))
        for i in best_global_act_seq
    ]
    print(
        f"\nBest vertex elimination sequence after {EPISODES} episodes: "
        f"{best_pairs} with {best_global_return} score."
    )


if __name__ == "__main__":
    main()
