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
from alphagrad.vertexgame.approx_game_3 import MAX_TOKENS, VertexEliminationEnv


def get_args(fn_str, key):  # StructDtypeshape or whatevr??? real data in data_gen?
    basic_args = {
        "Simple": (5.0, 7.0),
        "Lighthouse": (0.02,) * 4,
        "Helmholtz": (jnp.array([0.05, 0.15, 0.25, 0.35]),),
        "RobotArm_6DOF": (0.02,) * 6,
        "RoeFlux_1d": (0.01, 0.02, 0.02, 0.01, 0.03, 0.03),
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
    if fn_str == "Helmholtz":

        @jax.jit
        def fn(keys):
            return (jrand.uniform(keys[0], (4,)),)

    if fn_str.endswith("NeuralNetwork"):

        @jax.jit
        def fn(keys):
            if fn_str.startswith("Vmapped"):
                shape = (16,)
            else:
                shape = ()
            r1 = jrand.uniform(keys[0], shape)
            th1 = jrand.uniform(keys[1], shape, minval=-jnp.pi, maxval=jnp.pi)
            r2 = jrand.uniform(keys[2], shape)
            th2 = jrand.uniform(keys[3], shape, minval=-jnp.pi, maxval=jnp.pi)

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

    return fn


# ---------------------------------------------------------------------------
# Agent (Decoupled PPO Multi-Head Critic)
# ---------------------------------------------------------------------------


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
        # Value head now outputs a vector of shape (num_rewards,)
        self.value_head = MLP(embd_dim, num_rewards, value_dims, key=k4)

    def __call__(self, tokens, key=None, inference=False):
        if tokens.ndim == 1:
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            enc_key = key if key is not None else jrand.PRNGKey(0)
            x = self.encoder(x, key=enc_key)
            summary = jnp.mean(x, axis=0)
            logits = self.policy_head(summary)
            value = self.value_head(summary)  # Outputs multi-dim value
            return logits, value
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
    action_idx = action - 1
    prob_dist = jnn.softmax(logits, axis=-1)
    log_prob = jnp.log(prob_dist[action_idx] + 1e-7)
    return log_prob, prob_dist, value, entropy(prob_dist)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
def get_advantages_decoupled(
    rewards, dones, values, next_values, discounts, gae_lambda
):
    """
    Computes GAE independently for each reward stream.
    Inputs are vectors: rewards, values, next_values have shape (ROLLOUT_LENGTH, NUM_REWARDS)
    """

    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward, done, value, next_value, discount = traj

        mask = 1.0 - done

        mask_vec = jnp.broadcast_to(mask, reward.shape)
        discount_vec = jnp.broadcast_to(discount, reward.shape)

        episodic_return = reward + discount_vec * episodic_return * mask_vec

        value_raw = inverse_reward_normalization_fn(value)
        next_value_raw = inverse_reward_normalization_fn(next_value)

        delta = reward + next_value_raw * discount_vec * mask_vec - value_raw
        advantage = delta + discount_vec * gae_lambda * lastgaelam * mask_vec

        estim_return = advantage + value_raw
        # Concatenate on the feature axis to easily separate later
        return (episodic_return, advantage), jnp.concatenate(
            [episodic_return, estim_return, advantage], axis=-1
        )

    inputs = (rewards, dones, values, next_values, discounts)
    # Scan in reverse
    rev_inputs = jax.tree.map(lambda x: x[::-1], inputs)
    init_val = jnp.zeros_like(rewards[0])
    _, output = lax.scan(loop_fn, (init_val, init_val), rev_inputs)
    # Reverse output to match trajectory timeline
    output = jax.tree.map(lambda x: x[::-1], output)
    return output


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
    parser.add_argument("--name", type=str, default="Decoupled_PPO_MultiReward")
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

    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"Valid set: {env.valid_vertices}"
    )

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

    # ---- Hyperparameters ----
    ENTROPY_WEIGHT = 0.01
    VALUE_WEIGHT = 0.5
    EPISODES = args.episodes
    NUM_ENVS = os.cpu_count() or 64
    LR = 3e-4
    GAE_LAMBDA = 0.95
    EPS = 0.2
    MINIBATCHES = 32

    NUM_REWARDS = 2  # Hardcoded for 2 decoupled rewards (e.g. FMAS + Accuracy)
    OBS_SHAPE = MAX_TOKENS
    NUM_ACTIONS = 3 * total_v
    ROLLOUT_LENGTH = num_valid

    print(
        f"NUM_ACTIONS={NUM_ACTIONS}, ROLLOUT_LENGTH={ROLLOUT_LENGTH}, "
        f"MINIBATCHES={MINIBATCHES}"
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
        num_rewards=NUM_REWARDS,
        policy_dims=[64, 32],
        value_dims=[64, 32],
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
            rewards = env_out.reward  # Must return a vector of shape (NUM_REWARDS,)
            done = env_out.terminated.astype(jnp.float32)
            discount = 1.0

            _, next_value = agent(next_state.tokens, key=next_net_key, inference=True)

            new_sample = jnp.concatenate(
                (
                    state.tokens.astype(jnp.float32),
                    jnp.expand_dims(action_idx + 1, -1).astype(jnp.float32),
                    jnp.atleast_1d(rewards),
                    jnp.expand_dims(done, -1),
                    jnp.atleast_1d(value),
                    jnp.atleast_1d(next_value),
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

    # Feature tracking constants based on our concat structure
    BASE_LEN = OBS_SHAPE + 3 + 3 * NUM_REWARDS + NUM_ACTIONS

    def loss(agent, trajectories, keys):
        tokens = trajectories[:, :OBS_SHAPE].astype(jnp.int32)
        actions = trajectories[:, OBS_SHAPE].astype(jnp.int32)

        # We don't strictly need these for standard PPO loss, but mapped offsets for clarity
        rewards = trajectories[:, OBS_SHAPE + 1 : OBS_SHAPE + 1 + NUM_REWARDS]
        old_prob_dist = trajectories[
            :,
            OBS_SHAPE + 2 + 3 * NUM_REWARDS : OBS_SHAPE
            + 2
            + 3 * NUM_REWARDS
            + NUM_ACTIONS,
        ]

        # New appended features calculated sequentially
        episodic_returns = trajectories[:, BASE_LEN : BASE_LEN + NUM_REWARDS]
        estim_returns = trajectories[
            :, BASE_LEN + NUM_REWARDS : BASE_LEN + 2 * NUM_REWARDS
        ]
        final_adv = trajectories[:, BASE_LEN + 2 * NUM_REWARDS]

        log_probs, prob_dist, values, entropies = get_log_probs_and_value(
            agent, tokens, actions, keys
        )

        old_log_probs = jnp.log(
            jax.vmap(lambda p, a: p[a - 1])(old_prob_dist, actions) + 1e-7
        )
        ratio = jnp.exp(log_probs - old_log_probs)

        num_triggers = get_num_clipping_triggers(ratio, EPS)
        trigger_ratio = num_triggers / len(ratio)

        # Apply the final aggregated Decoupled Advantage
        clipping_objective = jnp.minimum(
            ratio * final_adv,
            jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * final_adv,
        )
        ppo_loss = jnp.mean(-clipping_objective)
        entropy_loss = jnp.mean(entropies)

        # Value loss incorporates all reward heads natively
        value_loss = jnp.mean(
            jnp.sum((values - reward_normalization_fn(estim_returns)) ** 2, axis=-1)
        )

        # We can extract fit variance similarly across the sum
        explained_var = explained_variance(
            jnp.sum(final_adv), jnp.sum(estim_returns, axis=-1)
        )
        kl_div = jnp.mean(optax.kl_divergence(jnp.log(prob_dist + 1e-7), old_prob_dist))

        total_loss = (
            ppo_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy_loss
        )

        return total_loss, (
            kl_div,
            entropy_loss,
            0.0,  # Placeholder for backward compatibility
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
    elim_order_table = wandb.Table(columns=["episode", "return", "elimination order"])

    pbar = tqdm(range(EPISODES))
    samplecounts = 0
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

        # Slice features explicitly for readability
        r_slice = trajectories[:, :, OBS_SHAPE + 1 : OBS_SHAPE + 1 + NUM_REWARDS]
        d_slice = trajectories[:, :, OBS_SHAPE + 1 + NUM_REWARDS]
        v_slice = trajectories[
            :, :, OBS_SHAPE + 2 + NUM_REWARDS : OBS_SHAPE + 2 + 2 * NUM_REWARDS
        ]
        nv_slice = trajectories[
            :, :, OBS_SHAPE + 2 + 2 * NUM_REWARDS : OBS_SHAPE + 2 + 3 * NUM_REWARDS
        ]
        disc_slice = trajectories[:, :, OBS_SHAPE + 2 + 3 * NUM_REWARDS + NUM_ACTIONS]

        adv_data = get_advantages_decoupled(
            r_slice, d_slice, v_slice, nv_slice, disc_slice, GAE_LAMBDA
        )

        # adv_data shape: (NUM_ENVS, ROLLOUT_LENGTH, 3 * NUM_REWARDS)
        episodic_returns = adv_data[:, :, 0:NUM_REWARDS]
        estim_returns = adv_data[:, :, NUM_REWARDS : 2 * NUM_REWARDS]
        raw_advantages = adv_data[:, :, 2 * NUM_REWARDS : 3 * NUM_REWARDS]

        mean_adv = jnp.mean(raw_advantages, axis=(0, 1), keepdims=True)
        std_adv = jnp.std(raw_advantages, axis=(0, 1), keepdims=True) + 1e-7
        norm_adv_per_reward = (raw_advantages - mean_adv) / std_adv

        summed_adv = jnp.sum(norm_adv_per_reward, axis=-1, keepdims=True)

        batch_mean = jnp.mean(summed_adv)
        batch_std = jnp.std(summed_adv) + 1e-7
        final_adv = (summed_adv - batch_mean) / batch_std

        trajectories = jnp.concatenate(
            [trajectories, episodic_returns, estim_returns, final_adv], axis=-1
        )

        batches = shuffle_and_batch(trajectories, MINIBATCHES, subkey)

        for i in range(MINIBATCHES):
            train_key, key = jrand.split(key)
            agent, opt_state, metrics = train_agent(
                agent, opt_state, batches[i], train_key
            )
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
        # Calculate sum total reward across objectives for evaluation
        rewards_summed = jnp.sum(r_slice, axis=-1)

        total_rewards = jnp.sum(rewards_summed, axis=1)
        max_idx = jnp.argmax(total_rewards)
        best_reward = total_rewards[max_idx]
        best_act_seq = actions[max_idx]

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
                "mean_return": float(jnp.mean(total_rewards)),
                "KL divergence": float(kl_div),
                "entropy evolution": float(policy_entropy),
                "explained variance": float(explained_var),
                "sample count": samplecounts,
                "ppo loss": float(ppo_loss),
                "value loss": float(value_loss),
                "total loss": float(total_loss),
            }
        )

        pbar.set_description(
            f"ent: {policy_entropy:.4f}, best: {best_reward:.1f}, mean: {jnp.mean(total_rewards):.1f}"
        )

    wandb.log({"Elimination order": elim_order_table})
    best_pairs = [
        (int((i - 1) % total_v) + 1, int((i - 1) // total_v))
        for i in best_global_act_seq
    ]
    print(
        f"\nBest vertex elimination sequence after {EPISODES} episodes: {best_pairs} with {best_global_return} score."
    )


if __name__ == "__main__":
    main()