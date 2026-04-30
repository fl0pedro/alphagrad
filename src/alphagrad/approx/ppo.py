import argparse
import inspect
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
from graphax import examples
from tqdm import tqdm

import heapq
from alphagrad.approx.env import MAX_TOKENS, VertexEliminationEnv
from alphagrad.transformer import MLP, Encoder, PositionalEncoder
from alphagrad.utils import entropy, explained_variance, symexp, symlog

class Trajectory(NamedTuple):
    tokens: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    next_value: jax.Array
    prob_dist: jax.Array
    discount: jax.Array
    action_mask: jax.Array


class TrainBatch(NamedTuple):
    tokens: jax.Array
    action: jax.Array
    old_prob_dist: jax.Array
    estim_returns: jax.Array
    norm_adv: jax.Array
    action_mask: jax.Array


def data_gen(fn_str):
    fn = None
    if fn_str == "Helmholtz":

        @jax.jit
        def fn(keys):
            x = jrand.uniform(keys[0], (4,))
            return (x / jnp.sum(x) * 0.9,)

        return fn

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

            y += 0.05 * jrand.normal(keys[4], y.shape)

            return x, y

        return fn

    if "Encoder" in fn_str or "Decoder" in fn_str:

        @jax.jit
        def fn(keys):
            if fn_str.startswith("Vmapped"):
                shape_x = (16, 4, 4)
                shape_y = (16, 4, 4)
            else:
                shape_x = (4, 4)
                shape_y = (4, 4)

            x = jrand.normal(keys[0], shape_x)

            y_base = jnp.sin(x * jnp.pi) + jnp.cos(x * jnp.pi)
            y = jax.nn.sigmoid(y_base) + 0.05 * jrand.normal(keys[1], shape_y)

            return x, y

        return fn


def _neural_network(x, y, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2


def get_args(fn_str, key):
    if fn_str.endswith("NeuralNetwork"):
        shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,)]
    elif fn_str.endswith("Perceptron"):
        shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,), (8,), (8,)]
    elif "EncoderDecoder" in fn_str:
        shapes = [(4, 4)] * 13 + [(4,)] * 8
    elif "Encoder" in fn_str:
        shapes = [(4, 4)] * 10 + [(4,)] * 6
    else:
        return {
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
        }[fn_str]

    if fn_str.startswith("Vmapped"):
        shapes[0] = (16, *shapes[0])
        if "Encoder" in fn_str or fn_str.endswith(("NeuralNetwork", "Perceptron")):
            shapes[1] = (16, *shapes[1])

    keys = jax.random.split(key, len(shapes))
    return [jax.random.normal(k, s) for k, s in zip(keys, shapes)]


def get_fn(fn_str):
    if fn_str.endswith("NeuralNetwork"):
        fn = _neural_network
    elif fn_str.endswith("Perceptron"):
        fn = examples.Perceptron
    else:
        fn = getattr(examples, fn_str, None)
        if fn is None:
            raise ValueError(f"Target function '{fn_str}' not found in examples.")

    if fn_str.startswith("Vmapped"):
        num_args = len(inspect.signature(fn).parameters)
        has_y = "Encoder" in fn_str or fn_str.endswith(("NeuralNetwork", "Perceptron"))

        mapped_axes = (0, 0) if has_y else (0,)
        static_axes = (None,) * (num_args - len(mapped_axes))

        fn = jax.vmap(fn, in_axes=mapped_axes + static_axes)

    return fn


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
            summary = jnp.sum(x * mask, axis=0) / jnp.maximum(
                jnp.sum(mask, axis=0), 1e-9
            )
            logits = self.policy_head(summary)
            value = self.value_head(summary)
            return logits, value
        else:
            batched_call = jax.vmap(self, in_axes=(0, None, None))
            return batched_call(tokens, key, inference)


def init_linear_weights(model, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    get_biases = lambda m: [
        x.bias
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x) and x.bias is not None
    ]

    weights = get_weights(model)
    biases = get_biases(model)
    init_fn = jnn.initializers.orthogonal(jnp.sqrt(2))

    new_weights = [
        init_fn(subkey, weight.shape)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_biases = [jnp.zeros_like(bias) for bias in biases]

    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_biases, new_model, new_biases)
    return new_model


def reward_normalization_fn(reward):
    return symlog(reward)


def inverse_reward_normalization_fn(reward):
    return symexp(reward)


def get_num_clipping_triggers(ratio, eps):
    _ratio = jnp.where(ratio <= 1.0 + eps, ratio, 0.0)
    _ratio = jnp.where(ratio >= 1.0 - eps, 1.0, 0.0)
    return jnp.sum(_ratio)


@partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
def get_log_probs_and_value(agent, tokens, action, action_mask, key):
    logits, value = agent(tokens, key=key)
    action_idx = action - 1
    masked_logits = jnp.where(action_mask > 0.5, logits, -1e9)
    prob_dist = jnn.softmax(masked_logits, axis=-1)
    log_prob = jnp.log(prob_dist[action_idx] + 1e-8)
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
        return (episodic_return, advantage), (episodic_return, estim_return, advantage)

    inputs = (rewards, dones, values, next_values, discounts)
    rev_inputs = jax.tree.map(lambda x: x[::-1], inputs)
    init_val = jnp.zeros_like(rewards[0])
    _, output = lax.scan(loop_fn, (init_val, init_val), rev_inputs)
    return jax.tree.map(lambda x: x[::-1], output)


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch(tree, minibatches, key):
    leaves, _ = jax.tree_util.tree_flatten(tree)
    num_envs, rollout_length = leaves[0].shape[:2]
    size = num_envs * rollout_length // minibatches
    valid_samples = size * minibatches

    indices = jrand.permutation(key, jnp.arange(num_envs * rollout_length))
    indices = indices[:valid_samples].reshape(minibatches, size)

    def _process(x):
        x = x.reshape(-1, *x.shape[2:])
        return x[indices]

    return jax.tree_util.tree_map(_process, tree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="approx-ppo")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seed", type=int, default=250197)
    parser.add_argument("--wandb", type=str, default="offline")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--example", type=str, default="Helmholtz")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--disable-sparsification", action="store_true")
    parser.add_argument("--reward-type", type=str, default="analytical")
    parser.add_argument("--rewards", nargs="+", type=str, default=["cmp", "mem", "acc"], choices=["cmp", "mem", "acc"])
    parser.add_argument("--lambda-cmp", type=float, default=1.0)
    parser.add_argument("--lambda-mem", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=10, help="Number of top trajectories to capture for each metric")
    parser.add_argument("--capture-perfect-grads", action="store_true", help="Capture trajectories with perfect accuracy (cosine similarity = 1.0)")
    parser.add_argument("--exec-on-gpu", action="store_true", help="Enforce execution on GPU 0 and data on GPU 1")
    args = parser.parse_args()

    if args.exec_on_gpu:
        try:
            gpus = jax.devices("gpu")
        except:
            gpus = []
        if len(gpus) < 2:
            raise RuntimeError(f"Requested --exec-on-gpu but only {len(gpus)} GPU(s) found. "
                               "Check your --gpus argument and CUDA_VISIBLE_DEVICES.")
        main_device = gpus[0]
    else:
        main_device = None

    if args.no_jit:
        jax.config.update("jax_disable_jit", True)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    target_fn = get_fn(args.example)
    xs = get_args(args.example, args_key)
    gen = data_gen(args.example)

    closed_jaxpr = jax.make_jaxpr(target_fn)(*xs)

    env_target_fun = target_fn if "acc" in args.rewards else None
    argnums = None
    if args.example.endswith("NeuralNetwork"):
        argnums = (2, 3, 4, 5)
    elif args.example.endswith("Perceptron"):
        argnums = (2, 3, 4, 5, 6, 7)
    else:
        argnums = (0,)


    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr, args=xs, argnums=argnums, num_envs=0, data_gen=gen, target_fun=env_target_fun,
        reward_type=args.reward_type, exec_on_gpu=args.exec_on_gpu,
    )

    total_v = len(closed_jaxpr.jaxpr.eqns)
    valid_vertices = jnp.array(env.valid_vertices, dtype=jnp.int32)
    num_valid = len(env.valid_vertices)

    print(
        f"Total vertices: {total_v}, Valid vertices: {num_valid}, "
        f"Valid set: {env.valid_vertices}"
    )

    sp_valid_mask_np = np.zeros((5, total_v), dtype=np.float32)

    for i, eqn in enumerate(closed_jaxpr.jaxpr.eqns):
        sp_valid_mask_np[0, i] = 1.0

        if args.disable_sparsification:
            continue

        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue

        out_ndim = len(eqn.outvars[0].aval.shape)
        invars = [v for v in eqn.invars if hasattr(v, "aval")]

        if invars:
            min_in_ndim = min(len(v.aval.shape) for v in invars)

            if out_ndim == 2 and min_in_ndim == 2:
                sp_valid_mask_np[1:5, i] = 1.0
            elif out_ndim == 1 and min_in_ndim == 2:
                sp_valid_mask_np[1:3, i] = 1.0
            elif out_ndim == 2 and min_in_ndim == 1:
                sp_valid_mask_np[1, i] = 1.0
                sp_valid_mask_np[3, i] = 1.0
            elif out_ndim == 1 and min_in_ndim == 1:
                sp_valid_mask_np[1, i] = 1.0

    sp_valid_mask = jnp.array(sp_valid_mask_np)

    ENTROPY_WEIGHT = 0.05
    VALUE_WEIGHT = 0.5
    EPISODES = args.episodes
    # Synchronize environment parallelism with model batch size for Vmapped examples
    if "Vmapped" in args.example:
        NUM_ENVS = 16
    else:
        NUM_ENVS = os.cpu_count() or 64
    LR = 3e-4
    GAE_LAMBDA = 0.95
    EPS = 0.2
    MINIBATCHES = 32
    PPO_EPOCHS = 2

    NUM_REWARDS = len(args.rewards)
    OBS_SHAPE = MAX_TOKENS
    NUM_ACTIONS = 5 * total_v
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

    def get_policy_weight(agent):
        return agent.policy_head.layers[-2].weight

    agent = eqx.tree_at(get_policy_weight, agent, get_policy_weight(agent) * 0.1)

    if args.exec_on_gpu:
        agent = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, main_device) if eqx.is_array(x) else x,
            agent
        )

    def generate_eval_samples(env_obj, key):
        config = env_obj.config
        args = env_obj.args

        def get_one_sample(k):
            dk, wk = jrand.split(k)
            e_args = list(args)
            if config.data_gen is not None:
                data = config.data_gen(jrand.split(dk, 5))
                for i, d in enumerate(data):
                    e_args[i] = d
            if config.argnums:
                w_keys = jrand.split(wk, len(config.argnums))
                for i, arg_idx in enumerate(config.argnums):
                    if config.data_gen is not None and arg_idx < len(data):
                        continue
                    curr_val = e_args[arg_idx]
                    e_args[arg_idx] = jrand.normal(
                        w_keys[i], curr_val.shape, curr_val.dtype
                    )

            return tuple(e_args)

        keys = jrand.split(key, 10)
        stacked_args = jax.vmap(get_one_sample)(keys)
        return stacked_args


    def reset_envs(env_obj):
        def _single_reset(_):
            return env_obj.reset()

        return jax.vmap(_single_reset)(jnp.arange(NUM_ENVS))

    LAMBDA_CMP = args.lambda_cmp
    LAMBDA_MEM = args.lambda_mem

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def rollout_fn(agent, env_obj, rollout_length, env_state, key):
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

            env_out = env_obj.step(state, env_action)
            next_state = env_out.state
            raw_rewards = env_out.reward
            # Store raw rewards for stable GAE
            rewards = jnp.array([
                raw_rewards[0],
                raw_rewards[1],
                raw_rewards[2]
            ])[:NUM_REWARDS]
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
                action_mask=available_flat,
            )

            return next_state, (transition, raw_rewards)

        final_state, (traj, all_raw_rewards) = lax.scan(step_fn, env_state, keys)
        return final_state, traj, all_raw_rewards[-1]


    schedule = optax.cosine_decay_schedule(LR, EPISODES * PPO_EPOCHS * MINIBATCHES, 0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(schedule, b1=0.9, eps=1e-7),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    def loss(agent, batch: TrainBatch, keys):
        log_probs, prob_dist, values, entropies = get_log_probs_and_value( # only measure entropy at the beginning?
            agent, batch.tokens, batch.action, batch.action_mask, keys
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
            jnp.sum(
                (values - reward_normalization_fn(batch.estim_returns)) ** 2, axis=-1
            )
        )

        explained_var = explained_variance(
            batch.norm_adv, jnp.sum(batch.estim_returns, axis=-1)
        )
        kl_div = jnp.mean(
            optax.kl_divergence(jnp.log(prob_dist + 1e-7), batch.old_prob_dist)
        )

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

    def train_episode(agent, opt_state, env_states, env_obj, key):
        subkey, key = jrand.split(key)
        rollout_key, key = jrand.split(key)
        rollout_keys = jrand.split(rollout_key, NUM_ENVS)

        env_states, traj, total_rewards_full = rollout_fn(agent, env_obj, ROLLOUT_LENGTH, env_states, rollout_keys)


        _, estim_returns, advantages = get_advantages(
            traj.reward,
            traj.done,
            traj.value,
            traj.next_value,
            traj.discount,
            GAE_LAMBDA,
        )

        def normalize(x):
            return (x - jnp.mean(x)) / (jnp.std(x) + 1e-7)

        norm_adv_components = jax.vmap(normalize, in_axes=-1, out_axes=-1)(
            advantages.reshape(-1, advantages.shape[-1])
        ).reshape(advantages.shape)
        
        weights = jnp.array([args.lambda_cmp, 1.0, args.lambda_mem])[:NUM_REWARDS]
        norm_adv = jnp.sum(norm_adv_components * weights, axis=-1)

        full_batch = TrainBatch(
            tokens=traj.tokens,
            action=traj.action,
            old_prob_dist=traj.prob_dist,
            estim_returns=estim_returns,
            norm_adv=norm_adv,
            action_mask=traj.action_mask,
        )

        dynamic_carry, static_carry = eqx.partition((agent, opt_state), eqx.is_array)

        def train_epoch(carry, epoch_key):
            batches = shuffle_and_batch(full_batch, MINIBATCHES, epoch_key)
            mb_keys = jrand.split(epoch_key, MINIBATCHES)

            def train_minibatch(c, batch_and_key):
                comb_agent, comb_opt_state = eqx.combine(c, static_carry)
                batch, t_key = batch_and_key
                keys = jrand.split(t_key, batch.tokens.shape[0])
                grads, metrics = eqx.filter_grad(loss, has_aux=True)(
                    comb_agent, batch, keys
                )
                updates, new_opt_state = optimizer.update(
                    grads, comb_opt_state, comb_agent
                )
                new_agent = eqx.apply_updates(comb_agent, updates)
                next_carry, _ = eqx.partition((new_agent, new_opt_state), eqx.is_array)
                return next_carry, metrics

            return lax.scan(train_minibatch, carry, (batches, mb_keys))

        epoch_keys = jrand.split(subkey, PPO_EPOCHS)
        dynamic_carry, metrics_seq = lax.scan(train_epoch, dynamic_carry, epoch_keys)

        agent, opt_state = eqx.combine(dynamic_carry, static_carry)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics_seq)

        # total_rewards_full shape: (NUM_ENVS, 3)
        return agent, opt_state, env_states, metrics, total_rewards_full, traj.action

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
        "top_n_total": [],  # (value, episode, rewards, actions)
        "top_n_cmp": [],
        "top_n_mem": [],
        "top_n_acc": [],
    }

    def host_log(ep, all_rets, all_seqs, mean_r, mets):
        ep = int(ep)
        all_rets = np.array(all_rets)
        all_seqs = np.array(all_seqs)
        mean_r = np.atleast_1d(np.array(mean_r))

        host_state["samplecounts"] += NUM_ENVS * ROLLOUT_LENGTH
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
        ) = [float(m) for m in mets]

        # Categories mapping in all_rets (raw_rewards from env.py):
        # 0: -cmp, 1: error (acc), 2: -mem
        weights = np.array([args.lambda_cmp, 1.0, args.lambda_mem])[:NUM_REWARDS]
        
        for i in range(all_rets.shape[0]):
            rets = all_rets[i]
            seq = all_seqs[i]
            
            # 1. Total Reward
            total_ret = float(np.sum(rets[:NUM_REWARDS] * weights))
            if len(host_state["top_n_total"]) < args.top_n:
                heapq.heappush(host_state["top_n_total"], (total_ret, ep, list(rets), list(seq)))
            else:
                heapq.heappushpop(host_state["top_n_total"], (total_ret, ep, list(rets), list(seq)))
            
            # 2. CMP (minimize FLOPs -> maximize -cmp)
            cmp_val = float(rets[0])
            if len(host_state["top_n_cmp"]) < args.top_n:
                heapq.heappush(host_state["top_n_cmp"], (cmp_val, ep, list(rets), list(seq)))
            else:
                heapq.heappushpop(host_state["top_n_cmp"], (cmp_val, ep, list(rets), list(seq)))
                
            # 3. Memory (minimize memory -> maximize -mem)
            mem_val = float(rets[2])
            if len(host_state["top_n_mem"]) < args.top_n:
                heapq.heappush(host_state["top_n_mem"], (mem_val, ep, list(rets), list(seq)))
            else:
                heapq.heappushpop(host_state["top_n_mem"], (mem_val, ep, list(rets), list(seq)))
                
            # 4. Accuracy (maximize cosine similarity)
            acc_val = float(rets[1])
            if args.capture_perfect_grads or acc_val < 0.999999: # Allow perfect grads only if flag is set
                if len(host_state["top_n_acc"]) < args.top_n:
                    heapq.heappush(host_state["top_n_acc"], (acc_val, ep, list(rets), list(seq)))
                else:
                    heapq.heappushpop(host_state["top_n_acc"], (acc_val, ep, list(rets), list(seq)))

        # Update global best for traditional tracking
        best_of_batch_idx = np.argmax(np.sum(all_rets[:, :NUM_REWARDS] * weights, axis=-1))
        best_of_batch_ret = float(np.sum(all_rets[best_of_batch_idx, :NUM_REWARDS] * weights))
        
        if best_of_batch_ret > host_state["best_global_return"]:
            host_state["best_global_return"] = best_of_batch_ret
            host_state["best_global_act_seq"] = all_seqs[best_of_batch_idx]

        log_dict = {
            "best_return": host_state["best_global_return"],
            "mean_return": mean_r[0],
            "KL divergence": kl_div,
            "entropy evolution": policy_entropy,
            "explained variance": explained_var,
            "sample count": host_state["samplecounts"],
            "ppo loss": ppo_loss,
            "value loss": value_loss,
            "total loss": total_loss,
        }
        if len(mean_r) > 1:
            for i in range(1, len(mean_r)):
                log_dict[f"mean_return_{i}"] = mean_r[i]

        wandb.log(log_dict)
        pbar.update(1)
        
        # Display best from CURRENT episode in pbar
        b_ret_unnorm = np.abs(all_rets[best_of_batch_idx])
        mean_r_unnorm = np.abs(mean_r)
        b_ret_desc = ", ".join([f"{float(x):.1f}" for x in b_ret_unnorm])
        means_str = ", ".join([f"{float(x):.2f}" for x in mean_r_unnorm])
        desc = f"ent:{policy_entropy:.3f} best:{b_ret_desc} means:{means_str}"
        pbar.set_description(desc)

    for ep in range(EPISODES):
        ep_key, key = jrand.split(key)
        ep_eval_key, ep_key = jrand.split(ep_key)
        
        eval_samples = generate_eval_samples(env, ep_eval_key)
        env_episode = eqx.tree_at(lambda e: e.eval_args_samples, env, eval_samples)

        env_states = reset_envs(env_episode)
        agent, opt_state, _, metrics, total_rewards_full, actions = train_episode(
            agent, opt_state, env_states, env_episode, ep_key
        )


        host_log(ep, total_rewards_full, actions, jnp.mean(total_rewards_full[:, :NUM_REWARDS], axis=0), metrics)

    pbar.close()

    def print_top_n(name, heap, reverse_val=True):
        print(f"\nTop {args.top_n} trajectories for {name}:")
        sorted_items = sorted(heap, key=lambda x: x[0], reverse=reverse_val)
        
        table = wandb.Table(columns=["rank", "episode", "total_reward", "cmp", "acc", "mem", "sequence"])
        
        weights = np.array([args.lambda_cmp, 1.0, args.lambda_mem])[:NUM_REWARDS]

        for rank, (val, ep, rets, seq) in enumerate(sorted_items, 1):
            total_ret = np.sum(np.array(rets)[:NUM_REWARDS] * weights)
            cmp_val = -rets[0]
            acc_val = rets[1]
            mem_val = -rets[2]
            
            # Recover pairs from seq (seq is list of action_idx + 1)
            action_pairs = [
                (int((i - 1) % total_v) + 1, int((i - 1) // total_v)) for i in seq
            ]
            
            print(f"{rank}. Ep {ep} | Total Reward: {total_ret:.2f} | CMP: {cmp_val:.1f} | Acc: {acc_val:.4f} | Mem: {mem_val:.1f}")
            print(f"   Sequence (vertex, sp_type): {action_pairs}")
            
            table.add_data(rank, ep, total_ret, cmp_val, acc_val, mem_val, str(action_pairs))
            
        wandb.log({f"Top N {name}": table})

    print_top_n("Total Reward", host_state["top_n_total"])
    print_top_n("CMP (Lowest FLOPs)", host_state["top_n_cmp"])
    print_top_n("Memory (Lowest Bytes)", host_state["top_n_mem"])
    print_top_n("Accuracy (Highest Cosine Similarity)", host_state["top_n_acc"])

    wandb.log({"Elimination order": elim_order_table})


if __name__ == "__main__":
    main()
