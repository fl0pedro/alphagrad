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
from graphax import examples, jacve
from graphax.core import vertex_elimination_jaxpr
from jax import Array
from tqdm import tqdm

from alphagrad.approx.env import VertexEliminationEnv
from alphagrad.transformer.models import ApproxModel
from alphagrad.utils import entropy, explained_variance, symexp, symlog


class Metrics(NamedTuple):
    kl_div: Array
    policy_entropy: Array
    fit_quality: Array | float
    explained_var: Array
    ppo_loss: Array
    fmas_value_loss: Array
    acc_value_loss: Array
    total_loss: Array
    clipping_trigger_ratio: Array


class TrainBatch(NamedTuple):
    tokens: Array
    actions: Array
    old_prob_dist: Array
    fmas_estim_returns: Array
    fmas_adv: Array
    acc_estim_returns: Array
    acc_adv: Array
    final_adv: Array


class Trajectory(NamedTuple):
    tokens: Array
    rewards: Array
    terminated: Array
    idx: Array
    fmas_value: Array
    acc_value: Array
    next_fmas_value: Array
    next_acc_value: Array
    prob_dist: Array
    discount: Array


def get_args(fn_str, key):
    basic_args = {
        "Simple": [(1,)] * 2,
        "Lighthouse": [(1,)] * 4,
        "Helmholtz": [(4,)],
        "RobotArm_6DOF": [(1,)] * 6,
        "RoeFlux_1d": [(1,)] * 6,
        "RoeFlux_3d": [(1,), (6,), (1,), (1,), (6,), (1,)],
        "BlackScholes_Jacobian": [(1,)] * 5,
    }
    shapes: list[tuple[int, ...]] = basic_args.get(fn_str, [])
    if fn_str.endswith("NeuralNetwork") or fn_str.endswith("Perceptron"):
        shapes = [(4,), (4,), (8, 4), (8,), (4, 8), (4,)]
        if fn_str.endswith("Perceptron"):
            shapes.extend([(), ()])
        if fn_str.startswith("Vmapped"):
            shapes[0] = (16,) + shapes[0]
            shapes[1] = (16,) + shapes[1]
    elif fn_str.startswith("Encoder"):
        shapes = [(4, 4), (2, 4), (4, 4) * 6, (4, 4), (4,), (2, 4), (2, 1)]
        if fn_str.endswith("Decoder"):
            shapes = shapes[:8] + [(4, 4) * 3] + shapes[8:]

    args = []
    for shape in shapes:
        key, subkey = jrand.split(key)
        args.append(jrand.uniform(subkey, shape, jnp.float32))

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
    if fn_str.endswith("NeuralNetwork") or fn_str.endswith("Perceptron"):

        @partial(jax.jit, static_argnums=(1,))
        def nn_mlp_data(key, size):
            k1, k2 = jrand.split(key)
            keys = jrand.split(k1, 4)
            shape = (size, 16) if fn_str.startswith("Vmapped") else (size,)
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
            other_args = jax.vmap(lambda k: get_args(fn_str, k))(jrand.split(k2, size))
            return (x, y, *other_args[2:])

        return nn_mlp_data
    else:

        @partial(jax.jit, static_argnums=(1,))
        def uniform_data(key, size):
            keys = jrand.split(key, 2)
            return jax.vmap(lambda _: get_args(fn_str, keys[0]))(jnp.arange(size))

        return uniform_data


def reward_normalization_fn(reward):
    return symlog(reward)


def inverse_reward_normalization_fn(reward):
    return symexp(reward)


def get_num_clipping_triggers(ratio, eps):
    _ratio = jnp.where(ratio <= 1.0 + eps, ratio, 0.0)
    _ratio = jnp.where(ratio >= 1.0 - eps, 1.0, 0.0)
    return jnp.sum(_ratio)


@partial(jax.vmap, in_axes=(None, 0, 0, 0))
def get_log_probs_and_value(agent, tokens, action, key):
    logits, fmas_value, acc_value = agent(tokens, key=key)
    action_idx = action - 1
    prob_dist = jnn.softmax(logits, axis=-1)
    log_prob = jnp.log(prob_dist[action_idx] + 1e-7)
    return log_prob, prob_dist, fmas_value, acc_value, entropy(prob_dist)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
def get_advantages(reward, done, value, next_value, discount, gae_lambda):
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

    inputs = (reward, done, value, next_value, discount)
    rev_inputs = jax.tree.map(lambda x: x[::-1], inputs)
    init_val = jnp.zeros_like(reward[0])
    _, output = lax.scan(loop_fn, (init_val, init_val), rev_inputs)
    return jax.tree.map(lambda x: x[::-1], output)


@partial(jax.jit, static_argnums=1)
def shuffle_and_batch(trajectories, minibatches, key):
    def _shuffle(x):
        num_envs, rollout_length = x.shape[:2]
        size = num_envs * rollout_length // minibatches
        valid_samples = size * minibatches
        x = x.reshape(-1, *x.shape[2:])
        x = jrand.permutation(key, x, axis=0)[:valid_samples]
        return x.reshape(minibatches, size, *x.shape[1:])

    return jax.tree.map(_shuffle, trajectories)


ENTROPY_WEIGHT = 0.01
VALUE_WEIGHT = 0.5
ACC_WEIGHT = 0.5  # Missing in original broken snippet
NUM_ENVS = os.cpu_count() or 64
LR = 3e-4
GAE_LAMBDA = 0.95
EPS = 0.2
MINIBATCHES = 32
NUM_REWARDS = 2


def train(
    key,
    episodes,
    seq_len,
    env,
    total_v,
    valid_vertices,
    num_valid,
    sp_valid_mask,
    gen,
    target_fn,
    exact_jac,
    eval_device_str,
):
    agent_key, key = jrand.split(key)
    agent = ApproxModel(
        vocab_size=256,
        embd_dim=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        num_actions=3 * total_v,
        policy_dims=[64, 32],
        value_dims=[64, 32],
        seq_len=seq_len,
        key=agent_key,
    )

    def reset_envs():
        def _single_reset(_):
            return env.reset()

        return jax.vmap(_single_reset)(jnp.arange(NUM_ENVS))

    schedule = optax.cosine_decay_schedule(LR, episodes, 0.0)
    optimizer = optax.chain(
        optax.adam(schedule, b1=0.9, eps=1e-7),
        optax.clip_by_global_norm(0.5),
    )
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))

    dyn_agent, static_agent = eqx.partition(agent, eqx.is_array)
    dyn_opt_state, static_opt_state = eqx.partition(opt_state, eqx.is_array)

    key, jac_key = jrand.split(key)
    # Generate batched arguments once (or you can move this into the loop if needed)
    eval_args = gen(jac_key, NUM_ENVS)

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def rollout_fn(current_dyn_agent, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, key):
            active_agent = eqx.combine(current_dyn_agent, static_agent)
            net_key, next_net_key, act_key = jrand.split(key, 3)

            logits, fmas_value, acc_value = active_agent(
                state.tokens, key=net_key, inference=True
            )

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
            env_action = sp_type * seq_len + target_vertex

            env_out = env.step(state, env_action)
            next_state = env_out.state

            _, next_fmas_value, next_acc_value = active_agent(
                next_state.tokens, key=next_net_key, inference=True
            )

            new_sample = Trajectory(
                tokens=state.tokens,
                rewards=env_out.reward,
                terminated=env_out.terminated,
                idx=action_idx + 1,
                fmas_value=fmas_value,
                acc_value=acc_value,
                next_fmas_value=next_fmas_value,
                next_acc_value=next_acc_value,
                prob_dist=prob_dist,
                discount=jnp.array(1.0, dtype=jnp.float32),
            )
            return next_state, new_sample

        return lax.scan(step_fn, env_state, keys)

    def loss_fn(eval_agent, batch: TrainBatch, keys):
        log_probs, prob_dist, fmas_val, acc_val, entropies = get_log_probs_and_value(
            eval_agent, batch.tokens, batch.actions, keys
        )
        old_log_probs = jnp.log(
            jax.vmap(lambda p, a: p[a - 1])(batch.old_prob_dist, batch.actions) + 1e-7
        )
        ratio = jnp.exp(log_probs - old_log_probs)
        num_triggers = get_num_clipping_triggers(ratio, EPS)
        trigger_ratio = num_triggers / len(ratio)

        clipping_objective = jnp.minimum(
            ratio * batch.final_adv,
            jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * batch.final_adv,
        )
        ppo_loss = jnp.mean(-clipping_objective)
        entropy_loss = jnp.mean(entropies)

        fmas_loss = jnp.mean(
            (fmas_val - reward_normalization_fn(batch.fmas_estim_returns)) ** 2
        )
        acc_loss = jnp.mean(
            (acc_val - reward_normalization_fn(batch.acc_estim_returns)) ** 2
        )

        explained_var = explained_variance(
            jnp.sum(batch.fmas_adv), jnp.sum(batch.fmas_estim_returns, axis=-1)
        )
        kl_div = jnp.mean(
            optax.kl_divergence(jnp.log(prob_dist + 1e-7), batch.old_prob_dist)
        )

        total_loss = (
            ppo_loss
            + VALUE_WEIGHT * fmas_loss
            + ACC_WEIGHT * acc_loss
            - ENTROPY_WEIGHT * entropy_loss
        )

        return total_loss, Metrics(
            kl_div,
            entropy_loss,
            explained_var,
            explained_var,
            ppo_loss,
            VALUE_WEIGHT * fmas_loss,
            ACC_WEIGHT * acc_loss,
            total_loss,
            trigger_ratio,
        )

    def host_eval_jacve(orders, sp_types, *eval_args_tuple):
        device = (
            jax.devices(eval_device_str)[0]
            if eval_device_str
            else jax.devices("cpu")[0]
        )

        # Enforce CPU execution for exact/approx graphs if requested to save GPU mem
        with jax.default_device(device):
            cossims = np.zeros(NUM_ENVS, dtype=np.float32)
            mses = np.zeros(NUM_ENVS, dtype=np.float32)

            for i in range(NUM_ENVS):
                o_list = [int(x) for x in orders[i]]
                sp_list = [int(x) for x in sp_types[i]]
                e_args = [a[i] for a in eval_args_tuple]

                exact_out = exact_jac(*e_args)
                jac_exact_dense = jax.tree.map(lambda x: jnp.array(x), exact_out)
                leaves_exact = jax.tree.leaves(jac_exact_dense)

                out_approx = jacve(
                    target_fn,
                    o_list,
                    argnums=env.argnums,
                    sparse_representation=True,
                    sparsity_types=sp_list,
                )(*e_args)
                jac_approx_dense = jax.tree.map(lambda x: jnp.array(x), out_approx)
                leaves_approx = jax.tree.leaves(jac_approx_dense)

                c, m = [], []
                for e, a in zip(leaves_exact, leaves_approx):
                    e_np, a_np = np.array(e), np.array(a)
                    if e_np.shape != a_np.shape:
                        # shape mismatch means it's definitely wrong or compressed
                        c.append(-1.0)
                        m.append(1e3)
                        continue

                    denom = np.maximum(
                        np.linalg.norm(e_np) * np.linalg.norm(a_np), 1e-7
                    )
                    c.append(float(-np.sum(e_np * a_np) / denom))
                    m.append(float(np.mean((e_np - a_np) ** 2)))

                cossims[i] = np.mean(c)
                mses[i] = np.mean(m)

        return cossims, mses

    def host_log(ep, b_ret, m_ret, mtx):
        wandb.log(
            {
                "best_fma_return": float(b_ret),
                "mean_fma_return": float(m_ret),
                "KL divergence": float(mtx.kl_div),
                "ppo loss": float(mtx.ppo_loss),
                "fmas loss": float(mtx.fmas_value_loss),
                "acc loss": float(mtx.acc_value_loss),
                "total loss": float(mtx.total_loss),
            }
        )
        print(
            f"Ep {ep:03d} | Mean FMA: {m_ret:.1f} | Best FMA: {b_ret:.1f} | Loss: {mtx.total_loss:.4f}"
        )

    @eqx.filter_jit
    def episode_step(carry, episode_idx):
        (
            c_dyn_agent,
            c_dyn_opt,
            c_key,
            samplecounts,
            best_global_return,
            best_global_act_seq,
        ) = carry
        subkey, c_key = jrand.split(c_key)
        rollout_key, c_key = jrand.split(c_key)

        env_states = reset_envs()
        env_states, trajectories = rollout_fn(
            c_dyn_agent, num_valid, env_states, jrand.split(rollout_key, NUM_ENVS)
        )

        cossim_errs, mse_errs = jax.pure_callback(
            host_eval_jacve,
            (jax.ShapeDtypeStruct((NUM_ENVS,), jnp.float32),) * 2,
            env_states.order,
            env_states.sparsity_types,
            *eval_args,
        )

        episodic_acc_broadcast = jnp.broadcast_to(
            jnp.expand_dims(cossim_errs, -1), trajectories.rewards[..., 0].shape
        )

        _, fmas_estim, fmas_adv_raw = get_advantages(
            trajectories.rewards[..., 0],
            trajectories.terminated,
            trajectories.fmas_value,
            trajectories.next_fmas_value,
            trajectories.discount,
            GAE_LAMBDA,
        )

        # Contextual bandit style accuracy advantage (mirrors your original intent)
        acc_adv_raw = episodic_acc_broadcast - trajectories.acc_value

        def normalize(adv):
            m = jnp.mean(adv)
            s = jnp.std(adv) + 1e-7
            return (adv - m) / s

        final_adv = normalize(fmas_adv_raw) + normalize(acc_adv_raw)

        train_batch = TrainBatch(
            tokens=trajectories.tokens,
            actions=trajectories.idx,
            old_prob_dist=trajectories.prob_dist,
            fmas_estim_returns=fmas_estim,
            fmas_adv=normalize(fmas_adv_raw),
            acc_estim_returns=episodic_acc_broadcast,
            acc_adv=normalize(acc_adv_raw),
            final_adv=final_adv,
        )

        batches = shuffle_and_batch(train_batch, MINIBATCHES, subkey)

        def scan_train_agent(scan_carry, i):
            scan_dyn_ag, scan_dyn_opt, scan_batches, scan_key = scan_carry
            ag = eqx.combine(scan_dyn_ag, static_agent)
            op = eqx.combine(scan_dyn_opt, static_opt_state)

            train_key, next_key = jrand.split(scan_key)
            minibatch = jax.tree.map(lambda x: x[i], scan_batches)
            keys = jrand.split(train_key, minibatch.tokens.shape[0])

            grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(ag, minibatch, keys)
            updates, new_op = optimizer.update(grads, op, ag)
            new_ag = eqx.apply_updates(ag, updates)

            new_dyn_ag, _ = eqx.partition(new_ag, eqx.is_array)
            new_dyn_op, _ = eqx.partition(new_op, eqx.is_array)

            return (new_dyn_ag, new_dyn_op, scan_batches, next_key), metrics

        (c_dyn_agent, c_dyn_opt, _, c_key), batch_metrics = lax.scan(
            scan_train_agent,
            (c_dyn_agent, c_dyn_opt, batches, c_key),
            jnp.arange(MINIBATCHES),
        )

        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), batch_metrics)
        samplecounts += NUM_ENVS * num_valid

        # Tracking fmas reward (index 0)
        rewards_summed = jnp.sum(trajectories.rewards[..., 0], axis=-1)
        max_idx = jnp.argmax(rewards_summed)
        best_reward = rewards_summed[max_idx]

        new_best_global = jnp.maximum(best_global_return, best_reward)
        new_best_seq = jnp.where(
            best_reward > best_global_return,
            trajectories.idx[max_idx],
            best_global_act_seq,
        )

        jax.debug.callback(
            host_log, episode_idx, best_reward, jnp.mean(rewards_summed), avg_metrics
        )

        return (
            c_dyn_agent,
            c_dyn_opt,
            c_key,
            samplecounts,
            new_best_global,
            new_best_seq,
        ), None

    init_carry = (
        dyn_agent,
        dyn_opt_state,
        key,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(-jnp.inf, dtype=jnp.float32),
        jnp.zeros(num_valid, dtype=jnp.int32),
    )
    final_carry = init_carry
    for ep in tqdm(range(episodes), desc="Episodes"):
        final_carry, _ = episode_step(final_carry, ep)
    _, _, _, _, best_return, best_seq = final_carry
    return best_return, best_seq, total_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ppo-approx")
    parser.add_argument("--seed", type=int, default=250197)
    parser.add_argument("--wandb", type=str, default="disabled")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--example", type=str, default="Helmholtz")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument(
        "--eval-device", type=str, default="cpu", choices=["cpu", "gpu", ""]
    )
    args = parser.parse_args()

    wandb.init(
        project="...",
        name=args.name,
        config=vars(args),
        mode="disabled" if args.wandb == "disabled" else "offline",
    )

    key = jrand.PRNGKey(args.seed)
    key, args_key = jrand.split(key)

    target_fn = get_fn(args.example)
    fn_args = get_args(args.example, args_key)
    gen = data_gen(args.example)

    closed_jaxpr = jax.make_jaxpr(target_fn)(*fn_args)
    total_v = len(closed_jaxpr.jaxpr.eqns)
    argnums = tuple(range(len(fn_args)))

    mode_fmas = {}
    for mode in {"fwd", "rev"}:
        _, aux = vertex_elimination_jaxpr(
            closed_jaxpr.jaxpr,
            mode,
            closed_jaxpr.literals,
            *fn_args,
            argnums=argnums,
            count_ops=True,
            sparse_representation=True,
        )
        # compile this specific exact function to use in callbacks
        
        mode_fmas[mode] = (
            jax.jit(jacve(target_fn, mode, argnums=argnums, sparse_representation=True)),
            aux["fmas"],
        )

    mode, (exact_jac, fmas) = min(mode_fmas.items(), key=lambda x: x[1][1])

    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr,
        args=fn_args,
        num_envs=0,
        seq_len=args.seq_len,
        target_fun=None,  # Pass None to ensure stage=2 does not run evaluation inside the step
    )

    valid_vertices = jnp.array(env.valid_vertices, dtype=jnp.int32)
    num_valid = len(env.valid_vertices)

    print(f"Total vertices: {total_v}, Valid vertices: {num_valid}")

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

    best_global_return, best_global_act_seq, total_v = train(
        key,
        args.episodes,
        args.seq_len,
        env,
        total_v,
        valid_vertices,
        num_valid,
        sp_valid_mask,
        gen,
        target_fn,
        exact_jac,
        args.eval_device,
    )

    best_pairs = [
        (int((i - 1) % total_v) + 1, int((i - 1) // total_v))
        for i in np.array(best_global_act_seq)
    ]
    print(f"\nBest sequence: {best_pairs} | Score: {float(best_global_return)}")


if __name__ == "__main__":
    main()
