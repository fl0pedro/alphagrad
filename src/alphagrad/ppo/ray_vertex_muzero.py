import os
import argparse
from pathlib import Path
import ray
import jax
import jax.numpy as jnp
import numpy as np
import optax
import mctx
from flax import nnx
from tqdm import tqdm
from typing import Any, Dict, Tuple

from alphagrad.vertexgame.ray_vertex_game import RayVertexGame, MAX_TOKENS
from alphagrad.transformer.gated_deltanet.pallas_model import PallasGatedDeltaNet

def symlog_jnp(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)

def symexp_jnp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

# ---------------------------------------------------------------------------
# MuZero Agent Components (Flax/NNX)
# ---------------------------------------------------------------------------

class Representation(nnx.Module):
    def __init__(self, vocab_size, embd_dim, num_layers, num_heads, hidden_dim, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(vocab_size, embd_dim, rngs=rngs)
        self.pos_enc = nnx.Param(jax.random.normal(rngs.params(), (1, MAX_TOKENS, embd_dim)))
        
        # Similar to PPO agent, we'll use PallasGatedDeltaNet
        self.num_heads = num_heads
        def make_linear(in_dim, out_dim):
            return nnx.Linear(in_dim, out_dim, use_bias=False, rngs=rngs)

        self.gdn_weights = nnx.Dict({
            "q_proj": make_linear(embd_dim, embd_dim),
            "k_proj": make_linear(embd_dim, embd_dim),
            "v_proj": make_linear(embd_dim, embd_dim),
            "b_proj": make_linear(embd_dim, num_heads),
            "a_proj": make_linear(embd_dim, num_heads),
            "g_proj": make_linear(embd_dim, embd_dim),
            "o_proj": make_linear(embd_dim, embd_dim),
            "o_norm": nnx.Param(jnp.ones((embd_dim // num_heads,))),
        })
        self.A_log = nnx.Param(jnp.zeros((num_heads,)))
        self.dt_bias = nnx.Param(jnp.zeros((num_heads,)))

    def __call__(self, tokens):
        x = self.embedding(tokens)
        x = x + self.pos_enc[:, :x.shape[1], :]
        
        gdn = PallasGatedDeltaNet(
            hidden_size=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.gdn_weights["q_proj"].kernel,
            k_proj_weight=self.gdn_weights["k_proj"].kernel,
            v_proj_weight=self.gdn_weights["v_proj"].kernel,
            b_proj_weight=self.gdn_weights["b_proj"].kernel,
            a_proj_weight=self.gdn_weights["a_proj"].kernel,
            g_proj_weight=self.gdn_weights["g_proj"].kernel,
            o_proj_weight=self.gdn_weights["o_proj"].kernel,
            o_norm_weight=self.gdn_weights["o_norm"][...],
            A_log=self.A_log[...],
            dt_bias=self.dt_bias[...],
        )
        out, _ = gdn.forward(x)
        return out[:, -1, :] # (batch, dim)

class Prediction(nnx.Module):
    def __init__(self, embd_dim, p_dims, v_dims, num_actions, rngs: nnx.Rngs):
        p_layers = []
        curr = embd_dim
        for d in p_dims:
            p_layers.append(nnx.Linear(curr, d, rngs=rngs))
            p_layers.append(nnx.relu)
            curr = d
        p_layers.append(nnx.Linear(curr, num_actions, rngs=rngs))
        self.policy = nnx.List(p_layers)

        v_layers = []
        curr = embd_dim
        for d in v_dims:
            v_layers.append(nnx.Linear(curr, d, rngs=rngs))
            v_layers.append(nnx.relu)
            curr = d
        v_layers.append(nnx.Linear(curr, 1, rngs=rngs))
        self.value = nnx.List(v_layers)

    def __call__(self, h):
        logits = h
        for layer in self.policy:
            logits = layer(logits) if isinstance(layer, nnx.Linear) else layer(logits)
        
        val = h
        for layer in self.value:
            val = layer(val) if isinstance(layer, nnx.Linear) else layer(val)
        return logits, val.squeeze(-1)

class Dynamics(nnx.Module):
    def __init__(self, embd_dim, d_dims, num_actions, rngs: nnx.Rngs):
        self.action_emb = nnx.Embed(num_actions + 1, embd_dim, rngs=rngs)
        curr = embd_dim * 2
        d_layers = []
        for d in d_dims:
            d_layers.append(nnx.Linear(curr, d, rngs=rngs))
            d_layers.append(nnx.relu)
            curr = d
        self.shared = nnx.List(d_layers)
        self.next_h_net = nnx.Linear(curr, embd_dim, rngs=rngs)
        self.reward_net = nnx.Linear(curr, 1, rngs=rngs)

    def __call__(self, h, action):
        a_emb = self.action_emb(action)
        x = jnp.concatenate([h, a_emb], axis=-1)
        feat = x
        for layer in self.shared:
            feat = layer(feat) if isinstance(layer, nnx.Linear) else layer(feat)
        return self.next_h_net(feat), self.reward_net(feat).squeeze(-1)

class MuZeroAgent(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.repr_net = Representation(
            config["vocab_size"], config["embd_dim"], config["num_layers"], 
            config["num_heads"], config["hidden_dim"], rngs
        )
        self.pred_net = Prediction(
            config["embd_dim"], config["p_dims"], config["v_dims"], config["num_actions"], rngs
        )
        self.dyna_net = Dynamics(
            config["embd_dim"], config["d_dims"], config["num_actions"], rngs
        )

    def initial_inference(self, obs):
        h = self.repr_net(obs)
        policy_logits, value = self.pred_net(h)
        return h, policy_logits, value

    def recurrent_inference(self, h, action):
        next_h, reward = self.dyna_net(h, action)
        policy_logits, value = self.pred_net(next_h)
        return next_h, reward, policy_logits, value

# ---------------------------------------------------------------------------
# MuZero Learner (GPU Actor)
# ---------------------------------------------------------------------------

@nnx.jit
def _muzero_initial_inference(agent, obs):
    return agent.initial_inference(obs)

@nnx.jit
def _muzero_recurrent_inference(agent, h, action):
    return agent.recurrent_inference(h, action)

@nnx.jit
def _muzero_train_step(agent, optimizer, batch, unroll_steps):
    def loss_fn(agent):
        obs = batch["obs"]
        h, policy_logits, value = agent.initial_inference(obs)
        
        total_loss = 0
        for i in range(unroll_steps):
            p_target = batch["policy_targets"][:, i]
            v_target = batch["value_targets"][:, i]
            
            # Cross entropy for policy
            p_loss = optax.softmax_cross_entropy(policy_logits, p_target).mean()
            v_loss = jnp.mean((value - symlog_jnp(v_target))**2)
            
            total_loss += p_loss + v_loss
            
            if i < unroll_steps - 1:
                a_batch = batch["actions"][:, i]
                r_target = batch["reward_targets"][:, i]
                
                h, reward, policy_logits, value = agent.recurrent_inference(h, a_batch)
                
                r_loss = jnp.mean((reward - symlog_jnp(r_target))**2)
                total_loss += r_loss
        return total_loss

    loss, grads = nnx.value_and_grad(loss_fn)(agent)
    optimizer.update(agent, grads)
    return loss

@ray.remote(num_gpus=1)
class MuZeroLearner:
    def __init__(self, config):
        self.config = config
        self.rngs = nnx.Rngs(0)
        self.agent = MuZeroAgent(config, self.rngs)
        self.optimizer = nnx.Optimizer(self.agent, optax.adam(1e-3), wrt=nnx.Param)

    def initial_inference(self, obs):
        return _muzero_initial_inference(self.agent, obs)

    def recurrent_inference(self, h, action):
        return _muzero_recurrent_inference(self.agent, h, action)

    def train_step(self, batch, unroll_steps):
        return _muzero_train_step(self.agent, self.optimizer, batch, unroll_steps)

    def get_state(self):
        return nnx.state(self.agent)

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, trajectory):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = trajectory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in idxs]

# ---------------------------------------------------------------------------
# Rollout Worker
# ---------------------------------------------------------------------------

@ray.remote
def get_env_info(config):
    env = RayVertexGame(config)
    return env.total_v

@ray.remote(num_cpus=1)
class MuZeroWorker:
    def __init__(self, config):
        self.config = config
        self.game = RayVertexGame(config["env_config"])
        self.num_actions = config["num_actions"]
        self.rngs = nnx.Rngs(0)
        self.agent = MuZeroAgent(config["agent_config"], self.rngs)

    def set_state(self, state):
        nnx.update(self.agent, state)

    def self_play(self):
        @nnx.jit
        def initial_inference(obs):
            return self.agent.initial_inference(obs)

        @nnx.jit
        def recurrent_inference(h, action):
            return self.agent.recurrent_inference(h, action)

        def rec_inf(params, key, action, h):
            # h: (B, dim), action: (B,)
            a_jnp = action.astype(jnp.int32) + 1 
            next_h, reward, policy, value = recurrent_inference(h, a_jnp)
            return mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.ones(h.shape[0]) * 0.99,
                prior_logits=policy,
                value=value
            ), next_h

        trajectory = []
        obs = self.game.reset()
        done = False
        
        obs_jnp = obs[None, :]
        h, policy_logits, value = initial_inference(obs_jnp)
        
        root = mctx.RootFnOutput(
            prior_logits=policy_logits,
            value=value,
            embedding=h
        )

        while not done:
            key = jax.random.PRNGKey(np.random.randint(0, 1000000))
            available = np.array(self.game.available[...])
            invalid_actions = jnp.where(jnp.array(available) > 0.5, False, True)[None, :]
            
            policy_output = mctx.gumbel_muzero_policy(
                params=None,
                rng_key=key,
                root=root,
                recurrent_fn=rec_inf,
                num_simulations=self.config.get("num_simulations", 20),
                invalid_actions=invalid_actions,
                max_num_considered_actions=self.num_actions
            )
            
            action_idx = int(policy_output.action[0])
            policy_target = np.array(policy_output.action_weights[0])
            next_obs, reward, done = self.game.step(jnp.array(action_idx + 1))
            
            trajectory.append((np.array(obs), action_idx, float(reward), float(policy_output.root_value[0]), policy_target))
            obs = next_obs
            
            if not done:
                obs_jnp = obs[None, :]
                h, policy_logits, value = initial_inference(obs_jnp)
                root = mctx.RootFnOutput(
                    prior_logits=policy_logits,
                    value=value,
                    embedding=h
                )
                
        return trajectory

def find_workspace_root():
    curr = Path(__file__).resolve().parent
    for parent in [curr] + list(curr.parents):
        if (parent / "graphax").exists() and (parent / "alphagrad").exists():
            return parent
    return Path(__file__).resolve().parents[4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--unroll-steps", type=int, default=3)
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=0)
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 4
    root = find_workspace_root()
    
    if not ray.is_initialized():
        ray.init(
            num_cpus=8 if args.local_test else cpu_count, 
            num_gpus=args.num_gpus,
            runtime_env={
                "working_dir": str(root),
                "env_vars": {"PYTHONPATH": f"{root}/alphagrad/src:{root}/graphax/src"},
            }
        )

    target_fn_name = "Helmholtz"
    xs_np = [np.random.uniform(0, 0.1, (4,)).astype(np.float32)]
    env_config = {"target_fn": target_fn_name, "args": xs_np, "sparse": False}
    total_v = ray.get(get_env_info.remote(env_config))
    
    agent_config = {
        "vocab_size": 256,
        "embd_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "hidden_dim": 128,
        "num_actions": total_v,
        "p_dims": [128, 64],
        "v_dims": [64],
        "d_dims": [128, 64],
    }

    learner = MuZeroLearner.remote(agent_config)
    buffer = ReplayBuffer(capacity=500)
    num_workers = 4 if args.local_test else max(1, cpu_count - 1)
    
    worker_config = {
        "env_config": env_config,
        "num_actions": total_v,
        "num_simulations": args.simulations,
    }
    workers = [MuZeroWorker.remote(worker_config) for _ in range(num_workers)]

    pbar = tqdm(range(args.episodes))
    for ep in pbar:
        # Sync weights to workers
        state = ray.get(learner.get_state.remote())
        ray.get([w.set_state.remote(state) for w in workers])

        trajectories = ray.get([w.self_play.remote() for w in workers])
        for traj in trajectories:
            buffer.push(traj)
            
        if len(buffer.buffer) >= args.batch_size:
            batch_trajectories = buffer.sample(args.batch_size)
            unroll_steps = args.unroll_steps
            
            # Prepare batch for JAX
            obs, actions, rewards, values, policies = [], [], [], [], []
            for traj in batch_trajectories:
                if len(traj) < unroll_steps:
                    traj = traj + [traj[-1]] * (unroll_steps - len(traj))
                start = np.random.randint(0, len(traj) - unroll_steps + 1)
                segment = traj[start : start + unroll_steps]
                
                obs.append(segment[0][0])
                actions.append([s[1] for s in segment])
                rewards.append([s[2] for s in segment])
                values.append([s[3] for s in segment])
                policies.append([s[4] for s in segment])
            
            batch = {
                "obs": jnp.array(obs),
                "actions": jnp.array(actions),
                "reward_targets": jnp.array(rewards),
                "value_targets": jnp.array(values),
                "policy_targets": jnp.array(policies)
            }
            
            loss = ray.get(learner.train_step.remote(batch, unroll_steps))
            pbar.set_description(f"Loss: {loss:.4f}")

    ray.shutdown()

if __name__ == "__main__":
    main()
