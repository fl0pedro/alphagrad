import argparse
import os
from pathlib import Path
import ray
import numpy as np
from tqdm import tqdm

def symlog_np(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

def symexp_np(x):
    return np.sign(x) * np.exp(np.abs(x) - 1)

def get_advantages(rewards, dones, values, next_values, discounts, gae_lambda):
    batch_size, T = rewards.shape
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(T)):
        mask = 1.0 - dones[:, t]
        
        value_raw = symexp_np(values[:, t])
        next_value_raw = symexp_np(next_values[:, t])
        
        delta = rewards[:, t] + next_value_raw * discounts[:, t] * mask - value_raw
        advantage = delta + discounts[:, t] * gae_lambda * lastgaelam * mask
        
        lastgaelam = advantage
        advantages[:, t] = advantage
        returns[:, t] = advantage + value_raw
        
    return advantages, returns

def build_ppo_agent():
    import torch
    import torch.nn as nn
    from fla.layers import GatedDeltaNet
    
    class GatedDeltaNetPPOAgent(nn.Module):
        def __init__(self, vocab_size, embd_dim, num_layers, num_heads,
                     hidden_dim, num_actions, policy_dims, value_dims):
            super().__init__()
            MAX_TOKENS=1024
            self.num_actions = num_actions
            self.embedding = nn.Embedding(vocab_size, embd_dim)
            self.pos_enc = nn.Parameter(torch.randn(1, MAX_TOKENS, embd_dim))
            
            self.gdn = GatedDeltaNet(
                hidden_size=embd_dim,
                num_heads=num_heads,
                head_dim=embd_dim // num_heads,
            )
            
            p_layers = []
            curr = embd_dim
            for d in policy_dims:
                p_layers.extend([nn.Linear(curr, d), nn.ReLU()])
                curr = d
            p_layers.append(nn.Linear(curr, num_actions))
            self.policy_head = nn.Sequential(*p_layers)

            v_layers = []
            curr = embd_dim
            for d in value_dims:
                v_layers.extend([nn.Linear(curr, d), nn.ReLU()])
                curr = d
            v_layers.append(nn.Linear(curr, 1))
            self.value_head = nn.Sequential(*v_layers)

        def forward(self, tokens):
            x = self.embedding(tokens)
            x = x + self.pos_enc[:, :x.size(1), :]
            
            out = self.gdn(x)[0]
            summary = out.mean(dim=1)
            
            logits = self.policy_head(summary)
            value = self.value_head(summary)
            return logits, value.squeeze(-1)
            
    return GatedDeltaNetPPOAgent

@ray.remote(num_gpus=0, runtime_env={"py_executable": "/Users/assmuth/dsnn/.ray_cpu_venv/bin/python", "env_vars": {"JAX_PLATFORMS":"cpu"}})
def get_env_info(config):
    from alphagrad.vertexgame.ray_vertex_game import RayVertexGame
    env = RayVertexGame(config)
    return env.total_v

@ray.remote
def sample_max_tokens(config):
    from graphax.core import extract_jaxpr
    ...
    max_tokens = ...
    return max_tokens
    


@ray.remote(runtime_env={"py_executable": "/Users/assmuth/dsnn/.ray_gpu_venv/bin/python"})
class PPOLearner:
    def __init__(self, config):
        import torch
        import torch.nn.functional as F
        self.torch = torch
        self.F = F
        
        import warnings
        warnings.filterwarnings("ignore", message=".*Triton is not supported.*")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        GatedDeltaNetPPOAgent = build_ppo_agent()
        self.agent = GatedDeltaNetPPOAgent(
            config["vocab_size"], config["embd_dim"], config["num_layers"], 
            config["num_heads"], config["hidden_dim"], config["num_actions"],
            config["policy_dims"], config["value_dims"]
        ).to(self.device).bfloat16()
        
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config["lr"])
        self.eps = config["eps"]
        self.value_weight = config["value_weight"]
        self.entropy_weight = config["entropy_weight"]

    def symlog_torch(self, x):
        return self.torch.sign(x) * self.torch.log(self.torch.abs(x) + 1)

    def get_status(self):
        return {
            "PID": os.getpid(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def train_step(self, trajectories_np):
        tokens = self.torch.from_numpy(trajectories_np["tokens"]).to(self.device).long()
        actions = self.torch.from_numpy(trajectories_np["actions"]).to(self.device).long()
        old_log_probs = self.torch.from_numpy(trajectories_np["log_probs"]).to(self.device).to(self.torch.bfloat16)
        advantages = self.torch.from_numpy(trajectories_np["advantages"]).to(self.device).to(self.torch.bfloat16)
        returns = self.torch.from_numpy(trajectories_np["returns"]).to(self.device).to(self.torch.bfloat16)
        
        logits, values = self.agent(tokens)
        
        log_probs = self.F.log_softmax(logits, dim=-1)
        curr_log_probs = log_probs.gather(1, (actions - 1).unsqueeze(-1)).squeeze(-1)
        
        ratio = self.torch.exp(curr_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = self.torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages
        ppo_loss = -self.torch.min(surr1, surr2).mean()
        
        value_targets = self.symlog_torch(returns)
        value_loss = self.F.mse_loss(values, value_targets)
        
        probs = self.F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        loss = ppo_loss + self.value_weight * value_loss - self.entropy_weight * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "ppo_loss": ppo_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }

    def weights(self):
        return {k: v.detach().cpu().to(self.torch.float32).numpy() for k, v in self.agent.state_dict().items()}

@ray.remote(num_cpus=1, runtime_env={"py_executable": "/Users/assmuth/dsnn/.ray_cpu_venv/bin/python", "env_vars": {"JAX_PLATFORMS":"cpu"}})
class RolloutWorker:
    def __init__(self, config):
        from alphagrad.vertexgame.ray_vertex_game import RayVertexGame
        self.env = RayVertexGame(config["env_config"])
        self.total_v = config["num_actions"]

    def sample(self, rollout_length, learner):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from alphagrad.transformer.gated_deltanet.model import GatedDeltaNet, GatedDeltaNetConfig
        from flax import nnx

        seed = np.random.randint(0, 100000)
        rngs = nnx.Rngs(seed) # using the seed you already generated
        
        rng = jax.random.PRNGKey(seed)
        state = self.env.reset()
        weights = ray.get(learner.weights.remote())

        gdn_config = GatedDeltaNetConfig()
        gdn = GatedDeltaNet(gdn_config, rngs=rngs)

        _, state = nnx.split(gdn)
        for pt_key, pt_tensor in weights.items():
            pt_tensor = jnp.array(pt_tensor, dtype=jnp.bfloat16)
            if "q_proj.weight" in pt_key:
                state.q_proj.kernel = pt_tensor.T
            elif "k_proj.weight" in pt_key:
                state.k_proj.kernel = pt_tensor.T
            elif "v_proj.weight" in pt_key:
                state.v_proj.kernel = pt_tensor.T
            elif "o_proj.weight" in pt_key:
                state.o_proj.kernel = pt_tensor.T
            elif "a_proj.weight" in pt_key:
                state.a_proj.kernel = pt_tensor.T
            elif "b_proj.weight" in pt_key:
                state.b_proj.kernel = pt_tensor.T
            elif "g_proj.weight" in pt_key:
                state.g_proj.kernel = pt_tensor.T
            elif "q_conv.weight" in pt_key or "q_conv.conv.weight" in pt_key:
                state.q_conv.conv.kernel = pt_tensor.transpose(2, 1, 0)
            elif "k_conv.weight" in pt_key or "k_conv.conv.weight" in pt_key:
                state.k_conv.conv.kernel = pt_tensor.transpose(2, 1, 0)
            elif "v_conv.weight" in pt_key or "v_conv.conv.weight" in pt_key:
                state.v_conv.conv.kernel = pt_tensor.transpose(2, 1, 0)
            elif "o_norm.weight" in pt_key:
                state.o_norm.weight = pt_tensor
            elif "A_log" in pt_key:
                state.A_log = pt_tensor
            elif "dt_bias" in pt_key:
                state.dt_bias = pt_tensor
            else:
                print(pt_key)

        nnx.update(gdn, state)

        def _step_fn(carry, _):
            state, key = carry
            
            logits, value = gdn(state.tokens)            
            available = jnp.zeros(self.total_v)
            valid_arr = jnp.array(self.env.valid_vertices, dtype=jnp.int32)
            
            def check_valid(i, avail):
                v = valid_arr[i]
                pos = jnp.argwhere(state.order == v, size=1).squeeze()
                is_used = pos < state.step_count
                return jax.lax.cond(is_used, lambda x: x, lambda x: x.at[v-1].set(1.0), avail)
            
            available = jax.lax.fori_loop(0, self.env.num_valid, check_valid, available)
            masked_logits = jnp.where(available > 0.5, logits, -1e9)
            probs = jax.nn.softmax(masked_logits)
            
            key, action_key = jax.random.split(key)
            action_idx = jax.random.categorical(action_key, masked_logits)
            action = action_idx + 1
            log_prob = jnp.log(probs[action_idx] + 1e-7)
            
            out = self.env.step(state, action)
            next_state, reward, done = out.state, out.reward, out.terminated
            
            next_state = jax.lax.cond(done, lambda _: self.env.reset(), lambda _: next_state, None)
            
            traj = {
                "obs": state.tokens,
                "action": action,
                "reward": reward,
                "done": done,
                "value": value,
                "log_prob": log_prob,
                "discount": jnp.array(0.99, dtype=jnp.float32)
            }
            return (next_state, key), traj


        state = self.env.reset()
        _jitted_scan = jax.jit(lambda s, k: jax.lax.scan(_step_fn, (s, k), None, length=rollout_length))
        (final_state, final_rng), trajs_jnp = _jitted_scan(state, rng)
        
        trajs = []
        for i in range(rollout_length):
            trajs.append({
                "obs": np.asarray(trajs_jnp["obs"][i]),
                "action": int(trajs_jnp["action"][i]),
                "reward": float(trajs_jnp["reward"][i]),
                "done": bool(trajs_jnp["done"][i]),
                "value": float(trajs_jnp["value"][i]),
                "log_prob": float(trajs_jnp["log_prob"][i]),
                "discount": float(trajs_jnp["discount"][i]),
            })
            
        return trajs

def find_workspace_root():
    curr = Path(__file__).resolve().parent
    for parent in [curr] + list(curr.parents):
        if (parent / "graphax").exists() and (parent / "alphagrad").exists():
            return parent
    return Path(__file__).resolve().parents[4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    cpu_count = os.cpu_count() or 4
    root = find_workspace_root()

    if not ray.is_initialized():
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        new_paths = "alphagrad/src:graphax/src"
        merged_pythonpath = f"{new_paths}:{current_pythonpath}" if current_pythonpath else new_paths

        env_vars = {
            "PYTHONPATH": merged_pythonpath,
        }
        if os.environ.get("UV_CACHE_DIR"):
            env_vars["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]

        ray_kwargs = {
            "num_cpus": 8 if args.local_test else cpu_count,
            "runtime_env": {
                "working_dir": str(root),
                "env_vars": env_vars,
                "py_executable": "/Users/assmuth/dsnn/.ray_venv/bin/python",
                "excludes": [
                    ".ray_*",
                    "python3.13", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
                    "**/pyproject.toml", "**/uv.lock",
                    "dann", "snnax", "synaptax", "torchneuromorphic", "graphax-og", "graphax_bak",
                    "wandb", "dist", "build", "target", "eval", "docs", "tests", "~",
                    "*.zip", "*.lp", "*.mps", "*.log", "**/.*_cache"
                ]
            }
        }
        if args.num_gpus is not None:
            ray_kwargs["num_gpus"] = args.num_gpus
            
        ray.init(**ray_kwargs)

    # target_fn_name = "g"
    # xs_np = [np.random.uniform(0, 1, (1,)).astype(np.float32) for _ in range(15)]
    target_fn_name = "Helmholtz"
    xs_np = [np.random.uniform(0, 1, (4,)).astype(np.float32)]
    
    env_config = {
        "target_fn": target_fn_name,
        "args": xs_np,
        "sparse": False,
    }
    
    total_v = ray.get(get_env_info.remote(env_config))
    
    agent_config = {
        "vocab_size": 8192,
        "embd_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "hidden_dim": 128,
        "num_actions": total_v,
        "policy_dims": [128, 64],
        "value_dims": [64],
        "lr": 3e-4,
        "eps": 0.2,
        "value_weight": 0.5,
        "entropy_weight": 0.01,
        "env_config": env_config
    }

    num_gpus = 0 if args.local_test else 1
    learner = PPOLearner.options(num_gpus=num_gpus).remote(agent_config)
    num_workers = 4 if args.local_test else max(1, cpu_count - 1)
    workers = [RolloutWorker.remote(agent_config) for _ in range(num_workers)]

    pbar = tqdm(range(args.episodes))
    rollout_length = 20
    
    for ep in pbar:
        if ep == 0:
            status = ray.get(learner.get_status.remote())
            print(f"DEBUG: Learner Status: {status}")
        futures = [w.sample.remote(rollout_length, learner) for w in workers]
        results = ray.get(futures)
        all_trajs = [item for sublist in results for item in sublist]
        obs_batch = np.array([t["obs"] for t in all_trajs])
        actions = np.array([t["action"] for t in all_trajs])
        rewards = np.array([t["reward"] for t in all_trajs]).reshape(num_workers, rollout_length)
        dones = np.array([t["done"] for t in all_trajs]).reshape(num_workers, rollout_length)
        values = np.array([t["value"] for t in all_trajs]).reshape(num_workers, rollout_length)
        log_probs = np.array([t["log_prob"] for t in all_trajs])
        discounts = np.array([t["discount"] for t in all_trajs]).reshape(num_workers, rollout_length)
        
        next_values = np.zeros_like(values)
        next_values[:, :-1] = values[:, 1:]
        
        advs, rets = get_advantages(rewards, dones, values, next_values, discounts, 0.95)
        
        train_data = {
            "tokens": obs_batch,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advs.flatten(),
            "returns": rets.flatten()
        }
        metrics = ray.get(learner.train_step.remote(train_data))
        pbar.set_description(f"Loss: {metrics['loss']:.4f} PPO: {metrics['ppo_loss']:.4f}")

    print("PPO training finished!")
    ray.shutdown()

if __name__ == "__main__":
    main()