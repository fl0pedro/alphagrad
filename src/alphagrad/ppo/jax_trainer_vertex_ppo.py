import argparse
import os

import ray
ray.init(
    runtime_env={
        "env_vars": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        }
    }
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from ray.train import ScalingConfig, RunConfig
from ray.train.v2.jax import JaxTrainer

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from alphagrad.vertexgame.ray_vertex_game import RayVertexGame, MAX_TOKENS
from alphagrad.transformer.gated_deltanet.model import GatedDeltaNet
from alphagrad.transformer.gated_deltanet.configuration import GatedDeltaNetConfig


class GatedDeltaNetPPOAgent(nnx.Module):
    def __init__(self, vocab_size, embd_dim, num_actions, config: GatedDeltaNetConfig, policy_dims, value_dims, rngs: nnx.Rngs):
        self.num_actions = num_actions
        self.embedding = nnx.Embed(vocab_size, embd_dim, rngs=rngs)
        self.pos_enc = nnx.Param(jax.random.normal(rngs.params(), (1, MAX_TOKENS, embd_dim)))
        
        self.gdn = GatedDeltaNet(config, rngs=rngs)
        
        p_layers = []
        curr = config.hidden_size
        for d in policy_dims:
            p_layers.extend([nnx.Linear(curr, d, rngs=rngs), nnx.relu])
            curr = d
        p_layers.append(nnx.Linear(curr, num_actions, rngs=rngs))
        self.policy_head = nnx.Sequential(*p_layers)

        v_layers = []
        curr = config.hidden_size
        for d in value_dims:
            v_layers.extend([nnx.Linear(curr, d, rngs=rngs), nnx.relu])
            curr = d
        v_layers.append(nnx.Linear(curr, 1, rngs=rngs))
        self.value_head = nnx.Sequential(*v_layers)

    def __call__(self, tokens):
        x = self.embedding(tokens)
        x = x + self.pos_enc.value[:, :x.shape[1], :]
        
        out = self.gdn(x)
        summary = jnp.mean(out, axis=1)
        
        logits = self.policy_head(summary)
        value = self.value_head(summary)
        return logits, jnp.squeeze(value, -1)


def get_advantages(rewards, dones, values, next_values, discounts, gae_lambda):
    def loop_fn(carry, traj):
        episodic_return, lastgaelam = carry
        reward, done, value, next_value, discount = traj
        
        mask = 1.0 - done
        delta = reward + next_value * discount * mask - value
        advantage = delta + discount * gae_lambda * lastgaelam * mask
        return (reward + discount * episodic_return * mask, advantage), advantage

    inputs = jnp.stack([rewards, dones, values, next_values, discounts], axis=1)
    _, advantages = jax.lax.scan(loop_fn, (0.0, 0.0), inputs[::-1])
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def train_loop_per_worker(config):
    """
    SPMD JAX training loop executed on every Ray worker.
    """
    # Environment Setup
    env = RayVertexGame(config["env_config"])
    total_v = env.total_v
    rollout_length = config["rollout_length"]

    # Model Initialization
    rngs = nnx.Rngs(params=jax.random.PRNGKey(jax.process_index() + config.get("seed", 42)))
    gdn_config = GatedDeltaNetConfig(
        hidden_size=config["hidden_dim"],
        head_dim=config["embd_dim"] // config["num_heads"],
        num_heads=config["num_heads"],
        vocab_size=config["vocab_size"],
    )
    
    agent = GatedDeltaNetPPOAgent(
        config["vocab_size"], config["embd_dim"], total_v,
        gdn_config, config["policy_dims"], config["value_dims"], rngs=rngs
    )
    
    # Optimizer Setup
    optimizer = nnx.Optimizer(agent, optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=config["lr"])
    ))

    @nnx.jit
    def train_step(agent, optimizer, batch):
        def loss_fn(agent):
            logits, values = jax.vmap(agent)(batch["tokens"])
            
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            curr_log_probs = jnp.take_along_axis(log_probs, batch["actions"][..., None] - 1, axis=-1).squeeze(-1)
            
            ratio = jnp.exp(curr_log_probs - batch["log_probs"])
            surr1 = ratio * batch["advantages"]
            surr2 = jnp.clip(ratio, 1.0 - config["eps"], 1.0 + config["eps"]) * batch["advantages"]
            ppo_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            
            value_loss = jnp.mean(jnp.square(values - batch["returns"]))
            
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
            
            total_loss = ppo_loss + config["value_weight"] * value_loss - config["entropy_weight"] * entropy
            return total_loss, {"ppo_loss": ppo_loss, "value_loss": value_loss, "entropy": entropy}
        
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(agent)
        
        # Cross-device gradient aggregation for distributed training
        grads = jax.lax.pmean(grads, axis_name="batch") if jax.device_count() > 1 else grads
        
        optimizer.update(grads)
        metrics["loss"] = loss
        return metrics

    # Distributed Training Loop
    state = env.reset()
    key = rngs.params()

    for ep in range(config["episodes"]):
        trajectories = []
        
        # 1. Rollout Phase
        for _ in range(rollout_length):
            logits, value = agent(state.tokens)
            probs = jax.nn.softmax(logits)
            
            key, action_key = jax.random.split(key)
            action_idx = jax.random.categorical(action_key, logits)
            action = action_idx + 1
            log_prob = jnp.log(probs[action_idx] + 1e-7)
            
            out = env.step(state, action)
            next_state, reward, done = out.state, out.reward, out.terminated
            
            trajectories.append({
                "obs": state.tokens,
                "action": action,
                "reward": reward,
                "done": done,
                "value": value,
                "log_prob": log_prob,
                "discount": jnp.array(0.99, dtype=jnp.float32)
            })
            
            state = jax.lax.cond(done, lambda _: env.reset(), lambda _: next_state, None)

        # 2. Advantage Calculation
        obs_batch = jnp.stack([t["obs"] for t in trajectories])
        actions = jnp.array([t["action"] for t in trajectories])
        rewards = jnp.array([t["reward"] for t in trajectories])
        dones = jnp.array([t["done"] for t in trajectories])
        values = jnp.array([t["value"] for t in trajectories])
        log_probs = jnp.array([t["log_prob"] for t in trajectories])
        discounts = jnp.array([t["discount"] for t in trajectories])

        _, next_value = agent(state.tokens)
        next_values = jnp.append(values[1:], next_value)

        advs, rets = get_advantages(rewards, dones, values, next_values, discounts, 0.95)

        train_data = {
            "tokens": obs_batch,
            "actions": actions,
            "log_probs": log_probs,
            "advantages": advs,
            "returns": rets
        }

        # 3. Optimization Phase
        metrics = train_step(agent, optimizer, train_data)

        # 4. Reporting
        ray.train.report({
            "episode": ep,
            "loss": float(metrics["loss"]),
            "ppo_loss": float(metrics["ppo_loss"]),
            "value_loss": float(metrics["value_loss"]),
            "entropy": float(metrics["entropy"]),
            "mean_reward": float(jnp.mean(rewards))
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)

    import graphax.examples as examples
    target_fn_name = "Helmholtz"
    xs_np = [jnp.array(jax.random.uniform(key, (4,)).astype(jnp.float32))]

    env_config = {
        "target_fn": target_fn_name,
        "args": xs_np,
        "sparse": False,
    }

    config = {
        "vocab_size": 256,
        "embd_dim": 64,
        "num_layers": 2,
        "num_heads": 2,
        "hidden_dim": 128,
        "policy_dims": [128, 64],
        "value_dims": [64],
        "lr": 3e-4,
        "eps": 0.2,
        "value_weight": 0.5,
        "entropy_weight": 0.01,
        "env_config": env_config,
        "episodes": args.episodes,
        "rollout_length": 20,
        "seed": 42
    }

    scaling_config = ScalingConfig(
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        resources_per_worker={"GPU": 1} if args.use_gpu else {"CPU": 1}
    )

    trainer = JaxTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=RunConfig(name="gdn_ppo_jax_trainer")
    )
    
    result = trainer.fit()
    print(f"Training finished. Final metrics: {result.metrics}")

if __name__ == "__main__":
    main()