import os
import argparse
from functools import partial
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx
import optax
import distrax
from tqdm import tqdm
import wandb

from alphagrad.approx.env import VertexEliminationEnv, MAX_TOKENS
from alphagrad.transformer import MLP, Encoder, PositionalEncoder

class GFNState(eqx.Module):
    logZ: jax.Array

class TransformerGFNAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    policy_head: MLP
    gfn_state: GFNState
    num_actions: int = eqx.field(static=True)

    def __init__(self, vocab_size, embd_dim, num_layers, num_heads, hidden_dim, num_actions, policy_dims, seq_len, key):
        k1, k2, k3 = jrand.split(key, 3)
        self.num_actions = num_actions
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k2)
        self.policy_head = MLP(embd_dim, num_actions, policy_dims, key=k3)
        self.gfn_state = GFNState(logZ=jnp.zeros((1,)))

    def __call__(self, tokens, key=None, inference=False):
        if tokens.ndim == 1:
            mask = (tokens != 0)[..., None]
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            enc_key = key if key is not None else jrand.PRNGKey(0)
            x = self.encoder(x, key=enc_key)
            summary = jnp.sum(x * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1e-9)
            logits = self.policy_head(summary)
            return logits
        else:
            batched_call = jax.vmap(self, in_axes=(0, None, None))
            return batched_call(tokens, key, inference)

class GFNTrajectory(NamedTuple):
    tokens: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    log_pf: jax.Array
    log_pb: jax.Array

@partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
def get_log_probs(agent, tokens, action, available_mask, key):
    logits = agent(tokens, key=key)
    masked_logits = jnp.where(available_mask > 0.5, logits, -1e9)
    prob_dist = jnn.softmax(masked_logits, axis=-1)
    log_prob = jnp.log(prob_dist[action] + 1e-8)
    return log_prob

def trajectory_balance_loss(agent, batch: GFNTrajectory, beta: float):
    # Sum log P_F over the sequence
    sum_log_pf = jnp.sum(batch.log_pf * (1.0 - batch.done), axis=-1)
    
    # Uniform backward policy approximation: P_B(s_{t-1} | s_t) = 1 / t
    # For a graph of N valid vertices, there are t valid parents at step t.
    sum_log_pb = jnp.sum(batch.log_pb * (1.0 - batch.done), axis=-1)
    
    # Terminal reward R(x) = exp(-beta * cost) -> log R(x) = -beta * cost
    # Assuming the environment returns total FMAs as the final reward 
    terminal_fmas = jnp.sum(batch.reward, axis=-1) 
    log_reward = -beta * terminal_fmas
    
    logZ = agent.gfn_state.logZ[0]
    
    loss = jnp.mean((logZ + sum_log_pf - sum_log_pb - log_reward) ** 2)
    return loss, (loss, logZ, jnp.mean(terminal_fmas))

def train_agent(agent, opt_state, optimizer, trajectories, beta):
    grad_fn = eqx.filter_grad(trajectory_balance_loss, has_aux=True)
    grads, metrics = grad_fn(agent, trajectories, beta)
    updates, opt_state = optimizer.update(grads, opt_state, agent)
    agent = eqx.apply_updates(agent, updates)
    return agent, opt_state, metrics

def setup_gfn(env, total_v, args):
    NUM_ACTIONS = env.action_space_size # Set to total_v for vertex, total_v**3 for triplets
    
    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def rollout_fn(agent, rollout_length, env_state, key):
        keys = jrand.split(key, rollout_length)

        def step_fn(state, key):
            net_key, act_key = jrand.split(key, 2)
            logits = agent(state.tokens, key=net_key, inference=True)
            
            # Masking logic adapting to vertex or triplet dimensions
            available_flat = env.get_available_mask(state)
            masked_logits = jnp.where(available_flat > 0.5, logits, -1e9)
            prob_dist = jnn.softmax(masked_logits, axis=-1)

            distribution = distrax.Categorical(probs=prob_dist)
            action_idx = distribution.sample(seed=act_key)
            log_pf = jnp.log(prob_dist[action_idx] + 1e-8)
            
            # Uniform backward probability
            step_count = jnp.maximum(state.step_count, 1.0)
            log_pb = -jnp.log(step_count)

            env_out = env.step(state, action_idx)
            
            transition = GFNTrajectory(
                tokens=state.tokens.astype(jnp.int32),
                action=action_idx,
                reward=jnp.atleast_1d(env_out.reward),
                done=jnp.array(env_out.terminated, dtype=jnp.float32),
                log_pf=log_pf,
                log_pb=log_pb
            )
            return env_out.state, transition

        return lax.scan(step_fn, env_state, keys)
    
    return rollout_fn

# Example Training Loop Integration
def run(args, env, total_v):
    key = jrand.PRNGKey(args.seed)
    agent_key, key = jrand.split(key)
    
    agent = TransformerGFNAgent(
        vocab_size=256, embd_dim=32, num_layers=2, num_heads=2,
        hidden_dim=64, num_actions=env.action_space_size, 
        policy_dims=[64, 32], seq_len=MAX_TOKENS, key=agent_key
    )
    
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))
    
    rollout_fn = setup_gfn(env, total_v, args)
    
    for ep in tqdm(range(args.episodes)):
        rollout_key, train_key, key = jrand.split(key, 3)
        env_states = env.reset()
        
        env_states, traj = rollout_fn(agent, env.rollout_length, env_states, jrand.split(rollout_key, args.num_envs))
        
        agent, opt_state, metrics = train_agent(agent, opt_state, optimizer, traj, args.beta)
        
        loss, logZ, mean_fmas = metrics
        wandb.log({"TB Loss": loss, "logZ": logZ, "Mean FMAs": mean_fmas})
    
# v_k = action_idx % total_v
# v_j = (action_idx // total_v) % total_v
# v_i = action_idx // (total_v ** 2)

class TransformerGFNAgent(eqx.Module):
    embedding: eqx.nn.Embedding
    pos_enc: PositionalEncoder
    encoder: Encoder
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    gfn_state: GFNState

    def __init__(self, vocab_size, embd_dim, num_layers, num_heads, hidden_dim, seq_len, key):
        k1, k2, k3, k4 = jrand.split(key, 4)
        self.embedding = eqx.nn.Embedding(vocab_size, embd_dim, key=k1)
        self.pos_enc = PositionalEncoder(embd_dim, seq_len)
        self.encoder = Encoder(num_layers, num_heads, embd_dim, hidden_dim, key=k2)
        
        # Pointer Network projections
        self.query_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k3)
        self.key_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k4)
        self.gfn_state = GFNState(logZ=jnp.zeros((1,)))

    def __call__(self, tokens, key=None, inference=False):
        if tokens.ndim == 1:
            mask = (tokens != 0)[..., None]
            x = jax.vmap(self.embedding)(tokens)
            x = self.pos_enc(x)
            enc_key = key if key is not None else jrand.PRNGKey(0)
            
            # Encoded sequence (L, D)
            encoded = self.encoder(x, key=enc_key)
            
            # Context vector (summary)
            context = jnp.sum(encoded * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1e-9)
            
            # Pointer Network Logits
            query = self.query_proj(context)            # (D,)
            keys = jax.vmap(self.key_proj)(encoded)     # (L, D)
            
            # Logits over the input sequence length L
            logits = jnp.einsum('ld,d->l', keys, query)
            
            # Zero-out logits for padding tokens
            logits = jnp.where(mask.squeeze(-1), logits, -1e9)
            
            return logits
        else:
            batched_call = jax.vmap(self, in_axes=(0, None, None))
            return batched_call(tokens, key, inference)

# # Instead of one query_proj, project to 3 queries
#         self.query_proj = eqx.nn.Linear(embd_dim, embd_dim * 3, key=k3)
#         self.key_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k4)

#     def __call__(self, tokens, key=None, inference=False):
#         # ... [encoder logic] ...
        
#         # Pointer Network Logits for Triplets
#         queries = self.query_proj(context).reshape(3, -1) # (3, D)
#         keys = jax.vmap(self.key_proj)(encoded)           # (L, D)
        
#         # Logits for v_i, v_j, v_k over the input sequence length L
#         logits_i = jnp.einsum('ld,d->l', keys, queries[0])
#         logits_j = jnp.einsum('ld,d->l', keys, queries[1])
#         logits_k = jnp.einsum('ld,d->l', keys, queries[2])
        
#         # You would then mask and sample v_i, then v_j, then v_k

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand

class TripletPointerNetwork(eqx.Module):
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    num_pointers: int = eqx.field(static=True)

    def __init__(self, embd_dim, num_pointers=3, key=None):
        k1, k2 = jrand.split(key, 2)
        self.num_pointers = num_pointers
        # Project context to all queries at once to maximize GPU utilization
        self.query_proj = eqx.nn.Linear(embd_dim, embd_dim * num_pointers, key=k1)
        self.key_proj = eqx.nn.Linear(embd_dim, embd_dim, key=k2)

    def __call__(self, context, encoded_tokens, mask):
        """
        context: (D,) - The graph summary vector
        encoded_tokens: (L, D) - The transformer encoder outputs
        mask: (L,) - Boolean mask (True = valid vertex token)
        """
        # 1. Generate all 3 queries in a single matmul -> (3, D)
        queries = self.query_proj(context).reshape(self.num_pointers, -1)

        # 2. Generate keys for all sequence tokens -> (L, D)
        keys = jax.vmap(self.key_proj)(encoded_tokens) 

        # 3. Compute Attention Logits efficiently 
        # q = num_pointers (3), d = embd_dim, l = seq_len
        logits = jnp.einsum('qd,ld->ql', queries, keys) # Output: (3, L)

        # 4. Apply masking
        # Expand mask to (1, L) to broadcast across all 3 query distributions
        logits = jnp.where(mask[None, :], logits, -1e9)

        return logits

import jax
import jax.numpy as jnp
import jax.nn as jnn
import distrax

@jax.jit
def sample_valid_triplet(logits, adj_matrix, base_mask, key):
    """
    logits: (3, L) from the TripletPointerNetwork
    adj_matrix: (L, L) boolean/float array where adj[a, b] > 0 if edge a -> b exists
    base_mask: (L,) boolean array of currently valid vertices in the graph
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # 1. Compute Reachability to prevent NaNs
    # valid_paths[a, b] > 0 if there is a path of length 2 from a to b
    valid_paths = jnp.matmul(adj_matrix, adj_matrix) 
    
    # 2. Sample v_i
    # v_i is only valid if it's in the base mask AND has at least one valid 2-hop path
    mask_i = base_mask & (jnp.sum(valid_paths, axis=-1) > 0)
    logits_i = jnp.where(mask_i, logits[0], -1e9)
    v_i = distrax.Categorical(logits=logits_i).sample(seed=k1)

    # 3. Sample v_j
    # v_j must be valid, AND have an incoming edge from the sampled v_i, 
    # AND have at least one outgoing edge to complete the triplet
    mask_j = base_mask & (adj_matrix[v_i] > 0) & (jnp.sum(adj_matrix, axis=-1) > 0)
    logits_j = jnp.where(mask_j, logits[1], -1e9)
    v_j = distrax.Categorical(logits=logits_j).sample(seed=k2)

    # 4. Sample v_k
    # v_k must be valid, AND have an incoming edge from the sampled v_j
    # (Optional: Add `& (jnp.arange(L) != v_i)` to prevent v_i -> v_j -> v_i cycles if needed)
    mask_k = base_mask & (adj_matrix[v_j] > 0)
    logits_k = jnp.where(mask_k, logits[2], -1e9)
    v_k = distrax.Categorical(logits=logits_k).sample(seed=k3)

    # Compute joint log probability for the TB Loss (P_F)
    log_prob_i = jnn.log_softmax(logits_i)[v_i]
    log_prob_j = jnn.log_softmax(logits_j)[v_j]
    log_prob_k = jnn.log_softmax(logits_k)[v_k]
    
    log_pf = log_prob_i + log_prob_j + log_prob_k

    triplet = jnp.array([v_i, v_j, v_k])
    
    return triplet, log_pf