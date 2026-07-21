"""Shared policy factory for the vertex-elimination RL trainers.

PPO (``ppo_ray_worker``) and Gumbel-AlphaZero (``az_gumbel``) train the SAME
network — :class:`MicroPPOAgent`: a causal-Palimpsa encoder over the
append-only tokenized jaxpr, a ``PointerVertexPolicy`` over the eliminable
vertices, the autoregressive micro-action heads, and a per-channel value head.
PPO consumes the value head directly; GAZ reuses the vertex logits as its
search prior and the value head as the leaf evaluator — no extra network, only
a different consumer. Building both through one factory keeps the two trainers
on an identical policy with one set of defaults, so they cannot drift.

Encoder defaults to causal ``"palimpsa"``. The Pallas kernel is GPU-only; on a
CPU host the ``palimpsa()`` dispatcher (``transformer.palimpsa_pallas``)
transparently falls back to the verified pure-JAX reference, so
``build_policy(...)`` constructs AND runs off-GPU for smoke tests without
changing GPU numerics.

NOTE: ``MicroPPOAgent`` still lives in ``ppo_ray_worker`` (the class's home,
with the training loop). This factory is the shared entry point external
trainers call so they instantiate the identical class; ``ppo_ray_worker`` may
adopt it once the class is relocated here in a follow-up.
"""
from alphagrad.approx.ppo_ray_worker import MicroPPOAgent, NUM_REWARDS

__all__ = ["build_policy", "MicroPPOAgent", "NUM_REWARDS"]


def build_policy(
    *,
    vocab_size: int,
    embd_dim: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    num_vertices: int,
    value_dims: tuple[int, ...],
    key,
    max_substeps: int = 16,
    policy: str = "palimpsa",
) -> MicroPPOAgent:
    """Construct the shared policy network used by both PPO and Gumbel-AZ.

    Single construction point so the two trainers never diverge on architecture
    or defaults. ``policy`` selects the token-mixer backbone (default causal
    Palimpsa); see the module docstring for the CPU/GPU dispatch.
    """
    return MicroPPOAgent(
        vocab_size=vocab_size,
        embd_dim=embd_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_vertices=num_vertices,
        value_dims=value_dims,
        key=key,
        max_substeps=max_substeps,
        policy=policy,
    )
