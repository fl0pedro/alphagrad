import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ppo_ray_worker.SimplePPOAgent was removed (the Ray PPO line is retired). "
    "Kept for provenance; delete once the subsystem is confirmed gone for good.",
    allow_module_level=True,
)

"""Smoke test: ``SimplePPOAgent`` builds with the 4-way op-type head
plus ``quant_dtype_head`` after Phase 6's QUANT propagation.

Verifies the JAX-side shape contract without running a full rollout:
the env build / Ray spawn is too heavy for a unit test. The
end-to-end smoke lives in the sbatch scripts (``--episodes 2``).
"""

from __future__ import annotations

import jax
import jax.random as jrand
import pytest


def test_simple_ppo_agent_has_quant_dtype_head():
    """4-way op-type head + non-None quant_dtype_head when dynamic_substeps is on."""
    from alphagrad.approx.ppo_ray_worker import SimplePPOAgent
    from graphax.sparse.micro_actions import NUM_QUANT_DTYPES

    agent = SimplePPOAgent(
        vocab_size=64,
        embd_dim=16,
        num_layers=1,
        num_heads=1,
        hidden_dim=16,
        num_vertices=8,
        policy_dims=(16, 16),
        value_dims=(16, 16),
        key=jrand.PRNGKey(0),
        dynamic_substeps=True,
        num_factors=4,
    )
    assert agent.quant_dtype_head is not None
    assert agent.num_quant_dtypes == NUM_QUANT_DTYPES
    # The op-type head produces 4 logits — DIAG / COMPRESS / QUANT / END.
    # We verify by running a forward pass on a dummy context vector
    # and inspecting the output dimension.
    import jax.numpy as jnp
    ctx = jnp.zeros((16,), dtype=jnp.float32)  # embd_dim=16
    op_logits = agent.op_type_head(ctx)
    assert op_logits.shape == (4,)
    q_logits = agent.quant_dtype_head(ctx)
    assert q_logits.shape == (NUM_QUANT_DTYPES,)


def test_simple_ppo_agent_static_branch_no_quant_head():
    """When dynamic_substeps is off, micro heads (including quant) stay None."""
    from alphagrad.approx.ppo_ray_worker import SimplePPOAgent

    agent = SimplePPOAgent(
        vocab_size=64,
        embd_dim=16,
        num_layers=1,
        num_heads=1,
        hidden_dim=16,
        num_vertices=8,
        policy_dims=(16, 16),
        value_dims=(16, 16),
        key=jrand.PRNGKey(0),
        dynamic_substeps=False,
        num_factors=4,
    )
    assert agent.quant_dtype_head is None
    assert agent.op_type_head is None


def test_all_logits_returns_seven_tuple():
    """all_logits' tuple shape grew from 6 to 7 to accommodate q_logits.
    Downstream act_step / loss_fn destructure on this shape."""
    from alphagrad.approx.env import MAX_TOKENS
    from alphagrad.approx.ppo_ray_worker import SimplePPOAgent
    import jax.numpy as jnp

    agent = SimplePPOAgent(
        vocab_size=64,
        embd_dim=16,
        num_layers=1,
        num_heads=1,
        hidden_dim=16,
        num_vertices=8,
        policy_dims=(16, 16),
        value_dims=(16, 16),
        key=jrand.PRNGKey(0),
        dynamic_substeps=True,
        num_factors=4,
    )
    tokens = jnp.ones((MAX_TOKENS,), dtype=jnp.int32)
    out = agent.all_logits(tokens, key=jrand.PRNGKey(1))
    assert len(out) == 7
    # (vertex_logits, value, op_l, i_l, j_l, f_l, q_l)
    vertex_logits, value, op_l, i_l, j_l, f_l, q_l = out
    assert op_l.shape == (4,)
    assert q_l.shape == (8,) or q_l.shape[0] > 0  # NUM_QUANT_DTYPES > 0


def test_micro_actions_to_rule_specs_accepts_quant_dtypes():
    """The env-side translator gained a ``quant_dtypes`` kwarg; the
    PPO act_step now passes it on every emit. Verify the contract."""
    from alphagrad.approx.env import micro_actions_to_rule_specs_jax
    from alphagrad.approx.heads import OP_QUANT
    import jax.numpy as jnp

    # Single-step QUANT spec. Quant doesn't use i/j/factor — those
    # slots become no-ops.
    op_types = jnp.array([OP_QUANT], dtype=jnp.int32)
    i_indices = jnp.array([0], dtype=jnp.int32)
    j_indices = jnp.array([0], dtype=jnp.int32)
    factors = jnp.array([0], dtype=jnp.int32)
    quant_dtypes = jnp.array([2], dtype=jnp.int32)  # arbitrary dtype index
    # Minimal axis_state stub: (MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)
    from alphagrad.approx.env import MAX_AXES_PER_VERTEX
    axis_state = jnp.zeros((MAX_AXES_PER_VERTEX, 16), dtype=jnp.int32)

    specs = micro_actions_to_rule_specs_jax(
        op_types=op_types,
        i_indices=i_indices,
        j_indices=j_indices,
        factors=factors,
        axis_state_for_vertex=axis_state,
        quant_dtypes=quant_dtypes,
    )
    # specs is (MAX_RULES_PER_VERTEX, 3) int32. First row should be the
    # QUANT spec: [QUANT_SENTINEL, dtype_idx, 0].
    from alphagrad.approx.env import QUANT_SENTINEL
    assert int(specs[0, 0]) == QUANT_SENTINEL
    assert int(specs[0, 1]) == 2  # quant_dtypes[0]
    assert int(specs[0, 2]) == 0
