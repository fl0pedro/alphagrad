"""Tests for the shared utilities in `alphagrad.approx.common` and the env's
backwards-compatible legacy action conversion."""

from __future__ import annotations


import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    NUM_AXIS_PAIRS,
    StepAction,
    _legacy_sp_to_specs,
    sp_type_to_map,
)
from alphagrad.approx.common import (
    build_legacy_sp_valid_mask,
    build_pair_factor_valid_mask,
    build_pair_valid_mask,
    build_vertex_valid_static,
    get_advantages,
    get_num_clipping_triggers,
    inverse_reward_normalization_fn,
    reward_normalization_fn,
    shuffle_and_batch,
    vertex_avail_at_step,
    vertex_axis_dims,
)


def test_legacy_sp_to_specs():
    print("\n[env] _legacy_sp_to_specs")
    # Each sp_type ∈ {1..4} should map to one rule with the matching (base_idx1, base_idx2)
    # and factor=-1; sp_type 0 should produce all-unused.
    for sp_type, (b1, b2) in sp_type_to_map.items():
        specs = _legacy_sp_to_specs(jnp.array(sp_type, dtype=jnp.int32))
        assert specs.shape == (MAX_RULES_PER_VERTEX, 3)
        row0 = specs[0].tolist()
        assert row0 == [b1, b2, -1], f"sp_type={sp_type} -> {row0}, expected {[b1, b2, -1]}"
        assert jnp.all(specs[1:, 0] == -1)
        print(f"  sp_type={sp_type}  rules={specs.tolist()}")
    # Dense (sp_type=0)
    specs0 = _legacy_sp_to_specs(jnp.array(0, dtype=jnp.int32))
    assert jnp.all(specs0[:, 0] == -1)
    print(f"  sp_type=0  rules={specs0.tolist()}")


def test_step_action_vs_int():
    print("\n[env] StepAction shape parity with legacy int")
    # The legacy path turns `sp_type * MAX_TOKENS + target_vertex` into a rule_specs
    # via `_legacy_sp_to_specs`; constructing a StepAction directly with the same
    # specs must produce the exact same array.
    sp_type = jnp.array(2, dtype=jnp.int32)
    legacy_specs = _legacy_sp_to_specs(sp_type)
    target_vertex = jnp.array(3, dtype=jnp.int32)
    step_action = StepAction(target_vertex=target_vertex, rule_specs=legacy_specs)
    assert step_action.rule_specs.shape == (MAX_RULES_PER_VERTEX, 3)
    assert jnp.array_equal(step_action.rule_specs, legacy_specs)
    print(f"  legacy sp_type=2 -> StepAction.rule_specs={step_action.rule_specs.tolist()}")


def fake_jaxpr(total_v=4):
    """Tiny synthetic jaxpr-like stand-in just for shape inspection helpers."""
    class FakeAval:
        def __init__(self, shape):
            self.shape = shape

    class FakeVar:
        def __init__(self, shape):
            self.aval = FakeAval(shape)

    class FakeEqn:
        def __init__(self, in_shapes, out_shape):
            self.invars = [FakeVar(s) for s in in_shapes]
            self.outvars = [FakeVar(out_shape)]

    class FakeJaxpr:
        def __init__(self, eqns):
            self.eqns = eqns

    eqns = [
        FakeEqn([(4, 4), (4, 4)], (4, 4)),  # 2D out, 2D in -> all 4 pairs
        FakeEqn([(4,), (4, 4)], (4,)),       # 1D out, min_in=1 -> only (0,0)
        FakeEqn([(4, 4), (4,)], (4, 4)),     # 2D out, min_in=1 -> (0,0) and (1,0)
        FakeEqn([(4, 4)], (4,)),             # 1D out, min_in=2 -> (0,0) and (0,1)
    ]
    return FakeJaxpr(eqns[:total_v])


def test_masks():
    print("\n[common] mask builders")
    jaxpr = fake_jaxpr(4)
    out_n, in_n = vertex_axis_dims(jaxpr, 4)
    print(f"  out_ndims={out_n.tolist()}, min_in_ndims={in_n.tolist()}")
    assert out_n.tolist() == [2, 1, 2, 1]
    assert in_n.tolist() == [2, 1, 1, 2]

    # build_pair_valid_mask: STOP always set; the four real pairs gate on shapes.
    mask5 = build_pair_valid_mask(jaxpr, total_v=4, num_pair_choices=5, pair_stop_idx=4)
    assert mask5.shape == (4, 5)
    assert jnp.all(mask5[:, 4] == 1.0), "STOP should be valid for every vertex"
    expected = np.array([
        [1, 1, 1, 1, 1],  # (2,2) -> all four real pairs valid
        [1, 0, 0, 0, 1],  # (1,1) -> only (0,0)
        [1, 0, 1, 0, 1],  # (2,1) -> (0,0) and (1,0)
        [1, 1, 0, 0, 1],  # (1,2) -> (0,0) and (0,1)
    ], dtype=np.float32)
    assert jnp.array_equal(mask5, jnp.array(expected)), f"got {mask5.tolist()}"
    print(f"  pair_valid_mask\n{mask5}")

    # disable_sparsification: only STOP is valid.
    mask_off = build_pair_valid_mask(
        jaxpr, total_v=4, num_pair_choices=5, pair_stop_idx=4, disable_sparsification=True
    )
    assert jnp.all(mask_off[:, :4] == 0.0)
    assert jnp.all(mask_off[:, 4] == 1.0)

    # Legacy 5-row mask (PPO old layout) and 3-row mask (alpha0/mu0/gdpo layout) both build.
    leg5 = build_legacy_sp_valid_mask(jaxpr, 4, num_sp_types=5, use_min_in_ndim=True)
    assert leg5.shape == (5, 4)
    leg3 = build_legacy_sp_valid_mask(jaxpr, 4, num_sp_types=3, use_min_in_ndim=False)
    assert leg3.shape == (3, 4)
    print(f"  legacy 5-row mask shape ok; 3-row mask shape ok")


def test_pair_factor_valid_mask():
    """Per-(vertex, pair, factor) validity mask used by the autoreg head.

    For each (vertex, pair) the mask should:
    * always allow factor in {-1, 0, 1} (these go through specialised
      paths in `apply_dynamic_sparsity` that don't crash);
    * allow factor K > 1 only if K divides both axis sizes for that pair;
    * keep at least one factor enabled in the STOP slot so the
      categorical never produces NaN softmax.
    """
    print("\n[common] build_pair_factor_valid_mask")
    jaxpr = fake_jaxpr(4)  # vertex 0 has out (4,4), in (4,4); 4 % 2 == 0 etc.
    factor_table = jnp.array([-1, 1, 2, 4, 3])  # 3 doesn't divide 4
    pair_stop_idx = 4
    num_pair_choices = pair_stop_idx + 1  # 4 pair indices + STOP
    mask = build_pair_factor_valid_mask(
        jaxpr, total_v=4, num_pair_choices=num_pair_choices,
        factor_table=factor_table, pair_stop_idx=pair_stop_idx,
    )
    assert mask.shape == (4, num_pair_choices, 5)

    # vertex 0 has out (4,4), inv[0] (4,4). All four real pairs reference
    # axes of size 4. So {-1, 0, 1, 2, 4} are valid; 3 doesn't divide 4.
    for p in range(4):
        assert float(mask[0, p, 0]) == 1.0, "factor=-1 always valid"
        assert float(mask[0, p, 1]) == 1.0, "factor=1 (no-op) always valid"
        assert float(mask[0, p, 2]) == 1.0, "factor=2 divides 4"
        assert float(mask[0, p, 3]) == 1.0, "factor=4 divides 4"
        assert float(mask[0, p, 4]) == 0.0, "factor=3 does NOT divide 4"

    # STOP slot keeps factor 0 valid (any factor is fine; it's ignored
    # downstream when pair_idx == STOP).
    assert float(mask[0, pair_stop_idx, 0]) == 1.0

    # vertex 1 has out (4,), in (4,4): only pair 0 (= (0,0)) is valid.
    # All others should have all factors masked off.
    assert float(mask[1, 0, 0]) == 1.0  # pair 0 with factor=-1
    for p in (1, 2, 3):
        for f in range(5):
            assert float(mask[1, p, f]) == 0.0, (
                f"vertex 1 pair {p} factor {f} should be masked"
            )

    # Sums per (v, p) should be > 0 for valid slots, exactly 1 for STOP.
    sums = jnp.sum(mask, axis=-1)
    assert float(sums[0, pair_stop_idx]) == 1.0
    print(f"  vertex 0 valid factors per pair: {mask[0].tolist()}")
    print(f"  vertex 1 valid factors per pair: {mask[1].tolist()}")


def test_vertex_avail_at_step():
    print("\n[common] vertex_avail_at_step")
    from collections import namedtuple
    State = namedtuple("State", ["order", "step_count"])
    total_v = 5
    valid_static = jnp.array([1, 1, 1, 0, 1], dtype=jnp.float32)  # vertex 4 (idx=3) cannot be eliminated
    # After 2 steps, vertices 1 and 3 have been chosen.
    state = State(order=jnp.array([1, 3, 5, 2, 4], jnp.int32), step_count=jnp.array(2, jnp.int32))
    avail = vertex_avail_at_step(state, valid_static, total_v=total_v, num_valid=5)
    expected = jnp.array([0, 1, 0, 0, 1], dtype=jnp.float32)
    assert jnp.array_equal(avail, expected), f"avail={avail.tolist()} expected={expected.tolist()}"
    print(f"  avail={avail.tolist()}")


def test_gae():
    print("\n[common] get_advantages")
    rollout, num_envs, num_rewards = 5, 3, 2
    key = jrand.PRNGKey(0)
    keys = jrand.split(key, 5)
    rewards = jrand.normal(keys[0], (num_envs, rollout, num_rewards))
    dones = jnp.zeros((num_envs, rollout, num_rewards))
    values = jrand.normal(keys[2], (num_envs, rollout, num_rewards))
    next_values = jrand.normal(keys[3], (num_envs, rollout, num_rewards))
    discounts = jnp.full((num_envs, rollout), 0.99)
    _, estim_returns, advantages = get_advantages(rewards, dones, values, next_values, discounts, 0.95)
    assert estim_returns.shape == (num_envs, rollout, num_rewards)
    assert advantages.shape == (num_envs, rollout, num_rewards)
    assert jnp.all(jnp.isfinite(estim_returns))
    assert jnp.all(jnp.isfinite(advantages))
    print(f"  estim_returns mean={float(jnp.mean(estim_returns)):.3f}, "
          f"advantage mean={float(jnp.mean(advantages)):.3f}")


def test_norm_roundtrip():
    print("\n[common] reward_normalization round-trip (symlog/symexp)")
    x = jnp.array([-1234.5, -1.0, 0.0, 1.0, 1234.5])
    y = inverse_reward_normalization_fn(reward_normalization_fn(x))
    assert jnp.allclose(x, y, atol=1e-3), f"x={x.tolist()}  y={y.tolist()}"
    print(f"  symlog→symexp ok on {x.tolist()}")


def test_shuffle_and_batch():
    print("\n[common] shuffle_and_batch tree handling")
    num_envs, rollout, total_v = 3, 4, 6
    tree = dict(
        a=jnp.arange(num_envs * rollout * total_v).reshape(num_envs, rollout, total_v),
        b=jnp.arange(num_envs * rollout).reshape(num_envs, rollout),
    )
    minibatches = 4
    out = shuffle_and_batch(tree, minibatches, jrand.PRNGKey(7))
    assert out["a"].shape == (minibatches, num_envs * rollout // minibatches, total_v)
    assert out["b"].shape == (minibatches, num_envs * rollout // minibatches)
    # All elements must be a permutation of the original (no data loss when divisible).
    assert sorted(out["b"].flatten().tolist()) == list(range(num_envs * rollout))
    print(f"  a:{out['a'].shape}, b:{out['b'].shape}, permutation preserved")


def main():
    print("=== Common-utility tests ===")
    test_legacy_sp_to_specs()
    test_step_action_vs_int()
    test_masks()
    test_pair_factor_valid_mask()
    test_vertex_avail_at_step()
    test_gae()
    test_norm_roundtrip()
    test_shuffle_and_batch()
    print("\nALL COMMON TESTS OK")


if __name__ == "__main__":
    main()
