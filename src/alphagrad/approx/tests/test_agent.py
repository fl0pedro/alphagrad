"""Agent-level smoke tests that exercise every variant without the broken graphax env."""

from __future__ import annotations


import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.approx.env import (
    MAX_TOKENS,
    MAX_RULES_PER_VERTEX,
    NUM_AXIS_PAIRS,
    StepAction,
)
from alphagrad.approx.ppo import (
    NUM_PAIR_CHOICES,
    PAIR_STOP,
    Agent,
    AutoregRulePolicy,
    MLPVertexPolicy,
    PointerVertexPolicy,
    SingleRulePolicy,
    Trajectory,
    TrainBatch,
    _build_agent,
    _build_factor_table,
    _scale_output_heads,
    _select_variant,
    _variant_label,
    build_rule_specs,
    build_legacy_rule_specs,
    make_argparser,
    old_log_prob_for_action,
)
from alphagrad.approx.common import init_linear_weights


def make_args(no_ptr: bool, not_autoreg: bool, **overrides):
    """Build a Namespace of args that mirrors what argparse would yield."""
    p = make_argparser()
    cli = ["--no-jit"] if False else []
    if no_ptr:
        cli.append("--no-ptr")
    if not_autoreg:
        cli.append("--not-autoreg")
    cli += [
        "--episodes", "1",
        "--num-envs", "2",
        "--minibatches", "2",
        "--ppo-epochs", "1",
        "--max-rules", "3",
        "--factors=-1,1,2",
        "--embd-dim", "16",
        "--num-heads", "2",
        "--num-layers", "1",
        "--hidden-dim", "32",
        "--policy-dims", "32,16",
        "--value-dims", "32,16",
        "--wandb", "disabled",
        "--rewards", "cmp",
        "--dataset", "none",
    ]
    args = p.parse_args(cli)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def build_one(no_ptr: bool, not_autoreg: bool, total_v: int = 6, seed: int = 0):
    args = make_args(no_ptr, not_autoreg)
    use_pointer, use_autoreg = _select_variant(args)
    label = _variant_label(use_pointer, use_autoreg)
    factor_table, factors_py, num_factors, max_rules = _build_factor_table(
        args, use_autoreg
    )
    key = jrand.PRNGKey(seed)
    agent_key, init_key, _ = jrand.split(key, 3)
    agent = _build_agent(
        use_pointer, use_autoreg, args, total_v,
        num_factors=num_factors, max_rules=max_rules, key=agent_key,
    )
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, args.head_init_scale, use_pointer, use_autoreg)
    return label, agent, factor_table, max_rules, num_factors


def fake_state(total_v: int, num_factors: int, key):
    """Build a single (tokens, vertex_avail, pair_valid, pair_factor) tuple."""
    tk_key, _mask_key = jrand.split(key)
    tokens = jrand.randint(tk_key, (MAX_TOKENS,), 1, 256, dtype=jnp.int32)
    tokens = tokens.at[50:].set(0)
    vertex_avail = jnp.ones((total_v,), dtype=jnp.float32)
    pair_valid = jnp.ones((total_v, NUM_PAIR_CHOICES), dtype=jnp.float32)
    pair_factor = jnp.ones((total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32)
    return tokens, vertex_avail, pair_valid, pair_factor


def test_variant(no_ptr: bool, not_autoreg: bool):
    total_v = 6
    label, agent, factor_table, max_rules, num_factors = build_one(no_ptr, not_autoreg, total_v)
    print(f"\n[{label}]  max_rules={max_rules}  num_factors={num_factors}")

    keys = jrand.split(jrand.PRNGKey(42), 4)
    tokens, vertex_avail, pair_valid, pair_factor = fake_state(total_v, num_factors, keys[0])

    # 1) sample_action
    out = agent.sample_action(tokens, vertex_avail, pair_valid, pair_factor, keys[1])
    (vertex_idx, pair_seq, factor_seq, vertex_dist, pair_dists, factor_dists,
     value, _v_context) = out
    assert vertex_dist.shape == (total_v,)
    assert pair_seq.shape == (max_rules,)
    assert factor_seq.shape == (max_rules,)
    assert pair_dists.shape == (max_rules, NUM_PAIR_CHOICES)
    assert factor_dists.shape == (max_rules, num_factors)
    assert jnp.all(jnp.isfinite(vertex_dist)) and jnp.allclose(jnp.sum(vertex_dist), 1.0, atol=1e-5)
    assert jnp.all(jnp.isfinite(pair_dists)) and jnp.allclose(jnp.sum(pair_dists, axis=-1), 1.0, atol=1e-5)
    assert jnp.all(jnp.isfinite(factor_dists)) and jnp.allclose(jnp.sum(factor_dists, axis=-1), 1.0, atol=1e-5)
    print(f"  sample shapes ok, value={value!r}")

    # 2) to_env_action: ensure StepAction has correct shape
    step_action = agent.to_env_action(vertex_idx, pair_seq, factor_seq, factor_table)
    assert step_action.target_vertex.shape == ()
    assert step_action.rule_specs.shape == (MAX_RULES_PER_VERTEX, 3)
    assert step_action.target_vertex.dtype == jnp.int32
    assert step_action.rule_specs.dtype == jnp.int32
    print(f"  step_action.target_vertex={int(step_action.target_vertex)} rule_specs={step_action.rule_specs.tolist()}")

    # 3) evaluate_action returns the SAME log-prob the rollout's stored dists imply.
    eval_out = agent.evaluate_action(
        tokens, vertex_idx, pair_seq, factor_seq, vertex_avail, pair_valid, pair_factor, keys[2]
    )
    new_log_prob, new_entropy, new_value, new_v_dist, new_p_dists, new_f_dists = eval_out
    old_log_prob = old_log_prob_for_action(
        vertex_idx, pair_seq, factor_seq, vertex_dist, pair_dists, factor_dists
    )
    eval_same_key = agent.evaluate_action(
        tokens, vertex_idx, pair_seq, factor_seq, vertex_avail, pair_valid, pair_factor, keys[1]
    )
    new_log_prob_same, *_ = eval_same_key
    diff = float(jnp.abs(new_log_prob_same - old_log_prob))
    print(f"  evaluate_action logprob: same_key={float(new_log_prob_same):.6f}, "
          f"old_dist_recompute={float(old_log_prob):.6f}, diff={diff:.6e}")
    assert diff < 1e-3, f"log-prob mismatch for variant {label}: {diff}"

    # 4) build_rule_specs / build_legacy_rule_specs round-trip via to_env_action handles STOP.
    forced_pair = pair_seq.at[1].set(PAIR_STOP)
    forced_specs = build_rule_specs(forced_pair, factor_seq, factor_table)
    assert jnp.all(forced_specs[1:, 0] == -1)
    assert jnp.all(forced_specs[1:, 2] == 0)
    print(f"  STOP propagation ok")

    # 5) sample_action under filter_jit (the path used in the rollout)
    import equinox as eqx
    @eqx.filter_jit
    def _sample(agent, tokens, vmask, pmask, fmask, k):
        return agent.sample_action(tokens, vmask, pmask, fmask, k)
    out2 = _sample(agent, tokens, vertex_avail, pair_valid, pair_factor, keys[3])
    assert out2[0].shape == ()
    print(f"  filter_jit sample_action ok")
    return label


def test_autoreg_respects_factor_mask():
    """The autoregressive policy must never sample a factor that the
    `pair_factor_mask` has zeroed out for the chosen pair. Pin this with
    a mask that allows only ONE factor per pair across many samples."""
    print("\n[autoreg] factor head respects pair_factor_mask")
    total_v = 4
    label, agent, factor_table, max_rules, num_factors = build_one(
        no_ptr=False, not_autoreg=False, total_v=total_v,
    )
    print(f"  variant={label}, num_factors={num_factors}, max_rules={max_rules}")
    assert num_factors >= 2, "test needs >=2 factors to be meaningful"

    keys = jrand.split(jrand.PRNGKey(123), 32)
    tokens, vertex_avail, pair_valid, _ = fake_state(total_v, num_factors, keys[0])

    # Allow ONLY factor index 1 for every (vertex, pair). Factor 0 is the
    # agent's "default first" choice; if the mask works, the agent must
    # always emit factor index 1 (or another non-zero index allowed by the
    # mask) for non-STOP slots — never index 0.
    pair_factor = jnp.zeros((total_v, NUM_PAIR_CHOICES, num_factors), dtype=jnp.float32)
    pair_factor = pair_factor.at[:, :, 1].set(1.0)
    # STOP slot: keep factor 0 valid (it's the unused-factor placeholder).
    pair_factor = pair_factor.at[:, PAIR_STOP, 0].set(1.0)

    bad_factors = 0
    for k in keys[1:]:
        out = agent.sample_action(tokens, vertex_avail, pair_valid, pair_factor, k)
        _, pair_seq, factor_seq, *_ = out
        for slot in range(int(max_rules)):
            p = int(pair_seq[slot])
            f = int(factor_seq[slot])
            if p == PAIR_STOP:
                continue  # factor irrelevant
            if f != 1:
                bad_factors += 1
    assert bad_factors == 0, (
        f"autoreg head sampled {bad_factors} forbidden factors "
        "despite the mask zeroing them out"
    )
    print(f"  no forbidden factors sampled across {len(keys)-1} draws")


def main():
    print("=== Agent variant smoke tests ===")
    labels = []
    for no_ptr, not_autoreg in [(False, False), (False, True), (True, False), (True, True)]:
        labels.append(test_variant(no_ptr, not_autoreg))
    test_autoreg_respects_factor_mask()
    print("\nALL VARIANTS OK:", labels)


if __name__ == "__main__":
    main()
