"""End-to-end loss + gradient test on a synthetic batch.

Exercises the refactored loss path (vmapped `evaluate_action`, joint log-prob
ratio, KL across heads, value loss) without going through the env. Done for
all four (vertex-policy, rule-policy) combinations.
"""

from __future__ import annotations


import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax

from alphagrad.approx.env import MAX_TOKENS
from alphagrad.approx.ppo import (
    NUM_PAIR_CHOICES,
    NUM_VALUE_HEADS,
    PAIR_STOP,
    TrainBatch,
    _build_agent,
    _build_factor_table,
    _scale_output_heads,
    _select_variant,
    _variant_label,
    make_argparser,
    old_log_prob_for_action,
)
from alphagrad.approx.common import (
    init_linear_weights,
    reward_normalization_fn,
    get_num_clipping_triggers,
)
from alphagrad.utils import entropy as entropy_fn, explained_variance


def build_one(no_ptr, not_autoreg, total_v=6, batch_size=8, seed=0):
    cli = []
    if no_ptr:
        cli.append("--no-ptr")
    if not_autoreg:
        cli.append("--not-autoreg")
    cli += [
        "--max-rules", "3",
        "--factors=-1,1,2",
        "--embd-dim", "16",
        "--num-heads", "2",
        "--num-layers", "1",
        "--hidden-dim", "32",
        "--policy-dims", "32,16",
        "--value-dims", "32,16",
        "--wandb", "disabled",
        "--rewards", "cmp", "mem",
        "--dataset", "none",
    ]
    args = make_argparser().parse_args(cli)
    use_pointer, use_autoreg = _select_variant(args)
    factor_table, factors_py, num_factors, max_rules = _build_factor_table(args, use_autoreg)
    key = jrand.PRNGKey(seed)
    agent_key, init_key, batch_key = jrand.split(key, 3)
    agent = _build_agent(
        use_pointer, use_autoreg, args, total_v,
        num_factors=num_factors, max_rules=max_rules, key=agent_key,
    )
    agent = init_linear_weights(agent, init_key)
    agent = _scale_output_heads(agent, args.head_init_scale, use_pointer, use_autoreg)
    return _variant_label(use_pointer, use_autoreg), agent, factor_table, max_rules, num_factors, batch_key


def fake_batch(batch_size, total_v, max_rules, num_factors, num_pair_choices, num_value_heads, key):
    keys = jrand.split(key, 10)
    tokens = jrand.randint(keys[0], (batch_size, MAX_TOKENS), 1, 256, dtype=jnp.int32)
    tokens = tokens.at[:, 100:].set(0)
    # Synthetic per-token equation IDs: cycle through small range over the
    # non-pad prefix; padding tokens get -1. Mirrors what `_callback` produces.
    arange_t = jnp.arange(MAX_TOKENS)
    fake_eids = jnp.where(arange_t < 100, (arange_t // 5) % 8, -1).astype(jnp.int32)
    eqn_ids = jnp.broadcast_to(fake_eids, (batch_size, MAX_TOKENS))
    vertex_idx = jrand.randint(keys[1], (batch_size,), 0, total_v, dtype=jnp.int32)
    pair_seq = jrand.randint(keys[2], (batch_size, max_rules), 0, num_pair_choices, dtype=jnp.int32)
    factor_seq = jrand.randint(keys[3], (batch_size, max_rules), 0, num_factors, dtype=jnp.int32)

    # Old policy dists: uniform, since this is a freshly-initialised "old" snapshot.
    old_v = jnp.ones((batch_size, total_v)) / total_v
    old_p = jnp.ones((batch_size, max_rules, num_pair_choices)) / num_pair_choices
    old_f = jnp.ones((batch_size, max_rules, num_factors)) / num_factors

    estim_returns = jrand.normal(keys[4], (batch_size, num_value_heads))
    norm_adv = jrand.normal(keys[5], (batch_size,))
    vertex_avail = jnp.ones((batch_size, total_v))
    # F preference vector — uniform 1/K as a stand-in for the per-env Dirichlet sample.
    preference = jnp.full(
        (batch_size, num_value_heads), 1.0 / num_value_heads, dtype=jnp.float32,
    )
    # Dynamic-substeps fields — zero-filled because this synthetic test
    # exercises the legacy rule head only. Shapes mirror the production
    # rollout_fn placeholders (see _dyn_zero_* in ppo.main).
    max_substeps = max_rules  # use the same bound for the test
    max_primes = 9
    max_axes_per_vertex = 8
    # Op-type vocabulary: DIAG / COMPRESS / QUANT / END = 4.
    num_ops = 4
    max_exponent = 30
    num_compress_kinds = 6
    num_quant_dtypes = 28
    micro_op_seq = jnp.zeros((batch_size, max_substeps), dtype=jnp.int32)
    micro_i_seq = jnp.zeros((batch_size, max_substeps), dtype=jnp.int32)
    micro_j_seq = jnp.zeros((batch_size, max_substeps), dtype=jnp.int32)
    micro_exp_seq = jnp.zeros((batch_size, max_substeps, max_primes), dtype=jnp.int32)
    micro_factor_seq = jnp.zeros((batch_size, max_substeps), dtype=jnp.int32)
    micro_compress_kind_seq = jnp.zeros((batch_size, max_substeps), dtype=jnp.int32)
    micro_quant_dtype_seq = jnp.zeros((batch_size, max_substeps), dtype=jnp.int32)
    micro_op_dists = jnp.zeros((batch_size, max_substeps, num_ops), dtype=jnp.float32)
    micro_i_dists = jnp.zeros(
        (batch_size, max_substeps, max_axes_per_vertex), dtype=jnp.float32,
    )
    micro_j_dists = jnp.zeros(
        (batch_size, max_substeps, max_axes_per_vertex), dtype=jnp.float32,
    )
    micro_exp_dists = jnp.zeros(
        (batch_size, max_substeps, max_primes, max_exponent + 1),
        dtype=jnp.float32,
    )
    micro_kind_dists = jnp.zeros(
        (batch_size, max_substeps, num_compress_kinds), dtype=jnp.float32,
    )
    micro_quant_dists = jnp.zeros(
        (batch_size, max_substeps, num_quant_dtypes), dtype=jnp.float32,
    )
    return TrainBatch(
        tokens=tokens,
        eqn_ids=eqn_ids,
        preference=preference,
        vertex_idx=vertex_idx,
        pair_seq=pair_seq,
        factor_seq=factor_seq,
        micro_op_seq=micro_op_seq,
        micro_i_seq=micro_i_seq,
        micro_j_seq=micro_j_seq,
        micro_exp_seq=micro_exp_seq,
        micro_factor_seq=micro_factor_seq,
        micro_compress_kind_seq=micro_compress_kind_seq,
        micro_quant_dtype_seq=micro_quant_dtype_seq,
        old_vertex_dist=old_v,
        old_pair_dists=old_p,
        old_factor_dists=old_f,
        old_micro_op_dists=micro_op_dists,
        old_micro_i_dists=micro_i_dists,
        old_micro_j_dists=micro_j_dists,
        old_micro_exp_dists=micro_exp_dists,
        old_micro_kind_dists=micro_kind_dists,
        old_micro_quant_dists=micro_quant_dists,
        estim_returns=estim_returns,
        norm_adv=norm_adv,
        vertex_avail_mask=vertex_avail,
    )


def make_loss_fn(pair_valid_mask, pair_factor_mask, eps=0.2, value_weight=0.5, entropy_weight=0.05):
    """Replicate the structure of ppo.main()'s `loss`, returning (scalar_loss, metrics_dict)."""
    def loss(agent, batch, keys):
        eval_batched = jax.vmap(
            lambda toks, vidx, pseq, fseq, vmask, k: agent.evaluate_action(
                toks, vidx, pseq, fseq, vmask, pair_valid_mask, pair_factor_mask, k,
            )
        )
        log_probs, entropies, values, vertex_dist, pair_dists, factor_dists = eval_batched(
            batch.tokens, batch.vertex_idx, batch.pair_seq, batch.factor_seq,
            batch.vertex_avail_mask, keys,
        )
        old_log_probs = jax.vmap(old_log_prob_for_action)(
            batch.vertex_idx, batch.pair_seq, batch.factor_seq,
            batch.old_vertex_dist, batch.old_pair_dists, batch.old_factor_dists,
        )
        ratio = jnp.exp(log_probs - old_log_probs)
        clipping = jnp.minimum(
            ratio * batch.norm_adv,
            jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * batch.norm_adv,
        )
        ppo_loss = jnp.mean(-clipping)
        entropy_loss = jnp.mean(entropies)
        value_loss = jnp.mean(
            jnp.sum((values - reward_normalization_fn(batch.estim_returns)) ** 2, axis=-1)
        )
        total = ppo_loss + value_weight * value_loss - entropy_weight * entropy_loss
        return total, dict(
            ppo_loss=ppo_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            ratio_min=jnp.min(ratio),
            ratio_max=jnp.max(ratio),
            num_clip=get_num_clipping_triggers(ratio, eps),
        )
    return loss


def test_variant(no_ptr, not_autoreg):
    total_v = 6
    batch_size = 8
    label, agent, factor_table, max_rules, num_factors, batch_key = build_one(
        no_ptr, not_autoreg, total_v=total_v, batch_size=batch_size
    )
    print(f"\n[{label}]  max_rules={max_rules}  num_factors={num_factors}")

    batch = fake_batch(
        batch_size, total_v, max_rules, num_factors, NUM_PAIR_CHOICES,
        num_value_heads=NUM_VALUE_HEADS, key=batch_key,
    )
    pair_valid = jnp.ones((total_v, NUM_PAIR_CHOICES))
    pair_factor = jnp.ones((total_v, NUM_PAIR_CHOICES, num_factors))
    loss_fn = make_loss_fn(pair_valid, pair_factor)

    keys = jrand.split(jrand.PRNGKey(99), batch_size)
    (loss_val, metrics), grads = eqx.filter_value_and_grad(
        lambda a: loss_fn(a, batch, keys), has_aux=True
    )(agent)
    print(f"  loss={float(loss_val):.4f}  ppo={float(metrics['ppo_loss']):.4f} "
          f"value={float(metrics['value_loss']):.4f} ent={float(metrics['entropy_loss']):.4f}")
    print(f"  ratio in [{float(metrics['ratio_min']):.3f}, {float(metrics['ratio_max']):.3f}], "
          f"clip_count={int(metrics['num_clip'])}")

    # Sanity: loss is finite, gradients are finite and non-zero somewhere.
    assert jnp.isfinite(loss_val), f"non-finite loss in {label}"
    leaves = [
        x for x in jax.tree_util.tree_leaves(grads)
        if eqx.is_inexact_array(x)
    ]
    grad_norms = [float(jnp.linalg.norm(x)) for x in leaves]
    assert all(jnp.isfinite(jnp.array(grad_norms))), f"non-finite grads in {label}"
    nonzero = sum(1 for n in grad_norms if n > 0)
    print(f"  {nonzero}/{len(grad_norms)} grad tensors are nonzero")
    assert nonzero > 0, f"no gradient flow in {label}"

    # Optimiser step: confirms the agent pytree is compatible with optax/eqx.apply_updates.
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(agent, eqx.is_inexact_array))
    updates, _ = optimizer.update(grads, opt_state, agent)
    new_agent = eqx.apply_updates(agent, updates)
    # New agent's parameters must still be valid: re-evaluate loss on the same batch.
    new_loss, _ = loss_fn(new_agent, batch, keys)
    assert jnp.isfinite(new_loss)
    print(f"  post-step loss={float(new_loss):.4f}  delta={float(new_loss - loss_val):+.4f}")
    return label


def main():
    print("=== Loss/grad smoke tests ===")
    labels = []
    for no_ptr, not_autoreg in [(False, False), (False, True), (True, False), (True, True)]:
        labels.append(test_variant(no_ptr, not_autoreg))
    print("\nALL LOSS/GRAD VARIANTS OK:", labels)


if __name__ == "__main__":
    main()
