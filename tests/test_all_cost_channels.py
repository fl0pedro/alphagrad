"""Pin that every cost channel populates on a successful env step.

The callback in `env._callback` carries a 10-channel reward vector:

    [0] muls_adds_fmas   — graphax symbolic op count
    [1] flops            — XLA cost_analysis
    [2] latency_ns       — wall-clock latency, only when measure_latency
    [3] max_io_sum       — graphax mem accumulator
    [4] bytes_accessed   — XLA cost_analysis
    [5] peak_memory      — ResourceMonitor peak HBM
    [6] cosine_sim       — only at terminal
    [7] frob_residual    — only at terminal
    [8] xla_peak_memory  — deterministic XLA memory_analysis peak estimate
    [9] bkstep_acc       — B_kstep trainability accuracy, only at terminal

The user explicitly asked us to test that ALL SIX cost channels (0..5)
populate when measure_latency=True — so downstream comparison and wandb
logging have signal even when --rewards selects only one of them.

The 144/192-sentinel bug observed in RQ1 (CPU pool size 4 vs num_envs 16)
zeroed *every* channel; these tests fail loudly if that or any similar
regression re-zeros the cost vector.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    REWARD_NAMES,
    StepAction,
    VertexEliminationEnv,
)


def _terminal_action_for(env) -> StepAction:
    """Drive the env to its terminal step in one go: pick the first valid
    vertex, plant no approximation rules. With the env at the first
    pre-terminal vertex this would emit cosine_sim and frob_residual; we
    only need the COST channels to populate, so just step once."""
    target_v = jnp.asarray(env.valid_vertices[0], dtype=jnp.int32)
    rule_specs = (
        jnp.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=jnp.int32)
        .at[..., 2].set(0)
    )
    return StepAction(target_vertex=target_v, rule_specs=rule_specs)


def _env_for(target_fn, args, *, measure_latency: bool):
    """Build the smallest possible VertexEliminationEnv that runs the full
    callback (i.e. has target_fun so cost_analysis / ResourceMonitor are
    actually exercised). Helmholtz is enough — same shape envelope as
    VmappedNN's per-step jaxpr."""
    closed_jaxpr = jax.make_jaxpr(target_fn)(*args)
    return VertexEliminationEnv.from_jaxpr(
        closed_jaxpr,
        args=args,
        argnums=(0,),
        num_envs=0,
        target_fun=target_fn,
        cmp_type="flops",
        mem_type="peak_memory",
        measure_latency=measure_latency,
    )


def test_all_six_cost_channels_populate_when_measure_latency_on():
    """The vital assertion: with --measure-latency on, every cost-family
    index (0..5) is non-zero after one successful env step. A pool-
    starvation / timeout sentinel would zero ALL of them; a half-
    configured callback (latency disabled) would zero #2 alone.
    """
    from graphax import examples

    print("\n[env] all 6 cost channels populate (measure_latency=True)")
    x = jnp.array([0.05, 0.15, 0.25, 0.35], dtype=jnp.float32)
    env = _env_for(examples.Helmholtz, (x,), measure_latency=True)

    state = env.reset()
    out = env.step(state, _terminal_action_for(env))
    reward = out.reward

    # Reward vector encodes cost as NEGATIVE; channel populated iff != 0.
    cost = {REWARD_NAMES[i]: float(reward[i]) for i in range(6)}
    print(f"  cost channels = {cost}")

    nonzero = {k: v for k, v in cost.items() if v != 0.0}
    missing = [k for k, v in cost.items() if v == 0.0]
    assert not missing, (
        f"cost channels read 0.0 — broken: {missing}. Full vector: {cost}"
    )
    print(f"  all 6 channels populated; {len(nonzero)} nonzero values")


def test_latency_is_zero_when_measure_latency_off():
    """Inverse check: with measure_latency=False, latency_ns is hardcoded
    to 0.0 (env.py:1356-1360) but every other cost channel still populates.
    Confirms the conditional is the only differentiator — so adding
    --measure-latency to the sbatch is sufficient to enable all six.
    """
    from graphax import examples

    print("\n[env] latency_ns=0 when measure_latency=False (but others populate)")
    x = jnp.array([0.05, 0.15, 0.25, 0.35], dtype=jnp.float32)
    env = _env_for(examples.Helmholtz, (x,), measure_latency=False)

    state = env.reset()
    out = env.step(state, _terminal_action_for(env))
    reward = out.reward
    cost = {REWARD_NAMES[i]: float(reward[i]) for i in range(6)}
    print(f"  cost channels = {cost}")

    assert cost["latency_ns"] == 0.0, (
        f"latency_ns should be 0 when measure_latency=False; got {cost['latency_ns']}"
    )
    # The remaining 5 cost channels (muls, flops, max_io, bytes, peak_mem)
    # are still populated regardless of the latency flag.
    other_channels = [
        "muls_adds_fmas", "flops", "max_io_sum",
        "bytes_accessed", "peak_memory",
    ]
    missing = [k for k in other_channels if cost[k] == 0.0]
    assert not missing, (
        f"non-latency cost channels should populate even when measure_latency=False; "
        f"missing: {missing}. Full vector: {cost}"
    )
    print(f"  latency_ns correctly zero; {len(other_channels)} other channels populated")


def test_target_fun_none_early_return_zeros_jit_channels():
    """Pin the *broken* behaviour: when target_fun is None (legacy
    pre-fix codepath), env._callback early-returns and 4 of the 6 cost
    channels (flops, latency_ns, bytes_accessed, peak_memory) read 0.
    The fix at all trainer-side env builders is to ALWAYS pass
    target_fun=target_fn regardless of args.rewards. This test exists
    so a future "performance optimisation" PR that re-adds the
    conditional fails loudly here.
    """
    from graphax import examples

    print("\n[env] target_fun=None reproduces the broken zeroing")
    x = jnp.array([0.05, 0.15, 0.25, 0.35], dtype=jnp.float32)
    # Pass target_fun=None — replicates the legacy
    # ``target_fn if "acc" in args.rewards else None`` path.
    closed_jaxpr = jax.make_jaxpr(examples.Helmholtz)(x)
    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr,
        args=(x,),
        argnums=(0,),
        num_envs=0,
        target_fun=None,           # <- the broken path
        cmp_type="flops",
        mem_type="peak_memory",
        measure_latency=True,
    )
    state = env.reset()
    out = env.step(state, _terminal_action_for(env))
    cost = {REWARD_NAMES[i]: float(out.reward[i]) for i in range(6)}
    # Quality channels too — the 2-phase schedule docs commit to these
    # being 0 (NOT 1.0) so phase 1 doesn't false-trigger anti-degeneracy.
    quality = {REWARD_NAMES[i]: float(out.reward[i]) for i in (6, 7)}
    print(f"  with target_fun=None: cost channels = {cost}")
    print(f"  with target_fun=None: quality channels = {quality}")

    # Symbolic counters still populate (graphax vertex_elimination_jaxpr).
    assert cost["muls_adds_fmas"] != 0.0
    assert cost["max_io_sum"] != 0.0
    # JIT-exec channels are zeroed by the early-return.
    for k in ("flops", "latency_ns", "bytes_accessed", "peak_memory"):
        assert cost[k] == 0.0, (
            f"with target_fun=None, channel {k!r} should be 0 (legacy "
            f"early-return); got {cost[k]}. The fix is to set "
            f"env_target_fun = target_fn in the trainer-side env builders."
        )
    # cossim must be 0 (no signal), NOT 1.0 (which would false-trigger
    # anti-degeneracy ceilings). This is the 2-phase schedule contract.
    assert quality["cosine_sim"] == 0.0, (
        f"cossim must be 0 in early-return (was 1.0 in pre-2026-05-24 code); "
        f"got {quality['cosine_sim']}"
    )
    assert quality["frob_residual"] == 0.0
    print("  confirmed: 4 cost + 2 quality channels zeroed when target_fun=None")


def main():
    print("=== all 6 cost channels populate tests ===")
    test_all_six_cost_channels_populate_when_measure_latency_on()
    test_latency_is_zero_when_measure_latency_off()
    test_target_fun_none_early_return_zeros_jit_channels()
    print("\nALL COST-CHANNEL TESTS OK")


if __name__ == "__main__":
    main()
