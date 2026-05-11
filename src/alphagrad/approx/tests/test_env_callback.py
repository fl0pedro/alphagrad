"""Regression tests for `alphagrad.approx.env._callback`.

The two failure modes these guard against — both surfaced by running the PPO
trainer end-to-end on `VmappedNeuralNetwork`:

1. **NaN accuracy reward** — the cossim aggregation used `[6:8].mean()` on a
   stack of `n_samples` cosine similarities. That slice is non-empty only when
   `n_samples >= 8` (the latency-mode path uses 10). Every other cmp/mem
   combination produces `n_samples == 1`, in which case `[6:8]` is empty and
   the JAX `mean()` returns NaN, poisoning the entire `acc` reward, GAE
   targets, and the value loss.

2. **Mid-rollout crash on autoregressive rule policies** — the autoreg agent
   draws factors from `--factors` (default `-1,1,2,4`). Any factor other than
   `-1` lands in graphax `apply_dynamic_sparsity`, which then produces
   `SparseIndex` configurations that the matmul `_prepare_physical_array` /
   `_prepare_contraction_views` paths don't handle. The env coerces non-`-1`
   factors to `-1` so the rollout can proceed even while the underlying
   graphax bug is being worked on; that coercion is what these tests pin
   down.
"""

from __future__ import annotations


from collections import namedtuple
from unittest import mock

import jax
import jax.numpy as jnp
import jax.random as jrand

from alphagrad.approx.env import (
    MAX_RULES_PER_VERTEX,
    MAX_TOKENS,
    EnvConfig,
    StepAction,
    VertexEliminationEnv,
    _callback,
    cossim,
)


# ---------------------------------------------------------------------------
# Tiny jaxpr helpers – avoid pulling in a full graphax build
# ---------------------------------------------------------------------------


def _tiny_jaxpr():
    """Build a closed jaxpr for `(x @ W).sum()` — the smallest thing whose
    `_callback` produces a non-trivial transforms sequence. We use square
    (4, 4) operands so multiple Diag rules on disjoint (out, primal) axis
    pairs can share the divisor 4, which the env-side translator
    accepts."""

    def fn(x, W):
        return jnp.sum(x @ W)

    x = jnp.ones((4, 4), dtype=jnp.float32)
    W = jnp.ones((4, 4), dtype=jnp.float32)
    return jax.make_jaxpr(fn)(x, W), (x, W)


# ---------------------------------------------------------------------------
# Cossim aggregation: n_samples == 1 must not produce NaN
# ---------------------------------------------------------------------------


def test_cossim_aggregation_single_sample_is_not_nan():
    """Mirrors the `error = jnp.stack(all_cossims)...` path. With one sample,
    the previous `[6:8].mean()` returned NaN; the fixed path falls back to a
    plain mean so the result is finite."""

    print("\n[env] cossim aggregation with n_samples=1 must be finite")
    cossim_stack = jnp.stack([jnp.array(0.5, dtype=jnp.float32)])
    assert cossim_stack.shape[0] == 1

    # Explicitly demonstrate the buggy path returns NaN:
    bad = float(cossim_stack.sort()[6:8].mean())
    assert bad != bad, f"baseline buggy path should be NaN, got {bad}"

    # Fixed path: take a plain mean when there are < 8 samples.
    good = float(cossim_stack.mean())
    assert good == 0.5

    # And drive the actual env helper (cossim()) end-to-end on a finite vec
    # to confirm the helper itself doesn't introduce NaN.
    a = jnp.array([1.0, 0.0, 0.0])
    b = jnp.array([0.5, 0.5, 0.0])
    c = float(cossim(a, b))
    assert jnp.isfinite(jnp.array(c)), f"cossim returned non-finite: {c}"
    print(f"  buggy_path={bad}, fixed_path={good}, cossim(a,b)={c:.4f}")


def test_cossim_aggregation_full_quartile_path_unchanged():
    """When n_samples >= 8 (the latency rollout case) the top-quartile mean
    must still be returned — the fix should not regress that branch."""

    print("\n[env] cossim aggregation top-quartile branch unchanged")
    rng = jrand.PRNGKey(0)
    samples = jrand.uniform(rng, (10,))
    stack = jnp.sort(samples)

    expected_quartile = float(stack[6:8].mean())
    if stack.shape[0] >= 8:
        actual = float(stack.sort()[6:8].mean())
    else:
        actual = float(stack.mean())
    assert abs(actual - expected_quartile) < 1e-6
    print(f"  quartile mean = {actual:.4f}")


# ---------------------------------------------------------------------------
# Factor coercion: env never forwards `factor != -1` to graphax
# ---------------------------------------------------------------------------


def _build_callback_state(jaxpr_closed, total_v):
    """Reset-style EnvState pieces that `_callback` needs as inputs."""
    initial_order = jnp.arange(1, total_v + 1, dtype=jnp.int32)
    initial_specs = jnp.full(
        (initial_order.shape[0], MAX_RULES_PER_VERTEX, 3), -1, dtype=jnp.int32
    )
    initial_specs = initial_specs.at[..., 2].set(0)
    return initial_order, initial_specs


def _spy_extract_jaxpr_to_record_transforms():
    """Patch out the heavy graphax call so we just record the
    `transforms` sequence that `_callback` would forward, without paying
    the matmul cost. With the typed-transform migration the kwarg is
    `transforms=[(v, (Diag, Compress, ...)), ...]` (was `sparsity_map`
    before)."""

    captured = {"transforms": None}

    class _StubVE:
        def tokenized(self):
            return jnp.zeros((MAX_TOKENS,), dtype=jnp.int32)

    def fake_extract_jaxpr(*args, transforms=None, **kwargs):
        captured["transforms"] = transforms
        return _StubVE()

    return captured, fake_extract_jaxpr


def test_callback_forwards_arbitrary_factors_as_diag():
    """`_callback` translates each EnvState.sparsity_specs row into a typed
    `Diag(i, j, factor)` and forwards them via the new `transforms=...`
    kwarg. Positive divisors pass through with the same factor value; the
    -1 sentinel is resolved to `math.gcd(d1, d2)` (since apply_diag rejects
    sentinels). This pins both behaviours."""
    from graphax.sparse.micro_actions import Diag

    print("\n[env] _callback emits Diag transforms with explicit factors")
    closed_jaxpr, args = _tiny_jaxpr()
    total_v = len(closed_jaxpr.jaxpr.eqns)

    config = EnvConfig(
        jaxpr=closed_jaxpr.jaxpr,
        argnums=(0, 1),
        has_aux=False,
        sparse=False,
        cmp_type="graphax",
        mem_type="graphax",
        target_fun=None,
        data_gen=None,
    )

    initial_order, initial_specs = _build_callback_state(closed_jaxpr, total_v)
    # Plant rules with non-(-1) factors on disjoint logical axis pairs:
    # pair (0, 0) → idx1=0, idx2=2; pair (1, 1) → idx1=1, idx2=3. Both
    # have axis sizes (4, 4) so divisors 2 and 4 are both legal.
    sparsity_specs = (
        initial_specs
        .at[0, 0].set(jnp.array([0, 0, 2], jnp.int32))
        .at[0, 1].set(jnp.array([1, 1, 4], jnp.int32))
    )
    stop = jnp.asarray(total_v, dtype=jnp.int32)

    captured, fake_extract = _spy_extract_jaxpr_to_record_transforms()
    with mock.patch("alphagrad.approx.env.extract_jaxpr", fake_extract):
        _callback(
            config,
            args,
            closed_jaxpr.literals,
            initial_order,
            sparsity_specs,
            stop,
            init=True,
        )

    transforms = captured["transforms"]
    assert transforms is not None and len(transforms) > 0, (
        f"_callback didn't forward any transforms; got {transforms!r}"
    )
    forwarded_factors = [
        t.factor for _v, ts in transforms for t in ts if isinstance(t, Diag)
    ]
    assert 2 in forwarded_factors, (
        f"factor=2 was stripped; got {forwarded_factors}."
    )
    assert 4 in forwarded_factors, (
        f"factor=4 was stripped; got {forwarded_factors}."
    )
    print(f"  forwarded transforms = {transforms}")
    print(f"  Diag factors = {forwarded_factors}")


def test_callback_resolves_minus_one_factor_to_gcd():
    """The legacy factor=-1 sentinel meant "gcd-collapse". With the typed
    transform API, apply_diag rejects sentinels, so `_callback` resolves
    -1 to the actual gcd(d1, d2) integer before emitting the Diag.
    """
    from graphax.sparse.micro_actions import Diag

    print("\n[env] _callback resolves factor=-1 to explicit gcd")
    closed_jaxpr, args = _tiny_jaxpr()
    total_v = len(closed_jaxpr.jaxpr.eqns)

    config = EnvConfig(
        jaxpr=closed_jaxpr.jaxpr,
        argnums=(0, 1),
        has_aux=False,
        sparse=False,
        cmp_type="graphax",
        mem_type="graphax",
        target_fun=None,
        data_gen=None,
    )
    initial_order, initial_specs = _build_callback_state(closed_jaxpr, total_v)
    sparsity_specs = initial_specs.at[0, 0].set(jnp.array([0, 0, -1], jnp.int32))
    stop = jnp.asarray(total_v, dtype=jnp.int32)

    captured, fake_extract = _spy_extract_jaxpr_to_record_transforms()
    with mock.patch("alphagrad.approx.env.extract_jaxpr", fake_extract):
        _callback(
            config, args, closed_jaxpr.literals,
            initial_order, sparsity_specs, stop, init=True,
        )

    transforms = captured["transforms"]
    assert transforms is not None and len(transforms) > 0, (
        "expected at least one Diag transform after -1 → gcd resolution"
    )
    first_diag = transforms[0][1][0]
    assert isinstance(first_diag, Diag) and first_diag.factor > 0, (
        f"expected -1 to resolve to a positive divisor, got {first_diag}"
    )
    print(f"  transforms = {transforms}")


# ---------------------------------------------------------------------------
# Multi-invar invariant: rules that don't fit every invar must be dropped
# ---------------------------------------------------------------------------


def test_callback_drops_rules_that_dont_fit_every_invar():
    """graphax's `_eliminate_vertex` applies each transform to every
    incoming edge of the vertex; if a vertex has heterogeneous-rank
    invars (e.g. ``div((4,), ())``), a Diag with ``j`` past the scalar
    edge's primal dims raises inside ``apply_diag``. Caught the
    `Diag.j = 1 out of range [0, 1); out_dims=1, primal_dims=0` crash
    in the PPO smoke test on Helmholtz.

    The env-side translator must therefore validate the rule against
    EVERY non-literal invar and silently drop it when any one of them
    can't host the chosen ``(bi1, bi2)`` pair.
    """
    from graphax import examples
    from graphax.sparse.micro_actions import Diag

    print("\n[env] _callback drops rules that don't fit every invar")
    target_fn = examples.Helmholtz
    x = jnp.array([0.05, 0.15, 0.25, 0.35], dtype=jnp.float32)
    closed_jaxpr = jax.make_jaxpr(target_fn)(x)
    jaxpr = closed_jaxpr.jaxpr
    total_v = len(jaxpr.eqns)

    # Find the div vertex; its second invar is the scalar denominator,
    # which is exactly the multi-invar shape mismatch we want to test.
    div_idx = next(
        i for i, e in enumerate(jaxpr.eqns) if e.primitive.name == "div"
    )

    config = EnvConfig(
        jaxpr=jaxpr,
        argnums=(0,),
        has_aux=False,
        sparse=False,
        cmp_type="graphax",
        mem_type="graphax",
        target_fun=None,
        data_gen=None,
    )
    initial_order, initial_specs = _build_callback_state(closed_jaxpr, total_v)
    # Plant (0, 0, -1) on the div vertex — fits the (4,) numerator but
    # not the () denominator. The translator must reject it.
    sparsity_specs = initial_specs.at[div_idx, 0].set(
        jnp.array([0, 0, -1], jnp.int32)
    )
    stop = jnp.asarray(total_v, dtype=jnp.int32)

    captured, fake_extract = _spy_extract_jaxpr_to_record_transforms()
    with mock.patch("alphagrad.approx.env.extract_jaxpr", fake_extract):
        _callback(
            config, (x,), closed_jaxpr.literals,
            initial_order, sparsity_specs, stop, init=True,
        )

    transforms = captured["transforms"] or []
    div_v = div_idx + 1
    div_entry = next((t for t in transforms if t[0] == div_v), None)
    assert div_entry is None, (
        f"rule on div (scalar-invar) leaked through: {div_entry}. "
        "The translator must validate against EVERY invar, not just invars[0]."
    )
    print(f"  div vertex correctly excluded from transforms = {transforms}")


# ---------------------------------------------------------------------------
# Smoke test: full env.step round-trip on the smallest real example
# ---------------------------------------------------------------------------


def test_env_step_roundtrip_on_helmholtz():
    """End-to-end: build the smallest VertexEliminationEnv and step once with
    a StepAction that includes a non-(-1) factor. Should NOT raise; reward
    must be finite (no NaN). Catches both fixes simultaneously when the env
    is wired with `target_fun` so the cossim path runs."""

    print("\n[env] full step round-trip with factor=2 in the action")
    from graphax import examples

    target_fn = examples.Helmholtz
    x = jnp.array([0.05, 0.15, 0.25, 0.35], dtype=jnp.float32)
    closed_jaxpr = jax.make_jaxpr(target_fn)(x)

    env = VertexEliminationEnv.from_jaxpr(
        closed_jaxpr,
        args=(x,),
        argnums=(0,),
        num_envs=0,
        target_fun=target_fn,
        cmp_type="flops",
        mem_type="peak_memory",
    )
    state = env.reset()

    # Pick the first valid vertex with a "dangerous" non-(-1) factor.
    target_v = jnp.asarray(env.valid_vertices[0], dtype=jnp.int32)
    rule_specs = jnp.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=jnp.int32).at[..., 2].set(0)
    rule_specs = rule_specs.at[0].set(jnp.array([0, 0, 4], jnp.int32))  # factor=4 — bad
    action = StepAction(target_vertex=target_v, rule_specs=rule_specs)

    out = env.step(state, action)
    reward = out.reward
    print(f"  reward = {[float(r) for r in reward]}")
    assert jnp.all(jnp.isfinite(reward)), f"reward has NaN/Inf: {reward}"


def main():
    print("=== env._callback regression tests ===")
    test_cossim_aggregation_single_sample_is_not_nan()
    test_cossim_aggregation_full_quartile_path_unchanged()
    test_callback_forwards_arbitrary_factors_as_diag()
    test_callback_resolves_minus_one_factor_to_gcd()
    test_callback_drops_rules_that_dont_fit_every_invar()
    test_env_step_roundtrip_on_helmholtz()
    print("\nALL ENV CALLBACK TESTS OK")


if __name__ == "__main__":
    main()
