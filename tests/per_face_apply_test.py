"""C3: per-face application, the per-path skip, and quant contraction coupling.

Per-VERTEX application hands graphax one rule list that lands uniformly on every
face, so a rule must fit all of them (or it raises / gets masked away for the
whole vertex). Per-FACE wraps the rules in a callable graphax invokes once per
face with that face's live operand, so a rule lands only where it is legal —
and a face where nothing is legal is left exact, which IS the per-path skip.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.micro_actions import Compress, Diag, Quant

from alphagrad.approx.common.masks import (
    couple_quant_rules,
    make_live_masked_hook,
    rule_is_legal,
)


# --------------------------------------------------------------------------- #
# quant contraction coupling
# --------------------------------------------------------------------------- #

def test_second_quant_in_a_turn_inherits_the_first_dtype():
    """int8 only contracts with int8, so the post operand must inherit rather
    than pick freely (one quantization decision per turn)."""
    rules = (Quant(dtype="int8"), Quant(dtype="float32"))
    coupled, in_force = couple_quant_rules(rules)
    assert in_force == "int8"
    assert [r.dtype for r in coupled] == ["int8", "int8"]


def test_coupling_keeps_the_policys_sign_choice():
    rules = (Quant(dtype="int8"), Quant(dtype="uint8", scale_sign=-1))
    coupled, _ = couple_quant_rules(rules)
    assert coupled[1].dtype == "int8"
    assert getattr(coupled[1], "scale_sign", 1) == -1


def test_coupling_is_a_noop_for_consistent_or_nonquant_rules():
    rules = (Diag(0, 1, 2), Quant(dtype="int8"), Quant(dtype="int8"))
    coupled, in_force = couple_quant_rules(rules)
    assert in_force == "int8"
    assert isinstance(coupled[0], Diag)
    assert [type(r) for r in coupled] == [type(r) for r in rules]


def test_dtype_already_in_force_from_earlier_in_the_turn_is_respected():
    coupled, in_force = couple_quant_rules((Quant(dtype="float32"),),
                                           applied_dtype="int8")
    assert in_force == "int8" and coupled[0].dtype == "int8"


# --------------------------------------------------------------------------- #
# per-face application
# --------------------------------------------------------------------------- #

def _mlp(x, W1, W2):
    return jnp.tanh(x @ W1) @ W2


_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1)
_ARGNUMS = (1, 2)


def _jac(transforms):
    from graphax import jacve
    return jacve(_mlp, "rev", argnums=_ARGNUMS, transforms=transforms)(*_ARGS)


def _flat(j):
    return np.concatenate([np.ravel(np.asarray(l))
                           for l in jax.tree_util.tree_leaves(j)])


def test_per_face_hook_never_raises_and_stays_finite():
    """The hook applies a rule only where legal, so it cannot produce the
    TRANSFORM DID NOT FIT that a uniform per-vertex list can."""
    cj = jax.make_jaxpr(_mlp)(*_ARGS)
    nv = len(cj.jaxpr.eqns)
    stats: dict = {}
    # A deliberately over-broad rule set: most of these fit only some faces.
    rules = (Diag(0, 2, 2), Compress((0,), "mean"), Quant(dtype="float32"))
    spec = [(v, (make_live_masked_hook(rules, stats=stats),))
            for v in range(1, nv + 1)]
    out = _jac(spec)
    flat = _flat(out)
    assert np.all(np.isfinite(flat)), "per-face application must stay finite"
    assert stats, "the hook should have been consulted"
    # Both outcomes are expected: some faces take the rule, others skip it.
    assert stats.get("applied", 0) + stats.get("skipped", 0) > 0


def test_per_face_differs_from_exact():
    """If per-face application changed nothing it would be pointless."""
    cj = jax.make_jaxpr(_mlp)(*_ARGS)
    nv = len(cj.jaxpr.eqns)
    exact = _flat(_jac(None))
    rules = (Diag(0, 2, 2),)
    spec = [(v, (make_live_masked_hook(rules),)) for v in range(1, nv + 1)]
    approx = _flat(_jac(spec))
    assert exact.shape == approx.shape
    assert not np.allclose(exact, approx), (
        "per-face application produced a byte-identical Jacobian — the rules "
        "never landed anywhere"
    )


def test_a_face_with_no_legal_rule_is_left_exact():
    """The per-path SKIP: an all-illegal rule set must leave the Jacobian
    untouched rather than raise."""
    cj = jax.make_jaxpr(_mlp)(*_ARGS)
    nv = len(cj.jaxpr.eqns)
    exact = _flat(_jac(None))
    stats: dict = {}
    # Diag with a factor that divides nothing here -> illegal on every face.
    impossible = (Diag(0, 1, 7919),)     # 7919 is prime, divides no dim
    spec = [(v, (make_live_masked_hook(impossible, stats=stats),))
            for v in range(1, nv + 1)]
    out = _flat(_jac(spec))
    assert np.allclose(exact, out), "an all-illegal rule set must be a no-op"
    assert stats.get("applied", 0) == 0
    assert stats.get("skipped", 0) > 0


def test_env_exposes_the_per_face_switch():
    from alphagrad.approx.env import EnvConfig, consume_per_face_stats
    from alphagrad.approx.ppo import make_argparser

    cfg = EnvConfig(jaxpr=None, argnums=(), has_aux=False, sparse=False,
                    cmp_type="flops", mem_type="peak_memory")
    assert cfg.per_face is False, "per-face must be opt-in"
    a = make_argparser().parse_args(["--example", "X", "--per-face"])
    assert a.per_face is True
    assert "applied_fraction" in consume_per_face_stats()


# --------------------------------------------------------------------------- #
# Observation token budget (the nn256 truncation)
# --------------------------------------------------------------------------- #

def _run_env_import(extra_env):
    """Import ``env`` in a FRESH interpreter with EXACTLY ``extra_env`` of the
    project's own knobs set, and report what it froze.

    THE CHILD ENVIRONMENT IS SCRUBBED OF EVERY ``ALPHAGRAD_*`` VARIABLE, and
    that is the whole point of the helper rather than an optimisation. These
    cases assert what ``env.py`` freezes FROM A GIVEN CONFIGURATION, including
    the empty one ("the default is 32768"). It used to pass ``dict(os.environ)``
    through, so under ``pytest tests/`` the child inherited whatever any module
    had assigned at ITS import: pytest imports every test module before running
    the first test, so by the time this test RUNS the variable holds the last
    such write. Two modules wrote it -- ``delta_obs_emission_test`` 1024 and
    ``policy_regression_gate.py`` 4096 -- and the value the child actually
    inherited was **4096** (measured: job 64984 E6 printed the post-collection
    environment). Either value makes the default case assert 32768 against a
    smaller number and fail. Run alone, nothing had assigned it and the same
    case passed. A test of a default must not be able to inherit one.

    ``JAX_*`` / ``XLA_*`` / ``CUDA_*`` and PATH-like variables are kept: the
    child still has to find an interpreter and a CPU backend.
    """
    import os
    import subprocess
    import sys
    src = (
        "from alphagrad.approx.env import "
        "LEGACY_STREAM_TOKENS, MAX_DELTA_TOKENS;"
        "print(LEGACY_STREAM_TOKENS, MAX_DELTA_TOKENS)"
    )
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env["ALPHAGRAD_DISABLE_RESOURCE_MONITOR"] = "1"
    env.update(extra_env)
    return subprocess.run([sys.executable, "-c", src], capture_output=True,
                          text=True, env=env, timeout=300)


def test_total_stream_cap_is_gone_and_setting_it_is_a_hard_error():
    """There is no total-stream token budget any more.

    The encoder is a recurrence -- the base stream is consumed once into a
    fixed ``(L, H, d, d)`` carry and every step after it is an extend by that
    step's DELTA -- so the whole stream is never materialised and a cap on its
    length buys nothing. It also actively harmed: the slice kept the OLDEST
    tokens, so a saturated buffer froze the observation for the rest of the
    episode. The knob must be a HARD ERROR rather than a silent no-op,
    because three smoke scripts used to set it.
    """
    out = _run_env_import({"ALPHAGRAD_MAX_TOKENS": "8192"})
    assert out.returncode != 0, "ALPHAGRAD_MAX_TOKENS must not be accepted"
    assert "ALPHAGRAD_MAX_TOKENS is GONE" in out.stderr


def test_delta_budget_is_the_only_bound_and_both_widths_are_settable():
    """The per-step DELTA buffer is the one bound JAX's static shapes need."""
    out = _run_env_import({"ALPHAGRAD_LEGACY_STREAM_TOKENS": "8192",
                           "ALPHAGRAD_MAX_DELTA_TOKENS": "16384"})
    assert out.returncode == 0, out.stderr[-600:]
    legacy, delta = out.stdout.strip().split()[-2:]
    assert int(legacy) == 8192
    assert int(delta) == 16384

    # Default: sized from the measured TLM distribution (worst single delta
    # 25737 tokens), not from a guess.
    out = _run_env_import({})
    assert out.returncode == 0, out.stderr[-600:]
    assert int(out.stdout.strip().split()[-1]) == 32768


def test_delta_overflow_is_never_silent():
    """A delta that does not fit raises by default; ``clip`` prints."""
    from alphagrad.approx import env as _env

    big = _env.MAX_DELTA_TOKENS + 1
    with pytest.raises(ValueError, match="token DELTA truncated"):
        _env._record_delta_truncation(big)
