"""The gradient oracle's precision context, its reference cache and its help
text (ticket dsnn-df8).

Three pins. None of them needs a GPU.

1. BOTH SIDES RUN AT "highest". The defect the ticket names is a float32
   matmul default: on this hardware a float32 dot is TF32, and a TF32
   ``jax.grad`` reference sits 1.054e-3 from the float64 truth while the bar
   it guards is 1e-3. The measurement that proves the fix needs a GPU (on the
   CPU there is no TF32, so the two precisions agree bit for bit and no CPU
   test can fail for the right reason). What a CPU test CAN pin is that the
   context is actually active on BOTH sides -- the elimination and
   ``jax.grad`` -- which is exactly what was missing. The precision reader is
   monkeypatched and must report "highest" twice.

2. THE REFERENCE IS COMPUTED ONCE PER PROBE BATCH. ``jax.grad`` of the target
   does not depend on the elimination order, so a second order must HIT the
   cache; a new probe batch (a new episode under ``--walk-rotate``) must MISS
   it.

3. THE HELP STATES THE TOLERANCE THE CODE USES. The ``--grad-oracle`` help
   said "rel L2 > 1e-4" and "aborts the run" while the code used 1e-3 and
   refused the order and carried on. Both halves are pinned against the
   constant itself, so the help cannot drift again.
"""
import types

import jax
import jax.numpy as jnp
import pytest

import alphagrad.approx.env as envmod


# --------------------------------------------------------------- the target
# Small, but with TWO real matmuls, so the elimination has a contraction to
# get wrong and the matmul precision is not a dead letter.
_X = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
_Y = jax.random.normal(jax.random.PRNGKey(2), (8, 3))
_W1 = jax.random.normal(jax.random.PRNGKey(3), (4, 6)) * 0.5
_W2 = jax.random.normal(jax.random.PRNGKey(4), (6, 3)) * 0.5

_ARGS = [_X, _Y, _W1, _W2]
_ARGNUMS = (2, 3)


def _target(x, y, w1, w2):
    h = jnp.tanh(x @ w1)
    return jnp.sum((h @ w2 - y) ** 2)


def _data_gen(keys):
    """Two probe slots, drawn from the keys `_probe_batch` splits for us."""
    return (jax.random.normal(keys[0], (8, 4)),
            jax.random.normal(keys[1], (8, 3)))


def _complete_order():
    """A complete elimination order of ``_target``, built the way
    ``tests/rule_replay_test.py`` builds one."""
    from graphax.core import _build_graph
    cj = jax.make_jaxpr(_target)(*_ARGS)
    _, _, _, vo = _build_graph(cj.jaxpr, _ARGS, list(cj.literals), _ARGNUMS)
    valid = [i for i, eqn in enumerate(cj.jaxpr.eqns, 1)
             if eqn.outvars[0] not in cj.jaxpr.outvars or i in vo]
    return list(reversed(valid))


#: The oracle only needs to know that an exact executable EXISTS -- it builds
#: its own at "highest" (see ``env._grad_oracle_exact``), so a marker is
#: enough here and a wrongly reused executable would show up as a TypeError.
_EXACT_EXISTS = object()


@pytest.fixture
def cfg(monkeypatch):
    """A fresh oracle state and one config shared by the whole test."""
    monkeypatch.delenv("ALPHAGRAD_GRAD_ORACLE", raising=False)
    monkeypatch.delenv("ALPHAGRAD_GRAD_ORACLE_TOL", raising=False)
    monkeypatch.delenv("ALPHAGRAD_WALK_ROTATE", raising=False)
    monkeypatch.delenv("ALPHAGRAD_WALK_EPISODE", raising=False)
    envmod._GRAD_ORACLE_DONE.clear()
    envmod._GRAD_ORACLE_REF.clear()
    envmod._GRAD_ORACLE_REF_STATS.update(hits=0, misses=0)
    envmod._GRAD_ORACLE_LAST_PRECISION.update(plan=None, reference=None)
    envmod._PROBE_BATCH.clear()
    return types.SimpleNamespace(
        target_fun=_target, argnums=_ARGNUMS, has_aux=False, sparse=True,
        scalar_target=True, data_gen=_data_gen)


# ------------------------------------------------ 1. the precision context
def test_both_sides_of_the_oracle_run_at_the_oracle_precision(cfg, monkeypatch):
    """The reader is called once per side and must report "highest" twice.

    The ambient precision is asserted NOT to be "highest" first, so a green
    test cannot come from a process that happens to be configured that way.
    """
    assert envmod._matmul_precision() != envmod._GRAD_ORACLE_PRECISION

    real = envmod._matmul_precision
    seen = []

    def spy():
        seen.append(real())
        return seen[-1]

    monkeypatch.setattr(envmod, "_matmul_precision", spy)
    envmod._grad_oracle_check(cfg, _EXACT_EXISTS, list(_ARGS), None,
                              _complete_order())

    assert seen == [envmod._GRAD_ORACLE_PRECISION,
                    envmod._GRAD_ORACLE_PRECISION], (
        "the oracle must read the live matmul precision once before the "
        f"elimination and once before jax.grad; got {seen}")
    assert envmod._GRAD_ORACLE_LAST_PRECISION == {
        "plan": envmod._GRAD_ORACLE_PRECISION,
        "reference": envmod._GRAD_ORACLE_PRECISION}
    # The context is a context: it must not leak past the check.
    assert envmod._matmul_precision() != envmod._GRAD_ORACLE_PRECISION


def test_the_check_passes_on_an_exact_order(cfg):
    """The bar is the shipped one and an exact order clears it, so the two
    cache tests below are exercising a check that actually ran."""
    order = _complete_order()
    envmod._grad_oracle_check(cfg, _EXACT_EXISTS, list(_ARGS), None, order)
    assert (tuple(order), str(None)) in envmod._GRAD_ORACLE_DONE
    assert envmod._GRAD_ORACLE_STATS["checks"] > 0


# ------------------------------------------------- 2. the reference cache
def test_the_reference_is_computed_once_and_reused_by_the_next_order(cfg):
    order = _complete_order()
    envmod._grad_oracle_check(cfg, _EXACT_EXISTS, list(_ARGS), None, order)
    assert envmod._GRAD_ORACLE_REF_STATS == {"hits": 0, "misses": 1}

    other = list(reversed(order))
    assert other != order, "need two DIFFERENT orders to test the reuse"
    envmod._grad_oracle_check(cfg, _EXACT_EXISTS, list(_ARGS), None, other)
    assert envmod._GRAD_ORACLE_REF_STATS == {"hits": 1, "misses": 1}, (
        "jax.grad does not depend on the elimination order, so the second "
        "order must reuse the first order's reference")
    assert len(envmod._GRAD_ORACLE_REF) == 1


def test_a_new_probe_batch_misses_the_reference_cache(cfg, monkeypatch):
    """A new episode under ``--walk-rotate`` draws a NEW probe batch, and a
    reference computed on the old batch is then the wrong answer."""
    order = _complete_order()
    envmod._grad_oracle_check(cfg, _EXACT_EXISTS, list(_ARGS), None, order)
    assert envmod._GRAD_ORACLE_REF_STATS == {"hits": 0, "misses": 1}

    monkeypatch.setenv("ALPHAGRAD_WALK_ROTATE", "1")
    monkeypatch.setenv("ALPHAGRAD_WALK_EPISODE", "1")
    other = list(reversed(order))
    envmod._grad_oracle_check(cfg, _EXACT_EXISTS, list(_ARGS), None, other)
    assert envmod._GRAD_ORACLE_REF_STATS == {"hits": 0, "misses": 2}, (
        "a new probe batch must invalidate the cached reference")
    assert len(envmod._GRAD_ORACLE_REF) == 2


# ------------------------------------------------------- 3. the help text
def test_the_grad_oracle_help_states_the_tolerance_the_code_uses():
    from alphagrad.approx.ppo import make_argparser

    actions = [a for a in make_argparser()._actions
               if "--grad-oracle" in (a.option_strings or ())]
    assert len(actions) == 1
    help_text = actions[0].help

    assert f"{envmod.grad_oracle_tol():.0e}" in help_text, help_text
    assert "1e-4" not in help_text, help_text
    # A refusal refuses ONE ORDER. It does not abort the run: the exception is
    # a plain RuntimeError, the measure path sentinels that plan and records
    # it as refused, and the next plan is measured.
    assert "abort" not in help_text.lower(), help_text
    assert "continue" in help_text.lower(), help_text


def test_the_tolerance_constant_is_the_shipped_bar():
    assert envmod._GRAD_ORACLE_TOL == 1e-3
    assert envmod.grad_oracle_tol() == 1e-3


def test_the_tolerance_env_override_reaches_the_help(monkeypatch):
    from alphagrad.approx.ppo import make_argparser

    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_TOL", "2e-3")
    assert envmod.grad_oracle_tol() == 2e-3
    action = [a for a in make_argparser()._actions
              if "--grad-oracle" in (a.option_strings or ())][0]
    assert "2e-03" in action.help, action.help
