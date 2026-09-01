"""A4 -- seeds are NOT vertices. Two conventions, pinned.

This file pins the two decisions of workstream A4 so neither can drift back:

  1. ``--seed-vertices`` is DEPRECATED, no launcher passes it, and it is
     kept ACCEPTED only so an archived run's graph can be rebuilt. It warns
     when used. It used to ALSO require ``--measure-grad``; that requirement
     is gone, because ``--measure-grad`` is now itself a no-op and an error
     message naming a no-op as the fix is worse than no error. See
     ``test_seed_vertices_no_longer_requires_measure_grad``.

  2. THE GRADIENT-COVERAGE LEAF SET excludes 0-d leaves, agreeing with
     ``env._walk_argnums``. Before A4 the two contradicted each other:
     ``_walk_argnums`` excluded the 0-d tangent seed from the Adam walk while
     ``_leaf_norms`` counted it, which put ``--reject-frozen-grads`` (default
     ON) in a position to sentinel a whole plan on the SEED's gradient.

``--measure-grad`` IS now redundant -- a DEPRECATED NO-OP kept accepted for
the 372 files that name it -- and the last two tests record that: it moves
nothing, and it is still an accepted CLI option.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_MAX_EQNS", "512")
# This file drives the guard's INPUTS directly; never inherit an armed guard.
os.environ["ALPHAGRAD_GRAD_COVERAGE"] = "0"
os.environ["ALPHAGRAD_REJECT_FROZEN_GRADS"] = "0"

import math                                                     # noqa: E402
import pathlib                                                  # noqa: E402
from types import SimpleNamespace as NS                         # noqa: E402

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

from alphagrad.approx.common.examples import (                  # noqa: E402
    grad_target_setup, grad_target_fn, infer_argnums)
from alphagrad.approx.env import (                              # noqa: E402
    _leaf_norms, _grad_coverage, _walk_argnums)

EXAMPLE = "MLP"          # any name; only infer_argnums() consumes it here


def _fn(*a):
    return a[0]


# --------------------------------------------------------------------------
# 1. the coupling is loud
# --------------------------------------------------------------------------

def test_seed_vertices_no_longer_requires_measure_grad():
    """The gate that WAS here demanded ``--measure-grad`` alongside, so that a
    replay rebuilt the archived QUALITY CHANNEL (loss_drop) as well as the
    archived graph. That condition no longer exists: the traced target is
    unconditionally the scalar loss, so loss_drop is the DEFAULT channel for
    every target that has one and the replay gets it without the flag.

    Requiring a flag that does nothing would make the error message false, so
    the requirement is dropped and ``--seed-vertices`` stands alone.
    """
    a = NS(measure_grad=False, seed_vertices=True)
    xs = (np.zeros((2, 2)),)
    fn, out_xs, argnums = grad_target_setup(a, _fn, xs, EXAMPLE)
    assert len(out_xs) == len(xs) + 1          # the tangent seed is appended
    assert grad_target_fn(a, _fn, EXAMPLE) is not _fn


def test_seed_vertices_no_longer_requires_measure_grad_dict():
    """The CPU measure-actor passes a dict, not a Namespace -- and it MUST
    build the identical graph, so it must take the identical branch."""
    a = {"measure_grad": False, "seed_vertices": True}
    xs = (np.zeros((2, 2)),)
    fn, out_xs, argnums = grad_target_setup(a, _fn, xs, EXAMPLE)
    assert len(out_xs) == len(xs) + 1
    assert grad_target_fn(a, _fn, EXAMPLE) is not _fn


def test_no_seed_vertices_is_the_default_and_adds_nothing():
    """The A4 launcher flag set: --measure-grad, no --seed-vertices.

    No appended argument, no appended argnum -- i.e. none of the 2 extra
    eliminable vertices (tangent-seed add, adjoint reduce_sum).
    """
    xs = (np.zeros((2, 2)), np.zeros((2,)))
    a = NS(measure_grad=True, seed_vertices=False)
    fn, out_xs, argnums = grad_target_setup(a, _fn, xs, EXAMPLE)
    assert len(out_xs) == len(xs)
    assert tuple(argnums) == tuple(infer_argnums(EXAMPLE))
    # ...and nothing is wrapped around the target either: the registered
    # target IS model + loss, so grad_target_fn is the identity here.
    assert grad_target_fn(a, _fn, EXAMPLE) is _fn


def test_seed_vertices_with_measure_grad_still_replays():
    """Kept ACCEPTED (not deleted) so landscape_map / ls_verify can rebuild an
    archived run's graph. The seed layout is (weights..., tangent seed t) with
    t appended as a 0-d arg -- the layout ``_walk_argnums`` and ``_leaf_norms``
    both key off."""
    xs = (np.zeros((2, 2)), np.zeros((2,)))
    a = NS(measure_grad=True, seed_vertices=True)
    fn, out_xs, argnums = grad_target_setup(a, _fn, xs, EXAMPLE)
    assert len(out_xs) == len(xs) + 1
    assert jnp.ndim(out_xs[-1]) == 0                       # the seed is 0-d
    assert tuple(argnums)[-1] == len(xs)


# --------------------------------------------------------------------------
# 2. the coverage leaf-set convention
# --------------------------------------------------------------------------

def test_leaf_norms_marks_0d_leaves_uncountable():
    """0-d leaf -> nan, not a norm and not 0.0.

    nan (rather than dropping the entry) keeps positions aligned with
    ``config.argnums``, which the forensics records index by; nan (rather than
    0.0) keeps the logged norms honest -- an exclusion is not a measured zero.
    """
    out = (jnp.ones((3, 4)), jnp.asarray(2.0))
    n = _leaf_norms(out, has_aux=False)
    assert len(n) == 2
    assert n[0] == pytest.approx(math.sqrt(12.0), rel=1e-6)
    assert math.isnan(n[1])


def test_coverage_leaf_set_agrees_with_walk_argnums():
    """THE POINT OF A4 (3): one convention, stated in two places, now equal.

    ``_walk_argnums`` drops 0-d argnums from the Adam walk because stepping the
    tangent seed moves every weight by ``t*ones`` and saturates the net. The
    coverage leaf set must make the SAME exclusion or the guard is scoring a
    slot the rest of the system has already declared not-a-weight.
    """
    cfg = NS(argnums=(0, 1, 2))
    base_args = [np.zeros((2, 2)), np.zeros((3,)), np.zeros(())]
    walk = _walk_argnums(cfg, base_args)
    assert walk == (0, 1)

    norms = _leaf_norms(
        (jnp.zeros((2, 2)), jnp.zeros((3,)), jnp.asarray(0.0)), has_aux=False)
    countable = tuple(i for i, v in enumerate(norms) if not math.isnan(v))
    assert countable == walk


def test_seed_leaf_cannot_sentinel_a_plan():
    """The guard consequence, stated as a behaviour.

    ``--reject-frozen-grads`` (default ON) sentinels the WHOLE plan as soon as
    ``frac_zeroed > 0``. A zeroed SEED leaf must not be able to do that; a
    zeroed WEIGHT leaf still must.
    """
    # weight leaf healthy, 0-d seed leaf zeroed by the plan -> guard inert
    cov = _grad_coverage([1.0, 0.0], [1.0, float("nan")])
    assert cov["defined"] is True
    assert cov["n_counted"] == 1
    assert cov["n_uncountable"] == 1
    assert cov["frac_zeroed"] == 0.0
    assert cov["min_leaf_ratio"] == pytest.approx(1.0)

    # the SAME numbers with the seed counted (the pre-A4 convention) would
    # have destroyed the plan -- this is what changed, and nothing else did
    cov_old = _grad_coverage([1.0, 0.0], [1.0, 1.0])
    assert cov_old["frac_zeroed"] == pytest.approx(0.5)

    # a genuinely frozen WEIGHT leaf still fires with the new convention
    cov_real = _grad_coverage([0.0, 1.0], [1.0, float("nan")])
    assert cov_real["frac_zeroed"] == pytest.approx(1.0)


@pytest.mark.parametrize("exact", [
    [float("nan"), 1.0, 5.0],      # nan first
    [1.0, float("nan"), 5.0],      # nan middle
    [1.0, 5.0, float("nan")],      # nan last
])
def test_eps_scale_is_not_poisoned_by_the_nan_leaf(exact):
    """``max()`` over a list containing nan is ORDER-DEPENDENT in Python
    (``max([1.0, nan]) == 1.0`` but ``max([nan, 1.0]) is nan``), so the eps
    scale must be taken over the finite entries only -- otherwise the leaf-set
    convention would silently make eps nan and uncount everything."""
    cov = _grad_coverage([1.0, 1.0, 1.0], exact)
    assert cov["defined"] is True
    assert cov["n_counted"] == 2
    assert math.isfinite(cov["min_leaf_ratio"])
    assert cov["min_leaf_ratio"] == pytest.approx(min(1.0, 1.0 / 5.0))


def test_all_leaves_uncountable_is_undefined_not_a_verdict():
    """Fail-soft is preserved: if the convention leaves nothing countable the
    record is ``defined: False`` and the guard can never fire on it."""
    cov = _grad_coverage([1.0], [float("nan")])
    assert cov["defined"] is False
    assert cov["frac_zeroed"] == 0.0
    assert cov["n_uncountable"] == 1


# --------------------------------------------------------------------------
# 3. regression fence + the record on --measure-grad
# --------------------------------------------------------------------------

def test_no_in_repo_launcher_passes_seed_vertices():
    """A4 (1): no launcher sets the flag. Comments may still discuss it."""
    root = pathlib.Path(__file__).resolve().parents[4]
    offenders = []
    for p in sorted(list(root.glob("*.sh")) + list(root.glob("*.sbatch"))):
        for i, line in enumerate(
                p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "--seed-vertices" in line:
                offenders.append("%s:%d" % (p.name, i))
    assert not offenders, "launchers still pass --seed-vertices: %s" % offenders


def test_measure_grad_is_a_deprecated_no_op():
    """SUPERSEDED A4's third finding, deliberately, and then the other two.

    A4 recorded ``--measure-grad`` as load-bearing in three ways, and none
    survives:

      * "``scalar_loss_fn`` wraps the target BEFORE tracing" -- the registered
        target IS model + loss now (``common.examples.get_fn``), so there is
        nothing to wrap and no other mode to select. ``grad_target_setup`` is
        the IDENTITY here.
      * "it flips the quality-metric default to ``loss_drop``" -- that default
        now follows ``EnvConfig.scalar_target``, a fact read off the traced
        jaxpr, which is true for exactly the targets on which the loss-drop
        walk is defined.
      * "it gates ``--seed-vertices``" -- see the first test in this file.

    ``test_jacobian_equals_grad.py`` owns the positive statement (every
    registered target is scalar and ``jacve`` of it == ``jax.grad``); this
    asserts only that the flag moves nothing.
    """
    xs = (np.zeros((2, 2)),)
    off = grad_target_setup(NS(measure_grad=False, seed_vertices=False),
                            _fn, xs, EXAMPLE)
    on = grad_target_setup(NS(measure_grad=True, seed_vertices=False),
                           _fn, xs, EXAMPLE)
    assert off[0] is _fn                 # the identity...
    assert on[0] is _fn                  # ...with the flag on OR off
    probe = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(np.asarray(off[0](probe)),
                                  np.asarray(on[0](probe)))
    assert off[1:] == on[1:]             # same args, same argnums


def test_seed_vertices_is_still_an_accepted_cli_option():
    """A4 kept the flag ACCEPTED for archived-run replay; that must not rot
    into "removed" the next time someone tidies the argparser."""
    import argparse
    from alphagrad.approx.args_ppo import add_ppo_args
    opts = set()
    for a in add_ppo_args(argparse.ArgumentParser())._actions:
        opts.update(a.option_strings)
    assert "--seed-vertices" in opts
    assert "--measure-grad" in opts
