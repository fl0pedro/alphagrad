"""Drive real eliminations with a masked chooser in every face slot.

This is the end-to-end claim the per-edge masks exist to support: an exploring
policy picks a micro-action for each of lhs / rhs / res on every local path,
and the run neither crashes nor loses provenance.

Two things are asserted that unit tests on the mask alone cannot show:

* nothing raises. The lhs/rhs/res operands are join INTERMEDIATES, so their
  index structure is unknowable before ``eliminate`` runs -- legality has to be
  decided inside the slot, against the live tensor.
* the choices are RECORDED. graphax's older tensor-returning callable path
  applied a transform without ever logging it; returning the chosen action
  routes it through the same record path as a literal one.
"""
import jax
import jax.numpy as jnp
import pytest
from graphax import IncrementalJaxpr

from alphagrad.approx.common.masks import masked_micro_chooser

CASES = [
    ("sin_mul", lambda x, y: jnp.sin(x * y) + x, (jnp.ones((4, 4)), jnp.ones((4, 4)))),
    ("chain", lambda x, y: jnp.tanh(x @ y), (jnp.ones((4, 6)), jnp.ones((6, 4)))),
    ("mix", lambda x, y: jnp.sum(jnp.exp(x) * y), (jnp.ones((6,)), jnp.ones((6,)))),
]


def _drive(fn, args, order, pick):
    cj = jax.make_jaxpr(fn)(*args)
    chooser = masked_micro_chooser(pick)
    incr = IncrementalJaxpr(cj.jaxpr, argnums=(0, 1), consts=cj.literals, args=args)
    n_faces = 0
    for v in order:
        keys = list(incr.faces(int(v)))
        n_faces += len(keys)
        incr.eliminate(
            int(v), face_transforms={k: (chooser, chooser, chooser) for k in keys})
    return incr, n_faces


def _orders(fn, args):
    n = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
    return [("fwd", list(range(1, n + 1))), ("rev", list(range(n, 0, -1)))]


@pytest.mark.parametrize("name,fn,args", CASES, ids=[c[0] for c in CASES])
def test_every_slot_choosing_an_action_never_raises_and_stays_finite(name, fn, args):
    for label, order in _orders(fn, args):
        incr, n_faces = _drive(fn, args, order, lambda st, acts: acts[0] if acts else None)
        assert n_faces > 0, f"{name}/{label}: no faces to drive"

        # Evaluate through the recovered jaxpr -- the AOJ's tensors are traced,
        # so reaching into `.val` from out here leaks a tracer.
        jx, consts, _ = incr.current_jaxpr()
        outs = jax.jit(lambda *a: jax.core.eval_jaxpr(jx, consts, *a))(*args)
        for leaf in jax.tree_util.tree_leaves(outs):
            assert bool(jnp.all(jnp.isfinite(leaf))), f"{name}/{label} non-finite"


@pytest.mark.parametrize("name,fn,args", CASES, ids=[c[0] for c in CASES])
def test_chosen_actions_are_recorded(name, fn, args):
    for label, order in _orders(fn, args):
        offered = []

        def pick(st, actions, _o=offered):
            _o.append(len(actions))
            return actions[0] if actions else None

        incr, _ = _drive(fn, args, order, pick)
        assert offered, f"{name}/{label}: the chooser was never consulted"
        n_chosen = sum(1 for k in offered if k > 0)
        assert n_chosen > 0, f"{name}/{label}: no slot had a legal action"
        assert len(incr.transform_records()) >= n_chosen, (
            f"{name}/{label}: chose {n_chosen} actions but recorded "
            f"{len(incr.transform_records())} -- provenance was dropped")


def test_a_slot_with_no_legal_action_is_left_exact():
    """`pick` returning None must leave the operand untouched, not error."""
    fn, args = CASES[0][1], CASES[0][2]
    incr, n_faces = _drive(fn, args, _orders(fn, args)[0][1], lambda st, acts: None)
    assert n_faces > 0
    assert incr.transform_records() == [], "nothing was chosen, so nothing recorded"
