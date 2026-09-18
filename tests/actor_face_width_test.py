"""The measure actor must derive the trainer's face width (dsnn-dfw.36).

``ppo.main`` derives the per-graph face bound and calls
``env.configure_max_faces`` before any shape is built from it. The measure
actor rebuilds the SAME graph in another process through
``cpu_approx_worker._build_env_from_args`` and used to leave ``env.MAX_FACES``
at the module default 16, so every terminal measurement of a vertex with more
faces than that raised "N faces exceed the derived bound 16" while the
trainer's own bound was far larger.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax.random as jrand
import pytest

import alphagrad.approx.env as E
from alphagrad.approx.common import get_args, get_fn
from alphagrad.approx.common.examples import grad_target_setup
from alphagrad.approx.cpu_approx_worker import (
    _build_env_from_args, _traced_inlined,
)

EXAMPLE = "Perceptron"
MODULE_DEFAULT = 16


def _args_dict(**over):
    d = dict(example=EXAMPLE, dataset="none", dataset_size=1,
             cmp_type="graphax", mem_type="graphax", seed=0,
             face_actions=True, per_face=True, rewards=["cmp"],
             intermediate_rewards=False)
    d.update(over)
    return d


def _derived_bound():
    fn = get_fn(EXAMPLE)
    xs = get_args(EXAMPLE, jrand.PRNGKey(0), dataset=None)
    ns = SimpleNamespace(**_args_dict())
    fn, xs, argnums = grad_target_setup(ns, fn, xs, EXAMPLE)
    cj = _traced_inlined(fn, xs)
    return int(E.derived_max_faces(cj.jaxpr, argnums, cj.literals, xs))


@pytest.fixture(autouse=True)
def _restore_width(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_MAX_FACES", raising=False)
    keep = E.MAX_FACES
    yield
    E.MAX_FACES = keep


def test_the_actor_env_build_configures_the_derived_face_width():
    want = _derived_bound()
    assert want != MODULE_DEFAULT, (
        f"{EXAMPLE} derives a face bound of {want}, which is the module "
        f"default: this test cannot tell the two apart. Pick an example "
        f"whose bound is not {MODULE_DEFAULT}.")
    E.MAX_FACES = MODULE_DEFAULT
    _build_env_from_args(_args_dict(), None)
    assert E.MAX_FACES == want


def test_the_actor_env_build_leaves_the_width_alone_without_face_actions():
    E.MAX_FACES = MODULE_DEFAULT
    _build_env_from_args(_args_dict(face_actions=False, per_face=False), None)
    assert E.MAX_FACES == MODULE_DEFAULT


def test_an_explicit_face_width_override_wins():
    want = _derived_bound()
    os.environ["ALPHAGRAD_MAX_FACES"] = str(want + 7)
    try:
        E.MAX_FACES = want + 7
        _build_env_from_args(_args_dict(), None)
        assert E.MAX_FACES == want + 7
    finally:
        os.environ.pop("ALPHAGRAD_MAX_FACES", None)
