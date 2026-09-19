"""The measure actor must derive the trainer's face width (dsnn-dfw.36).

``ppo.main`` derives the per-graph face bound and calls
``env.configure_max_faces`` before any shape is built from it. The measure
actor rebuilds the SAME graph in another process through
``cpu_approx_worker._build_env_from_args`` and used to leave ``env.MAX_FACES``
at the module default 16, so every terminal measurement of a vertex with more
faces than that raised "N faces exceed the derived bound 16" while the
trainer's own bound was 1920 on the 3-block TransformerLM.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

import alphagrad.approx.env as E
from alphagrad.approx.cpu_approx_worker import _build_env_from_args

EXAMPLE = "NeuralNetwork"
MODULE_DEFAULT = 16


def _args_dict(**over):
    d = dict(example=EXAMPLE, dataset="none", dataset_size=1,
             cmp_type="graphax", mem_type="graphax", seed=0,
             face_actions=True, per_face=True, rewards=["cmp"],
             intermediate_rewards=False)
    d.update(over)
    return d


def _bound_of(env):
    return int(E.derived_max_faces(
        env.config.jaxpr, env.config.argnums, env.consts, env.args))


@pytest.fixture(autouse=True)
def _restore_width(monkeypatch):
    monkeypatch.delenv("ALPHAGRAD_MAX_FACES", raising=False)
    keep = E.MAX_FACES
    yield
    E.MAX_FACES = keep


def test_the_actor_env_build_configures_the_derived_face_width():
    E.MAX_FACES = MODULE_DEFAULT
    env = _build_env_from_args(_args_dict(), None)
    want = _bound_of(env)
    assert want != MODULE_DEFAULT, (
        f"{EXAMPLE} derives a face bound of {want}, which is the module "
        f"default: this test cannot tell the two apart. Pick an example "
        f"whose bound is not {MODULE_DEFAULT}.")
    assert E.MAX_FACES == want


def test_the_actor_env_build_leaves_the_width_alone_without_face_actions():
    E.MAX_FACES = MODULE_DEFAULT
    _build_env_from_args(_args_dict(face_actions=False, per_face=False), None)
    assert E.MAX_FACES == MODULE_DEFAULT


def test_an_explicit_face_width_override_wins(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_MAX_FACES", "37")
    E.MAX_FACES = 37
    _build_env_from_args(_args_dict(), None)
    assert E.MAX_FACES == 37
