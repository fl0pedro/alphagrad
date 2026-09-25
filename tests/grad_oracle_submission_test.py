import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as E                                # noqa: E402
from alphagrad.approx.common.rsnn_shd import RSNN_HEAD_SLOTS    # noqa: E402


def _full_rollout_config():
    return types.SimpleNamespace(
        target_fun=lambda *a: 0.0, scalar_target=True,
        data_gen=types.SimpleNamespace(full_rollout=True))


# dsnn-dfw.202: the submission pulled every given slot to the host to overwrite it, 13.5 GiB at B=64.
def test_the_full_rollout_submission_fetches_only_the_head_slots(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE", "reference")
    monkeypatch.delenv("ALPHAGRAD_WALK_ROTATE", raising=False)
    head = [jnp.full((2, 3), float(i)) for i in range(RSNN_HEAD_SLOTS)]
    given = [jnp.zeros((2, 4, 5, 6)), jnp.zeros((2, 3, 5), jnp.bfloat16)]
    asked = []
    real = jax.device_get

    def spy(x):
        asked.extend(jax.tree_util.tree_leaves(x))
        return real(x)

    monkeypatch.setattr(jax, "device_get", spy)
    seed, a, batch = E.grad_oracle_submission(_full_rollout_config(), head + given, 0)
    assert batch is None, "the full-rollout tuple has no leading batch axis to cut"
    assert not [x for x in asked if any(x is g for g in given)], asked
    rng = np.random.default_rng(seed)
    want = [np.asarray(h) for h in head] + [
        rng.standard_normal(g.shape, dtype=np.float32).astype(g.dtype)
        for g in given]
    assert len(a) == len(want)
    for got, exp in zip(a, want):
        assert isinstance(got, np.ndarray), type(got)
        assert got.dtype == exp.dtype and got.shape == exp.shape
        np.testing.assert_array_equal(got.astype(np.float32),
                                      exp.astype(np.float32))


# dsnn-dfw.226: the float64 CPU check of a free-order TLM plan at B=64 asked for 2.6 to 4.2 TB;
# the check runs on the first N recordings of the batch instead (owner ruling 2026-09-25).
class _BatchedGen:
    # A Vmapped target's data generator: x (B, 3) and y (B,) per draw.
    def __init__(self, b):
        self.b = int(b)

    def __call__(self, keys):
        x = jnp.arange(self.b * 3, dtype=jnp.float32).reshape(self.b, 3)
        y = jnp.arange(self.b, dtype=jnp.float32)
        return x, y


class _UnbatchedGen:
    def __call__(self, keys):
        return jnp.ones((4,), jnp.float32), jnp.ones((5,), jnp.float32)


def _config(gen):
    return types.SimpleNamespace(
        target_fun=lambda *a: 0.0, scalar_target=True, data_gen=gen)


def _base(b):
    return [jnp.zeros((b, 3)), jnp.zeros((b,)), jnp.full((3, 2), 1.5)]


@pytest.fixture
def clean_probe(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE", "reference")
    monkeypatch.delenv("ALPHAGRAD_WALK_ROTATE", raising=False)
    E._PROBE_BATCH.clear()
    yield
    E._PROBE_BATCH.clear()


def test_the_submission_cuts_every_data_slot_to_the_first_n_recordings(clean_probe):
    seed, a, batch = E.grad_oracle_submission(_config(_BatchedGen(8)), _base(8), 0, batch=3)
    assert batch == 3
    assert a[0].shape == (3, 3) and a[1].shape == (3,)
    np.testing.assert_array_equal(a[0], np.arange(24, dtype=np.float32).reshape(8, 3)[:3])
    np.testing.assert_array_equal(a[1], np.arange(3, dtype=np.float32))
    np.testing.assert_array_equal(a[2], np.full((3, 2), 1.5)), "a weight slot is untouched"


def test_the_submission_reports_the_whole_batch_when_nothing_is_cut(clean_probe):
    seed, a, batch = E.grad_oracle_submission(_config(_BatchedGen(8)), _base(8), 0)
    assert batch == 8 and a[0].shape == (8, 3)
    seed, a, batch = E.grad_oracle_submission(_config(_BatchedGen(8)), _base(8), 0, batch=0)
    assert batch == 8 and a[0].shape == (8, 3)
    seed, a, batch = E.grad_oracle_submission(_config(_BatchedGen(8)), _base(8), 0, batch=16)
    assert batch == 8 and a[0].shape == (8, 3), "a request above the batch cuts nothing"


def test_a_cut_on_data_slots_without_a_shared_batch_axis_raises(clean_probe):
    with pytest.raises(ValueError, match="share no batch axis"):
        E.grad_oracle_submission(_config(_UnbatchedGen()),
                                 [jnp.zeros((4,)), jnp.zeros((5,))], 0, batch=2)
    seed, a, batch = E.grad_oracle_submission(_config(_UnbatchedGen()),
                                              [jnp.zeros((4,)), jnp.zeros((5,))], 0)
    assert batch is None


def test_the_batch_flag_default_and_its_publication(monkeypatch):
    from alphagrad.approx.ppo import make_argparser

    actions = {a.option_strings[0]: a for a in make_argparser()._actions
               if a.option_strings}
    assert actions["--grad-oracle-batch"].default == E.grad_oracle_batch_default()
    assert actions["--grad-oracle-batch"].default >= 1
    assert "ALPHAGRAD_GRAD_ORACLE_BATCH" in actions["--grad-oracle-batch"].help
    assert actions["--grad-oracle-cores"].default == 0
    assert "spare" in actions["--grad-oracle-cores"].help.lower() or \
        "leave" in actions["--grad-oracle-cores"].help.lower()
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_BATCH", "12")
    assert E.grad_oracle_batch() == 12
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_BATCH", "0")
    assert E.grad_oracle_batch() == 0
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_BATCH", "-1")
    with pytest.raises(ValueError):
        E.grad_oracle_batch()
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_BATCH", "nonsense")
    with pytest.raises(ValueError):
        E.grad_oracle_batch()
    monkeypatch.delenv("ALPHAGRAD_GRAD_ORACLE_BATCH")
    assert E.grad_oracle_batch() == E.grad_oracle_batch_default()
