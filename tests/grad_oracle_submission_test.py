import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402

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
    seed, a = E.grad_oracle_submission(_full_rollout_config(), head + given, 0)
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
