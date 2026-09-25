# dsnn-dfw.37, job 66313: --ray-measure 0 measured on the trainer's env, built
# sparse by default, so the cosine read SparseTensor leaves; a blocked-dense
# leaf there raised GradientStructureMismatch "stale layout metadata".
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from graphax.sparse.indexes import Index                        # noqa: E402
from graphax.sparse.tensor import SparseTensor                  # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402


def _toy_env():
    rng = np.random.default_rng(0)
    w = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(w, v):
        return jnp.sum(jnp.tanh(w @ v) ** 2)

    args = [w, x]
    env = env_mod.VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(toy)(*args), args=args, argnums=(0, 1), num_envs=0,
        target_fun=toy, measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)
    samples = tuple(jnp.asarray(np.stack([np.asarray(a)])) for a in args)
    return env, samples


def _exact_plan(n):
    specs = np.full((n, env_mod.MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n, env_mod.MAX_FACES, env_mod.FACE_SLOTS, 3), -1,
                    np.int32)
    skips = np.zeros((n, env_mod.MAX_FACES), np.int32)
    return jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips)


@pytest.fixture
def in_process(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "jac_cosine")
    monkeypatch.setenv("ALPHAGRAD_DIRECT_MEASURE", "1")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    monkeypatch.setattr(env_mod, "_device_bytes_limit",
                        lambda d: 16_000_000_000)
    env_mod.set_measure_timeout_s(300.0)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()
    yield
    env_mod.set_measure_timeout_s(None)
    env_mod.consume_refused_counts()
    env_mod.consume_plan_records()


def test_the_in_process_cosine_reads_dense_leaves_from_a_sparse_env(
        in_process, monkeypatch):
    seen = []
    real = env_mod._quality_metrics

    def spy(jac_exact, jac_approx, *a, **kw):
        seen.append(tuple(type(x).__name__
                          for x in env_mod._gradient_leaves(jac_approx)))
        return real(jac_exact, jac_approx, *a, **kw)

    monkeypatch.setattr(env_mod, "_quality_metrics", spy)
    env, samples = _toy_env()
    assert env.config.sparse is True
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    specs, faces, skips = _exact_plan(len(order))
    env_mod._callback(env.config, env.args, env.consts, jnp.asarray(order),
                      specs, faces, skips, len(order), *samples)
    print(f"[dfw37] candidate leaves at the cosine: {seen}")
    assert seen, "the in-process measurement scored no cosine"
    assert all(t != "SparseTensor" for leaves in seen for t in leaves), seen
    assert env_mod.consume_refused_counts() == {}


def test_a_blocked_dense_leaf_still_has_no_lazy_contraction():
    exact = jnp.ones((128, 128), jnp.float32)
    blocked = SparseTensor(
        (), (Index(0, 32, 0, block_size=4), Index(1, 128, 1)),
        jnp.ones((32, 128), jnp.float32))
    with pytest.raises(env_mod.GradientStructureMismatch) as exc:
        env_mod._gradient_similarity([exact], [blocked], "grad_cosine")
    print(f"[dfw37] {exc.value}")
    assert "stale layout metadata" in str(exc.value)
