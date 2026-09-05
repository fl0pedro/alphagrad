"""ONE order table for the trainer and the sweep tool (ticket dsnn-3qm.64).

On the Helmholtz example (the sweep test's target): ``common/order.py`` and
``landscape_map`` return the same Markowitz and reverse tables, the tables
are permutations of the valid vertices, and a rollout under the pin visits
exactly the table, one legal vertex per step. The vertex head's masked
distribution under a pin is a point mass for ANY logits, so its KL against
any other logits is 0.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from types import SimpleNamespace as NS

from alphagrad.approx.common import masks as M
from alphagrad.approx.common.order import fixed_order_for_env, markowitz_order
from alphagrad.approx.tools.landscape_map import (
    make_argparser, build_env, markowitz_order as lm_markowitz,
    rev_order as lm_rev)


@pytest.fixture(scope="module")
def helmholtz():
    args = make_argparser().parse_args(
        ["--example", "Helmholtz", "--out-dir", "/tmp/test_order_table",
         "--dry-run"])
    env, _, closed = build_env(args)
    return env, closed


def test_trainer_and_sweep_share_the_markowitz_table(helmholtz):
    env, closed = helmholtz
    a = fixed_order_for_env("markowitz", env)
    b = lm_markowitz(env)
    c = markowitz_order(closed.jaxpr, env.config.argnums, env.consts,
                        env.args, env.valid_vertices)
    assert a.tolist() == b.tolist() == c.tolist()
    assert sorted(a.tolist()) == sorted(int(v) for v in env.valid_vertices)


def test_trainer_and_sweep_share_the_reverse_table(helmholtz):
    env, _ = helmholtz
    a = fixed_order_for_env("reverse", env)
    assert a.tolist() == lm_rev(env).tolist()
    assert a.tolist() == sorted((int(v) for v in env.valid_vertices), reverse=True)


def test_pin_visits_exactly_the_table(helmholtz):
    env, closed = helmholtz
    total_v = len(closed.jaxpr.eqns)
    valid = [int(v) for v in env.valid_vertices]
    static = M.build_vertex_valid_static(valid, total_v)
    table = fixed_order_for_env("markowitz", env)
    chosen = np.zeros(len(valid), dtype=np.int32)
    for k in range(len(valid)):
        st = NS(order=jnp.asarray(chosen), step_count=jnp.asarray(k, jnp.int32))
        a = np.asarray(M.vertex_avail_at_step(st, static, total_v, len(valid),
                                              fixed_order=table))
        assert int(a.sum()) == 1, (k, a)
        chosen[k] = int(np.argmax(a)) + 1
    assert chosen.tolist() == table.tolist()
    st = NS(order=jnp.asarray(chosen), step_count=jnp.asarray(len(valid), jnp.int32))
    a = np.asarray(M.vertex_avail_at_step(st, static, total_v, len(valid),
                                          fixed_order=table))
    assert int(a.sum()) == 0


def test_vertex_head_kl_is_structurally_zero_under_a_pin():
    from alphagrad.approx.ppo import _mask_vertex_logits
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    logits_a = jax.random.normal(k1, (7,))
    logits_b = jax.random.normal(k2, (7,)) * 10.0
    mask = jnp.zeros((7,)).at[3].set(1.0)
    la = jax.nn.log_softmax(_mask_vertex_logits(logits_a, mask))
    lb = jax.nn.log_softmax(_mask_vertex_logits(logits_b, mask))
    pa = jnp.exp(la)
    kl = jnp.sum(jnp.where(pa > 0, pa * (la - lb), 0.0))
    assert float(kl) == 0.0
    assert float(pa[3]) == 1.0
