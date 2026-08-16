"""REMAT IS A MEMORY TRADE, NOT A SEMANTIC ONE. This pins that.

``ppo._dynamic_loss_fn`` advances the palimpsa carry K times per sample
(``--grad-window K``) in an unrolled Python loop. Each ``carry_stream.advance``
is a whole ``encode_extend`` over the delta window, and without remat
reverse-mode AD stores every one of those K forwards -- a ``(window, E)`` row
block per K per sample -- which is why peak memory tracked K almost linearly
(TLM, measured: 8971 MiB at K=1 -> 21785 at K=16).

Wrapping the body in ``jax.checkpoint`` stores only the K boundary
``(carry, vmem_sums, vmem_counts)`` triples and recomputes each forward when
the cotangent arrives. The recomputation replays the SAME jaxpr on the SAME
inputs, so this must be bitwise identical in BOTH directions:

  * the forward outputs, and
  * the cotangents w.r.t. every encoder parameter.

Anything less than bit-identical here means the wrapper changed what is
computed, not where it lives, and the fix must be reverted rather than
re-goldened. Asserted at K = 1, 2, 4 (K=1 is the shipped default; K>1 is where
the residual stack actually grows).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")
os.environ.setdefault("ALPHAGRAD_INCR_TOKEN_VOCAB", "512")
os.environ.setdefault("ALPHAGRAD_INCREMENTAL_TOKENS", "1")
os.environ.setdefault("ALPHAGRAD_EXTEND_CHUNK", "8")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx.common import carry_stream as CS  # noqa: E402

TOTAL_V = 6
EMBD = 32
DELTA_W = 32
KMAX = 4


@pytest.fixture(scope="module")
def setup():
    from alphagrad.approx import ppo as P
    from alphagrad.approx.common.agent_factory import (
        apply_policy_arch, build_and_init_agent)

    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns, dynamic_substeps=True, unified_head=False, no_approx_head=True,
        face_actions=False, unified_face_head=False, live_faces=False,
        max_substeps=1, axis_group_embedding=False,
    )
    ns.embd_dim = EMBD
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.vocab_size = 512
    ns.preference_conditioned = False
    agent = build_and_init_agent(ns, TOTAL_V, num_factors=4, max_rules=4,
                                 seed=3)

    rng = np.random.default_rng(0)
    # K deltas of DIFFERENT lengths -- the cond/chunk path has to skip a
    # different number of pad steps per k, which is exactly where a
    # residual-vs-recompute discrepancy would show up if there were one.
    counts = np.array([DELTA_W, DELTA_W // 2, DELTA_W - 3, 1], np.int32)
    d_tok = jnp.asarray(rng.integers(1, 200, (KMAX, DELTA_W)).astype(np.int32))
    d_eqn = jnp.asarray(
        (rng.integers(0, 4, (KMAX, DELTA_W))).astype(np.int32))
    owner = jnp.asarray(np.array([1, 3, 2, 4], np.int32))
    part = jnp.asarray(
        np.eye(TOTAL_V + 1, dtype=np.float32)[[2, 0, 5, 1]])
    return dict(agent=agent, d_tok=d_tok, d_eqn=d_eqn,
                d_cnt=jnp.asarray(counts), owner=owner, part=part)


def _loop(agent, K, d_tok, d_eqn, d_cnt, owner, part, *, remat):
    """The production loop, with and without the wrapper."""
    def _advance_k(carry, vs, vc, dt, de, dc, ow, pa):
        return CS.advance(
            agent, carry, vs, vc, dt, de, dc, ow,
            window=DELTA_W, participants=pa,
            chunk=None, budget=jnp.asarray(DELTA_W, jnp.int32),
        )

    step = jax.checkpoint(_advance_k) if remat else _advance_k
    carry = agent.carry_init()
    vs, vc = CS.zero_memory(TOTAL_V, EMBD)
    for k in range(K):
        carry, vs, vc = step(carry, vs, vc, d_tok[k], d_eqn[k], d_cnt[k],
                             owner[k], part[k])
    return carry, vs, vc


def _scalar(agent, K, s, remat):
    carry, vs, vc = _loop(agent, K, s["d_tok"], s["d_eqn"], s["d_cnt"],
                          s["owner"], s["part"], remat=remat)
    # A scalar that touches every output: the carry (M, I, cumhist, nvalid),
    # the vertex sums and the counts.
    return (jnp.sum(carry.M * 1.0) + jnp.sum(carry.I * 2.0)
            + jnp.sum(carry.cumhist * 3.0) + carry.nvalid
            + jnp.sum(vs * 4.0) + jnp.sum(vc * 5.0))


@pytest.mark.parametrize("K", [1, 2, 4])
def test_forward_is_bit_identical(setup, K):
    a = setup["agent"]
    off = _loop(a, K, setup["d_tok"], setup["d_eqn"], setup["d_cnt"],
                setup["owner"], setup["part"], remat=False)
    on = _loop(a, K, setup["d_tok"], setup["d_eqn"], setup["d_cnt"],
               setup["owner"], setup["part"], remat=True)
    leaves_off = jax.tree_util.tree_leaves(off)
    leaves_on = jax.tree_util.tree_leaves(on)
    assert len(leaves_off) == len(leaves_on) > 0
    for i, (x, y) in enumerate(zip(leaves_off, leaves_on)):
        assert np.array_equal(np.asarray(x), np.asarray(y)), (
            f"K={K} forward leaf {i} differs: "
            f"max|d|={np.max(np.abs(np.asarray(x) - np.asarray(y)))}")


@pytest.mark.parametrize("K", [1, 2, 4])
def test_gradient_is_bit_identical(setup, K):
    a = setup["agent"]
    s = setup
    g_off = eqx.filter_grad(lambda ag: _scalar(ag, K, s, False))(a)
    g_on = eqx.filter_grad(lambda ag: _scalar(ag, K, s, True))(a)
    lo = [x for x in jax.tree_util.tree_leaves(g_off)
          if eqx.is_inexact_array(x)]
    ln = [x for x in jax.tree_util.tree_leaves(g_on)
          if eqx.is_inexact_array(x)]
    assert len(lo) == len(ln) > 0
    tot = sum(float(jnp.sum(jnp.abs(x))) for x in lo)
    # The gradient must be REAL, or "identical" is trivially satisfied by two
    # zeros -- the exact failure mode palimpsa_base_grad_test exists for.
    assert tot > 0.0, f"K={K}: the no-remat gradient is identically zero"
    for i, (x, y) in enumerate(zip(lo, ln)):
        assert np.array_equal(np.asarray(x), np.asarray(y)), (
            f"K={K} grad leaf {i} differs: "
            f"max|d|={np.max(np.abs(np.asarray(x) - np.asarray(y)))} "
            f"(|g|={np.max(np.abs(np.asarray(x)))})")
