"""The temporal rule: what happens to the state carried into the gradient window.

TWO RULES (``--temporal-rule``, CONTEXT.md's Temporal order).

  ``bptt``  the carry enters as a CONSTANT. The gradient is the truncated one.
  ``rtrl``  the carried influence matrix ``J = d state / d W`` enters as a
            GIVEN EDGE from the weights to the carried state. Eliminating that
            vertex multiplies ``J`` by the step's state-to-state Jacobian,
            which is ONE step of real-time recurrent learning, and the gradient
            is exact through the WHOLE prefix.

THE E-PROP EQUATION THESE TESTS COMPARE AGAINST. The cell is
``graphax.examples.neuromorphic.ada_lif`` (Bellec et al.):

    U^t = alpha * U^(t-1) + W x^t
    A^t = theta + beta * a^(t-1)
    s^t = sigmoid(U^t - A^t)
    a^t = rho * a^(t-1) - s^t

Zenke and Neftci (arXiv 2010.11931) write real-time recurrent learning as
``G^t = H^t G^(t-1) + F^t`` with ``H^t = d h^t / d h^(t-1)`` the state-to-state
Jacobian, and approximate ``H^t`` by its BLOCK DIAGONAL: the internal dynamics
of a neuron survive, the coupling through other neurons' spikes is dropped.
What is left is an eligibility trace, local to one synapse. For the cell above
the trace of synapse ``(j, i)`` is

    eps_U[j,i,t] = alpha * eps_U[j,i,t-1] + x[i,t]
    psi[j,t]     = sigmoid'(U[j,t] - theta - beta * a[j,t-1])
    eps_s[j,i,t] = psi[j,t] * (eps_U[j,i,t] - beta * eps_a[j,i,t-1])
    eps_a[j,i,t] = rho * eps_a[j,i,t-1] - eps_s[j,i,t]

with both traces zero at ``t = 0``. :func:`test_eligibility_trace_is_the_carried_block`
shows that this recursion IS the within-layer carried Jacobian block, exactly,
and that its off-diagonal is exactly zero -- which is why ``Diag`` on that edge
costs nothing there and why the approximation lives in the CROSS-LAYER blocks.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve
from graphax.core import _inline_call_primitives, _stable_var_index
from graphax.examples.neuromorphic import (
    SHD_CARRY_BLOCKS,
    SHD_CARRY_DIAGONAL_BLOCKS,
    SNN_CARRY_SCOPE,
    ada_lif,
    attach_carried_jacobians,
)
from graphax.sparse.micro_actions import Diag, apply_diag, diag

from alphagrad.approx.common import examples as ex
from alphagrad.approx.common import temporal_order as to
from alphagrad.approx.common.datasets import (
    SHD_FRAME_MS,
    resolve_shd_bin_ms,
    shd_time_bins,
)
from alphagrad.approx.common.masks import diag_pair_factor_space, diag_valid_mask
from alphagrad.approx.common.snn_shd import (
    TEMPORAL_RULE_ORDER,
    resolve_fixed_temporal_order,
    resolve_temporal_rule,
)

ARGNUMS = (8, 9, 10)
BIN_MS = 100          # T = 10, so a test runs in seconds and not minutes


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / n) if n else float(np.linalg.norm(a - b))


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# A SMALL THREE-LAYER ADAPTIVE LIF, written here rather than registered.
#
# The registered SHD targets are 700-128-20 and their carried Jacobian is
# 218 MB, which is the right size for a measurement and the wrong size for a
# unit test. This one is n-n-n with n = 4, so every claim below is checked on
# arrays a person can print.
# ---------------------------------------------------------------------------

N_UNITS = 4
ALPHA, BETA, RHO, THETA = 0.9, 0.8, 0.95, 0.3
PARAMS = (jnp.array(ALPHA), jnp.array(BETA), jnp.array(RHO), jnp.array(THETA))


def _small_weights(key, n=N_UNITS):
    k = jax.random.split(key, 3)
    return tuple(jax.random.normal(k[i], (n, n)) * 0.5 for i in range(3))


def _small_sequence(key, steps, n=N_UNITS):
    k = jax.random.split(key, 2)
    seq = jax.random.bernoulli(k[0], 0.3, (steps, n)).astype(jnp.float64)
    tgt = jax.nn.one_hot(jax.random.randint(k[1], (), 0, n), n).astype(jnp.float64)
    return seq, tgt


def _small_step(W1, W2, W3, states, x):
    U1, U2, U3, a1, a2, a3 = states
    i1 = W1 @ x
    U1, a1, s1 = ada_lif(U1, a1, i1, *PARAMS)
    i2 = W2 @ s1
    U2, a2, s2 = ada_lif(U2, a2, i2, *PARAMS)
    i3 = W3 @ s2
    U3, a3, s3 = ada_lif(U3, a3, i3, *PARAMS)
    return (U1, U2, U3, a1, a2, a3), s3


def _small_prefix(seq, n_pre, n=N_UNITS):
    """``(W1, W2, W3) -> the six carried states`` after ``n_pre`` steps."""
    def run(W1, W2, W3):
        st = tuple(jnp.zeros((n,)) for _ in range(6))
        for t in range(n_pre):
            st, _ = _small_step(W1, W2, W3, st, seq[t])
        return st
    return run


def _small_target(seq_window, tgt, states, weights, *carried):
    """The same contract the SHD targets have, at n-n-n."""
    W1, W2, W3 = weights
    states = attach_carried_jacobians(states, weights, carried)
    loss = 0.0
    n_win = int(seq_window.shape[0])
    for t in range(n_win):
        with to_step_scope(t):
            states, s3 = _small_step(W1, W2, W3, states, seq_window[t])
            loss = loss + jnp.mean(0.5 * (s3 - tgt) ** 2)
    return loss / n_win


def to_step_scope(t):
    from graphax.examples.neuromorphic import snn_step_scope
    return snn_step_scope(t)


def _small_carried(seq, n_pre, weights, zero_cross=False, n=N_UNITS):
    """The fifteen-entry attachment tuple for the small net."""
    run = _small_prefix(seq, n_pre, n)
    jac = jax.jacrev(run, argnums=(0, 1, 2))(*weights)
    blocks = []
    for s_i, w_i in SHD_CARRY_BLOCKS:
        b = jac[s_i][w_i]
        if zero_cross and (s_i, w_i) not in SHD_CARRY_DIAGONAL_BLOCKS:
            b = jnp.zeros_like(b)
        blocks.append(b)
    return tuple(weights) + tuple(blocks), jac


def _small_full_loss(seq, tgt, n_win):
    """The loss of the last ``n_win`` steps, differentiated through EVERY step."""
    T = int(seq.shape[0])

    def loss(W1, W2, W3):
        st = tuple(jnp.zeros((N_UNITS,)) for _ in range(6))
        acc = 0.0
        for t in range(T):
            st, s3 = _small_step(W1, W2, W3, st, seq[t])
            if t >= T - n_win:
                acc = acc + jnp.mean(0.5 * (s3 - tgt) ** 2)
        return acc / n_win
    return loss


@pytest.fixture(scope="module")
def x64():
    """Float64 for every claim that says EXACT.

    The attachment is exact by construction, and at float32 the distance
    between it and the reference is set by the conditioning of a ten-step
    spiking prefix (measured 3.6e-4 on the 700-128-20 target), not by the
    design. A test that says "exact" has to measure the design.
    """
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# 1. The eligibility trace IS the within-layer carried Jacobian block
# ---------------------------------------------------------------------------

def test_eligibility_trace_is_the_carried_block(x64):
    """The e-prop recursion of the module docstring, on a 4-neuron layer.

    Claim, both halves: ``d U[j,t] / d W[k,i]`` and ``d a[j,t] / d W[k,i]`` are
    EXACTLY zero for ``k != j``, and on the diagonal ``k == j`` they are exactly
    the two eligibility traces. So the block-diagonal approximation of the
    state-to-state Jacobian costs NOTHING inside one layer -- which is Zenke and
    Neftci's point that a neuron's internal dynamics are already block diagonal
    -- and the trace is not an approximation of the carried block, it IS it.
    """
    n, steps = N_UNITS, 6
    key = jax.random.PRNGKey(0)
    W = jax.random.normal(jax.random.split(key, 2)[0], (n, n)) * 0.5
    x = jax.random.bernoulli(
        jax.random.split(key, 2)[1], 0.3, (steps, n)).astype(jnp.float64)

    def one_layer(W):
        U = jnp.zeros((n,))
        a = jnp.zeros((n,))
        for t in range(steps):
            U, a, _ = ada_lif(U, a, W @ x[t], *PARAMS)
        return U, a

    jU, ja = jax.jacrev(one_layer)(W)          # each (n, n, n): [j, k, i]

    # The hand-written e-prop trace, exactly the four lines of the docstring.
    epsU = np.zeros((n, n))                    # [j, i]
    epsa = np.zeros((n, n))
    U = np.zeros(n)
    a = np.zeros(n)
    xs = np.asarray(x, np.float64)
    Wn = np.asarray(W, np.float64)
    for t in range(steps):
        a_prev = a.copy()
        U = ALPHA * U + Wn @ xs[t]
        z = U - THETA - BETA * a_prev
        s = 1.0 / (1.0 + np.exp(-z))
        psi = s * (1.0 - s)                    # sigmoid'
        epsU = ALPHA * epsU + xs[t][None, :]
        eps_s = psi[:, None] * (epsU - BETA * epsa)
        epsa = RHO * epsa - eps_s
        a = RHO * a_prev - s

    jU = np.asarray(jU, np.float64)
    ja = np.asarray(ja, np.float64)
    off = ~np.eye(n, dtype=bool)
    # The off-diagonal of the (out neuron, weight row) pair is EXACTLY zero.
    assert np.max(np.abs(jU[off])) == 0.0
    assert np.max(np.abs(ja[off])) == 0.0
    diagU = np.einsum("jji->ji", jU)
    diaga = np.einsum("jji->ji", ja)
    assert _rel(epsU, diagU) < 1e-12
    assert _rel(epsa, diaga) < 1e-12


def test_diag_on_the_carried_edge_is_that_block_diagonal(x64):
    """``Diag(0, 1, n)`` on a carried-Jacobian edge keeps exactly that block.

    The edge is ``d state / d W`` with logical dims ``(state) x (W row, W col)``,
    so pair ``(0, 1)`` ties the state index to the weight's ROW index -- the
    presynaptic-to-postsynaptic pairing an eligibility trace has -- and factor
    ``n`` makes every block one element. That is the e-prop store: ``n * n``
    numbers where the exact block is ``n * n * n``.
    """
    n, steps = N_UNITS, 6
    key = jax.random.PRNGKey(0)
    W = jax.random.normal(key, (n, n)) * 0.5
    x = jax.random.bernoulli(
        jax.random.PRNGKey(1), 0.3, (steps, n)).astype(jnp.float64)

    def one_layer(W):
        U = jnp.zeros((n,))
        a = jnp.zeros((n,))
        for t in range(steps):
            U, a, _ = ada_lif(U, a, W @ x[t], *PARAMS)
        return U

    J = jax.jacrev(one_layer)(W)                 # (n, n, n)

    # The edge, as graphax stores one: state out, W's two dims primal.
    from graphax.sparse.indexes import DenseIndex
    from graphax.sparse.tensor import SparseTensor
    st = SparseTensor(
        out_dims=[DenseIndex(id=0, size=n, block_size=None, axis=0)],
        primal_dims=[DenseIndex(id=1, size=n, block_size=None, axis=1),
                     DenseIndex(id=2, size=n, block_size=None, axis=2)],
        val=J)
    mask = diag_valid_mask(st, 8)
    assert mask[0, 1], "the (state, W row) pair must admit Diag"
    base, span = diag_pair_factor_space(st, 0, 1)
    assert (base, span) == (1, n)

    out = apply_diag(st, Diag(i=0, j=1, factor=n))
    dense = np.asarray(out.dense(), np.float64).reshape(n, n, n)
    want = np.asarray(J, np.float64).copy()
    want[~np.eye(n, dtype=bool)] = 0.0
    assert np.array_equal(dense, want)
    # ... and it is LOSSLESS here, because the block was already diagonal.
    assert np.array_equal(dense, np.asarray(J, np.float64))


# ---------------------------------------------------------------------------
# 2. Exactness of the two rules, on the small net and on the SHD target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_win", [1, 2])
def test_small_rtrl_is_exact_and_bptt_is_the_truncation(x64, n_win):
    """RTRL equals the gradient through the WHOLE prefix; BPTT does not."""
    key = jax.random.PRNGKey(3)
    W = _small_weights(jax.random.split(key, 2)[0])
    seq, tgt = _small_sequence(jax.random.split(key, 2)[1], 8)
    T = int(seq.shape[0])
    n_pre = T - n_win
    states = _small_prefix(seq, n_pre)(*W)
    states = tuple(jax.lax.stop_gradient(s) for s in states)
    window = seq[n_pre:]
    carried, _ = _small_carried(seq, n_pre, W)

    def bptt(W1, W2, W3):
        return _small_target(window, tgt, states, (W1, W2, W3))

    def rtrl(W1, W2, W3):
        return _small_target(window, tgt, states, (W1, W2, W3), *carried)

    # The forward value does not move: the attachment is value-neutral.
    assert float(bptt(*W)) == float(rtrl(*W))

    exact = jax.grad(_small_full_loss(seq, tgt, n_win), argnums=(0, 1, 2))(*W)
    g_rtrl = jax.grad(rtrl, argnums=(0, 1, 2))(*W)
    g_bptt = jax.grad(bptt, argnums=(0, 1, 2))(*W)
    for a, b in zip(g_rtrl, exact):
        assert _rel(a, b) < 1e-11
    # BPTT is a real truncation, not a numerically equal one.
    assert _rel(g_bptt[0], exact[0]) > 1e-3


@pytest.mark.parametrize("n_win", [1, 2])
def test_small_elimination_matches_the_rule_it_runs(x64, n_win):
    """``jacve`` on the exact plan reproduces each rule's own gradient."""
    key = jax.random.PRNGKey(3)
    W = _small_weights(jax.random.split(key, 2)[0])
    seq, tgt = _small_sequence(jax.random.split(key, 2)[1], 8)
    T = int(seq.shape[0])
    n_pre = T - n_win
    states = tuple(jax.lax.stop_gradient(s)
                   for s in _small_prefix(seq, n_pre)(*W))
    window = seq[n_pre:]
    carried, _ = _small_carried(seq, n_pre, W)

    for name, extra in (("bptt", ()), ("rtrl", carried)):
        def fn(W1, W2, W3, _e=extra):
            return _small_target(window, tgt, states, (W1, W2, W3), *_e)

        jx = jax.make_jaxpr(fn)(*W).jaxpr
        valid = [i for i, e in enumerate(jx.eqns, 1)
                 if e.outvars[0] not in jx.outvars]
        order = list(reversed(valid))
        jac = jacve(fn, order, argnums=(0, 1, 2))(*W)
        ref = jax.grad(fn, argnums=(0, 1, 2))(*W)
        for a, b in zip(jac, ref):
            assert _rel(a, b) < 1e-11, name


def test_shd_rtrl_is_exact_through_the_prefix(x64):
    """The registered 700-128-20 target, at 100 ms bins, window 1 and 2.

    This is the claim in production shape: the plan the search runs under
    ``--temporal-rule rtrl`` computes the gradient of the window's loss through
    the WHOLE recording, not through the window.
    """
    from alphagrad.approx.common.snn_shd import _spike_sequence, _weights

    for n_win in (1, 2):
        xs = ex.get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(1), dataset=None,
                         grad_window=n_win, dataset_size=-1, bin_ms=BIN_MS,
                         temporal_rule="rtrl")
        fn = ex.get_fn("ADALIF_SNN_SHD")
        assert len(xs) == 15 + 3 + len(SHD_CARRY_BLOCKS)

        key = jax.random.split(jax.random.PRNGKey(1), 2)
        seq, tgt = _spike_sequence(key[0], None, -1, BIN_MS)
        W = _weights(key[1])
        T = int(seq.shape[0])
        h, n_out = 128, 20
        a_, b_, r_, th_ = (jnp.array(0.9), jnp.array(0.8),
                           jnp.array(0.95), jnp.array(0.3))

        def full(W1, W2, W3):
            U1 = jnp.zeros((h,)); U2 = jnp.zeros((h,)); U3 = jnp.zeros((n_out,))
            a1 = jnp.zeros((h,)); a2 = jnp.zeros((h,)); a3 = jnp.zeros((n_out,))
            acc = 0.0
            for t in range(T):
                i1 = W1 @ seq[t]
                U1, a1, s1 = ada_lif(U1, a1, i1, a_, b_, r_, th_)
                i2 = W2 @ s1
                U2, a2, s2 = ada_lif(U2, a2, i2, a_, b_, r_, th_)
                i3 = W3 @ s2
                U3, a3, s3 = ada_lif(U3, a3, i3, a_, b_, r_, th_)
                if t >= T - n_win:
                    acc = acc + jnp.mean(0.5 * (s3 - tgt) ** 2)
            return acc / n_win

        ref = jax.grad(full, argnums=(0, 1, 2))(*W)
        got = jax.grad(fn, argnums=ARGNUMS)(*xs)
        for a, b in zip(got, ref):
            assert _rel(a, b) < 1e-11


def test_bptt_graph_is_byte_for_byte_the_one_the_branch_built():
    """No rule, and ``--temporal-rule bptt``, build the SAME argument tuple.

    ``bptt`` names what the branch already did. If it changed the tuple, every
    archived truncated-BPTT result would stop reproducing.
    """
    a = ex.get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(1), dataset=None,
                    grad_window=1, dataset_size=-1, bin_ms=BIN_MS)
    b = ex.get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(1), dataset=None,
                    grad_window=1, dataset_size=-1, bin_ms=BIN_MS,
                    temporal_rule="bptt")
    assert len(a) == len(b) == 15
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# ---------------------------------------------------------------------------
# 3. The e-prop rule as a PLAN: Diag on the carried edges, cross-layer dropped
# ---------------------------------------------------------------------------

def _carry_face_keys(jx):
    """``{(in_edge idx, out_edge idx)}`` of every carried-Jacobian face.

    A carried-Jacobian face eliminates the contraction vertex of the carry
    block: its in edge comes from the weight-delta vertex and carries ``J``,
    its out edge is the addition that forms the state the window reads.
    """
    vidx = _stable_var_index(jx)
    out_owner = {}
    for e in jx.eqns:
        for ov in e.outvars:
            out_owner[ov] = e
    keys = {}
    for i, e in enumerate(jx.eqns):
        ns = str(getattr(e.source_info, "name_stack", ""))
        if SNN_CARRY_SCOPE not in ns or e.primitive.name != "dot_general":
            continue
        # the in edge: the weight-delta variable this contraction reads
        src = [v for v in e.invars
               if v in out_owner and out_owner[v].primitive.name == "sub"]
        # the out edge: the addition that consumes this contraction
        dst = [f.outvars[0] for f in jx.eqns
               if e.outvars[0] in f.invars]
        assert len(src) == 1 and len(dst) == 1
        keys[i + 1] = (vidx[src[0]], vidx[dst[0]])
    return keys


def test_eprop_plan_equals_the_block_diagonal_gradient(x64):
    """A PLAN that is e-prop: ``Diag`` on the carried edges, cross layers gone.

    Two independent constructions of the same number:

      the PLAN      ``jacve`` of the exact order with ``Diag(0, 1, n)`` on every
                    carried-Jacobian face, and the six cross-layer blocks
                    dropped (they are the coupling Zenke and Neftci remove).
      the RULE      ``jax.grad`` of the same target with the cross-layer blocks
                    zeroed in the arguments -- no elimination involved.

    They must agree. And the e-prop gradient must DIFFER from the exact one,
    or the approximation would not be one.
    """
    key = jax.random.PRNGKey(3)
    W = _small_weights(jax.random.split(key, 2)[0])
    seq, tgt = _small_sequence(jax.random.split(key, 2)[1], 8)
    T, n_win = int(seq.shape[0]), 1
    n_pre = T - n_win
    states = tuple(jax.lax.stop_gradient(s)
                   for s in _small_prefix(seq, n_pre)(*W))
    window = seq[n_pre:]
    carried, _ = _small_carried(seq, n_pre, W)
    carried_bd, _ = _small_carried(seq, n_pre, W, zero_cross=True)

    def fn(W1, W2, W3):
        return _small_target(window, tgt, states, (W1, W2, W3), *carried)

    def fn_bd(W1, W2, W3):
        return _small_target(window, tgt, states, (W1, W2, W3), *carried_bd)

    jx = jax.make_jaxpr(fn)(*W).jaxpr
    valid = [i for i, e in enumerate(jx.eqns, 1)
             if e.outvars[0] not in jx.outvars]
    order = list(reversed(valid))
    faces = _carry_face_keys(jx)
    assert len(faces) == len(SHD_CARRY_BLOCKS)

    hooks = {k: (diag(0, 1, N_UNITS), None, None) for k in faces.values()}
    got = jacve(fn_bd, order, argnums=(0, 1, 2), face_transforms=hooks)(*W)
    rule = jax.grad(fn_bd, argnums=(0, 1, 2))(*W)
    for a, b in zip(got, rule):
        assert _rel(a, b) < 1e-11

    exact = jax.grad(fn, argnums=(0, 1, 2))(*W)
    # The approximation is a real one: the cross-layer blocks carry signal.
    assert _rel(rule[0], exact[0]) > 1e-3
    assert _cos(rule[0], exact[0]) < 1.0


def test_within_layer_diag_alone_is_lossless(x64):
    """``Diag(0, 1, n)`` on the six WITHIN-layer faces changes nothing.

    The within-layer block is already diagonal in (state, weight row) -- test 1
    -- so the plan that diagonalises only those faces is the EXACT plan. This
    is what makes the cross-layer blocks, and not the diagonal ones, the place
    the approximation has to live.
    """
    key = jax.random.PRNGKey(3)
    W = _small_weights(jax.random.split(key, 2)[0])
    seq, tgt = _small_sequence(jax.random.split(key, 2)[1], 8)
    T, n_win = int(seq.shape[0]), 1
    n_pre = T - n_win
    states = tuple(jax.lax.stop_gradient(s)
                   for s in _small_prefix(seq, n_pre)(*W))
    window = seq[n_pre:]
    carried, _ = _small_carried(seq, n_pre, W)

    def fn(W1, W2, W3):
        return _small_target(window, tgt, states, (W1, W2, W3), *carried)

    jx = jax.make_jaxpr(fn)(*W).jaxpr
    valid = [i for i, e in enumerate(jx.eqns, 1)
             if e.outvars[0] not in jx.outvars]
    order = list(reversed(valid))
    faces = _carry_face_keys(jx)
    within = {SHD_CARRY_BLOCKS.index(b) for b in SHD_CARRY_DIAGONAL_BLOCKS}
    keys = [v for i, (_, v) in enumerate(sorted(faces.items())) if i in within]
    assert len(keys) == len(SHD_CARRY_DIAGONAL_BLOCKS)
    hooks = {k: (diag(0, 1, N_UNITS), None, None) for k in keys}
    got = jacve(fn, order, argnums=(0, 1, 2), face_transforms=hooks)(*W)
    ref = jax.grad(fn, argnums=(0, 1, 2))(*W)
    for a, b in zip(got, ref):
        assert _rel(a, b) < 1e-11


# ---------------------------------------------------------------------------
# 4. The step tag of the carry block, and the order it forces
# ---------------------------------------------------------------------------

def test_carry_block_carries_step_zero():
    """Every carry-block equation is tagged step 0, the earliest step there is.

    Real-time recurrent learning eliminates the earliest copy first, so the
    carried Jacobian has to be in that copy's group; otherwise ``forward``
    would put the RTRL step LAST, which is the opposite of the rule.
    """
    xs = ex.get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(1), dataset=None,
                     grad_window=2, dataset_size=-1, bin_ms=BIN_MS,
                     temporal_rule="rtrl")
    fn = ex.get_fn("ADALIF_SNN_SHD")
    cj = jax.make_jaxpr(fn)(*xs)
    jx, _ = _inline_call_primitives(cj.jaxpr, cj.literals)
    tags = to.step_tags(jx)
    carry = [i for i, e in enumerate(jx.eqns)
             if SNN_CARRY_SCOPE in str(getattr(e.source_info, "name_stack", ""))]
    # 3 weight deltas + 12 contractions + 12 additions
    assert len(carry) == 3 + 2 * len(SHD_CARRY_BLOCKS)
    assert all(int(tags[i]) == 0 for i in carry)


def test_rule_forces_the_matching_temporal_order():
    assert resolve_fixed_temporal_order("bptt", "free") == "reverse"
    assert resolve_fixed_temporal_order("rtrl", "free") == "forward"
    assert resolve_fixed_temporal_order("bptt", "reverse") == "reverse"
    assert resolve_fixed_temporal_order("rtrl", "forward") == "forward"
    assert resolve_fixed_temporal_order(None, "free") == "free"
    assert TEMPORAL_RULE_ORDER == {"bptt": "reverse", "rtrl": "forward"}


@pytest.mark.parametrize("rule,fixed", [("bptt", "forward"), ("rtrl", "reverse")])
def test_contradicting_order_raises(rule, fixed):
    with pytest.raises(ValueError, match="contradict"):
        resolve_fixed_temporal_order(rule, fixed)


@pytest.mark.parametrize("example", ["NeuralNetwork", "TransformerLM", None])
def test_rule_on_a_target_without_time_steps_raises(example):
    with pytest.raises(ValueError, match="NO time steps"):
        resolve_temporal_rule(example, "rtrl")
    with pytest.raises(ValueError, match="NO time steps"):
        resolve_temporal_rule(example, "bptt")


def test_rtrl_on_the_fully_unrolled_target_raises():
    """``ADALIF_SNN_SEQ`` has no detached prefix, so it has nothing to carry."""
    with pytest.raises(ValueError, match="DETACHED PREFIX"):
        resolve_temporal_rule("ADALIF_SNN_SEQ", "rtrl")
    # bptt is fine there: the carry that enters is a zero state.
    assert resolve_temporal_rule("ADALIF_SNN_SEQ", "bptt") == "bptt"


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="not one of"):
        resolve_temporal_rule("ADALIF_SNN_SHD", "eprop")


def test_no_rule_is_the_default():
    assert resolve_temporal_rule("ADALIF_SNN_SHD", None) is None
    assert resolve_temporal_rule("NeuralNetwork", None) is None


# ---------------------------------------------------------------------------
# 5. The bin width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bin_ms,steps", [(10, 100), (20, 50), (100, 10), (1000, 1)])
def test_bin_width_sets_the_step_count(bin_ms, steps):
    assert shd_time_bins(bin_ms) == steps
    assert steps * bin_ms == SHD_FRAME_MS


def test_bin_width_must_divide_the_frame():
    for bad in (7, 3, 300, 999):
        with pytest.raises(ValueError, match="does not divide"):
            resolve_shd_bin_ms(bad)
    for bad in (0, -10):
        with pytest.raises(ValueError, match=">= 1 ms"):
            resolve_shd_bin_ms(bad)
    assert resolve_shd_bin_ms(None) == 10


def test_bin_width_reaches_the_target_shape():
    for bin_ms, steps in ((10, 100), (100, 10)):
        xs = ex.get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(1), dataset=None,
                         grad_window=1, dataset_size=-1, bin_ms=bin_ms)
        assert xs[0].shape == (1, 700)
        xs = ex.get_args("ADALIF_SNN_SHD", jax.random.PRNGKey(1), dataset=None,
                         grad_window=steps, dataset_size=-1, bin_ms=bin_ms)
        assert xs[0].shape == (steps, 700)


def test_window_past_the_step_count_raises():
    from alphagrad.approx.common.snn_shd import resolve_grad_window
    assert resolve_grad_window("ADALIF_SNN_SHD", 10, bin_ms=100) == 10
    with pytest.raises(ValueError, match="exceeds the SHD sequence length 10"):
        resolve_grad_window("ADALIF_SNN_SHD", 11, bin_ms=100)
    assert resolve_grad_window("ADALIF_SNN_SHD", 100, bin_ms=10) == 100


def test_bin_width_off_an_shd_target_raises():
    from alphagrad.approx.common.snn_shd import resolve_bin_ms
    with pytest.raises(ValueError, match="reads no SHD recording"):
        resolve_bin_ms("NeuralNetwork", 100)
    with pytest.raises(ValueError, match="reads no SHD recording"):
        resolve_bin_ms("ADALIF_SNN_SEQ", 100)
    assert resolve_bin_ms("ADALIF_SNN_SHD", 100) == 100


def test_binned_cache_is_per_bin_width(tmp_path, monkeypatch):
    """One cache entry and one file per width, and the name says the width.

    A file written at one width must never be read back at another. The only
    way to make that impossible is to let the name say what is in it, so a
    stale file is simply not found.
    """
    import numpy as np

    from alphagrad.approx.common import datasets as ds

    monkeypatch.setenv("DSNN_SHD_DIR", str(tmp_path))
    monkeypatch.setattr(ds, "_SHD_BINNED", {})

    n, ch = 3, ds.SHD_CHANNELS
    times = [np.array([0.001, 0.015, 0.5, 0.995, 1.5]) for _ in range(n)]
    units = [np.array([0, 1, 2, 3, 4]) for _ in range(n)]
    labels = np.arange(n, dtype=np.uint8)

    calls = []

    def fake_bin(path, bin_ms):
        calls.append(int(bin_ms))
        nb = ds.SHD_FRAME_MS // int(bin_ms)
        x = np.zeros((n, nb, ch), dtype=np.uint8)
        for i in range(n):
            keep = times[i] < nb * (bin_ms / 1000.0)
            b = (times[i][keep] / (bin_ms / 1000.0)).astype(np.int64)
            np.add.at(x[i], (b, units[i][keep]), 1)
        return x, labels

    monkeypatch.setattr(ds, "_bin_shd", fake_bin)
    monkeypatch.setattr(ds, "_download_shd", lambda cache, subset: tmp_path / "x.h5")

    x10, _ = ds._shd_binned("train", 10)
    x100, _ = ds._shd_binned("train", 100)
    assert x10.shape == (n, 100, ch)
    assert x100.shape == (n, 10, ch)
    assert calls == [10, 100]

    # a second read of either width hits the in-process cache
    ds._shd_binned("train", 10)
    ds._shd_binned("train", 100)
    assert calls == [10, 100]

    names = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert names == ["shd_train_binned_100x700_10ms_uint8.npz",
                     "shd_train_binned_10x700_100ms_uint8.npz"]

    # and from disk, in a fresh process cache
    monkeypatch.setattr(ds, "_SHD_BINNED", {})
    y10, _ = ds._shd_binned("train", 10)
    y100, _ = ds._shd_binned("train", 100)
    assert calls == [10, 100]
    assert np.array_equal(y10, x10)
    assert np.array_equal(y100, x100)
    # the 100 ms bins hold the same spikes, folded ten to one
    assert int(y10.sum()) == int(y100.sum())
