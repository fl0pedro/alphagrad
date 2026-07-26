"""The vertex memory must be EXACTLY incremental, or ratio-1 breaks.

The append-only path folds each elimination's rows into the memory as they
arrive. The loss then has to reproduce the same state. That only holds if
folding in chunks equals folding all at once — which is why the pooling is a
plain sum/count (associative) rather than anything gated or recurrent.

Also pinned here: the memory's summary equals the full-sequence mean the value
heads used to get from enc_x, so switching to the memory does not silently
change what the critic sees.
"""
import os
from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alphagrad.approx import vertex_memory as vm

V, E = 6, 8


def _rows(n, seed=0, width=E):
    return jnp.asarray(
        np.random.default_rng(seed).standard_normal((n, width)), jnp.float32)


def _ids(n, seed=1):
    # a mix of real vertices and structural (-1) tokens
    r = np.random.default_rng(seed).integers(-1, V, size=n)
    return jnp.asarray(r, jnp.int32)


def test_chunked_fold_equals_single_pass():
    """THE property the append-only path rests on."""
    n = 40
    rows, ids = _rows(n), _ids(n)

    s1, c1 = vm.init(V, E)
    s1, c1 = vm.update(s1, c1, rows, ids)

    s2, c2 = vm.init(V, E)
    for lo, hi in [(0, 7), (7, 7), (7, 23), (23, 40)]:   # incl. an EMPTY chunk
        s2, c2 = vm.update(s2, c2, rows[lo:hi], ids[lo:hi])

    np.testing.assert_allclose(np.asarray(s1), np.asarray(s2), rtol=0, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))


def test_padding_contributes_nothing():
    """A padded delta must fold identically to the unpadded one."""
    n, D = 9, 20
    rows, ids = _rows(n, 2), _ids(n, 3)

    s1, c1 = vm.update(*vm.init(V, E), rows, ids)

    rows_p = jnp.concatenate([rows, jnp.ones((D - n, E), jnp.float32) * 99.0])
    ids_p = jnp.concatenate([ids, jnp.zeros(D - n, jnp.int32)])
    valid = jnp.arange(D) < n
    s2, c2 = vm.update(*vm.init(V, E), rows_p, ids_p, valid)

    np.testing.assert_allclose(np.asarray(s1), np.asarray(s2), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(c1), np.asarray(c2))


def test_structural_tokens_land_in_the_global_slot():
    rows = jnp.ones((4, E), jnp.float32)
    ids = jnp.asarray([-1, -1, 0, 2], jnp.int32)
    s, c = vm.update(*vm.init(V, E), rows, ids)
    assert float(c[V]) == 2.0, "eqn_id<0 must fold into the trailing slot"
    assert float(c[0]) == 1.0 and float(c[2]) == 1.0


def test_summary_equals_the_full_sequence_mean():
    """What the value heads used to compute from enc_x directly."""
    n = 33
    rows, ids = _rows(n, 4), _ids(n, 5)
    s, c = vm.update(*vm.init(V, E), rows, ids)
    np.testing.assert_allclose(
        np.asarray(vm.summary(s, c)),
        np.asarray(jnp.mean(rows, axis=0)), rtol=1e-5, atol=1e-5)


def test_occupancy_marks_only_touched_slots():
    rows = jnp.ones((3, E), jnp.float32)
    ids = jnp.asarray([1, 1, 4], jnp.int32)
    s, c = vm.update(*vm.init(V, E), rows, ids)
    occ = np.asarray(vm.occupancy(c))
    assert occ[1] and occ[4] and not occ[0] and not occ[V]


def test_read_is_the_per_slot_mean():
    rows = jnp.asarray([[2.0] * E, [4.0] * E, [9.0] * E], jnp.float32)
    ids = jnp.asarray([1, 1, 3], jnp.int32)
    s, c = vm.update(*vm.init(V, E), rows, ids)
    r = np.asarray(vm.read(s, c))
    np.testing.assert_allclose(r[1], np.full(E, 3.0), rtol=1e-6)
    np.testing.assert_allclose(r[3], np.full(E, 9.0), rtol=1e-6)
    np.testing.assert_allclose(r[0], np.zeros(E))  # untouched reads as zero


def _agent(seed=0, policy="palimpsa"):
    from alphagrad.approx.ppo import _build_agent
    prev = os.environ.get("ALPHAGRAD_POLICY")
    os.environ["ALPHAGRAD_POLICY"] = policy
    try:
        args = SimpleNamespace(
            vocab_size=512, embd_dim=32, num_layers=2, num_heads=2,
            hidden_dim=64, value_dims="32", op_embd_dim=8, max_substeps=1)
        return _build_agent(args, total_v=V, num_factors=4, max_rules=4,
                            key=jr.PRNGKey(seed))
    finally:
        if prev is None:
            os.environ.pop("ALPHAGRAD_POLICY", None)
        else:
            os.environ["ALPHAGRAD_POLICY"] = prev


def test_pointer_from_memory_has_the_right_shapes():
    agent = _agent()
    n = 25
    rows = _rows(n, 6, width=32) * 0.1
    ids = _ids(n, 7)
    s, c = vm.update(*vm.init(V, 32), rows, ids)
    logits, reprs = agent.vertex_policy.from_vertex_memory(
        vm.read(s, c), vm.occupancy(c))
    assert logits.shape == (V,), logits.shape
    assert reprs.shape == (V, 32), reprs.shape
    assert np.all(np.isfinite(np.asarray(logits)))


def test_pointer_from_memory_is_chunk_order_invariant():
    """Two different chunkings of the same stream give the same logits."""
    agent = _agent()
    n = 30
    rows, ids = _rows(n, 8, width=32) * 0.1, _ids(n, 9)

    def _logits(chunks):
        s, c = vm.init(V, 32)
        for lo, hi in chunks:
            s, c = vm.update(s, c, rows[lo:hi], ids[lo:hi])
        return np.asarray(agent.vertex_policy.from_vertex_memory(
            vm.read(s, c), vm.occupancy(c))[0])

    a = _logits([(0, 30)])
    b = _logits([(0, 5), (5, 6), (6, 19), (19, 30)])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
