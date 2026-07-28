"""Single-vertex oracle probing (ALPHAGRAD_ORACLE_ONE_VERTEX=1).

The rollout used to probe legality masks for EVERY vertex and then read a
single row (`oracle_*_all[vertex_idx + 1]`). The perf path defers the probe
until the vertex has been sampled and probes only that one. These tests pin
the property that makes the swap safe:

    masks(candidates=[v])[v]  ==  masks()[v]        (bit-identical row)
    face_masks(v)             is independent of which other vertices were probed

i.e. probing is per-vertex PURE — no cross-vertex state accumulates in the
oracle — for a FRESH oracle and for one advanced along a prefix WITH rules.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.common.masks import LiveVertexMaskOracle

CASES = [
    ("sin_mul", lambda x, y: jnp.sin(x * y) + x,
     (jnp.ones((4, 4)), jnp.ones((4, 4)))),
    ("chain", lambda x, y: jnp.tanh(x @ y),
     (jnp.ones((4, 6)), jnp.ones((6, 4)))),
    ("branchy", lambda x, y: jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y),
     (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)),
]
IDS = [c[0] for c in CASES]


def _oracle(fn, args):
    cj = jax.make_jaxpr(fn)(*args)
    return LiveVertexMaskOracle(cj.jaxpr, cj.literals, args, (0, 1)), cj.jaxpr


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_candidate_restriction_is_row_identical(name, fn, args):
    """The row for v must not depend on whether other vertices were probed."""
    oracle, jaxpr = _oracle(fn, args)
    all_pair, all_comp = oracle.masks()
    for v in range(1, len(jaxpr.eqns) + 1):
        one_pair, one_comp = oracle.masks(candidates=[v])
        np.testing.assert_array_equal(
            one_pair[v], all_pair[v],
            err_msg=f"{name} v{v}: DIAG row differs under candidate restriction")
        np.testing.assert_array_equal(
            one_comp[v], all_comp[v],
            err_msg=f"{name} v{v}: COMPRESS row differs under restriction")
        # Every OTHER row must stay zero — the policy only reads row v, but a
        # non-zero elsewhere would mean the restriction leaked state.
        others = [u for u in range(one_pair.shape[0]) if u != v]
        assert not one_pair[others].any(), f"{name} v{v}: leaked DIAG rows"
        assert not one_comp[others].any(), f"{name} v{v}: leaked COMPRESS rows"


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_face_masks_probe_order_independent(name, fn, args):
    """face_masks(v) must be identical whether or not other vertices were
    probed first — the perf path probes exactly one vertex per callback."""
    o_seq, jaxpr = _oracle(fn, args)
    nv = len(jaxpr.eqns)
    # Reference: probe every vertex in order (what the old all-vertex host did).
    ref = {}
    for v in range(1, nv + 1):
        fp, fc, n = o_seq.face_masks(v, 8)
        ref[v] = (np.asarray(fp).copy(), np.asarray(fc).copy(), int(n))
    # Perf path: a FRESH oracle per vertex, probing only that vertex.
    for v in range(1, nv + 1):
        o_one, _ = _oracle(fn, args)
        fp, fc, n = o_one.face_masks(v, 8)
        np.testing.assert_array_equal(np.asarray(fp), ref[v][0],
                                      err_msg=f"{name} v{v}: face pair differs")
        np.testing.assert_array_equal(np.asarray(fc), ref[v][1],
                                      err_msg=f"{name} v{v}: face comp differs")
        assert int(n) == ref[v][2], f"{name} v{v}: face count differs"


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_identical_after_an_advanced_prefix(name, fn, args):
    """The real rollout probes AFTER advancing along the elimination prefix.
    Restriction must still be row-identical on the advanced graph."""
    nv = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
    if nv < 3:
        pytest.skip("needs >=3 vertices to advance a prefix")
    prefix = [1, 2]
    o_all, jaxpr = _oracle(fn, args)
    for v in prefix:
        o_all.advance(v)
    all_pair, all_comp = o_all.masks()

    remaining = [v for v in range(1, nv + 1) if v not in prefix]
    for v in remaining:
        o_one, _ = _oracle(fn, args)
        for pv in prefix:
            o_one.advance(pv)
        one_pair, one_comp = o_one.masks(candidates=[v])
        np.testing.assert_array_equal(
            one_pair[v], all_pair[v],
            err_msg=f"{name} v{v}: DIAG row differs after prefix advance")
        np.testing.assert_array_equal(
            one_comp[v], all_comp[v],
            err_msg=f"{name} v{v}: COMPRESS row differs after prefix advance")
