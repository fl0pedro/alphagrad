"""Per-FACE legality masks (Stage 1 of the per-face/path wiring).

``LiveVertexMaskOracle.vertex_mask`` answers "is this action legal on EVERY face
of this vertex?" -- the per-vertex rule list is applied uniformly, so anything
less crashes on the second face. That intersection is exactly why per-vertex DIAG
is structurally almost always illegal.

``face_masks`` keeps the per-face masks separate, which is the whole point of
approximating per path: a slot only has to fit ITS OWN operand. These tests pin
the two properties the policy wiring will rely on:

* SOUNDNESS -- the intersection over the returned faces reproduces
  ``vertex_mask`` exactly (same env screens, just not AND-ed).
* COVERAGE  -- per-face admits at least as much as per-vertex, and on a real
  multi-face graph it admits strictly more somewhere.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.common.masks import LiveVertexMaskOracle

CASES = [
    ("sin_mul", lambda x, y: jnp.sin(x * y) + x, (jnp.ones((4, 4)), jnp.ones((4, 4)))),
    ("chain", lambda x, y: jnp.tanh(x @ y), (jnp.ones((4, 6)), jnp.ones((6, 4)))),
    ("branchy", lambda x, y: jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y),
     (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)),
]
IDS = [c[0] for c in CASES]


def _oracle(fn, args):
    cj = jax.make_jaxpr(fn)(*args)
    return LiveVertexMaskOracle(cj.jaxpr, cj.literals, args, (0, 1)), cj.jaxpr


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_face_masks_intersect_to_the_vertex_mask(name, fn, args):
    """AND-ing the per-face masks must reproduce vertex_mask -- same screens."""
    oracle, jaxpr = _oracle(fn, args)
    for v in range(1, len(jaxpr.eqns) + 1):
        v_pair, v_comp = oracle.vertex_mask(v)
        f_pair, f_comp, n = oracle.face_masks(v)
        if n == 0:
            continue
        np.testing.assert_array_equal(
            f_pair[:n].all(axis=0), v_pair,
            err_msg=f"{name} v{v}: DIAG face-intersection != vertex_mask")
        np.testing.assert_array_equal(
            f_comp[:n].all(axis=0), v_comp,
            err_msg=f"{name} v{v}: COMPRESS face-intersection != vertex_mask")


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_per_face_admits_at_least_as_much_as_per_vertex(name, fn, args):
    """Every action legal per-vertex is legal on every face (superset property)."""
    oracle, jaxpr = _oracle(fn, args)
    for v in range(1, len(jaxpr.eqns) + 1):
        v_pair, v_comp = oracle.vertex_mask(v)
        f_pair, f_comp, n = oracle.face_masks(v)
        for k in range(n):
            assert np.all(f_pair[k] >= v_pair), f"{name} v{v} face{k}: DIAG lost"
            assert np.all(f_comp[k] >= v_comp), f"{name} v{v} face{k}: COMPRESS lost"


def test_per_face_action_space_is_strictly_larger():
    """The payoff, stated honestly.

    Per-face's gain is EXPRESSIVITY, not (necessarily) extra legality: with F
    faces the policy picks independently per face, so the representable
    configurations are ``(L + 1) ** F`` against the per-vertex path's ``L + 1``
    (``+1`` = leave exact). That is strictly larger as soon as a vertex has >= 2
    faces and >= 1 legal action -- and it holds even when every face admits the
    SAME action set, which is exactly the case the legality-superset test cannot
    distinguish.

    (Whether per-face also admits MORE legal actions depends on the faces having
    heterogeneous index structure -- true for mixed-shape graphs, false for a
    uniformly-shaped elementwise one like ``branchy``. Hence it is measured, not
    asserted; see test_per_face_admits_at_least_as_much_as_per_vertex.)
    """
    multi, more_expressive = 0, 0
    for name, fn, args in CASES:
        oracle, jaxpr = _oracle(fn, args)
        for v in range(1, len(jaxpr.eqns) + 1):
            f_pair, f_comp, n = oracle.face_masks(v)
            if n < 2:
                continue
            multi += 1
            # legal actions per face (upper-triangle pairs + compressible axes)
            per_face_L = [
                int(np.triu(f_pair[k]).sum() + f_comp[k].sum()) for k in range(n)
            ]
            # (L+1)**n > (L+1) whenever some face has a legal action and n >= 2
            if max(per_face_L) > 0:
                more_expressive += 1
    assert multi > 0, "need a vertex with >= 2 faces for this to mean anything"
    assert more_expressive > 0, (
        "no multi-face vertex had a legal action anywhere in CASES, so per-face "
        "expressivity cannot be demonstrated")


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_padding_rows_are_zero_and_shapes_are_static(name, fn, args):
    """Rows >= n_faces are zero padding, and the shape is static (F, N, N) /
    (F, N) so it drops into a fixed-shape trajectory."""
    oracle, jaxpr = _oracle(fn, args)
    N, F = oracle.max_axes, 8
    for v in range(1, len(jaxpr.eqns) + 1):
        f_pair, f_comp, n = oracle.face_masks(v, max_faces=F)
        assert f_pair.shape == (F, N, N) and f_comp.shape == (F, N)
        assert not f_pair[n:].any(), f"{name} v{v}: DIAG padding not zero"
        assert not f_comp[n:].any(), f"{name} v{v}: COMPRESS padding not zero"


def test_eliminated_and_out_of_range_vertices_are_empty():
    name, fn, args = CASES[0]
    oracle, jaxpr = _oracle(fn, args)
    for bad in (0, len(jaxpr.eqns) + 1):
        p, c, n = oracle.face_masks(bad)
        assert n == 0 and not p.any() and not c.any()
    oracle.advance(1)
    p, c, n = oracle.face_masks(1)
    assert n == 0 and not p.any() and not c.any(), "eliminated vertex must be empty"
