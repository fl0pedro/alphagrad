"""elimrl env tests (M0, CPU-fast).

(a) VERTEX-macro reverse trajectory == jacve('rev') == jax.grad numerically.
(b) mixed FACE/VERTEX trajectory == jax.jacrev numerically; history replays
    to an identical Jacobian (the worker's elim_plan contract).
(c) vertex_only exactly reproduces jacve legality (graphax _checkify_order).
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
import pytest

from graphax import jacve
from graphax.core import _build_graph, _checkify_order

from alphagrad.elimrl.env import ElimEnv, rev_policy


def _small():
    k = jax.random.split(jax.random.PRNGKey(0), 3)

    def fn(x, y, W1, b1, W2, b2):
        return jnp.sum((jnn.softmax(W2 @ jnn.relu(W1 @ x + b1) + b2) - y) ** 2)

    args = (jax.random.normal(k[0], (12,)), jnn.one_hot(2, 5),
            jax.random.normal(k[1], (10, 12)) / 5, jnp.zeros(10),
            jax.random.normal(k[2], (5, 10)) / 5, jnp.zeros(5))
    return fn, args, (2, 3, 4, 5)


def _flat(x):
    """jacve regroups into one tuple per outvar when n>1; flatten."""
    out = []
    for e in (x if isinstance(x, (list, tuple)) else (x,)):
        if isinstance(e, tuple):
            out.extend(e)
        else:
            out.append(e)
    return out


def _assert_blocks_close(got, ref, atol=1e-5, rtol=1e-4):
    got = [np.asarray(g) for g in got]
    ref = [np.asarray(r) for r in ref]
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol)


# -- (a) ---------------------------------------------------------------------
def test_vertex_macro_reverse_equals_jacve_and_grad():
    fn, args, an = _small()
    env = ElimEnv(fn, args, an, vertex_only=True)
    for vid in sorted(env.jacve_vertices, reverse=True):
        env.apply(("V", vid))
    assert env.done
    jac = env.jacobian()
    assert all(j is not None for j in jac)
    jv = _flat(jacve(fn, "rev", argnums=an)(*args))
    gr = jax.grad(fn, argnums=an)(*args)
    _assert_blocks_close(jac, jv)
    _assert_blocks_close(jac, list(gr))


# -- (b) ---------------------------------------------------------------------
def test_mixed_face_vertex_trajectory_matches_jacrev_and_replays():
    fn, args, an = _small()
    env = ElimEnv(fn, args, an)
    i = 0
    while not env.done:
        if i % 3 != 2:                       # two FACE micro-steps ...
            faces = env.faces()
            env.apply(("F",) + faces[i % len(faces)])
        else:                                # ... then one VERTEX macro
            env.apply(("V", env.legal_vertices()[-1]))
        i += 1
    jac = env.jacobian()
    assert all(j is not None for j in jac)

    ref = jax.jacrev(fn, argnums=an)(*args)
    _assert_blocks_close(jac, list(ref), atol=1e-4)

    # replay determinism: the recorded history is the worker's plan format.
    env2 = ElimEnv(fn, args, an)
    for act in env.history:
        env2.apply(tuple(act))
    assert env2.done
    for a_, b_ in zip(jac, env2.jacobian()):
        np.testing.assert_array_equal(np.asarray(a_), np.asarray(b_))


def test_rev_policy_full_face_trajectory_matches_grad():
    fn, args, an = _small()
    env = ElimEnv(fn, args, an)
    while not env.done:
        env.apply(rev_policy(env))
    _assert_blocks_close(env.jacobian(), list(jax.grad(fn, argnums=an)(*args)))


# -- (c) ---------------------------------------------------------------------
def test_vertex_only_reproduces_jacve_legality():
    fn, args, an = _small()
    env = ElimEnv(fn, args, an, vertex_only=True)

    closed = jax.make_jaxpr(fn)(*args)
    _, _, _, vo = _build_graph(closed.jaxpr, args, closed.literals)
    expect = set(_checkify_order("fwd", closed.jaxpr, vo))

    assert set(env.jacve_vertices) == expect
    assert set(env.legal_vertices()) == expect

    vid = max(expect)
    env.apply(("V", vid))
    assert set(env.legal_vertices()) == expect - {vid}
    assert vid in env.eliminated_vertices()

    # FACE actions are disabled in the jacve search space
    with pytest.raises(ValueError):
        env.apply(("F", 0, 1))
    # eliminating the same vertex twice is illegal (jacve semantics)
    with pytest.raises(ValueError):
        env.apply(("V", vid))

    st = env.state()
    assert st.faces == () and st.vertex_only and not st.done


# -- state-object interface (M2 encoder contract) ------------------------------
def test_state_interface_fields():
    fn, args, an = _small()
    env = ElimEnv(fn, args, an)
    st = env.state()
    assert st.nodes and st.faces and st.edges and not st.done
    node = st.nodes[0]
    for f in ("nid", "kind", "src_vertex", "snk_vertex", "val_shape",
              "val_dtype", "out_dims", "primal_dims"):
        assert hasattr(node, f)
    face = st.faces[0]
    for f in ("u", "w", "i_vertex", "j_vertex", "k_vertex", "absorb", "target"):
        assert hasattr(face, f)
    assert isinstance(st.eliminated, frozenset)
    # absorb-vs-create is state-determined and consistent with the engine
    for fm in st.faces:
        assert fm.absorb == (fm.target is not None)
        assert fm.absorb == (env.absorb_target(fm.u, fm.w) is not None)
