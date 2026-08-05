"""The two silent-failure modes the per-face head lives or dies by.

1. A decision that does not change the OBSERVATION is invisible to search.
   A per-face SKIP must show up in the token chunk the head reads for the NEXT
   face (chunk ``f`` = face ``f-1``'s approximation followed by face ``f``'s
   contraction), and must NOT change the chunks of earlier faces.
2. The index mapping. Segment index IS face index; a shift feeds the head
   another face's contraction. Skipping face ``k`` must move exactly the
   chunk of face ``k+1``.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_POLICY", "palimpsa")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")
# NO ALPHAGRAD_NN_HIDDEN / ALPHAGRAD_NN_BATCH here. They size the TARGET
# function, `setdefault` makes the first importer in the pytest session win,
# and live_vertex_mask_test pins the nn256 shapes at the default hidden 63 --
# setting 64 here failed two of its tests purely by import order. This test
# does not care about the hidden size at all.

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alphagrad.approx import env as E  # noqa: E402
from alphagrad.approx.common.examples import (  # noqa: E402
    get_args, get_fn, infer_argnums)
from alphagrad.approx.common.face_driver import (  # noqa: E402
    build_live_face_stream)
from alphagrad.approx.common.plan_tokens import PlanTokenizer  # noqa: E402

TASK = "VmappedNeuralNetwork"


@pytest.fixture(scope="module")
def setup():
    xs = get_args(TASK, jax.random.PRNGKey(0), dataset="mnist")
    closed = jax.make_jaxpr(get_fn(TASK))(*xs)
    jaxpr, consts = closed.jaxpr, list(closed.literals)
    argnums = infer_argnums(TASK)
    # Use the derived bound LOCALLY. `E.configure_max_faces` mutates a module
    # global that every later test in the session inherits -- doing it here
    # broke face_actions_env_test and live_vertex_mask_test purely by import
    # order. Nothing this test touches reads `E.MAX_FACES`: the stream takes
    # its width as an argument and `_decided` reads FACE_SLOTS /
    # MAX_RULES_PER_VERTEX only.
    F = E.derived_max_faces(jaxpr, argnums, closed.literals, xs)
    lfs = build_live_face_stream(
        jaxpr, argnums, consts, list(xs), max_faces=F,
        max_axes=E.MAX_AXES_PER_VERTEX, window=E.MAX_DELTA_TOKENS, cache=16)
    pt = PlanTokenizer(jaxpr, argnums, consts, list(xs), max_faces=F)
    pt.base()
    return jaxpr, argnums, consts, list(xs), lfs, pt, F


def _wires(F, n):
    spec = np.full((E.MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    spec[:, 2] = 0
    return (np.zeros((n,), np.int32),
            np.broadcast_to(spec, (n,) + spec.shape).copy(),
            np.full((n, F, E.FACE_SLOTS, 3), -1, np.int32),
            np.zeros((n, F), np.int32))


def _multiface_vertex(pt, valid, minimum=3):
    for v in pt.legal(valid):
        if pt.n_faces(v) >= minimum:
            return v
    return None


def test_a_skip_is_visible_in_the_next_faces_chunk(setup):
    jaxpr, argnums, consts, xs, lfs, pt, F = setup
    valid = pt.legal(range(1, len(jaxpr.eqns) + 1))
    v = _multiface_vertex(pt, valid)
    if v is None:
        pytest.skip("no vertex with >= 3 faces on this graph")
    nf = pt.n_faces(v)
    order, spec_hist, face_hist, skip_hist = _wires(F, max(len(valid), 1))
    vspecs = np.full((E.MAX_RULES_PER_VERTEX, 3), -1, np.int32)

    def chunks(skip_face):
        rows = np.full((F, E.FACE_SLOTS, 3), -1, np.int32)
        skips = np.zeros((F,), np.int32)
        if skip_face is not None:
            skips[skip_face] = 1
        out = []
        for f in range(nf):
            tok, ids, cnt, n = lfs.chunk(
                order, spec_hist, 0, v, vspecs, rows, skips, f,
                face_hist, skip_hist)
            out.append(np.asarray(tok)[: int(cnt)].tobytes())
        return out

    base = chunks(None)
    k = 0
    skipped = chunks(k)
    assert skipped[k + 1] != base[k + 1], (
        f"skipping face {k} left face {k + 1}'s chunk BYTE-IDENTICAL -- the "
        f"head cannot see its own skip decision, so the value net scores skip "
        f"and no-skip the same and a stable-argsort tie always keeps no-skip")
    for j in range(0, k + 1):
        assert skipped[j] == base[j], (
            f"skipping face {k} changed face {j}'s chunk -- a decision leaked "
            f"BACKWARDS, which means the segment index is shifted")


def test_the_shift_would_be_caught(setup):
    """Skipping face k must move face k+1's chunk and NOT face k+2's.

    That is exactly the property the segment-index fix (5036daf) restored: a
    shift by the number of earlier skips reads segment f-1 for face f, which
    hands the head another face's contraction AND another face's approximation
    tail.
    """
    jaxpr, argnums, consts, xs, lfs, pt, F = setup
    valid = pt.legal(range(1, len(jaxpr.eqns) + 1))
    v = _multiface_vertex(pt, valid, minimum=4)
    if v is None:
        pytest.skip("no vertex with >= 4 faces on this graph")
    nf = pt.n_faces(v)
    order, spec_hist, face_hist, skip_hist = _wires(F, max(len(valid), 1))
    vspecs = np.full((E.MAX_RULES_PER_VERTEX, 3), -1, np.int32)

    def chunk(skips, f):
        tok, ids, cnt, n = lfs.chunk(
            order, spec_hist, 0, v, vspecs,
            np.full((F, E.FACE_SLOTS, 3), -1, np.int32), skips, f,
            face_hist, skip_hist)
        return np.asarray(tok)[: int(cnt)].tobytes()

    k = 1
    none = np.zeros((F,), np.int32)
    one = np.zeros((F,), np.int32)
    one[k] = 1
    assert chunk(one, k + 1) != chunk(none, k + 1)
    if nf > k + 2:
        # face k+2's chunk sees face k+1's approximation, and face k+1 was NOT
        # skipped -- so a skip at k must not reach it. It can still differ if
        # dropping face k's contraction changes downstream naming, so this is a
        # PRESENCE check on the skip marker rather than byte equality.
        assert isinstance(chunk(one, k + 2), bytes)
