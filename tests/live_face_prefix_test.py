"""The per-face token stream must READ the graph the measurement BUILDS.

Two failures were live in the PPO arm and both were silent -- a healthy
health line, a converging run, an observation that described a different
object from the one being scored.

BUG 1 -- the prefix cache key omitted the face wires. ``LiveFaceStream``
keyed its cached prefix tokenizer on ``(order[:n], specs[:n])``. Under
``--live-faces`` ppo.py sets the per-vertex specs to all-exact END rows for
every vertex (approximation is purely per-face), so ``specs[:n]`` is a
CONSTANT and the key degenerated to the elimination ORDER: two plans that
differed only in what the face head decided were served ONE tokenizer.

BUG 2 -- the prefix replay dropped the prefix's approximations. The replay
loop applied only the per-vertex ``rules`` (always empty under
``--live-faces``) and passed NO face transforms, so every chunk the head read
was computed on an EXACT prefix while the measurement
(``env._face_transforms_for_order`` -> ``ft_by_vertex``) built an
approximated one.

Both directions are demonstrated in each test: passing the history arrays
(the fix) against NOT passing them (the old code path, still reachable
through the defaults), so a regression that quietly stops threading the
history through ppo.py's host callbacks fails HERE rather than in a
three-day campaign.

The reference is env's OWN builder -- ``_face_transforms_for_order``, the
thing the measured elimination is handed -- never a re-derivation, because a
re-derivation would relocate the bug into the test.

The graph is `per_face_apply_test`'s ``_mlp``: its first vertex is the one
graph in the test suite where a per-FACE ``Diag`` is PROVEN to land on the
live operand (``test_per_face_differs_from_exact``). On smaller graphs every
face hands its slots an operand where nothing is legal, ``make_live_masked_hook``
correctly skips, and a golden test would then be pinning the fail-soft path
and calling it a pass.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alphagrad.approx.env as E
from alphagrad.approx.env import FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX
from alphagrad.approx.live_faces import LiveFaceStream
from graphax import IncrementalPathTokenizer


def _mlp(x, W1, W2):
    return jnp.tanh(x @ W1) @ W2


ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1)
ARGNUMS = (1, 2)
VOCAB = 512


def _mk():
    cj = jax.make_jaxpr(_mlp)(*ARGS)
    cfg = SimpleNamespace(jaxpr=cj.jaxpr, argnums=ARGNUMS, per_face=True)
    return cfg, list(cj.literals), list(ARGS)


def _stream(cfg, consts, args):
    return LiveFaceStream(cfg.jaxpr, ARGNUMS, consts, args, vocab=VOCAB,
                          max_faces=MAX_FACES, max_axes=8, window=8192)


def _exact_specs(T):
    """All-exact per-vertex rows -- what ppo.py writes under --live-faces, and
    the reason the old key degenerated to the vertex order."""
    return -np.ones((T, MAX_RULES_PER_VERTEX, 3), np.int32)


def _exact_faces(T):
    return np.full((T, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)


def _exact_skips(T):
    return np.zeros((T, MAX_FACES), np.int32)


def _ref_face0(cfg, consts, args, order, specs, faces, skips, n, v_cur):
    """Face 0 of ``v_cur`` on the prefix graph THE MEASUREMENT BUILDS.

    ``honor_last_compress=False`` puts every prefix vertex on ``is_last=False``
    -- the decode the terminal measurement gives a vertex that is not the last
    of the order, and the one ``_tokenizer_at`` replays with. The searched
    decisions below are DIAG, for which ``is_last`` is irrelevant, so nothing
    here depends on that choice.
    """
    ft_ref = E._face_transforms_for_order(
        cfg, consts, args,
        [int(v) for v in order[:n]],
        [np.asarray(specs[k]).tolist() for k in range(n)],
        [np.asarray(faces[k]).tolist() for k in range(n)],
        [np.asarray(skips[k]).tolist() for k in range(n)],
        honor_last_compress=False,
    )
    ref = IncrementalPathTokenizer(cfg.jaxpr, ARGNUMS, list(consts),
                                   list(args), vocab_size=VOCAB)
    ref.base_tokens()
    for k in range(n):
        v = int(order[k])
        # per-vertex rules are empty (all-exact rows), so the ONLY thing
        # separating the two plans is this dict.
        ref.eliminate(v, (), ft_ref.get(v))
    toks = [int(t) for t in ref.eliminate(int(v_cur), ())]
    segs = ref.last_face_segments()
    if not segs:
        return None
    s, split, _e = segs[0]
    return toks[s:split]


_FOUND: list = []


def _discriminating():
    """A prefix face decision that CHANGES what the next vertex's face 0
    reads -- found with ENV's builder, not with the code under test.

    Searched rather than hardcoded: which (face, slot, rule) triples survive
    ``make_live_masked_hook``'s live-operand legality check is a property of
    graphax's index structure, not of anything this module owns.
    """
    if _FOUND:
        return _FOUND[0]
    cfg, consts, args = _mk()
    T = len(cfg.jaxpr.eqns)
    order = np.arange(1, T + 1, dtype=np.int32)
    specs = _exact_specs(T)
    skips = _exact_skips(T)
    base_faces = _exact_faces(T)

    trials = [np.array([i, j, fac], np.int32)
              for fac in (-1, 2, 4)
              for i in range(2) for j in range(2)]

    for n in range(1, T):
        v_cur = int(order[n])
        try:
            plain = _ref_face0(cfg, consts, args, order, specs, base_faces,
                               skips, n, v_cur)
        except Exception:
            continue
        if not plain:
            continue
        for k in range(n):
            for face in range(min(3, MAX_FACES)):
                for slot in range(FACE_SLOTS):
                    for tr in trials:
                        faces = base_faces.copy()
                        faces[k, face, slot] = tr
                        try:
                            got = _ref_face0(cfg, consts, args, order, specs,
                                             faces, skips, n, v_cur)
                        except Exception:
                            continue
                        if got is not None and got != plain:
                            _FOUND.append(
                                (cfg, consts, args, order, specs, faces,
                                 base_faces, skips, n, v_cur, plain, got,
                                 (k, face, slot, tuple(int(x) for x in tr))))
                            return _FOUND[0]
    return None


@pytest.fixture(scope="module")
def disc():
    d = _discriminating()
    if d is None:
        pytest.skip("no prefix face decision changed the next vertex's "
                    "face-0 tokens on this graph")
    return d


# --------------------------------------------------------------------------- #
def test_prefix_key_separates_plans_differing_only_in_face_decisions(disc):
    """GOLDEN. Same vertex order, different face decisions -> different chunk.

    Old code: byte-identical, because the key and the replay both ignored the
    face wires. That direction is asserted too, so this is a discriminator
    rather than a coincidence.
    """
    (cfg, consts, args, order, specs, faces_a, faces_b, skips, n, v_cur,
     _exact_toks, _approx_toks, where) = disc
    lfs = _stream(cfg, consts, args)
    vspecs = -np.ones((MAX_RULES_PER_VERTEX, 3), np.int32)
    rows0 = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    sk0 = np.zeros((MAX_FACES,), np.int32)

    # --- OLD behaviour: no history -> ONE tokenizer serves both plans -----
    lfs._chunks.clear()
    lfs._prefix.clear()
    old_a = lfs.chunk(order, specs, n, v_cur, vspecs, rows0, sk0, 0)
    lfs._chunks.clear()
    old_b = lfs.chunk(order, specs, n, v_cur, vspecs, rows0, sk0, 0)
    assert np.array_equal(old_a[0], old_b[0])
    assert len(lfs._prefix) == 1, "the degenerate key is the bug being fixed"

    # --- NEW behaviour: the face wires are in the key AND in the replay ---
    lfs._chunks.clear()
    lfs._prefix.clear()
    new_a = lfs.chunk(order, specs, n, v_cur, vspecs, rows0, sk0, 0,
                      faces_a, skips)
    new_b = lfs.chunk(order, specs, n, v_cur, vspecs, rows0, sk0, 0,
                      faces_b, skips)
    assert len(lfs._prefix) == 2, (
        f"two plans with the same order {order[:n].tolist()} and different "
        f"face wires {where} still share one prefix tokenizer")
    assert int(new_a[2]) > 0 and int(new_b[2]) > 0
    assert not np.array_equal(new_a[0], new_b[0]), (
        f"face decision {where} left the next vertex's chunk unchanged")


def test_prefix_replay_matches_the_measurements_graph(disc):
    """CONSISTENCY. The chunk the policy reads is computed on the SAME
    approximated prefix ``_face_transforms_for_order`` hands the measurement.

    Both directions: with the history the chunk equals the measurement's view;
    without it, it equals the EXACT-prefix view instead -- which is precisely
    what the rollout was feeding the head.
    """
    (cfg, consts, args, order, specs, faces_a, faces_b, skips, n, v_cur,
     want_exact, want_approx, where) = disc
    lfs = _stream(cfg, consts, args)
    vspecs = -np.ones((MAX_RULES_PER_VERTEX, 3), np.int32)
    rows0 = np.full((MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    sk0 = np.zeros((MAX_FACES,), np.int32)

    assert want_approx != want_exact  # the discriminator is real

    lfs._chunks.clear()
    lfs._prefix.clear()
    tok, _ids, cnt, _nf = lfs.chunk(order, specs, n, v_cur, vspecs, rows0,
                                    sk0, 0, faces_a, skips)
    got = [int(x) for x in tok[:int(cnt)]]
    assert got == want_approx, (
        f"chunk read a graph the measurement never builds (decision {where}): "
        f"{len(got)} tokens vs {len(want_approx)}")

    # ... and the OLD path reproduces the divergence it was suffering from.
    lfs._chunks.clear()
    lfs._prefix.clear()
    tok_o, _i, cnt_o, _n = lfs.chunk(order, specs, n, v_cur, vspecs, rows0,
                                     sk0, 0)
    assert [int(x) for x in tok_o[:int(cnt_o)]] == want_exact


def test_face_count_is_prefix_face_aware(disc):
    """``n_faces`` is the rollout while_loop's TRIP COUNT, enumerated on the
    same replayed prefix -- so it takes the history too. A trip count read off
    a graph the measurement never builds is an action the loss cannot score."""
    (cfg, consts, args, order, specs, faces_a, _fb, skips, n, v_cur,
     *_rest) = disc
    lfs = _stream(cfg, consts, args)
    lfs._prefix.clear()
    k_hist = lfs.n_faces(order, specs, n, v_cur, faces_a, skips)
    assert len(lfs._prefix) == 1
    # The same call WITHOUT the history must not be served the approximated
    # tokenizer: distinct key, distinct replay.
    k_none = lfs.n_faces(order, specs, n, v_cur)
    assert len(lfs._prefix) == 2
    assert k_hist > 0 and k_none > 0
