"""One elimination per vertex returns the chunks the per-face path returns.

``LiveFaceStream.chunk_ex`` eliminates each (prefix, vertex) once and
re-eliminates only a decided face alone (dsnn-dfw.119, owner ruling
2026-09-23: the faces of one vertex do not depend on each other).
``chunk_ex_per_face`` is the path it replaces: one full elimination of the
vertex per face, with faces 0..f-1 decided. Every output of every face must
be bit-identical between the two.

The reproduction replays the three TLM plans of job 67489 episode 237
(``golden/onelim_plans_faces.jsonl``, the plans of probe job 67624) in a fresh
interpreter, because ``ALPHAGRAD_MAX_FACES=128`` and the TLM shape pins must
be in force before ``env.py`` is imported. The small-graph case drives SKIP
decisions, which arm the vertex's approx flag and so exercise the one-face
eliminations of undecided faces as well.
"""
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PLANS = HERE / "golden" / "onelim_plans_faces.jsonl"
WIKITEXT = os.environ.get("DSNN_WIKITEXT_DIR",
                          "/Scratch/assmuth/mrg/cache/dsnn_wikitext")


def test_three_tlm_plans_every_face_identical():
    if not os.path.isdir(WIKITEXT):
        pytest.skip(f"the TLM target needs the wikitext cache at {WIKITEXT}")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env.update({
        "JAX_PLATFORMS": "cpu",
        "DSNN_WIKITEXT_DIR": WIKITEXT,
        "ALPHAGRAD_TLM_SEQ": "32",
        "ALPHAGRAD_TLM_DMODEL": "128",
        "ALPHAGRAD_TLM_VOCAB": "1024",
        "ALPHAGRAD_MAX_FACES": "128",
        "ALPHAGRAD_SKIP_COST_ANALYSIS": "1",
        "ALPHAGRAD_SKIP_COUNT_OPS": "1",
    })
    r = subprocess.run(
        [sys.executable, str(HERE / "_onelim_tlm_replay.py"), str(PLANS)],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "PROBE-OK" in r.stdout, (
        f"rc={r.returncode}\n--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr[-6000:]}")


def _perceptron():
    import jax
    import jax.random as jrand
    from graphax.examples import Perceptron

    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (2, 4))
    y = jrand.normal(jrand.fold_in(key, 5), (2, 3))
    W1 = jrand.normal(jrand.fold_in(key, 1), (4, 6))
    b1 = jrand.normal(jrand.fold_in(key, 2), (6,))
    W2 = jrand.normal(jrand.fold_in(key, 3), (6, 3))
    b2 = jrand.normal(jrand.fold_in(key, 4), (3,))
    gamma = jrand.normal(jrand.fold_in(key, 6), (6,))
    beta = jrand.normal(jrand.fold_in(key, 7), (6,))
    args = (x, y, W1, b1, W2, b2, gamma, beta)
    cj = jax.make_jaxpr(Perceptron)(*args)
    return cj.jaxpr, list(cj.literals), list(args)


def _decisions(rng, F, S, n_faces):
    from alphagrad.approx.env import QUANT_SENTINEL
    from alphagrad.approx.common.plan_log import quant_dtype_id
    rows = -np.ones((F, S, 3), np.int32)
    rows[:, :, 2] = 0
    skips = np.zeros((F,), np.int32)
    q = quant_dtype_id("bfloat16")
    for j in range(n_faces):
        u = rng.random()
        if u < 0.25:
            skips[j] = 1
        elif u < 0.6:
            rows[j, int(rng.integers(S))] = (QUANT_SENTINEL, q, 1)
    return rows, skips


def test_small_graph_decided_and_skipped_faces_identical():
    from alphagrad.approx.env import MAX_FACES, wire_slots
    from alphagrad.approx.live_faces import LiveFaceStream

    jaxpr, consts, args = _perceptron()
    argnums = (2, 3, 4, 5)
    V = len(jaxpr.eqns)
    F, S, MR = MAX_FACES, wire_slots(), 16

    def mk(one_elim):
        s = LiveFaceStream(jaxpr, argnums, consts, args, vocab=256,
                           max_faces=F, max_axes=8, window=8192)
        s.one_elim = one_elim
        return s

    old, new = mk(False), mk(True)
    order = np.zeros((V,), np.int32)
    specs = -np.ones((V, MR, 3), np.int32)
    vspecs = -np.ones((MR, 3), np.int32)
    rng = np.random.default_rng(119)
    multi = 0
    for v in range(1, V + 1):
        exact = _decisions(rng, F, S, 0)
        nf = int(old.chunk_ex(order, specs, 0, v, vspecs, *exact, 0)[2])
        multi += nf >= 2
        # Three decision draws per vertex: each changes what faces 0..f-1
        # carry, so the second and third walk restart the render cursor.
        for _draw in range(3):
            rows, skips = _decisions(rng, F, S, nf)
            for f in range(nf):
                a = old.chunk_ex(order, specs, 0, v, vspecs, rows, skips, f)
                b = new.chunk_ex(order, specs, 0, v, vspecs, rows, skips, f)
                for name, x, y in zip(("tokens", "count", "n_faces", "ends",
                                       "ekey", "cvx", "head", "wrok"), a, b):
                    x, y = np.asarray(x), np.asarray(y)
                    assert x.dtype == y.dtype and np.array_equal(x, y), (
                        f"vertex {v} face {f}: {name} differs")
    assert multi >= 2, "the graph has no multi-face vertex left to test"
    assert new.stats["onelim_fallback"] == 0, new.last_onelim_error
    assert new.stats["face_elims"] > 0
    assert new.stats["elims"] < old.stats["elims"]
