# The merged count + slot-legality callback read off the chunk stream's one
# elimination per (prefix, vertex) returns what its own recording elimination
# returned, for every vertex (dsnn-dfw.131). The TLM replay runs in a fresh
# interpreter because ALPHAGRAD_MAX_FACES=128 and the TLM pins must be in force
# before env.py is imported.
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PLANS = HERE / "golden" / "onelim_plans_faces.jsonl"
WIKITEXT = os.environ.get("DSNN_WIKITEXT_DIR",
                          "/Scratch/assmuth/mrg/cache/dsnn_wikitext")


def test_three_tlm_plans_every_vertex_identical():
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
        [sys.executable, str(HERE / "_onelim_legality_replay.py"), str(PLANS)],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "REPLAY-OK" in r.stdout, (
        f"rc={r.returncode}\n--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr[-6000:]}")


@pytest.mark.parametrize("kind", ["forward", "reverse", "shuffled"])
def test_perceptron_random_skip_and_quant_identical(kind):
    import _onelim_legality_replay as R
    from onelim_face_chunk_test import _decisions, _perceptron
    from alphagrad.approx.env import MAX_FACES, wire_slots

    jaxpr, consts, args = _perceptron()
    argnums = (2, 3, 4, 5)
    V = len(jaxpr.eqns)
    F, S, MR = MAX_FACES, wire_slots(), 16
    order = np.arange(1, V + 1, dtype=np.int32)
    seed = 131 + ("forward", "reverse", "shuffled").index(kind)
    if kind == "reverse":
        order = order[::-1].copy()
    elif kind == "shuffled":
        order = np.random.default_rng(seed).permutation(order)
    specs = -np.ones((V, MR, 3), np.int32)

    walks = []
    for slot_one_elim in (False, True):
        s, cb = R.callbacks(jaxpr, argnums, consts, args, slot_one_elim,
                            F, 8, 8192)
        fspecs = -np.ones((V, F, S, 3), np.int32)
        fspecs[..., 2] = 0
        fskips = np.zeros((V, F), np.int32)
        rng = np.random.default_rng(seed)

        def decide(n, nf, fspecs=fspecs, fskips=fskips, rng=rng):
            fspecs[n], fskips[n] = _decisions(rng, F, S, nf)

        leg, chunks, _tl, _tc = R.walk(s, cb, order, specs, fspecs, fskips,
                                       decide=decide)
        walks.append((leg, chunks, s, fskips))

    (lo, co, so, _ko), (ln, cn, sn, kn) = walks
    assert R.first_difference(lo, ln, R.LEG_FIELDS) is None
    assert R.first_difference(co, cn, R.CHUNK_FIELDS) is None
    if kind == "forward":
        assert sum(int(out[5]) >= 2 for _n, _v, out in ln) >= 2
    assert int(kn.sum()) > 0
    assert sn.stats["slot_onelim"] > 0, sn.last_slot_onelim_error
    assert sn.stats["slot_probe"] == sn.stats["slot_onelim_fallback"]
    assert sn.stats["count_onelim"] > 0
    assert sn.stats["elims"] <= so.stats["elims"]


def test_the_shared_count_still_refuses_a_vertex_wider_than_the_wire():
    from types import SimpleNamespace
    from alphagrad.approx import live_faces as LF
    from alphagrad.approx.env import wire_slots

    tk = SimpleNamespace(ij=SimpleNamespace(faces=lambda v: []))
    s = LF.LiveFaceStream.__new__(LF.LiveFaceStream)
    s.max_faces = 64
    s.stats = {"failures": 0}
    s.slot_one_elim = True
    s._tokenizer_at = lambda *a, **k: tk
    order, specs = np.zeros((3,), np.int32), np.zeros((3, 1, 3), np.int32)

    def hist(w):
        return (-np.ones((3, w, wire_slots(), 3), np.int32),
                np.zeros((3, w), np.int32))

    frh, fsh = hist(3)
    st = SimpleNamespace(tk=tk, keys=[(i, i + 1) for i in range(5)])
    key = ((order[:2].tobytes(), specs[:2].tobytes(), 1, None)
           + LF._hist_key_parts(frh, fsh, 2))
    s._onelim = {key: st}
    with pytest.raises(RuntimeError, match="face wire"):
        s.n_faces(order, specs, 2, 1, frh, fsh)
    frh8, fsh8 = hist(8)
    s._onelim = {key[:4] + LF._hist_key_parts(frh8, fsh8, 2): st}
    assert s.n_faces(order, specs, 2, 1, frh8, fsh8) == 5
    assert s.stats["count_onelim"] == 2
