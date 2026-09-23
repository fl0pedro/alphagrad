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


@pytest.mark.parametrize("reverse", [False, True])
def test_perceptron_random_skip_and_quant_identical(reverse):
    import _onelim_legality_replay as R
    from onelim_face_chunk_test import _decisions, _perceptron
    from alphagrad.approx.env import MAX_FACES, wire_slots

    jaxpr, consts, args = _perceptron()
    argnums = (2, 3, 4, 5)
    V = len(jaxpr.eqns)
    F, S, MR = MAX_FACES, wire_slots(), 16
    order = np.arange(1, V + 1, dtype=np.int32)
    if reverse:
        order = order[::-1].copy()
    specs = -np.ones((V, MR, 3), np.int32)

    walks = []
    for slot_one_elim in (False, True):
        s, cb = R.callbacks(jaxpr, argnums, consts, args, slot_one_elim,
                            F, 8, 8192)
        fspecs = -np.ones((V, F, S, 3), np.int32)
        fspecs[..., 2] = 0
        fskips = np.zeros((V, F), np.int32)
        rng = np.random.default_rng(131 + int(reverse))

        def decide(n, nf, fspecs=fspecs, fskips=fskips, rng=rng):
            fspecs[n], fskips[n] = _decisions(rng, F, S, nf)

        leg, chunks, _tl, _tc = R.walk(s, cb, order, specs, fspecs, fskips,
                                       decide=decide)
        walks.append((leg, chunks, s, fskips))

    (lo, co, _so, _ko), (ln, cn, sn, kn) = walks
    assert R.first_difference(lo, ln, R.LEG_FIELDS) is None
    assert R.first_difference(co, cn, R.CHUNK_FIELDS) is None
    assert sum(int(out[5]) >= 2 for _n, _v, out in ln) >= 2
    assert int(kn.sum()) > 0
    assert sn.stats["slot_onelim"] > 0, sn.last_slot_onelim_error
    assert sn.stats["slot_probe"] == sn.stats["slot_onelim_fallback"]
    assert sn.stats["count_onelim"] > 0
