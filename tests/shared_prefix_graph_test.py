# dsnn-dfw.189: one prefix graph per environment changes no token, chunk or mask.
import json
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
DATA = {
    "DSNN_WIKITEXT_DIR": "/Scratch/assmuth/mrg/cache/dsnn_wikitext",
    "DSNN_MNIST_DIR": "/Scratch/assmuth/mrg/cache/dsnn_mnist",
    "DSNN_SHD_DIR": "/Scratch/assmuth/mrg/cache/dsnn_shd",
}
PINS = {
    "tlm": {"ALPHAGRAD_TLM_SEQ": "32", "ALPHAGRAD_TLM_DMODEL": "128",
            "ALPHAGRAD_TLM_VOCAB": "1024"},
    "nn256": {"ALPHAGRAD_NN_HIDDEN": "256"},
    "rtrl": {},
}
CASES = [("tlm", "markowitz", 3, 0), ("tlm", "free", 2, 3),
         ("nn256", "free", 3, 1), ("rtrl", "free", 2, 2)]


@pytest.mark.parametrize("target,kind,envs,seed", CASES,
                         ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_shared_prefix_graph_is_invisible_over_whole_episodes(
        target, kind, envs, seed):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env.update({"JAX_PLATFORMS": "cpu", "ALPHAGRAD_SKIP_COST_ANALYSIS": "1",
                "ALPHAGRAD_SKIP_COUNT_OPS": "1"})
    for k, v in DATA.items():
        env.setdefault(k, v)
    env.update(PINS[target])
    r = subprocess.run(
        [sys.executable, str(HERE / "_shared_prefix_replay.py"), target,
         kind, str(envs), str(seed)],
        env=env, capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0 and "REPLAY-OK" in r.stdout, (
        f"rc={r.returncode}\n--- stdout ---\n{r.stdout[-8000:]}\n"
        f"--- stderr ---\n{r.stderr[-6000:]}")
    line = next(x for x in r.stdout.splitlines() if x.startswith("STATS "))
    st = json.loads(line[len("STATS "):])
    sep, sha = st["separate"], st["shared"]
    assert sep["chain"]["elims"] > 0 and sep["face"]["prefix_shared"] == 0
    assert sep["face"]["prefix_ext"] > 0
    # The step callback eliminates each vertex once, and after the first
    # step the face callbacks read its tokenizer instead of extending one.
    assert sha["chain"]["elims"] == 0, sha
    assert sha["face"]["prefix_ext"] == 0, sha
    assert sha["face"]["prefix_miss"] - sha["face"]["prefix_shared"] == 1, sha
    assert sha["stream"]["ext"] > 0 and sha["shared"]["poison"] == 0, sha
