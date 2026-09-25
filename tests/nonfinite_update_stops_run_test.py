# dsnn-dfw.90 (3): a NaN parameter after the update stops the run at that episode.
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

_PKG = Path(importlib.util.find_spec("alphagrad").submodule_search_locations[0])
_PPO = _PKG / "approx" / "ppo.py"

# The environment and the flags of tools/smoke.sh for one episode, plus --grad-oracle off: without a plan log the oracle refuses to start.
_ENV = {
    "JAX_PLATFORMS": "cpu",
    "ALPHAGRAD_EXTEND_CHUNK": "256",
    "ALPHAGRAD_EXTEND_UNROLL": "32",
    "ALPHAGRAD_DELTA_OVERFLOW": "clip",
    "ALPHAGRAD_POLICY": "palimpsa",
    "ALPHAGRAD_INCREMENTAL_TOKENS": "1",
    "ALPHAGRAD_UNIFIED_FACE_ENUM": "1",
    "ALPHAGRAD_SKIP_COUNT_OPS": "1",
    "ALPHAGRAD_SKIP_COST_ANALYSIS": "1",
    "ALPHAGRAD_HEALTH_EPISODES": "99",
}
_ARGV = (
    "--variant full --face-actions --unified-face-head --live-faces "
    "--set-pointer --dynamic-substeps --max-substeps 1 --incremental-encode "
    "--grad-window 0 --dataset none "
    "--cmp-type flops --mem-type peak_memory --terminal-rewards-only "
    "--rewards cmp mem --lambda-cmp 1 --lambda-mem 1 --lambda-frob 1 "
    "--advantage-norm popart --episodes 1 --seed 42 --num-envs 2 "
    "--minibatches 1 --vocab-size 512 --wandb disabled "
    "--example Helmholtz --name nan_update --grad-oracle off").split()


def test_a_nan_parameter_after_the_update_stops_the_run(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ALPHAGRAD_")}
    env.update(_ENV)
    r = subprocess.run([sys.executable, str(_PPO), *_ARGV, "--lr", "nan"],
                       env=env, cwd=tmp_path, capture_output=True, text=True,
                       timeout=3000)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out[-4000:]
    assert re.search(r"NonFiniteUpdate: episode \d+: total_loss=", out), \
        out[-4000:]
