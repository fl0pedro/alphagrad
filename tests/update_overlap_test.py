# dsnn-dfw.190: --update-overlap runs the update of episode e-2 on a helper thread beside rollout e.
# Two short CPU trainer runs through the deep pipeline, with and without the flag, on the
# deterministic configuration of ppo_resume_equivalence_test.py plus one CPU measure actor.

from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TRAINER = _REPO / "src" / "alphagrad" / "approx" / "ppo.py"

_COMMON = [
    "--variant", "full", "--face-actions", "--unified-face-head",
    "--live-faces", "--set-pointer", "--dynamic-substeps",
    "--max-substeps", "1", "--incremental-encode", "--grad-window", "0",
    "--dataset", "none", "--cmp-type", "flops", "--mem-type", "peak_memory",
    "--terminal-rewards-only", "--rewards", "cmp", "mem",
    "--lambda-cmp", "1", "--lambda-mem", "1", "--lambda-frob", "1",
    "--advantage-norm", "popart", "--seed", "42", "--num-envs", "2",
    "--minibatches", "1", "--vocab-size", "512", "--wandb", "disabled",
    "--example", "Helmholtz", "--grad-oracle", "off",
    "--ray-measure", "1", "--measure-pipeline", "1",
    "--tokenize-where", "local",
]

_ENV = {
    "JAX_PLATFORMS": "cpu",
    "ALPHAGRAD_POLICY": "palimpsa",
    "ALPHAGRAD_INCREMENTAL_TOKENS": "1",
    "ALPHAGRAD_UNIFIED_FACE_ENUM": "1",
    "ALPHAGRAD_FACE_ENUM_CACHE": "1",
    "ALPHAGRAD_SKIP_COUNT_OPS": "1",
    "ALPHAGRAD_SKIP_COST_ANALYSIS": "1",
    "ALPHAGRAD_EXTEND_CHUNK": "256",
    "ALPHAGRAD_EXTEND_UNROLL": "32",
    "ALPHAGRAD_DELTA_OVERFLOW": "clip",
    "ALPHAGRAD_FORCE_REV": "0",
    "ALPHAGRAD_HEALTH_EPISODES": "99",
    "ALPHAGRAD_BATCHED_CALLBACK": "1",
}

_EVENT = re.compile(
    r"^\[(measure-pipeline|update-overlap)\] ep=(\d+) .*"
    r"overlapped=(rollout\d+|nothing)")


def _start(work: Path, tag: str, *extra: str):
    cwd = work / f"dir_{tag}"
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALPHAGRAD_")}
    env.update(_ENV)
    env["ALPHAGRAD_EQ_DUMP"] = str(work / tag)
    env["RAY_TMPDIR"] = str(work / f"ray_{tag}")
    return subprocess.run(
        [sys.executable, str(_TRAINER), *_COMMON, "--name", "overlap", *extra],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=5400)


def _run(work: Path, tag: str, *extra: str) -> list:
    r = _start(work, tag, *extra)
    if r.returncode != 0:
        raise AssertionError(
            f"the trainer failed for {tag} (rc={r.returncode})\n"
            f"--- stdout (tail) ---\n{r.stdout[-6000:]}\n"
            f"--- stderr (tail) ---\n{r.stderr[-6000:]}")
    events = []
    for line in r.stdout.splitlines():
        m = _EVENT.match(line)
        if m:
            events.append((m.group(1), int(m.group(2)), m.group(3)))
    return events


def _dump(work: Path, tag: str, ep: int) -> dict:
    path = work / f"{tag}.ep{ep}.pkl"
    assert path.exists(), f"no episode dump at {path}"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _differs(a: dict, b: dict) -> list:
    out = []
    assert sorted(a) == sorted(b), (sorted(a), sorted(b))
    for name in sorted(a):
        x, y = a[name], b[name]
        if isinstance(x, list):
            if len(x) != len(y) or any(
                    np.asarray(u).tobytes() != np.asarray(v).tobytes()
                    for u, v in zip(x, y)):
                out.append(name)
        elif isinstance(x, np.ndarray):
            if x.tobytes() != np.asarray(y).tobytes():
                out.append(name)
        elif x != y:
            out.append(name)
    return out


@pytest.mark.slow
def test_the_update_of_e_minus_2_runs_beside_rollout_e_and_the_drains_keep_episode_order(tmp_path):
    work = tmp_path
    run = ("--episodes", "5", "--checkpoint-every", "3")
    on = _run(work, "on", *run, "--update-overlap", "1")
    off = _run(work, "off", *run)
    assert off == [
        ("measure-pipeline", 0, "rollout1"),
        ("measure-pipeline", 1, "rollout2"),
        ("measure-pipeline", 2, "nothing"),
        ("measure-pipeline", 3, "rollout4"),
        ("measure-pipeline", 4, "nothing"),
    ], off
    assert on == [
        ("measure-pipeline", 0, "rollout1"),
        ("measure-pipeline", 1, "rollout2"),
        ("update-overlap", 0, "rollout2"),
        ("update-overlap", 1, "nothing"),
        ("measure-pipeline", 2, "nothing"),
        ("measure-pipeline", 3, "rollout4"),
        ("update-overlap", 3, "nothing"),
        ("measure-pipeline", 4, "nothing"),
    ], on
    # Rollouts 0 and 1 draw under the initial policy in both schedules, and updates 0 and 1 read the same inputs.
    for ep in (0, 1):
        assert _differs(_dump(work, "on", ep), _dump(work, "off", ep)) == [], ep
    # Rollout 2 draws under the initial policy with the overlap and under update 0 without it.
    assert "params" in _differs(_dump(work, "on", 2), _dump(work, "off", 2))
    for ep in (3, 4):
        _dump(work, "on", ep)
        _dump(work, "off", ep)


@pytest.mark.slow
def test_the_overlap_without_the_deep_pipeline_is_refused(tmp_path):
    r = _start(tmp_path, "refused", "--episodes", "1", "--checkpoint-every",
               "0", "--update-overlap", "1", "--tokenize-where", "pool")
    assert r.returncode != 0, r.stdout[-3000:]
    assert "--update-overlap 1 refused: it needs the deep pipeline" in r.stderr, \
        r.stderr[-3000:]
    assert not (tmp_path / "refused.ep0.pkl").exists()
