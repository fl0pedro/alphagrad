"""THE READOUT, END TO END (owner ruling 2026-09-26 Q2 c, dsnn-dfw.291).

A short CPU run trains two episodes on Helmholtz, writes its final checkpoint
and reads it out with `--readout 2`: two sampled plans and the argmax plan,
measured through the run's own measurement path, which here is the thesis
rows' path, the deep pipeline with one CPU measure actor. `tools/readout.py`
then reads the same checkpoint out twice: at the run's own seed, where every
record must be the trainer's, and from the run directory at another seed,
where the argmax plan must be the same plan and the sampled plans must not.
The policy a readout read is named by the sha256 of its array leaves, and it
must be the policy the run's last episode ended with, which the episode dump
(ALPHAGRAD_EQ_DUMP) holds.

The configuration is update_overlap_test.py's: the deterministic one of
ppo_resume_equivalence_test.py plus one measure actor, with the plan log on.
The reward is the analytic cost model, so every number a record carries is
reproducible across processes. One more run takes the synchronous path
(--measure-pipeline 0, no measure actor), where training never compiles the
rollout program on its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TRAINER = _REPO / "src" / "alphagrad" / "approx" / "ppo.py"
_TOOL = _REPO / "tools" / "readout.py"

_BASE = [
    "--variant", "full", "--face-actions", "--unified-face-head",
    "--live-faces", "--set-pointer", "--dynamic-substeps",
    "--max-substeps", "1", "--incremental-encode", "--grad-window", "0",
    "--dataset", "none", "--cmp-type", "flops", "--mem-type", "peak_memory",
    "--terminal-rewards-only", "--rewards", "cmp", "mem",
    "--lambda-cmp", "1", "--lambda-mem", "1", "--lambda-frob", "1",
    "--advantage-norm", "popart", "--seed", "42", "--num-envs", "2",
    "--minibatches", "1", "--vocab-size", "512", "--wandb", "disabled",
    "--example", "Helmholtz", "--grad-oracle", "off",
    "--name", "readout", "--episodes", "2", "--checkpoint-every", "2",
    "--plan-log", "auto",
]
#: The thesis rows' measurement path: the deep pipeline with a measure actor.
_POOLED = ["--ray-measure", "1", "--measure-pipeline", "1",
           "--tokenize-where", "local"]
#: The run's own arguments, as the trainer and the tool are both handed them.
_RUN = _BASE + _POOLED + ["--readout", "2"]

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

#: What a record says about its plan and its measurement, every field of which
#: this configuration reproduces in another process.
_SAME = ("record", "draw", "index", "rollout", "env", "status", "plan_hash",
         "plan", "rewards", "quality", "latency_log_ratio",
         "memory_log_ratio", "memory_source", "refused", "checkpoint",
         "checkpoint_episode", "seed", "run_name", "params_sha256")


def _start(work: Path, tag: str, argv, dump: bool = False):
    cwd = work / f"dir_{tag}"
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALPHAGRAD_")}
    env.update(_ENV)
    if dump:
        env["ALPHAGRAD_EQ_DUMP"] = str(work / tag)
    # Ray's socket paths must fit the 107 bytes of AF_UNIX, and pytest's tmp_path is too deep
    # for them, so the Ray session lives in a short directory under /tmp and goes with the run.
    ray_tmp = tempfile.mkdtemp(prefix=f"ray_{tag}_", dir="/tmp")
    env["RAY_TMPDIR"] = ray_tmp
    try:
        r = subprocess.run([sys.executable, *map(str, argv)], cwd=str(cwd),
                           env=env, capture_output=True, text=True,
                           timeout=5400)
    finally:
        shutil.rmtree(ray_tmp, ignore_errors=True)
    return cwd, r


def _ok(r, tag):
    if r.returncode != 0:
        raise AssertionError(
            f"{tag} failed (rc={r.returncode})\n"
            f"--- stdout (tail) ---\n{r.stdout[-6000:]}\n"
            f"--- stderr (tail) ---\n{r.stderr[-6000:]}")


def _records(path: Path) -> list:
    from alphagrad.approx.common import readout as ro
    return ro.load_records(str(path))


def _digest(leaves) -> str:
    # ppo._params_digest over the leaves the episode dump holds.
    h = hashlib.sha256()
    for leaf in leaves:
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _final_params_digest(work: Path, tag: str, last_ep: int) -> str:
    with open(work / f"{tag}.ep{last_ep}.pkl", "rb") as fh:
        return _digest(pickle.load(fh)["params"])


def _files_digest(ckpt: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ckpt.iterdir())}


def _check_readout(recs, n_plans, ckpt: Path, params: str, seed: int):
    assert [r["draw"] for r in recs] == ["sampled"] * n_plans + ["argmax"]
    assert [r["index"] for r in recs] == list(range(n_plans + 1))
    for r in recs:
        assert r["record"] == "readout"
        assert r["status"] == "measured", r
        assert r["plan_record"]["plan_hash"] == r["plan_hash"]
        assert r["checkpoint"] == os.path.realpath(ckpt)
        assert r["checkpoint_episode"] == 2
        assert r["seed"] == seed
        # The policy the readout read is the one the run's last episode ended with.
        assert r["params_sha256"] == params


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return tmp_path_factory.mktemp("readout")


@pytest.fixture(scope="module")
def trained(work):
    """The trainer's own readout at the end of a two-episode run."""
    cwd, r = _start(work, "train", [_TRAINER, *_RUN], dump=True)
    _ok(r, "the training run with --readout 2")
    ckpt = cwd / "ppo_ckpt_ep000000002"
    assert ckpt.is_dir(), sorted(p.name for p in cwd.iterdir())
    return {"cwd": cwd, "ckpt": ckpt, "stdout": r.stdout,
            "records": _records(cwd / "readout.jsonl"),
            "params": _final_params_digest(work, "train", 1)}


@pytest.mark.slow
def test_the_run_reads_its_final_policy_out_after_training(trained):
    recs = trained["records"]
    _check_readout(recs, 2, trained["ckpt"], trained["params"], 42)
    assert "[readout] 3 records ->" in trained["stdout"]
    # The two environments of the argmax rollout drew the one argmax plan, so the second
    # was served the first one's measurement.
    assert "readout argmax rollout 1: 2 of 2 plan records joined" in \
        trained["stdout"]


@pytest.mark.slow
def test_the_tool_reads_the_same_plans_out_of_the_same_checkpoint(work,
                                                                  trained):
    before = _files_digest(trained["ckpt"])
    out = work / "tool_same"
    _, r = _start(work, "tool_same", [_TOOL, "--out", out, trained["ckpt"],
                                      "--", *_RUN])
    _ok(r, "tools/readout.py at the run's seed")
    # The readout reads the checkpoint and changes nothing in it.
    assert _files_digest(trained["ckpt"]) == before
    recs = _records(out / "readout.jsonl")
    _check_readout(recs, 2, trained["ckpt"], trained["params"], 42)
    for a, b in zip(trained["records"], recs):
        assert {k: a[k] for k in _SAME} == {k: b[k] for k in _SAME}


@pytest.mark.slow
def test_the_argmax_plan_does_not_move_with_the_seed(work, trained):
    out = work / "tool_seed"
    # The run directory this time, which is read out at its final checkpoint.
    _, r = _start(work, "tool_seed", [_TOOL, "--out", out, "--readout-seed",
                                      "7", trained["cwd"], "--", *_RUN])
    _ok(r, "tools/readout.py at another seed")
    recs = _records(out / "readout.jsonl")
    _check_readout(recs, 2, trained["ckpt"], trained["params"], 7)
    theirs, ours = trained["records"][-1], recs[-1]
    assert (ours["plan_hash"], ours["plan"]) == \
        (theirs["plan_hash"], theirs["plan"])
    assert [r["plan_hash"] for r in recs[:-1]] != \
        [r["plan_hash"] for r in trained["records"][:-1]]


@pytest.mark.slow
def test_a_readout_of_another_configuration_is_refused(work, trained):
    out = work / "tool_other"
    other = [*_RUN, "--lambda-cmp", "2"]
    _, r = _start(work, "tool_other", [_TOOL, "--out", out, trained["ckpt"],
                                       "--", *other])
    assert r.returncode != 0, r.stdout[-3000:]
    assert "not a state of this run" in r.stderr, r.stderr[-3000:]
    assert "lambda_cmp: 1.0 in the checkpoint, 2.0 on the command line" in \
        r.stderr, r.stderr[-3000:]
    assert not (out / "readout.jsonl").exists()


@pytest.mark.slow
def test_a_run_without_its_final_checkpoint_has_no_readout(work):
    _, r = _start(work, "no_ckpt", [_TRAINER, *_BASE, *_POOLED, "--readout",
                                    "2", "--checkpoint-every", "0"])
    assert r.returncode != 0, r.stdout[-3000:]
    assert "--checkpoint-every 0 writes none" in r.stderr, r.stderr[-3000:]


@pytest.mark.slow
def test_the_synchronous_path_reads_out_too(work):
    cwd, r = _start(work, "sync", [_TRAINER, *_BASE, "--readout", "2"],
                    dump=True)
    _ok(r, "the synchronous training run with --readout 2")
    ckpt = cwd / "ppo_ckpt_ep000000002"
    _check_readout(_records(cwd / "readout.jsonl"), 2, ckpt,
                   _final_params_digest(work, "sync", 1), 42)
