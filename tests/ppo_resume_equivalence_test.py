"""A RESUME CONTINUES THE RUN IT LOADED, BIT FOR BIT.

This is the end-to-end half of the checkpoint work. `ppo_checkpoint_test.py`
pins the file format; this file pins the only property that matters, which is
that the episodes after a resume are the SAME episodes the uninterrupted run
would have produced.

HOW IT IS MEASURED. `ALPHAGRAD_EQ_DUMP=<prefix>` makes the trainer pickle,
after every episode, the policy parameters, the optimiser state, the episode's
metrics, its sampled actions, its reward vectors, the global step and the
scalar return. That dump already exists for exactly this class of question
(any change to the env or the rollout has to be proven trajectory-identical
against its parent commit), and it is a superset of the quantities the policy
regression gate records: it holds the parameters themselves. Two runs are
compared leaf by leaf on the RAW BYTES of every array, never with a tolerance.

WHAT THE TWO RUNS ARE.

* the straight run: `--episodes 4 --checkpoint-every 2`, which trains
  episodes 0..3 and writes a checkpoint after episode 1 and after episode 3.
* the resumed run: `--resume <that first checkpoint> --episodes 4
  --checkpoint-every 2`, which trains episodes 2..3.

Episodes 2 and 3 must agree. The first two episodes need no comparison here:
they are the same process's, and that the process is reproducible at all is
what `test_checkpointing_off_and_on_give_the_same_short_run` proves
separately. A comparison that did not separate those two questions could not
say which of them had failed. The 50-against-25-plus-25 form the owner asked
for is the same property over two separate processes and is run on the
cluster; see the report of 2026-09-15.

THE CONFIGURATION is the canonical deterministic CPU one from `tools/smoke.sh`
with the policy gate's environment pins: `--cmp-type flops --mem-type
peak_memory`, so the reward is the analytic cost model and no wall clock
enters it. A run with `--measure-latency` is not reproducible by anybody and
would test nothing here.

The suite cost is three short CPU trainer runs. Each is two to four episodes
on Helmholtz, the same target `tools/smoke.sh` uses.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TRAINER = _REPO / "src" / "alphagrad" / "approx" / "ppo.py"

#: The canonical deterministic CPU configuration (tools/smoke.sh COMMON).
_COMMON = [
    "--variant", "full", "--face-actions", "--unified-face-head",
    "--live-faces", "--set-pointer", "--dynamic-substeps",
    "--max-substeps", "1", "--incremental-encode", "--grad-window", "0",
    "--dataset", "none", "--cmp-type", "flops", "--mem-type", "peak_memory",
    "--terminal-rewards-only", "--rewards", "cmp", "mem",
    "--lambda-cmp", "1", "--lambda-mem", "1", "--lambda-frob", "1",
    "--advantage-norm", "popart", "--seed", "42", "--num-envs", "2",
    "--minibatches", "1", "--vocab-size", "512", "--wandb", "disabled",
    "--example", "Helmholtz",
    # THE GRADIENT ORACLE IS OFF because the plan log is. The oracle reads the
    # distinct elimination orders of an oracle-due episode off the plan-log
    # records (agent/oracle-cap, dsnn-dfw.22) and ppo.py refuses the pair
    # rather than check nothing. This file is about resume equivalence.
    "--grad-oracle", "off",
]

#: The environment the configuration is pinned by. Inherited ALPHAGRAD_*
#: variables are STRIPPED, for the reason `policy_regression_gate_test.py`
#: gives: a run whose configuration is partly the caller's shell is not a
#: run two invocations can be compared on.
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
}


def _run(work_dir: Path, tag: str, *extra: str, name: str = "eqrun") -> Path:
    """One trainer run. Returns ITS OWN directory, where its checkpoints are.

    Every run gets a directory of its own because with `--wandb disabled` the
    run directory is the working directory, and two runs sharing one would
    prune each other's checkpoints. Under wandb each run has its own already.

    `--name` is the SAME for every run of one comparison, and `tag` names only
    the episode dumps and the directory. A resume checks the whole argument
    namespace, `--name` included, so two legs of one run that disagreed about
    it would be refused -- correctly, and this file's first version was.
    """
    cwd = work_dir / f"dir_{tag}"
    cwd.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALPHAGRAD_")}
    env.update(_ENV)
    env["ALPHAGRAD_EQ_DUMP"] = str(work_dir / tag)
    r = subprocess.run(
        [sys.executable, str(_TRAINER), *_COMMON, "--name", name, *extra],
        cwd=str(cwd), env=env, capture_output=True, text=True,
        timeout=5400,
    )
    if r.returncode != 0:
        raise AssertionError(
            f"the trainer failed for {tag} (rc={r.returncode})\n"
            f"--- stdout (tail) ---\n{r.stdout[-6000:]}\n"
            f"--- stderr (tail) ---\n{r.stderr[-6000:]}")
    return cwd


def _dump(work_dir: Path, tag: str, ep: int) -> dict:
    path = work_dir / f"{tag}.ep{ep}.pkl"
    assert path.exists(), f"no episode dump at {path}"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _diff(a: dict, b: dict, ep: int) -> list:
    """Every field of two episode dumps that is not byte-identical."""
    out = []
    assert sorted(a) == sorted(b), (sorted(a), sorted(b))
    for name in sorted(a):
        x, y = a[name], b[name]
        if isinstance(x, list):
            if len(x) != len(y):
                out.append(f"ep{ep} {name}: {len(x)} leaves against {len(y)}")
                continue
            for i, (u, v) in enumerate(zip(x, y)):
                u = np.asarray(u)
                v = np.asarray(v)
                if u.dtype != v.dtype or u.shape != v.shape:
                    out.append(f"ep{ep} {name}[{i}]: {u.dtype}{u.shape} "
                               f"against {v.dtype}{v.shape}")
                elif u.tobytes() != v.tobytes():
                    d = np.abs(u.astype(np.float64) - v.astype(np.float64))
                    out.append(f"ep{ep} {name}[{i}]: {int(np.sum(u != v))} of "
                               f"{u.size} words differ, largest absolute "
                               f"difference {float(np.max(d)):.3e}")
        elif isinstance(x, np.ndarray):
            if x.tobytes() != np.asarray(y).tobytes():
                out.append(f"ep{ep} {name}: arrays differ")
        elif x != y:
            out.append(f"ep{ep} {name}: {x!r} against {y!r}")
    return out


def _checkpoint(work_dir: Path, episode: int) -> Path:
    from alphagrad.approx.common import checkpoint as ckpt
    path = work_dir / ckpt.checkpoint_dir_name(episode)
    assert path.is_dir(), (
        f"no checkpoint for {episode} episodes in {work_dir}; found "
        f"{sorted(p.name for p in work_dir.iterdir())}")
    return path


@pytest.mark.slow
def test_checkpointing_off_and_on_give_the_same_short_run(tmp_path):
    """TWO things at once, and they cost one pair of runs.

    First, the control: two runs of the same command must agree, or nothing
    else in this file means anything. A mismatch in the resume test could
    otherwise not be attributed -- it might be the resume, and it might be
    that the trainer is not reproducible under this configuration at all.

    Second, the owner's condition on the default: `--checkpoint-every 50` on
    a run SHORTER than 50 episodes must be identical to `--checkpoint-every
    0`. The only checkpoint such a run takes is the one at the end, which is
    after the last episode, so no drain ever happens inside the loop and the
    schedule is untouched. This is the pair that proves it.
    """
    work = tmp_path
    _run(work, "ctl_off", "--episodes", "2", "--checkpoint-every", "0")
    _run(work, "ctl_default", "--episodes", "2", "--checkpoint-every", "50")
    problems = []
    for ep in (0, 1):
        problems += _diff(_dump(work, "ctl_off", ep),
                          _dump(work, "ctl_default", ep), ep)
    assert not problems, (
        "checkpointing off and at its default disagreed on a 2-episode run:\n"
        + "\n".join(problems))
    # And the default did write its end-of-run checkpoint.
    _checkpoint(work / "dir_ctl_default", 2)
    assert not (work / "dir_ctl_off").joinpath(
        "ppo_ckpt_ep000000002").exists()


@pytest.mark.slow
def test_a_resumed_run_continues_bit_for_bit(tmp_path):
    work = tmp_path
    straight_dir = _run(work, "straight",
                        "--episodes", "4", "--checkpoint-every", "2")
    mid = _checkpoint(straight_dir, 2)
    _checkpoint(straight_dir, 4)
    _run(work, "resumed", "--episodes", "4", "--checkpoint-every", "2",
         "--resume", str(mid))

    problems = []
    for ep in (2, 3):
        problems += _diff(_dump(work, "straight", ep),
                          _dump(work, "resumed", ep), ep)
    assert not problems, (
        "the resumed run diverged from the straight one:\n"
        + "\n".join(problems))

    # The resumed run must not have re-run the episodes it loaded.
    assert not (work / "resumed.ep0.pkl").exists()
    assert not (work / "resumed.ep1.pkl").exists()


def test_a_resume_refuses_a_command_line_that_does_not_match(tmp_path):
    """No trainer run: the refusal is made before anything is built, which is
    the whole point of taking the argument namespace at parse time."""
    from alphagrad.approx.common import checkpoint as ckpt

    work = tmp_path
    saved = {"seed": 42, "episodes": 4, "resume": "", "checkpoint_every": 2,
             "lr": 0.0003}
    path = ckpt.save_ppo_checkpoint(
        str(work), episode=2, tree={"x": np.zeros((1,), np.float32)},
        meta={"args": saved, "wandb_run_id": "", "pareto_archive": {},
              "episode_bin": {}, "window_bin": {}, "host_state": {}})
    meta = ckpt.read_ppo_meta(path)
    assert meta["args"] == saved

    import argparse
    ok = argparse.Namespace(seed=42, episodes=9, resume=path,
                            checkpoint_every=2, lr=0.0003)
    ckpt.check_resume_args(meta["args"], ok)
    bad = argparse.Namespace(seed=7, episodes=9, resume=path,
                             checkpoint_every=2, lr=0.0003)
    with pytest.raises(ckpt.CheckpointError, match="seed"):
        ckpt.check_resume_args(meta["args"], bad)
