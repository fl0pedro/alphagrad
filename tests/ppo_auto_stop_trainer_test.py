"""THE TRAINER REALLY STOPS, AND ONLY WHEN IT IS ASKED TO.

Ticket ``dsnn-dfw.6``. `ppo_auto_stop_test.py` pins the decision on synthetic
windows; this file pins the wiring: that the rows the trainer records are the
rows that decision reads, that a stop writes the four artefacts at the
quiescent point, that a resume re-checks at the NEXT check point, and that a
run without the flag never stops.

WHY THE CHECK POINT IS 4 AND NOT 250. The ruled check points are after 250 and
after 500 episodes, with a window of 100. The smallest arm here runs an
episode in a few seconds, so a test at the ruled values would be an hour of
suite time per case. The check points and the window size are therefore two
SUPPRESSed test-only arguments with the ruled values as their defaults
(``--auto-stop-check-at``, ``--auto-stop-window``). They are part of the
argument namespace, so a resume checks them like every other argument, and no
launcher sets them. `ppo_auto_stop_test.py` pins that the defaults are 250,500
and 100.

WHY THIS ARM SETTLES. ``--approx-profile none`` removes both approximation
heads, and ``--fixed-order markowitz`` (the default) pins the elimination
order. Every episode's terminal plan is then the same plan, its measured cost
is the same cost, and its scalar return is the same number. So the archive
admits its first point and nothing after it, the return does not move, the
terminal plan does not change, and no terminal plan holds an approximation --
which is exactly arm B's predicted behaviour, in miniature and on purpose.
This is a test of the stop rule, not a claim about a real arm.

WHY THE QUALITY CHANNEL IS IN THE REWARD HERE (``--rewards cmp mem acc``,
which is also the trainer's own default, where `ppo_resume_equivalence_test`
uses the two cost channels alone). Under the campaign's paired-log cost form
a plan that IS rev-exact scores exactly 0 on both cost slots, so with the
cost channels alone this arm's raw scalar return is float noise around zero,
and a relative move between two such windows is 40 percent of nothing. With
quality in, a settled window returns a steady order-1 number. See the note
over `RETURN_TOLERANCE` in `common/auto_stop.py`; no thesis arm drops the
quality channel.

THE CONFIGURATION is the canonical deterministic CPU one from `tools/smoke.sh`
with the policy gate's environment pins, the same one
`ppo_resume_equivalence_test.py` uses, so no wall clock enters the reward.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_TRAINER = _REPO / "src" / "alphagrad" / "approx" / "ppo.py"

#: The canonical deterministic CPU configuration (tools/smoke.sh COMMON),
#: with both approximation heads removed so the arm settles at once.
_COMMON = [
    "--variant", "full", "--face-actions", "--unified-face-head",
    "--live-faces", "--set-pointer", "--dynamic-substeps",
    "--max-substeps", "1", "--incremental-encode", "--grad-window", "0",
    "--dataset", "none", "--cmp-type", "flops", "--mem-type", "peak_memory",
    "--terminal-rewards-only", "--rewards", "cmp", "mem", "acc",
    "--lambda-cmp", "1", "--lambda-mem", "1", "--lambda-frob", "1",
    "--lambda-acc", "2",
    "--advantage-norm", "popart", "--seed", "42", "--num-envs", "2",
    "--minibatches", "1", "--vocab-size", "512", "--wandb", "disabled",
    "--example", "Helmholtz", "--approx-profile", "none",
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
}


def _run(work_dir: Path, tag: str, *extra: str, expect_rc: int = 0,
         name: str = "asrun"):
    """One trainer run in a directory of its own. Returns (dir, result)."""
    cwd = work_dir / f"dir_{tag}"
    cwd.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALPHAGRAD_")}
    env.update(_ENV)
    env["ALPHAGRAD_EQ_DUMP"] = str(work_dir / tag)
    r = subprocess.run(
        [sys.executable, str(_TRAINER), *_COMMON, "--name", name, *extra],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=5400)
    if expect_rc is not None and r.returncode != expect_rc:
        raise AssertionError(
            f"the trainer for {tag} returned {r.returncode}, expected "
            f"{expect_rc}\n--- stdout (tail) ---\n{r.stdout[-8000:]}\n"
            f"--- stderr (tail) ---\n{r.stderr[-8000:]}")
    return cwd, r


def _reason(cwd: Path) -> dict:
    path = cwd / "auto_stop.json"
    assert path.exists(), (
        f"no auto_stop.json in {cwd}; found "
        f"{sorted(p.name for p in cwd.iterdir())}")
    with open(path) as fh:
        return json.load(fh)


def _checkpoints(cwd: Path):
    from alphagrad.approx.common import checkpoint as ckpt
    return [Path(p).name for p in ckpt.list_checkpoints(str(cwd))]


def _episode_dumps(work: Path, tag: str):
    return sorted(int(p.name.split(".ep")[1].split(".pkl")[0])
                  for p in work.glob(f"{tag}.ep*.pkl"))


@pytest.mark.slow
def test_a_settled_run_stops_at_its_check_point(tmp_path):
    """Eight episodes asked for, four run, and the reason on disk says why."""
    cwd, r = _run(tmp_path, "stop",
                  "--episodes", "8", "--checkpoint-every", "2",
                  "--auto-stop", "--auto-stop-check-at", "4",
                  "--auto-stop-window", "2")
    assert "[auto-stop] STOPPING after 4 episodes" in r.stdout

    # It ran episodes 0..3 and not one more.
    assert _episode_dumps(tmp_path, "stop") == [0, 1, 2, 3]

    doc = _reason(cwd)
    assert doc["stop"] is True
    assert doc["check_point"] == 4
    assert doc["window"] == 2
    assert doc["recent_window"] == [2, 3]
    assert doc["previous_window"] == [0, 1]
    c = doc["conditions"]
    assert c["archive_admitted_nothing"] is True
    assert c["return_moved_less_than_tolerance"] is True
    assert c["collapse_or_plan_frozen"] is True
    # Both halves of the third condition hold on this arm: the order is
    # pinned and both approximation heads are removed.
    assert c["plan_did_not_change"] is True
    assert c["collapse_identity"] is True
    n = doc["numbers"]
    assert n["admitted_recent"] == 0
    assert n["relative_return_move"] < 0.02
    assert n["max_approximations_in_any_terminal_plan"] == 0
    assert n["total_approximations_in_window"] == 0
    assert isinstance(n["terminal_plan_digest"], str)

    # THE FINAL CHECKPOINT IS AT THE EPISODE THE RUN REACHED, not at
    # --episodes. A checkpoint claiming 8 episodes were done would resume
    # after the end of the run.
    names = _checkpoints(cwd)
    assert "ppo_ckpt_ep000000004" in names
    assert "ppo_ckpt_ep000000008" not in names

    # AND THE RUN TOOK THE NORMAL END-OF-RUN PATH. A stop leaves the loop; it
    # does not exit the process. The top-N tables are printed after the loop,
    # so their presence is the proof that the front dump, the top-N and the
    # elimination-order table of a stopped arm all ran. Without them the
    # readout of an auto-stopped arm would have no input.
    assert "Top 10 trajectories" in r.stdout


@pytest.mark.slow
def test_a_resumed_run_rechecks_at_the_next_check_point(tmp_path):
    """The check point a resume STARTS at is not re-decided; the next one is.

    This is also the only test of the auto-stop history in the checkpoint:
    the window at check point 6 reaches back to episode 2, and the resumed
    leg never ran episodes 2 and 3. Without the restored history the decision
    at 6 would have had nothing to compare.
    """
    first, r1 = _run(tmp_path, "leg1",
                     "--episodes", "8", "--checkpoint-every", "2",
                     "--auto-stop", "--auto-stop-check-at", "4,6",
                     "--auto-stop-window", "2")
    assert _reason(first)["check_point"] == 4
    assert _episode_dumps(tmp_path, "leg1") == [0, 1, 2, 3]
    mid = first / "ppo_ckpt_ep000000004"
    assert mid.is_dir(), _checkpoints(first)

    second, r2 = _run(tmp_path, "leg2",
                      "--episodes", "8", "--checkpoint-every", "2",
                      "--auto-stop", "--auto-stop-check-at", "4,6",
                      "--auto-stop-window", "2", "--resume", str(mid))
    # It did NOT stop again at 4: it ran episodes 4 and 5 first.
    assert _episode_dumps(tmp_path, "leg2") == [4, 5]
    doc = _reason(second)
    assert doc["check_point"] == 6
    assert doc["stop"] is True
    assert doc["recent_window"] == [4, 5]
    assert doc["previous_window"] == [2, 3]
    assert doc["incomplete"] is False
    assert "ppo_ckpt_ep000000006" in _checkpoints(second)


@pytest.mark.slow
def test_a_run_without_the_flag_never_stops(tmp_path):
    """THE GATE ARM. The check points and the window are set to values this
    run reaches, and the arm is the one that settles at episode 0, so the
    ONLY thing keeping the run alive is that --auto-stop is off. It runs
    every episode it was given and writes no reason file.

    The policy regression gate itself does not run the trainer at all -- it
    traces the policy path in its own interpreter, with no episodes and no
    loop -- so this is the strongest trainer-level statement available about
    a run that does not ask to stop. `ppo_auto_stop_test.py` pins the other
    half: no gate file names the flag.
    """
    cwd, r = _run(tmp_path, "noflag",
                  "--episodes", "6", "--checkpoint-every", "2",
                  "--auto-stop-check-at", "4", "--auto-stop-window", "2")
    assert "[auto-stop]" not in r.stdout
    assert not (cwd / "auto_stop.json").exists()
    assert _episode_dumps(tmp_path, "noflag") == [0, 1, 2, 3, 4, 5]
    assert "ppo_ckpt_ep000000006" in _checkpoints(cwd)


def test_auto_stop_with_checkpointing_off_is_refused_by_the_trainer(tmp_path):
    """The refusal is made right after `parse_args`, so this costs one process
    start and no episode."""
    _, r = _run(tmp_path, "refuse", "--episodes", "8",
                "--checkpoint-every", "0", "--auto-stop",
                "--auto-stop-check-at", "4", "--auto-stop-window", "2",
                expect_rc=None)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "nothing to resume from" in out
    assert "AutoStopError" in out or "auto-stop" in out
