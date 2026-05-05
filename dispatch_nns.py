"""Two-phase dispatch for the VmappedNeuralNetwork PPO experiments.

Phase 1 (calibration): observe the natural distribution of cmp/acc/mem rewards
with `--lambda-cmp 1 --lambda-mem 1`. Two strategies (`--calibration-mode`):

    joint       — one PPO run per experiment, using the same --rewards as the
                  full training. Mirrors the production setup; cheap.
    per-reward  — one PPO run per reward in --rewards (e.g. cmp, mem, acc) per
                  experiment, with --rewards set to that single reward. Each
                  run sees its reward in isolation, without cross-objective
                  interference. More expensive: N waves of len(EXPERIMENTS).

Phase 2 (full training): `compute_lambdas` derives lambda_cmp / lambda_mem from
the calibration JSONs, and the experiments re-launch in parallel with those
lambdas and the full episode budget.

The `EXPERIMENTS` list defines the agent variants compared in each run:
    ptr_autoreg : default (pointer vertex + autoregressive RuleDecoder)
    ptr_single  : --not-autoreg (pointer vertex + single per-vertex sp head)
    mlp_single  : --no-ptr --not-autoreg (MLP vertex + single sp head)

Each experiment is pinned to one GPU. Edit `EXPERIMENTS` and `compute_lambdas`
to match your setup.

Usage:
    uv run alphagrad/dispatch_nns.py
    uv run alphagrad/dispatch_nns.py --calibration-mode per-reward
    uv run alphagrad/dispatch_nns.py --dataset none           # synthetic 4-D data
    uv run alphagrad/dispatch_nns.py --dataset mnist          # MNIST
    uv run alphagrad/dispatch_nns.py --skip-calibration logs_dispatch  # reuse stats
    uv run alphagrad/dispatch_nns.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PPO_SCRIPT = REPO_ROOT / "src" / "alphagrad" / "approx" / "ppo.py"
DEFAULT_LOG_DIR = REPO_ROOT.parent / "logs_dispatch"  # ~/dsnn/logs_dispatch


@dataclass
class Experiment:
    """One PPO configuration. `gpus` is CUDA_VISIBLE_DEVICES for this job."""

    tag: str
    gpus: str
    extra_args: list[str] = field(default_factory=list)


# ---- Edit this list to define the agent variants run per dispatch ----------
# Each entry is one PPO run pinned to its own GPU. With 4 GPUs available and
# 3 variants, we leave GPU 3 idle (room for a 4th variant later).
EXPERIMENTS: list[Experiment] = [
    Experiment(tag="ptr_autoreg", gpus="0", extra_args=[]),
    Experiment(tag="ptr_single",  gpus="1", extra_args=["--not-autoreg"]),
    Experiment(tag="mlp_single",  gpus="2", extra_args=["--no-ptr", "--not-autoreg"]),
]


# ---- Edit this function to change how lambdas are computed -----------------
def compute_lambdas(stats_by_label: dict[str, dict]) -> tuple[float, float]:
    """Return (lambda_cmp, lambda_mem) given the calibration stats for ONE experiment.

    `stats_by_label` keys depend on --calibration-mode:
        joint:      {"joint": <stats JSON dict>}  — has cmp/acc/mem stats together.
        per-reward: {"cmp": <stats>, "mem": <stats>, "acc": <stats>}
                    — read each reward's stats from its own isolated run.

    Default formula: balance the std of each reward's contribution against
    `acc` (which keeps weight 1.0 in ppo.py): lambda_x = std(acc) / std(x).
    Replace with whatever you actually use; the full stats dicts are visible.
    """
    eps = 1e-9
    if "joint" in stats_by_label:
        r = stats_by_label["joint"]["rewards"]
        s_cmp = r["cmp"]["std"]
        s_acc = r["acc"]["std"]
        s_mem = r["mem"]["std"]
    else:
        s_cmp = stats_by_label["cmp"]["rewards"]["cmp"]["std"]
        s_acc = stats_by_label["acc"]["rewards"]["acc"]["std"]
        s_mem = stats_by_label["mem"]["rewards"]["mem"]["std"]
    return s_acc / max(s_cmp, eps), s_acc / max(s_mem, eps)


# ----------------------------------------------------------------------------


@dataclass
class Job:
    """A single ppo.py invocation."""
    exp: Experiment
    label: str           # e.g. "joint", "cmp", "mem", "acc", or "full"
    cmd: list[str]
    out_path: Path
    stats_out: Path | None


def build_job(exp: Experiment, args: argparse.Namespace, *, episodes: int,
              wave_dir: Path, label: str,
              extra_override: list[str] | None,
              lambdas: tuple[float, float] | None,
              collect_stats: bool) -> Job:
    cmd = [
        "uv", "run", str(PPO_SCRIPT),
        "--name", f"{args.name_prefix}_{exp.tag}_{label}",
        "--top-n", str(args.top_n),
        "--episodes", str(episodes),
        "--example", args.example,
        "--dataset", args.dataset,
        "--dataset-size", str(args.dataset_size),
        "--num-eval-samples", str(args.num_eval_samples),
    ]
    if args.exec_on_gpu:
        cmd.append("--exec-on-gpu")
    cmd.extend(exp.extra_args)
    # The phase-2 reward set comes from --rewards. Overrides (per-reward
    # calibration) are appended last so argparse takes the override.
    cmd.extend(["--rewards", *args.rewards])
    if extra_override is not None:
        cmd.extend(extra_override)
    if lambdas is not None:
        l_cmp, l_mem = lambdas
        cmd.extend(["--lambda-cmp", repr(l_cmp), "--lambda-mem", repr(l_mem)])
    stats_out = wave_dir / f"{exp.tag}_stats.json" if collect_stats else None
    if stats_out is not None:
        cmd.extend(["--stats-out", str(stats_out)])
    if args.ppo_extra:
        cmd.extend(args.ppo_extra)
    return Job(exp=exp, label=label, cmd=cmd,
               out_path=wave_dir / f"{exp.tag}.out", stats_out=stats_out)


def launch_wave(jobs: list[Job], label: str, dry_run: bool) -> None:
    """Launch every job in parallel, wait, raise on any nonzero exit."""
    print(f"\n--- wave: {label} ({len(jobs)} job{'s' if len(jobs) != 1 else ''}) ---",
          flush=True)
    for j in jobs:
        j.out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [{j.exp.tag}] CUDA_VISIBLE_DEVICES={j.exp.gpus} -> {j.out_path}")
        print(f"    {shlex.join(j.cmd)}")
    sys.stdout.flush()

    if dry_run:
        return

    procs = []
    for j in jobs:
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = j.exp.gpus
        log_file = open(j.out_path, "w")
        proc = subprocess.Popen(
            j.cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT.parent),  # match `cd ~/dsnn` from the sbatch
        )
        procs.append((proc, j, log_file))

    failed: list[str] = []
    for proc, j, log_file in procs:
        rc = proc.wait()
        log_file.close()
        if rc != 0:
            failed.append(f"{j.exp.tag} (rc={rc})")
    if failed:
        raise RuntimeError(
            f"wave '{label}' had failures: {', '.join(failed)}. See logs."
        )


def calibration_waves(args: argparse.Namespace, log_root: Path) -> tuple[
        list[tuple[str, list[Job]]], dict[str, dict[str, Path]]]:
    """Build calibration waves and the (exp.tag -> {label -> stats_path}) index.

    A wave is a list of jobs that can run in parallel without GPU conflicts.
    Different waves run sequentially. Within a wave, every experiment runs once.
    """
    waves: list[tuple[str, list[Job]]] = []
    # exp.tag -> {label -> stats_path}
    stats_index: dict[str, dict[str, Path]] = {e.tag: {} for e in EXPERIMENTS}

    if args.calibration_mode == "joint":
        wave_dir = log_root / "calibration"
        jobs = [
            build_job(e, args, episodes=args.calibration_episodes,
                      wave_dir=wave_dir, label="joint",
                      extra_override=None, lambdas=None,
                      collect_stats=True)
            for e in EXPERIMENTS
        ]
        for e, j in zip(EXPERIMENTS, jobs):
            stats_index[e.tag]["joint"] = j.stats_out  # type: ignore[assignment]
        waves.append(("calibration:joint", jobs))
    else:  # per-reward
        for r in args.rewards:
            wave_dir = log_root / f"calibration_{r}"
            jobs = [
                build_job(e, args, episodes=args.calibration_episodes,
                          wave_dir=wave_dir, label=r,
                          extra_override=["--rewards", r], lambdas=None,
                          collect_stats=True)
                for e in EXPERIMENTS
            ]
            for e, j in zip(EXPERIMENTS, jobs):
                stats_index[e.tag][r] = j.stats_out  # type: ignore[assignment]
            waves.append((f"calibration:{r}", jobs))

    return waves, stats_index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Phase budgets
    p.add_argument("--calibration-episodes", type=int, default=30,
                   help="Episodes for each calibration job. Default: 30.")
    p.add_argument("--full-episodes", type=int, default=500,
                   help="Episodes for the lambda-weighted phase. Default: 500.")

    # Calibration strategy
    p.add_argument("--calibration-mode", choices=["joint", "per-reward"],
                   default="per-reward",
                   help="per-reward (default): one calibration per reward in "
                        "--rewards, each with --rewards set to that single reward "
                        "(isolated calibrations). joint: one calibration per "
                        "experiment using the full --rewards.")
    p.add_argument("--rewards", nargs="+", default=["cmp", "mem", "acc"],
                   choices=["cmp", "mem", "acc"],
                   help="Reward set for the full-training phase. In per-reward "
                        "mode, also the set iterated over for calibration.")

    # ppo.py passthrough
    p.add_argument("--example", type=str, default="VmappedNeuralNetwork")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "none"])
    p.add_argument("--dataset-size", type=int, default=-1)
    p.add_argument("--num-eval-samples", type=int, default=10)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--name-prefix", type=str, default="dispatch",
                   help="Prefix for the wandb `--name` of every spawned job.")
    p.add_argument("--exec-on-gpu", action="store_true",
                   help="Pass --exec-on-gpu to ppo.py (uses 2 GPUs per job: "
                        "model on the first, env eval on the second). Off by "
                        "default — each job uses one GPU.")
    p.add_argument("--ppo-extra", nargs=argparse.REMAINDER, default=[],
                   metavar="...",
                   help="Anything after this flag is forwarded verbatim to every "
                        "ppo.py invocation. Must come last on the command line.")

    # Logging / control
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                   help=f"Where to write per-wave log/stats subdirs. Default: {DEFAULT_LOG_DIR}.")
    p.add_argument("--skip-calibration", type=Path, default=None, metavar="LOG_DIR",
                   help="Skip phase 1 and read calibration stats from this LOG_DIR "
                        "(must contain the calibration_<label>/ subdirs from a prior run).")
    p.add_argument("--skip-full", action="store_true",
                   help="Run only calibration and print computed lambdas.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands but launch nothing.")
    args = p.parse_args()

    log_root = args.log_dir.resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: calibration -------------------------------------------------
    waves, stats_index = calibration_waves(args, log_root)

    if args.skip_calibration is None:
        print(f"\n=== Phase 1: calibration "
              f"(mode={args.calibration_mode}, "
              f"{args.calibration_episodes} episodes/wave, lambdas=1) ===",
              flush=True)
        t0 = time.time()
        for label, jobs in waves:
            launch_wave(jobs, label, args.dry_run)
        print(f"calibration phase done in {time.time() - t0:.1f}s", flush=True)
    else:
        # Re-point stats paths at the user-supplied directory.
        prior = args.skip_calibration.resolve()
        print(f"\n=== Phase 1: skipped, reading stats from {prior} ===")
        for tag, by_label in stats_index.items():
            for label, path in list(by_label.items()):
                rel = path.relative_to(log_root)
                stats_index[tag][label] = prior / rel
        for tag, by_label in stats_index.items():
            for label, path in by_label.items():
                if not path.exists():
                    print(f"missing stats file: {path}", file=sys.stderr)
                    return 2

    if args.dry_run:
        print("\n[dry-run] skipping lambda computation and phase 2.")
        return 0

    # ---- Compute lambdas ------------------------------------------------------
    print("\n=== Computed lambdas ===")
    lambdas_per_exp: list[tuple[float, float]] = []
    for exp in EXPERIMENTS:
        stats_by_label = {
            label: json.load(open(path))
            for label, path in stats_index[exp.tag].items()
        }
        l_cmp, l_mem = compute_lambdas(stats_by_label)
        lambdas_per_exp.append((l_cmp, l_mem))
        # Brief diagnostic: which std went into each lambda.
        if "joint" in stats_by_label:
            r = stats_by_label["joint"]["rewards"]
            print(f"  {exp.tag}: lambda_cmp={l_cmp:.10g}  lambda_mem={l_mem:.10g}"
                  f"  (joint std cmp={r['cmp']['std']:.3g}, "
                  f"acc={r['acc']['std']:.3g}, mem={r['mem']['std']:.3g})")
        else:
            sc = stats_by_label["cmp"]["rewards"]["cmp"]["std"]
            sa = stats_by_label["acc"]["rewards"]["acc"]["std"]
            sm = stats_by_label["mem"]["rewards"]["mem"]["std"]
            print(f"  {exp.tag}: lambda_cmp={l_cmp:.10g}  lambda_mem={l_mem:.10g}"
                  f"  (per-reward std cmp={sc:.3g}, acc={sa:.3g}, mem={sm:.3g})")

    # Save the computed lambdas for traceability.
    with open(log_root / "computed_lambdas.json", "w") as f:
        json.dump({
            e.tag: {"lambda_cmp": l_cmp, "lambda_mem": l_mem}
            for e, (l_cmp, l_mem) in zip(EXPERIMENTS, lambdas_per_exp)
        }, f, indent=2)

    if args.skip_full:
        print("\n--skip-full: stopping after lambda computation.")
        return 0

    # ---- Phase 2: full training ----------------------------------------------
    full_dir = log_root / "full"
    full_jobs = [
        build_job(e, args, episodes=args.full_episodes,
                  wave_dir=full_dir, label="full",
                  extra_override=None, lambdas=lams,
                  collect_stats=True)
        for e, lams in zip(EXPERIMENTS, lambdas_per_exp)
    ]
    print(f"\n=== Phase 2: full training "
          f"({args.full_episodes} episodes, computed lambdas) ===", flush=True)
    t0 = time.time()
    launch_wave(full_jobs, "full", dry_run=False)
    print(f"full phase done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
