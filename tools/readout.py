"""Read out the policy of a finished run from its checkpoint (dsnn-dfw.291).

THE READOUT (owner ruling 2026-09-26, Q2 c): the final checkpoint is loaded,
plans are sampled from the policy with no update, and they are measured with
the same instrument the run's training episodes were measured with: N sampled
plans and the argmax plan (the argmax at every step), one record per plan in
readout.jsonl, and the readout/* fields in the wandb summary of the run. A run
launched with --readout N does this itself at its end. This tool does it for a
run that did not, through the trainer's own code path: `ppo.main` builds the
run from the run's own arguments, reads the checkpoint out exactly as the
trainer does, and trains no episode.

The run's own arguments follow `--`, as the run was launched, with --readout N
added when the run did not carry it. They must match the checkpoint's saved
namespace except --readout and the arguments a resume may change
(`checkpoint.PPO_READOUT_EXEMPT_ARGS`); anything else raises, and so does an
argument this build defines that the checkpoint does not carry.

    $PY tools/readout.py $RUN_DIR -- <the run's arguments> --readout 64
    $PY tools/readout.py $RUN_DIR/ppo_ckpt_ep000002000 -- <the run's arguments>

A run directory is read at its final checkpoint and raises when the run did
not finish; a checkpoint directory is read as named. The records go to the run
directory the checkpoint sits in unless --out names another, and a readout
never overwrites one. Run it as an sbatch job on the node profile the run used:
the arguments after `--` pin the trainer's cores and narrow its GPUs when ppo
is imported, as they do for the run itself.
"""
from __future__ import annotations

import argparse
import os
import sys


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [--out DIR] [--readout-seed S] RUN_OR_CHECKPOINT "
              "-- RUN_ARGS...")
    p.add_argument("checkpoint",
                   help="the run directory of a finished run, or one of its "
                        "ppo_ckpt_* directories")
    p.add_argument("--out", default="",
                   help="directory for readout.jsonl; the run directory the "
                        "checkpoint sits in when not given")
    p.add_argument("--readout-seed", type=int, default=None,
                   help="the seed the plans are sampled under; the run's own "
                        "--seed when not given, which is the seed the run's "
                        "own readout samples under")
    return p


def split_argv(argv):
    """(the tool's arguments, the run's arguments), split at the first `--`."""
    if "--" not in argv:
        if "-h" in argv or "--help" in argv:
            return list(argv), []
        raise SystemExit(
            "tools/readout.py needs the run's own arguments after `--`: "
            "RUN_OR_CHECKPOINT -- <the run's arguments> [--readout N]")
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def main(argv=None) -> int:
    own, run = split_argv(sys.argv[1:] if argv is None else list(argv))
    a = make_argparser().parse_args(own)
    from alphagrad.approx import ppo
    from alphagrad.approx.common import readout as readout_mod

    args = ppo.make_argparser().parse_args(run)
    path = readout_mod.resolve_checkpoint(a.checkpoint, args.episodes)
    print(f"[readout] tools/readout.py: {path}, --readout {args.readout}, "
          f"seed {args.seed if a.readout_seed is None else a.readout_seed}",
          flush=True)
    ppo.main(args, readout_checkpoint=path, readout_out=a.out or None,
             readout_seed=a.readout_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
