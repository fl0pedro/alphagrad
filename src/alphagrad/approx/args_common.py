"""JAX-free CLI arguments shared by `ppo_ray.py` and `az_gumbel.py`.

The knobs BOTH trainers actually read, factored out of `ppo_args.py`
and `az_args.py` so the unified runner (`run.py`) composes each
algorithm's surface as ``add_common_args(p, ...) + add_<algo>_args(p)``
instead of two parsers drifting apart.

The two trainers disagree on the per-algorithm *values* of these
flags — az seeds with 7 and PPO with 250197, az defaults to the mnist
dataset and PPO to none — so the defaults are keyword parameters here
rather than hard-coded. Each algorithm's `COMMON_DEFAULTS` in
`args_ppo.py` / `args_az.py` supplies them, which keeps every composed
parser byte-identical (option strings, dest, default, type, nargs,
const, choices, required, help, action class) to the parser that
algorithm had before the split.

Flags that only LOOK shared stay per-algorithm: `--wandb` is a
`store_true` in az but a `str` with `choices` in PPO, so it cannot be
expressed once without changing one of the two surfaces.

Kept JAX-free (argparse only) so the parser can be built — and
`--help` printed — without triggering JAX init, same hygiene as
`ppo_args.py`.
"""

from __future__ import annotations

import argparse


def add_common_args(
    p: argparse.ArgumentParser,
    *,
    seed: int,
    dataset: str,
    latency_inner_reps: int,
    lr: float = 3e-4,
    dataset_type: type | None = str,
    latency_inner_reps_help: str | None = None,
) -> argparse.ArgumentParser:
    """Add the flags both trainers read to `p` and return `p`.

    `dataset_type` and `latency_inner_reps_help` exist only so the two
    pre-split surfaces survive verbatim: az declares `--dataset`
    without a `type=` (argparse stores ``None``, i.e. raw string) while
    PPO declares `type=str`, and PPO documents `--latency-inner-reps`
    in terms of its own measurement loop while az leaves it bare.
    """
    # Run / task
    p.add_argument("--seed", type=int, default=seed)
    p.add_argument("--dataset", type=dataset_type, default=dataset)

    # Optimizer
    p.add_argument("--lr", type=float, default=lr)

    # Measurement
    p.add_argument(
        "--latency-inner-reps", type=int, default=latency_inner_reps,
        help=latency_inner_reps_help,
    )

    return p
