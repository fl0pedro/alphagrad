"""JAX-free CLI argparser for `ppo_ray.py`.

Thin composition layer, kept so every existing importer (`ppo_ray.py`,
the sbatch launchers via ``python -m alphagrad.approx.ppo_ray``) keeps
working unchanged. The arguments themselves now live in two modules:

* `args_common.add_common_args` — the flags az_gumbel shares
  (`--seed`, `--dataset`, `--lr`, `--latency-inner-reps`), with PPO's
  defaults supplied from `args_ppo.COMMON_DEFAULTS`;
* `args_ppo.add_ppo_args` — everything else, PPO-only.

`make_argparser` composes the two, which is exactly what the unified
runner (`run.py ppo ...`) does, so the two entry points cannot drift.
The composed surface (option strings, dest, default, type, nargs,
const, choices, required, help, action class) is identical to the
single-file parser this module used to build; only the order the flags
appear in `--help` changed.

Kept JAX-free so the driver in `ppo_ray.py` can construct the parser
without triggering JAX init.
"""

from __future__ import annotations

import argparse

from alphagrad.approx.args_common import add_common_args
from alphagrad.approx.args_ppo import COMMON_DEFAULTS, add_ppo_args


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ray-actor PPO trainer (first-cut) for vertex elimination.",
    )
    add_common_args(p, **COMMON_DEFAULTS)
    add_ppo_args(p)
    return p
