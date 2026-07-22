"""JAX-free CLI argparser for `az_gumbel.py`.

Thin composition layer, kept so every existing importer (`az_gumbel.py`
itself, which parses at import time, plus any launcher running
``python -m alphagrad.approx.az_gumbel``) keeps working unchanged. The
arguments themselves now live in two modules:

* `args_common.add_common_args` — the flags ppo_ray shares (`--seed`,
  `--dataset`, `--lr`, `--latency-inner-reps`), with az's defaults
  supplied from `args_az.COMMON_DEFAULTS`;
* `args_az.add_az_args` — everything else, az-only (the search knobs
  `--n-candidates` / `--rollout-depth` have no PPO counterpart).

`make_argparser` composes the two, which is exactly what the unified
runner (`run.py az ...`) does, so the two entry points cannot drift.
The composed surface (option strings, dest, default, type, nargs,
const, choices, required, help, action class) is identical to the
single-file parser this module used to build; only the order the flags
appear in `--help` changed.

Kept JAX-free (argparse + os only) so the parser can be built — and
`--help` printed — without triggering JAX init, same hygiene as
`ppo_args.py`.

Search behaviour beyond these flags is env-var driven
(`ALPHAGRAD_GAZ_*`, `ALPHAGRAD_POPART_*`); see `az_gumbel.py`.
"""

from __future__ import annotations

import argparse

from alphagrad.approx.args_az import COMMON_DEFAULTS, add_az_args
from alphagrad.approx.args_common import add_common_args


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sampled + Gumbel AlphaZero trainer for vertex elimination.",
    )
    add_common_args(p, **COMMON_DEFAULTS)
    add_az_args(p)
    return p
