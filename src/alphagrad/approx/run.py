"""Unified entry point for the two approx RL trainers.

    python -m alphagrad.approx.run ppo [flags...]   ->  ppo_ray.py
    python -m alphagrad.approx.run az  [flags...]   ->  az_gumbel.py

SUBCOMMANDS rather than an `--algo` flag, so each algorithm keeps its
own `--help`. Every subcommand's parser is composed the same way:
`args_common.add_common_args` (the flags both trainers read, with the
algorithm's own defaults from its `COMMON_DEFAULTS`) plus that
algorithm's `add_*_args`. `ppo_args.make_argparser` and
`az_args.make_argparser` compose the identical pair, so the runner and
the direct `python -m alphagrad.approx.ppo_ray` entry points cannot
drift apart.

PURE DISPATCH — no algorithm logic lives here. This module owns the
subcommand table, the parser composition and the handover; everything
downstream of that belongs to the trainer.

Two properties of the trainers shape the handover:

* `az_gumbel` parses `sys.argv` at IMPORT time (its module-level `A`
  exports `ALPHAGRAD_MS_*` before any search function is defined), so
  `sys.argv` is rewritten to the algorithm's own argv BEFORE the
  import — otherwise that parse would choke on the subcommand token.
* Each trainer's `main()` runs a prologue that its `_run(args)` hook
  presumes has already happened: `ppo_ray.main` adds the driver-only
  knobs of `ppo_ray._extend_argparser` (`--ray-address`,
  `--num-cpu-workers`, `--actor-num-gpus`, ...), stamps
  `args.t_start` and calls `ray.init`/`ray.shutdown` around `_run`;
  `az_gumbel.main` relies on the import-time env exports above.
  Reproducing either here would put driver logic in the dispatcher, so
  the handover targets `main()` — the trainer-owned wrapper whose only
  job is to parse argv and call `_run(args)`. `_Algo.inner` records
  that inner hook and it is checked before dispatch, so a trainer that
  loses its `_run(args) -> int` contract fails loudly here.

Because those driver-only knobs are owned by the trainer and not by
`args_ppo`, flags the composed parser does not know are forwarded
verbatim instead of being rejected; the trainer's own parser is the
authority on its full surface and still rejects genuine typos.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from alphagrad.approx.args_az import COMMON_DEFAULTS as _AZ_COMMON_DEFAULTS
from alphagrad.approx.args_az import add_az_args
from alphagrad.approx.args_common import add_common_args
from alphagrad.approx.args_ppo import COMMON_DEFAULTS as _PPO_COMMON_DEFAULTS
from alphagrad.approx.args_ppo import add_ppo_args


@dataclass(frozen=True)
class _Algo:
    """One row of the dispatch table."""

    #: Importable module path of the trainer. Imported LAZILY, inside
    #: the dispatch branch — both trainers are expensive to import and
    #: `az_gumbel` has import-time side effects.
    module: str
    #: Module attribute the runner calls (takes no args, returns int).
    entry: str
    #: The `(args) -> int` hook `entry` funnels into. Not called here;
    #: checked so the contract can't rot silently.
    inner: str
    #: Adds the algorithm-only flags.
    add_args: Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
    #: This algorithm's values for the shared `add_common_args` flags.
    common_defaults: dict = field(repr=False)
    #: One-liner for the top-level subcommand listing.
    help: str
    #: Header for `run.py <algo> --help`.
    description: str


_ALGOS: dict[str, _Algo] = {
    "ppo": _Algo(
        module="alphagrad.approx.ppo_ray",
        entry="main",
        inner="_run",
        add_args=add_ppo_args,
        common_defaults=_PPO_COMMON_DEFAULTS,
        help="Ray-actor PPO trainer (ppo_ray.py).",
        description="Ray-actor PPO trainer (first-cut) for vertex elimination.",
    ),
    "az": _Algo(
        module="alphagrad.approx.az_gumbel",
        entry="main",
        inner="_run",
        add_args=add_az_args,
        common_defaults=_AZ_COMMON_DEFAULTS,
        help="Sampled + Gumbel AlphaZero trainer (az_gumbel.py).",
        description=(
            "Sampled + Gumbel AlphaZero trainer for vertex elimination."
        ),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser with one subcommand per algorithm.

    Cheap and JAX-free: only the `args_*` modules are touched, never a
    trainer.
    """
    p = argparse.ArgumentParser(
        prog="python -m alphagrad.approx.run",
        description="Unified runner for the approx RL trainers.",
        epilog="Driver-only flags owned by the trainer itself (e.g. "
               "--ray-address / --num-cpu-workers / --actor-num-gpus for "
               "ppo) are forwarded through to it; see "
               "`python -m alphagrad.approx.ppo_ray --help` for those.",
    )
    sub = p.add_subparsers(dest="algo", required=True, metavar="{ppo,az}")
    for name, algo in _ALGOS.items():
        sp = sub.add_parser(name, help=algo.help, description=algo.description)
        add_common_args(sp, **algo.common_defaults)
        algo.add_args(sp)
    return p


def _algo_argv(argv: list[str], algo: str) -> list[str]:
    """Everything after the subcommand token — the trainer's own argv.

    The top-level parser carries no options of its own, so the first
    occurrence of the subcommand name is always the token argparse
    dispatched on.
    """
    return argv[argv.index(algo) + 1:]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Unknown flags are the trainer's driver-only knobs; it validates them.
    args, _forwarded = build_parser().parse_known_args(argv)
    algo = _ALGOS[args.algo]

    saved_argv = list(sys.argv)
    sys.argv = [saved_argv[0], *_algo_argv(argv, args.algo)]
    try:
        module = importlib.import_module(algo.module)
        if not callable(getattr(module, algo.inner, None)):
            raise RuntimeError(
                f"{algo.module} no longer exposes a callable "
                f"{algo.inner}(args) -> int; the runner dispatches to "
                f"{algo.entry}() on the assumption that it wraps it."
            )
        return int(getattr(module, algo.entry)())
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    sys.exit(main())
