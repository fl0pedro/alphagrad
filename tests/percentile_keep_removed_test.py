"""--percentile-keep and its trimmed-mean sibling --latency-winsor were
DELETED 2026-09-14 (owner ruling, small fixes #1). Both were read only by
the legacy Ray/CPU workers (ppo_ray_worker.py, cpu_approx_worker.py,
verify_pareto_solution.py); the campaign path (env.py's
`_aggregate_samples`) always aggregated the noisy-channel pool with a
plain MEDIAN and never read either flag -- `VertexEliminationEnv.from_jaxpr`
swallows both names into its `**_compat` catch-all without ever turning
them into an `EnvConfig` field. The median stays; the flags go. This pins
the parser-level guard so neither can silently reappear.
"""
import argparse

import pytest

from alphagrad.approx.args_ppo import add_ppo_args
from alphagrad.approx.ppo_args import make_argparser


def _ppo_ray_parser() -> argparse.ArgumentParser:
    return add_ppo_args(argparse.ArgumentParser())


def test_percentile_keep_flag_is_gone():
    p = _ppo_ray_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--percentile-keep", "0.6"])


def test_latency_winsor_flag_is_also_gone():
    """The trimmed-mean sibling named in --percentile-keep's old help text
    ("aggregate ... with a symmetric winsorized mean ... instead of
    --percentile-keep"): confirmed unread by the campaign path too, so it
    was deleted alongside --percentile-keep rather than left dangling."""
    p = _ppo_ray_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--latency-winsor", "0.2"])


def test_the_assembled_ray_ppo_cli_rejects_it_too():
    """`ppo_args.make_argparser` (add_common_args + add_ppo_args) is the
    parser ppo_ray.py actually builds; pin that surface too, not just the
    raw `add_ppo_args` one."""
    p = make_argparser()
    with pytest.raises(SystemExit):
        p.parse_args(["--percentile-keep", "0.6"])
    with pytest.raises(SystemExit):
        p.parse_args(["--latency-winsor", "0.2"])


def test_the_other_flags_from_the_same_block_still_parse():
    """A sanity check that the deletion did not take neighbours with it:
    --ref-num-data-points / --ref-reps-per-point (owner ruling 2026-09-14,
    same date, different ticket) and --latency-warmup are untouched."""
    p = _ppo_ray_parser()
    ns = p.parse_args([
        "--ref-num-data-points", "3", "--ref-reps-per-point", "7",
        "--latency-warmup", "2",
    ])
    assert ns.ref_num_data_points == 3
    assert ns.ref_reps_per_point == 7
    assert ns.latency_warmup == 2
    assert not hasattr(ns, "percentile_keep")
    assert not hasattr(ns, "latency_winsor")
