"""The refusal RATE is visible, per episode and per kind.

The trainer drops an excluded environment from the update entirely, which is
correct and SILENT: a run whose refusal rate walks from 2 percent to 40
percent trains on fewer and fewer environments while every panel still looks
healthy. These counters are what makes that readable. They ride the drain
every measure actor already answers (`consume_collapse_stats`), NOT the plan
log, because the rate has to be readable with `--plan-log` off.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import alphagrad.approx.env as env                              # noqa: E402


def _drain():
    env.consume_refused_counts()


def test_the_kind_is_the_reason_prefix():
    assert env.refusal_kind("oom:approx compile") == "oom"
    assert env.refusal_kind("untraceable:measurement") == "untraceable"
    assert env.refusal_kind("raised:JaxRuntimeError") == "raised"
    assert env.refusal_kind("oracle:XlaRuntimeError") == "oracle"
    assert env.refusal_kind("muls-cap") == "muls-cap"
    assert env.refusal_kind("") == "unknown"


def test_counts_accumulate_per_kind_and_pop_on_drain():
    _drain()
    env._record_refusal("oom:approx compile", scored=True)
    env._record_refusal("oom:measurement", scored=True)
    env._record_refusal("raised:JaxRuntimeError", scored=True)
    env._record_refusal("oracle:XlaRuntimeError", scored=False)
    out = env.consume_refused_counts()
    assert out == {"oom": 2, "raised": 1, "oracle": 1, "total": 4,
                   "scored": 3, "excluded": 1}
    # POPPED: a second drain of the same episode reports nothing.
    assert env.consume_refused_counts() == {}


def test_the_oracle_is_counted_apart_from_the_plan():
    """The owner keeps the float64 oracle even though it can exhaust the
    device on its own compile. The refusal is then the APPARATUS failing, not
    the plan, and the two must never share one number."""
    _drain()
    env._record_refusal("oracle:XlaRuntimeError", scored=False)
    env._record_refusal("raised:XlaRuntimeError", scored=True)
    out = env.consume_refused_counts()
    assert out["oracle"] == 1
    assert out["raised"] == 1


def test_the_counters_roll_back_with_a_discarded_attempt():
    """An episode whose token stream overflows is DISCARDED and repeated.
    Its refusals must not be counted twice."""
    _drain()
    env._record_refusal("oom:approx compile", scored=True)
    snap = env.episode_telemetry_snapshot()
    env._record_refusal("oom:measurement", scored=True)
    env._record_refusal("raised:JaxRuntimeError", scored=True)
    env.episode_telemetry_restore(snap)
    out = env.consume_refused_counts()
    assert out == {"oom": 1, "total": 1, "scored": 1}
