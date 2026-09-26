# Owner ruling 2026-09-26, Q12: every measurement records p10, p25, p50, p75 and p90 of the latency
# over its timed windows, for the candidate and for the reference, in the paired-reference record and in
# the plan record. The median stays the latency. A one-run sample fills every field with its reading.
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as E                           # noqa: E402
from measure_instrument_test import (                           # noqa: E402,F401
    _one_instrument_toy_env, _paired_log_cpu, _walk)

QS = (10, 25, 50, 75, 90)
FIELDS = [f"{h}_latency_p{q}_ns" for h in ("candidate", "ref") for q in QS]


def test_the_quantiles_of_a_sample_and_their_record_fields():
    s = [float(x) for x in range(1, 11)]
    assert E.latency_quantiles(s) == {f"p{q}": float(np.percentile(s, q)) for q in QS}
    assert E.latency_quantiles([7.0]) == {f"p{q}": 7.0 for q in QS}
    assert E.latency_quantiles([]) is None
    assert E.latency_quantile_fields("ref", E.latency_quantiles([7.0])) == {
        f"ref_latency_p{q}_ns": 7.0 for q in QS}
    assert E.latency_quantile_fields("candidate", None) == {
        f"candidate_latency_p{q}_ns": None for q in QS}


def _measure(monkeypatch, lat_ns=None):
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    if lat_ns is not None:
        real = E._time_one_rep

        def fixed(ex, eval_args, devices, inner):
            _l, p, s, o = real(ex, eval_args, devices, inner)
            return lat_ns, p, s, o
        monkeypatch.setattr(E, "_time_one_rep", fixed)
    E.consume_plan_records()
    env = _one_instrument_toy_env(num_data_points=2, reps_per_point=2,
                                  ref_num_data_points=3, ref_reps_per_point=5)
    _walk(env, sorted(int(v) for v in np.asarray(env.valid_vertices)))
    out = E.consume_plan_records()
    recs = [r for r in out["records"] if "plan_hash" in r]
    assert recs, "no plan record"
    return recs, out["paired_ref"]["records"]


def test_every_measured_plan_records_both_halves_quantiles(_paired_log_cpu, monkeypatch):
    recs, refs = _measure(monkeypatch)
    for r in recs:
        for half in ("candidate", "ref"):
            q = [r[f"{half}_latency_p{p}_ns"] for p in QS]
            assert all(x is not None for x in q), (half, q)
            assert q == sorted(q), (half, q)
        assert r["candidate_latency_p50_ns"] == pytest.approx(r["candidate_latency_ns"], rel=1e-6)
        assert r["ref_latency_p50_ns"] == pytest.approx(r["ref_latency_ns"], rel=1e-6)
    for pr in refs:
        assert all(pr[f] is not None for f in FIELDS), pr


def test_a_one_run_sample_fills_every_field_with_its_reading(_paired_log_cpu, monkeypatch):
    # Past the 1 s budget the warm run is the whole sample, for both halves.
    recs, _refs = _measure(monkeypatch, lat_ns=2.0e9)
    for r in recs:
        assert r["measure_windows"] == 1 and r["ref_measure_windows"] == 1
        assert [r[f] for f in FIELDS] == [2.0e9] * len(FIELDS)
