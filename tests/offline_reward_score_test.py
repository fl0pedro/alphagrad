"""dsnn-3qm.42 -- the offline reward scorer on a synthetic landscape.

The scorer must (1) reproduce the trainer's paired-log reward for both forms,
(2) re-base sweep ratios on the order's own identity, (3) pick the absorber
and the baseline the gate G6 contrast is measured against, and (4) raise,
not guess, when the baseline is missing.
"""
from __future__ import annotations

import json
import math

import pytest

from alphagrad.approx.tools import offline_reward_score as S


def _summary(tmp_path):
    """Three plans on an order whose identity costs 10x latency and 4x
    memory against rev-exact: a clean saver, a gradient killer, and a
    low-quality saver."""
    def e(lat, mem, q, op="quant"):
        return {"op": op, "quality": {"mean": q}, "latency_ratio": {"mean": lat},
                "static_temp_ratio": {"mean": mem}, "mem_ratio": {"mean": mem}}
    summ = {
        "identity [gpu0]": e(10.0, 4.0, 1.0, "identity"),
        "identity [gpu1]": e(10.0, 4.0, 1.0, "identity"),
        "singleton:quant:k1.f0:lhs:bf16 [gpu0]": e(8.0, 3.0, 0.99),     # -20% lat, -25% mem
        "singleton:skip:k2.f0:v9/log [gpu0]": e(1.0, 0.4, 0.0),         # absorber
        "singleton:reduce:k3.f0:new:ax0 [gpu0]": e(5.0, 2.0, 0.7),      # cheap but q < tau
    }
    p = tmp_path / "summary_COMBINED.json"
    p.write_text(json.dumps({"summary": summ, "extra": {}, "notes": []}))
    return str(p)


def test_rebases_on_the_orders_identity(tmp_path):
    pts = {p.plan_id: p for p in S.points_from_sweep_summary(_summary(tmp_path))}
    ident = pts["identity"]
    assert ident.kind == "identity" and ident.dlat == 0.0 and ident.dmem == 0.0
    q = pts["singleton:quant:k1.f0:lhs:bf16"]
    assert q.dlat == pytest.approx(math.log(0.8)) and q.dmem == pytest.approx(math.log(0.75))
    raw = {p.plan_id: p for p in S.points_from_sweep_summary(_summary(tmp_path), reference="rev-exact")}
    assert raw["identity"].dlat == pytest.approx(math.log(10.0))


def test_reward_forms_match_the_trainer():
    p = S.Point("x", "s", "quant", math.log(0.8), math.log(0.75), 0.9)
    cost = -math.log(0.8) - math.log(0.75)
    assert S.reward(p, form="P0", lam_lat=1, lam_mem=1, lam_q=4, tau=0.95) == pytest.approx(cost + 4 * 0.9)
    assert S.reward(p, form="P1", lam_lat=1, lam_mem=1, lam_q=4, tau=0.95) == pytest.approx(cost - 4 * 0.05)
    assert S.reward(p, form="P1", lam_lat=1, lam_mem=1, lam_q=4, tau=0.8) == pytest.approx(cost)
    with pytest.raises(ValueError):
        S.reward(p, form="P2", lam_lat=1, lam_mem=1, lam_q=1, tau=0.9)


def test_contrast_uses_absorber_and_baseline(tmp_path):
    pts = S.points_from_sweep_summary(_summary(tmp_path))
    rows = S.score_grid(pts, forms=("P1",), lam_qs=(1.0, 8.0), taus=(0.9,), absorber_q=0.05)
    by_lq = {r["lambda_q"]: r for r in rows}
    # lambda_q = 1: the gradient killer pays 0.9 for its q = 0 and still wins
    # on cost (log 10 + log 10 ~ 4.6), so the absorber is the floor.
    r1 = by_lq[1.0]
    assert r1["n_absorbers"] == 1 and r1["R_absorber"] > r1["R_baseline"]
    assert r1["best_plan"] == "singleton:quant:k1.f0:lhs:bf16"
    assert r1["contrast"] < 0, "a lambda_q that lets the absorber win must show a negative contrast"
    # lambda_q = 8: the absorber pays 7.2 and drops below the baseline; the
    # saver wins by its cost saving (its q clears tau, so no hinge).
    r8 = by_lq[8.0]
    assert r8["R_absorber"] < r8["R_baseline"] == 0.0
    assert r8["contrast"] == pytest.approx(-math.log(0.8) - math.log(0.75))
    # the q = 0.7 plan is never feasible at tau = 0.9
    assert r8["n_feasible"] == 1


def test_missing_baseline_raises():
    pts = [S.Point("a", "s", "quant", -0.1, -0.1, 0.99)]
    with pytest.raises(ValueError):
        S.score_grid(pts)


def test_plan_log_points_use_the_logs_exact_plans(tmp_path):
    names = ["muls_adds_fmas", "flops", "latency_ns", "max_io_sum", "bytes_accessed",
             "peak_memory", "quality", "grad_coverage", "fidelity", "bkstep_acc", "sparsity"]
    def rec(lat, mem, q, n_req):
        r = [0.0] * len(names)
        r[2], r[5], r[6] = -lat, -mem, q
        return {"reward_names": names, "rewards": r, "requested": {"total": n_req}}
    p = tmp_path / "plan_log_x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        rec(100.0, 50.0, 1.0, 0), rec(100.0, 50.0, 1.0, 0), rec(80.0, 40.0, 0.95, 3)]) + "\n")
    pts = S.points_from_plan_log(str(p), source="w1")
    assert [pt.kind for pt in pts] == ["identity", "identity", "plan"]
    assert pts[2].dlat == pytest.approx(math.log(0.8)) and pts[2].dmem == pytest.approx(math.log(0.8))
    rows = S.score_grid(pts, forms=("P0",), lam_qs=(4.0,), taus=(0.9,))
    assert rows[0]["best_plan"] == "w1#2"


def test_cli_writes_markdown_and_json(tmp_path):
    out = tmp_path / "report.md"
    rc = S.main(["--sweep-summary", f"toy={_summary(tmp_path)}", "--lambda-q", "4",
                 "--tau", "0.9", "--out", str(out)])
    assert rc == 0 and out.exists()
    text = out.read_text()
    assert "| P0 | 4 | 0.9 |" in text and "| P1 | 4 | 0.9 |" in text
    data = json.loads((tmp_path / "report.json").read_text())
    assert set(data) == {"union"} and len(data["union"]) == 2
