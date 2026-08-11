"""elimrl measurement-worker tests (M0, CPU).

(d) budget: a plan whose predicted memory exceeds the budget is scored
    infeasible WITHOUT executing; a killed worker respawns and the next
    measurement succeeds; a hard timeout kills + respawns + scores infeasible.
"""

import pytest

from alphagrad.elimrl.env import ElimEnv
from alphagrad.elimrl.baselines import tiny_target
from alphagrad.elimrl.measure_worker import MeasureClient

TINY = {"builder": "alphagrad.elimrl.baselines:tiny_target"}


@pytest.fixture(scope="module")
def client():
    c = MeasureClient(env={"JAX_PLATFORMS": "cpu"},
                      startup_timeout=300.0, default_timeout=300.0)
    yield c
    c.close()


def test_budget_infeasible_scored_without_executing(client):
    res = client.measure(target=TINY, method="jax.jacrev",
                         budget_bytes=1, reps=2, inner=2)
    assert res["status"] == "infeasible"
    assert res["reason"] == "predicted_memory"
    assert res["executed"] is False
    assert res["mem_total_bytes"] > 1
    # the SAME worker keeps serving correct measurements afterwards
    res2 = client.measure(target=TINY, method="jax.grad", reps=2, inner=2)
    assert res2["status"] == "ok" and res2["executed"] and res2["latency_ns"] > 0


def test_measure_ok_and_numerically_checked(client):
    res = client.measure(target=TINY, method="jacve", order="rev",
                         reps=3, inner=2,
                         check_against={"method": "jax.grad"})
    assert res["status"] == "ok"
    assert res["latency_ns"] > 0
    assert res["mem_total_bytes"] is not None
    assert res["check_maxdiff"] < 1e-4
    assert res["check_cos"] > 0.999999


def test_elim_plan_roundtrip_through_worker(client):
    fn, args, an = tiny_target()
    env = ElimEnv(fn, args, an, vertex_only=True)
    for vid in sorted(env.jacve_vertices, reverse=True):
        env.apply(("V", vid))
    assert env.done
    plan = [list(a) for a in env.history]        # JSON wire format
    res = client.measure(target=TINY, method="elim_plan", plan=plan,
                         vertex_only=True, reps=2, inner=2,
                         check_against={"method": "jacve", "order": "rev"})
    assert res["status"] == "ok", res
    assert res["check_maxdiff"] < 1e-4


def test_jacfe_rev_policy_through_worker(client):
    res = client.measure(target=TINY, method="jacfe", order="rev",
                         reps=2, inner=2,
                         check_against={"method": "jacve", "order": "rev"})
    assert res["status"] == "ok", res
    assert res["check_maxdiff"] < 1e-4


def test_crashed_worker_respawns_and_next_measurement_succeeds(client):
    before = client.respawns
    res = client.request({"cmd": "crash"})
    assert res["status"] == "infeasible" and res["reason"] == "worker_died"
    assert client.respawns == before + 1
    res2 = client.measure(target=TINY, method="jax.grad", reps=2, inner=2)
    assert res2["status"] == "ok" and res2["latency_ns"] > 0


def test_timeout_kills_respawns_and_scores_infeasible(client):
    before = client.respawns
    res = client.request({"cmd": "sleep", "seconds": 120}, timeout=3)
    assert res["status"] == "infeasible" and res["reason"] == "timeout"
    assert client.respawns == before + 1
    res2 = client.measure(target=TINY, method="jax.grad", reps=2, inner=2)
    assert res2["status"] == "ok" and res2["latency_ns"] > 0
