# dsnn-dfw.226: the float64 CPU elimination of a free-order TLM plan at B=64 needs 341 GiB to
# 4.4 TiB of host memory (job 68216, XLA memory_analysis of the 16 orders of job 68195), and a
# program that fits the address space but not the RAM killed the oracle worker (job 68195).
# The program's bytes are known before it runs, so a check over the budget is refused there.
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest                                                   # noqa: E402

import alphagrad.approx.env as E                                # noqa: E402


def _exe(temp_gib, args_gib=0.02, out_mib=2.0):
    ma = types.SimpleNamespace(
        temp_size_in_bytes=int(temp_gib * 2 ** 30),
        argument_size_in_bytes=int(args_gib * 2 ** 30),
        output_size_in_bytes=int(out_mib * 2 ** 20),
        generated_code_size_in_bytes=12345)
    return types.SimpleNamespace(memory_analysis=lambda: ma)


def test_a_program_over_the_budget_is_refused_before_it_runs():
    with pytest.raises(E.GradientOracleRefused) as exc:
        E._grad_oracle_check_fits(_exe(438.6), [63, 78, 83, 23, 1, 57, 61], budget_gb=256)
    msg = str(exc.value)
    assert "438.6 GiB" in msg and "256 GiB" in msg and "(63, 78, 83, 23, 1, 57)" in msg
    assert "not run" in msg and "missing" in msg


def test_a_program_under_the_budget_runs_and_reports_its_bytes():
    need = E._grad_oracle_check_fits(_exe(42.6), [1, 2, 3], budget_gb=256)
    assert abs(need / 2 ** 30 - 42.62) < 0.01
    assert E._grad_oracle_check_fits(_exe(4000.0), [1, 2, 3], budget_gb=0) > 0, "0 = no bar"


def test_the_budget_reader_and_its_flag(monkeypatch):
    from alphagrad.approx.ppo import make_argparser

    actions = {a.option_strings[0]: a for a in make_argparser()._actions if a.option_strings}
    assert actions["--grad-oracle-host-budget-gb"].default == E.grad_oracle_host_budget_gb_default()
    assert E.grad_oracle_host_budget_gb_default() == 256.0
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_HOST_BUDGET_GB", "120.5")
    assert E.grad_oracle_host_budget_gb() == 120.5
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_HOST_BUDGET_GB", "-1")
    with pytest.raises(ValueError):
        E.grad_oracle_host_budget_gb()
    monkeypatch.delenv("ALPHAGRAD_GRAD_ORACLE_HOST_BUDGET_GB")
    assert E.grad_oracle_host_budget_gb() == 256.0


def test_the_exact_elimination_gates_on_the_budget_before_running():
    src = inspect.getsource(E._grad_oracle_exact)
    assert src.index("_grad_oracle_check_fits(exe, order)") < src.index("out = exe(*args)")


def test_a_refusal_inside_the_check_is_an_error_not_a_stop(tmp_path):
    from alphagrad.approx.common.grad_oracle_async import AsyncGradOracle
    from alphagrad.approx.ppo import _grad_oracle_boundary

    def check(order, probe_seed, episode):
        E._grad_oracle_check_fits(_exe(877.2), order, budget_gb=256)
        return "pass", 1e-14

    oracle = AsyncGradOracle(check, timeout_s=60.0)
    try:
        oracle.submit(0, 1, [{"order": (63, 78, 83), "plan_hashes": ["h"]}])
        end = __import__("time").monotonic() + 20.0
        while oracle._done.qsize() == 0 and __import__("time").monotonic() < end:
            __import__("time").sleep(0.01)
        out = _grad_oracle_boundary(oracle, str(tmp_path / "p.jsonl"), 1, 1e-3,
                                    log=lambda _l: None)
        assert [r["status"] for r in out] == ["error"]
        assert "877.2 GiB" in out[0]["error"]
        assert oracle.counts()["fail"] == 0
    finally:
        oracle.close()
