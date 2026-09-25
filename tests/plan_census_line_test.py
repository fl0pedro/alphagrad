from __future__ import annotations

import ast
import inspect
import re

import numpy as np

import alphagrad.approx.ppo as ppo


def _rows(slot, quality=None, approx=None):
    n = len(slot)
    approx = np.asarray([0] * n if approx is None else approx, np.float64)
    return {
        "quality": np.asarray([1.0] * n if quality is None else quality,
                              np.float64),
        "latency_ns": np.asarray(slot, np.float64),
        "approx_count": approx,
        "req_diag": approx,
        "req_compress": np.zeros(n),
        "req_quant": np.zeros(n),
        "approx_applied_est": approx,
        "skip_count": np.zeros(n),
    }


def test_slower_plans_under_paired_log_are_no_win_and_no_microseconds():
    # The slot holds -log(candidate/reference). Every plan is slower than the reference;
    # the first, 7 percent slower with no approximation, is below 0.9x the median (job 67852).
    rows = _rows([-0.071, -0.15, -0.2, -0.25])
    line = ppo._plan_census_line(0, rows, np.ones(4, bool), True)
    assert "WIN none" in line, line
    assert not re.search(r"lat_us|\dus\b", line), line
    assert "lat_logratio med=+0.175 [+0.071,+0.25]" in line, line


def test_a_faster_plan_under_paired_log_is_a_win_by_its_log_ratio():
    rows = _rows([0.105, -0.2, -0.3])
    line = ppo._plan_census_line(0, rows, np.ones(3, bool), True)
    assert "WIN env=0 lat_logratio=-0.105 q=+1.000 ops=0" in line, line


def test_absolute_costs_keep_microseconds_and_the_median_rule():
    rows = _rows([-50e3, -100e3, -120e3])
    line = ppo._plan_census_line(0, rows, np.ones(3, bool), False)
    assert "lat_us med=100 [50,120]" in line, line
    assert "WIN env=0 lat=50us q=+1.000 ops=0" in line, line


def test_host_log_prints_the_helper_line_with_the_cost_form():
    tree = ast.parse(inspect.getsource(ppo))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "host_log")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "_plan_census_line"]
    assert calls
    assert all(ast.unparse(c.args[3]) == "_paired_costs" for c in calls)
    assert not any(isinstance(n, ast.Constant) and n.value == "lat_us "
                   for n in ast.walk(fn))


def _with_excluded_row():
    # Rows 0 to 2 are live. Row 3 is excluded: the sentinel in every cost slot.
    rows = _rows([-0.05, -0.1, -0.2], quality=[0.9, 0.95, 1.0], approx=[1, 2, 3])
    excluded = {"quality": 0.0, "latency_ns": ppo.SENTINEL_COST,
                "approx_count": 30.0, "req_diag": 30.0, "req_compress": 0.0,
                "req_quant": 0.0, "approx_applied_est": 30.0, "skip_count": 9.0}
    rows = {k: np.append(v, excluded[k]) for k, v in rows.items()}
    return rows, np.array([True, True, True, False])


def test_an_excluded_row_does_not_enter_the_summary():
    rows, live = _with_excluded_row()
    for paired_log in (True, False):
        line = ppo._plan_census_line(4, rows, live, paired_log)
        assert "live=3" in line, line
        assert not re.search(r"1e\+0?7|1e\+10|30", line), line
        assert "q med=+0.95 [+0.9,+1]" in line, line
        assert "req med=2 [1,3]" in line and "req d/c/q=6/0/0" in line, line
        assert "skips=0" in line, line
    line = ppo._plan_census_line(4, rows, live, True)
    assert "lat_logratio med=+0.1 [+0.05,+0.2]" in line, line
    line = ppo._plan_census_line(4, rows, live, False)
    assert "lat_us med=0.0001 [5e-05,0.0002]" in line, line


def test_a_batch_with_no_live_row_prints_no_numbers():
    rows, _ = _with_excluded_row()
    line = ppo._plan_census_line(2, rows, np.zeros(4, bool), True)
    assert line == "[plan ep=2] n=4 live=0 | no live row", line
