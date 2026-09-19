"""The distribution-aware archive rule (ticket dsnn-dfw.44)."""
import json

import numpy as np
import pytest

from alphagrad.approx.common.pareto_archive import RatioBandArchive

OBJ = ("latency", "peak_memory")


def _d(lat, mem=(0.0, 0.0, 0.0)):
    return {"latency": {"q05": lat[0], "median": lat[1], "q95": lat[2]},
            "peak_memory": {"q05": mem[0], "median": mem[1], "q95": mem[2]}}


def _arch(**kw):
    return RatioBandArchive(OBJ, **kw)


def test_median_beyond_q95_is_dominated():
    a = _arch()
    assert a.add(_d((-0.10, -0.08, -0.06)), [1], 0)
    # worse than the point's q95 on latency, inside its band on memory
    assert not a.add(_d((0.01, 0.03, 0.05)), [2], 1)
    assert len(a.pts) == 1


def test_median_below_q05_removes_the_point():
    a = _arch()
    assert a.add(_d((-0.02, 0.00, 0.02)), [1], 0)
    assert a.add(_d((-0.30, -0.28, -0.26)), [2], 1)
    assert len(a.pts) == 1
    assert a.seqs == [[2]]


def test_inside_the_band_merges_and_counts():
    a = _arch()
    assert a.add(_d((-0.10, -0.08, -0.06)), [1], 0)
    assert not a.add(_d((-0.09, -0.07, -0.065)), [2], 1)
    assert len(a.pts) == 1
    assert a.counts == [2]
    assert a.n_merged == 1
    assert a.seqs == [[1]]


def test_non_dominated_pair_is_kept():
    a = _arch()
    assert a.add(_d((-0.10, -0.08, -0.06), (0.10, 0.12, 0.14)), [1], 0)
    assert a.add(_d((0.10, 0.12, 0.14), (-0.10, -0.08, -0.06)), [2], 1)
    assert len(a.pts) == 2


def test_drift_floor_widens_the_band_into_a_merge():
    a = _arch()
    assert a.add(_d((-0.001, 0.000, 0.001)), [1], 0)
    a.set_drift_floor(0.20)
    lo, hi = a.band(0)
    assert lo[0] == pytest.approx(-0.1)
    assert hi[0] == pytest.approx(0.1)
    # 0.05 is beyond q95 but inside the widened band
    assert not a.add(_d((0.049, 0.050, 0.051)), [2], 1)
    assert a.counts == [2]


def test_drift_floor_never_narrows_a_band():
    a = _arch()
    a.add(_d((-0.50, 0.00, 0.50)), [1], 0)
    a.set_drift_floor(0.10)
    lo, hi = a.band(0)
    assert lo[0] == pytest.approx(-0.5)
    assert hi[0] == pytest.approx(0.5)


def test_non_finite_drift_floor_keeps_the_last_reading():
    a = _arch()
    a.set_drift_floor(0.3)
    a.set_drift_floor(float("nan"))
    assert a.drift_floor == pytest.approx(0.3)
    with pytest.raises(ValueError):
        a.set_drift_floor(-1.0)


def test_cap_drops_the_widest_band():
    a = _arch(cap=3)
    # three mutually non-dominated points; the second has the widest band
    assert a.add(_d((-0.31, -0.30, -0.29), (0.29, 0.30, 0.31)), ["a"], 0)
    assert a.add(_d((-0.20, -0.10, 0.00), (-0.05, 0.05, 0.15)), ["b"], 0)
    assert a.add(_d((0.29, 0.30, 0.31), (-0.31, -0.30, -0.29)), ["c"], 0)
    assert len(a.pts) == 3
    assert a.add(_d((-0.26, -0.25, -0.24), (0.18, 0.19, 0.20)), ["d"], 1)
    assert len(a.pts) == 3
    assert a.n_dropped_cap == 1
    assert ["b"] not in a.seqs
    assert ["d"] in a.seqs


def test_admitted_count_is_the_auto_stop_contract():
    a = _arch()
    sols = [(_d((-0.10, -0.08, -0.06)), [1], 1.0),
            (_d((-0.09, -0.07, -0.065)), [2], 1.0),
            (_d((0.20, 0.22, 0.24), (-0.30, -0.28, -0.26)), [3], 1.0)]
    assert a.add_many(sols, 0) == 2
    assert a.n_merged == 1


def test_quality_floor_refuses_an_infeasible_plan():
    a = _arch(quality_floor=0.9)
    assert not a.add(_d((-0.50, -0.48, -0.46)), [1], 0, quality=0.80)
    assert a.add(_d((-0.10, -0.08, -0.06)), [2], 0, quality=0.90)
    assert len(a.pts) == 1


def test_missing_or_broken_distribution_raises():
    a = _arch()
    with pytest.raises(ValueError):
        a.add(None, [1], 0)
    with pytest.raises(ValueError):
        a.add(_d((0.10, 0.05, 0.20)), [1], 0)          # q05 above the median
    with pytest.raises(ValueError):
        a.add(_d((float("nan"), 0.0, 0.1)), [1], 0)


def test_hypervolume_and_size_stay_meaningful():
    a = _arch()
    assert a.hypervolume() == 0.0
    a.add(_d((-0.02, 0.00, 0.02)), [1], 0)
    hv1 = a.hypervolume()
    a.add(_d((-0.32, -0.30, -0.28)), [2], 1)
    hv2 = a.hypervolume()
    assert np.isfinite(hv1) and np.isfinite(hv2)
    assert hv2 > hv1
    assert len(a.pts) == 1


def test_dump_front_writes_the_bands(tmp_path):
    a = _arch()
    a.set_drift_floor(0.05)
    a.add(_d((-0.10, -0.08, -0.06)), [[1, []], [2, []]], 7)
    p = tmp_path / "pareto_front.json"
    a.dump_front(str(p), extra={"episode": 7})
    doc = json.loads(p.read_text())
    assert doc["objectives"] == list(OBJ)
    assert doc["drift_floor"] == pytest.approx(0.05)
    row = doc["front"][0]
    for k in ("obj", "q05", "q95", "band_lo", "band_hi"):
        assert set(row[k]) == set(OBJ)
    assert row["band_lo"]["latency"] < row["q05"]["latency"]
    assert row["band_hi"]["latency"] > row["q95"]["latency"]
    assert row["n"] == 1 and row["episode"] == 7


def test_point_keeps_the_half_of_the_axis_the_reward_floors_away():
    # ticket dsnn-dfw.44 part 1: --paired-cost-floor reference maps every
    # plan at or below parity onto reward 0; the archive's coordinate must not.
    from alphagrad.approx.env import log_ratio_quantiles, paired_log_costs
    cand = [65.0e3] * 20
    ref = [100.0e3] * 160
    d = log_ratio_quantiles(cand, ref, 100.0, 0.0)
    assert d["median"] == pytest.approx(np.log(0.65), abs=1e-9)
    assert d["n"] == 20 * 160
    d_lat, _d_mem, _n = paired_log_costs(65.0e3, 1.0, 100.0e3, 1.0)
    assert d_lat == 0.0
    a = _arch()
    assert a.add({"latency": d, "peak_memory": _d((0.0, 0.0, 0.0))["latency"]},
                 [1], 0)
    assert a.pts[0][0] == pytest.approx(np.log(0.65), abs=1e-9)


def test_log_ratio_quantiles_without_windows_is_a_degenerate_band():
    from alphagrad.approx.env import log_ratio_quantiles
    d = log_ratio_quantiles((), (), 1.0, -0.4)
    assert d == {"q05": -0.4, "median": -0.4, "q95": -0.4, "n": 0}


def test_log_ratio_quantiles_spread_covers_both_halves():
    from alphagrad.approx.env import log_ratio_quantiles
    rng = np.random.default_rng(0)
    cand = list(1.0e5 * np.exp(rng.normal(0.0, 0.05, 20)))
    ref = list(1.0e5 * np.exp(rng.normal(0.0, 0.05, 160)))
    d = log_ratio_quantiles(cand, ref, 100.0, 0.0)
    assert d["q05"] < d["median"] < d["q95"]
    assert d["q95"] - d["q05"] > 0.1
