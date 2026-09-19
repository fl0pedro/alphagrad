"""The distribution-aware archive rule (ticket dsnn-dfw.44, second design).

The band is a 90 percent distribution-free interval for the MEDIAN of the
plan's per-window paired log ratios, there is no drift floor, and a merged
measurement POOLS its windows into the point it landed in.
"""
import json

import numpy as np
import pytest

from alphagrad.approx.common.pareto_archive import (
    RatioBandArchive, _median_order_stat_lo, median_band)

OBJ = ("latency", "peak_memory")
FLAT = np.linspace(-1.0, 1.0, 20)          # median 0, band x_(6)..x_(15)


def _d(lat, mem=(0.0,)):
    return {"latency": np.asarray(lat, dtype=np.float64),
            "peak_memory": np.asarray(mem, dtype=np.float64)}


def _arch(**kw):
    return RatioBandArchive(OBJ, **kw)


# ---- the interval itself ------------------------------------------------

def test_order_statistic_is_the_binomial_one():
    # 20 windows: P(Bin(20,1/2) <= 5) = 0.0207 <= 0.05, P(<= 6) = 0.0577 > 0.05
    assert _median_order_stat_lo(20) == 6
    assert _median_order_stat_lo(8) == 2
    assert _median_order_stat_lo(6) == 1
    # too few windows to exclude any order statistic
    assert _median_order_stat_lo(4) == 0
    assert _median_order_stat_lo(1) == 0
    with pytest.raises(ValueError):
        _median_order_stat_lo(0)


def test_median_band_is_the_order_statistic_pair():
    a = np.arange(20, dtype=np.float64)
    med, lo, hi = median_band(a)
    assert med == pytest.approx(9.5)
    assert lo == pytest.approx(5.0)          # x_(6), 1-based
    assert hi == pytest.approx(14.0)         # x_(15)
    with pytest.raises(ValueError):
        median_band([])


def test_a_short_sample_gets_the_whole_range():
    med, lo, hi = median_band([3.0, 1.0, 2.0])
    assert (med, lo, hi) == (2.0, 1.0, 3.0)


def test_the_band_narrows_as_the_sample_grows():
    m1, l1, h1 = median_band(FLAT)
    m2, l2, h2 = median_band(np.concatenate([FLAT, FLAT]))
    assert (h2 - l2) < (h1 - l1)


# ---- the archive rule ---------------------------------------------------

def test_median_beyond_the_band_is_dominated():
    a = _arch()
    assert a.add(_d(FLAT - 5.0), [1], 0)
    hi = a.hi[0][0]
    assert a.add(_d(FLAT + (hi + 1.0)), [2], 1) is False
    assert len(a.pts) == 1


def test_median_below_the_band_removes_the_point():
    a = _arch()
    assert a.add(_d(FLAT), [1], 0)
    lo = a.lo[0][0]
    assert a.add(_d(FLAT + (lo - 1.0)), [2], 1)
    assert len(a.pts) == 1
    assert a.seqs == [[2]]


def test_inside_the_band_pools_the_windows_and_narrows_it():
    a = _arch()
    assert a.add(_d(FLAT), [1], 0)
    w0 = a.band_width(0)
    assert a.add(_d(FLAT * 0.5), [2], 1) is False      # median 0, inside
    assert a.counts == [2]
    assert a.n_merged == 1
    assert a.seqs == [[1]]
    assert a.samples[0][0].size == 40
    assert a.band_width(0) < w0


def test_pooling_moves_the_point_to_the_pooled_median():
    a = _arch()
    a.add(_d(FLAT), [1], 0)
    assert a.pts[0][0] == pytest.approx(0.0)
    a.add(_d(FLAT + 0.2), [2], 1)
    assert a.pts[0][0] == pytest.approx(0.1, abs=0.06)


def test_the_pooled_sample_is_capped():
    a = _arch(pool_cap=25)
    a.add(_d(FLAT), [1], 0)
    a.add(_d(FLAT), [2], 1)
    assert a.samples[0][0].size == 25
    a.add(_d(FLAT), [3], 2)
    assert a.samples[0][0].size == 25
    assert a.counts == [3]


def test_there_is_no_drift_floor_any_more():
    a = _arch()
    assert not hasattr(a, "drift_floor")
    assert not hasattr(a, "set_drift_floor")


def test_non_dominated_pair_is_kept():
    a = _arch()
    assert a.add(_d(FLAT - 5.0, FLAT + 5.0), [1], 0)
    assert a.add(_d(FLAT + 5.0, FLAT - 5.0), [2], 1)
    assert len(a.pts) == 2


def test_cap_drops_the_widest_band():
    a = _arch(cap=3)
    tight = FLAT * 0.01
    assert a.add(_d(tight - 3.0, tight + 3.0), ["a"], 0)
    assert a.add(_d(FLAT * 3.0 - 1.0, FLAT * 3.0 + 0.5), ["b"], 0)
    assert a.add(_d(tight + 3.0, tight - 3.0), ["c"], 0)
    assert len(a.pts) == 3
    widest = int(np.argmax([a.band_width(i) for i in range(3)]))
    assert a.seqs[widest] == ["b"]
    assert a.add(_d(tight - 2.5, tight + 2.0), ["d"], 1)
    assert len(a.pts) == 3
    assert a.n_dropped_cap == 1
    assert ["b"] not in a.seqs and ["d"] in a.seqs


def test_admitted_count_is_the_auto_stop_contract():
    a = _arch()
    sols = [(_d(FLAT), [1], 1.0),
            (_d(FLAT * 0.5), [2], 1.0),
            (_d(FLAT + 5.0, FLAT - 5.0), [3], 1.0)]
    assert a.add_many(sols, 0) == 2
    assert a.n_merged == 1


def test_quality_floor_refuses_an_infeasible_plan():
    a = _arch(quality_floor=0.9)
    assert a.add(_d(FLAT - 9.0), [1], 0, quality=0.80) is False
    assert a.add(_d(FLAT), [2], 0, quality=0.90)
    assert len(a.pts) == 1


def test_missing_or_broken_windows_raise():
    a = _arch()
    with pytest.raises(ValueError):
        a.add(None, [1], 0)
    with pytest.raises(ValueError):
        a.add({"latency": np.array([]), "peak_memory": np.array([0.0])}, [1], 0)
    with pytest.raises(ValueError):
        a.add(_d([0.0, float("nan"), 1.0]), [1], 0)
    with pytest.raises(ValueError):
        _arch(cap=0)
    with pytest.raises(ValueError):
        _arch(pool_cap=0)


def test_hypervolume_and_size_stay_meaningful():
    a = _arch()
    assert a.hypervolume() == 0.0
    a.add(_d(FLAT), [1], 0)
    hv1 = a.hypervolume()
    a.add(_d(FLAT - 4.0), [2], 1)
    hv2 = a.hypervolume()
    assert np.isfinite(hv1) and np.isfinite(hv2)
    assert hv2 > hv1
    assert len(a.pts) == 1


def test_dump_front_writes_the_bands(tmp_path):
    a = _arch()
    a.add(_d(FLAT), [[1, []], [2, []]], 7)
    p = tmp_path / "pareto_front.json"
    a.dump_front(str(p), extra={"episode": 7})
    doc = json.loads(p.read_text())
    assert doc["objectives"] == list(OBJ)
    assert doc["pool_cap"] == 512
    assert "median" in doc["band"]
    row = doc["front"][0]
    for k in ("obj", "band_lo", "band_hi"):
        assert set(row[k]) == set(OBJ)
    assert row["band_lo"]["latency"] < row["obj"]["latency"]
    assert row["band_hi"]["latency"] > row["obj"]["latency"]
    assert row["n"] == 1 and row["episode"] == 7
    assert row["windows"]["latency"] == 20


# ---- the member plans of a point ---------------------------------------

def test_a_merge_keeps_both_plans_as_members():
    a = _arch()
    a.add(_d(FLAT), ["A"], 0)
    a.add(_d(FLAT * 0.5), ["B"], 3)
    assert a.counts == [2]
    assert [m["seq"] for m in a.members[0]] == [["A"], ["B"]]
    assert [m["windows"] for m in a.members[0]] == [20, 20]
    assert [m["first_episode"] for m in a.members[0]] == [0, 3]


def test_the_representative_is_the_member_with_most_windows():
    a = _arch()
    a.add(_d(FLAT), ["A"], 0)
    a.add(_d(FLAT * 0.5), ["B"], 1)
    # a tie goes to the earlier member, so A still represents the point
    assert a.seqs[0] == ["A"]
    a.add(_d(FLAT * 0.5), ["B"], 2)
    # B now carries 40 of the 60 pooled windows and takes over
    assert a.seqs[0] == ["B"]
    b = [m for m in a.members[0] if m["seq"] == ["B"]][0]
    assert b["windows"] == 40 and b["n"] == 2 and b["last_episode"] == 2


def test_a_member_carries_the_face_actions_not_only_the_order():
    a = _arch()
    plan = {"seq": [[1, ["diag(0)"]], [2, []]],
            "faces": [{"k": 0, "f": [1], "rows": [[[3, 0]]], "skips": [0]}]}
    a.add(_d(FLAT), plan, 0)
    a.add(_d(FLAT * 0.5), ["other"], 1)
    assert a.members[0][0]["seq"] == plan
    assert a.members[0][0]["seq"]["faces"][0]["rows"] == [[[3, 0]]]
    assert a.front()[0]["members"][0]["seq"] == plan


def test_members_ride_the_front_and_the_dump(tmp_path):
    a = _arch()
    a.add(_d(FLAT), ["A"], 0)
    a.add(_d(FLAT * 0.5), ["B"], 1)
    row = a.front()[0]
    assert row["num_members"] == 2
    assert [m["seq"] for m in row["members"]] == [["A"], ["B"]]
    p = tmp_path / "f.json"
    a.dump_front(str(p))
    doc = json.loads(p.read_text())
    assert doc["front"][0]["num_members"] == 2
    assert doc["front"][0]["members"][1]["seq"] == ["B"]


def test_the_member_list_is_bounded_by_the_window_budget():
    a = _arch(pool_cap=30)
    a.add(_d(FLAT), ["A"], 0)          # 20 windows pooled
    a.add(_d(FLAT * 0.5), ["B"], 1)    # 10 more, the pool is now full
    assert a.samples[0][0].size == 30
    assert [m["windows"] for m in a.members[0]] == [20, 10]
    a.add(_d(FLAT * 0.5), ["C"], 2)    # contributes nothing, not a member
    assert [m["seq"] for m in a.members[0]] == [["A"], ["B"]]
    assert a.counts == [3]
    a.add(_d(FLAT * 0.5), ["B"], 3)    # already a member, still counted
    assert [m["n"] for m in a.members[0]] == [1, 2]


def test_members_survive_the_checkpoint_round_trip():
    from alphagrad.approx.common import checkpoint as ckpt
    a = _arch()
    a.add(_d(FLAT), [[1, ["diag(0)"]]], 0)
    a.add(_d(FLAT * 0.5), [[2, []]], 1)
    a.add(_d(FLAT * 0.5), [[2, []]], 2)
    doc = json.loads(json.dumps(ckpt.pareto_archive_to_json(a)))
    b = _arch()
    ckpt.pareto_archive_from_json(b, doc)
    assert b.counts == a.counts
    assert [m["seq"] for m in b.members[0]] == [[[1, ["diag(0)"]]], [[2, []]]]
    assert [m["windows"] for m in b.members[0]] == [20, 40]
    assert b.seqs == a.seqs == [[[2, []]]]
    # a restored archive keeps merging into the members it was given
    b.add(_d(FLAT * 0.5), [[2, []]], 3)
    assert [m["n"] for m in b.members[0]] == [1, 3]


def test_a_dropped_point_takes_its_members_with_it():
    a = _arch(cap=1)
    a.add(_d(FLAT), ["A"], 0)
    a.add(_d(FLAT - 9.0), ["B"], 1)
    assert len(a.members) == 1
    assert [m["seq"] for m in a.members[0]] == [["B"]]


# ---- the env side -------------------------------------------------------

def test_the_pair_partner_is_the_reference_median():
    from alphagrad.approx.env import paired_window_log_ratios
    # mean 3.25e5, median 1e5: only the median puts the ratio at 0
    ref = [1.0e5] * 120 + [1.0e6] * 40
    w = paired_window_log_ratios([1.0e5] * 20, ref, 100.0, 0.0)
    assert w.size == 20
    assert np.allclose(w, 0.0)


def test_a_channel_without_windows_is_one_reading():
    from alphagrad.approx.env import paired_window_log_ratios
    w = paired_window_log_ratios((), (), 1.0, -0.4)
    assert w.tolist() == [-0.4]


def test_window_record_carries_the_ratios_and_the_band():
    from alphagrad.approx.env import (paired_window_log_ratios,
                                      window_ratio_record)
    w = paired_window_log_ratios([1.0e5] * 20, [1.0e5] * 160, 100.0, 0.0)
    rec = window_ratio_record(w)
    assert rec["n"] == 20 and len(rec["windows"]) == 20
    assert rec["lo"] <= rec["median"] <= rec["hi"]
    assert median_band(rec["windows"])[0] == pytest.approx(rec["median"])


def test_point_keeps_the_half_of_the_axis_the_reward_floors_away():
    # ticket dsnn-dfw.44 part 1: --paired-cost-floor reference maps every plan
    # at or below parity onto reward 0; the archive's coordinate must not.
    from alphagrad.approx.env import paired_window_log_ratios, paired_log_costs
    w = paired_window_log_ratios([65.0e3] * 20, [100.0e3] * 160, 100.0, 0.0)
    assert float(np.median(w)) == pytest.approx(np.log(0.65), abs=1e-9)
    d_lat, _d_mem, _n = paired_log_costs(65.0e3, 1.0, 100.0e3, 1.0)
    assert d_lat == 0.0
    a = _arch()
    assert a.add({"latency": w, "peak_memory": np.array([0.0])}, [1], 0)
    assert a.pts[0][0] == pytest.approx(np.log(0.65), abs=1e-9)
