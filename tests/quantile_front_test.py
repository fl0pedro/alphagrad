# dsnn-dfw.293 (owner ruling 2026-09-26, Q12 i and a): the front reads the latency quantiles of each
# plan. Point A leaves the front only when a point B is at least as good on quality and peak memory
# and B's latency median lies below A's p10. A median inside the other point's p10 to p90 band keeps
# both. The front has no cap. A record without quantiles falls back to its median alone. Each point
# carries its five quantiles in the dumps. The band archive before it merged the plans inside one
# 90 percent median band into one point and held 64 points at most.
from __future__ import annotations

import inspect
import json
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import ppo                                # noqa: E402
from alphagrad.approx.common import checkpoint as ckpt          # noqa: E402
from alphagrad.approx.common.pareto_archive import (            # noqa: E402
    QUANTILE_KEYS, QuantileFrontArchive, RatioBandArchive)
from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX      # noqa: E402

THREE = ["latency", "peak_memory", "quality"]
REF_P50_NS = 2.0e5
REF_WATERMARK = 1.6e6


def _front():
    return QuantileFrontArchive(THREE, banded=["latency"],
                                senses=["min", "min", "max"])


def _q(p10, p25, p50, p75, p90):
    return {"latency": dict(zip(QUANTILE_KEYS, (p10, p25, p50, p75, p90)))}


def _dist(med, mem=0.0, quality=1.0):
    return {"latency": [med], "peak_memory": [mem], "quality": [quality]}


# A: the better median, and at least as good on memory and quality.
A = (_q(-0.30, -0.25, -0.20, -0.15, -0.10), 0.0, 1.0)
# B: its median lies inside A's band, and A's median lies inside B's band.
B_OVERLAP = (_q(-0.26, -0.22, -0.18, -0.12, -0.05), 0.0, 1.0)
# C: its band lies wholly above A's median.
C_ABOVE = (_q(-0.10, -0.05, 0.00, 0.05, 0.10), 0.0, 1.0)


def _add(front, name, point, ep=0):
    q, mem, quality = point
    return front.add(_dist(q["latency"]["p50"], mem, quality), [name], ep,
                     quantiles=q)


def _names(front):
    return sorted(s[0] for s in front.seqs)


@pytest.mark.parametrize("first", ["a", "b"])
def test_two_points_with_overlapping_bands_both_stay(first):
    front = _front()
    order = [("a", A), ("b", B_OVERLAP)]
    if first == "b":
        order.reverse()
    assert [_add(front, n, p) for n, p in order] == [True, True]
    assert _names(front) == ["a", "b"]


@pytest.mark.parametrize("first", ["a", "c"])
def test_a_point_whose_band_lies_wholly_above_another_median_leaves(first):
    front = _front()
    if first == "a":
        assert _add(front, "a", A) is True
        assert _add(front, "c", C_ABOVE) is False, "c is refused on arrival"
    else:
        assert _add(front, "c", C_ABOVE) is True
        assert _add(front, "a", A) is True
    assert _names(front) == ["a"]


@pytest.mark.parametrize("mem, quality", [(-0.1, 1.0), (0.0, 1.0 + 1e-6)])
def test_a_point_better_on_memory_or_quality_stays_above_the_median(mem, quality):
    front = _front()
    assert _add(front, "a", A) is True
    assert _add(front, "c", (C_ABOVE[0], mem, quality)) is True
    assert _names(front) == ["a", "c"]


def test_the_boundary_a_median_on_the_p10_keeps_the_point():
    front = _front()
    assert _add(front, "a", A) is True
    at_p10 = _q(-0.20, -0.10, 0.00, 0.10, 0.20)
    assert _add(front, "d", (at_p10, 0.0, 1.0)) is True, "-0.20 is not below -0.20"
    assert _names(front) == ["a", "d"]


def test_the_front_has_no_cap(tmp_path):
    front = _front()
    n = 300
    for i in range(n):
        # A trade-off: each point is faster and larger than the one before.
        med = -0.001 * i
        assert _add(front, f"p{i}", (_q(med - 1e-4, med, med, med, med + 1e-4),
                                     0.001 * i, 1.0), ep=i)
    assert len(front.pts) == n and front.cap is None
    front.dump_front(str(tmp_path / "front.json"))
    doc = json.loads((tmp_path / "front.json").read_text())
    assert doc["cap"] is None and doc["num_points"] == n
    assert doc["rule"] == "quantile" and doc["banded"] == ["latency"]
    assert doc["dropped_at_cap"] == 0 and doc["merged_measurements"] == 0


def test_a_record_without_quantiles_falls_back_to_its_median():
    front = _front()
    assert front.add(_dist(-0.20), ["m"], 0) is True
    assert front.quantiles == [{"latency": None}]
    assert front.lo[0][0] == front.hi[0][0] == pytest.approx(-0.20)
    # Its band is its median: a point with quantiles whose median lies below it takes it off.
    assert _add(front, "q", (_q(-0.40, -0.30, -0.25, -0.22, -0.21), 0.0, 1.0)) is True
    assert _names(front) == ["q"]
    # A lower median inside the band of a point with quantiles keeps both: -0.30 is not below -0.40.
    assert front.add(_dist(-0.30), ["m2"], 1) is True
    assert _names(front) == ["m2", "q"]


def test_a_plan_measured_again_is_counted_on_its_point():
    front = _front()
    assert _add(front, "a", A, ep=3) is True
    faster = (_q(-0.9, -0.8, -0.7, -0.6, -0.5), 0.0, 1.0)
    assert _add(front, "a", faster, ep=7) is False
    assert front.counts == [2] and front.n_repeats == 1
    assert front.members[0][0]["n"] == 2 and front.members[0][0]["last_episode"] == 7
    assert front.pts[0][0] == pytest.approx(-0.20), "the point keeps its first measurement"


def test_quantiles_out_of_order_or_incomplete_raise():
    front = _front()
    with pytest.raises(ValueError, match="non-decreasing"):
        _add(front, "x", (_q(-0.1, -0.2, -0.3, -0.4, -0.5), 0.0, 1.0))
    with pytest.raises(ValueError, match="five quantiles"):
        front.add(_dist(-0.2), ["y"], 0, quantiles={"latency": {"p50": -0.2}})
    with pytest.raises(ValueError, match="banded"):
        QuantileFrontArchive(THREE, banded=["flops"])


def _rewards(quality):
    r = [0.0] * NUM_REWARDS
    r[REWARD_INDEX["quality"]] = quality
    return r


def _record(order, lat_x, quality, spread=(0.8, 0.9, 1.0, 1.1, 1.3)):
    # A plan record as d65dcda writes it: the five candidate quantiles and the reference's p50.
    rec = {"order": list(order), "rewards": _rewards(quality),
           "mem_channel": "watermark", "mem_peak_source": "runtime_delta",
           "mem_watermark_bytes": REF_WATERMARK, "ref_watermark_bytes": REF_WATERMARK,
           "ref_latency_p50_ns": REF_P50_NS,
           "ratio_log": {"latency": {"windows": [math.log(lat_x)] * 5},
                         "memory": {"windows": [0.0] * 5}}}
    for p, s in zip((10, 25, 50, 75, 90), spread):
        rec[f"candidate_latency_p{p}_ns"] = lat_x * s * REF_P50_NS
    return rec


def _args():
    return ppo.make_argparser().parse_args(
        ["--cmp-type", "latency", "--mem-type", "peak_memory", "--cost-form",
         "paired-log", "--pareto-dump-every", "10"])


def _admit(archive, rec, args, ep):
    sample, source = ppo._band_sample(rec, args)
    quantiles, detail = ppo._latency_quantiles(rec, args)
    return archive.add(sample, [[v, []] for v in rec["order"]], ep,
                       mem_source=source, quantiles=quantiles, detail=detail)


def test_the_trainer_reads_the_quantile_fields_of_the_record():
    args = _args()
    rec = _record((1, 2, 3), 0.5, 1.0)
    quantiles, detail = ppo._latency_quantiles(rec, args)
    want = {f"p{p}": math.log(0.5 * s)
            for p, s in zip((10, 25, 50, 75, 90), (0.8, 0.9, 1.0, 1.1, 1.3))}
    assert quantiles["latency"] == pytest.approx(want, abs=1e-12)
    assert detail["ref_latency_p50_ns"] == REF_P50_NS
    assert detail["candidate_latency_ns"]["p90"] == pytest.approx(0.5 * 1.3 * REF_P50_NS)
    # A refused plan and an older record have no quantiles: the median alone.
    older = dict(rec)
    for p in (10, 25, 50, 75, 90):
        older[f"candidate_latency_p{p}_ns"] = None
    assert ppo._latency_quantiles(older, args) == (None, None)
    assert ppo._latency_quantiles({"order": [1]}, args) == (None, None)


def test_each_front_point_carries_its_five_quantiles_in_the_dumps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _args()
    archive = ppo._band_archive(args)
    assert isinstance(archive, QuantileFrontArchive) and archive.cap is None
    assert archive.banded == ["latency"] and archive.obj_names == THREE
    # b's median 0.52 lies inside a's band 0.40 to 0.65, and a's inside b's 0.42 to 0.68.
    assert _admit(archive, _record((1, 2, 3), 0.50, 1.0), args, 4)
    assert _admit(archive, _record((2, 1, 3), 0.52, 1.0), args, 5)
    # c's band starts at 0.8 x 0.9 = 0.72, above a's median 0.50: it leaves on arrival.
    assert not _admit(archive, _record((3, 2, 1), 0.90, 1.0), args, 6)
    ppo._dump_pareto(archive, args, 6)
    doc = json.loads((tmp_path / "pareto_front.json").read_text())
    assert doc["rule"] == "quantile" and doc["cap"] is None and doc["num_points"] == 2
    for point in doc["front"]:
        q = point["quantiles"]["latency"]
        assert list(q) == list(QUANTILE_KEYS)
        assert point["obj"]["latency"] == pytest.approx(q["p50"])
        assert (point["band_lo"]["latency"], point["band_hi"]["latency"]) == (
            pytest.approx(q["p10"]), pytest.approx(q["p90"]))
        assert list(point["detail"]["candidate_latency_ns"]) == list(QUANTILE_KEYS)
    best = json.loads((tmp_path / "best_sequences.json").read_text())
    for row in best["best_per_channel"].values():
        assert list(row["quantiles"]["latency"]) == list(QUANTILE_KEYS)


def test_the_quantile_front_survives_the_checkpoint_and_refuses_another_kind():
    args = _args()
    archive = ppo._band_archive(args)
    _admit(archive, _record((1, 2, 3), 0.50, 1.0), args, 4)
    _admit(archive, _record((2, 1, 3), 0.52, 1.0), args, 5)
    _admit(archive, _record((1, 2, 3), 0.40, 1.0), args, 9)
    doc = json.loads(json.dumps(ckpt.pareto_archive_to_json(archive)))
    assert doc["kind"] == "quantile-front" and doc["cap"] is None
    back = ppo._band_archive(args)
    ckpt.pareto_archive_from_json(back, doc)
    assert back.quantiles == archive.quantiles and back.details == archive.details
    assert back.n_repeats == archive.n_repeats == 1 and back.cap is None
    assert [list(p) for p in back.pts] == [list(p) for p in archive.pts]
    assert _admit(back, _record((3, 1, 2), 0.51, 1.0), args, 10), "the restored front admits"
    with pytest.raises(ckpt.CheckpointError, match="ratio-band"):
        ckpt.pareto_archive_from_json(
            RatioBandArchive(THREE, senses=["min", "min", "max"]), doc)
    other = QuantileFrontArchive(THREE, banded=["peak_memory"],
                                 senses=["min", "min", "max"])
    with pytest.raises(ckpt.CheckpointError, match="banded"):
        ckpt.pareto_archive_from_json(other, doc)


def test_the_trainer_feeds_the_record_quantiles_to_the_front():
    src = inspect.getsource(ppo.main)
    assert "+ _latency_quantiles(_r, args))" in src
    assert "quantiles=_d[2]" in src and "detail=_d[3]" in src
    assert 'log_dict["pareto/repeats"]' in src
