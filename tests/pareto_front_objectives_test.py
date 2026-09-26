# dsnn-dfw.281 (owner rulings 2026-09-26, QB and Q8): the trainer's front and its dumps rank on
# three objectives: quality (the gradient cosine, slot 6), latency and peak memory. Peak memory is
# the device watermark when one was measured, else the static estimate temp + args + out, and each
# point records its source. A plan of lower quality stays on the front when it is faster or
# smaller, a plan under tau and a plan of an early episode included. The band archive of
# dsnn-dfw.44 had two objectives and used quality only as the admission floor (dsnn-dfw.107). Since
# dsnn-dfw.293 the latency of a point is banded by its quantiles (quantile_front_test.py).
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
    QuantileFrontArchive, RatioBandArchive)
from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX      # noqa: E402

THREE = ["latency", "peak_memory", "quality"]
SPREAD = np.linspace(-0.01, 0.01, 20)
REF_WATERMARK = 1.6e6
REF_LATENCY_NS = 1.8e5


def _args():
    return ppo.make_argparser().parse_args(
        ["--cmp-type", "latency", "--mem-type", "peak_memory", "--cost-form",
         "paired-log", "--quality-floor", "0.9", "--pareto-dump-every", "10"])


def _rewards(quality):
    rewards = [0.0] * NUM_REWARDS
    rewards[REWARD_INDEX["quality"]] = quality
    return rewards


def _measured(order, latency_ratio, watermark_ratio, quality):
    return {"order": list(order), "rewards": _rewards(quality),
            "mem_channel": "watermark", "mem_peak_source": "runtime_delta",
            "mem_watermark_bytes": watermark_ratio * REF_WATERMARK,
            "ref_watermark_bytes": REF_WATERMARK,
            "ratio_log": {"latency": {"windows": (np.log(latency_ratio) + SPREAD).tolist()},
                          "memory": {"windows": (np.log(watermark_ratio) + SPREAD).tolist()}}}


def _refused(order, temp, args, out):
    # A plan refused before it ran: no watermark, its static bytes, the deadline as its latency.
    return {"order": list(order), "rewards": _rewards(0.0), "refused": "gate",
            "mem_channel": "watermark", "mem_peak_source": "not_measured",
            "mem_watermark_bytes": None, "ref_watermark_bytes": REF_WATERMARK,
            "mem_temp_bytes": temp, "mem_args_bytes": args, "mem_output_bytes": out,
            "refusal_latency_ns": 300e9, "ref_latency_ns": REF_LATENCY_NS,
            "ratio_log": None}


# name: (order, episode, latency x, watermark x, quality)
PLANS = {
    "exact": ((1, 2, 3), 80, 0.58, 1.00, 1.0),
    "quant": ((2, 1, 3), 500, 0.46, 1.08, 0.999998),
    "under_tau": ((3, 1, 2), 30, 0.40, 1.10, 0.85),
    "early_small": ((1, 3, 2), 0, 1.50, 0.74, 0.93),
    "dominated": ((3, 2, 1), 40, 0.90, 1.20, 0.95),
}
STATIC = ((2, 3, 1), 12, (0.2e6, 0.4e6, 0.2e6))


def _front(args):
    archive = ppo._band_archive(args)
    admitted = {}
    for name in ("early_small", "under_tau", "exact", "dominated", "quant"):
        order, ep, lat, mem, q = PLANS[name]
        sample, source = ppo._band_sample(_measured(order, lat, mem, q), args)
        admitted[name] = archive.add(sample, [[v, []] for v in order], ep,
                                     quality=q, mem_source=source)
    order, ep, (temp, a, out) = STATIC
    sample, source = ppo._band_sample(_refused(order, temp, a, out), args)
    admitted["static"] = archive.add(sample, [[v, []] for v in order], ep,
                                     quality=0.0, mem_source=source)
    return archive, admitted


def _names(archive):
    by_order = {PLANS[n][0]: n for n in PLANS}
    by_order[STATIC[0]] = "static"
    return {by_order[tuple(v for v, _c in s)] for s in archive.seqs}


def test_the_trainer_front_has_three_objectives_and_no_quality_floor():
    archive = ppo._band_archive(_args())
    assert archive.obj_names == THREE
    assert archive.senses == ["min", "min", "max"]
    assert archive.quality_floor is None


def test_a_lower_quality_point_stays_when_it_is_faster_or_smaller():
    archive, admitted = _front(_args())
    assert admitted["dominated"] is False
    assert _names(archive) == {"exact", "quant", "under_tau", "early_small", "static"}
    q = {n: PLANS[n][4] for n in _names(archive) if n != "static"}
    assert min(q.values()) == 0.85, "a point under tau 0.9 is on the front"
    assert 0 in archive.eps, "a point of episode 0 is on the front"


def test_a_point_without_a_watermark_enters_with_its_static_bytes_and_its_source():
    archive, admitted = _front(_args())
    assert admitted["static"] is True
    i = [k for k, s in enumerate(archive.seqs)
         if tuple(v for v, _c in s) == STATIC[0]][0]
    total = sum(STATIC[2])
    assert archive.pts[i][1] == pytest.approx(math.log(total / REF_WATERMARK), abs=1e-12)
    assert archive.pts[i][0] == pytest.approx(math.log(300e9 / REF_LATENCY_NS), abs=1e-12)
    assert archive.pts[i][2] == 0.0
    assert archive.mem_sources[i] == {"static": 1}
    others = [archive.mem_sources[k] for k in range(len(archive.pts)) if k != i]
    assert others and all(m == {"watermark": 1} for m in others)


def test_the_static_fallback_of_a_measured_plan_is_the_static_total():
    args = _args()
    rec = _measured((1, 2, 3), 0.6, 1.0, 0.99)
    rec.update(mem_peak_source="static_fallback", mem_watermark_bytes=None,
               mem_temp_bytes=1.0e6, mem_args_bytes=0.5e6, mem_output_bytes=0.1e6)
    sample, source = ppo._band_sample(rec, args)
    assert source == "static"
    assert sample["peak_memory"] == [pytest.approx(math.log(1.6e6 / REF_WATERMARK))]
    assert len(sample["latency"]) == 20
    assert ppo._band_sample({"order": [1], "ratio_log": None}, args) == (None, None)


def test_the_dump_carries_three_values_and_the_memory_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _args()
    archive, _admitted = _front(args)
    ppo._dump_pareto(archive, args, 510)
    doc = json.loads((tmp_path / "pareto_front.json").read_text())
    assert doc["objectives"] == THREE
    assert doc["senses"] == dict(zip(THREE, ["min", "min", "max"]))
    assert doc["quality_floor"] == 0.9 and doc["num_points"] == 5
    for point in doc["front"]:
        for key in ("obj", "band_lo", "band_hi"):
            assert list(point[key]) == THREE
            assert all(np.isfinite(v) for v in point[key].values())
        assert point["mem_source"] in ({"watermark": 1}, {"static": 1})
        assert point["quantiles"] == {"latency": None}, "no quantile fields: the median alone"
    got = sorted((round(p["obj"]["quality"], 6), list(p["mem_source"])[0])
                 for p in doc["front"])
    assert got == [(0.0, "static"), (0.85, "watermark"), (0.93, "watermark"),
                   (0.999998, "watermark"), (1.0, "watermark")]
    best = json.loads((tmp_path / "best_sequences.json").read_text())
    assert all(len(r["obj"]) == 3 for r in best["best_per_channel"].values())


def test_a_max_objective_prefers_the_higher_reading():
    a = RatioBandArchive(["latency", "quality"], senses=["min", "max"])
    assert a.add({"latency": SPREAD, "quality": [0.9]}, ["low"], 0)
    assert a.add({"latency": SPREAD, "quality": [0.99]}, ["high"], 1)
    assert a.seqs == [["high"]], "the same latency with a higher quality dominates"
    assert a.add({"latency": SPREAD, "quality": [0.95]}, ["lower"], 2) is False
    assert a.add({"latency": SPREAD - 1.0, "quality": [0.5]}, ["fast"], 3)
    assert sorted(map(tuple, a.seqs)) == [("fast",), ("high",)]
    assert np.isfinite(a.hypervolume()) and a.hypervolume() > 0.0
    with pytest.raises(ValueError, match="senses"):
        RatioBandArchive(["latency", "quality"], senses=["min", "up"])


def test_the_senses_and_sources_survive_the_checkpoint_and_a_mismatch_raises():
    args = _args()
    archive, _admitted = _front(args)
    doc = json.loads(json.dumps(ckpt.pareto_archive_to_json(archive)))
    back = ppo._band_archive(args)
    ckpt.pareto_archive_from_json(back, doc)
    assert back.senses == archive.senses and len(back.pts) == 5
    assert back.mem_sources == archive.mem_sources
    other = QuantileFrontArchive(THREE, banded=["latency"], senses=["min"] * 3)
    with pytest.raises(ckpt.CheckpointError, match="senses"):
        ckpt.pareto_archive_from_json(other, doc)


def test_the_trainer_feeds_the_front_from_the_plan_record():
    src = inspect.getsource(ppo.main)
    assert "return _band_archive(args)" in src
    assert "_smp, _msrc = _band_sample(_r, args)" in src
    assert "mem_source=_d[1]" in src
    sample, source = ppo._band_sample(_measured((1, 2, 3), 0.5, 1.0, 0.97), _args())
    assert source == "watermark" and sample["quality"] == [0.97]
    assert sorted(sample) == sorted(THREE)
