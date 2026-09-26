# dsnn-dfw.281 (owner ruling 2026-09-26, QB): the trainer's front and its dumps rank on four
# objectives: the latency ratio, the logged watermark ratio (slot 5), the gradient cosine (slot 6)
# and the trained memory objective (slot 11). A plan of lower quality stays on the front when it is
# faster or smaller, a plan under tau and a plan of an early episode included. The band archive of
# dsnn-dfw.44 had two objectives and used quality only as the admission floor (dsnn-dfw.107).
from __future__ import annotations

import inspect
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import ppo                                # noqa: E402
from alphagrad.approx.common import checkpoint as ckpt          # noqa: E402
from alphagrad.approx.common.pareto_archive import RatioBandArchive  # noqa: E402
from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX      # noqa: E402

FOUR = ["latency", "peak_memory", "quality", "mem_objective"]
SPREAD = np.linspace(-0.01, 0.01, 20)


def _args():
    return ppo.make_argparser().parse_args(
        ["--cmp-type", "latency", "--mem-type", "peak_memory", "--cost-form",
         "paired-log", "--quality-floor", "0.9", "--pareto-dump-every", "10"])


def _record(order, latency_ratio, watermark_ratio, quality, mem_objective):
    rewards = [0.0] * NUM_REWARDS
    rewards[REWARD_INDEX["quality"]] = quality
    rewards[REWARD_INDEX["mem_objective"]] = mem_objective
    lat = (np.log(latency_ratio) + SPREAD).tolist()
    mem = (np.log(watermark_ratio) + SPREAD).tolist()
    return {"order": list(order), "rewards": rewards,
            "ratio_log": {"latency": {"windows": lat}, "memory": {"windows": mem}}}


# name: (order, episode, latency x, watermark x, quality, slot 11)
PLANS = {
    "exact": ((1, 2, 3), 80, 0.58, 1.00, 1.0, 0.00),
    "quant": ((2, 1, 3), 500, 0.46, 1.08, 0.999998, 0.58),
    "under_tau": ((3, 1, 2), 30, 0.40, 1.10, 0.85, 0.20),
    "early_small": ((1, 3, 2), 0, 1.50, 0.74, 0.93, -1.00),
    "dominated": ((3, 2, 1), 40, 0.90, 1.20, 0.95, -0.10),
}


def _front(args):
    archive = ppo._band_archive(args)
    admitted = {}
    for name in ("early_small", "under_tau", "exact", "dominated", "quant"):
        order, ep, lat, mem, q, m = PLANS[name]
        sample = ppo._band_sample(_record(order, lat, mem, q, m), args)
        admitted[name] = archive.add(sample, [[v, []] for v in order], ep, quality=q)
    return archive, admitted


def _names(archive):
    by_order = {PLANS[n][0]: n for n in PLANS}
    return {by_order[tuple(v for v, _c in s)] for s in archive.seqs}


def test_the_trainer_front_has_four_objectives_and_no_quality_floor():
    archive = ppo._band_archive(_args())
    assert archive.obj_names == FOUR
    assert archive.senses == ["min", "min", "max", "max"]
    assert archive.quality_floor is None


def test_a_lower_quality_point_stays_when_it_is_faster_or_smaller():
    archive, admitted = _front(_args())
    assert admitted["dominated"] is False
    assert _names(archive) == {"exact", "quant", "under_tau", "early_small"}
    q = {n: PLANS[n][4] for n in _names(archive)}
    assert min(q.values()) == 0.85, "a point under tau 0.9 is on the front"
    assert 0 in archive.eps, "a point of episode 0 is on the front"


def test_the_dump_carries_all_four_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _args()
    archive, _admitted = _front(args)
    ppo._dump_pareto(archive, args, 510)
    doc = json.loads((tmp_path / "pareto_front.json").read_text())
    assert doc["objectives"] == FOUR
    assert doc["senses"] == dict(zip(FOUR, ["min", "min", "max", "max"]))
    assert doc["quality_floor"] == 0.9 and doc["num_points"] == 4
    for point in doc["front"]:
        for key in ("obj", "band_lo", "band_hi"):
            assert list(point[key]) == FOUR
            assert all(np.isfinite(v) for v in point[key].values())
        assert point["band_lo"]["quality"] <= point["obj"]["quality"] <= point["band_hi"]["quality"]
    got = sorted((round(p["obj"]["quality"], 6), round(p["obj"]["mem_objective"], 6))
                 for p in doc["front"])
    assert got == [(0.85, 0.2), (0.93, -1.0), (0.999998, 0.58), (1.0, 0.0)]
    lat = {round(p["obj"]["quality"], 6): p["obj"]["latency"] for p in doc["front"]}
    assert lat[0.85] == pytest.approx(np.log(0.40), abs=1e-9)
    best = json.loads((tmp_path / "best_sequences.json").read_text())
    assert all(len(r["obj"]) == 4 for r in best["best_per_channel"].values())
    assert best["best_overall"]["obj"][2] == pytest.approx(0.85)


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


def test_the_senses_survive_the_checkpoint_and_a_mismatch_raises():
    args = _args()
    archive, _admitted = _front(args)
    doc = json.loads(json.dumps(ckpt.pareto_archive_to_json(archive)))
    back = ppo._band_archive(args)
    ckpt.pareto_archive_from_json(back, doc)
    assert back.senses == archive.senses and len(back.pts) == 4
    other = RatioBandArchive(FOUR, senses=["min"] * 4)
    with pytest.raises(ckpt.CheckpointError, match="senses"):
        ckpt.pareto_archive_from_json(other, doc)


def test_the_trainer_feeds_the_front_from_the_plan_record():
    src = inspect.getsource(ppo.main)
    assert "return _band_archive(args)" in src
    assert "_smp = _band_sample(_r, args)" in src
    rec = _record((1, 2, 3), 0.5, 1.0, 0.97, 0.4)
    sample = ppo._band_sample(rec, _args())
    assert sample["quality"] == [0.97] and sample["mem_objective"] == [0.4]
    assert ppo._band_sample({"order": [1], "ratio_log": None}, _args()) is None
