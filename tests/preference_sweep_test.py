"""THE INFERENCE-TIME PREFERENCE SWEEP: the loader, the pin, the front writer.

Three contracts, one per group (ticket dsnn-dfw.86):

* THE LOADER. The argument namespace rebuilt from a checkpoint's `meta.json`
  writes back that namespace exactly -- the round trip is what makes the load
  a READ of that run rather than this build's defaults wearing its name. A
  namespace this build cannot account for raises, and an override outside the
  sweep's own surface raises.
* THE PIN. The grid is the latency-memory edge, both corners included, and a
  weight that asks for more plans than an episode measures raises rather than
  rounding. The vector the policy reads is the one the trainer's own
  `_lag_preferences` makes of the pinned one, so the pin is checked THROUGH
  that function: the quality coordinate is lambda and the two cost
  coordinates are the pin, renormalised.
* THE FRONT WRITER. The file the sweep dumps is a `pareto_front.json` of the
  same archive class the trainer dumps, with the bands in log-ratio space,
  and the comparison reads both fronts under ONE recorded nadir.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from alphagrad.approx.common import preference_sweep as psweep
from alphagrad.approx.common.checkpoint import CheckpointError, args_to_json


# ---------------------------------------------------------------------------
# A parser and a saved namespace, standing in for ppo.make_argparser and a
# run's meta.json. The real pair is exercised by the round-trip test below
# against ppo.make_argparser itself.
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="run")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--auto-stop", action="store_true")
    p.add_argument("--grad-oracle", choices=["reference", "off"],
                   default="reference")
    p.add_argument("--wandb", type=str, default="offline")
    p.add_argument("--rewards", nargs="+", default=["cmp", "mem", "acc"])
    p.add_argument("--preference-conditioned", action="store_true")
    p.add_argument("--preference-sweep-checkpoint", type=str, default="")
    p.add_argument("--preference-sweep-weights", type=str, default="edge:11")
    p.add_argument("--preference-sweep-plans", type=int, default=16)
    p.add_argument("--preference-sweep-out", type=str, default="")
    return p


def _saved(**over) -> dict:
    base = {"name": "condC", "seed": 250201, "episodes": 2000,
            "num_envs": 16, "checkpoint_every": 50, "resume": "",
            "auto_stop": False, "grad_oracle": "reference",
            "wandb": "online", "rewards": ["cmp", "mem", "acc"],
            "preference_conditioned": True}
    base.update(over)
    return {"args": base, "episode": 2000, "format": 1}


# ---------------------------------------------------------------------------
# THE LOADER.
# ---------------------------------------------------------------------------
def test_loader_round_trips_the_saved_namespace():
    args = psweep.load_sweep_args(_saved(), _parser())
    got = args_to_json(args)
    for name in psweep.SWEEP_ONLY_ARGS:
        got.pop(name)
    assert got == _saved()["args"]
    assert psweep.sweep_args_round_trip(_saved(), _parser())


def test_loader_keeps_every_saved_value_not_only_the_non_defaults():
    args = psweep.load_sweep_args(_saved(), _parser())
    assert args.seed == 250201
    assert args.episodes == 2000
    assert args.preference_conditioned is True
    assert args.rewards == ["cmp", "mem", "acc"]


def test_loader_applies_the_sweep_overrides():
    args = psweep.load_sweep_args(_saved(), _parser(), overrides={
        "checkpoint_every": 0, "grad_oracle": "off", "wandb": "offline",
        "preference_sweep_checkpoint": "/ckpt", "preference_sweep_plans": 32})
    assert args.checkpoint_every == 0
    assert args.grad_oracle == "off"
    assert args.wandb == "offline"
    assert args.preference_sweep_checkpoint == "/ckpt"
    assert args.preference_sweep_plans == 32
    # Everything else is still the run's own.
    assert args.seed == 250201


def test_loader_refuses_an_argument_this_build_does_not_define():
    meta = _saved()
    meta["args"]["a_flag_from_another_build"] = 3
    with pytest.raises(CheckpointError, match="not defined here"):
        psweep.load_sweep_args(meta, _parser())


def test_loader_refuses_a_missing_argument_rather_than_defaulting_it():
    meta = _saved()
    del meta["args"]["seed"]
    with pytest.raises(CheckpointError, match="absent from the checkpoint"):
        psweep.load_sweep_args(meta, _parser())


def test_loader_refuses_an_override_of_the_trained_state():
    with pytest.raises(CheckpointError, match="may not override"):
        psweep.load_sweep_args(_saved(), _parser(),
                               overrides={"seed": 7})


def test_loader_refuses_a_checkpoint_without_a_namespace():
    with pytest.raises(CheckpointError, match="no argument namespace"):
        psweep.load_sweep_args({"episode": 10}, _parser())


def test_round_trip_against_the_real_trainer_parser():
    ppo = pytest.importorskip("alphagrad.approx.ppo")
    parser = ppo.make_argparser()
    saved = args_to_json(parser.parse_args([]))
    for name in psweep.SWEEP_ONLY_ARGS:
        saved.pop(name, None)
    assert psweep.sweep_args_round_trip({"args": saved, "episode": 1}, parser)


# ---------------------------------------------------------------------------
# THE PIN.
# ---------------------------------------------------------------------------
def test_edge_grid_walks_the_latency_memory_edge_corner_to_corner():
    w = psweep.edge_weights(11)
    assert len(w) == 11
    assert w[0] == (0.0, 1.0, 0.0)
    assert w[-1] == (1.0, 0.0, 0.0)
    assert all(abs(a + b - 1.0) < 1e-12 for a, b, _ in w)
    assert [a for a, _, _ in w] == sorted(a for a, _, _ in w)


def test_edge_grid_refuses_fewer_than_two_points():
    with pytest.raises(ValueError, match="at least the two corners"):
        psweep.edge_weights(1)


def test_explicit_weights_parse_and_are_checked_against_the_head_count():
    w = psweep.parse_weights("1,0,0;0.5,0.5,0", 3)
    assert w == [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0)]
    with pytest.raises(ValueError, match="conditioned on 3"):
        psweep.parse_weights("1,0", 3)


def test_schedule_repeats_a_weight_until_the_plans_are_measured():
    w = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert psweep.sweep_schedule(w, 16, 16) == w
    assert psweep.sweep_schedule(w, 32, 16) == [w[0], w[0], w[1], w[1]]


def test_schedule_refuses_a_plan_count_an_episode_cannot_measure():
    with pytest.raises(ValueError, match="not a multiple"):
        psweep.sweep_schedule([(1.0, 0.0, 0.0)], 24, 16)


def test_the_pinned_vector_reaches_the_policy_as_lambda_and_the_pin():
    """The pin THROUGH the trainer's own preference transform.

    Under `--reward-mode lagrangian` the vector the policy reads is
    `_lag_preferences` of the pinned one: the quality coordinate is lambda
    and the two cost coordinates are the pin, renormalised to sum to 1. A
    sweep that pinned (t, 1-t, 0) and did not check this would report a
    preference the policy never saw.
    """
    ppo = pytest.importorskip("alphagrad.approx.ppo")
    jnp = pytest.importorskip("jax.numpy")
    pinned = jnp.asarray([[0.25, 0.75, 0.0]] * 4, dtype=jnp.float32)
    head_weights = np.asarray([1.0, 1.0, 16.0], dtype=np.float32)
    out = np.asarray(ppo._lag_preferences(pinned, head_weights, 16.0, True))
    assert out.shape == (4, 3)
    assert np.allclose(out[:, 0], 0.25)
    assert np.allclose(out[:, 1], 0.75)
    assert np.allclose(out[:, 2], 16.0)
    # The corner survives the renormalisation as the corner.
    corner = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    out = np.asarray(ppo._lag_preferences(corner, head_weights, 12.0, True))
    assert np.allclose(out[0], [1.0, 0.0, 12.0])


# ---------------------------------------------------------------------------
# THE FRONT WRITER AND THE COMPARISON.
# ---------------------------------------------------------------------------
def _plan(w, lat, mem, quality=0.99, n=8):
    return {"episode": 0, "env": 0, "w": list(w), "preference": list(w),
            "order": [1, 2, 3], "seq": [[1, []], [2, []], [3, []]],
            "quality": quality, "refused": None,
            "band": {"latency": {"windows": [lat] * n, "median": lat,
                                 "lo": lat, "hi": lat, "n": n},
                     "memory": {"windows": [mem] * n, "median": mem,
                                "lo": mem, "hi": mem, "n": n}}}


def test_front_writer_dumps_the_schema_the_archive_dumps(tmp_path):
    plans = [_plan((1.0, 0.0, 0.0), -0.5, 0.2),
             _plan((0.0, 1.0, 0.0), 0.2, -0.6)]
    plans[1]["order"] = [3, 2, 1]
    plans[1]["seq"] = [[3, []], [2, []], [1, []]]
    archive = psweep.swept_front_from_plans(
        plans, ("latency", "peak_memory"), quality_floor=0.9)
    path = tmp_path / "preference_sweep_front.json"
    archive.dump_front(str(path), extra={"final": True})
    doc = json.loads(path.read_text())
    assert doc["objectives"] == ["latency", "peak_memory"]
    assert doc["num_points"] == 2
    assert "log ratio" in doc["space"]
    for point in doc["front"]:
        assert set(point["obj"]) == {"latency", "peak_memory"}
        assert set(point["band_lo"]) == {"latency", "peak_memory"}
        assert point["seq"]
    assert doc["final"] is True


def test_front_writer_excludes_a_refused_plan_and_a_plan_below_the_floor():
    good = _plan((1.0, 0.0, 0.0), -0.5, 0.2)
    refused = _plan((0.5, 0.5, 0.0), -0.9, -0.9)
    refused["refused"] = "sentinelled"
    refused["band"] = None
    poor = _plan((0.0, 1.0, 0.0), -0.9, -0.9, quality=0.1)
    poor["order"] = [2, 1, 3]
    poor["seq"] = [[2, []], [1, []], [3, []]]
    archive = psweep.swept_front_from_plans(
        [good, refused, poor], ("latency", "peak_memory"), quality_floor=0.9)
    assert len(archive.pts) == 1
    assert np.allclose(archive.pts[0], [-0.5, 0.2])


def test_plan_records_round_trip_as_jsonl(tmp_path):
    plans = [_plan((1.0, 0.0, 0.0), -0.5, 0.2),
             _plan((0.0, 1.0, 0.0), 0.2, -0.6)]
    path = tmp_path / "plans.jsonl"
    psweep.write_plan_records(str(path), plans)
    assert psweep.load_plan_records(str(path)) == plans


def test_front_points_reads_a_pareto_front_document():
    doc = {"objectives": ["latency", "peak_memory"],
           "front": [{"obj": {"latency": -0.5, "peak_memory": 0.2}},
                     {"obj": {"latency": 0.1, "peak_memory": -0.4}}]}
    pts = psweep.front_points(doc, ("latency", "peak_memory"))
    assert pts.shape == (2, 2)
    assert np.allclose(pts[0], [-0.5, 0.2])
    with pytest.raises(ValueError, match="not in one space"):
        psweep.front_points(doc, ("latency", "flops"))


def test_set_coverage_counts_weak_domination_in_minimisation():
    a = np.array([[-1.0, -1.0]])
    b = np.array([[0.0, 0.0], [-2.0, 5.0]])
    assert psweep.set_coverage(a, b) == 0.5
    assert psweep.set_coverage(b, a) == 0.0
    assert psweep.set_coverage(a, np.empty((0, 2))) == 0.0


def test_the_comparison_uses_one_recorded_nadir_for_both_fronts():
    swept = np.array([[-1.0, 0.0], [0.0, -1.0]])
    archive = np.array([[-0.5, -0.5]])
    out = psweep.compare_fronts(swept, archive, ("latency", "peak_memory"))
    ref = np.asarray(out["nadir"])
    assert ref.shape == (2,)
    # Below every point of both fronts in maximisation space, so both
    # hypervolumes are finite and measured against the same corner.
    both = -np.concatenate([swept, archive])
    assert np.allclose(ref, both.min(axis=0) - 1.0)
    assert np.all(both > ref)
    assert out["hypervolume_swept"] > 0.0
    assert out["hypervolume_archive"] > 0.0
    assert out["coverage_archive_of_swept"] == 0.0
    assert out["num_swept"] == 2 and out["num_archive"] == 1


def test_the_nadir_does_not_move_when_the_fronts_are_swapped():
    a = np.array([[-1.0, 0.0]])
    b = np.array([[0.0, -1.0]])
    assert np.allclose(psweep.shared_nadir(a, b), psweep.shared_nadir(b, a))


# ---------------------------------------------------------------------------
# THE LIKE-FOR-LIKE ARCHIVE WINDOW.
# ---------------------------------------------------------------------------
def _dated_front():
    return {"objectives": ["latency", "peak_memory"], "episode": 2000,
            "front": [
                {"obj": {"latency": +0.07, "peak_memory": -0.45},
                 "episode": 105},
                {"obj": {"latency": -0.05, "peak_memory": +0.00},
                 "episode": 1851},
                {"obj": {"latency": -0.08, "peak_memory": +0.00},
                 "episode": 1927},
                {"obj": {"latency": +0.03, "peak_memory": -0.01},
                 "episode": 1800},
            ]}


def test_the_archive_window_keeps_only_the_points_admitted_late():
    """The like-for-like set for a checkpoint's swept front.

    A `pareto_front.json` is a lifetime union. A point admitted at episode
    105 stays on it whether or not the final policy can still produce that
    plan, so the union is not what one checkpoint can be held to.
    """
    doc = _dated_front()
    late = psweep.front_window(doc, 200)
    assert late["window_since_episode"] == 1800
    assert late["num_points"] == 3
    assert [p["episode"] for p in late["front"]] == [1851, 1927, 1800]
    assert psweep.front_window(doc, 100)["num_points"] == 2
    # The window is a READ. The document it came from is untouched.
    assert len(doc["front"]) == 4


def test_the_archive_window_refuses_a_front_that_cannot_date_itself():
    with pytest.raises(ValueError, match="carries no 'episode'"):
        psweep.front_window(
            {"objectives": ["latency", "peak_memory"], "front": []}, 200)
    with pytest.raises(ValueError, match="at least one episode"):
        psweep.front_window({"episode": 10, "front": []}, 0)


def test_the_window_changes_the_coverage_the_sweep_is_judged_against():
    doc = {"objectives": ["latency", "peak_memory"], "episode": 2000,
           "front": [
               {"obj": {"latency": +0.5, "peak_memory": -0.5},
                "episode": 100},
               {"obj": {"latency": +0.1, "peak_memory": +0.1},
                "episode": 1990},
           ]}
    swept = np.array([[0.0, 0.0]])
    full = psweep.front_points(doc, ("latency", "peak_memory"))
    late = psweep.front_points(psweep.front_window(doc, 200),
                               ("latency", "peak_memory"))
    assert psweep.set_coverage(swept, full) == 0.5
    assert psweep.set_coverage(swept, late) == 1.0


def test_hypervolume_of_negates_once_and_is_empty_safe():
    ref = psweep.shared_nadir(np.array([[-1.0, -1.0]]))
    assert psweep.hypervolume_of(np.empty((0, 2)), ref) == 0.0
    assert psweep.hypervolume_of(np.array([[-1.0, -1.0]]),
                                 ref) == pytest.approx(1.0)


def test_per_weight_table_reports_the_band_and_the_feasible_fraction():
    plans = [_plan((1.0, 0.0, 0.0), -0.5, 0.2, quality=0.95),
             _plan((1.0, 0.0, 0.0), -0.3, 0.3, quality=0.5),
             _plan((0.0, 1.0, 0.0), 0.1, -0.4, quality=0.99)]
    refused = _plan((0.0, 1.0, 0.0), 0.0, 0.0)
    refused["refused"] = "no-band"
    refused["band"] = None
    rows = psweep.per_weight_table(plans + [refused], quality_floor=0.9)
    assert len(rows) == 2
    by_w = {tuple(r["w"]): r for r in rows}
    hot = by_w[(1.0, 0.0, 0.0)]
    assert hot["n"] == 2
    assert hot["feasible_fraction"] == 0.5
    assert hot["latency_median"] == pytest.approx(-0.4)
    cold = by_w[(0.0, 1.0, 0.0)]
    assert cold["n"] == 1
    assert cold["refused"] == 1
    assert cold["feasible_fraction"] == 1.0


def test_per_weight_band_is_the_median_interval_not_the_envelope():
    """One wild plan widens the SAMPLE, it does not set the band.

    The band is the 90 percent median interval of the weight's plans. Taking
    the union of their own bands instead would let a single memory blow-up
    -- a real measurement, kept in the table and the file -- claim the whole
    axis and hide every other weight.
    """
    plans = [_plan((0.5, 0.5, 0.0), -0.5, 0.0) for _ in range(8)]
    wild = _plan((0.5, 0.5, 0.0), -0.5, 5.0)
    wild["band"]["memory"]["hi"] = 9.0
    wild["band"]["memory"]["lo"] = 4.0
    rows = psweep.per_weight_table(plans + [wild], quality_floor=0.9)
    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 9
    assert row["memory_median"] == 0.0
    assert row["memory_hi"] <= 5.0
    assert row["memory_lo"] == 0.0
