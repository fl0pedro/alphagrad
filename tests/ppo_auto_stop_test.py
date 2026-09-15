"""THE AUTO-STOP DECISION, ON SYNTHETIC WINDOWS.

Ticket ``dsnn-dfw.6``. The trainer end of the feature is in
``ppo_auto_stop_trainer_test.py``; this file pins the decision itself, which
is the part a reader of an auto-stopped arm has to be able to trust.

WHY SYNTHETIC WINDOWS. The decision is a function of 200 episode rows and
nothing else. Driving a real trainer to a chosen window would take an hour per
case and would still not let a test ask "what happens when the return moves by
exactly 2 percent". So every case here builds the rows directly, and the
trainer test proves separately that the rows the trainer produces are the rows
this decision reads.

THE SHAPE OF EVERY CASE. A window of 5 and a check point at 10, so the
previous window is episodes 0..4 and the recent window is 5..9. `_fill` writes
one constant row over a range; a case changes ONE thing away from a baseline
in which all three conditions hold, and asserts that the run no longer stops.
That is the only way to show that all three are load-bearing.
"""

from __future__ import annotations

import argparse
import json

import pytest

from alphagrad.approx.common import auto_stop as autostop


_W = 5
_CHECK = 10


def _monitor(**kw):
    kw.setdefault("window", _W)
    kw.setdefault("points", (_CHECK,))
    return autostop.AutoStopMonitor(**kw)


def _fill(mon, lo, hi, *, admitted=0, ret=100.0, quality=0.9, approx=3,
          plan="P", skips=0, n_envs=2):
    """One constant row over the episode indices `lo`..`hi` inclusive."""
    for ep in range(lo, hi + 1):
        mon.record(
            episode=ep,
            admitted=admitted,
            scalar_return=ret,
            qualities=[quality] * n_envs,
            plan_hashes=[autostop.plan_hash(plan if isinstance(plan, list)
                                            else [plan])] * n_envs,
            n_approx=[approx] * n_envs,
            n_skip=[skips] * n_envs,
        )


def _settled(**recent):
    """The baseline in which all three conditions hold."""
    mon = _monitor()
    _fill(mon, 0, _W - 1, ret=100.0)
    _fill(mon, _W, 2 * _W - 1, **{"ret": 100.0, **recent})
    return mon


# ---------------------------------------------------------------------------
# 1. All three together, and each one alone.
# ---------------------------------------------------------------------------

def test_all_three_conditions_together_stop_the_run():
    d = _settled().decide(_CHECK)
    assert d["stop"] is True
    c = d["reason"]["conditions"]
    assert c["archive_admitted_nothing"] is True
    assert c["return_moved_less_than_tolerance"] is True
    assert c["collapse_or_plan_frozen"] is True
    assert c["plan_did_not_change"] is True
    assert "STOPS" in d["message"]


def test_an_archive_that_still_admits_does_not_stop():
    """Condition 1 alone. Everything else is the settled baseline."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    _fill(mon, _W, 2 * _W - 2)
    _fill(mon, 2 * _W - 1, 2 * _W - 1, admitted=1)
    d = mon.decide(_CHECK)
    assert d["stop"] is False
    assert d["reason"]["conditions"]["archive_admitted_nothing"] is False
    assert d["reason"]["numbers"]["admitted_recent"] == 1
    # The other two still hold, so the archive is the only reason.
    assert d["reason"]["conditions"]["return_moved_less_than_tolerance"]
    assert d["reason"]["conditions"]["collapse_or_plan_frozen"]


def test_a_return_that_still_moves_does_not_stop():
    """Condition 2 alone: 10 percent against the previous window."""
    d = _settled(ret=110.0).decide(_CHECK)
    assert d["stop"] is False
    assert d["reason"]["conditions"]["return_moved_less_than_tolerance"] is False
    assert d["reason"]["numbers"]["relative_return_move"] == pytest.approx(0.1)
    assert d["reason"]["conditions"]["archive_admitted_nothing"]
    assert d["reason"]["conditions"]["collapse_or_plan_frozen"]


def test_a_policy_that_is_still_moving_does_not_stop():
    """Condition 3 alone: no collapse, and the terminal plan keeps changing."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    for ep in range(_W, 2 * _W):
        _fill(mon, ep, ep, plan=f"plan-{ep}")
    d = mon.decide(_CHECK)
    assert d["stop"] is False
    c = d["reason"]["conditions"]
    assert c["archive_admitted_nothing"] is True
    assert c["return_moved_less_than_tolerance"] is True
    assert c["collapse_or_plan_frozen"] is False
    assert c["collapse_detector_fired"] is False
    assert c["plan_did_not_change"] is False
    assert d["reason"]["numbers"]["distinct_terminal_plans_in_window"] >= 2


def test_two_of_three_are_not_enough():
    """Conditions 1 and 2 hold and 3 does not; and 2 and 3 hold and 1 does
    not. Neither pair stops, which is what "ALL THREE" means."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    for ep in range(_W, 2 * _W):
        _fill(mon, ep, ep, plan=f"plan-{ep}")
    assert mon.decide(_CHECK)["stop"] is False

    mon = _monitor()
    _fill(mon, 0, _W - 1)
    _fill(mon, _W, 2 * _W - 1, admitted=1)
    assert mon.decide(_CHECK)["stop"] is False


# ---------------------------------------------------------------------------
# 2. The two collapse modes.
# ---------------------------------------------------------------------------

def test_arm_a_collapse_to_zero_quality_fires_the_detector():
    """Arm A's prediction. The plan keeps changing, so condition 3 can only
    be met by the collapse detector."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    for ep in range(_W, 2 * _W):
        _fill(mon, ep, ep, quality=0.004, plan=f"plan-{ep}")
    d = mon.decide(_CHECK)
    assert d["stop"] is True
    c = d["reason"]["conditions"]
    assert c["collapse_quality"] is True
    assert c["collapse_identity"] is False
    assert c["plan_did_not_change"] is False
    assert d["reason"]["numbers"][
        "quality_median_of_episode_medians"] == pytest.approx(0.004)
    # Every plan of the window counted, and every one was under the threshold.
    assert d["reason"]["numbers"]["quality_plans_below_threshold"] == 2 * _W


def test_arm_b_stay_at_the_identity_fires_the_detector():
    """Arm B's prediction: not one approximation anywhere in the window. The
    quality is high and the plan keeps changing, so again only the detector
    can meet condition 3."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    for ep in range(_W, 2 * _W):
        _fill(mon, ep, ep, quality=1.0, approx=0, plan=f"plan-{ep}")
    d = mon.decide(_CHECK)
    assert d["stop"] is True
    c = d["reason"]["conditions"]
    assert c["collapse_identity"] is True
    assert c["collapse_quality"] is False
    assert c["plan_did_not_change"] is False
    assert d["reason"]["numbers"][
        "max_approximations_in_any_terminal_plan"] == 0


def test_one_approximation_anywhere_in_the_window_unfires_arm_b():
    """The identity collapse is ZERO approximations in EVERY terminal plan.
    A single one in a single environment of a single episode is enough to say
    the policy has left the identity."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    _fill(mon, _W, 2 * _W - 2, quality=1.0, approx=0, plan="same")
    for ep in (2 * _W - 1,):
        mon.record(episode=ep, admitted=0, scalar_return=100.0,
                   qualities=[1.0, 1.0],
                   plan_hashes=[autostop.plan_hash(["same"])] * 2,
                   n_approx=[0, 1], n_skip=[0, 0])
    d = mon.decide(_CHECK)
    assert d["reason"]["conditions"]["collapse_identity"] is False
    # The two plans of that episode now differ in content only through the
    # count, not the hash, so the plan condition still holds and the run
    # still stops -- on condition 3c, not on the detector.
    assert d["reason"]["conditions"]["plan_did_not_change"] is True
    assert d["stop"] is True


def test_a_skip_is_not_an_approximation():
    """CONTEXT.md: Skip is a distinct action class. A window of pure-skip
    plans is still "zero approximations applied", and the count of skips is
    reported beside it so a reader sees what the plans were."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    for ep in range(_W, 2 * _W):
        _fill(mon, ep, ep, quality=1.0, approx=0, skips=4, plan=f"p{ep}")
    d = mon.decide(_CHECK)
    assert d["reason"]["conditions"]["collapse_identity"] is True
    assert d["reason"]["numbers"]["total_skips_in_window"] == 4 * 2 * _W
    assert d["reason"]["numbers"]["total_approximations_in_window"] == 0


def test_a_quality_median_exactly_at_the_threshold_is_not_a_collapse():
    """Below 0.05, strictly. The threshold value itself is a plan that still
    computes a gradient."""
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    for ep in range(_W, 2 * _W):
        _fill(mon, ep, ep, quality=0.05, plan=f"p{ep}")
    d = mon.decide(_CHECK)
    assert d["reason"]["conditions"]["collapse_quality"] is False
    assert d["stop"] is False


# ---------------------------------------------------------------------------
# 3. The 2 percent boundary.
# ---------------------------------------------------------------------------

def test_a_return_move_of_exactly_two_percent_does_not_stop():
    """100 -> 102 is 0.02 exactly in binary floating point (2/100), and the
    rule is STRICTLY less than the tolerance."""
    d = _settled(ret=102.0).decide(_CHECK)
    assert d["reason"]["numbers"]["relative_return_move"] == 0.02
    assert d["reason"]["conditions"]["return_moved_less_than_tolerance"] is False
    assert d["stop"] is False


def test_a_return_move_just_under_two_percent_stops():
    d = _settled(ret=101.9).decide(_CHECK)
    assert d["reason"]["numbers"]["relative_return_move"] < 0.02
    assert d["reason"]["conditions"]["return_moved_less_than_tolerance"] is True
    assert d["stop"] is True


def test_the_move_is_relative_and_signed_moves_count_the_same():
    """A fall of 2 percent is as much of a move as a rise of 2 percent."""
    assert _settled(ret=98.0).decide(_CHECK)["reason"][
        "numbers"]["relative_return_move"] == 0.02
    assert _settled(ret=98.1).decide(_CHECK)["stop"] is True


def test_a_previous_window_at_exactly_zero_is_handled_by_a_named_rule():
    """A relative move against 0 has no value. Two zero windows have not
    moved; a move away from zero is not a small move."""
    mon = _monitor()
    _fill(mon, 0, _W - 1, ret=0.0)
    _fill(mon, _W, 2 * _W - 1, ret=0.0)
    assert mon.decide(_CHECK)["reason"][
        "numbers"]["relative_return_move"] == 0.0
    assert mon.decide(_CHECK)["stop"] is True

    mon = _monitor()
    _fill(mon, 0, _W - 1, ret=0.0)
    _fill(mon, _W, 2 * _W - 1, ret=1.0)
    d = mon.decide(_CHECK)
    assert d["reason"]["numbers"]["relative_return_move"] == float("inf")
    assert d["stop"] is False


# ---------------------------------------------------------------------------
# 4. The window itself.
# ---------------------------------------------------------------------------

def test_an_incomplete_window_does_not_decide():
    """Reachable on a resume from a checkpoint inside the previous window.
    The run continues and the message names the missing episodes."""
    mon = _monitor()
    _fill(mon, _W, 2 * _W - 1)
    d = mon.decide(_CHECK)
    assert d["stop"] is False
    assert d["reason"]["incomplete"] is True
    assert d["reason"]["missing_count"] == _W
    assert "not observed" in d["message"]


def test_the_windows_are_the_two_blocks_before_the_check_point():
    d = _settled().decide(_CHECK)
    assert d["reason"]["recent_window"] == [_W, 2 * _W - 1]
    assert d["reason"]["previous_window"] == [0, _W - 1]


def test_the_history_is_capped_at_two_windows():
    mon = _monitor()
    _fill(mon, 0, 99)
    assert len(mon.rows) == 2 * _W
    assert min(mon.rows) == 100 - 2 * _W
    assert max(mon.rows) == 99


def test_an_episode_with_no_live_environment_contributes_no_quality():
    mon = _monitor()
    _fill(mon, 0, _W - 1)
    for ep in range(_W, 2 * _W):
        mon.record(episode=ep, admitted=0, scalar_return=100.0,
                   qualities=[],
                   plan_hashes=[autostop.plan_hash(["x"])],
                   n_approx=[3], n_skip=[0])
    d = mon.decide(_CHECK)
    assert d["reason"]["numbers"][
        "quality_median_of_episode_medians"] is None
    assert d["reason"]["conditions"]["collapse_quality"] is False
    # The plan condition still carries the decision.
    assert d["stop"] is True


def test_a_non_finite_scalar_return_raises():
    mon = _monitor()
    with pytest.raises(autostop.AutoStopError, match="non-finite"):
        mon.record(episode=0, admitted=0, scalar_return=float("nan"),
                   qualities=[1.0], plan_hashes=["a"], n_approx=[0])


# ---------------------------------------------------------------------------
# 5. The plan hash.
# ---------------------------------------------------------------------------

def test_two_equal_plans_hash_equal_and_two_different_ones_do_not():
    a = [(1, ["diag(0, 1, 4)"]), (2, [])]
    b = [(1, ["diag(0, 1, 4)"]), (2, [])]
    c = [(1, ["diag(0, 1, 8)"]), (2, [])]
    assert autostop.plan_hash(a) == autostop.plan_hash(b)
    assert autostop.plan_hash(a) != autostop.plan_hash(c)


def test_the_face_wires_are_part_of_the_plan():
    """A Plan is the order TOGETHER WITH the approximation assignment
    (CONTEXT.md). Two plans with the same order and different face wires are
    two plans."""
    seq = [(1, []), (2, [])]
    one = {"seq": seq, "faces": [{"k": 0, "f": [0], "rows": [[[1, 2, 3]]],
                                  "skips": [0]}]}
    two = {"seq": seq, "faces": [{"k": 0, "f": [0], "rows": [[[1, 2, 4]]],
                                  "skips": [0]}]}
    assert autostop.plan_hash(one) != autostop.plan_hash(two)
    assert autostop.plan_hash(one) != autostop.plan_hash(seq)


def test_the_hash_is_a_short_hex_string():
    h = autostop.plan_hash([(1, [])])
    assert len(h) == 16
    assert all(ch in "0123456789abcdef" for ch in h)


# ---------------------------------------------------------------------------
# 6. Persistence through the checkpoint.
# ---------------------------------------------------------------------------

def test_the_history_round_trips_through_json():
    mon = _settled()
    doc = json.loads(json.dumps(mon.to_json()))
    back = _monitor()
    back.load_json(doc)
    assert back.rows == mon.rows
    assert back.decide(_CHECK)["stop"] is True


def test_a_resume_without_a_history_raises():
    with pytest.raises(autostop.AutoStopError, match="no auto-stop history"):
        _monitor().load_json(None)


def test_a_history_taken_under_another_window_raises():
    doc = _settled().to_json()
    other = autostop.AutoStopMonitor(window=7, points=(_CHECK,))
    with pytest.raises(autostop.AutoStopError, match="window"):
        other.load_json(doc)


def test_a_history_with_other_check_points_raises():
    doc = _settled().to_json()
    other = _monitor(points=(_CHECK, 99))
    with pytest.raises(autostop.AutoStopError, match="check points"):
        other.load_json(doc)


def test_a_history_of_another_version_raises():
    doc = _settled().to_json()
    doc["version"] = autostop.HISTORY_VERSION + 1
    with pytest.raises(autostop.AutoStopError, match="version"):
        _monitor().load_json(doc)


# ---------------------------------------------------------------------------
# 7. The artefacts a stop writes.
# ---------------------------------------------------------------------------

def test_the_reason_file_is_strict_json_and_names_every_condition(tmp_path):
    reason = _settled().decide(_CHECK)["reason"]
    path = autostop.write_auto_stop_json(str(tmp_path), reason)
    with open(path) as fh:
        doc = json.load(fh)
    assert doc["stop"] is True
    assert doc["check_point"] == _CHECK
    assert set(doc["conditions"]) == {
        "archive_admitted_nothing", "return_moved_less_than_tolerance",
        "collapse_or_plan_frozen", "collapse_detector_fired",
        "collapse_quality", "collapse_identity", "plan_did_not_change"}
    assert doc["numbers"]["mean_return_recent"] == 100.0
    assert path.endswith(autostop.AUTO_STOP_FILENAME)


def test_the_wandb_summary_fields_are_flat_scalars():
    fields = autostop.summary_fields(_settled().decide(_CHECK)["reason"])
    assert fields["auto_stop/stopped"] is True
    assert fields["auto_stop/episodes"] == _CHECK
    assert fields["auto_stop/archive_admitted_nothing"] is True
    assert fields["auto_stop/mean_return_recent"] == 100.0
    for name, value in fields.items():
        assert name.startswith("auto_stop/")
        assert isinstance(value, (bool, int, float, str)), (name, value)


def test_the_plan_log_record_says_it_is_not_a_plan():
    rec = autostop.plan_log_record(_settled().decide(_CHECK)["reason"])
    assert rec["record"] == "auto_stop"
    assert "episode" not in rec
    assert "plan_index" not in rec
    assert rec["auto_stop"]["check_point"] == _CHECK


# ---------------------------------------------------------------------------
# 8. The flags, and the refusals.
# ---------------------------------------------------------------------------

def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=50)
    autostop.add_auto_stop_args(p)
    return p


def test_auto_stop_is_off_by_default_with_the_ruled_check_points():
    ns = _parser().parse_args([])
    assert ns.auto_stop is False
    assert autostop.check_points(ns) == (250, 500)
    assert autostop.window_size(ns) == 100
    # And off, the refusal never fires, whatever else is on the line.
    ns = _parser().parse_args(["--checkpoint-every", "0"])
    autostop.check_auto_stop_args(ns)


def test_auto_stop_with_checkpointing_off_is_refused():
    ns = _parser().parse_args(["--auto-stop", "--checkpoint-every", "0",
                               "--episodes", "1000"])
    with pytest.raises(autostop.AutoStopError, match="nothing to resume"):
        autostop.check_auto_stop_args(ns)


def test_the_ruled_thesis_command_line_is_accepted():
    ns = _parser().parse_args(["--auto-stop", "--checkpoint-every", "50",
                               "--episodes", "1000"])
    autostop.check_auto_stop_args(ns)
    assert autostop.check_points(ns) == (250, 500)


def test_a_check_point_that_cannot_hold_two_windows_is_refused():
    ns = _parser().parse_args(["--auto-stop", "--checkpoint-every", "50",
                               "--episodes", "1000",
                               "--auto-stop-check-at", "150"])
    with pytest.raises(autostop.AutoStopError, match="window of 100"):
        autostop.check_auto_stop_args(ns)


def test_the_test_only_knobs_move_the_check_points_and_the_window():
    ns = _parser().parse_args(["--auto-stop", "--checkpoint-every", "2",
                               "--episodes", "8",
                               "--auto-stop-check-at", "4",
                               "--auto-stop-window", "2"])
    autostop.check_auto_stop_args(ns)
    assert autostop.check_points(ns) == (4,)
    assert autostop.window_size(ns) == 2


@pytest.mark.parametrize("bad", ["", "abc", "0", "-5"])
def test_a_check_point_list_that_is_not_episode_counts_raises(bad):
    ns = _parser().parse_args(["--auto-stop-check-at", bad])
    with pytest.raises(autostop.AutoStopError):
        autostop.check_points(ns)


def test_a_window_of_zero_raises():
    ns = _parser().parse_args(["--auto-stop-window", "0"])
    with pytest.raises(autostop.AutoStopError, match="positive"):
        autostop.window_size(ns)


# ---------------------------------------------------------------------------
# 9. The gate arm.
# ---------------------------------------------------------------------------

def test_the_gate_arm_carries_no_auto_stop_flag():
    """`tests/policy_regression_gate.py` traces the policy path in its own
    interpreter: it has no episodes, no optimiser and no trainer loop, so a
    check point cannot be reached there at all. This pins the other half of
    "never on the gate arm": no gate file mentions the flag, so no gate run
    can turn it on by accident."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("tests/policy_regression_gate.py",
                 "tests/policy_regression_gate_test.py",
                 "tools/smoke.sh"):
        path = root / name
        if not path.exists():
            continue
        text = path.read_text()
        assert "--auto-stop" not in text, f"{name} names --auto-stop"
        assert "auto_stop" not in text, f"{name} names auto_stop"


def test_the_gate_arm_command_line_parses_to_auto_stop_off():
    """The canonical deterministic configuration of `tools/smoke.sh`, with
    the checkpoint flag the gate arm runs at. --auto-stop is absent, so the
    trainer builds no monitor, and the refusal that --checkpoint-every 0
    would otherwise trigger does not fire either."""
    ns = _parser().parse_args(["--episodes", "2", "--checkpoint-every", "0"])
    assert ns.auto_stop is False
    autostop.check_auto_stop_args(ns)
