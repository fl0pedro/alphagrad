"""THE ASYNCHRONOUS GRADIENT ORACLE (owner ruling 2026-09-18).

The ruling: the oracle is a SANITY CHECK, not part of the scoring. It runs on
one worker thread in the trainer process, on the CPU device, in float64, AFTER
the episode whose orders it checks. The trainer never waits for it.

What "never waits" has to mean, and what these tests pin:

(a) THE EPISODE PACE DOES NOT MOVE. A check that takes longer than an episode
    is still running while the next episodes go by, and its answer arrives
    later carrying the episode it belongs to.
(b) A FAIL STOPS THE RUN AT AN EPISODE BOUNDARY, AFTER THE CHECKPOINT. In
    float64 on the CPU every order sits at 1e-14, so a disagreement is a real
    graphax defect. The checkpoint is written first, so the run resumes from
    it; the order of the two calls is pinned in ppo.py's own source, because
    it is a property of the call site.
(c) A TIMEOUT COUNTS AND DOES NOT RAISE. It is missing data, not a bad plan.
(d) THE LATE RECORD REFERENCES THE RIGHT PLAN. The plan log is append-only, so
    the answer is its own line, joined by the plan's content hash.
(e) THE GATE ARM IS UNTOUCHED. Under ``--grad-oracle off`` there is no oracle
    object, no thread, no key and no record.

Nothing here needs a GPU and nothing here runs a campaign: the scheduling is
separated from the check on purpose, so a fake check can be slow, can fail and
can hang without costing a compile.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
import time

import numpy as np
import pytest

from alphagrad.approx.common.grad_oracle_async import (
    AsyncGradOracle, result_record)


_ORDER_A = (3, 2, 1)
_ORDER_B = (1, 2, 3)
_HASH_A = "aa" * 16
_HASH_B = "bb" * 16


def _job(order, *hashes):
    return {"order": order, "plan_hashes": list(hashes)}


def _wait_for(predicate, timeout=20.0):
    """Poll until `predicate` holds. Returns True, or False on the timeout."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ===========================================================================
# (a) THE TRAINER DOES NOT BLOCK
# ===========================================================================
def test_submitting_returns_at_once_and_the_episodes_keep_their_pace():
    """The fake check sleeps longer than an episode. The episodes still run at
    their own pace, and the oracle is still working when they are done."""
    episode_s = 0.02
    check_s = 10 * episode_s
    started = threading.Event()

    def slow_check(order, probe_seed, episode):
        started.set()
        time.sleep(check_s)
        return "pass", 1e-14

    oracle = AsyncGradOracle(slow_check, timeout_s=60.0)
    try:
        t0 = time.monotonic()
        oracle.submit(0, 1234, [_job(_ORDER_A, _HASH_A)])
        submit_s = time.monotonic() - t0
        assert submit_s < episode_s, (
            f"submit took {submit_s:.4f}s, which is an episode's worth of "
            f"time; the trainer must not wait for the oracle at all")
        assert started.wait(5.0), "the worker thread never started the check"

        # Five episodes go by while the check runs.
        t1 = time.monotonic()
        for ep in range(1, 6):
            time.sleep(episode_s)
            assert oracle.take_results() == [], (
                "the check cannot be finished yet; nothing may be reported")
        five_eps = time.monotonic() - t1
        assert five_eps < check_s, (
            f"five episodes took {five_eps:.3f}s against one check's "
            f"{check_s:.3f}s -- the loop waited for the oracle")
        assert oracle.counts()["pending"] == 1

        # And the answer lands later, at a boundary of its own.
        got = []
        assert _wait_for(lambda: bool(got.extend(oracle.take_results()) or got))
        assert [r["episode"] for r in got] == [0]
        assert oracle.counts()["pending"] == 0
    finally:
        oracle.close()


def test_the_late_result_carries_the_episode_it_belongs_to():
    """The answer for episode 3 says episode 3, however many boundaries have
    passed. Without that the late record would be filed under the episode that
    happened to be running when it landed."""
    def check(order, probe_seed, episode):
        time.sleep(0.05)
        return "pass", 2e-14

    oracle = AsyncGradOracle(check, timeout_s=60.0)
    try:
        oracle.submit(3, 777, [_job(_ORDER_A, _HASH_A)])
        got = []
        assert _wait_for(lambda: bool(got.extend(oracle.take_results()) or got))
        assert len(got) == 1
        assert got[0]["episode"] == 3
        assert got[0]["order"] == _ORDER_A
        assert got[0]["status"] == "pass"
        assert oracle.counts() == {
            "pass": 1, "fail": 0, "timeout": 0, "pending": 0,
            "submitted": 1, "late": 0}
    finally:
        oracle.close()


def test_an_order_is_checked_once_per_process():
    """"Once per process and order" is the oracle's own rule and it survives
    the move to a thread: the second submission of the same order answers from
    the memo without running the check again."""
    calls = []

    def check(order, probe_seed, episode):
        calls.append(order)
        return "pass", 3e-14

    oracle = AsyncGradOracle(check, timeout_s=60.0)
    try:
        oracle.submit(0, 1, [_job(_ORDER_A, _HASH_A)])
        assert _wait_for(lambda: oracle.counts()["pass"] == 1
                         or bool(oracle.take_results()))
        oracle.submit(50, 2, [_job(_ORDER_A, _HASH_B)])
        got = []
        assert _wait_for(lambda: bool(got.extend(oracle.take_results()) or got))
        assert calls == [_ORDER_A], (
            "the exact gradient of an order does not depend on the episode; "
            "the second submission must not re-run the check")
        assert got[0]["from_memo"] is True
        assert got[0]["episode"] == 50
        assert got[0]["plan_hashes"] == [_HASH_B]
    finally:
        oracle.close()


# ===========================================================================
# (b) A FAIL RAISES AT THE NEXT EPISODE BOUNDARY, AFTER THE CHECKPOINT
# ===========================================================================
def test_a_fail_raises_at_the_boundary(tmp_path):
    from alphagrad.approx.env import GradientOracleFailure
    from alphagrad.approx.ppo import _grad_oracle_boundary

    def check(order, probe_seed, episode):
        return "fail", 4.2e-2

    oracle = AsyncGradOracle(check, timeout_s=60.0)
    path = str(tmp_path / "plan_log.jsonl")
    try:
        oracle.submit(7, 99, [_job(_ORDER_A, _HASH_A)])
        assert _wait_for(lambda: oracle._done.qsize() > 0)
        with pytest.raises(GradientOracleFailure) as exc:
            _grad_oracle_boundary(oracle, path, 8, 1e-3, log=lambda _l: None)
        msg = str(exc.value)
        assert "4.2e-02" in msg or "0.042" in msg, msg
        assert str(_ORDER_A) in msg, msg
        assert "episode 7" in msg, msg
        assert "checkpoint at episode 8" in msg, msg
        assert oracle.counts()["fail"] == 1
        # THE RECORD IS WRITTEN BEFORE THE RAISE. A failure that stopped the
        # run without leaving its own line would be a defect with no evidence.
        rows = [json.loads(l) for l in open(path) if l.strip()]
        assert [r["oracle"]["status"] for r in rows] == ["fail"]
        assert rows[0]["plan_hash"] == _HASH_A
        assert rows[0]["episode"] == 7
        assert rows[0]["oracle"]["checked_at_episode"] == 8
    finally:
        oracle.close()


def test_a_check_that_raises_is_a_fail_and_not_a_silent_skip(tmp_path):
    """An oracle that cannot run is a defect of the same apparatus. It is
    reported as a fail with its exception text, not swallowed."""
    from alphagrad.approx.env import GradientOracleFailure
    from alphagrad.approx.ppo import _grad_oracle_boundary

    def check(order, probe_seed, episode):
        raise ValueError("the elimination did not build")

    oracle = AsyncGradOracle(check, timeout_s=60.0)
    try:
        oracle.submit(1, 5, [_job(_ORDER_A, _HASH_A)])
        assert _wait_for(lambda: oracle._done.qsize() > 0)
        with pytest.raises(GradientOracleFailure) as exc:
            _grad_oracle_boundary(oracle, str(tmp_path / "p.jsonl"), 2, 1e-3,
                                  log=lambda _l: None)
        assert "the elimination did not build" in str(exc.value)
    finally:
        oracle.close()


def test_the_boundary_runs_after_the_checkpoint_in_the_episode_loop():
    """THE ORDER OF THE TWO CALLS, PINNED IN THE SOURCE.

    A fail raises. If it raised before `_ckpt_write`, the run would stop with
    no checkpoint at that episode and every episode since the last one would be
    lost -- which is the difference between a defect that costs an hour and one
    that costs a day. The order is a property of the CALL SITE, so it is
    checked at the call site.
    """
    import alphagrad.approx.ppo as ppo

    src = inspect.getsource(ppo.main)
    loop = src[src.index("for ep in range(_ep_start, args.episodes):"):]
    # The first iteration's body, up to the next rollout dispatch.
    head = loop[:loop.index("set_walk_episode")]
    assert "_ckpt_write(ep)" in head, head[:400]
    assert "_grad_oracle_boundary(" in head, head[:400]
    assert head.index("_ckpt_write(ep)") < head.index("_grad_oracle_boundary("), (
        "the gradient oracle's boundary must run AFTER the checkpoint: it is "
        "the one call in the loop that may stop the run, and the run has to "
        "be resumable from the checkpoint of that same episode")


def test_the_drain_at_exit_runs_after_the_final_checkpoint():
    """Same rule at the end of the run, for the same reason."""
    import alphagrad.approx.ppo as ppo

    src = inspect.getsource(ppo.main)
    tail = src[src.index("_ckpt_write(_EPISODES_DONE[0])"):]
    assert "_GRAD_ORACLE.drain(" in tail, tail[:400]


def test_no_oracle_state_rides_the_checkpoint():
    """A RESUME STARTS WITH NOTHING PENDING, and must.

    The checkpoint is the arithmetic state of the run. The oracle's pending
    checks live in a thread of a process that is about to end; a resumed run is
    a new process with a new thread and no backlog, and it re-submits from its
    own episodes. A checkpoint that carried the backlog would make a resume
    raise on a failure the fixed code no longer has.
    """
    import alphagrad.approx.ppo as ppo

    src = inspect.getsource(ppo.main)
    tree = src[src.index("def _ckpt_tree():"):src.index("def _ckpt_meta():")]
    meta = src[src.index("def _ckpt_meta():"):src.index("_CKPT_DIR = ")]
    for half in (tree, meta):
        assert "_GRAD_ORACLE" not in half, half
        assert "oracle" not in half.lower(), half


# ===========================================================================
# (c) A TIMEOUT COUNTS AND DOES NOT RAISE
# ===========================================================================
def test_a_timeout_counts_and_does_not_raise(tmp_path):
    """The check never answers. After the timeout the trainer counts it, writes
    it and goes on -- an unanswered sanity check is missing data, and the
    owner's rule stops the run for a wrong gradient, not for a slow one."""
    from alphagrad.approx.ppo import _grad_oracle_boundary

    release = threading.Event()

    def hanging_check(order, probe_seed, episode):
        release.wait(30.0)
        return "pass", 1e-14

    oracle = AsyncGradOracle(hanging_check, timeout_s=0.05)
    path = str(tmp_path / "plan_log.jsonl")
    try:
        oracle.submit(2, 5, [_job(_ORDER_A, _HASH_A)])
        assert _wait_for(lambda: oracle.counts()["pending"] == 1)
        time.sleep(0.1)
        # No raise.
        out = _grad_oracle_boundary(oracle, path, 3, 1e-3, log=lambda _l: None)
        assert [r["status"] for r in out] == ["timeout"]
        assert oracle.counts()["timeout"] == 1
        assert oracle.counts()["fail"] == 0
        assert oracle.counts()["pending"] == 0
        rows = [json.loads(l) for l in open(path) if l.strip()]
        assert rows[0]["oracle"]["status"] == "timeout"
        assert rows[0]["oracle"]["rel_l2"] is None
        assert rows[0]["episode"] == 2
    finally:
        release.set()
        oracle.close()


def test_an_answer_that_arrives_after_its_timeout_is_dropped_and_counted():
    """A disowned check's late answer must not be filed: the boundary it
    belonged to has passed and the run has already recorded what it knew. It is
    counted under `late` so the drop is visible."""
    release = threading.Event()

    def slow(order, probe_seed, episode):
        release.wait(10.0)
        return "pass", 1e-14

    oracle = AsyncGradOracle(slow, timeout_s=0.05)
    try:
        oracle.submit(0, 1, [_job(_ORDER_A, _HASH_A)])
        assert _wait_for(lambda: oracle.counts()["pending"] == 1)
        time.sleep(0.1)
        assert [r["status"] for r in oracle.take_results()] == ["timeout"]
        release.set()
        assert _wait_for(lambda: oracle.counts()["late"] == 1
                         or bool(oracle.take_results()))
        assert oracle.take_results() == []
        c = oracle.counts()
        assert (c["pass"], c["timeout"], c["late"]) == (0, 1, 1)
    finally:
        release.set()
        oracle.close()


def test_the_drain_at_exit_accounts_for_every_check():
    """A run ends with pass + fail + timeout equal to what it submitted."""
    release = threading.Event()
    seen = []

    def check(order, probe_seed, episode):
        seen.append(order)
        if order == _ORDER_B:
            release.wait(10.0)
        return "pass", 1e-14

    oracle = AsyncGradOracle(check, timeout_s=0.3)
    try:
        oracle.submit(0, 1, [_job(_ORDER_A, _HASH_A), _job(_ORDER_B, _HASH_B)])
        out = oracle.drain(0.5)
        c = oracle.counts()
        assert c["pending"] == 0
        assert c["pass"] + c["fail"] + c["timeout"] == c["submitted"] == 2
        assert sorted(r["status"] for r in out) == ["pass", "timeout"]
    finally:
        release.set()
        oracle.close()


def test_the_timeout_default_and_its_flag():
    from alphagrad.approx.env import grad_oracle_timeout
    from alphagrad.approx.ppo import make_argparser

    assert grad_oracle_timeout() == 600.0
    actions = [a for a in make_argparser()._actions
               if "--grad-oracle-timeout" in (a.option_strings or ())]
    assert len(actions) == 1
    assert actions[0].default == 600.0
    assert "timeout" in actions[0].help.lower()
    assert "not stop the run" in actions[0].help.lower()


def test_the_timeout_env_override(monkeypatch):
    from alphagrad.approx.env import grad_oracle_timeout

    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_TIMEOUT", "12")
    assert grad_oracle_timeout() == 12.0
    monkeypatch.setenv("ALPHAGRAD_GRAD_ORACLE_TIMEOUT", "nonsense")
    assert grad_oracle_timeout() == 600.0


# ===========================================================================
# (d) THE LATE RECORD REFERENCES THE RIGHT PLAN
# ===========================================================================
def _wires(order):
    """The five wires of a plan with no approximation on any face."""
    n = len(order)
    return (np.asarray(order, dtype=np.int32),
            np.full((n, 2, 3), -1, dtype=np.int32),
            np.full((n, 1, 1, 3), -1, dtype=np.int32),
            np.zeros((n, 1), dtype=np.int32),
            None)


def test_every_terminal_plan_record_carries_its_content_hash():
    """The join key exists on EVERY terminal plan, dedupe on or off. A key that
    exists only under a flag is not a key."""
    import alphagrad.approx.env as envmod

    order, rules, faces, skips, joins = _wires(_ORDER_A)
    envmod._PLAN_RECORDS.clear()
    envmod._record_terminal_plan(
        order=order, rule_specs=rules, face_specs=faces, face_skips=skips,
        face_joins=joins, reward_vec=np.zeros(len(envmod.REWARD_NAMES)),
        face_before=None, face_after={}, counts_from_trace=False)
    assert len(envmod._PLAN_RECORDS) == 1
    rec = envmod._PLAN_RECORDS[0]
    envmod._PLAN_RECORDS.clear()
    assert rec["plan_hash"] == envmod._plan_content_key(
        order, rules, faces, skips, joins).hex()
    assert re.fullmatch(r"[0-9a-f]{32}", rec["plan_hash"])


def test_the_late_record_names_the_plan_and_the_episode(tmp_path):
    """END TO END OF THE JOIN: a plan record's hash, the job built from it, the
    answer, the late record, and the two lines meeting on `plan_hash`."""
    import alphagrad.approx.env as envmod
    from alphagrad.approx.ppo import _grad_oracle_boundary, _grad_oracle_jobs

    order, rules, faces, skips, joins = _wires(_ORDER_A)
    envmod._PLAN_RECORDS.clear()
    envmod._record_terminal_plan(
        order=order, rule_specs=rules, face_specs=faces, face_skips=skips,
        face_joins=joins, reward_vec=np.zeros(len(envmod.REWARD_NAMES)),
        face_before=None, face_after={}, counts_from_trace=False)
    plan = dict(envmod._PLAN_RECORDS[0])
    envmod._PLAN_RECORDS.clear()
    plan["episode"] = 4

    jobs = _grad_oracle_jobs([plan])
    assert jobs == [{"order": tuple(int(v) for v in order),
                     "plan_hashes": [plan["plan_hash"]]}]

    oracle = AsyncGradOracle(lambda o, s, e: ("pass", 7e-15), timeout_s=60.0)
    path = str(tmp_path / "plan_log.jsonl")
    try:
        oracle.submit(4, 31337, jobs)
        assert _wait_for(lambda: oracle._done.qsize() > 0)
        _grad_oracle_boundary(oracle, path, 6, 1e-3, log=lambda _l: None)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "oracle_result"
        assert row["plan_hash"] == plan["plan_hash"]
        assert row["episode"] == plan["episode"] == 4
        assert row["order"] == [int(v) for v in order]
        assert row["oracle"] == {
            "status": "pass", "rel_l2": 7e-15, "checked_at_episode": 6,
            "tol": 1e-3, "seconds": row["oracle"]["seconds"],
            "device": "cpu", "dtype": "float64", "error": None}
    finally:
        oracle.close()


def test_two_plans_with_one_order_share_one_check_and_get_one_record_each():
    """The oracle checks the ORDER. Two plans that differ only in their
    approximation rules are one job and two late records."""
    from alphagrad.approx.ppo import _grad_oracle_jobs

    jobs = _grad_oracle_jobs([
        {"order": list(_ORDER_A), "plan_hash": _HASH_A},
        {"order": list(_ORDER_A), "plan_hash": _HASH_B},
        {"order": list(_ORDER_B), "plan_hash": "cc" * 16},
    ])
    assert len(jobs) == 2
    by_order = {j["order"]: j["plan_hashes"] for j in jobs}
    assert by_order[_ORDER_A] == [_HASH_A, _HASH_B]
    assert by_order[_ORDER_B] == ["cc" * 16]

    recs = [result_record(
        {"episode": 1, "order": _ORDER_A, "status": "pass", "rel_l2": 1e-14,
         "seconds": 0.5, "checked_at_episode": 2, "error": None},
        plan_hash=h, tol=1e-3) for h in by_order[_ORDER_A]]
    assert [r["plan_hash"] for r in recs] == [_HASH_A, _HASH_B]


def test_a_record_without_a_hash_or_an_order_is_skipped_not_guessed():
    from alphagrad.approx.ppo import _grad_oracle_jobs

    assert _grad_oracle_jobs([
        {"order": list(_ORDER_A)},
        {"plan_hash": _HASH_A},
        {},
        "not a record",
    ]) == []


# ===========================================================================
# (e) THE GATE ARM: --grad-oracle off CHANGES NOTHING
# ===========================================================================
def test_the_boundary_is_a_no_op_without_an_oracle(tmp_path):
    """Under `--grad-oracle off` the oracle object is None and every call site
    is inert -- no file is touched, nothing raises, nothing is logged."""
    from alphagrad.approx.ppo import _grad_oracle_boundary

    path = tmp_path / "plan_log.jsonl"
    assert _grad_oracle_boundary(None, str(path), 3, 1e-3) == []
    assert not path.exists()


def test_the_oracle_is_built_only_when_the_flag_is_on():
    """The object, the thread and the wandb keys all hang off one condition."""
    import alphagrad.approx.ppo as ppo

    src = inspect.getsource(ppo.main)
    assert 'if args.grad_oracle != "off":' in src
    build = src[src.index('if args.grad_oracle != "off":'):]
    build = build[:build.index("def _oracle_prune(")]
    assert "AsyncGradOracle(" in build
    # And the wandb keys are inside `if _GRAD_ORACLE is not None`.
    log = src[src.index("if _GRAD_ORACLE is not None:"):]
    assert 'log_dict[f"oracle/{_or_k}"]' in log[:600]


def test_the_measurement_callback_no_longer_runs_the_oracle():
    """THE ORACLE LEFT THE SCORING PATH. `_callback_measured` compiles, times
    and scores; it does not check gradients, and `refused/oracle` therefore has
    no producer left."""
    import alphagrad.approx.env as envmod

    src = inspect.getsource(envmod._callback_measured)
    assert "_grad_oracle_check(" not in src
    assert "_grad_oracle_rel_l2(" not in src
    assert "grad_oracle_cpu_check(" not in src
    assert not hasattr(envmod, "_grad_oracle_check"), (
        "the synchronous in-callback oracle is gone; a leftover copy is a "
        "second implementation of the check")
    assert not hasattr(envmod, "_LAST_RAISE_SOURCE")
    whole = inspect.getsource(envmod._callback)
    assert '"oracle"' not in whole, (
        "no path may still label a refusal as the oracle's")
