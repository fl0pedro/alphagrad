# dsnn-dfw.301: the trainer joins each plan record to its environment by (episode, env_index), with
# plan_hash as the fallback. A record with measured_from takes the numbers of the plan it names.
# Until 59d5d0d the join was the elimination order alone. On a fixed-order arm every plan of an
# episode has the same order, so every environment took the latency, memory and quality of the
# first record.
from __future__ import annotations

import inspect
import math
import os
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ALPHAGRAD_SKIP_COST_ANALYSIS", "1")
os.environ.setdefault("ALPHAGRAD_SKIP_COUNT_OPS", "1")

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

from alphagrad.approx import env as env_mod                     # noqa: E402
from alphagrad.approx import ppo                                # noqa: E402
from alphagrad.approx.env import NUM_REWARDS, REWARD_INDEX      # noqa: E402

REF_P50, WM = 2.0e5, 1.6e6
SPREAD = (0.8, 0.9, 1.0, 1.1, 1.3)
ORDER = [4, 7, 5]


def _args():
    return ppo.make_argparser().parse_args(
        ["--cmp-type", "latency", "--mem-type", "peak_memory", "--cost-form",
         "paired-log"])


def _record(env_index, plan_hash, lat_x, mem_x, quality, pid=11, **extra):
    rec = {"order": list(ORDER), "plan_hash": plan_hash, "pid": pid,
           "rewards": [0.0] * NUM_REWARDS, "measured_from": None,
           "mem_channel": "watermark", "mem_peak_source": "runtime_delta",
           "mem_watermark_bytes": mem_x * WM, "ref_watermark_bytes": WM,
           "ref_latency_p50_ns": REF_P50,
           "ratio_log": {"latency": {"windows": [math.log(lat_x * s) for s in SPREAD]},
                         "memory": {"windows": [math.log(mem_x)] * 5}}}
    rec["rewards"][REWARD_INDEX["quality"]] = quality
    for p, s in zip((10, 25, 50, 75, 90), SPREAD):
        rec[f"candidate_latency_p{p}_ns"] = lat_x * s * REF_P50
    if env_index is not None:
        rec["env_index"] = env_index
    rec.update(extra)
    return rec


def _dedupe(env_index, plan_hash, pid=11):
    # A plan the actor had already measured this episode: no timing of its own.
    return {"order": list(ORDER), "plan_hash": plan_hash, "pid": pid,
            "env_index": env_index, "measured_from": 0, "ratio_log": None,
            "rewards": [0.0] * NUM_REWARDS}


def test_two_environments_with_one_order_each_get_their_own_numbers():
    args = _args()
    # Env 1 is slower and smaller, so the front keeps both plans.
    recs = [_record(0, "aa", 0.50, 1.00, 0.99), _record(1, "bb", 0.90, 0.80, 0.80)]
    dists, counts = ppo._ratio_dists(recs, args, 3, 2)
    assert sorted(dists) == [0, 1] and counts["by_env_index"] == 2
    (s0, m0, q0, d0), (s1, m1, q1, d1) = dists[0], dists[1]
    assert np.median(s0["latency"]) == pytest.approx(math.log(0.50))
    assert np.median(s1["latency"]) == pytest.approx(math.log(0.90))
    assert s0["peak_memory"] != s1["peak_memory"] and (m0, m1) == ("watermark", "watermark")
    assert s0["quality"] == [0.99] and s1["quality"] == [0.80]
    assert q0["latency"]["p10"] == pytest.approx(math.log(0.40))
    assert q1["latency"]["p10"] == pytest.approx(math.log(0.72))
    assert d0["candidate_latency_ns"] != d1["candidate_latency_ns"]
    # Through the front: two points, each with the numbers of its own plan.
    front = ppo._band_archive(args)
    for e in (0, 1):
        smp, src, q, d = dists[e]
        front.add(smp, [[v, [e]] for v in ORDER], 3, mem_source=src, quantiles=q, detail=d)
    assert sorted(round(float(p[0]), 6) for p in front.pts) == [
        round(math.log(0.50), 6), round(math.log(0.90), 6)]


def _states(orders, faces_on):
    from alphagrad.approx.env import FACE_SLOTS, MAX_FACES, MAX_RULES_PER_VERTEX
    n_env, n = len(orders), len(orders[0])
    specs = np.full((n_env, n, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((n_env, n, MAX_FACES, FACE_SLOTS, 3), -1, np.int32)
    for e, on in enumerate(faces_on):
        if on:
            faces[e, 0, 0, 0] = (0, 1, 2)
    skips = np.zeros((n_env, n, MAX_FACES), np.int32)
    return types.SimpleNamespace(order=np.asarray(orders, np.int32), sparsity_specs=specs,
                                 face_specs=faces, face_skips=skips, face_joins=None)


def test_a_record_without_env_index_joins_by_plan_hash():
    states = _states([ORDER, ORDER], faces_on=[False, True])
    hashes = ppo._plan_hashes(states, 2)
    assert hashes[0] != hashes[1], "one order, two approximations, two plans"
    recs = [_record(None, hashes[1], 0.90, 1.20, 0.80),
            _record(None, hashes[0], 0.50, 1.00, 0.99)]
    rows, counts = ppo._join_plan_records(recs, 3, 2, states)
    assert rows[0] is recs[1] and rows[1] is recs[0]
    assert counts["by_plan_hash"] == 2 and counts["unjoined"] == 0
    # Without the end states there is nothing to match, and the records are counted.
    rows, counts = ppo._join_plan_records(recs, 3, 2, None)
    assert rows == {} and counts["unjoined"] == 2


def test_a_record_with_measured_from_takes_the_numbers_of_the_plan_it_names():
    timed = _record(0, "aa", 0.50, 1.00, 0.99, pid=11)
    other = _record(2, "aa", 0.70, 1.00, 0.99, pid=12)
    recs = [other, timed, _dedupe(1, "aa", pid=11), _dedupe(3, "zz", pid=11)]
    rows, counts = ppo._join_plan_records(recs, 3, 4)
    assert rows[1] is timed, "the plan measured by the same actor"
    assert rows[0] is timed and rows[2] is other
    assert 3 not in rows and counts["measured_from"] == 2 and counts["unresolved"] == 1


def test_the_join_counts_every_record_it_does_not_place():
    recs = [_record(0, "aa", 0.5, 1.0, 0.9), _record(0, "bb", 0.6, 1.0, 0.9),
            _record(1, "cc", 0.5, 1.0, 0.9, episode=2), _record(5, "dd", 0.5, 1.0, 0.9)]
    rows, counts = ppo._join_plan_records(recs, 3, 2)
    assert list(rows) == [0] and rows[0] is recs[0]
    assert (counts["duplicate"], counts["other_episode"], counts["unjoined"]) == (1, 1, 1)


def _toy_env():
    from alphagrad.approx.env import VertexEliminationEnv

    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.standard_normal((16, 16), dtype=np.float32) / 4.0)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 16, dtype=np.float32))

    def toy(v):
        return jnp.sum(jnp.tanh(W @ v) ** 2)

    return VertexEliminationEnv.from_jaxpr(
        jax.make_jaxpr(toy)(x), args=[x], argnums=(0,), num_envs=0,
        target_fun=toy, measure_latency=True, terminal_rewards_only=True,
        latency_inner_reps=1)


def test_the_trainer_hash_of_a_plan_is_the_hash_its_record_carries(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_QUALITY_METRIC", "none")
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "0")
    env = _toy_env()
    order = sorted(int(v) for v in np.asarray(env.valid_vertices))
    states = _states([order], faces_on=[False])
    env_mod.consume_plan_records()
    env_mod._callback(
        env.config, env.args, env.consts, jnp.asarray(states.order[0]),
        jnp.asarray(states.sparsity_specs[0]), jnp.asarray(states.face_specs[0]),
        jnp.asarray(states.face_skips[0]), len(order),
        *[jnp.asarray(s) for s in (np.stack([np.linspace(-1.0, 1.0, 16, dtype=np.float32)]),)])
    recs = env_mod.consume_plan_records()["records"]
    assert recs and recs[-1]["plan_hash"] == ppo._plan_hashes(states, 1)[0]


def test_the_trainer_joins_by_environment_row_at_every_site():
    src = inspect.getsource(ppo.main)
    assert "_rdists.get(_key)" not in src and "_bands.get(_key)" not in src
    assert "_rd, _rd_join = _ratio_dists(" in src and "_dist = _rdists.get(i)" in src
    assert "_join_plan_records(_recs, ep, num_envs, _es)" in src
    assert "_band = _bands.get(i)" in src
    assert 'plan_states=_fe.get("plan_states")' in src
    assert "plan_states=_end_states" in src and "plan_states=_wend" in src
