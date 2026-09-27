"""THE READOUT'S HOST HALF (owner ruling 2026-09-26 Q2 c, dsnn-dfw.291).

`common/readout.py` is host arithmetic: the rollouts a readout takes, what a
plan's measurement came to, the records, the wandb summary, the file, and the
checkpoint a run directory is read out at. `checkpoint.check_readout_args` is
the namespace check a readout of a checkpoint makes. And the argmax plan:
under `ppo.argmax_key()` every draw the policy makes is its argmax, pinned
here on the JAX and distrax samplers the heads call and on the face head.

The end-to-end readout, through the trainer and through tools/readout.py, is
tests/readout_test.py.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pytest

from alphagrad.approx.common import checkpoint as ckpt
from alphagrad.approx.common import readout as ro


# ---------------------------------------------------------------- the rollouts

def test_n_plans_take_n_over_e_sampled_rollouts_and_one_argmax_rollout():
    assert ro.schedule(64, 16) == [ro.SAMPLED] * 4 + [ro.ARGMAX]
    assert ro.schedule(2, 2) == [ro.SAMPLED, ro.ARGMAX]


@pytest.mark.parametrize("n,e", [(3, 2), (60, 16), (0, 16), (16, 0)])
def test_a_plan_count_that_no_whole_rollout_count_measures_raises(n, e):
    with pytest.raises(ValueError):
        ro.schedule(n, e)


# ------------------------------------------------------------------ the records

_PROV = {"checkpoint": "/run/ppo_ckpt_ep000002000", "checkpoint_episode": 2000,
         "seed": 250197, "run_name": "C_nn256_s250197",
         "params_sha256": "ab" * 32}


_MEASURED = object()


def _rec(draw, index, *, plan_hash, quality=0.9, lat=-0.2, mem=-0.1,
         plan_record=_MEASURED, dropped=False):
    if plan_record is _MEASURED:
        plan_record = {"plan_hash": plan_hash}
    return ro.record(
        draw=draw, index=index, rollout=index // 2, env=index % 2,
        provenance=_PROV, plan_hash=plan_hash, plan={"seq": [[1, []]]},
        rewards=[0.0] * 6 + [quality] + [0.0] * 5, quality=quality,
        numbers={"latency_log_ratio": lat, "memory_log_ratio": mem,
                 "memory_source": "watermark", "latency_quantiles": None},
        plan_record=plan_record, dropped=dropped)


def test_a_plan_status_is_measured_failed_dropped_or_missing():
    assert ro.plan_status({"plan_hash": "a"}, False) == "measured"
    assert ro.plan_status({"refused": "oom: 2 GiB"}, False) == "failed"
    assert ro.plan_status({"refused": "quality-undefined:grad_cosine"},
                          True) == "dropped"
    assert ro.plan_status(None, False) == "missing"
    assert set(ro.STATUSES) == {"measured", "failed", "dropped", "missing"}


def test_every_record_is_marked_and_carries_its_plan_record():
    r = _rec(ro.SAMPLED, 0, plan_hash="a", plan_record={"plan_hash": "a",
                                                        "pid": 7})
    assert r["record"] == ro.RECORD == "readout"
    assert r["draw"] == ro.SAMPLED and r["index"] == 0
    assert r["plan_record"] == {"plan_hash": "a", "pid": 7}
    for k, v in _PROV.items():
        assert r[k] == v
    json.dumps(r, allow_nan=False)


def test_a_failed_plan_keeps_its_scores_and_a_dropped_one_has_no_quality():
    failed = _rec(ro.SAMPLED, 0, plan_hash="f", quality=0.0, lat=5.9,
                  plan_record={"refused": "deadline:300s"})
    assert failed["status"] == "failed"
    assert failed["quality"] == 0.0 and failed["latency_log_ratio"] == 5.9
    assert failed["refused"] == "deadline:300s"
    dropped = _rec(ro.SAMPLED, 1, plan_hash="d", quality=-1e10,
                   plan_record={"refused": "quality-undefined:grad_cosine"},
                   dropped=True)
    assert dropped["status"] == "dropped" and dropped["quality"] is None
    missing = _rec(ro.SAMPLED, 2, plan_hash="m", quality=0.7, lat=None,
                   mem=None, plan_record=None)
    assert missing["status"] == "missing" and missing["quality"] == 0.7
    assert missing["plan_record"] is None and missing["refused"] is None


def test_the_summary_counts_the_sample_and_names_the_argmax_plan():
    recs = [
        _rec(ro.SAMPLED, 0, plan_hash="a", quality=0.95, lat=-0.4, mem=-0.2),
        _rec(ro.SAMPLED, 1, plan_hash="a", quality=0.93, lat=-0.3, mem=-0.1),
        _rec(ro.SAMPLED, 2, plan_hash="b", quality=0.0, lat=5.0, mem=0.5,
             plan_record={"refused": "oom: card"}),
        _rec(ro.SAMPLED, 3, plan_hash="c", quality=-1e10,
             plan_record={"refused": "quality-undefined:grad_cosine"},
             dropped=True),
        _rec(ro.SAMPLED, 4, plan_hash="d", quality=0.5, lat=None, mem=None,
             plan_record=None),
        _rec(ro.ARGMAX, 5, plan_hash="a", quality=0.94, lat=-0.35, mem=-0.15),
    ]
    s = ro.summary(recs, quality_floor=0.9)
    assert s["readout/plans"] == 6 and s["readout/sampled"] == 5
    assert (s["readout/measured"], s["readout/failed"], s["readout/dropped"],
            s["readout/missing"]) == (2, 1, 1, 1)
    assert s["readout/distinct"] == 4
    assert s["readout/argmax_in_sample"] is True
    # A failed plan is scored at its failed-plan values; a dropped one is not scored.
    assert s["readout/quality_median"] == pytest.approx(0.93)
    assert s["readout/latency_log_ratio_median"] == pytest.approx(-0.3)
    assert s["readout/feasible_fraction"] == pytest.approx(2 / 3)
    assert s["readout/argmax/quality"] == pytest.approx(0.94)
    assert s["readout/argmax/status"] == "measured"
    assert s["readout/argmax/plan_hash"] == "a"
    assert s["readout/checkpoint_episode"] == 2000
    assert s["readout/seed"] == 250197
    rows = ro.table_rows(recs)
    assert len(rows) == 6 and len(rows[0]) == len(ro.TABLE_COLUMNS)


def test_a_summary_without_exactly_one_argmax_plan_raises():
    with pytest.raises(ValueError):
        ro.summary([_rec(ro.SAMPLED, 0, plan_hash="a")])


def test_a_readout_never_overwrites_another_and_reads_back(tmp_path):
    recs = [_rec(ro.SAMPLED, 0, plan_hash="a"),
            _rec(ro.ARGMAX, 1, plan_hash="b")]
    path = str(tmp_path / ro.READOUT_FILE)
    ro.write_records(path, recs)
    assert ro.load_records(path) == recs
    with pytest.raises(FileExistsError):
        ro.write_records(path, recs)
    other = tmp_path / "plan_log.jsonl"
    other.write_text(json.dumps({"plan_hash": "a"}) + "\n")
    with pytest.raises(ValueError):
        ro.load_records(str(other))


# ------------------------------------------------------------ the checkpoint

def _ckpt_dir(run_dir, episode):
    d = os.path.join(run_dir, ckpt.checkpoint_dir_name(episode))
    os.makedirs(d)
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump({"format": ckpt.PPO_CKPT_FORMAT, "episode": episode}, fh)
    return d


def test_a_checkpoint_directory_is_read_out_as_named(tmp_path):
    d = _ckpt_dir(str(tmp_path), 50)
    assert ro.resolve_checkpoint(d, 1000) == d


def test_a_run_directory_is_read_out_at_its_final_checkpoint(tmp_path):
    _ckpt_dir(str(tmp_path), 950)
    last = _ckpt_dir(str(tmp_path), 1000)
    assert ro.resolve_checkpoint(str(tmp_path), 1000) == last


def test_a_run_that_did_not_finish_has_no_final_checkpoint(tmp_path):
    _ckpt_dir(str(tmp_path), 950)
    with pytest.raises(ckpt.CheckpointError, match="did not finish"):
        ro.resolve_checkpoint(str(tmp_path), 1000)
    with pytest.raises(ckpt.CheckpointError):
        ro.resolve_checkpoint(str(tmp_path / "nowhere"), 1000)


def test_an_auto_stopped_run_ends_where_it_stopped(tmp_path):
    last = _ckpt_dir(str(tmp_path), 500)
    with open(tmp_path / "auto_stop.json", "w") as fh:
        json.dump({"check_point": 500}, fh)
    assert ro.resolve_checkpoint(str(tmp_path), 1000) == last


def _ns(**kw):
    base = {"seed": 250197, "episodes": 2000, "lr": 3e-4, "readout": 0,
            "resume": "", "gpus": "0", "name": "run"}
    base.update(kw)
    return argparse.Namespace(**base)


def test_a_readout_may_differ_from_its_run_in_readout_alone():
    saved = ckpt.args_to_json(_ns())
    ckpt.check_readout_args(saved, _ns(readout=64))
    # A run launched before the readout existed carries no --readout.
    del saved["readout"]
    ckpt.check_readout_args(saved, _ns(readout=64, gpus="1"))


@pytest.mark.parametrize("change", [{"lr": 1e-3}, {"seed": 1}, {"name": "x"}])
def test_a_readout_of_another_configuration_raises(change):
    saved = ckpt.args_to_json(_ns())
    with pytest.raises(ckpt.CheckpointError, match="not a state of this run"):
        ckpt.check_readout_args(saved, _ns(readout=64, **change))


def test_an_argument_the_checkpoint_does_not_carry_raises():
    saved = ckpt.args_to_json(_ns())
    saved["an_argument_of_another_build"] = 1
    with pytest.raises(ckpt.CheckpointError):
        ckpt.check_readout_args(saved, _ns(readout=64))
    saved = ckpt.args_to_json(_ns())
    del saved["lr"]
    with pytest.raises(ckpt.CheckpointError):
        ckpt.check_readout_args(saved, _ns(readout=64))


# ------------------------------------------------------------ the argmax plan

@pytest.fixture(scope="module")
def jx():
    import distrax
    import jax
    import jax.numpy as jnp
    import jax.random as jrand

    from alphagrad.approx.ppo import argmax_key
    return argparse.Namespace(jax=jax, jnp=jnp, jrand=jrand, distrax=distrax,
                              key=argmax_key)


def test_every_sampler_the_heads_call_draws_its_argmax(jx):
    jnp, jrand = jx.jnp, jx.jrand
    rng = np.random.default_rng(0)
    k = jx.key()
    # Split and fold_in stay argmax keys, so every key the rollout derives is one.
    keys = [k, jrand.split(k, 3)[1], jrand.fold_in(k, 7),
            jrand.split(jrand.fold_in(k, 2), 4)[3]]
    for t in range(300):
        kk = keys[t % len(keys)]
        n = int(rng.integers(2, 12))
        logits = (rng.normal(size=n) * rng.choice([0.01, 1.0, 30.0])
                  ).astype(np.float32)
        live = rng.random(n) < 0.7
        live[int(rng.integers(n))] = True
        masked = np.where(live, logits, -np.inf).astype(np.float32)
        want = int(np.argmax(masked))
        assert int(jrand.categorical(kk, jnp.asarray(masked))) == want
        probs = jx.jax.nn.softmax(jnp.asarray(masked))
        assert int(jx.distrax.Categorical(probs=probs).sample(seed=kk)) == \
            int(np.argmax(np.asarray(probs)))
        p = float(rng.random())
        assert bool(jrand.uniform(kk) < p) == (p > 0.5)
        assert bool(jrand.bernoulli(kk, p)) == (p > 0.5)


def test_the_argmax_holds_under_jit_and_vmap(jx):
    jax, jnp, jrand = jx.jax, jx.jnp, jx.jrand
    logits = jnp.asarray(np.random.default_rng(1).normal(size=(8, 9)),
                         jnp.float32)
    draw = jax.jit(jax.vmap(lambda kk, lg: jrand.categorical(kk, lg)))
    got = draw(jrand.split(jx.key(), 8), logits)
    assert np.array_equal(np.asarray(got), np.asarray(jnp.argmax(logits, -1)))


def test_the_face_head_draws_its_argmax_decision(jx):
    """Every field of UnifiedFaceHead.sample, against the argmax of its logits.

    The choose layout carries every field there is: the skip, the quant bit,
    the join bit, and op / i / j / axis / reduce_fn per slot, with j masked
    by the drawn i and a quantised operand slot's op forced to none.
    """
    import jax.nn as jnn

    from alphagrad.approx import unified_face_head as ufh
    jnp, jrand = jx.jnp, jx.jrand
    head = ufh.UnifiedFaceHead(16, key=jrand.PRNGKey(3), approx_add="choose")
    lay = head.layout
    n = lay.n_slots
    rng = np.random.default_rng(2)

    def _masked_argmax(z, m):
        z = np.where(np.asarray(m) > 0.5, np.asarray(z), -np.inf)
        return 0 if np.all(np.isinf(z)) else int(np.argmax(z))

    seen_skip, seen_quant = set(), set()
    for t in range(40):
        ctx = jnp.asarray(rng.normal(size=16) * 3.0, jnp.float32)
        masks = {name: jnp.asarray(
                     (rng.random((n, width)) < 0.7).astype(np.float32))
                 for name, width in (("op_mask", ufh.NUM_APPROX_OPS),
                                     ("i_mask", ufh.MAX_PAIR_IDX),
                                     ("j_mask", ufh.MAX_PAIR_IDX),
                                     ("axis_mask", ufh.NUM_REDUCE_AXES))}
        _, f, _, _, _ = head.sample(ctx, jrand.fold_in(jx.key(), t), **masks)
        z = np.asarray(head.logits(ctx))
        likelier = lambda logit: bool(  # noqa: E731
            0.5 < float(jnn.sigmoid(jnp.float32(logit))))
        skip = int(likelier(z[ufh.O_SKIP]))
        quant = int(likelier(z[ufh.O_QUANT])) * (1 - skip)
        assert int(f.skip) == skip
        assert int(f.quant) == quant
        assert int(f.join) == int(likelier(z[lay.choose_index]))
        seen_skip.add(skip)
        seen_quant.add(quant)
        for s in range(n):
            b = lay.slot_base(s)
            op = _masked_argmax(z[b + ufh.S_OP:b + ufh.S_I],
                                masks["op_mask"][s])
            if s in ufh.QUANT_SLOTS and quant:
                op = ufh.OP_NONE
            i = _masked_argmax(z[b + ufh.S_I:b + ufh.S_J], masks["i_mask"][s])
            jm = ufh.j_mask_given_i(jnp.int32(i), masks["j_mask"][s])
            assert int(f.op[s]) == op
            assert int(f.i[s]) == i
            assert int(f.j[s]) == _masked_argmax(
                z[b + ufh.S_J:b + ufh.S_AXIS], jm)
            assert int(f.axis[s]) == _masked_argmax(
                z[b + ufh.S_AXIS:b + ufh.S_RFN], masks["axis_mask"][s])
            assert int(f.reduce_fn[s]) == int(np.argmax(
                z[b + ufh.S_RFN:b + ufh.SLOT_WIDTH]))
    # Both values of both bits occurred, so neither comparison is vacuous.
    assert seen_skip == {0, 1} and seen_quant == {0, 1}
