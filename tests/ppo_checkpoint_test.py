"""THE PPO TRAINER'S EXACT CHECKPOINT: every field, round-tripped.

The contract these tests pin is the one `--resume` rests on: a checkpoint
written at a quiescent point of the episode pipeline and read back gives the
SAME state, bit for bit, and any way in which it cannot raises instead of
returning a state that is nearly right.

Three groups:

* the arithmetic half (`state.eqx`): parameters, optimiser state, PopArt,
  the duals, the RNG key, the step counter. Compared as BIT PATTERNS, not
  with a tolerance -- a checkpoint that moved a parameter by one ULP would
  make a resumed run diverge from the run it continues, slowly and silently.
* the bookkeeping half (`meta.json`): the Pareto archive, the two bin
  policies, the top-N heaps and the best-so-far record, the wandb run id and
  the whole argument namespace.
* the refusals: an argument that differs, an `--episodes` that shrinks, a
  format that does not match, a directory that is not a checkpoint, a
  non-array leaf that changed, and a bin policy configured differently.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
import pytest

from alphagrad.approx.common import checkpoint as ckpt


# ---------------------------------------------------------------------------
# A stand-in for the agent: an equinox module with float32 and bf16 leaves and
# one non-array leaf, which is what the real agent looks like to equinox.
# ---------------------------------------------------------------------------
class _Tiny(eqx.Module):
    w: jax.Array
    b: jax.Array
    half: jax.Array
    tag: str

    def __init__(self, key):
        k1, k2 = jrand.split(key)
        self.w = jrand.normal(k1, (4, 3), dtype=jnp.float32)
        self.b = jrand.normal(k2, (3,), dtype=jnp.float32)
        self.half = jnp.asarray([1.5, -2.25], dtype=jnp.bfloat16)
        self.tag = "tiny"


def _bits(x):
    """The raw bytes of an array leaf. Equality here is bit equality."""
    a = np.asarray(x)
    return (str(a.dtype), tuple(a.shape), a.tobytes())


def _tree_bits(tree):
    return [_bits(leaf) for leaf in jax.tree_util.tree_leaves(tree)
            if isinstance(leaf, (jax.Array, np.ndarray))]


def _example_tree(seed=0, *, with_probes=True):
    key = jrand.PRNGKey(seed)
    k_agent, k_probe, k_run = jrand.split(key, 3)
    agent = _Tiny(k_agent)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    opt_state = opt.init(eqx.filter(agent, eqx.is_inexact_array))
    probes = _Tiny(k_probe) if with_probes else None
    probe_opt = optax.adam(1e-3)
    probe_opt_state = (probe_opt.init(eqx.filter(probes, eqx.is_inexact_array))
                       if with_probes else None)
    return {
        "agent": agent,
        "opt_state": opt_state,
        "probes": probes,
        "probe_opt_state": probe_opt_state,
        "vprobes": None,
        "vprobe_opt_state": None,
        "popart_m1": jnp.asarray([0.125, -3.5, 7.0], dtype=jnp.float32),
        "popart_m2": jnp.asarray([1.0, 12.25, 49.5], dtype=jnp.float32),
        "popart_w": jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float32),
        "global_step": jnp.asarray(1234, dtype=jnp.int32),
        "key": k_run,
        # A float whose decimal form is not exact, so a lossy path shows up.
        "lag_lambda": 0.1 + 0.2,
        "kl_ref_coef": 1.0 / 3.0,
    }


def _example_meta(**over):
    meta = {
        "args": {"episodes": 50, "seed": 42, "name": "unit", "resume": "",
                 "checkpoint_every": 25, "lr": 0.0003},
        "wandb_run_id": "abc123",
        "pareto_archive": {
            "obj_names": ["flops", "peak_memory", "cosine_sim"],
            "obj_idx": [0, 1, 6],
            "quality_floor": None,
            "pts": [], "seqs": [], "eps": [], "all_candidates": [],
            "seen": [], "hv_ref": None,
        },
        "episode_bin": {"log2": 12, "last_used": 11, "overflowed": False,
                        "recent": [10, 20], "initial": 11, "window": 4,
                        "cap": 20, "floor": 0},
        "window_bin": {"log2": 11, "last_used": 11, "overflowed": True,
                       "recent": [5], "initial": 11, "window": 4,
                       "cap": 15, "floor": 10},
        "host_state": {"samplecounts": 7, "best_global_return": -1.5,
                       "best_global_act_seq": None,
                       "top_n_total": [], "top_n_cmp": [],
                       "top_n_mem": [], "top_n_acc": []},
    }
    meta.update(over)
    return meta


# ---------------------------------------------------------------------------
# 1. The arithmetic half.
# ---------------------------------------------------------------------------

def test_every_arithmetic_field_comes_back_bit_for_bit(tmp_path):
    tree = _example_tree(seed=3)
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=25, tree=tree, meta=_example_meta())

    # A template built independently of the saved values: every array leaf is
    # a DIFFERENT number, so a field the loader forgot shows up as a mismatch
    # instead of matching by accident.
    template = _example_tree(seed=99)
    assert _tree_bits(template) != _tree_bits(tree)

    back = ckpt.load_ppo_tree(path, template)
    assert _tree_bits(back) == _tree_bits(tree)

    # And named field by named field, so a renamed key cannot pass.
    for name in tree:
        if name in ("lag_lambda", "kl_ref_coef"):
            continue
        assert _tree_bits(back[name]) == _tree_bits(tree[name]), name


def test_the_two_host_side_duals_round_trip_exactly(tmp_path):
    tree = _example_tree()
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=1, tree=tree, meta=_example_meta())
    back = ckpt.load_ppo_tree(path, _example_tree(seed=99))
    assert back["lag_lambda"] == tree["lag_lambda"]
    assert back["kl_ref_coef"] == tree["kl_ref_coef"]
    assert isinstance(back["lag_lambda"], float)
    assert isinstance(back["kl_ref_coef"], float)
    # 0.1 + 0.2 is not 0.3. A path that went through a decimal string with
    # fewer digits would return 0.3 here and look fine.
    assert back["lag_lambda"] != 0.3


def test_the_rng_key_round_trips_as_the_same_words(tmp_path):
    tree = _example_tree(seed=5)
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=2, tree=tree, meta=_example_meta())
    back = ckpt.load_ppo_tree(path, _example_tree(seed=99))
    assert np.array_equal(np.asarray(back["key"]), np.asarray(tree["key"]))
    # The key is the whole point: the next split has to give the same subkey.
    a = jrand.split(back["key"], 3)
    b = jrand.split(tree["key"], 3)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_an_absent_probe_tree_round_trips_as_absent(tmp_path):
    tree = _example_tree(with_probes=False)
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=3, tree=tree, meta=_example_meta())
    back = ckpt.load_ppo_tree(path, _example_tree(seed=9, with_probes=False))
    assert back["probes"] is None
    assert back["probe_opt_state"] is None


def test_a_tree_whose_structure_changed_raises(tmp_path):
    tree = _example_tree(with_probes=True)
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=4, tree=tree, meta=_example_meta())
    with pytest.raises(Exception):
        ckpt.load_ppo_tree(path, _example_tree(with_probes=False))


def test_a_changed_non_array_leaf_raises_instead_of_being_kept(tmp_path):
    """equinox does not WRITE a non-array leaf, so a load would silently keep
    the template's value for it. The manifest is what refuses that."""
    tree = _example_tree()
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=5, tree=tree, meta=_example_meta())
    other = _example_tree(seed=99)
    other["agent"] = eqx.tree_at(lambda m: m.tag, other["agent"],
                                 "something-else",
                                 is_leaf=lambda x: isinstance(x, str))
    with pytest.raises(ckpt.CheckpointError, match="non-array leaves"):
        ckpt.load_ppo_tree(path, other)


def test_the_manifest_does_not_depend_on_an_address():
    """Two PROCESSES must agree on the manifest, so it can carry no address.
    A function leaf is the case that broke: its repr holds one."""
    tree = {"fn": (lambda: None), "x": jnp.zeros((2,))}
    tags = [t for _, t in ckpt._non_array_manifest(tree)]
    assert tags, "a function leaf is not array-like and must be listed"
    assert all("0x" not in t for t in tags), tags


# ---------------------------------------------------------------------------
# 2. The bookkeeping half.
# ---------------------------------------------------------------------------

def test_the_meta_half_round_trips_every_field(tmp_path):
    meta = _example_meta()
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=25, tree=_example_tree(), meta=meta)
    back = ckpt.read_ppo_meta(path)
    assert back["episode"] == 25
    assert back["format"] == ckpt.PPO_CKPT_FORMAT
    for name, value in meta.items():
        assert back[name] == value, name


def test_the_pareto_archive_round_trips_points_sequences_and_episodes():
    from alphagrad.approx.common.pareto_archive import ParetoArchive

    def _fresh():
        return ParetoArchive(obj_names=("flops", "peak_memory", "cosine_sim"),
                             obj_idx=(0, 1, 2), quality_floor=None)

    live = _fresh()
    rng = np.random.default_rng(0)
    for ep in range(6):
        vec = rng.normal(size=8)
        live.add(vec, [f"vertex-{ep}", ep], ep)
    assert live.pts, "the fixture admitted nothing; it would test nothing"

    doc = json.loads(json.dumps(ckpt.pareto_archive_to_json(live)))
    back = _fresh()
    ckpt.pareto_archive_from_json(back, doc)

    assert len(back.pts) == len(live.pts)
    for a, b in zip(back.pts, live.pts):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert back.seqs == live.seqs
    assert back.eps == live.eps
    assert back.all_candidates == live.all_candidates
    assert back._seen == live._seen
    if live._hv_ref is None:
        assert back._hv_ref is None
    else:
        assert np.array_equal(back._hv_ref, live._hv_ref)


def test_the_archive_refuses_to_load_a_different_objective_set():
    from alphagrad.approx.common.pareto_archive import ParetoArchive

    live = ParetoArchive(obj_names=("flops", "peak_memory", "cosine_sim"),
                         obj_idx=(0, 1, 2))
    doc = ckpt.pareto_archive_to_json(live)
    other = ParetoArchive(obj_names=("flops", "peak_memory", "loss_drop"),
                          obj_idx=(0, 1, 2))
    with pytest.raises(ckpt.CheckpointError, match="objectives"):
        ckpt.pareto_archive_from_json(other, doc)


def test_both_bin_policies_round_trip_their_whole_state():
    from alphagrad.approx.common import episode_stream as epstream

    live = epstream.BinPolicy(11, history=4, margin=1.25, cap=20)
    live.record(900)
    live.record(1800)
    live.record_overflow(5000)
    live.pick()
    saved = json.loads(json.dumps(ckpt.bin_policy_to_json(live)))

    back = epstream.BinPolicy(11, history=4, margin=1.25, cap=20)
    ckpt.bin_policy_from_json(back, saved)
    assert back.log2 == live.log2
    assert back.last_used == live.last_used
    assert back.overflowed == live.overflowed
    assert list(back.recent) == list(live.recent)
    # The point of restoring it: the NEXT pick has to agree.
    assert back.pick() == live.pick()


def test_a_bin_policy_built_with_a_different_history_refuses_the_state():
    from alphagrad.approx.common import episode_stream as epstream

    live = epstream.BinPolicy(11, history=4, margin=1.25, cap=20)
    saved = ckpt.bin_policy_to_json(live)
    other = epstream.BinPolicy(11, history=8, margin=1.25, cap=20)
    with pytest.raises(ckpt.CheckpointError, match="window"):
        ckpt.bin_policy_from_json(other, saved)


def test_the_top_n_heaps_and_the_best_so_far_round_trip():
    live = {
        "samplecounts": 190,
        "collapsed_total": 3,
        "best_global_return": -1.2345678901234567e-8,
        "best_global_act_seq": ["v1", "v2"],
        "top_n_total": [(1.5, 2, [0.25, -0.5], ["a"]),
                        (0.5, 1, [1.0, 2.0], ["b"])],
        "top_n_cmp": [(-3.0, 0, [1.0], ["c"])],
        "top_n_mem": [],
        "top_n_acc": [],
    }
    doc = json.loads(json.dumps(ckpt.host_state_to_json(live)))
    back = {"_wall_t0": 12.0}
    for name in ("top_n_total", "top_n_cmp", "top_n_mem", "top_n_acc"):
        back[name] = []
    ckpt.host_state_from_json(back, doc)
    assert back["samplecounts"] == live["samplecounts"]
    assert back["collapsed_total"] == live["collapsed_total"]
    assert back["best_global_return"] == live["best_global_return"]
    assert back["best_global_act_seq"] == live["best_global_act_seq"]
    for name in ("top_n_total", "top_n_cmp", "top_n_mem", "top_n_acc"):
        assert back[name] == live[name], name
    # The wall-clock origin belongs to THIS process and is not restored.
    assert back["_wall_t0"] == 12.0
    assert "_wall_t0" not in doc


# ---------------------------------------------------------------------------
# 3. The refusals.
# ---------------------------------------------------------------------------

def _ns(**kw):
    base = dict(episodes=50, seed=42, name="unit", resume="",
                checkpoint_every=25, checkpoint_keep_at=[], lr=0.0003,
                live_faces=True, rewards=["cmp", "mem"])
    base.update(kw)
    return argparse.Namespace(**base)


def test_matching_arguments_are_accepted():
    saved = ckpt.args_to_json(_ns())
    ckpt.check_resume_args(saved, _ns(resume="/somewhere/ppo_ckpt_ep000000025"))


def test_a_different_checkpoint_keep_at_is_exempt():
    """dsnn-dfw.117: --checkpoint-keep-at only decides which OLD checkpoints
    survive pruning. It reads no state and steps no gradient, so a resume
    may change it freely, same as --episodes and --resume."""
    saved = ckpt.args_to_json(_ns(checkpoint_keep_at=[25]))
    ckpt.check_resume_args(
        saved, _ns(resume="/x", checkpoint_keep_at=[25, 50]))


def test_a_longer_episodes_is_accepted_and_a_shorter_one_is_not():
    saved = ckpt.args_to_json(_ns(episodes=50))
    ckpt.check_resume_args(saved, _ns(episodes=200, resume="/x"))
    with pytest.raises(ckpt.CheckpointError, match="cannot shorten"):
        ckpt.check_resume_args(saved, _ns(episodes=10, resume="/x"))


@pytest.mark.parametrize("field,value", [
    ("seed", 43),
    ("lr", 0.0004),
    ("checkpoint_every", 10),
    ("live_faces", False),
    ("rewards", ["cmp"]),
    ("name", "other"),
])
def test_any_other_changed_argument_raises(field, value):
    saved = ckpt.args_to_json(_ns())
    with pytest.raises(ckpt.CheckpointError, match=field):
        ckpt.check_resume_args(saved, _ns(resume="/x", **{field: value}))


def test_an_argument_that_only_one_side_has_raises():
    saved = ckpt.args_to_json(_ns())
    saved.pop("lr")
    with pytest.raises(ckpt.CheckpointError, match="lr"):
        ckpt.check_resume_args(saved, _ns(resume="/x"))

    saved = ckpt.args_to_json(_ns())
    saved["a_flag_this_build_lost"] = 1
    with pytest.raises(ckpt.CheckpointError, match="a_flag_this_build_lost"):
        ckpt.check_resume_args(saved, _ns(resume="/x"))


def test_an_argument_of_an_unknown_type_raises_rather_than_being_dropped():
    with pytest.raises(ckpt.CheckpointError, match="cannot be written"):
        ckpt.args_to_json(_ns(weird=object()))


# dsnn-dfw.274 (owner ruling 2026-09-26): other GPU numbers resume, another GPU model does not.
_BLACKWELL = {
    "device_kind": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "nvidia_smi_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation "
                       "Edition"}
_H100 = {"device_kind": "NVIDIA H100 80GB HBM3",
         "nvidia_smi_name": "NVIDIA H100 80GB HBM3"}


class _Device:
    def __init__(self, device_kind, platform):
        self.device_kind, self.platform = device_kind, platform


def test_a_resume_on_other_gpu_numbers_of_the_same_model_passes():
    saved = ckpt.args_to_json(_ns(gpus="4", measure_gpus="5,6,7"))
    ckpt.check_resume_args(saved, _ns(resume="/x", gpus="0",
                                      measure_gpus="1,2,3"))
    ckpt.check_resume_gpu_model({"trainer_gpu_model": dict(_BLACKWELL)},
                                dict(_BLACKWELL))
    ckpt.check_resume_gpu_model(
        {"trainer_gpu_model": dict(_BLACKWELL, nvidia_smi_name=None)},
        dict(_BLACKWELL))


def test_a_resume_on_another_gpu_model_raises_and_names_both():
    with pytest.raises(ckpt.CheckpointError) as e:
        ckpt.check_resume_gpu_model({"trainer_gpu_model": dict(_BLACKWELL)},
                                    dict(_H100))
    assert _BLACKWELL["device_kind"] in str(e.value), str(e.value)
    assert _H100["device_kind"] in str(e.value), str(e.value)
    with pytest.raises(ckpt.CheckpointError, match="RTX 6000 Ada"):
        ckpt.check_resume_gpu_model(
            {"trainer_gpu_model": dict(_BLACKWELL)},
            dict(_BLACKWELL, nvidia_smi_name="NVIDIA RTX 6000 Ada Generation"))


def test_a_checkpoint_without_the_gpu_model_field_raises():
    with pytest.raises(ckpt.CheckpointError, match="trainer_gpu_model"):
        ckpt.check_resume_gpu_model(_example_meta(), dict(_BLACKWELL))
    with pytest.raises(ckpt.CheckpointError, match="trainer_gpu_model"):
        ckpt.check_resume_gpu_model(
            _example_meta(trainer_gpu_model={"nvidia_smi_name": "x"}),
            dict(_BLACKWELL))


def test_the_trainer_gpu_model_is_the_device_kind_and_the_nvidia_smi_name():
    def smi(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="0, NVIDIA A\n4, NVIDIA B\n", stderr="")

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    assert ckpt.trainer_gpu_model(_Device("B", "gpu"), "4", run=smi) == {
        "device_kind": "B", "nvidia_smi_name": "NVIDIA B"}
    assert ckpt.trainer_gpu_model(_Device("cpu", "cpu"), "0", run=smi) == {
        "device_kind": "cpu", "nvidia_smi_name": None}
    assert ckpt.trainer_gpu_model(_Device("B", "gpu"), "4", run=missing) == {
        "device_kind": "B", "nvidia_smi_name": None}
    assert ckpt.trainer_gpu_model(_Device("B", "gpu"), "7", run=smi) == {
        "device_kind": "B", "nvidia_smi_name": None}


def test_reading_something_that_is_not_a_checkpoint_raises(tmp_path):
    with pytest.raises(ckpt.CheckpointError, match="needs a checkpoint"):
        ckpt.read_ppo_meta("")
    with pytest.raises(ckpt.CheckpointError, match="not a directory"):
        ckpt.read_ppo_meta(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ckpt.CheckpointError, match="no meta.json"):
        ckpt.read_ppo_meta(str(empty))


def test_a_checkpoint_of_another_format_raises(tmp_path):
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=1, tree=_example_tree(), meta=_example_meta())
    meta_path = os.path.join(path, "meta.json")
    with open(meta_path) as fh:
        doc = json.load(fh)
    doc["format"] = ckpt.PPO_CKPT_FORMAT + 1
    with open(meta_path, "w") as fh:
        json.dump(doc, fh)
    with pytest.raises(ckpt.CheckpointError, match="format"):
        ckpt.read_ppo_meta(path)


def test_a_checkpoint_without_its_state_file_raises(tmp_path):
    path = ckpt.save_ppo_checkpoint(
        str(tmp_path), episode=1, tree=_example_tree(), meta=_example_meta())
    os.remove(os.path.join(path, "state.eqx"))
    with pytest.raises(ckpt.CheckpointError, match="no state.eqx"):
        ckpt.load_ppo_tree(path, _example_tree())


# ---------------------------------------------------------------------------
# 4. The directory: naming, ordering, and keeping the last two.
# ---------------------------------------------------------------------------

def test_only_the_last_two_checkpoints_are_kept(tmp_path):
    run_dir = str(tmp_path / "run")
    written = []
    for ep in (10, 20, 30, 40):
        written.append(ckpt.save_ppo_checkpoint(
            run_dir, episode=ep, tree=_example_tree(),
            meta=_example_meta(), keep=2))
    kept = ckpt.list_checkpoints(run_dir)
    assert kept == written[-2:], kept
    assert not os.path.exists(written[0])
    assert not os.path.exists(written[1])


def test_a_pinned_episode_survives_pruning_beyond_the_newest_keep(tmp_path):
    """dsnn-dfw.117: `keep_at` exempts specific episodes from the
    newest-`keep` pruning; every other stale checkpoint is still removed."""
    run_dir = str(tmp_path / "run")
    written = {}
    for ep in (10, 20, 30, 40, 50):
        written[ep] = ckpt.save_ppo_checkpoint(
            run_dir, episode=ep, tree=_example_tree(), meta=_example_meta(),
            keep=2, keep_at={20})
    kept = ckpt.list_checkpoints(run_dir)
    assert kept == [written[20], written[40], written[50]], kept
    assert not os.path.exists(written[10])
    assert not os.path.exists(written[30])


def test_several_pinned_episodes_all_survive(tmp_path):
    run_dir = str(tmp_path / "run")
    written = {}
    for ep in (10, 20, 30, 40, 50, 60):
        written[ep] = ckpt.save_ppo_checkpoint(
            run_dir, episode=ep, tree=_example_tree(), meta=_example_meta(),
            keep=2, keep_at={10, 30})
    kept = ckpt.list_checkpoints(run_dir)
    assert kept == [written[10], written[30], written[50], written[60]], kept
    assert not os.path.exists(written[20])
    assert not os.path.exists(written[40])


def test_an_unset_keep_at_prunes_exactly_as_before(tmp_path):
    run_dir = str(tmp_path / "run")
    written = []
    for ep in (10, 20, 30, 40):
        written.append(ckpt.save_ppo_checkpoint(
            run_dir, episode=ep, tree=_example_tree(), meta=_example_meta(),
            keep=2, keep_at=None))
    kept = ckpt.list_checkpoints(run_dir)
    assert kept == written[-2:], kept


def test_the_directory_names_sort_by_episode(tmp_path):
    run_dir = str(tmp_path / "run")
    for ep in (2, 10, 100, 1000):
        ckpt.save_ppo_checkpoint(run_dir, episode=ep, tree=_example_tree(),
                                 meta=_example_meta(), keep=0)
    names = [os.path.basename(p) for p in ckpt.list_checkpoints(run_dir)]
    assert names == sorted(names)
    assert [ckpt.read_ppo_meta(os.path.join(run_dir, n))["episode"]
            for n in names] == [2, 10, 100, 1000]


def test_a_rewritten_checkpoint_replaces_the_old_one(tmp_path):
    run_dir = str(tmp_path / "run")
    first = ckpt.save_ppo_checkpoint(
        run_dir, episode=7, tree=_example_tree(seed=1), meta=_example_meta())
    second = ckpt.save_ppo_checkpoint(
        run_dir, episode=7, tree=_example_tree(seed=2), meta=_example_meta())
    assert first == second
    back = ckpt.load_ppo_tree(second, _example_tree(seed=99))
    assert _tree_bits(back) == _tree_bits(_example_tree(seed=2))
    # No staging directory survives a completed write.
    assert not os.path.exists(second + ".writing")


# ---------------------------------------------------------------------------
# 5. The flags themselves, on the real trainer's parser.
# ---------------------------------------------------------------------------

def test_the_trainer_defines_the_two_flags_with_the_ruled_defaults():
    p = argparse.ArgumentParser()
    ckpt.add_checkpoint_args(p)
    ns = p.parse_args([])
    assert ns.checkpoint_every == 50
    assert ns.resume == ""
    ns = p.parse_args(["--checkpoint-every", "0", "--resume", "/a/b"])
    assert ns.checkpoint_every == 0
    assert ns.resume == "/a/b"


# ---------------------------------------------------------------------------
# 6. --checkpoint-keep-at (dsnn-dfw.117): the flag and its parse-time checks.
# ---------------------------------------------------------------------------

def test_checkpoint_keep_at_defaults_to_empty():
    from alphagrad.approx.ppo import make_argparser as _ppo_argparser

    ns = _ppo_argparser().parse_args([])
    assert ns.checkpoint_keep_at == []


def test_checkpoint_keep_at_collects_every_value_given():
    from alphagrad.approx.ppo import make_argparser as _ppo_argparser

    ns = _ppo_argparser().parse_args(
        ["--checkpoint-every", "50", "--checkpoint-keep-at", "50", "100"])
    assert ns.checkpoint_keep_at == [50, 100]


def test_a_pinned_episode_that_is_not_a_multiple_raises():
    from alphagrad.approx.ppo import _validate_checkpoint_keep_at

    with pytest.raises(SystemExit, match="checkpoint-keep-at"):
        _validate_checkpoint_keep_at(50, [30])
    with pytest.raises(SystemExit, match="checkpoint-keep-at"):
        _validate_checkpoint_keep_at(50, [-50])
    with pytest.raises(SystemExit, match="checkpoint-keep-at"):
        _validate_checkpoint_keep_at(50, [0])


def test_a_nonempty_list_with_checkpointing_off_raises():
    from alphagrad.approx.ppo import _validate_checkpoint_keep_at

    with pytest.raises(SystemExit, match="checkpoint-every"):
        _validate_checkpoint_keep_at(0, [50])


def test_valid_pins_return_a_frozenset_and_empty_is_a_no_op():
    from alphagrad.approx.ppo import _validate_checkpoint_keep_at

    assert _validate_checkpoint_keep_at(50, [50, 100]) == frozenset({50, 100})
    assert _validate_checkpoint_keep_at(50, []) == frozenset()
    assert _validate_checkpoint_keep_at(0, []) == frozenset()
    assert _validate_checkpoint_keep_at(0, None) == frozenset()
