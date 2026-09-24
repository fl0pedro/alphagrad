import json

import jax.numpy as jnp
import numpy as np
import pytest

import alphagrad.approx.env as envmod
from alphagrad.approx.common import carry_plan as CP
from alphagrad.approx.common import masks as M
from alphagrad.approx.common import plan_log as plog

T_PIN = 7
RULE = "rtrl"
RAISED = "synthetic refusal on the container's program"
_BUILT: dict = {}


def _env(out_dir):
    import jax.random as jrand
    import alphagrad.approx.tools.landscape_map as lm
    if "env" not in _BUILT:
        args = lm.make_argparser().parse_args([
            "--example", "RSNN_SHD", "--dataset", "none",
            "--temporal-rule", RULE, "--step-position", str(T_PIN),
            "--num-eval-samples", "1", "--num-data-points", "1",
            "--reps-per-point", "1", "--out-dir", str(out_dir)])
        env, _eval, _cj = lm.build_env(args)
        _key, args_key = jrand.split(jrand.PRNGKey(args.seed))
        _BUILT.update(env=env, args_ns=args, key=args_key)
    env = _BUILT["env"]
    if not CP.armed(env.config):
        CP.register(_BUILT["args_ns"], _BUILT["key"], "RSNN_SHD", RULE,
                    env.config, env.args, env.consts, dataset=None,
                    dataset_size=-1, step_position=T_PIN)
    return env


def _quant_on_the_carry(env):
    import alphagrad.approx.tools.landscape_map as lm
    order = sorted((int(v) for v in env.valid_vertices), reverse=True)
    mask = CP.carry_scope_mask(env.config.jaxpr)
    specs, faces, skips = lm.empty_plan(len(order))
    row = [envmod.QUANT_SENTINEL, plog.quant_dtype_id("bfloat16"), 0]
    for k, v in enumerate(order):
        if mask[v - 1]:
            faces[k, 0, 0] = row
            faces[k, 0, 1] = row
    return order, specs, faces, skips


def _arm(mp, env):
    from graphax.core import FaceTransformIllegal
    mp.setenv("ALPHAGRAD_PLAN_LOG", "1")
    mp.setattr(M, "_PER_FACE_MASKS", [False])
    real = envmod._face_transforms_for_order

    def refuse_on_the_container_program(config, *a, **k):
        if config is not env.config:
            raise FaceTransformIllegal(RAISED)
        return real(config, *a, **k)

    mp.setattr(envmod, "_face_transforms_for_order",
               refuse_on_the_container_program)


def _measure(env, order, specs, faces, skips, *samples):
    return envmod._callback(
        env.config, env.args, env.consts, jnp.asarray(order),
        jnp.asarray(specs), jnp.asarray(faces), jnp.asarray(skips),
        int(len(order)), *samples)


@pytest.fixture(scope="module")
def refused(tmp_path_factory):
    from graphax.core import FaceTransformIllegal
    env = _env(tmp_path_factory.mktemp("planlogrec"))
    order, specs, faces, skips = _quant_on_the_carry(env)
    own = CP.container_for_plan(env.config, order, faces, skips, specs)
    own_bytes = CP.carry_at_rest_bytes(CP.measurement_env(own, env.config),
                                       RULE)
    exact_bytes = CP.carry_at_rest_bytes(CP._entry(env.config)["base"], RULE)
    with pytest.MonkeyPatch.context() as mp:
        _arm(mp, env)
        # What an exact plan measured before this one leaves in the process.
        mp.setattr(envmod, "_PLAN_CARRY", ["exact"])
        mp.setattr(envmod, "_PLAN_CARRY_BYTES", [exact_bytes])
        envmod.consume_plan_records()
        with pytest.raises(FaceTransformIllegal, match=RAISED):
            _measure(env, order, specs, faces, skips)
        records = envmod.consume_plan_records()["records"]
    assert len(records) == 1, records
    return {"env": env, "record": records[0], "own": own,
            "own_bytes": own_bytes, "exact_bytes": exact_bytes,
            "wires": (order, specs, faces, skips)}


def test_a_refused_plan_record_names_its_own_carry_container_and_bytes(
        refused):
    rec = refused["record"]
    assert rec["refused"] == "raised:FaceTransformIllegal"
    assert refused["own"] == "quant"
    assert refused["own_bytes"] != refused["exact_bytes"]
    assert (rec.get("carry_container"), rec.get("carry_bytes")) == (
        refused["own"], refused["own_bytes"])


def test_a_refused_plan_record_decodes_and_replays_to_the_same_raise(
        refused, monkeypatch):
    from graphax.core import FaceTransformIllegal
    rec = json.loads(json.dumps(plog.jsonable(refused["record"]),
                                allow_nan=False))
    order, specs, faces, skips = plog.decode_wires(rec)
    for got, want in zip((order, specs, faces, skips), refused["wires"]):
        np.testing.assert_array_equal(got, np.asarray(want))
    env = refused["env"]
    _arm(monkeypatch, env)
    envmod.consume_plan_records()
    with pytest.raises(FaceTransformIllegal, match=RAISED):
        _measure(env, order, specs, faces, skips)
    again = envmod.consume_plan_records()["records"]
    assert len(again) == 1, again
    for k in ("refused", "plan_hash", "carry_container", "carry_bytes"):
        assert again[0].get(k) == rec.get(k), k
    assert rec["faces_truncated"] == 0
    assert rec["replayable"] is True and rec["sentinelled"] is True


def test_a_dedupe_hit_record_carries_the_carry_bytes_of_its_container(
        refused, monkeypatch):
    env = refused["env"]
    order, specs, faces, skips = refused["wires"]
    samples = tuple(env.eval_args_samples)
    cached = CP.measurement_env(refused["own"], env.config)
    # A missed hit raises at the container step instead of measuring.
    _arm(monkeypatch, env)
    monkeypatch.setenv("ALPHAGRAD_MEASURE_DEDUPE", "1")
    monkeypatch.setattr(envmod, "_PLAN_CARRY_BYTES", [refused["exact_bytes"]])
    envmod.consume_plan_records()
    # This episode measured the same plan before, as its plan 0.
    envmod._roll_measure_episode(envmod._episode_measure_key(samples))
    monkeypatch.setitem(
        envmod._PLAN_DEDUPE,
        envmod._plan_content_key(order, specs, faces, skips, None),
        (0, [0.0] * envmod.NUM_REWARDS))
    _measure(env, order, specs, faces, skips, *samples)
    records = envmod.consume_plan_records()["records"]
    assert len(records) == 1 and records[0]["measured_from"] == 0, records
    assert CP.measurement_env(refused["own"], env.config) is cached
    assert (records[0].get("carry_container"),
            records[0].get("carry_bytes")) == (refused["own"],
                                               refused["own_bytes"])


def test_a_record_refused_before_the_container_step_keeps_no_carry_bytes(
        refused, monkeypatch):
    env = refused["env"]
    order, specs, faces, skips = refused["wires"]
    _arm(monkeypatch, env)
    monkeypatch.setattr(envmod, "_PLAN_CARRY_BYTES", [refused["exact_bytes"]])

    def overflow_in_the_tokenizer(n):
        raise ValueError("synthetic refusal in the tokenizer")

    monkeypatch.setattr(envmod, "_record_token_length",
                        overflow_in_the_tokenizer)
    envmod.consume_plan_records()
    with pytest.raises(ValueError, match="synthetic refusal in the tokenizer"):
        _measure(env, order, specs, faces, skips)
    records = envmod.consume_plan_records()["records"]
    assert len(records) == 1, records
    assert records[0]["refused"] == "raised:ValueError"
    assert records[0].get("carry_container") == refused["own"]
    assert "carry_bytes" not in records[0]


def test_decode_refuses_only_a_truncated_record():
    order = np.arange(1, 4, dtype=np.int32)
    specs = np.full((3, 2, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((3, 4, 3, 3), -1, np.int32)
    skips = np.zeros((3, 4), np.int32)
    faces[0, 1, 0] = (1, 0, 2)
    skips[2, 3] = 1
    old = plog.encode_wires(order, specs, faces, skips)
    # How jobs 67851 and 67852 wrote a refused record with a complete wire.
    old.update(refused="raised:FaceTransformIllegal", sentinelled=True,
               replayable=False)
    o2, s2, f2, k2 = plog.decode_wires(old)
    for got, want in zip((o2, s2, f2, k2), (order, specs, faces, skips)):
        np.testing.assert_array_equal(got, want)
    cut = plog.encode_wires(order, specs, faces, skips, max_faces_recorded=1)
    assert cut["faces_truncated"] == 1
    with pytest.raises(ValueError, match="TRUNCATED"):
        plog.decode_wires(cut)
