import argparse
import inspect
import json
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx                                           # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import alphagrad.approx.env as E                                # noqa: E402
from alphagrad.approx.common import plan_log as plog            # noqa: E402
from alphagrad.approx.env import (                              # noqa: E402
    EnvConfig, MAX_FACES, MAX_RULES_PER_VERTEX, wire_slots,
)


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)


def _setup():
    cj = jax.make_jaxpr(_fn)(*ARGS)
    cfg = EnvConfig(jaxpr=cj.jaxpr, argnums=(0, 1), has_aux=False,
                    sparse=True, cmp_type="graphax", mem_type="graphax",
                    per_face=True)
    return cfg, tuple(cj.literals), tuple(ARGS)


def _wires(V, seed=0):
    rng = np.random.default_rng(seed)
    order = np.asarray(rng.permutation(np.arange(1, V + 1)), np.int32)
    specs = np.full((V, MAX_RULES_PER_VERTEX, 3), -1, np.int32)
    specs[..., 2] = 0
    faces = np.full((V, MAX_FACES, wire_slots(), 3), -1, np.int32)
    faces[..., 2] = 0
    skips = np.zeros((V, MAX_FACES), np.int32)
    for i in range(V):
        if rng.random() < 0.4:
            specs[i, 0] = (0, 0, 2)
        for f in range(3):
            r = rng.random()
            if r < 0.3:
                faces[i, f, 0] = (0, 0, 2)
            elif r < 0.4:
                skips[i, f] = 1
    return order, specs, faces, skips


def _run(run_dir):
    from alphagrad.approx.common import repro_bundle as RB
    args = argparse.Namespace(seed=250197, example="toy")
    config = {"commit/alphagrad": "ec5b614", "commit/graphax": "eea5a2c",
              "lr": 1.0}
    return lambda: RB.run_identity(
        args, config, SimpleNamespace(id="w4nd8id0", dir=str(run_dir)))


def _knobs(monkeypatch):
    monkeypatch.setenv("ALPHAGRAD_PLAN_LOG", "1")
    monkeypatch.setenv("ALPHAGRAD_INCREMENTAL_TOKENS", "1")
    monkeypatch.delenv("ALPHAGRAD_UNIFIED_FACE_ENUM", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    E.consume_plan_records()
    E._INCR_STREAM_CACHE.clear()
    E._FACE_ENUM_CACHE.clear()


def _identity(bundle):
    assert bundle["seed"] == 250197
    assert bundle["commits"] == {"alphagrad": "ec5b614", "graphax": "eea5a2c"}
    assert bundle["flags"] == {"seed": 250197, "example": "toy"}
    assert bundle["wandb_id"] == "w4nd8id0"


def test_a_refused_measurement_writes_a_bundle_with_the_plan_order_and_wires(
        tmp_path, monkeypatch):
    from alphagrad.approx.common import repro_bundle as RB
    from alphagrad.approx.cpu_approx_worker import CpuApproximationServer
    _knobs(monkeypatch)
    monkeypatch.setenv("ALPHAGRAD_DISABLE_JIT_DISK_CACHE", "1")
    monkeypatch.setattr(E, "_MEASURE_OOM_CONSUMER", [False])
    cfg, consts, args = _setup()
    V = len(cfg.jaxpr.eqns)
    order, specs, faces, skips = _wires(V)
    orig = E._record_token_length

    def _raise_at_the_terminal(raw_len):
        orig(raw_len)
        if int(E._PLAN_LOG_TERMINALS[0]) > 0:
            raise ValueError("injected raise inside the measurement")

    monkeypatch.setattr(E, "_record_token_length", _raise_at_the_terminal)
    srv = CpuApproximationServer(SimpleNamespace(
        config=cfg, args=args, consts=consts, eval_args_samples=None))
    out = srv.evaluate(order, specs, V, face_specs=faces, face_skips=skips,
                       env_row=5)
    assert "injected raise" in srv.last_eval_error[1]
    reward = np.asarray(out[-1])
    assert np.all(np.delete(reward, E.REWARD_INDEX["cosine_sim"])
                  == np.float32(-1e10))
    records = E.consume_plan_records()["records"]
    assert [r["refused"] for r in records] == ["raised:ValueError"]
    for r in records:
        r["actor"] = 3
        r["episode"] = 7
    log = tmp_path / "logs" / "job-4242.log"
    log.parent.mkdir()
    log.write_text("")
    monkeypatch.setattr(RB, "job_log", lambda: str(log))
    paths = RB.write_refused(records, _run(tmp_path / "run"))
    assert len(paths) == 1
    assert os.path.dirname(paths[0]) == str(log.parent)
    assert os.path.basename(paths[0]).startswith("repro_4242_ep7_env5_")
    with open(paths[0]) as fh:
        bundle = json.load(fh)
    want = plog.encode_wires(order, specs, faces, skips,
                             compress_sentinel=E.COMPRESS_SENTINEL,
                             quant_sentinel=E.QUANT_SENTINEL)
    assert want["faces"]
    assert bundle["plan"]["order"] == [int(v) for v in order]
    assert bundle["plan"]["rules"] == plog.jsonable(want["rules"])
    assert bundle["plan"]["faces"] == plog.jsonable(want["faces"])
    assert bundle["episode"] == 7 and bundle["env_index"] == 5
    assert bundle["exception"]["class"] == "ValueError"
    assert bundle["written_next_to"] == "job log"
    _identity(bundle)


def test_b_a_raise_in_a_pure_callback_inside_jit_writes_a_bundle_and_raises(
        tmp_path, monkeypatch):
    from alphagrad.approx.common import repro_bundle as RB
    _knobs(monkeypatch)
    monkeypatch.setattr(RB, "job_log", lambda: None)
    cfg, consts, args = _setup()
    V = len(cfg.jaxpr.eqns)
    order, specs, faces, skips = _wires(V)

    def _edge_shape(*a, **k):
        raise RuntimeError("Existing edge shape (128, 32) does not match "
                           "expected shape (32, 128)!")

    monkeypatch.setattr(E, "_incremental_stream_tokens", _edge_shape)
    # The patched raise site is on the separate tokenizer path; the shared prefix (dsnn-dfw.189) bypasses it.
    monkeypatch.setattr(E, "_SHARED_PREFIX", False)

    def _host(x):
        E._callback(cfg, args, consts, jnp.asarray(order), jnp.asarray(specs),
                    jnp.asarray(faces), jnp.asarray(skips), 2)
        return x

    @eqx.filter_jit
    def rollout(x):
        return jax.pure_callback(
            _host, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    guarded = RB.guard(rollout, source="rollout", episode=3,
                       run=_run(run_dir), records=E._PLAN_RECORDS)
    with pytest.raises(Exception, match="Existing edge shape") as info:
        guarded(jnp.ones((4,), jnp.float32))
    files = sorted(run_dir.glob("repro_4242_ep3_envna_*.json"))
    assert len(files) == 1, sorted(p.name for p in run_dir.iterdir())
    bundle = json.loads(files[0].read_text())
    assert bundle["source"] == "rollout" and bundle["episode"] == 3
    assert bundle["plan"] is None and bundle["env_index"] is None
    assert bundle["exception"]["class"] == type(info.value).__name__
    assert "Existing edge shape" in bundle["exception"]["message"]
    assert "in _callback_measured" in bundle["exception"]["message"]
    assert "Traceback" in bundle["exception"]["traceback"]
    assert bundle["written_next_to"] == "run directory"
    _identity(bundle)


def test_the_trainer_sends_every_rollout_update_and_refusal_to_the_writer():
    from alphagrad.approx import ppo
    src = inspect.getsource(ppo.main)
    for call in ('_repro_guard(_episode_rollout_jit, "rollout", ep)(',
                 '_repro_guard(train_episode, "rollout and update", ep)(',
                 '_repro_guard(_episode_update_jit, "update",',
                 'rollout_fn, "popart warm-up rollout", ep)(',
                 '_repro.write_refused('):
        assert call in src, f"the trainer does not call {call!r}"
