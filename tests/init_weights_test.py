"""Tests for --init-weights (weights-only warm start across targets).

Spec: agent-prompts/init-weights.prompt.md
Issue: dsnn-dfw.320
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

from alphagrad.approx.common import checkpoint as ckpt
from alphagrad.approx.ppo import make_argparser, _build_agent


_REPO = Path(__file__).resolve().parents[1]
_TRAINER = _REPO / "src" / "alphagrad" / "approx" / "ppo.py"


class _TinyModel(eqx.Module):
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


class _MismatchShapeModel(eqx.Module):
    w: jax.Array
    b: jax.Array
    half: jax.Array
    tag: str

    def __init__(self, key):
        k1, k2 = jrand.split(key)
        self.w = jrand.normal(k1, (5, 3), dtype=jnp.float32)  # mismatched shape (5,3) vs (4,3)
        self.b = jrand.normal(k2, (3,), dtype=jnp.float32)
        self.half = jnp.asarray([1.5, -2.25], dtype=jnp.bfloat16)
        self.tag = "tiny"


class _MismatchDtypeModel(eqx.Module):
    w: jax.Array
    b: jax.Array
    half: jax.Array
    tag: str

    def __init__(self, key):
        k1, k2 = jrand.split(key)
        self.w = jnp.zeros((4, 3), dtype=jnp.int32)  # mismatched dtype int32 vs float32
        self.b = jrand.normal(k2, (3,), dtype=jnp.float32)
        self.half = jnp.asarray([1.5, -2.25], dtype=jnp.bfloat16)
        self.tag = "tiny"


def _make_dummy_checkpoint(path: str, model, episode: int = 2, target: str = "Helmholtz"):
    os.makedirs(path, exist_ok=True)
    tree = {
        "agent": model,
        "opt_state": [jnp.zeros((4, 3))],
        "popart_m1": jnp.array(0.0),
        "popart_m2": jnp.array(1.0),
        "popart_w": jnp.array(1.0),
        "global_step": jnp.array(100),
        "lag_lambda": float(1.0),
        "kl_ref_coef": float(0.0),
    }
    meta = {
        "args": {"name": "dummy_source", "example": target},
        "episode": int(episode),
        "wandb_run_id": "dummy_wandb_123",
        "pareto_archive": {"pts": []},
    }
    return ckpt.save_ppo_checkpoint(path, episode=episode, tree=tree, meta=meta)


def test_init_weights_flag_defined():
    p = make_argparser()
    ns = p.parse_args([])
    assert hasattr(ns, "init_weights")
    assert ns.init_weights == ""
    ns2 = p.parse_args(["--init-weights", "/tmp/ckpt"])
    assert ns2.init_weights == "/tmp/ckpt"


def test_load_ppo_agent_success(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    model_orig = _TinyModel(jrand.PRNGKey(0))
    saved_path = _make_dummy_checkpoint(ckpt_dir, model_orig)

    template = _TinyModel(jrand.PRNGKey(42))
    loaded = ckpt.load_ppo_agent(saved_path, template)

    assert np.array_equal(np.asarray(loaded.w), np.asarray(model_orig.w))
    assert np.array_equal(np.asarray(loaded.b), np.asarray(model_orig.b))
    assert np.array_equal(np.asarray(loaded.half), np.asarray(model_orig.half))
    assert loaded.tag == "tiny"


def test_load_ppo_agent_shape_mismatch_refused(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    model_orig = _TinyModel(jrand.PRNGKey(0))
    saved_path = _make_dummy_checkpoint(ckpt_dir, model_orig)

    template_mismatch = _MismatchShapeModel(jrand.PRNGKey(42))
    with pytest.raises(ckpt.CheckpointError) as exc_info:
        ckpt.load_ppo_agent(saved_path, template_mismatch)

    err = str(exc_info.value)
    assert "parameter shape mismatch at leaf" in err
    assert ".w" in err
    assert "Likely argument:" in err


def test_load_ppo_agent_dtype_mismatch_refused(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    model_orig = _TinyModel(jrand.PRNGKey(0))
    saved_path = _make_dummy_checkpoint(ckpt_dir, model_orig)

    template_mismatch = _MismatchDtypeModel(jrand.PRNGKey(42))
    with pytest.raises(ckpt.CheckpointError) as exc_info:
        ckpt.load_ppo_agent(saved_path, template_mismatch)

    err = str(exc_info.value)
    assert "parameter dtype mismatch at leaf" in err
    assert ".w" in err
    assert "Likely argument:" in err


def test_init_weights_mutually_exclusive_with_resume(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    model_orig = _TinyModel(jrand.PRNGKey(0))
    saved_path = _make_dummy_checkpoint(ckpt_dir, model_orig)

    cmd = [
        sys.executable, str(_TRAINER),
        "--example", "Helmholtz",
        "--dataset", "none",
        "--init-weights", saved_path,
        "--resume", saved_path,
        "--wandb", "disabled",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "--init-weights and --resume are mutually exclusive" in (res.stderr + res.stdout)


def test_init_weights_mutually_exclusive_with_readout(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    model_orig = _TinyModel(jrand.PRNGKey(0))
    saved_path = _make_dummy_checkpoint(ckpt_dir, model_orig)

    cmd = [
        sys.executable, str(_TRAINER),
        "--example", "Helmholtz",
        "--dataset", "none",
        "--init-weights", saved_path,
        "--readout", "2",
        "--wandb", "disabled",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "--init-weights cannot be combined with --readout" in (res.stderr + res.stdout)


_FAST_COMMON = [
    "--variant", "full", "--face-actions", "--unified-face-head",
    "--live-faces", "--set-pointer", "--dynamic-substeps",
    "--max-substeps", "1", "--incremental-encode", "--grad-window", "0",
    "--dataset", "none", "--cmp-type", "flops", "--mem-type", "peak_memory",
    "--terminal-rewards-only", "--rewards", "cmp", "mem",
    "--lambda-cmp", "1", "--lambda-mem", "1", "--lambda-frob", "1",
    "--advantage-norm", "popart", "--seed", "42", "--num-envs", "2",
    "--minibatches", "1", "--vocab-size", "512", "--wandb", "disabled",
    "--grad-oracle", "off",
]

_FAST_ENV = {
    "JAX_PLATFORMS": "cpu",
    "ALPHAGRAD_POLICY": "palimpsa",
    "ALPHAGRAD_INCREMENTAL_TOKENS": "1",
    "ALPHAGRAD_UNIFIED_FACE_ENUM": "1",
    "ALPHAGRAD_SKIP_COUNT_OPS": "1",
    "ALPHAGRAD_SKIP_COST_ANALYSIS": "1",
    "ALPHAGRAD_EXTEND_CHUNK": "256",
    "ALPHAGRAD_DELTA_OVERFLOW": "clip",
    "ALPHAGRAD_FORCE_REV": "0",
}


def test_init_weights_end_to_end_transfer(tmp_path):
    # Step 1: Run source run on Helmholtz for 2 episodes, saving checkpoint every 1 episode
    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("ALPHAGRAD_")}
    env.update(_FAST_ENV)
    env["ALPHAGRAD_EQ_DUMP"] = str(source_dir / "dump")

    cmd1 = [
        sys.executable, str(_TRAINER),
        *_FAST_COMMON,
        "--example", "Helmholtz",
        "--name", "source_run",
        "--episodes", "2",
        "--checkpoint-every", "1",
    ]
    r1 = subprocess.run(cmd1, cwd=str(source_dir), env=env, capture_output=True, text=True, timeout=600)
    assert r1.returncode == 0, f"Source run failed:\n{r1.stdout}\n{r1.stderr}"

    ckpts = ckpt.list_checkpoints(str(source_dir))
    assert len(ckpts) >= 1
    ckpt_ep2 = ckpts[-1]
    assert ckpt.read_ppo_meta(ckpt_ep2)["episode"] == 2

    # Step 2: Start transfer run on NeuralNetwork with --init-weights pointing to ckpt_ep2
    target_dir = tmp_path / "target_run"
    target_dir.mkdir()
    env2 = dict(env)
    env2["ALPHAGRAD_EQ_DUMP"] = str(target_dir / "dump")

    cmd2 = [
        sys.executable, str(_TRAINER),
        *_FAST_COMMON,
        "--example", "NeuralNetwork",
        "--name", "transfer_run",
        "--episodes", "1",
        "--checkpoint-every", "1",
        "--init-weights", ckpt_ep2,
    ]
    r2 = subprocess.run(cmd2, cwd=str(target_dir), env=env2, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, f"Transfer run failed:\n{r2.stdout}\n{r2.stderr}"

    # Verify stdout confirmation of init-weights
    assert "[checkpoint] init-weights: successfully loaded network parameters" in r2.stdout

    # Verify target checkpoint metadata records init_weights_source
    target_ckpts = ckpt.list_checkpoints(str(target_dir))
    assert len(target_ckpts) >= 1
    target_meta = ckpt.read_ppo_meta(target_ckpts[-1])
    assert "init_weights_source" in target_meta
    iw_src = target_meta["init_weights_source"]
    assert iw_src is not None
    assert iw_src["episode"] == 2
    assert iw_src["target"] == "Helmholtz"
    assert iw_src["run_name"] == "source_run"
    assert os.path.realpath(ckpt_ep2) == iw_src["checkpoint_path"]
