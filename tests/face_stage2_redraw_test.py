#!/usr/bin/env python3
"""The stage-2 redraw must keep the loop's own decisions (dsnn-dfw.11).

STAGE 2 (finding 75) re-draws every face on device from its STORED context
against the host-composed slot-2 masks. Slot 2's mask is the only input that
moves, so the SKIP bit and the slot-0 / slot-1 rows have to come back exactly
as ``Agent._face_loop`` drew them: same key, same context, same masks, same
logits. Anything less and the trajectory stores an action the behaviour policy
never drew.

The redraw used to run as ``jax.vmap`` over the whole face WIDTH (1920 lanes
on the 3-block TLM for a vertex with two live faces). On the GPU that batched
head returned saturated logits -- measured at ``--face-none-bias 2``, step 0,
vertex 39: skip probability 1.0e-6 inside the vmap against 0.116905 from one
call on the SAME stored context in the same process -- so every face came back
NONE, no face was ever skipped, and a bias-2 run applied no approximation at
all while every counter looked healthy. It now visits the LIVE faces in the
same ``lax.while_loop`` the sampling loop draws them in.

The two claims below are the ones that failed on the GPU. They are backend
independent, so this module pins them wherever it runs.

Runs alone (``pytest tests/face_stage2_redraw_test.py``): it imports
``policy_regression_gate``, which pins ALPHAGRAD_* at module scope.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_SCALE_PINS = {k: os.environ.get(k) for k in
               ("ALPHAGRAD_MAX_DELTA_TOKENS", "ALPHAGRAD_MAX_FACES")}
import policy_regression_gate as _gate                            # noqa: E402

import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import jax.random as jrand                                        # noqa: E402

from graphax.examples import Perceptron                           # noqa: E402

from alphagrad.approx import face_action as _rec                  # noqa: E402
from alphagrad.approx.common import carry_stream as CS            # noqa: E402
from alphagrad.approx.common.face_driver import (                 # noqa: E402
    bind_sizes_callback, bind_step_callbacks)
from alphagrad.approx.common.masks import vertex_avail_at_step    # noqa: E402
from alphagrad.approx.common.order import fixed_order_for_env     # noqa: E402
from alphagrad.approx import env as _env_frozen                   # noqa: E402,F401

for _k, _v in _SCALE_PINS.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v


def _perceptron_args():
    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (2, 4))
    y = jrand.normal(jrand.fold_in(key, 5), (2, 3))
    W1 = jrand.normal(jrand.fold_in(key, 1), (4, 6))
    b1 = jrand.normal(jrand.fold_in(key, 2), (6,))
    W2 = jrand.normal(jrand.fold_in(key, 3), (6, 3))
    b2 = jrand.normal(jrand.fold_in(key, 4), (3,))
    gamma = jrand.normal(jrand.fold_in(key, 6), (6,))
    beta = jrand.normal(jrand.fold_in(key, 7), (6,))
    return (x, y, W1, b1, W2, b2, gamma, beta)


_ARGS = _perceptron_args()
_ARGNUMS = (2, 3, 4, 5)
_STEPS = 6
_SEED = 0


@pytest.fixture(scope="module", autouse=True)
def _put_the_per_face_mask_setting_back():
    # `build_case` turns --per-face-masks on process-wide; put it back.
    from alphagrad.approx.common import masks as _M
    flags = (_M._PER_FACE_MASKS[0], _M._PER_FACE_REPAIR_AXIS[0])
    names = ("ALPHAGRAD_PER_FACE_MASKS", "ALPHAGRAD_PER_FACE_REPAIR_AXIS")
    envs = {k: os.environ.get(k) for k in names}
    try:
        yield
    finally:
        _M._PER_FACE_MASKS[0], _M._PER_FACE_REPAIR_AXIS[0] = flags
        for k, v in envs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _build_case(seed):
    _gate._mlp = Perceptron
    _gate._ARGS = _ARGS
    _gate._ARGNUMS = _ARGNUMS
    _gate.SEED = seed
    return _gate.build_case()


def _rows(seed=_SEED, steps=_STEPS):
    """One rollout; per step the record with stage 2 ON and with it OFF.

    The two calls share the step's key, state and callbacks, so the face loop
    inside them draws the same faces from the same contexts -- the ONLY
    difference is whether the stage-2 host pass and its redraw run.
    """
    case = _build_case(seed)
    env, agent = case["env"], case["agent"]
    total_v, num_valid = case["total_v"], case["num_valid"]

    state = env.reset()
    base_tok, base_n = env.base_observation()
    base_w = max(int(base_n), 1)
    try:
        base_own = env.base_owners()
    except Exception:                                     # pragma: no cover
        base_own = None
    enc_carry, base_s, base_c = CS.init_carry(
        agent, base_tok[:base_w], base_n,
        window=base_w, total_v=total_v, embd_dim=_gate.EMBD,
        base_owners=base_own, path="rollout")
    base_mem = (base_s, base_c)
    vmem_s, vmem_c = CS.zero_memory(total_v, _gate.EMBD)

    fixed_order = fixed_order_for_env("markowitz", env)
    n_steps = min(int(steps), max(1, num_valid - 1))
    keys = jrand.split(jrand.PRNGKey(seed), max(1, num_valid - 1))
    part = jnp.zeros((total_v + 1,), jnp.float32)

    out = []
    for t in range(n_steps):
        delta_owner = jnp.where(
            state.step_count > 0,
            state.order[jnp.maximum(state.step_count - 1, 0)],
            jnp.asarray(-1, jnp.int32)).astype(jnp.int32)
        enc_carry, vmem_s, vmem_c = CS.advance(
            agent, enc_carry, vmem_s, vmem_c,
            state.delta_tokens, state.delta_count,
            delta_owner, window=case["window"], participants=part,
            path="rollout")
        precomputed = CS.heads(agent, vmem_s, vmem_c,
                               base_mem=base_mem, preference=None)
        avail = vertex_avail_at_step(
            state, case["vertex_valid_static"], total_v, num_valid,
            fixed_order=fixed_order)

        chunk_fn, count_fn = bind_step_callbacks(
            case["chunk_cb"], case["count_cb"],
            state.order, state.sparsity_specs, state.step_count,
            state.face_specs, state.face_skips)
        sizes_fn = bind_sizes_callback(
            case["sizes_cb"],
            state.order, state.sparsity_specs, state.step_count,
            state.face_specs, state.face_skips)

        def decide_fn(_v, _skips, _rows_, _o=state.order,
                      _s=state.sparsity_specs, _k=state.step_count,
                      _fh=state.face_specs, _kh=state.face_skips):
            return case["decide_cb"](_o, _s, _k, _v, _fh, _kh, _skips, _rows_)

        def _sample(decide):
            return agent.sample_action_dynamic(
                None, avail, state.axis_state, state.axis_valid_mask,
                case["factor_tables"], case["op_legality"], keys[t],
                preference=None, precomputed=precomputed,
                face_chunk_fn=chunk_fn, face_count_fn=count_fn,
                face_sizes_fn=sizes_fn, face_decide_fn=decide,
                enc_carry=enc_carry)

        with_two = _sample(decide_fn)
        without = _sample(None)
        assert with_two[11] is not None and without[11] is not None
        fa2, fv = with_two[11][0], with_two[11][5]
        fa1 = without[11][0]
        out.append((fa1, fa2, np.asarray(fv)))

        env_action = agent.to_env_action_dynamic(
            with_two[0], with_two[1], state.axis_state, face_action=fa2)
        state = env.step(state, env_action).state
    return case, out


@pytest.fixture(scope="module")
def rollout():
    return _rows()


def test_the_redraw_keeps_every_skip_bit_the_loop_drew(rollout):
    _case, rows = rollout
    live_faces = 0
    for t, (fa1, fa2, fv) in enumerate(rows):
        live = fv > 0.5
        live_faces += int(live.sum())
        np.testing.assert_array_equal(
            np.asarray(fa2.skip)[live], np.asarray(fa1.skip)[live],
            err_msg=f"step {t}: the stage-2 redraw changed a skip bit")
    assert live_faces > 0, "no live face in the rollout: nothing was pinned"


def test_the_redraw_keeps_the_two_exact_slots(rollout):
    # Slots 0 (lhs) and 1 (rhs) read masks nothing at this vertex moves, so
    # stage 2 must reproduce them; only slot 2 (res:new) may differ.
    _case, rows = rollout
    for t, (fa1, fa2, fv) in enumerate(rows):
        live = fv > 0.5
        for field in ("op_type", "i", "j", "compress_kind", "quant_dtype"):
            a = np.asarray(getattr(fa1, field))[live][:, :2]
            b = np.asarray(getattr(fa2, field))[live][:, :2]
            np.testing.assert_array_equal(
                b, a, err_msg=f"step {t}: stage 2 moved {field} on slot 0/1")


def test_the_redraw_leaves_the_padding_faces_canonical(rollout):
    case, rows = rollout
    pol = case["agent"].face_path_policy
    zero = _rec.zeros(pol.approx_add, pol.max_faces)
    for t, (_fa1, fa2, fv) in enumerate(rows):
        dead = ~(fv > 0.5)
        if not dead.any():
            continue
        np.testing.assert_array_equal(
            np.asarray(fa2.skip)[dead], np.asarray(zero.skip)[dead],
            err_msg=f"step {t}: a padding face carries a skip")
        for field in _rec.names(pol.approx_add)[1:]:
            np.testing.assert_array_equal(
                np.asarray(getattr(fa2, field))[dead],
                np.asarray(getattr(zero, field))[dead],
                err_msg=f"step {t}: a padding face carries {field}")
