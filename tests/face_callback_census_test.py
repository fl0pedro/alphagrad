#!/usr/bin/env python3
"""HOW MANY HOST ROUND TRIPS ONE ROLLOUT STEP MAKES, counted (owner ruling
2026-09-15, item 1).

``face_driver`` now keeps a census: one counter per host body, bumped once per
``pure_callback`` INVOCATION. Every face body is dispatched with
``vmap_method="broadcast_all"``, so one invocation serves the whole batch and
the census IS the number of device-to-host round trips the face path makes.

WHAT THIS FILE PINS.

1. THE MERGE. ``make_face_slot_legality_callback(with_count=True)`` serves the
   face COUNT and the per-slot MASKS in ONE call. Over a whole episode the
   merged route fires ``faces.count_legality`` once per step and
   ``faces.live_count`` NEVER; the unmerged route fires ``faces.live_count``
   and ``faces.live_slot_legality`` once each per step. One round trip per
   step is removed, and the operand set those two shared -- the prefix order,
   the rule specs and the two face-history wires -- crosses the bus once
   instead of twice.

2. THE MERGE IS EXACT. The same seeded episode on both routes produces the
   same vertex, the same face action, the same counts and the same face
   tokens, step for step. The count is still
   :meth:`LiveFaceStream.n_faces` in both, which is why.

3. THE REST OF THE CENSUS, WHICH IS NOT ONE. The ruling asked for a rollout
   step to make exactly one ``pure_callback`` for all host work. It cannot,
   and this test says so in numbers rather than in a comment. Per step the
   face path fires:

     * ONE ``faces.count_legality`` (the merge above), before the draw;
     * ONE ``faces.live_chunk`` PER LIVE FACE, from inside
       ``UnifiedPolicy._face_loop``'s ``lax.while_loop``: face f's chunk is
       read on the graph faces 0..f-1 of the SAME vertex have already been
       approximated on, so it cannot be computed before face f-1 is drawn on
       the device;
     * ONE ``faces.vertex_decide``, after that loop, whose operands are the
       loop's own output (``fa.skip`` and the stage-1 rows).

   Plus the env step callback in ``env.step``, which runs after the action is
   complete and is therefore later still. The three moments are separated by
   device-side draws, so merging them means moving the neural face draw to the
   host, which is a different change with a different blast radius.

THIS FILE IS MEANT TO RUN ALONE (``pytest tests/face_callback_census_test.py``),
like ``tests/one_stream_claim_test.py`` and for the same reason: it drives the
real policy path through ``policy_regression_gate.build_case``, and that module
pins the interpreter at import time. The import block below restores the two
scale knobs so a shared run has nothing to collide on.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# THE GATE PINS THE INTERPRETER AT MODULE SCOPE, AND IT HAS TO BE IMPORTED
# BEFORE ANY ALPHAGRAD IMPORT. `env.py` freezes MAX_FACES / MAX_DELTA_TOKENS
# into module constants at its first import, so a gate import that lands after
# one is ten dead writes and a graph built at the wrong scale. Snapshot the two
# scale knobs and put them back after env.py has frozen them, exactly as
# tests/one_stream_claim_test.py does and for the same two reasons: a
# standalone run still freezes env.py under the gate's pins, and a shared run
# is left with no cross-module configuration conflict for the collection guard
# to fail on.
_SCALE_PINS = {k: os.environ.get(k) for k in
               ("ALPHAGRAD_MAX_DELTA_TOKENS", "ALPHAGRAD_MAX_FACES")}
import policy_regression_gate as _gate                            # noqa: E402

import jax.numpy as jnp                                           # noqa: E402
import jax.random as jrand                                        # noqa: E402
import numpy as np                                                # noqa: E402

from alphagrad.approx.common.face_driver import (                 # noqa: E402
    bind_sizes_callback,
    bind_step_callbacks,
    consume_callback_census,
    make_face_slot_legality_callback,
)
# This import has to be here, and only here: env.py must have frozen its
# constants before the two scale knobs go back.
from alphagrad.approx import env as _env_frozen                   # noqa: E402,F401

for _k, _v in _SCALE_PINS.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v


STEPS = 6


def _case(with_count):
    """The gate's own case, with the slot-legality callback rebuilt.

    ``build_case`` is the one place agent / env / live-face-stream
    construction lives (see its docstring); only the callback under test is
    swapped, so nothing about the policy path is a second hand-copy.
    """
    case = _gate.build_case()
    case["sizes_cb"] = make_face_slot_legality_callback(
        case["stream"], max_faces=case["max_faces"],
        max_axes=case["max_axes"], with_count=with_count)
    return case


def _run(case, steps=STEPS):
    """``(records, census)`` for a seeded episode on this case's callbacks.

    The step loop is ``policy_regression_gate.run_trace``'s, trimmed to what
    a census and an equality need.
    """
    from alphagrad.approx.common import carry_stream as CS
    from alphagrad.approx.common.masks import vertex_avail_at_step

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
        base_owners=base_own)
    base_mem = (base_s, base_c)
    vmem_s, vmem_c = CS.zero_memory(total_v, _gate.EMBD)

    n_steps = max(1, min(int(steps), num_valid - 1))
    keys = jrand.split(jrand.PRNGKey(_gate.SEED), n_steps)
    part = jnp.zeros((total_v + 1,), jnp.float32)

    consume_callback_census()
    records = []
    per_step_census = []
    for t in range(n_steps):
        delta_owner = jnp.where(
            state.step_count > 0,
            state.order[jnp.maximum(state.step_count - 1, 0)],
            jnp.asarray(-1, jnp.int32)).astype(jnp.int32)
        enc_carry, vmem_s, vmem_c = CS.advance(
            agent, enc_carry, vmem_s, vmem_c,
            state.delta_tokens, state.delta_count,
            delta_owner, window=case["window"], participants=part)
        precomputed = CS.heads(agent, vmem_s, vmem_c,
                               base_mem=base_mem, preference=None)
        avail = vertex_avail_at_step(
            state, case["vertex_valid_static"], total_v, num_valid)

        chunk_fn, count_fn = bind_step_callbacks(
            case["chunk_cb"], case["count_cb"],
            state.order, state.sparsity_specs, state.step_count,
            state.face_specs, state.face_skips)
        sizes_fn = bind_sizes_callback(
            case["sizes_cb"],
            state.order, state.sparsity_specs, state.step_count,
            state.face_specs, state.face_skips)

        def decide_fn(_v, _skips, _rows, _o=state.order,
                      _s=state.sparsity_specs, _k=state.step_count,
                      _fh=state.face_specs, _kh=state.face_skips):
            return case["decide_cb"](_o, _s, _k, _v, _fh, _kh, _skips, _rows)

        # The census of THIS step alone.
        consume_callback_census()
        (vertex_idx, micro, _vd, _od, _id_, _jd, _ed, _kd,
         _mq, _mp, _mc, face_out, _value,
         _vctx) = agent.sample_action_dynamic(
            None, avail, state.axis_state, state.axis_valid_mask,
            case["factor_tables"], case["op_legality"], keys[t],
            preference=None, precomputed=precomputed,
            face_chunk_fn=chunk_fn, face_count_fn=count_fn,
            face_sizes_fn=sizes_fn, face_decide_fn=decide_fn,
            enc_carry=enc_carry,
        )
        per_step_census.append(consume_callback_census())

        assert face_out is not None, "no face head ran -- the census is empty"
        (fa, _lp, _ent, _pair, _comp, f_valid, f_cnt, f_dt,
         _ends) = face_out[:9]
        records.append(dict(
            vertex=int(vertex_idx),
            n_live=int(np.sum(np.asarray(f_valid) > 0.5)),
            valid=np.asarray(f_valid, np.float64).copy(),
            counts=np.asarray(f_cnt, np.int64).copy(),
            tokens=np.asarray(f_dt, np.int64).copy(),
            skip=np.asarray(fa.skip, np.int64).copy(),
            op_type=np.asarray(fa.op_type, np.int64).copy(),
        ))

        # The participation mask of the delta the NEXT step consumes, exactly
        # as `run_trace` computes it: drop it and the carry diverges from the
        # gate's after one step, and with it every draw.
        part = agent.participation_mask(
            total_v, jnp.asarray(vertex_idx, jnp.int32),
            face_out[8], face_out[5])
        env_action = agent.to_env_action_dynamic(
            vertex_idx, micro, state.axis_state, face_action=fa)
        state = env.step(state, env_action).state

    return records, per_step_census


# ------------------------------------------------------------------ the merge
def test_the_count_and_the_legality_are_one_call_per_step():
    """ONE round trip serves both, and the count callback never fires."""
    _case_m = _case(with_count=True)
    _recs, census = _run(_case_m)
    assert census, "the episode ran no steps"
    for t, c in enumerate(census):
        assert c.get("faces.count_legality", 0) == 1, (
            f"step {t}: the merged count+legality callback fired "
            f"{c.get('faces.count_legality', 0)} times, want exactly 1 "
            f"(census {c})")
        assert c.get("faces.live_count", 0) == 0, (
            f"step {t}: the separate face-count callback fired "
            f"{c['faces.live_count']} times on the merged route; the count "
            f"rides out of the legality call (census {c})")
        assert c.get("faces.live_slot_legality", 0) == 0, (
            f"step {t}: the unmerged legality key was bumped on the merged "
            f"route (census {c})")


def test_the_unmerged_route_pays_two_calls_per_step():
    """The control: without ``with_count`` it is two round trips."""
    _recs, census = _run(_case(with_count=False))
    for t, c in enumerate(census):
        assert c.get("faces.live_count", 0) == 1, (
            f"step {t}: census {c}")
        assert c.get("faces.live_slot_legality", 0) == 1, (
            f"step {t}: census {c}")
        assert c.get("faces.count_legality", 0) == 0, (
            f"step {t}: census {c}")


def test_the_merge_changes_no_value():
    """Same episode, both routes, step for step identical.

    The count is :meth:`LiveFaceStream.n_faces` on both routes -- the merged
    call does NOT read ``face_slot_legality``'s own ``min(len(keys), F)``,
    which clamps and soft-fails where ``n_faces`` raises -- so the face
    validity mask, and everything the head draws under it, is the same array.
    """
    merged, _c1 = _run(_case(with_count=True))
    plain, _c2 = _run(_case(with_count=False))
    assert len(merged) == len(plain)
    for t, (a, b) in enumerate(zip(merged, plain)):
        assert a["vertex"] == b["vertex"], f"step {t}: vertex"
        assert a["n_live"] == b["n_live"], f"step {t}: live face count"
        np.testing.assert_array_equal(a["valid"], b["valid"],
                                      err_msg=f"step {t}: f_valid")
        np.testing.assert_array_equal(a["counts"], b["counts"],
                                      err_msg=f"step {t}: face counts")
        np.testing.assert_array_equal(a["tokens"], b["tokens"],
                                      err_msg=f"step {t}: face tokens")
        np.testing.assert_array_equal(a["skip"], b["skip"],
                                      err_msg=f"step {t}: face skip")
        np.testing.assert_array_equal(a["op_type"], b["op_type"],
                                      err_msg=f"step {t}: face op_type")


# ------------------------------------------------- the rest of the census
def test_the_step_census_is_what_the_face_path_can_be():
    """The whole per-step census, named -- and it is not one call.

    See this module's docstring, point 3. The chunk callback runs once per
    LIVE face inside the device face loop, and the vertex decision runs after
    it on the loop's own output, so neither can join the call that fires
    before the draw.
    """
    recs, census = _run(_case(with_count=True))
    for t, (r, c) in enumerate(zip(recs, census)):
        assert set(c) <= {"faces.count_legality", "faces.live_chunk",
                          "faces.vertex_decide"}, (
            f"step {t}: an unexpected host body fired: {c}")
        assert c.get("faces.count_legality", 0) == 1, f"step {t}: {c}"
        assert c.get("faces.vertex_decide", 0) == 1, f"step {t}: {c}"
        # One chunk call per live face. The loop trips `n_live` times; a
        # vertex with no live face trips none.
        assert c.get("faces.live_chunk", 0) == r["n_live"], (
            f"step {t}: {c.get('faces.live_chunk', 0)} chunk callbacks for "
            f"{r['n_live']} live faces (census {c})")
    # And the episode saw at least one vertex with a live face, or the
    # chunk assertion above is vacuous.
    assert any(r["n_live"] > 0 for r in recs), (
        "no vertex in this episode had a live face")
