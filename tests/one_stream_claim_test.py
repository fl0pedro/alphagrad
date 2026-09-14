#!/usr/bin/env python3
"""Owner ruling Q2 (2026-09-13): is the stored face stream of step t equal to
the next step's delta, truncated?

``ppo.Trajectory.face_delta_tokens[t]`` (built by ``Agent._face_loop`` from
``LiveFaceStream`` chunks, during sampling, before the vertex's face
decisions are final) and ``ppo.Trajectory.delta_tokens[t+1]`` (built by the
env's OWN incremental tokenizer, from the real ``env.step`` transition that
step t's decisions produced) are two arrays written by TWO DIFFERENT
tokenizer instances over what should be the same prefix. Nothing in the repo
asserted, before this file, that they agree. ``tests/trajectory_layout_test.py``
proved the MECHANISM (a face's chunk carries its PREDECESSOR's echo, so the
concatenation of a vertex's chunks is a prefix of the vertex's own emission,
missing the last face's echo) against a freshly-built REFERENCE tokenizer on
one hand-picked vertex. This file checks the same claim against the actual
two-tokenizer pair, across a real multi-step rollout, using the real
``--face-actions --unified-face-head --live-faces --per-face-masks
--dynamic-substeps`` policy path (``--incremental-encode`` has been mandatory
since stage 2, see ``ppo.main``):

    face_delta_tokens[t][i] == delta_tokens[t+1][i]   for i < sum(face_counts[t])
    face_delta_tokens[t][i] == 0 (pad)                for i >= sum(face_counts[t])
    sum(face_counts[t]) <= delta_count[t+1]

THE ROLLOUT. ``ppo.rollout_fn`` is a closure nested inside ``ppo.main`` (it
reads ``main``'s parsed ``args`` and is built inside a decorator stack), so it
cannot be called on its own. ``tests/policy_regression_gate.py`` exists
because of exactly this: it drives the SAME functions ``rollout_fn``'s
``step_fn`` calls -- ``carry_stream.advance``, ``Agent.sample_action_dynamic``
(which runs ``_face_loop`` internally), ``env.step`` -- one Python step at a
time, built from ``build_and_init_agent`` / ``build_live_face_stream`` /
``make_face_callbacks`` exactly as ``ppo.main`` builds them. This module
reuses that harness (``build_case``) rather than re-implementing agent/env/
stream construction a second time; it only swaps in its own target graph,
seed and a ``--fixed-order markowitz`` table (``common.order
.fixed_order_for_env``, ticket dsnn-3qm.64) and writes its own short step
loop, because the gate's own loop records a JSON-able trace this test does
not need and does not pin ``fixed_order``.

The graph is ``graphax.examples.Perceptron`` at the argnums
``tests/trajectory_layout_test.py`` uses -- proven there (module docstring,
item 2) to have a vertex whose faces echo an approximation under BOTH
decision channels, ``skip`` (``face_skips[f] == 1``) and a slot ``rule``
(a real op != ``OP_END``), since the 2e06129e chooser change made graphax
record an ``approx`` block for either one. Twelve weight-init seeds are
pooled (the vertex ORDER is pinned by markowitz on every seed, so only the
face head's draws vary) so the rollout is checked against BOTH channels
rather than against whichever one an untrained head happens to draw first.

``tests/policy_regression_gate.py`` pins ``ALPHAGRAD_*``/environment
configuration at IMPORT TIME (its own header explains why: ``env.py`` reads
``MAX_FACES`` / ``MAX_DELTA_TOKENS`` at module scope). That is only safe in a
process that collects NOTHING else (see ``_pytest_config_guard.py``), which is
why this file is meant to run alone (``pytest tests/one_stream_claim_test.py``),
exactly as the sbatch template does.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import policy_regression_gate as _gate                            # noqa: E402

import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import jax.random as jrand                                         # noqa: E402

from graphax import IncrementalPathTokenizer                       # noqa: E402
from graphax.examples import Perceptron                           # noqa: E402

from alphagrad.approx.heads import OP_END                          # noqa: E402
from alphagrad.approx.common.order import fixed_order_for_env     # noqa: E402
from alphagrad.approx.common.masks import vertex_avail_at_step    # noqa: E402
from alphagrad.approx.common import carry_stream as CS             # noqa: E402
from alphagrad.approx.common.face_driver import (                 # noqa: E402
    bind_sizes_callback, bind_step_callbacks)
from alphagrad.approx.common.token_vocab import incr_token_vocab  # noqa: E402


# --------------------------------------------------------------------------
# THE GRAPH. Identical to ``tests/trajectory_layout_test.py``'s ``_perceptron``
# (module docstring, item 2): the one graph in this suite proven to echo an
# approximation under both the ``skip`` and the ``rule`` decision channel.
# --------------------------------------------------------------------------
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


_PERCEPTRON_ARGS = _perceptron_args()
_PERCEPTRON_ARGNUMS = (2, 3, 4, 5)

# Weight-init seeds pooled for face-decision coverage. The vertex order is
# pinned (markowitz), so only the face head's draws differ between them.
_SEEDS = tuple(range(12))


def _build_case(seed):
    """One ``policy_regression_gate.build_case()`` over OUR graph/seed.

    ``build_case`` reads its target/args/argnums/seed off its own module
    globals at call time, so overwriting them here and calling it gives the
    IDENTICAL agent/env/live-face-stream construction the face tests already
    exercise, instead of a second hand-copy of ~80 lines of setup.
    """
    _gate._mlp = Perceptron
    _gate._ARGS = _PERCEPTRON_ARGS
    _gate._ARGNUMS = _PERCEPTRON_ARGNUMS
    _gate.SEED = seed
    return _gate.build_case()


def _run_episode(seed):
    """One seeded rollout through the REAL policy path, fixed markowitz order.

    Mirrors ``policy_regression_gate.run_trace``'s step loop (same calls, same
    order), plus ``--fixed-order markowitz`` (``run_trace`` does not pin one).
    Returns ``(case, records)`` where each record is what step t's Trajectory
    row would store (``face_counts``, ``face_delta_tokens``, the sampled
    ``FaceAction``'s ``skip``/``op_type``) paired with the successor
    ``delta_tokens``/``delta_count`` the SAME transition's real ``env.step``
    produced -- i.e. what Trajectory would store as step t+1's leaves.
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
        base_owners=base_own)
    base_mem = (base_s, base_c)
    vmem_s, vmem_c = CS.zero_memory(total_v, _gate.EMBD)

    fixed_order = fixed_order_for_env("markowitz", env)
    steps = max(1, num_valid - 1)
    keys = jrand.split(jrand.PRNGKey(seed), steps)
    part = jnp.zeros((total_v + 1,), jnp.float32)

    records = []
    for t in range(steps):
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

        def decide_fn(_v, _skips, _rows, _o=state.order,
                      _s=state.sparsity_specs, _k=state.step_count,
                      _fh=state.face_specs, _kh=state.face_skips):
            return case["decide_cb"](_o, _s, _k, _v, _fh, _kh, _skips, _rows)

        (vertex_idx, micro, _vd, _od, _id_, _jd, _ed, _kd,
         _mqlp, _mpv, _mcv, face_out, _value,
         _vctx) = agent.sample_action_dynamic(
            None, avail, state.axis_state, state.axis_valid_mask,
            case["factor_tables"], case["op_legality"], keys[t],
            preference=None, precomputed=precomputed,
            face_chunk_fn=chunk_fn, face_count_fn=count_fn,
            face_sizes_fn=sizes_fn, face_decide_fn=decide_fn,
            enc_carry=enc_carry,
        )

        face_action = None
        rec = None
        if face_out is not None:
            (fa, _f_logp, _f_ent, _f_pair, _f_comp, f_valid, f_cnt, f_dt,
             _f_ends) = face_out[:9]
            n_live = int(np.sum(np.asarray(f_valid) > 0.5))
            rec = dict(
                seed=seed, t=t, n_live=n_live,
                f_cnt=np.asarray(f_cnt, np.int64),
                f_dt=np.asarray(f_dt, np.int64),
                skip=np.asarray(fa.skip, np.int64),
                op_type=np.asarray(fa.op_type, np.int64),
            )
            face_action = fa

        env_action = agent.to_env_action_dynamic(
            vertex_idx, micro, state.axis_state, face_action=face_action)
        part = agent.participation_mask(
            total_v, jnp.asarray(int(vertex_idx), jnp.int32),
            (jnp.zeros((case["max_faces"], 2), jnp.int32)
             if face_out is None else face_out[8]),
            (jnp.zeros((case["max_faces"],), jnp.float32)
             if face_out is None else face_out[5]))

        env_out = env.step(state, env_action)
        state = env_out.state

        if rec is not None:
            rec["next_dt"] = np.asarray(state.delta_tokens, np.int64)
            rec["next_dc"] = int(state.delta_count)
            records.append(rec)

    return case, records


def _tokenizer_for(case):
    """A throwaway ``IncrementalPathTokenizer`` for ``.decode()`` ONLY.

    ``decode`` reads only the static vocabulary tables built at ``__init__``
    (never the elimination history), so a freshly constructed tokenizer over
    the SAME jaxpr/argnums/consts/args decodes exactly as the two live
    tokenizers under test would -- it is not a third source of tokens, only a
    dictionary for the diagnostic.
    """
    env = case["env"]
    cfg = env.config
    vocab = incr_token_vocab(None)
    return IncrementalPathTokenizer(
        cfg.jaxpr, cfg.argnums, list(env.consts), list(env.args),
        vocab_size=vocab)


def _check_pair(rec, tok):
    """Assert the three relations for one (seed, t) pair. Fails LOUDLY, with
    the first differing index, the two token ids and their decode, on the
    first mismatch."""
    f_cnt, f_dt = rec["f_cnt"], rec["f_dt"]
    next_dt, next_dc = rec["next_dt"], rec["next_dc"]
    total = int(f_cnt.sum())
    W = f_dt.shape[0]

    if total > next_dc:
        pytest.fail(
            "sum(face_counts[t]) exceeds delta_count[t+1]: the stored face "
            f"stream is LONGER than the delta it is supposed to prefix.\n"
            f"seed={rec['seed']} step={rec['t']} "
            f"sum(face_counts)={total} delta_count[t+1]={next_dc}")

    for i in range(total):
        a, b = int(f_dt[i]), int(next_dt[i])
        if a != b:
            lo, hi = max(0, i - 4), min(W, i + 5)
            ctx_face = [int(x) for x in f_dt[lo:hi]]
            ctx_next = [int(x) for x in next_dt[lo:hi]]
            pytest.fail(
                "THE CLAIM IS FALSE: face_delta_tokens[t] diverges from "
                "delta_tokens[t+1] inside the declared live prefix. The two "
                "streams come from two different tokenizer instances (the "
                "env's incremental stream and LiveFaceStream's own prefix "
                "tokenizer) and they disagree.\n"
                f"seed={rec['seed']} step={rec['t']} "
                f"first differing index={i} of {total} "
                f"(sum(face_counts)={total}, delta_count[t+1]={next_dc})\n"
                f"face_delta_tokens[t][{i}]={a}   "
                f"delta_tokens[t+1][{i}]={b}\n"
                f"context [{lo}:{hi}) face_delta_tokens  ={ctx_face}\n"
                f"context [{lo}:{hi}) delta_tokens[t+1]  ={ctx_next}\n"
                f"decode(face_delta_tokens context) ={tok.decode(ctx_face)!r}\n"
                f"decode(delta_tokens[t+1] context)  ={tok.decode(ctx_next)!r}\n"
                f"face_counts[t]={f_cnt.tolist()}")

    tail = f_dt[total:]
    bad = np.flatnonzero(tail != 0)
    if bad.size:
        i = total + int(bad[0])
        pytest.fail(
            "face_delta_tokens[t] carries a NON-ZERO token past "
            "sum(face_counts[t]) -- the stored buffer is not zero-padded "
            "where the claim says it must be.\n"
            f"seed={rec['seed']} step={rec['t']} index={i} "
            f"value={int(f_dt[i])} sum(face_counts)={total}")

    return total, next_dc


def test_the_stored_face_stream_of_step_t_equals_the_next_steps_delta_truncated_to_its_length():
    """Owner ruling Q2: face_delta_tokens[t] == delta_tokens[t+1][:sum(face_counts[t])],
    zero-padded past that length, with sum(face_counts[t]) <= delta_count[t+1],
    for every step t with a successor in the same episode.

    Checked against a REAL rollout (real Agent, real LiveFaceStream, real
    env.step), pooled over twelve weight-init seeds on a fixed (markowitz)
    vertex order, so the face head's draws vary while the graph and the order
    stay fixed. Fails on the FIRST mismatch, anywhere, with the index, the two
    token ids and their decode. If it does not fail, this also asserts that
    both approximation channels (SKIP and a live slot rule) were genuinely
    exercised somewhere in the pool -- otherwise a pass would only mean "no
    approximation ever ran," which proves nothing about the claim.
    """
    tok = None
    total_pairs = 0
    ratios = []
    gap_steps = 0
    any_skip = False
    any_rule = False

    for seed in _SEEDS:
        case, records = _run_episode(seed)
        if tok is None:
            tok = _tokenizer_for(case)
        assert records, f"seed {seed}: no step recorded a face decision at all"
        for rec in records:
            total, next_dc = _check_pair(rec, tok)
            total_pairs += 1
            if next_dc > 0:
                ratios.append(total / next_dc)
            if total < next_dc:
                gap_steps += 1

            n_live = rec["n_live"]
            if n_live:
                skip = rec["skip"][:n_live]
                op = rec["op_type"][:n_live]
                if np.any(skip == 1):
                    any_skip = True
                not_skipped = op[skip == 0]
                if not_skipped.size and np.any(not_skipped != OP_END):
                    any_rule = True

    assert total_pairs > 0, "no (seed, step) pair produced a face decision"
    assert any_skip, (
        f"no face used the SKIP bit across seeds {_SEEDS}; this pool cannot "
        "claim coverage of the echo the skip channel emits")
    assert any_rule, (
        f"no live, non-skipped face drew a real slot rule (op_type != "
        f"OP_END) across seeds {_SEEDS}; this pool cannot claim coverage of "
        "the echo a rule emits")

    max_ratio = max(ratios) if ratios else float("nan")
    print(f"[one-stream] pairs_checked={total_pairs} "
          f"max(sum(face_counts)/delta_count[t+1])={max_ratio:.6f} "
          f"steps_with_gap={gap_steps} any_skip={any_skip} any_rule={any_rule}",
          flush=True)
    # Also written to a file: `-q -x` (the sbatch template) does not show
    # captured stdout for a passing test.
    pathlib.Path(__file__).with_name("_onestream_stats.txt").write_text(
        f"pairs_checked={total_pairs}\n"
        f"max_ratio={max_ratio:.6f}\n"
        f"steps_with_gap={gap_steps}\n"
        f"any_skip={any_skip}\n"
        f"any_rule={any_rule}\n"
        f"seeds={list(_SEEDS)}\n")
