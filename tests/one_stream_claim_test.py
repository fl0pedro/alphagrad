#!/usr/bin/env python3
"""What the stored face stream actually is (owner ruling Q2, 2026-09-13,
settled 2026-09-14).

The question was whether ``ppo.Trajectory.face_delta_tokens[t]`` (built by
``Agent._face_loop`` from ``LiveFaceStream`` chunks DURING SAMPLING, before
the vertex's face decisions are final) equals ``ppo.Trajectory
.delta_tokens[t+1]`` (built by the env's OWN incremental tokenizer, from the
real ``env.step`` transition step t's decisions produced), truncated to
``sum(face_counts[t])``. It does not, in general -- and this file now PINS
the actual fact, established against a real rollout (real Agent, real
LiveFaceStream, real ``env.step``, ``--face-actions --unified-face-head
--live-faces --per-face-masks --dynamic-substeps`` -- ``--incremental-encode``
has been mandatory since stage 2, see ``ppo.main``) rather than asserting the
(false) equality and failing by design:

1. For every step t where every one of the vertex's live faces was left
   UNDECIDED (no ``skip``, no live slot rule -- i.e. every face is still
   exactly what the head saw) AND ``sum(face_counts[t]) > 0``, the two
   streams DO agree on that shared prefix::

       face_delta_tokens[t][:sum(face_counts[t])] == delta_tokens[t+1][:sum(face_counts[t])]

   This is the case the original claim was implicitly built on: with nothing
   decided, there is no approximation echo anywhere, so the face's chunk
   (its raw contraction) and the real emission are the same tokens.

2. For at least one step where a face WAS decided (skip or a live slot
   rule), the two streams DIVERGE, and the first differing index is at or
   after the START of the decided face's own chunk (``cumsum(face_counts)``
   up to that face) -- never inside an earlier, still-undecided face's
   share of the buffer.

THE CONCRETE INSTANCE (cluster job 65383, commit 9581de20, seed 0, step 4).
One live face (``n_live=1``), decided SKIP. ``face_counts[t] = [610, 0, ...]``
(the chunk read the face's full raw contraction while undecided),
``delta_count[t+1] = 12`` (the real emission is just the SKIP marker). First
differing index 6: ``face_delta_tokens[t][6] = 143`` vs
``delta_tokens[t+1][6] = 10``. Decoded context: the face-stream side reads
``'&#c&#14fns#b#1a=reshape'`` (the raw, undecided contraction -- it names a
real op, ``reshape``); the delta side reads ``'&#c&#14{}approxSKIP{'`` (the
actual emission -- ``approx SKIP {}``). Exactly the divergence
``Agent._face_loop``'s own docstring predicts ("chunk f's contraction is
deliberately unhooked (the face is undecided when read)... emission-window
replay measured ratio/max_log 778 at epoch 0"), and the live-rollout
confirmation of the mechanism ``tests/trajectory_layout_test.py`` had already
isolated against a hand-built reference on one hand-picked vertex.

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
record an ``approx`` block for either one. Weight-init seeds are widened
until BOTH relations above have at least one witness (in practice seed 0
alone supplies both: several undecided early steps and the decided SKIP at
step 4) -- never skipped or xfailed if a seed pool comes up short; the search
just widens.

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

# Weight-init seeds are tried 0, 1, 2, ... (the vertex order is pinned by
# markowitz, so only the face head's draws differ between them) until both
# relations have a witness. This is a cap on the WIDENING, not a fixed pool:
# exhausting it is a hard failure ("re-seed, do not skip"), never a skip.
_MAX_SEEDS = 60


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


def _decided_faces(rec):
    """Which of this step's LIVE faces were actually decided, in face-index
    order: ``skip == 1``, or a live slot rule (``op_type != OP_END`` on a
    face that was not skipped). Empty iff every live face is still exactly
    what the head saw -- undecided."""
    n_live = rec["n_live"]
    if n_live == 0:
        return []
    skip = rec["skip"][:n_live]
    op = rec["op_type"][:n_live]
    return [f for f in range(n_live)
            if skip[f] == 1 or np.any(op[f] != OP_END)]


def _first_difference(rec, total):
    """The first index in ``[0, total)`` where ``face_delta_tokens[t]`` and
    ``delta_tokens[t+1]`` disagree, or ``None`` if they agree throughout."""
    f_dt, next_dt = rec["f_dt"], rec["next_dt"]
    for i in range(total):
        if int(f_dt[i]) != int(next_dt[i]):
            return i
    return None


def _decode_ctx(rec, tok, i):
    """Decoded +/- 4 token window around index ``i`` of both streams, for a
    failure message."""
    f_dt, next_dt = rec["f_dt"], rec["next_dt"]
    W = f_dt.shape[0]
    lo, hi = max(0, i - 4), min(W, i + 5)
    ctx_face = [int(x) for x in f_dt[lo:hi]]
    ctx_next = [int(x) for x in next_dt[lo:hi]]
    return (lo, hi, ctx_face, ctx_next,
            tok.decode(ctx_face), tok.decode(ctx_next))


def _assert_undecided_prefix_matches(rec, tok, total):
    """RELATION (1): with every live face still undecided, the stored face
    stream and the successor's delta must agree on the shared prefix. Fails
    LOUDLY, with the first differing index, the two token ids and their
    decode, on the first mismatch."""
    diff = _first_difference(rec, total)
    if diff is None:
        return
    a, b = int(rec["f_dt"][diff]), int(rec["next_dt"][diff])
    lo, hi, ctx_face, ctx_next, dec_face, dec_next = _decode_ctx(rec, tok, diff)
    pytest.fail(
        "RELATION (1) IS VIOLATED: every live face of this step was left "
        "UNDECIDED (no skip, no live slot rule), so face_delta_tokens[t] and "
        "delta_tokens[t+1] were expected to agree on their shared prefix -- "
        "they do not.\n"
        f"seed={rec['seed']} step={rec['t']} n_live={rec['n_live']} "
        f"first differing index={diff} of {total} "
        f"(sum(face_counts)={total}, delta_count[t+1]={rec['next_dc']})\n"
        f"face_delta_tokens[t][{diff}]={a}   delta_tokens[t+1][{diff}]={b}\n"
        f"context [{lo}:{hi}) face_delta_tokens ={ctx_face}\n"
        f"context [{lo}:{hi}) delta_tokens[t+1] ={ctx_next}\n"
        f"decode(face_delta_tokens context) ={dec_face!r}\n"
        f"decode(delta_tokens[t+1] context)  ={dec_next!r}\n"
        f"face_counts[t]={rec['f_cnt'].tolist()}")


def _assert_decided_diverges_at_the_right_place(rec, tok, total, decided):
    """RELATION (2): with at least one live face decided, face_delta_tokens[t]
    and delta_tokens[t+1] must actually diverge, and not before the FIRST
    decided face's own chunk starts (``cumsum(face_counts)`` up to that
    face) -- an earlier, still-undecided face's share of the buffer is
    exactly the raw contraction the real emission also contains. Fails
    LOUDLY, with the same index/ids/decode diagnostic, if either half is
    violated."""
    f_cnt = rec["f_cnt"]
    f0 = decided[0]
    start = int(f_cnt[:f0].sum())
    diff = _first_difference(rec, total)

    if diff is None:
        pytest.fail(
            "RELATION (2) IS VIOLATED: a face was decided but "
            "face_delta_tokens[t] and delta_tokens[t+1] agree everywhere in "
            f"[0, {total}) anyway.\n"
            f"seed={rec['seed']} step={rec['t']} n_live={rec['n_live']} "
            f"decided faces={decided} decided_chunk_start={start} "
            f"sum(face_counts)={total} delta_count[t+1]={rec['next_dc']}\n"
            f"face_counts[t]={f_cnt.tolist()} "
            f"skip[:n_live]={rec['skip'][:rec['n_live']].tolist()} "
            f"op_type[:n_live]={rec['op_type'][:rec['n_live']].tolist()}")

    if diff < start:
        lo, hi, ctx_face, ctx_next, dec_face, dec_next = _decode_ctx(
            rec, tok, diff)
        a, b = int(rec["f_dt"][diff]), int(rec["next_dt"][diff])
        pytest.fail(
            "RELATION (2) IS VIOLATED: the streams diverged BEFORE the "
            "decided face's own chunk -- an earlier, still-undecided face's "
            "share of the buffer should have matched the real emission.\n"
            f"seed={rec['seed']} step={rec['t']} n_live={rec['n_live']} "
            f"decided faces={decided} decided_chunk_start={start} "
            f"first differing index={diff} of {total}\n"
            f"face_delta_tokens[t][{diff}]={a}   "
            f"delta_tokens[t+1][{diff}]={b}\n"
            f"context [{lo}:{hi}) face_delta_tokens ={ctx_face}\n"
            f"context [{lo}:{hi}) delta_tokens[t+1] ={ctx_next}\n"
            f"decode(face_delta_tokens context) ={dec_face!r}\n"
            f"decode(delta_tokens[t+1] context)  ={dec_next!r}\n"
            f"face_counts[t]={f_cnt.tolist()}")


def test_the_stored_face_stream_is_what_the_head_read_before_the_decision_and_diverges_from_the_next_delta_exactly_at_a_decided_face():
    """The positive pin (owner ruling Q2, settled 2026-09-14): the stored
    face stream is the tokens the head read BEFORE each face's decision, not
    a prefix of the next step's delta in general.

    (1) Every step whose live faces are ALL still undecided: the stored face
        stream and the next step's delta agree on their shared prefix (no
        decision, no divergence -- see ``_assert_undecided_prefix_matches``).
    (2) At least one step where a face WAS decided (skip or a live slot
        rule): the two streams diverge, and never before that decided face's
        own chunk (see ``_assert_decided_diverges_at_the_right_place``).

    Checked against a REAL rollout (real Agent, real LiveFaceStream, real
    env.step) on a fixed (markowitz) vertex order. Weight-init seeds widen
    from 0 until both relations have a witness (``_MAX_SEEDS`` caps the
    widening as a hard failure, never a skip -- "re-seed, do not skip").
    Fails on the FIRST violation of (1), anywhere, with the index, the two
    token ids and their decode; fails the same way if (2)'s witness turns out
    to violate its own half, or if no decided step exists inside the cap.
    """
    tok = None
    undecided_checked = 0
    decided_checked = 0
    last_seed = -1

    for seed in range(_MAX_SEEDS):
        last_seed = seed
        case, records = _run_episode(seed)
        if tok is None:
            tok = _tokenizer_for(case)
        for rec in records:
            total = int(rec["f_cnt"].sum())
            if total == 0:
                continue
            decided = _decided_faces(rec)
            if not decided:
                _assert_undecided_prefix_matches(rec, tok, total)
                undecided_checked += 1
            else:
                _assert_decided_diverges_at_the_right_place(
                    rec, tok, total, decided)
                decided_checked += 1
        if undecided_checked > 0 and decided_checked > 0:
            break

    assert undecided_checked > 0, (
        f"no step across seeds 0..{last_seed} had every live face left "
        "undecided with a non-empty chunk; re-seed (raise _MAX_SEEDS), do "
        "not skip -- relation (1) has no witness")
    assert decided_checked > 0, (
        f"no step across seeds 0..{last_seed} decided ANY face (skip or a "
        "live slot rule); re-seed (raise _MAX_SEEDS), do not skip -- "
        "relation (2) has no witness")

    print(f"[one-stream] seeds_tried=0..{last_seed} "
          f"undecided_steps_checked={undecided_checked} "
          f"decided_steps_checked={decided_checked}", flush=True)
    # Also written to a file: `-q -x` (the sbatch template) does not show
    # captured stdout for a passing test.
    pathlib.Path(__file__).with_name("_onestream_stats.txt").write_text(
        f"seeds_tried=0..{last_seed}\n"
        f"undecided_steps_checked={undecided_checked}\n"
        f"decided_steps_checked={decided_checked}\n")
