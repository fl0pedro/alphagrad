"""Bit-identical POLICY-PATH regression gate (Phase 0, part 3).

WHY THIS EXISTS
---------------
Phases 1-2 re-pad tensors (face bucketing) and restructure heads. Those are
exactly the edits that silently change MASKING: runs v22-v28 of this project
were lost to a ``-1e9`` mask sentinel colliding with the set-pointer sentinel,
which drove the measured cosine to 0 for six campaigns before anyone noticed.
Nothing crashed. The health lines looked fine.

This module is the tripwire. It runs a SHORT, seeded, fully deterministic
rollout through the REAL policy path --

    env.reset -> carry_stream.init_carry (base stream)
      per step:
        carry_stream.advance   (this step's token delta)
        carry_stream.heads     (vertex logits / contexts / value)
        Agent.sample_action_dynamic
          -> _face_loop  -> LiveFaceStream pure_callbacks (live per-face
             token chunks, prefix cache, face-enum cache)
          -> FacePathPolicy / UnifiedFacePolicy sampling
        Agent.to_env_action_dynamic
        env.step

-- and records a SEMANTIC golden artefact, then asserts bit-identical equality
against it on every later run.

WHAT IS COMPARED, AND WHY IT IS SHAPE-INDEPENDENT
-------------------------------------------------
Bucketing changes PADDING. A golden that pinned raw padded arrays would fail on
every bucket-width change for no reason, and would be deleted within a week.
So the golden stores the SEMANTIC content only:

* the realized vertex choice (and the 1-based target the env actually got),
* the set of AVAILABLE vertices, as a sorted index list (not a padded mask),
* the vertex mask's FILL VALUE for illegal slots, and the probability mass
  that leaks onto them,
* the per-face rule WIRES, for the live faces only (not the padded tail),
* the per-face legality masks, as sorted ``(i, j)`` / ``i`` index lists,
* every log-prob / entropy / value, as exact float32 BIT PATTERNS,
* the per-face token chunk counts and a hash of the tokens the head actually
  read (the live prefix of the emission window, never the padded tail),
* the env's own per-step token delta count.

Re-padding, re-bucketing, or widening any of ``MAX_FACES`` /
``MAX_DELTA_TOKENS`` / ``FACE_SLOTS``' padding tail leaves ALL of the above
untouched. Changing a mask sentinel, reordering a head's outputs, or shifting
the face index does not.

Floats are compared BIT-EXACTLY (``float32`` bit patterns, not ``allclose``).
That is the established bar in this codebase: ``tests/face_bucket_test.py``
already proves the full-width and bucketed face draws are bitwise identical, so
a tolerance here would only hide the class of bug this gate exists to catch.

RUNNING IT
----------
Check against the committed golden (this is the gate; exit 0 = pass)::

    JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py

Re-record the golden (ONLY after a deliberate, reviewed semantic change)::

    JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py --record

Dump the live trace as JSON without comparing::

    JAX_PLATFORMS=cpu uv run --no-sync python tests/policy_regression_gate.py --dump /tmp/t.json

It is also collected by pytest through ``policy_regression_gate_test.py``.

No GPU, no measurement. The rollout deliberately stops one step BEFORE the
terminal vertex, so ``env.step`` never reaches the measurement callback --
the policy path is exercised for real, the cost model is not exercised at all.

DETERMINISM CAVEAT
------------------
The golden pins float32 bit patterns produced by XLA:CPU. It is reproducible on
a fixed (jax, jaxlib, CPU) triple; the recorded ``fingerprint`` block reports
that triple and a MISMATCH is printed as a warning next to any diff, so a
jaxlib bump is never mistaken for a masking regression.
"""
from __future__ import annotations

import os

# Every one of these must be set BEFORE alphagrad/graphax import: env.py reads
# MAX_FACES / MAX_DELTA_TOKENS at module scope, and ppo._build_agent reads
# ALPHAGRAD_POLICY. Pinned (not `setdefault`-from-ambient) so the golden cannot
# silently depend on whatever the caller's shell happened to export.
os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cpu")
os.environ["GRAPHAX_ALLOW_PARTIAL_ORDER"] = "1"
os.environ["ALPHAGRAD_POLICY"] = "palimpsa"
os.environ["ALPHAGRAD_SKIP_COST_ANALYSIS"] = "1"
os.environ["ALPHAGRAD_SKIP_COUNT_OPS"] = "1"
os.environ["ALPHAGRAD_MAX_FACES"] = "16"
os.environ["ALPHAGRAD_MAX_DELTA_TOKENS"] = "1024"
os.environ["ALPHAGRAD_INCR_TOKEN_VOCAB"] = "512"
os.environ["ALPHAGRAD_INCREMENTAL_TOKENS"] = "1"
os.environ["ALPHAGRAD_FACE_ENUM_CACHE"] = "1"
os.environ["ALPHAGRAD_UNIFIED_FACE_ENUM"] = "1"
# The gate must pin the SAMPLED path, not a forced one.
os.environ["ALPHAGRAD_FORCE_REV"] = "0"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrand  # noqa: E402
import numpy as np  # noqa: E402

GOLDEN = Path(__file__).with_name("golden") / "policy_gate_golden.json"

SEED = 20260810
ROLLOUT_STEPS = None  # None => (num_valid - 1): stop before the terminal step
EMBD = 32


# --------------------------------------------------------------- the target
# `_mlp` is the graph `live_face_prefix_test` uses, and for a reason quoted
# from there: its first vertex is the one graph in the suite where a per-FACE
# Diag is PROVEN to land on the live operand. On smaller graphs every face
# hands its slots an operand where nothing is legal, `make_live_masked_hook`
# correctly skips, and a golden test would then be pinning the FAIL-SOFT path
# and calling it a pass. `assert_nontrivial` below re-checks that property on
# the recorded trace rather than trusting the comment.
# BRANCHED, not a chain. A chain MLP gives every vertex in-degree 1 and
# out-degree 1, i.e. exactly ONE face each, and a one-face-per-step golden
# cannot see a face-INDEX shift (the single most likely Phase-1 bucketing
# bug) because there is no index to shift. `h` here has out-degree 2, so at
# least one step decides several faces and their ORDER is pinned.
def _mlp(x, W1, W2, W3):
    h = jnp.tanh(x @ W1)
    a = jnp.tanh(h @ W2)
    b = jnp.tanh(h @ W3)
    return a * b


_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1,
         jnp.ones((32, 16)) * 0.1, jnp.ones((32, 16)) * 0.1)
_ARGNUMS = (1, 2, 3)


# ------------------------------------------------------------- bit encoding
def _f32bits(x) -> str:
    """Exact float32 bit pattern as 8 hex chars. NaN keeps its payload."""
    return struct.pack(">f", np.float32(x)).hex()


def _f32list(a) -> list:
    return [_f32bits(v) for v in np.asarray(a, np.float32).ravel().tolist()]


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(
        np.asarray(a, np.int32)).tobytes()).hexdigest()[:16]


def _idx(mask, thresh=0.5) -> list:
    """A 1-D 0/1 mask as a sorted index list -- padding-independent."""
    return [int(i) for i in np.flatnonzero(np.asarray(mask) > thresh)]


def _pairs(mask2d, thresh=0.5) -> list:
    """A 2-D 0/1 mask as a sorted ``[i, j]`` list -- padding-independent."""
    ii, jj = np.nonzero(np.asarray(mask2d) > thresh)
    return [[int(i), int(j)] for i, j in zip(ii, jj)]


# ------------------------------------------------------------------- set-up
def build_case():
    """Everything one deterministic rollout needs. Mirrors ``ppo.main``'s
    setup for the ``--live-faces --face-actions --dynamic-substeps`` arm the
    campaigns run; the pieces come from the SHARED modules ppo.py itself
    imports (``common.agent_factory``, ``common.carry_stream``,
    ``common.face_driver``, ``common.masks``), never from a local copy."""
    from alphagrad.approx.env import (
        MAX_AXES_PER_VERTEX, MAX_DELTA_TOKENS, MAX_FACES,
        VertexEliminationEnv,
    )
    from alphagrad.approx.common.agent_factory import (
        apply_policy_arch, build_and_init_agent)
    from alphagrad.approx.common.face_driver import (
        build_live_face_stream, make_face_callbacks)
    from alphagrad.approx.common.masks import build_vertex_valid_static
    from alphagrad.approx.heads import NUM_OPS, precompute_factor_tables
    from alphagrad.approx import ppo as P

    # az_gumbel's requirement, and ours: the quant hardware scan must be warmed
    # EAGERLY before any trace, or a first-touch inside the face while_loop
    # hands a later trace a dead tracer (UnexpectedTracerError).
    from graphax.sparse.micro_actions import report_hardware_scan
    report_hardware_scan()

    closed = jax.make_jaxpr(_mlp)(*_ARGS)
    env = VertexEliminationEnv.from_jaxpr(
        closed, args=list(_ARGS), argnums=_ARGNUMS, num_envs=0,
        target_fun=_mlp,
        # --face-actions implies per-face legality masking for the per-vertex
        # micro rules too (ppo.main sets `per_face=args.per_face or
        # args.face_actions`).
        per_face=True,
        # STAGE 2: the env emits the per-step token DELTA, not the stream.
        delta_obs=True,
    )
    total_v = len(closed.jaxpr.eqns)
    valid = [int(v) for v in np.asarray(env.valid_vertices)]
    num_valid = len(valid)

    ns = P.make_argparser().parse_args([])
    apply_policy_arch(
        ns,
        dynamic_substeps=True, unified_head=False, no_approx_head=False,
        face_actions=True, unified_face_head=True, live_faces=True,
        max_substeps=1, axis_group_embedding=False,
    )
    ns.embd_dim = EMBD
    ns.num_layers = 2
    ns.hidden_dim = 32
    ns.vocab_size = 512
    ns.preference_conditioned = False
    agent = build_and_init_agent(
        ns, total_v, num_factors=4, max_rules=4, seed=SEED)
    assert agent.face_path_policy is not None or \
        getattr(agent, "unified_face_policy", None) is not None, \
        "no face head was built -- the gate would cover nothing"

    stream = build_live_face_stream(
        closed.jaxpr, _ARGNUMS, list(closed.literals), list(_ARGS),
        max_faces=MAX_FACES, max_axes=MAX_AXES_PER_VERTEX,
        window=MAX_DELTA_TOKENS, cache=64,
    )
    chunk_cb, count_cb = make_face_callbacks(stream, window=MAX_DELTA_TOKENS)

    return dict(
        env=env, agent=agent, stream=stream,
        chunk_cb=chunk_cb, count_cb=count_cb,
        total_v=total_v, num_valid=num_valid, valid=valid,
        vertex_valid_static=build_vertex_valid_static(valid, total_v),
        factor_tables=precompute_factor_tables(64),
        op_legality=jnp.ones((NUM_OPS,), jnp.float32),
        max_faces=int(MAX_FACES), window=int(MAX_DELTA_TOKENS),
        max_axes=int(MAX_AXES_PER_VERTEX),
    )


# ----------------------------------------------------------------- the roll
def run_trace(case=None, steps=None):
    """One seeded rollout -> the semantic trace (a plain JSON-able dict)."""
    from alphagrad.approx.common import carry_stream as CS
    from alphagrad.approx.common.face_driver import bind_step_callbacks
    from alphagrad.approx.common.masks import vertex_avail_at_step
    from alphagrad.approx.ppo import _mask_vertex_logits

    case = case or build_case()
    env, agent = case["env"], case["agent"]
    total_v, num_valid = case["total_v"], case["num_valid"]
    if steps is None:
        steps = ROLLOUT_STEPS if ROLLOUT_STEPS is not None else num_valid - 1
    steps = max(1, int(steps))

    state = env.reset()
    base_tok, base_eqn, base_n = env.base_observation()
    base_w = max(int(base_n), 1)
    try:
        base_own = env.base_owners()
    except Exception:                                    # pragma: no cover
        base_own = None
    enc_carry, vmem_s, vmem_c = CS.init_carry(
        agent, base_tok[:base_w], base_eqn[:base_w], base_n,
        window=base_w, total_v=total_v, embd_dim=EMBD, base_owners=base_own)
    residual = jnp.zeros((total_v, EMBD), jnp.float32)

    keys = jrand.split(jrand.PRNGKey(SEED), steps)
    out_steps = []
    for t in range(steps):
        delta_owner = jnp.where(
            state.step_count > 0,
            state.order[jnp.maximum(state.step_count - 1, 0)],
            jnp.asarray(-1, jnp.int32)).astype(jnp.int32)
        enc_carry, vmem_s, vmem_c = CS.advance(
            agent, enc_carry, vmem_s, vmem_c,
            state.delta_tokens, state.delta_eqns, state.delta_count,
            delta_owner, window=case["window"])
        precomputed = CS.heads(agent, vmem_s, vmem_c,
                               vertex_features=None, residual_state=residual,
                               preference=None)
        avail = vertex_avail_at_step(
            state, case["vertex_valid_static"], total_v, num_valid)

        chunk_fn, count_fn = bind_step_callbacks(
            case["chunk_cb"], case["count_cb"],
            state.order, state.sparsity_specs, state.step_count,
            state.face_specs, state.face_skips)

        (vertex_idx, micro, vertex_dist, _od, _id_, _jd, _ed, _kd,
         micro_qlp, micro_pair_v, micro_comp_v, face_out, value,
         v_context) = agent.sample_action_dynamic(
            None, avail, state.axis_state, state.axis_valid_mask,
            case["factor_tables"], case["op_legality"], keys[t],
            eqn_ids=None, vertex_features=None, residual_state=residual,
            preference=None, precomputed=precomputed,
            face_chunk_fn=chunk_fn, face_count_fn=count_fn,
            enc_carry=enc_carry,
        )
        v = int(vertex_idx)

        rec = {
            "step": t,
            "vertex": v,
            "avail_vertices": _idx(avail),
            # only the AVAILABLE vertices' probabilities: the padded/illegal
            # tail is exactly what a sentinel bug corrupts, and its mass is
            # asserted separately below.
            "vertex_probs_live": [
                [int(i), _f32bits(np.asarray(vertex_dist)[i])]
                for i in _idx(avail)],
            "vertex_prob_illegal_mass": _f32bits(
                float(np.sum(np.where(np.asarray(avail) > 0.5, 0.0,
                                      np.asarray(vertex_dist))))),
            # THE SENTINEL LITERAL, pinned. `-inf -> -1e9` is invisible in
            # `vertex_dist` (float32 `exp(-1e9 - max)` flushes to exactly
            # 0.0, so the softmax is bit-identical) yet it is precisely the
            # edit that cost campaigns v22-v28. So the gate records the fill
            # VALUE the mask writes into the illegal slots, not only its
            # downstream effect.
            "vertex_mask_fill": sorted({
                _f32bits(x) for x in np.asarray(
                    _mask_vertex_logits(precomputed[0], avail)
                )[np.asarray(avail) <= 0.5]}),
            "value": _f32list(value),
            "micro_op_type": [int(x) for x in np.asarray(micro.op_type)],
            "micro_i": [int(x) for x in np.asarray(micro.i)],
            "micro_j": [int(x) for x in np.asarray(micro.j)],
            "micro_quant_logp": _f32list(micro_qlp),
            "micro_pair_valid": _pairs(micro_pair_v),
            "micro_compress_valid": _idx(micro_comp_v),
        }

        if face_out is not None:
            (fa, f_logp, f_ent, f_pair, f_comp, f_valid, f_cnt, f_dt,
             f_de) = face_out
            n_live = int(np.sum(np.asarray(f_valid) > 0.5))
            cnt = np.asarray(f_cnt, np.int32)
            tot = int(cnt[:n_live].sum()) if n_live else 0
            rec["face"] = {
                "n_live": n_live,
                "logp": _f32bits(f_logp),
                "entropy": _f32bits(f_ent),
                # per-face WIRES, live faces only -- the padded tail is
                # canonical by construction and carries no decision.
                "wires": [
                    {
                        "skip": int(np.asarray(fa.skip)[f]),
                        "op_type": [int(x) for x in np.asarray(fa.op_type)[f]],
                        "i": [int(x) for x in np.asarray(fa.i)[f]],
                        "j": [int(x) for x in np.asarray(fa.j)[f]],
                        "factor": [int(x) for x in np.asarray(fa.factor)[f]],
                        "compress_kind": [
                            int(x) for x in np.asarray(fa.compress_kind)[f]],
                        "quant_dtype": [
                            int(x) for x in np.asarray(fa.quant_dtype)[f]],
                        "quant_sign": [
                            int(x)
                            for x in np.asarray(fa.quant_scale_sign)[f]],
                        "quant_frac": _f32list(
                            np.asarray(fa.quant_scale_frac)[f]),
                    }
                    for f in range(n_live)
                ],
                # the LEGALITY masks, as index lists: this is the field the
                # v22-v28 sentinel collision silently emptied.
                "pair_valid": [_pairs(np.asarray(f_pair)[f])
                               for f in range(n_live)],
                "comp_valid": [_idx(np.asarray(f_comp)[f])
                               for f in range(n_live)],
                # the live-faces STREAM's own output: how many tokens each
                # face read, and a hash of the tokens themselves (live prefix
                # of the emission window only, never the padded tail).
                "chunk_counts": [int(x) for x in cnt[:n_live]],
                "chunk_tokens_sha": _sha(np.asarray(f_dt)[:tot]),
                "chunk_eqns_sha": _sha(np.asarray(f_de)[:tot]),
            }
            face_action = fa
        else:
            rec["face"] = None
            face_action = None

        env_action = agent.to_env_action_dynamic(
            vertex_idx, micro, state.axis_state, face_action=face_action)
        rec["target_vertex"] = int(env_action.target_vertex)
        rec["rule_specs"] = [
            [int(c) for c in row]
            for row in np.asarray(env_action.rule_specs)
            if int(row[0]) >= 0 or int(row[1]) >= 0
        ]
        if face_action is not None:
            n_live = rec["face"]["n_live"]
            rec["face_rows"] = [
                [[int(c) for c in slot] for slot in face]
                for face in np.asarray(env_action.face_rows)[:n_live]
            ]
            rec["face_skip"] = [
                int(x) for x in np.asarray(env_action.face_skip)[:n_live]]

        env_out = env.step(state, env_action)
        state = env_out.state
        residual = agent.update_residual(residual, vertex_idx, v_context)
        rec["env_delta_count"] = int(state.delta_count)
        rec["env_order_prefix"] = [
            int(x) for x in np.asarray(state.order)[:int(state.step_count)]]
        out_steps.append(rec)

    import jaxlib
    return {
        "fingerprint": {
            "jax": jax.__version__,
            "jaxlib": getattr(jaxlib, "__version__", "?"),
            "python": "%d.%d" % sys.version_info[:2],
            "platform": str(jax.devices()[0].platform),
        },
        "config": {
            "seed": SEED, "steps": steps, "total_v": total_v,
            "valid_vertices": case["valid"], "embd_dim": EMBD,
            "max_faces": case["max_faces"], "window": case["window"],
        },
        "steps": out_steps,
    }


# ------------------------------------------------------------ non-triviality
def assert_nontrivial(trace):
    """A golden that pins the fail-soft path is worse than no golden: it goes
    green forever while the thing it claims to protect is dead. These are the
    minimum conditions for the trace to MEAN anything."""
    problems = []
    steps = trace["steps"]
    if not steps:
        problems.append("no steps recorded")
    faces = [s["face"] for s in steps if s.get("face")]
    if not faces:
        problems.append("the face head was never consulted (face_out is None "
                        "on every step) -- the gate covers no approx path")
    if not any(f["n_live"] > 0 for f in faces):
        problems.append("every step had ZERO live faces -- the live-faces "
                        "stream never produced a decision")
    if not any(sum(f["chunk_counts"]) > 0 for f in faces):
        problems.append("no face read a single token -- the LiveFaceStream "
                        "callbacks are not on the exercised path")
    if not any(f["pair_valid"] and any(p for p in f["pair_valid"])
               for f in faces):
        problems.append("no face had ANY legal DIAG pair -- this is the "
                        "fail-soft path; pick a graph where a per-face Diag "
                        "lands on the live operand")
    if len({s["vertex"] for s in steps}) < min(2, len(steps)):
        problems.append("the rollout never moved off one vertex")
    # OP_END == 3 (heads.OP_END). A trace of nothing but END wires would pin
    # the do-nothing branch of the head.
    if not any(o != 3 for f in faces for w in f["wires"] for o in w["op_type"]):
        problems.append("every sampled wire is END -- the approximation head "
                        "never emitted a rule, so the golden pins only the "
                        "no-op branch")
    # Without a multi-face step a face-INDEX shift is invisible: there is no
    # ordering for the golden to pin.
    if not any(f["n_live"] >= 2 for f in faces):
        problems.append("no step decided 2+ faces -- a face-index shift "
                        "would be undetectable in this trace")
    if problems:
        raise AssertionError(
            "the recorded trace is TRIVIAL and would not catch anything:\n  - "
            + "\n  - ".join(problems))


# ------------------------------------------------------------------ compare
def _walk(prefix, a, b, out, limit):
    if len(out) >= limit:
        return
    if type(a) is not type(b) and not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append((prefix, a, b))
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((f"{prefix}.{k}", "<absent in golden>", b[k]))
            elif k not in b:
                out.append((f"{prefix}.{k}", a[k], "<absent in live>"))
            else:
                _walk(f"{prefix}.{k}", a[k], b[k], out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((f"{prefix} (length)", len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(f"{prefix}[{i}]", x, y, out, limit)
    elif a != b:
        out.append((prefix, a, b))


def compare(golden, live, limit=25):
    """``(ok, report)``. The report names the FIRST differing step and the
    exact field -- never a bare ``assert x == y`` on two 40 kB blobs."""
    lines = []
    gf, lf = golden.get("fingerprint", {}), live.get("fingerprint", {})
    if gf != lf:
        lines.append(
            "NOTE  toolchain fingerprint differs from the golden's:\n"
            f"        golden {gf}\n        live   {lf}\n"
            "      float32 bit patterns are only reproducible on a fixed "
            "(jax, jaxlib, platform) triple. If the ONLY diffs below are "
            "float bit patterns, suspect the toolchain before the code; if "
            "any INTEGER field (a vertex, a wire, a mask index, a count) "
            "differs, the toolchain is irrelevant -- that is a real "
            "regression.")
    if golden.get("config") != live.get("config"):
        d = []
        for k in sorted(set(golden.get("config", {}))
                        | set(live.get("config", {}))):
            g, l = golden.get("config", {}).get(k), live.get("config", {}).get(k)
            if g != l:
                d.append(f"          {k}: golden={g!r}  live={l!r}")
        lines.append("FAIL  the rollout CONFIG changed, so the two traces are "
                     "not comparable:\n" + "\n".join(d))

    gs, ls = golden.get("steps", []), live.get("steps", [])
    if len(gs) != len(ls):
        lines.append(f"FAIL  step count: golden={len(gs)} live={len(ls)}")
    for t, (g, l) in enumerate(zip(gs, ls)):
        diffs = []
        _walk("", g, l, diffs, limit)
        if diffs:
            lines.append(f"FAIL  first divergence at STEP {t} "
                         f"({len(diffs)} differing field(s), showing "
                         f"{min(len(diffs), limit)}):")
            for path, gv, lv in diffs[:limit]:
                p = path.lstrip(".") or "<root>"
                lines.append(f"        {p}")
                lines.append(f"            golden : {gv!r}")
                lines.append(f"            live   : {lv!r}")
            lines.append(_hint(diffs))
            break
    ok = not any(x.startswith("FAIL") for x in lines)
    return ok, "\n".join(lines)


def _hint(diffs):
    """Point at the usual suspect for the FIELDS that moved."""
    paths = " ".join(p for p, _, _ in diffs)
    if ("pair_valid" in paths or "comp_valid" in paths or "avail" in paths
            or "mask_fill" in paths or "illegal_mass" in paths):
        return ("      HINT  a LEGALITY/AVAILABILITY mask changed. This is "
                "the v22-v28 failure mode: a mask sentinel colliding with "
                "the set-pointer sentinel emptied these lists and drove "
                "cosine to 0 for six campaigns. Check the mask fill value "
                "(-inf, never -1e9) and the padding of the face masks.")
    if "wires" in paths or "face_rows" in paths or "op_type" in paths:
        return ("      HINT  the realized RULE WIRES changed. Either a head's "
                "outputs were reordered, or the face INDEX shifted (face f's "
                "decision landed on face f+-1).")
    if "chunk" in paths:
        return ("      HINT  the live-faces TOKEN STREAM changed: the head is "
                "reading a different graph from the one it read before. "
                "Check the prefix-cache KEY (it must carry the face wires) "
                "and the prefix replay's face transforms.")
    if "logp" in paths or "prob" in paths or "value" in paths:
        return ("      HINT  only distributions moved. If every integer field "
                "is intact this is an arithmetic/ordering change in a head "
                "(or a toolchain bump -- see the fingerprint note).")
    return ""


# --------------------------------------------------------------------- main
def check(golden_path=GOLDEN):
    """Used by the pytest wrapper. Raises AssertionError with the report."""
    if not Path(golden_path).exists():
        raise AssertionError(
            f"no golden at {golden_path}; record one with\n"
            f"    JAX_PLATFORMS=cpu uv run --no-sync python "
            f"tests/policy_regression_gate.py --record")
    golden = json.loads(Path(golden_path).read_text())
    live = run_trace()
    assert_nontrivial(live)
    ok, report = compare(golden, live)
    if not ok:
        raise AssertionError(
            "POLICY-PATH REGRESSION GATE FAILED\n" + report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", action="store_true",
                    help="(re)write the golden from this build")
    ap.add_argument("--dump", metavar="PATH",
                    help="write the live trace as JSON and exit")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)

    live = run_trace(steps=a.steps)
    assert_nontrivial(live)
    blob = json.dumps(live, indent=1, sort_keys=True)

    if a.dump:
        Path(a.dump).write_text(blob)
        print(f"[gate] trace written to {a.dump}")
        return 0
    if a.record:
        Path(a.golden).parent.mkdir(parents=True, exist_ok=True)
        Path(a.golden).write_text(blob)
        n = len(live["steps"])
        nf = sum(s["face"]["n_live"] for s in live["steps"] if s["face"])
        print(f"[gate] RECORDED {a.golden}\n"
              f"       {n} steps, {nf} live face decisions, "
              f"{sum(sum(s['face']['chunk_counts']) for s in live['steps'] if s['face'])}"
              f" face tokens read")
        return 0

    if not Path(a.golden).exists():
        print(f"[gate] NO GOLDEN at {a.golden} -- record one with --record",
              file=sys.stderr)
        return 2
    golden = json.loads(Path(a.golden).read_text())
    ok, report = compare(golden, live)
    if report:
        print(report)
    if ok:
        n = len(live["steps"])
        nf = sum(s["face"]["n_live"] for s in live["steps"] if s["face"])
        print(f"[gate] PASS -- bit-identical to {a.golden} "
              f"({n} steps, {nf} live face decisions)")
        return 0
    print("[gate] FAIL -- the policy path changed semantically.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
