from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, NamedTuple, Sequence

import jax
import jax._src.core as core
import jax.numpy as jnp
import jax.random as jrand
from jax import Array, jit
from jax.experimental import io_callback
from jax.tree_util import register_pytree_node_class

import numpy as np

from alphagrad.approx.common.relations import compute_eqn_ids_from_tokens
from graphax.core import _build_graph, extract_jaxpr, jacve, vertex_elimination_jaxpr
from graphax.jaxpr import get_vocab as _graphax_get_vocab
from graphax.sparse.micro_actions import COMPRESS_KINDS, Compress, Diag
from jax_memory_monitor import ResourceMonitor

import math as _math

# Cache the graphax vocabulary used by `compute_eqn_ids_from_tokens` — the
# tokenizer always uses the same digit_base, so the vocab is constant and
# rebuilding it on every callback is pure overhead.
_TOKEN_VOCAB, _, _ = _graphax_get_vocab()

MAX_TOKENS = 4096
# Upper bound on rule_specs rows per vertex. In dynamic-substeps mode this
# also bounds the number of typed micro-actions per vertex that survive
# :func:`micro_actions_to_rule_specs_jax` — set it to the same scale as
# the policy's ``max_substeps`` (≈ 2 × MAX_AXES_PER_VERTEX) so the
# translator doesn't silently truncate DIAG / COMPRESS rows the policy
# emitted. Memory cost is O(total_v × MAX_RULES_PER_VERTEX × 3) int32.
MAX_RULES_PER_VERTEX = 16
NUM_AXIS_PAIRS = 4

# Per-vertex axis-state observation surface. The policy's dynamic action
# space (DIAG / COMPRESS / END) emits indices into a per-vertex axis set;
# this is the static observation that feeds heads.py's `AxisSetEncoder`.
# `MAX_AXES_PER_VERTEX` is the JAX-static upper bound on axes any vertex
# can have — most graphax ops have ≤ 4-6 axes (out_ndim + min_in_ndim);
# 8 leaves headroom without bloating state. `AXIS_FEATURE_DIM`'s four
# fields are [size, is_output, is_compressed, group_id]: only the first
# two are populated today (the others are placeholders for the future
# DIAG/COMPRESS state updates that micro_actions wiring will fill in).
MAX_AXES_PER_VERTEX = 8
AXIS_FEATURE_DIM = 4
_AXIS_FEAT_SIZE = 0
_AXIS_FEAT_IS_OUTPUT = 1
_AXIS_FEAT_IS_COMPRESSED = 2
_AXIS_FEAT_GROUP_ID = 3

# Canonical 8-component reward vector layout. The env reports raw reward values
# in the convention "higher is better": every cost component is stored *negated*
# (so r = -cost), `cosine_sim` is in [0, 1] (1 = identical Jacobian), and
# `frob_residual` is stored as `-||J_e - J_a||_F / ||J_e||_F` so larger residuals
# correspond to lower reward. Downstream code can therefore treat all 8 entries
# uniformly as "reward to maximize".
#
# Compute family (indices 0..5):
#   0 muls_adds_fmas   — graphax `adds + muls + fmas` op count from VE.
#   1 flops            — XLA cost-analysis FLOPs of the compiled approx fn.
#   2 latency_ns       — wall-clock latency in ns (only populated when
#                        `EnvConfig.measure_latency` is True; else 0).
#   3 max_io_sum       — graphax `mem` accumulator = sum over Jacobian
#                        accumulations of `max(in_size, out_size, edge_out_size)
#                        * itemsize`. (This is the "sum of max-input/max-output
#                        sizes per Jacobian accumulation" metric in the spec.)
#   4 bytes_accessed   — XLA cost-analysis bytes-accessed of the approx fn.
#   5 peak_memory      — peak HBM bytes during a single execution of the approx
#                        fn, captured via `ResourceMonitor`.
# Quality family (indices 6..7):
#   6 cosine_sim       — cosine similarity between flattened approximated and
#                        exact Jacobians, averaged over the calibration samples.
#   7 frob_residual    — relative Frobenius residual ||J_e - J_a||_F / ||J_e||_F.
NUM_REWARDS = 8
REWARD_NAMES: tuple[str, ...] = (
    "muls_adds_fmas",
    "flops",
    "latency_ns",
    "max_io_sum",
    "bytes_accessed",
    "peak_memory",
    "cosine_sim",
    "frob_residual",
)
REWARD_INDEX = {name: i for i, name in enumerate(REWARD_NAMES)}
COMPUTE_REWARD_INDICES = tuple(range(0, 6))  # cost components
QUALITY_REWARD_INDICES = (6, 7)             # cosine, frobenius

# Sentinel reward returned when a per-vertex transform sequence matches an
# entry in the in-file blacklist (used during exploration to penalise
# pathological configurations). The blacklist is no longer wired up after
# the typed-transform migration; the array is kept for potential reuse.
_SENTINEL_BAD_REWARD = jnp.array(
    [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1.0, -1e10],
    dtype=jnp.float32,
)

# Axis pair index -> (base_idx1, base_idx2). base_idx1 picks output axis 0/1; base_idx2 picks input axis 0/1.
axis_pair_idx_to_base = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}

# Sentinel used in `sparsity_specs[v, slot, 0]` to flag a COMPRESS sub-step
# (vs the regular `bi1 >= 0` DIAG payload or `bi1 == -1` end-of-sequence
# marker). The value is encoded as ``-2`` so existing `bi1 < 0` guards still
# recognise the slot as "not a DIAG", and `_callback` dispatches on the
# specific sentinel value.
#
# Caveat — COMPRESS through the vertex elimination DAG is only partial:
# `_callback` emits the correct `graphax.sparse.micro_actions.Compress`,
# but graphax's `_eliminate_vertex` assumes every edge keeps its nominal
# `(out_dims, primal_dims)` shape. Compress is lossy and reduces
# `val.ndim`, so a Compress applied to a vertex whose edge feeds into a
# subsequent elimination step trips the shape-preservation assertion at
# `core.py:417`. Practical implications:
#   * COMPRESS works end-to-end when it lands on the LAST vertex of the
#     elimination order (no downstream edge to matmul against).
#   * Earlier vertices in the order will assert. Hold off on
#     ``--allow-compress`` unless you've ordered the agent to only emit
#     COMPRESS on the final vertex, or are prepared to do the graphax
#     pre_transforms / shape-bookkeeping work.
# The reverse direction — graphax silently dropping a transform whose
# axes don't fit `val.ndim` at all (e.g. axis 1 on a 1-D val) — has been
# fixed (graphax commit `fa0a088`).
COMPRESS_SENTINEL = -2


class EnvState(NamedTuple):
    order: Array
    # (N, MAX_RULES_PER_VERTEX, 3) int32. Per-slot row layout depends on
    # the leading column:
    #   row[0] >= 0:                 DIAG with `(bi1=row[0], bi2=row[1],
    #                                factor=row[2])`. bi1/bi2 are
    #                                base-axis positions (output-side /
    #                                primal-side, respectively).
    #   row[0] == COMPRESS_SENTINEL: COMPRESS with axis `row[1]` (physical
    #                                index into the SparseTensor edge:
    #                                out axes 0..out_len-1, then primal
    #                                axes out_len.. ). row[2] unused.
    #   row[0] == -1:                end-of-sequence sentinel; every slot
    #                                past it is treated as unused.
    sparsity_specs: Array
    tokens: Array
    eqn_ids: Array  # (MAX_TOKENS,) int32; per-token equation ID, -1 for non-eqn tokens
    # Per-vertex axis state — observation surface for the dynamic action
    # space. `axis_state` is a packed int32 array of (size, is_output,
    # is_compressed, group_id) per axis slot; `axis_valid_mask` flags
    # which slots carry a real axis (vs. padding up to MAX_AXES_PER_VERTEX).
    # After `step()` the row for the just-eliminated vertex is updated to
    # reflect DIAG group_ids / shrunk sizes and COMPRESS marks
    # (see `_apply_rules_to_axis_state`). Downstream propagation across
    # the jaxpr DAG (where vertex `v`'s output axes feed into vertex
    # `v'`'s input axes later in the order) is intentionally not
    # implemented — the agent never revisits an eliminated vertex, and
    # the policy carries its own per-substep axis state through the
    # heads.py scan, so the missing signal is "useful debug metadata"
    # not "training signal".
    axis_state: Array          # (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM) int32
    axis_valid_mask: Array     # (total_v, MAX_AXES_PER_VERTEX) float32
    step_count: Array
    max_steps: int
    reward: Array  # (NUM_REWARDS,) float32; see REWARD_NAMES for layout
    terminated: bool


class EnvOut(NamedTuple):
    state: EnvState
    reward: Array
    terminated: bool


class StepAction(NamedTuple):
    target_vertex: Array  # scalar int32
    rule_specs: Array  # (MAX_RULES_PER_VERTEX, 3) int32; row [base_idx1, base_idx2, factor]; base_idx1 < 0 marks unused


class EnvConfig(NamedTuple):
    jaxpr: core.Jaxpr
    argnums: tuple[int, ...]
    has_aux: bool
    sparse: bool
    # cmp_type / mem_type used to gate which compute/memory metric was measured.
    # The env now reports the full 8-component reward vector every step, so they
    # are kept only as *primary-metric hints* for legacy CLI/host-side reporting.
    # New callers should ignore them and pick the desired component from the
    # reward vector explicitly via REWARD_INDEX.
    cmp_type: Literal["graphax", "flops", "latency"]
    mem_type: Literal["graphax", "bytes_accessed", "peak_memory"]
    target_fun: Callable | None = None
    data_gen: Callable | None = None
    exec_on_gpu: bool = False
    # Latency requires running the compiled fn 10x per step, which roughly 10xs
    # rollout-to-reward time. Off by default; flip on when the latency component
    # of the reward is actually being weighted.
    measure_latency: bool = False
    # Skip the expensive jacve-compile/exec branch on every step EXCEPT the
    # terminal one. Tokens/eqn_ids are still produced (the agent needs them as
    # the next observation), but the reward vector is zero on intermediate
    # steps and fully populated only when the order is complete. This is the
    # paper-native form for AlphaZero / GDPO / GFlowNet and works fine for PPO
    # / MuZero (just yields a sparse reward signal).
    terminal_rewards_only: bool = False


def _get_partials(order, sparsity_specs, stop):
    v_stop = int(stop)
    partial_order = order[:v_stop] if v_stop < len(order) else order
    partial_specs = (
        sparsity_specs[:v_stop] if v_stop < len(sparsity_specs) else sparsity_specs
    )
    return partial_order, partial_specs


def _apply_rules_to_axis_state(axis_state_v: Array, rule_specs: Array) -> Array:
    """Mutate one vertex's axis_state to reflect the rules just applied.

    Single-vertex update — the downstream / symbolic-shape propagation
    across the jaxpr DAG (where compressed / paired axes of vertex `v`
    show up as input axes of vertex `v'` later in the order) is the
    "deeper" piece called out in the env's roadmap and is **not**
    implemented here. The reason is that nothing in this scope actually
    needs the downstream signal: once a vertex is eliminated the agent
    never revisits it, and the policy carries its own per-substep axis
    state through `_features_after_diag` / `_features_after_compress` in
    `heads.py`. What this function buys is:

    * Honest top-N / replay output — `EnvState.axis_state[v]` after
      `step()` shows what the rules did to vertex `v`, which is helpful
      for debugging.
    * A baseline for the future cross-vertex propagation: when that
      lands, it will read the per-vertex mutations from here and
      forward them along the data-flow edges.

    Per rule:

    * DIAG ``(bi1, bi2, factor)`` — the output axis at relative position
      ``bi1`` and the primal axis at relative position ``bi2`` are paired
      under a fresh `group_id`. Their sizes shrink by `factor` (so an
      original (4, 4) pair with factor=2 becomes (2, 2)).
    * COMPRESS ``(SENTINEL, axis, _)`` — the axis at the recorded token
      position is marked `is_compressed = 1` and its size collapses to 1.
    * Unused / END sentinel rows leave the state unchanged.

    Implementation is JAX-traceable so callers inside the jitted
    `step()` can use it. ``rule_specs`` is iterated with
    ``lax.fori_loop`` and per-slot updates are gated by ``jnp.where``.
    """
    is_output = axis_state_v[:, _AXIS_FEAT_IS_OUTPUT]
    n_out = jnp.sum(is_output).astype(jnp.int32)

    # The fresh group_id starts after the largest existing one — that
    # way we don't overwrite groups recorded by earlier steps on this
    # same vertex (if any) and the per-vertex group sequence stays
    # monotonic. `_AXIS_FEAT_GROUP_ID` defaults to -1 (ungrouped), so
    # max(-1, ...) + 1 = 0 on the first DIAG.
    init_gid = jnp.max(axis_state_v[:, _AXIS_FEAT_GROUP_ID]) + 1

    def _body(slot, carry):
        state, gid = carry
        row = rule_specs[slot]
        bi1 = row[0]
        bi2 = row[1]
        factor = row[2]

        is_diag = bi1 >= 0
        is_compress = bi1 == COMPRESS_SENTINEL

        # DIAG: pair the (bi1, n_out + bi2) axes under `gid` and shrink
        # both sizes by `factor`. Clamp factor to >= 1 so the dummy
        # path (`factor == 0` from the unused row) leaves sizes alone.
        diag_out_tok = jnp.clip(bi1, 0, axis_state_v.shape[0] - 1)
        diag_prim_tok = jnp.clip(n_out + bi2, 0, axis_state_v.shape[0] - 1)
        safe_factor = jnp.maximum(factor, 1)

        def _apply_diag(s):
            s = s.at[diag_out_tok, _AXIS_FEAT_GROUP_ID].set(gid)
            s = s.at[diag_prim_tok, _AXIS_FEAT_GROUP_ID].set(gid)
            s = s.at[diag_out_tok, _AXIS_FEAT_SIZE].set(
                jnp.maximum(s[diag_out_tok, _AXIS_FEAT_SIZE] // safe_factor, 1)
            )
            s = s.at[diag_prim_tok, _AXIS_FEAT_SIZE].set(
                jnp.maximum(s[diag_prim_tok, _AXIS_FEAT_SIZE] // safe_factor, 1)
            )
            return s

        # COMPRESS: mark axis is_compressed, collapse size to 1.
        comp_tok = jnp.clip(bi2, 0, axis_state_v.shape[0] - 1)

        def _apply_compress(s):
            s = s.at[comp_tok, _AXIS_FEAT_IS_COMPRESSED].set(1)
            s = s.at[comp_tok, _AXIS_FEAT_SIZE].set(1)
            return s

        state = jax.lax.cond(is_diag, _apply_diag, lambda s: s, state)
        state = jax.lax.cond(is_compress, _apply_compress, lambda s: s, state)
        new_gid = jnp.where(is_diag, gid + 1, gid)
        return state, new_gid

    final_state, _ = jax.lax.fori_loop(
        0, MAX_RULES_PER_VERTEX, _body, (axis_state_v, init_gid)
    )
    return final_state


def compute_static_axis_state(jaxpr, total_v: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex axis features extracted statically from the jaxpr.

    Each vertex's axes are concatenated as ``(out_dims..., primal_dims...)``
    into a fixed-size slot of ``MAX_AXES_PER_VERTEX``. The primal proxy
    is the first non-literal input variable (matching the convention used
    by ``vertex_axis_dims`` in common/masks.py). Returns:

    * ``axis_state`` — ``(total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)``
      int32. Field layout: ``[size, is_output, is_compressed, group_id]``.
      ``is_compressed`` and ``group_id`` are placeholders (0 / -1) until
      the heads.py wiring lands and the env starts mutating them per
      sub-step.
    * ``axis_valid_mask`` — ``(total_v, MAX_AXES_PER_VERTEX)`` float32.
      ``1.0`` for slots carrying a real axis.

    Vertices with no shape info (literal-only inputs, etc.) get an
    all-zero / all-invalid row — same convention as
    ``vertex_axis_dims``.
    """
    axis_state = np.zeros(
        (total_v, MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM), dtype=np.int32,
    )
    axis_state[..., _AXIS_FEAT_GROUP_ID] = -1  # ungrouped sentinel
    axis_valid = np.zeros((total_v, MAX_AXES_PER_VERTEX), dtype=np.float32)

    for v_idx, eqn in enumerate(jaxpr.eqns):
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue
        out_shape = eqn.outvars[0].aval.shape
        invars = [v for v in eqn.invars if hasattr(v, "aval")]
        primal_shape = invars[0].aval.shape if invars else ()

        slot = 0
        for size in out_shape:
            if slot >= MAX_AXES_PER_VERTEX:
                break
            axis_state[v_idx, slot, _AXIS_FEAT_SIZE] = int(size)
            axis_state[v_idx, slot, _AXIS_FEAT_IS_OUTPUT] = 1
            axis_valid[v_idx, slot] = 1.0
            slot += 1
        for size in primal_shape:
            if slot >= MAX_AXES_PER_VERTEX:
                break
            axis_state[v_idx, slot, _AXIS_FEAT_SIZE] = int(size)
            axis_state[v_idx, slot, _AXIS_FEAT_IS_OUTPUT] = 0
            axis_valid[v_idx, slot] = 1.0
            slot += 1

    return axis_state, axis_valid


# ---------------------------------------------------------------------------
# Typed MicroAction -> EnvState.sparsity_specs row translator
# ---------------------------------------------------------------------------
#
# The heads.py policy emits a typed `MicroAction(op_type, i, j, exponents,
# factor)` sequence per vertex. The env's _callback turns the stored
# specs (one (base_idx1, base_idx2, factor) row per slot) into typed
# graphax.sparse.micro_actions.Diag entries before handing them to
# graphax's `transforms` API. This translator converts the policy's
# typed action sequence into the 3-tuple row format the EnvState carries
# in `sparsity_specs` — the rest of the env then dispatches normally.
# COMPRESS micro-actions are silently dropped today since the graphax
# `transforms` API only handles Diag end-to-end through the env's reward
# path (a Compress callable would need to be threaded all the way to
# graphax's per-vertex transform list — straightforward but not yet wired).


def micro_actions_to_rule_specs(
    op_types,
    i_indices,
    j_indices,
    factors,
    *,
    axis_state_for_vertex,
    compress_kinds=None,
):
    """Translate a sub-episode's typed micro-actions into legacy rule_specs.

    Args:
        op_types: (S,) int32 — per-sub-step op type (heads.py OP_DIAG /
            OP_COMPRESS / OP_END).
        i_indices: (S,) int32 — axis-token index for `i` (DIAG and COMPRESS).
        j_indices: (S,) int32 — axis-token index for `j` (DIAG only).
        factors: (S,) int32 — explicit positive factor (DIAG only),
            already collapsed from the prime-exponent head.
        axis_state_for_vertex: ``(MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)``
            int32 — used to map axis-token indices to the legacy
            ``base_idx1 / base_idx2`` (out_axis_position /
            primal_axis_position) representation. ``is_output`` of each
            axis token (column ``_AXIS_FEAT_IS_OUTPUT``) determines which
            side of the pair it lands on; the relative position is the
            running count of output-or-primal axes encountered before it.

    Returns:
        rule_specs: ``(MAX_RULES_PER_VERTEX, 3)`` int32 — same layout
        the env consumes. DIAG rows are ``[bi1, bi2, factor]``;
        COMPRESS rows are ``[COMPRESS_SENTINEL, physical_axis, kind_idx]``
        where ``kind_idx`` indexes :data:`COMPRESS_KINDS`.
        Slots past the first ``OP_END`` (or past ``MAX_RULES_PER_VERTEX``,
        whichever comes first) are filled with the unused sentinel
        ``[-1, -1, 0]``.
    """
    # Lazy import to avoid circular dependency at module import time —
    # heads.py imports nothing from env.py but env.py only needs the
    # heads.py constants when this translator is actually invoked.
    from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END

    op_types_arr = np.asarray(op_types)
    i_arr = np.asarray(i_indices)
    j_arr = np.asarray(j_indices)
    f_arr = np.asarray(factors)
    if compress_kinds is None:
        k_arr = np.zeros_like(op_types_arr)
    else:
        k_arr = np.asarray(compress_kinds)
    axis_state_np = np.asarray(axis_state_for_vertex)

    n_out = int(np.sum(axis_state_np[:, _AXIS_FEAT_IS_OUTPUT]))
    # Build a per-token-index → (is_output, relative_position) map matching
    # `compute_static_axis_state`'s layout: out axes come first in slots
    # 0..n_out-1, then primal axes in slots n_out..n_out+n_primal-1.
    def _to_base(token_idx: int) -> tuple[int, int]:
        token_idx = int(token_idx)
        is_out = int(axis_state_np[token_idx, _AXIS_FEAT_IS_OUTPUT])
        rel = token_idx if is_out else token_idx - n_out
        return (rel, is_out)

    specs = np.full((MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, 2] = 0  # factor=0 default for unused slots (matches reset path)

    slot = 0
    for s_idx, op in enumerate(op_types_arr.tolist()):
        if op == OP_END:
            break
        if op == OP_COMPRESS:
            # COMPRESS encodes a single axis to reduce. The env stores it
            # as `(COMPRESS_SENTINEL, physical_axis, kind_idx)` in the
            # sparsity specs row; `_callback` recognises the sentinel and
            # emits `Compress(axes=(physical_axis,), kind=COMPRESS_KINDS[kind_idx])`.
            # The axis-token index equals the physical position because
            # tokens are arranged as (outs..., primals...) matching the
            # SparseTensor edge layout.
            if slot >= MAX_RULES_PER_VERTEX:
                break
            physical_axis = int(i_arr[s_idx])
            specs[slot, 0] = COMPRESS_SENTINEL
            specs[slot, 1] = physical_axis
            specs[slot, 2] = int(k_arr[s_idx])
            slot += 1
            continue
        if op != OP_DIAG:
            raise ValueError(f"Unknown op_type {op!r} at sub-step {s_idx}.")
        if slot >= MAX_RULES_PER_VERTEX:
            break
        rel_i, is_out_i = _to_base(i_arr[s_idx])
        rel_j, is_out_j = _to_base(j_arr[s_idx])
        # Legacy rule_specs layout: row [base_idx1, base_idx2, factor]
        # where base_idx1 indexes the output axis and base_idx2 indexes
        # the primal axis. If both i and j are on the same side, the
        # mapping isn't lossless — log + fall through to OP_END so the
        # rest of the sub-episode doesn't poison the spec. This case
        # will go away when the typed action becomes the canonical form.
        if is_out_i == is_out_j:
            # Both axes on the same side (both output or both primal). The
            # env's sparsity_specs row format strictly pairs one output axis
            # with one primal axis; no representation for this. Terminate
            # the sub-episode here — the policy is expected to mask these
            # out before sampling, but a stray pair shouldn't crash the env.
            break
        # bi1 = output-side axis position, bi2 = primal-side axis position.
        if is_out_i:
            bi1, bi2 = rel_i, rel_j
        else:
            bi1, bi2 = rel_j, rel_i
        specs[slot, 0] = bi1
        specs[slot, 1] = bi2
        specs[slot, 2] = int(f_arr[s_idx])
        slot += 1

    return specs


def micro_actions_to_rule_specs_jax(
    op_types,
    i_indices,
    j_indices,
    factors,
    axis_state_for_vertex,
    compress_kinds=None,
):
    """JAX-traceable MicroAction → rule_specs (DIAG and COMPRESS).

    Differs from :func:`micro_actions_to_rule_specs` in that the entire
    transform is JAX-tracer-friendly — no Python loops over sub-steps,
    no exceptions. It is *intended* for the rollout's JIT-compiled
    sample-then-step path; the Python translator stays for host-side
    code paths (e.g. tests, debugging, top-N replay).

    Semantics:

    * Each DIAG sub-step's `(i, j, factor)` becomes one rule_specs row
      `[bi1, bi2, factor]` where bi1/bi2 are out-side / primal-side
      relative positions.
    * Each COMPRESS sub-step writes a `[COMPRESS_SENTINEL, axis, kind_idx]`
      row where ``axis`` is the policy's i-index (which equals the
      physical axis position in the SparseTensor edge) and ``kind_idx``
      indexes :data:`graphax.sparse.micro_actions.COMPRESS_KINDS`. The
      env's `_callback` recognises the sentinel and emits a graphax
      `Compress(axes=(axis,), kind=COMPRESS_KINDS[kind_idx])`.
    * `op_type == OP_END` and every sub-step after the first END are
      marked unused.
    * The output is truncated to ``MAX_RULES_PER_VERTEX`` rows; trailing
      sub-steps beyond the legacy capacity are dropped. The policy's
      ``max_substeps`` should be ≤ ``MAX_RULES_PER_VERTEX`` to avoid
      silent truncation, or the trainer should accept the truncation
      (the dropped DIAGs become no-ops from the env's perspective).

    Args:
        op_types: (max_substeps,) int32 — heads.py OP_* values.
        i_indices, j_indices: (max_substeps,) int32 — axis-token indices
            into axis_state_for_vertex.
        factors: (max_substeps,) int32 — the integer factor produced
            by the prime-exponent head (already collapsed from exponents).
        axis_state_for_vertex: ``(MAX_AXES_PER_VERTEX, AXIS_FEATURE_DIM)``
            int32 — per-vertex axis features from EnvState.

    Returns:
        rule_specs: ``(MAX_RULES_PER_VERTEX, 3)`` int32 in the legacy
        env layout ``[base_idx1, base_idx2, factor]``. Unused rows are
        ``[-1, -1, 0]``.
    """
    # Lazy import — heads.py imports nothing from env.py, but env.py
    # only needs the heads.py constants when this translator runs.
    from alphagrad.approx.heads import OP_COMPRESS, OP_DIAG, OP_END

    is_output = axis_state_for_vertex[:, _AXIS_FEAT_IS_OUTPUT].astype(jnp.int32)
    n_out = jnp.sum(is_output)

    if compress_kinds is None:
        compress_kinds = jnp.zeros_like(op_types)

    is_end_per = (op_types == OP_END)
    prior_ends = (
        jnp.cumsum(is_end_per.astype(jnp.int32)) - is_end_per.astype(jnp.int32)
    )
    active = (prior_ends == 0)
    is_diag = (op_types == OP_DIAG)
    is_compress = (op_types == OP_COMPRESS)

    def _row(s_idx):
        i = i_indices[s_idx]
        j = j_indices[s_idx]
        is_out_i = is_output[i] > 0
        is_out_j = is_output[j] > 0
        # Relative position within out vs primal axes mirrors the
        # compute_static_axis_state layout: out axes come first.
        rel_i = jnp.where(is_out_i, i, i - n_out)
        rel_j = jnp.where(is_out_j, j, j - n_out)
        # DIAG spec: base_idx1 = output side, base_idx2 = primal side.
        # If both axes are on the same side, the row format can't
        # express the pair → mark unused.
        same_side = is_out_i == is_out_j
        diag_bi1 = jnp.where(is_out_i, rel_i, rel_j)
        diag_bi2 = jnp.where(is_out_i, rel_j, rel_i)
        diag_used = active[s_idx] & is_diag[s_idx] & (~same_side)

        # COMPRESS spec: bi1 = COMPRESS_SENTINEL (-2), bi2 = physical axis
        # position in the SparseTensor edge. Tokens are arranged as
        # (out axes..., primal axes...) which matches the edge's physical
        # layout, so token index `i` IS the physical axis index. bi2 is
        # plumbed through `_callback`, which double-checks the axis exists
        # in every invar's edge before emitting Compress.
        compress_used = active[s_idx] & is_compress[s_idx]
        compress_bi1 = jnp.asarray(COMPRESS_SENTINEL, dtype=jnp.int32)
        compress_bi2 = i.astype(jnp.int32)

        # Compose the row. Priority: COMPRESS over DIAG over unused
        # (these branches are mutually exclusive because is_compress and
        # is_diag look at the same op_type slot). For DIAG the third
        # column carries the integer factor; for COMPRESS it carries the
        # `compress_kind` index into :data:`COMPRESS_KINDS`.
        bi1 = jnp.where(
            compress_used, compress_bi1,
            jnp.where(diag_used, diag_bi1, -1),
        ).astype(jnp.int32)
        bi2 = jnp.where(
            compress_used, compress_bi2,
            jnp.where(diag_used, diag_bi2, -1),
        ).astype(jnp.int32)
        f = jnp.where(
            compress_used, compress_kinds[s_idx],
            jnp.where(diag_used, factors[s_idx], 0),
        ).astype(jnp.int32)
        return jnp.stack([bi1, bi2, f])

    rows = jax.vmap(_row)(jnp.arange(op_types.shape[0]))

    # Truncate to MAX_RULES_PER_VERTEX. If max_substeps < MAX_RULES we
    # pad the trailing rows with [-1, -1, 0].
    rows_truncated = rows[:MAX_RULES_PER_VERTEX]
    pad_needed = MAX_RULES_PER_VERTEX - rows_truncated.shape[0]
    if pad_needed > 0:
        pad = jnp.tile(
            jnp.array([-1, -1, 0], dtype=jnp.int32), (pad_needed, 1),
        )
        rows_truncated = jnp.concatenate([rows_truncated, pad], axis=0)
    return rows_truncated


# Lookup row used to convert a legacy scalar sp_type ∈ {0..4} into a single-rule (MAX_RULES, 3) spec.
_LEGACY_SP_TO_RULE_ROW = jnp.array(
    [
        [-1, -1, 0],   # sp 0: unused
        [0, 0, -1],    # sp 1 -> (0,0)
        [0, 1, -1],    # sp 2 -> (0,1)
        [1, 0, -1],    # sp 3 -> (1,0)
        [1, 1, -1],    # sp 4 -> (1,1)
    ],
    dtype=jnp.int32,
)


def _legacy_sp_to_specs(sp_type: Array) -> Array:
    """Convert a scalar legacy sp_type ∈ {0..4} into (MAX_RULES_PER_VERTEX, 3) rule specs."""
    first = _LEGACY_SP_TO_RULE_ROW[sp_type]  # (3,)
    pad = jnp.tile(jnp.array([-1, -1, 0], dtype=jnp.int32), (MAX_RULES_PER_VERTEX - 1, 1))
    return jnp.concatenate([first[None, :], pad], axis=0)


@jax.jit
def cossim(target, preds):
    target = target / jnp.maximum(
        jnp.linalg.norm(target, keepdims=True), jnp.sqrt(1e-7)
    )
    preds = preds / jnp.maximum(jnp.linalg.norm(preds, keepdims=True), jnp.sqrt(1e-7))
    return jnp.sum(target * preds)


sp_type_to_map = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}

# things to try:
# error = MSE, cossim, Frobenius Norm
# other = {log, no log} x {div, no div}


def _flatten_jacobians(jac):
    """Concatenate all leaves of a (possibly nested) jacobian pytree to a flat 1-d array."""
    leaves = jax.tree_util.tree_leaves(jac)
    if not leaves:
        return None
    flats = [jnp.ravel(l) for l in leaves]
    return jnp.concatenate(flats)


def _quality_metrics(jac_exact, jac_approx):
    """`(cosine_sim, relative_frobenius)` of `jac_approx` against `jac_exact`.

    Returns the trivial `(1.0, 0.0)` (perfect agreement) when either side has
    no leaves, mismatched shapes, or zero size, mirroring the original `error`
    fallback so a degenerate plan can't poison downstream normalisation.
    """
    flat_exact = _flatten_jacobians(jac_exact)
    flat_approx = _flatten_jacobians(jac_approx)
    if flat_exact is None or flat_approx is None:
        return jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)
    if flat_approx.shape != flat_exact.shape or flat_approx.size == 0:
        return jnp.array(1.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)
    cos = cossim(flat_exact, flat_approx)
    exact_norm = jnp.linalg.norm(flat_exact)
    resid_norm = jnp.linalg.norm(flat_exact - flat_approx)
    rel_frob = resid_norm / jnp.maximum(exact_norm, jnp.sqrt(1e-7))
    return cos, rel_frob


def _aggregate_samples(values, want_top_quartile: bool):
    """Reduce a list of per-sample scalars to a single jnp scalar.

    With ≥8 samples and `want_top_quartile`, takes the top-quartile mean
    (matching legacy behaviour for latency); otherwise falls back to a plain
    mean. Handles the empty-list case by returning `0.0`.
    """
    if not values:
        return jnp.array(0.0, dtype=jnp.float32)
    stack = jnp.stack([jnp.asarray(v, dtype=jnp.float32) for v in values])
    if want_top_quartile and stack.shape[0] >= 8:
        return stack.sort()[6:8].mean()
    return stack.mean()


def _callback(
    config: EnvConfig,
    args,
    consts,
    order,
    sparsity_specs,
    stop,
    *eval_samples,
    init: bool = False,
):
    """Stage A reward harness: returns `(tokens, rewards)` where `rewards` is
    the canonical `(NUM_REWARDS,)` float32 vector documented at the top of this
    file. Every component is computed every (non-init) call, except `latency`
    which is gated behind `config.measure_latency`. When
    `config.terminal_rewards_only` is on, intermediate steps return tokens
    only — every reward component is zeroed so the heavy jacve compile/exec
    is skipped entirely until the elimination order is complete.
    """
    partial_order, partial_specs = _get_partials(order, sparsity_specs, stop)
    is_terminal = int(stop) >= len(order)

    o_list = [int(x) for x in partial_order.tolist()]
    specs_list = partial_specs.tolist()  # list of MAX_RULES x 3 lists

    # Build the per-vertex `transforms` sequence consumed by graphax's
    # typed-transform API. Each row in `sparsity_specs` is
    # ``[base_idx1, base_idx2, factor]``; we resolve the logical axis
    # indices and the legacy -1 (gcd) / 0 (drop) / 1 (no-op) sentinels
    # into explicit Diag(i, j, factor) entries with a strictly positive
    # integer factor — the only form graphax's apply_diag accepts. Slots
    # with factor=0 (legacy drop-axes) or factor=1 (legacy no-op) are
    # silently skipped: drop has no replacement under the new API, and
    # no-op is dead weight. The rule must also fit **every** non-literal
    # invar of the eqn: graphax's `_eliminate_vertex` applies each
    # transform to every incoming edge, and apply_diag raises if the
    # primal axis index is out of range for any of them (e.g. a div by
    # a scalar denominator has one (n,) edge and one () edge — a Diag
    # with j=1 only fits the first).
    transforms: list[tuple[int, tuple]] = []
    last_v_idx = len(o_list) - 1
    for v_idx, v in enumerate(o_list):
        eqn = config.jaxpr.eqns[v - 1]
        if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
            continue

        out_shape = eqn.outvars[0].aval.shape
        out_len = len(out_shape)
        primal_shapes = [
            iv.aval.shape for iv in eqn.invars if hasattr(iv, "aval")
        ]
        if not primal_shapes:
            continue  # no non-literal inputs → no edges to transform

        rules: list = []  # mixed list[Diag | Compress]
        used_axes: set[int] = set()
        for slot in range(MAX_RULES_PER_VERTEX):
            row = specs_list[v_idx][slot]
            bi1 = int(row[0])
            bi2 = int(row[1])
            factor = int(row[2])
            if bi1 == -1:
                break  # end-of-sequence sentinel
            if bi1 == COMPRESS_SENTINEL:
                # COMPRESS slot: row[1] is the *physical* axis index in the
                # SparseTensor edge (same layout as Diag's idx: out axes 0..
                # out_len-1, primal axes out_len..out_len+primal_dims-1).
                # row[2] is the kind index into COMPRESS_KINDS. Skip the
                # slot if the axis index doesn't fit every invar's edge —
                # graphax's apply_compress will validate too, but raising
                # would crash the io_callback.
                #
                # COMPRESS reduces ``val.ndim``, which trips graphax's
                # shape-preservation assertion in ``_eliminate_vertex``
                # when the compressed edge feeds into a subsequent
                # elimination. Until graphax learns to propagate the
                # reduced shape, restrict COMPRESS rows to the **last**
                # vertex of the partial elimination order — the only
                # vertex with no downstream elimination step within this
                # callback.
                if v_idx != last_v_idx:
                    continue
                axis_idx = bi2
                kind_idx = factor  # row[2] reused as kind index for COMPRESS
                fits_all = True
                if axis_idx < 0:
                    fits_all = False
                elif axis_idx < out_len:
                    pass  # output-side axis, always present
                else:
                    primal_pos = axis_idx - out_len
                    fits_all = all(primal_pos < len(ps) for ps in primal_shapes)
                if not fits_all:
                    continue
                if axis_idx in used_axes:
                    continue
                if not (0 <= kind_idx < len(COMPRESS_KINDS)):
                    # Unknown kind — fall back to the default "mean" rather
                    # than dropping the row, since the axis-removal effect is
                    # the dominant signal.
                    kind_idx = 0
                used_axes.add(axis_idx)
                rules.append(
                    Compress(axes=(axis_idx,), kind=COMPRESS_KINDS[kind_idx])
                )
                continue
            if bi1 < 0:
                # Any other negative bi1 is reserved for future sentinels;
                # skip without aborting the sequence so a new sentinel
                # introduced upstream doesn't silently break older specs.
                continue
            idx1 = bi1            # logical output-side axis
            idx2 = out_len + bi2  # logical primal-side axis
            # Skip rules that would reuse an axis (graphax expected bipartite
            # disjoint pairs; the legacy translator filtered them, do it here
            # so apply_diag's stricter checks don't crash).
            if idx1 in used_axes or idx2 in used_axes or idx1 == idx2:
                continue
            if not (0 <= bi1 < out_len):
                continue
            n1 = int(out_shape[bi1])
            # The rule must fit every primal edge of this vertex.
            n2_list: list[int] = []
            fits_all = True
            for ps in primal_shapes:
                if not (0 <= bi2 < len(ps)):
                    fits_all = False
                    break
                n2_list.append(int(ps[bi2]))
            if not fits_all:
                continue
            if factor == 0 or factor == 1:
                continue  # drop-axes and no-op have no equivalent in the new API
            if factor == -1:
                # Joint gcd across the out axis and every primal axis.
                from functools import reduce as _reduce
                factor = _reduce(_math.gcd, [n1] + n2_list)
            # apply_diag requires factor | gcd(n_i, n_j) on every edge it
            # touches. Silently skip mismatches rather than crash.
            if (
                factor <= 0
                or n1 % factor != 0
                or any(n2 % factor != 0 for n2 in n2_list)
            ):
                continue
            used_axes.add(idx1)
            used_axes.add(idx2)
            rules.append(Diag(i=idx1, j=idx2, factor=factor))
        if rules:
            transforms.append((int(v), tuple(rules)))

    ve = extract_jaxpr(
        config.jaxpr,
        config.argnums,
        o_list,
        config.sparse,
        args,
        consts,
        transforms=transforms,
    )
    tokens = ve.tokenized()[:MAX_TOKENS]
    tokens = jnp.pad(tokens, (0, MAX_TOKENS - tokens.shape[0]))

    # Compute per-token equation IDs once for the relational-bias encoder
    # (Stage B.1). Cheap (single Python scan over a length-≤4096 numpy array)
    # and adds (MAX_TOKENS,) int32 to EnvState.
    tokens_np = np.asarray(tokens)
    eqn_ids_np = compute_eqn_ids_from_tokens(tokens_np, _TOKEN_VOCAB)
    eqn_ids = jnp.asarray(eqn_ids_np, dtype=jnp.int32)

    if init:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    # Terminal-only fast path: every reward component is sparse — only the
    # final step (when the elimination order is complete) gets a non-zero
    # signal, so we skip the expensive jacve compile/exec on every prior
    # step. Cumsum-style returns (alpha0/mu0) and per-rollout aggregations
    # (gdpo) collapse to the terminal reward; gfn already reads only the
    # last step. PPO sees a sparse-reward MDP, which GAE handles natively.
    if config.terminal_rewards_only and not is_terminal:
        return tokens, eqn_ids, jnp.zeros(NUM_REWARDS, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Compute family — graphax counters (always) → muls_adds_fmas, max_io_sum.
    # ------------------------------------------------------------------
    _, aux = vertex_elimination_jaxpr(
        config.jaxpr,
        o_list,
        consts,
        *args,
        argnums=config.argnums,
        count_ops=True,
        sparse_representation=config.sparse,
        transforms=transforms,
    )
    muls_adds_fmas = float(aux["adds"] + aux["muls"] + aux["fmas"])
    max_io_sum = float(aux["mem"])

    # If no `target_fun` is supplied, we can't compile/execute. Skip every
    # execution-derived metric and return a partial reward vector.
    if config.target_fun is None:
        rewards = jnp.array(
            [
                -muls_adds_fmas, 0.0, 0.0, -max_io_sum,
                0.0, 0.0, 1.0, 0.0,
            ],
            dtype=jnp.float32,
        )
        return tokens, eqn_ids, rewards

    # ------------------------------------------------------------------
    # Compile both the approximated and exact jacobian functions once.
    # ------------------------------------------------------------------
    callback_device = None
    if config.exec_on_gpu:
        gpu_devices = jax.devices("gpu")
        if len(gpu_devices) >= 2:
            callback_device = gpu_devices[1]

    args_for_lower = (
        jax.device_put(args, callback_device)
        if (config.exec_on_gpu and callback_device is not None)
        else args
    )

    def compiled(transforms_arg=None):
        return (
            jax.jit(
                jacve(
                    config.target_fun,
                    o_list,
                    argnums=config.argnums,
                    has_aux=config.has_aux,
                    sparse_representation=config.sparse,
                    transforms=transforms_arg,
                ),
                keep_unused=True,
            )
            .lower(*args_for_lower)
            .compile()
        )

    compiled_approx = compiled(transforms)
    compiled_exact = compiled()

    # XLA cost analysis — flops + bytes accessed. Falls back to 0 when the
    # backend doesn't expose them (CPU sometimes returns an empty dict).
    cost_analysis = compiled_approx.cost_analysis() or {}
    flops = float(cost_analysis.get("flops", 0))
    bytes_accessed = float(cost_analysis.get("bytes accessed", 0))

    # ------------------------------------------------------------------
    # Execution loop — runs once for peak_memory + quality, or 10x when
    # `measure_latency` is on (the latency reading is noisy enough that the
    # top-quartile-mean smoothing from the original code is worth keeping).
    # ------------------------------------------------------------------
    n_samples = 10 if config.measure_latency else 1

    monitoring_devices: list = []
    for x in jax.tree_util.tree_leaves(args):
        if hasattr(x, "devices"):
            monitoring_devices.extend(list(x.devices()))
    unique_devices = list(set(monitoring_devices))
    if (
        config.exec_on_gpu
        and callback_device is not None
        and callback_device not in unique_devices
    ):
        unique_devices.append(callback_device)
    if not unique_devices:
        unique_devices = jax.local_devices()

    out_approxs: list = []
    out_exacts: list = []
    latency_samples: list[float] = []
    peak_mem_samples: list[float] = []

    for i in range(n_samples):
        if eval_samples:
            eval_args_i = [arg[i] for arg in eval_samples]
        else:
            eval_args_i = list(args)
        if config.exec_on_gpu and callback_device is not None:
            eval_args_i = [jax.device_put(d, callback_device) for d in eval_args_i]

        with ResourceMonitor(devices=unique_devices) as monitor:
            out_approx = compiled_approx(*eval_args_i)
            # JAX dispatches asynchronously; without a block here the
            # monitor exits before the device finishes the work and both
            # the time and memory readings are dominated by dispatch
            # overhead (peak comes back as 0 bytes). Forcing the result to
            # land synchronizes the device queue so the tracker sees the
            # full peak allocation and the timer captures real wall-clock.
            out_approx = jax.block_until_ready(out_approx)
        # Key by name instead of unpacking ``.values()`` so this stays
        # robust to dict-order / API tweaks in jax_memory_monitor.
        latency_s = float(monitor.stats.get("time", 0.0))
        peak_bytes = float(monitor.stats.get("memory", 0.0))
        latency_samples.append(latency_s * 1e9)  # → ns
        peak_mem_samples.append(peak_bytes)

        out_exact = compiled_exact(*eval_args_i)
        out_approxs.append(out_approx)
        out_exacts.append(out_exact)

    latency_ns = (
        float(_aggregate_samples(latency_samples, want_top_quartile=True))
        if config.measure_latency
        else 0.0
    )
    peak_memory = float(max(peak_mem_samples)) if peak_mem_samples else 0.0

    # ------------------------------------------------------------------
    # Quality family — cosine similarity + relative Frobenius residual.
    # ------------------------------------------------------------------
    cosines: list = []
    frobs: list = []
    for out_approx, out_exact in zip(out_approxs, out_exacts):
        jac_approx = out_approx[1] if config.has_aux else out_approx
        jac_exact = out_exact[1] if config.has_aux else out_exact
        cos, rel_frob = _quality_metrics(jac_exact, jac_approx)
        cosines.append(cos)
        frobs.append(rel_frob)

    cosine_sim = float(_aggregate_samples(cosines, want_top_quartile=True))
    frob_residual = float(_aggregate_samples(frobs, want_top_quartile=True))

    rewards = jnp.array(
        [
            -muls_adds_fmas,
            -flops,
            -latency_ns,
            -max_io_sum,
            -bytes_accessed,
            -peak_memory,
            cosine_sim,
            -frob_residual,
        ],
        dtype=jnp.float32,
    )

    return tokens, eqn_ids, rewards


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class VertexEliminationEnv:
    config: EnvConfig
    args: tuple
    consts: tuple
    valid_vertices: tuple
    # Static per-vertex axis state derived from `config.jaxpr` at __init__
    # time. Stored as a JAX array so it travels through reset/step without
    # recomputation; values are constant across all episodes for a given
    # env. The pair `(axis_state_static, axis_valid_static)` matches the
    # shape contract documented on EnvState's `axis_state` / `axis_valid_mask`
    # fields.
    axis_state_static: Array | None = None
    axis_valid_static: Array | None = None
    num_envs: int | None = None
    eval_args_samples: tuple | None = None

    def __init__(
        self,
        config: EnvConfig,
        args: Sequence,
        consts: Sequence,
        valid_vertices: tuple | None = None,
        num_envs: int | None = None,
        eval_args_samples: tuple | None = None,
        axis_state_static: Array | None = None,
        axis_valid_static: Array | None = None,
    ):
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "args", tuple(args))
        object.__setattr__(self, "consts", tuple(consts))
        object.__setattr__(self, "eval_args_samples", eval_args_samples)

        if num_envs is None:
            num_envs = jax.local_device_count()
        object.__setattr__(self, "num_envs", num_envs)

        if valid_vertices is None:
            _, _, _, vo_vertices = _build_graph(
                config.jaxpr, args, consts, config.argnums
            )
            valid = []
            for i, eqn in enumerate(config.jaxpr.eqns, 1):
                if eqn.outvars[0] not in config.jaxpr.outvars or i in vo_vertices:
                    valid.append(i)
            valid_vertices = tuple(valid)
        object.__setattr__(self, "valid_vertices", valid_vertices)

        if axis_state_static is None or axis_valid_static is None:
            total_v = len(config.jaxpr.eqns)
            axis_state_np, axis_valid_np = compute_static_axis_state(
                config.jaxpr, total_v,
            )
            axis_state_static = jnp.asarray(axis_state_np, dtype=jnp.int32)
            axis_valid_static = jnp.asarray(axis_valid_np, dtype=jnp.float32)
        object.__setattr__(self, "axis_state_static", axis_state_static)
        object.__setattr__(self, "axis_valid_static", axis_valid_static)

    @classmethod
    def from_jaxpr(
        cls,
        jaxpr: core.ClosedJaxpr,
        argnums=None,
        args=None,
        has_aux=False,
        sparse=False,
        num_envs=None,
        data_gen: Callable | None = None,
        target_fun: Callable | None = None,
        cmp_type: str = "flops",
        mem_type: str = "peak_memory",
        exec_on_gpu: bool = False,
        measure_latency: bool = False,
        terminal_rewards_only: bool = False,
    ):
        assert (argnums is None and args is None) or not (args is None or args is None)
        config = EnvConfig(
            jaxpr=jaxpr.jaxpr,
            argnums=tuple(range(len(jaxpr.invars)))
            if argnums is None
            else tuple(argnums),
            has_aux=has_aux,
            sparse=sparse,
            cmp_type=cmp_type,
            mem_type=mem_type,
            target_fun=target_fun,
            data_gen=data_gen,
            exec_on_gpu=exec_on_gpu,
            measure_latency=measure_latency,
            terminal_rewards_only=terminal_rewards_only,
        )
        return cls(
            config,
            args=jaxpr.invars if args is None else args,
            consts=jaxpr.literals,
            num_envs=num_envs,
        )

    def tree_flatten(self):
        children = (
            self.args, self.consts, self.eval_args_samples,
            self.axis_state_static, self.axis_valid_static,
        )
        aux_data = (self.config, self.valid_vertices, self.num_envs)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, consts, eval_args_samples, axis_state_static, axis_valid_static = (
            children
        )
        config, valid_vertices, num_envs = aux_data
        return cls(
            config, args, consts, valid_vertices, num_envs, eval_args_samples,
            axis_state_static=axis_state_static,
            axis_valid_static=axis_valid_static,
        )

    def tokenize(self, init: bool = False):
        return partial(_callback, self.config, init=init)

    @property
    def _callback_shape(self):
        return (
            jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32),
            jax.ShapeDtypeStruct((MAX_TOKENS,), jnp.int32),
            jax.ShapeDtypeStruct((NUM_REWARDS,), jnp.float32),
        )

    def reset(self, num_envs: int | None = None) -> EnvState:
        if num_envs is None:
            num_envs = getattr(self, "num_envs", None)

        initial_order = jnp.array(self.valid_vertices, dtype=jnp.int32)
        initial_specs = jnp.full(
            (initial_order.shape[0], MAX_RULES_PER_VERTEX, 3),
            -1,
            dtype=jnp.int32,
        )
        initial_specs = initial_specs.at[..., 2].set(0)  # factor=0 default for unused rows

        tokens, eqn_ids, _ = io_callback(
            self.tokenize(init=True),
            self._callback_shape,
            self.args,
            self.consts,
            initial_order,
            initial_specs,
            0,
            *(self.eval_args_samples if self.eval_args_samples is not None else ()),
        )

        max_steps_val = initial_order.shape[0]
        step_count = jnp.array(0, dtype=jnp.int32)
        max_steps = max_steps_val
        reward = jnp.zeros(NUM_REWARDS, dtype=jnp.float32)
        terminated = jnp.array(False, dtype=jnp.bool_)

        state = EnvState(
            order=initial_order,
            sparsity_specs=initial_specs,
            tokens=tokens,
            eqn_ids=eqn_ids,
            axis_state=self.axis_state_static,
            axis_valid_mask=self.axis_valid_static,
            step_count=step_count,
            max_steps=max_steps,
            reward=reward,
            terminated=terminated,
        )

        if num_envs is not None and num_envs > 0:
            state = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)), state
            )

        return state

    @jit
    def step(self, state: EnvState, action) -> EnvOut:
        # Action may be either a `StepAction` (multi-rule) or a legacy scalar int
        # encoded as `sp_type * MAX_TOKENS + target_vertex`.
        if isinstance(action, StepAction):
            target_vertex = jnp.asarray(action.target_vertex, dtype=jnp.int32)
            rule_specs = jnp.asarray(action.rule_specs, dtype=jnp.int32)
        else:
            action = jnp.asarray(action, dtype=jnp.int32)
            sp_type = action // MAX_TOKENS
            target_vertex = action % MAX_TOKENS
            rule_specs = _legacy_sp_to_specs(sp_type)

        idx = state.step_count
        new_step = idx + 1
        curr_order = state.order
        curr_specs = state.sparsity_specs

        pos = jnp.argwhere(curr_order == target_vertex, size=1).squeeze()

        indices = jnp.arange(curr_order.shape[0])
        shifted = jnp.where((indices > idx) & (indices <= pos), indices - 1, indices)

        new_order = curr_order[shifted.astype(jnp.int32)].at[idx].set(target_vertex)
        new_specs = curr_specs[shifted.astype(jnp.int32)].at[idx].set(rule_specs)

        tokens, eqn_ids, reward = io_callback(
            self.tokenize(),
            self._callback_shape,
            self.args,
            self.consts,
            new_order,
            new_specs,
            new_step,
            *(self.eval_args_samples if self.eval_args_samples is not None else ()),
        )

        terminated = new_step >= state.max_steps

        # Single-vertex axis_state mutation: the just-acted-on vertex's
        # row records the DIAG pairings / COMPRESS markings the agent
        # committed to. See `_apply_rules_to_axis_state` for the per-rule
        # semantics and for why downstream propagation is deliberately
        # deferred. `target_vertex` is 1-indexed (matches the env's
        # vertex IDs); axis_state is 0-indexed by equation, so subtract 1.
        v_idx = target_vertex - jnp.int32(1)
        updated_axis_v = _apply_rules_to_axis_state(
            state.axis_state[v_idx], rule_specs,
        )
        new_axis_state = state.axis_state.at[v_idx].set(updated_axis_v)

        new_state = EnvState(
            order=new_order,
            sparsity_specs=new_specs,
            tokens=tokens,
            eqn_ids=eqn_ids,
            axis_state=new_axis_state,
            axis_valid_mask=state.axis_valid_mask,
            step_count=new_step,
            max_steps=state.max_steps,
            reward=reward,
            terminated=terminated,
        )

        def _step_process(_):
            return EnvOut(new_state, reward, terminated)

        def _step_done(_):
            return EnvOut(
                state,
                jnp.zeros(NUM_REWARDS, jnp.float32),
                jnp.array(True, dtype=jnp.bool_),
            )

        return jax.lax.cond(state.terminated, _step_done, _step_process, None)
