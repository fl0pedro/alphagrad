"""Per-vertex instrumentation for the data-dependent encoder feed (Stage B.2.A).

The architecture wants the policy to be data-dependent: the same graph G with
different parameter / data distributions can warrant different elimination
plans. We achieve this by extracting a small set of per-vertex features that
summarise *what each forward equation actually does* on the calibration
samples, then conditioning the agent's vertex / rule heads on those features.

This module ships the cheapest, most reusable subset:

* **Static features** (4 floats per vertex) — derived purely from the jaxpr:
  ``op_type_id``, ``out_ndim``, ``log1p(out_size)``, ``log1p(in_total_size)``.
  The op-type id indexes a small vocabulary of primitives; everything outside
  the vocab maps to ``OP_TYPE_UNKNOWN``. The continuous fields are stored on a
  log scale because vertex output sizes span many orders of magnitude.

* **Dynamic features** (4 floats per vertex) — moments of each vertex's
  activation across the calibration samples: ``mean``, ``std``, ``abs_max``,
  ``sparsity_rate``. We run a plain Python eval of the forward jaxpr on each
  sample (one extra forward pass per sample, negligible vs the existing
  exact + approx Jacobian cost in `_callback`), collect each vertex's first
  output, and average the per-sample moments.

The result is an `(total_v, NUM_VERTEX_FEATURES) = (V, 8)` numpy array. The
op-type id lives in column 0 as a float so the whole feature tensor stays
homogeneous; the agent embeds it back to int32 before lookup.

Stage B.3 will replace the per-sample mean with a Set Transformer aggregator;
Stage B.2.B will add Hutchinson-style Jacobian sketches and the off-diagonal
energy ratio. Both slot in by extending `compute_vertex_features` without
touching the agent-side wiring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core


# Op-type vocabulary. Most jaxpr primitives map here; anything else collapses
# to `OP_TYPE_UNKNOWN`. Indices are small enough that the agent's op-embedding
# table is tiny (`OP_TYPE_VOCAB_SIZE * op_dim` parameters).
OP_TYPE_UNKNOWN: int = 0
OP_TYPE_VOCAB: dict[str, int] = {
    "add": 1, "sub": 2, "mul": 3, "div": 4,
    "neg": 5, "abs": 6, "sqrt": 7, "rsqrt": 8,
    "exp": 9, "log": 10, "log1p": 11, "expm1": 12,
    "sin": 13, "cos": 14, "tan": 15, "tanh": 16,
    "logistic": 17, "relu": 18,
    "reduce_sum": 19, "reduce_max": 20, "reduce_min": 21,
    "reduce_prod": 22, "reduce_mean": 23,
    "dot_general": 24, "matmul": 25,
    "broadcast_in_dim": 26, "reshape": 27, "transpose": 28,
    "squeeze": 29, "concatenate": 30, "slice": 31,
    "convert_element_type": 32, "stop_gradient": 33,
    "integer_pow": 34, "pow": 35,
    "select_n": 36, "select": 37,
    "lt": 38, "gt": 39, "le": 40, "ge": 41, "eq": 42, "ne": 43,
    "min": 44, "max": 45, "clamp": 46,
    "iota": 47, "argmin": 48, "argmax": 49,
    "dynamic_slice": 50, "dynamic_update_slice": 51,
    "gather": 52, "scatter": 53,
    "sigmoid": 54, "softmax": 55,
    "real": 56, "imag": 57, "conj": 58, "complex": 59,
    "floor": 60, "ceil": 61, "round": 62,
    "and": 63, "or": 64, "xor": 65, "not": 66,
    "shift_left": 67, "shift_right": 68,
    "bitcast_convert_type": 69, "device_put": 70,
}
OP_TYPE_VOCAB_SIZE: int = max(OP_TYPE_VOCAB.values()) + 1


# Feature layout. Column 0 is the op-type id (cast to int32 by the agent).
# When `graphax.instrumentation` is toggled on, the last two columns are
# populated from `extract_jacobian_features`; otherwise they remain zero so
# the agent's projection picks up no extra signal — same numerical behaviour
# as before B.2.B.
NUM_STATIC_FEATURES: int = 4
NUM_DYNAMIC_FEATURES: int = 4
NUM_INSTRUMENTATION_FEATURES: int = 2
NUM_VERTEX_FEATURES: int = (
    NUM_STATIC_FEATURES + NUM_DYNAMIC_FEATURES + NUM_INSTRUMENTATION_FEATURES
)
VERTEX_FEATURE_NAMES: tuple[str, ...] = (
    "op_type_id",       # categorical; embedded via OP_TYPE_VOCAB
    "out_ndim",         # number of axes of the eqn's output
    "log1p_out_size",   # log1p of the eqn's output total element count
    "log1p_in_size",    # log1p of the eqn's combined input element count
    "act_mean",         # per-sample mean of the activation, averaged
    "act_std",          # per-sample std of the activation, averaged
    "act_abs_max",      # per-sample max(abs(.)), averaged
    "act_sparsity",     # per-sample fraction of exact zeros, averaged
    "log1p_jac_frob_sq",  # B.2.B: log1p of ||J_v||_F^2, analytical or Hutchinson
    "jac_off_diag_ratio", # B.2.B: off-diag energy ratio in [0, 1]
)


def _eqn_op_id(eqn) -> int:
    return OP_TYPE_VOCAB.get(eqn.primitive.name, OP_TYPE_UNKNOWN)


def _eqn_static_features(eqn) -> tuple[float, float, float, float]:
    """The (op_type_id, out_ndim, log1p_out_size, log1p_in_size) tuple."""
    if not eqn.outvars or not hasattr(eqn.outvars[0], "aval"):
        return float(_eqn_op_id(eqn)), 0.0, 0.0, 0.0
    out_shape = eqn.outvars[0].aval.shape
    out_size = int(np.prod(out_shape)) if out_shape else 1
    in_size = 0
    for v in eqn.invars:
        if hasattr(v, "aval") and v.aval.shape:
            in_size += int(np.prod(v.aval.shape))
        elif hasattr(v, "aval"):
            in_size += 1
    return (
        float(_eqn_op_id(eqn)),
        float(len(out_shape)),
        float(np.log1p(out_size)),
        float(np.log1p(in_size)),
    )


def eval_with_intermediates(
    jaxpr: core.Jaxpr, consts: tuple, args: tuple
) -> list:
    """Walk `jaxpr` equation-by-equation and return per-eqn first-output values.

    Returns a list of length `len(jaxpr.eqns)`; entry `i` is the array value
    bound to ``jaxpr.eqns[i].outvars[0]`` after that equation runs, or
    ``None`` if the eqn dropped its output / failed to evaluate.

    No JIT — this is only used inside `_callback` (which is itself an
    `io_callback`, i.e. already running as plain Python). Failures on
    individual equations (e.g. unsupported control-flow primitives) record
    ``None`` and continue rather than blowing up the whole feature pipeline.
    """
    env: dict = {}
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val
    for var, val in zip(jaxpr.invars, args):
        env[var] = val

    def read(v):
        return v.val if isinstance(v, core.Literal) else env[v]

    out_per_eqn: list = []
    for eqn in jaxpr.eqns:
        try:
            in_vals = [read(v) for v in eqn.invars]
            out_vals = eqn.primitive.bind(*in_vals, **eqn.params)
            outs = out_vals if eqn.primitive.multiple_results else (out_vals,)
            for var, val in zip(eqn.outvars, outs):
                if not isinstance(var, core.DropVar):
                    env[var] = val
            first_var = eqn.outvars[0]
            out_per_eqn.append(
                None if isinstance(first_var, core.DropVar) else env[first_var]
            )
        except Exception:
            out_per_eqn.append(None)
    return out_per_eqn


def _moments_of(arr: np.ndarray) -> tuple[float, float, float, float]:
    """`(mean, std, abs_max, sparsity_rate)` of a flattened array.

    Non-finite entries (``nan``, ``±inf``) are dropped from the mean / std /
    abs_max computation so that one bad sample (e.g. ``log(0)`` when a
    calibration draw ends up near a primitive's domain boundary) doesn't
    poison the whole feature row. Sparsity is computed over the original
    array since ``inf`` clearly isn't a structural zero.
    """
    flat = arr.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(finite)),
        float(np.std(finite)),
        float(np.max(np.abs(finite))),
        float(np.mean(flat == 0.0)),
    )


def _unstack_eval_samples(eval_samples) -> list[tuple]:
    """Split a vmapped pytree of args (leaves shaped `(num_samples, ...)`)
    into a list of `num_samples` plain-args tuples."""
    leaves, treedef = jax.tree_util.tree_flatten(eval_samples)
    if not leaves:
        return []
    num_samples = leaves[0].shape[0]
    out = []
    for i in range(num_samples):
        sliced_leaves = [l[i] for l in leaves]
        out.append(jax.tree_util.tree_unflatten(treedef, sliced_leaves))
    return out


def compute_per_sample_vertex_features(
    jaxpr: core.Jaxpr,
    consts: tuple,
    base_args: tuple,
    eval_samples,
    argnums: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Per-sample variant of :func:`compute_vertex_features` (Stage B.3).

    Returns a `(num_samples, total_v, NUM_VERTEX_FEATURES)` numpy array — no
    aggregation across samples. The Set Transformer aggregator in the agent
    handles the cross-sample pool, so per-sample variability becomes signal
    rather than being averaged away.

    When ``eval_samples`` is None or empty, returns a `(1, total_v, F)` array
    populated only with static features (matches the static-only fallback in
    :func:`compute_vertex_features`).
    """
    total_v = len(jaxpr.eqns)

    # Static features are sample-invariant; compute once and broadcast.
    static = np.zeros((total_v, NUM_STATIC_FEATURES), dtype=np.float32)
    for i, eqn in enumerate(jaxpr.eqns):
        op_id, out_ndim, log_out, log_in = _eqn_static_features(eqn)
        static[i, 0] = op_id
        static[i, 1] = out_ndim
        static[i, 2] = log_out
        static[i, 3] = log_in

    sample_arg_tuples = _unstack_eval_samples(eval_samples) if eval_samples is not None else []
    if not sample_arg_tuples:
        out = np.zeros((1, total_v, NUM_VERTEX_FEATURES), dtype=np.float32)
        out[0, :, :NUM_STATIC_FEATURES] = static
        return out

    # Optional: graphax instrumentation toggle for the last two cols.
    try:
        from graphax.instrumentation import (
            extract_jacobian_features,
            is_enabled as _instrument_enabled,
        )
    except ImportError:
        _instrument_enabled = lambda: False  # noqa: E731
        extract_jacobian_features = None
    use_instr = _instrument_enabled() and extract_jacobian_features is not None

    out = np.zeros(
        (len(sample_arg_tuples), total_v, NUM_VERTEX_FEATURES), dtype=np.float32,
    )
    for s_idx, sample_args in enumerate(sample_arg_tuples):
        out[s_idx, :, :NUM_STATIC_FEATURES] = static
        merged = list(base_args)
        if argnums is not None:
            for idx, slot in enumerate(argnums):
                if idx < len(sample_args):
                    merged[slot] = sample_args[idx]
        else:
            merged = list(sample_args) + list(base_args[len(sample_args):])
        try:
            outs = eval_with_intermediates(jaxpr, consts, tuple(merged))
        except Exception:
            continue
        for i, val in enumerate(outs):
            if val is None:
                continue
            try:
                arr = np.asarray(val)
            except Exception:
                continue
            mean, std, abs_max, sparsity = _moments_of(arr)
            out[s_idx, i, NUM_STATIC_FEATURES + 0] = mean
            out[s_idx, i, NUM_STATIC_FEATURES + 1] = std
            out[s_idx, i, NUM_STATIC_FEATURES + 2] = abs_max
            out[s_idx, i, NUM_STATIC_FEATURES + 3] = sparsity

        if use_instr:
            try:
                feats = extract_jacobian_features(
                    jaxpr, consts, tuple(merged), n_probes=8,
                )
            except Exception:
                continue
            if feats is None:
                continue
            n = min(total_v, feats.frob_sq.shape[0])
            log_frob = np.log1p(np.maximum(feats.frob_sq[:n], 0.0))
            ratio = feats.off_diag_ratio[:n]
            valid = np.isfinite(log_frob) & np.isfinite(ratio)
            log_frob = np.where(valid, log_frob, 0.0)
            ratio = np.where(valid, ratio, 0.0)
            out[s_idx, :n, NUM_STATIC_FEATURES + NUM_DYNAMIC_FEATURES + 0] = log_frob
            out[s_idx, :n, NUM_STATIC_FEATURES + NUM_DYNAMIC_FEATURES + 1] = ratio

    return out


def compute_vertex_features(
    jaxpr: core.Jaxpr,
    consts: tuple,
    base_args: tuple,
    eval_samples=None,
    argnums: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Build the `(total_v, NUM_VERTEX_FEATURES)` feature tensor.

    `base_args` is the env's frozen-arg tuple (used to fill in non-data
    arguments when the eval samples only cover a subset). When `eval_samples`
    is None or empty, dynamic features fall back to zeros and the policy is
    conditioned only on static (jaxpr-only) information.
    """
    total_v = len(jaxpr.eqns)
    features = np.zeros((total_v, NUM_VERTEX_FEATURES), dtype=np.float32)

    # Static features come from the jaxpr alone.
    for i, eqn in enumerate(jaxpr.eqns):
        op_id, out_ndim, log_out, log_in = _eqn_static_features(eqn)
        features[i, 0] = op_id
        features[i, 1] = out_ndim
        features[i, 2] = log_out
        features[i, 3] = log_in

    if eval_samples is None:
        return features

    sample_arg_tuples = _unstack_eval_samples(eval_samples)
    if not sample_arg_tuples:
        return features

    accum = np.zeros((total_v, NUM_DYNAMIC_FEATURES), dtype=np.float32)
    counts = np.zeros((total_v,), dtype=np.float32)

    for sample_args in sample_arg_tuples:
        # The eval-samples generator may only fill in `argnums`-flagged slots;
        # everything else falls back to `base_args` to keep the forward pass
        # well-defined.
        merged = list(base_args)
        if argnums is not None:
            for idx, slot in enumerate(argnums):
                if idx < len(sample_args):
                    merged[slot] = sample_args[idx]
        else:
            merged = list(sample_args) + list(base_args[len(sample_args):])
        try:
            outs = eval_with_intermediates(jaxpr, consts, tuple(merged))
        except Exception:
            continue
        for i, val in enumerate(outs):
            if val is None:
                continue
            try:
                arr = np.asarray(val)
            except Exception:
                continue
            mean, std, abs_max, sparsity = _moments_of(arr)
            accum[i, 0] += mean
            accum[i, 1] += std
            accum[i, 2] += abs_max
            accum[i, 3] += sparsity
            counts[i] += 1.0

    safe_counts = np.maximum(counts[:, None], 1.0)
    features[:, NUM_STATIC_FEATURES:NUM_STATIC_FEATURES + NUM_DYNAMIC_FEATURES] = (
        accum / safe_counts
    )

    # B.2.B follow-up: when the env-var toggle is set, additionally populate
    # the last two columns with `(log1p(frob_sq), off_diag_ratio)` per vertex,
    # averaged over the calibration samples. Off-path: zero columns, matching
    # the pre-B.2.B behaviour exactly.
    try:
        from graphax.instrumentation import (
            extract_jacobian_features,
            is_enabled as _instrument_enabled,
        )
    except ImportError:
        _instrument_enabled = lambda: False  # noqa: E731
        extract_jacobian_features = None

    if _instrument_enabled() and extract_jacobian_features is not None:
        instr_accum = np.zeros((total_v, NUM_INSTRUMENTATION_FEATURES), dtype=np.float32)
        instr_counts = np.zeros((total_v,), dtype=np.float32)
        for sample_args in sample_arg_tuples:
            merged = list(base_args)
            if argnums is not None:
                for idx, slot in enumerate(argnums):
                    if idx < len(sample_args):
                        merged[slot] = sample_args[idx]
            else:
                merged = list(sample_args) + list(base_args[len(sample_args):])
            try:
                feats = extract_jacobian_features(
                    jaxpr, consts, tuple(merged), n_probes=8,
                )
            except Exception:
                continue
            if feats is None:
                continue
            n = min(total_v, feats.frob_sq.shape[0])
            # Hutchinson probes can produce non-finite values when activations
            # are near a primitive's domain boundary (e.g. log near 0). Skip
            # those samples per-vertex so a single bad draw doesn't poison
            # the whole feature row.
            log_frob = np.log1p(np.maximum(feats.frob_sq[:n], 0.0))
            ratio = feats.off_diag_ratio[:n]
            valid = np.isfinite(log_frob) & np.isfinite(ratio)
            log_frob = np.where(valid, log_frob, 0.0)
            ratio = np.where(valid, ratio, 0.0)
            instr_accum[:n, 0] += log_frob
            instr_accum[:n, 1] += ratio
            instr_counts[:n] += valid.astype(np.float32)
        safe_instr = np.maximum(instr_counts[:, None], 1.0)
        features[:, NUM_STATIC_FEATURES + NUM_DYNAMIC_FEATURES:] = (
            instr_accum / safe_instr
        )

    return features
