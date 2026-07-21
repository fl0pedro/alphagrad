"""Shared seq -> (order, specs, n_rules) builder.

Rebuilds an elimination order + per-vertex sparsification rule specs
from a recorded solution ``seq`` (list of ``(action_idx, calls)`` pairs
with string-encoded micro-action calls). Lifted out of
``verify_pareto_solution.py`` so ``measure_server``, ``az_gumbel`` and
tests can import it without pulling the replay script (which pins
``JAX_PLATFORMS=cpu`` at import time — this module must NOT do that).
"""
import re

import numpy as np

from alphagrad.approx.env import (
    micro_actions_to_rule_specs,
    MAX_RULES_PER_VERTEX,
)
from alphagrad.approx.heads import OP_DIAG, OP_COMPRESS, OP_QUANT
from graphax.sparse.micro_actions import COMPRESS_KINDS, QUANT_DTYPES


# ---------------------------------------------------------------- seq parsing
_DIAG = re.compile(r"diag\((\d+),\s*(\d+),\s*(-?\d+)\)")
_COMP = re.compile(r"compress\('([^']+)',\s*(\d+)\)")
_QUANT = re.compile(r"quant\('([^']+)'\)")


def parse_calls(calls):
    """Decoded string calls -> per-substep micro-action arrays."""
    op, i, j, fac, kind, quant = [], [], [], [], [], []
    for c in calls:
        m = _DIAG.fullmatch(c)
        if m:
            op.append(OP_DIAG); i.append(int(m[1])); j.append(int(m[2]))
            fac.append(int(m[3])); kind.append(0); quant.append(0); continue
        m = _COMP.fullmatch(c)
        if m:
            op.append(OP_COMPRESS); i.append(int(m[2])); j.append(0); fac.append(0)
            kind.append(COMPRESS_KINDS.index(m[1])); quant.append(0); continue
        m = _QUANT.fullmatch(c)
        if m:
            op.append(OP_QUANT); i.append(0); j.append(0); fac.append(0)
            kind.append(0); quant.append(QUANT_DTYPES.index(m[1])); continue
        raise ValueError(f"unparseable call {c!r}")
    return op, i, j, fac, kind, quant


def build_order_specs(seq, env):
    # The recorded ``seq`` vertex is the agent's 0-based ACTION INDEX into
    # ``env.valid_vertices`` (see ppo_ray_worker: ``act_step`` returns the raw
    # ``vertex_action`` and the env applies ``vertex_id = vertex_action + 1``).
    # The env's elimination order + ``axis_state_static`` are keyed by the
    # 1-indexed jaxpr vertex id, so resolve each action index through
    # ``valid_vertices`` before use. Passing the raw 0-based index straight
    # through (the old behaviour) shifts the whole order by one, drops the
    # real last vertex, and indexes ``axis_static[-1]`` for vertex 0 -> the
    # measured Jacobian is garbage (frob_residual=-1, cosine_sim=0).
    valid = np.asarray(env.valid_vertices, dtype=np.int32)
    resolved = [int(valid[int(v)]) for v, _ in seq]
    axis_static = np.asarray(env.axis_state_static)
    specs = np.full((len(resolved), MAX_RULES_PER_VERTEX, 3), -1, dtype=np.int32)
    specs[:, :, 2] = 0
    n_rules = 0
    for k, ((_, calls), vid) in enumerate(zip(seq, resolved)):
        if not calls:
            continue
        op, i, j, fac, kind, quant = parse_calls(calls)
        n_rules += len(op)
        specs[k] = micro_actions_to_rule_specs(
            np.array(op), np.array(i), np.array(j), np.array(fac),
            axis_state_for_vertex=axis_static[vid - 1],
            compress_kinds=np.array(kind), quant_dtypes=np.array(quant),
        )
    return np.array(resolved, dtype=np.int32), specs, n_rules
