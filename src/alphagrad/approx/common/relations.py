"""Relation extraction for the relational-bias encoder (Stage B.1).

The architecture spec calls for typed relational biases over five relations
(predecessor / successor / shared-tensor / same-scope / shape-compatible).
This module provides the cheapest, most jit-friendly subset:

* `same_eqn`  — tokens i and j live in the same residual-jaxpr equation.
* `earlier`   — eqn_id[i] > eqn_id[j] (i.e. token j is in an earlier equation
                in topological order, so it's a *potential* predecessor of i).
* `later`     — symmetrical, eqn_id[i] < eqn_id[j].

The "earlier / later" bias is a topological-order proxy for true
predecessor / successor: every actual predecessor of vertex i comes earlier
in the equation list (jaxpr eqns are emitted in topological order), so the
proxy is a superset of the true predecessor relation. The model is free to
learn that not every "earlier" eqn is a real predecessor — the bias only
shifts the prior, it doesn't enforce structure.

We don't materialise the (T, T) relation tensor here; the encoder layer
derives it on the fly from a per-token `(T,) int32` eqn_id array. That keeps
EnvState memory at one int32 per token instead of `T*T*R` floats, and the
encoder still gets the full structural signal at attention time.
"""

from __future__ import annotations

import numpy as np

# Subset implemented today; see module docstring.
RELATION_NAMES: tuple[str, ...] = ("same_eqn", "earlier", "later")
NUM_RELATIONS: int = len(RELATION_NAMES)


def compute_eqn_ids_from_tokens(
    tokens: np.ndarray, vocab: dict[str, int]
) -> np.ndarray:
    """Per-token equation ID, derived from the graphax tokenizer's structure.

    The tokenizer emits a sequence of the form
    ``invars... [~ defs...] { eqn_0 \n eqn_1 \n ... eqn_{E-1} } [pad...]``.
    We scan once: tokens before ``{`` get id ``-1`` (invars / defs / boundary
    markers); tokens between ``{`` and ``}`` are tagged with their equation
    index, with ``\n`` acting as the equation separator. The closing ``}`` and
    every padding token also get ``-1``.

    Returns a numpy int32 array of the same length as `tokens`. The ``-1``
    sentinel is what the encoder layer uses to mask out non-equation tokens
    when lifting eqn-level relations to per-token biases.
    """
    LBRACE = vocab["{"]
    RBRACE = vocab["}"]
    NEWLINE = vocab["\n"]

    out = np.full(tokens.shape, -1, dtype=np.int32)
    in_eqns = False
    eqn_id = 0
    for i, t in enumerate(tokens):
        t_int = int(t)
        if t_int == 0:
            # Padding token — leave as -1 and stop scanning (the rest of the
            # buffer is guaranteed to be padding).
            break
        if not in_eqns:
            if t_int == LBRACE:
                in_eqns = True
            continue
        if t_int == RBRACE:
            in_eqns = False
            continue
        out[i] = eqn_id
        if t_int == NEWLINE:
            eqn_id += 1
    return out
