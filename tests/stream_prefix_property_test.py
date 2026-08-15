"""Does the append-only token stream ACTUALLY have the prefix property?

`_incremental_stream_tokens` caches a tokenizer per elimination prefix and
extends it, on the stated invariant: "The stream for a prefix is a byte-wise
prefix of the stream for any extension."

That invariant is what the ancestor-extension fast path, the face-enum prefix
cache and any prefix memo for the oracle rest on. It USED TO BE FALSE for
COMPRESS: `decode_vertex_rule_specs` took an `is_last` flag and emitted
Compress only when it was set, so the vertex at index k-1 was tokenized WITH
its Compress at prefix length k and WITHOUT it at length k+1 -- the streams
diverged (measured: token 681 of 931 on this graph), and BOTH prefix caches
carried a COMPRESS carve-out because of it (v40: ext=1/431).

The gate is GONE (2026-08-15). Measured before removing it: the same decoded
COMPRESS applied at all 23 positions of the NeuralNetwork plan and 11 sampled
positions of the TransformerLM plan, raw and hook-wrapped, raised NOTHING and
genuinely changed the Jacobian -- the gate was discarding real approximations,
not preventing a failure.

So the property now holds for EVERY rule kind and these tests ASSERT it. If a
future change re-introduces any position sensitivity in the decode, the
`compress` case here fails first.
"""
import inspect
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.micro_actions import Compress

from alphagrad.approx.env import (
    COMPRESS_SENTINEL, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs,
)


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)


def _stream_for(prefix, specs_by_v, vocab=248):
    """Cold replay: tokenize exactly `prefix`, decoding each vertex's rules
    the way `_callback` does for a partial order."""
    from graphax import IncrementalPathTokenizer

    cj = jax.make_jaxpr(_fn)(*ARGS)
    tk = IncrementalPathTokenizer(
        cj.jaxpr, (0, 1), list(cj.literals), list(ARGS), vocab_size=vocab)
    stream = [int(t) for t in tk.base_tokens()]
    for v in prefix:
        rules = decode_vertex_rule_specs(cj.jaxpr, int(v), specs_by_v[int(v)])
        stream += [int(t) for t in tk.eliminate(int(v), tuple(rules))]
    return stream


def _rows(kind):
    """MAX_RULES-shaped spec rows: 'none', 'diag', or 'compress'."""
    rows = [[-1, -1, 0] for _ in range(MAX_RULES_PER_VERTEX)]
    if kind == "diag":
        rows[0] = [0, 0, 2]
    elif kind == "compress":
        rows[0] = [COMPRESS_SENTINEL, 0, 0]
    return rows


@pytest.mark.parametrize("kind", ["none", "diag", "compress"])
def test_prefix_property(kind):
    """stream(prefix[:k]) must be a byte-prefix of stream(prefix[:k+1]),
    at EVERY k and for EVERY rule kind -- COMPRESS included."""
    prefix = [1, 2, 3]
    specs = {v: _rows(kind) for v in prefix}
    for k in range(1, len(prefix)):
        short = _stream_for(prefix[:k], specs)
        long = _stream_for(prefix[:k + 1], specs)
        if long[: len(short)] != short:
            i = next((i for i, (a, b) in enumerate(zip(short, long))
                      if a != b), len(short))
            pytest.fail(
                f"PREFIX PROPERTY VIOLATED for {kind} at k={k}: streams "
                f"diverge at token {i} of {len(short)} "
                f"(short={short[max(0, i-2):i+3]}, "
                f"long={long[max(0, i-2):i+3]}). Ancestor extension in "
                f"_incremental_stream_tokens and the _FACE_ENUM_CACHE "
                f"pop-extend are UNSOUND in this state.")


def test_decode_has_no_position_parameter():
    """The gate is gone at the level of the signature, not just its default.

    A `decode_vertex_rule_specs(..., is_last=...)` that silently defaulted to
    True would leave every caller free to re-introduce the divergence.
    """
    params = inspect.signature(decode_vertex_rule_specs).parameters
    assert "is_last" not in params, (
        f"decode_vertex_rule_specs regrew a position parameter: {list(params)}")


def test_compress_decodes_at_any_position():
    """The same COMPRESS row decodes to the same rule for every vertex."""
    cj = jax.make_jaxpr(_fn)(*ARGS)
    rows = _rows("compress")
    got = [decode_vertex_rule_specs(cj.jaxpr, v, rows)
           for v in (1, 2, 3)]
    assert all(len(g) == 1 and isinstance(g[0], Compress) for g in got), got
    assert got[0] == got[1] == got[2], got
