"""Does the append-only token stream ACTUALLY have the prefix property?

`_incremental_stream_tokens` caches a tokenizer per elimination prefix and
extends it, on the stated invariant: "The stream for a prefix is a byte-wise
prefix of the stream for any extension."

That invariant is what the ancestor-extension fast path (and any prefix memo
for the oracle / face-key enumeration) rests on. But
`decode_vertex_rule_specs` takes `is_last`, and COMPRESS is honored ONLY on
the last vertex of a partial order:

    env.py: "COMPRESS: ... Only honored on the LAST vertex of the partial
             order (val.ndim reduction upstream of a later elimination trips
             graphax's shape assertion)."

So the vertex at index k-1 is decoded WITH its COMPRESS when the prefix has
length k, and WITHOUT it when the prefix has length k+1. If that changes the
emitted tokens, the length-k stream is NOT a prefix of the length-k+1 stream
and the fast path silently produces a different observation than a cold
replay.

These tests decide it empirically, separately for:
  * DIAG-only rules   (expected: prefix property HOLDS)
  * COMPRESS rules    (the suspect case)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GRAPHAX_ALLOW_PARTIAL_ORDER", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagrad.approx.env import (
    COMPRESS_SENTINEL, MAX_RULES_PER_VERTEX, decode_vertex_rule_specs,
)


def _fn(x, y):
    return jnp.tanh(jnp.sin(x) * y) + jnp.exp(jnp.sin(x) * y)


ARGS = (jnp.ones((4, 4)) * 0.5, jnp.ones((4, 4)) * 0.4)


def _stream_for(prefix, specs_by_v, vocab=248):
    """Cold replay: tokenize exactly `prefix`, decoding rules with `is_last`
    relative to THIS prefix (what `_callback` does for a partial order)."""
    from graphax import IncrementalPathTokenizer

    cj = jax.make_jaxpr(_fn)(*ARGS)
    tk = IncrementalPathTokenizer(
        cj.jaxpr, (0, 1), list(cj.literals), list(ARGS), vocab_size=vocab)
    stream = [int(t) for t in tk.base_tokens()]
    last = len(prefix) - 1
    for k, v in enumerate(prefix):
        rules = decode_vertex_rule_specs(
            cj.jaxpr, int(v), specs_by_v[int(v)], is_last=(k == last))
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
    """Is stream(prefix[:k]) a byte-prefix of stream(prefix[:k+1])?"""
    prefix = [1, 2, 3]
    specs = {v: _rows(kind) for v in prefix}
    short = _stream_for(prefix[:2], specs)
    long = _stream_for(prefix[:3], specs)
    holds = long[: len(short)] == short
    if kind == "compress" and not holds:
        # Localise the divergence for the report.
        i = next(i for i, (a, b) in enumerate(zip(short, long)) if a != b)
        pytest.fail(
            f"PREFIX PROPERTY VIOLATED for {kind}: streams diverge at token "
            f"{i} of {len(short)} (short={short[i-2:i+3]}, "
            f"long={long[i-2:i+3]}). Ancestor extension is UNSOUND here — "
            f"the k-th vertex is decoded with is_last=True in the short "
            f"prefix and is_last=False in the long one."
        )
    assert holds, f"prefix property violated for kind={kind}"


def test_islast_changes_decoded_rules_for_compress():
    """Directly: does is_last actually change the decoded rule list?"""
    cj = jax.make_jaxpr(_fn)(*ARGS)
    rows = _rows("compress")
    as_last = decode_vertex_rule_specs(cj.jaxpr, 1, rows, is_last=True)
    as_mid = decode_vertex_rule_specs(cj.jaxpr, 1, rows, is_last=False)
    # This is the mechanism under test; report both either way.
    print(f"\nis_last=True  -> {as_last}\nis_last=False -> {as_mid}")
    assert as_last is not None and as_mid is not None
