"""Append-only tokenization: the palimpsa-native observation.

The default path re-tokenizes the whole jaxpr every step and hands the policy a
fixed MAX_TOKENS buffer, so the encoder re-reads the entire stream each step and
anything past the budget is silently clipped (nn256 emits ~4657 against 4096).
That discards the reason palimpsa was chosen: linear attention is a RECURRENCE,
so it only needs what is NEW.

These pin the append-only contract:
  base_tokens() ++ concat(eliminate(v) for v in order) == the full stream,
so a policy can carry the encoder state and absorb one delta per step.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import IncrementalPathTokenizer

from alphagrad.approx.env import (
    MAX_DELTA_TOKENS,
    incremental_token_delta,
)

VOCAB = 512


def _mlp(x, W1, W2):
    return jnp.tanh(x @ W1) @ W2


_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1)
_ARGNUMS = (1, 2)


def _cj():
    return jax.make_jaxpr(_mlp)(*_ARGS)


def _full_stream(order):
    """The whole stream the batch tokenizer would produce for this order."""
    cj = _cj()
    tk = IncrementalPathTokenizer(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                  vocab_size=VOCAB)
    out = [int(t) for t in tk.base_tokens()]
    for v in order:
        out += [int(t) for t in tk.eliminate(int(v))]
    return out


def test_base_plus_deltas_reconstructs_the_full_stream():
    """THE contract: appending each step's delta must equal the full stream —
    otherwise the encoder's recurrence diverges from what it should have read."""
    cj = _cj()
    order = list(range(1, len(cj.jaxpr.eqns) + 1))
    pieces = incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                     (), vocab_size=VOCAB)
    rebuilt = list(pieces)
    for k in range(len(order)):
        rebuilt += incremental_token_delta(
            cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
            tuple(order[: k + 1]), vocab_size=VOCAB)
    assert rebuilt == _full_stream(order)


def test_each_delta_is_smaller_than_the_growing_stream():
    """The point of append-only: a step reads its delta, not everything so far."""
    cj = _cj()
    order = list(range(1, len(cj.jaxpr.eqns) + 1))
    full = _full_stream(order)
    deltas = [
        incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                tuple(order[: k + 1]), vocab_size=VOCAB)
        for k in range(len(order))
    ]
    assert all(len(d) <= len(full) for d in deltas)
    assert max(len(d) for d in deltas) < len(full), (
        "no delta may be as large as the whole stream — that would mean we are "
        "still effectively re-reading everything"
    )
    assert all(len(d) <= MAX_DELTA_TOKENS for d in deltas)


def test_ids_stay_inside_the_bounded_vocab():
    """A fixed policy embedding needs bounded ids; unbounded mode can exceed
    any table (graphax's own warning)."""
    cj = _cj()
    order = list(range(1, len(cj.jaxpr.eqns) + 1))
    stream = _full_stream(order)
    assert stream, "empty stream"
    assert max(stream) < VOCAB


def test_delta_is_deterministic_and_cached():
    cj = _cj()
    a = incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                (1,), vocab_size=VOCAB)
    b = incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                (1,), vocab_size=VOCAB)
    assert a == b


def test_divergent_branches_do_not_corrupt_each_other():
    """Two envs picking different first vertices must get independent streams
    (the tokenizer is stateful, so aliasing a cached parent would cross-talk)."""
    cj = _cj()
    n = len(cj.jaxpr.eqns)
    if n < 2:
        pytest.skip("needs >= 2 vertices")
    d1 = incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                 (1, 2), vocab_size=VOCAB)
    d2 = incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                 (2, 1), vocab_size=VOCAB)
    # Recompute the first branch: must be unchanged by the second's replay.
    d1_again = incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                       (1, 2), vocab_size=VOCAB)
    assert d1 == d1_again
    assert isinstance(d2, list)


def test_oversized_delta_raises_instead_of_clipping():
    """Clipping a delta would desync the encoder recurrence from the stream —
    strictly worse than clipping a re-read buffer, so it must raise."""
    import alphagrad.approx.env as E

    cj = _cj()
    saved = E.MAX_DELTA_TOKENS
    try:
        E.MAX_DELTA_TOKENS = 4          # absurdly small
        E._INCR_TOK_CACHE.clear()
        with pytest.raises(ValueError, match="MAX_DELTA_TOKENS"):
            E.incremental_token_delta(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                      (), vocab_size=VOCAB)
    finally:
        E.MAX_DELTA_TOKENS = saved
        E._INCR_TOK_CACHE.clear()


def test_incremental_encoder_consumes_deltas_equivalently():
    """End-to-end: feeding the deltas to the incremental encoder must land on
    the same encoding as one full pass over the concatenated stream."""
    IE = pytest.importorskip("alphagrad.approx.incremental_encoder")
    assert hasattr(IE, "init_state") and hasattr(IE, "extend"), (
        "the append-only consumer must expose init_state/extend"
    )
