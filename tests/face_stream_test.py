"""The AUTOREGRESSIVE per-face loop: emit -> re-encode -> decide -> emit.

The previous version of these tests asserted only that "tokens precede the
decision" — which passed while the driver emitted SYNTHETIC ids and never
advanced the encoder at all. These assert the two properties that actually
matter:

  * the emitted ids come from the tokenizer's REAL vocabulary (`path`, `approx`,
    `SKIP`/`DIAG`/`COMPRESS`/`QUANT`), so the policy reads the same stream the
    tokenizer would have produced; and
  * the palimpsa carry is ADVANCED between decisions, so a decision is
    conditioned on the tokens describing the thing being decided.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from graphax import IncrementalJaxpr, IncrementalPathTokenizer
from graphax.jaxpr import get_vocab
from graphax.sparse.micro_actions import Diag

from alphagrad.approx.face_stream import (
    AutoregressiveFaceDriver,
    face_transforms_for,
)

VOCAB, NAMES, _ = get_vocab(10)
T_PATH, T_APPROX, T_SKIP = VOCAB["path"], VOCAB["approx"], VOCAB["SKIP"]
T_DIAG, T_COMPRESS, T_QUANT = VOCAB["DIAG"], VOCAB["COMPRESS"], VOCAB["QUANT"]


def _mlp(x, W1, W2):
    return jnp.tanh(x @ W1) @ W2


_ARGS = (jnp.ones((2, 8)), jnp.ones((8, 32)) * 0.1, jnp.ones((32, 4)) * 0.1)
_ARGNUMS = (1, 2)


class _FakeEncoder:
    """Stands in for incremental_encoder: records every extend() call so the
    test can prove the carry advanced by exactly the new tokens."""

    def __init__(self):
        self.calls: list[list[int]] = []

    def init_state(self, agent):
        return {"n": 0}

    def extend(self, agent, state, new_tokens):
        self.calls.append(list(new_tokens))
        return {"n": state["n"] + len(new_tokens)}


def _drive(policy, max_substeps=1, with_encoder=True, slots=("lhs", "rhs", "res")):
    cj = jax.make_jaxpr(_mlp)(*_ARGS)
    tk = IncrementalPathTokenizer(cj.jaxpr, _ARGNUMS, cj.literals, _ARGS,
                                  vocab_size=512)
    list(tk.base_tokens())
    incr = IncrementalJaxpr(cj.jaxpr, argnums=_ARGNUMS, consts=cj.literals,
                            args=_ARGS)
    enc = _FakeEncoder() if with_encoder else None
    d = AutoregressiveFaceDriver(policy, tk, encoder=enc,
                                 agent=(object() if with_encoder else None),
                                 max_substeps=max_substeps)
    for v in range(1, len(cj.jaxpr.eqns) + 1):
        incr.eliminate(v, face_transforms=face_transforms_for(incr, v, d, slots))
    return d, enc


# --------------------------------------------------------------------------- #
# the properties the old tests missed
# --------------------------------------------------------------------------- #

def test_encoder_is_advanced_between_decisions():
    """THE fix: the carry must be extended before the policy is consulted, and
    again after the action is emitted. Encode-once made this zero."""
    seen_at_decision = []

    def policy(ctx):
        seen_at_decision.append(ctx["enc_state"]["n"])
        return ctx["legal"][0] if ctx["legal"] else None

    d, enc = _drive(policy)
    assert enc.calls, "the encoder was never extended — the loop is not closed"
    assert d.stats["encodes"] == len(enc.calls)
    # every decision saw a strictly non-empty carry
    assert all(n > 0 for n in seen_at_decision), (
        "a decision was made against an empty encoder state"
    )
    # and the carry only ever grows (append-only recurrence)
    sizes = [len(c) for c in enc.calls]
    assert all(s > 0 for s in sizes), "an empty delta was pushed to the encoder"


def test_emitted_ids_are_real_tokenizer_vocabulary():
    """Not synthetic ids: `path` and `approx` must be the tokenizer's own."""
    d, _ = _drive(lambda ctx: ctx["legal"][0] if ctx["legal"] else None)
    assert T_PATH in d.tokens, "no real `path` token was emitted"
    assert T_APPROX in d.tokens, "no real `approx` token was emitted"
    # every id must be a legal vocabulary id for this tokenizer
    assert max(d.tokens) < 512, "an id escaped the bounded vocabulary"


def test_skip_emits_approx_SKIP_and_no_new_accumulation():
    """Spec: a skip emits `approx SKIP` and NO accumulation (edge deleted)."""
    d, _ = _drive(lambda ctx: None)
    assert d.stats["skipped"] > 0 and d.stats["applied"] == 0
    assert T_SKIP in d.tokens, "skip was not tokenized — it is invisible"
    # one `path` per face (the pre-decision accumulation), and NO second
    # accumulation after the skip.
    assert d.tokens.count(T_PATH) == d.stats["faces"]


def test_applied_action_emits_head_then_new_accumulation():
    """Spec: a non-skip approx is followed by the NEW post-approx accumulation,
    so the next decision sees the result."""
    d, _ = _drive(lambda ctx: ctx["legal"][0] if ctx["legal"] else None)
    if d.stats["applied"] == 0:
        pytest.skip("no face admitted a legal action on this graph")
    # faces contribute one `path` each; every APPLIED action contributes a
    # second one (the post-approx accumulation).
    assert d.tokens.count(T_PATH) == d.stats["faces"] + d.stats["applied"]
    assert any(t in (T_DIAG, T_COMPRESS, T_QUANT) for t in d.tokens)


def test_stream_is_append_only():
    prefixes = []

    def policy(ctx):
        prefixes.append(list(ctx["tokens"]))
        return ctx["legal"][0] if ctx["legal"] else None

    d, _ = _drive(policy)
    for earlier, later in zip(prefixes, prefixes[1:]):
        assert later[: len(earlier)] == earlier, "stream rewritten, not appended"


def test_only_legal_actions_are_offered():
    from alphagrad.approx.common.masks import rule_is_legal

    offered = []

    def policy(ctx):
        offered.extend((ctx["operand"], a) for a in ctx["legal"])
        return ctx["legal"][0] if ctx["legal"] else None

    _drive(policy)
    assert offered, "no actions were ever offered"
    for st, a in offered:
        assert rule_is_legal(st, a), f"offered an illegal action: {a}"


def test_decisions_can_differ_per_face():
    def policy(ctx):
        return ctx["legal"][0] if (ctx["face_index"] % 2 == 0 and ctx["legal"]) else None

    d, _ = _drive(policy)
    if d.stats["faces"] < 2:
        pytest.skip("needs >= 2 faces")
    assert d.stats["applied"] >= 1 and d.stats["skipped"] >= 1


def test_runs_without_an_encoder():
    d, enc = _drive(lambda ctx: None, with_encoder=False)
    assert enc is None and d.enc_state is None
    assert d.tokens and d.stats["encodes"] == 0
